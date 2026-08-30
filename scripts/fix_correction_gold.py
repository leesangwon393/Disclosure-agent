#!/usr/bin/env python3
"""정정 비교 문항의 정답을 **후보 여럿**으로 다시 만든다.

## 무엇이 문제였나

    Q  "현대건설의 단일판매ㆍ공급계약체결 공시가 정정된 내역이 있는가?
        있다면 최초 공시와 최종 정정본 사이에 무엇이 달라졌는지 설명해줘."

질문이 **어느 공시인지 특정하지 않는다.** 현대건설의 해당 유형 정정 공시는
70건이고, 값이 실제로 바뀐 정정 체인만 10개다. 그런데 정답지는 그중 하나의
값만 정답으로 인정했다. 모델은 다른 체인을 설명했고 오답 처리됐다 —
**틀린 게 아니라 다른 걸 고른 것이다.**

## 어떻게 고치나

(회사, 공시유형)의 **모든 정정 체인**을 후보로 넣는다(`required_any`).
하나라도 온전히 설명하면 정답이다. 각 후보는 그 체인에서 실제로 바뀐 항목의
**최종 정정본 값**이다 — "최신 정정본 수치를 답하는가"가 원래 채점 기준이다.

기존에 손으로 확인한 정답이 있으면 후보에 **더한다**(지우지 않는다).
자동 추출이 못 잡는 형태가 있다 — 예를 들어 두산로보틱스(S025)의 최종 정정은
값이 바뀐 게 아니라 **삭제**(합의해제)라 수치 비교로는 안 잡힌다.

## 건수도 같이 기록한다

`chain_count` / `corrected_doc_count` 를 넣어 둔다. "정정된 공시가 70건
있습니다" 처럼 전체 규모를 밝히는 답이 더 낫기 때문인데, **아직 필수로는
걸지 않는다** — 그 능력을 재본 적이 없으므로 먼저 측정한다.
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_eval_suite_v2 import clean_key, is_bad_key, norm_report  # noqa: E402

MAX_TOKENS_PER_CHAIN = 4

# 문서 하나로 끝나는 정정("[기재정정]...")에서 정답으로 쓸 대표 수치 항목.
# 이런 문서는 `correction_group_id` 가 자기 자신이라 문서 간 체인 열거에서
# 통째로 빠졌다. 실제로는 문서 안에 `정정전/정정후` 표로 기록된 정정이고,
# 모델이 그걸 설명해도 오답이 됐다(2026-08-31, S023·S024 실측).
#
# 파서가 `정정전` 칸을 표 헤더("정정후")로 잡아버려 **이전 값은 facts 에
# 없다.** 그래서 최종값 + 항목명만 요구한다 — 채점 기준이 원래
# "최신 정정본 수치를 답하는가" 이므로 그것으로 충분하다.
SINGLE_DOC_VALUE_KEYS = (
    "계약금액", "확정계약금액", "계약금액총액", "투자금액", "양수금액", "양도금액",
    "취득금액", "처분금액", "권면총액", "발행가액", "신주발행가액",
)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def load_manifest(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["doc_id"]] = d
    return out


def chains_for(db, man, company: str, kind: str) -> list[dict]:
    """(회사, 공시유형)에서 값이 바뀐 정정 체인을 전부 찾는다."""
    gids = [r[0] for r in db.execute(
        "SELECT DISTINCT correction_group_id FROM facts "
        "WHERE company=? AND correction_group_id IS NOT NULL", (company,))]
    out = []
    for gid in gids:
        rows = db.execute(
            "SELECT doc_id, filing_date, key_norm, value_text, is_latest "
            "FROM facts WHERE correction_group_id=? AND value_num IS NOT NULL",
            (gid,)).fetchall()
        if not rows:
            continue
        docs = {r[0] for r in rows}
        if kind and not any(d in man and norm_report(nfc(man[d]["report_nm"])) == kind
                            for d in docs):
            continue
        dates = {r[0]: (r[1] or "") for r in rows}
        latest = {r[0] for r in rows if r[4]}
        first = min(dates, key=lambda d: dates[d])
        final = (max(latest, key=lambda d: dates[d]) if latest
                 else max(dates, key=lambda d: dates[d]))
        if first == final:
            continue

        def values(doc):
            acc = collections.defaultdict(set)
            for d, _dt, key, val, _l in rows:
                if d == doc and not is_bad_key(key):
                    acc[clean_key(key)].add(val)
            return {k: list(v)[0] for k, v in acc.items() if len(v) == 1}

        before, after = values(first), values(final)
        changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
        if not changed:
            continue
        # 값만 넣으면 **토큰 1개짜리 후보**가 생긴다. 그러면 서술형 문항이
        # "숫자 하나만 언급하면 만점"이 된다(2026-08-31 발견, 4문항 6그룹).
        # 채점 기준이 원래 "변경 항목을 특정하는가" 이므로 항목명도 함께 요구한다.
        tokens = [after[k] for k in changed][:MAX_TOKENS_PER_CHAIN]
        if len(tokens) < 2:
            tokens = tokens + changed[:1]
        out.append({"first_doc": first, "final_doc": final,
                    "changed": changed, "tokens": tokens})
    return out


def single_doc_corrections(db, man, company: str, kind: str, covered: set) -> list[dict]:
    """문서 하나로 끝나는 정정본을 후보로 만든다."""
    out = []
    for d in man.values():
        doc_id = d["doc_id"]
        if doc_id in covered or not d.get("is_correction"):
            continue
        if nfc(d["corp_name"]) != company:
            continue
        if kind and norm_report(nfc(d.get("report_nm", ""))) != kind:
            continue
        rows = db.execute(
            "SELECT key_norm, value_text FROM facts "
            "WHERE doc_id=? AND value_num IS NOT NULL", (doc_id,)).fetchall()
        picked = []
        for want in SINGLE_DOC_VALUE_KEYS:
            for key, val in rows:
                if clean_key(key) == want and val and val not in picked:
                    picked.append(val)
                    picked.append(want)
                    break
            if picked:
                break
        if picked:
            out.append({"final_doc": doc_id, "tokens": picked[:MAX_TOKENS_PER_CHAIN]})
    return out


def corrected_doc_count(man, company: str, kind: str) -> int:
    return sum(1 for d in man.values()
               if nfc(d["corp_name"]) == company and d.get("is_correction")
               and (not kind or norm_report(nfc(d.get("report_nm", ""))) == kind))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", action="append", required=True,
                    help="여러 번 줄 수 있다: --suite eval/suite_v1.jsonl")
    ap.add_argument("--facts", default="artifacts_v2/facts/facts.sqlite")
    ap.add_argument("--manifest", default="corpus/manifest.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = sqlite3.connect(args.facts)
    man = load_manifest(Path(args.manifest))

    for suite_path in args.suite:
        path = Path(suite_path)
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        touched = 0
        for row in rows:
            is_correction_q = (row.get("generator") == "correction"
                               or "정정된 내역" in (row.get("query") or ""))
            if not is_correction_q:
                continue
            company = nfc(row.get("company") or "")
            if not company:
                for d in row.get("gold_doc_ids") or []:
                    if d in man:
                        company = nfc(man[d]["corp_name"])
                        break
            kind = ""
            for d in row.get("gold_doc_ids") or []:
                if d in man:
                    kind = norm_report(nfc(man[d].get("report_nm", "")))
                    break
            if not company:
                continue

            chains = chains_for(db, man, company, kind)
            covered = {d for c in chains for d in (c["first_doc"], c["final_doc"])}
            singles = single_doc_corrections(db, man, company, kind, covered)
            groups = [c["tokens"] for c in chains + singles if c["tokens"]]
            # 손으로 확인한 기존 정답은 지우지 않고 후보에 더한다.
            if row.get("required_all"):
                groups.append(list(row["required_all"]))
            if not groups:
                continue
            # 중복 제거(순서 유지)
            seen, uniq = set(), []
            for g in groups:
                key = tuple(g)
                if key not in seen:
                    seen.add(key)
                    uniq.append(g)
            row["required_any"] = uniq
            row["chain_count"] = len(chains)
            row["corrected_doc_count"] = corrected_doc_count(man, company, kind)
            row["answer_source"] = "chains_20260830"
            touched += 1
            print(f"  [{row.get('id')}] {company}/{kind or '-'}: "
                  f"후보 {len(uniq)}개 (문서간 체인 {len(chains)} + 단일문서 "
                  f"{len(singles)}, 정정공시 {row['corrected_doc_count']}건)")
        print(f"{path}: {touched}문항 갱신")
        if not args.dry_run and touched:
            path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                            encoding="utf-8")
    if args.dry_run:
        print("\n(--dry-run: 파일을 쓰지 않았다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
