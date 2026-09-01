#!/usr/bin/env python3
"""멀티기업 질문 귀속 채점 — 값을 몰라도 잴 수 있는 것만 잰다.

## 왜 별도 채점기인가

"삼성전자와 삼성SDI의 매출액을 비교해줘" 에서 나온 실패는 **틀린 숫자**가
아니라 **틀린 출처**였다. 삼성전자 값을 삼성SDI 사업보고서의 '최대주주
재무현황' 표에서 가져왔다. 값만 보는 채점기는 이걸 못 잡는다 — 숫자 자체는
그 표에 적힌 그대로라 맞아 보이기 때문이다.

정답 값을 몰라도 다음 두 가지는 잴 수 있다.

  커버리지  질문에 나온 회사마다 **그 회사 문서를 실제로 인용**했는가
  순도      질문에 없는 회사의 문서를 인용하지 않았는가

report_id -> 회사는 manifest 로 확정된다. 답변이 어느 표기(현대차/현대자동차)
를 썼는지와 무관하므로 별칭 회사도 그대로 잴 수 있다.

## 쓰는 법

    python3 scripts/score_attribution.py --gold eval/suite_multi.jsonl \
        --answers results/multi1/answers.jsonl --out results/multi1

HCX 를 부르지 않는다. 이미 받아둔 답변만 다시 읽으므로 몇 초면 끝난다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# "근거: report_id(exchange_20251208800031), report_id(...)" 형식.
# 형식이 흔들려도 문서 ID 자체는 접두사가 정해져 있어 본문 어디서든 잡힌다.
_DOC_ID = re.compile(r"\b((?:periodic|major|exchange|holding)_\d{8,})\b")


def load_manifest(path: Path) -> dict[str, str]:
    """doc_id -> 회사명."""
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row.get("doc_id")
            if doc_id:
                out[doc_id] = row.get("corp_name") or ""
    return out


def cited_ids(answer: str) -> list[str]:
    seen: list[str] = []
    for match in _DOC_ID.finditer(answer or ""):
        if match.group(1) not in seen:
            seen.append(match.group(1))
    return seen


def grade_one(answer: str, companies: list[str], owner: dict[str, str],
              *, pipeline_ids: list[str] | None = None) -> dict:
    """`pipeline_ids` 가 있으면 그것을 쓴다 — 본문 파싱보다 정확하다.

    2026-09-01 첫 측정에서 28문항 중 8문항이 '근거 0건'으로 나왔는데, 실제로는
    제대로 답한 문항이었다. 모델이 프롬프트 지시("근거: report_id(...)")를
    안 지키고 "근거: [EVIDENCE 1]" 이라고만 썼기 때문이다. 채점기가 본문
    형식에 매달리면 시스템이 아니라 채점기를 측정하게 된다.
    """
    ids = list(pipeline_ids) if pipeline_ids else cited_ids(answer)
    cited_companies = [owner.get(i, "?") for i in ids]
    covered = [c for c in companies if c in cited_companies]
    # 질문에 없는 회사 문서. manifest 에 없는 ID(?)는 따로 센다 — 지어낸
    # 문서 ID 일 수도 있어서 순도와 섞으면 원인이 흐려진다.
    foreign = sorted({c for c in cited_companies if c and c != "?" and c not in companies})
    unknown = [i for i in ids if i not in owner]
    return {
        "n_companies": len(companies),
        "n_cited_docs": len(ids),
        "covered": len(covered),
        "coverage": round(len(covered) / len(companies), 4) if companies else None,
        "missing": " / ".join(c for c in companies if c not in cited_companies),
        "foreign": " / ".join(foreign),
        "n_foreign": len(foreign),
        "unknown_ids": " / ".join(unknown),
        "clean": int(not foreign and len(covered) == len(companies)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/suite_multi.jsonl")
    ap.add_argument("--answers", required=True,
                    help="score_answers.py 가 남긴 answers.jsonl")
    ap.add_argument("--manifest", default="corpus/manifest.jsonl")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    gold = {}
    with Path(args.gold).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                gold[row["id"]] = row

    answers = {}
    with Path(args.answers).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                answers[row.get("id")] = row

    owner = load_manifest(Path(args.manifest))

    rows = []
    for qid, g in gold.items():
        if qid not in answers:
            continue
        row = answers[qid]
        rows.append({"id": qid, "n_asked": g["n_companies"], "note": g.get("note", ""),
                     "query": g["query"][:70],
                     "source": "pipeline" if row.get("cited_ids") else "본문파싱",
                     **grade_one(row.get("answer") or "", g["companies"], owner,
                                 pipeline_ids=row.get("cited_ids"))})

    if not rows:
        print("대조할 답변이 없다 — --answers 경로를 확인하라.")
        return 1

    n = len(rows)
    clean = sum(r["clean"] for r in rows)
    cov = sum(r["coverage"] for r in rows) / n
    foreign_rows = [r for r in rows if r["n_foreign"]]
    print(f"\n{'='*66}\n멀티기업 귀속 채점  ({n}문항)\n{'='*66}")
    print(f"  회사 커버리지 (질문한 회사 문서를 인용한 비율)  {cov:6.1%}")
    print(f"  완전 정상 (전부 인용 + 남의 회사 없음)          {clean}/{n} = {clean/n:.1%}")
    print(f"  남의 회사 문서를 인용한 문항                    {len(foreign_rows)}건")
    parsed = sum(1 for r in rows if r["source"] == "본문파싱")
    if parsed:
        print(f"  ⚠️  {parsed}문항은 파이프라인 인용 기록이 없어 본문에서 긁었다 — "
              f"과소평가일 수 있다(옛 결과 폴더).")

    print(f"\n  회사 수별:")
    for k in sorted({r["n_asked"] for r in rows}):
        sub = [r for r in rows if r["n_asked"] == k]
        c = sum(r["coverage"] for r in sub) / len(sub)
        print(f"    {k}곳  n={len(sub):2d}  커버리지 {c:6.1%}  "
              f"완전정상 {sum(r['clean'] for r in sub)}/{len(sub)}")

    bad = [r for r in rows if not r["clean"]]
    if bad:
        print(f"\n  문제 문항 {len(bad)}건:")
        for r in bad:
            print(f"    {r['id']} ({r['n_asked']}곳) 커버 {r['covered']}/{r['n_companies']}"
                  f"{'  빠짐: ' + r['missing'] if r['missing'] else ''}"
                  f"{'  남의회사: ' + r['foreign'] if r['foreign'] else ''}")

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "attribution.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), restval="")
            w.writeheader()
            w.writerows(rows)
        (out_dir / "attribution.json").write_text(json.dumps(
            {"n": n, "coverage": round(cov, 4), "clean": clean,
             "clean_rate": round(clean / n, 4), "foreign_rows": len(foreign_rows)},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  저장: {out_dir}/attribution.csv, attribution.json")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
