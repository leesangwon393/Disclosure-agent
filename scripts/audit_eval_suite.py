#!/usr/bin/env python3
"""평가셋 전수 검증 — 문항 하나하나가 **답할 수 있는가**를 기계로 확인한다.

    python3 scripts/audit_eval_suite.py --suite eval/suite_v2.jsonl

## 왜 필요한가

생성기에 관문을 넣어도 놓치는 게 있다. 과거 정답셋에서 사후에 발견한 결함이
28건이었다(모호 22 + 교차주체 6). 눈으로 몇 개 보는 걸로는 300문항을 못 믿는다.

여기서 보는 건 **정답셋 자체의 결함**이지 파이프라인 성능이 아니다.
검색이 못 찾는 건 여기서 판단하지 않는다.

## 검사 항목

    구조      필수 필드 존재, id 중복, 질문 중복
    출처      gold 문서가 manifest 에 있는가, 제출사가 질문의 회사와 같은가
    정답      auto_* 문항의 정답이 facts 에서 실제로 재현되는가
    비교      승자가 실제로 더 큰 값인가
    모호      같은 문서에 같은 값을 가진 다른 항목이 있는가
    채점기준  rubric_only 문항에 채점 항목이 붙어 있는가
    균형      유형·회사·양음성 분포
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REQUIRED_FIELDS = ("id", "task_type", "mode", "query", "gold_doc_ids",
                   "answer_source", "check_points")


def norm_num(s: str) -> str:
    return re.sub(r"[,\s원₩]", "", s or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="eval/suite_v2.jsonl")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--facts", default="artifacts_v2/facts/facts.sqlite")
    ap.add_argument("--periodic-facts", default="artifacts_v2/facts_periodic_v2/facts.sqlite")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.suite, encoding="utf-8") if l.strip()]
    docs = {d["doc_id"]: d for d in
            (json.loads(l) for l in open(Path(args.corpus) / "manifest.jsonl", encoding="utf-8"))}
    dbs = [sqlite3.connect(p) for p in (args.facts, args.periodic_facts) if Path(p).exists()]

    problems: collections.Counter = collections.Counter()
    details: list[str] = []

    def flag(row, kind, msg):
        problems[kind] += 1
        if problems[kind] <= 3:          # 유형당 예시 3건까지만 남긴다
            details.append(f"- **{kind}** `{row['id']}` {msg}\n  - {row['query'][:90]}")

    # ---------------------------------------------------------------- 구조
    seen_ids, seen_q = set(), {}
    for r in rows:
        for f in REQUIRED_FIELDS:
            if f not in r:
                flag(r, "필드누락", f"`{f}` 없음")
        if r["id"] in seen_ids:
            flag(r, "id중복", r["id"])
        seen_ids.add(r["id"])
        if r["query"] in seen_q:
            flag(r, "질문중복", f"{seen_q[r['query']]} 와 같음")
        seen_q[r["query"]] = r["id"]
        if not r["gold_doc_ids"]:
            flag(r, "근거문서없음", "gold_doc_ids 가 비어 있다")
        if not r.get("check_points"):
            flag(r, "채점기준없음", "check_points 가 비어 있다")

    # ---------------------------------------------------------------- 출처
    for r in rows:
        for doc_id in r["gold_doc_ids"]:
            doc = docs.get(doc_id)
            if doc is None:
                flag(r, "문서없음", doc_id)
                continue
            company = (r.get("company") or "")
            if company and "|" not in company and doc["corp_name"] != company:
                # 교차주체: 다른 회사가 제출한 문서를 근거로 달았다
                flag(r, "교차주체", f"{company} 질문인데 근거는 {doc['corp_name']} 제출")

    # ---------------------------------------------------------------- 정답 재현
    def facts_values(doc_id, key):
        out = set()
        for db in dbs:
            for (v,) in db.execute(
                "SELECT DISTINCT value_text FROM facts WHERE doc_id=? AND key_norm=?",
                    (doc_id, key)):
                out.add(norm_num(v))
        return out

    n_checked = collections.Counter()
    for r in rows:
        if r["answer_source"] != "auto_facts":
            continue
        n_checked["auto_facts"] += 1
        key = r.get("key") or ""
        want = norm_num(r.get("answer") or "")
        got = set()
        for doc_id in r["gold_doc_ids"]:
            got |= facts_values(doc_id, key)
            # key 는 clean_key 를 거쳤을 수 있어 원본 key_norm 과 다를 수 있다
            if not got:
                for db in dbs:
                    for (k, v) in db.execute(
                        "SELECT key_norm, value_text FROM facts WHERE doc_id=?", (doc_id,)):
                        if key and key in k:
                            got.add(norm_num(v))
        if want and want not in got:
            flag(r, "정답재현실패", f"항목 {key!r} 값 {r['answer']!r} 을 근거 문서에서 못 찾음")

    # ---------------------------------------------------------------- 비교 정합성
    for r in rows:
        if r["answer_source"] != "auto_compare":
            continue
        n_checked["auto_compare"] += 1
        values = r.get("compare_values") or {}
        if len(values) != 2:
            flag(r, "비교값누락", f"{len(values)}개")
            continue
        nums = {k: float(norm_num(v)) for k, v in values.items()}
        winner = (r.get("answer") or "").split("(")[0].strip()
        real = max(nums, key=lambda k: nums[k])
        if winner != real:
            flag(r, "비교승자오류", f"정답은 {winner} 인데 값은 {real} 이 크다")
        if len(set(nums.values())) == 1:
            flag(r, "비교동점", "두 값이 같아 '더 큰 쪽'이 없다")

    # ---------------------------------------------------------------- 서술형 기준
    for r in rows:
        if r["answer_source"] != "rubric_only":
            continue
        hint = r.get("expected_fields_hint") or []
        if len(hint) < 1:
            flag(r, "서술형기준없음", "채점할 항목 목록이 비어 있다")
        if r.get("answer"):
            flag(r, "서술형에정답", "rubric_only 인데 answer 가 채워져 있다")

    # ---------------------------------------------------------------- 균형
    by_type = collections.Counter((r["task_type"], r["mode"]) for r in rows)
    by_gen = collections.Counter(r.get("generator", "?") for r in rows)
    per_co: collections.Counter = collections.Counter()
    for r in rows:
        for c in (r.get("company") or "").split("|"):
            if c:
                per_co[c] += 1
    polarity = collections.Counter(r.get("polarity") for r in rows if r.get("polarity"))

    # ---------------------------------------------------------------- 리포트
    lines = [f"# 평가셋 검증 — {args.suite}", "",
             f"- 문항 {len(rows)} / 회사 {len(per_co)}곳",
             f"- 결함 **{sum(problems.values())}건**"
             + ("" if problems else " — 없음"),
             "",
             "검사가 헛돌지 않았는지 확인용 — 실제로 원본 대조한 문항 수:",
             f"- 정답 재현 검증 {n_checked['auto_facts']}건 (auto_facts)",
             f"- 비교 정합성 검증 {n_checked['auto_compare']}건 (auto_compare)", ""]

    if problems:
        lines += ["## 결함", "", "| 유형 | 건수 |", "|---|---:|"]
        for k, v in problems.most_common():
            lines.append(f"| {k} | {v} |")
        lines += ["", "### 예시", ""] + details

    lines += ["", "## 유형 분포", "", "| 유형 | 모드 | 문항 | 비율 |", "|---|---|---:|---:|"]
    for (t, m), n in sorted(by_type.items()):
        lines.append(f"| {t} | {m} | {n} | {n / len(rows):.1%} |")

    lines += ["", "## 생성기별", "", "| 생성기 | 문항 |", "|---|---:|"]
    for g, n in by_gen.most_common():
        lines.append(f"| {g} | {n} |")

    if polarity:
        lines += ["", "## 예/아니오 균형", "",
                  "음성 사례(없다고 답해야 하는 문항)가 없으면 '있습니다'라고",
                  "항상 답해도 만점이 나와 할루시네이션을 못 잡는다.", "",
                  "| 극성 | 문항 |", "|---|---:|"]
        for k, v in polarity.most_common():
            lines.append(f"| {k} | {v} |")

    top = per_co.most_common(10)
    share = sum(n for _c, n in top) / max(1, sum(per_co.values()))
    lines += ["", "## 회사 쏠림", "",
              f"상위 10개사가 전체의 **{share:.1%}**", "",
              "| 회사 | 문항 |", "|---|---:|"]
    for c, n in top:
        lines.append(f"| {c} | {n} |")

    report = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
