#!/usr/bin/env python3
"""suite_v2 의 서술형 107문항에 `required_all` 을 채운다.

## 왜 필요한가

서술형은 정답이 문자열 하나가 아니라 `answer` 가 비어 있고, 채점기는 그런
문항을 '채점 불가'로 분류한다. 296문항 중 107문항(36%)이 그 상태여서
suite_v2 로는 사실상 189문항만 재고 있었다.

`required_all` = **답변에 반드시 등장해야 하는 값들**. 전부 나오면 정답,
아니면 몇 개 나왔는지를 `required_coverage` 로 본다.

## 값을 어디서 가져오나 — 유형마다 다르다

    correction  문항이 이미 갖고 있다. `after`(최종 정정본 값) + 바뀐 항목명.
                채점 기준이 "최신 정정본 수치를 답하는가" 이므로 정확히 이것이다.
    funding     수치사전에서 **자금 용도별 금액**만 뽑는다. "자금조달 내역"이
                묻는 게 그것이다.
    summary     `expected_fields_hint` 에 해당하는 값 + 계약금액류 화이트리스트.

## 아무 수치나 쓰면 안 된다

수치사전에는 `참석 = 5`(이사회 참석 인원), `1주당액면가액 = 5,000` 같은
값도 들어 있다. 이런 걸 필수 항목에 넣으면 정답을 오답으로 만든다.
그래서 **키 화이트리스트**로 거른다. 화이트리스트에 걸리는 값이 하나도
없으면 그 문항은 **채점 불가로 그대로 둔다** — 억지로 만들지 않는다.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

# 자금조달 질문이 묻는 것 = 용도별 조달 금액.
FUNDING_KEYS = {
    "시설자금", "운영자금", "채무상환자금", "타법인증권취득자금", "기타자금",
    "영업양수자금", "현물출자가액", "권면총액", "발행가액", "납입금액",
}

# 요약 질문에서 답에 꼭 나와야 할 만한 값.
SUMMARY_KEYS = {
    "계약금액", "확정계약금액", "계약금액총액", "매출액대비", "최근매출액",
    "투자금액", "자기자본", "자기자본대비", "해지금액", "취득금액", "처분금액",
}

# 넣으면 안 되는 것 — 답변에 나올 이유가 없거나 잡음이다.
BLOCKED_KEYS = {
    "참석", "불참", "1주당액면가액", "없음", "보통주식", "기타주식",
    "7-2.기준주가에대한할인율또는할증율", "최근일종가", "1주일평균",
    "1개월평균", "산술평균d=(a+b+c)/3",
}

MAX_TOKENS = 5           # 너무 많으면 정답도 통과 못 한다
MIN_DIGITS = 3           # 두 자리 이하 수는 우연히 맞을 수 있다
_DIGIT = re.compile(r"\d")


def _norm_key(s: str) -> str:
    return re.sub(r"\s", "", (s or "")).lower()


def _iso(d: str) -> str:
    d = (d or "").strip()
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 and d.isdigit() else ""


def _usable(value_text: str, value_num, value_date) -> str:
    """토큰으로 쓸 수 있는 값이면 그 표기를, 아니면 빈 문자열."""
    if value_date:
        return _iso(value_date)
    if value_num is None:
        return ""
    text = (value_text or "").strip()
    if not text or len(text) > 30:
        return ""
    if len(_DIGIT.findall(text)) < MIN_DIGITS and "." not in text:
        return ""
    return text


def collect(cur, doc_ids, allowed: set[str], hints: set[str]) -> list[str]:
    out: list[str] = []
    for doc_id in doc_ids:
        rows = cur.execute(
            "SELECT key_norm, value_text, value_num, value_date FROM facts WHERE doc_id=?",
            (doc_id,)).fetchall()
        for key, vt, vn, vd in rows:
            k = _norm_key(key)
            if k in {_norm_key(b) for b in BLOCKED_KEYS}:
                continue
            if k not in {_norm_key(a) for a in allowed} and k not in {_norm_key(h) for h in hints}:
                continue
            token = _usable(vt, vn, vd)
            if token and token not in out:
                out.append(token)
    return out


def build(row: dict, cur) -> list[str]:
    gen = row.get("generator")
    ids = row.get("gold_doc_ids") or []
    hints = set(row.get("expected_fields_hint") or [])

    if gen == "correction":
        after = row.get("after") or {}
        tokens = [str(v).strip() for v in after.values() if str(v).strip()]
        tokens += [str(f).strip() for f in (row.get("changed_fields") or []) if str(f).strip()]
        return list(dict.fromkeys(tokens))[:MAX_TOKENS]
    if gen == "funding":
        return collect(cur, ids, FUNDING_KEYS, set())[:MAX_TOKENS]
    if gen == "summary":
        return collect(cur, ids, SUMMARY_KEYS, hints)[:MAX_TOKENS]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="eval/suite_v2.jsonl")
    ap.add_argument("--facts", default="artifacts_v2/facts/facts.sqlite")
    ap.add_argument("--periodic-facts", default="artifacts_v2/facts_periodic_v2/facts.sqlite")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.suite)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    con = sqlite3.connect(args.facts)
    cur = con.cursor()
    try:
        cur.execute(f"ATTACH DATABASE '{args.periodic_facts}' AS pf")
    except sqlite3.Error:
        pass

    stat = Counter()
    filled = 0
    for row in rows:
        if row.get("answer_source") != "rubric_only":
            continue
        if row.get("required_all"):
            stat[(row["generator"], "이미있음")] += 1
            continue
        tokens = build(row, cur)
        if tokens:
            row["required_all"] = tokens
            row["answer_source"] = "rubric_auto_20260830"
            filled += 1
            stat[(row["generator"], f"{len(tokens)}개")] += 1
        else:
            stat[(row["generator"], "만들지못함")] += 1

    for k, v in sorted(stat.items()):
        print(f"  {k[0]:12} {k[1]:8} {v:3}")
    print(f"\n채운 문항: {filled}")

    gradeable = sum(1 for r in rows
                    if (r.get("answer") or "").strip() or r.get("required_all"))
    print(f"채점 가능: {gradeable}/{len(rows)}")

    if args.dry_run:
        print("\n(--dry-run: 파일을 쓰지 않았다)")
        return 0
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    print(f"저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
