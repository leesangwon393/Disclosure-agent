#!/usr/bin/env python3
"""정답셋 청소 — 답할 수 없는 질문과 잘못된 정답을 걸러낸다.

## 왜 필요한가

`scripts/score_answers.py --mode retrieval` 로 314문항을 재보니(2026-08-30)
상한도달 72.0% 였는데, 실패를 들여다보니 **상당수가 우리 시스템의 문제가
아니라 정답셋의 결함**이었다.

### 결함 1 — 답할 수 없는 질문

`gold_periodic`/`gold_form` 은 표의 행 라벨을 그대로 질문으로 만들었다.

    "셀트리온 자본이(가) 어떻게 돼?"
    "SK하이닉스 합계이(가) 어떻게 돼?"
    "시프트업 기말이(가) 어떻게 돼?"

무엇의 합계인지 질문에 없어서 **사람도 답할 수 없다.** 실측으로도
질문이 공시를 지정하면 76.5%, 지정하지 않으면 62.4% 로 갈렸고,
key 라벨이 gold 문서 안에 10회 이상 등장하면 50.6% 까지 떨어졌다.

### 결함 2 — 다른 법인의 표에서 뽑힌 정답

삼성중공업 반기보고서 질문의 정답이 **삼성전자 수치**였다.

    법인 또는 단체의 명칭  삼성전자주식회사   자산총계 448,424,507 ...

타법인 출자현황 표의 행을 그 보고서 제출회사의 값으로 붙인 것이다.
우리 시스템이 삼성중공업의 진짜 자산총계를 찾아오면 **오답으로 채점된다.**

## 판정 규칙

    AMBIGUOUS   질문이 공시/기간을 지정하지 않고, key 라벨이 gold 문서 안에
                ambiguity_threshold 회 이상 등장한다 -> 무엇을 묻는지 정할 수 없다
    CROSS_ENTITY 정답 문자열 바로 앞 문맥에 질문 회사가 아닌 법인명이 있다

둘 다 **원본을 지우지 않는다.** clean 파일을 따로 쓰고, 제외한 문항은
사유·근거와 함께 리포트에 전부 남긴다. 규칙이 과했는지 사람이 볼 수 있어야
한다 — 정답셋을 조용히 줄이면 점수만 올라간다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# 질문이 어떤 공시/시점을 보는지 지정하는 표현
QUALIFIER = re.compile(r"보고서|공시|계약체결|취득결정|발행결정|신규시설투자|투자판단|대량보유|\(20\d{2}|20\d{2}년")
# 다른 법인의 수치임을 드러내는 표 헤더
FOREIGN_ENTITY = re.compile(r"법인 또는 단체의 명칭\s+(\S[^\n]{1,28}?)\s+(?:자산총계|부채총계|자본총계|매출액|당기순|구 분)")

# threshold 민감도 실측(2026-08-30, 314문항). 유지 집단의 상한도달률이
# 어느 값에서도 75~76% 로 거의 같다 — 즉 **임계값을 조정해 점수를 만들 수
# 없다.** 그래서 가장 보수적인(적게 지우는) 쪽을 고른다.
#
#   threshold  제외   제외집단 상한률   유지 n   유지 상한률
#           2    56          57.1%      258        75.2%
#           3    52          53.8%      262        75.6%
#           5    37          40.5%      277        76.2%
#          10    28          32.1%      286        75.9%   <- 채택
#          20    21          14.3%      293        76.1%
#
# 10 에서 걸리는 건 "합계"(381회) "자본"(114회) "기말"(149회) 처럼 누가 봐도
# 답을 특정할 수 없는 것들이고, "자기자본"(3회) 같은 애매한 건은 남는다.
# 제외 집단의 상한률이 32.1% 로 0% 가 아니라는 점이 중요하다 — 실패한 문항만
# 골라 지운 게 아니라는 뜻이다.
DEFAULT_AMBIGUITY_THRESHOLD = 10
CONTEXT_WINDOW = 1200


def _plain_text(path: Path) -> str:
    text = ""
    for name in sorted(os.listdir(path)):
        if name.endswith(".xml"):
            text += (path / name).read_text(encoding="utf-8", errors="replace")
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))


def judge(row: dict, doc_text: str, *, threshold: int) -> tuple[str, str] | None:
    """제외 사유와 근거를 돌려준다. 문제없으면 None."""
    answers = [a for a in (row.get("answers") or []) if a] or [row.get("answer")]
    answer = answers[0] if answers else None

    # 결함 2 — 다른 법인의 표
    if answer and answer in doc_text:
        i = doc_text.index(answer)
        context = doc_text[max(0, i - CONTEXT_WINDOW):i]
        last = None
        for m in FOREIGN_ENTITY.finditer(context):
            last = m
        if last:
            other = last.group(1)
            if row["company"].replace(" ", "") not in other.replace(" ", ""):
                return "CROSS_ENTITY", f"정답 앞 문맥의 표 주체가 '{other}' 다 (질문 회사: {row['company']})"

    # 결함 1 — 답할 수 없는 질문
    if not QUALIFIER.search(row["query"]):
        occurrences = doc_text.count(row["key"])
        if occurrences >= threshold:
            return "AMBIGUOUS", f"질문이 공시를 지정하지 않는데 '{row['key']}' 라벨이 gold 문서에 {occurrences}회 등장한다"

    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/gold_passages.jsonl")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default="eval/gold_clean_report.md")
    ap.add_argument("--threshold", type=int, default=DEFAULT_AMBIGUITY_THRESHOLD)
    args = ap.parse_args()

    from disclosure_rag.common.manifest_loader import load_manifest
    from disclosure_rag.common.unicode_utils import PathResolver

    manifest = {d.doc_id: d for d in load_manifest(args.corpus)}
    resolver = PathResolver(args.corpus)

    gold_path = Path(args.gold)
    out_path = Path(args.out or gold_path.with_name(gold_path.stem + "_clean.jsonl"))
    rows = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    cache: dict[str, str] = {}
    kept, dropped = [], []
    for row in rows:
        gid = (row.get("gold_report_ids") or [None])[0]
        if not gid or gid not in manifest:
            kept.append(row)
            continue
        if gid not in cache:
            cache[gid] = _plain_text(Path(resolver.resolve(manifest[gid].file_path)))
        verdict = judge(row, cache[gid], threshold=args.threshold)
        if verdict is None:
            kept.append(row)
        else:
            dropped.append((row, *verdict))

    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")

    reasons: dict[str, int] = {}
    for _, reason, _ in dropped:
        reasons[reason] = reasons.get(reason, 0) + 1
    lines = [
        f"# 정답셋 청소 리포트 — `{gold_path.name}`", "",
        f"- 원본 {len(rows)}문항 -> 유지 **{len(kept)}** / 제외 **{len(dropped)}** ({len(dropped)/max(len(rows),1):.1%})",
        f"- ambiguity threshold = {args.threshold}", "",
        "| 사유 | 건수 |", "|---|---:|",
    ]
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {reason} | {count} |")
    lines += ["", "## 제외한 문항 전체", "",
              "| 사유 | 회사 | key | 질문 | 근거 |", "|---|---|---|---|---|"]
    for row, reason, why in dropped:
        q = row["query"].replace("|", "/")
        lines.append(f"| {reason} | {row['company']} | {row['key']} | {q} | {why} |")
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"유지 {len(kept)} / 제외 {len(dropped)}  ->  {out_path}")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"   {reason}: {count}")
    print(f"리포트: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
