#!/usr/bin/env python3
"""조각(passage) 수준 정답 세트 자동 생성.

왜 필요한가
-----------
기존 `eval/gold_queries.json` 의 정답은 **문서 수준**(`gold_report_ids`)이다.

    {"query": "삼성전자 2025년 사업보고서에서 매출액 알려줘",
     "gold_report_ids": ["periodic_20260310002820"]}

이러면 "그 문서에서 나온 조각이면 아무거나 정답"이 된다. **청크 크기를 바꿔도
점수가 거의 안 움직인다** — 크게 자르든 작게 자르든 같은 문서에서 나오니까.
Stage 1 에서 후보 간 차이가 0.05 밖에 안 났던 것도 실험 설계가 아니라
**정답 라벨의 해상도** 문제일 가능성이 크다.

어떻게 만드나
-------------
공시 질문의 답은 원문에 있는 **특정 문자열**이다. 그래서 사람이 라벨링할 필요가 없다.

    질문: "삼성전자 반도체 위탁생산 계약금액이 얼마야?"
    정답: "22,764,764,160,000"          <- 조각에 이 문자열이 있으면 정답
    출처: exchange_20250728800035

폼 문서(exchange 1,469 / holding 1,083 / major 598 = 전체의 75%)는 서식이 고정이라
`KeyValueNode` 의 (key, value) 가 그대로 (질문 대상, 정답) 이 된다.
DART 가 붙여준 `ACODE` 가 있는 항목을 우선 고른다 — 기계가 "이건 구조화 필드"라고
표시해 준 것이므로 질문거리로 안전하다.

**외부 LLM 을 쓰지 않는다.** 질문은 서식 항목명으로 템플릿 생성하므로,
"Synthetic Data 생성에 외부 LLM 허용 여부" 미해결 이슈에 걸리지 않는다.

한계 (정직하게)
---------------
- 템플릿 질문이라 실제 사용자 어투보다 문어체다. **검색기 비교용**으로는 충분하지만
  최종 성능 수치로 인용하면 안 된다.
- 정답 문자열이 코퍼스에 여러 번 나오면 신호가 흐려진다 -> `--max-doc-freq` 로 거른다.
- 폼 문서 기반이라 periodic(서술형 장문서) 질의는 적게 나온다. 보완하려면
  사람이 쓴 질의를 섞을 것.

사용:
  python3 scripts/make_gold_passages.py \
      --corpus-root ~/Desktop/미래에셋/데이터/corpus \
      --out eval/gold_passages.jsonl --n-companies 10 --per-company 30
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from disclosure_rag.correction.correction_graph_builder import build_correction_index  # noqa: E402
from disclosure_rag.facts.extractor import extract_facts  # noqa: E402
from disclosure_rag.common.manifest_loader import load_manifest  # noqa: E402
from disclosure_rag.common.unicode_utils import PathResolver  # noqa: E402
from disclosure_rag.parsing.document_detector import parse_documents_for_row  # noqa: E402

# 질문 템플릿 — 항목명(key)을 그대로 쓰되 어투만 몇 가지로 돌린다.
_TEMPLATES = [
    "{company} {report} {key} 얼마야?",
    "{company}의 {report}에서 {key} 알려줘",
    "{company} {key}이(가) 어떻게 돼?",
]

# 질문거리가 되는 항목명인지 — 한글이 최소 2자는 있어야 사람이 물어볼 수 있는 말이 된다.
# 회귀(실측): 이 게이트가 없어서 재무제표의 (당기값, 전기값) 쌍이 (항목, 값)으로 잡혀
#   "LG씨엔에스 50,513,660 얼마야?" -> "47,197,274"
#   "삼성중공업 (1,401,120)이(가) 어떻게 돼?" -> "(1,401,120)"   (질문=답)
# 같은 쓰레기 질의가 3,356개 생성됐다. 재무제표는 다열 표라 KeyValueNode 분류가
# key-value 로 성립하지 않는 경우가 많다.
_HANGUL2 = re.compile(r"[가-힣].*[가-힣]", re.S)
# 표의 각주·기호 항목("(*)", "(주1)", "(*2)")도 질문이 안 된다.
_FOOTNOTE_KEY = re.compile(r"^[(（]?\s*[*주※]\s*\d*\s*[)）]?$")


def _iter_nodes(sec: SectionNode):
    for c in sec.children:
        if isinstance(c, SectionNode):
            yield from _iter_nodes(c)
        else:
            yield c


def _is_good_answer(value: str) -> bool:
    v = value.strip()
    if not v or v in _BAD_VALUES or len(v) < 7:
        return False
    if _NUMERIC.match(v):
        return True
    # 숫자를 포함한 충분히 긴 문자열 (예: "2026년 3분기 ~ 2029년 1분기")
    return len(v) >= 10 and bool(_HAS_DIGITS.search(v))


def _clean_key(key: str) -> str:
    k = re.sub(r"^\s*[0-9가-힣]{1,3}[.)]\s*", "", key).strip()   # "가. ", "1) " 제거
    k = re.sub(r"\s+", " ", k)
    return k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--out", default="eval/gold_passages.jsonl")
    ap.add_argument("--groups", default="exchange,major,holding")
    ap.add_argument("--n-companies", type=int, default=10, help="0=전체")
    ap.add_argument("--per-company", type=int, default=30, help="회사당 질의 상한")
    ap.add_argument("--max-doc-freq", type=int, default=3,
                    help="정답 문자열이 이 개수 넘는 문서에 나오면 버린다(신호 희석 방지)")
    ap.add_argument("--max-answers", type=int, default=4,
                    help="한 (회사,항목) 이 이보다 많은 값을 가지면 질문이 모호한 것으로 보고 버린다")
    ap.add_argument("--seed", type=int, default=20260823)
    args = ap.parse_args()

    corpus = Path(args.corpus_root).expanduser()
    manifest = load_manifest(str(corpus))
    resolver = PathResolver(str(corpus))
    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    rnd = random.Random(args.seed)

    companies = sorted({r.corp_name for r in manifest})
    if args.n_companies:
        companies = rnd.sample(companies, min(args.n_companies, len(companies)))
    target = set(companies)
    rows = [r for r in manifest if r.corp_name in target and r.doc_group in groups]
    rnd.shuffle(rows)
    print(f"대상 회사 {len(target)}개 / 문서 {len(rows)}건", file=sys.stderr)

    corrections = build_correction_index(manifest, resolver)
    cand: list[dict] = []
    value_docs: dict[str, set[str]] = defaultdict(set)
    per_company: Counter = Counter()
    dropped = Counter()

    for row in rows:
        if per_company[row.corp_name] >= args.per_company:
            continue
        corr = corrections.get(row.doc_id)
        if corr is None:
            continue
        try:
            docs = parse_documents_for_row(row, resolver)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {row.doc_id}: {type(e).__name__}", file=sys.stderr)
            continue
        for parsed in docs:
            if parsed.report_subtype == "unsupported_pdf_html":
                continue
            # facts 추출기와 **같은 경로**를 쓴다 — 필터를 두 곳에서 관리하지 않는다.
            for fact in extract_facts(parsed, row, corr):
                key = fact.key_norm
                # ① 정답은 숫자 또는 날짜여야 한다. 서술형 답은 채점 기준이 모호하다.
                if fact.value_num is None and fact.value_date is None:
                    dropped["답이 숫자·날짜가 아님"] += 1
                    continue
                # ② 항목명에 한글이 최소 2자 (숫자 항목명·각주 기호 제거)
                if not _HANGUL2.search(key) or _FOOTNOTE_KEY.match(key):
                    dropped["항목명이 질문이 안 됨"] += 1
                    continue
                # ③ 질문과 답이 같으면 안 된다
                if fact.value_text.strip() == fact.key.strip():
                    dropped["질문=답"] += 1
                    continue
                # ④ 답이 너무 짧으면 다른 조각에도 우연히 들어 있다
                if len(fact.value_text.strip()) < 7:
                    dropped["답이 너무 짧음"] += 1
                    continue
                value_docs[fact.value_text.strip()].add(row.doc_id)
                cand.append({
                    "company": row.corp_name, "report": row.report_nm or "",
                    "doc_group": row.doc_group, "report_id": row.doc_id,
                    "key": key, "answer": fact.value_text.strip(),
                    "field_code": fact.field_code, "unit": fact.unit_code,
                    "unit_value": fact.unit_value,
                    "priority": 0 if fact.field_code else 1,
                })
                per_company[row.corp_name] += 1
                if per_company[row.corp_name] >= args.per_company:
                    break

    # ⑤ 정답 문자열이 여러 문서에 흔하면 containment 신호가 흐려진다 -> 제거
    kept = [c for c in cand if len(value_docs[c["answer"]]) <= args.max_doc_freq]
    dropped["답이 여러 문서에 흔함"] = len(cand) - len(kept)

    # ⑥ 같은 (회사, 항목) 이 여러 값을 가지는 경우를 처리한다.
    #    실측: 재무제표는 한 문서 안에 "자산총계"가 연결/별도 x 당기/전기로 여러 번 나온다.
    #      Q: 자본총계 얼마야?  A: 690,327,520
    #      Q: 자본총계 어떻게 돼? A: 540,887,863   <- 같은 질문, 다른 답
    #    질문만으로는 어느 것인지 알 수 없으므로 **답을 복수 허용**한다. 검색 평가에서
    #    필요한 것은 "유효한 답이 든 조각을 찾았는가" 이므로 이게 정직한 정의다.
    #    단 값이 너무 많으면(>N) 그 항목은 애초에 질문이 모호한 것이라 버린다.
    grouped: dict[tuple, dict] = {}
    for c in kept:
        sig = (c["company"], c["key"])
        g = grouped.setdefault(sig, {**c, "answers": [], "report_ids": []})
        if c["answer"] not in g["answers"]:
            g["answers"].append(c["answer"])
        if c["report_id"] not in g["report_ids"]:
            g["report_ids"].append(c["report_id"])
        g["priority"] = min(g["priority"], c["priority"])

    groups = [g for g in grouped.values() if len(g["answers"]) <= args.max_answers]
    dropped["항목이 모호함(값 과다)"] = len(grouped) - len(groups)
    groups.sort(key=lambda g: (g["priority"], -len(g["answers"])))

    out_rows = []
    for i, g in enumerate(groups):
        tpl = _TEMPLATES[i % len(_TEMPLATES)]
        out_rows.append({
            "id": i + 1,
            "query": tpl.format(company=g["company"], report=g["report"], key=g["key"]),
            "answers": g["answers"],              # ← 조각 수준 채점 (하나라도 포함되면 정답)
            "answer": g["answers"][0],            # 하위호환
            "gold_report_ids": g["report_ids"],   # ← 문서 수준(비교용)
            "company": g["company"], "doc_group": g["doc_group"],
            "key": g["key"], "field_code": g["field_code"],
            "unit": g["unit"], "unit_value": g["unit_value"],
            "n_answers": len(g["answers"]),
        })

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_group = Counter(r["doc_group"] for r in out_rows)
    with_code = sum(1 for r in out_rows if r["field_code"])
    multi = sum(1 for r in out_rows if r["n_answers"] > 1)
    print(json.dumps({
        "생성": len(out_rows), "후보": len(cand), "복수정답_질의": multi,
        "회사수": len({r["company"] for r in out_rows}),
        "그룹별": dict(by_group), "ACODE_보유": with_code, "출력": str(out),
        "탈락사유": dict(dropped),
    }, ensure_ascii=False, indent=2))
    print("\n샘플 10개:", file=sys.stderr)
    for r in out_rows[:10]:
        print(f"  Q: {r['query'][:60]:62s} A: {r['answer'][:22]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
