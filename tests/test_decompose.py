"""Stage 4 질문 분해.

실패 방향이 둘이다.
  - 안 쪼개면: 두 회사 비교에서 한쪽 근거만 모아 반쪽 답변
  - 잘못 쪼개면: 검색 횟수만 늘고 각 하위 질의의 몫이 줄어 정답이 밀린다
그래서 (1) 언제 쪼개나 (2) 무슨 축으로 쪼개나 (3) 어떻게 합치나 를 각각 본다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

from disclosure_rag.agent.decompose import (  # noqa: E402
    build_sub_queries,
    decompose_and_search,
    merge_results,
)
from disclosure_rag.agent.query_plan import QueryPlan  # noqa: E402


def chunk(cid, rid="r"):
    return SimpleNamespace(chunk_id=cid, report_id=rid)


# --------------------------------------------------------------------------- 언제 쪼개나

def test_single_target_is_not_decomposed():
    plan = QueryPlan(answer_mode="closed", task="lookup", companies=["삼성전자"])
    subs = build_sub_queries(plan, "삼성전자 순자산액은?")
    assert len(subs) == 1 and subs[0].kind == "base"


def test_closed_two_company_question_is_decomposed():
    """S007~S014 — answer_mode 가 closed 인데도 쪼개야 한다."""
    plan = QueryPlan(answer_mode="closed", task="compare",
                     companies=["삼성전자", "한미반도체"],
                     needs_multiple_documents=True, operations=["조회:A", "조회:B", "비교"])
    subs = build_sub_queries(plan, "삼성전자와 한미반도체 중 최대 계약금액은?")
    assert [s.kind for s in subs] == ["company", "company"]
    assert [s.plan.companies for s in subs] == [["삼성전자"], ["한미반도체"]]


def test_two_report_kinds_are_decomposed():
    """S015~S022 — 회사는 하나인데 체결·해지 공시를 둘 다 봐야 한다."""
    plan = QueryPlan(answer_mode="closed", task="lookup", companies=["한전기술"],
                     report_kinds=["단일판매공급계약체결", "단일판매공급계약해지"],
                     needs_multiple_documents=True)
    subs = build_sub_queries(plan, "한전기술이 체결한 계약 중 해지된 것이 있는가?")
    assert [s.kind for s in subs] == ["report_kind", "report_kind"]
    assert [s.plan.report_kinds for s in subs] == [["단일판매공급계약체결"],
                                                   ["단일판매공급계약해지"]]


def test_two_periods_are_decomposed():
    plan = QueryPlan(answer_mode="open", task="compare", companies=["삼성전자"],
                     periods=["2024", "2026"], needs_multiple_documents=True)
    subs = build_sub_queries(plan, "2024년과 2026년 사업보고서를 비교해줘")
    assert [s.plan.periods for s in subs] == [["2024"], ["2026"]]


# --------------------------------------------------------------------------- 무슨 축으로

def test_only_one_axis_is_split():
    """회사 2 × 유형 2 를 곱해 4번 검색하지 않는다 — 검색 횟수만 늘어난다."""
    plan = QueryPlan(companies=["A", "B"], report_kinds=["체결", "해지"],
                     periods=["2023", "2024"], needs_multiple_documents=True)
    subs = build_sub_queries(plan, "질문")
    assert len(subs) == 2
    assert all(s.kind == "company" for s in subs)      # 회사가 최우선


def test_correction_diff_is_not_split_but_gets_more_budget():
    """정정 diff 는 축이 하나다. 버전 정리(Stage 7)가 최초·최종을 남기므로
    여기서는 검색량만 키운다."""
    plan = QueryPlan(answer_mode="mixed", task="correction_diff",
                     companies=["현대건설"], latest_policy="first_and_final",
                     needs_multiple_documents=True)
    subs = build_sub_queries(plan, "최초 공시와 최종 정정본의 차이는?")
    assert len(subs) == 1
    assert subs[0].top_k == 24


def test_sub_query_keeps_the_original_question():
    """원 질문을 버리고 '삼성전자 계약금액' 같은 짧은 문장을 만들면
    '최대'인지 '최초'인지가 사라진다."""
    plan = QueryPlan(companies=["삼성전자", "한미반도체"], needs_multiple_documents=True)
    subs = build_sub_queries(plan, "각각 공시한 계약 중 최대 계약금액은?")
    assert "최대 계약금액" in subs[0].text
    assert "삼성전자" in subs[0].text


def test_sub_plans_do_not_recurse():
    plan = QueryPlan(companies=["A", "B"], needs_multiple_documents=True,
                     operations=["조회:A", "조회:B"])
    for s in build_sub_queries(plan, "질문"):
        assert not s.plan.needs_decomposition


def test_original_plan_is_not_mutated():
    plan = QueryPlan(companies=["A", "B"], needs_multiple_documents=True)
    build_sub_queries(plan, "질문")
    assert plan.companies == ["A", "B"]
    assert plan.needs_multiple_documents is True


def test_budget_is_divided_across_targets():
    """하위 질의마다 최소 8칸은 준다 — 나눠서 2칸씩이면 아무것도 못 찾는다."""
    plan = QueryPlan(companies=["A", "B"], needs_multiple_documents=True)
    subs = build_sub_queries(plan, "질문")
    assert all(s.top_k >= 8 for s in subs)


# --------------------------------------------------------------------------- 병합

def test_merge_is_round_robin_not_by_score():
    """점수로 줄 세우면 한쪽 회사 근거가 통째로 밀려 비교가 불가능해진다."""
    a = SimpleNamespace(text="A"), 0.99
    a2 = SimpleNamespace(text="A2"), 0.98
    b = SimpleNamespace(text="B"), 0.10
    sq_a = build_sub_queries(QueryPlan(companies=["A"]), "q")[0]
    sq_b = build_sub_queries(QueryPlan(companies=["B"]), "q")[0]
    sq_b.label = "company:B"
    merged, counts, empty = merge_results([(sq_a, [a, a2]), (sq_b, [b])])
    assert [c.text for c, _s in merged] == ["A", "B", "A2"]   # 번갈아
    assert empty == []


def test_merge_deduplicates_same_chunk():
    c = chunk("same")
    sq1 = build_sub_queries(QueryPlan(companies=["A"]), "q")[0]
    sq2 = build_sub_queries(QueryPlan(companies=["B"]), "q")[0]
    sq2.label = "b"
    merged, _c, _e = merge_results([(sq1, [(c, 0.9)]), (sq2, [(c, 0.8)])])
    assert len(merged) == 1


def test_empty_sub_query_is_recorded():
    """'두 회사 비교인데 한 회사 근거만 있음'을 Stage 11 이 알아야 한다.
    근거 건수만 보면 통과해버린다."""
    sq1 = build_sub_queries(QueryPlan(companies=["A"]), "q")[0]
    sq2 = build_sub_queries(QueryPlan(companies=["B"]), "q")[0]
    sq2.label = "company:B"
    _m, counts, empty = merge_results([(sq1, [(chunk("x"), 0.9)]), (sq2, [])])
    assert empty == ["company:B"]
    assert counts["company:B"] == 0


# --------------------------------------------------------------------------- 통합

def test_decompose_and_search_calls_once_per_sub_query():
    calls = []

    def fake(text, plan, k):
        calls.append((plan.companies, k))
        return [(chunk(f"c{len(calls)}"), 0.5)]

    plan = QueryPlan(companies=["A", "B"], needs_multiple_documents=True)
    res = decompose_and_search(plan, "질문", fake)
    assert len(calls) == 2
    assert res.decomposed and len(res.merged) == 2
    assert not res.partial


def test_one_failing_sub_query_does_not_kill_the_rest():
    def flaky(text, plan, k):
        if plan.companies == ["B"]:
            raise RuntimeError("검색 실패")
        return [(chunk("ok"), 0.5)]

    plan = QueryPlan(companies=["A", "B"], needs_multiple_documents=True)
    res = decompose_and_search(plan, "질문", flaky)
    assert len(res.merged) == 1
    assert res.partial and res.empty_labels == ["company:B"]


def test_undecomposed_question_searches_once():
    calls = []

    def fake(text, plan, k):
        calls.append(text)
        return []

    plan = QueryPlan(answer_mode="closed", task="lookup", companies=["A"])
    res = decompose_and_search(plan, "원 질문", fake)
    assert calls == ["원 질문"]
    assert not res.decomposed


# --------------------------------------------------------------------------- 정답셋 회귀

SUITE = ROOT / "eval" / "suite_v1.jsonl"
SCHEMA = ROOT / "config" / "field_schema.json"
CORPUS = ROOT / "corpus"


@pytest.mark.skipif(not (SUITE.exists() and SCHEMA.exists() and CORPUS.exists()),
                    reason="정답셋/명세/코퍼스 미구비")
def test_gold_suite_decomposition_counts():
    from disclosure_rag.agent.field_schema import FieldSchema
    from disclosure_rag.agent.query_plan import RulePlanBuilder
    from disclosure_rag.entity.entity_extractor import EntityExtractor

    builder = RulePlanBuilder(
        schema=FieldSchema.load(SCHEMA),
        extractor=EntityExtractor(corpus_root=str(CORPUS),
                                  metric_terms_path=str(ROOT / "config" / "metric_terms.txt")),
    )
    rows = [json.loads(l) for l in SUITE.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {}
    for r in rows:
        plan = builder.build(r["query"])
        by_id[r["id"]] = build_sub_queries(plan, r["query"])

    # 두 회사 비교 -> 회사 축으로 2개
    for i in range(7, 15):
        subs = by_id[f"S{i:03d}"]
        assert len(subs) == 2 and all(s.kind == "company" for s in subs), f"S{i:03d}"

    # 체결/해지 -> 공시유형 축으로 2개
    for i in range(15, 23):
        subs = by_id[f"S{i:03d}"]
        assert len(subs) == 2 and all(s.kind == "report_kind" for s in subs), f"S{i:03d}"

    # 단일 조회 -> 쪼개지 않는다
    for i in range(1, 7):
        assert len(by_id[f"S{i:03d}"]) == 1, f"S{i:03d}"

    # S025·S026 은 "주요사항보고서"만 말해 19종에 걸린다 -> 쪼개지 않는다
    assert len(by_id["S025"]) == 1 and len(by_id["S026"]) == 1

    total = sum(len(v) for v in by_id.values())
    assert total == 57, total          # 19문항이 2개로 늘어난다 (38 + 19)


# --------------------------------------------------------------------------- 분해 상한

def test_too_many_axes_falls_back_to_a_single_wide_search():
    """S025 형태 — 질문이 대상을 특정하지 못해 19종에 걸린다.
    그대로 쪼개면 검색을 19번 하고 각자 8칸씩만 받아 정답이 밀린다."""
    from disclosure_rag.agent.decompose import MAX_SUB_QUERIES
    plan = QueryPlan(answer_mode="mixed", task="correction_diff", companies=["두산로보틱스"],
                     report_kinds=[f"주요사항보고서(유형{i})" for i in range(19)],
                     needs_multiple_documents=True)
    subs = build_sub_queries(plan, "주요사항보고서가 정정된 내역이 있는가?")
    assert len(subs) == 1
    assert subs[0].kind == "base"
    assert subs[0].top_k >= 24          # 대신 넓게 찾는다
    assert MAX_SUB_QUERIES == 3


def test_at_the_cap_it_still_splits():
    plan = QueryPlan(companies=["A", "B", "C"], needs_multiple_documents=True)
    assert len(build_sub_queries(plan, "질문")) == 3


def test_just_over_the_cap_does_not_split():
    plan = QueryPlan(companies=["A", "B", "C", "D"], needs_multiple_documents=True)
    subs = build_sub_queries(plan, "질문")
    assert len(subs) == 1 and subs[0].kind == "base"
