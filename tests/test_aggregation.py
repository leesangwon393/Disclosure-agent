"""집계 질의 — "최대 계약금액은 얼마인가".

실측 실패(v2 38문항 S007~S014, 8문항 전부 오답)에서 두 가지 원인이 나왔다.

  1. Facts 가 **최신순** 10건만 줘서 최댓값이 잘려나감
     삼성바이오로직스 54건 중 최댓값 2,074,694,843,000
     -> 최신 10건의 최댓값 1,110,278,292,000 이 답으로 나갔다

  2. 최댓값이 목록에 **있었는데도** 모델이 못 골랐다
     한전기술 10건 중 7번째가 최댓값인데 5번째 값을 답했다

1은 값 기준 정렬로, 2는 파이썬이 계산해서 명시하는 것으로 막는다.
"""
from __future__ import annotations

import pytest

from disclosure_rag.agent.answer_generator import build_answer_prompt
from disclosure_rag.agent.evidence import build_evidence_pack_from_retrieval
from disclosure_rag.agent.query_plan import QueryPlan, detect_aggregation
from disclosure_rag.facts.multi_store import MultiFactStore
from disclosure_rag.facts.store import Fact, FactStore


# --------------------------------------------------------------------------- 판정

def test_detects_max_and_min():
    assert detect_aggregation("최대 계약금액은 얼마인가?") == "max"
    assert detect_aggregation("가장 큰 계약은?") == "max"
    assert detect_aggregation("최소 투자금액은?") == "min"
    assert detect_aggregation("가장 적은 금액은?") == "min"


def test_plain_lookup_is_not_an_aggregation():
    assert detect_aggregation("순자산액은 얼마인가?") == "none"


def test_count_wins_over_max():
    """'총 몇 건'은 집계지만 최댓값이 아니다."""
    assert detect_aggregation("계약이 총 몇 건인가?") == "count"


# --------------------------------------------------------------------------- 값 기준 정렬

def _fact(doc, value, date):
    return dict(doc_id=doc, chunk_id=f"{doc}::c1", company="삼성바이오로직스",
                corp_code="1", doc_group="exchange", doc_subtype=None,
                report_name="계약", filing_date=date, period=None,
                is_correction=False, is_latest=True, correction_group_id=doc,
                group_label=None, key="계약금액", key_norm="계약금액",
                value_text=f"{value:,}", value_num=float(value), value_unit=None,
                value_date=None, field_code=None, unit_code=None,
                unit_value=None, section_path=[])


@pytest.fixture
def store(tmp_path):
    st = FactStore(tmp_path / "f.sqlite")
    # 최댓값이 **가장 오래된** 공시에 있다 — 최신순으로 자르면 놓친다
    st.insert_many([Fact(**_fact("old", 2_074_694_843_000, "20200101"))]
                   + [Fact(**_fact(f"new{i}", 1_000_000_000 + i, f"2025010{i}"))
                      for i in range(1, 6)])
    return st


def test_date_order_misses_the_maximum(store):
    rows = store.lookup(key="계약금액", limit=3, order_by="date")
    assert max(r["value_num"] for r in rows) < 2_074_694_843_000


def test_value_order_puts_the_maximum_first(store):
    rows = store.lookup(key="계약금액", limit=3, order_by="value_desc")
    assert rows[0]["value_num"] == 2_074_694_843_000


def test_value_asc_puts_the_minimum_first(store):
    rows = store.lookup(key="계약금액", limit=3, order_by="value_asc")
    assert rows[0]["value_num"] == 1_000_000_001


def test_non_numeric_values_do_not_block_the_top(tmp_path):
    """값이 숫자가 아닌 행이 앞을 막으면 최댓값이 안 보인다."""
    st = FactStore(tmp_path / "g.sqlite")
    row = _fact("a", 0, "20240101")
    row.update(value_text="해당사항없음", value_num=None)
    st.insert_many([Fact(**row), Fact(**_fact("b", 500, "20240101"))])
    assert st.lookup(key="계약금액", order_by="value_desc")[0]["value_num"] == 500
    assert st.lookup(key="계약금액", order_by="value_asc")[0]["value_num"] == 500


def test_invalid_order_by_is_rejected(store):
    with pytest.raises(ValueError):
        store.lookup(key="계약금액", order_by="랜덤")


def test_multi_store_keeps_the_value_order(tmp_path):
    """저장소가 여러 개일 때 A의 10등이 B의 1등보다 앞에 오면 안 된다."""
    a = FactStore(tmp_path / "a.sqlite")
    a.insert_many([Fact(**_fact("a1", 100, "20250101"))])
    b = FactStore(tmp_path / "b.sqlite")
    b.insert_many([Fact(**_fact("b1", 999, "20200101"))])
    rows = MultiFactStore([a, b]).lookup(key="계약금액", order_by="value_desc")
    assert [r["value_num"] for r in rows] == [999.0, 100.0]


# --------------------------------------------------------------------------- 계산해서 명시

def _row(company, value, doc):
    return {"company": company, "item": "계약금액", "value": f"{value:,}",
            "value_num": float(value), "report_id": doc, "report_name": "계약",
            "filing_date": "20240101"}


def test_pack_marks_the_computed_maximum():
    """S011 실제 실패 — 모델이 목록 7번째의 최댓값 대신 5번째를 골랐다."""
    facts = [_row("한전기술", 373_449_426_066, "d1"),
             _row("한전기술", 1_250_850_298_678, "d2"),
             _row("한전기술", 32_323_000_000, "d3")]
    pack = build_evidence_pack_from_retrieval("질문", [], facts=facts, aggregation="max")
    text = pack.prompt_text
    assert "▶ 한전기술 계약금액 최대값: 1,250,850,298,678" in text
    assert "목록에서 직접 고르지 말고" in text


def test_maximum_is_computed_per_company():
    """비교 질문이라 회사별로 따로 계산해야 한다."""
    facts = [_row("두산에너빌리티", 4_928_968_079_043, "d1"),
             _row("두산에너빌리티", 100, "d2"),
             _row("한전기술", 1_250_850_298_678, "d3"),
             _row("한전기술", 50, "d4")]
    text = build_evidence_pack_from_retrieval("질문", [], facts=facts,
                                              aggregation="max").prompt_text
    assert "▶ 두산에너빌리티 계약금액 최대값: 4,928,968,079,043" in text
    assert "▶ 한전기술 계약금액 최대값: 1,250,850,298,678" in text


def test_minimum_is_marked_for_min_questions():
    facts = [_row("A", 100, "d1"), _row("A", 5, "d2")]
    text = build_evidence_pack_from_retrieval("질문", [], facts=facts,
                                              aggregation="min").prompt_text
    assert "최소값: 5" in text


def test_no_mark_for_plain_lookups():
    facts = [_row("A", 100, "d1"), _row("A", 5, "d2")]
    text = build_evidence_pack_from_retrieval("질문", [], facts=facts,
                                              aggregation="none").prompt_text
    assert "▶" not in text


def test_non_numeric_rows_are_skipped_when_computing():
    facts = [_row("A", 100, "d1"), {"company": "A", "item": "계약금액",
                                    "value": "해당사항없음", "value_num": None,
                                    "report_id": "d2"}]
    text = build_evidence_pack_from_retrieval("질문", [], facts=facts,
                                              aggregation="max").prompt_text
    assert "최대값: 100" in text


def test_prompt_tells_the_model_not_to_pick_by_itself():
    p = build_answer_prompt(QueryPlan(answer_mode="closed", task="compare",
                                      aggregation="max"))
    assert "▶ 표시가 이미 계산된 답" in p
    assert "직접 비교해서 고르지 마세요" in p


def test_prompt_has_no_aggregate_block_for_plain_lookups():
    p = build_answer_prompt(QueryPlan(answer_mode="closed", task="lookup"))
    assert "▶ 표시가 이미 계산된 답" not in p


# --------------------------------------------------------------------------- 프롬프트 크기

def test_aggregate_lists_few_but_computes_over_all():
    """집계 조회는 50건까지 가져온다. 그걸 다 프롬프트에 넣으면 비교 질문에서
    100줄이 되어 HCX 응답이 18초 -> 80~170초로 느려졌다(실측).
    계산은 전부로, 표시는 몇 줄만."""
    facts = [_row("A", 1_000 + i, f"d{i}") for i in range(50)]
    text = build_evidence_pack_from_retrieval("질문", [], facts=facts,
                                              aggregation="max").prompt_text
    assert "▶ A 계약금액 최대값: 1,049" in text      # 50건 전부에서 계산
    assert text.count("- 계약금액:") <= 8            # 표시는 8줄까지
    assert "생략" in text                            # 생략 사실을 밝힌다


def test_plain_lookup_lists_everything():
    """집계가 아니면 줄이지 않는다 — 나열 자체가 답인 경우가 있다."""
    facts = [_row("A", 1_000 + i, f"d{i}") for i in range(20)]
    text = build_evidence_pack_from_retrieval("질문", [], facts=facts,
                                              aggregation="none").prompt_text
    assert text.count("- 계약금액:") == 20
