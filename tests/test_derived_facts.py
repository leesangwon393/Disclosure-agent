"""파생 계산은 파이썬이 한다 — 모델에게 산수를 시키지 않는다.

실측(2026-09-01, suite_v2): 계산이 붙는 최대/최소 문항은 96%, 계산이 안 붙는
합계 50% · 비율 40% · 평균 0%. 그래서 합계·평균·차이·건수·순위·증감률·
CAGR·비율을 모두 파이썬이 먼저 계산해 `▶▶` 로 못 박는다.
"""

from datetime import date

import pytest

from disclosure_rag.agent.derived_facts import (
    days_between,
    derive,
    derive_calculations,
    describe_span,
    months_between,
    parse_dates,
    wanted_operations,
)


def row(item, value, *, filing="20260315", company="A", period="2025", unit="백만원"):
    return {"item": item, "value_num": value, "filing_date": filing,
            "company": company, "period": period, "value_unit": unit}


ROWS = [
    row("매출액", 1000.0, filing="20230315", period="2022"),
    row("매출액", 1500.0),
    row("영업이익", 150.0),
]


# --- 무엇을 계산해 달라는 질문인가 -------------------------------------------

@pytest.mark.parametrize("query,expected", [
    ("매출액 합계는 얼마인가", ["sum"]),
    ("평균 급여는", ["avg"]),
    ("두 회사의 차이는", ["diff"]),
    ("계약이 몇 건인가", ["count"]),
    ("큰 순서대로 나열해줘", ["rank"]),
    ("매출액은 얼마인가", []),
])
def test_reads_which_operation_the_question_asks_for(query, expected):
    assert wanted_operations(query) == expected


def test_no_operation_means_no_lines():
    assert derive("매출액은 얼마인가", ROWS) == []


def test_no_numbers_means_no_lines():
    assert derive("매출액 합계는", [{"item": "매출액", "value_num": None}]) == []


# --- 합계·평균·차이·건수·순위 ------------------------------------------------

def test_sum_only_adds_the_same_item():
    lines = derive("매출액 합계는", ROWS)
    assert any("매출액 합계: 2,500" in ln for ln in lines)
    # 영업이익은 값이 하나뿐이라 합계를 내지 않는다 — 반쪽짜리 합계는 안 만든다.
    assert not any("영업이익 합계" in ln for ln in lines)


def test_average_is_computed_not_guessed():
    lines = derive("평균은", ROWS)
    assert any("매출액 평균: 1,250.00" in ln for ln in lines)


def test_difference_uses_max_minus_min():
    lines = derive("차이는 얼마인가", ROWS)
    assert any("최대-최소 차이: 500" in ln for ln in lines)


def test_count_reports_every_item():
    lines = derive("몇 건인가", ROWS)
    assert any("매출액 건수: 2건" in ln for ln in lines)
    assert any("영업이익 건수: 1건" in ln for ln in lines)


def test_rank_is_descending():
    rows = [row("매출액", 300.0, company="가"), row("매출액", 900.0, company="나")]
    lines = derive("큰 순서대로", rows)
    assert lines and lines[0].index("나") < lines[0].index("가")


def test_amount_is_spelled_out_in_korean_units():
    lines = derive("매출액 합계는", ROWS)
    assert any("25억원" in ln for ln in lines)


def test_small_won_amounts_get_no_conversion():
    rows = [row("주식수", 12.0, unit="원"), row("주식수", 8.0, unit="원")]
    lines = derive("합계는", rows)
    assert lines == ["▶▶ 주식수 합계: 20 (2건)"]


# --- 증감률 · CAGR -----------------------------------------------------------

def test_growth_rate_is_computed():
    lines = derive_calculations("매출액이 전년 대비 얼마나 증가했나", ROWS)
    assert lines == ["▶▶ 매출액 증감률: +50.00% (1,000 -> 1,500, 증감 +500)"]


def test_cagr_replaces_growth_rate_instead_of_doubling_up():
    """`연평균성장률` 은 `성장률` 을 포함한다. 둘 다 내보내면 모델이
    아무거나 골라 적는다 — CAGR 을 물었으면 CAGR 만 답한다."""
    lines = derive_calculations("연평균성장률은 얼마인가", ROWS)
    assert len(lines) == 1
    assert "연평균성장률(CAGR, 3년): +14.47%" in lines[0]


def test_one_data_point_gets_no_growth_rate():
    assert derive_calculations("전년 대비 증감률은", [row("매출액", 100.0)]) == []


def test_undated_rows_get_no_growth_rate():
    rows = [row("매출액", 100.0, filing=""), row("매출액", 200.0, filing="")]
    assert derive_calculations("전년 대비 증감률은", rows) == []


# --- 비율 --------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "매출액에서 영업이익이 차지하는 비율은",
    "매출액 대비 영업이익 비율은",
    "매출액 중 영업이익 비중은",
    "영업이익의 매출액 대비 비율은",
])
def test_ratio_puts_the_part_on_top(query):
    lines = derive_calculations(query, ROWS)
    assert lines == ["▶▶ 매출액 대비 영업이익 비율: 10.00% (영업이익 150 / 매출액 1,500)"]


def test_ratio_needs_two_named_items():
    assert derive_calculations("매출액 비율은", ROWS) == []


def test_ratio_without_a_direction_word_is_skipped():
    """`매출액 영업이익 비율` 은 어느 쪽이 분자인지 알 수 없다."""
    assert derive_calculations("매출액 영업이익 비율", ROWS) == []


def test_ratio_never_divides_two_companies():
    rows = [row("매출액", 1000.0, company="가"), row("영업이익", 100.0, company="나")]
    assert derive_calculations("매출액 중 영업이익 비중", rows) == []


def test_ratio_never_divides_two_periods():
    rows = [row("매출액", 1000.0, period="2024"), row("영업이익", 100.0, period="2025")]
    assert derive_calculations("매출액 중 영업이익 비중", rows) == []


def test_ratio_never_divides_two_units():
    rows = [row("매출액", 1000.0, unit="백만원"), row("영업이익", 100.0, unit="억원")]
    assert derive_calculations("매출액 중 영업이익 비중", rows) == []


def test_ratio_is_not_this_year_over_last_year():
    """같은 항목의 올해/작년을 나눈 값은 비중이 아니다. 예전 코드가 그걸
    `비율` 이라고 붙여 내보냈다(2026-09-01 수정)."""
    rows = [row("매출액", 1000.0, filing="20230315", period="2022"),
            row("매출액", 1500.0)]
    assert derive_calculations("매출액 비중은", rows) == []


# --- 날짜 --------------------------------------------------------------------

def test_dates_are_read_in_every_korean_form():
    got = parse_dates("2024년 3월 15일 ~ 2026.01.02 그리고 2025-12-31")
    assert got == [date(2024, 3, 15), date(2026, 1, 2), date(2025, 12, 31)]


def test_impossible_dates_are_dropped():
    assert parse_dates("2024.02.30") == []


def test_span_counts_days_and_whole_months():
    assert days_between(date(2024, 1, 1), date(2024, 3, 1)) == 60
    assert months_between(date(2024, 1, 31), date(2024, 2, 28)) == 0
    assert months_between(date(2024, 1, 1), date(2026, 7, 1)) == 30


def test_span_is_described_for_the_prompt():
    text = describe_span(date(2024, 1, 1), date(2024, 12, 31))
    assert text.startswith("▶▶ 기간: 2024-01-01 ~ 2024-12-31")
    assert "365일" in text and "11개월" in text
