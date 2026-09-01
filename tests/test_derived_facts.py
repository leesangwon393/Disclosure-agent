"""파생 계산은 파이썬이 한다 — 모델에게 산수를 시키지 않는다.

실측(2026-09-01, suite_v2): 계산이 붙는 최대/최소 문항은 96%, 계산이 안 붙는
합계 50% · 비율 40% · 평균 0%.

`▶▶` 줄에는 "계산이 끝난 값이니 그대로 베껴라" 가 붙는다. 그래서 **틀린
`▶▶` 는 없는 것보다 나쁘다.** 아래 테스트의 절반은 "이럴 때는 계산하지
않는다" 를 지키는지 보는 것이다.
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
    statement_kind,
    unit_of,
    wanted_operations,
)


def row(item, value, *, filing="20260315", company="A", period="2025",
        unit="백만원", section=(), doc=None):
    return {"item": item, "value_num": value, "filing_date": filing,
            "company": company, "period": period, "unit": unit,
            "section_path": list(section),
            "report_id": doc or f"{company}-{filing}-{item}-{value}"}


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


@pytest.mark.parametrize("query", ["연평균성장률은 얼마인가", "가중평균 유통주식수는"])
def test_a_word_that_merely_contains_average_does_not_ask_for_an_average(query):
    """`연평균성장률` 의 `평균` 이 평균 계산을 켜서 CAGR 질문에 엉뚱한
    평균이 붙었다(2026-09-01 발견)."""
    assert "avg" not in wanted_operations(query)


def test_no_operation_means_no_lines():
    assert derive("매출액은 얼마인가", ROWS) == []


def test_no_numbers_means_no_lines():
    assert derive("매출액 합계는", [{"item": "매출액", "value_num": None}]) == []


# --- 섞으면 안 되는 축: 단위 · 회사 · 시점 · 연결/별도 -----------------------

def test_the_unit_key_the_pipeline_actually_uses_is_read():
    """Facts 저장소는 `value_unit`, 생성기로 넘어가는 dict 는 `unit` 이다.
    `value_unit` 만 보던 코드는 실제 파이프라인에서 단위를 한 번도 못 읽었다."""
    assert unit_of({"unit": "백만원"}) == "백만원"
    assert unit_of({"value_unit": "천원"}) == "천원"
    assert unit_of({"unit": "(단위: 백만원)"}) == "백만원"
    assert unit_of({"unit": "단위:주"}) == "주"
    assert unit_of({}) == ""


def test_consolidated_and_separate_statements_are_different_groups():
    assert statement_kind({"section_path": ["III. 재무에 관한 사항", "연결재무제표"]}) == "연결"
    assert statement_kind({"section_path": ["III. 재무에 관한 사항", "재무제표"]}) == "별도"
    assert statement_kind({"section_path": ["VII. 주주에 관한 사항"]}) == ""


def test_different_units_are_never_added_together():
    rows = [row("영업비용", 3_112_850, company="크래프톤", unit="백만원"),
            row("영업비용", 255_698_325, company="펄어비스", unit="천원")]
    lines = derive("두 회사 영업비용 합계는", rows)
    assert not any("합계" in ln for ln in lines)


def test_every_computed_line_carries_its_unit():
    rows = [row("매출액", 1000.0, company="가"), row("매출액", 500.0, company="나")]
    for line in derive("매출액 합계와 평균, 차이, 큰 순서대로", rows):
        assert "백만원" in line


def test_two_values_for_the_same_company_and_period_stop_the_calculation():
    """어느 쪽이 맞는지 모르면 계산하지 않는다. 반쪽짜리 합계는 오답보다 나쁘다."""
    rows = [row("매출액", 100.0, doc="x"), row("매출액", 200.0, doc="y")]
    assert derive("매출액 합계는", rows) == []


def test_the_newest_filing_wins_for_the_same_company_and_period():
    rows = [row("매출액", 100.0, filing="20240315"),
            row("매출액", 120.0, filing="20260315"),
            row("매출액", 80.0, company="나")]
    lines = derive("매출액 합계는", rows)
    assert any("합계: 200백만원" in ln for ln in lines)


# --- 합계·평균·차이·건수·순위 ------------------------------------------------

def test_sum_only_adds_the_same_item():
    rows = [row("매출액", 1000.0, company="가"), row("매출액", 1500.0, company="나"),
            row("영업이익", 150.0, company="가")]
    lines = derive("매출액 합계는", rows)
    assert any("매출액 합계: 2,500백만원" in ln for ln in lines)
    # 영업이익은 회사가 하나뿐이라 합계를 내지 않는다.
    assert not any("영업이익 합계" in ln for ln in lines)


def test_average_is_computed_not_guessed():
    rows = [row("매출액", 1000.0, company="가"), row("매출액", 1500.0, company="나")]
    assert any("매출액 평균: 1,250.00백만원" in ln for ln in derive("평균은", rows))


def test_difference_uses_max_minus_min():
    rows = [row("매출액", 1000.0, company="가"), row("매출액", 1500.0, company="나")]
    assert any("최대-최소 차이: 500백만원" in ln for ln in derive("차이는 얼마인가", rows))


def test_count_reports_documents_not_rows():
    """같은 값이 여러 청크에서 나온다. 행 수를 세면 건수가 부풀려진다."""
    rows = [row("계약금액", 100.0, doc="d1"), row("계약금액", 100.0, doc="d1"),
            row("계약금액", 200.0, doc="d2")]
    assert derive("몇 건인가", rows) == ["▶▶ 계약금액 건수: 2건"]


def test_rank_is_descending():
    rows = [row("매출액", 300.0, company="가"), row("매출액", 900.0, company="나")]
    lines = derive("큰 순서대로", rows)
    assert lines and lines[0].index("나") < lines[0].index("가")


def test_amount_is_spelled_out_in_korean_units():
    rows = [row("매출액", 1000.0, company="가"), row("매출액", 1500.0, company="나")]
    assert any("25억원" in ln for ln in derive("매출액 합계는", rows))


def test_a_non_monetary_unit_gets_no_won_conversion():
    """주(株)를 원으로 환산하면 안 된다."""
    rows = [row("주식수", 12.0, company="가", unit="주"),
            row("주식수", 8.0, company="나", unit="주")]
    assert derive("합계는", rows) == ["▶▶ 주식수 합계: 20주 (2건)"]


# --- 증감률 · CAGR -----------------------------------------------------------

def test_growth_rate_is_computed():
    lines = derive_calculations("매출액이 전년 대비 얼마나 증가했나", ROWS)
    assert lines == ["▶▶ A 매출액 증감률: +50.00% "
                     "(1,000백만원 -> 1,500백만원, 증감 +500백만원)"]


def test_growth_rate_never_joins_two_companies():
    rows = [row("매출액", 1000.0, company="가", filing="20230315", period="2022"),
            row("매출액", 1500.0, company="가"),
            row("매출액", 9000.0, company="나")]
    lines = derive_calculations("전년 대비 증감률은", rows)
    assert len(lines) == 1 and "가 매출액" in lines[0]


def test_the_same_filing_date_at_both_ends_stops_the_calculation():
    """어느 값이 '전' 이고 어느 값이 '후' 인지 정할 수 없다. 예전에는 입력
    순서에 따라 증감률 부호가 뒤집혔다(2026-09-01 발견)."""
    rows = [row("매출액", 30_937_013.0, doc="x"), row("매출액", 41_895_681.0, doc="y")]
    assert derive_calculations("전년 대비 증감률은", rows) == []


def test_cagr_replaces_growth_rate_instead_of_doubling_up():
    lines = derive_calculations("연평균성장률은 얼마인가", ROWS)
    assert len(lines) == 1
    assert "연평균성장률(CAGR, 3년): +14.47%" in lines[0]


def test_a_value_that_turned_negative_gets_no_cagr_and_does_not_crash():
    """(end/begin)**(1/n) 이 복소수가 되어 round() 가 TypeError 를 던졌다 —
    요청 전체가 죽었다(2026-09-01 발견)."""
    rows = [row("매출액", 299_266.0, filing="20240329", period="2023"),
            row("매출액", -2_823_006.0, filing="20260515")]
    assert derive_calculations("연평균성장률은", rows) == []


def test_one_data_point_gets_no_growth_rate():
    assert derive_calculations("전년 대비 증감률은", [row("매출액", 100.0)]) == []


def test_undated_rows_get_no_growth_rate():
    rows = [row("매출액", 100.0, filing=""), row("매출액", 200.0, filing="")]
    assert derive_calculations("전년 대비 증감률은", rows) == []


# --- 비율 --------------------------------------------------------------------

RATIO_ROWS = [row("매출액", 1500.0), row("영업이익", 150.0)]
RATIO_LINE = ("▶▶ 매출액 대비 영업이익 비율: 10.00% "
              "(영업이익 150백만원 / 매출액 1,500백만원)")


@pytest.mark.parametrize("query", [
    "매출액에서 영업이익이 차지하는 비율은",
    "매출액 대비 영업이익 비율은",
    "매출액 중 영업이익 비중은",
    "영업이익의 매출액 대비 비율은",
])
def test_ratio_puts_the_part_on_top(query):
    assert derive_calculations(query, RATIO_ROWS) == [RATIO_LINE]


def test_a_question_with_no_direction_word_is_skipped():
    """`매출액의 몇 퍼센트가 영업이익인가` 는 조사 `의` 만으로는 분자를
    가릴 수 없다. 예전 규칙은 분자·분모를 뒤집어 3,942.99% 를 내놨다."""
    assert derive_calculations("매출액의 몇 퍼센트가 영업이익인가", RATIO_ROWS) == []
    assert derive_calculations("매출액 영업이익 비율", RATIO_ROWS) == []


def test_ratio_needs_two_named_items():
    assert derive_calculations("매출액 비율은", RATIO_ROWS) == []


def test_an_item_name_inside_a_longer_word_does_not_count():
    """질문의 `총자산` 안에서 별개 항목 `자산` 이 분모로 잡혔다."""
    rows = [row("자산", 50.0), row("유형자산", 300.0)]
    assert derive_calculations("총자산 중 유형자산이 차지하는 비율은", rows) == []


def test_a_particle_after_the_item_name_still_counts():
    """한국어는 이름 뒤에 조사가 붙는다 — `영업이익이` 도 `영업이익` 이다."""
    assert derive_calculations("매출액에서 영업이익이 차지하는 비중은", RATIO_ROWS)


@pytest.mark.parametrize("other", [
    {"company": "나"},
    {"period": "2024"},
    {"unit": "억원"},
    {"section": ["III. 재무에 관한 사항", "연결재무제표"]},
])
def test_ratio_never_divides_across_a_mismatched_axis(other):
    rows = [row("매출액", 1500.0), row("영업이익", 150.0, **other)]
    assert derive_calculations("매출액 중 영업이익 비중", rows) == []


def test_ratio_is_not_this_year_over_last_year():
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
