"""한국 공시 숫자 표기 — 읽기와 환산 (2026-09-01)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from disclosure_rag.common.korean_number import (  # noqa: E402
    describe_amount, format_korean_amount, normalize_unit_text, same_ratio,
    to_won, unit_multiplier,
)


def test_reads_the_unit_line_of_a_table():
    """표 머리는 "(단위 : 백만원, %)" 처럼 붙어 온다."""
    assert normalize_unit_text("(단위 : 백만원, %)") == "백만원"
    assert normalize_unit_text("(단위:천원)") == "천원"
    assert normalize_unit_text("(단위 : 억원, %)") == "억원"
    # 금액 단위가 아니면 빈 값 — 없는 단위를 지어내지 않는다
    assert normalize_unit_text("(단위 : 주, %)") == ""
    assert normalize_unit_text(None) == ""


def test_bigger_units_win_over_smaller_ones():
    """"백만원" 이 "만원" 보다 먼저 걸려야 한다."""
    assert unit_multiplier("백만원") == 1_000_000
    assert unit_multiplier("만원") == 10_000
    assert unit_multiplier("십억원") == 1_000_000_000


def test_the_thousand_fold_trap():
    """같은 3,112,850 이 표에 따라 1,000배 차이가 난다."""
    assert to_won(3_112_850, "백만원") == 3_112_850_000_000
    assert to_won(3_112_850, "천원") == 3_112_850_000


def test_reads_an_amount_the_way_a_person_would():
    assert format_korean_amount(22_764_764_160_000).startswith("22조 7,647억")
    assert format_korean_amount(0) == "0원"
    assert format_korean_amount(-1_0000_0000).startswith("-1억")


def test_describes_a_table_value_with_both_forms():
    """원문 숫자를 지우지 않는다 — 괄호로 같이 남긴다."""
    text = describe_amount(3_112_850, "백만원")
    assert "3조 1,128억" in text
    assert "3,112,850백만원" in text


def test_unknown_unit_produces_nothing():
    """단위를 모르면 환산을 지어내지 않는다."""
    assert describe_amount(1234, None) == ""
    assert describe_amount(1234, "(단위 : 주)") == ""


def test_percent_and_decimal_are_the_same_ratio():
    """G0146: 정답지 0.0430, 답변 4.30%. 후자가 더 정확한데 오답 처리됐다."""
    assert same_ratio(0.0430, 4.30)
    assert same_ratio(4.30, 0.0430)
    assert not same_ratio(1.0, 2.0)
    assert not same_ratio(0.05, 4.30)
