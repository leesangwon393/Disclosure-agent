"""기간(period) 필터 회귀 테스트.

2026-08-30 이전 구현은 `chunk.period not in self.periods` 정확일치였다.
실측상 chunk 의 period 는 정기공시만 "YYYY-MM" 이고 major/exchange/holding 은
전건 None 이라, 이 필터는 **어떤 입력을 넣어도 사실상 0건**이었다.
아래 테스트는 그 상태로 되돌아가면 반드시 깨진다.
"""
from __future__ import annotations

from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.retrieval.metadata_filter import (
    RetrievalFilter,
    filter_chunks,
    normalize_period_tokens,
)


def _c(cid: str, *, period: str | None, filing_date: str, report_type: str) -> ChunkSchema:
    return ChunkSchema(
        chunk_id=cid, report_id="r_" + cid, text="본문", raw_text="본문",
        company="삼성전자", period=period, filing_date=filing_date, report_type=report_type,
    )


PERIODIC_2024 = _c("p1", period="2024-12", filing_date="20250310", report_type="periodic")
PERIODIC_2024Q1 = _c("p2", period="2024-03", filing_date="20240515", report_type="periodic")
PERIODIC_2025 = _c("p3", period="2025-12", filing_date="20260310", report_type="periodic")
MAJOR_2024 = _c("m1", period=None, filing_date="20241118", report_type="major")
HOLDING_2025 = _c("h1", period=None, filing_date="20250120", report_type="holding")

ALL = [PERIODIC_2024, PERIODIC_2024Q1, PERIODIC_2025, MAJOR_2024, HOLDING_2025]


# --- normalize_period_tokens -------------------------------------------------

def test_natural_language_year_becomes_year_token():
    assert normalize_period_tokens("2024년") == ["2024"]


def test_quarter_resolves_to_fiscal_month():
    """70개사 전부 12월 결산이라 분기 → 기준월이 하나로 정해진다."""
    assert normalize_period_tokens("2026년 1분기") == ["2026-03"]
    assert normalize_period_tokens("2024년 상반기") == ["2024-06"]


def test_year_month_forms_are_unified():
    assert normalize_period_tokens("2024.3월") == ["2024-03"]
    assert normalize_period_tokens("2024-03") == ["2024-03"]


def test_multi_year_query_keeps_both():
    assert normalize_period_tokens("2023년과 2025년 비교") == ["2023", "2025"]


def test_unusable_tokens_return_none_not_empty_filter():
    """'최근 3년' '1분기' 는 연도가 없어 필터로 못 쓴다 → 기간 필터를 걸지 않는다.

    빈 리스트를 돌려주면 호출부가 `periods=[]` 로 필터를 걸어 전건 탈락한다.
    """
    assert normalize_period_tokens("최근 3년") is None
    assert normalize_period_tokens("1분기") is None
    assert normalize_period_tokens(None) is None


# --- 연도 단위 매칭 (옛 구현이 영구 0건이던 케이스) --------------------------

def test_year_request_matches_year_month_chunk():
    got = filter_chunks(ALL, RetrievalFilter(periods=["2024"]))
    assert {c.chunk_id for c in got} == {"p1", "p2", "m1"}, (
        "'2024' 로 걸면 2024-12/2024-03 정기공시와 2024년 접수 major 가 잡혀야 한다"
    )


def test_year_request_is_not_empty():
    """옛 정확일치 구현에서는 이 결과가 반드시 0건이었다."""
    assert filter_chunks(ALL, RetrievalFilter(periods=["2024"]))


# --- period 가 없는 문서 유형 (옛 구현이 통째로 날리던 케이스) ---------------

def test_year_month_request_keeps_period_less_docs_by_filing_year():
    got = filter_chunks(ALL, RetrievalFilter(periods=["2024-12"]))
    assert "p1" in {c.chunk_id for c in got}
    assert "m1" in {c.chunk_id for c in got}, (
        "major/exchange/holding 은 period 가 없다. 월로 거르면 전건 사라진다"
    )
    assert "p2" not in {c.chunk_id for c in got}, "정기공시는 월까지 정확히 본다"


def test_other_year_is_excluded():
    got = filter_chunks(ALL, RetrievalFilter(periods=["2025"]))
    assert {c.chunk_id for c in got} == {"p3", "h1"}


def test_no_period_filter_returns_everything():
    assert len(filter_chunks(ALL, RetrievalFilter())) == len(ALL)


def test_period_filter_combines_with_other_filters():
    got = filter_chunks(ALL, RetrievalFilter(periods=["2024"], doc_groups=["periodic"]))
    assert {c.chunk_id for c in got} == {"p1", "p2"}
