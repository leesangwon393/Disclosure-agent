"""Phase 12 회귀 테스트: Entity Extraction + Query Normalize."""

from __future__ import annotations

from pathlib import Path

import pytest

from disclosure_rag.entity.entity_extractor import EntityExtractor
from disclosure_rag.entity.query_normalizer import normalize_query

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
pytestmark = pytest.mark.skipif(not CORPUS_ROOT.is_dir(), reason="corpus/ 없음")


@pytest.fixture(scope="module")
def extractor():
    return EntityExtractor(corpus_root=CORPUS_ROOT, metric_terms_path=CONFIG_ROOT / "metric_terms.txt")


def test_spec_example_two_companies(extractor):
    """§35 명세 예시를 그대로 재현."""
    e = extractor.extract("삼성전자랑 SK하이닉스 최근 3년 영업이익률 비교해줘")
    assert e.companies == ["삼성전자", "SK하이닉스"]
    assert e.company_count == 2
    assert "최근 3년" in e.period
    assert "영업이익률" in e.metrics
    assert e.explicit_correction is False


def test_common_name_alias_resolution(extractor):
    """§2 README: 현대차→현대자동차, KT→케이티, 엔씨소프트→NC, LIG넥스원→LIG디펜스앤에어로스페이스."""
    assert extractor.extract("현대차 매출 얼마야").companies == ["현대자동차"]
    assert extractor.extract("KT 실적 알려줘").companies == ["케이티"]
    assert extractor.extract("엔씨소프트 영업이익").companies == ["NC"]
    assert extractor.extract("LIG넥스원 주요사항보고서").companies == ["LIG디펜스앤에어로스페이스"]


def test_explicit_correction_detection(extractor):
    e = extractor.extract("삼성전자 정정 전후 영업이익이 어떻게 달라졌어?")
    assert e.explicit_correction is True


def test_report_name_detection(extractor):
    e = extractor.extract("삼성전자 사업보고서에서 매출액 찾아줘")
    assert e.report_name == "사업보고서"


def test_no_company_mentioned(extractor):
    e = extractor.extract("영업이익 얼마야?")
    assert e.companies == []
    assert e.company_count == 0


def test_query_normalize_single_company(extractor):
    e = extractor.extract("삼성전자 2025년 영업이익 얼마야?")
    normalized = normalize_query(e)
    assert normalized == "[COMPANY] 2025년 영업이익 얼마야?"


def test_query_normalize_two_companies_numbered(extractor):
    e = extractor.extract("삼성전자와 SK하이닉스 매출 비교해줘")
    normalized = normalize_query(e)
    assert normalized == "[COMPANY_1]와 [COMPANY_2] 매출 비교해줘"


def test_query_normalize_no_company_returns_unchanged(extractor):
    e = extractor.extract("영업이익 얼마야?")
    assert normalize_query(e) == "영업이익 얼마야?"


def test_query_normalize_repeated_company_reuses_number(extractor):
    e = extractor.extract("삼성전자와 SK하이닉스 비교, 삼성전자가 더 커?")
    normalized = normalize_query(e)
    assert normalized.count("[COMPANY_1]") == 2
    assert normalized.count("[COMPANY_2]") == 1


# --------------------------------------------------------------------------- 다른 회사 오인 (2026-08-31)

@pytest.mark.parametrize("query, wrong", [
    ("카카오뱅크의 사업보고서상 직원 수는 몇 명인가?", "카카오"),
    ("카카오모빌리티의 최근 영업이익은 얼마인가?", "카카오"),
    ("카카오페이증권의 자본총계를 알려줘.", "카카오"),
])
def test_longer_company_name_is_not_mistaken_for_its_prefix(extractor, query, wrong):
    """'카카오뱅크' 를 '카카오' 로 잡으면 **다른 회사 답을 준다.**

    코퍼스에 없는 회사이므로 아무것도 안 잡히는 게 맞다. 그래야 범위 게이트가
    "코퍼스에 없는 회사입니다" 로 끝내고 HCX 도 안 부른다.
    (gold_abstention 160문항 중 3건이 이 경로로 틀린 답을 내고 있었다.)
    """
    assert wrong not in extractor.extract(query).companies
    assert extractor.extract(query).companies == []


@pytest.mark.parametrize("query, expected", [
    ("삼성전자와 SK하이닉스 비교", ["삼성전자", "SK하이닉스"]),
    ("현대차의 2024년 매출액", ["현대자동차"]),
    ("삼성전자를 알려줘", ["삼성전자"]),
    ("삼성전자에서 나온 공시", ["삼성전자"]),
    ("삼성전자는 얼마인가", ["삼성전자"]),
])
def test_particles_after_a_company_name_are_still_the_same_company(extractor, query, expected):
    """조사는 이름의 일부가 아니다 — 여기서 막으면 정상 질문이 전부 죽는다."""
    assert extractor.extract(query).companies == expected
