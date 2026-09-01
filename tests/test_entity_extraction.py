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


# --------------------------------------------------------------------------- 조사 뒤 글자 (2026-09-01)

@pytest.mark.parametrize("query, wrong", [
    ("삼성전자로지텍의 자산총계는?", "삼성전자"),
    ("한화오션에코텍의 직원 수는?", "한화오션"),
    ("한화오션디지털의 매출액은?", "한화오션"),
])
def test_a_particle_followed_by_more_letters_is_part_of_another_name(extractor, query, wrong):
    """'로'·'에' 는 조사이기도 하고 이름의 첫 글자이기도 하다.

    조사 뒤에 글자가 더 이어지면 그건 조사가 아니다.
        "삼성전자로 인해"  -> '로' 뒤가 공백 -> 조사   -> 삼성전자
        "삼성전자로지텍"    -> '로' 뒤가 '지' -> 이름   -> 다른 회사
    """
    assert wrong not in extractor.extract(query).companies


@pytest.mark.parametrize("query, expected", [
    ("삼성전자로 인해 발생한 손실", ["삼성전자"]),
    ("삼성전자에서 나온 공시", ["삼성전자"]),
    ("삼성전자에 대한 질문", ["삼성전자"]),
])
def test_a_real_particle_still_passes(extractor, query, expected):
    """조사 판정을 조이면서 정상 문장을 죽이면 안 된다."""
    assert extractor.extract(query).companies == expected


# --------------------------------------------------------------------------- 해외 자회사

@pytest.mark.parametrize("query, wrong", [
    ("현대로템 USA의 매출은?", "현대로템"),
    ("LS ELECTRIC AMERICA Inc.의 매출은?", "엘에스일렉트릭"),
    ("삼성전자 Egypt S.A.E의 직원 수는?", "삼성전자"),
    ("삼성중공업 유한공사의 매출", "삼성중공업"),
    ("현대자동차 CHINA 실적", "현대자동차"),
])
def test_an_overseas_affiliate_is_a_different_company(extractor, query, wrong):
    """"현대로템 주식회사"(같은 회사)와 "현대로템 USA"(다른 회사)는 구조가 같다.

    조사 규칙으로는 못 가르므로 지명·해외 법인격을 목록으로 둔다.
    """
    assert wrong not in extractor.extract(query).companies


@pytest.mark.parametrize("query, expected", [
    ("삼성전자 주식회사의 매출", ["삼성전자"]),
    ("현대로템 주식회사", ["현대로템"]),
    ("삼성전자 사업보고서에 기재된 계약금액", ["삼성전자"]),
    ("삼성전자 2024년 실적", ["삼성전자"]),
    ("엘에스일렉트릭 반기보고서", ["엘에스일렉트릭"]),
])
def test_ordinary_words_after_a_company_name_are_not_blocked(extractor, query, expected):
    """목록에 없는 말(사업보고서·매출액·연도)은 그대로 통과해야 한다."""
    assert extractor.extract(query).companies == expected


# ------------------------------------------- 지명은 그 자체로 다른 회사가 아니다
#
# 2026-09-01 교차 검수: 지명으로 시작하기만 하면 무조건 다른 회사로 봐서,
# 유니버스 76개 이름 **전부** 가 "{회사} 유럽 매출액은?" 에서 사라졌다.
# 회사명이 없으니 되묻기로 끝난다 — 정상 질문에 답을 못 하는 쪽이 더 나쁘다.

@pytest.mark.parametrize("query, expected", [
    ("삼성전자 유럽 매출액은 얼마인가?", ["삼성전자"]),
    ("현대자동차 베트남 공장 투자금액은?", ["현대자동차"]),
    ("삼성전자 유럽시장 매출은?", ["삼성전자"]),
    ("현대자동차 차이나 사업 실적은?", ["현대자동차"]),
])
def test_a_place_name_modifying_the_next_word_keeps_the_company(extractor, query, expected):
    assert extractor.extract(query).companies == expected


@pytest.mark.parametrize("query, wrong", [
    ("삼성전자 아메리카의 매출은?", "삼성전자"),
    ("삼성전자 유럽법인의 매출은?", "삼성전자"),
    ("현대자동차 베트남법인 실적", "현대자동차"),
])
def test_a_place_name_acting_as_the_subject_is_a_different_company(extractor, query, wrong):
    """`아메리카의` 처럼 지명이 조사를 달고 주어 노릇을 하면 그건 현지법인이다."""
    assert wrong not in extractor.extract(query).companies


# --------------------------------------------------- 연도와 분기는 한 덩어리다
#
# 쪼개면 `normalize_period_tokens` 가 토큰마다 따로 해석해 "1분기" 를 버리고
# 연도만 남긴다 — 분기 필터가 통째로 풀린다(2026-09-01 발견).

@pytest.mark.parametrize("query, expected", [
    ("삼성전자의 2024년 1분기 매출액은?", ["2024년 1분기"]),
    ("삼성전자의 2024년 3분기 영업이익은?", ["2024년 3분기"]),
    ("삼성전자의 2024년 상반기 매출액은?", ["2024년 상반기"]),
    ("삼성전자의 2024년도 1분기 매출은?", ["2024년도 1분기"]),
    ("삼성전자의 2024년 05월 공시는?", ["2024년 05월"]),
    ("삼성전자의 2024년 매출액은?", ["2024년"]),
])
def test_a_year_and_its_quarter_stay_one_token(extractor, query, expected):
    assert extractor.extract(query).period == expected


def test_the_quarter_survives_normalization():
    from disclosure_rag.retrieval.metadata_filter import normalize_period_tokens
    assert normalize_period_tokens(["2024년 1분기"]) == ["2024-03"]
    assert normalize_period_tokens(["2024년 3분기"]) == ["2024-09"]
