"""Answer Validator 회귀 테스트 (2026-08-30 개정분).

실측으로 확인된 오탐/침묵 4건을 박제한다.

1. `"2025년"` 의 연도를 근거 없는 숫자로 잡던 오탐
2. `"근거" in answer` 폴백 때문에 has_citation 이 항상 참이던 문제
3. 근거가 0건이면 경고조차 없이 만점 통과하던 침묵
4. 계산해서 나온 값을 전부 근거 없는 숫자로 잡던 오탐
"""
from __future__ import annotations

from disclosure_rag.agent.evidence import Citation, EvidencePack
from disclosure_rag.agent.validator import validate_answer
from disclosure_rag.entity.entity_extractor import ExtractedEntities


def _cit(report_id="periodic_20260310002820", chunk_id="C1", is_correction=False):
    return Citation(
        chunk_id=chunk_id, report_id=report_id, company="삼성전자",
        report_name="사업보고서", filing_date="20260310", section_path=["III. 재무에 관한 사항"],
        is_correction=is_correction, is_latest=True,
    )


def _pack(prompt_text: str, citations=None, tool_results=None):
    return EvidencePack(
        question="질문", prompt_text=prompt_text,
        citations=list(citations if citations is not None else [_cit()]),
        tool_results_summary=list(tool_results or []),
    )


_NO_ENTITIES = ExtractedEntities(raw_query="질문")


# --- 1. 연도 오탐 ------------------------------------------------------------

def test_year_in_period_context_is_not_ungrounded():
    """실측 오탐: `['2025', '2026']` 이 '근거 없는 숫자'로 경고됐다."""
    pack = _pack("[EVIDENCE 1]\n순자산액: 224,787,773,988,054\nreport_id: periodic_20260310002820\n")
    answer = "2025년 사업보고서 기준 순자산액은 224,787,773,988,054원입니다. 근거: periodic_20260310002820"
    res = validate_answer(answer, pack, _NO_ENTITIES)
    assert res.numbers_grounded, res.ungrounded_numbers
    assert res.period_claims == {"2025"}, "연도는 버리지 말고 period_claims 로 남긴다"


def test_year_not_in_period_context_is_still_checked():
    """`"2025"` 가 기간 표현 없이 단독 수치로 나오면 그대로 검사 대상이다."""
    pack = _pack("[EVIDENCE 1]\n내용 없음\nreport_id: periodic_20260310002820\n")
    res = validate_answer("계약 건수는 2025건입니다. periodic_20260310002820", pack, _NO_ENTITIES)
    assert not res.numbers_grounded


# --- 2. has_citation 폴백 제거 -----------------------------------------------

def test_word_geunngeo_alone_is_not_a_citation():
    """`ANSWER_SYSTEM_PROMPT` 가 '근거:' 를 쓰라고 시키므로 이 단어만으로는 인용이 아니다."""
    pack = _pack("[EVIDENCE 1]\n내용\nreport_id: periodic_20260310002820\n")
    res = validate_answer("답변입니다. 근거: 사업보고서", pack, _NO_ENTITIES)
    assert not res.has_citation


def test_real_report_id_counts_as_citation():
    pack = _pack("[EVIDENCE 1]\n내용\nreport_id: periodic_20260310002820\n")
    res = validate_answer("답변입니다. 근거: periodic_20260310002820", pack, _NO_ENTITIES)
    assert res.has_citation


# --- 3. 근거 0건 침묵 --------------------------------------------------------

def test_no_evidence_is_never_a_pass():
    """근거 0건 + 숫자 0개 답변이 만점 통과하던 경로."""
    pack = _pack("[USER QUESTION]\n질문\n", citations=[])
    res = validate_answer("제공된 근거로는 확인할 수 없습니다.", pack, _NO_ENTITIES)
    assert not res.has_any_evidence
    assert not res.passed
    assert any("근거가 하나도 없다" in w for w in res.warnings)


# --- 4. 유도 검산 ------------------------------------------------------------

def test_growth_rate_stated_with_inputs_is_verified():
    pack = _pack(
        "[EVIDENCE 1]\n2025년 영업이익: 1,200\n2024년 영업이익: 1,000\n"
        "report_id: periodic_20260310002820\n"
    )
    answer = (
        "영업이익은 1,000에서 1,200으로 20.0% 증가했습니다. "
        "근거: periodic_20260310002820"
    )
    res = validate_answer(answer, pack, _NO_ENTITIES)
    assert res.numbers_grounded, res.ungrounded_numbers
    assert "20.0" in res.derived_numbers


def test_number_without_stated_inputs_is_not_silently_passed():
    """입력 수치를 안 밝힌 계산값은 검산이 되는 척하지 않는다."""
    pack = _pack(
        "[EVIDENCE 1]\n2025년 영업이익: 1,200\n2024년 영업이익: 1,000\n"
        "report_id: periodic_20260310002820\n"
    )
    res = validate_answer("영업이익이 37.5% 증가했습니다. 근거: periodic_20260310002820", pack, _NO_ENTITIES)
    assert not res.numbers_grounded


# --- 5. 단위 재환산 ----------------------------------------------------------

def test_unit_rescale_is_flagged_not_passed():
    """`ANSWER_SYSTEM_PROMPT` 는 단위 재환산을 금지한다. 통과시키면 자릿수 오류가 샌다."""
    pack = _pack("[EVIDENCE 1]\n순자산액: 7,661,584,000,000\nreport_id: periodic_20260310002820\n")
    res = validate_answer("순자산액은 7,661,584백만원입니다. 근거: periodic_20260310002820", pack, _NO_ENTITIES)
    assert not res.numbers_grounded
    assert res.rescaled_numbers


# --- 6. 정정 근거 완전성 -----------------------------------------------------

def test_correction_question_needs_both_versions():
    pack = _pack(
        "[EVIDENCE 1]\n내용\nreport_id: major_20241118000328\n",
        citations=[_cit(report_id="major_20241118000328", is_correction=True)],
    )
    res = validate_answer("정정 후 금액은 ... major_20241118000328", pack,
                          ExtractedEntities(raw_query="정정 질문", explicit_correction=True))
    assert res.correction_evidence_complete is False
    assert not res.passed
