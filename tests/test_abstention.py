from types import SimpleNamespace

import pytest

from disclosure_rag.agent.abstention import (
    decide_abstention,
    decide_from_evidence_pack,
    fields_found_in_text,
)
from disclosure_rag.agent.query_plan import QueryPlan


def test_zero_evidence_always_abstains_even_for_open():
    plan = QueryPlan(answer_mode="open", expected_fields=["계약금액"])
    decision = decide_abstention(plan, evidence_count=0)
    assert decision.action == "abstain"
    assert decision.reason == "evidence_absent"
    assert "확인되지 않습니다" in decision.finalize("지어낸 답")


def test_closed_missing_required_field_abstains():
    plan = QueryPlan(answer_mode="closed", expected_fields=["계약금액", "계약상대"])
    decision = decide_abstention(plan, evidence_count=2, found_fields=["계약금액"])
    assert decision.action == "abstain"
    assert decision.missing_fields == ("계약상대",)
    assert "계약상대" in decision.finalize("")


@pytest.mark.parametrize("mode", ["open", "mixed"])
def test_open_or_mixed_missing_field_returns_partial_with_limit(mode):
    plan = QueryPlan(answer_mode=mode, expected_fields=["투자금액", "투자목적"])
    decision = decide_abstention(plan, evidence_count=1, found_fields=["투자금액"])
    assert decision.action == "partial"
    answer = decision.finalize("확인된 투자금액은 100원입니다.")
    assert "100원" in answer
    assert "투자목적" in answer
    assert "한계" in answer


def test_all_required_fields_present_answers():
    plan = QueryPlan(answer_mode="closed", expected_fields=["계약금액"])
    decision = decide_abstention(plan, evidence_count=1, found_fields=["계약 금액"])
    assert decision.action == "answer", "공백·구두점 표기 차이는 정규화돼야 한다"


def test_no_expected_fields_fails_open_when_evidence_exists():
    plan = QueryPlan(answer_mode="closed", expected_fields=[])
    assert decide_abstention(plan, evidence_count=1).action == "answer"


def test_negative_evidence_count_is_rejected():
    with pytest.raises(ValueError):
        decide_abstention(QueryPlan(), evidence_count=-1)


def test_text_fallback_requires_literal_normalized_field_name():
    expected = ["계약금액", "계약상대"]
    got = fields_found_in_text(expected, "- 계약 금액: 100원\n- 판매지역: 한국")
    assert got == ["계약금액"]


def test_evidence_pack_adapter_counts_fact_rows_but_not_calculator():
    plan = QueryPlan(answer_mode="closed", expected_fields=["계약금액"])
    pack = SimpleNamespace(
        citations=[], prompt_text="[TOOL RESULT] 계약금액: 100",
        tool_results_summary=[
            {"tool": "lookup_fact", "result": {"results": [
                {"doc_id": "d1", "chunk_id": "c1", "item": "계약금액", "value": "100"}
            ]}},
            {"tool": "calculate_ratio", "result": {"value": 10}},
        ],
    )
    decision = decide_from_evidence_pack(plan, pack)
    assert decision.evidence_count == 1
    assert decision.action == "answer"


def test_calculator_only_is_zero_document_evidence():
    pack = SimpleNamespace(
        citations=[], prompt_text="", tool_results_summary=[
            {"tool": "calculate_ratio", "result": {"value": 10}},
        ],
    )
    assert decide_from_evidence_pack(QueryPlan(answer_mode="open"), pack).action == "abstain"


def test_document_id_without_chunk_content_is_not_evidence():
    pack = SimpleNamespace(
        citations=[], prompt_text="[TOOL RESULT]\nget_latest_report: found",
        tool_results_summary=[
            {"tool": "get_latest_report", "result": {"found": True, "doc_id": "d1"}},
        ],
    )
    assert decide_from_evidence_pack(QueryPlan(answer_mode="closed"), pack).evidence_count == 0


def test_question_text_does_not_fake_field_coverage():
    plan = QueryPlan(answer_mode="closed", expected_fields=["직원수"])
    pack = SimpleNamespace(
        citations=[object()],
        prompt_text=("[USER QUESTION]\n이 공시의 직원수는?\n\n"
                     "[EVIDENCE 1]\n계약금액: 100원"),
        tool_results_summary=[],
    )
    decision = decide_from_evidence_pack(plan, pack)
    assert decision.action == "abstain"
    assert decision.missing_fields == ("직원수",)


# ===========================================================================
# 분해·정정 신호 연결 (2026-08-30)
# ===========================================================================
#
# 아래 두 경우는 '근거 건수'로는 절대 안 잡힌다. 건수만 보고 통과시키면
# 한쪽 근거만으로 비교 결론을 내거나, 비교 대상이 없는데 비교했다고 답한다.

from types import SimpleNamespace  # noqa: E402

from disclosure_rag.agent.abstention import decide_from_processed  # noqa: E402
from disclosure_rag.agent.evidence_processor import process_evidence  # noqa: E402
from disclosure_rag.agent.query_plan import QueryPlan as _QP  # noqa: E402


def _chunk(cid, rid="r", fields=None, order=0):
    return SimpleNamespace(
        chunk_id=cid, report_id=rid, correction_order=order, is_latest=True,
        raw_text="", text="",
        field_codes=[SimpleNamespace(key=k, text=v, unit=None)
                     for k, v in (fields or {}).items()],
    )


def test_missing_comparison_target_blocks_a_closed_answer():
    """S007~S014 — 삼성전자 근거 10건, 한미반도체 0건. 건수로는 통과한다."""
    plan = _QP(answer_mode="closed", task="compare",
               companies=["삼성전자", "한미반도체"], expected_fields=["계약금액"])
    ev = [_chunk(f"c{i}", fields={"계약금액": "1,000"}) for i in range(10)]
    processed = process_evidence(plan, ev)
    dec = decide_from_processed(
        plan, processed,
        decompose_result=SimpleNamespace(empty_labels=["company:한미반도체"], merged=ev),
    )
    assert dec.should_abstain
    assert dec.reason == "target_evidence_missing"
    assert dec.evidence_count == 10          # 근거는 많았다
    assert "한미반도체" in dec.message


def test_missing_target_becomes_partial_for_open_questions():
    plan = _QP(answer_mode="open", task="compare",
               companies=["A", "B"], expected_fields=["계약금액"])
    ev = [_chunk("c1", fields={"계약금액": "1,000"})]
    dec = decide_from_processed(
        plan, process_evidence(plan, ev),
        decompose_result=SimpleNamespace(empty_labels=["company:B"], merged=ev),
    )
    assert dec.action == "partial"
    assert dec.can_generate                  # 확인된 내용은 답한다
    assert "B" in dec.message


def test_incomplete_version_pair_blocks_correction_diff():
    """최종본만 10건 — 비교 대상이 없는데 '달라졌다'고 답하면 안 된다."""
    plan = _QP(answer_mode="closed", task="correction_diff",
               expected_fields=["계약금액"])
    ev = [_chunk(f"c{i}", "fix", {"계약금액": "150"}, order=2) for i in range(10)]
    dec = decide_from_processed(plan, process_evidence(plan, ev))
    assert dec.should_abstain
    assert dec.reason == "version_pair_incomplete"
    assert dec.incomplete_pairs == ("계약금액",)


def test_complete_pair_passes():
    plan = _QP(answer_mode="mixed", task="correction_diff", expected_fields=["계약금액"])
    ev = [_chunk("c1", "orig", {"계약금액": "100"}, order=0),
          _chunk("c2", "fix", {"계약금액": "150"}, order=2)]
    dec = decide_from_processed(plan, process_evidence(plan, ev))
    assert dec.action == "answer"


def test_empty_target_outranks_missing_field():
    """한쪽 근거만으로 비교 결론을 내는 게 항목 하나 빠지는 것보다 위험하다."""
    plan = _QP(answer_mode="closed", task="compare", companies=["A", "B"],
               expected_fields=["계약금액", "계약상대"])
    ev = [_chunk("c1", fields={"계약금액": "1,000"})]      # 계약상대도 없다
    dec = decide_from_processed(
        plan, process_evidence(plan, ev),
        decompose_result=SimpleNamespace(empty_labels=["company:B"], merged=ev),
    )
    assert dec.reason == "target_evidence_missing"


def test_pair_signal_ignored_when_task_is_not_correction_diff():
    plan = _QP(answer_mode="closed", task="lookup", expected_fields=["계약금액"])
    ev = [_chunk("c1", fields={"계약금액": "100"})]
    dec = decide_from_processed(plan, process_evidence(plan, ev))
    assert dec.action == "answer"


def test_found_fields_come_from_stage9_not_prompt_text():
    """판정을 한 곳으로 모은다 — 충분성과 거부 게이트가 어긋나면 안 된다."""
    plan = _QP(answer_mode="closed", task="lookup", expected_fields=["계약금액"])
    ev = [_chunk("c1", fields={"계약금액": "1,000"})]
    processed = process_evidence(plan, ev)
    dec = decide_from_processed(plan, processed)
    assert dec.found_fields == ("계약금액",)
    assert dec.action == "answer"


def test_zero_evidence_still_wins_over_everything():
    plan = _QP(answer_mode="closed", task="compare", companies=["A", "B"])
    dec = decide_from_processed(
        plan, process_evidence(plan, []),
        decompose_result=SimpleNamespace(empty_labels=["company:A", "company:B"], merged=[]),
    )
    assert dec.reason == "evidence_absent"
