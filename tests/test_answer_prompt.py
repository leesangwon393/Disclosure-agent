"""Stage 14 답변 프롬프트 분기.

한 프롬프트로 closed/open 을 다 시키면 open 답변이 값 하나만 뱉거나 closed
답변에 불필요한 서술이 붙는다. 모드별로 실제 지시가 갈리는지 본다.
"""
from __future__ import annotations

from disclosure_rag.agent.answer_generator import (
    ANSWER_SYSTEM_PROMPT,
    answer_token_budget,
    build_answer_prompt,
)
from disclosure_rag.agent.query_plan import QueryPlan


def test_no_plan_keeps_the_existing_prompt():
    """기존 호출부가 그대로 동작해야 한다."""
    assert build_answer_prompt(None) == ANSWER_SYSTEM_PROMPT


def test_measured_rules_survive_in_every_mode():
    """공통부의 규칙들은 실측 실패에서 나온 것이라 모드와 무관하게 남아야 한다.
    (자릿수 오류, 여러 공시 값 나열, 묻지 않은 결론 금지)"""
    for mode in ("closed", "open", "mixed"):
        p = build_answer_prompt(QueryPlan(answer_mode=mode))
        assert "글자 그대로 옮기세요" in p
        assert "전부 제시하세요" in p
        assert "질문에 없는 결론을 덧붙이지 마세요" in p


def test_closed_asks_for_a_single_value():
    p = build_answer_prompt(QueryPlan(answer_mode="closed", task="lookup"))
    assert "단일 값" in p
    assert "덧붙이지 마세요" in p
    assert "항목별로 줄을 나눠" not in p


def test_open_asks_for_item_by_item_and_marks_gaps():
    p = build_answer_prompt(QueryPlan(answer_mode="open", task="summarize"))
    assert "항목별로 줄을 나눠" in p
    assert "(확인되지 않음)" in p


def test_mixed_asks_for_a_verdict_first():
    """S023~S026 — '있는가?' 에 먼저 단정하고 그 다음 설명."""
    p = build_answer_prompt(QueryPlan(answer_mode="mixed", task="correction_diff"))
    assert "첫 줄에서 예/아니오를 단정" in p
    assert "항목별로 줄을 나눠" in p


def test_correction_diff_requires_unchanged_items_too():
    """바뀐 것만 쓰면 나머지를 확인했는지 알 수 없다."""
    p = build_answer_prompt(QueryPlan(answer_mode="mixed", task="correction_diff"))
    assert "변동 없음" in p
    assert "최초 공시 값 / 최종 정정본 값" in p


def test_compare_forbids_a_conclusion_from_one_side():
    """S007~S014 — 한쪽 값만으로 '더 크다'고 답하면 오답."""
    p = build_answer_prompt(QueryPlan(answer_mode="closed", task="compare"))
    assert "한쪽 대상의 근거가 없으면 결론을 내지 말고" in p


def test_calculate_forbids_mental_arithmetic():
    p = build_answer_prompt(QueryPlan(answer_mode="closed", task="calculate"))
    assert "직접 암산하지 마세요" in p


def test_count_requires_listing_what_was_counted():
    p = build_answer_prompt(QueryPlan(answer_mode="closed", task="count"))
    assert "센 대상을 나열" in p


def test_expected_fields_are_named_for_open_answers():
    """'주요 내용을 정리'는 사람마다 다르게 읽히지만 항목 목록은 그렇지 않다."""
    plan = QueryPlan(answer_mode="open", task="summarize",
                     expected_fields=["투자금액", "투자목적", "투자기간"])
    p = build_answer_prompt(plan)
    assert "반드시 다뤄야 하는 항목" in p
    for f in plan.expected_fields:
        assert f"- {f}" in p


def test_expected_fields_are_not_listed_for_closed():
    """closed 의 요구 항목은 질문이 지목한 지표 하나다 — 목록을 붙이면
    묻지 않은 항목까지 답하게 된다."""
    plan = QueryPlan(answer_mode="closed", task="lookup", expected_fields=["계약금액"])
    assert "반드시 다뤄야 하는 항목" not in build_answer_prompt(plan)


def test_token_budget_scales_with_mode():
    """closed 는 값 하나라 길 필요가 없고, open 은 항목 수만큼 늘어난다."""
    assert answer_token_budget(QueryPlan(answer_mode="closed")) == 800
    assert answer_token_budget(QueryPlan(answer_mode="open")) > 800
    assert answer_token_budget(QueryPlan(answer_mode="mixed")) == \
        answer_token_budget(QueryPlan(answer_mode="open"))
    assert answer_token_budget(None) == 800


def test_unknown_mode_gets_the_safer_middle_budget():
    """모드를 못 정했으면 open 쪽으로 기운다 — 항목이 빠지는 쪽이 더 비싸다."""
    b = answer_token_budget(QueryPlan(answer_mode="unknown"))
    assert 800 < b <= answer_token_budget(QueryPlan(answer_mode="open"))


# --------------------------------------------- closed 블록과 공통규칙 4의 충돌
#
# 실측 실패(S001, 2026-08-30): 삼성전자 자기주식취득결정 공시가 6건이고
# 순자산액이 전부 다르다. 같은 프롬프트·같은 근거로
#   v2_off5 -> 6건을 공시일과 함께 전부 나열 (정답)
#   v2_off6 -> 최신 1건만 답변          (오답)
# 공통규칙 4("여러 공시에 있으면 전부 제시")와 closed 10/12("값 하나만,
# 다른 항목 덧붙이지 마")가 충돌해 어느 쪽을 따를지가 실행마다 갈렸다.

def test_closed_block_defers_to_the_list_all_rule():
    from disclosure_rag.agent.query_plan import QueryPlan
    prompt = build_answer_prompt(QueryPlan(answer_mode="closed", task="lookup"))
    assert "공통규칙 4가" in prompt and "우선" in prompt, (
        "closed 프롬프트가 '여러 공시면 전부 나열' 예외를 명시해야 한다"
    )
    assert "여러 시점" in prompt


def test_closed_block_still_forbids_unrelated_fields():
    """예외를 넣느라 '묻지 않은 항목 금지'가 사라지면 안 된다."""
    from disclosure_rag.agent.query_plan import QueryPlan
    prompt = build_answer_prompt(QueryPlan(answer_mode="closed", task="lookup"))
    assert "묻지 않은 다른 항목" in prompt


def test_open_and_mixed_blocks_are_untouched():
    from disclosure_rag.agent.query_plan import QueryPlan
    for mode in ("open", "mixed"):
        prompt = build_answer_prompt(QueryPlan(answer_mode=mode, task="summarize"))
        assert "(확인되지 않음)" in prompt
        assert "공통규칙 4가" not in prompt      # closed 전용 문구다
