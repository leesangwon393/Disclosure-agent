from disclosure_rag.agent.query_plan import QueryPlan
from disclosure_rag.agent.scope_gate import apply_scope_gate, evaluate_scope


class _Registry:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def types_for(self, name):
        return set(self.mapping.get(name, ()))


def test_market_price_is_the_only_kind_of_entity_independent_early_refusal():
    plan = QueryPlan(companies=["삼성전자"])
    decision = evaluate_scope(plan, "삼성전자 현재 주가가 얼마야?", _Registry({"삼성전자": {"universe"}}))
    assert decision.scope == "hard_out_scope"
    assert decision.should_refuse
    assert decision.matched_rule == "market_price"


def test_disclosure_question_that_mentions_stock_price_is_not_blocked():
    plan = QueryPlan(companies=["삼성전자"])
    decision = evaluate_scope(
        plan, "공시에 주가에 미칠 영향이 기재됐는지 확인해줘.",
        _Registry({"삼성전자": {"universe"}}),
    )
    assert decision.scope == "in_scope"
    assert not decision.should_refuse


def test_non_universe_submitter_is_in_scope():
    plan = QueryPlan(companies=["LG전자"])
    decision = evaluate_scope(plan, "LG전자가 제출한 대량보유보고서를 찾아줘.",
                              _Registry({"LG전자": {"submitter", "shareholder"}}))
    assert decision.scope == "in_scope"
    assert decision.action == "proceed"


def test_unknown_company_is_possibly_scope_not_refused():
    plan = QueryPlan(companies=["쿠팡"])
    decision = evaluate_scope(plan, "쿠팡 사업보고서 매출액은?", _Registry())
    assert decision.scope == "possibly_scope"
    assert decision.action == "proceed"


def test_no_company_is_possibly_scope_and_proceeds():
    decision = evaluate_scope(QueryPlan(), "매출액은 얼마야?", _Registry())
    assert decision.scope == "possibly_scope"
    assert decision.action == "proceed"
    assert decision.needs_clarification
    assert "회사명" in decision.clarification_message


def test_broad_summary_suggests_clarification_without_refusing():
    plan = QueryPlan(companies=["삼성전자"], answer_mode="open", task="summarize")
    decision = evaluate_scope(plan, "삼성전자 공시 알려줘.", _Registry({"삼성전자": {"universe"}}))
    assert decision.action == "proceed"
    assert decision.needs_clarification
    assert "공시 종류" in decision.clarification_message


def test_apply_writes_only_scope_contract_fields():
    plan = QueryPlan(companies=["삼성전자"])
    decision = apply_scope_gate(plan, "삼성전자 최신 뉴스를 알려줘.", _Registry({"삼성전자": {"universe"}}))
    assert decision.should_refuse
    assert plan.scope == "hard_out_scope"
    assert plan.scope_reason
    assert plan.source["scope"] == "rule"


# ---------------------------------------- 코퍼스 밖 회사 · 역질문 (2026-08-31)
#
# gold_abstention 160문항 실측에서 두 유형이 거의 전멸이었다:
#   ambiguous     5%  — 무엇을 묻는지 특정 못 하면 **빈 답변**이 나갔다(12건)
#   wrong_entity 12.5% — 코퍼스 밖 회사인데 검색하고 HCX 까지 부른 뒤 본문에서 거부
# scope_gate 가 판단은 하고 있었는데 ask_v2 가 그 값을 쓰지 않았다.

from disclosure_rag.agent.scope_gate import unknown_subject  # noqa: E402


class _NamedRegistry:
    """이름 목록을 갖는 레지스트리(위쪽 `_Registry` 와 이름이 겹치지 않게)."""
    def __init__(self, names): self._names = set(names)
    def types_for(self, name): return {"corp"} if name in self._names else set()


_REG = _NamedRegistry({"삼성전자", "한미반도체"})


def test_named_company_outside_the_corpus_is_detected():
    """이름은 댔는데 코퍼스에 없는 회사 — '회사명이 필요합니다'는 틀린 답이다."""
    assert unknown_subject("쿠팡의 최근 사업보고서 매출액은 얼마인가?", _REG) == "쿠팡"
    assert unknown_subject("야놀자가 체결한 공급계약 금액을 알려줘.", _REG) == "야놀자"


def test_known_company_is_not_flagged():
    assert unknown_subject("삼성전자의 순자산액은 얼마인가?", _REG) is None


def test_pronoun_subject_is_not_a_company():
    """`그것의 …` 는 역질문 대상이지 '없는 회사'가 아니다."""
    assert unknown_subject("그것의 현재 값과 이전 값을 비교해줘.", _REG) is None


def test_question_without_a_named_subject_is_not_flagged():
    for q in ("그 회사 계약금액은 얼마야?", "작년 실적이 어떻게 됐어?",
              "매출액은 얼마야?", "계약이 바뀌었어?", "두 회사를 비교해줘."):
        assert unknown_subject(q, _REG) is None, q


# --------------------------------------------------------------------------- 역질문 오탐 (2026-08-31)

def test_short_but_clear_question_is_not_asked_back(registry_fixture_or_none=None):
    """"삼성전자의 2024년 매출액은?" 같은 문장으로 되물으면 안 된다.

    규칙만으로는 answer_mode/task 가 unknown 이지만, 그 빈칸은 계획 보완(2b)이
    채운다. 되물을 일이 아니다. ("...얼마인가?" 로 끝나면 규칙이 채우고
    "...은?" 으로 끝나면 못 채웠다 — 문장 끝 표현 하나로 답을 안 주던 버그.)
    """
    from disclosure_rag.agent.query_plan import QueryPlan
    from disclosure_rag.agent.scope_gate import evaluate_scope

    plan = QueryPlan(companies=["삼성전자"], periods=["2024-12"])   # mode/task 미정
    registry = _NamedRegistry({"삼성전자": ["corp"]})

    filled = evaluate_scope(plan, "삼성전자의 2024년 매출액은?", registry,
                            can_fill_blanks=True)
    assert not filled.needs_clarification

    # 보완이 불가능한 상황(클라이언트 없음)에서는 예전대로 되묻는다
    unfilled = evaluate_scope(plan, "삼성전자의 2024년 매출액은?", registry)
    assert unfilled.needs_clarification


def test_missing_company_still_asks_back_even_when_blanks_are_fillable():
    """회사명이 없는 건 HCX 가 채워줄 수 있는 빈칸이 아니다."""
    from disclosure_rag.agent.query_plan import QueryPlan
    from disclosure_rag.agent.scope_gate import evaluate_scope

    plan = QueryPlan(companies=[], answer_mode="closed", task="lookup")
    decision = evaluate_scope(plan, "그것의 매출액은?", _NamedRegistry({}),
                              can_fill_blanks=True)
    assert decision.needs_clarification
