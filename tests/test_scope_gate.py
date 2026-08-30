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
