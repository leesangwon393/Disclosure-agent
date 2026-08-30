"""근거 충분성 게이트 테스트.

실측 문제: 에이전트가 tool 을 한 번 호출하고 멈춘다(상원 100문항 배치에서
`iterations=2` 가 74건). 이 게이트는 모델 판단이 아니라 **규칙**으로 커버리지를
검사하므로 API 호출 없이 검증할 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from disclosure_rag.agent.sufficiency import missing_coverage, nudge_message
from disclosure_rag.entity.entity_extractor import ExtractedEntities


@dataclass
class _Call:
    result: dict = field(default_factory=dict)


def _search(*items) -> _Call:
    return _Call(result={"results": list(items)})


def _ev(company, period="2025-12", is_correction=False, filing_date="20260310"):
    return {"company": company, "period": period, "is_correction": is_correction,
            "filing_date": filing_date}


def _ent(**kw):
    kw.setdefault("raw_query", "질문")
    return ExtractedEntities(**kw)


# --- 다중 회사 -------------------------------------------------------------

def test_second_company_missing_is_detected():
    ent = _ent(companies=["삼성전자", "SK하이닉스"], company_count=2)
    calls = [_search(_ev("삼성전자"))]
    missing = missing_coverage(ent, calls)
    assert missing and "SK하이닉스" in missing[0]


def test_both_companies_present_is_sufficient():
    ent = _ent(companies=["삼성전자", "SK하이닉스"], company_count=2)
    calls = [_search(_ev("삼성전자"), _ev("SK하이닉스"))]
    assert missing_coverage(ent, calls) == []


def test_single_company_question_is_never_nudged():
    ent = _ent(companies=["삼성전자"], company_count=1)
    assert missing_coverage(ent, [_search(_ev("삼성전자"))]) == []


def test_total_search_failure_is_not_a_nudge():
    """전부 못 찾은 건 '더 찾아봐' 로 풀 문제가 아니다 — 다른 원인이다."""
    ent = _ent(companies=["삼성전자", "SK하이닉스"], company_count=2)
    assert missing_coverage(ent, [_search()]) == []


# --- 기간 -------------------------------------------------------------------

def test_second_period_missing_is_detected():
    ent = _ent(companies=["삼성전자"], period=["2023년", "2025년"])
    calls = [_search(_ev("삼성전자", period="2025-12"))]
    missing = missing_coverage(ent, calls)
    assert missing and "2023" in missing[0]


def test_both_periods_present_is_sufficient():
    ent = _ent(companies=["삼성전자"], period=["2023년", "2025년"])
    calls = [_search(_ev("삼성전자", period="2023-12"), _ev("삼성전자", period="2025-12"))]
    assert missing_coverage(ent, calls) == []


def test_filing_date_year_also_counts():
    """major/exchange 는 period 가 없다 — filing_date 연도로 판정한다."""
    ent = _ent(companies=["삼성전자"], period=["2023년", "2025년"])
    calls = [_search(_ev("삼성전자", period=None, filing_date="20230515"),
                     _ev("삼성전자", period=None, filing_date="20250515"))]
    assert missing_coverage(ent, calls) == []


# --- 정정 -------------------------------------------------------------------

def test_correction_question_needs_both_versions():
    ent = _ent(companies=["삼성전자"], explicit_correction=True)
    calls = [_search(_ev("삼성전자", is_correction=True))]
    missing = missing_coverage(ent, calls)
    assert missing and "원본" in missing[0]


def test_correction_question_with_both_is_sufficient():
    ent = _ent(companies=["삼성전자"], explicit_correction=True)
    calls = [_search(_ev("삼성전자", is_correction=True), _ev("삼성전자", is_correction=False))]
    assert missing_coverage(ent, calls) == []


# --- 메시지 -----------------------------------------------------------------

def test_nudge_message_names_the_missing_items():
    """막연히 '더 찾아봐' 가 아니라 빠진 항목의 이름을 준다."""
    msg = nudge_message(["아직 근거가 없는 회사: SK하이닉스"])
    assert "SK하이닉스" in msg
    assert "없다고 하세요" in msg, "자료가 없으면 없다고 하도록 명시해야 지어내지 않는다"


# --- 루프 통합 (HCX 스텁) ----------------------------------------------------

class _StubClient:
    """스크립트된 응답을 순서대로 돌려주는 가짜 HCX."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.seen_messages = []

    def chat(self, messages, **kwargs):
        self.seen_messages.append([dict(m) for m in messages])
        return self._responses.pop(0) if self._responses else {"role": "assistant", "content": "끝"}


def _tool_msg(name, args):
    return {"role": "assistant", "content": "",
            "toolCalls": [{"id": "c1", "function": {"name": name, "arguments": args}}]}


def _make_tools(payloads):
    """호출 순서대로 payloads 를 돌려주는 search_disclosures 스텁."""
    from disclosure_rag.agent.tools import ToolDef

    calls = {"n": 0}

    def handler(**kwargs):
        i = min(calls["n"], len(payloads) - 1)
        calls["n"] += 1
        return {"results": payloads[i]}

    return [ToolDef(name="search_disclosures", description="검색",
                    parameters={"type": "object", "properties": {}}, handler=handler)]


class _StubExtractor:
    def __init__(self, entities):
        self._entities = entities

    def extract(self, question):
        return self._entities


def _run(entities, responses, payloads, **kw):
    from disclosure_rag.agent.agent_loop import run_agent_loop

    client = _StubClient(responses)
    trace = run_agent_loop(client, _make_tools(payloads), "질문",
                           entity_extractor=_StubExtractor(entities), **kw)
    return client, trace


def test_loop_nudges_when_a_company_is_missing():
    """모델이 한 번 검색하고 멈춰도, 두 번째 회사가 없으면 다시 돌린다."""
    ent = _ent(companies=["삼성전자", "SK하이닉스"], company_count=2)
    client, trace = _run(
        ent,
        responses=[_tool_msg("search_disclosures", {"q": "삼성"}),
                   {"role": "assistant", "content": "삼성전자만 답함"},   # 여기서 멈추려 함
                   _tool_msg("search_disclosures", {"q": "하이닉스"}),
                   {"role": "assistant", "content": "둘 다 답함"}],
        payloads=[[_ev("삼성전자")], [_ev("SK하이닉스")]],
    )
    assert len(trace.nudges) == 1, "추가 검색 요청이 나가지 않았다"
    assert "SK하이닉스" in trace.nudges[0]
    assert len(trace.tool_calls) == 2, "재검색이 실제로 실행되지 않았다"
    assert trace.stopped_reason == "no_more_tool_calls"
    assert trace.final_assistant_message == "둘 다 답함"


def test_loop_does_not_nudge_when_coverage_is_complete():
    ent = _ent(companies=["삼성전자", "SK하이닉스"], company_count=2)
    client, trace = _run(
        ent,
        responses=[_tool_msg("search_disclosures", {"q": "둘 다"}),
                   {"role": "assistant", "content": "완료"}],
        payloads=[[_ev("삼성전자"), _ev("SK하이닉스")]],
    )
    assert trace.nudges == []
    assert len(trace.tool_calls) == 1


def test_nudge_budget_is_respected():
    """근거가 계속 안 채워져도 무한히 재촉하지 않는다."""
    ent = _ent(companies=["삼성전자", "SK하이닉스"], company_count=2)
    stop = {"role": "assistant", "content": "그래도 삼성만"}
    client, trace = _run(
        ent,
        responses=[_tool_msg("search_disclosures", {"q": "1"}), stop,
                   _tool_msg("search_disclosures", {"q": "2"}), stop,
                   _tool_msg("search_disclosures", {"q": "3"}), stop],
        payloads=[[_ev("삼성전자")]],
        max_nudges=1,
    )
    assert len(trace.nudges) == 1
    assert trace.stopped_reason == "no_more_tool_calls"
