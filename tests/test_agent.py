"""Phase 15~19 회귀 테스트: HCX Agent Loop, Tools, Evidence Pack, Answer, Validator.

실제 HCX API(.env 의 HCX_API_KEY) 를 호출한다 — 네트워크/키 필요, 느림, 계정
tier 에 따라 rate-limit 영향을 받을 수 있다. 전부 slow 마킹.

핵심 실측 사실(중요, 회귀 방지용으로 여기 남겨둠):
- system prompt 가 길면(예전 버전 ~400자, 6줄 bullet) tool-calling 2번째 턴부터
  HCX API 가 결정적으로(재시도해도 실패) 400 "Unsupported function" 을 반환했다.
  AGENT_SYSTEM_PROMPT 를 짧게 유지해야 한다 — 이 테스트가 그 회귀를 잡는다
  (max_iterations>=2 가 필요한 시나리오를 반드시 하나는 포함해야 함).
- Coarse-to-Fine 검색에서 HCX 가 period 포맷을 잘못 추측해 필터가 0건이 되는
  케이스가 실측으로 흔했다 — search_disclosures 의 자동 필터 완화 재시도로 처리.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from disclosure_rag.agent.calculation import calculate_cagr, calculate_growth_rate, calculate_ratio
from disclosure_rag.agent.evidence import build_evidence_pack
from disclosure_rag.agent.validator import validate_answer
from disclosure_rag.chunking.chunk_schema import filter_leaf_chunks
from disclosure_rag.common.manifest_loader import load_manifest
from disclosure_rag.common.unicode_utils import PathResolver
from disclosure_rag.correction.correction_graph_builder import build_correction_index
from disclosure_rag.entity.entity_extractor import EntityExtractor
from disclosure_rag.pipeline import build_all_chunks
from disclosure_rag.retrieval.bm25_retriever import BM25Retriever
from disclosure_rag.retrieval.tokenizers import build_tokenizer

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
pytestmark = pytest.mark.skipif(not CORPUS_ROOT.is_dir(), reason="corpus/ 없음")

SAMPLE_DOC_IDS = {
    "periodic_20240312000736", "major_20241118000171",
    "exchange_20250728800035", "exchange_20250731800028",
    "holding_20241025000530",
}


def test_calculation_tools_are_deterministic():
    assert calculate_growth_rate(100, 150)["growth_rate_pct"] == 50.0
    assert calculate_ratio(50, 200)["ratio_pct"] == 25.0
    r = calculate_cagr(100, 200, 3)
    assert 25.0 < r["cagr_pct"] < 26.5  # (2^(1/3)-1)*100 ≈ 25.99


def test_calculation_tools_handle_zero_denominator():
    assert calculate_growth_rate(0, 10)["error"]
    assert calculate_ratio(10, 0)["error"]


def _try_setup():
    if not ENV_PATH.is_file():
        pytest.skip(".env 없음 — HCX API 키 필요")
    try:
        from disclosure_rag.agent.hcx_client import HCXClient
        client = HCXClient(env_path=ENV_PATH)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"HCX client 초기화 불가: {e}")

    manifest = load_manifest(CORPUS_ROOT)
    resolver = PathResolver(CORPUS_ROOT)
    rows = [r for r in manifest if r.doc_id in SAMPLE_DOC_IDS]
    chunks = filter_leaf_chunks(build_all_chunks(str(CORPUS_ROOT), rows=rows, validate=False))
    correction_index = build_correction_index(manifest, resolver)
    tok = build_tokenizer("kiwi", user_dict_path=CONFIG_ROOT / "financial_terms.txt")
    bm25 = BM25Retriever(chunks, tok)

    from disclosure_rag.agent.tools import build_all_tools
    tools = build_all_tools(bm25, manifest, correction_index)
    extractor = EntityExtractor(corpus_root=CORPUS_ROOT, metric_terms_path=CONFIG_ROOT / "metric_terms.txt")
    return client, tools, extractor


@pytest.mark.slow
def test_agent_single_lookup_finds_correct_number():
    """실측 검증된 케이스: 삼성전자 반도체 위탁생산 계약금액 = 22,764,764,160,000원.
    (교차검증: corpus/raw 원문 XML 의 '계약금액(원)' 필드와 일치함)"""
    client, tools, extractor = _try_setup()
    from disclosure_rag.agent.ask import ask

    result = ask(client, tools, "삼성전자 반도체 위탁생산 계약금액 얼마야?", entity_extractor=extractor, max_iterations=4)
    assert "22,764,764,160,000" in result.answer or "22조" in result.answer
    assert result.validation.numbers_grounded
    assert result.validation.has_citation


@pytest.mark.slow
def test_agent_correction_analysis_uses_both_versions_and_two_plus_turns():
    """회귀 테스트: AGENT_SYSTEM_PROMPT 가 길어지면 2턴째부터 API 가 결정적으로
    실패했던 문제를 잡는다 — 이 시나리오는 반드시 다중 턴(get_correction_history
    이후 최소 1번 이상의 추가 tool 호출)이 필요하다.
    실측 검증된 정답: 계약상대가 '글로벌 대형기업' -> '테슬라(Tesla, Inc.)' 로 정정됨
    (교차검증: corpus/raw 원문 XML 의 정정사항 표와 일치함)."""
    client, tools, extractor = _try_setup()
    from disclosure_rag.agent.ask import ask

    result = ask(
        client, tools, "삼성전자 단일판매공급계약체결 공시 정정 전후로 뭐가 바뀌었어?",
        entity_extractor=extractor, max_iterations=6,
    )
    assert result.trace.iterations >= 2, "다중 턴 tool-calling 이 실행되지 않음 (회귀 의심)"
    assert "테슬라" in result.answer or "Tesla" in result.answer
    report_ids_cited = {c.report_id for c in result.evidence_pack.citations}
    assert "exchange_20250728800035" in report_ids_cited
    assert "exchange_20250731800028" in report_ids_cited


def test_validator_catches_hallucinated_citation():
    """실측 재현: 답변이 evidence 에 없는 report_id 를 인용하면 has_citation=False 가 돼야 한다."""
    from disclosure_rag.agent.evidence import EvidencePack

    pack = EvidencePack(question="q", prompt_text="[EVIDENCE 1]\n내용: 영업이익 100억원\nreport_id: real_doc_1\n", citations=[])
    answer = "영업이익은 100억원입니다.\n근거: report_id(fake_doc_999)"
    from disclosure_rag.entity.entity_extractor import ExtractedEntities

    entities = ExtractedEntities(raw_query="q")
    result = validate_answer(answer, pack, entities)
    assert result.has_citation is False  # citations 리스트가 비어있으므로


def test_validator_flags_ungrounded_number():
    from disclosure_rag.agent.evidence import EvidencePack
    from disclosure_rag.entity.entity_extractor import ExtractedEntities

    pack = EvidencePack(question="q", prompt_text="[EVIDENCE 1]\n내용: 영업이익 100000원\nreport_id: r1\nchunk_id: c1\n", citations=[])
    answer = "영업이익은 999999999원입니다. 근거: r1"
    result = validate_answer(answer, pack, ExtractedEntities(raw_query="q"))
    assert not result.numbers_grounded
    assert "999999999" in result.ungrounded_numbers


def test_validator_ignores_approx_paren_restatement():
    """회귀(2026-08-16, 회사 일반화 스모크테스트): "7,661,584백만원 (약 7조
    6,615억원)"처럼 같은 숫자를 조/억 단위로 다시 풀어 쓴 괄호 안 숫자가
    evidence 원문과 문자 그대로 안 겹친다는 이유로 "근거 없는 숫자"로 오탐됐다.
    "(약 ...)" 괄호는 근사 재표기이므로 grounding 검사에서 제외해야 한다."""
    from disclosure_rag.agent.evidence import EvidencePack
    from disclosure_rag.entity.entity_extractor import ExtractedEntities

    pack = EvidencePack(
        question="q", prompt_text="[EVIDENCE 1]\n내용: 매출액 7,661,584백만원\nreport_id: r1\nchunk_id: c1\n",
        citations=[],
    )
    answer = "매출액은 7,661,584백만원(약 7조 6,615억원)입니다. 근거: r1"
    result = validate_answer(answer, pack, ExtractedEntities(raw_query="q"))
    assert result.numbers_grounded, f"괄호 안 재표기가 오탐됨: {result.ungrounded_numbers}"


def test_validator_has_citation_from_correction_history_tool_result():
    """회귀(2026-08-16): get_correction_history 만 호출돼 evidence_pack.citations
    가 비어있는 경우(search_disclosures 를 안 씀), 답변이 tool 결과의 report_id
    를 정확히 인용했는데도 무조건 has_citation=False 로 잡혔다."""
    from disclosure_rag.agent.evidence import EvidencePack
    from disclosure_rag.entity.entity_extractor import ExtractedEntities

    pack = EvidencePack(
        question="q", prompt_text="[TOOL RESULT]\nget_correction_history: ...\n",
        citations=[],
        tool_results_summary=[{
            "tool": "get_correction_history", "arguments": {},
            "result": {"correction_groups": [{"chain": [{"doc_id": "major_20250519000120"}]}]},
        }],
    )
    answer = "정정이 4번 있었습니다.\n근거: report_id(major_20250519000120)"
    result = validate_answer(answer, pack, ExtractedEntities(raw_query="q"))
    assert result.has_citation is True


class _StubHCXClient:
    """agent_loop 의 duplicate-tool-call 방지 로직을 실제 API 없이 검증하기
    위한 스텁 — 스크립트된 응답을 순서대로 반환한다."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)

    def chat(self, messages, *, tools=None, **kwargs):
        return self._responses.pop(0)


def test_agent_loop_skips_redundant_identical_tool_calls():
    """회귀(2026-08-16, 회사 일반화 스모크테스트): "몇 건이야?" 같은 카운팅
    질문이 calculation route 로 오분류되면, agent 가 calculate_cagr 을 완전히
    동일한(무의미한) 인자로 여러 번 연속 호출하다 포기하는 경우가 실측 재현됐다
    (n_years=0 처럼 이미 실패가 확정된 입력을 계속 재시도). 이름+인자가 완전히
    같은 tool 호출은 실제로 재실행하지 않아야 한다."""
    from disclosure_rag.agent.agent_loop import run_agent_loop
    from disclosure_rag.agent.tools import ToolDef

    call_count = {"n": 0}

    def handler(x: int) -> dict:
        call_count["n"] += 1
        return {"value": x}

    dummy_tool = ToolDef(
        name="dummy_tool", description="테스트용",
        parameters={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        handler=handler,
    )
    repeated_call_msg = {
        "role": "assistant", "content": "",
        "toolCalls": [{"id": "id", "type": "function", "function": {"name": "dummy_tool", "arguments": {"x": 5}}}],
    }
    responses = [dict(repeated_call_msg) for _ in range(3)] + [{"role": "assistant", "content": "완료.", "toolCalls": None}]
    client = _StubHCXClient(responses)
    extractor = EntityExtractor(corpus_root=CORPUS_ROOT, metric_terms_path=CONFIG_ROOT / "metric_terms.txt")

    trace = run_agent_loop(client, [dummy_tool], "테스트 질문", entity_extractor=extractor, max_iterations=6)

    assert len(trace.tool_calls) == 3, "호출 기록 자체는 투명하게 3번 다 남아야 함"
    assert call_count["n"] == 1, f"handler 가 {call_count['n']}번 실제 실행됨 — 중복 방지가 동작 안 함"
    assert "이미" in trace.tool_calls[1].result.get("note", ""), "2번째부터는 중복 안내가 붙어야 함"
