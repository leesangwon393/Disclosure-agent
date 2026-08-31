"""Step 6 Dual Channel 회귀 테스트.

핵심 불변식은 두 가지다.
1) expected_fields에 실제 정형 항목이 있으면 HCX 선택과 무관하게 Facts를 조회한다.
2) Facts는 BM25/Dense/Sparse 순위에 섞지 않아 기존 context precision을 건드리지 않는다.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from disclosure_rag.agent.dual_channel import DualChannelRetriever
from disclosure_rag.agent.query_plan import QueryPlan
from disclosure_rag.agent.tools import (
    make_planned_search_disclosures_tool,
    run_dual_channel_search,
)
from disclosure_rag.agent.version_dedup import deduplicate_scored
from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.correction.correction_graph_builder import CorrectionRecord
from disclosure_rag.facts.store import FactStore
from disclosure_rag.retrieval.hybrid_retriever import HybridRetriever, HybridSearchTrace


def _chunk(chunk_id: str, report_id: str, **kwargs) -> ChunkSchema:
    base = {
        "text": chunk_id,
        "raw_text": chunk_id,
        "company": "삼성전자",
        "report_type": "exchange",
        "is_latest": True,
    }
    base.update(kwargs)
    return ChunkSchema(chunk_id=chunk_id, report_id=report_id, **base)


def _fact(report_id: str = "r1", item: str = "계약금액") -> dict:
    return {
        "doc_id": report_id,
        "chunk_id": report_id + "_fact",
        "company": "삼성전자",
        "key_norm": item,
        "value_text": "100억원",
        "value_num": 10_000_000_000,
        "value_unit": "원",
        "value_date": None,
        "report_name": "단일판매ㆍ공급계약체결",
        "filing_date": "20250102",
        "period": "2024-12",
        "is_correction": False,
        "is_latest": True,
        "correction_group_id": report_id,
        "section_path": ["계약 내용"],
    }


class FakeUnstructured:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def search_with_trace(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return HybridSearchTrace(
            results=list(self.results),
            channel_counts={"bm25": 8, "dense": 8, "sparse": 8},
            fused_count=len(self.results),
        )


class FakeFacts:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.calls = []

    def distinct_keys(self, **_kwargs):
        return [("계약금액", 10), ("최근매출액", 9)]

    def lookup(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.rows)


def _plan(**kwargs) -> QueryPlan:
    base = dict(
        answer_mode="closed",
        task="lookup",
        companies=["삼성전자"],
        report_types=["exchange"],
        expected_fields=["계약금액"],
    )
    base.update(kwargs)
    return QueryPlan(**base)


def test_structured_expected_field_always_executes_facts_and_logs(caplog):
    facts = FakeFacts([_fact()])
    dual = DualChannelRetriever(
        FakeUnstructured([(_chunk("c1", "r1"), 0.8)]), facts,
        available_fact_keys=["계약금액"],
    )

    with caplog.at_level(logging.INFO):
        result = dual.search("계약금액은?", _plan())

    assert result.facts_executed is True
    assert facts.calls and facts.calls[0]["key"] == "계약금액"
    assert "facts_executed=True" in caplog.text
    assert result.reports[0].report_id == "r1"
    assert result.reports[0].channels == ["unstructured", "facts"]


def test_unknown_expected_field_does_not_run_facts():
    facts = FakeFacts([_fact()])
    dual = DualChannelRetriever(
        FakeUnstructured([]), facts, available_fact_keys=["계약금액"],
    )

    result = dual.search("해지 조건은?", _plan(expected_fields=["해지조건"]))

    assert result.facts_executed is False
    assert facts.calls == []
    assert result.facts == []


def test_facts_are_separate_and_never_receive_a_fusion_score():
    dual = DualChannelRetriever(
        FakeUnstructured([(_chunk("c1", "r1"), 0.75)]),
        FakeFacts([_fact()]), available_fact_keys=["계약금액"],
    )

    payload = dual.search("계약금액", _plan()).to_dict()

    assert payload["unstructured_results"][0]["score"] == 0.75
    assert "score" not in payload["facts"][0]
    assert payload["reports"][0]["channels"] == ["unstructured", "facts"]


def test_unstructured_ranking_and_context_precision_input_are_unchanged():
    """Dual Channel은 Step 5 dedup 이후의 본문 목록을 그대로 보존한다."""
    raw = [
        (_chunk("c-final", "final", correction_group_id="original",
                correction_order=1, is_latest=True), 0.9),
        (_chunk("c-old", "original", correction_group_id="original",
                correction_order=0, is_latest=False), 0.8),
        (_chunk("c-other", "other"), 0.7),
    ]
    baseline, _ = deduplicate_scored(raw, "latest_only")
    dual = DualChannelRetriever(
        FakeUnstructured(raw), FakeFacts([_fact("fact-only")]),
        available_fact_keys=["계약금액"],
    )

    result = dual.search("계약금액", _plan(latest_policy="latest_only"))

    assert [(c.chunk_id, s) for c, s in result.unstructured_results] == [
        (c.chunk_id, s) for c, s in baseline
    ]
    # Facts 전용 문서는 report bundle 뒤에 붙지만 본문 평가 대상에는 들어가지 않는다.
    assert "fact-only" not in [c.report_id for c, _ in result.unstructured_results]
    assert "fact-only" in [r.report_id for r in result.reports]


def test_structured_and_unstructured_really_run_in_parallel():
    facts_started = threading.Event()

    class WaitingUnstructured(FakeUnstructured):
        def search_with_trace(self, query, **kwargs):
            assert facts_started.wait(timeout=1), "Facts가 끝날 때까지 검색 worker가 살아 있어야 함"
            return super().search_with_trace(query, **kwargs)

    class SignallingFacts(FakeFacts):
        def lookup(self, **kwargs):
            facts_started.set()
            return super().lookup(**kwargs)

    dual = DualChannelRetriever(
        WaitingUnstructured([(_chunk("c1", "r1"), 1.0)]),
        SignallingFacts([_fact()]), available_fact_keys=["계약금액"],
    )
    assert dual.search("계약금액", _plan()).reports


def test_period_filters_on_reporting_period_not_filing_date():
    facts = FakeFacts([_fact()])
    dual = DualChannelRetriever(
        FakeUnstructured([]), facts, available_fact_keys=["계약금액"],
    )

    dual.search("2024년 계약금액", _plan(periods=["2024"]))

    assert facts.calls[0]["period"] == "2024"
    assert "date_from" not in facts.calls[0]


def test_correction_graph_is_merged_by_report_id_even_without_search_hit():
    record = CorrectionRecord(
        doc_id="correction-only", correction_group_id="original",
        correction_order=1, is_correction=True, is_latest=True,
        resolution_source="rule",
    )
    manifest = [SimpleNamespace(
        doc_id="correction-only", corp_name="삼성전자", doc_group="exchange",
        report_nm="[기재정정]단일판매ㆍ공급계약체결", rcept_dt="20250102",
    )]
    dual = DualChannelRetriever(
        FakeUnstructured([]), FakeFacts([]), correction_index={record.doc_id: record},
        manifest=manifest, available_fact_keys=["계약금액"],
    )

    result = dual.search("최초와 최종을 비교", _plan(latest_policy="first_and_final"))

    assert [r.report_id for r in result.reports] == ["correction-only"]
    assert result.reports[0].channels == ["correction_graph"]


def test_tools_entrypoint_uses_plan_without_exposing_fields_to_hcx():
    dual = DualChannelRetriever(
        FakeUnstructured([(_chunk("c1", "r1"), 1.0)]),
        FakeFacts([_fact()]), available_fact_keys=["계약금액"],
    )
    plan = _plan()

    direct = run_dual_channel_search(dual, "계약금액", plan)
    tool = make_planned_search_disclosures_tool(dual, plan)
    via_tool = tool.handler(query="계약금액")

    assert direct["diagnostics"]["facts_executed"] is True
    assert via_tool["diagnostics"]["facts_executed"] is True
    assert "expected_fields" not in tool.parameters["properties"]
    assert "latest_policy" not in tool.parameters["properties"]


def test_fact_store_connection_is_safe_across_request_threads(tmp_path):
    store = FactStore(tmp_path / "facts.sqlite")
    store.conn.execute(
        "INSERT INTO facts(doc_id,key,key_norm,value_text,period) VALUES(?,?,?,?,?)",
        ("r1", "계약금액", "계약금액", "100억원", "2024-12"),
    )
    store.conn.commit()

    with ThreadPoolExecutor(max_workers=1) as pool:
        rows = pool.submit(store.lookup, key="계약금액", period="2024").result()

    assert rows[0]["doc_id"] == "r1"
    store.close()


def test_fact_period_falls_back_to_filing_year_when_document_has_no_period(tmp_path):
    store = FactStore(tmp_path / "event-facts.sqlite")
    store.conn.execute(
        "INSERT INTO facts(doc_id,key,key_norm,value_text,period,filing_date) "
        "VALUES(?,?,?,?,?,?)",
        ("event", "계약금액", "계약금액", "50억원", None, "20240401"),
    )
    store.conn.commit()

    assert store.lookup(key="계약금액", period="2024")[0]["doc_id"] == "event"
    assert store.lookup(key="계약금액", period="2024-12")[0]["doc_id"] == "event"
    assert store.lookup(key="계약금액", period="2023") == []
    store.close()


class _StaticSearcher:
    def __init__(self, rows):
        self.rows = rows

    def search(self, _query, *, k, flt):
        return self.rows[:k]


def test_hybrid_trace_contains_only_ranked_search_channels():
    chunks = [_chunk("c1", "r1"), _chunk("c2", "r2")]
    hybrid = HybridRetriever(
        _StaticSearcher([(chunks[0], 2.0), (chunks[1], 1.0)]),
        dense=_StaticSearcher([(chunks[1], 0.9)]),
        sparse=_StaticSearcher([(chunks[0], 0.8)]),
        fusion="rrf",
    )

    trace = hybrid.search_with_trace("계약", k=2)
    normal = hybrid.search("계약", k=2)

    assert trace.results == normal
    assert set(trace.channel_counts) == {"bm25", "dense", "sparse"}
    assert "facts" not in trace.channel_counts


def test_hybrid_trace_records_per_stage_timings():
    """단계별 소요 시간이 실제로 기록되는지.

    '검색이 느리다'까지는 알아도 어느 단계가 느린지 모르면 추측으로 고치게
    된다. 이 계측이 빠지면 그 상태로 되돌아간다.
    """
    chunks = [_chunk("c1", "r1"), _chunk("c2", "r2")]
    hybrid = HybridRetriever(
        _StaticSearcher([(chunks[0], 2.0), (chunks[1], 1.0)]),
        dense=_StaticSearcher([(chunks[1], 0.9)]),
        sparse=_StaticSearcher([(chunks[0], 0.8)]),
        fusion="rrf",
    )

    trace = hybrid.search_with_trace("계약", k=2)

    assert {"bm25", "dense", "sparse", "fusion", "total"} <= set(trace.timings_ms)
    assert "rerank" not in trace.timings_ms          # reranker 가 없으니 없어야 한다
    assert all(value >= 0 for value in trace.timings_ms.values())


def test_hybrid_trace_timings_include_rerank_when_present():
    chunks = [_chunk("c1", "r1"), _chunk("c2", "r2")]

    class _Reranker:
        def rerank(self, _query, candidates, *, top_k):
            return candidates[:top_k]

    hybrid = HybridRetriever(
        _StaticSearcher([(chunks[0], 2.0), (chunks[1], 1.0)]),
        reranker=_Reranker(),
    )

    trace = hybrid.search_with_trace("계약", k=1)

    assert "rerank" in trace.timings_ms
    assert trace.reranked is True
