"""Phase 13+14 회귀 테스트: Semantic Router + Eval.

BGE-M3 로딩이 필요해 느리다. CPU 경합 시 매우 느려질 수 있으므로 seperate 로 표시.
"""

from __future__ import annotations

import pytest

from disclosure_rag.router.eval import evaluate_router, threshold_sweep
from disclosure_rag.router.eval_dataset import EVAL_SET
from disclosure_rag.router.routes import ROUTE_UTTERANCES
from disclosure_rag.router.semantic_router_wrapper import NoRouter, SemanticRouterAdapter


def _try_build_router(threshold: float = 0.5):
    try:
        from disclosure_rag.retrieval.embeddings import BgeM3EmbeddingProvider

        provider = BgeM3EmbeddingProvider(device="cpu")
        return SemanticRouterAdapter(provider, threshold=threshold)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"BGE-M3 모델 로딩 불가: {e}")


def test_routes_cover_six_intents():
    assert set(ROUTE_UTTERANCES.keys()) == {
        "single_lookup", "correction_analysis", "multi_compare",
        "calculation", "ownership_analysis", "event_analysis",
    }
    for name, utterances in ROUTE_UTTERANCES.items():
        assert len(utterances) >= 10, f"{name}: utterance 수가 너무 적음"


def test_eval_set_disjoint_from_training_utterances():
    """§48: 등록 utterance 와 평가셋은 반드시 분리돼야 한다."""
    all_training = {u for utts in ROUTE_UTTERANCES.values() for u in utts}
    eval_queries = {ex.query for ex in EVAL_SET}
    assert not (all_training & eval_queries)


def test_no_router_always_falls_back():
    router = NoRouter()
    result = router.route("[COMPANY] 영업이익 얼마야?")
    assert result.route is None


@pytest.mark.slow
def test_semantic_router_routes_clear_single_lookup_query():
    router = _try_build_router(threshold=0.3)
    result = router.route("[COMPANY] 영업이익 알려줘")
    assert result.route == "single_lookup"


@pytest.mark.slow
def test_semantic_router_routes_clear_correction_query():
    router = _try_build_router(threshold=0.3)
    result = router.route("[COMPANY] 정정공시에서 뭐가 바뀌었어?")
    assert result.route == "correction_analysis"


@pytest.mark.slow
def test_semantic_router_high_threshold_increases_fallback():
    """threshold 를 극단적으로 높이면 fallback rate 가 올라가야 한다 (sanity check)."""
    router = _try_build_router(threshold=0.3)
    low_report = evaluate_router(router, EVAL_SET)
    router.set_threshold(0.99)
    high_report = evaluate_router(router, EVAL_SET)
    assert high_report.fallback_rate >= low_report.fallback_rate


@pytest.mark.slow
def test_router_eval_report_has_expected_fields():
    router = _try_build_router(threshold=0.3)
    report = evaluate_router(router, EVAL_SET)
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.macro_f1 <= 1.0
    assert 0.0 <= report.fallback_rate <= 1.0
    assert report.n == len(EVAL_SET)
    assert len(report.confusion_matrix) == len(report.labels)


@pytest.mark.slow
def test_threshold_sweep_runs_without_rebuilding():
    router = _try_build_router(threshold=0.3)
    results = threshold_sweep(router, EVAL_SET, thresholds=[0.2, 0.5, 0.8])
    assert set(results.keys()) == {0.2, 0.5, 0.8}
    for t, report in results.items():
        assert report.n == len(EVAL_SET)
