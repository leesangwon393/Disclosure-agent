"""experiments/metrics.py 회귀 테스트. Stage 1~5 전체가 이 모듈에 의존하므로
잘못되면 모든 실험 결과가 잘못된다 — 특히 NDCG>1 버그(report-level gold 인데
chunk 단위로 relevance 를 중복 카운트해서 발생)를 회귀로 고정한다."""

from __future__ import annotations

from disclosure_rag.experiments.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank


def test_ndcg_never_exceeds_1_even_with_many_chunks_from_same_gold_report():
    """회귀: gold report 하나에서 chunk 10개가 전부 top-10 에 뽑히는 극단적 경우에도
    NDCG<=1 이어야 한다 (예전엔 1.26 같은 값이 나왔음)."""
    retrieved = ["r1"] * 10
    gold = {"r1"}
    assert ndcg_at_k(retrieved, gold, 10) <= 1.0
    assert ndcg_at_k(retrieved, gold, 10) == 1.0  # r1 이 1위이므로 완벽한 랭킹


def test_ndcg_perfect_ranking_is_1():
    retrieved = ["r1", "r2", "r3", "x", "x"]
    gold = {"r1", "r2", "r3"}
    assert ndcg_at_k(retrieved, gold, 5) == 1.0


def test_ndcg_no_relevant_is_0():
    retrieved = ["x", "y", "z"]
    gold = {"r1"}
    assert ndcg_at_k(retrieved, gold, 5) == 0.0


def test_recall_at_k_fraction_of_gold_covered():
    retrieved = ["r1", "x", "r2", "y"]
    gold = {"r1", "r2", "r3"}
    assert recall_at_k(retrieved, gold, 10) == 2 / 3


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "r1", "y"], {"r1"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"r1"}) == 0.0


def test_hit_at_k():
    assert hit_at_k(["x", "r1"], {"r1"}, 1) == 0.0
    assert hit_at_k(["x", "r1"], {"r1"}, 2) == 1.0
