"""순위 품질 지표 — 손으로 계산 가능한 케이스만 쓴다.

지표 코드가 틀리면 '개선됐다'는 착각을 만든다. 실제로 ndcg_at_k 는
report-level dedup 을 빠뜨려 NDCG>1 이 나온 전력이 있다(metrics.py docstring).
그래서 여기서는 **경계값과 상한 위반**을 집중적으로 본다.
"""
from __future__ import annotations

import math

import pytest

from disclosure_rag.experiments.metrics import (
    average_precision_at_k,
    first_relevant_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# --------------------------------------------------------------------------- precision@k

def test_precision_all_relevant():
    assert precision_at_k(["A", "A", "A"], {"A"}, 3) == 1.0


def test_precision_none_relevant():
    assert precision_at_k(["X", "Y", "Z"], {"A"}, 3) == 0.0


def test_precision_counts_chunks_not_documents():
    """같은 gold report 의 chunk 가 여러 칸을 차지해도 그 칸들은 쓸모 있는
    근거다 — 낭비로 세면 안 된다."""
    assert precision_at_k(["A", "A", "X", "Y"], {"A"}, 4) == 0.5


def test_precision_shorter_than_k_uses_actual_length():
    """검색 결과가 k 보다 적으면 실제 길이로 나눈다(0 으로 나누지 않는다)."""
    assert precision_at_k(["A"], {"A"}, 10) == 1.0


def test_precision_empty_is_zero():
    assert precision_at_k([], {"A"}, 10) == 0.0


# --------------------------------------------------------------------------- average precision

def test_ap_perfect_ranking():
    assert average_precision_at_k(["A", "B"], {"A", "B"}, 10) == 1.0


def test_ap_rewards_early_hits():
    early = average_precision_at_k(["A", "X", "X"], {"A"}, 3)
    late = average_precision_at_k(["X", "X", "A"], {"A"}, 3)
    assert early == 1.0
    assert late == pytest.approx(1 / 3)
    assert early > late


def test_ap_deduplicates_by_report():
    """dedup 이 없으면 [A,A,A] 가 3.0 이 되어 상한 1 을 넘는다."""
    assert average_precision_at_k(["A", "A", "A"], {"A"}, 3) == 1.0


def test_ap_never_exceeds_one_with_many_duplicate_chunks():
    retrieved = ["A"] * 20
    assert average_precision_at_k(retrieved, {"A"}, 20) <= 1.0


def test_ap_two_golds_interleaved():
    # i=2 에서 A 처음(1/2), i=4 에서 B 처음(2/4) -> (0.5+0.5)/2 = 0.5
    assert average_precision_at_k(["X", "A", "X", "B"], {"A", "B"}, 4) == pytest.approx(0.5)


def test_ap_partial_coverage_is_penalized():
    """gold 2건 중 1건만 찾으면 완벽한 순위여도 만점이 아니다."""
    assert average_precision_at_k(["A", "X"], {"A", "B"}, 10) == pytest.approx(0.5)


def test_ap_k_truncates():
    """k 밖의 gold 는 세지 않는다."""
    assert average_precision_at_k(["X", "X", "A"], {"A"}, 2) == 0.0


def test_ap_empty_gold_is_zero():
    assert average_precision_at_k(["A"], set(), 10) == 0.0


# --------------------------------------------------------------------------- first_relevant_rank

def test_first_rank_is_one_based():
    assert first_relevant_rank(["A"], {"A"}) == 1


def test_first_rank_finds_earliest():
    assert first_relevant_rank(["X", "Y", "A", "A"], {"A"}) == 3


def test_first_rank_none_when_absent():
    """0 이 아니라 None 이어야 한다 — 0 으로 평균내면 '1등'처럼 보인다."""
    assert first_relevant_rank(["X", "Y"], {"A"}) is None


def test_first_rank_is_consistent_with_mrr():
    retrieved = ["X", "A", "B"]
    gold = {"A"}
    r = first_relevant_rank(retrieved, gold)
    assert r is not None
    assert reciprocal_rank(retrieved, gold) == pytest.approx(1.0 / r)


# --------------------------------------------------------------------------- 기존 지표와의 정합성

def test_recall_and_precision_disagree_on_ranking():
    """이 지표들을 새로 넣는 이유 — recall 은 같은데 순위 품질이 다른 경우를
    구분해야 한다."""
    good = ["A", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    bad = ["X", "X", "X", "X", "X", "X", "X", "X", "X", "A"]
    gold = {"A"}
    assert recall_at_k(good, gold, 10) == recall_at_k(bad, gold, 10) == 1.0
    assert precision_at_k(good, gold, 10) == precision_at_k(bad, gold, 10)  # 둘 다 0.1
    assert average_precision_at_k(good, gold, 10) > average_precision_at_k(bad, gold, 10)
    assert ndcg_at_k(good, gold, 10) > ndcg_at_k(bad, gold, 10)


def test_ndcg_still_bounded_by_one():
    """회귀 방지 — 과거에 dedup 누락으로 NDCG>1 이 나온 적이 있다."""
    assert ndcg_at_k(["A"] * 10, {"A"}, 10) <= 1.0


def test_all_metrics_are_zero_on_empty_retrieval():
    for fn in (precision_at_k, average_precision_at_k, recall_at_k, ndcg_at_k):
        assert fn([], {"A"}, 10) == 0.0
    assert reciprocal_rank([], {"A"}) == 0.0
    assert first_relevant_rank([], {"A"}) is None


def test_ndcg_matches_hand_calculation():
    """gold={A}, A 가 2등 -> DCG = 1/log2(3), IDCG = 1/log2(2) = 1"""
    assert ndcg_at_k(["X", "A"], {"A"}, 10) == pytest.approx(1 / math.log2(3))
