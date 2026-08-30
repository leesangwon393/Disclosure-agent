"""Fusion 계약 테스트 — RRF 가 왜 top-5 에서 손해였는지를 코드로 고정한다.

실측 배경: Stage 4 승자는 normalized_weighted 였는데 구현이 저장소에 없었고
프로덕션은 RRF 를 쓰고 있었다(R@10 0.860 vs 0.900). RRF 는 점수를 버리고 순위만
쓰기 때문에, BM25 가 압도적으로 확신하는 1등이 Dense 의 애매한 1등에 밀린다.
"""
from __future__ import annotations

from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.retrieval.fusion import (
    normalized_weighted_fusion,
    reciprocal_rank_fusion,
)


def _c(cid: str) -> ChunkSchema:
    return ChunkSchema(chunk_id=cid, report_id="r", text=cid, raw_text=cid)


A, B, C, D, E = (_c(x) for x in "ABCDE")


def test_weighted_fusion_keeps_a_confident_bm25_hit_on_top():
    """BM25 가 압도적 점수로 1등을 준 문서가 top-1 을 지켜야 한다.

    실패 시나리오(RRF): 두 리스트에 **모두** 등장하기만 하면 순위 합이 커져서,
    한쪽이 압도적으로 확신하는 단독 1등을 이긴다. 공시에서는 이게 곧
    "BM25 가 접수번호/계약금액으로 정확히 찾은 문서가, Dense 가 애매하게 올린
    비슷비슷한 정정본에 밀리는" 상황이다.
    """
    bm25 = [(A, 42.0), (B, 1.1)]     # A 가 압도적 1등, B 는 간신히 2등
    dense = [(B, 0.71), (D, 0.70)]   # B 가 1등이지만 D 와 거의 동점(= 정보량 낮음)

    weighted = normalized_weighted_fusion({"bm25": bm25, "dense": dense},
                                          weights={"bm25": 0.6, "dense": 0.4})
    assert weighted[0][0].chunk_id == "A", "가중 융합이 확신 있는 BM25 1등을 지키지 못했다"

    # RRF 는 점수를 버리므로 "두 리스트에 다 나온" B 가 A 를 이긴다.
    rrf_scores = {c.chunk_id: s for c, s in reciprocal_rank_fusion([bm25, dense])}
    assert rrf_scores["B"] > rrf_scores["A"], (
        "이 픽스처에서 RRF 가 A 를 밀어내는 것이 정상이다 — 그래서 weighted 로 바꿨다."
    )


def test_all_equal_scores_are_neutralized_not_maximized():
    """한 retriever 의 점수가 전부 같으면 순위 정보가 없다.
    1.0 으로 정규화하면 그 리스트가 통째로 최고점이 되어버리므로 0.5 로 중립화한다."""
    flat = [(A, 5.0), (B, 5.0)]
    other = [(C, 1.0), (A, 0.0)]
    fused = dict((c.chunk_id, s) for c, s in
                 normalized_weighted_fusion({"bm25": flat, "dense": other},
                                            weights={"bm25": 0.5, "dense": 0.5}))
    assert fused["A"] == fused["B"], "동점 리스트 안에서는 순위 차이가 생기면 안 된다"
    assert fused["C"] > fused["B"], "정보가 있는 쪽(other)의 1등이 이겨야 한다"


def test_missing_retriever_falls_back_cleanly():
    """Dense 인덱스가 아직 없어도(=임베딩 전) BM25 단독으로 정상 동작해야 한다.
    임베딩 여부 결정을 뒤로 미룰 수 있는 근거."""
    bm25 = [(A, 3.0), (B, 2.0)]
    fused = normalized_weighted_fusion({"bm25": bm25, "dense": []})
    assert [c.chunk_id for c, _ in fused] == ["A", "B"]


def test_weights_are_renormalized_over_active_retrievers():
    """가중치 합이 1이 아니어도, 일부 retriever 가 비어도 결과가 스케일에 휘둘리지 않는다."""
    bm25 = [(A, 1.0), (B, 0.0)]
    f1 = normalized_weighted_fusion({"bm25": bm25}, weights={"bm25": 0.6, "dense": 0.4})
    f2 = normalized_weighted_fusion({"bm25": bm25}, weights={"bm25": 6.0, "dense": 4.0})
    assert [(c.chunk_id, round(s, 6)) for c, s in f1] == [(c.chunk_id, round(s, 6)) for c, s in f2]


def test_top_k_by_route_is_above_the_losing_k():
    """route 별 top_k 가 하이브리드가 지던 k=5 보다 커야 한다."""
    from disclosure_rag.agent.tools import DEFAULT_TOP_K, TOP_K_BY_ROUTE, top_k_for_route

    assert DEFAULT_TOP_K > 5
    assert all(v > 5 for v in TOP_K_BY_ROUTE.values())
    assert top_k_for_route("multi_compare") > top_k_for_route("single_lookup")
    assert top_k_for_route("존재하지_않는_route") == DEFAULT_TOP_K
