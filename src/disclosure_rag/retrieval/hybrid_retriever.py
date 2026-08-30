"""§52 Hybrid Retrieval: Metadata Filter -> (BM25 + Dense + Sparse) -> Fusion -> (선택) Rerank.

=== KIM 브랜치 변경점 (2026-08-23) ===
1. 기본 fusion 을 RRF -> normalized_weighted 로 바꿨다 (Stage 4 승자, fusion.py 참조).
2. sparse(BGE-M3 lexical) 경로를 추가했다. 임베딩 1회로 dense/sparse/colbert 가
   같은 forward pass 에서 나오므로 **추가 연산 비용이 거의 없다**.
3. reranker 자리에 cross-encoder 뿐 아니라 ColBERT late-interaction 도 꽂을 수 있게 했다.
   표·숫자를 벡터 하나로 뭉개는 문제의 정면 해결책이고, 이미 BGE-M3 안에 들어 있다.

Dense/Sparse/Reranker 를 빼면 자동으로 "BM25 only" 로 동작한다 — 임베딩 없이도
파이프라인 전체가 굴러가므로 임베딩 여부 결정을 뒤로 미룰 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.retrieval.bm25_retriever import BM25Retriever
from disclosure_rag.retrieval.fusion import (
    DEFAULT_WEIGHTS,
    normalized_weighted_fusion,
    reciprocal_rank_fusion,
)
from disclosure_rag.retrieval.metadata_filter import RetrievalFilter


class _Searcher(Protocol):
    def search(self, query: str, *, k: int, flt: RetrievalFilter | None) -> list[tuple[ChunkSchema, float]]: ...


class _Reranker(Protocol):
    def rerank(self, query: str, candidates: list[tuple[ChunkSchema, float]], *, top_k: int)\
        -> list[tuple[ChunkSchema, float]]: ...


@dataclass(frozen=True)
class HybridSearchTrace:
    """Unstructured 채널 검색 결과와 채널별 진단.

    Facts는 이 타입에 들어올 수 없다. 점수 없는 SQLite 조회를
    실수로 fusion에 섞는 것을 인터페이스 수준에서 막는다.
    """

    results: list[tuple[ChunkSchema, float]]
    channel_counts: dict[str, int] = field(default_factory=dict)
    fused_count: int = 0
    reranked: bool = False


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: _Searcher | None = None,
        reranker: _Reranker | None = None,
        sparse: _Searcher | None = None,
        *,
        fusion: str = "weighted",           # "weighted" | "rrf"
        weights: dict[str, float] | None = None,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.fusion = fusion
        self.weights = dict(weights or DEFAULT_WEIGHTS)

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        flt: RetrievalFilter | None = None,
        candidate_k: int = 50,
        rrf_k: int = 60,
        rerank_top_n: int = 50,
    ) -> list[tuple[ChunkSchema, float]]:
        return self.search_with_trace(
            query, k=k, flt=flt, candidate_k=candidate_k, rrf_k=rrf_k,
            rerank_top_n=rerank_top_n,
        ).results

    def search_with_trace(
        self,
        query: str,
        *,
        k: int = 10,
        flt: RetrievalFilter | None = None,
        candidate_k: int = 50,
        rrf_k: int = 60,
        rerank_top_n: int = 50,
    ) -> HybridSearchTrace:
        """BM25/Dense/Sparse만 검색·융합하고 채널 진단을 돌려준다."""
        named: dict[str, list[tuple[ChunkSchema, float]]] = {
            "bm25": self.bm25.search(query, k=candidate_k, flt=flt)
        }
        if self.dense is not None:
            named["dense"] = self.dense.search(query, k=candidate_k, flt=flt)
        if self.sparse is not None:
            named["sparse"] = self.sparse.search(query, k=candidate_k, flt=flt)

        pool = max(k, rerank_top_n if self.reranker is not None else k)
        if len(named) == 1:
            fused = named["bm25"][:pool]
        elif self.fusion == "rrf":
            fused = reciprocal_rank_fusion(list(named.values()), k=rrf_k, top_k=pool)
        else:
            fused = normalized_weighted_fusion(named, weights=self.weights, top_k=pool)

        fused_count = len(fused)
        if self.reranker is not None:
            results = self.reranker.rerank(query, fused, top_k=k)
        else:
            results = fused[:k]
        return HybridSearchTrace(
            results=results,
            channel_counts={name: len(rows) for name, rows in named.items()},
            fused_count=fused_count,
            reranked=self.reranker is not None,
        )
