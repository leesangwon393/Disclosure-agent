"""Reranker (§54): Hybrid Retrieval Top-N 후보를 Query+Candidate 쌍으로 정밀
재평가한다. optional 로 만들어 "Hybrid only" vs "Hybrid + Reranker" 비교가
가능하게 한다 (§74)."""

from __future__ import annotations

from typing import Protocol

from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.common.device import pick_device

Candidates = list[tuple[ChunkSchema, float]]


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: Candidates, *, top_k: int = 5) -> Candidates: ...


class CrossEncoderReranker:
    name = "bge-reranker-v2-m3"

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        # 회귀 발견: max_length 를 안 주면 CrossEncoder 가 truncation 없이 그대로
        # 토크나이즈한다. corpus 에 드물게 존재하는 매우 긴 chunk(malformed XML
        # 표 잔여 케이스, 최대 26,027자 확인됨)를 만나면 quadratic attention
        # 비용 때문에 reranking 1건이 수십 분씩 걸려 실측으로 확인됨. 512 토큰
        # 이면 chunk 의 핵심 내용은 대부분 포함되므로 정확도 손실은 미미하다.
        # Apple Silicon 에서 cuda 로 잡히지 않도록 명시한다
        self._model = CrossEncoder(model_name, device=pick_device(device), max_length=max_length)

    def rerank(self, query: str, candidates: Candidates, *, top_k: int = 5) -> Candidates:
        if not candidates:
            return []
        pairs = [(query, c.text) for c, _ in candidates]
        scores = self._model.predict(pairs)
        reranked = sorted(
            zip((c for c, _ in candidates), scores), key=lambda pair: pair[1], reverse=True,
        )
        return [(c, float(s)) for c, s in reranked[:top_k]]


class NoOpReranker:
    """Reranker 를 끈 baseline (Hybrid only) 비교용."""

    name = "none"

    def rerank(self, query: str, candidates: Candidates, *, top_k: int = 5) -> Candidates:
        return candidates[:top_k]
