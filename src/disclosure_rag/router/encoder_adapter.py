"""우리 EmbeddingProvider(§2, Phase 9) 를 aurelio-labs semantic-router 의
DenseEncoder 인터페이스에 맞게 감싼다. Router 도 embedding 기반 컴포넌트지만,
LLM 이 아니므로 HCX Agent 와는 역할이 분리된 별도 컴포넌트다 (§37)."""

from __future__ import annotations

from pydantic import PrivateAttr
from semantic_router.encoders import DenseEncoder

from disclosure_rag.retrieval.embeddings import EmbeddingProvider


class ProviderBackedEncoder(DenseEncoder):
    _provider: EmbeddingProvider = PrivateAttr()

    def __init__(self, provider: EmbeddingProvider, **kwargs):
        super().__init__(name=f"provider:{provider.name}", **kwargs)
        self._provider = provider

    def __call__(self, docs: list[str]) -> list[list[float]]:
        return self._provider.embed(docs)
