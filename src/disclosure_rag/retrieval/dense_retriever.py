"""Dense Retriever: EmbeddingProvider + QdrantVectorStore 를 묶은 검색 인터페이스.
BM25Retriever 와 동일한 search() 시그니처를 유지해 Fusion(Phase 10)에서 대칭적으로
다룰 수 있게 한다."""

from __future__ import annotations

from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.retrieval.embeddings import EmbeddingProvider
from disclosure_rag.retrieval.metadata_filter import RetrievalFilter
from disclosure_rag.retrieval.qdrant_store import QdrantVectorStore


class DenseRetriever:
    def __init__(self, chunks: list[ChunkSchema], embedding_provider: EmbeddingProvider, store: QdrantVectorStore):
        self.chunks_by_id = {c.chunk_id: c for c in chunks}
        self.embedding_provider = embedding_provider
        self.store = store

    @classmethod
    def build(
        cls,
        chunks: list[ChunkSchema],
        embedding_provider: EmbeddingProvider,
        store: QdrantVectorStore,
        *,
        batch_size: int = 32,
    ) -> "DenseRetriever":
        texts = [c.text for c in chunks]
        vectors = embedding_provider.embed(texts, batch_size=batch_size)
        store.upsert_chunks(chunks, vectors)
        return cls(chunks, embedding_provider, store)

    def search(
        self, query: str, *, k: int = 10, flt: RetrievalFilter | None = None,
    ) -> list[tuple[ChunkSchema, float]]:
        qvec = self.embedding_provider.embed_query(query)
        pairs = self.store.search(qvec, k=k, flt=flt)
        results = []
        for chunk_id, score in pairs:
            chunk = self.chunks_by_id.get(chunk_id)
            if chunk is not None:
                results.append((chunk, score))
        return results
