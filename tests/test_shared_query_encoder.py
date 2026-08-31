"""질의 인코딩 공유 — 같은 결과를 절반의 모델 호출로."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from disclosure_rag.retrieval.embeddings import SharedQueryEncoder  # noqa: E402


class _FakeProvider:
    """BGE-M3 대신. 호출 횟수를 센다."""

    name, dim = "fake", 4

    def __init__(self):
        self.calls: list[dict] = []

    def encode_all(self, texts, *, batch_size=32, dense=True, sparse=True, colbert=False, **kw):
        self.calls.append({"texts": list(texts), "dense": dense, "sparse": sparse})
        return {
            "dense_vecs": [[len(t), 1.0, 2.0, 3.0] for t in texts],
            "lexical_weights": [{"1": 0.5, "2": 0.25} for _ in texts],
        }

    def embed(self, texts, *, batch_size=32):
        return [[len(t), 1.0, 2.0, 3.0] for t in texts]


def test_dense_and_sparse_share_one_forward_pass():
    enc = SharedQueryEncoder(_FakeProvider())

    dense = enc.embed_query("삼성전자 매출액")
    lexical = enc.lexical_query("삼성전자 매출액")

    assert dense == [8.0, 1.0, 2.0, 3.0]
    assert lexical == {"1": 0.5, "2": 0.25}
    assert len(enc._provider.calls) == 1          # 두 번 태우지 않는다
    assert enc._provider.calls[0]["dense"] and enc._provider.calls[0]["sparse"]


def test_repeated_query_hits_the_cache():
    enc = SharedQueryEncoder(_FakeProvider())
    enc.embed_query("같은 질문")
    enc.embed_query("같은 질문")
    assert len(enc._provider.calls) == 1
    assert enc.hits == 1 and enc.misses == 1


def test_cache_is_bounded():
    enc = SharedQueryEncoder(_FakeProvider(), cache_size=2)
    for text in ("a", "b", "c"):
        enc.embed_query(text)
    assert len(enc._cache) == 2
    assert "a" not in enc._cache          # 가장 오래된 것이 빠진다


def test_unknown_attributes_fall_through_to_the_provider():
    enc = SharedQueryEncoder(_FakeProvider())
    assert enc.name == "fake" and enc.dim == 4


def test_batch_embedding_is_not_cached():
    """코퍼스 임베딩까지 캐시에 담으면 메모리가 터진다."""
    enc = SharedQueryEncoder(_FakeProvider())
    enc.embed(["a", "b", "c"])
    assert enc._cache == {}
