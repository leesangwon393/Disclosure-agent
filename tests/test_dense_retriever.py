"""Phase 9 회귀 테스트: BGE-M3 embedding provider + Qdrant vector store.

BGE-M3 모델(~2.3GB) 다운로드/로딩이 필요해 다른 테스트보다 느리다.
network/모델 캐시가 없는 환경에서는 skip 한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from disclosure_rag.chunking.chunk_schema import filter_leaf_chunks
from disclosure_rag.common.manifest_loader import load_manifest
from disclosure_rag.pipeline import build_all_chunks
from disclosure_rag.retrieval.dense_retriever import DenseRetriever
from disclosure_rag.retrieval.metadata_filter import RetrievalFilter
from disclosure_rag.retrieval.qdrant_store import QdrantVectorStore, build_qdrant_filter

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
pytestmark = pytest.mark.skipif(not CORPUS_ROOT.is_dir(), reason="corpus/ 없음")

SAMPLE_DOC_IDS = {
    "periodic_20240312000736",
    "major_20241118000171",
    "exchange_20250728800035",
    "holding_20241025000530",
}


def _try_load_bge_m3():
    try:
        from disclosure_rag.retrieval.embeddings import BgeM3EmbeddingProvider

        return BgeM3EmbeddingProvider(device="cpu")  # MPS 는 다른 프로세스와 동시 사용 시 OOM 발생 확인됨
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"BGE-M3 모델 로딩 불가 (네트워크/캐시 없음): {e}")


@pytest.fixture(scope="module")
def sample_chunks():
    """검색 인덱스에는 leaf chunk 만 넣는다 (parent 는 매우 길어질 수 있어
    그대로 임베딩하면 비정상적으로 느려짐 — 실측으로 확인된 회귀)."""
    manifest = load_manifest(CORPUS_ROOT)
    rows = [r for r in manifest if r.doc_id in SAMPLE_DOC_IDS]
    all_chunks = build_all_chunks(str(CORPUS_ROOT), rows=rows, validate=False)
    return filter_leaf_chunks(all_chunks)


def test_qdrant_filter_translation():
    flt = RetrievalFilter(companies=["삼성전자"], doc_groups=["major"], latest_only=True)
    qf = build_qdrant_filter(flt)
    assert qf is not None
    keys = {c.key for c in qf.must}
    assert {"company", "report_type", "is_latest"} <= keys


def test_qdrant_filter_none_when_empty():
    assert build_qdrant_filter(None) is None
    assert build_qdrant_filter(RetrievalFilter()) is None


def test_qdrant_in_memory_upsert_and_search():
    """실제 embedding 없이 dummy 벡터로 Qdrant store 배선만 검증 (빠름)."""
    from disclosure_rag.chunking.chunk_schema import ChunkSchema

    chunks = [
        ChunkSchema(
            chunk_id="c1", report_id="r1", text="t1", raw_text="t1",
            company="삼성전자", report_type="major", is_correction=False, is_latest=True,
        ),
        ChunkSchema(
            chunk_id="c2", report_id="r2", text="t2", raw_text="t2",
            company="SK하이닉스", report_type="major", is_correction=False, is_latest=True,
        ),
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    store = QdrantVectorStore(dim=2, in_memory=True, collection_name="test")
    store.upsert_chunks(chunks, vectors)

    results = store.search([1.0, 0.0], k=2)
    assert results[0][0] == "c1"

    flt = RetrievalFilter(companies=["SK하이닉스"])
    results_filtered = store.search([1.0, 0.0], k=2, flt=flt)
    assert [r[0] for r in results_filtered] == ["c2"]


@pytest.mark.slow
def test_bge_m3_dense_retriever_finds_relevant_chunk(sample_chunks):
    provider = _try_load_bge_m3()
    store = QdrantVectorStore(dim=provider.dim, in_memory=True, collection_name="test_dense")
    retriever = DenseRetriever.build(sample_chunks, provider, store)

    # semantic query: "R&D 에 얼마 썼어" 는 "연구개발비" 와 표현이 다르지만
    # 의미가 같아야 Dense 가 잡아내야 한다 (§33).
    results = retriever.search("R&D에 얼마나 투자했어?", k=5)
    assert results
    texts = [c.raw_text for c, _ in results]
    assert any("연구개발" in t for t in texts), f"Dense retrieval 이 의미 매칭 실패: {texts}"
