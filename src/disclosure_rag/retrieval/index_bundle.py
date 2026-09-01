"""artifacts 에서 검색기 한 벌을 조립한다.

    artifacts/
      l1/     chunks.jsonl.gz ...        (build_snapshot.py)
      emb/    dense_*.npz, sparse_*.jsonl.gz, colbert_*.npz   (embed_corpus.py)
      facts/  facts.sqlite               (build_facts.py)
      index/  bm25/                      (build_index.py)

설계 원칙: **없는 것은 조용히 건너뛴다.** 임베딩을 아직 안 돌렸으면 dense/sparse 없이
BM25 단독으로 동작한다 — 임베딩 여부 결정을 뒤로 미룰 수 있게 하려는 것이고,
`HybridRetriever(dense=None)` 이 원래 그렇게 설계돼 있다.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.l1 import load_chunks, load_parent_map
from disclosure_rag.retrieval.bm25_retriever import BM25Retriever
from disclosure_rag.retrieval.hybrid_retriever import HybridRetriever
from disclosure_rag.retrieval.parent_expander import ParentExpander
from disclosure_rag.retrieval.tokenizers import build_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class IndexBundle:
    chunks: list[ChunkSchema]
    retriever: HybridRetriever
    parent_expander: ParentExpander
    modes: list[str] = field(default_factory=list)   # 실제로 붙은 검색 경로
    fact_store: object | None = None


def _load_dense(emb_dir: Path, chunks: list[ChunkSchema], provider, qdrant_path: str | None):
    """기본은 numpy 전수검색. Qdrant 로컬 모드는 62만 point 에서 질의당 21초가
    걸린다(2026-08-30 실측, 2회 재현). 둘 다 exact 검색이라 결과는 동일하다.
    Qdrant 로 되돌리려면 DENSE_BACKEND=qdrant."""
    import os
    if os.environ.get("DENSE_BACKEND", "numpy") != "qdrant":
        from disclosure_rag.retrieval.numpy_dense_retriever import NumpyDenseRetriever
        try:
            return NumpyDenseRetriever(chunks, provider, emb_dir)
        except FileNotFoundError:
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("[INDEX] numpy dense 실패(%s) — Qdrant 로 진행", type(e).__name__)
    return _load_dense_qdrant(emb_dir, chunks, provider, qdrant_path)


def _load_dense_qdrant(emb_dir: Path, chunks: list[ChunkSchema], provider, qdrant_path: str | None):
    """dense_*.npz -> Qdrant. shard 를 순회하며 upsert 한다(전부 메모리에 올리지 않는다)."""
    import numpy as np
    from disclosure_rag.retrieval.dense_retriever import DenseRetriever
    from disclosure_rag.retrieval.qdrant_store import QdrantVectorStore

    shards = sorted(emb_dir.glob("dense_*.npz"))
    if not shards:
        return None
    by_id = {c.chunk_id: c for c in chunks}
    store = QdrantVectorStore(path=qdrant_path or str(emb_dir.parent / "index" / "qdrant"))

    # ── 이미 적재돼 있으면 건너뛴다 ──────────────────────────────────────
    # Qdrant 는 디스크에 남는다. build_index.py 가 한 번 넣어두면 서버를 다시
    # 띄워도 그대로 있는데, 예전 코드는 매번 62만 벡터를 다시 upsert 해서
    # **기동할 때마다 15~20분**을 버렸다(평가 서버에선 치명적).
    # 개수가 같고 표본 chunk_id 가 실제로 들어있으면 재적재하지 않는다.
    # 강제로 다시 넣으려면 REINDEX_DENSE=1.
    import os
    expected = 0
    for sh in shards:
        with np.load(sh, allow_pickle=True) as z:
            expected += len(z["chunk_ids"])
    have = store.count()
    if (not os.environ.get("REINDEX_DENSE")) and have >= expected > 0:
        with np.load(shards[-1], allow_pickle=True) as z:
            probe = [str(z["chunk_ids"][0]), str(z["chunk_ids"][-1])]
        if all(store.has_chunk(cid) for cid in probe):
            logger.info("[INDEX] dense %d vector 이미 적재됨 — 재적재 건너뜀 "
                        "(다시 넣으려면 REINDEX_DENSE=1)", have)
            return DenseRetriever(chunks, provider, store)
        logger.warning("[INDEX] point 수(%d)는 맞지만 내용이 다르다 — 다시 적재한다", have)

    total = 0
    STEP = 2_000  # .tolist() 로 부푸는 파이썬 float 을 오래 들고 있지 않는다
    for sh in shards:
        z = np.load(sh, allow_pickle=True)
        ids, vecs = list(z["chunk_ids"]), z["vectors"]
        for b in range(0, len(ids), STEP):
            cs, vs = [], []
            for i, v in zip(ids[b:b + STEP], vecs[b:b + STEP]):
                c = by_id.get(i)
                if c is not None:
                    cs.append(c); vs.append(v.astype("float32").tolist())
            if cs:
                store.upsert_chunks(cs, vs)
                total += len(cs)
            del cs, vs
        del z, ids, vecs
    logger.info("[INDEX] dense %d vector 적재 (shard %d개)", total, len(shards))
    return DenseRetriever(chunks, provider, store)


def _load_sparse(emb_dir: Path, chunks: list[ChunkSchema], provider):
    """sparse_*.jsonl.gz -> SparseRetriever (BGE-M3 learned lexical weights).

    2026-08-23: 예전엔 `weights_by_id` 에 55만 개 dict 를 전부 담고(약 9.7GB) 넘겼다.
    지금은 `from_shards` 가 파일을 흘려 읽으며 numpy CSR 로 바로 눌러 담는다.
    """
    from disclosure_rag.retrieval.sparse_retriever import SparseRetriever

    shards = sorted(emb_dir.glob("sparse_*.jsonl.gz")) + sorted(emb_dir.glob("sparse_*.jsonl"))
    if not shards:
        return None

    # provider 가 SharedQueryEncoder 면 dense 와 forward pass 를 나눠 쓴다.
    # 아니면 예전처럼 sparse 만 따로 뽑는다.
    encode_query = getattr(provider, "lexical_query", None)
    if encode_query is None:
        def encode_query(text: str) -> dict:   # noqa: F811
            out = provider.encode_all([text], batch_size=1, dense=False,
                                      sparse=True, colbert=False)
            w = out["lexical_weights"][0]
            return {str(k): float(v) for k, v in w.items()}

    return SparseRetriever.from_shards(chunks, shards, encode_query)


def load_bundle(
    artifacts: str | Path = "artifacts_v2",
    *,
    bm25_tokenizer: str = "kiwi",
    use_dense: bool = True,
    use_sparse: bool = True,
    use_facts: bool = True,
    # 2026-08-27 314문항 실측: rrf 가 weighted 를 전 지표에서 이긴다
    #   hit@5 0.583 vs 0.535 / MRR 0.465 vs 0.409. 비용 차이는 없다.
    fusion: str = "rrf",
    weights: dict[str, float] | None = None,
    parent_max_chars: int = 4000,
) -> IndexBundle:
    root = Path(artifacts)
    l1, emb, idx = root / "l1", root / "emb", root / "index"

    chunks = load_chunks(l1, leaf_only=True, render_text=True)
    parent_map = load_parent_map(l1)
    logger.info("[INDEX] leaf %d / parent %d", len(chunks), len(parent_map))

    tok = build_tokenizer(bm25_tokenizer)
    bm25_dir = idx / "bm25"
    if (bm25_dir / "chunk_ids.json").exists():
        bm25 = BM25Retriever.load(bm25_dir, chunks, tok)
        logger.info("[INDEX] BM25 인덱스 로드")
    else:
        bm25 = BM25Retriever(chunks, tok)
        logger.info("[INDEX] BM25 인덱스 신규 구축 (build_index.py 로 저장해두면 다음부터 빨라진다)")

    modes = ["bm25"]
    dense = sparse = None
    if (use_dense or use_sparse) and emb.exists():
        provider = None
        try:
            from disclosure_rag.retrieval.embeddings import (
                BgeM3MultiProvider, SharedQueryEncoder,
            )
            # 질의 인코딩을 dense/sparse 가 한 번에 나눠 쓰게 감싼다.
            # 검색 점수는 그대로고 질의당 모델 호출만 2회 -> 1회가 된다.
            provider = SharedQueryEncoder(BgeM3MultiProvider())
        except Exception as e:  # noqa: BLE001
            logger.warning("[INDEX] BGE-M3 로드 실패(%s) — BM25 단독으로 진행", type(e).__name__)
        if provider is not None:
            if use_dense:
                dense = _load_dense(emb, chunks, provider, str(idx / "qdrant"))
                if dense:
                    modes.append("dense")
            if use_sparse:
                sparse = _load_sparse(emb, chunks, provider)
                if sparse:
                    modes.append("sparse")

    retriever = HybridRetriever(bm25, dense=dense, sparse=sparse,
                                fusion=fusion, weights=weights)

    fact_store = None
    if use_facts:
        # facts 가 두 파일로 나뉘어 있다 — 서식 공시(25,207건)와 사업보고서
        # (1,070,486건). 표 구조가 달라 추출 규칙이 달라서 따로 뽑았다.
        # 조회할 때 합친다(MultiFactStore). 자세한 이유는 그 모듈 참조.
        #
        # periodic 은 정리본(v2)이 있으면 그쪽만 쓴다 — 둘 다 열면 같은 사실이
        # 두 번 나온다.
        from disclosure_rag.facts.multi_store import MultiFactStore

        candidates = [root / "facts" / "facts.sqlite"]
        # 정기공시 수치사전은 여러 판이 남아 있다. **최신 판을 먼저** 본다.
        # v4 = 제3자(최대주주 등) 수치에 주인 이름을 붙인 판(2026-09-01).
        # FACTS_PERIODIC 환경변수로 특정 판을 지목할 수 있다(A/B 비교용).
        import os
        forced = os.environ.get("FACTS_PERIODIC", "").strip()
        names = ([forced] if forced else
                 ["facts_periodic_v4", "facts_periodic_v3",
                  "facts_periodic_v2", "facts_periodic"])
        for name in names:
            path = root / name / "facts.sqlite"
            if path.exists():
                candidates.append(path)
                logger.info("[INDEX] 정기공시 수치사전: %s", name)
                break
        fact_store = MultiFactStore.from_paths(candidates)
        if fact_store is not None:
            modes.append("facts")
            logger.info("[INDEX] facts 저장소 %d개",
                        len(getattr(fact_store, "stores", [fact_store])))

    return IndexBundle(
        chunks=chunks, retriever=retriever,
        parent_expander=ParentExpander(parent_map, max_chars=parent_max_chars,
                              table_map=ParentExpander.build_table_map(chunks),
                              chunk_by_id={c.chunk_id: c for c in chunks}),
        modes=modes, fact_store=fact_store,
    )
