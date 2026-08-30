"""dense 검색을 numpy 행렬곱으로 직접 한다. Qdrant 로컬 모드 대체.

왜 만들었나 (2026-08-30 실측)
-----------------------------
Qdrant 로컬 모드는 62만 point 에서 **질의당 21초**가 걸렸다(2회 재현, 워밍업 아님).
전체 검색 26초의 80%다. 로컬 모드는 HNSW 색인 없이 전수 비교를 하는데,
같은 전수 비교를 numpy 로 하면 BLAS 가 붙어 훨씬 빠르다.

    62.6만 x 1024 float32 행렬 @ 질의벡터 = 6.4억 FLOP → 수십~수백 ms

정확도는 **완전히 동일**하다. 둘 다 전수 비교(exact)이고, 벡터가 L2 정규화돼
있으므로 내적 = 코사인 유사도다.

메모리: float32 로 들고 있으면 2.6GB. Qdrant 로컬이 쓰던 메모리를 대신 쓰는
것이므로 순증가는 그보다 작다. `DENSE_FP16=1` 이면 fp16(1.3GB)으로 들고
질의 때마다 샤드 단위로 변환한다(메모리 절반, 속도는 조금 손해).
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import numpy as np

from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.retrieval.metadata_filter import RetrievalFilter

logger = logging.getLogger(__name__)


class NumpyDenseRetriever:
    """BM25Retriever / DenseRetriever 와 같은 search() 시그니처를 유지한다."""

    name = "dense"

    def __init__(self, chunks: list[ChunkSchema], embedding_provider, emb_dir: str | Path):
        self.embedding_provider = embedding_provider
        by_id = {c.chunk_id: c for c in chunks}

        shards = sorted(glob.glob(str(Path(emb_dir) / "dense_*.npz")))
        if not shards:
            raise FileNotFoundError(f"dense_*.npz 가 없습니다: {emb_dir}")

        keep_fp16 = os.environ.get("DENSE_FP16", "") not in ("", "0", "false", "no")
        dtype = np.float16 if keep_fp16 else np.float32

        mats, kept = [], []
        for sh in shards:
            with np.load(sh, allow_pickle=True) as z:
                ids, vecs = list(z["chunk_ids"]), z["vectors"]
            # 스냅샷에 없는 조각(있으면 안 되지만)은 버린다 — 인덱스가 어긋나면
            # 엉뚱한 chunk 를 돌려주게 되므로 여기서 확실히 정렬한다.
            sel = [i for i, cid in enumerate(ids) if cid in by_id]
            if len(sel) != len(ids):
                logger.warning("[DENSE] %s: %d/%d 만 스냅샷과 매칭",
                               Path(sh).name, len(sel), len(ids))
            mats.append(np.asarray(vecs[sel], dtype=dtype))
            kept.extend(by_id[ids[i]] for i in sel)

        self.matrix = np.concatenate(mats, axis=0)
        self.chunks = kept
        del mats
        logger.info("[DENSE] numpy 행렬 %s %s (%.2fGB)", self.matrix.shape,
                    self.matrix.dtype, self.matrix.nbytes / 1024 ** 3)

    def search(
        self, query: str, *, k: int = 10, flt: RetrievalFilter | None = None,
    ) -> list[tuple[ChunkSchema, float]]:
        q = np.asarray(self.embedding_provider.embed_query(query), dtype=np.float32)

        if self.matrix.dtype == np.float32:
            scores = self.matrix @ q
        else:
            # fp16 저장 모드: 샤드로 잘라 변환하며 계산(메모리 절반)
            scores = np.empty(self.matrix.shape[0], dtype=np.float32)
            STEP = 50_000
            for s in range(0, self.matrix.shape[0], STEP):
                e = min(s + STEP, self.matrix.shape[0])
                scores[s:e] = self.matrix[s:e].astype(np.float32) @ q

        # 후보 pool 결정.
        #
        # 2026-08-30 수정: 좁은 필터(report_ids/companies)일 때 pool 상한을
        # 두면 **통과 대상이 pool 안에 하나도 없어 빈 결과**가 나온다.
        # 예: report_ids 로 문서 1건(약 560 chunk)을 지정하면 전체 626,497 중
        # 0.09% 인데, 전역 상위 1,000개 안에 그 문서 chunk 가 없으면 0건이 된다.
        # 실측(60문항): 에이전트 경로 정답문서 회수 66.7% vs 필터 없는 검색 94.4%.
        #
        # 좁은 필터일 때는 전체를 정렬한다. 626k float32 argsort 는 수십 ms 라
        # 감당 가능하고, 무엇보다 **정확**하다(근사가 아니다).
        if flt is None:
            pool = k
        elif flt.is_selective:
            pool = len(scores)
        else:
            pool = min(len(scores), max(k * 20, 1000))

        if pool >= len(scores):
            order = np.argsort(-scores)
        else:
            top = np.argpartition(-scores, pool)[:pool]
            order = top[np.argsort(-scores[top])]

        out: list[tuple[ChunkSchema, float]] = []
        for i in order:
            c = self.chunks[int(i)]
            if flt is not None and not flt.matches(c):
                continue
            out.append((c, float(scores[i])))
            if len(out) >= k:
                break
        return out
