"""BGE-M3 learned sparse (lexical weights) retriever.

BM25 와 뭐가 다른가
-------------------
BM25 는 **단어 빈도 통계**로 가중치를 정한다(1994년 공식, 학습 없음).
learned sparse 는 같은 "토큰 -> 가중치" 형태지만 **가중치를 모델이 학습**한다.
- 문서 안에서 어떤 토큰이 중요한지 문맥을 보고 판단한다
- BM25 의 강점(정확한 문자열 일치, 숫자·고유명사 구분)을 유지하면서
  dense 의 일반화를 일부 가져온다

공시 도메인에서 특히 유효한 이유: 접수번호·종목코드·계약금액 같은 **정확 일치가
중요한 토큰**을 dense 처럼 뭉개지 않으면서, "매출액"처럼 어디에나 나오는 토큰의
가중치는 낮춰준다.

메모리 — 2026-08-23 재작성 이유
--------------------------------
예전 구현은 역색인을 `dict[str, list[tuple[int, float]]]` 로 들고 있었다.
우리 코퍼스는 leaf 551,596개 × 조각당 평균 189 term = **1억 424만 개 posting** 이다.
파이썬 객체로는 tuple 하나가 60바이트를 넘으므로 **약 20GB** 가 필요했다.
(같은 종류의 착각으로 ColBERT 전체 저장을 시도했다가 맥북이 꺼졌다.)

지금은 numpy CSR 역색인이다:
  posting 1개 = doc_id(int32) + weight(float32) = **8바이트**
  1억 424만 개 x 8B = **약 0.83GB**
검색도 파이썬 루프가 아니라 슬라이스 + fancy-index 누산이라 훨씬 빠르다.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.retrieval.metadata_filter import RetrievalFilter

logger = logging.getLogger(__name__)

LexWeights = dict[str, float]  # token(보통 token_id 문자열) -> weight

_FLUSH = 4_000_000  # 이만큼 모이면 numpy 배열로 눌러 담는다(파이썬 리스트 체류시간 최소화)


class SparseRetriever:
    """BGE-M3 lexical_weights 기반 numpy CSR 역색인 검색기.

    chunks[i] 와 weights[i] 는 같은 순서여야 한다.
    query_encoder(text) -> LexWeights 를 주입한다(BgeM3MultiProvider 로 만든다).

    전체 코퍼스를 올릴 때는 `from_shards()` 를 써라. `__init__` 은 weights 를
    전부 들고 있어야 하지만 `from_shards` 는 샤드 파일을 흘려 읽는다.
    """

    name = "sparse"

    def __init__(
        self,
        chunks: list[ChunkSchema],
        weights: Iterable[LexWeights],
        query_encoder: Callable[[str], LexWeights],
    ):
        self.chunks = chunks
        self._encode = query_encoder
        self._build(len(chunks), enumerate(weights))

    # ------------------------------------------------------------------ 생성
    @classmethod
    def from_shards(
        cls,
        chunks: list[ChunkSchema],
        shard_paths: list[Path],
        query_encoder: Callable[[str], LexWeights],
    ) -> "SparseRetriever | None":
        """sparse_*.jsonl(.gz) 를 **흘려 읽으며** 역색인을 만든다.

        예전 방식은 `weights_by_id` dict 에 55만 개 dict 를 전부 담은 뒤
        생성자에 넘겼다(약 9.7GB). 여기서는 한 줄 읽고 바로 배열에 눌러 담는다.
        """
        self = cls.__new__(cls)
        self.chunks = chunks
        self._encode = query_encoder
        row_of = {c.chunk_id: i for i, c in enumerate(chunks)}

        def pairs() -> Iterator[tuple[int, LexWeights]]:
            seen = 0
            skipped_shards = 0
            for sh in shard_paths:
                op = gzip.open if sh.suffix == ".gz" else open
                try:
                    with op(sh, "rt", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            d = json.loads(line)
                            row = row_of.get(d["chunk_id"])
                            if row is None:      # 스냅샷과 임베딩이 어긋난 조각은 건너뛴다
                                continue
                            seen += 1
                            yield row, d["w"]
                except (OSError, EOFError, json.JSONDecodeError) as exc:
                    # 샤드 파일 하나가 손상돼도(디스크 문제·불완전 복사 등) 전체
                    # sparse 인덱스를 못 쓰게 만들면 안 된다 — BM25/dense/facts는
                    # 멀쩡한데 sparse 샤드 1개 때문에 검색 전체가 죽는 건 손실이
                    # 훨씬 크다(2026-09-01 발견: sparse_0005.jsonl.gz gzip 자체
                    # 손상). 그 샤드의 나머지 chunk만 sparse 점수 없이 빠진다.
                    skipped_shards += 1
                    logger.warning(
                        "[SPARSE] 샤드 손상으로 건너뜀: %s (%s: %s) — 이 샤드의 "
                        "chunk는 sparse 채널 없이 BM25/dense로만 검색된다",
                        sh, type(exc).__name__, exc,
                    )
                    continue
            logger.info("[SPARSE] 샤드 %d개에서 %d chunk 적재%s", len(shard_paths), seen,
                        f" (손상되어 건너뛴 샤드 {skipped_shards}개)" if skipped_shards else "")

        self._build(len(chunks), pairs())
        if self._docs.size == 0:
            return None
        return self

    def _build(self, n_docs: int, pair_iter: Iterable[tuple[int, LexWeights]]) -> None:
        self._n_docs = n_docs
        vocab: dict[str, int] = {}
        d_parts: list[np.ndarray] = []
        t_parts: list[np.ndarray] = []
        v_parts: list[np.ndarray] = []
        bd: list[int] = []
        bt: list[int] = []
        bv: list[float] = []

        def flush() -> None:
            if not bd:
                return
            d_parts.append(np.asarray(bd, dtype=np.int32))
            t_parts.append(np.asarray(bt, dtype=np.int32))
            v_parts.append(np.asarray(bv, dtype=np.float32))
            bd.clear(); bt.clear(); bv.clear()

        for row, w in pair_iter:
            for tok, val in w.items():
                v = float(val)
                if v <= 0.0:
                    continue
                key = str(tok)
                t = vocab.get(key)
                if t is None:
                    t = len(vocab)
                    vocab[key] = t
                bd.append(row); bt.append(t); bv.append(v)
            if len(bd) >= _FLUSH:
                flush()
        flush()

        if not d_parts:
            self._vocab = {}
            self._starts = np.zeros(1, dtype=np.int64)
            self._docs = np.empty(0, dtype=np.int32)
            self._vals = np.empty(0, dtype=np.float32)
            return

        docs = np.concatenate(d_parts); d_parts.clear()
        terms = np.concatenate(t_parts); t_parts.clear()
        vals = np.concatenate(v_parts); v_parts.clear()

        # term 순으로 정렬해 CSR 로 만든다. (term 마다 posting 이 연속 구간이 된다)
        order = np.argsort(terms, kind="stable")
        terms = terms[order]; docs = docs[order]; vals = vals[order]
        del order

        # vocab 은 0..n_terms-1 을 빠짐없이 쓰므로 bincount 로 바로 offset 을 만든다.
        counts = np.bincount(terms, minlength=len(vocab))
        starts = np.zeros(len(vocab) + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        del terms, counts

        self._vocab = vocab
        self._starts = starts
        self._docs = docs
        self._vals = vals
        mb = (docs.nbytes + vals.nbytes + starts.nbytes) / 1024 ** 2
        logger.info("[SPARSE] %d chunk / %d token / posting %d개 (%.0fMB)",
                    n_docs, len(vocab), docs.size, mb)

    # ------------------------------------------------------------------ 검색
    def search(
        self, query: str, *, k: int = 10, flt: RetrievalFilter | None = None,
    ) -> list[tuple[ChunkSchema, float]]:
        qw: LexWeights = self._encode(query)
        if not qw or self._docs.size == 0:
            return []

        scores = np.zeros(self._n_docs, dtype=np.float32)
        hit = False
        for tok, q_val in qw.items():
            qv = float(q_val)
            if qv <= 0.0:
                continue
            t = self._vocab.get(str(tok))
            if t is None:
                continue
            s, e = self._starts[t], self._starts[t + 1]
            if s == e:
                continue
            # 한 term 의 posting 안에서 doc 은 중복되지 않으므로 fancy-index += 가 안전하다
            scores[self._docs[s:e]] += qv * self._vals[s:e]
            hit = True
        if not hit:
            return []

        nz = np.flatnonzero(scores)
        if nz.size == 0:
            return []
        ranked = nz[np.argsort(-scores[nz], kind="stable")]

        out: list[tuple[ChunkSchema, float]] = []
        for i in ranked:
            c = self.chunks[int(i)]
            if flt is not None and not flt.matches(c):
                continue
            out.append((c, float(scores[i])))
            if len(out) >= k:
                break
        return out
