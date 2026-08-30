"""BM25S baseline retriever (§8, §32, §51~52).

Metadata Filtering 은 §51 대로 "Coarse-to-Fine" 로 구현한다: 정확한 field-filter
가 가능한 inverted index 를 새로 만드는 대신(baseline 이므로), BM25S 전체
인덱스에서 넉넉히 overfetch 한 뒤 metadata 로 post-filter 하고 top-k 를 자른다.
후보가 부족하면 overfetch 배수를 늘려 재시도한다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import bm25s

from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.retrieval.metadata_filter import RetrievalFilter
from disclosure_rag.retrieval.tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self, chunks: list[ChunkSchema], tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.chunks_by_id = {c.chunk_id: c for c in chunks}
        self._ids = [c.chunk_id for c in chunks]

        corpus_tokens = [tokenizer.tokenize(c.text) for c in chunks]
        self._bm25 = bm25s.BM25(corpus=self._ids)
        self._bm25.index(corpus_tokens, show_progress=False)

    def _resolve_id(self, raw) -> str | None:
        """bm25s 는 인덱스를 어떻게 만들었느냐에 따라 corpus 항목을 문자열/정수/dict 로
        돌려준다(저장 후 load_corpus=True 로 읽으면 dict). 어느 쪽이든 chunk_id 로 정규화한다."""
        if isinstance(raw, dict):
            raw = raw.get("text", raw.get("id"))
        if isinstance(raw, str):
            return raw
        try:
            return self._ids[int(raw)]
        except (TypeError, ValueError, IndexError):
            return None

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        flt: RetrievalFilter | None = None,
        overfetch_multiplier: int = 20,
        max_overfetch: int = 2000,
    ) -> list[tuple[ChunkSchema, float]]:
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            logger.warning("[BM25] query 토큰화 결과가 비어있음: %r", query)
            return []

        # 2026-08-30: 좁은 필터(report_ids/companies)면 overfetch 상한을 풀어준다.
        # 상한 2,000 은 전체 626,497 의 0.3% 라, 문서 1건으로 좁히면 그 문서
        # chunk 가 상위 2,000 안에 없어 **필터가 통과시킬 후보 자체가 0개**가 된다.
        # 아래 while 루프가 fetch_k 를 4배씩 늘리므로, 상한만 풀면 필요한 만큼만
        # 커진다(대부분 한두 번 확장에서 끝난다).
        if flt is not None and flt.is_selective:
            max_overfetch = max(max_overfetch, len(self._ids))

        # 회사가 지정됐으면 **그 회사 문서만 점수를 남긴다**(weight_mask).
        # 전역 상위 N을 먼저 뽑고 파이썬에서 거르면, 대상 회사의 청크가 드문
        # 공시일수록 fetch_k 를 4배씩 키우며 626,497건까지 훑게 된다. 그게
        # 지연시간의 주범이었다(2026-08-31 실측: lookup_form 검색 1회 67초,
        # 정기공시는 같은 코드로 3.4초 — 회사당 청크 밀도 차이).
        #
        # 마스크를 씌우면 통과 대상이 상위에 모이므로 fetch_k 를 키울 일이 없다.
        # 결과는 동일하다 — 어차피 필터가 떨어뜨릴 문서를 미리 0점으로 만들 뿐이다.
        mask = self._company_mask(flt)
        if mask is not None:
            allowed = int(mask.sum())
            if allowed == 0:
                return []
            max_overfetch = min(max_overfetch, max(allowed, k))

        fetch_k = min(max(k * overfetch_multiplier, k), max_overfetch, len(self._ids))
        while True:
            ids, scores = self._bm25.retrieve([query_tokens], k=fetch_k,
                                              show_progress=False,
                                              **({"weight_mask": mask} if mask is not None else {}))
            candidates = list(zip(ids[0].tolist(), scores[0].tolist()))

            results: list[tuple[ChunkSchema, float]] = []
            for raw_id, score in candidates:
                chunk = self.chunks_by_id.get(self._resolve_id(raw_id))
                if chunk is None or score <= 0:
                    continue
                if flt is not None and not flt.matches(chunk):
                    continue
                results.append((chunk, score))
                if len(results) >= k:
                    return results

            if fetch_k >= min(max_overfetch, len(self._ids)):
                return results
            fetch_k = min(fetch_k * 4, max_overfetch, len(self._ids))


    def _company_mask(self, flt):
        """회사 필터를 bm25s 의 weight_mask 로. 회사 지정이 없으면 None."""
        names = list(getattr(flt, "companies", None) or []) if flt is not None else []
        if not names:
            return None
        cache = getattr(self, "_mask_cache", None)
        if cache is None:
            cache = self._mask_cache = {}
        key = tuple(sorted(names))
        if key in cache:
            return cache[key]
        import numpy as _np
        wanted = set(names)
        mask = _np.zeros(len(self._ids), dtype=_np.float32)
        for i, cid in enumerate(self._ids):
            chunk = self.chunks_by_id.get(cid)
            if chunk is not None and getattr(chunk, "company", None) in wanted:
                mask[i] = 1.0
        cache[key] = mask
        return mask

    # ------------------------------------------------------------------ 영속화
    # 기존 파이프라인은 프로세스를 켤 때마다 45만 chunk 를 처음부터 다시 색인했다
    # (Kiwi 형태소 분석 포함). 평가 서버 기동 시간이 그대로 늘어난다.
    def save(self, path: str | Path) -> None:
        """bm25s 인덱스 + chunk_id 순서를 디스크에 저장한다."""
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(d / "bm25s"))
        (d / "chunk_ids.json").write_text(
            json.dumps(self._ids, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, chunks: list[ChunkSchema], tokenizer: Tokenizer
             ) -> "BM25Retriever":
        """저장된 인덱스를 읽는다. chunks 는 L1 스냅샷에서 온 것이어야 하며,
        저장 당시와 chunk_id 집합이 다르면 **조용히 어긋나지 않도록 즉시 실패**한다."""
        import bm25s as _bm25s

        d = Path(path)
        ids = json.loads((d / "chunk_ids.json").read_text(encoding="utf-8"))
        by_id = {c.chunk_id: c for c in chunks}
        missing = [i for i in ids[:5000] if i not in by_id]
        if missing:
            raise ValueError(
                f"인덱스와 스냅샷이 불일치한다(예: {missing[:3]}). "
                "청킹을 바꿨다면 인덱스를 다시 만들어야 한다.")
        obj = cls.__new__(cls)
        obj.tokenizer = tokenizer
        obj._ids = ids
        obj.chunks_by_id = by_id
        obj._bm25 = _bm25s.BM25.load(str(d / "bm25s"), load_corpus=True)
        return obj
