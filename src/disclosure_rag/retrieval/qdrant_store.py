"""Dense Vector 저장소 (사용자 결정 #6): Qdrant.

chunk vector 와 metadata payload 를 같이 저장해 company/period/report_type/
correction 상태 등으로 filter 가 가능하게 한다. 서버 없이도 쓸 수 있게 embedded
(local path 또는 in-memory) 모드를 기본으로 지원한다.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.retrieval.metadata_filter import RetrievalFilter

_NAMESPACE = uuid.UUID("d15c105e-0000-4000-8000-000000000000")  # "disclosure" 고정 namespace


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def _payload_of(chunk: ChunkSchema) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "report_id": chunk.report_id,
        "parent_chunk_id": chunk.parent_chunk_id,
        "company": chunk.company,
        "corp_code": chunk.corp_code,
        "report_type": chunk.report_type,
        "report_subtype": chunk.report_subtype,
        "report_name": chunk.report_name,
        "period": chunk.period,
        "filing_date": chunk.filing_date,
        "filing_date_int": int(chunk.filing_date) if chunk.filing_date else None,
        "section_path": chunk.section_path,
        "content_type": chunk.content_type,
        "is_correction": chunk.is_correction,
        "correction_group_id": chunk.correction_group_id,
        "correction_order": chunk.correction_order,
        "is_latest": chunk.is_latest,
    }


def build_qdrant_filter(flt: RetrievalFilter | None) -> Filter | None:
    if flt is None:
        return None
    must: list[FieldCondition] = []
    if flt.companies:
        must.append(FieldCondition(key="company", match=MatchAny(any=flt.companies)))
    if flt.doc_groups:
        must.append(FieldCondition(key="report_type", match=MatchAny(any=flt.doc_groups)))
    if flt.doc_subtypes:
        must.append(FieldCondition(key="report_subtype", match=MatchAny(any=flt.doc_subtypes)))
    if flt.periods:
        # period payload 는 정기공시만 "YYYY-MM" 이고 나머지 3종은 없다.
        # 연도 단위 요청("2024")을 서버측 MatchAny 로 표현할 수 없으므로 결산월
        # 4종으로 전개한다(70개사 전부 12월 결산 → 03/06/09/12).
        # 한계: period 가 없는 major/exchange/holding 은 이 조건에서 탈락한다.
        #       numpy/bm25 경로(RetrievalFilter.matches)는 filing_date 연도로
        #       대체 판정하지만 Qdrant payload 필터로는 표현이 안 된다.
        #       Qdrant 는 현재 기본 백엔드가 아니다(DENSE_BACKEND=qdrant 일 때만).
        expanded: list[str] = []
        for want in flt.periods:
            want = (want or "").strip()
            if not want:
                continue
            if len(want) == 4:
                expanded.extend(f"{want}-{mm}" for mm in ("03", "06", "09", "12"))
            else:
                expanded.append(want)
        if expanded:
            must.append(FieldCondition(key="period", match=MatchAny(any=expanded)))
    if flt.filing_date_from or flt.filing_date_to:
        must.append(
            FieldCondition(
                key="filing_date_int",
                range=Range(
                    gte=int(flt.filing_date_from) if flt.filing_date_from else None,
                    lte=int(flt.filing_date_to) if flt.filing_date_to else None,
                ),
            )
        )
    if flt.latest_only:
        must.append(FieldCondition(key="is_latest", match=MatchValue(value=True)))
    if not flt.include_corrections:
        must.append(FieldCondition(key="is_correction", match=MatchValue(value=False)))
    return Filter(must=must) if must else None


class QdrantVectorStore:
    def __init__(
        self,
        *,
        collection_name: str = "disclosure_chunks",
        dim: int = 1024,
        path: str | None = None,
        url: str | None = None,
        in_memory: bool = False,
    ):
        if in_memory:
            self.client = QdrantClient(":memory:")
        elif url:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(path=path or "./qdrant_data")
        self.collection_name = collection_name
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def count(self) -> int:
        """컬렉션에 들어있는 point 개수. 재적재가 필요한지 판단하는 데 쓴다."""
        try:
            if not self.client.collection_exists(self.collection_name):
                return 0
            return int(self.client.count(self.collection_name, exact=True).count)
        except Exception:  # noqa: BLE001
            return 0

    def has_chunk(self, chunk_id: str) -> bool:
        """특정 chunk 가 이미 적재돼 있는지. 개수만 같고 내용이 다른 경우를 거른다."""
        try:
            got = self.client.retrieve(self.collection_name, ids=[_point_id(chunk_id)])
            return bool(got)
        except Exception:  # noqa: BLE001
            return False

    def upsert_chunks(
        self, chunks: list[ChunkSchema], vectors: list[list[float]], *, batch_size: int = 256,
    ) -> None:
        assert len(chunks) == len(vectors)
        # PointStruct 를 전부 먼저 만들면 벡터 1024개가 파이썬 float 객체로 부풀어
        # 2만 건 기준 1GB 가까이 잡힌다. 배치 단위로 만들고 바로 버린다.
        for i in range(0, len(chunks), batch_size):
            points = [
                PointStruct(id=_point_id(c.chunk_id), vector=v, payload=_payload_of(c))
                for c, v in zip(chunks[i:i + batch_size], vectors[i:i + batch_size])
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            del points

    def search(
        self, query_vector: list[float], *, k: int = 10, flt: RetrievalFilter | None = None,
    ) -> list[tuple[str, float]]:
        """chunk_id, score 쌍을 반환한다 (chunk 본체는 caller 가 별도 lookup)."""
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
            query_filter=build_qdrant_filter(flt),
            with_payload=True,
        )
        return [(pt.payload["chunk_id"], pt.score) for pt in result.points]
