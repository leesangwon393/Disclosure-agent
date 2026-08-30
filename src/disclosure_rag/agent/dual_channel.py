"""Stage 6: Structured ∥ Unstructured Dual Channel Retrieval.

Unstructured는 HybridRetriever(BM25/Dense/Sparse -> fusion -> rerank), Structured는
Facts SQLite와 정정 그래프다. Facts에는 검색 점수가 없으므로 fusion에
넣지 않고, 두 결과를 report_id로만 묶는다.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Iterable

from disclosure_rag.agent.field_schema import normalize_field_key, normalize_report_kind
from disclosure_rag.agent.query_plan import QueryPlan
from disclosure_rag.agent.version_dedup import deduplicate_scored, deduplicate_versions
from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.retrieval.metadata_filter import RetrievalFilter, normalize_period_tokens

logger = logging.getLogger(__name__)

# Facts 에 어떤 항목이 있는지 미리 훑을 때의 상한. 서식 공시만 있을 땐 680종
# 이었지만 사업보고서를 붙이면 40,479종이 된다(실측).
FACT_KEY_LIMIT = 100_000


@dataclass
class ReportEvidence:
    report_id: str
    unstructured: list[tuple[ChunkSchema, float]] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)
    correction: dict | None = None

    @property
    def channels(self) -> list[str]:
        out = []
        if self.unstructured:
            out.append("unstructured")
        if self.facts:
            out.append("facts")
        if self.correction:
            out.append("correction_graph")
        return out

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "channels": self.channels,
            "unstructured_chunk_ids": [chunk.chunk_id for chunk, _ in self.unstructured],
            "fact_count": len(self.facts),
            "correction": self.correction,
        }


@dataclass
class DualChannelResult:
    query: str
    unstructured_results: list[tuple[ChunkSchema, float]]
    facts: list[dict]
    reports: list[ReportEvidence]
    corrections: list[dict]
    facts_executed: bool
    structured_fields: list[str]
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "unstructured_results": [_chunk_dict(chunk, score)
                                     for chunk, score in self.unstructured_results],
            # score 필드를 붙이지 않는다. Facts는 fusion/ranking 대상이 아니다.
            "facts": [_fact_dict(row) for row in self.facts],
            "reports": [report.to_dict() for report in self.reports],
            "corrections": list(self.corrections),
            "diagnostics": {
                **self.diagnostics,
                "facts_executed": self.facts_executed,
                "structured_fields": list(self.structured_fields),
            },
        }


def _chunk_dict(chunk: ChunkSchema, score: float) -> dict:
    return {
        "chunk_id": chunk.chunk_id, "report_id": chunk.report_id,
        "company": chunk.company, "report_type": chunk.report_type,
        "report_name": chunk.report_name, "period": chunk.period,
        "filing_date": chunk.filing_date, "section_path": chunk.section_path,
        "content_type": chunk.content_type, "is_correction": chunk.is_correction,
        "correction_group_id": chunk.correction_group_id, "is_latest": chunk.is_latest,
        "text": chunk.raw_text, "score": round(float(score), 4),
    }


def _fact_dict(row: dict) -> dict:
    """생성기에 필요한 값과 provenance만 남긴다."""
    return {
        "report_id": row.get("doc_id"), "doc_id": row.get("doc_id"),
        "chunk_id": row.get("chunk_id"), "company": row.get("company"),
        "item": row.get("key_norm"), "value": row.get("value_text"),
        "value_num": row.get("value_num"), "unit": row.get("value_unit"),
        "date": row.get("value_date"), "report_name": row.get("report_name"),
        "filing_date": row.get("filing_date"), "period": row.get("period"),
        "is_correction": bool(row.get("is_correction")), "is_latest": row.get("is_latest"),
        "section_path": row.get("section_path") or [],
    }


def retrieval_filter_from_plan(plan: QueryPlan) -> RetrievalFilter:
    return RetrievalFilter(
        companies=list(plan.companies) or None,
        doc_groups=list(plan.report_types) or None,
        periods=normalize_period_tokens(plan.periods),
        latest_only=(plan.latest_policy == "latest_only"),
    )


class DualChannelRetriever:
    def __init__(
        self, unstructured, fact_store, *, correction_index: dict | None = None,
        manifest: Iterable | None = None, facts_per_field: int = 10,
        available_fact_keys: Iterable[str] | None = None,
    ):
        self.unstructured = unstructured
        self.fact_store = fact_store
        self.correction_index = correction_index or {}
        self.manifest = list(manifest or [])
        self.facts_per_field = facts_per_field
        if available_fact_keys is not None:
            keys = available_fact_keys
        elif fact_store is not None:
            # 사업보고서 facts 를 붙이면서 항목 종류가 680 -> 41,000 대로 늘었다.
            # 상한이 10,000 이면 뒤쪽 항목이 잘려 "Facts 에 없다"고 오판한다.
            keys = (key for key, _count in fact_store.distinct_keys(limit=FACT_KEY_LIMIT))
        else:
            keys = []
        self._fact_key_by_norm = {
            normalize_field_key(key): key for key in keys if normalize_field_key(key)
        }

    def structured_fields(self, plan: QueryPlan) -> list[str]:
        """expected_fields 중 Facts에 실제 존재하는 항목만 반환한다."""
        out = []
        for field_name in plan.expected_fields:
            normalized = normalize_field_key(field_name)
            if normalized in self._fact_key_by_norm and self._fact_key_by_norm[normalized] not in out:
                out.append(self._fact_key_by_norm[normalized])
        return out

    # 집계 질문일 때의 조회 한도. 값 기준으로 정렬해도 상한이 작으면 회사별
    # 최댓값 하나만 겨우 들어와 비교 근거가 얇아진다. 정렬이 이미 값 순이므로
    # 넉넉히 가져와도 상위권은 안 흔들린다.
    AGGREGATION_LIMIT = 50

    _ORDER_BY = {"max": "value_desc", "min": "value_asc"}

    def _lookup_facts(self, plan: QueryPlan, fields: list[str]) -> list[dict]:
        if not fields:
            return []
        companies: list[str | None] = list(plan.companies) or [None]
        doc_groups: list[str | None] = list(plan.report_types) or [None]
        periods: list[str | None] = list(plan.periods) or [None]
        latest_only = plan.latest_policy == "latest_only"
        # "최대 계약금액은?" 같은 질문에서 최신순으로 자르면 최댓값이 잘려나간다
        # (실측: 삼성바이오로직스 54건 중 최댓값이 최신 10건 밖). 값 기준으로
        # 정렬하고 한도도 올린다.
        order_by = self._ORDER_BY.get(getattr(plan, "aggregation", "none"), "date")
        per_field = (self.AGGREGATION_LIMIT if order_by != "date"
                     else self.facts_per_field)
        rows: list[dict] = []
        seen: set[tuple] = set()
        for company in companies:
            for key in fields:
                for doc_group in doc_groups:
                    for period in periods:
                        found = self.fact_store.lookup(
                            company=company, key=key, doc_group=doc_group,
                            period=period, latest_only=latest_only,
                            order_by=order_by, limit=per_field,
                        )
                        for row in found:
                            signature = (
                                row.get("doc_id"), row.get("chunk_id"), row.get("key_norm"),
                                row.get("value_text"),
                            )
                            if signature not in seen:
                                seen.add(signature)
                                # version_dedup은 report_id를 공통 키로 읽는다. Facts의
                                # 원래 provenance(doc_id)는 그대로 두고 alias만 보탠다.
                                rows.append({**row, "report_id": row.get("doc_id")})

        # latest_only는 SQL에서도 적용하지만 first_and_final/all_versions는
        # 정정 체인을 실제 정책대로 정리해야 한다. 점수나 재정렬은 생기지 않는다.
        kept, _report = deduplicate_versions(rows, plan.latest_policy)
        return kept

    def _correction_rows(self, plan: QueryPlan) -> list[dict]:
        if not self.correction_index or not self.manifest or not plan.companies:
            return []
        candidates = [row for row in self.manifest if row.corp_name in plan.companies]
        if plan.report_types:
            candidates = [row for row in candidates if row.doc_group in plan.report_types]
        if plan.report_kinds:
            wanted_kinds = {normalize_report_kind(kind) for kind in plan.report_kinds}
            candidates = [
                row for row in candidates
                if any(kind in normalize_report_kind(
                    getattr(row, "doc_subtype", None) or row.report_nm
                ) for kind in wanted_kinds)
            ]
        if plan.periods:
            wanted_years = {period[:4] for period in plan.periods if len(period) >= 4}
            candidates = [
                row for row in candidates
                if str(getattr(row, "base_year", "") or row.rcept_dt[:4]) in wanted_years
            ]
        records = []
        for row in candidates:
            record = self.correction_index.get(row.doc_id)
            if record is None:
                continue
            if plan.latest_policy == "latest_only":
                keep = record.is_latest
            elif plan.latest_policy == "first_and_final":
                keep = record.correction_order == 0 or record.is_latest
            else:
                keep = True
            if keep:
                item = asdict(record)
                item.update({"company": row.corp_name, "report_name": row.report_nm,
                             "filing_date": row.rcept_dt, "report_type": row.doc_group})
                records.append(item)
        return records

    def search(
        self, query: str, plan: QueryPlan, *, k: int | None = None,
        flt: RetrievalFilter | None = None, candidate_k: int = 50,
        rerank_top_n: int = 50,
    ) -> DualChannelResult:
        """Unstructured를 worker에서 돌리는 동안 현재 thread에서 Facts를 조회한다.

        sqlite connection을 생성한 thread에서만 쓰기 위해 Facts를 worker로
        넘기지 않는다. 그래도 두 I/O/계산 경로는 실제로 병렬 실행된다.
        """
        top_k = k if k is not None else plan.top_k()
        retrieval_filter = flt or retrieval_filter_from_plan(plan)
        fields = self.structured_fields(plan)
        started = time.perf_counter()
        unstructured_error = None
        facts_error = None

        def run_unstructured():
            if hasattr(self.unstructured, "search_with_trace"):
                trace = self.unstructured.search_with_trace(
                    query, k=top_k, flt=retrieval_filter, candidate_k=candidate_k,
                    rerank_top_n=rerank_top_n,
                )
                deduped, dedup_report = deduplicate_scored(
                    trace.results, plan.latest_policy,
                )
                return deduped, {
                    "channel_counts": trace.channel_counts,
                    "fused_count": trace.fused_count,
                    "reranked": trace.reranked,
                    "version_dedup": asdict(dedup_report),
                }
            raw = self.unstructured.search(query, k=top_k, flt=retrieval_filter)
            deduped, dedup_report = deduplicate_scored(raw, plan.latest_policy)
            return deduped, {"version_dedup": asdict(dedup_report)}

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="unstructured-retrieval") as pool:
            future = pool.submit(run_unstructured)
            facts_started = time.perf_counter()
            try:
                facts = self._lookup_facts(plan, fields)
            except Exception as exc:  # noqa: BLE001
                logger.exception("[DUAL] Facts 채널 실패: %s", exc)
                facts, facts_error = [], f"{type(exc).__name__}: {exc}"
            facts_ms = (time.perf_counter() - facts_started) * 1000
            try:
                unstructured_results, unstructured_diag = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.exception("[DUAL] Unstructured 채널 실패: %s", exc)
                unstructured_results, unstructured_diag = [], {}
                unstructured_error = f"{type(exc).__name__}: {exc}"

        corrections = self._correction_rows(plan)
        correction_by_id = {row["doc_id"]: row for row in corrections}
        reports_by_id: dict[str, ReportEvidence] = {}
        report_order: list[str] = []
        for chunk, score in unstructured_results:
            if chunk.report_id not in reports_by_id:
                reports_by_id[chunk.report_id] = ReportEvidence(report_id=chunk.report_id)
                report_order.append(chunk.report_id)
            reports_by_id[chunk.report_id].unstructured.append((chunk, score))
        for row in facts:
            report_id = row.get("report_id") or row.get("doc_id")
            if not report_id:
                continue
            if report_id not in reports_by_id:
                reports_by_id[report_id] = ReportEvidence(report_id=report_id)
                report_order.append(report_id)
            reports_by_id[report_id].facts.append(row)
        # 정정 그래프도 Structured 채널의 일부다. 검색/Facts에 잡히지 않은
        # 정정 문서는 뒤에만 추가해 Unstructured의 순위는 건드리지 않는다.
        if plan.latest_policy != "latest_only":
            for row in corrections:
                report_id = row["doc_id"]
                if report_id not in reports_by_id:
                    reports_by_id[report_id] = ReportEvidence(report_id=report_id)
                    report_order.append(report_id)
        for report_id in report_order:
            reports_by_id[report_id].correction = correction_by_id.get(report_id)

        elapsed_ms = (time.perf_counter() - started) * 1000
        diagnostics = {
            **unstructured_diag,
            "unstructured_count": len(unstructured_results), "fact_count": len(facts),
            "report_count": len(report_order), "facts_ms": round(facts_ms, 2),
            "elapsed_ms": round(elapsed_ms, 2),
            "unstructured_error": unstructured_error, "facts_error": facts_error,
            "facts_order_by": self._ORDER_BY.get(getattr(plan, "aggregation", "none"), "date"),
        }
        logger.info(
            "[DUAL] facts_executed=%s structured_fields=%s fact_rows=%d "
            "unstructured=%d reports=%d elapsed_ms=%.1f",
            bool(fields), fields, len(facts), len(unstructured_results), len(report_order), elapsed_ms,
        )
        return DualChannelResult(
            query=query, unstructured_results=unstructured_results, facts=facts,
            reports=[reports_by_id[report_id] for report_id in report_order],
            corrections=corrections, facts_executed=bool(fields),
            structured_fields=fields, diagnostics=diagnostics,
        )


__all__ = [
    "DualChannelRetriever", "DualChannelResult", "ReportEvidence",
    "retrieval_filter_from_plan",
]
