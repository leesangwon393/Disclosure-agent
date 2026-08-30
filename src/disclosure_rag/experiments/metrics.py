"""Stage 1~5 공용 Retrieval 평가 지표.

Gold 는 report_id(문서) 단위로 매긴다(정확한 chunk-level gold label 은 수작업
비용이 커 이번 실험 범위에서 report-level 로 근사한다 — 이 근사가 관련
report 전체를 커버했는지는 판별 가능하지만, "그 report 안의 어떤 chunk 가
가장 좋은가"는 구분하지 못한다는 한계를 명시적으로 남긴다).

relevance(chunk) = 1  if chunk.report_id in gold_report_ids  else 0
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from disclosure_rag.chunking.chunk_schema import ChunkSchema


@dataclass
class QueryResult:
    query_id: int
    retrieved_report_ids: list[str]  # rank 순서, 중복 허용(같은 report 의 여러 chunk)
    gold_report_ids: list[str]
    latency_sec: float


def _relevant_mask(retrieved_report_ids: list[str], gold: set[str]) -> list[int]:
    return [1 if rid in gold else 0 for rid in retrieved_report_ids]


def recall_at_k(retrieved_report_ids: list[str], gold: set[str], k: int) -> float:
    """top-k 안에서 발견된 서로 다른 gold report 수 / 전체 gold report 수."""
    if not gold:
        return 0.0
    covered = set(retrieved_report_ids[:k]) & gold
    return len(covered) / len(gold)


def reciprocal_rank(retrieved_report_ids: list[str], gold: set[str]) -> float:
    for i, rid in enumerate(retrieved_report_ids, start=1):
        if rid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_report_ids: list[str], gold: set[str], k: int) -> float:
    """Gold 가 report 단위라 report 하나에서 chunk 여러 개가 뽑힐 수 있다 — 같은
    report 를 두 번째부터 relevant(rel=1) 로 다시 세면 DCG 가 IDCG(report 개수 기준
    상한) 를 넘어 NDCG>1 이 나오는 버그가 있었다(실측 발견). "새로운 gold report 를
    처음 커버한 자리"만 relevant 로 센다(report-level dedup)."""
    seen: set[str] = set()
    rel = []
    for rid in retrieved_report_ids[:k]:
        if rid in gold and rid not in seen:
            rel.append(1)
            seen.add(rid)
        else:
            rel.append(0)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal_n = min(k, len(gold))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(retrieved_report_ids: list[str], gold: set[str], k: int) -> float:
    return 1.0 if set(retrieved_report_ids[:k]) & gold else 0.0


@dataclass
class AggregateMetrics:
    n_queries: int
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float | None
    mrr: float
    ndcg_at_10: float
    hit_at_1: float | None
    hit_at_3: float | None
    ndcg_at_5: float | None
    mean_latency_sec: float
    p95_latency_sec: float

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def evaluate_search_fn(search_fn, queries: list[dict], *, k_max: int = 20, top_n_per_report: int = 3) -> tuple[AggregateMetrics, list[QueryResult]]:
    """search_fn(query_text, k) -> list[ChunkSchema] (rank 순서).

    queries: [{"id":..., "query":..., "gold_report_ids":[...]}]
    """
    per_query: list[QueryResult] = []
    for q in queries:
        gold = set(q["gold_report_ids"])
        t0 = time.time()
        chunks: list[ChunkSchema] = search_fn(q["query"], k_max)
        latency = time.time() - t0
        retrieved_report_ids = [c.report_id for c in chunks]
        per_query.append(QueryResult(
            query_id=q["id"], retrieved_report_ids=retrieved_report_ids,
            gold_report_ids=q["gold_report_ids"], latency_sec=latency,
        ))

    def agg(fn, k=None):
        vals = [fn(r.retrieved_report_ids, set(r.gold_report_ids), k) if k is not None else fn(r.retrieved_report_ids, set(r.gold_report_ids)) for r in per_query]
        return sum(vals) / len(vals) if vals else 0.0

    latencies = sorted(r.latency_sec for r in per_query)
    p95_idx = max(0, int(len(latencies) * 0.95) - 1)

    metrics = AggregateMetrics(
        n_queries=len(per_query),
        recall_at_5=agg(recall_at_k, 5),
        recall_at_10=agg(recall_at_k, 10),
        recall_at_20=agg(recall_at_k, 20) if k_max >= 20 else None,
        mrr=agg(reciprocal_rank),
        ndcg_at_10=agg(ndcg_at_k, 10),
        hit_at_1=agg(hit_at_k, 1),
        hit_at_3=agg(hit_at_k, 3),
        ndcg_at_5=agg(ndcg_at_k, 5),
        mean_latency_sec=sum(latencies) / len(latencies) if latencies else 0.0,
        p95_latency_sec=latencies[p95_idx] if latencies else 0.0,
    )
    return metrics, per_query


# --------------------------------------------------------------------------- 순위 품질
#
# 2026-08-30 추가. recall/hit 은 "정답 문서를 가져왔는가"만 본다 — **몇 등으로**
# 가져왔는지는 안 본다. 실측에서 정기공시가 evidence_hit 94.1% 인데 상한도달은
# 71.5% 였다. 22.6%p 는 "가져오긴 했는데 하위권에 처박혀 못 쓴" 경우이고,
# 지금까지 그걸 직접 재는 지표가 없었다. 아래 세 개가 그 자리를 메운다.


def precision_at_k(retrieved_report_ids: list[str], gold: set[str], k: int) -> float:
    """top-k **조각** 중 gold report 에서 나온 비율.

    컨텍스트 예산 효율을 잰다 — LLM 에 넣는 k 칸 중 몇 칸이 정답 문서인가.
    여기서는 report-level dedup 을 하지 **않는다**: 같은 gold report 의 chunk 가
    여러 칸을 차지해도 그 칸들은 실제로 쓸모 있는 근거이므로 낭비가 아니다.
    """
    top = retrieved_report_ids[:k]
    if not top:
        return 0.0
    return sum(1 for rid in top if rid in gold) / len(top)


def average_precision_at_k(retrieved_report_ids: list[str], gold: set[str], k: int) -> float:
    """순위를 반영한 정밀도 (RAGAS 의 context precision 계열).

    gold report 를 **처음** 커버한 자리에서만 precision 을 누적한다.
    ndcg_at_k 와 같은 이유로 report-level dedup 이 필수다 — dedup 없이 세면
    report 하나가 chunk 를 여러 개 내는 순간 AP 가 1 을 넘는다.

    앞자리에서 gold 를 맞힐수록 높다:
        gold={A}, [A, X, X] -> 1.00
        gold={A}, [X, X, A] -> 0.33
    """
    seen: set[str] = set()
    hits = 0
    acc = 0.0
    for i, rid in enumerate(retrieved_report_ids[:k], start=1):
        if rid in gold and rid not in seen:
            seen.add(rid)
            hits += 1
            acc += hits / i
    denom = min(k, len(gold))
    return acc / denom if denom else 0.0


def first_relevant_rank(retrieved_report_ids: list[str], gold: set[str]) -> int | None:
    """첫 gold chunk 의 등수(1-based). 못 찾으면 None.

    평균을 낼 때 못 찾은 경우를 0 이나 k+1 로 채우면 숫자가 왜곡되므로
    None 을 그대로 돌려주고, 집계 쪽에서 '찾은 것만'의 중앙값을 쓴다.
    """
    for i, rid in enumerate(retrieved_report_ids, start=1):
        if rid in gold:
            return i
    return None
