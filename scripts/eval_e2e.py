#!/usr/bin/env python3
"""검색 계층 E2E 평가 — 어떤 조합이 실제로 좋은지 잰다.

비교하는 것 (있는 것만 자동으로):
    bm25            BM25 단독 (임베딩 0)
    dense           Dense 단독
    sparse          BGE-M3 learned sparse 단독
    hybrid_rrf      RRF 융합            (기존 프로덕션 방식)
    hybrid_weighted normalized weighted (Stage 4 승자, KIM 브랜치 기본)
    +rerank         위 최고 조합 + 리랭커

특히 확인할 것 세 가지
---------------------
1. **Dense 를 얹으면 실제로 오르는가.** 상원 실측에서 하이브리드가 BM25 단독에
   **지는 구간이 있었다**(R@5 0.706 vs 0.661). 안 오르면 안 넣으면 된다 —
   `HybridRetriever(dense=None)` 이라 되돌리는 비용이 0이다.
2. **RRF vs weighted.** Stage 4 승자(weighted)가 프로덕션에 반영돼 있지 않았다.
3. **리랭커 재측정.** Stage 11 최고점(R@5 0.820)인데 258배 지연으로 기각됐고,
   그 지연의 원인이 26,027자 outlier chunk 였을 가능성이 크다. KIM 브랜치에서
   최대 2,059자로 잘렸으므로 **다시 재야 한다.** `BM25 + 리랭커` 조합은 아직
   실험된 적이 없다.

채점은 조각 수준(정답 문자열 포함)과 문서 수준을 모두 낸다.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

K_LIST = (1, 3, 5, 10, 20)


def _dcg(rels): return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def _metrics(flags: list[int]) -> dict:
    out = {f"hit@{k}": (1.0 if any(flags[:k]) else 0.0) for k in K_LIST}
    out["mrr"] = next((1.0 / i for i, r in enumerate(flags, 1) if r), 0.0)
    # ideal 을 "정답 1개"로 잡으면 안 된다(정답 조각이 여러 개일 수 있는 설정).
    out["ndcg@10"] = _dcg(flags[:10]) / (_dcg(sorted(flags[:10], reverse=True)) or 1.0)
    return out


def _agg(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in rows[0]}


def _run(name, search_fn, gold, results, *, k=max(K_LIST)):
    per_p, per_r, lat = [], [], []
    for g in gold:
        t = time.time()
        hits = search_fn(g["query"], k)
        lat.append(time.time() - t)
        answers = g.get("answers") or [g["answer"]]
        ids = set(g["gold_report_ids"])
        per_p.append(_metrics([1 if any(a in c.raw_text for a in answers) else 0 for c, _ in hits]))
        per_r.append(_metrics([1 if c.report_id in ids else 0 for c, _ in hits]))
    lat.sort()
    rec = {"method": name, "passage": _agg(per_p), "report": _agg(per_r),
           "latency_mean_sec": round(sum(lat) / len(lat), 4),
           "latency_p95_sec": round(lat[int(len(lat) * .95) - 1], 4)}
    results.append(rec)
    p = rec["passage"]
    print(f"{name:20s} hit@1={p['hit@1']:.3f} hit@5={p['hit@5']:.3f} mrr={p['mrr']:.3f} "
          f"ndcg@10={p['ndcg@10']:.3f} | {rec['latency_mean_sec']*1000:7.1f}ms", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts_v2")
    ap.add_argument("--gold", default="eval/gold_passages.jsonl")
    ap.add_argument("--out", default="results/e2e")
    ap.add_argument("--rerank", action="store_true", help="리랭커도 잰다(느리다)")
    ap.add_argument("--rerank-top-n", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="질의 수 제한(빠른 확인용)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from disclosure_rag.retrieval.fusion import (normalized_weighted_fusion,
                                                 reciprocal_rank_fusion)
    from disclosure_rag.retrieval.index_bundle import load_bundle

    gold = [json.loads(l) for l in Path(args.gold).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        gold = gold[: args.limit]
    b = load_bundle(args.artifacts)
    print(f"검색 경로 {b.modes} | leaf {len(b.chunks):,} | 질의 {len(gold)}\n")

    R = b.retriever
    results: list[dict] = []

    _run("bm25", lambda q, k: R.bm25.search(q, k=k), gold, results)
    if R.dense is not None:
        _run("dense", lambda q, k: R.dense.search(q, k=k), gold, results)
    if R.sparse is not None:
        _run("sparse", lambda q, k: R.sparse.search(q, k=k), gold, results)

    if R.dense is not None or R.sparse is not None:
        def _named(q, k):
            d = {"bm25": R.bm25.search(q, k=50)}
            if R.dense is not None:
                d["dense"] = R.dense.search(q, k=50)
            if R.sparse is not None:
                d["sparse"] = R.sparse.search(q, k=50)
            return d
        _run("hybrid_rrf",
             lambda q, k: reciprocal_rank_fusion(list(_named(q, k).values()), top_k=k),
             gold, results)
        _run("hybrid_weighted",
             lambda q, k: normalized_weighted_fusion(_named(q, k), weights=R.weights, top_k=k),
             gold, results)

    if args.rerank:
        try:
            from disclosure_rag.retrieval.reranker import CrossEncoderReranker
            rr = CrossEncoderReranker()
            base = (lambda q, n: R.search(q, k=n, candidate_k=args.rerank_top_n))
            _run(f"best+rerank(top{args.rerank_top_n})",
                 lambda q, k: rr.rerank(q, base(q, args.rerank_top_n), top_k=k), gold, results)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] 리랭커: {type(e).__name__}: {e}", file=sys.stderr)

    results.sort(key=lambda r: -r["passage"]["mrr"])
    out = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(
        {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "modes": b.modes,
         "n_leaf": len(b.chunks), "n_queries": len(gold), "weights": R.weights,
         "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 검색 E2E 비교", "", f"- 검색 경로 {b.modes} / leaf {len(b.chunks):,} / 질의 {len(gold)}",
             f"- fusion 가중치 {R.weights}", "",
             "| 방법 | hit@1 | hit@5 | **MRR** | nDCG@10 | 지연(ms) | (문서수준 MRR) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        p = r["passage"]
        lines.append(f"| {r['method']} | {p['hit@1']:.3f} | {p['hit@5']:.3f} | **{p['mrr']:.3f}** "
                     f"| {p['ndcg@10']:.3f} | {r['latency_mean_sec']*1000:.0f} | {r['report']['mrr']:.3f} |")
    lines += ["", "## 판단 기준", "",
              "- `hybrid_*` 가 `bm25` 를 **못 넘으면 Dense 를 넣지 않는다.** 되돌리는 비용은 0이다",
              "  (`HybridRetriever(dense=None)`).",
              "- `hybrid_weighted` < `hybrid_rrf` 면 가중치를 다시 잡는다(현재 BM25 쪽으로 기울여 둠).",
              "- 리랭커는 지연이 300초 예산 안에 들어오는지 함께 본다. 평가는 sequential 호출이다."]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n결과: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
