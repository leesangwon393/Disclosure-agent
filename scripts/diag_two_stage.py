#!/usr/bin/env python3
"""2단계 검색 + 필터 순서 수정의 효과를 실제 인덱스로 잰다. HCX 호출 0회.

## 배경 (2026-08-30 실측)

60문항 full 실행에서 같은 질문에 대해

    필터 없이 그냥 검색   정답문서 회수 94.4%
    에이전트가 검색       정답문서 회수 66.7%

에이전트는 `company` 필터를 걸고 검색한다. 그런데 리트리버가 "전체에서 상위
N개를 뽑고 그 다음에 필터로 거른다" 순서라, 좁은 필터에서는 통과 대상이 상위
N개 안에 거의 없어 결과가 무너진다. 이 스크립트는 네 가지를 같은 질문 집합에
돌려 그 가설을 확인하고, 2단계 검색의 이득을 분리해 잰다.

    A. 필터 없음                 (기존 채점기와 동일한 조건 = 상한 기준선)
    B. company 필터 + 옛 pool 규칙  (버그 재현 — 코드가 아니라 여기서 직접 재현한다)
    C. company 필터 + 새 규칙       (필터 순서 수정만의 효과)
    D. 2단계 검색                 (C + 문서 확정 후 재검색)

측정값은 채점기와 같은 정의다.

    evidence_hit    반환된 chunk 중 gold 문서의 것이 있는가
    answer_ceiling  반환된 chunk 본문 안에 정답 문자열이 있는가 (= 답할 수 있었는가)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger("diag_two_stage")


def _norm(s: str) -> str:
    return re.sub(r"[,\s원₩]", "", s or "")


def _golds(row: dict) -> list[str]:
    out = [a for a in (row.get("answers") or []) if a]
    if row.get("answer"):
        out.append(row["answer"])
    return list(dict.fromkeys(out))


def _score(results, row) -> tuple[int, int]:
    gold_ids = set(row.get("gold_report_ids") or row.get("gold_doc_ids") or [])
    golds = _golds(row)
    ev = int(any(c.report_id in gold_ids for c, _ in results))
    ceil = int(any(_norm(g) and _norm(g) in _norm(c.raw_text) for c, _ in results for g in golds))
    return ev, ceil


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/gold_passages_clean.jsonl")
    ap.add_argument("--artifacts", default=os.environ.get("ARTIFACTS", "artifacts_v2"))
    ap.add_argument("--sample", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--no-reranker", action="store_true")
    ap.add_argument("--out", default="results/diag_two_stage")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    import numpy as np

    from disclosure_rag.agent.tools import make_search_disclosures_tool
    from disclosure_rag.retrieval.index_bundle import load_bundle
    from disclosure_rag.retrieval.metadata_filter import RetrievalFilter

    rows = [json.loads(l) for l in Path(args.gold).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if _golds(r) and r.get("company")]
    import random
    rng = random.Random(args.seed)
    if args.sample and args.sample < len(rows):
        rows = rng.sample(rows, args.sample)
    logger.info("표본 %d문항", len(rows))

    t0 = time.time()
    bundle = load_bundle(args.artifacts)
    if not args.no_reranker:
        try:
            from disclosure_rag.retrieval.reranker import CrossEncoderReranker
            bundle.retriever.reranker = CrossEncoderReranker()
        except Exception as e:  # noqa: BLE001
            logger.warning("리랭커 없이 진행 (%s)", type(e).__name__)
    logger.info("인덱스 적재 %.0fs", time.time() - t0)
    retriever = bundle.retriever

    # --- B: 옛 pool 규칙 재현. 코드를 되돌리지 않고 여기서 직접 흉내낸다.
    def old_pool_search(query, k, flt):
        dense = getattr(retriever, "dense", None)
        if dense is None or not hasattr(dense, "matrix"):
            return retriever.search(query, k=k, flt=flt)   # numpy dense 가 아니면 비교 불가
        q = np.asarray(dense.embedding_provider.embed_query(query), dtype=np.float32)
        scores = dense.matrix @ q if dense.matrix.dtype == np.float32 else \
            np.concatenate([dense.matrix[s:s + 50000].astype(np.float32) @ q
                            for s in range(0, dense.matrix.shape[0], 50000)])
        pool = min(len(scores), max(k * 20, 1000))          # 옛 규칙
        top = np.argpartition(-scores, pool)[:pool]
        order = top[np.argsort(-scores[top])]
        out = []
        for i in order:
            c = dense.chunks[int(i)]
            if flt is not None and not flt.matches(c):
                continue
            out.append((c, float(scores[i])))
            if len(out) >= k:
                break
        return out

    two_stage = make_search_disclosures_tool(retriever, default_k=args.k, two_stage=True)
    one_stage = make_search_disclosures_tool(retriever, default_k=args.k, two_stage=False)

    def via_tool(tool, row):
        out = tool.handler(query=row["query"], company=row["company"], top_k=args.k)
        ids = {e["chunk_id"] for e in out["results"]}
        # tool 은 dict 를 돌려주므로 채점을 위해 chunk 객체가 필요 없다 — 직접 계산
        gold_ids = set(row.get("gold_report_ids") or [])
        golds = _golds(row)
        ev = int(any(e["report_id"] in gold_ids for e in out["results"]))
        ceil = int(any(_norm(g) and _norm(g) in _norm(e.get("text") or "")
                       for e in out["results"] for g in golds))
        return ev, ceil

    agg = {name: [0, 0] for name in ("A_필터없음", "B_옛규칙+회사필터", "C_새규칙+회사필터", "D_2단계검색")}
    per_q = []
    for i, row in enumerate(rows, 1):
        company_flt = RetrievalFilter(companies=[row["company"]])
        rec = {"query": row["query"], "company": row["company"]}

        for name, results in (
            ("A_필터없음", retriever.search(row["query"], k=args.k)),
            ("B_옛규칙+회사필터", old_pool_search(row["query"], args.k, company_flt)),
            ("C_새규칙+회사필터", retriever.search(row["query"], k=args.k, flt=company_flt)),
        ):
            ev, ceil = _score(results, row)
            agg[name][0] += ev; agg[name][1] += ceil
            rec[name] = f"ev={ev} ceil={ceil}"

        ev, ceil = via_tool(two_stage, row)
        agg["D_2단계검색"][0] += ev; agg["D_2단계검색"][1] += ceil
        rec["D_2단계검색"] = f"ev={ev} ceil={ceil}"
        per_q.append(rec)

        if i % 10 == 0 or i == len(rows):
            # 주의: 조립한 문자열에 % 가 남아 있으면 logger 가 포맷으로 해석해 터진다.
            # 완성된 문자열을 인자 없이 넘긴다.
            logger.info("[%d/%d] %s", i, len(rows),
                        " | ".join(f"{n} ceil {v[1]/i:.0%}" for n, v in agg.items()))

    n = len(rows)
    print()
    print(f"{'방식':<20}{'정답문서 회수':>14}{'상한도달':>12}")
    print("-" * 48)
    for name, (ev, ceil) in agg.items():
        print(f"{name:<20}{ev/n:>13.1%}{ceil/n:>12.1%}")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"n": n, "k": args.k, "reranker": not args.no_reranker,
         "results": {name: {"evidence_hit": round(ev / n, 4), "answer_ceiling": round(ceil / n, 4)}
                     for name, (ev, ceil) in agg.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "per_question.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in per_q), encoding="utf-8")
    print(f"\n저장: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
