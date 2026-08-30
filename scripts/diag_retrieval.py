#!/usr/bin/env python3
"""검색이 왜 정답을 못 찾는지 / 왜 느린지 한 번에 진단한다.

    python3 scripts/diag_retrieval.py

인덱스 적재가 5분이라 한 번 올려서 전부 본다.
  ① 정답이 든 조각이 각 경로에서 몇 등인지 (bm25/dense/sparse/융합/리랭커)
  ② 단계별 소요시간 (질의 인코딩 / 각 검색 / 융합 / 리랭커)
  ③ facts 층으로 조회하면 바로 나오는지
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ARTIFACTS = "artifacts_v2"

CASES = [
    {
        "q": "삼성전자의 주요사항보고서(자기주식취득결정)에 기재된 순자산액은 얼마인가?",
        "answer": "224,787,773,988,054",
        "company": "삼성전자",
        "key": "순자산액",
    },
]


def rank_of(ranked, needle: str) -> str:
    for i, (c, s) in enumerate(ranked, 1):
        if needle in (c.raw_text or ""):
            return f"{i}등 (score {s:.4f})"
    return f"{len(ranked)}등 밖"


def main() -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="  [%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")

    t = time.time()
    from disclosure_rag.serving import api as serving_api
    serving_api.ARTIFACTS = ARTIFACTS
    serving_api._warm()
    b = serving_api._state["bundle"]
    print(f"\n인덱스 적재 {time.time()-t:.0f}초 | 경로 {b.modes} | "
          f"융합 {b.retriever.fusion} | 리랭커 {'ON' if b.retriever.reranker else 'OFF'}")

    R = b.retriever
    for case in CASES:
        q, ans = case["q"], case["answer"]
        print(f"\n{'='*70}\n질문: {q}\n정답 문자열: {ans}\n{'='*70}")

        # ── 0) 정답이 코퍼스에 있긴 한가 ────────────────────────────────
        holders = [c for c in b.chunks if ans in (c.raw_text or "")]
        print(f"\n[0] 정답을 담은 조각: {len(holders)}개")
        for c in holders[:5]:
            print(f"    {c.chunk_id}  ({c.company} / {c.report_name})")
        if not holders:
            print("    ❌ 코퍼스에 정답 문자열이 없다 — 검색 문제가 아니라 파싱 문제")
            continue

        # ── 1) 경로별 순위 + 시간 ──────────────────────────────────────
        print(f"\n[1] 경로별 순위 (후보 50개 기준)")
        timings = {}
        t0 = time.time(); bm = R.bm25.search(q, k=50); timings["bm25"] = time.time()-t0
        print(f"    bm25    {rank_of(bm, ans):22s} {timings['bm25']*1000:7.0f}ms")

        dn = sp = None
        if R.dense is not None:
            t0 = time.time(); dn = R.dense.search(q, k=50); timings["dense"] = time.time()-t0
            print(f"    dense   {rank_of(dn, ans):22s} {timings['dense']*1000:7.0f}ms")
        if R.sparse is not None:
            t0 = time.time(); sp = R.sparse.search(q, k=50); timings["sparse"] = time.time()-t0
            print(f"    sparse  {rank_of(sp, ans):22s} {timings['sparse']*1000:7.0f}ms")

        from disclosure_rag.retrieval.fusion import reciprocal_rank_fusion
        named = [x for x in (bm, dn, sp) if x is not None]
        t0 = time.time(); fused = reciprocal_rank_fusion(named, top_k=50)
        timings["fusion"] = time.time()-t0
        print(f"    rrf융합  {rank_of(fused, ans):22s} {timings['fusion']*1000:7.0f}ms")

        if R.reranker is not None:
            t0 = time.time(); rr = R.reranker.rerank(q, fused, top_k=50)
            timings["rerank"] = time.time()-t0
            print(f"    +리랭커 {rank_of(rr, ans):22s} {timings['rerank']*1000:7.0f}ms")
            print(f"\n    리랭커 상위 5개 점수:")
            for c, s in rr[:5]:
                mark = "★정답" if ans in (c.raw_text or "") else "     "
                print(f"      {mark} {s:.4f}  {c.chunk_id}  {(c.raw_text or '')[:50]!r}")

        print(f"\n[2] 시간 합계 {sum(timings.values()):.2f}초")
        for k, v in timings.items():
            print(f"    {k:8s} {v*1000:7.0f}ms  ({v/sum(timings.values())*100:4.1f}%)")

        # ── 2.5) 근거 확장 → 컨텍스트에 정답이 살아남는가 ────────────────
        # 검색이 정답을 12개 안에 넣었는데 답변이 "확인할 수 없습니다"였다.
        # 그렇다면 검색 다음 단계에서 잘렸을 가능성이 크다.
        print(f"\n[2.5] 근거 확장 → HCX 로 가는 컨텍스트")
        hits12 = R.search(q, k=12, candidate_k=50)
        print(f"    검색 12건 중 정답 보유: "
              f"{sum(1 for c, _ in hits12 if ans in (c.raw_text or ''))}건")
        for i, (c, sc) in enumerate(hits12, 1):
            if ans in (c.raw_text or ""):
                pos = (c.raw_text or "").find(ans)
                print(f"      {i}등 {c.chunk_id}  조각길이 {len(c.raw_text)}자, "
                      f"정답 위치 {pos}자 지점")

        ev = b.parent_expander.expand(hits12, budget_chars=12000)
        n_ev_with = sum(1 for e in ev if ans in (e.get("text") or ""))
        print(f"    근거 {len(ev)}건 / 합계 {sum(len(e['text']) for e in ev):,}자"
              f" | 정답 포함 근거: {n_ev_with}건")
        for i, e in enumerate(ev, 1):
            mark = "★" if ans in (e.get("text") or "") else " "
            print(f"      {mark} [{i}] {e['chunk_id']}  {len(e['text']):,}자")

        ctx = serving_api._build_context(ev)
        print(f"\n    최종 컨텍스트 {len(ctx):,}자")
        if ans in ctx:
            pos = ctx.find(ans)
            print(f"    ✅ 정답이 컨텍스트 안에 있다 ({pos:,}자 지점)")
            print(f"       주변: ...{ctx[max(0,pos-120):pos+60]}...")
            print(f"    -> 검색·확장은 정상. **HCX 가 근거를 읽고도 못 찾은 것**")
        else:
            print(f"    ❌ 정답이 컨텍스트에서 사라졌다")
            print(f"    -> 근거 확장/컨텍스트 조립 단계에서 잘렸다")
            for i, e in enumerate(ev, 1):
                if ans in (e.get("text") or ""):
                    continue
            # 어느 조각이 잘렸는지: 원본 조각엔 있는데 확장 결과엔 없는 경우
            for c, _ in hits12:
                if ans in (c.raw_text or ""):
                    match = [e for e in ev if e["chunk_id"] == c.chunk_id]
                    if not match:
                        print(f"       {c.chunk_id}: 근거 목록에서 아예 빠짐(중복 parent 제거?)")
                    elif ans not in match[0]["text"]:
                        print(f"       {c.chunk_id}: 근거에 있으나 텍스트가 잘림 "
                              f"({len(match[0]['text']):,}자)")

        # ── 3) facts 층으로는 바로 나오나 ──────────────────────────────
        print(f"\n[3] facts 층 조회")
        if b.fact_store is None:
            print("    fact_store 없음")
        else:
            t0 = time.time()
            rows = b.fact_store.lookup(company=case["company"], key=case["key"], limit=5)
            el = time.time() - t0
            print(f"    lookup(company={case['company']!r}, key={case['key']!r}) "
                  f"-> {len(rows)}건 {el*1000:.0f}ms")
            for r in rows[:5]:
                hit = "★" if ans in str(r.get("value_text", "")) else " "
                print(f"      {hit} {r.get('value_text')}  ({r.get('report_name')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
