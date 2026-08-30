#!/usr/bin/env python3
"""실험 (b) — 청크 크기·표 표현을 실제로 정한다.

배경
----
`CHILD_TARGET_TOKENS=600 / CHILD_MAX_TOKENS=1000` 은 **한 번도 ablation 된 적이 없다.**
설계 문서의 "400~800 목표"를 인용한 값이다. Stage 1 은 *전략*(section-aware vs fixed)
비교였지 *크기* 튜닝이 아니었다.

그리고 문헌은 우리가 오히려 작을 수 있다고 시사한다:
  - Snowflake (SEC 10-K/Q 23,000 PDF, 500 질의): **1,800자 내외가 최적**,
    지나치게 크면 10~20% 하락. 우리 leaf p50 = 1,417자.
  - arXiv 2605.00318 (STC, MAUD 39,231건): 표를 **행 단위 key-value 블록**으로 표현하면
    hybrid MRR 0.358 -> 0.595, BM25 R@1 0.366 -> 0.754. 단 KV 표현만 하고 구조 인식
    분할을 안 하면 오히려 baseline 보다 나빴다 -> 이득의 본체는 "행 경계 준수"다.

Stage 1 의 결함 3개를 반복하지 않는다
-------------------------------------
1. **모든 조건에 컨텍스트 헤더를 동일하게 적용한다.** Stage 1 은 fixed_500 baseline 만
   `[회사]/[공시]/[Section]` 헤더가 없어서, "청킹 전략의 승리"와 "메타데이터 부착의
   승리"가 분리되지 않았다. (Snowflake 는 메타데이터 부착만으로 50~60% -> 72~75% 라고
   보고한다 — 즉 그 혼입은 결과를 통째로 뒤집을 수 있는 크기다.)
   여기서는 render_search_text() 가 모든 조건에 자동 적용되므로 구조적으로 통제된다.
2. **여러 회사로 한다.** Stage 1 은 삼성전자 1개사 33문서였다.
3. **이 스크립트를 저장소에 남긴다.** Stage 1 은 실험 스크립트가 세션 scratchpad 에만
   있었고 결과만 남아서 재현이 불가능했다. 결과 폴더에 실행 인자도 함께 기록한다.

채점
----
`--gold` 로 두 방식을 모두 낸다.
  - **passage**: 검색된 조각의 raw_text 에 정답 문자열이 있으면 정답 (조각 수준)
  - **report**: 검색된 조각의 report_id 가 gold 문서면 정답 (기존 방식, 문서 수준)
문서 수준 지표는 청크 크기에 거의 반응하지 않는다 — 비교용으로만 같이 출력한다.

임베딩이 필요 없다
------------------
BM25 인덱스는 몇 초면 만들어진다. 크기를 5가지로 바꿔도 **임베딩 0시간**이다.
여기서 후보를 2개로 좁힌 뒤 그것만 dense 로 검증하면 된다.

사용:
  python3 scripts/exp_chunk_size.py \
      --corpus-root ~/Desktop/미래에셋/데이터/corpus \
      --gold eval/gold_passages.jsonl \
      --sizes 300,450,600,900,1200 --table-styles grid,kv \
      --n-companies 10 --out results/chunk_size
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from disclosure_rag.chunking.chunk_schema import filter_leaf_chunks, set_token_counter  # noqa: E402
from disclosure_rag.chunking.chunkers import ChunkConfig, chunk_document  # noqa: E402
from disclosure_rag.common.manifest_loader import load_manifest  # noqa: E402
from disclosure_rag.common.unicode_utils import PathResolver  # noqa: E402
from disclosure_rag.correction.correction_graph_builder import build_correction_index  # noqa: E402
from disclosure_rag.parsing.document_detector import parse_documents_for_row  # noqa: E402
from disclosure_rag.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from disclosure_rag.retrieval.tokenizers import build_tokenizer  # noqa: E402

K_LIST = (1, 3, 5, 10, 20)


# ------------------------------------------------------------------ 지표
def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def _metrics(rel_flags: list[int]) -> dict:
    """rel_flags: 순위대로 0/1. 정답이 여러 조각일 수 있으므로 recall 은
    "top-k 안에 정답 조각이 하나라도 있는가"(=hit)로 잰다 — 조각 수준에서는
    정답 조각의 총 개수를 알 수 없으므로 이게 정직한 정의다."""
    out = {}
    for k in K_LIST:
        out[f"hit@{k}"] = 1.0 if any(rel_flags[:k]) else 0.0
    rr = 0.0
    for i, r in enumerate(rel_flags, start=1):
        if r:
            rr = 1.0 / i
            break
    out["mrr"] = rr
    # nDCG: 정답 조각이 몇 개인지 알 수 없는 설정이라("정답 문자열이 든 조각이면 정답"),
    # ideal 을 "정답 1개"로 잡으면 안 된다. 실측에서 hit@5=0.538 인데 nDCG=0.954 가
    # 나오는 모순이 있었다. 검색된 것 중 최선의 순서를 ideal 로 삼는다.
    ideal = _dcg(sorted(rel_flags[:10], reverse=True)) or 1.0
    out["ndcg@10"] = _dcg(rel_flags[:10]) / ideal
    return out


def _aggregate(per_query: list[dict]) -> dict:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: round(sum(q[k] for q in per_query) / len(per_query), 4) for k in keys}


# ------------------------------------------------------------------ 본체
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--gold", default="eval/gold_passages.jsonl")
    ap.add_argument("--out", default="results/chunk_size")
    ap.add_argument("--sizes", default="300,450,600,900,1200",
                    help="target_tokens 목록. max_tokens 는 target*1.67 로 둔다(현행 600/1000 비율)")
    ap.add_argument("--table-styles", default="grid,kv")
    ap.add_argument("--table-rows", default="20", help="표 분할 행 수 목록")
    ap.add_argument("--tokenizer", default="kiwi", help="BM25 토크나이저: kiwi|char_2gram|whitespace")
    ap.add_argument("--n-companies", type=int, default=10, help="0=전체")
    ap.add_argument("--max-docs", type=int, default=400,
                    help="파싱 결과를 메모리에 들고 조건별로 재사용하므로 문서 수가 메모리를 정한다. "
                         "정기공시(사업보고서)는 1건이 수백 MB 트리가 되므로 함부로 늘리지 말 것. "
                         "0=제한없음")
    ap.add_argument("--groups", default="",
                    help="건초더미에 넣을 doc_group 제한 (예: exchange,major,holding). 비우면 전체")
    ap.add_argument("--real-tokenizer", action="store_true",
                    help="BGE-M3 실제 토크나이저로 토큰을 센다(권장). 없으면 heuristic")
    ap.add_argument("--seed", type=int, default=20260823)
    args = ap.parse_args()

    # --- 토큰 카운터: 크기 실험이므로 '토큰'의 정의가 실제와 같아야 의미가 있다 ---
    tok_name = "heuristic(chars/2.0)"
    if args.real_tokenizer:
        try:
            from transformers import AutoTokenizer  # type: ignore
            tk = AutoTokenizer.from_pretrained("BAAI/bge-m3")
            set_token_counter(lambda s: len(tk.encode(s, add_special_tokens=False)))
            tok_name = "BAAI/bge-m3"
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] 실제 토크나이저 실패({type(e).__name__}) — heuristic 으로 진행", file=sys.stderr)

    corpus = str(Path(args.corpus_root).expanduser())
    gold = [json.loads(l) for l in Path(args.gold).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not gold:
        print("gold 가 비어 있다. scripts/make_gold_passages.py 를 먼저 돌릴 것.", file=sys.stderr)
        return 2

    manifest = load_manifest(corpus)
    resolver = PathResolver(corpus)
    corrections = build_correction_index(manifest, resolver)

    # 평가셋에 등장하는 회사 = 반드시 포함. 나머지는 무작위로 채워 '건초더미'를 만든다.
    # (검색 난이도를 실제에 가깝게 하려면 정답이 없는 문서도 인덱스에 있어야 한다.)
    need = {g["company"] for g in gold}
    need_docs = {rid for g in gold for rid in g["gold_report_ids"]}
    rnd = random.Random(args.seed)
    others = sorted({r.corp_name for r in manifest} - need)
    if args.n_companies:
        extra = rnd.sample(others, max(0, min(args.n_companies - len(need), len(others))))
    else:
        extra = others
    companies = need | set(extra)

    pool = [r for r in manifest if r.corp_name in companies]
    if args.groups:
        want = {g.strip() for g in args.groups.split(",") if g.strip()}
        pool = [r for r in pool if r.doc_group in want or r.doc_id in need_docs]

    # 정답이 들어 있는 문서는 반드시 포함하고, 나머지는 '건초더미'로 채운다.
    # 파싱 트리를 메모리에 들고 있으므로 문서 수가 곧 메모리다 — 상한을 둔다.
    must = [r for r in pool if r.doc_id in need_docs]
    rest = [r for r in pool if r.doc_id not in need_docs]
    rnd.shuffle(rest)
    rows = must + rest
    if args.max_docs and len(rows) > args.max_docs:
        keep_rest = max(0, args.max_docs - len(must))
        rows = must + rest[:keep_rest]
        print(f"[제한] --max-docs {args.max_docs} 적용: 정답문서 {len(must)}건 + 건초더미 {keep_rest}건",
              file=sys.stderr)
    print(f"회사 {len(companies)}개 / 문서 {len(rows)}건 / 질의 {len(gold)}개 "
          f"| 토큰카운터={tok_name}", file=sys.stderr)

    # --- 파싱은 조건마다 반복할 필요가 없다. 한 번만 하고 재사용한다(실험 시간 절약) ---
    t0 = time.time()
    parsed_cache: list[tuple] = []
    for i, row in enumerate(rows, 1):
        c = corrections.get(row.doc_id)
        if c is None:
            continue
        try:
            for parsed in parse_documents_for_row(row, resolver):
                if parsed.report_subtype != "unsupported_pdf_html":
                    parsed_cache.append((parsed, row, c))
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {row.doc_id}: {type(e).__name__}", file=sys.stderr)
        if i % 50 == 0:
            print(f"  파싱 {i}/{len(rows)} ({time.time()-t0:.0f}s)", file=sys.stderr)
    print(f"파싱 완료 {len(parsed_cache)}건 ({time.time()-t0:.0f}s)", file=sys.stderr)

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    styles = [x.strip() for x in args.table_styles.split(",") if x.strip()]
    row_caps = [int(x) for x in args.table_rows.split(",") if x.strip()]

    tokenizer = build_tokenizer(args.tokenizer)
    results = []

    for target in sizes:
        for style in styles:
            for rcap in row_caps:
                cfg = ChunkConfig(
                    target_tokens=target,
                    max_tokens=int(target * 1.67),
                    whole_doc_max_tokens=int(target * 1.67),
                    table_max_rows=rcap,
                    table_style=style,
                )
                t1 = time.time()
                chunks = []
                for parsed, row, corr in parsed_cache:
                    chunks.extend(chunk_document(parsed, row, corr, cfg))
                leaf = filter_leaf_chunks(chunks)
                build_t = time.time() - t1

                t2 = time.time()
                retriever = BM25Retriever(leaf, tokenizer)
                index_t = time.time() - t2

                per_q_passage, per_q_report, unresolved = [], [], 0
                t3 = time.time()
                for g in gold:
                    hits = retriever.search(g["query"], k=max(K_LIST))
                    answers = g.get("answers") or [g["answer"]]
                    gold_ids = set(g["gold_report_ids"])
                    flags_p = [1 if any(a in c.raw_text for a in answers) else 0 for c, _s in hits]
                    flags_r = [1 if c.report_id in gold_ids else 0 for c, _s in hits]
                    if not any(flags_p):
                        unresolved += 1
                    per_q_passage.append(_metrics(flags_p))
                    per_q_report.append(_metrics(flags_r))
                search_t = time.time() - t3

                lens = sorted(len(c.raw_text) for c in leaf)
                rec = {
                    "config": cfg.label, "target_tokens": target, "max_tokens": cfg.max_tokens,
                    "table_style": style, "table_max_rows": rcap,
                    "n_leaf": len(leaf), "n_parent": len(chunks) - len(leaf),
                    "leaf_chars_p50": lens[len(lens) // 2] if lens else 0,
                    "leaf_chars_p90": lens[int(len(lens) * .9)] if lens else 0,
                    "leaf_chars_max": lens[-1] if lens else 0,
                    "passage": _aggregate(per_q_passage),
                    "report": _aggregate(per_q_report),
                    "unresolved_queries": unresolved,
                    "sec": {"chunk": round(build_t, 1), "index": round(index_t, 1),
                            "search": round(search_t, 1)},
                }
                results.append(rec)
                p = rec["passage"]
                print(f"{cfg.label:28s} leaf={len(leaf):6d} p50={rec['leaf_chars_p50']:5d} "
                      f"| passage hit@5={p['hit@5']:.3f} mrr={p['mrr']:.3f} ndcg@10={p['ndcg@10']:.3f} "
                      f"| {build_t + index_t + search_t:.0f}s", flush=True)

    results.sort(key=lambda r: (-r["passage"].get("mrr", 0), -r["passage"].get("hit@5", 0)))
    out = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "token_counter": tok_name, "bm25_tokenizer": args.tokenizer,
        "n_companies": len(companies), "n_docs": len(rows), "n_parsed": len(parsed_cache),
        "n_queries": len(gold), "gold": str(Path(args.gold).resolve()),
        "note": "모든 조건에 동일한 컨텍스트 헤더가 적용됨(render_search_text). Stage 1 의 혼입 없음.",
        "results": results,
    }
    (out / "results.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 청크 크기·표 표현 실험", "",
             f"- 회사 {len(companies)} / 문서 {len(rows)} / 질의 {len(gold)}",
             f"- 토큰 카운터: `{tok_name}` · BM25 토크나이저: `{args.tokenizer}`",
             "- 모든 조건에 컨텍스트 헤더 동일 적용 (Stage 1 의 baseline 혼입 없음)", "",
             "| config | leaf | p50자 | p90자 | **hit@5** | **MRR** | nDCG@10 | (문서수준 MRR) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(
            f"| `{r['config']}` | {r['n_leaf']:,} | {r['leaf_chars_p50']} | {r['leaf_chars_p90']} "
            f"| {r['passage']['hit@5']:.3f} | **{r['passage']['mrr']:.3f}** | {r['passage']['ndcg@10']:.3f} "
            f"| {r['report']['mrr']:.3f} |")
    lines += ["", "## 읽는 법", "",
              "- **passage 지표가 판단 기준이다.** 문서수준(MRR) 은 청크 크기에 거의 반응하지 않아",
              "  비교용으로만 실었다 — 두 열의 차이가 작다면 그건 '크기가 무의미하다'가 아니라",
              "  '문서수준 라벨로는 크기를 못 잰다'는 뜻이다.",
              "- 상위 2개를 골라 dense 로 재검증한 뒤 `ChunkConfig` 기본값을 바꾼다.",
              "- `unresolved_queries` 가 크면 정답 문자열이 어떤 조각에도 없다는 뜻이다 —",
              "  파싱 유실을 의심하고 `tests/test_properties.py` 부터 확인할 것."]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n결과: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
