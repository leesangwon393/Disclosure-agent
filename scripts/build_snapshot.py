#!/usr/bin/env python3
"""L1 정규화 스냅샷 빌더 — "그릇".

기존 파이프라인은 파싱·청킹 결과를 **어디에도 저장하지 않고** 매 실행마다
원본 XML 5.2GB 부터 다시 만들었다(약 9분). 유일한 영속물이 파이썬 pickle 캐시였다.
이 스크립트가 그 결과를 불변 스냅샷으로 확정한다.

산출물 (--out 아래):
  documents.jsonl    문서 1건 = 1행. 파싱 성공/경고/청크수 포함
  chunks.jsonl       청크 1개 = 1행. **parent 포함 전량** (인덱싱 대상은 is_leaf=true)
  corrections.jsonl  정정 그래프 (group/order/is_latest)
  build_manifest.json  재현 정보 + 품질 지표

왜 저장하는가 (성능이 아니라 이 세 가지 때문):
  1) 대회가 전처리 산출물(파싱 결과·인덱스)을 제출 대상으로 명시했다.
     pickle 은 제출물로 부적합하다(파이썬 버전 종속, 열어볼 수 없음).
  2) 코퍼스 71.8% 훼손 버그가 오래 안 보였던 이유가 중간 산출물이 눈에 안 보였기
     때문이다. JSONL 이면 grep 으로 바로 확인된다.
  3) 나중에 상원님 결과와 합칠 때 **코드가 아니라 산출물을 diff** 할 수 있다.

사용:
  python3 scripts/build_snapshot.py --corpus-root ~/Desktop/미래에셋/데이터/corpus \
      --out artifacts/l1 [--limit-per-group 20] [--groups periodic,exchange] [--tokenizer]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from disclosure_rag.chunking.chunk_schema import (  # noqa: E402
    filter_leaf_chunks,
    set_token_counter,
    token_counter_is_exact,
)
from disclosure_rag.chunking.chunkers import chunk_document  # noqa: E402
from disclosure_rag.common.corpus_validator import validate_corpus  # noqa: E402
from disclosure_rag.common.manifest_loader import load_manifest  # noqa: E402
from disclosure_rag.common.unicode_utils import PathResolver  # noqa: E402
from disclosure_rag.correction.correction_graph_builder import build_correction_index  # noqa: E402
from disclosure_rag.parsing.document_detector import parse_documents_for_row  # noqa: E402


def _maybe_load_tokenizer() -> str | None:
    """실제 BGE-M3 토크나이저를 붙인다. 없으면 heuristic 으로 진행하고 그 사실을 기록한다."""
    try:
        from transformers import AutoTokenizer  # type: ignore
        tk = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        set_token_counter(lambda s: len(tk.encode(s, add_special_tokens=False)))
        return "BAAI/bge-m3"
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 실제 토크나이저 로드 실패({type(e).__name__}) — heuristic(chars/2.0)으로 진행", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--out", default="artifacts_v2/l1")
    ap.add_argument("--limit-per-group", type=int, default=0, help="0=전체")
    ap.add_argument("--groups", default="", help="쉼표구분. 비우면 전체")
    ap.add_argument("--tokenizer", action="store_true", help="실제 BGE-M3 토크나이저 사용")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--keep-text", action="store_true",
                    help="검색용 text 를 스냅샷에 그대로 저장(기본은 생략 — raw_text+메타로 재현 가능, 용량 29%% 절감)")
    ap.add_argument("--no-gzip", action="store_true")
    args = ap.parse_args()

    corpus = os.path.expanduser(args.corpus_root)
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)

    tok_name = _maybe_load_tokenizer() if args.tokenizer else None
    t0 = time.time()

    manifest = load_manifest(corpus)
    resolver = PathResolver(corpus)

    validation = None
    if not args.no_validate:
        report, _ = validate_corpus(corpus)
        validation = {"healthy": report.is_healthy(), "render": report.render()[:4000]}
        if not report.is_healthy():
            print("[WARN] corpus validation UNHEALTHY — 통계를 신뢰하기 전에 원인 확인", file=sys.stderr)

    correction_index = build_correction_index(manifest, resolver)

    rows = manifest
    if args.groups:
        want = {g.strip() for g in args.groups.split(",") if g.strip()}
        rows = [r for r in rows if r.doc_group in want]
    if args.limit_per_group:
        seen: Counter = Counter()
        picked = []
        for r in rows:
            if seen[r.doc_group] < args.limit_per_group:
                picked.append(r)
                seen[r.doc_group] += 1
        rows = picked

    ext = "" if args.no_gzip else ".gz"
    opener = (lambda p: open(p, "w", encoding="utf-8")) if args.no_gzip else (
        lambda p: gzip.open(p, "wt", encoding="utf-8", compresslevel=6))
    f_doc = opener(out / f"documents.jsonl{ext}")
    f_chunk = opener(out / f"chunks.jsonl{ext}")
    f_corr = opener(out / f"corrections.jsonl{ext}")

    n_chunks = n_leaf = n_parent = n_docs = n_failed = 0
    lengths: list[int] = []
    ctype = Counter()
    group_leaf = Counter()
    group_docs = Counter()
    n_refs = 0
    n_unit = 0
    warnings_total = 0

    for i, row in enumerate(rows, 1):
        correction = correction_index.get(row.doc_id)
        if correction is None:
            n_failed += 1
            f_doc.write(json.dumps({"doc_id": row.doc_id, "ok": False,
                                    "error": "correction record 없음"}, ensure_ascii=False) + "\n")
            continue
        f_corr.write(json.dumps({
            "doc_id": row.doc_id,
            "correction_group_id": correction.correction_group_id,
            "correction_order": correction.correction_order,
            "is_latest": correction.is_latest,
            "is_correction": row.is_correction,
        }, ensure_ascii=False) + "\n")

        try:
            parsed_docs = parse_documents_for_row(row, resolver)
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            f_doc.write(json.dumps({"doc_id": row.doc_id, "ok": False,
                                    "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False) + "\n")
            continue

        doc_chunks = 0
        doc_warn: list[str] = []
        for parsed in parsed_docs:
            if parsed.report_subtype == "unsupported_pdf_html":
                doc_warn.append("unsupported_pdf_html (PDF+HTML 대체수집 3건 중 하나)")
                continue
            doc_warn.extend(parsed.parse_warnings)
            chunks = chunk_document(parsed, row, correction)
            leaf_ids = {c.chunk_id for c in filter_leaf_chunks(chunks)}
            for c in chunks:
                # 용량 최적화 (실측 기준: field_codes 37% / text 29% / raw_text 28%)
                #  - null 필드 제거
                #  - 검색용 text 는 raw_text + 메타로 결정론적 재현 가능하므로 기본 생략
                #    (l1.py 의 load_chunks(render_text=True) 가 되살린다)
                rec = c.model_dump(exclude_none=True)
                if not args.keep_text:
                    rec.pop("text", None)
                rec["is_leaf"] = c.chunk_id in leaf_ids
                f_chunk.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks += 1
                doc_chunks += 1
                if rec["is_leaf"]:
                    n_leaf += 1
                    lengths.append(len(c.raw_text))
                    ctype[c.content_type] += 1
                    group_leaf[row.doc_group] += 1
                    n_refs += len(c.field_codes)
                    if c.unit_hint:
                        n_unit += 1
                else:
                    n_parent += 1

        warnings_total += len(doc_warn)
        n_docs += 1
        group_docs[row.doc_group] += 1
        f_doc.write(json.dumps({
            "doc_id": row.doc_id, "ok": True, "corp_name": row.corp_name,
            "corp_code": row.corp_code, "doc_group": row.doc_group,
            "doc_subtype": row.doc_subtype, "report_nm": row.report_nm,
            "rcept_dt": row.rcept_dt, "is_correction": row.is_correction,
            "file_path": row.file_path, "n_chunks": doc_chunks,
            "parse_warnings": doc_warn,
        }, ensure_ascii=False) + "\n")

        if i % 200 == 0:
            print(f"  {i}/{len(rows)} 문서, chunk {n_chunks}, {time.time()-t0:.0f}s", flush=True)

    for f in (f_doc, f_chunk, f_corr):
        f.close()

    lengths.sort()
    def q(p: float) -> int:
        return lengths[min(len(lengths) - 1, int(len(lengths) * p))] if lengths else 0

    build = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "branch": "kim",
        "corpus_root": corpus,
        "python": sys.version.split()[0],
        "text_field_stored": bool(args.keep_text),
        "gzip": not args.no_gzip,
        "token_counter": tok_name or "heuristic(chars/2.0)",
        "token_counter_exact": token_counter_is_exact(),
        "args": vars(args),
        "counts": {
            "documents_requested": len(rows), "documents_ok": n_docs, "documents_failed": n_failed,
            "chunks_total": n_chunks, "leaf": n_leaf, "parent": n_parent,
            "field_refs": n_refs, "parse_warnings": warnings_total,
        },
        "leaf_chars": {"p50": q(.5), "p90": q(.9), "p99": q(.99),
                       "max": lengths[-1] if lengths else 0,
                       "over_2000_pct": round(sum(1 for x in lengths if x > 2000) / max(1, len(lengths)) * 100, 2)},
        "leaf_content_type_pct": {k: round(v / max(1, n_leaf) * 100, 1) for k, v in ctype.most_common()},
        "leaf_per_doc_by_group": {g: round(group_leaf[g] / max(1, group_docs[g]), 1) for g in sorted(group_docs)},
        "unit_hint_leaf_pct": round(n_unit / max(1, n_leaf) * 100, 1),
        "elapsed_sec": round(time.time() - t0, 1),
        "validation": validation,
    }
    (out / "build_manifest.json").write_text(json.dumps(build, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: build[k] for k in
                      ("counts", "leaf_chars", "leaf_content_type_pct", "leaf_per_doc_by_group",
                       "unit_hint_leaf_pct", "token_counter", "elapsed_sec")},
                     ensure_ascii=False, indent=2))
    print(f"\n스냅샷: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
