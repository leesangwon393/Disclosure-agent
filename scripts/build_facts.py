#!/usr/bin/env python3
"""facts 층 빌더 — 폼 문서에서 정형 사실을 뽑아 JSONL + SQLite 로 만든다.

    python3 scripts/build_facts.py --corpus-root ~/Desktop/미래에셋/데이터/corpus \
        --out artifacts/facts [--groups exchange,major,holding] [--limit-per-group 0]

산출물:
    facts.jsonl(.gz)   사실 1건 = 1행 (사람이 읽고 grep 할 수 있는 형태 — 제출물)
    facts.sqlite       조회용 인덱스 (평가 서버가 이 파일 하나만 들고 가면 된다)
    facts_manifest.json  생성 정보 + 품질 지표

왜 폼 문서만인가: exchange/major/holding 3,150건(전체의 75%)은 서식이 고정이라
(항목, 값) 이 그대로 사실이 된다. periodic(재무제표)은 계정과목 정규화·연결/별도·
단위 처리가 훨씬 어려워 후순위다. --groups 로 나중에 확장할 수 있게 열어 둔다.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from disclosure_rag.chunking.chunkers import chunk_document  # noqa: E402
from disclosure_rag.common.manifest_loader import load_manifest  # noqa: E402
from disclosure_rag.common.unicode_utils import PathResolver  # noqa: E402
from disclosure_rag.correction.correction_graph_builder import build_correction_index  # noqa: E402
from disclosure_rag.facts.extractor import FORM_GROUPS, extract_facts, link_facts_to_chunks  # noqa: E402
from disclosure_rag.facts.store import FactStore  # noqa: E402
from disclosure_rag.parsing.document_detector import parse_documents_for_row  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--out", default="artifacts_v2/facts")
    ap.add_argument("--groups", default=",".join(FORM_GROUPS))
    ap.add_argument("--limit-per-group", type=int, default=0, help="0=전체")
    ap.add_argument("--no-gzip", action="store_true")
    args = ap.parse_args()

    corpus = str(Path(args.corpus_root).expanduser())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    groups = {g.strip() for g in args.groups.split(",") if g.strip()}

    t0 = time.time()
    manifest = load_manifest(corpus)
    resolver = PathResolver(corpus)
    corrections = build_correction_index(manifest, resolver)

    rows = [r for r in manifest if r.doc_group in groups]
    if args.limit_per_group:
        seen: Counter = Counter(); picked = []
        for r in rows:
            if seen[r.doc_group] < args.limit_per_group:
                picked.append(r); seen[r.doc_group] += 1
        rows = picked
    print(f"대상 문서 {len(rows)}건 ({sorted(groups)})", flush=True)

    ext = "" if args.no_gzip else ".gz"
    opener = (lambda p: open(p, "w", encoding="utf-8")) if args.no_gzip else (
        lambda p: gzip.open(p, "wt", encoding="utf-8", compresslevel=6))

    store = FactStore(out / "facts.sqlite")
    store.clear()

    n_facts = n_linked = n_docs = n_failed = 0
    by_group: Counter = Counter()
    top_keys: Counter = Counter()
    periodic_filter_stats: Counter = Counter()

    with opener(out / f"facts.jsonl{ext}") as f:
        for i, row in enumerate(rows, 1):
            corr = corrections.get(row.doc_id)
            if corr is None:
                n_failed += 1
                continue
            try:
                parsed_docs = parse_documents_for_row(row, resolver)
            except Exception as e:  # noqa: BLE001
                print(f"[SKIP] {row.doc_id}: {type(e).__name__}: {e}", file=sys.stderr)
                n_failed += 1
                continue

            doc_facts = []
            doc_chunks = []
            for parsed in parsed_docs:
                if parsed.report_subtype == "unsupported_pdf_html":
                    continue
                doc_facts.extend(extract_facts(
                    parsed, row, corr, filter_stats=periodic_filter_stats,
                ))
                doc_chunks.extend(chunk_document(parsed, row, corr))
            if not doc_facts:
                continue

            n_linked += link_facts_to_chunks(doc_facts, doc_chunks)
            store.insert_many(doc_facts)
            for fact in doc_facts:
                f.write(fact.model_dump_json(exclude_none=True) + "\n")
                by_group[fact.doc_group] += 1
                top_keys[fact.key_norm] += 1
            n_facts += len(doc_facts)
            n_docs += 1
            if i % 300 == 0:
                print(f"  {i}/{len(rows)} 문서, fact {n_facts}건 ({time.time()-t0:.0f}s)", flush=True)

    st = store.stats()
    store.close()
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "corpus_root": corpus, "groups": sorted(groups),
        "documents_scanned": len(rows), "documents_with_facts": n_docs, "documents_failed": n_failed,
        "facts": n_facts,
        "linked_to_chunk_pct": round(n_linked / max(1, n_facts) * 100, 1),
        "numeric_pct": round(st["numeric"] / max(1, st["n"]) * 100, 1),
        "dated_pct": round(st["dated"] / max(1, st["n"]) * 100, 1),
        "distinct_keys": st["keys"], "companies": st["companies"],
        "by_group": dict(by_group),
        "periodic_filter": dict(periodic_filter_stats),
        "top_keys": top_keys.most_common(25),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "facts_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "top_keys"}, ensure_ascii=False, indent=2))
    print("\n자주 나오는 항목 25개:")
    for k, c in meta["top_keys"]:
        print(f"  {c:6d}  {k}")
    print(f"\nfacts: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
