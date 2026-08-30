#!/usr/bin/env python3
"""L1 스냅샷(+임베딩 shard) -> 검색 인덱스 조립·저장.

    python3 scripts/build_index.py --artifacts artifacts [--tokenizer kiwi] [--no-dense]

BM25 인덱스를 파일로 저장한다. 기존 파이프라인은 프로세스를 켤 때마다 45만 chunk 를
Kiwi 형태소 분석부터 다시 돌렸다 — 평가 서버 기동 시간이 그대로 늘어난다.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from disclosure_rag.l1 import load_chunks  # noqa: E402
from disclosure_rag.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from disclosure_rag.retrieval.tokenizers import build_tokenizer  # noqa: E402


def _rss_gb() -> float:
    """이 프로세스가 지금까지 쓴 최대 메모리(GB). 단계마다 찍어서 눈으로 감시한다."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1024**3 if sys.platform == "darwin" else peak / 1024**2
    except Exception:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts_v2")
    ap.add_argument("--tokenizer", default="kiwi", help="kiwi|char_2gram|char_3gram|whitespace")
    ap.add_argument("--no-dense", action="store_true", help="dense/sparse 적재를 건너뛴다")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = Path(args.artifacts)
    t0 = time.time()
    chunks = load_chunks(root / "l1", leaf_only=True, render_text=True)
    n_leaf = len(chunks)
    print(f"leaf chunk {n_leaf:,}개 로드 ({time.time()-t0:.0f}s) | RSS {_rss_gb():.1f}GB", flush=True)

    t1 = time.time()
    bm25 = BM25Retriever(chunks, build_tokenizer(args.tokenizer))
    out = root / "index" / "bm25"
    bm25.save(out)
    print(f"BM25 인덱스 저장: {out} ({time.time()-t1:.0f}s) | RSS {_rss_gb():.1f}GB", flush=True)

    modes = ["bm25"]
    if not args.no_dense and (root / "emb").exists():
        # load_bundle 은 스냅샷을 **다시** 읽는다(설계상 단독 실행 가능해야 하므로).
        # 여기서 chunks/bm25 를 놓지 않으면 55만 ChunkSchema 4.8GB 가 두 벌 상주한다.
        # 다시 읽는 데 3분쯤 더 걸리지만 그게 메모리 부족으로 죽는 것보다 낫다.
        del chunks, bm25
        gc.collect()
        print(f"1단계 메모리 반납 | RSS {_rss_gb():.1f}GB", flush=True)

        from disclosure_rag.retrieval.index_bundle import load_bundle
        b = load_bundle(root, bm25_tokenizer=args.tokenizer)
        modes = b.modes
        print(f"검색 경로: {modes} | RSS {_rss_gb():.1f}GB", flush=True)

    (root / "index" / "index_manifest.json").write_text(json.dumps({
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_leaf_chunks": n_leaf, "bm25_tokenizer": args.tokenizer,
        "modes": modes, "elapsed_sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료 ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
