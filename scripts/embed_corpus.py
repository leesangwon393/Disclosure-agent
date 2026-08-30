#!/usr/bin/env python3
"""L1 스냅샷 -> BGE-M3 3모드 임베딩 (dense + sparse + colbert).

핵심 전제
---------
dense / sparse / colbert 는 **같은 forward pass 의 서로 다른 출력 헤드**다.
dense 만 뽑으나 셋 다 뽑으나 모델 연산은 사실상 동일하다.
전체 코퍼스 임베딩은 M5 Pro/MPS 실측 61.4ms/chunk 기준 8~13시간짜리 작업이므로,
**그 한 번으로 세 가지를 다 확보**해두면 이후 조합 실험을 재임베딩 없이 할 수 있다.

입력이 원본 XML 이 아니라 **L1 스냅샷**인 점이 기존 스크립트와 다르다.
파싱을 다시 돌리지 않으므로 (1) 9분이 절약되고 (2) 무엇보다 임베딩된 텍스트가
스냅샷과 **비트 단위로 동일**함이 보장된다. 기존에는 임베딩 시점의 파싱 결과가
남아 있지 않아 "이 벡터가 어떤 텍스트에서 나왔는지" 사후 확인이 불가능했다
(코퍼스 71.8% 훼손 버그가 늦게 발견된 구조적 이유이기도 하다).

사용:
  python3 scripts/embed_corpus.py --snapshot artifacts/l1 --out artifacts/emb \
      [--device mps] [--batch-size 64] [--limit 2000]
  (--colbert 는 코퍼스 전체 저장용이 아니다. 아래 '메모리' 참고)

산출물:
  dense_XXXX.npz        chunk_ids + float16 (N, 1024)
  sparse_XXXX.jsonl.gz  {"chunk_id":..., "w": {token_id: weight}}
  colbert_XXXX.npz      (--colbert + --force 일 때만. RAM 60GB+/디스크 800GB+ 필요 —
                         사실상 쓰면 안 된다. ColBERT 는 질의 시점 리랭커로 쓴다)
  progress.json         중단 후 이어서 실행
  embed_manifest.json   재현 정보 + 실측 처리량
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SHARD = 20_000


def _rss_gb() -> float:
    """현재 프로세스 상주 메모리(GB). 샤드마다 찍어서 누수를 눈으로 본다."""
    try:
        import resource, sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1024**3 if sys.platform == "darwin" else peak / 1024**2
    except Exception:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="build_snapshot.py 산출 디렉터리")
    ap.add_argument("--out", default="artifacts_v2/emb")
    ap.add_argument("--device", default=None, help="mps|cuda|cpu (기본: 자동)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="M5 Pro 24GB+ 면 64~128 이 32보다 20~40%% 빠른 경우가 많다")
    ap.add_argument("--max-length", type=int, default=8192,
                    help="모델 상한(8192)을 그대로 쓴다. 배치 안 최장 텍스트에 맞춰 동적 패딩되므로 "
                         "상한을 높게 둬도 실제 비용이 늘지 않는다 — 우리 조각 최대가 2,059자라 "
                         "8192 로 두면 잘림이 원천적으로 불가능하고 속도 손해도 없다")
    ap.add_argument("--colbert", action="store_true",
                    help="[위험] 코퍼스 전체 multi-vector 저장. RAM 60GB+/디스크 800GB+ 필요. "
                         "ColBERT 는 질의 시점에 후보 50건만 인코딩하는 리랭커로 쓰는 게 정석이다")
    ap.add_argument("--no-sparse", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bench-only", action="store_true", help="200건 벤치만 하고 종료")
    ap.add_argument("--force", action="store_true", help="잘림 경고를 무시하고 진행")
    args = ap.parse_args()

    import numpy as np
    from disclosure_rag.l1 import iter_leaf_texts, load_build_manifest
    from disclosure_rag.retrieval.embeddings import BgeM3MultiProvider

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from disclosure_rag.common.device import pick_device

    device = pick_device(args.device)
    print(f"device={device}  batch={args.batch_size}  max_length={args.max_length}")

    # ChunkSchema 객체를 만들지 않고 (id, text) 만 읽는다 — 전체 코퍼스의
    # field_refs 8,141만 개를 객체화하면 메모리가 감당이 안 된다.
    ids, texts = [], []
    for cid, text in iter_leaf_texts(args.snapshot):
        ids.append(cid); texts.append(text)
        if args.limit and len(ids) >= args.limit:
            break
    print(f"leaf chunk {len(texts):,}개 (인덱싱 대상). parent 는 임베딩하지 않는다.")

    # === ColBERT 전체 저장 가드 ==============================================
    # 2026-08-23 사고: --colbert 로 돌렸다가 샤드 0(2만 건)의 multi-vector 를
    # 파이썬 리스트로 들고 있는 순간 맥북이 메모리 부족으로 꺼졌다.
    # dense/sparse 는 조각당 벡터 1개지만, colbert 는 **토큰당 벡터 1개**다.
    if args.colbert:
        avg_tok = sum(len(t) for t in texts) / max(len(texts), 1) / 1.81  # 실측 1.81 자/토큰
        ram_gb = SHARD * avg_tok * 1024 * 4 / 1024**3      # fp32 리스트 (샤드 1개분)
        disk_gb = len(texts) * avg_tok * 1024 * 2 / 1024**3  # fp16 전체
        print(f"[colbert] 예상 RAM {ram_gb:.0f}GB/샤드, 디스크 {disk_gb:.0f}GB/전체", flush=True)
        if not args.force:
            print("[중단] ColBERT 전체 저장은 이 장비에서 불가능합니다.\n"
                  "       ColBERT 는 검색 후보 상위 50건만 질의 시점에 인코딩하는 리랭커로 쓰십시오\n"
                  "       (colbert_reranker.ColbertReranker 는 vector_lookup 주입식이라 그대로 됩니다).\n"
                  "       --colbert 없이 다시 실행하세요. 정말 강행하려면 --force.", flush=True)
            return 2


    # === 사전 점검: max_length 초과로 조용히 잘리는 청크가 없는지 확인 ===
    # 청크 크기는 heuristic(chars/2.0)으로 정해졌는데, 실제 BGE-M3 토크나이저의
    # 한국어 char/token 비율은 그것과 다르다. 추정이 낙관적이면 청크가 max_length 를
    # 넘어 **뒷부분이 조용히 잘린 채 임베딩된다** — 우리가 파싱에서 고쳐온 것과
    # 정확히 같은 종류의 silent 손실이다. 15시간을 태우기 전에 여기서 잡는다.
    try:
        from transformers import AutoTokenizer  # type: ignore
        tk = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        import random as _rnd
        sample = _rnd.Random(20260823).sample(texts, min(1500, len(texts)))
        lens = [len(tk.encode(t, add_special_tokens=True)) for t in sample]
        lens.sort()
        over = sum(1 for x in lens if x > args.max_length)
        cpt = sum(len(t) for t in sample) / max(1, sum(lens))
        print(f"[사전점검] 실제 토큰 p50={lens[len(lens)//2]} p90={lens[int(len(lens)*.9)]} "
              f"max={lens[-1]} | {cpt:.2f} chars/token (heuristic 은 2.00)")
        if over:
            pct = over / len(lens) * 100
            print(f"[사전점검] ⚠️  max_length={args.max_length} 초과 {pct:.1f}% — "
                  f"이만큼이 뒤가 잘린 채 임베딩된다. --max-length 를 {((lens[-1]//256)+1)*256} 이상으로 올리거나 "
                  f"chunk_schema.set_token_counter() 로 재청킹할 것.", file=sys.stderr)
            if not args.force:
                print("[중단] 그대로 진행하려면 --force", file=sys.stderr)
                return 2
        else:
            print(f"[사전점검] ✅ 잘리는 청크 없음")
    except ImportError:
        print("[사전점검] transformers 없음 — 토큰 길이 확인 생략(잘림 위험 미검증)", file=sys.stderr)

    provider = BgeM3MultiProvider(device=device)

    # --- 벤치마크: 무작정 몇 시간 태우기 전에 실측 ETA 를 먼저 뽑는다 ---
    n_bench = min(200, len(texts))
    t0 = time.time()
    provider.encode_all(texts[:n_bench], batch_size=args.batch_size, max_length=args.max_length,
                        sparse=not args.no_sparse, colbert=args.colbert)
    ms = (time.time() - t0) / n_bench * 1000
    eta_h = len(texts) * ms / 1000 / 3600
    print(f"[벤치] {n_bench}건 -> {ms:.1f} ms/chunk | 전체 {len(texts)}건 예상 {eta_h:.1f}시간 "
          f"(CPU 실측 591.7ms 대비 {591.7/ms:.1f}배)")
    if args.bench_only:
        return 0

    prog = out / "progress.json"
    start = json.loads(prog.read_text())["next_idx"] if prog.exists() else 0
    # progress.json 은 샤드를 **다 쓴 뒤** 갱신된다. 샤드 저장 도중에 죽으면
    # dense/sparse 파일은 멀쩡히 남았는데 progress 만 뒤처져 그 샤드를 다시 돈다.
    # 그래서 실제로 존재하는 산출물을 직접 훑어서 시작점을 올린다.
    scanned = 0
    while (out / f"dense_{scanned // SHARD:04d}.npz").exists() and (
        args.no_sparse or (out / f"sparse_{scanned // SHARD:04d}.jsonl.gz").exists()
    ):
        scanned += SHARD
    if scanned > start:
        print(f"[재개] 완성된 샤드 {scanned // SHARD}개 발견 -> idx={scanned} 부터")
        start = scanned
    if start:
        print(f"이어서 진행: idx={start}")

    t_all = time.time()
    for s in range(start, len(texts), SHARD):
        e = min(s + SHARD, len(texts))
        t0 = time.time()
        res = provider.encode_all(texts[s:e], batch_size=args.batch_size,
                                  max_length=args.max_length,
                                  sparse=not args.no_sparse, colbert=args.colbert)
        idx = s // SHARD

        np.savez_compressed(out / f"dense_{idx:04d}.npz",
                            chunk_ids=np.array(ids[s:e], dtype=object),
                            vectors=np.asarray(res["dense_vecs"], dtype=np.float16))

        if not args.no_sparse:
            with gzip.open(out / f"sparse_{idx:04d}.jsonl.gz", "wt", encoding="utf-8") as f:
                for cid, w in zip(ids[s:e], res["lexical_weights"]):
                    f.write(json.dumps(
                        {"chunk_id": cid, "w": {str(k): round(float(v), 4) for k, v in w.items()}},
                        ensure_ascii=False) + "\n")

        if args.colbert:
            vecs = res["colbert_vecs"]
            lens = np.array([len(v) for v in vecs], dtype=np.int32)
            np.savez_compressed(
                out / f"colbert_{idx:04d}.npz",
                chunk_ids=np.array(ids[s:e], dtype=object),
                flat=np.concatenate([np.asarray(v, dtype=np.float16) for v in vecs], axis=0),
                lengths=lens,
            )

        del res
        gc.collect()

        el = time.time() - t0
        print(f"shard {idx}: {e-s}건 {el:.0f}s ({el/(e-s)*1000:.1f} ms/chunk) "
              f"| RSS {_rss_gb():.1f}GB | 남은 예상 {(len(texts)-e)*(el/(e-s))/3600:.1f}시간",
              flush=True)
        prog.write_text(json.dumps({"next_idx": e}))

    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "snapshot": str(Path(args.snapshot).resolve()),
        "snapshot_build": load_build_manifest(args.snapshot).get("created_at"),
        "model": "BAAI/bge-m3", "device": device,
        "modes": {"dense": True, "sparse": not args.no_sparse, "colbert": bool(args.colbert)},
        "batch_size": args.batch_size, "max_length": args.max_length,
        "n_chunks": len(texts), "ms_per_chunk_bench": round(ms, 2),
        "elapsed_sec": round(time.time() - t_all, 1),
    }
    (out / "embed_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
