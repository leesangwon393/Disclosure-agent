#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 의미 블록 청킹(2026-08-25) 반영 → 전체 재구축. 약 9시간.
#
# 기존 artifacts/ 는 **건드리지 않는다.** 전부 artifacts_v2/ 에 만든다.
# 새 결과가 더 나쁘면 그냥 artifacts_v2/ 를 지우면 되고, 그 사이에도
# 지금 인덱스로 서비스가 가능하다.
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
CORPUS="$HOME/Desktop/미래에셋/데이터/corpus"
OUT=artifacts_v2
LOG="$OUT/logs"
mkdir -p "$LOG"

run () {   # run <이름> <명령...>
  local name="$1"; shift
  local f="$LOG/${name}.log"
  echo "════════ $name 시작 $(date '+%H:%M:%S') ════════" | tee -a "$LOG/_all.log"
  # >> : 재시작해도 이전 기록을 지우지 않는다(끊긴 지점 추적용)
  if "$@" >> "$f" 2>&1; then
    echo "✅ $name 완료 $(date '+%H:%M:%S')" | tee -a "$LOG/_all.log"
  else
    echo "❌ $name 실패 (exit $?) — $f 확인" | tee -a "$LOG/_all.log"
    tail -20 "$f" | tee -a "$LOG/_all.log"
    exit 1
  fi
}

# START_AT=3 ./rebuild_v2.sh  처럼 특정 단계부터 이어서 돌릴 수 있다.
# (배터리 방전 등으로 중간에 끊겼을 때 앞 단계를 다시 돌리지 않기 위함)
START_AT="${START_AT:-1}"
step () { [ "$1" -ge "$START_AT" ]; }

echo "시작 $(date '+%Y-%m-%d %H:%M:%S')  (START_AT=$START_AT)" | tee -a "$LOG/_all.log"

step 1 && run 1_snapshot python3 scripts/build_snapshot.py --corpus-root "$CORPUS" --out "$OUT/l1"
step 2 && run 2_facts    python3 scripts/build_facts.py    --corpus-root "$CORPUS" --out "$OUT/facts"
step 3 && run 3_embed    python3 scripts/embed_corpus.py   --snapshot "$OUT/l1"    --out "$OUT/emb"
step 4 && run 4_index    python3 scripts/build_index.py    --artifacts "$OUT"
step 5 && run 5_eval     python3 scripts/eval_e2e.py --artifacts "$OUT" --rerank \
                      --out "$OUT/results_e2e"

echo "════════ 전부 완료 $(date '+%Y-%m-%d %H:%M:%S') ════════" | tee -a "$LOG/_all.log"
[ -f "$LOG/5_eval.log" ] && grep -E "hit@" "$LOG/5_eval.log" | tee -a "$LOG/_all.log"
echo "비교 기준(기존 청킹): bm25 hit@5=0.455 / hybrid_rrf 0.580 / +rerank 0.643 mrr 0.507" \
  | tee -a "$LOG/_all.log"
