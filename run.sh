#!/usr/bin/env bash
# 무엇을 돌리든 이걸로 감싸면 전 과정이 artifacts/logs/ 에 남는다.
# 그 폴더는 Claude 가 직접 읽을 수 있으므로 터미널 내용을 붙여넣을 필요가 없다.
#
#   ./run.sh python3 scripts/exp_chunk_size.py --corpus-root ... 
#   ./run.sh -bg python3 scripts/embed_corpus.py ...      # 백그라운드(터미널 닫아도 계속)
#
set -uo pipefail
cd "$(dirname "$0")"

# 파이썬 출력이 파일로 리다이렉트되면 블록 버퍼링이 걸려 로그가 몇십 분씩 비어 보인다.
# (실측: 11시간짜리 임베딩이 첫 shard 전까지 로그가 헤더 230B 뿐이었다.)
export PYTHONUNBUFFERED=1
mkdir -p artifacts/logs

BG=0
if [ "${1:-}" = "-bg" ]; then BG=1; shift; fi

# `python3` 를 프로젝트 가상환경으로 바꿔치기한다.
#
# 2026-09-01 실패: `.venv` 를 activate 하지 않은 셸에서 run_tonight.sh 를 돌리자
# 시스템 python3.14 가 잡혀 "No module named pytest" 로 즉시 죽었다. 의존성은
# 전부 .venv(python3.11)에만 있다. activate 를 잊는 건 사람 잘못이 아니라
# 스크립트가 챙길 일이다.
case "${1:-}" in
  python3|python)
    if [ -x ".venv/bin/python3" ]; then
      shift; set -- ".venv/bin/python3" "$@"
    elif [ -d ".venv" ]; then
      echo "⚠️  .venv 는 있는데 .venv/bin/python3 을 못 쓴다 — 시스템 파이썬으로 돌린다." >&2
      echo "    의존성이 .venv 에만 있으면 곧바로 ModuleNotFoundError 로 죽는다." >&2
    fi
    ;;
esac

# 로그 이름은 실행하는 .py 파일에서 딴다. `-m pytest` 처럼 모듈 실행이면
# 모듈 이름을 쓴다 — 예전엔 전부 `_-m.log` 로 뭉뚱그려졌다.
NAME=run
for arg in "$@"; do
  case "$arg" in
    *.py) NAME=$(basename -- "$arg" .py); break ;;
  esac
done
if [ "$NAME" = "run" ]; then
  case " $* " in
    *" -m "*) NAME=$(echo "$*" | sed -n 's/.* -m \([A-Za-z0-9_.]*\).*/\1/p') ;;
  esac
  [ -z "$NAME" ] && NAME=run
fi
TS=$(date +%Y%m%d_%H%M%S)
LOG="artifacts/logs/${TS}_${NAME}.log"

{
  echo "=== CMD: $* ==="
  echo "=== START: $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "=== PY: $(python3 -V 2>&1) | $(python3 -c 'import sys;print(sys.prefix)') ==="
  echo
} > "$LOG"

run() {
  "$@" >> "$LOG" 2>&1
  code=$?
  { echo; echo "=== END: $(date '+%Y-%m-%d %H:%M:%S') exit=$code ==="; } >> "$LOG"
  return $code
}

ln -sfn "$(basename "$LOG")" artifacts/logs/latest.log

if [ "$BG" = "1" ]; then
  # macOS 에는 `setsid` 가 없다(리눅스 util-linux 전용). 예전 버전이 그걸 쓰면서
  # 에러를 /dev/null 로 버려 **조용히 아무것도 실행되지 않았다.**
  # nohup 만으로도 SIGHUP(터미널 닫힘)은 충분히 막힌다.
  # 중첩 따옴표를 피하려고 실행할 내용을 임시 스크립트로 떨궈서 돌린다.
  RUNNER=$(mktemp "${TMPDIR:-/tmp}/runsh.XXXXXX")
  {
    echo '#!/usr/bin/env bash'
    echo "cd $(printf '%q' "$PWD")"
    echo "export PYTHONUNBUFFERED=1"
    printf '%q ' "$@"; echo
    echo 'ec=$?'
    echo 'echo'
    echo 'echo "=== END: $(date "+%Y-%m-%d %H:%M:%S") exit=$ec ==="'
  } > "$RUNNER"
  chmod +x "$RUNNER"
  nohup "$RUNNER" >> "$LOG" 2>&1 &
  PID=$!
  sleep 2
  if kill -0 "$PID" 2>/dev/null; then
    echo "✅ 백그라운드 실행 중 (PID $PID) — 터미널 닫아도 계속됩니다"
  else
    echo "❌ 시작 직후 종료됐습니다. 로그를 확인하세요: $LOG"
    tail -20 "$LOG"
    exit 1
  fi
  echo "로그: $LOG"
  echo "진행 보기: tail -f $LOG"
  echo "살아있나: ps -p $PID   /   pgrep -fl embed_corpus"
else
  echo "로그: $LOG"
  run "$@"
  code=$?
  tail -25 "$LOG"
  exit $code
fi
