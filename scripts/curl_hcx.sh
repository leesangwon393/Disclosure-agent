#!/usr/bin/env bash
# 파이썬 없이 HCX 를 직접 때려본다.
# 목적: KeyboardInterrupt 가 파이썬(requests/urllib3) 쪽 문제인지,
#       네트워크·CLOVA 쪽 문제인지 가른다.
#   - curl 도 느리거나 끊긴다  -> 네트워크/서버 문제
#   - curl 은 멀쩡한데 파이썬만 -> 파이썬 환경 문제
set -u
cd "$(dirname "$0")/.."
[ -f .env ] || { echo ".env 가 없습니다"; exit 1; }
KEY=$(grep '^HCX_API_KEY=' .env | cut -d= -f2-)
MODEL=$(grep '^HCX_MODEL=' .env | cut -d= -f2-)
echo "model=$MODEL  key=${KEY:0:5}…${KEY: -4}"
# HCX-007 은 maxTokens 를 거부한다("Invalid parameter: maxTokens", 실측).
# maxCompletionTokens 를 써야 한다 — hcx_client.py 도 모델별로 이름을 바꿔 보낸다.
case "$MODEL" in
  *007*) MAXPARAM="maxCompletionTokens" ;;
  *)     MAXPARAM="maxTokens" ;;
esac
PAYLOAD='{"messages":[{"role":"user","content":"한 문장으로만 답해라. 대한민국의 수도는?"}],"'"$MAXPARAM"'":50,"temperature":0.3}'
echo "maxTokens 파라미터명: $MAXPARAM"
echo "─────────────────────────────────────────"

for i in 1 2 3; do
  echo "[${i}회차]"
  curl -sS -w '  → HTTP %{http_code} | DNS %{time_namelookup}s | 연결 %{time_connect}s | TLS %{time_appconnect}s | 첫바이트 %{time_starttransfer}s | 총 %{time_total}s\n' \
       -o /tmp/hcx_out.json \
       --max-time 40 \
       -X POST "https://clovastudio.stream.ntruss.com/v3/chat-completions/$MODEL" \
       -H "Authorization: Bearer $KEY" \
       -H "Content-Type: application/json" \
       -d "$PAYLOAD"
  echo "  응답: $(head -c 200 /tmp/hcx_out.json)"
  echo
done
