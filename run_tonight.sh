#!/usr/bin/env bash
# 296문항 -> 38문항 연속 측정. 자는 동안 돌리는 용도.
#
# 어젯밤 실패에서 배운 것: 코드를 고치는 도중에 실행을 시작하면 파이썬이
# 서로 안 맞는 중간 상태를 읽는다. 그래서 **먼저 테스트로 트리가 일관된지
# 확인하고**, 통과할 때만 본 측정에 들어간다. 2시간을 날리지 않기 위한 45초다.
set -u
cd "$(dirname "$0")"

echo "════════ 0. 사전 점검 (코드가 일관된가) ════════"
if ! ./run.sh python3 -m pytest -q -p no:cacheprovider \
        --ignore=tests/test_dense_retriever.py --ignore=tests/test_router.py; then
  echo "❌ 테스트 실패 — 측정을 시작하지 않는다. 위 로그를 확인하라."
  exit 1
fi
echo "✅ 사전 점검 통과"
echo

echo "════════ 1. suite_v2 296문항 (약 1시간 30분) ════════"
./run.sh python3 scripts/score_answers.py --gold eval/suite_v2.jsonl \
    --mode full --pipeline v2 --thinking off --yes --out results/v2suite_full2
echo "1단계 종료 코드: $?"
echo

echo "════════ 2. suite_v1 38문항 (약 15분) ════════"
./run.sh python3 scripts/score_answers.py --gold eval/suite_v1.jsonl \
    --mode full --pipeline v2 --thinking off --yes --out results/v2_off7
echo "2단계 종료 코드: $?"
echo

echo "════════ 전부 끝 $(date '+%Y-%m-%d %H:%M:%S') ════════"
for d in results/v2suite_full2 results/v2_off7; do
  [ -f "$d/metrics.json" ] && python3 -c "
import json,sys
d=json.load(open('$d/metrics.json',encoding='utf-8'))
print(f\"  $d  n={d['n']} graded_hit={d.get('graded_hit')} {d['labels']}\")"
done
