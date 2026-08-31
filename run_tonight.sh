#!/usr/bin/env bash
# 멀티기업 28 -> suite_v2 296 -> suite_v1 38 연속 측정. 자는 동안 돌리는 용도.
# 전체 약 2시간 15분.
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

# 멀티기업을 맨 앞에 둔다. 제일 짧고(30분) 이번에 고친 것을 직접 재는
# 유일한 묶음이라, 뒤에서 뭐가 잘못돼도 이건 건진다.
echo "════════ 1. 멀티기업 28문항 (약 30분) ════════"
./run.sh python3 scripts/score_answers.py --gold eval/suite_multi.jsonl \
    --mode full --pipeline v2 --thinking off --yes --out results/multi1
echo "1단계 종료 코드: $?"
# 값이 아니라 출처를 본다 — HCX 를 다시 부르지 않고 답변 파일만 읽는다.
./run.sh python3 scripts/score_attribution.py --gold eval/suite_multi.jsonl \
    --answers results/multi1/answers.jsonl --out results/multi1
echo

echo "════════ 2. suite_v2 296문항 (약 1시간 30분) ════════"
./run.sh python3 scripts/score_answers.py --gold eval/suite_v2.jsonl \
    --mode full --pipeline v2 --thinking off --yes --out results/v2suite_full3
echo "2단계 종료 코드: $?"
echo

echo "════════ 3. suite_v1 38문항 (약 15분) ════════"
./run.sh python3 scripts/score_answers.py --gold eval/suite_v1.jsonl \
    --mode full --pipeline v2 --thinking off --yes --out results/v2_off8b
echo "3단계 종료 코드: $?"
echo

echo "════════ 전부 끝 $(date '+%Y-%m-%d %H:%M:%S') ════════"
for d in results/multi1 results/v2suite_full3 results/v2_off8b; do
  [ -f "$d/metrics.json" ] && python3 -c "
import json,sys
d=json.load(open('$d/metrics.json',encoding='utf-8'))
print(f\"  $d  n={d['n']} graded_hit={d.get('graded_hit')} {d['labels']}\")
b=d.get('latency_breakdown') or {}
if b:
    top=sorted(((v['median_ms'],k) for k,v in b.items() if k!='total'), reverse=True)[:3]
    print('     느린 단계 top3(중앙값): ' + ', '.join(f'{k} {v:.0f}ms' for v,k in top))"
done

echo
[ -f results/multi1/attribution.json ] && python3 -c "
import json
d=json.load(open('results/multi1/attribution.json',encoding='utf-8'))
print(f\"  멀티기업 귀속: 커버리지 {d['coverage']:.1%}  완전정상 {d['clean']}/{d['n']}  남의회사 인용 {d['foreign_rows']}건\")"
