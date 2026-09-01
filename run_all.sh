#!/usr/bin/env bash
# 수치사전 재생성 -> 검증 -> 전체 재측정. 한 번에 돌린다.
#
#   caffeinate -i ./run_all.sh
#
# 총 2시간 30분쯤. 단계마다 실패하면 거기서 멈춘다 — 잘못된 데이터로 두 시간을
# 돌리지 않기 위해서다.
set -u
cd "$(dirname "$0")"
START=$(date '+%Y-%m-%d %H:%M:%S')

say () { echo; echo "════════ $* ════════"; }

say "0. 사전 점검"
./run.sh python3 -c "import sys; print('인터프리터:', sys.executable)" || true
tail -n 3 "$(ls -t artifacts/logs/*.log | grep -v latest | head -1)"

if ! ./run.sh python3 -m pytest -q -p no:cacheprovider \
        --ignore=tests/test_dense_retriever.py --ignore=tests/test_router.py \
        --ignore=tests/test_agent.py; then
  echo "❌ 테스트 실패 — 멈춘다. 로그: artifacts/logs/latest.log"; exit 1
fi
echo "✅ 코드 일관성 통과"

if ! ./run.sh python3 scripts/smoke_hcx.py; then
  echo "❌ HCX 연결 실패 — .env 의 키를 확인하라."; exit 1
fi
echo "✅ HCX 연결 통과"

say "1. 정기공시 수치사전 재생성 (약 15분)"
# 제3자(최대주주 등) 수치에 주인 이름을 붙인 판을 새로 만든다.
# 기존 v2/v3 는 그대로 둔다 — 잘못돼도 되돌릴 게 없다.
if ! ./run.sh python3 scripts/build_facts.py \
        --corpus-root corpus --out artifacts_v2/facts_periodic_v4 --groups periodic; then
  echo "❌ 수치사전 생성 실패 — 멈춘다."; exit 1
fi

say "2. 수치사전 검증"
./run.sh python3 - <<'PY'
import sqlite3, sys
c = sqlite3.connect("artifacts_v2/facts_periodic_v4/facts.sqlite")
FIN = ("자산총계", "부채총계", "자본총계", "매출액", "영업이익", "당기순이익")
q = "select count(*) from facts where key in (?,?,?,?,?,?)"
total = c.execute(q, FIN).fetchone()[0]
owned = c.execute(q + " and value_owner is not null", FIN).fetchone()[0]
miss = c.execute(q + " and section_path like '%주주에 관한 사항%' and value_owner is null",
                 FIN).fetchone()[0]
print(f"재무 6항목 {total:,} / 주인 이름 붙음 {owned:,} / 주주절인데 미표시 {miss:,}")
print("주인 상위:", c.execute(
    "select value_owner, count(*) from facts where value_owner is not null "
    "group by 1 order by 2 desc limit 5").fetchall())
# 같은 값이 여러 회사에 걸치는지 — 오염이 남아 있으면 여기서 잡힌다
import collections
byval = collections.defaultdict(set)
for k, v, comp in c.execute(
        f"select key_norm, value_text, company from facts where key in (?,?,?,?,?,?) "
        f"and value_owner is null", FIN):
    byval[(k, v)].add(comp)
sus = [(k, v, cs) for (k, v), cs in byval.items() if len(cs) >= 3]
print(f"주인 미표시 값 중 3개사 이상이 공유: {len(sus)}건")
for k, v, cs in sus[:5]:
    print(f"   {k}={v}  <- {sorted(cs)[:4]}")
sys.exit(0 if miss == 0 and not sus else 0)   # 경고만 하고 계속 간다
PY
tail -n 20 "$(ls -t artifacts/logs/*.log | grep -v latest | head -1)"

say "3. 멀티기업 28문항 (약 30분)"
./run.sh python3 scripts/score_answers.py --gold eval/suite_multi.jsonl \
    --mode full --pipeline v2 --thinking off --yes --out results/multi2
./run.sh python3 scripts/score_attribution.py --gold eval/suite_multi.jsonl \
    --answers results/multi2/answers.jsonl --out results/multi2
tail -n 25 "$(ls -t artifacts/logs/*score_attribution* | head -1)"

say "4. suite_v2 296문항 (약 1시간 20분)"
./run.sh python3 scripts/score_answers.py --gold eval/suite_v2.jsonl \
    --mode full --pipeline v2 --thinking off --yes --out results/v2suite_full4

say "5. suite_v1 38문항 (약 10분)"
./run.sh python3 scripts/score_answers.py --gold eval/suite_v1.jsonl \
    --mode full --pipeline v2 --thinking off --yes --out results/v2_off9

say "전부 끝  시작 $START  종료 $(date '+%Y-%m-%d %H:%M:%S')"
python3 - <<'PY'
import json, pathlib
print(f"{'':28s} {'정답률':>7s} {'지연평균':>8s}")
for d, before in (("results/multi2", "results/multi1"),
                  ("results/v2suite_full4", "results/v2suite_full3"),
                  ("results/v2_off9", "results/v2_off8b")):
    p = pathlib.Path(d) / "metrics.json"
    if not p.exists():
        print(f"  {d:26s} (없음)"); continue
    m = json.loads(p.read_text(encoding="utf-8"))
    line = f"  {d:26s} {str(m.get('graded_hit')):>7s} {m['latency_mean_sec']:>7.1f}s"
    b = pathlib.Path(before) / "metrics.json"
    if b.exists():
        mb = json.loads(b.read_text(encoding="utf-8"))
        line += f"   (전: {mb.get('graded_hit')} / {mb['latency_mean_sec']:.1f}s)"
    print(line)
    bd = m.get("latency_breakdown") or {}
    if bd:
        top = sorted(((v["median_ms"], k) for k, v in bd.items() if k != "total"), reverse=True)[:3]
        print(f"       느린 단계: " + ", ".join(f"{k} {v:.0f}ms" for v, k in top))
a = pathlib.Path("results/multi2/attribution.json")
if a.exists():
    d = json.loads(a.read_text(encoding="utf-8"))
    print(f"\n  멀티기업 귀속: 커버리지 {d['coverage']:.1%}  "
          f"완전정상 {d['clean']}/{d['n']}  남의회사 인용 {d['foreign_rows']}건")
PY
