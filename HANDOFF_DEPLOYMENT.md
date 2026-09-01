# 배포·정합성 작업 인수인계 (Claude Code 세션, 2026-08-31~09-01)

이 문서는 `HANDOFF.md`(KIM의 정답률 개선 작업)와 별개로, **서버 배포 + 배포 후 실측에서 나온 버그 수정** 트랙을 정리한다.

---

## 1. 서버 현황

- 네이버클라우드, `s2-g3a`(vCPU 2, Memory 32GB — 처음 8GB→16GB→32GB로 증설함), Ubuntu 24.04
- 공인 IP: `223.130.142.206`, ACG: 80/443/22/3389 전체 개방(0.0.0.0/0)
- 서비스 경로: `/data/Disclosure-agent` (git repo), 20GB 블록 스토리지를 `/data`에 마운트
- **systemd**: `disclosure-api.service` (enabled, active) — SSH 끊기거나 재부팅해도 자동 실행
  ```bash
  systemctl restart disclosure-api
  journalctl -u disclosure-api -f
  curl http://127.0.0.1/health   # 서버 안에서
  ```
- `.env`: `HCX_API_KEY`, `HCX_MODEL=HCX-007`, `ARTIFACTS=artifacts_v2`
- `artifacts_v2/`: `~/Desktop/공시 agent/pipeline_kim_v2/artifacts_v2`(l1/facts/emb) + `~/Desktop/마지막/Embedding/artifacts/index`(bm25) 조합으로 로컬에서 조립 후 scp. **로컬 `정호/artifacts_v2/`는 전부 심볼릭 링크**(원본 폴더 지우면 로컬 쪽도 깨짐, 서버는 실제 파일).

## 2. 이 세션에서 찾고 고친 버그 (시간순)

### 2-1. v2 파이프라인이 조용히 꺼져 있었음
- 원인: `corpus/manifest.jsonl`이 `.gitignore`에 있어 `git clone`만으로는 안 옴 → `agent/api.py:_prepare_v2()`가 `load_manifest()`에서 실패 → 조용히 구경로(단순 검색+생성, 게이트 없음)로 폴백
- 확인법: `think_trace`에 `plan:`/`scope=`/`hcx=N회`가 있으면 v2, `retrieve: ... (경로 [...])`만 있으면 구경로
- 조치: `corpus/manifest.jsonl`(2.4MB) 서버에 scp

### 2-2. DENSE_FP16 미적용 (OOM 관련)
- 원인: `.env` 로딩(`load_dotenv`)이 `hcx_client.py`에서만 일어나는데, 이건 dense 행렬을 다 읽은 **뒤**에 실행됨(`api.py`의 `_warm()` 흐름상) → `DENSE_FP16=1`이 dense 행렬 로딩 시점엔 반영 안 됨 → float32(2.39GB)로 로드(의도는 float16 1.2GB)
- systemd의 `EnvironmentFile`로 프로세스 시작 전에 env를 주입하면서 자연스럽게 해결됨(확인: `[DENSE] numpy 행렬 ... float16 (1.19GB)` 로그)

### 2-3. 회사명이 재무항목으로 오인식되는 버그 (제일 큰 것)
- 증상: "SK텔레콤의 리스부채는?" 질문에서 "SK텔레콤"(또는 그 접두어 "SK")이 `expected_fields`에 metric처럼 섞여 들어감
  - facts 조회 결과가 회사 주소·계열사 목록 등 잡음에 파묻힘
  - **더 심각한 부작용**: 그 가짜 필드가 "확인 안 됨"으로 남아 충분성 검사가 최대 재검색(2회, 검색 1회 90~100초)을 다 돌고 → 300초 근처 타임아웃 → 결국 거부
- 전수조사: 정기공시 facts 40,479개 항목 중 90개사·323종·3,557건이 회사명/문서유형 라벨 오염
- **1차 조치**(`scripts/build_field_schema.py`): registry 회사명 대조로 스키마 생성 단계에서 제외 — 데이터 레벨 패치, 완전하지 않음(그룹 약칭 "SK","한화" 등은 못 잡음)
- **2차 조치, Phase A**(`query_plan.py`, `evidence_processor.py`, `sufficiency.py`, `ask_v2.py`, 커밋 `43175ff`):
  - A-1: `RulePlanBuilder._expected_fields()`가 이미 resolve된 `plan.companies`를 받아 metric 후보에서 회사명+접두어를 제외(런타임 필터, blacklist 아님)
  - A-2: **더 근본적인 원인 발견** — `evidence_processor.process_evidence()`가 만드는 `by_field`(충분성 검사가 보는 "확보된 항목")는 **비정형 검색 결과만** 보고 facts SQL 조회 결과는 아예 안 봄. `check_sufficiency()`에 `facts_found_fields` 추가해 연결
  - 효과(37문항 재검증, 부분): 평균 응답시간 220초→143초, 타임아웃 비율 크게 감소. 한화오션 케이스는 완전 정답으로 전환

### 2-4. 서버가 `git pull`을 한 번도 제대로 받은 적이 없었음
- 증상: Phase A 배포했다고 재기동했는데 결과가 그대로였음
- 원인: `git status`에서 `config/field_schema.json`이 로컬 수정 상태(scp로 직접 덮어씀)라 pull이 계속 막혀 있었음. `git log`가 세션 최초 커밋(`3b70f3f`)에 멈춰있었던 걸 뒤늦게 발견 — **KIM의 19개 커밋도 이 시점까지 서버엔 전혀 없었음**
- 조치: `git checkout -- config/field_schema.json` 후 `git pull` → `43175ff`까지 정상 반영 확인

## 3. 검증된 것 / 아직 남은 것

### 검증됨
- SK텔레콤/한화오션 재검색 루프·타임아웃 해소 확인 (라이브 서버 직접 테스트)
- 한화오션 지급보증KRW: 완전 정답
- 494문항 실측 기준: **회사명 추출 실패 0건, 질문과 다른 회사 추출 0건** → Phase B(corp_code/ticker 기반 재설계) 불필요 결론

### 아직 안 고쳐진 것
1. **SK텔레콤 리스부채**: [FACT] 블록에 정답(6,503)이 있어도 답변이 재무상태표 본문의 다른 "리스부채"(383,890)를 고름. **정답지(gold) 자체가 부속 주석의 값을 잘못 집었을 가능성 있음** — 사람이 원문 대조 필요(`eval/suite_v2.jsonl` G0148)
2. **사업보고서(연간) period 매칭 실패** (신규 발견, 미수정): `facts/extractor.py`가 `period` 컬럼을 항상 `None`으로 둠 → 조회 시 `filing_date` 폴백을 쓰는데, 사업보고서는 회계연도 종료(예: 2023.12) 후 2~3개월 뒤(익년 3월)에 제출돼서 필터가 안 맞음. 분기/반기보고서는 제출이 빨라서(45일 이내) 우연히 맞았음. 영향받은 실측 사례: 삼성E&A 수권주식수(타임아웃), HMM 부채총계, 크래프톤 영업비용(다른 분기 값 섞여 나옴)
3. **compare 유형 일부**: "매출액" 같은 key_norm이 문서유형(exchange 계약공시 등)에 따라 다른 의미로 쓰이며 자릿수가 완전히 다른 값이 나오는 사례(셀트리온 vs HMM, JYP Ent vs 파마리서치) — 원인 미조사
4. **카카오페이증권 등 wrong_entity 1건**: 다른 회사 표 안의 자회사/거래상대방 이름을 독립 회사로 오인 — `EntityRegistry.types`(universe만 정식 인정)로 범위 게이트를 좁히면 될 걸로 보이나 미수정

### 검토 후 기각
- **Phase B~E(corp_code/metric_id 기반 재설계)**: 494문항 실측으로 근거 없음 확인. 남은 실패는 전부 위 2·3·4번(좁은 버그)로 설명됨. corp_code 인프라(`EntityRegistry.corp_code`, `manifest.corp_code`)는 이미 존재하지만 연결할 필요성이 낮음.

## 4. 다음에 할 일 (우선순위 제안)

1. period 매칭 버그(3-2) 수정 — `facts/extractor.py`가 `base_year`/`base_month`로 `period`를 채우도록. lookup_periodic 실패의 상당 부분을 설명할 가능성 높음
2. compare 유형 key_norm 다의성(3-3) 원인 조사
3. G0148 등 의심되는 gold answer 사람이 원문 대조
4. **기술제안서 착수** (아직 미착수, README 자체가 "마지막 날에 써도 된다"지만 미루면 위험)
5. 전체 재측정 1회(100~494문항)로 최종 수치 확정

## 5. 자주 쓰는 명령

```bash
# 서버 상태
curl http://223.130.142.206/health

# 질문 테스트
curl -G "http://223.130.142.206/answer" \
  --data-urlencode "question_id=TEST-001" \
  --data-urlencode "question=..." --max-time 290

# 배포(코드만 바뀐 경우)
ssh 서버: cd /data/Disclosure-agent && git pull && pip install -e . && systemctl restart disclosure-api
```

마감: **2026-09-06 23:59 KST**
