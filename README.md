# Disclosure-agent

공시자료를 근거로 사용자 질의에 답변하는 공시 에이전트.
제10회 2026 미래에셋증권 AI Festival 「공시 Agent」 과제 구현체.

DART 공시 4,204건(70개사)을 검색해 HyperCLOVA X 가 근거 기반으로 답한다.
**근거가 없으면 지어내지 않고 "확인되지 않습니다"라고 답하는 것**을 정답 경로로 둔다.

---

## 빠른 시작

```bash
# 1. 의존성
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. API 키
cp .env.example .env      # HCX_API_KEY 를 채운다

# 3. 산출물 배치 (아래 '무엇이 더 필요한가' 참고)
#    artifacts_v2/ 를 프로젝트 루트에 둔다

# 4. 서버 기동 — 약 3분 (62만 청크를 메모리에 올린다)
python3 -m uvicorn disclosure_rag.serving.api:app --host 0.0.0.0 --port 8000

# 5. 질문
curl -G http://127.0.0.1:8000/answer \
  --data-urlencode "question_id=T001" \
  --data-urlencode "question=삼성전자의 주요사항보고서(자기주식취득결정)에 기재된 순자산액은 얼마인가?"
```

`GET /health` 로 준비 상태를 볼 수 있다. 기동이 끝나기 전 요청에는 `503` 을 준다
(주최측 재시도 유도).

### 무엇이 더 필요한가 — 코드만으로는 안 돌아간다

| | 크기 | 어디서 | 왜 저장소에 없나 |
|---|---|---|---|
| `artifacts_v2/` | 3.7GB | 별도 전달 또는 `rebuild_v2.sh` | GitHub 파일당 100MB 한도. 코드로 재생성 가능 |
| `corpus/` | 5.2GB | 대회 제공 데이터 | 재배포 불가 |
| `.env` | — | 각자 발급 | 비밀정보 |

`corpus/raw` 는 대회 데이터 폴더를 가리키는 심볼릭 링크다.
`corpus/manifest.jsonl`, `corpus/universe.csv` 도 함께 있어야 한다.

---

## 파이프라인

**설계 원칙: 무엇을 검색할지는 규칙이 정하고, HCX 는 답변 문장만 만든다.**

구 경로(`agent/ask.py`)는 HCX 에게 도구 목록을 주고 무엇을 부를지 맡겼다.
실측상 100문항 중 54건이 도구를 1회만 부르고 끝냈고, 정정 이력 도구는 시켜도
안 불렀다. 그래서 신 경로에서는 검색 횟수와 대상을 규칙이 결정한다.

**현행 진입점은 `agent/ask_v2.py` 다.** `agent/ask.py` 는 비교 기준으로 남겨둔 구 경로다.

| 단계 | 하는 일 | 누가 |
|---:|---|---|
| 1 | 질의 이해 — 정규화, 회사·기간·지표 추출 | 규칙 |
| 2 | **QueryPlan** — 답의 모양·작업 유형·검색 대상·필수 항목 결정 | 규칙 → (미결시) HCX |
| 3 | **범위 게이트** — 코퍼스 밖 질문 즉시 거부 | 규칙 |
| 3.5 | **존재 전수 확인** — 목록 전체를 세어 있/없음 판정 | 규칙 |
| 4 | 질문 분해 — 회사 > 공시유형 > 기간 축으로 최대 3개 | 규칙 |
| 5~6 | **이중 채널 검색** — 정형(수치사전) ∥ 비정형(BM25+Dense+Sparse) | 규칙 |
| 7 | 버전 정리 — 정정본 중 최신 유효본만 남김 | 규칙 |
| 8 | 부모 확장 — leaf 청크 → 상위 문단 | 규칙 |
| 9 | 근거 구조화 — 항목별 값 추출, 정정 전후 짝 맞추기 | 규칙 |
| 10 | 충분성 검사 — 부족하면 4단계로 복귀(최대 2회) | 규칙 |
| 11 | **거부 게이트** — 여기서 막히면 HCX 를 아예 안 부른다 | 규칙 |
| 13 | Evidence Pack 조립 | 규칙 |
| 14 | **답변 생성** — answer_mode 별 프롬프트 | **HCX** |
| 15 | Validator — 근거에 없는 숫자 검출 | 규칙 |
| 16 | 부분답변이면 한계 명시 | 규칙 |

HCX 호출 지점은 셋뿐이고 그중 둘은 거의 안 탄다(14번만 필수).
**범위 게이트와 거부 게이트를 통과하지 못하면 HCX 를 부르지 않는다 — 지어낼 기회 자체가 없다.**

### 검색 구성

```
질의 ─┬─ 정형: Facts(sqlite)   공시 표에서 미리 뽑아둔 확정 값
      └─ 비정형: BM25(Kiwi) + Dense(BGE-M3, 62만 벡터) + Learned Sparse
                     └─ RRF 융합 → Cross-Encoder 리랭커(bge-reranker-v2-m3)
```

리랭커 실측(314문항): `hybrid_rrf` hit@5 0.583 → `+rerank` 0.618. 문항당 +4.8초.

---

## 왜 이렇게 만들었나 — 실측에서 나온 결정들

**정형/비정형 이중 채널.** "순자산액은 얼마인가" 같은 질문은 본문 검색보다
표에서 뽑아둔 값을 그대로 주는 게 정확하다. 그래서 `facts/`(주요사항보고서류)와
`facts_periodic_v2/`(정기공시)를 따로 만들어 함께 조회한다.

**최댓값은 Python 이 계산한다.** 값 10개를 주자 모델이 최댓값이 아닌 5번째 값을
골랐다(S011). 집계 질문은 `[FACT]` 블록에 `▶` 로 답을 미리 계산해 표시하고,
"목록에서 직접 고르지 말라"고 지시한다.

**정정공시는 최신 유효본만.** 정정본의 43%가 원본과 텍스트가 거의 같아서
정리하지 않으면 같은 내용이 top-k 를 채운다.

**thinking 은 기본 OFF.** HCX-007 은 reasoning 모델이고 기본이 ON 인데, 같은
프롬프트·같은 근거(13,542자)로 실측하니 ON 이면 "확인할 수 없습니다", OFF 면
정답이었다. `--thinking auto|on` 으로 A/B 할 수 있게 열어뒀다.

**'없음'과 '모름'을 구분한다.** 유사도 검색은 상위 k건만 본다. 거기 없다고
부재를 단정할 수 없어서 모델이 "확인할 수 없습니다"로 물러섰다. 그래서
존재 질문은 `manifest`(전체 목록)를 전수로 훑어 판정을 만들어 준다.
확인된 부재를 '모름'으로 답하는 것도 감점이기 때문이다.

---

## 평가

정답셋 3종이 `eval/` 에 있다.

| 파일 | 문항 | 무엇을 재나 |
|---|---:|---|
| `suite_v1.jsonl` | 38 | 대회 예시 질문 형태. **주력 벤치마크** |
| `suite_v2.jsonl` | 296 | 자동 생성 + 교차검증. 통계적으로 안정된 숫자용 |
| `gold_abstention.jsonl` | 160 | 범위 밖 질문에 거부/역질문/유보를 제대로 하는가 |

정답 형식이 네 가지이고 채점기가 각각 다르게 처리한다.

```
값     "224,787,773,988,054"            콤마 표기 차이 무시
비교   "삼성전자 (1,000 vs 500)"         양쪽 수치 + 결론 일치
예아니오 "아니오 (해지 공시 없음)"        가장 먼저 등장한 판정이 이긴다
서술형 required_all: ["9.90", ...]      필수 항목이 전부 나와야 정답
```

서술형은 표기 차이를 흡수한다 — 정답지 `9.90` 과 답변 `9.9%`, 정답지
`2024-04-24` 와 답변 `2024년 4월 24일` 은 같은 값으로 본다.

```bash
# 검색만 (HCX 안 씀, 크레딧 0원)
python3 scripts/score_answers.py --gold eval/suite_v1.jsonl --mode retrieval

# 전체 파이프라인 (HCX 사용)
python3 scripts/score_answers.py --gold eval/suite_v1.jsonl \
    --mode full --pipeline v2 --thinking off --yes --out results/v2_off5

# 이미 받아둔 답변을 다시 채점 (HCX 0회, 수초)
python3 scripts/score_answers.py --gold eval/suite_v1.jsonl \
    --rescore results/v2_off5 --out results/v2_off5_regrade

# 유보/거부 판정
python3 scripts/score_abstention.py --gold eval/gold_abstention.jsonl
```

결과는 `results/<이름>/` 에 `metrics.json` · `results.csv` · `answers.jsonl`
(답변 원문) · `failure_cases.jsonl` · `summary.md` 로 떨어진다.

**채점 기준이 없는 문항은 분모에서 뺀다.** 정답이 비어 있는 문항을 분모에 넣으면
자동 0점이 되어 실제보다 낮게 나온다 — `graded_hit_n` 과 `ungradeable_n` 을 함께 본다.

### 현재 성능 (suite_v1 38문항, 2026-08-31 측정)

`results/v2_off6_regrade/` — HCX-007, thinking off, 리랭커 ON.

| | |
|---|---:|
| **정답률** | **71.1%** (27/38) |
| 근거 회수 | 86.8% |
| 지어낸 숫자 없음 | 86.8% |
| 근거 표시 | 86.8% |
| 무리한 답 대신 유보 | 2.6% |
| 응답 평균 / 최대 | 18.1초 / 40.1초 |

유형별로 갈린다. 단일 값 조회는 사실상 잡혔고 서술형이 남았다.

```
lookup           13/14  93%      closed  21/22  95%
compare           8/11  73%      open     5/12  42%
summarize         5/ 9  56%      mixed    1/ 4  25%
correction_diff   1/ 4  25%
```

하루 동안의 추이 — 무엇을 고쳐서 얼마가 움직였는지.

| | 정답률 | 검색실패 | 무엇을 고쳤나 |
|---|---:|---:|---|
| v2_off4 | 52.6% | 6 | (기준) |
| v2_off5 | 65.8% | 6 | 채점기 분모 버그 + 서술형 정답 16개 + '없음/모름' 구분 |
| v2_off6 | 71.1% | 2 | 기간 필터(제출연도) + 개념어→공시유형 전개 |

**실패의 성격이 바뀌었다.** 검색실패 6건→2건, 답변실패 7건→9건.
근거를 못 가져와서 지던 문제가 근거는 왔는데 답을 못 쓰는 문제로 옮겨갔다.

---

## 산출물 재생성

```bash
./rebuild_v2.sh          # 1~5단계, 약 9시간 (임베딩이 대부분)
START_AT=4 ./rebuild_v2.sh   # 끊긴 지점부터 이어서
```

7단계를 순서대로 만든다 — 스냅샷 → 수치사전 → 정기공시 수치사전 →
임베딩 → 색인 → 회사명 레지스트리 → 검색 평가.

정기공시 수치사전과 레지스트리는 **빠져도 에러가 안 난다.** 대신 사업보고서류
수치 조회가 통째로 죽고 범위 게이트가 조용히 꺼진다. 그래서 안내가 아니라
스크립트 안에 넣어뒀다.

색인(`artifacts_v2/index/`)은 임베딩과 청크만 있으면 재생성된다 — 전달받을 필요 없다.

```bash
python3 scripts/build_index.py --artifacts artifacts_v2
```

---

## 구조

```
src/disclosure_rag/
  agent/        ask_v2(현행 진입점) · query_plan · dual_channel · decompose
                evidence_processor · sufficiency · abstention · existence
                answer_generator · validator · version_dedup · scope_gate
  retrieval/    bm25 · dense(numpy) · sparse · 융합 · 리랭커 · index_bundle
  facts/        수치사전 조회 (store · multi_store)
  correction/   정정 체인 구성
  chunking/     의미 블록 청킹
  parsing/      공시 XML 파싱
  entity/       회사·지표 추출, 레지스트리
  serving/      api.py (평가용 HTTP 서버)
scripts/        빌드 · 채점 · 정답셋 생성 · 진단
eval/           정답셋
config/         field_schema.json · metric_terms.txt
tests/          608개
```

---

## 환경변수

`.env.example` 에 전부 설명해뒀다. 자주 쓰는 것만:

| | 기본 | |
|---|---|---|
| `HCX_API_KEY` | — | 필수 |
| `ARTIFACTS` | `artifacts_v2` | 산출물 경로 |
| `DENSE_FP16` | 꺼짐 | 켜면 임베딩 메모리 절반(정확도 손실 없음). 8GB 서버에서 켠다 |
| `THINKING` | `off` | HCX reasoning 정책 |
| `USE_RERANKER` | `1` | 0 이면 끈다 |

메모리는 기동 후 약 3.2GB, 기동 순간 피크 5~6GB. `DENSE_FP16=1` 이면 피크가
2.5~3GB 로 떨어진다.

---

## 대회 제약 (위반 시 평가 제외)

- **LLM 은 HyperCLOVA 계열만.** 최종 답변 생성과 Agent 판단에 다른 LLM 을 쓰면 실격
- 제약 **밖**(자유): 임베딩, 리랭커, 파싱·구조화, BM25, 형태소 분석기, 규칙 기반 NLP
- 제공 코퍼스 외 데이터 금지. 외부 공시 API 실시간 호출 금지
- 근거가 없으면 "제공된 공시에서 확인되지 않습니다" 가 정답 경로
- 응답 제한 300초. 초과 전에 스스로 멈추고 그때까지의 근거로 답한다(부분 점수)

---

## 알려진 한계

남은 실패 11건의 원인은 넷이다.

| 원인 | 건수 | |
|---|---:|---|
| 긴 문서 안에서 청크 선택 | 3 | 사업보고서는 문서당 청크가 900~1,300개다. 문서는 찾는데 요약재무정보 표를 못 집는다 |
| 최종 정정본 미선택 | 3 | 정정 체인에서 중간 버전 값을 답한다 |
| 값 일부 누락 | 2 | 자금조달 문항. 문서는 맞고 금액 일부를 빠뜨린다 |
| 짧은 공시 순위 밀림 | 2 | 청크 3~4개짜리 공시가 62만 개 속에서 상위에 못 온다 |

그 외:

- `suite_v2` 296문항 중 20문항은 아직 채점 기준을 만들지 못했다(수치사전에 쓸 값이 없다). 억지로 만들지 않고 `채점불가`로 둔다
- `temperature=0.2` 라 같은 질문의 답이 실행마다 흔들린다. 38문항은 한 문항이 2.6%라 소수점 차이로 개선을 판단하면 안 된다 — `suite_v2` 296문항 측정이 필요하다
- 배포 리허설 미완 — 로컬 기동은 확인했으나 클라우드 서버 기동은 아직
