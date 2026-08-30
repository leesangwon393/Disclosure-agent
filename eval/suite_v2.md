# 평가셋 v2 — 296문항

- 생성 2026-08-30T11:15:05+00:00 / seed 20260830
- 회사 63곳 (유니버스 70곳 중)

## 유형별

| 유형 | 모드 | 문항 |
|---|---|---:|
| 검색·정보추출 | closed | 99 |
| 검색·정보추출 | open | 58 |
| 다중조회·비교·연산 | closed | 60 |
| 다중조회·비교·연산 | open | 26 |
| 복합문서추론 | closed | 30 |
| 복합문서추론 | open | 23 |

## 정답 출처

정답을 **어떻게 확정했는지**. `rubric_only` 는 정답 문장이 없는 대신
채점 항목이 붙어 있다(open_scoring 이 커버리지로 채점).

| 출처 | 문항 | 뜻 |
|---|---:|---|
| rubric_only | 107 | 서술형 — 정답 문장 없이 채점 항목만 |
| auto_facts | 99 | facts 층에서 값이 유일하게 확정 |
| auto_compare | 60 | 두 회사 최댓값을 계산해 비교 결론까지 확정 |
| auto_meta | 30 | manifest 에서 존재 여부가 확정 (예/아니오) |

## 생성기별

| 생성기 | 문항 |
|---|---:|
| compare | 60 |
| lookup_form | 59 |
| summary | 58 |
| lookup_periodic | 40 |
| termination | 30 |
| funding | 26 |
| correction | 23 |

## 버린 후보와 이유

품질 관문을 생성기 안에 뒀다. 사후 정리는 이미 오염된 걸 줄일 뿐이라,
여기서는 애초에 만들지 않는다.

| 이유 | 건수 |
|---|---:|
| summary:회사쏠림 | 302 |
| summary:채점기준부족 | 248 |
| correction:변경항목부족 | 114 |
| summary:항목쏠림 | 90 |
| correction:동일유형체인다수 | 57 |
| lookup_periodic:한글2자미만 | 54 |
| lookup_form:항목쏠림 | 47 |
| lookup_form:회사쏠림 | 32 |
| lookup_periodic:값중복 | 27 |
| lookup_form:구조라벨 | 6 |
| lookup_form:값중복 | 4 |
| lookup_periodic:값이항목명 | 4 |
| funding:회사쏠림 | 4 |
| lookup_periodic:구조라벨 | 2 |
| summary:질문중복 | 2 |
| lookup_form:질문중복 | 1 |
| correction:버전1개 | 1 |
| lookup_periodic:회사쏠림 | 1 |

## 회사 분포 (상위 15)

| 회사 | 문항 |
|---|---:|
| 한화오션 | 13 |
| 셀트리온 | 12 |
| 삼성E&A | 10 |
| 한화에어로스페이스 | 10 |
| 한미반도체 | 10 |
| 삼성중공업 | 9 |
| HD현대중공업 | 9 |
| OCI홀딩스 | 9 |
| 삼성바이오로직스 | 9 |
| 신한지주 | 9 |
| 한전기술 | 8 |
| 현대건설 | 8 |
| 한국항공우주 | 8 |
| 대우건설 | 8 |
| 레인보우로보틱스 | 8 |

## 쓰는 법

```bash
# 검색 상한 (HCX 없이, 빠름)
./run.sh python3 scripts/score_answers.py --gold eval/suite_v2.jsonl \
    --mode retrieval --k 10 --out results/v2suite_retrieval

# 최종 답변 (HCX 사용)
./run.sh python3 scripts/score_answers.py --gold eval/suite_v2.jsonl \
    --mode full --pipeline v2 --yes --out results/v2suite_full
```

