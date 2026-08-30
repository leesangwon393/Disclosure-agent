# AskV2 거부 능력 측정 인수인계

## 현재 상태

- 거부셋: `eval/gold_abstention.jsonl` 160문항, 원인별 40문항
- 거부 체크포인트: `results/abstention_v2/abstention_checkpoint.jsonl` 160/160 완료
- 정상 체크포인트: `results/abstention_v2/normal_checkpoint.jsonl` 38/38 완료
- 실행 종료: 오류 0건, 정상 종료
- 관련 회귀 테스트: 107개 통과

## 변경한 파일

- `eval/gold_abstention.jsonl`
- `scripts/score_abstention.py`
- `tests/test_gold_abstention.py`
- `tests/test_score_abstention.py`

`src/disclosure_rag/`, `scripts/score_answers.py`, `eval/suite_v1.jsonl`, `eval/gold_passages*.jsonl`은 수정하지 않았다.

## 재개 명령

```bash
./run.sh .venv/bin/python scripts/score_abstention.py \
  --pipeline v2 --yes --out results/abstention_v2
```

`--fresh`는 사용하지 않는다. 문항마다 체크포인트를 임시 파일에 쓴 뒤 원자적으로 교체하므로 중단 후 같은 명령을 실행하면 완료 문항을 자동으로 건너뛴다. 오류 행은 재개 대상으로 인정하지 않아 다시 실행된다.

## 최종 결과

- abstention_accuracy: 44.37% (71/160)
- false_refusal_rate: 2.63% (1/38, `S038`)
- 원인별: hard_out 100%, ambiguous 5%, no_answer_in_doc 60%, wrong_entity 12.5%
- 종료 단계: `scope_gate 41 / abstention_gate 30 / answered 89`
- 거부셋 실제 HCX 호출: 189회
- 실제로 거부된 질문의 HCX 호출: 30회
- 전체 실행 시간: 8,533.97초

최종 산출물은 `results/abstention_v2/metrics.json`, `summary.md`, `failure_cases.jsonl`, `results.jsonl`, `normal_results.jsonl`에 있다.

최초 중단 실행에서 로그로 복구한 28건은 `error: recovered_from_log`로 표시된다. 이 행들은 답변 본문과 계획 세부값은 없지만 action, stopped_at, hcx_calls, elapsed_sec가 보존돼 핵심 지표 계산에 사용할 수 있다.

## 후속 개선 우선순위

1. `ambiguous` 질문을 답변 전에 역질문으로 보내는 판정 강화
2. Entity Registry에 없는 회사를 조기 차단하되, universe 밖 코퍼스 등장 주체는 계속 허용
3. `no_answer_in_doc`에서 검색 조각 존재 여부가 아니라 expected field 충족 여부로 판정
4. `S038`의 검색 실패 원인을 수정한 뒤 정상 38문항 오거부 회귀 재측정
5. HCX 보완 호출이 scope gate보다 먼저 일어나는 경로를 제거해 거부 질문 HCX 호출 0회 달성
