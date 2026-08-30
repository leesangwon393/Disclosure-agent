# AskV2 거부 능력 측정 인수인계

## 현재 상태

- 거부셋: `eval/gold_abstention.jsonl` 160문항, 원인별 40문항
- 거부 체크포인트: `results/abstention_v2/abstention_checkpoint.jsonl` 76/160
- 정상 체크포인트: 아직 없음. 거부셋 완료 후 `eval/suite_v1.jsonl` 38문항을 실행한다.
- 관련 테스트: 56개 통과
- 마지막 완료 문항: `ABS-H040`; 다음은 `ABS-A013`

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

## 이미 확인된 부분 결과

76건 기준 종료 단계는 `scope_gate 40 / abstention_gate 3 / answered 33`, 실제 HCX 호출은 79회였다.

- hard_out 40건은 모두 거부했지만 계획 보완이 scope gate보다 먼저 실행되어 일부 HCX 호출이 발생했다.
- 기존 ambiguous 12건은 전부 answered로 통과했다.
- 기존 wrong_entity 12건도 전부 answered로 통과했다.
- 기존 no_answer_in_doc 12건 중 3건만 abstention_gate에서 막혔다.

최초 중단 실행에서 로그로 복구한 28건은 `error: recovered_from_log`로 표시된다. 이 행들은 답변 본문과 계획 세부값은 없지만 action, stopped_at, hcx_calls, elapsed_sec가 보존돼 핵심 지표 계산에 사용할 수 있다.

## 남은 작업

1. 거부셋 160건 완료
2. 정상셋 38건 완료 및 false_refusal_rate 계산
3. `results/abstention_v2/metrics.json`, `summary.md`, `failure_cases.jsonl` 확인
4. 최종 보고: 문항 수/분포 → abstention_accuracy → false_refusal_rate → HCX 호출 → 원인별 실패

