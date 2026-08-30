# AskV2 거부 능력 측정

- 거부셋: 160문항
- abstention_accuracy: 44.37%
- false_refusal_rate: 2.63% (1/38)
- 거부셋 HCX 호출: 189
- 실제 게이트 거부 중 HCX 호출: 30

## 원인별

| 원인 | 문항 | 거부 정확도 | 실패 |
|---|---:|---:|---:|
| ambiguous | 40 | 5.00% | 38 |
| hard_out | 40 | 100.00% | 0 |
| no_answer_in_doc | 40 | 60.00% | 16 |
| wrong_entity | 40 | 12.50% | 35 |

## 종료 단계

- abstention_gate: 30
- answered: 89
- scope_gate: 41
