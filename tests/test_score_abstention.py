import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from score_abstention import (  # noqa: E402
    action_from_result,
    aggregate_pipeline,
    prediction_from_result,
    recover_checkpoint_from_log,
    run_rows,
    score,
)


def test_abstention_and_exact_action_are_separate_metrics():
    gold = [
        {"id": "1", "reason": "hard_out", "expected_action": "refuse"},
        {"id": "2", "reason": "ambiguous", "expected_action": "clarify"},
    ]
    predictions = [
        {"id": "1", "action": "abstain"},
        {"id": "2", "action": "clarify"},
    ]
    metrics, details = score(gold, predictions)
    assert metrics["abstention_accuracy"] == 1.0
    assert metrics["action_accuracy"] == 0.5
    assert len(details) == 2


def test_missing_prediction_is_counted_as_failure():
    metrics, _ = score(
        [{"id": "1", "reason": "wrong_entity", "expected_action": "abstain"}],
        [],
    )
    assert metrics["n_predicted"] == 0
    assert metrics["abstention_accuracy"] == 0.0


def _result(stopped_at, *, hcx_calls=0, abstention_action="answer"):
    return SimpleNamespace(
        stopped_at=stopped_at,
        hcx_calls=hcx_calls,
        answer="테스트",
        plan=SimpleNamespace(answer_mode="closed", task="lookup", companies=["삼성전자"]),
        scope=SimpleNamespace(scope="in_scope", reason="test"),
        abstention=SimpleNamespace(action=abstention_action, reason="test"),
        decomposed=SimpleNamespace(merged=[]),
        evidence=[],
    )


def test_ask_v2_result_action_uses_actual_stop_stage():
    assert action_from_result(_result("scope_gate")) == "refuse"
    assert action_from_result(_result("abstention_gate")) == "abstain"
    assert action_from_result(_result("answered", abstention_action="partial")) == "partial"
    assert action_from_result(_result("answered")) == "answer"


def test_pipeline_metrics_report_false_refusal_stops_and_hcx_together():
    gold = [
        {"id": "A", "query": "q1", "reason": "hard_out", "expected_action": "refuse"},
        {"id": "B", "query": "q2", "reason": "wrong_entity", "expected_action": "abstain"},
    ]
    predictions = [
        prediction_from_result(gold[0], _result("scope_gate"), elapsed_sec=0.1),
        prediction_from_result(gold[1], _result("answered", hcx_calls=1), elapsed_sec=0.2),
    ]
    normal = [
        {"id": "S1", "action": "answer", "stopped_at": "answered", "hcx_calls": 1},
        {"id": "S2", "action": "abstain", "stopped_at": "abstention_gate", "hcx_calls": 0},
    ]
    metrics, details = aggregate_pipeline(gold, predictions, normal)
    assert metrics["abstention_accuracy"] == 0.5
    assert metrics["false_refusal_rate"] == 0.5
    assert metrics["stopped_at"] == {"answered": 1, "scope_gate": 1}
    assert metrics["hcx_calls_on_abstention_set"] == 1
    assert metrics["hcx_calls_on_pipeline_refusals"] == 0
    assert metrics["false_refusal_ids"] == ["S2"]
    assert len(details) == 2


def test_run_rows_counts_real_client_calls_even_if_result_counter_is_zero():
    class Client:
        calls = 0

    class Ask:
        client = Client()

        def run(self, _query):
            self.client.calls += 1
            return _result("abstention_gate", hcx_calls=0)

    rows = run_rows(Ask(), [{"id": "A", "query": "q"}], label="TEST")
    assert rows[0]["hcx_calls"] == 1


def test_run_rows_checkpoints_each_item_and_resumes_without_rerunning(tmp_path):
    class Client:
        calls = 0

    class Ask:
        client = Client()

        def run(self, _query):
            self.client.calls += 1
            return _result("scope_gate")

    ask = Ask()
    gold = [{"id": "A", "query": "q1"}, {"id": "B", "query": "q2"}]
    checkpoint = tmp_path / "checkpoint.jsonl"
    first = run_rows(ask, gold, label="TEST", checkpoint_path=checkpoint)
    assert ask.client.calls == 2
    assert [row["id"] for row in first] == ["A", "B"]
    saved = [__import__("json").loads(line) for line in checkpoint.read_text().splitlines()]
    assert [row["id"] for row in saved] == ["A", "B"]

    second = run_rows(ask, gold, label="TEST", checkpoint_path=checkpoint,
                      existing_rows=saved)
    assert ask.client.calls == 2
    assert second == first


def test_recover_checkpoint_from_interrupted_log(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "20:00:00 [ABSTAIN 1/2] scope_gate       HCX0  0.00s ABS-H001\n"
        "20:00:01 [ABSTAIN 2/2] answered         HCX2  1.25s ABS-A001\n"
        "20:00:02 [NORMAL 1/1] abstention_gate  HCX0  0.50s S001\n",
        encoding="utf-8",
    )
    gold = [{"id": "ABS-H001", "query": "q1"}, {"id": "ABS-A001", "query": "q2"}]
    recovered = recover_checkpoint_from_log(log, gold, label="ABSTAIN")
    assert [(row["id"], row["action"], row["hcx_calls"]) for row in recovered] == [
        ("ABS-H001", "refuse", 0), ("ABS-A001", "answer", 2),
    ]


# --------------------------------------- 본문 거부를 거부로 센다 (2026-08-31)
#
# 실측(160문항): 시스템이 본문에서 제대로 거부했는데 `stopped_at` 만 보고
# "답변함" 으로 분류된 것이 47건이었다. 인정하면 44.4% -> 73.7% 다.

from score_abstention import body_is_refusal  # noqa: E402


def test_body_refusal_is_recognised():
    for a in ("쿠팡의 매출액을 찾을 수 없어 제공할 수 없습니다.",
              "제공된 증거에는 해당 정보가 포함되어 있지 않습니다.",
              "확인할 수 없습니다.",
              "해당 자료가 없습니다."):
        assert body_is_refusal(a), a


def test_real_answer_is_not_a_refusal():
    for a in ("순자산액은 224,787,773,988,054원입니다.",
              "정정 내역이 있습니다. 계약금액이 변경되었습니다."):
        assert not body_is_refusal(a), a


def test_empty_answer_is_not_counted_as_a_refusal():
    """빈 답변은 거부가 아니라 실패다 — 섞으면 미구현이 숨는다."""
    assert not body_is_refusal("")
    assert not body_is_refusal("   ")
