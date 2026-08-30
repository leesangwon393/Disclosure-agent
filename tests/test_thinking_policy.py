"""HCX-007 reasoning(thinking) A/B 장치.

기본값은 **꺼짐**이다. 켜는 쪽이 나을 거라는 건 아직 가정이고, 이 모델에서
reasoning 이 오히려 나빴던 실측이 있다(diag_answer.py: 근거 13,542자에서
thinking ON 이면 "확인할 수 없습니다", OFF 면 정답).

그래서 여기서 지키는 건 두 가지다.
  1. 아무 것도 안 하면 지금까지와 똑같이 꺼져 있는가
  2. 켜기로 했을 때, 잘못된 값이어도 38문항이 통째로 죽지 않는가
"""
from __future__ import annotations

from disclosure_rag.agent.answer_generator import (
    DEFAULT_THINKING_EFFORT,
    THINKING_OFF,
    THINKING_TASKS,
    resolve_thinking,
)
from disclosure_rag.agent.query_plan import QueryPlan


# --------------------------------------------------------------------------- 정책

def test_default_policy_keeps_thinking_off():
    """기본값이 바뀌면 지금까지 측정한 모든 수치의 기준이 흔들린다."""
    assert resolve_thinking(QueryPlan(task="lookup")) == THINKING_OFF
    assert resolve_thinking(None) == THINKING_OFF
    assert resolve_thinking(QueryPlan(task="correction_diff")) == THINKING_OFF


def test_auto_turns_it_on_only_for_multi_step_tasks():
    on = {"effort": DEFAULT_THINKING_EFFORT}
    assert resolve_thinking(QueryPlan(task="correction_diff"), policy="auto") == on
    assert resolve_thinking(QueryPlan(task="compare"), policy="auto") == on
    # 값 하나 뽑기 — 실측으로 OFF 가 나았다
    assert resolve_thinking(QueryPlan(task="lookup"), policy="auto") == THINKING_OFF
    # 계산은 파이썬이 한다. 모델은 옮겨쓰기만 하므로 추론이 필요 없다
    assert resolve_thinking(QueryPlan(task="calculate"), policy="auto") == THINKING_OFF


def test_on_ignores_the_task():
    for task in ("lookup", "compare", "summarize"):
        assert resolve_thinking(QueryPlan(task=task), policy="on") != THINKING_OFF


def test_auto_with_no_plan_stays_off():
    assert resolve_thinking(None, policy="auto") == THINKING_OFF


def test_effort_value_is_overridable():
    """CLOVA 문서에 허용값이 없어 'low' 는 추측이다 — 바꿀 수 있어야 한다."""
    assert resolve_thinking(QueryPlan(task="compare"), policy="on",
                            effort="medium") == {"effort": "medium"}


def test_candidate_tasks_are_the_measured_ones():
    """suite_v1 기준 correction_diff 4 + compare 11 = 15문항(39%)."""
    assert THINKING_TASKS == {"correction_diff", "compare"}


# --------------------------------------------------------------------------- 400 폴백

class _FakeResponse:
    def __init__(self, status_code, text="", body=None):
        self.status_code, self.text, self._body = status_code, text, body or {}

    def json(self):
        return self._body


def test_rejected_thinking_value_falls_back_instead_of_killing_the_run(monkeypatch):
    """허용값을 모르는 채로 켰다가 38문항이 중간에 전부 죽으면 안 된다.

    400 이 나면 thinking 파라미터를 **빼고** 다시 보낸다 — HCX-007 은 기본이
    ON 이므로 결과적으로 reasoning 은 켜진 채 진행된다.
    """
    import disclosure_rag.agent.hcx_client as hc

    monkeypatch.setattr(hc, "_wait_for_slot", lambda *a, **k: None)
    client = hc.HCXClient(api_key="k", model="HCX-007")

    sent: list[dict] = []
    ok = {"status": {"code": "20000"}, "result": {"message": {"content": "답"}}}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.append(dict(json))
        if "thinking" in json:
            return _FakeResponse(400, text='{"message":"Invalid parameter: thinking"}')
        return _FakeResponse(200, body=ok)

    monkeypatch.setattr(hc.requests, "post", fake_post)
    out = client.chat([{"role": "user", "content": "q"}], thinking={"effort": "존재하지않는값"})

    assert out["content"] == "답"
    assert len(sent) == 2
    assert "thinking" in sent[0] and "thinking" not in sent[1]


def test_other_400_errors_do_not_trigger_the_fallback(monkeypatch):
    import disclosure_rag.agent.hcx_client as hc

    monkeypatch.setattr(hc, "_wait_for_slot", lambda *a, **k: None)
    monkeypatch.setattr(hc.time, "sleep", lambda *a: None)
    client = hc.HCXClient(api_key="k", model="HCX-007")

    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        return _FakeResponse(400, text='{"message":"Invalid parameter: messages"}')

    monkeypatch.setattr(hc.requests, "post", fake_post)
    try:
        client.chat([{"role": "user", "content": "q"}], max_retries=1)
    except hc.HCXError:
        pass
    assert calls["n"] == 2          # 정상 재시도만, 폴백 추가 호출 없음
