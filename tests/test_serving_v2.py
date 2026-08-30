"""평가용 API 가 신 파이프라인을 쓰는가.

여기서 틀리면 오늘 만든 12개 모듈이 서비스에서 하나도 안 걸린다.
실제로 그런 상태였다 — `api.py` 가 구 경로(`ask.py`)를 부르고 있었다.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def api(monkeypatch):
    """HCX·인덱스 없이 모듈만 올린다."""
    monkeypatch.setenv("PIPELINE", "v2")
    for mod in [m for m in sys.modules if m.startswith("disclosure_rag.serving")]:
        del sys.modules[mod]
    from disclosure_rag.serving import api as mod
    return mod


def test_v2_is_the_default_pipeline(api):
    """측정을 전부 v2 로 했으므로 서비스 기본값도 v2 여야 한다."""
    assert api.PIPELINE == "v2"


def test_answer_once_prefers_v2(api, monkeypatch):
    calls = []

    def fake_v2(qid, q, *, deadline):
        calls.append("v2")
        return {"question_id": qid, "question": q, "retrieved_context": "",
                "think_trace": "v2", "answer": "v2 답변"}

    def fake_agent(qid, q, *, deadline):
        calls.append("agent")
        return {}

    monkeypatch.setitem(api._state, "askv2", object())
    monkeypatch.setitem(api._state, "tools", [object()])
    monkeypatch.setattr(api, "_answer_with_v2", fake_v2)
    monkeypatch.setattr(api, "_answer_with_agent", fake_agent)

    out = api._answer_once("1", "삼성전자 계약금액은?", deadline=1e18)
    assert calls == ["v2"]            # 구 경로는 부르지 않는다
    assert out["answer"] == "v2 답변"


def test_v2_failure_falls_back_to_the_old_path(api, monkeypatch):
    def boom(qid, q, *, deadline):
        raise RuntimeError("v2 고장")

    def fake_agent(qid, q, *, deadline):
        return {"question_id": qid, "question": q, "retrieved_context": "",
                "think_trace": "agent", "answer": "구 경로 답변"}

    monkeypatch.setitem(api._state, "askv2", object())
    monkeypatch.setitem(api._state, "tools", [object()])
    monkeypatch.setattr(api, "_answer_with_v2", boom)
    monkeypatch.setattr(api, "_answer_with_agent", fake_agent)

    assert api._answer_once("1", "질문", deadline=1e18)["answer"] == "구 경로 답변"


def test_abstention_is_not_treated_as_failure(api, monkeypatch):
    """거부는 설계된 동작이다. 여기서 폴백하면 아래 경로가 없는 사실을 지어낸다."""
    def refuse(qid, q, *, deadline):
        return {"question_id": qid, "question": q, "retrieved_context": "",
                "think_trace": "stopped_at=abstention_gate",
                "answer": "제공된 DART 공시 근거에서는 확인되지 않습니다."}

    called = []
    monkeypatch.setitem(api._state, "askv2", object())
    monkeypatch.setitem(api._state, "tools", [object()])
    monkeypatch.setattr(api, "_answer_with_v2", refuse)
    monkeypatch.setattr(api, "_answer_with_agent",
                        lambda *a, **k: called.append("agent") or {})

    out = api._answer_once("1", "쿠팡 매출액은?", deadline=1e18)
    assert "확인되지 않습니다" in out["answer"]
    assert called == []               # 폴백하지 않았다


def test_response_has_the_five_required_fields(api, monkeypatch):
    """대회 응답 규격. 필드가 빠지면 채점 자체가 안 된다."""
    result = types.SimpleNamespace(
        answer="답", stopped_at="answered", hcx_calls=1, retries=0,
        plan=types.SimpleNamespace(answer_mode="closed", task="lookup",
                                   companies=["삼성전자"], report_kinds=["단일판매공급계약체결"],
                                   aggregation="none", latest_policy="latest_only"),
        scope=types.SimpleNamespace(scope="in_scope"),
        decomposed=types.SimpleNamespace(sub_queries=[1], merged=[1, 2], empty_labels=[]),
        sufficiency=types.SimpleNamespace(ok=True, reasons=[]),
        abstention=types.SimpleNamespace(action="answer", reason="sufficient"),
        validation_result=None,
        evidence_pack=types.SimpleNamespace(prompt_text="[EVIDENCE 1] ..."),
    )
    monkeypatch.setitem(api._state, "askv2",
                        types.SimpleNamespace(run=lambda q: result))
    out = api._answer_with_v2("42", "질문", deadline=1e18)
    assert set(out) == {"question_id", "question", "retrieved_context",
                        "think_trace", "answer"}
    assert out["question_id"] == "42"
    assert out["retrieved_context"].startswith("[EVIDENCE 1]")


def test_trace_records_where_it_stopped(api, monkeypatch):
    """거부가 '범위 밖'인지 '근거 없음'인지 사후에 가릴 수 있어야 한다."""
    result = types.SimpleNamespace(
        answer="확인되지 않습니다", stopped_at="scope_gate", hcx_calls=0, retries=0,
        plan=None, scope=types.SimpleNamespace(scope="hard_out_scope"),
        decomposed=None, sufficiency=None, abstention=None,
        validation_result=None, evidence_pack=None)
    monkeypatch.setitem(api._state, "askv2",
                        types.SimpleNamespace(run=lambda q: result))
    out = api._answer_with_v2("1", "삼성전자 주가", deadline=1e18)
    assert "stopped_at=scope_gate" in out["think_trace"]
    assert "hard_out_scope" in out["think_trace"]
    assert out["retrieved_context"] == ""      # 검색을 안 했으므로 비어야 한다
