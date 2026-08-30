"""HCX 호출 간격 조절과 재시도 — 실측 실패를 박제한다.

2026-08-30 60문항 배치에서 **6문항(10%)이 429 "Too many requests"** 로 통째로
날아갔다. 원인은 재시도 부족이 아니라 요청을 너무 촘촘히 보낸 것이었다.
"""
from __future__ import annotations

import time

import pytest
import requests

import disclosure_rag.agent.hcx_client as hcx


@pytest.fixture(autouse=True)
def _reset_pacing():
    hcx._LAST_CALL_AT = 0.0
    yield
    hcx._LAST_CALL_AT = 0.0


def _client(**kw):
    return hcx.HCXClient(api_key="k", model="HCX-007", **kw)


class _Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body or {"status": {"code": "20000"}, "result": {"message": {"role": "assistant", "content": "ok"}}}
        self.text = text

    def json(self):
        return self._body


# --- 간격 조절 --------------------------------------------------------------

def test_calls_are_paced(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    c = _client(min_interval_sec=0.2)
    t0 = time.monotonic()
    for _ in range(3):
        c.chat([{"role": "user", "content": "hi"}])
    assert time.monotonic() - t0 >= 0.4, "호출 간격 조절이 걸리지 않았다"


def test_pacing_is_shared_across_instances(monkeypatch):
    """RPM 은 API 키 단위다 — agent 용/answer 용 클라이언트가 따로여도 같은 지갑."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    a, b = _client(min_interval_sec=0.2), _client(min_interval_sec=0.2)
    t0 = time.monotonic()
    a.chat([{"role": "user", "content": "hi"}])
    b.chat([{"role": "user", "content": "hi"}])
    assert time.monotonic() - t0 >= 0.2, "인스턴스가 다르면 간격이 무시된다"


def test_pacing_can_be_disabled(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    c = _client(min_interval_sec=0.0)
    t0 = time.monotonic()
    for _ in range(3):
        c.chat([{"role": "user", "content": "hi"}])
    assert time.monotonic() - t0 < 0.2


# --- 네트워크 예외도 재시도 대상 --------------------------------------------

def test_network_error_is_retried(monkeypatch):
    """예전엔 requests.post 가 try 밖이라 타임아웃이 질문 전체를 죽였다."""
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectTimeout("boom")
        return _Resp()

    monkeypatch.setattr(requests, "post", flaky)
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    c = _client(min_interval_sec=0.0)
    out = c.chat([{"role": "user", "content": "hi"}])
    assert out["content"] == "ok"
    assert calls["n"] == 2


def test_network_error_finally_raises(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("x")))
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    with pytest.raises(hcx.HCXError) as e:
        _client(min_interval_sec=0.0).chat([{"role": "user", "content": "hi"}], max_retries=1)
    assert "네트워크" in str(e.value)


# --- 429 는 더 길게 쉰다 ----------------------------------------------------

def test_rate_limit_uses_a_longer_backoff(monkeypatch):
    waits: list[float] = []
    calls = {"n": 0}

    def rate_limited(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(status=429, text="Too many requests")
        return _Resp()

    monkeypatch.setattr(requests, "post", rate_limited)
    monkeypatch.setattr(time, "sleep", lambda s: waits.append(s))
    _client(min_interval_sec=0.0).chat([{"role": "user", "content": "hi"}])
    assert waits and waits[0] >= 10.0, f"429 백오프가 짧다: {waits}"


def test_other_error_keeps_the_short_backoff(monkeypatch):
    waits: list[float] = []
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        return _Resp(status=500, text="oops") if calls["n"] == 1 else _Resp()

    monkeypatch.setattr(requests, "post", flaky)
    monkeypatch.setattr(time, "sleep", lambda s: waits.append(s))
    _client(min_interval_sec=0.0).chat([{"role": "user", "content": "hi"}])
    assert waits and waits[0] < 10.0
