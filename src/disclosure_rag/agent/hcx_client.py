"""HyperCLOVA X (CLOVA Studio) Chat Completions 저수준 클라이언트 (§57).

Agent 의 reasoning/routing/최종 답변은 반드시 HCX 계열 모델만 쓴다 — 다른 외부
LLM 을 online Agent 용도로 쓰지 않는다 (§57). Semantic Router 는 embedding 기반
컴포넌트라 별도 취급(§37) — 이 클라이언트는 그와 무관하다.

실제 API 로 확인한 사실 (.env 의 실키로 라이브 테스트):
- endpoint: POST https://clovastudio.stream.ntruss.com/v3/chat-completions/{model}
- auth: `Authorization: Bearer {HCX_API_KEY}` (Bearer 방식 키 기준)
- tool calling 은 OpenAI 호환에 가까움 (messages/tools/tool_choice, 응답의
  message.toolCalls[].function.{name,arguments} — arguments 는 이미 dict 로 옴,
  OpenAI 처럼 JSON string 이 아님).
- `tools` 파라미터와 `maxTokens` 를 동시에 주면 400(Invalid parameter) 이 나는
  조합이 실측으로 확인됨 — tool-calling 요청에는 maxTokens 를 생략한다.
- 짧은 시간에 연속 호출하면(질문 1개 안에서도 tool-calling 라운드트립으로 호출이
  누적됨) 이후 요청이 400("Unsupported function")으로 실패하는 게 반복 관측됨 —
  단일 호출은 항상 성공하고 연속 호출 시에만 실패하는 패턴이라 계정/모델 tier 의
  RPM(분당 요청수) rate-limit 로 추정(에러 메시지 자체는 이유를 설명하지 않음).
  지수 백오프 재시도로 대부분 우회되지만, Phase 15+ 실사용 시 요청 빈도를
  스로틀링하거나 tier 상향이 필요할 수 있다.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 호출 간격 조절
#
# 2026-08-30 실측: 60문항 배치에서 **6문항(10%)이 429 "Too many requests"** 로
# 통째로 실패했다. 재시도(3s/6s/12s 백오프)로도 못 뚫은 게 아니라, 애초에
# 요청을 너무 촘촘히 보내서 rate limit 창에 계속 걸린 것이다.
#
# RPM 제한은 **API 키 단위**지 클라이언트 인스턴스 단위가 아니다. agent 용과
# answer 용으로 HCXClient 를 따로 만들어도 같은 지갑을 쓴다. 그래서 타이머를
# 모듈 레벨에 두고 이 프로세스의 모든 인스턴스가 공유하게 한다.
#
# 정확한 계정 RPM 을 모르므로 1.0초는 보수적 추정치다. 429 가 또 나면 올리고,
# 배치가 너무 느리면 실측하며 낮춘다.
_PACING_LOCK = threading.Lock()
_LAST_CALL_AT = 0.0


def _wait_for_slot(min_interval_sec: float) -> None:
    if min_interval_sec <= 0:
        return
    global _LAST_CALL_AT
    with _PACING_LOCK:
        now = time.monotonic()
        wait = _LAST_CALL_AT + min_interval_sec - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _LAST_CALL_AT = now

_ENDPOINT_TMPL = "https://clovastudio.stream.ntruss.com/v3/chat-completions/{model}"


class HCXError(RuntimeError):
    pass


class HCXClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        env_path: str | Path = ".env",
        timeout: float = 60.0,
        min_interval_sec: float = 1.0,
    ):
        if api_key is None or model is None:
            self._load_env(env_path)
        self.api_key = api_key or os.environ.get("HCX_API_KEY")
        self.model = model or os.environ.get("HCX_MODEL")
        if not self.api_key:
            raise HCXError("HCX_API_KEY 가 없습니다 (.env 확인)")
        if not self.model:
            raise HCXError("HCX_MODEL 이 없습니다 (.env 확인)")
        self.timeout = timeout
        self.min_interval_sec = min_interval_sec
        self._url = _ENDPOINT_TMPL.format(model=self.model)
        # HCX-007(reasoning 모델)은 thinking 모드 기본 on 상태에서 tools 와 같이
        # 쓰면 400 이 난다(실측, Stage 10) — 모델명으로 감지해 자동으로 꺼준다.
        # 호출부(agent_loop.py 등)를 모델별로 분기시키지 않기 위해 클라이언트
        # 레벨에서 흡수한다.
        self._default_thinking = {"effort": "none"} if "007" in self.model else None
        # HCX-007 은 max token 제한 파라미터 이름이 다른 모델과 다르다(실측,
        # Stage 12): "maxTokens"를 주면 400("Invalid parameter: maxTokens")이
        # 나고, 대신 "maxCompletionTokens"를 써야 정상 동작한다. 원인 불명
        # (reasoning 모델이라 내부적으로 토큰 예산을 다르게 다루는 것으로 추정)
        # — 호출부 무수정으로 흡수하기 위해 여기서 모델별로 파라미터 이름을 정한다.
        self._max_tokens_param = "maxCompletionTokens" if "007" in self.model else "maxTokens"

    def _load_env(self, env_path: str | Path) -> None:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=str(env_path))

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float = 0.3,
        top_p: float = 0.8,
        max_tokens: int | None = None,
        max_retries: int = 3,
        thinking: dict | None = None,
    ) -> dict:
        """단일 chat completion 호출. 원본 응답의 result.message 를 그대로 반환한다.

        실측: 드물게 동일 요청이 일시적으로 400("Unsupported function")을 내다가
        재시도하면 바로 성공하는 케이스가 관측됨(원인 불명, API 측 일시 오류로 추정)
        — 짧게 재시도한다.

        `thinking`: HCX-007(reasoning 모델)은 기본적으로 thinking 모드가 켜져
        있는데, 이 상태에서 `tools`를 같이 주면 400("Invalid parameter: tools,
        thinking")이 난다(실측, Stage 10). `thinking={"effort": "none"}`을
        명시하면 해결된다. HCX-DASH-002/HCX-005는 이 파라미터가 필요 없다
        (안 줘도 tool-calling 정상 동작) — 모델별로 호출부에서 필요할 때만 전달.

        `max_tokens`: HCX-007 은 파라미터 이름 자체가 다르다("maxTokens"를
        주면 400, "maxCompletionTokens"라고 줘야 함 — 실측, Stage 12).
        `self._max_tokens_param`(모델별 자동 결정)을 키로 써서 흡수한다."""
        payload: dict[str, Any] = {"messages": messages, "topP": top_p, "temperature": temperature}
        if thinking is None:
            thinking = self._default_thinking
        if thinking is not None:
            payload["thinking"] = thinking
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["toolChoice"] = tool_choice
            # 실측: tools + maxTokens 동시 사용 시 400 (§ 위 docstring)
        elif max_tokens is not None:
            payload[self._max_tokens_param] = max_tokens

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_error: Exception | None = None
        rate_limited = False
        thinking_fallback_done = False
        for attempt in range(max_retries + 1):
            _wait_for_slot(self.min_interval_sec)
            try:
                resp = requests.post(self._url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                # 2026-08-30: 예전엔 이 호출이 try 밖이라 ConnectTimeout/
                # ConnectionError 가 재시도 없이 그대로 질문 전체를 죽였다.
                # 상태코드 오류만 재시도하고 네트워크 오류는 안 하는 건 앞뒤가 안 맞는다.
                last_error = HCXError(f"HCX 네트워크 오류: {type(e).__name__}: {e}")
                rate_limited = False
            else:
                if resp.status_code == 200:
                    body = resp.json()
                    if body.get("status", {}).get("code") == "20000":
                        return body["result"]["message"]
                    last_error = HCXError(f"HCX API 오류: {body.get('status')}")
                    rate_limited = str(body.get("status", {}).get("code", "")).startswith("429")
                else:
                    last_error = HCXError(f"HCX API 오류 {resp.status_code}: {resp.text[:500]}")
                    rate_limited = resp.status_code == 429
                    # thinking 값이 안 먹으면 그 파라미터를 빼고 한 번 더 시도한다.
                    #
                    # 왜 필요한가: CLOVA 문서에 `effort` 의 허용값이 명시돼 있지
                    # 않다. 우리가 실측으로 아는 건 `{"effort": "none"}` 이
                    # 동작한다는 것뿐이고, ON 쪽 값("low" 등)은 추측이다.
                    # 38문항을 돌리다 중간에 400 으로 전부 죽는 것보다,
                    # **파라미터를 빼서 모델 기본값(HCX-007 은 thinking ON)**
                    # 으로 진행하는 편이 낫다.
                    if (resp.status_code == 400 and not thinking_fallback_done
                            and "thinking" in resp.text and "thinking" in payload):
                        removed = payload.pop("thinking")
                        thinking_fallback_done = True
                        logger.warning("[HCX] thinking=%s 가 거부됐다 — 파라미터를 빼고 "
                                       "모델 기본값으로 재시도한다", removed)
                        continue        # 대기 없이 바로 다시

            if attempt < max_retries:
                # 429 는 "잠깐 기다리면 풀리는" 오류다. 다른 오류보다 길게 쉰다.
                base = 10.0 if rate_limited else 3.0
                wait = base * (2 ** attempt)
                logger.warning("[HCX] chat 호출 실패(시도 %d/%d)%s, %.1fs 후 재시도: %s",
                               attempt + 1, max_retries + 1, " [rate limit]" if rate_limited else "",
                               wait, last_error)
                time.sleep(wait)
        raise last_error

    def chat_simple(self, prompt: str, *, system: str | None = None, max_tokens: int = 512, temperature: float = 0.3) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = self.chat(messages, max_tokens=max_tokens, temperature=temperature)
        return result.get("content", "")
