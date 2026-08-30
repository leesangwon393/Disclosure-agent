"""HCX Answer Generator (§65~66).

핵심 원칙: Evidence 에 있는 내용만 사용한다. 모델에게 공시 내용을 "기억"해서
답하도록 요구하지 않는다 — Evidence Pack 텍스트만 근거로 준다."""

from __future__ import annotations

from disclosure_rag.agent.evidence import EvidencePack
from disclosure_rag.agent.hcx_client import HCXClient

# 공통부. 아래 규칙들은 전부 **실측 실패에서 나온 것**이라 모드와 무관하게 지킨다.
# 기존 이름(ANSWER_SYSTEM_PROMPT)은 호환을 위해 그대로 둔다 — plan 을 주지 않으면
# 지금까지와 똑같이 이 프롬프트가 쓰인다.
ANSWER_SYSTEM_PROMPT = """당신은 금융공시(DART) 근거 기반 답변 생성기입니다.

## 하는 일
아래 [EVIDENCE]와 [TOOL RESULT]에서 질문에 해당하는 항목을 찾아 그 값을 답합니다.

## 숫자 표기 — 가장 중요
1. **근거에 적힌 숫자를 글자 그대로 옮기세요.** 쉼표 위치까지 그대로입니다.
   조/억/만 단위로 바꾸지 마세요. 반올림하지 마세요.
   (실측 실패: `224,787,773,988,054` 를 "224조 7877억 7398백54만 054원"으로
    바꿔 쓰는 바람에 자릿수가 틀렸습니다.)
2. 단위 환산이 꼭 필요하면 **원문 숫자를 먼저 쓰고** 괄호로 덧붙이세요.

## 근거 사용
3. [EVIDENCE]와 [TOOL RESULT]에 있는 내용만 사용하세요. 없는 숫자·날짜·사실을
   추측하거나 지어내지 마세요.
4. **같은 항목이 여러 공시에 나오면 어느 하나를 고르려 하지 말고, 공시일과 함께
   전부 제시하세요.** 질문이 시점을 특정하지 않았다면 그게 맞는 답입니다.
   (실측 실패: 값이 3개 공시에 흩어져 있자 "확인할 수 없습니다"라고 답했습니다.
    근거에 있는데도 그렇게 답하면 오답입니다.)
5. **질문에 없는 결론을 덧붙이지 마세요.** 질문이 "가장 최근"을 묻지 않았다면
   "따라서 가장 최근은 ~입니다" 같은 문장을 만들지 마세요. 시점이 모호하면
   값을 나열하고 **"질문에 시점이 명시되지 않아 해당하는 공시를 모두 제시합니다"**
   라고만 덧붙이세요.
   (실측 실패: 묻지도 않은 "가장 최근의 순자산액은 ~" 결론을 붙였습니다.
    채점자가 다른 시점을 기대하면 오답 처리됩니다.)
6. 근거에 **정말로 없을 때만** "제공된 근거로는 확인할 수 없습니다"라고 답하세요.
   찾기 어렵다는 이유로 이렇게 답하지 마세요.

## 정정공시
7. 정정 관련 질문이 아니라면 "정정 상태: 원본 (최신)" 또는 "정정본 (최신)"으로
   표시된 최신 유효본을 우선 사용하세요.
8. 정정 전후 비교 질문이면 원본과 정정본을 모두 명시적으로 비교하세요.

## 출력
9. 답변 마지막 줄에 "근거: report_id(chunk_id), ..." 형식으로 실제로 사용한
   근거를 나열하세요."""


def generate_answer(client: HCXClient, evidence_pack: EvidencePack, *,
                    max_tokens: int | None = None,
                    thinking: dict | None = None,
                    thinking_policy: str = "off",
                    correction_note: str | None = None,
                    plan=None) -> str:
    """`correction_note` 는 재생성 시 **원인별** 교정 지시다(§ask 재생성 루프).

    상원 레포는 재생성 지시가 항상 "암산하지 마" 로 고정돼 있어서, 문서 ID가
    근거 없는 숫자로 잡힌 경우에도 암산 교정 지시가 나갔다. 원인을 단정하지
    않는다.
    """
    if not evidence_pack.citations and not evidence_pack.tool_results_summary:
        return "제공된 근거로는 확인할 수 없습니다. (검색된 공시 근거가 없습니다.)"

    user_content = evidence_pack.prompt_text
    if correction_note:
        user_content = f"{user_content}\n\n[재작성 지시]\n{correction_note}\n"

    messages = [
        {"role": "system", "content": build_answer_prompt(plan)},
        {"role": "user", "content": user_content},
    ]
    if max_tokens is None:
        max_tokens = answer_token_budget(plan)
    if thinking is None:
        # 명시적으로 주지 않으면 정책으로 정한다. 기본 정책은 off 라
        # 지금까지와 동일하게 {"effort": "none"} 이 나간다.
        thinking = resolve_thinking(plan, policy=thinking_policy)
    # HCX-007(reasoning)의 thinking 을 끈다.
    # 실측(2026-08-30, scripts/diag_answer.py): 프롬프트·근거가 완전히 동일한데
    #   thinking ON  -> "확인할 수 없습니다"  (근거 13,542자 안에 정답이 있는데도)
    #   thinking OFF -> 정답
    # 긴 근거에서 값을 뽑는 작업에서 reasoning 모드가 오히려 방해한다.
    # 복합추론 유형에서 다시 켜야 할 수도 있으므로 인자로 열어둔다.
    result = client.chat(messages, max_tokens=max_tokens, temperature=0.2,
                         thinking=thinking)
    return result.get("content", "")


# ============================================================================
# answer_mode / task 별 프롬프트 분기 (Stage 14, 2026-08-30)
# ============================================================================
#
# 위 공통 프롬프트는 "값 하나 뽑기"에 맞춰져 있다. 그런데 suite_v1 38문항 중
# 16문항이 open(서술형)이고, 그 문항들은 요구가 다르다:
#
#     closed  "순자산액은 얼마인가"        -> 숫자 하나. 자릿수가 생명이다.
#     open    "주요 내용을 정리해줘"        -> 항목별 나열. 빠진 항목을 밝혀야 한다.
#     mixed   "있는가? 있다면 설명해줘"     -> 앞은 단정, 뒤는 나열.
#
# 한 프롬프트로 셋을 다 시키면 open 답변이 값 하나만 뱉거나, closed 답변에
# 불필요한 서술이 붙는다. 그래서 공통부 + 모드별 블록으로 조립한다.
#
# 비대칭 주의: closed 를 open 으로 오판하면 답변이 장황해질 뿐이지만, open 을
# closed 로 오판하면 항목이 통째로 빠진다. 그래서 mixed 는 open 쪽에 붙인다.

_MODE_CLOSED = """
## 이 질문의 답 형태 — 단일 값
10. 질문이 묻는 **값 하나**를 먼저 제시하세요. 배경 설명을 앞에 두지 마세요.
11. 단위를 반드시 함께 쓰세요(원/주/건/%). 근거에 단위가 있으면 그대로 옮기세요.
12. 값 이외의 항목을 덧붙이지 마세요. 묻지 않은 항목은 답변을 흐립니다."""

_MODE_OPEN = """
## 이 질문의 답 형태 — 항목별 정리
10. **항목별로 줄을 나눠** 제시하세요. 한 문단으로 뭉치지 마세요.
11. 항목마다 그 값의 근거 공시를 함께 표시하세요.
12. 요구된 항목 중 근거에서 확인되지 않은 것이 있으면, 지어내지 말고
    **"(확인되지 않음)"** 이라고 그 항목 자리에 적으세요. 항목을 조용히
    빼면 읽는 사람이 다 확인된 것으로 오해합니다.
13. 근거에 없는 배경지식·업계 상식·추론을 덧붙이지 마세요."""

_MODE_MIXED = """
## 이 질문의 답 형태 — 판정 + 정리
10. **첫 줄에서 예/아니오를 단정**하세요("정정 내역이 있습니다" 등).
    그 판단의 근거 공시를 바로 옆에 표시하세요.
11. 그 다음 줄부터 **항목별로 줄을 나눠** 설명하세요.
12. 요구된 항목 중 근거에서 확인되지 않은 것은 **"(확인되지 않음)"** 이라고
    적으세요. 조용히 빼지 마세요.
13. 근거에 없는 배경지식·추론을 덧붙이지 마세요."""

_TASK_BLOCKS = {
    "correction_diff": """
## 정정 전후 비교
14. **최초 공시 값 / 최종 정정본 값**을 항목마다 나란히 쓰세요.
15. 값이 바뀌지 않은 항목도 "변동 없음"으로 적으세요 — 바뀐 것만 쓰면
    나머지를 확인했는지 알 수 없습니다.
16. 정정 사유가 근거에 있으면 함께 제시하세요.""",
    "compare": """
## 비교
14. 비교 대상 **각각의 값을 먼저** 제시한 뒤 결론을 쓰세요.
15. 한쪽 대상의 근거가 없으면 결론을 내지 말고 **어느 쪽이 확인되지 않았는지**
    밝히세요. 한쪽 값만으로 "더 크다"고 답하면 오답입니다.""",
    "calculate": """
## 계산
14. 계산 결과는 이미 [TOOL RESULT]에 있습니다. **직접 암산하지 마세요.**
15. 사용한 피연산자와 그 출처 공시를 함께 밝히세요.""",
    "count": """
## 건수
14. 건수를 먼저 쓰고, **센 대상을 나열**하세요. 나열 없이 숫자만 쓰면
    검증이 불가능합니다.""",
}

# 답변 길이 예산. closed 는 값 하나라 길 필요가 없고, open 은 항목 수만큼 늘어난다.
_MAX_TOKENS = {"closed": 800, "open": 2400, "mixed": 2400, "unknown": 1600}


def build_answer_prompt(plan) -> str:
    """QueryPlan 을 보고 시스템 프롬프트를 조립한다.

    `plan` 이 None 이면 공통부만 쓴다 — 기존 호출부가 그대로 동작한다.
    """
    if plan is None:
        return ANSWER_SYSTEM_PROMPT
    mode = getattr(plan, "answer_mode", "unknown")
    block = {"closed": _MODE_CLOSED, "open": _MODE_OPEN, "mixed": _MODE_MIXED}.get(mode, "")
    prompt = ANSWER_SYSTEM_PROMPT + block + _TASK_BLOCKS.get(getattr(plan, "task", ""), "")

    if getattr(plan, "aggregation", "none") in ("max", "min"):
        # 실측 실패(S011): 값 10개를 주자 모델이 최댓값이 아닌 5번째 값을
        # 골랐다. 목록에서 최댓값 고르기는 모델이 자주 틀린다.
        word = "가장 큰" if plan.aggregation == "max" else "가장 작은"
        prompt += (f"\n\n## {word} 값 고르기\n"
                   "17. [FACT] 블록의 **▶ 표시가 이미 계산된 답**입니다. "
                   "그 값을 그대로 쓰세요.\n"
                   "18. 목록의 값들을 직접 비교해서 고르지 마세요. "
                   "▶ 가 없을 때만 근거에서 찾으세요.")

    fields = list(getattr(plan, "expected_fields", []) or [])
    if fields and mode in ("open", "mixed"):
        # 무엇을 다뤄야 하는지 이름으로 준다. "주요 내용을 정리"는 사람마다
        # 다르게 읽히지만 항목 목록은 그렇지 않다.
        header = "\n\n## 이 답변에 반드시 다뤄야 하는 항목\n"
        listed = "\n".join(f"- {f}" for f in fields)
        prompt += header + listed + "\n확인되지 않은 항목은 '(확인되지 않음)'이라고 적으세요."
    return prompt


def answer_token_budget(plan) -> int:
    if plan is None:
        return 800
    return _MAX_TOKENS.get(getattr(plan, "answer_mode", "unknown"), 1600)


# ============================================================================
# thinking(reasoning) A/B 장치 (2026-08-30)
# ============================================================================
#
# HCX-007 은 reasoning 모델이고 thinking 이 **기본 ON** 이다. 우리는 지금 전
# 경로에서 끄고 있는데, 그 판단의 근거는 한 종류의 질문에서만 잰 것이다:
#
#     scripts/diag_answer.py — 같은 프롬프트, 같은 근거 13,542자
#       thinking ON   -> "확인할 수 없습니다"   (근거 안에 정답이 있는데도)
#       thinking OFF  -> 정답
#
# 이건 `closed / lookup`(긴 근거에서 값 하나 뽑기)이다. 다단계 작업에서도
# 같은 결론일지는 **재보지 않았다.** suite_v1 기준 후보가 15문항(39%)이다:
#
#     correction_diff  4문항   최초본↔최종본을 항목별로 대조
#     compare         11문항   두 값을 비교해 결론
#
# 그래서 기본값은 그대로 OFF 로 두고, 정책만 바꿔 A/B 할 수 있게 해둔다.
# **재보기 전에는 켜지 않는다** — 이 모델에서 reasoning 이 오히려 나빴던
# 실측이 있으므로, 유형이 다르면 좋아질 거라는 건 아직 가정이다.

THINKING_OFF: dict = {"effort": "none"}

# ON 쪽 값은 **추측이다.** CLOVA 문서에 effort 허용값이 명시돼 있지 않고,
# 우리가 실측으로 아는 건 "none" 이 동작한다는 것뿐이다. 잘못된 값이면
# hcx_client 가 400 을 받고 **파라미터를 빼서** 모델 기본값(ON)으로 재시도한다.
# 즉 어느 쪽이든 thinking 은 켜진다.
DEFAULT_THINKING_EFFORT = "low"

# reasoning 이 도움 될 만한 작업. 가설이지 측정 결과가 아니다.
THINKING_TASKS = frozenset({"correction_diff", "compare"})


def resolve_thinking(plan, *, policy: str = "off",
                     effort: str = DEFAULT_THINKING_EFFORT) -> dict:
    """정책과 계획을 보고 thinking 설정을 정한다.

        off   전부 끈다 (기본값, 현재 운영 상태)
        auto  THINKING_TASKS 에 해당하는 작업만 켠다
        on    전부 켠다
    """
    if policy == "on":
        return {"effort": effort}
    if policy == "auto" and plan is not None and getattr(plan, "task", None) in THINKING_TASKS:
        return {"effort": effort}
    return dict(THINKING_OFF)
