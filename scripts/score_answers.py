#!/usr/bin/env python3
"""답변 채점기 — 검색이 아니라 **최종 답변**을 정답과 대조한다.

## 왜 새로 만들었나

`scripts/eval_e2e.py` 는 검색 계층만 잰다(정답 조각이 top-k 안에 들어왔는가).
대회가 채점하는 건 최종 답변이고, 지금까지 그걸 재는 코드가 없었다. 그래서
"검색은 되는데 답을 못 한다" 와 "애초에 못 찾는다" 가 구분되지 않았다.

## 실패를 네 갈래로 나눈다 — 이게 이 스크립트의 핵심 산출물이다

    정답        답변에 정답 수치가 있다
    답변실패    근거에는 정답 문서가 들어왔는데 답변이 틀렸다   -> 프롬프트/생성 문제
    검색실패    근거에 정답 문서가 아예 없다                    -> 검색/필터 문제
    거부        "확인할 수 없습니다"                            -> 위 둘 중 어느 쪽인지 함께 본다

이 구분이 없으면 프롬프트를 고쳐야 할 때 검색을 고치게 된다.

## 모드

    --mode retrieval  (기본) HCX 호출 0회. 크레딧 안 쓴다.
                      "근거만 보면 답할 수 있었는가" 상한을 잰다.
    --mode full       실제 파이프라인(ask). 문항당 HCX 3회 내외.

`--mode full` 은 크레딧을 쓴다. 20문항을 넘기면 예상 비용/시간을 출력하고
`--yes` 없이는 실행하지 않는다.

## 출력 (기존 results/ 컨벤션)

    results/<out>/config.json  metrics.json  results.csv  failure_cases.jsonl  summary.md
"""
from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal, InvalidOperation
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger("score_answers")

# 파이프라인이 실제로 내보내는 거부 문구. **세 곳(abstention/scope_gate/ask)의
# 문구를 전부 여기 반영해야 한다** — 2026-08-31 대조에서 8종 중 1종만 잡히고
# 있었다. 그 결과 HCX 를 안 부르고 거부한 문항이 '오답'으로 분류됐고,
# `_verdict` 도 "불명"을 돌려줘 예/아니오 채점이 어긋났다.
_REFUSAL = (
    "확인할 수 없습니다", "확인되지 않습니다", "찾을 수 없습니다", "확인이 어렵습니다",
    # abstention.py
    "확인할 수 없어 결론을 내지 않습니다", "확인할 수 없어 답변하지 않습니다",
    "확인되어 변경 내역을 단정하지 않습니다",
    "근거를 확인하지 못한 대상", "확인되지 않은 필수 항목", "정정 전후 중 한쪽만 확인된 항목",
    # scope_gate.py
    "확인할 수 있는 범위를 벗어납니다",
)

# 평균을 내는 비율형 지표. retrieval / full 양쪽에 없는 키는 자동으로 건너뛴다.
_RATE_KEYS = (
    "answer_hit", "evidence_hit", "answer_ceiling", "refusal",
    "numbers_grounded", "has_citation", "validation_passed",
    # 순위 품질 (2026-08-30 추가) — "가져왔는가"와 "몇 등으로 가져왔는가"를 분리한다
    "context_recall", "context_precision", "context_ap", "mrr", "ndcg_at_10",
    # 나열로 걸린 것과 정확히 맞힌 것의 구분 (2026-08-30)
    "answer_hit_exact",
)

# 채점 기준(정답 문장 또는 required_all)이 있는 문항만으로 평균 내는 지표.
# 전체를 분모로 쓰면 채점 불가 문항이 자동 0점이 되어 실제보다 낮게 나온다.
# 실측(2026-08-30): suite_v1 38문항 중 16문항에 정답이 없어 52.6% 로 찍혔고,
# 채점 가능한 22문항만 보면 90.9% 였다.
_GRADEABLE_KEYS = ("graded_hit",)

# 값이 None 일 수 있는 지표(해당 없음). 평균에서 None 을 빼고 센다.
_OPTIONAL_KEYS = (
    "field_coverage", "silent_omission_rate", "citation_recall", "citation_precision",
    "required_coverage",
)
_NUM = re.compile(r"\d[\d,]*\.?\d*")


# --------------------------------------------------------------------------- 정규화

def _norm(s: str) -> str:
    """콤마·공백·통화기호를 지운 비교용 문자열."""
    return re.sub(r"[,\s원₩]", "", s or "")


def _numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in _NUM.finditer(text or "")}


def _gold_answers(row: dict) -> list[str]:
    """정답 후보. 한 질문에 여러 문서가 걸리면 정답도 여러 개다(gold 파일 실측)."""
    out = [a for a in (row.get("answers") or []) if a]
    if row.get("answer"):
        out.append(row["answer"])
    return [a for a in dict.fromkeys(out)]


def _gold_report_ids(row: dict) -> set[str]:
    return set(row.get("gold_report_ids") or row.get("gold_doc_ids") or [])


# ------------------------------------------------------------------- 서술형 대조
#
# 서술형 질문("자금조달 내역을 유형별로 정리해줘")은 정답이 문자열 하나가 아니다.
# 그래서 정답지에 `required_all` — **반드시 등장해야 하는 항목들** — 을 넣고
# 포함 여부로 채점한다. 전부 나오면 정답, 아니면 몇 개 나왔는지를 따로 센다.
#
# 표기 차이를 흡수하지 않으면 맞은 답을 틀렸다고 한다. 실제로 갈리는 게 둘이다:
#   수치  정답지 "9.90"      vs 답변 "9.9%"        -> 수로 비교한다
#   날짜  정답지 "2024-04-24" vs 답변 "2024년 4월 24일" -> 표기 변형을 모두 본다

_DATE_ISO = re.compile(r"^(\d{4})[-./](\d{1,2})[-./](\d{1,2})$")
_DATE_KO = re.compile(r"^(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일$")
_NUMERIC_TOKEN = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")


_REPORT_ID_PAT = re.compile(r"\b(?:periodic|major|exchange|holding)_\d+")


def _body_only(answer: str) -> str:
    """채점용 본문 — **문서 ID만** 지운다. 뒷부분을 자르지 않는다.

    왜 지우나 (2026-08-31 실측)
    --------------------------
    `report_id(exchange_20240424800596)` 안에 `20240424` 가 들어 있어서
    날짜 정답 `2024-04-24` 가 **거부 답변에서도 만점**을 받았다.

        답변  "확인되지 않습니다. 근거: report_id(exchange_20240424800596)"
        정답  ["2024-04-24", "2026-10-30"]  -> 2/2 만점

    왜 자르면 안 되나 (같은 날, 자른 뒤 재현)
    ----------------------------------------
    처음엔 `"근거:"` 이후를 통째로 잘랐다. 그런데 모델은 항목마다 근거를
    다는 형식도 쓴다:

        - 현대글로비스: 3,365,500,000,000원
          - 근거: [exchange_20241231800103]
        - HMM: 1,282,363,356,560원
          ...
        따라서 더 큰 기업은 현대글로비스입니다.

    첫 `"근거:"` 에서 자르면 **뒤에 오는 값과 결론이 통째로 사라진다.**
    v2_off8 에서 S013·S037 두 문항이 맞는 답인데 오답으로 찍혔다.

    그래서 자르지 않고 문서 ID 토큰만 제거한다 — 오탐의 원인은 그것뿐이다.
    """
    text = (answer or "")
    text = _REPORT_ID_PAT.sub(" ", text)
    # `chunk_id: xxx::main::C1` 처럼 남는 식별자도 숫자를 품는다.
    text = re.sub(r"chunk_id\s*[:=]\s*\S+", " ", text)
    return text


def _norm_token(s: str) -> str:
    """비교용 정규화 — 콤마·공백·통화기호를 지우고 라틴 문자는 소문자로."""
    return re.sub(r"[,\s원₩]", "", (s or "")).lower()


def _date_forms(tok: str) -> list[str] | None:
    """날짜면 표기 변형 전부를, 날짜가 아니면 None 을 돌려준다."""
    t = (tok or "").strip()
    m = _DATE_ISO.match(t) or _DATE_KO.match(t)
    if not m:
        return None
    y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
    return [_norm_token(f) for f in (
        f"{y}-{mo:02d}-{d:02d}", f"{y}.{mo:02d}.{d:02d}", f"{y}/{mo:02d}/{d:02d}",
        f"{y}{mo:02d}{d:02d}",
        f"{y}년{mo}월{d}일", f"{y}년{mo:02d}월{d:02d}일",
    )]


def _decimals(text: str) -> set:
    """답변에 나온 수를 Decimal 집합으로. 콤마·소수점 표기 차이를 흡수한다."""
    out = set()
    for m in _NUM.finditer(text or ""):
        raw = m.group(0).replace(",", "").rstrip(".")
        try:
            out.add(Decimal(raw))
        except InvalidOperation:
            continue
    return out


def _token_present(tok: str, answer_norm: str, answer_nums: set) -> bool:
    tok = str(tok or "").strip()
    if not tok:
        return True
    forms = _date_forms(tok)
    if forms is not None:
        return any(f in answer_norm for f in forms)
    if _NUMERIC_TOKEN.match(tok):
        try:
            return Decimal(tok.replace(",", "")) in answer_nums
        except InvalidOperation:
            pass
    return _norm_token(tok) in answer_norm


def required_any_report(answer: str, groups) -> dict:
    """여러 정답 후보 중 **하나만** 맞으면 정답인 경우.

    질문이 대상을 특정하지 않을 때 쓴다. 실측(S023~S026, 2026-08-30):

        Q  "현대건설의 단일판매ㆍ공급계약체결 공시가 정정된 내역이 있는가?"
        현실  해당 유형 정정 공시 70건, 값이 바뀐 정정 체인 10개
        정답지  그중 1개 체인의 값만 정답으로 인정

    모델은 다른 체인을 설명했다. 틀린 게 아니라 **다른 걸 고른 것**이다.
    그래서 체인 전부를 후보로 두고, 하나라도 온전히 설명하면 정답으로 친다.

    통계는 가장 많이 맞은 후보 기준으로 낸다 — 부분 점수가 0 으로 뭉개지지 않게.
    """
    valid = [g for g in (groups or []) if [t for t in (g or []) if str(t).strip()]]
    if not valid:
        return {"n": 0, "matched": 0, "coverage": None, "missing": [], "n_groups": 0}
    reports = [required_report(answer, g) for g in valid]
    best = max(reports, key=lambda r: (r["matched"] == r["n"], r["coverage"] or 0))
    return {**best, "n_groups": len(valid),
            "n_full": sum(1 for r in reports if r["n"] and r["matched"] == r["n"])}


def required_report(answer: str, required) -> dict:
    """required_all 대조 결과. 빈 목록이면 coverage 는 None(해당 없음)."""
    items = [t for t in (required or []) if str(t).strip()]
    if not items:
        return {"n": 0, "matched": 0, "coverage": None, "missing": []}
    body = _body_only(answer)
    an = _norm_token(body)
    nums = _decimals(body)
    missing = [t for t in items if not _token_present(t, an, nums)]
    matched = len(items) - len(missing)
    return {"n": len(items), "matched": matched,
            "coverage": round(matched / len(items), 4), "missing": missing}


# --------------------------------------------------------------------------- 채점

_PAREN_NEG = re.compile(r"^\((?P<v>[\d,.]+)\)$")


def _as_decimal(text: str):
    """수치 표기를 Decimal 로. 회계 괄호 음수와 △ 도 받는다."""
    t = re.sub(r"[,\s원₩%]", "", (text or "").strip())
    if not t:
        return None
    neg = False
    m = _PAREN_NEG.match(t)
    if m:
        t, neg = m.group("v"), True
    elif t.startswith(("△", "▲", "-", "−")):
        t, neg = t[1:], True
    t = t.rstrip(".")
    try:
        d = Decimal(t)
    except InvalidOperation:
        return None
    return -d if neg else d


def _answer_hit(answer: str, golds: list[str]) -> bool:
    """정답 수치가 답변에 있는가. 표기 차이는 흡수한다.

    2026-08-31: 문자열 비교를 **Decimal 비교**로 바꿨다. 그전에는 같은
    채점기 안에서 규칙이 갈렸다 — 서술형 경로(`_token_present`)는 Decimal 로
    `9.90 == 9.9%` 를 흡수하는데, 값 경로는 문자열이라 오답이었다.
    실측으로 suite_v2 18문항이 이것 때문에 정답을 오답으로 받고 있었다
    (소수점 14건 `17.0` vs `17%`, 괄호 음수 4건 `(6,503)` vs `-6,503`).
    """
    if not golds:
        return False
    body = _body_only(answer)
    norm_answer = _norm(body)
    answer_decimals = {d for n in _numbers(body) if (d := _as_decimal(n)) is not None}
    for g in golds:
        ng = _norm(g)
        if not ng:
            continue
        gd = _as_decimal(g)
        if gd is not None:
            # 수치 정답은 **값 단위로 정확히** 일치해야 한다. 부분 문자열을
            # 허용하면 `1234` 가 `91,234,567` 안에서 정답이 된다(실측 확인).
            if gd in answer_decimals:
                return True
        elif len(ng) >= 2 and ng in norm_answer:
            # 비수치 정답(계약상대·사유 등)은 포함 여부로 본다.
            return True
    return False


# 답변에서 '값 후보'만 골라내기 위한 패턴.
_YEAR_TOKEN = re.compile(r"^(?:19|20)\d{2}$")


def _answer_numbers(answer: str) -> list[str]:
    """답변이 제시한 **값 후보**를 등장 순서대로. 중복은 한 번만.

    왜 필요한가 (2026-08-30 실측, v2 스모크 5문항)
    ---------------------------------------------
    `_answer_hit` 은 정답 숫자가 답변 **어딘가에** 있으면 정답으로 센다.
    그런데 실제 답변이 이랬다:

        질문: "삼성전자의 순자산액은 얼마인가?"   gold = 224,787,773,988,054
        답변: "254,330,082,981,146 / 236,396,657,259,591 / … / 224,787,773,988,054"
              -> 값 6개를 나열했고 정답은 6번째. 채점기는 '정답' 처리.

    사람 채점자는 이걸 정답으로 안 볼 가능성이 높다. 그래서 '정확히 맞힘'과
    '나열해서 걸린 것'을 구분할 재료를 남긴다. **기존 answer_hit 은 그대로
    둔다** — 지금까지의 수치와 비교 가능해야 하므로 정의를 바꾸지 않는다.

    제외 대상:
      - "근거:" 뒤쪽 (report_id 안의 긴 숫자열이 값으로 잡힌다)
      - report_id 토큰
      - 연도(19xx/20xx)와 한두 자리 목록 번호
    """
    body = (answer or "").split("근거:")[0]
    body = _REPORT_ID_PAT.sub(" ", body)
    out: list[str] = []
    for m in _NUM.finditer(body):
        raw = m.group(0)
        value = raw.replace(",", "")
        if _YEAR_TOKEN.match(value) or len(value.rstrip(".")) <= 2:
            continue
        if value not in out:
            out.append(value)
    return out


def _gold_position(answer: str, golds: list[str]) -> int:
    """답변이 나열한 값 중 정답이 **몇 번째**인가. 없으면 0."""
    numbers = _answer_numbers(answer)
    wanted = {_norm(g) for g in golds if _norm(g)}
    for i, value in enumerate(numbers, start=1):
        if _norm(value) in wanted:
            return i
    return 0


# ---------------------------------------------------------------------------
# 정답 형식이 세 가지다 (2026-08-30 실측으로 발견)
# ---------------------------------------------------------------------------
#
# `_answer_hit` 은 A형(단일 수치)만 처리한다. B·C형은 정답 문자열이 **사람이
# 읽는 요약**이라 답변에 그대로 등장할 수가 없다. suite_v1 22문항 중 16문항이
# 여기 해당해서, 맞는 답이 전부 오답 처리됐다.
#
#     S007  정답 "삼성전자 (삼성전자 22,764,764,160,000 vs 한미반도체 149,919,000,000)"
#           답변 "…삼성전자 22,764,764,160,000원, 한미반도체 149,919,000,000원.
#                 따라서 더 큰 기업은 삼성전자입니다"      -> 완전히 맞는데 오답 처리
#
#     S020  정답 "아니오 (해지 공시 없음)"
#           답변 "해당 사항 없음"                            -> 맞는데 오답 처리
#
# 기존 `answer_hit` 은 **정의를 바꾸지 않는다**(과거 수치와 비교해야 한다).
# `graded_hit` 을 따로 만들어 나란히 본다.

_COMPARE_GOLD = re.compile(r"^\s*(?P<winner>[^()]{1,40}?)\s*\((?P<body>.+)\)\s*$")
_YESNO_GOLD = re.compile(r"^\s*(?P<verdict>예|아니오)\s*[\(（]")

# 순서가 중요하다. 거부를 먼저 보지 않으면 "확인할 수 없습니다"가
# 부정("없")에 걸려 '아니오'로 잘못 읽힌다.
# 2026-08-31: 실측으로 놓치던 표현을 넣었다. 과거형("존재했습니다")·건수형
# ("2건입니다")·발견형("발견되지 않았습니다")이 전부 '불명' 이었다.
_NEG_PAT = re.compile(
    r"(존재하지\s*않|없습니다|없음|없었|아니오|아닙니다|해당\s*사항\s*없"
    r"|발견되지\s*않|확인되지\s*않았)")
_POS_PAT = re.compile(
    r"(존재합니다|존재한다|존재했|존재하며|존재하는|있습니다|있음|있었|네,|예,|맞습니다"
    r"|\d+\s*건(?:입니다|이\s*확인|을\s*확인|이\s*있))")
_WINNER_PAT = re.compile(r"[^.\n]*(?:더\s*큰|더\s*많|가장\s*큰|큽니다|많습니다|따라서)[^.\n]*")


def _verdict(answer: str) -> str:
    """답변이 내린 판정. 예 | 아니오 | 거부 | 불명.

    **가장 먼저 등장한 신호가 이긴다.** 순서를 고정해서 보면 이런 답변이
    잘못 읽힌다:

        "네, 존재합니다. 다만 일부 항목은 확인할 수 없습니다."
        -> 거부를 먼저 보면 '거부'가 된다. 실제로는 '예'다.

    open/mixed 프롬프트가 "확인되지 않은 항목은 '(확인되지 않음)'이라고
    적으세요"를 시키므로, 답변 뒤쪽에 확인 불가 문구가 붙는 건 정상이다.
    판정은 앞부분에서 내리라고 지시했으므로 앞 200자만 본다.
    """
    head = (answer or "")[:200]
    found = []
    for label, pat in (("아니오", _NEG_PAT), ("예", _POS_PAT)):
        m = pat.search(head)
        if m:
            found.append((m.start(), label))
    for marker in _REFUSAL:
        i = head.find(marker)
        if i >= 0:
            found.append((i, "거부"))
    if not found:
        return "거부" if _is_refusal(answer or "") else "불명"
    return min(found)[1]


def _stated_winner(answer: str, candidates: list[str]) -> str | None:
    """답변이 결론 문장에서 지목한 쪽. 못 찾으면 None.

    후보끼리 포함 관계면 **긴 쪽만 센다.** 정답 `JYP Ent (...)` 에서 후보를
    뽑으면 `JYP Ent` 와 `JYP` 가 함께 들어가는데, 둘 다 문장에 있으니
    `len(hits)==2` 가 되어 판정을 포기했다 — 완벽한 답변이 0점이 됐다.
    """
    for sentence in reversed(_WINNER_PAT.findall(answer or "")):
        hits = [c for c in candidates if c and c in sentence]
        maximal = [h for h in hits
                   if not any(h != other and h in other for other in hits)]
        if len(maximal) == 1:
            return maximal[0]
    return None


def grade_answer(answer: str, gold: str, golds: list[str],
                 required=None, any_groups=None) -> tuple[int, str]:
    """정답 형식을 판별해 채점한다. `(맞았나, 형식)` 을 돌려준다."""
    gold = (gold or "").strip()

    if any_groups:
        # D-2형 — 정답 후보가 여럿. 하나만 온전히 맞으면 정답이다.
        # required_all 이 함께 있으면 **그것도 후보 하나로 넣는다.** 예전엔
        # any_groups 가 있으면 required_all 을 아예 안 봤는데, 후보 중에
        # 토큰 1개짜리가 섞이면 "숫자 하나만 언급하면 만점"이 됐다.
        groups = list(any_groups)
        items0 = [t for t in (required or []) if str(t).strip()]
        if items0 and list(items0) not in [list(g) for g in groups]:
            groups.append(items0)
        rep = required_any_report(answer, groups)
        if rep["n"]:
            return int(rep["matched"] == rep["n"]), "required_any"

    items = [t for t in (required or []) if str(t).strip()]
    if items:
        # D형 — 서술형. 필수 항목이 **전부** 나와야 정답이다.
        # 부분 점수는 required_coverage 로 따로 본다.
        rep = required_report(answer, items)
        return int(rep["matched"] == rep["n"]), "required"

    m = _YESNO_GOLD.match(gold)
    if m:
        # C형 — 있나/없나. 거부는 '아니오'로 쳐주지 않는다.
        # S020 채점 기준이 "없다고 **정확히** 답하는가"이고, 별도 항목으로
        # "없는 사실을 지어내지 않는가"를 본다. 확인 불가와 부재는 다른 답이다.
        return int(_verdict(answer) == m.group("verdict")), "yes_no"

    m = _COMPARE_GOLD.match(gold)
    if m and " vs " in m.group("body"):
        # B형 — "승자 (A 값 vs B 값)"
        body = m.group("body")
        numbers = {_norm(x) for x in _NUM.findall(body)}
        answer_numbers = {_norm(x) for x in _numbers(answer)}
        if not numbers or not numbers <= answer_numbers:
            return 0, "compare"        # 양쪽 수치를 다 제시해야 한다
        winner = m.group("winner").strip()
        # 이름을 `split()[0]` 으로 자르면 공백 있는 회사명이 깨진다
        # (실측: gold `JYP Ent (...)` -> names ['JYP'] -> 완벽한 답도 0점).
        # 정답의 승자 문자열을 그대로 후보에 넣는다.
        names = [seg.strip().split(" ")[0] for seg in body.split(" vs ") if seg.strip()]
        names = list(dict.fromkeys([winner, *names]))
        stated = _stated_winner(_body_only(answer), names)
        if stated is None:
            # 예전에는 여기서 **정답 처리**했다. 그러면 승자를 아예 안 밝힌
            # 답변과 승자를 틀리게 쓴 답변이 모두 통과한다(실측 3종 재현).
            # 수치는 다 맞았으므로 오답이라고 단정하지도 않고, 별도 형식으로
            # 분리해 사람이 보게 한다.
            return 0, "compare_no_verdict"
        return int(stated == winner), "compare"

    return int(_answer_hit(answer, golds)), "value"


def _is_refusal(answer: str) -> bool:
    return any(m in (answer or "") for m in _REFUSAL)


def _label(answer_hit: bool, evidence_hit: bool, refusal: bool,
           *, gradeable: bool = True) -> str:
    if not gradeable:
        # 정답도 required_all 도 없는 문항. 맞았는지 틀렸는지 말할 수 없다.
        return "채점불가"
    if answer_hit:
        return "정답"
    if refusal:
        return "거부(검색실패)" if not evidence_hit else "거부(근거있음)"
    return "답변실패" if evidence_hit else "검색실패"


# --------------------------------------------------------------------------- 배선

def _load_bundle(artifacts: str, *, use_reranker: bool):
    from disclosure_rag.retrieval.index_bundle import load_bundle

    t0 = time.time()
    bundle = load_bundle(artifacts)
    if use_reranker:
        try:
            from disclosure_rag.retrieval.reranker import CrossEncoderReranker
            bundle.retriever.reranker = CrossEncoderReranker()
        except Exception as e:  # noqa: BLE001
            logger.warning("리랭커 적재 실패(%s) — 없이 진행", type(e).__name__)
    logger.info("인덱스 적재 %.0fs — 경로 %s / 융합 %s / 리랭커 %s",
                time.time() - t0, bundle.modes, bundle.retriever.fusion,
                "ON" if bundle.retriever.reranker else "OFF")
    return bundle


def _prepare_agent(bundle, corpus_root: str):
    from disclosure_rag.agent.fact_tools import build_fact_tools
    from disclosure_rag.agent.tools import build_all_tools
    from disclosure_rag.common.manifest_loader import load_manifest
    from disclosure_rag.common.unicode_utils import PathResolver
    from disclosure_rag.correction.correction_graph_builder import build_correction_index
    from disclosure_rag.entity.entity_extractor import EntityExtractor

    manifest = load_manifest(corpus_root)
    correction_index = build_correction_index(manifest, PathResolver(corpus_root))
    tools = build_all_tools(bundle.retriever, manifest, correction_index)
    if bundle.fact_store is not None:
        tools += build_fact_tools(bundle.fact_store)
    extractor = EntityExtractor(corpus_root=corpus_root,
                                metric_terms_path="config/metric_terms.txt")
    logger.info("도구 %d개: %s", len(tools), ", ".join(t.name for t in tools))
    return tools, extractor


def _prepare_v2(bundle, corpus_root: str, artifacts: str, thinking: str = "off"):
    """신 파이프라인 부품 조립. 인덱스는 이미 올라와 있는 것을 재사용한다."""
    from disclosure_rag.agent.ask_v2 import AskV2
    from disclosure_rag.agent.dual_channel import DualChannelRetriever
    from disclosure_rag.agent.field_schema import FieldSchema
    from disclosure_rag.agent.hcx_client import HCXClient
    from disclosure_rag.agent.query_plan import PlanValidator, RulePlanBuilder
    from disclosure_rag.common.manifest_loader import load_manifest
    from disclosure_rag.common.unicode_utils import PathResolver
    from disclosure_rag.correction.correction_graph_builder import build_correction_index
    from disclosure_rag.entity.entity_extractor import EntityExtractor
    from disclosure_rag.entity.entity_registry import EntityRegistry

    schema_path = Path("config/field_schema.json")
    schema = FieldSchema.load(schema_path) if schema_path.exists() else FieldSchema.empty()

    registry_path = Path(artifacts) / "registry" / "entities.json"
    registry = EntityRegistry.load(registry_path) if registry_path.exists() else None
    if registry is None:
        logger.warning("[V2] Entity Registry 없음(%s) — 범위 게이트를 건너뛴다", registry_path)

    manifest = load_manifest(corpus_root)
    corrections = build_correction_index(manifest, PathResolver(corpus_root))

    dual = DualChannelRetriever(
        bundle.retriever, bundle.fact_store,
        correction_index=corrections, manifest=manifest,
    )
    builder = RulePlanBuilder(
        schema=schema,
        extractor=EntityExtractor(corpus_root=corpus_root,
                                  metric_terms_path="config/metric_terms.txt"),
    )
    return AskV2(
        client=HCXClient(), dual_retriever=dual, plan_builder=builder,
        plan_validator=PlanValidator(registry=registry, schema=schema),
        registry=registry, parent_expander=bundle.parent_expander,
        thinking_policy=thinking,
    )


def _run_v2(ask, rows: list[dict]) -> list[dict]:
    """신 파이프라인으로 채점. v1 과 **같은 컬럼**을 낸다(직접 비교용)."""
    out = []
    for i, row in enumerate(rows, 1):
        golds = _gold_answers(row)
        gold_ids = _gold_report_ids(row)
        t = time.time()
        error = ""
        try:
            res = ask.run(row["query"])
            answer = res.answer
            cited = {c.report_id for c in res.citations}
            # 거부로 끝나면 Evidence Pack 이 없어 citations 가 비고, 그러면
            # 검색이 성공했는데도 '검색실패'로 라벨이 붙는다. 진단이 뒤집히므로
            # **검색이 무엇을 회수했는지**를 따로 본다.
            retrieved = {getattr(c, "report_id", None) for c, _s in (res.evidence or [])}
            retrieved.discard(None)
            stopped, hcx_calls, retries = res.stopped_at, res.hcx_calls, res.retries
            plan = res.plan
            validation = res.validation_result
            timing = dict(getattr(res, "timing_ms", None) or {})
        except Exception as e:  # noqa: BLE001
            logger.error("[%d] 실패 %s: %s", i, type(e).__name__, e)
            answer, cited, retrieved = "", set(), set()
            stopped, hcx_calls, retries, plan = "error", 0, 0, None
            validation = None
            timing = {}
            error = f"{type(e).__name__}: {e}"
        elapsed = time.time() - t

        hit = _answer_hit(answer, golds)
        required = [t for t in (row.get("required_all") or []) if str(t).strip()]
        any_groups = [g for g in (row.get("required_any") or []) if g]
        req = (required_any_report(answer, any_groups) if any_groups
               else required_report(answer, required))
        bonus = required_report(answer, row.get("bonus_any") or [])
        if any_groups or required:
            graded, gold_kind = grade_answer(answer, "", golds, required=required,
                                             any_groups=any_groups)
        elif golds:
            graded, gold_kind = grade_answer(answer, row.get("answer") or "", golds)
        else:
            # 정답 문장도 required_all 도 없다 — 채점할 기준 자체가 없는 문항.
            # 0 으로 두되 gradeable=0 이라 평균에서 빠진다.
            graded, gold_kind = 0, "ungradeable"
        gradeable = int(bool(golds) or bool(required) or bool(any_groups))
        numbers = _answer_numbers(answer)
        position = _gold_position(answer, golds)
        # 정확히 맞힘 = 정답이 **첫 값**이고, 나열한 값이 둘 이하.
        # 값 하나만 물었는데 다섯 개를 나열하면 사람 채점자는 감점한다.
        exact = int(bool(hit) and position == 1 and len(numbers) <= 2)
        # evidence_hit 은 '검색이 정답 문서를 회수했는가'다. 답변에 인용됐는지는
        # citation_hit 으로 따로 본다 — 둘이 갈리면 게이트가 막은 것이다.
        evidence_hit = bool(gold_ids & (cited | retrieved))
        citation_hit = bool(gold_ids & cited)
        refusal = _is_refusal(answer)
        # 서술형 채점 — 정답 문장이 있든 없든 항상 계산한다. 값이 맞았는지와
        # 요구 항목을 다 다뤘는지는 다른 질문이다.
        from disclosure_rag.agent.open_scoring import score_open_answer
        open_score = score_open_answer(
            answer,
            required_fields=getattr(plan, "expected_fields", []) or [],
            gold_doc_ids=list(gold_ids),
        ).to_dict()

        out.append({
            "id": row.get("id"), "query": row["query"], "company": row.get("company"),
            "doc_group": row.get("doc_group") or "?",
            "gold": golds[0] if golds else "", "n_gold": len(golds),
            # results.csv 는 사람이 읽는 용도라 줄인다. 재채점은 answers.jsonl 의
            # 원문으로 한다 — 잘린 답으로 채점하면 뒤쪽 항목을 놓친다.
            "answer": answer.replace("\n", " ")[:600],
            "answer_full": answer,
            "answer_hit": int(hit), "answer_hit_exact": exact,
            "graded_hit": graded, "gold_kind": gold_kind, "gradeable": gradeable,
            "required_n": req["n"], "required_matched": req["matched"],
            "required_coverage": req["coverage"],
            "required_missing": " / ".join(str(x) for x in req["missing"]),
            "bonus_n": bonus["n"], "bonus_matched": bonus["matched"],
            **{k: v for k, v in open_score.items() if k != "silent_fields"},
            "silent_fields": " / ".join(open_score["silent_fields"]),
            "n_answer_numbers": len(numbers), "gold_position": position,
            "evidence_hit": int(evidence_hit), "citation_hit": int(citation_hit),
            "refusal": int(refusal),
            "label": _label(bool(graded), evidence_hit, refusal, gradeable=bool(gradeable)),
            # v2 전용 진단
            "numbers_grounded": int(bool(validation and validation.numbers_grounded)),
            "has_citation": int(bool(validation and validation.has_citation)),
            "validation_passed": int(bool(validation and validation.passed)),
            "ungrounded": " / ".join(sorted(validation.ungrounded_numbers)[:5]) if validation else "",
            "stopped_at": stopped, "hcx_calls": hcx_calls, "retries": retries,
            "thinking": (res.thinking or {}).get("effort", "") if not error else "",
            "answer_mode": getattr(plan, "answer_mode", ""),
            "task": getattr(plan, "task", ""),
            "error": error, "elapsed_sec": round(elapsed, 2),
            "cited_ids": sorted(cited), "retrieved_ids": sorted(retrieved),
            # 지연 분해 — 어디가 느린지 추정 대신 측정값으로 본다(ms)
            **{f"ms_{name}": round(value, 1)
               for name, value in timing.items() if name != "searches"},
            "n_searches": int(timing.get("searches", 0)),
        })
        logger.info("[%d/%d] %-14s %-16s HCX%d %5.1fs  %s", i, len(rows),
                    out[-1]["label"], stopped, hcx_calls, elapsed, row["query"][:44])
    return out


# --------------------------------------------------------------------------- 실행

def _run_retrieval(bundle, rows: list[dict], k: int, *,
                   candidate_k: int = 50, rerank_top_n: int = 50) -> list[dict]:
    """candidate_k / rerank_top_n 을 노출하는 이유(2026-08-30):

    HybridRetriever 의 기본값은 candidate_k=50, rerank_top_n=50 이다. 즉 k 를
    10 -> 50 으로만 올리면 **같은 후보 50개를 더 많이 보여줄 뿐** 51등 이하는
    애초에 존재하지 않는다. "정답이 몇 등에 있나"를 깊게 진단하려면 후보 풀
    자체를 넓혀야 한다.
    """
    from disclosure_rag.experiments.metrics import (
        average_precision_at_k,
        first_relevant_rank,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
        reciprocal_rank,
    )

    out = []
    for i, row in enumerate(rows, 1):
        golds = _gold_answers(row)
        gold_ids = _gold_report_ids(row)
        t = time.time()
        hits = bundle.retriever.search(row["query"], k=k,
                                       candidate_k=candidate_k, rerank_top_n=rerank_top_n)
        elapsed = time.time() - t

        retrieved_ids = [c.report_id for c, _ in hits]
        evidence_hit = bool(gold_ids & set(retrieved_ids))
        # 근거만 보면 답할 수 있었는가 = 정답 문자열이 회수된 조각 안에 있는가
        ceiling = any(_norm(g) and _norm(g) in _norm(c.raw_text) for c, _ in hits for g in golds)

        # 순위 품질. evidence_hit 은 "가져왔는가"(0/1)만 보므로, 정답 문서를
        # 10등에 놓은 검색과 1등에 놓은 검색이 같은 점수를 받는다. 아래 지표들이
        # 그 차이를 드러낸다. gold ID 만 있으면 계산되므로 LLM 은 쓰지 않는다.
        rank = first_relevant_rank(retrieved_ids, gold_ids)

        out.append({
            "id": row.get("id"), "query": row["query"], "company": row.get("company"),
            "doc_group": row.get("doc_group") or "?",
            "gold": golds[0] if golds else "", "n_gold": len(golds),
            "n_gold_docs": len(gold_ids),
            "evidence_hit": int(evidence_hit), "answer_ceiling": int(ceiling),
            # context_recall = 회수된 gold 문서 / 전체 gold 문서.
            # evidence_hit(하나라도 걸리면 1)과 달리 부분 회수를 구분한다.
            "context_recall": round(recall_at_k(retrieved_ids, gold_ids, k), 4),
            # context_precision = top-k 조각 중 gold 문서에서 나온 비율(예산 효율)
            "context_precision": round(precision_at_k(retrieved_ids, gold_ids, k), 4),
            # context_ap = 순위 반영 정밀도. 앞자리에서 맞힐수록 높다
            "context_ap": round(average_precision_at_k(retrieved_ids, gold_ids, k), 4),
            "mrr": round(reciprocal_rank(retrieved_ids, gold_ids), 4),
            "ndcg_at_10": round(ndcg_at_k(retrieved_ids, gold_ids, 10), 4),
            # 못 찾으면 0. 평균내지 말 것 — 집계에서 '찾은 것만'의 중앙값을 쓴다
            "first_gold_rank": rank or 0,
            "label": "상한도달" if ceiling else ("근거만도달" if evidence_hit else "검색실패"),
            "elapsed_sec": round(elapsed, 3),
        })
        # 진행 상황을 남긴다 — 314문항이 20분 가까이 걸리는데 로그가 비어 있으면
        # 멈춘 건지 도는 건지 알 수가 없다(실측 불편).
        if i % 20 == 0 or i == len(rows):
            done = out[-20:]
            logger.info("[%d/%d] 최근 20건 상한도달 %d / 근거만 %d / 검색실패 %d (%.1fs/건)",
                        i, len(rows),
                        sum(1 for r in done if r["label"] == "상한도달"),
                        sum(1 for r in done if r["label"] == "근거만도달"),
                        sum(1 for r in done if r["label"] == "검색실패"),
                        sum(r["elapsed_sec"] for r in done) / len(done))
    return out


def _run_full(bundle, tools, extractor, rows: list[dict], *, max_iterations: int) -> list[dict]:
    from disclosure_rag.agent.ask import ask
    from disclosure_rag.agent.hcx_client import HCXClient

    client = HCXClient()
    out = []
    for i, row in enumerate(rows, 1):
        golds = _gold_answers(row)
        gold_ids = _gold_report_ids(row)
        t = time.time()
        error = ""
        try:
            res = ask(client, tools, row["query"],
                      entity_extractor=extractor, router=None, max_iterations=max_iterations)
            answer = res.answer
            cited = {c.report_id for c in res.evidence_pack.citations}
            validation = res.validation
            remediation = " | ".join(res.remediation)
            n_tool_calls = len(res.trace.tool_calls or [])
            iterations = res.trace.iterations
            n_nudges = len(getattr(res.trace, "nudges", []) or [])
        except Exception as e:  # noqa: BLE001
            logger.error("[%d] 실패 %s: %s", i, type(e).__name__, e)
            answer, cited, validation, remediation = "", set(), None, ""
            n_tool_calls = iterations = n_nudges = 0
            error = f"{type(e).__name__}: {e}"
        elapsed = time.time() - t

        hit = _answer_hit(answer, golds)
        evidence_hit = bool(gold_ids & cited)
        refusal = _is_refusal(answer)

        out.append({
            "id": row.get("id"), "query": row["query"], "company": row.get("company"),
            "doc_group": row.get("doc_group") or "?",
            "gold": golds[0] if golds else "", "n_gold": len(golds),
            "answer": answer.replace("\n", " ")[:600],
            "answer_hit": int(hit), "evidence_hit": int(evidence_hit),
            "citation_hit": int(bool(gold_ids & {r for r in gold_ids if r in answer})),
            "refusal": int(refusal),
            "label": _label(hit, evidence_hit, refusal),
            "numbers_grounded": int(bool(validation and validation.numbers_grounded)),
            "has_citation": int(bool(validation and validation.has_citation)),
            "validation_passed": int(bool(validation and validation.passed)),
            "warnings": " / ".join(validation.warnings) if validation else "",
            "remediation": remediation,
            "n_tool_calls": n_tool_calls, "iterations": iterations, "n_nudges": n_nudges,
            "error": error, "elapsed_sec": round(elapsed, 2),
        })
        logger.info("[%d/%d] %-14s %5.1fs  %s", i, len(rows), out[-1]["label"], elapsed,
                    row["query"][:48])
    return out


# --------------------------------------------------------------------------- 집계·저장

def _aggregate(rows: list[dict], mode: str) -> dict:
    n = len(rows) or 1
    labels: dict[str, int] = {}
    for r in rows:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
    lat = sorted(r["elapsed_sec"] for r in rows)
    metrics = {
        "mode": mode, "n": len(rows),
        "labels": labels,
        "label_rate": {k: round(v / n, 4) for k, v in labels.items()},
        "latency_mean_sec": round(sum(lat) / n, 2),
        "latency_p95_sec": lat[max(0, int(n * 0.95) - 1)] if lat else 0,
        "latency_max_sec": lat[-1] if lat else 0,
    }
    # 지연 분해 — 어느 단계가 병목인지 중앙값으로 본다. 평균은 한 문항이
    # 크게 튀면 통째로 왜곡되므로 중앙값을 같이 남긴다.
    stage_keys = sorted({k for r in rows for k in r if k.startswith("ms_")})
    if stage_keys:
        breakdown = {}
        for key in stage_keys:
            vals = sorted(r[key] for r in rows if isinstance(r.get(key), (int, float)))
            if not vals:
                continue
            breakdown[key[3:]] = {
                "median_ms": round(vals[len(vals) // 2], 1),
                "mean_ms": round(sum(vals) / len(vals), 1),
                "max_ms": round(vals[-1], 1),
                "n": len(vals),
            }
        metrics["latency_breakdown"] = breakdown

    for key in _RATE_KEYS:
        if rows and key in rows[0]:
            metrics[key] = round(sum(r[key] for r in rows) / n, 4)

    # 채점 기준이 있는 문항만 분모에 넣는다. 아래 세 줄이 없으면 정답이 비어 있는
    # 문항이 자동 0점이 되어 전체 점수를 끌어내린다.
    gradeable_rows = [r for r in rows if r.get("gradeable", 1)]
    for key in _GRADEABLE_KEYS:
        if rows and key in rows[0]:
            metrics[key] = (round(sum(r[key] for r in gradeable_rows) / len(gradeable_rows), 4)
                            if gradeable_rows else None)
            metrics[f"{key}_n"] = len(gradeable_rows)
    if rows:
        metrics["ungradeable_n"] = len(rows) - len(gradeable_rows)

    if rows and "answer_hit" in rows[0]:
        with_ev = [r for r in rows if r["evidence_hit"]]
        metrics["answer_hit_given_evidence"] = (
            round(sum(r["answer_hit"] for r in with_ev) / len(with_ev), 4) if with_ev else None
        )

    for key in _OPTIONAL_KEYS:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            metrics[key] = round(sum(vals) / len(vals), 4)
            metrics[f"{key}_n"] = len(vals)

    # 첫 정답 문서의 등수. 못 찾은 문항(0)을 평균에 넣으면 '1등에 가깝다'는
    # 거짓 신호가 되므로, 찾은 것만 모아 중앙값과 분포를 본다.
    if rows and "first_gold_rank" in rows[0]:
        found = sorted(r["first_gold_rank"] for r in rows if r["first_gold_rank"] > 0)
        metrics["first_gold_rank_found_n"] = len(found)
        metrics["first_gold_rank_median"] = found[len(found) // 2] if found else None
        metrics["first_gold_rank_p90"] = (
            found[max(0, int(len(found) * 0.9) - 1)] if found else None
        )
        metrics["first_gold_rank_hist"] = {
            "1": sum(1 for x in found if x == 1),
            "2-3": sum(1 for x in found if 2 <= x <= 3),
            "4-10": sum(1 for x in found if 4 <= x <= 10),
            "11+": sum(1 for x in found if x >= 11),
            "미발견": len(rows) - len(found),
        }

    # 공시유형별 분해. 정기공시와 주요사항보고서는 병목이 다르다(실측:
    # periodic 은 회수는 되는데 순위가 나쁘고, major 는 애초에 못 찾는다).
    # 전체 평균만 보면 이 둘이 상쇄돼 어느 쪽을 고쳐야 할지 알 수 없다.
    if rows and "doc_group" in rows[0]:
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(r.get("doc_group") or "?", []).append(r)
        by_group = {}
        for g, members in sorted(groups.items()):
            m = {"n": len(members)}
            for key in _RATE_KEYS:
                if key in members[0]:
                    m[key] = round(sum(x[key] for x in members) / len(members), 4)
            gm = [x for x in members if x.get("gradeable", 1)]
            for key in _GRADEABLE_KEYS:
                if key in members[0]:
                    m[key] = (round(sum(x[key] for x in gm) / len(gm), 4) if gm else None)
            m["gradeable_n"] = len(gm)
            found_g = sorted(x["first_gold_rank"] for x in members
                             if x.get("first_gold_rank", 0) > 0)
            if found_g:
                m["first_gold_rank_median"] = found_g[len(found_g) // 2]
            by_group[g] = m
        metrics["by_doc_group"] = by_group

    return metrics


def _write(out_dir: Path, config: dict, metrics: dict, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows and "answer_full" in rows[0]:
        # 답변 원문. results.csv 는 600자에서 자르므로 재채점은 이 파일로 한다.
        with (out_dir / "answers.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"id": r.get("id"), "query": r.get("query"),
                                    "answer": r.get("answer_full") or "",
                                    # 답변 본문의 인용 형식은 모델이 지키지 않을 때가
                                    # 많다("[EVIDENCE 1]"). 파이프라인이 실제로 붙인
                                    # 근거 ID 를 같이 남겨 채점이 본문 파싱에
                                    # 의존하지 않게 한다.
                                    "cited_ids": r.get("cited_ids") or [],
                                    "retrieved_ids": r.get("retrieved_ids") or []},
                                   ensure_ascii=False) + "\n")
    if rows:
        _SKIP = {"answer_full", "cited_ids", "retrieved_ids"}
        csv_rows = [{k: v for k, v in r.items() if k not in _SKIP} for r in rows]
        # 열 이름은 **모든 행의 합집합**이다. 첫 행 기준으로 잡으면, 조기
        # 종료한 문항이 1번으로 오는 순간 뒤 행의 ms_rerank 같은 열에서
        # DictWriter 가 ValueError 로 죽는다 — 측정 전체가 날아간다.
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in csv_rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with (out_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            w.writeheader()
            w.writerows(csv_rows)
    bad = [r for r in rows if r["label"] not in ("정답", "상한도달")]
    with (out_dir / "failure_cases.jsonl").open("w", encoding="utf-8") as f:
        for r in bad:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lines = [f"# 답변 채점 결과 ({metrics['mode']})", "",
             f"- 문항 {metrics['n']}건",
             f"- 지연 평균 {metrics['latency_mean_sec']}s / p95 {metrics['latency_p95_sec']}s / 최대 {metrics['latency_max_sec']}s",
             "", "## 분류", "", "| 분류 | 건수 | 비율 |", "|---|---:|---:|"]
    for k, v in sorted(metrics["labels"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} | {metrics['label_rate'][k]:.1%} |")
    lines += ["", "## 지표", "", "| 지표 | 값 |", "|---|---:|"]
    if metrics.get("ungradeable_n"):
        lines += ["", f"> 채점 기준(정답 또는 required_all)이 없는 문항 "
                      f"{metrics['ungradeable_n']}건은 graded_hit 분모에서 제외했다. "
                      f"분모 {metrics.get('graded_hit_n')}건.", ""]
    for k in ("graded_hit", "graded_hit_n", "required_coverage",
              "answer_hit", "answer_hit_exact", "answer_hit_given_evidence",
              "evidence_hit", "answer_ceiling",
              "refusal", "numbers_grounded", "has_citation", "validation_passed",
              "context_recall", "context_precision", "context_ap", "mrr", "ndcg_at_10",
              "field_coverage", "silent_omission_rate",
              "citation_recall", "citation_precision"):
        if metrics.get(k) is not None and k in metrics:
            lines.append(f"| {k} | {metrics[k]} |")

    if metrics.get("first_gold_rank_hist"):
        lines += ["", "## 첫 정답 문서 등수", "",
                  f"- 중앙값 {metrics.get('first_gold_rank_median')} / p90 {metrics.get('first_gold_rank_p90')}",
                  "", "| 등수 | 건수 |", "|---|---:|"]
        for band, cnt in metrics["first_gold_rank_hist"].items():
            lines.append(f"| {band} | {cnt} |")

    if metrics.get("by_doc_group"):
        cols = [c for c in (_GRADEABLE_KEYS + _RATE_KEYS)
                if c in next(iter(metrics["by_doc_group"].values()))]
        lines += ["", "## 공시유형별", "",
                  "| 유형 | n | " + " | ".join(cols) + " | 첫정답등수(중앙) |",
                  "|---" * (len(cols) + 3) + "|"]
        for g, m in metrics["by_doc_group"].items():
            cells = " | ".join(str(m.get(c, "")) for c in cols)
            lines.append(f"| {g} | {m['n']} | {cells} | {m.get('first_gold_rank_median', '')} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------ 재채점

_NUM_COLS = frozenset(
    _RATE_KEYS + _GRADEABLE_KEYS + _OPTIONAL_KEYS + (
        "elapsed_sec", "gradeable", "n_gold", "n_answer_numbers", "gold_position",
        "hcx_calls", "retries", "citation_hit", "evidence_hit", "refusal",
        "required_n", "required_matched", "bonus_n", "bonus_matched",
        "first_gold_rank", "n_required", "n_covered", "n_acknowledged", "n_silent",
    ))


def _coerce(col: str, raw: str):
    if col not in _NUM_COLS:
        return raw
    v = (raw or "").strip()
    if v in ("", "None", "null"):
        return None
    try:
        return int(v) if v.lstrip("-").isdigit() else float(v)
    except ValueError:
        return raw


def _rescore(src: Path, gold_rows: list[dict], out_dir: Path, mode: str) -> int:
    """이미 받아둔 답변을 다시 채점한다. HCX 0회, 파이프라인도 안 띄운다.

    채점 기준을 고칠 때마다 20분짜리 실행을 반복하지 않으려고 만들었다.
    답변 원문은 `answers.jsonl` 에서 읽는다 — `results.csv` 의 answer 는
    600자에서 잘려 있어 서술형 뒤쪽 항목을 놓친다.
    """
    csv_path = src / "results.csv"
    if not csv_path.exists():
        logger.error("results.csv 가 없다: %s", csv_path)
        return 2

    full: dict[str, str] = {}
    ans_path = src / "answers.jsonl"
    if ans_path.exists():
        for line in ans_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("id"):
                    full[str(d["id"])] = d.get("answer") or ""
                if d.get("query"):
                    full[d["query"]] = d.get("answer") or ""
        logger.info("답변 원문 %d건을 answers.jsonl 에서 읽었다", len(full))
    else:
        logger.warning("answers.jsonl 이 없다 — results.csv 의 600자 잘린 답변으로 "
                       "채점한다. 서술형 점수가 실제보다 낮게 나온다.")

    by_id = {str(r.get("id")): r for r in gold_rows if r.get("id")}
    by_query = {r["query"]: r for r in gold_rows if r.get("query")}

    with csv_path.open(encoding="utf-8-sig") as f:
        raw_rows = list(csv.DictReader(f))
    rows, missing_gold = [], 0
    for raw in raw_rows:
        r = {k: _coerce(k, v) for k, v in raw.items()}
        rid, query = str(r.get("id") or ""), r.get("query") or ""
        gold_row = by_id.get(rid) or by_query.get(query)
        if gold_row is None:
            missing_gold += 1
            rows.append(r)
            continue
        answer = full.get(rid) or full.get(query) or (r.get("answer") or "")
        golds = _gold_answers(gold_row)
        required = [t for t in (gold_row.get("required_all") or []) if str(t).strip()]
        any_groups = [g for g in (gold_row.get("required_any") or []) if g]
        req = (required_any_report(answer, any_groups) if any_groups
               else required_report(answer, required))
        bonus = required_report(answer, gold_row.get("bonus_any") or [])
        if any_groups or required:
            graded, kind = grade_answer(answer, "", golds, required=required,
                                        any_groups=any_groups)
        elif golds:
            graded, kind = grade_answer(answer, gold_row.get("answer") or "", golds)
        else:
            graded, kind = 0, "ungradeable"
        gradeable = int(bool(golds) or bool(required) or bool(any_groups))
        hit = _answer_hit(answer, golds)
        numbers = _answer_numbers(answer)
        position = _gold_position(answer, golds)
        refusal = _is_refusal(answer)
        r.update({
            "answer": answer.replace("\n", " ")[:600], "answer_full": answer,
            "gold": golds[0] if golds else "", "n_gold": len(golds),
            "answer_hit": int(hit),
            "answer_hit_exact": int(bool(hit) and position == 1 and len(numbers) <= 2),
            "graded_hit": graded, "gold_kind": kind, "gradeable": gradeable,
            "required_n": req["n"], "required_matched": req["matched"],
            "required_coverage": req["coverage"],
            "required_missing": " / ".join(str(x) for x in req["missing"]),
            "bonus_n": bonus["n"], "bonus_matched": bonus["matched"],
            "n_answer_numbers": len(numbers), "gold_position": position,
            "refusal": int(refusal),
            "label": _label(bool(graded), bool(r.get("evidence_hit")), refusal,
                            gradeable=bool(gradeable)),
        })
        rows.append(r)
    if missing_gold:
        logger.warning("정답셋에서 못 찾은 문항 %d건은 원본 값을 그대로 뒀다", missing_gold)

    metrics = _aggregate(rows, mode)
    _write(out_dir, {"rescored_from": str(src), "gold_rows": len(gold_rows)}, metrics, rows)
    logger.info("재채점 완료 -> %s/", out_dir)
    return 0


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/gold_passages.jsonl")
    ap.add_argument("--artifacts", default=os.environ.get("ARTIFACTS", "artifacts_v2"))
    ap.add_argument("--corpus", default=os.environ.get("CORPUS_ROOT", "corpus"))
    ap.add_argument("--out", default="")
    ap.add_argument("--mode", choices=("retrieval", "full"), default="retrieval")
    ap.add_argument("--thinking", choices=("off", "auto", "on"), default="off",
                    help="HCX-007 reasoning 사용 정책. off=전부 끔(기본, 현 운영 상태), "
                         "auto=correction_diff/compare 만 켬, on=전부 켬. "
                         "A/B 는 같은 정답셋으로 두 번 돌려 비교한다.")
    ap.add_argument("--pipeline", choices=("v1", "v2"), default="v1",
                    help="full 모드에서 쓸 파이프라인. v1=기존 에이전트 루프, "
                         "v2=결정론적 신 파이프라인. v1 은 비교 기준이므로 그대로 둔다.")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만 (빠른 확인용)")
    ap.add_argument("--sample", type=int, default=0,
                    help="doc_group 비율을 유지한 층화 표본 N개. full 모드 비용을 줄일 때 쓴다.")
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--candidate-k", type=int, default=50,
                    help="검색기별 1차 후보 수. 이걸 안 올리면 --k 를 키워도 "
                         "51등 이하는 볼 수 없다(진단용으로 200 권장).")
    ap.add_argument("--rerank-top-n", type=int, default=0,
                    help="리랭커에 넣을 후보 수. 0 이면 candidate-k 를 따른다.")
    ap.add_argument("--max-iterations", type=int, default=6)
    ap.add_argument("--no-reranker", action="store_true")
    ap.add_argument("--yes", action="store_true", help="full 모드 대량 실행 승인")
    ap.add_argument("--rescore", default="",
                    help="이미 나온 결과 폴더를 다시 채점한다(HCX 0회). "
                         "예: --rescore results/v2_off4 --out results/v2_off4_regrade")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    rows = [json.loads(line) for line in Path(args.gold).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("query")]
    # 2026-08-30: 정답 문장이 비어 있다고 버리면 안 된다. suite_v1 의 그런
    # 16문항이 **전부 open 형**이라, 지금까지 서술형 답변을 한 번도 채점하지
    # 못했다(전체의 42%). 정답 문장 없이도 항목 커버리지·근거 정확성은 잴 수
    # 있다(open_scoring 모듈). v2 경로에서만 살린다 — v1 은 비교 기준이라
    # 동작을 바꾸지 않는다.
    # required_all 이 있으면 정답 문장이 없어도 채점된다(서술형 D형).
    unanswered = [r for r in rows
                  if not _gold_answers(r) and not r.get("required_all")
                  and not r.get("required_any")]
    if unanswered and not (args.mode == "full" and args.pipeline == "v2"):
        logger.warning("정답이 비어 있는 %d문항은 제외한다(v2 full 모드에서만 채점)",
                       len(unanswered))
        # 판단 조건과 같은 기준으로 걸러야 한다. 예전엔 `_gold_answers` 만 봐서
        # required_all/required_any 로 채점 가능한 문항까지 버렸다 —
        # suite_v2 296문항 중 87문항이 조용히 사라졌다(2026-08-31 발견).
        rows = [r for r in rows
                if _gold_answers(r) or r.get("required_all") or r.get("required_any")]
    elif unanswered:
        logger.info("정답 문장이 없는 %d문항은 서술형 기준으로 채점한다", len(unanswered))
    if args.sample and args.sample < len(rows):
        # 앞에서 N개를 자르면 파일이 회사/유형 순으로 정렬돼 있어 표본이
        # 한쪽으로 쏠린다. doc_group 비율을 유지한 층화 표본을 쓴다.
        import collections as _c
        import random as _r
        rng = _r.Random(args.seed)
        by_group: dict[str, list] = _c.defaultdict(list)
        for r in rows:
            by_group[r.get("doc_group") or "?"].append(r)
        picked = []
        for group, members in sorted(by_group.items()):
            k = max(1, round(len(members) / len(rows) * args.sample))
            picked += rng.sample(members, min(k, len(members)))
        rng.shuffle(picked)
        rows = picked[: args.sample]
        logger.info("층화 표본 %d문항 (doc_group 비율 유지, seed=%d)", len(rows), args.seed)
    elif args.limit:
        rows = rows[: args.limit]
    logger.info("채점 대상 %d문항 (%s)", len(rows), args.gold)

    if args.mode == "full" and len(rows) > 80 and not args.yes:
        est_sec = len(rows) * 15
        print(f"\n[중단] full 모드 {len(rows)}문항 = HCX 약 {len(rows) * 3}회 호출, "
              f"예상 {est_sec // 60}분 {est_sec % 60}초. 크레딧을 씁니다.\n"
              f"       진행하려면 --yes 를, 일부만 보려면 --limit 를 붙이세요.\n")
        return 2

    out_dir = Path(args.out or f"results/answers_{args.mode}")

    if args.rescore:
        # 인덱스도 모델도 안 띄운다. 채점 기준만 다시 적용한다.
        return _rescore(Path(args.rescore), rows, out_dir, args.mode)

    bundle = _load_bundle(args.artifacts, use_reranker=not args.no_reranker)

    if args.mode == "retrieval":
        results = _run_retrieval(bundle, rows, args.k,
                                 candidate_k=args.candidate_k,
                                 rerank_top_n=args.rerank_top_n or args.candidate_k)
    elif args.pipeline == "v2":
        results = _run_v2(_prepare_v2(bundle, args.corpus, args.artifacts,
                                      thinking=args.thinking), rows)
    else:
        tools, extractor = _prepare_agent(bundle, args.corpus)
        results = _run_full(bundle, tools, extractor, rows, max_iterations=args.max_iterations)

    metrics = _aggregate(results, args.mode)
    config = {"gold": args.gold, "artifacts": args.artifacts, "mode": args.mode,
              "pipeline": args.pipeline, "thinking": args.thinking,
              "k": args.k, "candidate_k": args.candidate_k,
              "rerank_top_n": args.rerank_top_n or args.candidate_k,
              "limit": args.limit, "sample": args.sample, "seed": args.seed,
              "reranker": not args.no_reranker, "max_iterations": args.max_iterations}
    _write(out_dir, config, metrics, results)

    print("\n" + json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n저장: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
