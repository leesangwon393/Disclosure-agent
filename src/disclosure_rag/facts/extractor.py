"""폼 문서에서 정형 사실(fact) 행을 뽑는다.

왜 필요한가
-----------
지금은 "삼성전자 계약금액 얼마야?" 를 이렇게 답한다.

    질문 -> 벡터/BM25 검색 -> 표가 들어 있는 조각을 찾음 -> HCX 가 표를 읽고 숫자를 뽑음

세 군데(검색·표 파싱·LLM 읽기)에서 틀릴 수 있고, 특히 **dense 벡터는 22조와 30조를
구분하지 못한다**(숫자는 임베딩에서 거의 의미를 못 가진다). EACL 2026 벤치마크는
금융문서 검색 실패의 73%가 표 구조 불일치라고 보고한다.

그런데 우리 코퍼스는 이미 절반 이상이 **서식이 고정된 폼**이다.

    exchange 1,469 + holding 1,083 + major 598 = 3,150건 (전체의 75%)

폼 문서의 `KeyValueNode` 는 (항목, 값) 이 그대로 (질문 대상, 정답) 이다.
이걸 행으로 뽑아 두면:

    "계약금액 얼마야"          -> 검색 없이 SELECT
    "매출액 대비 몇 %"         -> 계산 도구에 **정확한** 숫자 투입
    "계약금액 1조 넘는 계약"    -> 검색으로는 불가능. WHERE value_num > 1e12

표를 임베딩에서 빼는 것이 아니다. 표 하나가 **두 곳으로 간다** —
① 검색용 조각(서술·맥락 질의용)  ② facts 행(숫자 질의용). 버리는 것이 없다.

근거 표시
---------
모든 fact 에 출처 `chunk_id` 가 붙는다. 숫자를 조회한 뒤 그 숫자가 나온 원문 조각을
그대로 근거로 제시할 수 있다(대회 평가 항목 「근거 공시 표시」).

범위
----
periodic(재무제표)은 계정과목 정규화·연결/별도 구분·단위 처리가 훨씬 어려워 후순위다.
기본값은 exchange/major/holding 만 대상으로 한다.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, MutableMapping

from pydantic import BaseModel, Field

from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.common.doc_tree import KeyValueNode, ParsedDocument, SectionNode
from disclosure_rag.common.manifest_loader import ManifestRow
from disclosure_rag.correction.correction_graph_builder import CorrectionRecord

FORM_GROUPS = ("exchange", "major", "holding")

# --- 값 정규화 패턴 ---
_NUM = re.compile(r"^-?[\d,]+(?:\.\d+)?$")
_NUM_IN_TEXT = re.compile(r"-?[\d,]+(?:\.\d+)?")
_SCALE = {"조": 1_000_000_000_000, "억": 100_000_000, "만": 10_000, "천": 1_000, "백만": 1_000_000}
_SCALE_RE = re.compile(r"^(-?[\d,]+(?:\.\d+)?)\s*(백만|조|억|만|천)\s*원?$")
_DATE_PATTERNS = [
    (re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"), lambda m: f"{m[1]}{m[2]}{m[3]}"),
    (re.compile(r"^(\d{4})[./](\d{1,2})[./](\d{1,2})$"), lambda m: f"{m[1]}{int(m[2]):02d}{int(m[3]):02d}"),
    (re.compile(r"^(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일$"), lambda m: f"{m[1]}{int(m[2]):02d}{int(m[3]):02d}"),
    (re.compile(r"^(\d{8})$"), lambda m: m[1]),
]
# 항목명 끝의 단위 표기: "계약금액(원)", "매출액대비(%)"
_KEY_UNIT = re.compile(r"[（(]\s*([^)）]{1,12})\s*[)）]\s*$")
# 항목 번호: "1. ", "가. ", "- ", "3) "
_KEY_PREFIX = re.compile(r"^\s*(?:[0-9]{1,2}|[가-힣])\s*[.)]\s*|^\s*[-·※]\s*")
_EMPTY_VALUES = {"", "-", "–", "해당사항 없음", "해당사항없음", "해당없음", "N/A", "없음", "미해당"}

# --- 노이즈 필터 ---
# 공시 서식의 머리말/서명란은 KV 로 파싱되지만 사실(fact)이 아니다.
# 실측(120문서 표본)에서 상위 항목에 "금융위원회 / 한국거래소 귀중", "회 사 명 :",
# "대 표 이 사 :", "(직 책) 재무 담당" 같은 서식 골격이 그대로 올라왔다.
_NOISE_KEY_PAT = re.compile(
    r"귀\s*중|금융위원회|한국거래소|^회\s*사\s*명|^대\s*표\s*이\s*사|^본\s*점\s*소\s*재\s*지"
    r"|^\(?\s*직\s*책|전화번호|작성책임자|담당자|^성\s*명|^주\s*소|^전\s*자\s*우\s*편"
    r"|^홈페이지|^결\s*재|^문서번호|^제\s*출\s*인"
)
# 항목명이 사실상 값인 경우(사업자등록번호가 key 로 올라오는 등)
_KEY_IS_VALUE = re.compile(r"^[\d,.\-/()%\s]+$")
_HANGUL_OR_ALNUM = re.compile(r"[가-힣A-Za-z0-9]")

# --- periodic 전용 pair-based 필터 ---
# 단어 하나만 보고 지우지 않는다. 아래 라벨이 key에 있더라도 값이 숫자/날짜면
# 정상 fact로 남긴다. 특히 gold가 실제로 묻는 "기초"·"기타"·"합계"가 그렇다.
_PERIODIC_HEADER_KEY = re.compile(
    r"^(?:"
    r"(?:(?:당|전|전전|직전|금)(?:기|분기|반기|연도|사업연도|회계연도)(?:초|말|누적)?)"
    r"|(?:당|전)?(?:3|6|9|12)개월|누적|제?\d+기"
    r"|구분|항목|내역|내용|분류|단위|비고|주석|합계|소계|계|기타|기초|기말"
    r")$"
)
# 기존 _KEY_IS_VALUE가 이미 순수 숫자 key를 제거한다. 여기서는 그 정규식이
# 놓치는 회계식 음수(△298,342)가 값 열에서 key 열로 밀린 경우를 잡는다.
_PERIODIC_ACCOUNTING_VALUE_KEY = re.compile(
    r"^[△▲▽]\s*[+-]?[\d,]+(?:\.\d+)?\s*(?:원|천원|백만원|억원|조원|%|주)?$"
)
_ACCOUNTING_WRAPPED = re.compile(r"^[（(](.*)[）)]$")


class Fact(BaseModel):
    """정형 사실 1건. `chunk_id` 로 원문 조각까지 되짚을 수 있다."""

    doc_id: str
    chunk_id: str | None = None
    company: str | None = None
    corp_code: str | None = None
    doc_group: str | None = None
    doc_subtype: str | None = None
    report_name: str | None = None
    filing_date: str | None = None          # YYYYMMDD
    period: str | None = None

    is_correction: bool = False
    is_latest: bool | None = None
    correction_group_id: str | None = None

    group_label: str | None = None          # 표에서 rowspan 으로 묶인 상위 라벨
    key: str                                # 원문 항목명
    key_norm: str                           # 번호·단위 제거한 정규화 항목명 (조회 키)

    value_text: str                         # 원문 값 문자열 (근거 표시용)
    value_num: float | None = None          # 숫자로 해석되면 채움 -> WHERE/ORDER BY 가능
    value_unit: str | None = None           # 원 / % / 주 ...
    value_date: str | None = None           # YYYYMMDD 로 해석되면 채움

    field_code: str | None = None           # DART TE[ACODE]
    unit_code: str | None = None            # DART TU[AUNIT]
    unit_value: str | None = None           # DART TU[AUNITVALUE] — 이미 정규화된 값
    section_path: list[str] = Field(default_factory=list)

    # 이 수치의 주인. 보통은 보고서를 낸 회사(company)지만, 「주주에 관한 사항」의
    # '최대주주 및 특수관계인 현황' 표처럼 **다른 법인의 재무현황**을 싣는 표가
    # 있다. 그 표에는 주인 이름이 같은 표 안에 적혀 있으므로 추출할 때 붙인다.
    # None 이면 company 가 주인이다.
    value_owner: str | None = None


# 표 안에 이 항목이 있으면, 그 표의 수치는 **그 이름의 법인 것**이다.
# (「VII. 주주에 관한 사항」의 최대주주의 개요 표)
#
# '성명' 은 일부러 넣지 않았다. 「VIII. 임원 및 직원 등에 관한 사항」의 임원
# 보수 표에도 성명이 있어서, 넣으면 임원 보수가 '남의 값'으로 분류되어
# "삼성전자 임원 보수는?" 이 막힌다. 최대주주가 개인이면 애초에 재무현황 표를
# 내지 않으므로(개인은 재무제표 제출 대상이 아니다) 실익도 없다.
# 우리 코퍼스 70개사에는 개인 최대주주 사례가 0건이다(2026-09-01 확인).
OWNER_NAME_KEYS = ("법인 또는 단체의 명칭", "법인또는단체의명칭")

# 「최대주주의 개요」 뒤에 이름 없이 따라붙는 요약재무정보 표를 알아보는 표지.
# 이 중 둘 이상이 한 표에 있으면 재무제표 표로 본다.
_FINANCIAL_KEYS = (
    "자산총계", "부채총계", "자본총계", "매출액", "영업이익", "당기순이익",
    "유동자산", "비유동자산", "유동부채", "비유동부채", "자본금", "이익잉여금",
    "부채와자본총계", "영업수익",
)


def _table_owner(kv) -> str | None:
    """이 표의 수치가 누구 것인지 표 안에서 찾는다.

    2026-09-01 실측: 이걸 안 해서 국민연금공단의 자산총계가 KB금융·신한지주·
    하나금융지주·POSCO홀딩스의 값으로 저장돼 있었다(네 곳 다 464,418). 값도
    문서도 맞았고 **주인만 없었다.**
    """
    for pair in kv.pairs:
        key = (pair.key or "").strip()
        if any(marker in key for marker in OWNER_NAME_KEYS):
            value = (pair.value or "").strip()
            if value and value not in _EMPTY_VALUES:
                return value
    return None


def _looks_like_financial_statement(kv) -> bool:
    """이름은 없지만 재무제표로 보이는 표인가.

    실제 문서(SK하이닉스 반기보고서)를 열어 확인한 형태:

        표1  법인 또는 단체의 명칭 = SK스퀘어 주식회사   ← 이름 있음
        표2  법인 또는 단체의 명칭 = SK주식회사         ← 이름 있음
        표3  구 분 = 제33기 반기 / [유동자산] ... 자산총계 199,360,274
        표4  구 분 = 제33기 반기 / ... (별도재무제표)

    표3·표4 는 표2(SK주식회사)의 상세 재무제표인데 **이름을 다시 적지 않는다.**
    사람이 읽으면 "위에서 말한 그 회사"인 게 당연하기 때문이다. 표3 의
    자산총계가 표2 와 정확히 같은 값인 것으로 확인했다.

    조건을 재무 항목 2개 이상으로 둔 이유: 임원 보수·주식 소유현황처럼 이름
    없는 표는 많은데, 그런 표에는 자산총계·부채총계가 같이 나오지 않는다.
    """
    hits = 0
    seen: set[str] = set()
    for pair in kv.pairs:
        key = (pair.key or "").strip().replace(" ", "").strip("[]")
        for marker in _FINANCIAL_KEYS:
            if marker in key and marker not in seen:
                seen.add(marker)
                hits += 1
                break
        if hits >= 2:
            return True
    return False


def normalize_key(key: str) -> tuple[str, str | None]:
    """항목명에서 번호와 단위를 떼어낸다. -> (key_norm, unit)

    "2. 계약금액(원)" -> ("계약금액", "원")
    "매출액대비(%)"   -> ("매출액대비", "%")
    """
    k = _KEY_PREFIX.sub("", key or "").strip()
    unit = None
    m = _KEY_UNIT.search(k)
    if m:
        unit = m.group(1).strip()
        k = k[: m.start()].strip()
    k = k.rstrip(":： ").strip()
    # 한글 서식은 자간을 띄어 쓰는 경우가 많다("최근 매출액" vs "최근매출액",
    # "회 사 명"). 공백을 전부 제거해야 같은 항목이 하나로 모인다.
    k = re.sub(r"\s+", "", k)
    return k, unit


# 한국 공시의 회계 표기 — 실측으로 확인한 것만 넣었다(2026-09-01, 청크 15만 개).
#
#   △59,917        음수. 표에 "[△는 부(-)의 값임]" 범례가 함께 온다 (690건+)
#   1%p            퍼센트포인트. `%` 와 **다른 단위**다 (164건, 전부 해석 실패)
#   63조 7,454억원  조·억 혼용. `_SCALE_RE` 는 단일 단위만 읽었다 (12,441건)
#   69,406주 (주1)  숫자 뒤 주석 표시 (938건)
#   ０             전각 숫자 (18건)
#
# 괄호 음수 `(4,935,379)` 는 이미 처리되고 있다(90,084건 음수 저장) — 손대지 않는다.
_MINUS_MARKS = ("△", "▲", "▽", "▼")
_FULLWIDTH_DIGITS = {ord("０") + i: ord("0") + i for i in range(10)}
_FOOTNOTE_TAIL = re.compile(r"\s*(?:\(\s*주\s*\d*\s*\)|주\s*\d+\s*\)|\*+|＊+)\s*$")
_PERCENT_POINT = re.compile(r"^(-?[\d,]+(?:\.\d+)?)\s*(?:%\s*[pP]|퍼센트\s*포인트|%포인트)$")
# "63조 7,454억원", "1조 2,345억 6,789만원" 처럼 큰 단위부터 이어 붙인 금액
_COMPOUND_UNIT = re.compile(r"([\d,]+(?:\.\d+)?)\s*(조|억|만|천)")
_COMPOUND_OK = re.compile(r"^-?(?:[\d,]+(?:\.\d+)?\s*(?:조|억|만|천)\s*)+[\d,]*\s*원?$")


def _strip_accounting_marks(text: str) -> tuple[str, bool]:
    """세모 부호와 주석 표시를 떼어내고 음수 여부를 돌려준다."""
    t = text.strip()
    negative = False
    while t[:1] in _MINUS_MARKS:
        negative = True
        t = t[1:].strip()
    t = _FOOTNOTE_TAIL.sub("", t).strip()
    return t, negative


def _parse_compound_amount(text: str) -> float | None:
    """"63조 7,454억원" -> 6_374_540_000_000. 단일 단위는 여기 안 온다."""
    if not _COMPOUND_OK.match(text):
        return None
    matches = list(_COMPOUND_UNIT.finditer(text))
    if len(matches) < 2:                     # 단일 단위는 기존 `_SCALE_RE` 가 본다
        return None
    total = 0.0
    for m in matches:
        try:
            total += float(m.group(1).replace(",", "")) * _SCALE[m.group(2)]
        except (ValueError, KeyError):
            return None
    # 단위 없이 남은 꼬리("… 6,789원")
    tail = text[matches[-1].end():].strip().rstrip("원").strip()
    if tail:
        try:
            total += float(tail.replace(",", ""))
        except ValueError:
            return None
    return total


def parse_value(value: str, *, key_unit: str | None = None, unit_value: str | None = None
                ) -> tuple[float | None, str | None, str | None]:
    """값 문자열을 (숫자, 단위, 날짜) 로 해석한다. 해석 실패는 None — 지어내지 않는다."""
    # 전각 숫자(０１２)만 반각으로 바꾼다.
    # NFKC 를 통째로 쓰면 안 된다 — `㈜LS` 가 `(주)LS` 로 바뀌어 회사 이름이
    # 훼손된다. 최대주주 이름(value_owner)이 그 값으로 저장되므로 치명적이다.
    v = unicodedata.normalize("NFC", value or "").strip().translate(_FULLWIDTH_DIGITS)
    if not v or v in _EMPTY_VALUES:
        return None, key_unit, None

    # 퍼센트포인트는 `%` 와 다른 단위다. 5%->7% 는 "2%p 상승" 이지 "2% 상승" 이 아니다.
    m = _PERCENT_POINT.match(v)
    if m:
        try:
            return float(m.group(1).replace(",", "")), "%p", None
        except ValueError:
            return None, key_unit, None

    # 세모 부호·주석 표시를 떼고 다시 본다 (서식공시에도 적용된다)
    stripped, negative = _strip_accounting_marks(v)
    if stripped != v:
        num, unit, date = parse_value(stripped, key_unit=key_unit, unit_value=unit_value)
        if num is not None:
            return (-num if negative else num), unit, date
        if date is not None:
            return None, unit, date
        v = stripped

    compound = _parse_compound_amount(v)
    if compound is not None:
        return compound, key_unit or "원", None

    # DART 가 이미 정규화해 준 값이 있으면 날짜 판정에 먼저 쓴다 (AUNITVALUE)
    for cand in (unit_value, v):
        if not cand:
            continue
        for pat, fmt in _DATE_PATTERNS:
            m = pat.match(cand.strip())
            if m:
                return None, key_unit, fmt(m)

    if _NUM.match(v):
        try:
            return float(v.replace(",", "")), key_unit, None
        except ValueError:
            return None, key_unit, None

    m = _SCALE_RE.match(v)
    if m:
        try:
            return float(m.group(1).replace(",", "")) * _SCALE[m.group(2)], key_unit or "원", None
        except (ValueError, KeyError):
            pass

    # "22,764,764,160,000원" 처럼 단위가 붙은 형태
    if v.endswith("원") or v.endswith("%") or v.endswith("주"):
        body, suffix = v[:-1].strip(), v[-1]
        if _NUM.match(body):
            try:
                return float(body.replace(",", "")), key_unit or suffix, None
            except ValueError:
                pass
    return None, key_unit, None


def parse_periodic_value(
    value: str, *, key_unit: str | None = None, unit_value: str | None = None,
) -> tuple[float | None, str | None, str | None]:
    """periodic 표의 회계식 음수까지 해석한다. 폼 공시 parse_value는 바꾸지 않는다.

    ``(1,234)``, ``△1,234``, ``(1,234천원)``은 재무제표에서 음수 표기다.
    이 값을 먼저 숫자로 인식해야 ``합계`` 같은 정상 항목을 헤더 잡음으로
    오판하지 않고, numeric 비율도 실제 데이터에 가깝게 측정할 수 있다.
    """
    parsed = parse_value(value, key_unit=key_unit, unit_value=unit_value)
    if parsed[0] is not None or parsed[2] is not None:
        return parsed

    text = (value or "").strip()
    negative = False
    wrapped = _ACCOUNTING_WRAPPED.fullmatch(text)
    if wrapped:
        text = wrapped.group(1).strip()
        negative = True
    if text.startswith(("△", "▲", "▽")):
        text = text[1:].strip()
        negative = True
    if text.startswith("+"):
        text = text[1:].strip()

    if text == (value or "").strip():
        return parsed
    num, unit, date = parse_value(text, key_unit=key_unit, unit_value=unit_value)
    if num is not None and negative:
        num = -abs(num)
    return num, unit, date


def periodic_noise_reason(
    key_norm: str, raw_value: str, *, value_num: float | None, value_date: str | None,
) -> str | None:
    """periodic의 (항목명, 값) 쌍이 확실한 표 방향 잡음이면 사유를 반환한다."""
    if _PERIODIC_ACCOUNTING_VALUE_KEY.fullmatch(key_norm or ""):
        return "numeric_key"
    if (_PERIODIC_HEADER_KEY.fullmatch(key_norm or "")
            and value_num is None and value_date is None):
        return "header_pair"
    return None


def is_meaningful_key(raw_key: str, key_norm: str) -> bool:
    """서식 골격(머리말·서명란)과 '값이 항목명 자리에 온 것'을 걸러낸다."""
    if len(key_norm) < 2 or len(key_norm) > 40:
        return False
    if _KEY_IS_VALUE.match(key_norm):          # "104-81-26688" 같은 것
        return False
    if len(_HANGUL_OR_ALNUM.findall(key_norm)) < 2:
        return False
    if _NOISE_KEY_PAT.search(raw_key) or _NOISE_KEY_PAT.search(key_norm):
        return False
    return True


def _iter_kv(sections: Iterable[SectionNode]):
    for sec in sections:
        for child in sec.children:
            if isinstance(child, SectionNode):
                yield from _iter_kv([child])
            elif isinstance(child, KeyValueNode):
                yield sec, child


def _iter_nodes(sections: Iterable[SectionNode]):
    """절 안의 자식들을 **문서에 적힌 순서대로** 돌려준다.

    `_iter_kv` 는 키-값 표만 준다. 그런데 최대주주 이름은 키-값 표가 아니라
    **일반 표의 '명 칭' 열**에 있는 경우가 있다(한화에어로스페이스 등 66개
    문서). 순서가 필요해서 따로 만들었다 — 이름이 먼저 나오고 재무현황이
    뒤에 오기 때문이다.
    """
    for sec in sections:
        for child in sec.children:
            if isinstance(child, SectionNode):
                yield from _iter_nodes([child])
            else:
                yield sec, child


# 최대주주 이름이 실린 일반 표를 알아보는 표지.
_OWNER_TABLE_HINTS = ("최대주주", "법인 기본정보", "법인기본정보")
# 그 표에서 이름이 들어 있는 열의 머리글.
_OWNER_COLUMN_HEADERS = ("명칭", "명 칭", "법인명", "상호", "회사명", "단체명")


def _owner_from_table(node) -> str | None:
    """일반 표에서 최대주주 이름을 꺼낸다.

    실제 형태(한화에어로스페이스 분기보고서):

        title_hint = "나. 최대주주의 기본정보(1) 법인 기본정보"
        머리글      명 칭 | 출자자수(명) | 대표이사 ... | 최대주주 ...
        본문        (주)한화 | 74,702 | 김동관 ...

    바로 뒤에 "(2) 최대주주(법인 또는 단체)의 최근 결산기 재무현황" 표가
    이름 없이 붙는다. 이 표를 안 읽으면 그 재무현황이 한화에어로스페이스
    자신의 값으로 저장된다.
    """
    hint = (getattr(node, "title_hint", None) or "")
    rows = getattr(node, "rows", None)
    if not rows or not any(marker in hint for marker in _OWNER_TABLE_HINTS):
        return None
    # 이름 열의 위치를 머리글에서 찾는다.
    col = None
    for row in rows[:3]:
        for cell in row:
            text = (getattr(cell, "text", "") or "").replace(" ", "")
            if text in (h.replace(" ", "") for h in _OWNER_COLUMN_HEADERS):
                col = getattr(cell, "col", None)
                break
        if col is not None:
            break
    if col is None:
        return None
    for row in rows:
        for cell in row:
            if getattr(cell, "col", None) != col or getattr(cell, "is_header", False):
                continue
            value = (getattr(cell, "text", "") or "").strip()
            if value and value not in _EMPTY_VALUES and value != "-":
                return value
    return None


def extract_facts(
    parsed: ParsedDocument, row: ManifestRow, correction: CorrectionRecord,
    *, filter_stats: MutableMapping[str, int] | None = None,
) -> list[Fact]:
    """ParsedDocument 하나에서 fact 행들을 뽑는다 (chunk_id 는 아직 비어 있음)."""
    facts: list[Fact] = []
    # 절이 바뀌면 초기화한다. 「주주에 관한 사항」에서 본 이름이 「임원 및
    # 직원」까지 따라가면 안 된다.
    carried_owner: str | None = None
    carried_section: tuple[str, ...] | None = None

    for section, node in _iter_nodes(parsed.sections):
        section_key = tuple(section.path)
        if section_key != carried_section:
            carried_owner, carried_section = None, section_key

        if not isinstance(node, KeyValueNode):
            # 키-값 표가 아니면 값을 뽑지 않는다. 다만 최대주주 이름이 실린
            # 일반 표면 이름만 기억해 둔다 — 뒤에 이름 없는 재무현황이 온다.
            found = _owner_from_table(node)
            if found:
                carried_owner = found
            continue
        kv = node

        # 표 하나를 통째로 보고 주인을 먼저 정한다. 행 단위로는 알 수 없다 —
        # 주인 이름은 표의 다른 행에 적혀 있기 때문이다.
        table_owner = _table_owner(kv)
        if table_owner:
            carried_owner = table_owner
        elif carried_owner and _looks_like_financial_statement(kv):
            # 이름 없이 따라붙은 재무제표 표 — 바로 앞에서 본 주인의 것이다.
            table_owner = carried_owner

        for pair in kv.pairs:
            raw_key, raw_val = (pair.key or "").strip(), (pair.value or "").strip()
            if not raw_key or raw_val in _EMPTY_VALUES:
                continue
            key_norm, key_unit = normalize_key(raw_key)
            if not key_norm or not is_meaningful_key(raw_key, key_norm):
                continue
            if row.doc_group == "periodic":
                num, unit, date = parse_periodic_value(
                    raw_val, key_unit=key_unit, unit_value=pair.unit_value,
                )
                reason = periodic_noise_reason(
                    key_norm, raw_val, value_num=num, value_date=date,
                )
                if reason:
                    if filter_stats is not None:
                        filter_stats[reason] = filter_stats.get(reason, 0) + 1
                    continue
            else:
                num, unit, date = parse_value(
                    raw_val, key_unit=key_unit, unit_value=pair.unit_value,
                )
            # 숫자도 날짜도 아니고 서술도 아닌 짧은 토막은 버린다(노이즈)
            if num is None and date is None and len(raw_val) < 2:
                continue
            facts.append(Fact(
                doc_id=row.doc_id, company=row.corp_name, corp_code=row.corp_code,
                doc_group=row.doc_group, doc_subtype=row.doc_subtype,
                report_name=parsed.document_name or row.report_nm,
                filing_date=row.rcept_dt, period=None,
                is_correction=row.is_correction,
                is_latest=correction.is_latest,
                correction_group_id=correction.correction_group_id,
                group_label=kv.group_label, key=raw_key, key_norm=key_norm,
                value_text=raw_val, value_num=num, value_unit=unit, value_date=date,
                field_code=pair.field_code, unit_code=pair.unit_code, unit_value=pair.unit_value,
                section_path=list(section.path),
                # 주인이 따로 적힌 표면 그 이름을, 아니면 None(=회사 자신).
                value_owner=(table_owner if table_owner and table_owner != row.corp_name
                             else None),
            ))
    return facts


def link_facts_to_chunks(facts: list[Fact], chunks: list[ChunkSchema]) -> int:
    """각 fact 에 출처 chunk_id 를 붙인다 — 근거 표시의 핵심.

    같은 문서의 조각들 중 **항목명과 값이 함께 들어 있는** 조각을 찾는다.
    (값만으로 찾으면 표 헤더나 다른 항목에 우연히 같은 숫자가 있을 때 틀린다.)
    못 찾으면 None 으로 남긴다 — 임의로 아무 조각이나 붙이지 않는다.
    """
    by_doc: dict[str, list[ChunkSchema]] = {}
    for c in chunks:
        by_doc.setdefault(c.report_id, []).append(c)

    linked = 0
    for f in facts:
        cands = by_doc.get(f.doc_id, [])
        best = None
        for c in cands:
            if f.value_text in c.raw_text and f.key in c.raw_text:
                best = c
                break
        if best is None:  # 항목명이 줄바꿈 등으로 갈렸을 수 있다 -> 값만으로 재시도
            for c in cands:
                if f.value_text in c.raw_text:
                    best = c
                    break
        if best is not None:
            f.chunk_id = best.chunk_id
            linked += 1
    return linked
