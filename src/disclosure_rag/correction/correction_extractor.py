"""정정공시 본문에서 "정정대상 원본"의 제출일/제목을 추출한다.

Phase 0 실측 근거 (§7): 모든 `[기재정정]` 문서 본문에는 예외 없이
    "1. 정정대상 공시서류 : <원본 제목>"
    "2. 정정대상 공시서류의 최초제출일 : <날짜>"   (periodic/major/holding)
    "2. 정정관련 공시서류제출일 : <날짜>"           (exchange, 문구가 다름)
텍스트가 존재한다. 날짜 표기는 "YYYY년 MM월 DD일" / "YYYY-MM-DD" / "YYYY.MM.DD"
3가지가 혼재한다 (실측). 전체 1,002건(xml) 대상 회귀 테스트 결과 1,001건 성공(99.9%),
유일한 실패는 원문 자체의 오타(major_20250828001452, "08년 28일") — 이런 known
edge case 는 regex 를 억지로 느슨하게 만들지 않고 overrides.py 의 명시적
override table 로 처리한다 (사용자 결정 #5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lxml import etree, html as lhtml

_DATE = r"(\d{4})\s*[-.년]\s*(\d{1,2})\s*[-.월]\s*(\d{1,2})\s*일?"
_PAT_TARGET_DATE = re.compile(r"정정대상\s*공시서류의\s*최초제출일\s*[:：]?\s*" + _DATE)
_PAT_TARGET_DATE_EXCHANGE = re.compile(r"정정관련\s*공시서류제출일\s*[:：]?\s*" + _DATE)
# 주의: 본문은 파싱 전 개행이 전부 공백으로 flatten 되므로(_flatten_text), "\n" 을
# 종료 앵커로 쓸 수 없다. 대신 항상 바로 뒤에 오는 "2. 정정대상/정정관련..." 필드
# 시작 지점을 lookahead 로 잡아 종료시킨다 (실측: 순서가 항상 1->2 로 고정).
_PAT_TARGET_TITLE = re.compile(r"정정대상\s*공시서류\s*[:：]?\s*(.+?)\s*(?=\d\s*\.\s*정정)")
_PAT_TARGET_TITLE_EXCHANGE = re.compile(r"정정관련\s*공시서류\s*[:：]?\s*(.+?)\s*(?=\d\s*\.\s*정정)")


@dataclass
class CorrectionExtraction:
    target_date: str | None  # YYYYMMDD
    target_title: str | None
    ok: bool


def _flatten_text(file_bytes: bytes, *, is_html: bool) -> str:
    try:
        if is_html:
            doc = lhtml.fromstring(file_bytes.decode("utf-8", errors="replace"))
        else:
            parser = etree.XMLParser(recover=True, huge_tree=True)
            doc = etree.fromstring(file_bytes, parser=parser)
        text = " ".join(doc.itertext())
    except Exception:  # noqa: BLE001
        text = file_bytes.decode("utf-8", errors="replace")
    return re.sub(r"\s+", " ", text)


def extract_correction_info(file_bytes: bytes, *, doc_group: str) -> CorrectionExtraction:
    is_html = doc_group == "exchange"
    text = _flatten_text(file_bytes, is_html=is_html)

    date_pat = _PAT_TARGET_DATE_EXCHANGE if is_html else _PAT_TARGET_DATE
    title_pat = _PAT_TARGET_TITLE_EXCHANGE if is_html else _PAT_TARGET_TITLE

    date_match = date_pat.search(text) or (_PAT_TARGET_DATE.search(text) if is_html else _PAT_TARGET_DATE_EXCHANGE.search(text))
    title_match = title_pat.search(text) or (_PAT_TARGET_TITLE.search(text) if is_html else _PAT_TARGET_TITLE_EXCHANGE.search(text))

    target_date = None
    if date_match:
        y, m, d = date_match.groups()
        target_date = f"{int(y):04d}{int(m):02d}{int(d):02d}"

    target_title = title_match.group(1).strip() if title_match else None

    return CorrectionExtraction(target_date=target_date, target_title=target_title, ok=target_date is not None)
