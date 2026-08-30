"""DART 원본 XML 의 malformation 을 파싱 *전에* 정리한다.

배경 (2026-08-16 실측, KIM 작업본)
==================================
DART 원본 XML 은 well-formed 가 아니다. `recover=True` 로 넘기면 lxml 이 태그
구조를 오정렬시켜 최상위 SECTION-1 이 TABLE/TBODY/TR 안에 파묻히고, `_walk()`가
BODY 직속 SECTION-1 만 문서 섹션으로 승격하므로 **문서 대부분이 조용히 유실**된다.

실측으로 확인한 malformation 은 3종이며 성격이 전부 다르다.

A. 이스케이프 안 된 `&`  — 예: `S&P`, `Battery Recycling & Production ETF`
   → **구조는 안 깨진다.** lxml 이 텍스트로 복구한다. 정상 문서(삼성전자)도
     972개를 갖고 있다. 즉 이것만 고쳐서는 유실 문제가 전혀 해결되지 않는다.
     (원래 이 버그의 원인으로 지목됐던 항목이지만 실측 결과 무죄였다.)

B. 이스케이프 안 된 `<`  — 예: `<신  설>`, `<메리츠화재>`, `<Manufacturing Excellence>`
   → **구조 붕괴의 진짜 원인.** lxml 이 `<신` 을 시작태그로, `설` 을 속성으로
     읽는다. 닫히지 않는 유령 element 가 생기고 이후 모든 형제 노드가 그 안에
     중첩된다. 삼성전자가 무사했던 건 그 문서의 맨 `<` 가 전부 `< TV 점유율 >`
     처럼 뒤에 공백이 있어 lxml 이 태그로 인식하지 않았기 때문이다 — 순전히 운.

C. 속성값 안의 여분 따옴표 — 전부 `ENG` 속성, TH/TE 태그
   예: `ENG="Other receivables and others""` / `ENG=""Other receivables and others"`
       `ENG="" KB Insurance Co., Ltd ""`
   → **문서 절단의 원인.** start tag 파싱이 실패하면 뒤따르는 `</TR>`, `</THEAD>`,
     `</TABLE>` 이 전부 mismatch 로 연쇄되며 스택이 DOCUMENT 까지 pop 된다.
     루트가 조기 종료되면 XML 규격상 이후 내용은 전부 폐기된다.
   → 실측: periodic 1,051건 중 79건 / 401회 발생. major·holding 은 0건.

**B 와 C 는 반드시 같이 고쳐야 한다.** B 만 고치면 현대자동차/KB금융은 오히려
본문이 절반으로 줄어든다 — B 의 유령 태그가 스택을 깊게 쌓아둔 덕분에 C 의 연쇄
pop 이 DOCUMENT 까지 도달하지 못하고 있었기 때문이다. 완충재만 걷어내면 순손실이다.
(실측: 현대차 813천자 → 422천자, KB금융 1,634천자 → 814천자)

효과 (각 사 최신 사업보고서, BODY 직속 SECTION-1 개수)
------------------------------------------------------
  현대자동차 2→14 · KB금융 3→14 · 메리츠금융지주 2→14 · 삼성SDI 7→14 ·
  LG에너지솔루션 4→14 · 삼성전자 14→14(유지) · 한미반도체 14→14(유지)
  lxml 에러도 전 문서 0건이 되고, 본문 텍스트 총량은 어느 문서도 줄지 않는다.

설계 원칙
---------
- **bytes 레벨에서만 처리한다.** str 로 디코드하지 않는다 (인코딩 UTF-8).
- **well-formed 인 부분은 바이트 단위로 그대로 통과시킨다.** 정상 시작태그는
  재작성하지 않고, 파싱이 실제로 실패한 태그만 수리한다 (최소 개입).
- 이미 올바른 엔티티(`&amp;` `&lt;` `&#123;` `&#x1F;` 등)는 건드리지 않는다.
- 치환 건수를 돌려줘서 호출부가 `parse_warnings` 에 남길 수 있게 한다 (§7 관측 가능성).
- `recover=True` 는 그대로 유지한다 — 여기서 못 잡는 malformation 이 남아 있을 수 있다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 알려진 DART 태그. 에러 0건으로 파싱되는 사업보고서들에서 채집한 기본 집합이다.
# 다만 **이것만 믿으면 안 된다** — 실제로 정정공시 전용 `<CORRECTION>` 태그가
# 이 목록에 없어서 본문 텍스트로 오인되는 회귀가 났었다(기존 테스트 4개 실패로
# 잡힘). 그래서 아래 `_markup_tags()` 가 문서마다 닫는 태그를 훑어 집합을
# 확장한다. 이 상수는 "닫는 태그가 유실될 만큼 심하게 깨진 문서"를 위한 바닥이다.
DART_TAGS: frozenset[bytes] = frozenset({
    b"A", b"BODY", b"COL", b"COLGROUP", b"COMPANY-NAME", b"CORRECTION", b"COVER",
    b"COVER-TITLE", b"DOCUMENT", b"DOCUMENT-NAME", b"EXTRACTION", b"FORMULA-VERSION",
    b"IMAGE", b"IMG", b"IMG-CAPTION", b"LIBRARY", b"P", b"PGBRK", b"SECTION-1",
    b"SECTION-2", b"SECTION-3", b"SECTION-4", b"SPAN", b"SUMMARY", b"TABLE",
    b"TABLE-GROUP", b"TBODY", b"TD", b"TE", b"TH", b"THEAD", b"TITLE", b"TR", b"TU",
})

# 문서 안에 실제로 닫는 태그가 있으면 그 이름은 마크업이다.
_CLOSE_TAG = re.compile(rb"</([A-Za-z][A-Za-z0-9._-]*)\s*>")
# 자기닫음 태그(`<PGBRK/>`)는 닫는 태그가 없으므로 따로 걷는다.
_SELF_CLOSING = re.compile(rb"<([A-Za-z][A-Za-z0-9._-]*)[^<>]*/>")


def _markup_tags(data: bytes) -> frozenset[bytes]:
    """이 문서에서 '태그로 취급할 이름' 집합.

    고정 목록 + 문서가 스스로 증명한 것(닫는 태그/자기닫음 태그)을 합친다.
    `<메리츠화재>`, `<Manufacturing Excellence>` 같은 본문 텍스트는 짝이 되는
    닫는 태그가 없으므로 여기 들어오지 않는다.
    """
    found = set(_CLOSE_TAG.findall(data))
    found.update(_SELF_CLOSING.findall(data))
    return frozenset(DART_TAGS | found)

# 이미 올바른 엔티티는 제외하고 맨 `&` 만 잡는다.
_BARE_AMP = re.compile(rb"&(?!(?:[a-zA-Z][a-zA-Z0-9]*|#[0-9]+|#[xX][0-9a-fA-F]+);)")
_TAGNAME = re.compile(rb"[A-Za-z][A-Za-z0-9._-]*")
_ATTRNAME = re.compile(rb"[A-Za-z_:][A-Za-z0-9._:-]*")
# 속성값의 진짜 끝: 닫는 따옴표 뒤에 (공백* 다음속성명 =) 또는 (공백* > 또는 />) 가 와야 한다.
_VALUE_END = re.compile(rb'"(?=\s*(?:[A-Za-z_:][A-Za-z0-9._:-]*\s*=|/?>))')
_WS = b" \t\r\n"

_LT = 0x3C   # '<'
_GT = 0x3E   # '>'
_EQ = 0x3D   # '='
_SLASH = 0x2F  # '/'
_QUOTE = 0x22  # '"'


@dataclass(frozen=True)
class SanitizeStats:
    """무엇을 몇 개 고쳤는지. 관측 가능성을 위해 반드시 기록한다."""

    bare_amp: int = 0      # A: `&` -> `&amp;`
    bare_lt: int = 0       # B: 본문의 `<` -> `&lt;`
    broken_attr: int = 0   # C: 따옴표 깨진 시작태그를 재작성한 횟수

    @property
    def total(self) -> int:
        return self.bare_amp + self.bare_lt + self.broken_attr

    def describe(self) -> str:
        return (
            f"XML 사전정리: 맨 & {self.bare_amp}건, 본문 < {self.bare_lt}건, "
            f"속성 따옴표 깨짐 {self.broken_attr}건 수정"
        )


def _is_markup(data: bytes, i: int, tags: frozenset[bytes]) -> bool:
    """data[i] == '<' 일 때 이게 마크업인가(태그/닫는태그/주석/PI), 아니면 본문 텍스트인가."""
    if i + 1 >= len(data):
        return False
    nxt = data[i + 1]
    if nxt in b"/!?":
        return True
    m = _TAGNAME.match(data, i + 1)
    return bool(m and m.group(0) in tags)


def _rewrite_start_tag(data: bytes, start: int) -> tuple[bytes | None, int, bool]:
    """시작태그 하나를 파싱한다.

    Returns (재작성된_바이트 | None, 태그_끝_오프셋, 수리했는지).
    수리가 불필요했으면 재작성 결과는 원본과 바이트 단위로 동일하다.
    파싱 자체가 불가능하면 (None, ?, False) — 호출부가 원본을 그대로 둔다.
    """
    n = len(data)
    m = _TAGNAME.match(data, start + 1)
    if m is None:
        return None, start, False
    pos = m.end()
    out = bytearray(data[start:pos])
    repaired = False
    # 직전 속성을 되돌리기 위한 기억: (out 내 시작 오프셋, 값 여는 따옴표 위치, 속성명+'=' 바이트)
    prev_attr: tuple[int, int, bytes] | None = None

    while True:
        ws_start = pos
        while pos < n and data[pos] in _WS:
            pos += 1
        if pos >= n:
            return None, pos, False
        if data[pos] == _GT:
            out += data[ws_start:pos] + b">"
            return bytes(out), pos + 1, repaired
        if data[pos] == _SLASH and pos + 1 < n and data[pos + 1] == _GT:
            out += data[ws_start:pos] + b"/>"
            return bytes(out), pos + 2, repaired

        am = _ATTRNAME.match(data, pos)
        eq = -1
        if am is not None:
            eq = am.end()
            while eq < n and data[eq] in _WS:
                eq += 1

        if am is None or eq >= n or data[eq] != _EQ:
            # 속성으로 읽히지 않는다 = 직전 속성의 값이 일찍 끊긴 것.
            # 예: `ENG="" KB Insurance Co., Ltd ""` — `""` 를 빈 값으로 읽고 나면
            # 남은 ` KB Insurance ...` 가 속성명 자리에 오게 된다.
            if prev_attr is None:
                return None, pos, False
            out_at, quote_at, prefix = prev_attr
            vm = _VALUE_END.search(data, quote_at + 1)
            if vm is None:
                return None, pos, False
            del out[out_at:]
            out += prefix + b'"' + data[quote_at + 1:vm.start()].replace(b'"', b"&quot;") + b'"'
            pos = vm.end()
            prev_attr = None
            repaired = True
            continue

        out += data[ws_start:pos]
        attr_out_at = len(out)
        prefix = data[pos:eq + 1]   # 속성명 + 공백 + '='
        pos = eq + 1
        while pos < n and data[pos] in _WS:
            pos += 1
        if pos >= n or data[pos] != _QUOTE:
            return None, pos, False

        quote_at = pos
        naive_end = data.find(b'"', quote_at + 1)
        if naive_end < 0:
            return None, pos, False
        after = naive_end + 1
        naive_ok = (
            after >= n
            or data[after] in _WS
            or data[after] == _GT
            or (data[after] == _SLASH and after + 1 < n and data[after + 1] == _GT)
        )
        if naive_ok:
            out += prefix + data[quote_at:after]
            prev_attr = (attr_out_at, quote_at, prefix)
            pos = after
            continue

        # 닫는 따옴표 직후에 잡문자 — 예: `ENG="Other receivables and others""`
        vm = _VALUE_END.search(data, quote_at + 1)
        if vm is None:
            return None, pos, False
        out += prefix + b'"' + data[quote_at + 1:vm.start()].replace(b'"', b"&quot;") + b'"'
        pos = vm.end()
        prev_attr = None
        repaired = True


def sanitize_dart_xml(file_bytes: bytes) -> tuple[bytes, SanitizeStats]:
    """malformed DART XML 을 파싱 가능한 형태로 정리한다.

    입력이 이미 well-formed 면 바이트 단위로 동일한 결과를 돌려준다
    (SanitizeStats.total == 0).
    """
    n_amp = len(_BARE_AMP.findall(file_bytes))
    data = _BARE_AMP.sub(b"&amp;", file_bytes) if n_amp else file_bytes

    tags = _markup_tags(data)

    out = bytearray()
    n = len(data)
    pos = 0
    copied_upto = 0
    n_lt = 0
    n_attr = 0

    while True:
        pos = data.find(b"<", pos)
        if pos < 0:
            break
        if not _is_markup(data, pos, tags):
            # B: 본문에 그냥 쓰인 `<`
            out += data[copied_upto:pos] + b"&lt;"
            n_lt += 1
            pos += 1
            copied_upto = pos
            continue
        if data[pos + 1] in b"/!?":
            pos += 1
            continue

        rewritten, end, repaired = _rewrite_start_tag(data, pos)
        if rewritten is None:
            # 여기서 못 고치는 형태 — 원본 그대로 두고 recover 파서에 맡긴다 (§7)
            pos += 1
            continue
        if repaired:
            # C: 실제로 깨져 있던 태그만 교체한다
            out += data[copied_upto:pos] + rewritten
            copied_upto = end
            n_attr += 1
        pos = end

    if n_lt == 0 and n_attr == 0:
        return data, SanitizeStats(bare_amp=n_amp)

    out += data[copied_upto:]
    return bytes(out), SanitizeStats(bare_amp=n_amp, bare_lt=n_lt, broken_attr=n_attr)
