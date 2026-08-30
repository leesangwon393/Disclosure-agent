"""회귀 테스트: malformed DART XML 사전정리 (2026-08-16 문서 대량 유실 버그).

전부 API 호출 없는 순수 유닛 테스트다. 코퍼스도 필요 없다 — 실제 원문에서
채집한 malformation 을 최소 XML 샘플로 재현한다.

배경과 실측 수치는 `parsing/xml_sanitizer.py` docstring 참고.
"""

from __future__ import annotations

from lxml import etree

from disclosure_rag.parsing.dart_xml_parser import parse_dart_xml
from disclosure_rag.parsing.xml_sanitizer import sanitize_dart_xml


def _parse(doc_bytes: bytes):
    return parse_dart_xml(
        doc_bytes,
        doc_id="test_doc",
        doc_group="periodic",
        doc_subtype="annual",
        report_subtype="main",
        source_path="test.xml",
    )


def _wrap(body_inner: bytes) -> bytes:
    return b"<DOCUMENT><DOCUMENT-NAME>\xed\x85\x8c\xec\x8a\xa4\xed\x8a\xb8</DOCUMENT-NAME><BODY>" + body_inner + b"</BODY></DOCUMENT>"


def _section(title: str, body: str = "본문") -> bytes:
    return f"<SECTION-1><TITLE>{title}</TITLE><P>{body}</P></SECTION-1>".encode()


def _section_with(title: str, inner: bytes) -> bytes:
    """실제 문서처럼 malformed content 를 SECTION-1 *안에* 넣는다.

    SECTION 밖에 두면 기존의 loose-content -> synthetic section 경로가 따로
    작동해서(§7) 이 테스트가 재려는 것과 다른 걸 재게 된다.
    """
    return f"<SECTION-1><TITLE>{title}</TITLE>".encode() + inner + b"</SECTION-1>"


# --------------------------------------------------------------------------
# A. 맨 `&`
# --------------------------------------------------------------------------

def test_bare_amp_is_escaped_and_counted():
    out, stats = sanitize_dart_xml(b"<P>S&P 500 and AT&T</P>")
    assert b"&amp;P" in out and b"&amp;T" in out
    assert stats.bare_amp == 2


def test_existing_entities_are_left_alone():
    src = b"<P>a &amp; b &lt; c &gt; d &quot;e&quot; &apos;f&apos; &#123; &#x1F;</P>"
    out, stats = sanitize_dart_xml(src)
    assert out == src, "이미 올바른 엔티티를 건드리면 안 된다"
    assert stats.total == 0


def test_wellformed_input_is_byte_identical():
    src = _wrap(_section("I. 회사의 개요"))
    out, stats = sanitize_dart_xml(src)
    assert out == src
    assert stats.total == 0


# --------------------------------------------------------------------------
# B. 맨 `<`  — 구조 붕괴의 진짜 원인
# --------------------------------------------------------------------------

def test_bare_lt_in_text_does_not_bury_following_sections():
    """`<신  설>` 같은 본문 `<` 뒤의 SECTION-1 이 유실되면 안 된다 (현대자동차 재현)."""
    body = (
        _section_with(
            "I. 회사의 개요",
            b"<TABLE><TBODY><TR><TD><\xec\x8b\xa0  \xec\x84\xa4></TD><TD>v</TD></TR></TBODY></TABLE>",
        )
        + _section("II. 사업의 내용")
        + _section("III. 재무에 관한 사항")
    )
    doc = _parse(_wrap(body))
    titles = [s.title for s in doc.sections]
    assert len(doc.sections) == 3, f"섹션이 표 안에 파묻힘 (회귀): {titles}"
    assert any(t.startswith("II.") for t in titles)
    assert any(t.startswith("III.") for t in titles)


def test_bare_lt_variants_are_escaped():
    """한글/ASCII 대문자로 시작하는 형태 모두 잡아야 한다."""
    out, stats = sanitize_dart_xml(
        b"<P><\xeb\xa9\x94\xeb\xa6\xac\xec\xb8\xa0\xed\x99\x94\xec\x9e\xac></P>"
        b"<P><Manufacturing Excellence></P>"
    )
    assert stats.bare_lt == 2
    assert b"&lt;Manufacturing Excellence>" in out
    assert etree.fromstring(b"<R>" + out + b"</R>") is not None


def test_unknown_but_real_tag_is_not_escaped():
    """고정 화이트리스트에 없는 실제 태그를 본문 텍스트로 오인하면 안 된다.

    회귀 이력: 정정공시 전용 `<CORRECTION>` 태그가 초기 화이트리스트에 없어서
    `&lt;CORRECTION>` 으로 이스케이프됐고, holding/major 정정공시의 섹션 구조가
    통째로 무너졌다(기존 테스트 4개 실패). 이후 문서 안의 닫는 태그를 걷어
    집합을 확장하도록 바꿨다.
    """
    src = _wrap(b"<NOTE-XYZ><TITLE>\xec\xa0\x95\xec\xa0\x95</TITLE><P>x</P></NOTE-XYZ>" + _section("I. 개요"))
    out, stats = sanitize_dart_xml(src)
    assert stats.bare_lt == 0, "닫는 태그가 있는 미지의 태그를 텍스트로 오인함"
    assert b"&lt;NOTE-XYZ" not in out


def test_self_closing_unknown_tag_is_not_escaped():
    out, stats = sanitize_dart_xml(b'<P>a</P><WEIRD-BREAK ATTR="1"/><P>b</P>')
    assert stats.bare_lt == 0


def test_real_tags_are_not_escaped():
    src = _wrap(_section("I. 개요") + b"<TABLE-GROUP ACLASS=\"X\"><TABLE><TBODY><TR><TE ACODE=\"C\">v</TE></TR></TBODY></TABLE></TABLE-GROUP>")
    out, stats = sanitize_dart_xml(src)
    assert stats.bare_lt == 0
    assert out == src


# --------------------------------------------------------------------------
# C. 속성값 안의 여분 따옴표 — 문서 절단의 원인
# --------------------------------------------------------------------------

def test_broken_attr_shape1_trailing_quote():
    """`ENG="value""` (현대자동차 실제 형태)."""
    out, stats = sanitize_dart_xml(b'<TH WIDTH="141" ENG="Other receivables and others"">x</TH>')
    assert stats.broken_attr == 1
    el = etree.fromstring(out)
    assert el.get("WIDTH") == "141"
    assert el.get("ENG") == 'Other receivables and others"'


def test_broken_attr_shape1_leading_quote():
    """`ENG=""value"` (현대자동차 실제 형태)."""
    out, stats = sanitize_dart_xml(b'<TH WIDTH="141" ENG=""Other  receivables and others">x</TH>')
    assert stats.broken_attr == 1
    el = etree.fromstring(out)
    assert el.get("WIDTH") == "141"
    assert el.get("ENG") == '"Other  receivables and others'


def test_broken_attr_shape2_wrapped_quotes():
    """`ENG="" value ""` (KB금융/CJ제일제당 실제 형태) — 빈 값으로 오인되는 케이스."""
    out, stats = sanitize_dart_xml(b'<TH COLSPAN="8" ENG="" KB Insurance Co., Ltd "">x</TH>')
    assert stats.broken_attr == 1
    el = etree.fromstring(out)
    assert el.get("COLSPAN") == "8"
    assert el.get("ENG") == '" KB Insurance Co., Ltd "'


def test_broken_attr_shape2_midtag():
    """깨진 속성이 태그 중간에 있어도 뒤 속성들이 살아야 한다 (CJ제일제당 형태)."""
    out, stats = sanitize_dart_xml(
        b'<TE ENG="" CJ telenix co.,Ltd "" VALIGN="MIDDLE" WIDTH="141">x</TE>'
    )
    assert stats.broken_attr == 1
    el = etree.fromstring(out)
    assert el.get("VALIGN") == "MIDDLE"
    assert el.get("WIDTH") == "141"


def test_broken_attr_does_not_truncate_document():
    """깨진 속성 뒤의 SECTION-1 이 통째로 폐기되면 안 된다 (KB금융/현대차 재현).

    이게 이 버그의 가장 위험한 형태다: start tag 파싱 실패가 연쇄 tag mismatch 를
    일으켜 루트가 조기 종료되고, 그 뒤 문서 전체가 조용히 사라진다.
    """
    body = (
        _section_with(
            "I. 회사의 개요",
            b'<TABLE><TBODY><TR><TH ENG="" KB Insurance Co., Ltd "">x</TH><TD>v</TD></TR></TBODY></TABLE>',
        )
        + _section("XI. 투자자 보호")
        + _section("XII. 상세표")
    )
    doc = _parse(_wrap(body))
    titles = [s.title for s in doc.sections]
    assert any(t.startswith("XII.") for t in titles), f"문서 뒷부분이 절단됨 (회귀): {titles}"


# --------------------------------------------------------------------------
# 통합 / 관측 가능성
# --------------------------------------------------------------------------

def test_all_three_malformations_together():
    """B 와 C 는 반드시 같이 고쳐야 한다 — 하나만 고치면 순손실이었다."""
    body = (
        _section_with(
            "I. 개요",
            b"<P>S&P \xeb\x93\xb1\xea\xb8\x89</P>"
            + b"<TABLE><TBODY><TR><TD><\xec\x8b\xa0  \xec\x84\xa4></TD>"
            + b'<TH ENG="Other receivables"">y</TH></TR></TBODY></TABLE>',
        )
        + _section("II. 사업의 내용")
        + _section("XII. 상세표")
    )
    doc = _parse(_wrap(body))
    titles = [s.title for s in doc.sections]
    assert len(doc.sections) == 3, titles
    assert any(t.startswith("XII.") for t in titles)


def test_sanitize_counts_are_reported_in_parse_warnings():
    """몇 개를 고쳤는지 관측 가능해야 한다 (§7)."""
    doc = _parse(_wrap(_section("I. 개요") + b"<P>S&P</P>"))
    assert any("XML 사전정리" in w for w in doc.parse_warnings), doc.parse_warnings


def test_clean_document_produces_no_sanitize_warning():
    doc = _parse(_wrap(_section("I. 개요")))
    assert not any("XML 사전정리" in w for w in doc.parse_warnings)
    assert len(doc.sections) == 1
