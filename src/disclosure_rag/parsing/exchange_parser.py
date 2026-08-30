"""exchange 공시 Parser.

실측 결과(Phase 0 §16): `.xml` 확장자지만 실제 내용은 100% HTML(`<html><table>`)
이다. 확장자로 판단하지 않고, 이 모듈이 명시적으로 HTML 파서(lxml.html)를 사용한다
(금지 사항 #2). 문서가 짧아(평균 9.7KB) 대부분 "공시 1건 = 섹션 1개"로 수렴한다.
"""

from __future__ import annotations

from lxml import html as lhtml

from disclosure_rag.common.doc_tree import ParsedDocument, SectionNode
from disclosure_rag.parsing.table_parser import RawCell, classify_grid, expand_grid


def _text_of(el) -> str:
    return "".join(el.itertext()).strip()


def _cell_indent(el) -> int:
    raw = "".join(el.itertext())
    n = 0
    for ch in raw:
        if ch in " \t\u00a0\u3000":
            n += 1
        elif ch in "\r\n":
            n = 0
        else:
            break
    return n


def _cell_to_raw(cell_el) -> RawCell:
    return RawCell(
        text=_text_of(cell_el),
        indent=_cell_indent(cell_el),
        rowspan=int(cell_el.get("rowspan", "1") or "1"),
        colspan=int(cell_el.get("colspan", "1") or "1"),
        is_header=(cell_el.tag == "th"),
    )


def _parse_html_table(table_el):
    raw_rows: list[list[RawCell]] = []
    for tr in table_el.iter("tr"):
        cells = [c for c in tr if c.tag in ("td", "th")]
        if cells:
            raw_rows.append([_cell_to_raw(c) for c in cells])
    if not raw_rows:
        return []
    grid = expand_grid(raw_rows)
    return classify_grid(grid)


def parse_exchange_html(
    file_bytes: bytes,
    *,
    doc_id: str,
    doc_subtype: str | None,
    source_path: str,
) -> ParsedDocument:
    warnings: list[str] = []
    try:
        # exchange 파일은 HTML meta 에 euc-kr 로 표기돼 있지만 실제 파일은 UTF-8 (§88 README 5).
        root = lhtml.fromstring(file_bytes.decode("utf-8", errors="replace"))
    except Exception as e:  # noqa: BLE001
        warnings.append(f"HTML parse 실패: {e}")
        return ParsedDocument(
            doc_id=doc_id, doc_group="exchange", doc_subtype=doc_subtype,
            report_subtype="main", source_path=source_path,
            document_name=None, sections=[], parse_warnings=warnings,
        )

    title_el = root.find(".//title")
    document_name = _text_of(title_el) if title_el is not None else None

    # 문서 안의 굵은 중앙정렬 span 이 실제 공시 제목(예: "단일판매ㆍ공급계약 체결")
    event_title = None
    for span in root.iter("span"):
        style = (span.get("style") or "")
        if "font-weight:bold" in style and "text-align:center" in style:
            text = _text_of(span)
            if text:
                event_title = text
                break
    section_title = event_title or document_name or "exchange 공시"

    section = SectionNode(title=section_title, level=1, path=[section_title])
    for table_el in root.iter("table"):
        section.children.extend(_parse_html_table(table_el))

    if not section.children:
        warnings.append(f"table 파싱 결과 없음 — 구조 확인 필요 (doc_id={doc_id})")

    return ParsedDocument(
        doc_id=doc_id, doc_group="exchange", doc_subtype=doc_subtype,
        report_subtype="main", source_path=source_path,
        document_name=document_name, sections=[section], parse_warnings=warnings,
    )
