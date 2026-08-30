"""Document Detector: manifest 의 doc_group 으로 라우팅한다.

파일 확장자(`.xml`)로 파서를 결정하지 않는다 (금지 사항 #2) — manifest 가 이미
`doc_group` 을 제공하므로 이를 신뢰하고, exchange 만 HTML 파서로 명시적으로
분기한다. 정기공시(annual) 의 첨부(감사보고서/연결감사보고서)처럼 폴더 안에
파일이 여러 개인 경우, 파일마다 별도 ParsedDocument 를 만들고 report_subtype 으로
구분한다 — 절대 버리지 않는다 (§14 "이 값들을 버리지 않는다" 원칙을 파일 단위로도 적용).

사용자 결정 #4: pdf+html 대체수집 3건은 현재 main pipeline 의 blocker 로 삼지
않는다. parser 가 처리할 수 없음을 명시적으로 report_subtype="unsupported_pdf_html"
로 기록하고 넘어간다 (silent 하게 버리지 않는다, §7).
"""

from __future__ import annotations

import logging

from disclosure_rag.common.doc_tree import ParsedDocument
from disclosure_rag.common.manifest_loader import ManifestRow
from disclosure_rag.common.unicode_utils import PathResolver
from disclosure_rag.parsing.dart_xml_parser import parse_dart_xml
from disclosure_rag.parsing.exchange_parser import parse_exchange_html

logger = logging.getLogger(__name__)


def _classify_attachment(document_name: str | None) -> str:
    if not document_name:
        return "attachment"
    if "연결" in document_name and "감사보고서" in document_name:
        return "consolidated_audit_report"
    if "감사보고서" in document_name:
        return "separate_audit_report"
    return "attachment"


def parse_documents_for_row(row: ManifestRow, resolver: PathResolver) -> list[ParsedDocument]:
    """manifest 한 행(=공시 1건, 폴더 1개)에서 실제 파일들을 전부 파싱한다."""
    doc_dir = resolver.resolve(row.file_path)
    if doc_dir is None or not doc_dir.is_dir():
        logger.warning("[DOCUMENT_DETECTOR] 폴더 resolve 실패: doc_id=%s file_path=%r", row.doc_id, row.file_path)
        return []

    if row.file_format == "pdf+html":
        logger.warning(
            "[DOCUMENT_DETECTOR] pdf+html 대체수집 문서 — main pipeline 미지원으로 스킵: doc_id=%s", row.doc_id,
        )
        return [
            ParsedDocument(
                doc_id=row.doc_id, doc_group=row.doc_group, doc_subtype=row.doc_subtype,
                report_subtype="unsupported_pdf_html", source_path=row.file_path,
                document_name=row.report_nm, sections=[],
                parse_warnings=["pdf+html 형식 — Phase 4 파서 미지원, main pipeline 완료 후 별도 처리 예정"],
            )
        ]

    xml_files = sorted(doc_dir.glob("*.xml"))
    if not xml_files:
        logger.warning("[DOCUMENT_DETECTOR] doc_id=%s 폴더에 xml 파일 없음: %s", row.doc_id, doc_dir)
        return []

    results: list[ParsedDocument] = []
    for fpath in xml_files:
        is_main = fpath.stem == row.rcept_no
        file_bytes = fpath.read_bytes()
        source_path = f"{row.file_path}/{fpath.name}"

        if row.doc_group == "exchange":
            parsed = parse_exchange_html(
                file_bytes, doc_id=row.doc_id, doc_subtype=row.doc_subtype, source_path=source_path,
            )
        else:
            parsed = parse_dart_xml(
                file_bytes,
                doc_id=row.doc_id,
                doc_group=row.doc_group,
                doc_subtype=row.doc_subtype,
                report_subtype="main" if is_main else "attachment",
                source_path=source_path,
            )
            if not is_main:
                parsed.report_subtype = _classify_attachment(parsed.document_name)

        if parsed.parse_warnings:
            for w in parsed.parse_warnings:
                logger.warning("[DOCUMENT_DETECTOR] doc_id=%s file=%s: %s", row.doc_id, fpath.name, w)

        results.append(parsed)

    return results
