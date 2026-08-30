"""전처리 파이프라인 오케스트레이터: manifest 전체 -> ChunkSchema 리스트.

Corpus -> Path Resolve -> Parse -> Correction Graph -> Chunk 의 전체 흐름을
연결한다. Phase 8(BM25)/Phase 9(Dense) 는 이 모듈이 만든 ChunkSchema 리스트를
입력으로 받는다.
"""

from __future__ import annotations

import logging

from disclosure_rag.chunking.chunk_schema import ChunkSchema, filter_leaf_chunks
from disclosure_rag.chunking.chunkers import chunk_document
from disclosure_rag.common.corpus_validator import validate_corpus
from disclosure_rag.common.manifest_loader import ManifestRow, load_manifest
from disclosure_rag.common.unicode_utils import PathResolver
from disclosure_rag.correction.correction_graph_builder import build_correction_index
from disclosure_rag.parsing.document_detector import parse_documents_for_row

logger = logging.getLogger(__name__)


def build_all_chunks(
    corpus_root: str,
    *,
    rows: list[ManifestRow] | None = None,
    validate: bool = True,
) -> list[ChunkSchema]:
    resolver = PathResolver(corpus_root)
    manifest = load_manifest(corpus_root)

    if validate:
        report, _ = validate_corpus(corpus_root)
        if not report.is_healthy():
            logger.warning("[PIPELINE] Corpus validation UNHEALTHY — 통계를 신뢰하지 말고 원인 파악 우선.\n%s", report.render())

    correction_index = build_correction_index(manifest, resolver)

    target_rows = rows if rows is not None else manifest
    all_chunks: list[ChunkSchema] = []
    for row in target_rows:
        parsed_docs = parse_documents_for_row(row, resolver)
        correction = correction_index.get(row.doc_id)
        if correction is None:
            logger.warning("[PIPELINE] doc_id=%s correction record 없음 — skip", row.doc_id)
            continue
        for parsed in parsed_docs:
            if parsed.report_subtype == "unsupported_pdf_html":
                continue
            chunks = chunk_document(parsed, row, correction)
            if not chunks:
                logger.warning(
                    "[PIPELINE] doc_id=%s (%s/%s) 파싱은 됐는데 chunk 0개 — 확인 필요",
                    row.doc_id, row.doc_group, parsed.report_subtype,
                )
            all_chunks.extend(chunks)

    return all_chunks


def build_retrieval_chunks(corpus_root: str, **kwargs) -> list[ChunkSchema]:
    """BM25/Dense 인덱스 구축용 — build_all_chunks() 와 달리 leaf chunk 만 반환한다.

    build_all_chunks() 의 전체 결과(parent 포함)는 Parent Expansion(§56, 아직 미구현)
    단계에서 필요하므로 별도로 보관해두되, Retriever 는 반드시 이 함수를 써야 한다
    (parent 를 그대로 임베딩하면 비정상적으로 느려지고 검색 품질도 떨어짐 — 실측 확인됨).
    """
    return filter_leaf_chunks(build_all_chunks(corpus_root, **kwargs))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING)
    root = sys.argv[1] if len(sys.argv) > 1 else "corpus"
    chunks = build_all_chunks(root)
    leaf_chunks = filter_leaf_chunks(chunks)
    print(f"총 chunk 수: {len(chunks)} (leaf/검색대상: {len(leaf_chunks)}, parent/context용: {len(chunks) - len(leaf_chunks)})")
