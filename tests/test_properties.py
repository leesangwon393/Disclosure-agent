"""속성 테스트 — 파싱·청킹의 **계약**을 고정한다.

왜 필요한가: 기존 테스트 세트(test_parsers/test_chunkers, 16개)에 의도적 회귀
7종을 주입했더니 **6종이 통과했다**. 표를 5행에서 자르든, 청킹을 사실상 꺼버리든,
토큰 상수를 4배로 하든 전부 초록불이었다. 기존 테스트는 "예전에 터진 버그 6개를
케이스별로 못 박은 회귀 방지 세트"이고 그 이상이 아니다.

여기서는 개별 케이스가 아니라 **분포·비율 수준의 계약**을 검사한다:
  1. 청크 크기 상한이 실제로 지켜지는가
  2. 원문 대비 커버리지 (내용을 조용히 버리고 있지 않은가)
  3. 표의 열 정렬 (헤더와 본문의 열 수가 같은가)
  4. 표의 1열 전폭 행이 보존되는가
  5. DART 기계판독 속성(AUNITVALUE 포함)이 보존되는가
  6. parent 참조 무결성

임계값은 넉넉하게 잡았다 — 목적은 "미세한 성능 변화 감지"가 아니라
"계층이 통째로 망가진 것을 즉시 잡는 것"이다.
"""
from __future__ import annotations

import os
import random
import re
from functools import lru_cache
from pathlib import Path

import pytest

from disclosure_rag.chunking.chunk_schema import filter_leaf_chunks
from disclosure_rag.chunking.chunkers import CHILD_MAX_TOKENS, chunk_document
from disclosure_rag.common.doc_tree import TableNode, TextNode
from disclosure_rag.common.manifest_loader import load_manifest
from disclosure_rag.common.unicode_utils import PathResolver
from disclosure_rag.correction.correction_graph_builder import build_correction_index
from disclosure_rag.parsing.document_detector import parse_documents_for_row

CORPUS = os.environ.get("CORPUS_ROOT", "corpus")
SEED = 20260823
N_PER_GROUP = {"periodic": 4, "major": 10, "exchange": 10, "holding": 10}

# 청크 상한(토큰) -> 문자 환산. heuristic(chars/2.0) 기준 + 구분자 오버헤드 10% 여유.
MAX_CHARS_ALLOWED = int(CHILD_MAX_TOKENS * 2.0 * 1.10)


@lru_cache(maxsize=1)
def _sample():
    if not Path(CORPUS).exists():
        pytest.skip(f"코퍼스 없음: {CORPUS} (CORPUS_ROOT 환경변수로 지정)")
    manifest = load_manifest(CORPUS)
    resolver = PathResolver(CORPUS)
    corrections = build_correction_index(manifest, resolver)
    by_group: dict[str, list] = {}
    for r in manifest:
        by_group.setdefault(r.doc_group, []).append(r)
    rnd = random.Random(SEED)
    rows = [r for g in sorted(by_group)
            for r in rnd.sample(by_group[g], min(N_PER_GROUP.get(g, 5), len(by_group[g])))]

    parsed_all, chunks_all = [], []
    for row in rows:
        for parsed in parse_documents_for_row(row, resolver):
            if parsed.report_subtype == "unsupported_pdf_html":
                continue
            parsed_all.append((row, parsed))
            chunks_all.extend(chunk_document(parsed, row, corrections[row.doc_id]))
    return parsed_all, chunks_all, resolver


def _iter_nodes(section):
    for c in section.children:
        if hasattr(c, "children"):
            yield from _iter_nodes(c)
        else:
            yield c


def _all_nodes(parsed):
    for s in parsed.sections:
        yield from _iter_nodes(s)


def _visible_chars(path: str) -> int:
    try:
        raw = Path(path).read_bytes().decode("utf-8", "replace")
    except OSError:
        return 0
    raw = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", raw)
    return len(re.sub(r"\s+", "", re.sub(r"<[^>]+>", " ", raw)))


# ------------------------------------------------------------------ 1. 크기
def test_leaf_chunk_size_is_actually_bounded():
    """상한이 '권고'가 아니라 '상한'이어야 한다.

    회귀 이력: 노드 하나가 예산을 넘으면 절대 쪼개지 않아 41,625자 leaf 가 존재했고,
    이 outlier 때문에 reranker 가 1,500쌍에 58분 걸렸다.
    """
    _, chunks, _ = _sample()
    leaf = filter_leaf_chunks(chunks)
    assert leaf, "leaf chunk 가 하나도 없다"
    lengths = sorted(len(c.raw_text) for c in leaf)
    over = [n for n in lengths if n > MAX_CHARS_ALLOWED]
    assert not over, (
        f"상한 {MAX_CHARS_ALLOWED}자 초과 leaf {len(over)}개 (최대 {lengths[-1]}자). "
        "packer.split_long_text 가 동작하지 않는다."
    )
    p50 = lengths[len(lengths) // 2]
    assert 300 <= p50 <= 2000, f"leaf p50={p50}자 — 청킹이 사실상 꺼졌거나 과분할됐다"


# ------------------------------------------------------------------ 2. 커버리지
@pytest.mark.parametrize("group,min_ratio", [("exchange", 0.95), ("holding", 0.90)])
def test_original_text_is_not_silently_dropped(group, min_ratio):
    """청크 문자 합 / 원문 가시 문자 >= 임계값.

    회귀 이력: 표의 1열 전폭 행을 로그 없이 버려서 거래소 공시 본문의 25%(중앙값)가
    청크에 없었다. 계약 단서조항("당사 지분 90% 해당금액")이 통째로 사라졌다.
    1.0 을 넘는 것은 헤더 반복 때문이라 정상이다.
    """
    parsed_all, chunks, _ = _sample()
    by_doc: dict[str, list] = {}
    for c in filter_leaf_chunks(chunks):
        by_doc.setdefault(c.report_id, []).append(c)

    num = den = 0
    for row, parsed in parsed_all:
        if row.doc_group != group:
            continue
        den += _visible_chars(os.path.join(CORPUS, parsed.source_path))
        num += sum(len(re.sub(r"\s+", "", c.raw_text)) for c in by_doc.get(row.doc_id, []))
    if den == 0:
        pytest.skip(f"{group} 표본 없음")
    ratio = num / den
    assert ratio >= min_ratio, f"{group} 커버리지 {ratio:.3f} < {min_ratio} — 본문이 유실되고 있다"


# ------------------------------------------------------------------ 3. 열 정렬
def test_table_rows_are_column_aligned():
    """정규 그리드여야 한다: 모든 행의 열 수가 동일.

    회귀 이력: expand_grid 로 펼친 뒤 RLE 로 다시 축약해 저장하는 바람에 본문 행의
    54.3%가 헤더와 열 수가 달랐다. " | " 렌더링에서 'N번째 필드 = 헤더 N번째 열'이
    깨지고, LLM 이 열을 세어 읽는 순간 틀린다. (EACL 2026: 금융문서 검색 실패의
    73%가 표 구조 불일치)
    """
    parsed_all, _, _ = _sample()
    total = bad = tables = 0
    for _row, parsed in parsed_all:
        for node in _all_nodes(parsed):
            if not isinstance(node, TableNode) or not node.rows:
                continue
            tables += 1
            ncol = len(node.rows[0])
            for r in node.rows:
                total += 1
                if len(r) != ncol:
                    bad += 1
    assert tables > 0, "표본에 TableNode 가 없다"
    assert bad == 0, f"열 수가 어긋난 행 {bad}/{total} — 정규 그리드가 깨졌다"


# ------------------------------------------------------------------ 4. 1열 행
def test_full_width_table_rows_are_preserved():
    """표의 1열 전폭 행이 TextNode 로 살아남아야 한다.

    DART/KRX 서식에서 이 자리는 '(단위: 백만원)', '기타 투자판단과 관련한 중요사항'
    같은 본문이다. 기존 코드는 `continue` 로 버렸다(holding 58.1%, periodic 18.8%).
    """
    parsed_all, _, _ = _sample()
    from_table = sum(
        1 for _row, parsed in parsed_all
        for n in _all_nodes(parsed)
        if isinstance(n, TextNode) and n.from_table_row
    )
    assert from_table > 0, "표의 1열 행이 하나도 보존되지 않았다 (table_parser 회귀)"


def test_unit_hint_is_populated():
    """단위 표기가 hint 로 승격돼야 한다.

    회귀 이력: unit_hint 가 전 코퍼스에서 항상 None(표 23,320개 중 0개)이라
    단위 삽입 로직이 죽은 코드였고, 금액 청크가 '원인지 백만원인지' 없이 인덱싱됐다.
    """
    parsed_all, _, _ = _sample()
    tables = [n for _r, p in parsed_all for n in _all_nodes(p) if isinstance(n, TableNode)]
    assert tables, "표본에 표가 없다"
    with_unit = sum(1 for t in tables if t.unit_hint)
    assert with_unit > 0, "unit_hint 가 하나도 채워지지 않았다"


# ------------------------------------------------------------------ 5. 기계판독 속성
def test_dart_machine_readable_attributes_survive_to_chunks():
    """ACODE/AUNIT/AUNITVALUE 가 청크까지 살아남아야 한다.

    회귀 이력: {셀 텍스트: 코드} 평면 dict 라 같은 텍스트면 덮어써졌고(major 80.1% 소실),
    AUNITVALUE 는 100% 폐기됐다. DART 가 이미 정규화해준 '20230421' 을 버리고
    LLM 이 '2023년 04월 21일' 을 다시 파싱하게 만드는 구조였다.
    """
    _, chunks, _ = _sample()
    refs = [r for c in chunks for r in c.field_codes]
    assert refs, "field_codes 가 하나도 없다"
    assert any(r.code for r in refs), "ACODE 가 하나도 보존되지 않았다"
    assert any(r.unit_value for r in refs), "AUNITVALUE 가 하나도 보존되지 않았다 (회귀)"
    # 위치 정보가 있어야 나중에 facts 층에서 열을 되짚을 수 있다
    assert any(r.row is not None and r.col is not None for r in refs), "위치 정보가 없다"


# ------------------------------------------------------------------ 6. 무결성
def test_parent_child_reference_integrity():
    _, chunks, _ = _sample()
    ids = {c.chunk_id for c in chunks}
    orphans = [c.chunk_id for c in chunks if c.parent_chunk_id and c.parent_chunk_id not in ids]
    assert not orphans, f"고아 parent 참조 {len(orphans)}건: {orphans[:3]}"
    assert len({c.chunk_id for c in chunks}) == len(chunks), "chunk_id 중복"
