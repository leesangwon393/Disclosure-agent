"""의미 블록 탐지/패킹 회귀 테스트 (2026-08-25 상원님 제보 대응).

여기서 지키는 계약:
  1. 구간 제목(값 없는 무번호 행)이 있으면 구간 단위로 묶인다
  2. 구간 제목이 없으면 번호매김/들여쓰기로 묶는다
  3. 신호가 전혀 없는 평평한 표는 행마다 독립 블록 (기존 동작 보존)
  4. 블록이 예산을 넘으면 쪼개되, 모든 조각에 소속 라벨을 남긴다
  5. 어떤 경우에도 max_tokens 예산을 지킨다
"""
from __future__ import annotations

from disclosure_rag.common.doc_tree import TableCell, TableNode
from disclosure_rag.chunking.chunk_schema import estimate_tokens, render_table_node
from disclosure_rag.parsing.table_parser import (
    detect_semantic_blocks, is_numbered_label, strip_numbering_prefix,
)


def _row(*cells, indent=0, origin=None):
    """(라벨, 값...) 로 행 하나. 라벨 셀에만 들여쓰기를 준다."""
    out = []
    for i, t in enumerate(cells):
        out.append(TableCell(text=t, row=0, col=i,
                             indent=indent if i == 0 else 0,
                             origin_id=(origin if origin is not None else -1)))
    return out


def test_numbering_detection():
    for t in ("1. 매출액", "12)", "(3) 자산", "가. 개요", "Ⅰ. 회사의 개요", "IV) 기타"):
        assert is_numbered_label(t), t
    for t in ("매출액", "합계", "", "2025년"):
        assert not is_numbered_label(t), t
    assert strip_numbering_prefix("1. 매출액") == "매출액"
    assert strip_numbering_prefix("매출액") == "매출액"


def test_flat_table_keeps_row_granularity():
    """신호 없는 평평한 표 -> 행마다 독립 블록 (기존 동작과 동일해야 한다)."""
    body = [_row("삼성전자", "100"), _row("SK하이닉스", "200"), _row("LG", "300")]
    assert detect_semantic_blocks(body) == [[0], [1], [2]]


def test_group_header_wins_over_numbering():
    """SK하이닉스 '지역별 재무 정보' 모양 — 구간 제목 기준으로 묶여야 한다.

    번호("1./2./3.")로 끊으면 같은 구간의 매출액과 영업이익이 갈라진다.
    """
    body = [
        _row("본국"),                       # 구간 제목 (값 없음, 번호 없음)
        _row("1. 매출액", indent=1),
        _row("계", "89,400", indent=4),
        _row("2. 영업이익", "43,995"),
        _row("합계"),                       # 다음 구간
        _row("1. 매출액", indent=1),
        _row("계", "192,972", indent=4),
        _row("2. 영업이익", indent=1),
        _row("계", "47,206", indent=4),
    ]
    blocks = detect_semantic_blocks(body)
    assert blocks == [[0, 1, 2, 3], [4, 5, 6, 7, 8]], blocks
    # 합계 구간의 매출액 계와 영업이익 계는 반드시 같은 블록
    tail = blocks[-1]
    assert 6 in tail and 8 in tail


def test_numbering_fallback_without_group_header():
    """구간 제목이 없으면 상원님 원안(번호매김)으로 끊는다."""
    body = [
        _row("1. 매출액", "100"),
        _row("소계", "10", indent=4),
        _row("2. 영업이익", "200"),
        _row("소계", "20", indent=4),
    ]
    assert detect_semantic_blocks(body) == [[0, 1], [2, 3]]


def test_rowspan_rows_stay_together():
    """같은 origin_id(rowspan 복제)는 한 블록."""
    body = [
        _row("A", "1", indent=0, origin=7),
        _row("A", "2", indent=0, origin=7),
        _row("B", "3", indent=0, origin=8),
    ]
    blocks = detect_semantic_blocks(body)
    assert [0, 1] in blocks


def test_render_keeps_group_together_and_respects_budget():
    node = TableNode(rows=[
        [TableCell(text="구분", is_header=True, col=0),
         TableCell(text="금액", is_header=True, col=1)],
        _row("합계"),
        _row("1. 매출액", "192,972"),
        _row("2. 영업이익", "47,206"),
    ], title_hint="지역별 재무 정보")
    for style in ("kv", "grid"):
        frags = render_table_node(node, max_rows_per_chunk=20,
                                  max_tokens_per_chunk=1000, style=style)
        assert any("192,972" in f and "47,206" in f for f in frags), (style, frags)
        for f in frags:
            assert estimate_tokens(f) <= 1000 * 1.1, (style, estimate_tokens(f))


def test_oversized_block_is_split_with_label():
    """한 블록이 예산을 넘으면 쪼개되, 각 조각에 소속 라벨이 남아야 한다."""
    rows = [[TableCell(text="구분", is_header=True, col=0),
             TableCell(text="값", is_header=True, col=1)]]
    rows.append(_row("초대형구간"))
    for i in range(300):
        rows.append(_row(f"항목{i}", f"{i}", indent=4))
    node = TableNode(rows=rows, title_hint="T")
    frags = render_table_node(node, max_rows_per_chunk=20,
                              max_tokens_per_chunk=300, style="kv")
    assert len(frags) > 1
    assert all("초대형구간" in f for f in frags), "쪼갠 조각이 소속 라벨을 잃었다"
    for f in frags:
        assert estimate_tokens(f) <= 300 * 1.2, estimate_tokens(f)


def test_empty_and_header_only_tables():
    assert render_table_node(TableNode(rows=[])) == []
    node = TableNode(rows=[[TableCell(text="헤더", is_header=True, col=0)]], title_hint="T")
    assert len(render_table_node(node)) == 1
