"""공통 Table 처리 (§21).

DART XML <TABLE> (TH/TD/TE/TU 셀) 과 exchange 의 위장 HTML <table> (<td>/<span>) 은
태그 어휘는 다르지만, 둘 다 "rowspan/colspan 이 섞인 2차원 그리드"라는 점은 동일하다.

그래서 두 원본 모두 먼저 RawCell 의 2차원 리스트(행 단위)로 변환한 뒤,
동일한 expand_grid() 로 rowspan/colspan 을 채워 넣은 정규 그리드를 만들고,
동일한 classify_grid() 로 KeyValueNode / TableNode / TextNode 를 결정한다.

절대 금지: Table 을 그냥 순서대로 이어붙여 plain text 로 flatten 하는 것 (금지 사항 #3).
Column-Value 관계, rowspan 으로 묶인 상위 라벨 관계를 반드시 보존한다.

=== KIM 브랜치 변경점 (2026-08-23) ===

[A] **정규 그리드를 유지한다.**
    기존: expand_grid 로 펼친 뒤 _rle() 로 다시 축약해 TableNode.rows 에 저장.
          -> 본문 행의 53.1%가 헤더와 열 수 불일치(실측, 그런 표가 61.0%).
             " | " 로 렌더링하면 "3번째 필드 = 헤더 3번째 열"이 과반에서 깨진다.
             EACL 2026 벤치마크는 금융문서 검색 실패의 73%가 표 구조 불일치라고
             보고한다 — 우리가 그 실패 모드를 스스로 만들고 있었다.
    변경: TableNode.rows 는 펼친 그리드 그대로(모든 행 열 수 동일). 반복 복제된
          칸은 dup_left/dup_up 플래그만 세워두고 렌더링에서 빈칸으로 낸다.
          KV/Table 분류 판단은 종전처럼 RLE 뷰에서 하므로 기존 회귀는 유지된다.

[B] **1열짜리 행을 더 이상 버리지 않는다.**
    기존: `else: continue` — 주석은 "구분선/타이틀류"라 했지만 DART/KRX 서식에서
          전폭 1열 행은 서술형 본문 자리다. 실측 폐기율:
            periodic 18.8%(63만자) / holding 58.1% / major 17.6% / exchange 13.7%
          사라진 것: "(단위: 백만원)", "제 41 기 1분기말 2024.03.31 현재",
                     "11. 기타 투자판단에 참고할 사항 …", 보유목적 서술
    변경: TextNode(from_table_row=True) 로 문서 순서 그대로 보존한다.

[C] **unit_hint / period_hint 를 실제로 채운다.**
    기존: unit_hint 가 전 코퍼스에서 항상 None(표 23,320개 중 0개) — 단위 삽입
          로직이 죽은 코드였고, 금액 청크가 "원인지 백만원인지" 없이 인덱싱됐다.
    변경: 표 안/직전의 단위·기수 표기를 정규식으로 찾아 hint 로 승격한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from disclosure_rag.common.doc_tree import (
    ContentNode,
    KeyValueNode,
    KVPair,
    TableCell,
    TableNode,
    TextNode,
)

KEY_VALUE_MAX_COLS = 3  # 이보다 열이 많으면 grid(table)로 취급

# 표 안에서 "단위" / "기수·기준일" 표기를 식별하는 패턴 (실측 문자열 기반).
#   "(단위: 백만원)"  "(원화단위: 백만원, 외화단위: 외화 천단위)"  "단위 : 원"
_UNIT_RE = re.compile(r"(?:^|[\s(（])[가-힣]{0,4}단위\s*[::]")
#   "제 41 기 1분기말 2024.03.31 현재"  "제41기"  "2024.03.31 현재"
_PERIOD_RE = re.compile(r"(제\s*\d+\s*기)|(\d{4}[.\-]\d{2}[.\-]\d{2}\s*(현재|기준))")
_MAX_HINT_LEN = 120  # 이보다 길면 본문이지 hint 가 아니다


def looks_like_unit_line(text: str) -> bool:
    t = text.strip()
    return bool(t) and len(t) <= _MAX_HINT_LEN and bool(_UNIT_RE.search(t))


def looks_like_period_line(text: str) -> bool:
    t = text.strip()
    return bool(t) and len(t) <= _MAX_HINT_LEN and bool(_PERIOD_RE.search(t))


@dataclass
class RawCell:
    text: str
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False
    indent: int = 0          # strip 전 선행 공백 길이(계층 신호)
    field_code: str | None = None
    unit_code: str | None = None
    unit_value: str | None = None


@dataclass
class _GridCell:
    origin_id: int
    cell: TableCell


def expand_grid(raw_rows: list[list[RawCell]]) -> list[list[_GridCell]]:
    """rowspan/colspan 을 채워 넣어 모든 행의 길이가 같은 정규 그리드로 만든다.

    각 칸에 (row, col) 좌표와 dup_left/dup_up 플래그를 심는다. dup_* 는 "같은 원본
    셀이 span 때문에 이 칸에도 복제됐다"는 표시이며, 렌더링에서 빈칸으로 처리해
    **열 수는 유지하면서 텍스트 중복만 없앤다**.
    """
    grid: list[list[_GridCell]] = []
    pending: dict[int, tuple[int, _GridCell]] = {}  # col -> (remaining_rowspan, gridcell)
    next_id = 0

    for raw_row in raw_rows:
        row: list[_GridCell] = []
        col = 0
        idx = 0
        while idx < len(raw_row) or col in pending:
            if col in pending:
                remaining, gc = pending[col]
                # rowspan 복제본은 원본 TableCell 을 공유하면 안 된다(좌표가 달라짐).
                dup = _GridCell(
                    origin_id=gc.origin_id,
                    cell=TableCell(
                        text=gc.cell.text, is_header=gc.cell.is_header,
                        field_code=gc.cell.field_code, unit_code=gc.cell.unit_code,
                        unit_value=gc.cell.unit_value, dup_up=True,
                        indent=gc.cell.indent, origin_id=gc.origin_id,
                    ),
                )
                row.append(dup)
                if remaining - 1 > 0:
                    pending[col] = (remaining - 1, gc)
                else:
                    del pending[col]
                col += 1
                continue
            raw = raw_row[idx]
            idx += 1
            span = max(raw.colspan, 1)
            for c in range(span):
                gc = _GridCell(
                    origin_id=next_id,
                    cell=TableCell(
                        text=raw.text.strip(),
                        is_header=raw.is_header,
                        field_code=raw.field_code,
                        unit_code=raw.unit_code,
                        unit_value=raw.unit_value,
                        dup_left=(c > 0),
                        indent=raw.indent,
                        origin_id=next_id,
                    ),
                )
                row.append(gc)
                if raw.rowspan > 1:
                    pending[col + c] = (raw.rowspan - 1, gc)
            next_id += 1
            col += span
        grid.append(row)

    max_cols = max((len(r) for r in grid), default=0)
    for r_i, row in enumerate(grid):
        while len(row) < max_cols:
            row.append(_GridCell(origin_id=-1, cell=TableCell(text="")))
        for c_i, gc in enumerate(row):
            gc.cell.row = r_i
            gc.cell.col = c_i
    return grid


def _rle(row: list[_GridCell]) -> list[_GridCell]:
    """같은 origin_id(=같은 원본 셀)가 rowspan/colspan 으로 반복된 것을 1개로 축약.
    **분류 판단에만 쓴다** — 저장되는 TableNode.rows 는 정규 그리드를 유지한다."""
    out: list[_GridCell] = []
    for gc in row:
        if not out or out[-1].origin_id != gc.origin_id:
            out.append(gc)
    return out


def _scan_hints(grid: list[list[_GridCell]]) -> tuple[str | None, str | None]:
    """표 안에 흩어져 있는 단위/기수 표기를 찾아 hint 로 승격한다 (변경점 [C])."""
    unit_hint: str | None = None
    period_hint: str | None = None
    seen: set[int] = set()
    for row in grid:
        for gc in row:
            if gc.origin_id in seen or gc.origin_id < 0:
                continue
            seen.add(gc.origin_id)
            t = gc.cell.text.strip()
            if not t:
                continue
            if unit_hint is None and looks_like_unit_line(t):
                unit_hint = t
            if period_hint is None and looks_like_period_line(t):
                period_hint = t
        if unit_hint and period_hint:
            break
    return unit_hint, period_hint


def classify_grid(
    grid: list[list[_GridCell]],
    *,
    title_hint: str | None = None,
    unit_hint: str | None = None,
    acode_group: str | None = None,
) -> list[ContentNode]:
    """정규 그리드를 KeyValueNode / TableNode / TextNode 로 분류한다.

    휴리스틱 (§21):
    - 열 수가 적고(<=KEY_VALUE_MAX_COLS) 행마다 의미가 완결되는 표 -> 행 단위 KeyValueNode.
      rowspan 으로 묶인 첫 컬럼은 group_label 로 별도 보존한다.
    - 열이 많은 grid(재무제표, 임원현황) -> TableNode 그대로 보존.
    - **1열짜리 전폭 행 -> TextNode 로 보존** (변경점 [B]. 기존에는 폐기됐다.)
    """
    # 분류 판단은 RLE 뷰에서 한다: colspan=3 짜리 값 셀 하나가 3열로 보여 실제로는
    # 3칸짜리 key-value 표인데 grid 로 잘못 분류되던 회귀를 그대로 방지한다.
    rle_rows = [_rle(row) for row in grid]
    max_cols = max((len(r) for r in rle_rows), default=0)
    if max_cols == 0:
        return []

    scanned_unit, scanned_period = _scan_hints(grid)

    if max_cols > KEY_VALUE_MAX_COLS:
        # 정규 그리드를 그대로 보존한다 (변경점 [A]).
        rows = [[gc.cell for gc in row] for row in grid]
        return [
            TableNode(
                rows=rows,
                title_hint=title_hint,
                unit_hint=unit_hint or scanned_unit,
                period_hint=scanned_period,
                acode_group=acode_group,
            )
        ]

    nodes: list[ContentNode] = []
    prev_group_origin: int | None = None
    current_kv: KeyValueNode | None = None

    def _new_kv(group_label: str | None) -> KeyValueNode:
        kv = KeyValueNode(group_label=group_label, acode_group=acode_group)
        nodes.append(kv)
        return kv

    for row in rle_rows:
        cells = [c for c in row if c.cell.text or c.origin_id != -1]
        if not cells or all(not c.cell.text for c in cells):
            continue  # 빈 spacer row (실측: 표 첫 행에 흔함)

        if len(cells) >= 3:
            group_cell, key_cell, *value_cells = cells
            value_text = " / ".join(c.cell.text for c in value_cells if c.cell.text)
            if group_cell.origin_id != prev_group_origin or current_kv is None:
                current_kv = _new_kv(group_cell.cell.text or None)
                prev_group_origin = group_cell.origin_id
            last = value_cells[-1] if value_cells else key_cell
            current_kv.pairs.append(
                KVPair(
                    key=key_cell.cell.text,
                    value=value_text,
                    field_code=key_cell.cell.field_code or last.cell.field_code,
                    unit_code=last.cell.unit_code or key_cell.cell.unit_code,
                    unit_value=last.cell.unit_value or key_cell.cell.unit_value,
                )
            )
        elif len(cells) == 2:
            key_cell, value_cell = cells
            if current_kv is None or current_kv.group_label is not None:
                current_kv = _new_kv(None)
                prev_group_origin = None
            current_kv.pairs.append(
                KVPair(
                    key=key_cell.cell.text,
                    value=value_cell.cell.text,
                    field_code=value_cell.cell.field_code or key_cell.cell.field_code,
                    unit_code=value_cell.cell.unit_code,
                    unit_value=value_cell.cell.unit_value,
                )
            )
        else:
            # === 변경점 [B]: 1열짜리 전폭 행을 보존한다 ===
            text = cells[0].cell.text.strip()
            if not text:
                continue
            nodes.append(TextNode(text=text, from_table_row=True))
            # 이 줄 다음부터는 새 key-value 블록으로 본다(서술문이 블록을 가른다).
            current_kv = None
            prev_group_origin = None

    return nodes


# ═══════════════════════════════════════════════════════════════════════════
# semantic block 탐지 — 표 안에서 "의미상 붙어있어야 하는 행 묶음"을 찾는다
#
# 왜 필요한가 (2026-08-25, 상원님 제보 + 실측 재현)
# ------------------------------------------------
# 기존 렌더링은 표를 **행 개수/토큰 예산으로만** 잘랐다. 그래서 재무제표처럼
#   합계
#     1. 매출액 → ... → 계 192,972,588
#     2. 영업이익 → ... → 계  47,206,319
# 로 된 표에서 "1.매출액 계"와 "2.영업이익 계"가 다른 조각으로 갈라진다.
# 갈라진 조각은 상위 라벨("합계")을 잃어서, 검색으로 찾아도 그 숫자가 전사
# 합계인지 특정 지역 값인지 알 수 없다.
#
# 주의: 갈라져도 뒤이은 패킹 단계에서 우연히 다시 합쳐지는 경우가 있다
# (토큰 예산이 남으면). 즉 지금 동작은 **우연에 기대고** 있다. 이 함수는 그걸
# 의도된 동작으로 바꾼다.
# ═══════════════════════════════════════════════════════════════════════════

_KOREAN_ENUM = "가나다라마바사아자차카타파하"

_NUMBERING_PATTERNS = [
    re.compile(r"^\d{1,3}[.)](?=\s|[^\d]|$)"),                    # "1. " / "1)" / "12."
    re.compile(r"^\(\d{1,3}\)"),                                    # "(1)"
    re.compile(rf"^[{_KOREAN_ENUM}][.)](?=\s|[^0-9]|$)"),           # "가." / "나)"
    re.compile(rf"^\([{_KOREAN_ENUM}]\)"),                          # "(가)"
    re.compile(r"^[IVXLCDM]{1,4}[.)](?=\s|$)"),                     # "I." / "IV)"
    re.compile(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ][.)]"),                            # "Ⅰ." (재무제표에 흔함)
]


def is_numbered_label(text: str) -> bool:
    t = text.strip()
    return bool(t) and any(p.match(t) for p in _NUMBERING_PATTERNS)


def strip_numbering_prefix(text: str) -> str:
    """\"1. 매출액\" -> \"매출액\". 매칭 안 되면 원문 그대로."""
    t = text.strip()
    for p in _NUMBERING_PATTERNS:
        m = p.match(t)
        if m:
            return t[m.end():].strip(" .)")
    return t


def _label_info(row: list[TableCell]) -> tuple[str, int, int]:
    """행의 라벨 셀(첫 비어있지 않은 셀)의 (텍스트, 들여쓰기, origin_id).

    dup_left 셀은 같은 행에서 원본(왼쪽)보다 항상 뒤에 오므로 탐색 순서에 영향이
    없다. dup_up 셀(rowspan 반복)이 행의 첫 셀이면 origin_id 를 원본과 똑같이
    물려받으므로 아래 "origin 동일 -> 같은 블록" 규칙이 자연스럽게 걸린다.
    """
    for c in row:
        if c.text.strip():
            return c.text.strip(), c.indent, c.origin_id
    return "", 0, -1


def _is_group_header(row: list[TableCell]) -> bool:
    """값 없이 라벨만 있는 행인가. 표 안의 '구간 제목' 행을 가려낸다.

    dup_left/dup_up(=colspan/rowspan 복제칸)은 세지 않는다. 구간 제목은 보통
    colspan 으로 행 전체를 차지해서, 복제칸을 세면 값이 있는 것처럼 보인다.
    """
    filled = [c for c in row if c.text.strip() and not (c.dup_left or c.dup_up)]
    return len(filled) <= 1


def detect_semantic_blocks(body_rows: list[list[TableCell]]) -> list[list[int]]:
    """본문 행들을 의미 블록(행 index 묶음)으로 나눈다. 원본 순서는 그대로 보존한다.

    3단계로 판정한다 — 위쪽이 더 강한 신호다.

    1) **구간 제목 행** (값 없이 라벨만, 번호 없음, 최상위 들여쓰기)
       실제 재무제표 표는 이 모양이 압도적으로 많다:
           본국(대한민국)          <- 구간 제목
             1. 매출액 / 외부매출액 / 계
             2. 영업이익
             3. 자산
           중국                   <- 다음 구간
       여기서 블록을 "1./2./3." 로 끊으면 '합계' 구간의 매출액과 영업이익이
       갈라진다(2026-08-25 SK하이닉스 재현 케이스). 구간 단위로 묶어야 맞다.
    2) 구간 제목이 없으면 **번호매김 + 들여쓰기** 로 끊는다.
    3) 둘 다 없는 평평한 표는 **행마다 독립 블록** — 기존 동작과 완전히 동일.
    """
    n = len(body_rows)
    if n == 0:
        return []

    infos = [_label_info(r) for r in body_rows]
    numbered = [is_numbered_label(t) for t, _i, _o in infos]
    indents = [i for _t, i, _o in infos]

    # --- 1단계: 구간 제목 행 찾기 ---
    header_like = [
        i for i in range(n)
        if infos[i][0] and not numbered[i] and _is_group_header(body_rows[i])
    ]
    starters: set[int] = set()
    if header_like:
        top = min(indents[i] for i in header_like)
        starters = {i for i in header_like if indents[i] == top}

    if starters:
        blocks: list[list[int]] = []
        for idx in range(n):
            if idx in starters or not blocks:
                blocks.append([idx])
            else:
                blocks[-1].append(idx)
        return blocks

    # --- 2/3단계: 번호매김·들여쓰기 ---
    has_numbering = any(numbered)
    has_indent = len(set(indents)) > 1
    # rowspan 으로 여러 행에 걸친 셀은 원본 표에서 '병합된 한 칸'이다. 번호도
    # 들여쓰기도 없더라도 이건 그 자체로 구조 신호다 — 갈라놓으면 뒤쪽 행이
    # 라벨을 잃는다. (참조 구현에는 이 검사가 빠져 있어 평평한 표로 오인된다.)
    has_span_run = any(
        infos[i][2] != -1 and infos[i][2] == infos[i - 1][2] for i in range(1, n)
    )
    if not has_numbering and not has_indent and not has_span_run:
        return [[i] for i in range(n)]

    blocks = [[0]]
    baseline = infos[0][1]
    for idx in range(1, n):
        text, indent, origin = infos[idx]
        prev_origin = infos[idx - 1][2]
        if text == "":
            blocks[-1].append(idx); continue
        if origin != -1 and origin == prev_origin:
            blocks[-1].append(idx); continue
        if has_numbering:
            starts_new = numbered[idx] and indent <= baseline
        elif has_indent:
            gmin = min(indents)
            starts_new = indent == gmin and any(infos[j][1] > gmin for j in blocks[-1])
        else:
            # 신호가 rowspan 뿐인 표. 위에서 '같은 origin -> 같은 블록'을 이미
            # 처리했으므로, 여기 왔다는 건 원본 셀이 바뀌었다는 뜻 = 새 블록.
            starts_new = True
        if starts_new:
            blocks.append([idx]); baseline = indent
        else:
            blocks[-1].append(idx)
    return blocks
