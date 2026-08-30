"""공통 Chunk Schema (§24) + node -> 검색용 text 렌더링 (§21, §26).

Parsing 방식은 4종 공시마다 다르지만, Retrieval 이후 사용하는 Chunk 구조는
전부 이 스키마로 통일한다. 실제 데이터에서 알 수 없는 값은 hallucinate 하지
않고 None(null) 을 쓴다.

=== KIM 브랜치 변경점 (2026-08-23) ===

[A] field_codes 를 dict[str,str] -> list[FieldRef] 로 재설계했다.
    기존은 `{셀 텍스트: 코드}` 평면 dict 라 (1) 위치 정보가 없고 (2) 같은 텍스트면
    덮어써졌다. 실측 소실률: major 80.1% / holding 40.5% / periodic 33.9%.
    그리고 AUNITVALUE(=DART 가 이미 정규화해준 값, 예 "20230421")는 100% 폐기됐다.
    이제 (code, unit, unit_value, text, row, col) 을 리스트로 보존한다.

[B] estimate_tokens 를 주입 가능하게 만들었다.
    기존 상수 2.0 은 주석 스스로 "실측 아님"이라 인정한 값인데, 같은 코퍼스 안에서
    산문(한글 68%/숫자 4%)과 표(한글 33%/숫자 20%)의 문자 구성이 크게 다르다.
    set_token_counter() 로 실제 BGE-M3 토크나이저를 꽂을 수 있게 하고,
    꽂지 않았을 때만 heuristic 을 쓴다(그리고 heuristic 임을 명시한다).
    -> scripts/calibrate_tokens.py 로 1회 보정 가능.

[C] render_table_node 가 **열 정렬을 보존**한다.
    정규 그리드를 그대로 " | " 로 내되, span 복제 칸은 빈칸으로 낸다.
    -> 열 수가 모든 행에서 동일해지므로 "N번째 필드 = 헤더 N번째 열"이 성립한다.
    단위/기수 표기를 preamble 로 반복 삽입한다(기존엔 unit_hint 가 항상 None 이라
    이 로직이 죽은 코드였다).
"""

from __future__ import annotations

from typing import Callable, Literal

from pydantic import BaseModel, Field

from disclosure_rag.parsing.table_parser import detect_semantic_blocks
from disclosure_rag.common.doc_tree import KeyValueNode, TableNode, TextNode

ContentType = Literal["text", "table", "key_value"]


class FieldRef(BaseModel):
    """DART 의 기계판독 속성 1건. 위치를 함께 보존해 덮어쓰기 소실을 막는다."""

    code: str | None = None        # TE[ACODE]
    unit: str | None = None        # TU[AUNIT]
    unit_value: str | None = None  # TU[AUNITVALUE] — 정규화된 값 (예: "20230421")
    text: str | None = None        # 셀 원문 텍스트
    row: int | None = None
    col: int | None = None
    key: str | None = None         # key-value 표에서의 key (해당 시)


class ChunkSchema(BaseModel):
    chunk_id: str
    report_id: str  # manifest doc_id
    parent_chunk_id: str | None = None

    text: str  # embedding/BM25 에 쓰는, [회사]/[공시]/[Section] 컨텍스트 포함 검색용 텍스트
    raw_text: str  # 컨텍스트 헤더 없는 순수 chunk 내용 (재조합/디버깅용)

    company: str | None = None
    corp_code: str | None = None

    report_type: str | None = None      # doc_group: periodic|major|exchange|holding
    report_subtype: str | None = None   # doc_subtype 또는 report_subtype(main/attachment 등)
    report_name: str | None = None      # document_name / report_nm

    period: str | None = None           # 예: "2024-12"
    filing_date: str | None = None      # rcept_dt YYYYMMDD

    section_path: list[str] = Field(default_factory=list)

    content_type: ContentType = "text"

    source_path: str | None = None

    is_correction: bool = False
    # 이 조각이 어느 표(들)에서 왔는지. 같은 표가 여러 조각으로 쪼개진 경우
    # 서로를 찾아 근거를 이어붙이기 위한 열쇠다(형제 조각 확장).
    # 값은 문서 안에서만 유일하면 되므로 "<doc_id>::T<n>" 형태로 만든다.
    table_ids: list[str] = Field(default_factory=list)
    correction_group_id: str | None = None
    correction_order: int | None = None
    is_latest: bool | None = None

    # DART TE[ACODE]/TU[AUNIT/AUNITVALUE] 구조화 필드 — 위치까지 보존 (변경점 [A])
    field_codes: list[FieldRef] = Field(default_factory=list)

    # 표 청크가 어떤 단위/기준기간 아래에 있었는지 (변경점 [C] 의 부산물).
    unit_hint: str | None = None
    period_hint: str | None = None


def filter_leaf_chunks(chunks: list["ChunkSchema"]) -> list["ChunkSchema"]:
    """BM25/Dense 인덱스에 넣을 "검색 대상" chunk 만 남긴다.

    §13/§20: Parent 는 Context 확장용이지 검색 대상이 아니다. "이 chunk_id 가
    다른 chunk 의 parent_chunk_id 로 나타나면 parent" 라는 규칙으로 leaf 를 가려낸다.

    주의: parent 를 **버리는 것이 아니라 인덱싱에서만 제외**한다. parent 본문은
    L1 스냅샷(chunks.jsonl)에 전량 저장되며 Parent Expansion 이 그걸 조회한다.
    (기존 코드는 parent 를 만들고 어디에도 저장하지 않아 전체 청크 문자의 49.4%를
     생성 즉시 폐기했고, Qdrant payload 의 parent_chunk_id 로 조회할 대상이 없었다.)
    """
    referenced_as_parent = {c.parent_chunk_id for c in chunks if c.parent_chunk_id}
    return [c for c in chunks if c.chunk_id not in referenced_as_parent]


# ---------------------------------------------------------------- 토큰 추정
TOKEN_CHARS_PER_TOKEN = 2.0  # heuristic 기본값 (실측 아님 — 아래 set_token_counter 참조)

_token_counter: Callable[[str], int] | None = None


def set_token_counter(fn: Callable[[str], int] | None) -> None:
    """실제 토크나이저를 주입한다. 예:

        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        set_token_counter(lambda s: len(tk.encode(s, add_special_tokens=False)))

    주입하지 않으면 heuristic 으로 동작하며, 그 사실이 build_manifest 에 기록된다.
    """
    global _token_counter
    _token_counter = fn


def token_counter_is_exact() -> bool:
    return _token_counter is not None


def estimate_tokens(text: str) -> int:
    """실제 토크나이저가 주입돼 있으면 그것을, 아니면 heuristic 을 쓴다."""
    if _token_counter is not None:
        return max(1, _token_counter(text))
    return max(1, int(len(text) / TOKEN_CHARS_PER_TOKEN))


# ---------------------------------------------------------------- 렌더링
def render_text_node(node: TextNode) -> str:
    return node.text


def render_kv_node(node: KeyValueNode) -> str:
    lines: list[str] = []
    if node.group_label:
        lines.append(f"[{node.group_label}]")
    for p in node.pairs:
        # AUNITVALUE 가 셀 텍스트와 다른 정보를 담고 있으면 함께 노출한다.
        # 예: 텍스트 "2023년 04월 21일" / unit_value "20230421"
        if p.unit_value and p.unit_value not in p.value:
            lines.append(f"- {p.key}: {p.value} ({p.unit_value})")
        else:
            lines.append(f"- {p.key}: {p.value}")
    return "\n".join(lines)


def _cell_text(cell) -> str:
    """span 으로 복제된 칸은 빈칸으로 낸다 — 열 수는 유지하고 텍스트만 생략."""
    if cell.dup_left or cell.dup_up:
        return ""
    return cell.text


TableStyle = Literal["grid", "kv"]


def _render_row_kv(row, header_cells) -> str:
    """행 하나를 `컬럼명: 값` 블록으로 낸다 (structure-aware chunking 계열 표현).

    arXiv 2605.00318(STC)은 표를 행 단위 key-value 블록으로 표현하면 검색이 크게
    오른다고 보고한다(MAUD 39,231건, hybrid MRR 0.358 -> 0.595, BM25 R@1 0.366 -> 0.754).
    단 같은 논문이 **KV 표현만 하고 구조 인식 분할을 안 하면 오히려 baseline 보다
    나빴다**고도 보고한다 — 표현이 아니라 "행 경계를 지키는 분할"이 이득의 본체다.
    우리는 이미 행 경계를 지키므로, 여기서는 표현 축만 A/B 로 비교한다.
    """
    parts = []
    for j, cell in enumerate(row):
        if cell.dup_left or cell.dup_up or not cell.text:
            continue
        head = header_cells[j] if j < len(header_cells) else ""
        parts.append(f"{head}: {cell.text}" if head else cell.text)
    return " / ".join(parts)


def render_table_node(
    node: TableNode,
    *,
    max_rows_per_chunk: int = 20,
    max_tokens_per_chunk: int = 1000,
    style: TableStyle = "grid",
) -> list[str]:
    """정규 그리드를 Title+Unit+Header+N행 단위로 나누고, 분할된 모든 조각에
    Title/Unit/Header 를 반복 삽입한다 (§21 "다음 Chunk 에도 정보를 반복한다").

    열 수가 모든 행에서 동일하므로 " | " 로 이어붙여도 헤더와의 대응이 유지된다
    (변경점 [C]). max_rows/max_tokens 를 크게 주면 표 전체를 1개로 낸다(parent 용).
    """
    if not node.rows:
        return []

    header_idx = {i for i, row in enumerate(node.rows) if row and all(c.is_header for c in row if c.text)}
    if header_idx:
        header_rows = [node.rows[i] for i in sorted(header_idx)]
        body_rows = [row for i, row in enumerate(node.rows) if i not in header_idx]
    else:
        header_rows = [node.rows[0]]
        body_rows = node.rows[1:]

    # 헤더 라벨: 여러 헤더 행이 있으면 열별로 이어붙여 하나의 라벨로 만든다
    header_labels: list[str] = []
    ncol = len(node.rows[0])
    for j in range(ncol):
        parts = []
        for hr in header_rows:
            if j < len(hr) and hr[j].text and not (hr[j].dup_left or hr[j].dup_up):
                parts.append(hr[j].text)
        header_labels.append(" ".join(parts))

    def fmt_row(row) -> str:
        if style == "kv":
            return _render_row_kv(row, header_labels)
        return " | ".join(_cell_text(c) for c in row)

    # kv 표현에서는 헤더 라벨이 각 행에 이미 붙으므로 헤더 행을 따로 반복하지 않는다
    header_lines = [] if style == "kv" else [" | ".join(_cell_text(c) for c in r) for r in header_rows]
    preamble_parts = [p for p in (node.title_hint, node.unit_hint, node.period_hint) if p]
    preamble = "\n".join(preamble_parts)

    fixed_overhead = estimate_tokens(preamble) + sum(estimate_tokens(h) for h in header_lines)

    # ── 의미 블록 우선 패킹 ────────────────────────────────────────────────
    # 우선순위: 1) 의미 블록 안 찢기 > 2) max_tokens 예산 >
    #           3) (구조 신호가 없는 평평한 표에서만) max_rows fallback
    #
    # 2026-08-25: 이전에는 행 개수/토큰만 보고 잘라서, 재무제표의
    # "합계 > 1.매출액 ... 계 / 2.영업이익 ... 계" 가 다른 조각으로 갈라졌다.
    # 갈라진 조각은 상위 라벨("합계")을 잃어 숫자의 소속을 알 수 없게 된다.
    def render_lines(extra_label: str | None, rows) -> str:
        parts = []
        if preamble:
            parts.append(preamble)
        parts.extend(header_lines)
        if extra_label:
            parts.append(f"[{extra_label}]")
        parts.extend(fmt_row(r) for r in rows)
        return "\n".join(parts)

    chunks: list[str] = []

    if body_rows:
        row_blocks = detect_semantic_blocks(body_rows)
        has_structure = any(len(b) > 1 for b in row_blocks)

        if not has_structure:
            # 평평한 표 -> 기존 동작 그대로(행 단위 max_rows + max_tokens)
            group, group_tokens = [], fixed_overhead
            for row in body_rows:
                row_tokens = estimate_tokens(fmt_row(row))
                too_many_rows = len(group) >= max_rows_per_chunk
                too_many_tokens = bool(group) and (group_tokens + row_tokens > max_tokens_per_chunk)
                if group and (too_many_rows or too_many_tokens):
                    chunks.append(render_lines(None, group))
                    group, group_tokens = [], fixed_overhead
                group.append(row)
                group_tokens += row_tokens
            if group:
                chunks.append(render_lines(None, group))
        else:
            # 블록 단위 패킹. max_rows 는 쓰지 않는다 — 블록을 쪼개는 유일한 이유는
            # 토큰 예산뿐이어야 한다.
            def block_label(block) -> str:
                for i in block:
                    lab = next((c.text.strip() for c in body_rows[i] if c.text.strip()), "")
                    if lab:
                        return lab
                return ""

            group_rows, group_tokens = [], fixed_overhead
            for block in row_blocks:
                b_rows = [body_rows[i] for i in block]
                b_tokens = sum(estimate_tokens(fmt_row(r)) for r in b_rows)

                if b_tokens + fixed_overhead > max_tokens_per_chunk:
                    # 블록 하나가 예산을 넘김 -> 쌓인 것 먼저 내보내고 이 블록만 쪼갠다.
                    # 쪼갠 모든 조각에 "[블록라벨 i/n]" 을 넣어 '계'만 남아도 소속을 안다.
                    if group_rows:
                        chunks.append(render_lines(None, group_rows))
                        group_rows, group_tokens = [], fixed_overhead
                    label = block_label(block)
                    subs, cur, cur_tok = [], [], fixed_overhead
                    for r in b_rows:
                        rt = estimate_tokens(fmt_row(r))
                        if cur and cur_tok + rt > max_tokens_per_chunk:
                            subs.append(cur); cur, cur_tok = [], fixed_overhead
                        cur.append(r); cur_tok += rt
                    if cur:
                        subs.append(cur)
                    total = len(subs)
                    for i, sub in enumerate(subs, start=1):
                        tag = f"{label} [{i}/{total}]" if label else f"[{i}/{total}]"
                        chunks.append(render_lines(tag, sub))
                    continue

                if group_rows and group_tokens + b_tokens > max_tokens_per_chunk:
                    chunks.append(render_lines(None, group_rows))
                    group_rows, group_tokens = [], fixed_overhead
                group_rows.extend(b_rows)
                group_tokens += b_tokens
            if group_rows:
                chunks.append(render_lines(None, group_rows))

    if not chunks:
        # body 가 없는 표(header/preamble 만) -> header-only chunk 1개는 보존
        parts = ([preamble] if preamble else []) + header_lines
        chunks.append("\n".join(parts))
    return chunks


def table_field_refs(node: TableNode) -> list[FieldRef]:
    """표에서 기계판독 속성을 위치까지 붙여 뽑는다 (변경점 [A])."""
    refs: list[FieldRef] = []
    for row in node.rows:
        for cell in row:
            if cell.dup_left or cell.dup_up:
                continue
            if cell.field_code or cell.unit_code or cell.unit_value:
                refs.append(
                    FieldRef(
                        code=cell.field_code, unit=cell.unit_code, unit_value=cell.unit_value,
                        text=cell.text or None, row=cell.row, col=cell.col,
                    )
                )
    return refs


def kv_field_refs(node: KeyValueNode) -> list[FieldRef]:
    refs: list[FieldRef] = []
    for p in node.pairs:
        if p.field_code or p.unit_code or p.unit_value:
            refs.append(
                FieldRef(
                    code=p.field_code, unit=p.unit_code, unit_value=p.unit_value,
                    text=p.value or None, key=p.key or None,
                )
            )
    return refs


def render_search_text(
    *,
    company: str | None,
    report_name: str | None,
    period: str | None,
    section_path: list[str],
    body_text: str,
) -> str:
    """§26: 검색용 텍스트에 [회사]/[공시]/[기간]/[Section]/[내용] 컨텍스트를 포함한다.

    문헌 근거: 문서 수준 메타데이터를 청크에 부착하면 금융문서 QA 정확도가
    50~60% -> 72~75% 로 오른다(Snowflake, SEC 23,000 PDF / 500 질의).
    """
    lines = []
    if company:
        lines.append(f"[회사]\n{company}")
    if report_name:
        lines.append(f"[공시]\n{report_name}")
    if period:
        lines.append(f"[기간]\n{period}")
    if section_path:
        lines.append("[Section]\n" + " > ".join(section_path))
    lines.append(f"[내용]\n{body_text}")
    return "\n\n".join(lines)
