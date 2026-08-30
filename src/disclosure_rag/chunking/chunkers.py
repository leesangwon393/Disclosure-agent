"""공시 유형별 Chunker (§12~20, Phase 5) + 공통 Chunk Schema 변환 (Phase 6).

- periodic / holding: Section 을 Parent 로, 그 아래 content 를 Child 로 묶는
  Parent-Child 구조 (§13, §20). Parent 자체는 검색 대상이 아니라 Context 확장용
  이며, "이 chunk_id 가 다른 chunk 의 parent_chunk_id 로 나타나면 parent" 라는
  규칙으로 구분한다(별도 스키마 필드 없이 §24 스키마 그대로 사용).
- major: 이벤트 단위 (§15). 문서가 짧으면 통째로 1 chunk, 길면 TABLE-GROUP 의
  group_label 단위(이미 "결정내용/금액/일정" 처럼 의미 단위로 쪼개져 있음)로 나눈다.
- exchange: 원문 그대로 1 chunk 를 우선한다 (§18). 표본 실측상 평균 9.7KB 로
  거의 항상 여기 해당한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from disclosure_rag.chunking.chunk_schema import ChunkSchema, estimate_tokens, render_search_text
from disclosure_rag.chunking.packer import pack_nodes
from disclosure_rag.common.doc_tree import ContentNode, ParsedDocument, SectionNode, TableNode
from disclosure_rag.common.manifest_loader import ManifestRow
from disclosure_rag.correction.correction_graph_builder import CorrectionRecord

CHILD_TARGET_TOKENS = 600
CHILD_MAX_TOKENS = 1000
WHOLE_DOC_MAX_TOKENS = 1000  # 이보다 작으면 major/exchange 는 "공시 1건 = chunk 1개"


@dataclass(frozen=True)
class ChunkConfig:
    """청킹 파라미터를 한 곳에 모은다 (실험 축).

    === 2026-08-23 실험으로 확정 (scripts/exp_chunk_size.py, 실제 BGE-M3 토크나이저) ===
    조건 12개(크기 6 x 표표현 2) / 회사 10 / 문서 587 / 조각수준 질의 314.
    Stage 1 의 결함 3개(baseline 헤더 혼입, 1개사, 스크립트 미보존)를 모두 회피했다.

    **크기: 600 유지.** MRR 범위가 0.389~0.422, 폭이 표준오차의 1.4배라 **차이가
    노이즈다.** 300~1200 사이 어디를 골라도 통계적으로 같다(1800 만 명확히 나쁘다 —
    hit@10 0.586). 성능으로 못 고르므로 비용으로 고른다:
      - 큰 청크(1200)는 임베딩 1회를 44% 아끼지만(leaf 32k vs 57k)
      - 작은 청크(600)는 **매 질의마다** 넘기는 컨텍스트가 절반이다(p50 1,242자 vs 2,296자)
    임베딩은 한 번이고 서빙은 매번이므로 600 이 유리하다. hit@5(0.554)·hit@10(0.656)
    도 600 이 1위였다.

    **표 표현: grid -> kv 로 변경.** 같은 크기끼리 짝지어 비교했을 때
    **hit@5 가 6/6 쌍 전부 kv 우세**(평균 +0.019, 부호검정 p=0.031), MRR 5/6 쌍 우세.
    크기가 클수록 격차가 커진다(1800에서 MRR +0.026) — 행이 많은 표에서 grid 는
    열 위치로만 헤더를 참조하는데 kv 는 행마다 컬럼명을 붙이기 때문으로 보인다.
    arXiv 2605.00318(STC)의 방향과 일치한다.
    """

    target_tokens: int = CHILD_TARGET_TOKENS
    max_tokens: int = CHILD_MAX_TOKENS
    whole_doc_max_tokens: int = WHOLE_DOC_MAX_TOKENS
    table_max_rows: int = 20
    table_style: str = "kv"  # "grid" | "kv" — 2026-08-23 실험으로 kv 채택

    @property
    def label(self) -> str:
        return f"t{self.target_tokens}_m{self.max_tokens}_r{self.table_max_rows}_{self.table_style}"


DEFAULT_CHUNK_CONFIG = ChunkConfig()

_TOC_DASH_RATIO_MIN = 0.6
_TOC_PAGENUM_RATIO_MIN = 0.6
_TOC_MIN_ROWS = 3
_TOC_DASH_LEADER_MIN_LEN = 10  # "리더 점선"(예: 35자 대시)과 재무표의 "-"(결측/0 표기) 를 구분


def _is_toc_table(node: ContentNode) -> bool:
    """목차(TOC) 페이지가 표(TableNode)로 파싱된 경우를 감지한다.

    실측(2026-08-15, 대회 참고 질의 스모크테스트): periodic 문서에서 SECTION
    밖 loose content 로 존재하는 목차 표가 "SECTION 밖 loose content 보존"
    수정(§10) 덕에 더 이상 유실되지는 않지만, 그대로 인덱싱하면 [제목 | 빈칸 |
    '---------------------------------' | 페이지번호] 행이 반복되는 텍스트가
    실제 section 제목과 키워드가 그대로 겹쳐 BM25에서 허위로 높은 점수를 받고,
    진짜 본문 chunk 를 top-k 밖으로 밀어낸다(실제 질의 "핵심 사업"에서 재현
    확인됨). 목차는 원문에 있는 모든 제목이 실제 section 으로도 그대로
    존재하므로 검색 인덱스에서 빼도 정보 손실이 없다 — 파싱 트리 자체에서는
    빼지 않고(§9-2 원문 보존 원칙 유지), chunk 로 만들 content 후보에서만 제외.

    주의(회귀 발견): 처음엔 셀 텍스트가 "-"로만 이루어졌는지만 봤는데, 재무표에서
    결측/0 을 "-" 한 글자로 표기하는 관행과 구분이 안 돼 내부통제 인력 현황 같은
    진짜 데이터 표를 오탐했다(실측: `periodic_20250319000665` "2. 내부통제에
    관한 사항"). 목차의 "리더 점선"은 길게 이어진 대시(예: 35자)라는 게 실측으로
    확인된 차이라 최소 길이 조건(`_TOC_DASH_LEADER_MIN_LEN`)을 추가해 구분한다."""
    if not isinstance(node, TableNode) or len(node.rows) < _TOC_MIN_ROWS:
        return False
    n = len(node.rows)
    dash_hits = 0
    pagenum_hits = 0
    for row in node.rows:
        texts = [c.text.strip() for c in row]
        if any(len(t) >= _TOC_DASH_LEADER_MIN_LEN and set(t) == {"-"} for t in texts):
            dash_hits += 1
        if texts and texts[-1].isdigit():
            pagenum_hits += 1
    return (dash_hits / n) >= _TOC_DASH_RATIO_MIN and (pagenum_hits / n) >= _TOC_PAGENUM_RATIO_MIN


def _period_str(row: ManifestRow) -> str | None:
    if row.base_year is None:
        return None
    if row.base_month is None:
        return str(row.base_year)
    return f"{row.base_year}-{row.base_month:02d}"


def _base_fields(row: ManifestRow, correction: CorrectionRecord, report_name: str | None) -> dict:
    return dict(
        report_id=row.doc_id,
        company=row.corp_name,
        corp_code=row.corp_code,
        report_type=row.doc_group,
        report_name=report_name or row.report_nm,
        period=_period_str(row),
        filing_date=row.rcept_dt,
        is_correction=row.is_correction,
        correction_group_id=correction.correction_group_id,
        correction_order=correction.correction_order,
        is_latest=correction.is_latest,
    )


def _content_bearing_sections(sections: list[SectionNode]):
    """direct content(ContentNode) 를 가진 SectionNode 를 전부 순회한다 (Parent 후보).
    목차 표(_is_toc_table)는 검색 인덱스 후보에서 제외한다(원문 트리 자체는 그대로 유지)."""
    for s in sections:
        direct_content = [c for c in s.children if not isinstance(c, SectionNode) and not _is_toc_table(c)]
        if direct_content:
            yield s, direct_content
        sub_sections = [c for c in s.children if isinstance(c, SectionNode)]
        yield from _content_bearing_sections(sub_sections)


def chunk_parent_child(
    parsed: ParsedDocument, row: ManifestRow, correction: CorrectionRecord,
    cfg: ChunkConfig = DEFAULT_CHUNK_CONFIG,
) -> list[ChunkSchema]:
    """periodic / holding 공용: Section = Parent, 그 아래 packed unit = Child (§13, §20)."""
    base = _base_fields(row, correction, parsed.document_name)
    chunks: list[ChunkSchema] = []
    parent_counter = 0

    for section, direct_content in _content_bearing_sections(parsed.sections):
        parent_counter += 1
        parent_id = f"{row.doc_id}::{parsed.report_subtype}::P{parent_counter}"
        section_path = section.path

        # Parent: Context 확장용 — 해당 section 의 전체 raw 내용을 그대로 이어붙인다.
        packed_full = pack_nodes(
            direct_content, target_tokens=10**9, max_tokens=10**9,
            table_max_rows=10**9, table_max_tokens=10**9, table_style=cfg.table_style,
        )
        parent_raw = "\n\n".join(u.text for u in packed_full) if packed_full else section.title
        parent_content_type = packed_full[0].content_type if packed_full else "text"
        chunks.append(
            ChunkSchema(
                chunk_id=parent_id,
                parent_chunk_id=None,
                raw_text=parent_raw,
                text=render_search_text(
                    company=base["company"], report_name=base["report_name"],
                    period=base["period"], section_path=section_path, body_text=parent_raw,
                ),
                section_path=section_path,
                content_type=parent_content_type,
                source_path=parsed.source_path,
                field_codes=[r for u in packed_full for r in u.field_refs],
                unit_hint=next((u.unit_hint for u in packed_full if u.unit_hint), None),
                period_hint=next((u.period_hint for u in packed_full if u.period_hint), None),
                **base,
            )
        )

        # Child: 실제 검색 대상, target/max token 예산으로 쪼갠다.
        child_units = pack_nodes(
            direct_content, target_tokens=cfg.target_tokens, max_tokens=cfg.max_tokens,
            table_max_rows=cfg.table_max_rows, table_style=cfg.table_style,
        )
        for i, unit in enumerate(child_units, start=1):
            child_id = f"{parent_id}::C{i}"
            chunks.append(
                ChunkSchema(
                    chunk_id=child_id,
                    parent_chunk_id=parent_id,
                    raw_text=unit.text,
                    text=render_search_text(
                        company=base["company"], report_name=base["report_name"],
                        period=base["period"], section_path=section_path, body_text=unit.text,
                    ),
                    section_path=section_path,
                    content_type=unit.content_type,
                    source_path=parsed.source_path,
                    field_codes=unit.field_refs,
                    unit_hint=unit.unit_hint,
                    period_hint=unit.period_hint,
                    # 표 순번은 pack_nodes 호출 단위(=섹션)로 매겨지므로
                    # parent_id 를 접두해 문서 안에서 유일하게 만든다.
                    table_ids=[f"{parent_id}::{k}" for k in unit.table_keys],
                    **base,
                )
            )

    return chunks


def _flatten_with_section(sections: list[SectionNode]) -> list[tuple[list[str], object]]:
    out: list[tuple[list[str], object]] = []
    for s in sections:
        for c in s.children:
            if isinstance(c, SectionNode):
                out.extend(_flatten_with_section([c]))
            elif not _is_toc_table(c):
                out.append((s.path, c))
    return out


def chunk_flat_whole_doc_preferred(
    parsed: ParsedDocument, row: ManifestRow, correction: CorrectionRecord,
    cfg: ChunkConfig = DEFAULT_CHUNK_CONFIG,
) -> list[ChunkSchema]:
    """major / exchange 공용: "공시 1건 = chunk 1개" 를 우선하고(§15, §18),
    너무 길 때만 group_label/문단 단위(이미 packer 가 나누는 단위)로 쪼갠다.
    Parent-Child 를 쓰지 않는다 — 전부 top-level(검색 대상) chunk."""
    base = _base_fields(row, correction, parsed.document_name)
    node_section_pairs = _flatten_with_section(parsed.sections)
    if not node_section_pairs:
        return []

    all_nodes = [n for _sp, n in node_section_pairs]
    whole_units = pack_nodes(
        all_nodes, target_tokens=10**9, max_tokens=10**9,
        table_max_rows=10**9, table_max_tokens=10**9, table_style=cfg.table_style,
    )
    whole_text = "\n\n".join(u.text for u in whole_units)

    chunks: list[ChunkSchema] = []
    # 대표 section_path: 실제 content 가 있는 마지막 section (major 의 첫 section 은
    # 보통 고정 안내문 뿐이라 content 가 없고, 두번째 section 이 진짜 이벤트다 — 실측 §4).
    representative_section_path = node_section_pairs[-1][0]

    if estimate_tokens(whole_text) <= cfg.whole_doc_max_tokens:
        field_codes = [r for u in whole_units for r in u.field_refs]
        content_type = whole_units[0].content_type if whole_units else "text"
        chunks.append(
            ChunkSchema(
                chunk_id=f"{row.doc_id}::{parsed.report_subtype}::C1",
                parent_chunk_id=None,
                raw_text=whole_text,
                text=render_search_text(
                    company=base["company"], report_name=base["report_name"],
                    period=base["period"], section_path=representative_section_path, body_text=whole_text,
                ),
                section_path=representative_section_path,
                content_type=content_type,
                source_path=parsed.source_path,
                field_codes=field_codes,
                unit_hint=next((u.unit_hint for u in whole_units if u.unit_hint), None),
                period_hint=next((u.period_hint for u in whole_units if u.period_hint), None),
                **base,
            )
        )
    else:
        units = pack_nodes(
            all_nodes, target_tokens=cfg.target_tokens, max_tokens=cfg.max_tokens,
            table_max_rows=cfg.table_max_rows, table_style=cfg.table_style,
        )
        for i, unit in enumerate(units, start=1):
            chunks.append(
                ChunkSchema(
                    chunk_id=f"{row.doc_id}::{parsed.report_subtype}::C{i}",
                    parent_chunk_id=None,
                    raw_text=unit.text,
                    text=render_search_text(
                        company=base["company"], report_name=base["report_name"],
                        period=base["period"], section_path=representative_section_path, body_text=unit.text,
                    ),
                    section_path=representative_section_path,
                    content_type=unit.content_type,
                    source_path=parsed.source_path,
                    field_codes=unit.field_refs,
                    unit_hint=unit.unit_hint,
                    period_hint=unit.period_hint,
                    table_ids=[f"{row.doc_id}::{parsed.report_subtype}::{k}"
                               for k in unit.table_keys],
                    **base,
                )
            )
    return chunks


def chunk_document(
    parsed: ParsedDocument, row: ManifestRow, correction: CorrectionRecord,
    cfg: ChunkConfig = DEFAULT_CHUNK_CONFIG,
) -> list[ChunkSchema]:
    if parsed.doc_group in ("periodic", "holding"):
        return chunk_parent_child(parsed, row, correction, cfg)
    if parsed.doc_group in ("major", "exchange"):
        return chunk_flat_whole_doc_preferred(parsed, row, correction, cfg)
    raise ValueError(f"알 수 없는 doc_group: {parsed.doc_group}")
