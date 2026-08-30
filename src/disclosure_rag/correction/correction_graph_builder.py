"""Correction Graph Builder (§27~29, Phase 7).

Correction Resolver 는 Retriever 와 역할이 다르다 — "어떤 Chunk 가 관련있는가"가
아니라 "어떤 Version 의 공시를 써야 하는가"를 결정한다. 따라서 similarity 로
최신 여부를 판단하지 않고, 여기서 만든 metadata graph 로만 결정한다 (§29 금지).

전략은 doc_group 별로 다르다 (Phase 0 실측 근거):

- periodic: (corp_name, doc_subtype, base_year, base_month) 키로 묶으면
  collision 이 0건이었다 (§7 실측) — manifest 필드만으로 100% 안전하게 묶인다.
  이 방식은 pdf+html 대체수집 문서(KB금융/한화오션 정정)도 텍스트 파싱 없이
  자동으로 처리된다 (텍스트 정규식이 필요 없으므로 사용자 결정 #4 의 blocker 가
  애초에 발생하지 않음).

- major / exchange / holding: 이런 안전한 manifest 키가 없다 (major 는
  doc_subtype 이 전부 None, holding 은 flr_nm 이 다수, exchange 는 짧은
  이벤트 반복). 대신 본문의 "정정대상 공시서류의 최초제출일" 텍스트(§7 실측
  99.9% 성공)로 원본의 rcept_dt 를 역산해 후보를 찾고, (corp_name, rcept_dt)
  가 겹치는 4~9% 구간(§7 실측 collision)은 "정정대상 공시서류" 제목을 후보들의
  report_nm 과 fuzzy 매칭해 타이브레이크한다. holding 은 추가로 flr_nm 을
  키에 포함한다(§29 대량보유자별 독립 보고).

절대 금지: "제목 하나만 보고 무조건 연결"(사용자 결정 #5 참고 규칙) — fuzzy
매칭은 collision 발생 시의 최후 타이브레이커일 뿐, 1차 키는 항상 날짜다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from disclosure_rag.common.manifest_loader import ManifestRow
from disclosure_rag.common.unicode_utils import PathResolver
from disclosure_rag.correction.correction_extractor import extract_correction_info
from disclosure_rag.correction.overrides import MANUAL_TARGET_DATE_OVERRIDES

logger = logging.getLogger(__name__)

FUZZY_MIN_SCORE = 40.0  # 이 밑이면 티어브레이크 결과를 신뢰하지 않고 unresolved 로 남김


@dataclass
class CorrectionRecord:
    doc_id: str
    correction_group_id: str
    correction_order: int
    is_correction: bool
    is_latest: bool
    resolution_source: str  # manifest_key | rule | manual_override | fuzzy_tiebreak | unresolved | unsupported_format | original


def _strip_correction_prefix(report_nm: str) -> str:
    return report_nm.replace("[기재정정]", "").replace("[첨부추가]", "").strip()


def _finalize_groups(members: dict[str, list[ManifestRow]], source_by_doc: dict[str, str]) -> dict[str, CorrectionRecord]:
    records: dict[str, CorrectionRecord] = {}
    for group_id, rows in members.items():
        rows_sorted = sorted(rows, key=lambda r: (r.rcept_dt, r.rcept_no))
        n = len(rows_sorted)
        for i, row in enumerate(rows_sorted):
            records[row.doc_id] = CorrectionRecord(
                doc_id=row.doc_id,
                correction_group_id=group_id,
                correction_order=i,
                is_correction=row.is_correction,
                is_latest=(i == n - 1),
                resolution_source=source_by_doc.get(row.doc_id, "original"),
            )
    return records


def _build_periodic_groups(rows: list[ManifestRow]) -> dict[str, CorrectionRecord]:
    groups: dict[tuple, list[ManifestRow]] = {}
    for r in rows:
        key = (r.corp_name, r.doc_subtype, r.base_year, r.base_month)
        groups.setdefault(key, []).append(r)

    members: dict[str, list[ManifestRow]] = {}
    source_by_doc: dict[str, str] = {}
    for key, group_rows in groups.items():
        group_rows_sorted = sorted(group_rows, key=lambda r: (r.rcept_dt, r.rcept_no))
        group_id = group_rows_sorted[0].doc_id
        members[group_id] = group_rows_sorted
        for r in group_rows_sorted:
            source_by_doc[r.doc_id] = "manifest_key"
    return _finalize_groups(members, source_by_doc)


def _build_event_groups(
    rows: list[ManifestRow],
    doc_group: str,
    resolver: PathResolver,
) -> dict[str, CorrectionRecord]:
    """정정 체인을 구축한다.

    실측 결과(Phase 0 은 periodic 만 검증했었는데, 이번 실행에서 major/exchange/
    holding 은 다른 동작을 함을 발견): periodic 은 모든 정정이 "최초 원본"의
    제출일을 직접 가리키지만, major/exchange/holding 은 **바로 직전 정정본**을
    가리키는 다단 체인(chain)이 존재한다 (예: 한화오션 exchange 정정이 또 다른
    exchange 정정을 가리킴). 그래서 후보를 원본(is_correction=False)으로만
    제한하면 안 되고, 같은 키의 모든 문서(원본+정정 전체)를 후보로 삼아 "직접
    부모"를 찾은 뒤, 부모가 없어질 때까지 chasing 해서 최종 root(=원본)를 구한다.
    """
    all_rows = list(rows)

    def cand_key(r: ManifestRow) -> tuple:
        return (r.corp_name, r.flr_nm) if doc_group == "holding" else (r.corp_name,)

    candidates_by_key: dict[tuple, list[ManifestRow]] = {}
    for r in all_rows:
        candidates_by_key.setdefault(cand_key(r), []).append(r)

    by_doc_id = {r.doc_id: r for r in all_rows}
    immediate_parent: dict[str, str | None] = {}
    source_by_doc: dict[str, str] = {}

    for row in all_rows:
        if not row.is_correction:
            immediate_parent[row.doc_id] = None
            source_by_doc[row.doc_id] = "original"
            continue

        if row.file_format != "xml":
            immediate_parent[row.doc_id] = None
            source_by_doc[row.doc_id] = "unsupported_format"
            logger.warning("[CORRECTION] doc_id=%s pdf+html 형식 — 텍스트 추출 불가, unresolved", row.doc_id)
            continue

        if row.doc_id in MANUAL_TARGET_DATE_OVERRIDES:
            target_date = MANUAL_TARGET_DATE_OVERRIDES[row.doc_id]
            target_title = None
            source = "manual_override"
        else:
            doc_dir = resolver.resolve(row.file_path)
            main_file = None
            if doc_dir is not None:
                exact = [f for f in doc_dir.glob("*.xml") if f.stem == row.rcept_no]
                main_file = exact[0] if exact else next(iter(doc_dir.glob("*.xml")), None)
            if main_file is None:
                immediate_parent[row.doc_id] = None
                source_by_doc[row.doc_id] = "unresolved"
                logger.warning("[CORRECTION] doc_id=%s 원본 파일 resolve 실패", row.doc_id)
                continue
            info = extract_correction_info(main_file.read_bytes(), doc_group=doc_group)
            target_date, target_title = info.target_date, info.target_title
            source = "rule" if info.ok else "unresolved"

        if target_date is None:
            immediate_parent[row.doc_id] = None
            source_by_doc[row.doc_id] = "unresolved"
            logger.warning("[CORRECTION] doc_id=%s target_date 추출 실패 (본문 패턴 없음)", row.doc_id)
            continue

        candidates = [
            c for c in candidates_by_key.get(cand_key(row), [])
            if c.rcept_dt == target_date and c.doc_id != row.doc_id
        ]

        chosen: ManifestRow | None = None
        if len(candidates) == 1:
            chosen = candidates[0]
        elif len(candidates) > 1:
            scored = sorted(
                candidates,
                key=lambda c: fuzz.token_sort_ratio(target_title or "", _strip_correction_prefix(c.report_nm)),
                reverse=True,
            )
            best_score = fuzz.token_sort_ratio(target_title or "", _strip_correction_prefix(scored[0].report_nm))
            logger.warning(
                "[CORRECTION] doc_id=%s (corp=%s date=%s) 후보 %d건 collision — fuzzy tiebreak score=%.1f: %s",
                row.doc_id, row.corp_name, target_date, len(candidates), best_score,
                [c.doc_id for c in scored],
            )
            if best_score >= FUZZY_MIN_SCORE:
                chosen = scored[0]
                source = "fuzzy_tiebreak"

        if chosen is None:
            immediate_parent[row.doc_id] = None
            source_by_doc[row.doc_id] = "unresolved"
            logger.warning(
                "[CORRECTION] doc_id=%s 원본/직전정정본 미발견 (corp=%s target_date=%s target_title=%r 후보수=%d)",
                row.doc_id, row.corp_name, target_date, target_title, len(candidates),
            )
            continue

        immediate_parent[row.doc_id] = chosen.doc_id
        source_by_doc[row.doc_id] = source

    def find_root(doc_id: str) -> str:
        seen: set[str] = set()
        cur = doc_id
        while immediate_parent.get(cur) is not None:
            if cur in seen:  # cycle guard (이론상 불가하지만 방어적으로)
                logger.warning("[CORRECTION] doc_id=%s parent chain 순환 감지 — cur=%s 에서 중단", doc_id, cur)
                break
            seen.add(cur)
            cur = immediate_parent[cur]
        return cur

    members: dict[str, list[ManifestRow]] = {}
    for row in all_rows:
        root_id = find_root(row.doc_id)
        members.setdefault(root_id, []).append(row)

    return _finalize_groups(members, source_by_doc)


def build_correction_index(manifest: list[ManifestRow], resolver: PathResolver) -> dict[str, CorrectionRecord]:
    """전체 manifest 에 대해 doc_id -> CorrectionRecord 매핑을 만든다."""
    index: dict[str, CorrectionRecord] = {}

    periodic_rows = [r for r in manifest if r.doc_group == "periodic"]
    index.update(_build_periodic_groups(periodic_rows))

    for grp in ("major", "exchange", "holding"):
        rows = [r for r in manifest if r.doc_group == grp]
        index.update(_build_event_groups(rows, grp, resolver))

    return index
