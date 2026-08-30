"""Phase 7 회귀 테스트: Correction Graph Builder.

핵심 발견 3가지를 회귀로 고정한다:
1. periodic 은 (corp_name, doc_subtype, base_year, base_month) 키만으로 collision
   없이 100% 묶인다 (pdf+html 대체수집 2건도 이 방식으로 자동 해결됨).
2. major/exchange/holding 은 정정이 "최초 원본"이 아니라 "직전 정정본"을 가리키는
   다단 체인이 존재한다 — root 까지 chasing 해야 한다.
3. "unresolved" 로 남는 정정의 절대다수는 원본이 corpus 수집 기간(2023-01-01~)
   이전에 제출되어 애초에 코퍼스에 없기 때문이다 — resolver 버그가 아니라
   corpus 자체의 특성이다. 이를 silent 하게 넘기지 않고 WARNING 으로 남긴다(§7).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from disclosure_rag.common.manifest_loader import load_manifest
from disclosure_rag.common.unicode_utils import PathResolver
from disclosure_rag.correction.correction_extractor import extract_correction_info
from disclosure_rag.correction.correction_graph_builder import build_correction_index

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
pytestmark = pytest.mark.skipif(not CORPUS_ROOT.is_dir(), reason="corpus/ 없음")


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(CORPUS_ROOT)


@pytest.fixture(scope="module")
def resolver():
    return PathResolver(CORPUS_ROOT)


@pytest.fixture(scope="module")
def index(manifest, resolver):
    return build_correction_index(manifest, resolver)


def test_every_manifest_row_has_a_record(manifest, index):
    assert len(index) == len(manifest) == 4204


def test_periodic_zero_collision_manifest_key_grouping(manifest, index):
    """§7 실측: periodic 은 전부 manifest_key 로 해결되어야 한다 (collision 0건)."""
    periodic_corr = [r for r in manifest if r.doc_group == "periodic" and r.is_correction]
    assert len(periodic_corr) == 159
    assert all(index[r.doc_id].resolution_source == "manifest_key" for r in periodic_corr)


def test_correction_chain_up_to_4_deep(manifest, index):
    """메리츠금융지주 2023Q3 분기보고서: 원본 + 정정 3건 = 4단, 전부 최초 원본으로 수렴."""
    rows = [
        r for r in manifest
        if r.corp_name == "메리츠금융지주" and r.doc_group == "periodic"
        and r.doc_subtype == "quarter" and r.base_year == 2023 and r.base_month == 9
    ]
    assert len(rows) == 4
    group_ids = {index[r.doc_id].correction_group_id for r in rows}
    assert len(group_ids) == 1  # 전부 같은 그룹
    orders = sorted(index[r.doc_id].correction_order for r in rows)
    assert orders == [0, 1, 2, 3]
    latest = [r for r in rows if index[r.doc_id].is_latest]
    assert len(latest) == 1
    assert latest[0].rcept_dt == max(r.rcept_dt for r in rows)


def test_exchange_correction_can_chain_to_prior_correction_not_original(manifest, index):
    """회귀 테스트: 한화오션 exchange 정정이 원본이 아니라 직전 정정본을 가리키는 실제 사례.
    exchange_20250113800354 -> exchange_20231211800451(그 자체가 [기재정정]) -> 그 원본.
    최종 correction_group_id 는 (root) 원본으로 수렴해야 하고, 두 정정 모두 같은 그룹이어야 한다."""
    child = next(r for r in manifest if r.doc_id == "exchange_20250113800354")
    parent = next(r for r in manifest if r.doc_id == "exchange_20231211800451")
    assert parent.is_correction is True  # parent 자체가 정정본임을 재확인 (구조 전제)
    assert index[child.doc_id].correction_group_id == index[parent.doc_id].correction_group_id


def test_manual_override_typo_case_resolved(manifest, index):
    """§7 실측: major_20250828001452 원문 오타("08년 28일")는 override table 로 해결된다."""
    rec = index["major_20250828001452"]
    assert rec.resolution_source == "manual_override"
    original = next(r for r in manifest if r.doc_id == "major_20250828001211")
    assert rec.correction_group_id == original.doc_id


def test_unresolved_corrections_mostly_predate_collection_window(manifest, resolver, index):
    """unresolved 의 절대다수는 원본이 2023-01-01 이전이라 코퍼스에 없기 때문이어야 한다
    (실측: exchange 287/300=95.7%, major 24/31=77.4%, holding 2/2=100%).
    이 비율이 크게 무너지면 resolver 로직 회귀(버그)를 의심해야 한다."""
    for doc_group in ("major", "exchange", "holding"):
        unresolved = [
            r for r in manifest
            if r.doc_group == doc_group and r.is_correction
            and index[r.doc_id].resolution_source == "unresolved"
        ]
        if not unresolved:
            continue
        before_window = 0
        for r in unresolved:
            d = resolver.resolve(r.file_path)
            fl = next(d.glob("*.xml"))
            info = extract_correction_info(fl.read_bytes(), doc_group=doc_group)
            if info.ok and info.target_date < "20230101":
                before_window += 1
        assert before_window / len(unresolved) >= 0.70, (
            f"{doc_group}: unresolved 중 window-이전 비율이 예상보다 낮음 "
            f"({before_window}/{len(unresolved)}) — 회귀 의심"
        )


def test_every_group_has_exactly_one_latest(manifest, index):
    groups: dict[str, list] = {}
    for r in manifest:
        groups.setdefault(index[r.doc_id].correction_group_id, []).append(r)
    for group_id, rows in groups.items():
        latest_count = sum(1 for r in rows if index[r.doc_id].is_latest)
        assert latest_count == 1, f"group {group_id}: is_latest 개수={latest_count}"


def test_overall_resolution_rate_floor(manifest, index):
    """전체 정정(1,004건) 중 최소 65% 는 rule/manifest_key/fuzzy/override 로 resolve 되어야 한다."""
    corrections = [r for r in manifest if r.is_correction]
    resolved = [r for r in corrections if index[r.doc_id].resolution_source != "unresolved"]
    rate = len(resolved) / len(corrections)
    assert rate >= 0.65, f"전체 정정 resolve율 회귀: {rate*100:.1f}%"
