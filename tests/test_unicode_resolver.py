"""Phase 1 회귀 테스트: Unicode NFC/NFD 리졸버 + Corpus validation.

실제 코퍼스(corpus/)를 대상으로 돌린다 — Phase 0 에서 실측한 결과
(전 문서 100% resolve, collision 0건)를 회귀 테스트로 고정한다.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from disclosure_rag.common.corpus_validator import validate_corpus
from disclosure_rag.common.manifest_loader import load_manifest, load_universe
from disclosure_rag.common.unicode_utils import PathResolver, normalize_nfc

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"

pytestmark = pytest.mark.skipif(not CORPUS_ROOT.is_dir(), reason="corpus/ 없음")


def test_normalize_nfc_roundtrip():
    nfd = unicodedata.normalize("NFD", "삼성전자")
    nfc = unicodedata.normalize("NFC", "삼성전자")
    assert nfd != nfc  # 실제로 바이트 표현이 다름을 확인
    assert normalize_nfc(nfd) == nfc
    assert normalize_nfc(None) is None


def test_raw_company_dirs_are_nfd_confirmed():
    """§2 에서 실측한 전제: raw/ 하위 폴더명은 NFD 다. 전제가 깨지면 리졸버 설계를 재검토해야 하므로 회귀 테스트로 고정."""
    sample_dir = CORPUS_ROOT / "raw" / "periodic"
    names = [p.name for p in sample_dir.iterdir() if not p.name.startswith(".")]
    assert names, "raw/periodic 하위 폴더가 없음"
    nfd_count = sum(1 for n in names if unicodedata.is_normalized("NFD", n) and not unicodedata.is_normalized("NFC", n))
    assert nfd_count / len(names) > 0.9, "raw 폴더명 정규화 형태 전제가 바뀜 (더 이상 NFD 아님?)"


def test_manifest_is_nfc():
    manifest = load_manifest(CORPUS_ROOT)
    assert len(manifest) == 4204
    for row in manifest[:50]:
        assert unicodedata.is_normalized("NFC", row.corp_name)
        assert unicodedata.is_normalized("NFC", row.file_path)


def test_path_resolver_resolves_all_company_dirs():
    resolver = PathResolver(CORPUS_ROOT)
    universe = load_universe(CORPUS_ROOT)
    for grp in ("periodic", "major", "exchange", "holding"):
        dir_map = resolver.company_dir_map(grp)
        for corp_name in universe["corp_name"]:
            assert normalize_nfc(corp_name) in dir_map, f"{grp}: {corp_name} 폴더 resolve 실패"
    assert resolver.collisions == []


def test_direct_string_equality_fails_without_normalization():
    """§2 가 경고한 안티패턴이 실제로 깨짐을 증명 (반면 normalize_nfc 사용시 성공)."""
    resolver = PathResolver(CORPUS_ROOT)
    manifest = load_manifest(CORPUS_ROOT)
    row = next(r for r in manifest if r.doc_group == "periodic")
    raw_dir = CORPUS_ROOT / "raw" / "periodic"
    physical_names = {p.name for p in raw_dir.iterdir()}
    # manifest.corp_name(NFC) 이 물리적 폴더명 집합(NFD)에 직접 포함되지 않음을 확인
    assert row.corp_name not in physical_names
    # 반면 정규화된 리졸버는 성공
    assert resolver.resolve(row.file_path) is not None


def test_full_corpus_validation_100_percent():
    """전체 4,204건 manifest 문서가 100% resolve 되어야 한다 (Phase 0 실측 결과 고정)."""
    report, manifest = validate_corpus(str(CORPUS_ROOT))
    assert report.manifest_documents == 4204
    assert report.resolved_documents == 4204
    assert report.missing_documents == 0
    assert report.path_collisions == 0
    assert report.is_healthy()
    for grp in ("periodic", "major", "exchange", "holding"):
        assert report.group_stats[grp].rate == 1.0
