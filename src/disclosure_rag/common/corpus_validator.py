"""Corpus Startup Validation (§8).

전처리 파이프라인 시작 시 반드시 먼저 돌려서, "데이터가 없다" 는 결론이
Unicode mismatch 때문인지 진짜로 데이터가 없는 것인지 구분한다 (§7, §8).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from disclosure_rag.common.manifest_loader import ManifestRow, load_manifest, load_universe
from disclosure_rag.common.unicode_utils import PathResolver, normalize_nfc

logger = logging.getLogger(__name__)

DOC_GROUPS = ("periodic", "major", "exchange", "holding")


@dataclass
class GroupStats:
    total: int = 0
    resolved: int = 0
    failed_doc_ids: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.resolved / self.total if self.total else 0.0


@dataclass
class ValidationReport:
    universe_companies: int = 0
    resolved_companies: dict[str, int] = field(default_factory=dict)  # doc_group -> count
    missing_companies: dict[str, list[str]] = field(default_factory=dict)  # doc_group -> names
    manifest_documents: int = 0
    resolved_documents: int = 0
    missing_documents: int = 0
    group_stats: dict[str, GroupStats] = field(default_factory=dict)
    path_collisions: int = 0

    def is_healthy(self, min_rate: float = 0.99) -> bool:
        if self.manifest_documents == 0:
            return False
        overall_rate = self.resolved_documents / self.manifest_documents
        return overall_rate >= min_rate

    def render(self) -> str:
        lines = ["=== Corpus Startup Validation Report ===", ""]
        lines.append(f"Companies in universe.csv: {self.universe_companies}")
        for grp in DOC_GROUPS:
            resolved = self.resolved_companies.get(grp, 0)
            missing = self.missing_companies.get(grp, [])
            lines.append(
                f"  [{grp}] resolved in raw: {resolved}/{self.universe_companies}"
                + (f"  MISSING: {missing}" if missing else "")
            )
        lines.append("")
        lines.append(f"Manifest documents: {self.manifest_documents}")
        lines.append(f"Resolved documents: {self.resolved_documents}")
        lines.append(f"Missing documents:  {self.missing_documents}")
        lines.append("")
        for grp in DOC_GROUPS:
            gs = self.group_stats.get(grp, GroupStats())
            lines.append(f"  {grp}: {gs.resolved}/{gs.total} resolved ({gs.rate*100:.2f}%)")
        lines.append("")
        lines.append(f"Path NFC collisions detected: {self.path_collisions}")
        lines.append("")
        lines.append("HEALTHY" if self.is_healthy() else "UNHEALTHY -- Unicode/path mismatch 의심, 통계 확정 금지")
        return "\n".join(lines)


def validate_corpus(corpus_root: str) -> tuple[ValidationReport, list[ManifestRow]]:
    resolver = PathResolver(corpus_root)
    universe = load_universe(corpus_root)
    manifest = load_manifest(corpus_root)

    report = ValidationReport()
    universe_names = set(normalize_nfc(n) for n in universe["corp_name"])
    report.universe_companies = len(universe_names)

    for grp in DOC_GROUPS:
        dir_map = resolver.company_dir_map(grp)
        raw_names = set(dir_map.keys())
        resolved = universe_names & raw_names
        missing = sorted(universe_names - raw_names)
        report.resolved_companies[grp] = len(resolved)
        if missing:
            report.missing_companies[grp] = missing
            logger.warning("[CORPUS_VALIDATION] %s: universe 기업 중 raw 폴더 없음: %s", grp, missing)

    report.manifest_documents = len(manifest)
    for grp in DOC_GROUPS:
        report.group_stats[grp] = GroupStats()

    for row in manifest:
        gs = report.group_stats.setdefault(row.doc_group, GroupStats())
        gs.total += 1
        resolved_path = resolver.resolve(row.file_path)
        if resolved_path is not None and resolved_path.is_dir():
            gs.resolved += 1
            report.resolved_documents += 1
        else:
            gs.failed_doc_ids.append(row.doc_id)
            report.missing_documents += 1
            logger.warning(
                "[CORPUS_VALIDATION] resolve 실패: doc_id=%s corp_name=%r doc_group=%s file_path=%r",
                row.doc_id, row.corp_name, row.doc_group, row.file_path,
            )

    report.path_collisions = len(resolver.collisions)
    return report, manifest


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING)
    root = sys.argv[1] if len(sys.argv) > 1 else "corpus"
    rep, _ = validate_corpus(root)
    print(rep.render())
