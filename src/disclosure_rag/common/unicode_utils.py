"""Unicode NFC/NFD 정규화 및 Corpus 실제 filesystem path 리졸버.

배경 (Phase 0 실측 결과):
- corpus/raw/ 하위 법인 폴더명은 NFD(자모분해형)로 저장되어 있다.
- manifest.jsonl 의 corp_name / file_path, universe.csv 의 corp_name 은 NFC(완성형)다.
- 파일명(접수번호 기반, 예: 20230515002335.xml)은 전부 숫자/영문이라
  NFC/NFD 이슈가 없다 — 이슈는 오직 "법인 폴더명" 세그먼트에만 존재한다.

따라서 이 모듈은:
1. normalize_nfc() : 모든 논리적 문자열 비교의 표준
2. PathResolver : manifest 의 NFC file_path 를 실제 NFD(또는 그 무엇이든) filesystem
   경로로 세그먼트 단위로 안전하게 변환한다. 폴더명이 통째로 NFD 인지, 일부만
   그런지 가정하지 않고 "실제 디렉토리 엔트리를 NFC 로 정규화해 비교"하는 방식으로
   모든 경우를 커버한다.

금지 사항(구현 원칙):
- `folder_name == manifest_corp_name` 같은 직접 비교 금지 (silent 0건 위험, §2 실측 확인됨)
- `Path(manifest_row["file_path"]).exists()` 같은 직접 경로 사용 금지
- resolve 실패를 조용히 빈 리스트로 넘기지 않는다 (§7) — 반드시 로그를 남긴다.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_nfc(value: str | None) -> str | None:
    """모든 논리적 문자열 비교(조인, 필터링, 메타데이터 저장)의 표준 정규화."""
    if value is None:
        return None
    return unicodedata.normalize("NFC", value)


def is_nfc(value: str) -> bool:
    return unicodedata.is_normalized("NFC", value)


@dataclass
class ResolveFailure:
    """Path resolve 실패 시 남기는 진단 정보. 정상적인 '빈 결과'와 구분하기 위한 근거."""

    requested_path: str
    failed_segment: str
    failed_segment_index: int
    available_siblings: list[str] = field(default_factory=list)

    def warn(self, *, extra: dict | None = None) -> None:
        logger.warning(
            "[CORPUS_PATH_RESOLVE_FAILED] requested_path=%r failed_segment=%r "
            "(segment #%d) available_siblings=%r extra=%r",
            self.requested_path,
            self.failed_segment,
            self.failed_segment_index,
            self.available_siblings,
            extra or {},
        )


class PathResolver:
    """corpus root 기준으로, manifest 의 NFC 논리 경로를 실제 filesystem Path 로 변환.

    핵심 아이디어: 세그먼트 단위로 내려가면서, 각 단계에서 실제 디렉토리 엔트리들을
    NFC 정규화해 목표 세그먼트(역시 NFC)와 비교한다. 물리적으로 그 엔트리가 NFD든
    NFC든 그 외 어떤 정규화 형태든 상관없이 always NFC 기준으로 매칭한다.

    같은 NFC 키로 정규화되는 서로 다른 실제 엔트리가 한 디렉토리에 동시에 존재하면
    collision 으로 간주하고 경고를 남긴다 (§4).
    """

    def __init__(self, corpus_root: str | Path):
        self.root = Path(corpus_root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"corpus_root 가 디렉토리가 아닙니다: {self.root}")
        self._dir_cache: dict[Path, dict[str, str]] = {}  # dir -> {nfc_name: actual_name}
        self.collisions: list[tuple[Path, str, str, str]] = []  # (dir, nfc_key, name1, name2)

    def _entries_nfc_map(self, directory: Path) -> dict[str, str]:
        if directory in self._dir_cache:
            return self._dir_cache[directory]
        mapping: dict[str, str] = {}
        try:
            entries = list(directory.iterdir())
        except (FileNotFoundError, NotADirectoryError):
            entries = []
        for entry in entries:
            key = normalize_nfc(entry.name)
            if key in mapping and mapping[key] != entry.name:
                self.collisions.append((directory, key, mapping[key], entry.name))
                logger.warning(
                    "[CORPUS_PATH_COLLISION] dir=%s nfc_key=%r entries=%r vs %r",
                    directory, key, mapping[key], entry.name,
                )
                continue
            mapping[key] = entry.name
        self._dir_cache[directory] = mapping
        return mapping

    def resolve(self, logical_path: str) -> Path | None:
        """logical_path (NFC 기준, corpus_root 상대경로) -> 실제 Path.

        실패 시 None 을 반환하되, 반드시 WARNING 로그를 남긴다 (silent failure 금지).
        """
        parts = Path(logical_path).parts
        cur = self.root
        for idx, seg in enumerate(parts):
            target = normalize_nfc(seg)
            mapping = self._entries_nfc_map(cur)
            actual_name = mapping.get(target)
            if actual_name is None:
                ResolveFailure(
                    requested_path=logical_path,
                    failed_segment=seg,
                    failed_segment_index=idx,
                    available_siblings=sorted(mapping.values())[:20],
                ).warn()
                return None
            cur = cur / actual_name
        return cur

    def resolve_or_raise(self, logical_path: str) -> Path:
        p = self.resolve(logical_path)
        if p is None:
            raise FileNotFoundError(f"corpus path resolve 실패: {logical_path}")
        return p

    def company_dir_map(self, doc_group: str) -> dict[str, Path]:
        """raw/<doc_group>/ 하위 법인 폴더들을 {정규화된 법인명: 실제 Path} 로 매핑."""
        group_dir = self.root / "raw" / doc_group
        result: dict[str, Path] = {}
        if not group_dir.is_dir():
            logger.warning("[CORPUS_PATH_RESOLVE_FAILED] doc_group 디렉토리 없음: %s", group_dir)
            return result
        for entry in group_dir.iterdir():
            if entry.name.startswith("."):
                continue
            result[normalize_nfc(entry.name)] = entry
        return result
