"""DART 코퍼스에 등장하는 주체와 공시 안에서의 역할을 관리한다.

Entity Registry는 질문에 등장한 회사나 사람이 우리 코퍼스에서 확인된
주체인지, 그리고 어떤 역할로 등장했는지를 QueryPlan이 규조화하는
오프라인 사전이다. 이 모듈은 저장 형식과 조회 API, 룰 기반 빌더를 함께
제공한다. 판정에 LLM을 사용하지 않는다.

counterparty 등 정형 역할은 facts.sqlite를 빠른 기본 소스로 사용한다.
facts는 chunks의 KeyValueNode에서 만든 손실 없는 색인이며 chunk_id를
보존한다. facts가 없는 환경을 위해 chunks.jsonl(.gz)를 직접 스캔하는
대체 경로도 제공한다.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Literal

from disclosure_rag.common.manifest_loader import load_manifest, load_universe

EntityType = Literal["universe", "submitter", "counterparty", "shareholder", "subsidiary"]
ENTITY_TYPES: tuple[EntityType, ...] = (
    "universe", "submitter", "counterparty", "shareholder", "subsidiary",
)

# 필드명이 이 집합과 정확히 일치할 때만 역할로 인정한다. 본문에 단어가
# 언급됐다는 이유로 역할을 붙이지 않는다.
ROLE_KEYS: dict[EntityType, frozenset[str]] = {
    "counterparty": frozenset({"계약상대", "계약상대방", "거래상대방", "처분상대방"}),
    "shareholder": frozenset({"최대주주명", "주요주주명", "주주명", "보고자명"}),
    "subsidiary": frozenset({"종속기업명", "종속회사명", "자회사명"}),
    "universe": frozenset(),
    "submitter": frozenset(),
}
_KEY_PREFIX = re.compile(r"^\s*(?:[0-9]{1,2}|[가-힣])\s*[.)]\s*|^\s*[-·※]\s*")
_KEY_UNIT = re.compile(r"[（(]\s*[^)）]{1,12}\s*[)）]\s*$")
_KV_LINE = re.compile(r"^\s*-\s*(?P<key>[^:：]{1,80})\s*[:：]\s*(?P<value>.+?)\s*$")
_SUBSIDIARY_ROW = re.compile(
    r"(?:^|/)\s*기업명\s*[:：]\s*(?P<value>.+?)\s*/\s*업종\s*[:：]",
    re.MULTILINE,
)
_SUBSIDIARY_CONTEXT = ("종속기업 현황", "종속회사 현황", "연결대상 종속회사 현황")
_SPACE = re.compile(r"\s+")
_CORP_MARKS = re.compile(r"(?:㈜|株)")
_EMPTY_OR_SECRET = re.compile(
    r"^(?:-|[–—]|해당사항\s*없음|해당없음|미정|미공개|비공개|N/?A)$",
    re.IGNORECASE,
)


def normalize_name(value: object) -> str:
    """표시용 이름을 NFC로 정규화한다."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value)).strip()
    text = _CORP_MARKS.sub("(주)", text)
    text = text.replace("（", "(").replace("）", ")")
    return _SPACE.sub(" ", text)


def lookup_key(value: object) -> str:
    """별칭 매칭용 보수적 key.

    법인 표기와 공백만 통일한다. 영문 대소문자는 통일하지만 구두점은
    남겨 서로 다른 법인을 과도하게 병합하지 않는다.
    """
    text = normalize_name(value).replace("주식회사", "(주)")
    return _SPACE.sub("", text).casefold()


def _role_key(raw: str) -> str:
    key = _KEY_PREFIX.sub("", normalize_name(raw))
    key = _KEY_UNIT.sub("", key).rstrip(":： ")
    return _SPACE.sub("", key)


def _valid_role_value(value: str) -> bool:
    value = normalize_name(value)
    if len(value) < 2 or len(value) > 300 or _EMPTY_OR_SECRET.match(value):
        return False
    return bool(re.search(r"[가-힣A-Za-z]", value))


def _entity_id(key: str) -> str:
    return "ent_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class EntityEvidence:
    source: str
    role: EntityType
    report_id: str | None = None
    chunk_id: str | None = None
    document_company: str | None = None
    field: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class EntityRecord:
    entity_id: str
    canonical_name: str
    normalized_name: str
    aliases: set[str] = field(default_factory=set)
    types: set[EntityType] = field(default_factory=set)
    source_counts: Counter = field(default_factory=Counter)
    evidence: list[EntityEvidence] = field(default_factory=list)
    corp_code: str | None = None
    stock_code: str | None = None

    def to_dict(self) -> dict:
        result = {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "normalized_name": self.normalized_name,
            "aliases": sorted(self.aliases, key=lambda x: (len(x), x)),
            "types": [kind for kind in ENTITY_TYPES if kind in self.types],
            "source_counts": dict(sorted(self.source_counts.items())),
            "evidence": [item.to_dict() for item in self.evidence],
        }
        if self.corp_code:
            result["corp_code"] = self.corp_code
        if self.stock_code:
            result["stock_code"] = self.stock_code
        return result

    @classmethod
    def from_dict(cls, obj: dict) -> "EntityRecord":
        return cls(
            entity_id=obj["entity_id"], canonical_name=obj["canonical_name"],
            normalized_name=obj.get("normalized_name") or lookup_key(obj["canonical_name"]),
            aliases=set(obj.get("aliases", [])), types=set(obj.get("types", [])),
            source_counts=Counter(obj.get("source_counts", {})),
            evidence=[EntityEvidence(**item) for item in obj.get("evidence", [])],
            corp_code=obj.get("corp_code"), stock_code=obj.get("stock_code"),
        )


class EntityRegistry:
    """이름/별칭에서 엔터티와 역할을 조회하는 불변에 가까운 사전."""

    def __init__(self, records: Iterable[EntityRecord], *, metadata: dict | None = None):
        self.records = list(records)
        self.metadata = metadata or {}
        self._by_id = {record.entity_id: record for record in self.records}
        self._by_alias: dict[str, EntityRecord] = {}
        for record in self.records:
            for alias in record.aliases | {record.canonical_name}:
                self._by_alias.setdefault(lookup_key(alias), record)

    def resolve(self, name: str) -> EntityRecord | None:
        return self._by_alias.get(lookup_key(name))

    def contains(self, name: str, entity_type: EntityType | None = None) -> bool:
        record = self.resolve(name)
        return bool(record and (entity_type is None or entity_type in record.types))

    def types_for(self, name: str) -> set[EntityType]:
        record = self.resolve(name)
        return set(record.types) if record else set()

    @property
    def summary(self) -> dict[str, int]:
        result = {"entities": len(self.records)}
        result.update({kind: sum(kind in record.types for record in self.records) for kind in ENTITY_TYPES})
        return result

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "generated_at": self.metadata.get("generated_at"),
            "sources": self.metadata.get("sources", {}),
            "summary": self.summary,
            "entities": [record.to_dict() for record in sorted(self.records, key=lambda r: r.canonical_name)],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EntityRegistry":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            [EntityRecord.from_dict(item) for item in obj["entities"]],
            metadata={"generated_at": obj.get("generated_at"), "sources": obj.get("sources", {})},
        )


class EntityRegistryBuilder:
    """순서와 중복 처리가 재현 가능한 룰 기반 빌더."""

    def __init__(self, *, max_evidence_per_role: int = 5):
        # submitter 근거 5개가 모든 자리를 차지해 counterparty 근거가
        # 사라지지 않도록 역할별로 상한을 둔다.
        self.max_evidence_per_role = max_evidence_per_role
        self._records: dict[str, EntityRecord] = {}
        self._alias_to_key: dict[str, str] = {}
        self.raw_counts: Counter = Counter()

    def add(
        self, name: str, entity_type: EntityType, *, source: str,
        aliases: Iterable[str] = (), evidence: EntityEvidence | None = None,
        corp_code: str | None = None, stock_code: str | None = None,
    ) -> EntityRecord | None:
        name = normalize_name(name)
        if not name:
            return None
        aliases = {normalize_name(alias) for alias in aliases if normalize_name(alias)} | {name}
        keys = [lookup_key(alias) for alias in aliases]
        existing_key = next((self._alias_to_key[key] for key in keys if key in self._alias_to_key), None)
        key = existing_key or lookup_key(name)
        if key not in self._records:
            self._records[key] = EntityRecord(
                entity_id=_entity_id(key), canonical_name=name, normalized_name=key,
            )
        record = self._records[key]
        record.aliases.update(aliases)
        record.types.add(entity_type)
        record.source_counts[source] += 1
        record.corp_code = record.corp_code or corp_code
        record.stock_code = record.stock_code or stock_code
        for alias_key in keys:
            self._alias_to_key[alias_key] = key
        role_evidence_count = sum(item.role == evidence.role for item in record.evidence) if evidence else 0
        if evidence and role_evidence_count < self.max_evidence_per_role:
            signature = (evidence.role, evidence.report_id, evidence.chunk_id, evidence.field)
            existing = {(e.role, e.report_id, e.chunk_id, e.field) for e in record.evidence}
            if signature not in existing:
                record.evidence.append(evidence)
        self.raw_counts[entity_type] += 1
        return record

    def add_universe(self, corpus_root: str | Path) -> None:
        for _, row in load_universe(corpus_root).iterrows():
            corp = normalize_name(row["corp_name"])
            listed = normalize_name(row.get("listed_name"))
            # 공시 표에서 흔한 법인표기도 유니버스 회사로 해결한다.
            legal_aliases = {corp, listed, f"{corp}(주)", f"(주){corp}", f"{corp} 주식회사"}
            self.add(
                corp, "universe", source="universe.csv", aliases=legal_aliases,
                corp_code=str(row.get("corp_code") or ""), stock_code=str(row.get("stock_code") or ""),
            )

    def add_manifest(self, corpus_root: str | Path) -> tuple[int, int]:
        submitters: set[str] = set()
        outside_universe: set[str] = set()
        for row in load_manifest(corpus_root):
            submitter = normalize_name(row.flr_nm)
            if not submitter:
                continue
            submitters.add(submitter)
            record = self.add(
                submitter, "submitter", source="manifest.jsonl",
                evidence=EntityEvidence(
                    source="manifest.jsonl", role="submitter", report_id=row.doc_id,
                    document_company=row.corp_name,
                ),
            )
            if record and "universe" not in record.types:
                outside_universe.add(submitter)
            if row.doc_group == "holding":
                self.add(
                    submitter, "shareholder", source="manifest:holding",
                    evidence=EntityEvidence(
                        source="manifest.jsonl", role="shareholder", report_id=row.doc_id,
                        document_company=row.corp_name,
                    ),
                )
        return len(submitters), len(outside_universe)

    def add_roles(self, mentions: Iterable[tuple[str, EntityEvidence]]) -> None:
        for value, evidence in mentions:
            if _valid_role_value(value):
                self.add(value, evidence.role, source=evidence.source, evidence=evidence)

    def build(self, *, sources: dict | None = None) -> EntityRegistry:
        return EntityRegistry(
            self._records.values(),
            metadata={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sources": sources or {},
            },
        )


def iter_role_mentions_from_facts(path: str | Path) -> Iterator[tuple[str, EntityEvidence]]:
    """facts.sqlite의 역할 필드를 스트리밍한다."""
    path = Path(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        keys = sorted({key for values in ROLE_KEYS.values() for key in values})
        placeholders = ",".join("?" for _ in keys)
        sql = (
            "SELECT doc_id, chunk_id, company, key_norm, value_text "
            f"FROM facts WHERE key_norm IN ({placeholders}) ORDER BY fact_id"
        )
        key_to_role = {key: role for role, values in ROLE_KEYS.items() for key in values}
        for doc_id, chunk_id, company, key, value in connection.execute(sql, keys):
            role = key_to_role.get(key)
            if role:
                yield value, EntityEvidence(
                    source="facts.sqlite", role=role, report_id=doc_id, chunk_id=chunk_id,
                    document_company=company, field=key,
                )
    finally:
        connection.close()


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def iter_role_mentions_from_chunks(path: str | Path) -> Iterator[tuple[str, EntityEvidence]]:
    """chunks.jsonl(.gz)의 key-value 라인을 직접 읽는 대체 경로."""
    path = Path(path)
    key_to_role = {key: role for role, values in ROLE_KEYS.items() for key in values}
    with _open_text(path) as stream:
        for line in stream:
            if not any(key in line for key in key_to_role):
                continue
            obj = json.loads(line)
            raw_text = obj.get("raw_text") or ""
            for raw_line in raw_text.splitlines():
                match = _KV_LINE.match(raw_line)
                if not match:
                    continue
                key = _role_key(match.group("key"))
                role = key_to_role.get(key)
                if role:
                    yield match.group("value"), EntityEvidence(
                        source="chunks.jsonl", role=role, report_id=obj.get("report_id"),
                        chunk_id=obj.get("chunk_id"), document_company=obj.get("company"), field=key,
                    )


def iter_subsidiary_mentions_from_chunks(path: str | Path) -> Iterator[tuple[str, EntityEvidence]]:
    """정기공시의 `종속기업 현황` 표에서 `기업명` 열만 읽는다.

    periodic은 facts 추출 대상이 아니므로 이 역할만 청크에서 보완한다.
    현황 문구와 `기업명: ... / 업종:` 구조가 모두 있을 때만 채택해
    단순한 '종속기업' 언급은 버린다.
    """
    path = Path(path)
    with _open_text(path) as stream:
        for line in stream:
            if not any(context in line for context in _SUBSIDIARY_CONTEXT):
                continue
            obj = json.loads(line)
            # parent/leaf에 동일 표가 중복 저장된다. leaf만 근거로 쓴다.
            if obj.get("is_leaf") is False:
                continue
            raw_text = obj.get("raw_text") or ""
            for match in _SUBSIDIARY_ROW.finditer(raw_text):
                value = match.group("value").strip()
                yield value, EntityEvidence(
                    source="chunks.jsonl:subsidiary_table", role="subsidiary",
                    report_id=obj.get("report_id"), chunk_id=obj.get("chunk_id"),
                    document_company=obj.get("company"), field="종속기업현황.기업명",
                )


def build_entity_registry(
    *, corpus_root: str | Path, facts_db: str | Path | None = None,
    chunks_path: str | Path | None = None, role_source: Literal["auto", "facts", "chunks"] = "auto",
    include_subsidiaries: bool = True,
) -> tuple[EntityRegistry, dict[str, int]]:
    """레지스트리를 빌드하고 완료 조건에 필요한 원시 카운트를 반환한다."""
    builder = EntityRegistryBuilder()
    builder.add_universe(corpus_root)
    submitters, outside = builder.add_manifest(corpus_root)

    chosen = role_source
    if chosen == "auto":
        chosen = "facts" if facts_db and Path(facts_db).is_file() else "chunks"
    if chosen == "facts":
        if not facts_db or not Path(facts_db).is_file():
            raise FileNotFoundError(f"facts DB not found: {facts_db}")
        mentions = iter_role_mentions_from_facts(facts_db)
        role_path = str(Path(facts_db))
    else:
        if not chunks_path or not Path(chunks_path).is_file():
            raise FileNotFoundError(f"chunks file not found: {chunks_path}")
        mentions = iter_role_mentions_from_chunks(chunks_path)
        role_path = str(Path(chunks_path))
    builder.add_roles(mentions)
    supplemental_chunks = None
    if include_subsidiaries and chosen == "facts":
        if not chunks_path or not Path(chunks_path).is_file():
            raise FileNotFoundError(f"chunks file not found for subsidiaries: {chunks_path}")
        builder.add_roles(iter_subsidiary_mentions_from_chunks(chunks_path))
        supplemental_chunks = str(Path(chunks_path))
    stats = {
        "universe": sum("universe" in r.types for r in builder._records.values()),
        "submitter_raw": submitters,
        "submitter_outside_universe_raw": outside,
        **{f"role_rows_{key}": value for key, value in sorted(builder.raw_counts.items())},
    }
    registry = builder.build(sources={
        "corpus_root": str(Path(corpus_root)), "role_source": chosen, "role_path": role_path,
        "supplemental_chunks": supplemental_chunks,
        "build_stats": stats,
    })
    return registry, stats
