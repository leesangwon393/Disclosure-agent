"""L1 스냅샷 로더.

build_snapshot.py 가 만든 JSONL(gzip 가능)을 다시 ChunkSchema 로 읽는다.
검색용 `text` 필드는 용량 때문에 기본적으로 저장하지 않으므로 여기서 재현한다
(raw_text + company/report_name/period/section_path 로 결정론적으로 복원됨).

파싱을 다시 돌리지 않고 인덱스만 재구축할 때 이 모듈을 쓴다.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator

from disclosure_rag.chunking.chunk_schema import ChunkSchema, render_search_text


def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _resolve(snapshot_dir: str | Path, stem: str) -> Path:
    d = Path(snapshot_dir)
    for cand in (d / f"{stem}.jsonl.gz", d / f"{stem}.jsonl"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{stem}.jsonl(.gz) 없음: {d}")


def iter_records(snapshot_dir: str | Path, stem: str) -> Iterator[dict]:
    with _open(_resolve(snapshot_dir, stem)) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_chunks(
    snapshot_dir: str | Path, *, leaf_only: bool = False, render_text: bool = True,
    include_field_codes: bool = False,
) -> list[ChunkSchema]:
    """L1 스냅샷에서 ChunkSchema 를 읽는다.

    include_field_codes 기본 False 인 이유(실측): 전체 코퍼스 스냅샷의 field_refs 가
    **81,418,455개**다(leaf 551,596개 기준 조각당 약 148개). 이걸 전부 pydantic
    객체로 만들면 수십 GB 와 수 분이 든다.

    그런데 **검색 계층은 field_codes 를 전혀 쓰지 않는다.** 구조화 필드는 facts 층
    (`artifacts/facts/facts.sqlite`)이 담당한다. 그래서 인덱싱·서빙 경로에서는
    기본적으로 버리고 읽는다. 필요하면 True 로 켠다.
    """
    out: list[ChunkSchema] = []
    for rec in iter_records(snapshot_dir, "chunks"):
        is_leaf = rec.pop("is_leaf", True)
        if leaf_only and not is_leaf:
            continue
        if not include_field_codes:
            rec.pop("field_codes", None)
        if render_text and "text" not in rec:
            rec["text"] = render_search_text(
                company=rec.get("company"), report_name=rec.get("report_name"),
                period=rec.get("period"), section_path=rec.get("section_path") or [],
                body_text=rec.get("raw_text", ""),
            )
        rec.setdefault("text", rec.get("raw_text", ""))
        out.append(ChunkSchema(**rec))
    return out


def iter_leaf_texts(snapshot_dir: str | Path):
    """(chunk_id, 검색용 text) 만 흘려보낸다. **ChunkSchema 객체를 만들지 않는다.**

    임베딩처럼 "id 와 텍스트만 있으면 되는" 경로용. 전체 코퍼스 기준 leaf 551,596개에
    field_refs 가 81,418,455개 달려 있어서(조각당 평균 148개) pydantic 으로 전부
    객체화하면 수십 GB 를 쓴다. 여기서는 dict 에서 필요한 두 개만 꺼내고 버린다.
    """
    for rec in iter_records(snapshot_dir, "chunks"):
        if not rec.get("is_leaf", True):
            continue
        text = rec.get("text")
        if not text:
            text = render_search_text(
                company=rec.get("company"), report_name=rec.get("report_name"),
                period=rec.get("period"), section_path=rec.get("section_path") or [],
                body_text=rec.get("raw_text", ""),
            )
        yield rec["chunk_id"], text


def load_parent_map(snapshot_dir: str | Path) -> dict[str, str]:
    """parent_chunk_id -> parent raw_text. Parent Expansion 용.
    기존 파이프라인은 parent 를 만들고 버려서 이 조회가 불가능했다."""
    return {
        rec["chunk_id"]: rec.get("raw_text", "")
        for rec in iter_records(snapshot_dir, "chunks")
        if not rec.get("is_leaf", True)
    }  # raw_text 만 읽으므로 field_codes 는 애초에 객체화되지 않는다


def load_build_manifest(snapshot_dir: str | Path) -> dict:
    return json.loads((Path(snapshot_dir) / "build_manifest.json").read_text(encoding="utf-8"))
