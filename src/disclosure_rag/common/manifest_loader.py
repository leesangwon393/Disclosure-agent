"""corpus/manifest.jsonl, universe.csv 로더.

- corp_code / stock_code 는 선행 0 이 있는 문자열이므로 반드시 str 로 로딩한다.
- 한글 metadata (corp_name 등) 는 로딩 시점에 NFC 정규화를 강제한다 (§25).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from disclosure_rag.common.unicode_utils import normalize_nfc

MANIFEST_STR_DTYPES = {"corp_code": str, "stock_code": str}


@dataclass(frozen=True)
class ManifestRow:
    doc_id: str
    corp_code: str
    corp_name: str
    listed_name: str
    stock_code: str
    industry: str
    sector: str
    doc_group: str  # periodic | major | exchange | holding
    doc_subtype: str | None
    report_nm: str
    is_correction: bool
    rcept_no: str
    rcept_dt: str  # YYYYMMDD
    flr_nm: str
    base_year: int | None
    base_month: int | None
    file_path: str  # NFC, corpus_root 상대경로
    file_format: str  # xml | pdf+html
    n_files: int

    @classmethod
    def from_json(cls, obj: dict) -> "ManifestRow":
        return cls(
            doc_id=obj["doc_id"],
            corp_code=obj["corp_code"],
            corp_name=normalize_nfc(obj["corp_name"]),
            listed_name=normalize_nfc(obj.get("listed_name")),
            stock_code=obj["stock_code"],
            industry=obj.get("industry"),
            sector=obj.get("sector"),
            doc_group=obj["doc_group"],
            doc_subtype=obj.get("doc_subtype"),
            report_nm=obj.get("report_nm", ""),
            is_correction=bool(obj.get("is_correction", False)),
            rcept_no=obj["rcept_no"],
            rcept_dt=obj["rcept_dt"],
            flr_nm=normalize_nfc(obj.get("flr_nm")),
            base_year=obj.get("base_year"),
            base_month=obj.get("base_month"),
            file_path=normalize_nfc(obj["file_path"]),
            file_format=obj.get("file_format", "xml"),
            n_files=int(obj.get("n_files", 1)),
        )


def load_manifest(corpus_root: str | Path) -> list[ManifestRow]:
    path = Path(corpus_root) / "manifest.jsonl"
    rows: list[ManifestRow] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(ManifestRow.from_json(json.loads(line)))
    return rows


def load_universe(corpus_root: str | Path) -> pd.DataFrame:
    path = Path(corpus_root) / "universe.csv"
    df = pd.read_csv(path, dtype={"corp_code": str, "stock_code": str}, encoding="utf-8-sig")
    df["corp_name"] = df["corp_name"].map(normalize_nfc)
    df["listed_name"] = df["listed_name"].map(normalize_nfc)
    return df
