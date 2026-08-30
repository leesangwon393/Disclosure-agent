"""facts SQLite 스토어 — 숫자 질의를 '검색'이 아니라 '조회'로 처리한다.

왜 SQLite 인가: 서버 프로세스가 필요 없고(평가 서버가 파일 하나만 들고 가면 된다),
`WHERE value_num > 1e12` 같은 **범위 질의**가 되며, 결정론적이다.
벡터 검색으로는 "계약금액 1조 넘는 계약 찾아줘" 를 원리적으로 못 한다.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterable

from disclosure_rag.facts.extractor import Fact

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id             INTEGER PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    chunk_id            TEXT,
    company             TEXT,
    corp_code           TEXT,
    doc_group           TEXT,
    doc_subtype         TEXT,
    report_name         TEXT,
    filing_date         TEXT,
    period              TEXT,
    is_correction       INTEGER,
    is_latest           INTEGER,
    correction_group_id TEXT,
    group_label         TEXT,
    key                 TEXT NOT NULL,
    key_norm            TEXT NOT NULL,
    value_text          TEXT NOT NULL,
    value_num           REAL,
    value_unit          TEXT,
    value_date          TEXT,
    field_code          TEXT,
    unit_code           TEXT,
    unit_value          TEXT,
    section_path        TEXT
);
CREATE INDEX IF NOT EXISTS ix_facts_company_key ON facts(company, key_norm);
CREATE INDEX IF NOT EXISTS ix_facts_key         ON facts(key_norm);
CREATE INDEX IF NOT EXISTS ix_facts_doc         ON facts(doc_id);
CREATE INDEX IF NOT EXISTS ix_facts_date        ON facts(filing_date);
CREATE INDEX IF NOT EXISTS ix_facts_num         ON facts(key_norm, value_num);
CREATE INDEX IF NOT EXISTS ix_facts_group       ON facts(doc_group, doc_subtype);
"""

_COLS = [
    "doc_id", "chunk_id", "company", "corp_code", "doc_group", "doc_subtype", "report_name",
    "filing_date", "period", "is_correction", "is_latest", "correction_group_id",
    "group_label", "key", "key_norm", "value_text", "value_num", "value_unit", "value_date",
    "field_code", "unit_code", "unit_value", "section_path",
]


class FactStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        # API startup thread에서 만든 Store를 요청 worker가 사용해도 안전하도록
        # thread마다 SQLite connection을 하나씩 가진다. check_same_thread=False로
        # 하나를 공유하는 방식보다 동시 요청 사이의 cursor 충돌도 피할 수 있다.
        self._local = threading.local()
        self.conn.executescript(_SCHEMA)

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------ 적재
    def insert_many(self, facts: Iterable[Fact], *, batch: int = 2000) -> int:
        sql = f"INSERT INTO facts ({','.join(_COLS)}) VALUES ({','.join('?' * len(_COLS))})"
        buf, n = [], 0
        for f in facts:
            d = f.model_dump()
            d["section_path"] = json.dumps(d.get("section_path") or [], ensure_ascii=False)
            d["is_correction"] = int(bool(d.get("is_correction")))
            d["is_latest"] = None if d.get("is_latest") is None else int(bool(d["is_latest"]))
            buf.append(tuple(d.get(c) for c in _COLS))
            if len(buf) >= batch:
                self.conn.executemany(sql, buf); n += len(buf); buf = []
        if buf:
            self.conn.executemany(sql, buf); n += len(buf)
        self.conn.commit()
        return n

    def clear(self) -> None:
        self.conn.execute("DELETE FROM facts")
        self.conn.commit()

    # ------------------------------------------------------------------ 조회
    def lookup(
        self,
        *,
        company: str | None = None,
        companies: list[str] | None = None,
        key: str | None = None,
        doc_group: str | None = None,
        doc_id: str | None = None,
        period: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        latest_only: bool = True,
        numeric_only: bool = False,
        exact_only: bool = False,
        order_by: str = "date",
        limit: int = 20,
    ) -> list[dict]:
        """항목명은 정확히 일치하면 그것을, 아니면 부분일치로 넓힌다.

        latest_only 기본 True: 정정공시가 43%(거래소 기준)라 최신 유효본만 보는 것이
        일반 조회의 기본이다. 정정 이력을 볼 때만 False 로 내린다.

        order_by
            "date"       최신 공시부터 (기본값, 일반 조회)
            "value_desc" 값이 큰 것부터  — "최대 계약금액" 같은 질문
            "value_asc"  값이 작은 것부터

        기본값이 날짜순이라 **최댓값을 묻는 질문에서 답을 놓친다.** 실측
        (v2 38문항 S007~S014): 삼성바이오로직스 계약금액 54건 중 최댓값이
        2,074,694,843,000 인데, 최신 10건만 주니 그 안의 최댓값
        1,110,278,292,000 이 답으로 나갔다. 모델은 받은 것 중 정확히 최대를
        골랐다 — 우리가 잘못 준 것이다.
        """
        if order_by not in ("date", "value_desc", "value_asc"):
            raise ValueError(f"order_by 는 date|value_desc|value_asc 여야 합니다: {order_by!r}")
        where, params = [], []
        if company:
            where.append("company = ?"); params.append(company)
        if companies:
            where.append(f"company IN ({','.join('?' * len(companies))})"); params += companies
        if doc_group:
            where.append("doc_group = ?"); params.append(doc_group)
        if doc_id:
            where.append("doc_id = ?"); params.append(doc_id)
        if period:
            # 기준기간이 있는 정기공시는 period를 우선한다. 기준기간이 없는
            # event 공시(exchange/major/holding)는 metadata_filter와 똑같이
            # filing_date 연도로 fallback한다.
            if len(period) == 4 and period.isdigit():
                where.append(
                    "(period LIKE ? OR ((period IS NULL OR period = '') AND filing_date LIKE ?))"
                )
                params.extend([period + "-%", period + "%"])
            else:
                where.append(
                    "(period = ? OR ((period IS NULL OR period = '') AND filing_date LIKE ?))"
                )
                params.extend([period, period[:4] + "%"])
        if date_from:
            where.append("filing_date >= ?"); params.append(date_from)
        if date_to:
            where.append("filing_date <= ?"); params.append(date_to)
        if min_value is not None:
            where.append("value_num >= ?"); params.append(min_value)
        if max_value is not None:
            where.append("value_num <= ?"); params.append(max_value)
        if numeric_only:
            where.append("value_num IS NOT NULL")
        if latest_only:
            where.append("(is_latest IS NULL OR is_latest = 1)")

        def _run(key_clause: str | None, key_params: list) -> list[dict]:
            w = list(where) + ([key_clause] if key_clause else [])
            p = list(params) + key_params
            sql = "SELECT * FROM facts"
            if w:
                sql += " WHERE " + " AND ".join(w)
            if order_by == "value_desc":
                # SQLite 는 DESC 에서 NULL 을 뒤로 보낸다 — 숫자가 아닌 값이
                # 앞을 막지 않는다.
                sql += " ORDER BY value_num DESC, filing_date DESC, fact_id ASC LIMIT ?"
            elif order_by == "value_asc":
                # ASC 는 NULL 이 앞으로 오므로 명시적으로 뒤로 민다.
                sql += (" ORDER BY value_num IS NULL, value_num ASC,"
                        " filing_date DESC, fact_id ASC LIMIT ?")
            else:
                sql += " ORDER BY filing_date DESC, fact_id ASC LIMIT ?"
            rows = self.conn.execute(sql, p + [limit]).fetchall()
            return [self._row(r) for r in rows]

        if not key:
            return _run(None, [])
        exact = _run("key_norm = ?", [key])
        if exact or exact_only:
            # exact_only 는 MultiFactStore 용이다. 저장소가 여러 개일 때
            # 각자 알아서 부분일치로 넓히면, 한쪽에 정확히 일치하는 값이
            # 있는데도 다른 쪽의 엉뚱한 부분일치가 섞여 들어온다.
            # 정확일치를 **모든 저장소에서 먼저** 시도할 수 있게 열어둔다.
            return exact
        return _run("(key_norm LIKE ? OR key LIKE ?)", [f"%{key}%", f"%{key}%"])

    def distinct_keys(self, *, company: str | None = None, doc_group: str | None = None,
                      limit: int = 100) -> list[tuple[str, int]]:
        """어떤 항목을 조회할 수 있는지 알려준다 (Agent 가 키 이름을 추측하지 않게)."""
        w, p = [], []
        if company:
            w.append("company = ?"); p.append(company)
        if doc_group:
            w.append("doc_group = ?"); p.append(doc_group)
        sql = "SELECT key_norm, COUNT(*) c FROM facts"
        if w:
            sql += " WHERE " + " AND ".join(w)
        sql += " GROUP BY key_norm ORDER BY c DESC LIMIT ?"
        return [(r["key_norm"], r["c"]) for r in self.conn.execute(sql, p + [limit])]

    def stats(self) -> dict:
        c = self.conn.execute(
            "SELECT COUNT(*) n, COUNT(value_num) numeric, COUNT(value_date) dated, "
            "COUNT(chunk_id) linked, COUNT(DISTINCT company) companies, "
            "COUNT(DISTINCT doc_id) docs, COUNT(DISTINCT key_norm) keys FROM facts"
        ).fetchone()
        return dict(c)

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        try:
            d["section_path"] = json.loads(d.get("section_path") or "[]")
        except json.JSONDecodeError:
            d["section_path"] = []
        d["is_correction"] = bool(d.get("is_correction"))
        if d.get("is_latest") is not None:
            d["is_latest"] = bool(d["is_latest"])
        return d

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            del self._local.conn
