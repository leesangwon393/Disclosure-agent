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
    section_path        TEXT,
    value_owner         TEXT
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
    "field_code", "unit_code", "unit_value", "section_path", "value_owner",
]


# 「VII. 주주에 관한 사항」의 '최대주주 및 특수관계인 현황' 표에 실리는 재무 항목.
# 이 값들의 주인은 보고서를 낸 회사가 아니라 **최대주주 법인**이다.
OTHER_PARTY_FINANCIAL_KEYS = (
    "자산총계", "부채총계", "자본총계", "매출액", "영업이익", "당기순이익",
)
OTHER_PARTY_SECTION = "주주에 관한 사항"
# 같은 표 안에서 그 수치의 주인 이름이 적혀 있는 항목.
OWNER_NAME_KEY = "법인 또는 단체의 명칭"


class FactStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        # API startup thread에서 만든 Store를 요청 worker가 사용해도 안전하도록
        # thread마다 SQLite connection을 하나씩 가진다. check_same_thread=False로
        # 하나를 공유하는 방식보다 동시 요청 사이의 cursor 충돌도 피할 수 있다.
        self._local = threading.local()
        # 이미 만들어진 저장소는 건드리지 않는다. 예전에는 열 때마다
        # CREATE TABLE/INDEX 를 다시 돌렸는데, 그건 **읽기 전용 artifacts 에서
        # 기동을 죽이고** 구버전 스키마에서는 없는 컬럼으로 인덱스를 만들려다
        # 실패한다(2026-09-01). 새로 만들 때만 스키마를 적용한다.
        existing = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='facts'"
        ).fetchone()
        if existing is None:
            self.conn.executescript(_SCHEMA)

    @property
    def columns(self) -> frozenset[str]:
        """이 저장소에 실제로 있는 컬럼.

        **읽기만 한다.** 예전엔 없는 컬럼을 ALTER TABLE 로 붙이려 했는데 그건
        틀린 설계였다(2026-09-01):
          - 평가 서버의 artifacts 는 읽기 전용일 수 있다. 조회하려고 열었다가
            쓰기를 시도하면 기동이 통째로 죽는다.
          - 실제로 마운트된 디스크에서 "disk I/O error" 로 실패하면서
            journal 파일만 남겼다.
        구버전 DB 는 컬럼 없이 그대로 두고, 질의문을 컬럼 유무에 맞춰 짠다.
        """
        cached = getattr(self, "_columns", None)
        if cached is None:
            try:
                cached = frozenset(r[1] for r in self.conn.execute("PRAGMA table_info(facts)"))
            except sqlite3.Error:
                cached = frozenset()
            self._columns = cached
        return cached

    @property
    def owner_by_chunk(self) -> dict[str, str]:
        """chunk_id -> 그 표에 적힌 수치의 주인(최대주주 법인명).

        「주주에 관한 사항」의 재무 표에는 값과 함께 '법인 또는 단체의 명칭'이
        같은 조각 안에 들어 있다. 그걸 읽어 두면 **값을 버리지 않고도** 주인을
        붙일 수 있다. 값을 막는 게 아니라 누구 것인지 적는 것이 맞는 해법이다.

        코퍼스 전체에서 770개 조각뿐이라 메모리에 들고 있어도 된다. SQL 상관
        서브쿼리로 매 조회마다 확인하면 100만 행 스캔이 되어 2분을 넘긴다
        (2026-09-01 실측: 타임아웃).
        """
        cached = getattr(self, "_owner_by_chunk", None)
        if cached is None:
            cached = {}
            try:
                for row in self.conn.execute(
                        "SELECT chunk_id, value_text FROM facts WHERE key = ?",
                        (OWNER_NAME_KEY,)):
                    if row["chunk_id"] and row["value_text"]:
                        cached.setdefault(row["chunk_id"], row["value_text"])
            except sqlite3.Error:  # 스키마가 다른 오래된 저장소
                cached = {}
            self._owner_by_chunk = cached
        return cached

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
        cols = [c for c in _COLS if c in self.columns] or list(_COLS)
        sql = f"INSERT INTO facts ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
        buf, n = [], 0
        for f in facts:
            d = f.model_dump()
            d["section_path"] = json.dumps(d.get("section_path") or [], ensure_ascii=False)
            d["is_correction"] = int(bool(d.get("is_correction")))
            d["is_latest"] = None if d.get("is_latest") is None else int(bool(d["is_latest"]))
            buf.append(tuple(d.get(c) for c in cols))
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
        corrections_only: bool = False,
        numeric_only: bool = False,
        value_owner: str = "self",
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
        if corrections_only:
            # 질문이 "[기재정정]..." 처럼 정정본을 지목했을 때만. 비정형 검색과
            # 조건을 맞춰 두 채널이 서로 다른 문서를 답하지 않게 한다.
            where.append("is_correction = 1")
        # ── 수치의 주인 ───────────────────────────────────────────────
        # 「VII. 주주에 관한 사항」의 재무 항목은 **최대주주 법인의 값**이다.
        # 그런데 company 컬럼에는 보고서를 낸 회사가 들어 있어서, 그대로 주면
        # 남의 재무제표를 그 회사 것으로 답한다.
        #
        #   KB금융 자산총계 464,418 = 신한지주 = 하나금융지주 = POSCO홀딩스
        #   (네 회사의 최대주주가 모두 국민연금공단이라 값이 같았다)
        #
        # 해결은 **막는 게 아니라 주인을 밝히는 것**이다. 그래서 값은 그대로
        # 두고, 회사 자신을 묻는 질문과 최대주주를 묻는 질문에 서로 다른 것을
        # 준다. "KB금융 최대주주의 자산총계는?" 도 답할 수 있어야 한다.
        #
        #   value_owner="self"   회사 자신의 값만 (기본)
        #   value_owner="other"  최대주주 등 제3자의 값만
        #   value_owner="any"    둘 다
        if value_owner not in ("self", "other", "any"):
            raise ValueError(
                f"value_owner 는 self|other|any 여야 합니다: {value_owner!r}")
        # 판정은 두 가지를 OR 로 본다.
        #   ① value_owner 컬럼 — 추출할 때 표에서 확인한 주인 (새로 만든 DB)
        #   ② 섹션+항목 규칙  — 그 컬럼이 없는 예전 DB 를 위한 대비책
        # 팀원이 들고 있는 artifacts 사본을 다시 만들지 않아도 동작해야 한다.
        key_ph = ",".join("?" * len(OTHER_PARTY_FINANCIAL_KEYS))
        by_section = f"(section_path LIKE ? AND key IN ({key_ph}))"
        owner_params = [f"%{OTHER_PARTY_SECTION}%", *OTHER_PARTY_FINANCIAL_KEYS]
        if "value_owner" in self.columns:
            by_column = "(value_owner IS NOT NULL AND value_owner <> IFNULL(company, ''))"
            third_party = f"({by_column} OR {by_section})"
        else:
            # 구버전 DB — 추출 시점 기록이 없으니 섹션 규칙만으로 판정한다.
            third_party = by_section
        if value_owner == "self":
            where.append(f"NOT {third_party}")
            params.extend(owner_params)
        elif value_owner == "other":
            where.append(third_party)
            params.extend(owner_params)

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
            return [self._annotate_owner(self._row(r)) for r in rows]

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

    def _annotate_owner(self, row: dict) -> dict:
        """이 수치가 누구 것인지 적어 준다.

        값을 지우는 대신 이름을 붙인다. 근거 블록이 "국민연금공단의 자산총계"
        라고 표시할 수 있어야, 최대주주를 묻는 질문에도 답하고 회사를 묻는
        질문에서 잘못 쓰이지도 않는다.
        """
        stored = row.get("value_owner")
        if stored:
            # 추출할 때 표에서 확인한 주인이다. 추정보다 이게 정확하다.
            row["value_owner_is_company"] = stored == row.get("company")
            return row
        section = row.get("section_path") or []
        is_third_party = (
            any(OTHER_PARTY_SECTION in str(part) for part in section)
            and row.get("key") in OTHER_PARTY_FINANCIAL_KEYS
        )
        if is_third_party:
            row["value_owner"] = self.owner_by_chunk.get(row.get("chunk_id") or "") or "최대주주"
            row["value_owner_is_company"] = False
        else:
            row["value_owner"] = row.get("company")
            row["value_owner_is_company"] = True
        return row

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
