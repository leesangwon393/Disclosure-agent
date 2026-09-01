"""「주주에 관한 사항」의 재무수치는 그 회사 것이 아니다 (2026-09-01)."""

from __future__ import annotations

import sqlite3

import pytest

from disclosure_rag.facts.store import FactStore

# 실제 코퍼스에서 그대로 가져온 값이다. KB금융·신한지주·하나금융지주·
# POSCO홀딩스의 자산총계가 전부 464,418 로 같았고, 그건 네 회사의 최대주주가
# 모두 국민연금공단이기 때문이었다.
ROWS = [
    # (doc_id, company, key, value_text, section_path)
    ("periodic_1", "KB금융", "자산총계", "464,418", '["VII. 주주에 관한 사항"]'),
    ("periodic_1", "KB금융", "법인 또는 단체의 명칭", "국민연금공단",
     '["VII. 주주에 관한 사항"]'),
    ("periodic_2", "KB금융", "자산총계", "751,234",
     '["III. 재무에 관한 사항", "3. 연결재무제표 주석"]'),
]


@pytest.fixture()
def store(tmp_path):
    s = FactStore(tmp_path / "facts.sqlite")
    for i, (doc_id, company, key, value, section) in enumerate(ROWS, 1):
        s.conn.execute(
            "INSERT INTO facts(fact_id,doc_id,chunk_id,company,key,key_norm,"
            "value_text,filing_date,is_latest,section_path) "
            "VALUES(?,?,?,?,?,?,?,?,1,?)",
            (i, doc_id, f"{doc_id}::c{i}", company, key, key, value, "20250314", section),
        )
    s.conn.commit()
    yield s
    s.close()


def test_largest_shareholders_numbers_are_not_returned_as_the_companys_own(store):
    """기본 조회에서 최대주주 표의 값이 나오면 안 된다.

    이게 새면 "KB금융의 자산총계는?" 에 국민연금공단의 464,418 이 나간다.
    [FACT] 블록은 '공시 표에서 뽑은 확정값'으로 제시되므로 모델이 의심 없이
    쓴다 — 비정형 근거보다 위험하다.
    """
    rows = store.lookup(company="KB금융", key="자산총계")
    values = [r["value_text"] for r in rows]
    assert "464,418" not in values
    assert values == ["751,234"]


def test_it_can_still_be_read_when_explicitly_asked_for(store):
    """최대주주 재무현황 자체를 묻는 질문까지 막지는 않는다."""
    rows = store.lookup(company="KB금융", key="자산총계", exclude_other_party=False)
    assert {r["value_text"] for r in rows} == {"464,418", "751,234"}


def test_rows_without_a_section_are_kept(store):
    """section_path 가 없는 저장소(서식 공시)를 통째로 날리면 안 된다."""
    store.conn.execute(
        "INSERT INTO facts(fact_id,doc_id,chunk_id,company,key,key_norm,"
        "value_text,filing_date,is_latest,section_path) "
        "VALUES(99,'exchange_1','exchange_1::c1','KB금융','계약금액','계약금액',"
        "'50억원','20250401',1,NULL)")
    store.conn.commit()
    rows = store.lookup(company="KB금융", key="계약금액")
    assert [r["value_text"] for r in rows] == ["50억원"]


def test_the_shareholders_name_is_still_readable(store):
    """최대주주 '이름'은 그 회사에 대한 정당한 사실이다 — 같이 날리면 안 된다.

    절을 통째로 막으면 "KB금융의 최대주주는?" 에 Facts 가 빈손이 된다.
    막을 것은 그 절의 **재무 항목**뿐이다.
    """
    rows = store.lookup(company="KB금융", key="법인 또는 단체의 명칭")
    assert [r["value_text"] for r in rows] == ["국민연금공단"]
