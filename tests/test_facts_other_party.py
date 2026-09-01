"""「주주에 관한 사항」의 재무수치는 그 회사 것이 아니다 (2026-09-01)."""

from __future__ import annotations

import sqlite3

import pytest

from disclosure_rag.facts.store import FactStore

# 실제 코퍼스에서 그대로 가져온 값이다. KB금융·신한지주·하나금융지주·
# POSCO홀딩스의 자산총계가 전부 464,418 로 같았고, 그건 네 회사의 최대주주가
# 모두 국민연금공단이기 때문이었다.
# 최대주주 재무현황 표에서는 값과 주인 이름이 **같은 조각** 안에 들어 있다.
# 그래서 값을 버리지 않고도 주인을 붙일 수 있다.
ROWS = [
    # (doc_id, chunk_id, company, key, value_text, section_path)
    ("periodic_1", "periodic_1::P27", "KB금융", "자산총계", "464,418",
     '["VII. 주주에 관한 사항"]'),
    ("periodic_1", "periodic_1::P27", "KB금융", "법인 또는 단체의 명칭", "국민연금공단",
     '["VII. 주주에 관한 사항"]'),
    ("periodic_2", "periodic_2::P8", "KB금융", "자산총계", "751,234",
     '["III. 재무에 관한 사항", "3. 연결재무제표 주석"]'),
]


@pytest.fixture()
def store(tmp_path):
    s = FactStore(tmp_path / "facts.sqlite")
    for i, (doc_id, chunk_id, company, key, value, section) in enumerate(ROWS, 1):
        s.conn.execute(
            "INSERT INTO facts(fact_id,doc_id,chunk_id,company,key,key_norm,"
            "value_text,filing_date,is_latest,section_path) "
            "VALUES(?,?,?,?,?,?,?,?,1,?)",
            (i, doc_id, chunk_id, company, key, key, value, "20250314", section),
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
    """최대주주 재무현황 자체를 묻는 질문에는 **답할 수 있어야 한다.**

    막는 것은 해법이 아니다. "KB금융 최대주주의 자산총계는?" 은 정당한
    질문이고, 그 값은 KB금융 사업보고서에 실제로 실려 있다.
    """
    rows = store.lookup(company="KB금융", key="자산총계", value_owner="other")
    assert [r["value_text"] for r in rows] == ["464,418"]
    # 누구 값인지가 함께 온다 — 이게 있어야 답변이 "국민연금공단의" 라고 쓴다
    assert rows[0]["value_owner"] == "국민연금공단"
    assert rows[0]["value_owner_is_company"] is False

    both = store.lookup(company="KB금융", key="자산총계", value_owner="any")
    assert {r["value_text"] for r in both} == {"464,418", "751,234"}


def test_the_companys_own_value_is_labelled_as_its_own(store):
    rows = store.lookup(company="KB금융", key="자산총계")
    assert rows[0]["value_owner"] == "KB금융"
    assert rows[0]["value_owner_is_company"] is True


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


def test_the_stored_owner_column_is_trusted_over_the_section_rule(tmp_path):
    """새로 만든 DB 는 추출 시점에 확인한 주인을 그대로 쓴다.

    섹션 규칙은 예전 artifacts 를 위한 대비책이지, 판정의 근거가 아니다.
    표 유형이 늘어날 때마다 규칙을 덧붙이는 방식으로는 못 따라간다.
    """
    s = FactStore(tmp_path / "facts.sqlite")
    s.conn.execute(
        "INSERT INTO facts(fact_id,doc_id,chunk_id,company,key,key_norm,value_text,"
        "filing_date,is_latest,section_path,value_owner) "
        "VALUES(1,'d1','d1::c1','삼성전자','매출액','매출액','999',"
        "'20250314',1,'[\"IX. 계열회사 등에 관한 사항\"]','삼성디스플레이')")
    s.conn.execute(
        "INSERT INTO facts(fact_id,doc_id,chunk_id,company,key,key_norm,value_text,"
        "filing_date,is_latest,section_path,value_owner) "
        "VALUES(2,'d1','d1::c2','삼성전자','매출액','매출액','300조',"
        "'20250314',1,'[\"III. 재무에 관한 사항\"]',NULL)")
    s.conn.commit()

    own = s.lookup(company="삼성전자", key="매출액")
    assert [r["value_text"] for r in own] == ["300조"]

    other = s.lookup(company="삼성전자", key="매출액", value_owner="other")
    assert [(r["value_text"], r["value_owner"]) for r in other] == [("999", "삼성디스플레이")]
    s.close()


def test_an_old_store_without_the_column_still_works_and_is_never_written_to(tmp_path):
    """구버전 artifacts 를 열 때 **쓰기를 시도하면 안 된다.**

    2026-09-01: 없는 컬럼을 ALTER TABLE 로 붙이려다 마운트된 디스크에서
    "disk I/O error" 로 실패하고 journal 파일만 남겼다. 평가 서버의 artifacts
    가 읽기 전용이면 기동 자체가 죽는다. 컬럼은 읽어서 확인만 하고, 질의문을
    거기에 맞춘다.
    """
    import sqlite3

    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL,"
        " chunk_id TEXT, company TEXT, key TEXT NOT NULL, key_norm TEXT NOT NULL,"
        " value_text TEXT NOT NULL, filing_date TEXT, is_latest INTEGER,"
        " is_correction INTEGER, value_num REAL, section_path TEXT);")
    old.execute(
        "INSERT INTO facts(fact_id,doc_id,chunk_id,company,key,key_norm,value_text,"
        "filing_date,is_latest,section_path) VALUES(1,'d','d::c','KB금융','자산총계',"
        "'자산총계','464,418','20250314',1,'[\"VII. 주주에 관한 사항\"]')")
    old.commit()
    old.close()

    s = FactStore(path)
    assert "value_owner" not in s.columns
    # 섹션 규칙만으로도 회사 자신의 값이 아님을 안다
    assert s.lookup(company="KB금융", key="자산총계") == []
    assert len(s.lookup(company="KB금융", key="자산총계", value_owner="other")) == 1
    s.close()

    # 스키마가 바뀌지 않았어야 한다
    check = sqlite3.connect(path)
    assert "value_owner" not in {r[1] for r in check.execute("PRAGMA table_info(facts)")}
    check.close()


# --------------------------------------------------------------------------- 이름 이어받기

def test_a_financial_table_that_follows_a_named_one_inherits_the_owner():
    """이름 없이 따라붙는 요약재무정보 표는 **앞 표의 주인** 것이다.

    실제 문서(SK하이닉스 반기보고서)에서 확인한 형태:
        표1  법인 또는 단체의 명칭 = SK스퀘어 주식회사
        표2  법인 또는 단체의 명칭 = SK주식회사
        표3  구 분 = 제33기 반기 / [유동자산]... 자산총계 199,360,274   ← 이름 없음
        표4  구 분 = 제33기 반기 / ... (별도재무제표)                  ← 이름 없음
    표3 의 자산총계는 표2 와 정확히 같은 값이다. 사람이 읽으면 "위에서 말한 그
    회사"인 게 당연해서 이름을 다시 안 적은 것이다. 이어받지 않으면 이 값들이
    SK하이닉스 것으로 저장된다.
    """
    from disclosure_rag.facts.extractor import _looks_like_financial_statement, _table_owner

    class _Pair:
        def __init__(self, key, value):
            self.key, self.value = key, value
            self.unit_value = self.field_code = self.unit_code = None

    class _KV:
        def __init__(self, pairs):
            self.pairs = [_Pair(k, v) for k, v in pairs]
            self.group_label = None

    named = _KV([("법인 또는 단체의 명칭", "SK주식회사"), ("자산총계", "199,360,274")])
    follow = _KV([("구 분", "제33기 반기"), ("[유동자산]", "66,289,793"),
                  ("자산총계", "199,360,274"), ("부채총계", "123,851,358")])
    unrelated = _KV([("성명", "홍길동"), ("보수총액", "1,200")])

    assert _table_owner(named) == "SK주식회사"
    assert _table_owner(follow) is None
    assert _looks_like_financial_statement(follow) is True
    # 임원 보수 표는 재무제표가 아니다 — 여기까지 이어받으면 "임원 보수" 질문이 막힌다
    assert _looks_like_financial_statement(unrelated) is False


def test_an_executives_name_is_not_treated_as_an_owner():
    """'성명' 을 주인 표지로 쓰면 임원 보수가 '남의 값' 이 된다.

    「VIII. 임원 및 직원 등에 관한 사항」에도 성명이 있다. 최대주주가 개인이면
    애초에 재무현황 표를 내지 않으므로(개인은 재무제표 제출 대상이 아니다)
    성명을 표지로 쓸 실익도 없다.
    """
    from disclosure_rag.facts.extractor import OWNER_NAME_KEYS
    assert not any("성명" in k for k in OWNER_NAME_KEYS)
