"""여러 facts 저장소를 하나처럼 조회.

facts 가 두 파일로 나뉘어 있다(서식 25,207건 / 사업보고서 1,070,486건).
합쳐서 다시 뽑는 대신 조회할 때 합치는데, 여기서 틀리면
  - 정확일치가 있는데 부분일치가 섞이거나
  - limit 이 두 배로 새거나
  - 같은 사실이 두 번 들어온다.
"""
from __future__ import annotations

import sqlite3

import pytest

from disclosure_rag.facts.multi_store import MultiFactStore
from disclosure_rag.facts.store import Fact, FactStore


def _store(tmp_path, name, rows):
    st = FactStore(tmp_path / f"{name}.sqlite")
    st.insert_many([Fact(**r) for r in rows])
    return st


def _fact(doc_id, key, value, date="20240101", **kw):
    base = dict(doc_id=doc_id, chunk_id=f"{doc_id}::c1", company="삼성전자",
                corp_code="0001", doc_group="exchange", doc_subtype=None,
                report_name="공시", filing_date=date, period=None,
                is_correction=False, is_latest=True, correction_group_id=doc_id,
                group_label=None, key=key, key_norm=key, value_text=value,
                value_num=None, value_unit=None, value_date=None,
                field_code=None, unit_code=None, unit_value=None, section_path=[])
    base.update(kw)
    return base


# --------------------------------------------------------------------------- 생성

def test_single_path_returns_a_plain_store(tmp_path):
    """저장소가 하나뿐이면 감쌀 이유가 없다."""
    _store(tmp_path, "a", [_fact("d1", "계약금액", "100")])
    got = MultiFactStore.from_paths([tmp_path / "a.sqlite", tmp_path / "없음.sqlite"])
    assert isinstance(got, FactStore)


def test_no_existing_path_returns_none(tmp_path):
    assert MultiFactStore.from_paths([tmp_path / "없음.sqlite"]) is None


def test_two_paths_are_wrapped(tmp_path):
    _store(tmp_path, "a", [_fact("d1", "계약금액", "100")])
    _store(tmp_path, "b", [_fact("d2", "자산총계", "200")])
    got = MultiFactStore.from_paths([tmp_path / "a.sqlite", tmp_path / "b.sqlite"])
    assert isinstance(got, MultiFactStore) and len(got.stores) == 2


# --------------------------------------------------------------------------- 조회

def test_lookup_reaches_both_stores(tmp_path):
    a = _store(tmp_path, "a", [_fact("d1", "계약금액", "100")])
    b = _store(tmp_path, "b", [_fact("d2", "자산총계", "200")])
    multi = MultiFactStore([a, b])
    assert multi.lookup(key="계약금액")[0]["value_text"] == "100"
    assert multi.lookup(key="자산총계")[0]["value_text"] == "200"


def test_exact_match_wins_over_a_partial_match_in_another_store(tmp_path):
    """한쪽에 정확일치가 있으면 다른 쪽 부분일치가 섞이면 안 된다.

    '자산총계'를 물었는데 다른 저장소의 '유동자산총계'가 답에 끼면 오답이 된다.
    """
    a = _store(tmp_path, "a", [_fact("d1", "유동자산총계", "999")])
    b = _store(tmp_path, "b", [_fact("d2", "자산총계", "200")])
    rows = MultiFactStore([a, b]).lookup(key="자산총계")
    assert [r["value_text"] for r in rows] == ["200"]


def test_partial_match_still_works_when_nothing_is_exact(tmp_path):
    a = _store(tmp_path, "a", [_fact("d1", "유동자산총계", "999")])
    b = _store(tmp_path, "b", [_fact("d2", "비유동자산총계", "888")])
    rows = MultiFactStore([a, b]).lookup(key="자산총계")
    assert {r["value_text"] for r in rows} == {"999", "888"}


def test_limit_is_not_multiplied_by_the_number_of_stores(tmp_path):
    a = _store(tmp_path, "a", [_fact(f"a{i}", "계약금액", str(i)) for i in range(10)])
    b = _store(tmp_path, "b", [_fact(f"b{i}", "계약금액", str(i)) for i in range(10)])
    assert len(MultiFactStore([a, b]).lookup(key="계약금액", limit=5)) == 5


def test_newest_first_across_stores(tmp_path):
    """FactStore 와 같은 정렬(filing_date DESC)을 유지해야 한다."""
    a = _store(tmp_path, "a", [_fact("d1", "계약금액", "옛것", date="20230101")])
    b = _store(tmp_path, "b", [_fact("d2", "계약금액", "새것", date="20250101")])
    rows = MultiFactStore([a, b]).lookup(key="계약금액")
    assert [r["value_text"] for r in rows] == ["새것", "옛것"]


def test_identical_facts_are_deduplicated(tmp_path):
    """정리 전/후 저장소를 둘 다 열면 같은 사실이 두 번 나온다."""
    row = _fact("d1", "계약금액", "100")
    a = _store(tmp_path, "a", [row])
    b = _store(tmp_path, "b", [row])
    assert len(MultiFactStore([a, b]).lookup(key="계약금액")) == 1


def test_filters_are_passed_through(tmp_path):
    a = _store(tmp_path, "a", [_fact("d1", "계약금액", "100", company="삼성전자")])
    b = _store(tmp_path, "b", [_fact("d2", "계약금액", "200", company="한미반도체")])
    rows = MultiFactStore([a, b]).lookup(key="계약금액", company="한미반도체")
    assert [r["value_text"] for r in rows] == ["200"]


# --------------------------------------------------------------------------- 항목 목록

def test_distinct_keys_sums_counts_across_stores(tmp_path):
    a = _store(tmp_path, "a", [_fact(f"a{i}", "계약금액", str(i)) for i in range(3)])
    b = _store(tmp_path, "b", [_fact(f"b{i}", "계약금액", str(i)) for i in range(2)]
                              + [_fact("b9", "자산총계", "9")])
    keys = dict(MultiFactStore([a, b]).distinct_keys(limit=100))
    assert keys["계약금액"] == 5
    assert keys["자산총계"] == 1


def test_stats_merges_sources(tmp_path):
    a = _store(tmp_path, "a", [_fact("d1", "계약금액", "100")])
    b = _store(tmp_path, "b", [_fact("d2", "자산총계", "200")])
    st = MultiFactStore([a, b]).stats()
    assert st["sources"] == 2 and len(st["by_source"]) == 2


# --------------------------------------------------------------------------- 호환

def test_exact_only_flag_does_not_change_default_behavior(tmp_path):
    """기존 호출부는 exact_only 를 모른다 — 기본값이 지금까지와 같아야 한다."""
    st = _store(tmp_path, "a", [_fact("d1", "유동자산총계", "999")])
    assert st.lookup(key="자산총계")              # 부분일치로 찾는다(기존 동작)
    assert st.lookup(key="자산총계", exact_only=True) == []
