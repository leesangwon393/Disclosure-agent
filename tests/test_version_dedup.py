"""Stage 7 버전 정리.

이 단계가 틀리면 두 방향으로 죽는다.
  - 너무 많이 버리면: 정정 diff 질문에서 최초본이 사라져 답이 안 나온다
  - 너무 적게 버리면: 거의 같은 정정본이 상위 k칸을 채워 정보량이 줄어든다
그래서 정책별로 '무엇이 남는가'를 하나씩 고정한다.
"""
from __future__ import annotations

from dataclasses import dataclass

from disclosure_rag.agent.version_dedup import (
    deduplicate_scored,
    deduplicate_versions,
)


@dataclass
class C:
    """청크 최소 형태."""
    report_id: str
    correction_group_id: str | None = None
    correction_order: int | None = 0
    is_latest: bool | None = True
    tag: str = ""


def _chain():
    """원본 1 + 정정 2 = 문서 3건짜리 그룹. 실제 데이터에 있는 형태다
    (major_20241115000375 그룹)."""
    g = "doc_a"
    return [
        C("doc_a", g, 0, False, "원본"),
        C("doc_b", g, 1, False, "정정1"),
        C("doc_c", g, 2, True, "최종"),
    ]


def ids(chunks):
    return [c.report_id for c in chunks]


# --------------------------------------------------------------------------- latest_only

def test_latest_only_keeps_the_final_version():
    kept, rep = deduplicate_versions(_chain(), "latest_only")
    assert ids(kept) == ["doc_c"]
    assert rep.dropped == 2


def test_latest_only_falls_back_when_final_is_not_retrieved():
    """최종본이 검색에 안 잡혔다고 전부 버리면 근거가 0건이 된다 —
    회수된 것 중 가장 최신을 남긴다."""
    partial = [c for c in _chain() if c.report_id != "doc_c"]
    kept, _ = deduplicate_versions(partial, "latest_only")
    assert ids(kept) == ["doc_b"]


def test_multiple_is_latest_picks_the_highest_order():
    """데이터 이상으로 is_latest 가 여러 건일 때도 하나만 남아야 한다."""
    chunks = [C("d0", "g", 0, True), C("d1", "g", 1, True)]
    kept, _ = deduplicate_versions(chunks, "latest_only")
    assert ids(kept) == ["d1"]


# --------------------------------------------------------------------------- first_and_final

def test_first_and_final_keeps_both_ends():
    """S023~S026 — 최초 공시와 최종 정정본을 비교하는 질문."""
    kept, _ = deduplicate_versions(_chain(), "first_and_final")
    assert ids(kept) == ["doc_a", "doc_c"]


def test_first_and_final_drops_only_the_middle():
    chain = _chain() + [C("doc_b2", "doc_a", 3, False, "정정3")]
    chain[2].is_latest = False
    chain[-1].is_latest = True
    kept, _ = deduplicate_versions(chain, "first_and_final")
    assert ids(kept) == ["doc_a", "doc_b2"]


def test_first_and_final_on_a_single_version_keeps_it():
    single = [C("only", "only", 0, True)]
    kept, _ = deduplicate_versions(single, "first_and_final")
    assert ids(kept) == ["only"]


# --------------------------------------------------------------------------- all_versions

def test_all_versions_changes_nothing():
    chain = _chain()
    kept, rep = deduplicate_versions(chain, "all_versions")
    assert ids(kept) == ids(chain)
    assert rep.dropped == 0 and not rep.changed


# --------------------------------------------------------------------------- 안전장치

def test_rank_order_is_preserved():
    """리랭킹 뒤에 도는 단계다 — 걸러내기만 하고 재정렬하면 안 된다."""
    chunks = [C("doc_c", "doc_a", 2, True), C("x", None, 0, True),
              C("doc_a", "doc_a", 0, False), C("y", None, 0, True)]
    kept, _ = deduplicate_versions(chunks, "latest_only")
    assert ids(kept) == ["doc_c", "x", "y"]


def test_chunks_without_group_info_are_never_dropped():
    """그룹 정보가 없으면 판단 근거가 없다 — 버리면 안 된다."""
    chunks = [C("a", None, None, None), C("b", None, None, None)]
    kept, rep = deduplicate_versions(chunks, "latest_only")
    assert ids(kept) == ["a", "b"] and rep.dropped == 0


def test_same_document_chunks_are_not_split():
    """같은 문서의 청크가 정책에 따라 갈리면 안 된다."""
    chunks = [C("doc_c", "doc_a", 2, True, "c1"), C("doc_c", "doc_a", 2, True, "c2"),
              C("doc_a", "doc_a", 0, False, "a1")]
    kept, _ = deduplicate_versions(chunks, "latest_only")
    assert [c.tag for c in kept] == ["c1", "c2"]


def test_independent_groups_do_not_interfere():
    chunks = [C("a1", "a1", 0, False), C("a2", "a1", 1, True),
              C("b1", "b1", 0, True)]
    kept, _ = deduplicate_versions(chunks, "latest_only")
    assert ids(kept) == ["a2", "b1"]


def test_empty_input():
    kept, rep = deduplicate_versions([], "latest_only")
    assert kept == [] and rep.dropped == 0


def test_scored_pairs_keep_their_scores():
    """검색기 출력은 (chunk, score) 형태다."""
    chain = _chain()
    scored = [(chain[0], 0.9), (chain[2], 0.7)]
    kept, _ = deduplicate_scored(scored, "latest_only")
    assert kept == [(chain[2], 0.7)]


def test_report_records_what_was_dropped():
    """무엇을 왜 버렸는지 남아야 디버깅이 된다."""
    _kept, rep = deduplicate_versions(_chain(), "latest_only")
    assert rep.policy == "latest_only"
    assert rep.dropped_report_ids == ["doc_a", "doc_b"]
    assert rep.groups_seen == 1
    assert rep.changed


def test_dict_chunks_are_accepted():
    chunks = [{"report_id": "d0", "correction_group_id": "g", "correction_order": 0,
               "is_latest": False},
              {"report_id": "d1", "correction_group_id": "g", "correction_order": 1,
               "is_latest": True}]
    kept, _ = deduplicate_versions(chunks, "latest_only")
    assert [c["report_id"] for c in kept] == ["d1"]
