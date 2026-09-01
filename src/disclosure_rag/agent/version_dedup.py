"""Stage 7: 정정 전/후본 정리.

## 왜 필요한가

DART 는 정정할 때 **새 문서를 발행한다**. 원본과 정정본이 따로 존재하고,
실측상 정정공시의 43%가 원본과 텍스트가 거의 같다. 정리하지 않으면 검색
상위권을 같은 내용이 채워서, 상위 k칸 중 실제로 볼 수 있는 정보량이 줄어든다.

## 왜 지금까지 안 됐나

정정 그래프는 이미 있고(`correction_graph_builder`), 도구로도 연결돼 있다
(`get_correction_history` / `get_latest_report`). 그런데 그건 **HCX 가 부를지
말지 결정하는 도구**다. 안 부르면 그만이다. 이 모듈은 정책에 따라 **항상
실행되는 결정론적 단계**로 승격시킨다.

## 재빌드가 필요 없다

필요한 정보가 이미 청크마다 박혀 있다(실측, 20만 청크 표본):

    correction_group_id  100%   원본 문서 id = 그룹 키
    correction_order     100%   0=원본, 1,2,…=정정 순서
    is_latest            100%   True 80.8% / False 19.2%

(과거에 correction_order 채움률을 15.4%로 잰 적이 있는데 그건 측정 오류였다 —
값 `0` 을 '비어 있음'으로 세는 falsy 검사 때문이었다. 실제로는 100% 채워져 있다.)

## 정책

    latest_only      최종본만          일반 조회
    first_and_final  최초 + 최종만      정정 diff 질문 (S023~S026)
    all_versions     전부              정정 이력 전체 질문

## 순서를 바꾸지 않는다

이 단계는 리랭킹 **뒤에** 온다. 입력 목록은 이미 관련도 순으로 정렬돼 있으므로
걸러내기만 하고 재정렬하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from disclosure_rag.agent.query_plan import LatestPolicy


def _attr(obj: Any, name: str, default=None):
    """청크 객체와 (doc, score) 튜플·dict 를 모두 받아준다."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class DedupReport:
    policy: LatestPolicy
    kept: int = 0
    dropped: int = 0
    groups_seen: int = 0
    dropped_report_ids: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.dropped > 0


def _group_key(chunk: Any) -> str | None:
    gid = _attr(chunk, "correction_group_id")
    return gid or None


def _order(chunk: Any) -> int:
    """정정 체인 안에서의 순서.

    `correction_order` 가 없으면 **접수일**로 대신한다. Facts(sqlite) 행에는
    그 컬럼이 아예 없어서, 예전에는 전부 0 이 되어 "가장 최신을 남긴다" 는
    대비책이 사실상 아무거나 남기고 있었다(2026-09-01 발견).
    """
    o = _attr(chunk, "correction_order")
    if o is not None:
        try:
            return int(o)
        except (TypeError, ValueError):
            pass
    filing = str(_attr(chunk, "filing_date") or "")
    return int(filing) if filing.isdigit() else 0


def _in_window(chunk: Any, periods: Sequence[str] | None) -> bool:
    """이 문서가 질문이 지목한 기간(YYYY-MM 또는 YYYY) 안에 있는가."""
    if not periods:
        return True
    filing = str(_attr(chunk, "filing_date") or "")
    period = str(_attr(chunk, "period") or "")
    for want in periods:
        want = str(want)
        if len(want) >= 7:                      # YYYY-MM
            head = want[:4] + want[5:7]         # YYYYMM
            if filing.startswith(head) or period.startswith(want):
                return True
        else:                                   # YYYY
            if filing.startswith(want) or period.startswith(want):
                return True
    return False


def _select_report_ids(chunks: Sequence[Any], policy: LatestPolicy,
                       periods: Sequence[str] | None = None) -> set[str] | None:
    """정책에 따라 **남길 문서 id 집합**을 정한다. None 이면 전부 남긴다.

    문서 단위로 정하고 청크는 그 결과를 따른다 — 같은 문서의 청크가 정책에
    따라 갈리면 안 되기 때문이다.
    """
    if policy == "all_versions":
        return None

    # 그룹 -> {report_id: (order, is_latest)}
    groups: dict[str, dict[str, tuple[int, Any]]] = {}
    ungrouped: set[str] = set()
    for c in chunks:
        rid = str(_attr(c, "report_id") or "")
        if not rid:
            continue
        gid = _group_key(c)
        if gid is None:
            ungrouped.add(rid)      # 그룹 정보가 없으면 판단 근거가 없다 -> 남긴다
            continue
        groups.setdefault(gid, {})[rid] = (_order(c), _attr(c, "is_latest"))

    # latest_in_window 는 문서별 접수일이 필요하므로 따로 모아 둔다.
    in_window: dict[str, bool] = {}
    for c in chunks:
        rid = str(_attr(c, "report_id") or "")
        if rid and rid not in in_window:
            in_window[rid] = _in_window(c, periods)

    keep: set[str] = set(ungrouped)
    for _gid, members in groups.items():
        if len(members) == 1:
            keep.update(members)
            continue
        orders = {rid: o for rid, (o, _l) in members.items()}
        if policy == "latest_in_window":
            # 질문이 지목한 기간 안에서 가장 늦은 1건. 그 안에 아무것도 없으면
            # 체인 전체에서 가장 늦은 1건으로 물러선다.
            #
            # **끄는 게 아니라 기준 시점을 옮기는 것**이다. 5월에 정정본이
            # 3건이면 5월 안에서 최신(05-24)을 고른다. 코퍼스 전체의 최종본
            # (2026-01-20)을 고르지 않는다.
            candidates = [rid for rid in orders if in_window.get(rid, True)]
            keep.add(max(candidates or list(orders), key=lambda r: orders[r]))
            continue
        if policy == "first_and_final":
            keep.add(min(orders, key=lambda r: orders[r]))
            keep.add(max(orders, key=lambda r: orders[r]))
            continue
        # latest_only
        latest = [rid for rid, (_o, is_latest) in members.items() if is_latest]
        if latest:
            # is_latest 가 여러 건이면(데이터 이상) order 가 가장 큰 것을 쓴다
            keep.add(max(latest, key=lambda r: orders[r]))
        else:
            # 이 그룹의 최종본이 검색 결과에 안 들어온 경우다. 그렇다고 전부
            # 버리면 근거가 0건이 되므로, 회수된 것 중 가장 최신을 남긴다.
            keep.add(max(orders, key=lambda r: orders[r]))
    return keep


def deduplicate_versions(
    chunks: Iterable[Any], policy: LatestPolicy = "latest_only",
    *, periods: Sequence[str] | None = None,
) -> tuple[list[Any], DedupReport]:
    """정책에 맞는 버전의 청크만 남긴다. 입력 순서는 그대로 유지한다.

    `periods` 는 `latest_in_window` 에서만 쓴다 — 질문이 지목한 기간이다.
    """
    items = list(chunks)
    report = DedupReport(policy=policy)
    keep = _select_report_ids(items, policy, periods)

    if keep is None:
        report.kept = len(items)
        return items, report

    report.groups_seen = len({_group_key(c) for c in items if _group_key(c)})
    out: list[Any] = []
    dropped: list[str] = []
    for c in items:
        rid = str(_attr(c, "report_id") or "")
        if not rid or rid in keep:
            out.append(c)
        else:
            dropped.append(rid)
    report.kept = len(out)
    report.dropped = len(dropped)
    report.dropped_report_ids = sorted(set(dropped))
    return out, report


def deduplicate_scored(
    scored: Iterable[tuple[Any, float]], policy: LatestPolicy = "latest_only",
    *, periods: Sequence[str] | None = None,
) -> tuple[list[tuple[Any, float]], DedupReport]:
    """`(chunk, score)` 목록용. 검색기 출력이 이 형태다."""
    pairs = list(scored)
    kept, report = deduplicate_versions([c for c, _s in pairs], policy, periods=periods)
    keep_ids = {id(c) for c in kept}
    return [p for p in pairs if id(p[0]) in keep_ids], report


__all__ = ["deduplicate_versions", "deduplicate_scored", "DedupReport"]
