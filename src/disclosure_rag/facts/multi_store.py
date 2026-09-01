"""여러 facts 저장소를 하나처럼 조회한다.

## 왜 필요한가

facts 가 두 파일로 나뉘어 있다.

    artifacts_v2/facts/facts.sqlite              서식 공시     25,207건 (17MB)
    artifacts_v2/facts_periodic_v2/facts.sqlite  사업보고서 1,070,486건 (635MB)

따로 뽑은 이유는 서식 공시(exchange/major/holding)와 재무제표(periodic)의 표
구조가 달라 추출 규칙과 잡음 필터가 다르기 때문이다. 합쳐서 다시 뽑을 수도
있지만, 그러면 두 시간 넘게 걸리고 서식 쪽 25,207건이 회귀할 위험을 진다.
**조회할 때 합치는 편이 싸고 안전하다.**

## 인터페이스를 안 바꾼다

`DualChannelRetriever` 가 쓰는 건 `lookup()` 과 `distinct_keys()` 둘뿐이다.
이 클래스가 같은 시그니처를 제공하므로 호출부는 바뀌지 않는다.

## 정확일치를 먼저, 모든 저장소에서

`FactStore.lookup()` 은 정확일치가 없으면 부분일치로 넓힌다. 저장소가 여러
개일 때 각자 그걸 하면 이런 일이 난다:

    질문 항목: "자산총계"
      periodic 에 정확일치 있음        -> 이게 답
      서식 쪽엔 없어서 부분일치로 넓힘  -> "유동자산총계" 같은 게 섞여 들어옴

그래서 **정확일치를 전 저장소에서 먼저 시도**하고, 하나도 없을 때만 부분일치로
내려간다(`exact_only` 인자).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


class MultiFactStore:
    def __init__(self, stores: Sequence):
        self.stores = [s for s in stores if s is not None]
        if not self.stores:
            raise ValueError("facts 저장소가 하나도 없습니다")

    # ------------------------------------------------------------------ 생성

    @classmethod
    def from_paths(cls, paths: Sequence[str | Path]) -> "MultiFactStore | None":
        """존재하는 파일만 연다. 하나도 없으면 None."""
        from disclosure_rag.facts.store import FactStore

        stores = []
        for path in paths:
            p = Path(path)
            if not p.exists():
                continue
            stores.append(FactStore(p))
            logger.info("[FACTS] %s 연결", p)
        if not stores:
            return None
        if len(stores) == 1:
            return stores[0]      # 하나뿐이면 감쌀 이유가 없다
        return cls(stores)

    # ------------------------------------------------------------------ 조회

    @staticmethod
    def _merge(batches: list[list[dict]], limit: int, order_by: str = "date") -> list[dict]:
        """저장소별 결과를 합친다.

        `FactStore.lookup` 과 **같은 기준**으로 다시 세운다. 여기서 정렬을
        빠뜨리면 저장소 A 의 10등이 저장소 B 의 1등보다 앞에 온다.
        `fact_id` 는 저장소마다 독립이라 비교 의미가 없어 저장소 순서를
        타이브레이커로 쓴다(안정 정렬).
        """
        rows: list[dict] = [row for batch in batches for row in batch]

        if order_by == "value_desc":
            rows.sort(key=lambda r: (r.get("value_num") is None,
                                     -(r.get("value_num") or 0.0),
                                     str(r.get("filing_date") or "")))
        elif order_by == "value_asc":
            rows.sort(key=lambda r: (r.get("value_num") is None,
                                     (r.get("value_num") or 0.0),
                                     str(r.get("filing_date") or "")))
        else:
            rows.sort(key=lambda r: str(r.get("filing_date") or ""), reverse=True)
        out, seen = [], set()
        for row in rows:
            sig = (row.get("doc_id"), row.get("chunk_id"),
                   row.get("key_norm"), row.get("value_text"))
            if sig in seen:
                continue
            seen.add(sig)
            out.append(row)
            if len(out) >= limit:
                break
        return out

    def lookup(self, *, key: str | None = None, limit: int = 20,
               order_by: str = "date", **kw) -> list[dict]:
        if key:
            exact = self._merge(
                [s.lookup(key=key, limit=limit, exact_only=True, order_by=order_by, **kw)
                 for s in self.stores], limit, order_by)
            if exact:
                return exact
        return self._merge(
            [s.lookup(key=key, limit=limit, order_by=order_by, **kw) for s in self.stores],
            limit, order_by)

    @property
    def owner_by_chunk(self) -> dict[str, str]:
        """저장소 여러 개의 '조각 -> 수치 주인' 지도를 합친다."""
        merged: dict[str, str] = {}
        for store in self.stores:
            try:
                merged.update(store.owner_by_chunk)
            except Exception:  # noqa: BLE001
                continue
        return merged

    def distinct_keys(self, *, limit: int = 100, **kw) -> list[tuple[str, int]]:
        """항목별 건수를 저장소끼리 **더해서** 돌려준다."""
        totals: dict[str, int] = {}
        for store in self.stores:
            for key, count in store.distinct_keys(limit=limit, **kw):
                totals[key] = totals.get(key, 0) + count
        return sorted(totals.items(), key=lambda kv: -kv[1])[:limit]

    def stats(self) -> dict:
        merged: dict = {"sources": len(self.stores), "by_source": []}
        for store in self.stores:
            st = store.stats()
            merged["by_source"].append({"path": getattr(store, "path", "?"), **st})
            for k, v in st.items():
                if isinstance(v, (int, float)):
                    merged[k] = merged.get(k, 0) + v
        return merged

    def close(self) -> None:
        for store in self.stores:
            close = getattr(store, "close", None)
            if close:
                close()


__all__ = ["MultiFactStore"]
