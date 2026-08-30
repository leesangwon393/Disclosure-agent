"""Parent Expansion (small-to-big).

검색은 **작은 조각**으로 해야 정확하고, 답변에는 **그 조각이 속한 섹션 전체**가 있어야
맥락이 산다. 그래서 child 로 찾고 parent 로 넓힌다.

기존 파이프라인은 parent 를 만들어 놓고 **어디에도 저장하지 않아서**(전체 청크 문자의
49.4%를 생성 즉시 폐기) Qdrant payload 의 `parent_chunk_id` 로 조회할 대상이 없었다.
L1 스냅샷이 parent 를 보존하면서 이 조회가 가능해졌다.

문헌 근거: arXiv 2511.18177 — small-to-big 검색이 65% 승률, 지연 오버헤드 0.2초.
"""

from __future__ import annotations

from disclosure_rag.chunking.chunk_schema import ChunkSchema

Ranked = list[tuple[ChunkSchema, float]]


class ParentExpander:
    """parent_map: chunk_id -> parent raw_text (l1.load_parent_map() 결과).

    max_chars: parent 가 매우 길 수 있으므로(섹션 전체) 상한을 둔다. 평가 규격상
    `retrieved_context` 길이 초과분은 채점에 반영되지 않으므로 무한정 넣으면 손해다.
    """

    @staticmethod
    def build_table_map(chunks) -> dict[str, list[str]]:
        """table_id -> 그 표에서 나온 조각 id 들(문서 내 순서 유지).

        한 표가 여러 조각으로 쪼개졌을 때 서로를 찾기 위한 색인이다. 조각 하나가
        표 두 개에 걸칠 수도 있어서(패킹이 표 경계를 넘어 묶는 경우) 값은 리스트다.
        """
        m: dict[str, list[str]] = {}
        for c in chunks:
            for tid in (getattr(c, "table_ids", None) or []):
                m.setdefault(tid, []).append(c.chunk_id)
        return m

    def __init__(self, parent_map: dict[str, str], *, max_chars: int = 4000,
                 table_map: dict[str, list[str]] | None = None,
                 chunk_by_id: dict | None = None):
        self.parent_map = parent_map
        self.max_chars = max_chars
        # 형제 조각 확장용. 없으면 기능이 꺼진 것과 같다(기존 동작 그대로).
        self.table_map = table_map or {}
        self.chunk_by_id = chunk_by_id or {}

    def expand_one(self, chunk: ChunkSchema) -> str:
        """조각 하나를 parent 본문으로 넓힌다. parent 가 없으면 원래 조각 그대로."""
        if not chunk.parent_chunk_id:
            return chunk.raw_text
        parent = self.parent_map.get(chunk.parent_chunk_id)
        if not parent:
            return chunk.raw_text
        if len(parent) <= self.max_chars:
            return parent
        # parent 가 너무 길면 **찾은 조각 주변**을 잘라낸다 — 앞부분을 무조건 자르면
        # 정작 검색된 내용이 빠질 수 있다.
        pos = parent.find(chunk.raw_text[:120])
        if pos < 0:
            return parent[: self.max_chars]
        half = self.max_chars // 2
        start = max(0, pos - half)
        return parent[start : start + self.max_chars]

    def _siblings_for(self, chunk, already: str) -> tuple[str, list[str]]:
        """이 조각이 표의 일부라면, 같은 표의 다른 조각 중 아직 안 들어간 것을 잇는다.

        왜 필요한가: 표는 토큰 예산 때문에 여러 조각으로 쪼개진다. "합계" 같은
        상위 라벨이 앞 조각에, 정답 숫자가 뒤 조각에 남으면 숫자의 소속을 알 수
        없다. parent 확장이 대개 이걸 덮지만, parent 가 매우 길어 잘려나가면
        놓친다 — 그때를 위한 안전망이다.
        """
        tids = getattr(chunk, "table_ids", None) or []
        if not tids or not self.table_map:
            return "", []
        parts, ids = [], []
        for tid in tids:
            for cid in self.table_map.get(tid, []):
                if cid == chunk.chunk_id:
                    continue
                sib = self.chunk_by_id.get(cid)
                if sib is None:
                    continue
                raw = getattr(sib, "raw_text", "") or ""
                if not raw or raw[:80] in already:   # 이미 들어있으면 건너뛴다
                    continue
                parts.append(raw)
                ids.append(cid)
        if not parts:
            return "", []
        joined = "\n".join(parts)
        return joined[: self.max_chars], ids

    def expand(self, ranked: Ranked, *, budget_chars: int | None = None,
               min_evidences: int = 5) -> list[dict]:
        """검색 결과를 (근거 텍스트 + 출처) 리스트로 만든다.

        같은 parent 를 가진 조각이 여러 개 나오면 **한 번만** 넣는다 — 중복 근거가
        컨텍스트 예산을 잡아먹는 것을 막는다.

        min_evidences: 근거 하나가 예산을 통째로 먹지 못하게 1인분 상한을 둔다.
        (실측: 3,000자 예산에 parent 하나가 3,000자를 차지해 나머지 근거가 전부
        탈락하는 일이 있었다. 섹션 전체가 parent 이므로 흔한 일이다.)
        """
        out: list[dict] = []
        used_parents: set[str] = set()
        total = 0
        share = (budget_chars // max(1, min_evidences)) if budget_chars else None
        for chunk, score in ranked:
            pid = chunk.parent_chunk_id
            if pid and pid in used_parents:
                continue
            text = self.expand_one(chunk)
            if share is not None and len(text) > share:
                # 1인분 상한 초과 -> 찾은 조각 주변만 남긴다(앞부분을 무조건 자르지 않는다)
                pos = text.find(chunk.raw_text[:120])
                if pos < 0:
                    text = text[:share]
                else:
                    start = max(0, pos - share // 2)
                    text = text[start : start + share]
            if budget_chars is not None and total + len(text) > budget_chars:
                text = text[: max(0, budget_chars - total)]
                if not text:
                    break
            if pid:
                used_parents.add(pid)
            total += len(text)
            sib_text, sib_ids = self._siblings_for(chunk, text)
            if sib_text:
                if budget_chars is not None and total + len(text) + len(sib_text) > budget_chars:
                    sib_text, sib_ids = "", []
                else:
                    text = text + "\n" + sib_text
            out.append({
                "table_sibling_ids": sib_ids,
                "chunk_id": chunk.chunk_id, "parent_chunk_id": pid,
                "report_id": chunk.report_id, "company": chunk.company,
                "report_name": chunk.report_name, "period": chunk.period,
                "filing_date": chunk.filing_date, "section_path": chunk.section_path,
                "is_correction": chunk.is_correction, "is_latest": chunk.is_latest,
                "score": round(float(score), 4), "text": text,
                "expanded": bool(pid and self.parent_map.get(pid)),
            })
            if budget_chars is not None and total >= budget_chars:
                break
        return out
