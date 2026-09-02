"""Reranker 파인튜닝용 학습 데이터를 만든다.

절차 (LLM 호출 0회 — 전부 규칙 기반):

    eval/gold_passages.jsonl (company/doc_group/key/gold_report_ids)
        → facts.sqlite 에서 (company, key_norm, doc_id ∈ gold_report_ids) 조회
          → chunk_id 가 연결된 행만 남김 (link_facts_to_chunks 로 이미 연결됨)
        → Positive 청크 확정
    → 같은 query 로 HybridRetriever.search() 실행 → top-K 후보
        → Positive 아닌 후보를 8가지 유형으로 분류 (Hard Negative) 또는
          easy negative 후보로 남김. 회사·기간·항목이 전부 일치하는데
          gold_report_ids 에 없는 것은 **학습에서 제외**(false negative 의심 —
          사람/Claude 검수용 별도 파일로 뺀다).

출력: out_dir/{train,val,test}.jsonl + out_dir/review_needed.jsonl

사용:
    python3 scripts/reranker/build_reranker_dataset.py \
        --gold eval/gold_passages_clean.jsonl --artifacts artifacts_v2 \
        --out eval/reranker_data

산출물 위치는 artifacts_v2/ 가 아니라 eval/ 이다 — artifacts_v2/ 는 전부
.gitignore 대상(재생성되는 대용량 산출물)이라, 재현 가능하더라도 작고
가치 있는 이 데이터셋(6.6MB)까지 거기 두면 커밋이 안 된다.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

EASY_NEGATIVES_PER_QUERY = 2
MAX_HARD_NEGATIVES_PER_QUERY = 8
TOP_K_CANDIDATES = 30


@dataclass
class Example:
    query: str
    chunk_id: str
    report_id: str
    text: str
    label: int
    neg_type: str | None = None
    query_id: str | None = None
    company: str | None = None
    period: str | None = None


def _statement_kind(section_path: list[str]) -> str:
    import unicodedata
    joined = unicodedata.normalize("NFC", " > ".join(str(p) for p in (section_path or [])))
    flat = joined.replace(" ", "")
    if "연결재무제표" in flat:
        return "연결"
    if "재무제표" in flat:
        return "별도"
    return ""


def _report_kind(chunk) -> str:
    return f"{chunk.report_type}:{chunk.report_subtype}"


def classify_negative(positive_meta: dict, candidate) -> str | None:
    """None 이면 easy negative 후보. 문자열이면 그 유형의 hard negative.

    positive_meta: {"company", "period", "key_norm", "section_path",
                     "is_correction", "correction_group_id", "is_latest", "doc_group"}
    """
    if candidate.company != positive_meta["company"]:
        # 회사가 다른데 기간·항목까지 겹치면 hard negative (유형 2), 아니면 easy.
        return "same_period_diff_company" if candidate.period == positive_meta["period"] else None

    # 여기부터 같은 회사
    if candidate.period != positive_meta["period"]:
        return "same_company_diff_year"
    if _statement_kind(candidate.section_path) != positive_meta.get("statement_kind", ""):
        return "statement_mismatch"
    if candidate.is_correction != positive_meta["is_correction"]:
        return "correction_mismatch"
    if candidate.report_type != positive_meta["doc_group"]:
        return "report_type_mismatch"
    if candidate.section_path != positive_meta["section_path"]:
        return "diff_section"
    return "topic_related_same_scope"  # 회사·기간·유형 다 같음 -> false negative 의심(리뷰로)


def load_positive_chunks(facts_dbs: list[Path], gold_rows: list[dict]) -> dict[int, list[dict]]:
    """gold 문항 id -> [{"chunk_id","report_id","company","period","doc_group",...}, ...]

    facts 는 두 DB로 나뉜다(서식공시 vs 정기공시, `facts/multi_store.py` 참고) —
    둘 다 봐야 한다. 서식 DB만 봤을 때 실측 286문항 중 188건이 매칭 실패했고,
    그중 186건이 doc_group=periodic 이었다(2026-09-02 발견).
    """
    conns = []
    for db in facts_dbs:
        if not db.exists():
            logger.warning("facts DB 없음, 건너뜀: %s", db)
            continue
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        conns.append(con)

    out: dict[int, list[dict]] = {}
    for g in gold_rows:
        report_ids = g.get("gold_report_ids") or []
        if not report_ids or not g.get("company") or not g.get("key"):
            continue
        placeholders = ",".join("?" * len(report_ids))
        rows: list[dict] = []
        for con in conns:
            found = con.execute(
                f"""SELECT DISTINCT chunk_id, doc_id, company, period, doc_group,
                           section_path, is_correction, correction_group_id, is_latest
                    FROM facts
                    WHERE company = ? AND key_norm = ? AND doc_id IN ({placeholders})
                      AND chunk_id IS NOT NULL""",
                (g["company"], _norm_key(g["key"]), *report_ids),
            ).fetchall()
            rows.extend(dict(r) for r in found)
        if rows:
            out[g["id"]] = rows
    for con in conns:
        con.close()
    return out


def _norm_key(key: str) -> str:
    from disclosure_rag.agent.field_schema import normalize_field_key
    return normalize_field_key(key)


def build(gold_path: Path, facts_dbs: list[Path], artifacts: Path, out_dir: Path,
         *, top_k: int = TOP_K_CANDIDATES, seed: int = 13) -> dict:
    from disclosure_rag.retrieval.index_bundle import load_bundle
    from disclosure_rag.retrieval.metadata_filter import RetrievalFilter

    gold_rows = [json.loads(line) for line in gold_path.open(encoding="utf-8")]
    logger.info("gold 문항 %d건 로드", len(gold_rows))

    positives_by_qid = load_positive_chunks(facts_dbs, gold_rows)
    logger.info("facts↔chunk 연결로 positive 확보된 문항 %d/%d건", len(positives_by_qid), len(gold_rows))

    bundle = load_bundle(str(artifacts))
    rng = random.Random(seed)

    all_examples: list[Example] = []
    review_needed: list[Example] = []
    skipped_no_positive = 0

    for g in gold_rows:
        qid = g["id"]
        pos_rows = positives_by_qid.get(qid)
        if not pos_rows:
            skipped_no_positive += 1
            continue

        pos_chunk_ids = {r["chunk_id"] for r in pos_rows}
        # 대표 positive 메타(첫 행 기준) — hard negative 판정 기준으로 쓴다.
        rep = pos_rows[0]
        positive_meta = {
            "company": rep["company"], "period": rep["period"], "doc_group": rep["doc_group"],
            "section_path": json.loads(rep["section_path"]) if rep["section_path"] else [],
            "is_correction": bool(rep["is_correction"]), "correction_group_id": rep["correction_group_id"],
            "is_latest": rep["is_latest"],
        }
        positive_meta["statement_kind"] = _statement_kind(positive_meta["section_path"])

        # 회사 필터 없이 전체 62만 조각에서 검색하면 정기공시 정답 조각이
        # 문서당 평균 560개 조각 속에 묻힌다(2026-09-02 발견: 필터 없이는
        # positive 16/198 밖에 안 살아남았다. 실제 서빙(dual_channel.py)도
        # QueryPlan에서 뽑은 회사로 항상 필터링하므로, 이게 리랭커가 실전에서
        # 보게 될 후보 분포에도 더 가깝다).
        flt = RetrievalFilter(companies=[g["company"]]) if g.get("company") else None
        candidates = bundle.retriever.search(g["query"], k=top_k, flt=flt)
        # 회사 필터를 걸면 "다른 회사" 계열 negative(유형 2)와 순수 easy
        # negative가 후보 풀에서 아예 사라진다. 별도로 무필터 검색을 한 번
        # 더 해서 그쪽 negative 후보를 보충한다(적은 수만 필요하다).
        unfiltered = bundle.retriever.search(g["query"], k=10)

        hard_by_type: dict[str, list] = defaultdict(list)
        easy_pool: list = []
        seen_chunk_ids: set[str] = set()

        for chunk, _score in list(candidates) + list(unfiltered):
            if chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.chunk_id)

            # facts.sqlite의 chunk_id 연결이 정기공시는 **parent 레벨**로 돼
            # 있다(2026-09-02 발견). 검색 인덱스엔 leaf만 있어서 정확 일치가
            # 거의 안 되고(positive 16/198), leaf.parent_chunk_id 로 견주면
            # 잡힌다. 서식공시(major 등)는 원래 leaf로 연결돼 있어 이 조건이
            # 덧붙는다고 잘못 잡힐 일은 없다(parent_chunk_id 도 pos_chunk_ids
            # 안에 없으면 그냥 False).
            if chunk.chunk_id in pos_chunk_ids or chunk.parent_chunk_id in pos_chunk_ids:
                all_examples.append(Example(
                    query=g["query"], chunk_id=chunk.chunk_id, report_id=chunk.report_id,
                    text=chunk.raw_text, label=1, query_id=str(qid),
                    company=chunk.company, period=chunk.period,
                ))
                continue

            neg_type = classify_negative(positive_meta, chunk)
            ex = Example(
                query=g["query"], chunk_id=chunk.chunk_id, report_id=chunk.report_id,
                text=chunk.raw_text, label=0, neg_type=neg_type, query_id=str(qid),
                company=chunk.company, period=chunk.period,
            )
            if neg_type == "topic_related_same_scope":
                review_needed.append(ex)  # false negative 의심 -> 학습에서 뺀다
            elif neg_type is not None:
                hard_by_type[neg_type].append(ex)
            else:
                easy_pool.append(ex)

        # 유형별로 최소 1개씩 우선, 상한까지 채운다 (한 유형이 후보를 독식하지 않게)
        hard_picked: list[Example] = []
        types = list(hard_by_type)
        rng.shuffle(types)
        while len(hard_picked) < MAX_HARD_NEGATIVES_PER_QUERY and any(hard_by_type.values()):
            for t in types:
                if hard_by_type[t]:
                    hard_picked.append(hard_by_type[t].pop(0))
                    if len(hard_picked) >= MAX_HARD_NEGATIVES_PER_QUERY:
                        break
        all_examples.extend(hard_picked)

        rng.shuffle(easy_pool)
        all_examples.extend(easy_pool[:EASY_NEGATIVES_PER_QUERY])

    logger.info("positive 없어 스킵된 문항: %d건", skipped_no_positive)
    logger.info("학습 후보 %d건 (positive %d / hard-neg %d / easy-neg %d), 리뷰 대기 %d건",
               len(all_examples),
               sum(1 for e in all_examples if e.label == 1),
               sum(1 for e in all_examples if e.label == 0 and e.neg_type not in (None, "topic_related_same_scope")),
               sum(1 for e in all_examples if e.label == 0 and e.neg_type is None),
               len(review_needed))

    # ---- query 단위 split (8절: leakage 방지) ----
    qids = sorted({e.query_id for e in all_examples})
    rng.shuffle(qids)
    n = len(qids)
    n_test = max(1, int(n * 0.15))
    n_val = max(1, int(n * 0.15))
    test_qids = set(qids[:n_test])
    val_qids = set(qids[n_test:n_test + n_val])
    train_qids = set(qids[n_test + n_val:])

    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_qids, "val": val_qids, "test": test_qids}
    counts = {}
    for name, qid_set in splits.items():
        rows = [e for e in all_examples if e.query_id in qid_set]
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
        counts[name] = {"queries": len(qid_set), "examples": len(rows)}

    with (out_dir / "review_needed.jsonl").open("w", encoding="utf-8") as f:
        for e in review_needed:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

    meta = {"n_gold": len(gold_rows), "n_with_positive": len(positives_by_qid),
            "skipped_no_positive": skipped_no_positive, "splits": counts,
            "review_needed": len(review_needed)}
    (out_dir / "dataset_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/gold_passages_clean.jsonl")
    ap.add_argument("--facts", action="append",
                    default=None,
                    help="facts.sqlite 경로. 여러 번 줄 수 있다(서식+정기공시). "
                         "안 주면 artifacts_v2/facts 와 artifacts_v2/facts_periodic_v2_fix "
                         "(없으면 facts_periodic_v2)를 자동으로 찾는다.")
    ap.add_argument("--artifacts", default="artifacts_v2")
    ap.add_argument("--out", default="eval/reranker_data")
    ap.add_argument("--top-k", type=int, default=TOP_K_CANDIDATES)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.facts:
        facts_dbs = [Path(p) for p in args.facts]
    else:
        root = Path(args.artifacts)
        facts_dbs = [root / "facts" / "facts.sqlite"]
        for name in ("facts_periodic_v2_fix", "facts_periodic_v2", "facts_periodic"):
            p = root / name / "facts.sqlite"
            if p.exists():
                facts_dbs.append(p)
                break
    logger.info("facts DB: %s", [str(p) for p in facts_dbs])

    meta = build(Path(args.gold), facts_dbs, Path(args.artifacts), Path(args.out),
                top_k=args.top_k)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
