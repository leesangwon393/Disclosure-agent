"""채점기 규칙 테스트 — 채점이 틀리면 이후 모든 판단이 틀린다.

특히 표기 차이(콤마·단위)를 오답 처리하면 개선이 있어도 안 보이고,
반대로 너무 느슨하면 없는 개선이 보인다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from score_answers import _answer_hit, _gold_answers, _is_refusal, _label, _norm  # noqa: E402


def test_comma_notation_is_ignored():
    """gold 는 `224,787,773,988,054`, 답변은 콤마 없이 쓸 수 있다."""
    assert _answer_hit("순자산액은 224787773988054원입니다", ["224,787,773,988,054"])
    assert _answer_hit("순자산액은 224,787,773,988,054원입니다", ["224787773988054"])


def test_any_of_multiple_gold_answers_counts():
    """한 질문에 여러 공시가 걸리면 정답도 여러 개다(gold 파일 실측)."""
    golds = ["27,211,951,962,825", "26,088,427,058,810", "29,360,152,532,579"]
    assert _answer_hit("26,088,427,058,810 입니다", golds)


def test_wrong_number_is_not_a_hit():
    assert not _answer_hit("순자산액은 224,787,773,988,055원입니다", ["224,787,773,988,054"])


def test_rescaled_number_is_not_a_hit():
    """`7,661,584백만원` 은 `7,661,584,000,000` 의 정답 표기가 아니다."""
    assert not _answer_hit("7,661,584백만원입니다", ["7,661,584,000,000"])


def test_substring_of_a_longer_number_is_not_a_hit():
    """`1234` 가 `91234567` 안에 있다고 정답 처리하면 안 된다."""
    assert not _answer_hit("금액은 91,234,567원입니다", ["1234"])


def test_norm_strips_currency_and_space():
    assert _norm("224,787,773,988,054 원") == "224787773988054"


def test_refusal_detection():
    assert _is_refusal("제공된 근거로는 확인할 수 없습니다.")
    assert not _is_refusal("영업이익은 1,200억원입니다.")


def test_label_separates_search_failure_from_answer_failure():
    """이 구분이 이 채점기의 핵심 산출물이다."""
    assert _label(True, True, False) == "정답"
    assert _label(False, True, False) == "답변실패"   # 근거는 왔는데 답을 못 함
    assert _label(False, False, False) == "검색실패"  # 근거가 아예 없음
    assert _label(False, False, True) == "거부(검색실패)"
    assert _label(False, True, True) == "거부(근거있음)"


def test_gold_answers_merges_answers_and_answer():
    row = {"answers": ["1", "2"], "answer": "1"}
    assert _gold_answers(row) == ["1", "2"]
    assert _gold_answers({"answer": "5"}) == ["5"]
    assert _gold_answers({"answer": "", "answers": []}) == []


# --------------------------------------------------------------------------- 집계 (2026-08-30)

from score_answers import _aggregate, _write  # noqa: E402


def _retrieval_row(**kw):
    base = {
        "id": "x", "query": "q", "company": "c", "doc_group": "periodic",
        "gold": "1", "n_gold": 1, "n_gold_docs": 1,
        "evidence_hit": 1, "answer_ceiling": 1,
        "context_recall": 1.0, "context_precision": 0.2, "context_ap": 1.0,
        "mrr": 1.0, "ndcg_at_10": 1.0, "first_gold_rank": 1,
        "label": "상한도달", "elapsed_sec": 0.5,
    }
    base.update(kw)
    return base


def test_aggregate_averages_rank_metrics():
    rows = [_retrieval_row(mrr=1.0), _retrieval_row(mrr=0.0)]
    m = _aggregate(rows, "retrieval")
    assert m["mrr"] == 0.5
    assert m["context_precision"] == 0.2


def test_first_gold_rank_median_excludes_not_found():
    """못 찾은 문항(0)을 평균에 넣으면 '1등에 가깝다'는 거짓 신호가 된다."""
    rows = [
        _retrieval_row(first_gold_rank=1),
        _retrieval_row(first_gold_rank=9),
        _retrieval_row(first_gold_rank=0, evidence_hit=0, label="검색실패"),
    ]
    m = _aggregate(rows, "retrieval")
    assert m["first_gold_rank_found_n"] == 2
    assert m["first_gold_rank_median"] in (1, 9)   # 짝수 표본이라 상위값
    assert m["first_gold_rank_hist"]["미발견"] == 1
    assert m["first_gold_rank_hist"]["1"] == 1
    assert m["first_gold_rank_hist"]["4-10"] == 1


def test_by_doc_group_splits_periodic_and_major():
    """전체 평균만 보면 두 병목이 상쇄돼 어느 쪽을 고칠지 알 수 없다."""
    rows = [
        _retrieval_row(doc_group="periodic", context_recall=1.0, context_ap=0.2, first_gold_rank=9),
        _retrieval_row(doc_group="periodic", context_recall=1.0, context_ap=0.2, first_gold_rank=9),
        _retrieval_row(doc_group="major", context_recall=0.0, context_ap=0.0,
                       first_gold_rank=0, evidence_hit=0, label="검색실패"),
    ]
    m = _aggregate(rows, "retrieval")
    g = m["by_doc_group"]
    assert set(g) == {"periodic", "major"}
    assert g["periodic"]["n"] == 2
    assert g["periodic"]["context_recall"] == 1.0      # 회수는 됨
    assert g["periodic"]["context_ap"] == 0.2          # 그런데 순위가 나쁨 = 순위 문제
    assert g["major"]["context_recall"] == 0.0         # 아예 못 찾음 = 발견 문제
    assert "first_gold_rank_median" not in g["major"]  # 찾은 게 없으면 생략


def test_aggregate_skips_missing_keys_for_full_mode():
    """full 모드 행에는 순위 지표가 없다 — KeyError 로 죽으면 안 된다."""
    rows = [{
        "id": "x", "query": "q", "company": "c", "doc_group": "major",
        "gold": "1", "n_gold": 1, "answer": "a",
        "answer_hit": 1, "evidence_hit": 1, "refusal": 0,
        "numbers_grounded": 1, "has_citation": 1, "validation_passed": 1,
        "label": "정답", "elapsed_sec": 1.0,
    }]
    m = _aggregate(rows, "full")
    assert m["answer_hit"] == 1.0
    assert "mrr" not in m
    assert "first_gold_rank_hist" not in m
    assert m["by_doc_group"]["major"]["n"] == 1


def test_write_renders_new_sections(tmp_path):
    rows = [_retrieval_row(), _retrieval_row(doc_group="major", first_gold_rank=0,
                                             evidence_hit=0, label="검색실패")]
    m = _aggregate(rows, "retrieval")
    _write(tmp_path, {"mode": "retrieval"}, m, rows)
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## 첫 정답 문서 등수" in text
    assert "## 공시유형별" in text
    assert "context_ap" in text


# --------------------------------------------------------------------------- v2 경로 (2026-08-30)

def _v2_row(**kw):
    base = {
        "id": "x", "query": "q", "company": "c", "doc_group": "exchange",
        "gold": "1", "n_gold": 1, "answer": "a",
        "answer_hit": 1, "evidence_hit": 1, "refusal": 0, "label": "정답",
        "stopped_at": "answered", "hcx_calls": 1, "retries": 0,
        "answer_mode": "closed", "task": "lookup", "error": "", "elapsed_sec": 1.0,
    }
    base.update(kw)
    return base


def test_v2_rows_aggregate_with_the_same_columns_as_v1():
    """v1 과 직접 비교하려면 지표 컬럼이 같아야 한다."""
    m = _aggregate([_v2_row(), _v2_row(answer_hit=0, label="답변실패")], "full")
    assert m["answer_hit"] == 0.5
    assert m["evidence_hit"] == 1.0
    assert m["by_doc_group"]["exchange"]["n"] == 2


def test_v2_aggregate_tolerates_missing_v1_only_columns():
    """v2 행에는 numbers_grounded 같은 v1 전용 컬럼이 없다 — 죽으면 안 된다."""
    m = _aggregate([_v2_row()], "full")
    assert "numbers_grounded" not in m
    assert m["answer_hit"] == 1.0


def test_v2_write_renders(tmp_path):
    rows = [_v2_row(), _v2_row(stopped_at="abstention_gate", hcx_calls=0,
                               answer_hit=0, refusal=1, label="거부(검색실패)")]
    _write(tmp_path, {"mode": "full", "pipeline": "v2"}, _aggregate(rows, "full"), rows)
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## 공시유형별" in text


# --------------------------------------------------------------------------- 나열 판별 (2026-08-30)

from score_answers import _answer_numbers, _gold_position  # noqa: E402


def test_answer_numbers_ignores_the_citation_line():
    """근거 줄의 report_id 안 숫자가 값으로 잡히면 개수가 엉망이 된다."""
    a = "순자산액은 224,787,773,988,054원입니다.\n근거: report_id(major_20241118000328)"
    assert _answer_numbers(a) == ["224787773988054"]


def test_answer_numbers_ignores_years_and_list_markers():
    a = "1. 2023년 공시: 23,926,907,424,811원  2. 2024년 공시: 23,823,475,847,161원"
    assert _answer_numbers(a) == ["23926907424811", "23823475847161"]


def test_answer_numbers_deduplicates():
    a = "236,396,657,259,591 / 236,396,657,259,591 / 224,787,773,988,054"
    assert _answer_numbers(a) == ["236396657259591", "224787773988054"]


def test_gold_position_finds_where_the_answer_buried_it():
    """v2 스모크 S001 실제 형태 — 정답이 여섯 번째였다."""
    a = ("순자산액: 254,330,082,981,146 원  236,396,657,259,591 원  "
         "111,111,111,111,111 원  222,222,222,222,222 원  333,333,333,333,333 원  "
         "224,787,773,988,054 원")
    assert _gold_position(a, ["224,787,773,988,054"]) == 6


def test_gold_position_is_zero_when_absent():
    assert _gold_position("확인되지 않습니다.", ["1,000"]) == 0


def test_exact_hit_requires_the_gold_first_and_few_numbers():
    """정의를 테스트로 고정한다 — 값 하나 물었는데 다섯 개 나열하면 감점 대상."""
    clean = "44,798,302,492,788 원\n근거: report_id(major_20240712000097)"
    listed = "1. 23,926,907,424,811원  2. 23,823,475,847,161원  3. 24,000,000,000,000원"
    assert _gold_position(clean, ["44,798,302,492,788"]) == 1
    assert len(_answer_numbers(clean)) == 1
    assert _gold_position(listed, ["23,823,475,847,161"]) == 2   # 첫 값이 아니다


def test_answer_hit_definition_is_unchanged():
    """기존 수치와 비교 가능해야 하므로 answer_hit 의 정의는 건드리지 않는다."""
    listed = "254,330,082,981,146 / 224,787,773,988,054"
    assert _answer_hit(listed, ["224,787,773,988,054"])       # 여전히 정답으로 센다


# --------------------------------------------------------------------------- 정답 형식 3종 (2026-08-30)
#
# suite_v1 22문항 중 16문항이 '사람이 읽는 요약' 형태의 정답을 갖고 있어,
# 문자열 대조로는 맞는 답이 전부 오답 처리됐다(실측 27.3% -> 68.2%).

from score_answers import grade_answer, _verdict  # noqa: E402


def test_value_gold_uses_the_existing_rule():
    hit, kind = grade_answer("순자산액은 224,787,773,988,054원입니다.",
                             "224,787,773,988,054", ["224,787,773,988,054"])
    assert (hit, kind) == (1, "value")


# --- B형: 비교 -------------------------------------------------------------

_S007_GOLD = "삼성전자 (삼성전자 22,764,764,160,000 vs 한미반도체 149,919,000,000)"


def test_compare_gold_accepts_a_correct_answer():
    a = ("삼성전자의 최대 계약금액은 22,764,764,160,000원이고, 한미반도체는 "
         "149,919,000,000원입니다. 따라서 더 큰 기업은 삼성전자입니다.")
    assert grade_answer(a, _S007_GOLD, [_S007_GOLD]) == (1, "compare")


def test_compare_requires_both_numbers():
    """한쪽 값만 제시하면 비교가 성립하지 않는다."""
    a = "삼성전자의 최대 계약금액은 22,764,764,160,000원입니다. 삼성전자가 더 큽니다."
    assert grade_answer(a, _S007_GOLD, [_S007_GOLD])[0] == 0


def test_compare_rejects_a_wrong_number():
    """S011/S014 실제 실패 — 최댓값이 아닌 값을 가져왔다."""
    a = ("삼성전자는 22,764,764,160,000원, 한미반도체는 999,999,999원입니다. "
         "따라서 삼성전자가 더 큽니다.")
    assert grade_answer(a, _S007_GOLD, [_S007_GOLD])[0] == 0


def test_compare_rejects_a_wrong_conclusion():
    """수치가 다 맞아도 결론이 뒤집히면 오답이다."""
    a = ("삼성전자는 22,764,764,160,000원, 한미반도체는 149,919,000,000원입니다. "
         "따라서 한미반도체가 더 큽니다.")
    assert grade_answer(a, _S007_GOLD, [_S007_GOLD])[0] == 0


def test_compare_accepts_when_the_conclusion_sentence_is_unrecognizable():
    """결론 표현이 다양해 탐지가 실패할 수 있다 — 그걸로 오답 처리하면 과하다."""
    a = "삼성전자 22,764,764,160,000원 / 한미반도체 149,919,000,000원"
    assert grade_answer(a, _S007_GOLD, [_S007_GOLD])[0] == 1


# --- C형: 예/아니오 --------------------------------------------------------

def test_yes_gold_accepts_an_affirmative_answer():
    a = "네, 해지된 계약이 존재합니다. 해지된 계약은 Ford Motor Company와의…"
    assert grade_answer(a, "예 (해지 공시 존재)", []) == (1, "yes_no")


def test_no_gold_accepts_a_negative_answer():
    """S020 실제 답변 — '해당 사항 없음'."""
    assert grade_answer("해당 사항 없음", "아니오 (해지 공시 없음)", [])[0] == 1


def test_yes_gold_rejects_a_negative_answer():
    assert grade_answer("존재하지 않습니다.", "예 (해지 공시 존재)", [])[0] == 0


def test_refusal_is_not_credited_as_no():
    """S021/S022 — '확인할 수 없습니다'는 '없습니다'와 다른 답이다.

    채점 기준이 "없다고 **정확히** 답하는가" 이고, 별도 항목으로 "없는 사실을
    지어내지 않는가"를 본다. 확인 불가와 부재는 구분해야 한다.
    """
    a = "제공된 근거로는 확인할 수 없습니다."
    assert _verdict(a) == "거부"
    assert grade_answer(a, "아니오 (해지 공시 없음)", [])[0] == 0


def test_verdict_reads_the_head_not_the_details():
    """앞부분에서 판정하고 뒤에 근거를 다는 게 프롬프트 지시다."""
    a = "네, 존재합니다. 다만 일부 항목은 확인할 수 없습니다."
    assert _verdict(a) == "예"


def test_negative_is_checked_before_positive():
    """'존재하지 않습니다'가 '존재합니다'에 걸리면 안 된다."""
    assert _verdict("해지된 계약은 존재하지 않습니다.") == "아니오"
