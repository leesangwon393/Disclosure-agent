"""채점기 규칙 테스트 — 채점이 틀리면 이후 모든 판단이 틀린다.

특히 표기 차이(콤마·단위)를 오답 처리하면 개선이 있어도 안 보이고,
반대로 너무 느슨하면 없는 개선이 보인다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from score_answers import (  # noqa: E402
    _answer_hit, _gold_answers, _is_refusal, _label, _norm,
    evidence_hit, korean_amounts, unit_stated,
)


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


def test_a_number_without_its_unit_is_not_a_hit():
    """`7,661,584원` 은 `7,661,584,000,000` 이 아니다 — 백만 배 틀렸다.

    2026-09-01: 이 자리에 있던 테스트는 `7,661,584백만원` 도 오답이라고
    못 박고 있었다. 그런데 7,661,584 x 1,000,000 = 7,661,584,000,000 —
    **같은 값이다**. 맞는 답을 오답으로 세던 규칙이라 고쳤다.
    """
    assert not _answer_hit("7,661,584원입니다", ["7,661,584,000,000"])
    assert not _answer_hit("7,661,584입니다", ["7,661,584,000,000"])
    assert _answer_hit("7,661,584백만원입니다", ["7,661,584,000,000"])


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


def test_compare_without_a_conclusion_is_not_counted_as_correct():
    """결론을 못 찾으면 예전엔 **정답 처리**했다. 그게 두 종류를 통과시켰다.

    실측(2026-08-31): 승자를 아예 안 밝힌 답변과 승자를 **틀리게** 쓴 답변이
    모두 정답으로 잡혔다. `_WINNER_PAT` 가 마침표 없는 문장에서 두 회사 이름을
    한꺼번에 잡아 `len(hits)==1` 이 깨지고 `stated=None` 이 되기 때문이다.
    비교 문항은 suite_v1+v2 합쳐 68개라 그대로 두면 점수가 부풀려진다.

    수치는 다 맞았으므로 '틀렸다'고 단정하지도 않는다 — 별도 형식으로
    분리해서 사람이 보게 한다.
    """
    a = "삼성전자 22,764,764,160,000원 / 한미반도체 149,919,000,000원"
    hit, kind = grade_answer(a, _S007_GOLD, [_S007_GOLD])
    assert (hit, kind) == (0, "compare_no_verdict")


def test_compare_with_a_wrong_winner_in_a_period_less_sentence():
    """마침표가 없어도 틀린 승자를 통과시키면 안 된다."""
    a = ("삼성전자 22,764,764,160,000원 vs 한미반도체 149,919,000,000원 "
         "-> 한미반도체가 더 큽니다")
    assert grade_answer(a, _S007_GOLD, [_S007_GOLD])[0] == 0


def test_compare_winner_name_with_a_space_is_matched():
    """`split()[0]` 로 자르면 `JYP Ent` 가 `JYP` 가 돼 완벽한 답도 0점이 됐다."""
    gold = "JYP Ent (JYP Ent 100,000,000 vs 파마리서치 50,000,000)"
    a = ("JYP Ent 는 100,000,000원, 파마리서치는 50,000,000원입니다. "
         "따라서 JYP Ent 가 더 큽니다.")
    assert grade_answer(a, gold, [gold]) == (1, "compare")


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


# --------------------------------------------------------------- 서술형(D형) 채점
#
# 2026-08-30 추가. 그전까지 서술형 16문항(suite_v1 의 42%)은 정답 문장이 없어
# 자동으로 0점 처리됐고, 그 0점이 전체 평균에 섞여 실제보다 낮은 숫자가 나왔다.

from score_answers import (  # noqa: E402
    _GRADEABLE_KEYS, _aggregate, grade_answer, required_report,
)


def test_required_all_needs_every_item():
    req = ["684,000", "확정발행가"]
    assert grade_answer("684,000원 확정발행가로 정정", "", [], required=req) == (1, "required")
    assert grade_answer("684,000원으로 정정", "", [], required=req) == (0, "required")


def test_required_coverage_gives_partial_credit():
    rep = required_report("금액은 1,000원", ["1,000", "가처분", "해제"])
    assert rep["n"] == 3 and rep["matched"] == 1
    assert rep["coverage"] == round(1 / 3, 4)
    assert rep["missing"] == ["가처분", "해제"]


def test_required_report_is_none_when_nothing_required():
    """required_all 이 없는 문항은 '0점'이 아니라 '해당 없음'이어야 한다."""
    assert required_report("아무 답", [])["coverage"] is None


def test_number_notation_differences_are_absorbed():
    """정답지 9.90, 답변 9.9% — 같은 값이다."""
    assert required_report("자기자본대비 9.9%", ["9.90"])["matched"] == 1
    assert required_report("자기자본대비 9%", ["9.0"])["matched"] == 1
    assert required_report("금액 5296200000000원", ["5,296,200,000,000"])["matched"] == 1


def test_number_match_is_exact_not_substring():
    """19.90 안에 9.90 이 들어 있다고 맞다고 하면 안 된다."""
    assert required_report("비율은 19.90% 입니다", ["9.90"])["matched"] == 0
    assert required_report("금액 5,296,200,000,001", ["5,296,200,000,000"])["matched"] == 0


def test_date_notation_differences_are_absorbed():
    """정답지 2024-04-24, 답변 '2024년 4월 24일' — 같은 날이다."""
    for ans in ("이사회결의일 2024년 4월 24일", "기간 2024.04.24 ~", "접수 20240424800596"):
        assert required_report(ans, ["2024-04-24"])["matched"] == 1, ans
    assert required_report("2024년 4월 25일", ["2024-04-24"])["matched"] == 0


def test_korean_date_token_matches_iso_answer():
    assert required_report("계약기간 2025-02-05 ~ 2026-12-31", ["2026년 12월 31일"])["matched"] == 1


def test_latin_tokens_are_case_insensitive():
    assert required_report("상대방은 vrnj co., ltd.", ["VRNJ"])["matched"] == 1
    assert required_report("공급물량 19 GWh", ["19GWh"])["matched"] == 1


def test_required_takes_precedence_over_empty_gold():
    """정답 문장이 비어 있어도 required_all 이 있으면 채점된다."""
    hit, kind = grade_answer("청주 M15X 건설", "", [], required=["청주 M15X"])
    assert (hit, kind) == (1, "required")


# ------------------------------------------------------- 채점 불가 문항의 분모 처리

def test_ungradeable_rows_get_their_own_label():
    assert _label(False, True, False, gradeable=False) == "채점불가"
    assert _label(False, True, False, gradeable=True) == "답변실패"


def _row(graded, gradeable, label="정답"):
    return {"label": label, "elapsed_sec": 1.0,
            "graded_hit": graded, "gradeable": gradeable}


def test_graded_hit_excludes_ungradeable_rows():
    """정답이 없는 문항을 분모에 넣으면 실제보다 낮게 나온다.

    실측(2026-08-30): 38문항 중 16문항에 정답이 없어 52.6% 로 찍혔고,
    채점 가능한 22문항만 보면 90.9% 였다.
    """
    rows = [_row(1, 1), _row(1, 1), _row(0, 0), _row(0, 0)]
    m = _aggregate(rows, "full")
    assert m["graded_hit"] == 1.0          # 2/2, 채점 불가 2건은 빠진다
    assert m["graded_hit_n"] == 2
    assert m["ungradeable_n"] == 2


def test_graded_hit_is_none_when_nothing_is_gradeable():
    m = _aggregate([_row(0, 0)], "full")
    assert m["graded_hit"] is None
    assert m["graded_hit_n"] == 0


def test_graded_hit_left_out_of_plain_rate_keys():
    """일반 평균 목록에 남아 있으면 분모가 두 번 계산된다."""
    from score_answers import _RATE_KEYS
    assert "graded_hit" not in _RATE_KEYS
    assert "graded_hit" in _GRADEABLE_KEYS


# ------------------------------------------------- 정답 후보가 여럿인 경우(D-2형)
#
# 질문이 대상을 특정하지 않을 때가 있다. 실측(S023~S026, 2026-08-30):
#   "현대건설의 단일판매ㆍ공급계약체결 공시가 정정된 내역이 있는가?"
#   -> 해당 유형 정정 공시 70건, 값이 바뀐 체인 10개. 정답지는 그중 1개만 인정.
#   모델은 다른 체인을 설명했고 오답이 됐다 — 틀린 게 아니라 다른 걸 고른 것이다.

from score_answers import required_any_report  # noqa: E402


def test_any_one_candidate_is_enough():
    groups = [["1,000,000", "계약금액"], ["2,000,000", "매출액대비"]]
    assert grade_answer("계약금액이 2,000,000원", "", [], any_groups=groups)[0] == 0
    hit, kind = grade_answer("매출액대비 기준 2,000,000", "", [], any_groups=groups)
    assert (hit, kind) == (1, "required_any")


def test_partial_credit_uses_the_best_candidate():
    """제일 많이 맞은 후보로 통계를 낸다 — 0 으로 뭉개면 개선이 안 보인다."""
    rep = required_any_report("계약금액은 1,000,000원", [["1,000,000", "계약금액", "없는값"],
                                                    ["9,999", "다른항목"]])
    assert rep["matched"] == 2 and rep["n"] == 3
    assert rep["n_groups"] == 2 and rep["n_full"] == 0


def test_full_match_count_is_reported():
    rep = required_any_report("계약금액 1,000,000원 / 매출액대비 5.0%",
                              [["1,000,000"], ["5.0"], ["없는값"]])
    assert rep["n_full"] == 2


def test_empty_candidate_list_is_not_applicable():
    rep = required_any_report("아무 답", [])
    assert rep["coverage"] is None and rep["n"] == 0


def test_blank_entries_inside_a_candidate_are_ignored():
    assert required_any_report("계약금액 1,000,000원", [["1,000,000", "", "  "]])["matched"] == 1


def test_required_any_takes_precedence_over_required_all():
    hit, kind = grade_answer("값은 500 이다", "", [], required=["999"],
                             any_groups=[["500"]])
    assert (hit, kind) == (1, "required_any")


def test_real_suite_correction_rows_have_multiple_candidates():
    """정정 문항이 후보 하나만 갖고 있으면 예전 문제로 되돌아간 것이다."""
    import json
    p = Path(__file__).resolve().parents[1] / "eval" / "suite_v1.jsonl"
    if not p.exists():
        pytest.skip("suite_v1 없음")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    corr = [r for r in rows if r.get("required_any")]
    assert corr, "정정 문항에 required_any 가 없다"
    for r in corr:
        assert r.get("chain_count") is not None
        assert r.get("corrected_doc_count", 0) > 0


# ------------------------------------------- 근거 표기 처리 (2026-08-31 두 번 고침)
#
# 1차: 근거줄의 `report_id(exchange_20240424800596)` 안 숫자가 날짜 정답과
#      우연히 맞아 **거부 답변이 만점**을 받았다. -> "근거:" 이후를 잘랐다.
# 2차: 모델은 항목마다 근거를 다는 형식도 쓴다. 첫 "근거:" 에서 자르니
#      **뒤에 오는 값과 결론이 통째로 사라져** 맞는 답이 오답이 됐다
#      (v2_off8 에서 S013·S037 실제 발생). -> 자르지 않고 문서 ID 만 지운다.

from score_answers import _body_only  # noqa: E402


def test_document_ids_do_not_satisfy_a_date_requirement():
    """거부만 한 답변이 근거줄 숫자로 날짜 항목을 채우면 안 된다."""
    bad = "확인되지 않습니다. 근거: report_id(exchange_20240424800596)"
    assert required_report(bad, ["2024-04-24"])["matched"] == 0


def test_inline_citations_do_not_truncate_the_answer():
    """항목마다 근거를 다는 형식에서 뒷부분이 잘리면 안 된다."""
    a = ("- 현대글로비스: 3,365,500,000,000원\n  - 근거: [exchange_20241231800103]\n"
         "- HMM: 1,282,363,356,560원\n  - 근거: [exchange_20231027800429]\n"
         "따라서 최대 계약금액이 더 큰 기업은 현대글로비스입니다.")
    gold = "현대글로비스 (현대글로비스 3,365,500,000,000 vs HMM 1,282,363,356,560)"
    assert grade_answer(a, gold, [gold]) == (1, "compare")


def test_inline_citations_keep_later_required_items():
    a = ("- 제목: 양해각서(MOU) 체결\n  - 근거: [exchange_20250205800058]\n"
         "- 주요내용: VRNJ Co.,Ltd. 와 협동 로봇 300대, 계약기간 2026년 12월 31일까지")
    rep = required_report(a, ["VRNJ", "300", "2026년 12월 31일", "MOU"])
    assert rep["matched"] == 4, rep["missing"]


def test_body_only_keeps_content_and_drops_ids():
    text = "값은 1,000원이다. 근거: report_id(major_20240101000001) chunk_id: x::main::C1"
    body = _body_only(text)
    assert "1,000원" in body and "근거" in body
    assert "20240101000001" not in body and "C1" not in body


# --------------------------------------------------------------------------- 지연 분해 (2026-08-31)

def test_csv_columns_are_the_union_of_all_rows(tmp_path):
    """첫 행에 없는 컬럼이 뒷 행에 있어도 저장이 죽으면 안 된다.

    조기 종료 문항(게이트)은 ms_rerank 가 없다. 그런 문항이 1번으로 오면
    첫 행 기준으로 헤더를 잡던 예전 코드는 DictWriter ValueError 로 죽어
    **측정 결과 전체가 저장되지 않는다.** 1시간 30분짜리 실행이 날아간다.
    """
    rows = [_v2_row(stopped_at="scope_gate", ms_total=12.0),
            _v2_row(ms_total=9000.0, ms_rerank=4200.0, ms_bm25=300.0)]
    _write(tmp_path, {"mode": "full", "pipeline": "v2"}, _aggregate(rows, "full"), rows)
    header = (tmp_path / "results.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert "ms_rerank" in header and "ms_total" in header


def test_latency_breakdown_lands_in_metrics():
    m = _aggregate([_v2_row(ms_total=1000.0, ms_rerank=400.0),
                    _v2_row(ms_total=3000.0, ms_rerank=1200.0)], "full")
    assert m["latency_breakdown"]["rerank"]["median_ms"] == 1200.0
    assert m["latency_breakdown"]["total"]["n"] == 2


# --------------------------------------------------------------- 정정 체인 (A-9a)
#
# 정답지에 원본 문서 ID 가 박혀 있는데 검색이 최신 정정본을 가져오면 예전
# 채점기는 '검색실패'로 찍었다. 실측 43건 중 14건이 그것이었다.

GROUPS = {"exchange_1": "exchange_1", "exchange_1r": "exchange_1",
          "exchange_2": "exchange_2"}


def test_the_same_document_is_a_hit():
    assert evidence_hit({"exchange_1"}, {"exchange_1"}, GROUPS)


def test_another_version_of_the_same_correction_chain_is_a_hit():
    assert evidence_hit({"exchange_1"}, {"exchange_1r"}, GROUPS)


def test_a_different_chain_is_not_a_hit():
    assert not evidence_hit({"exchange_1"}, {"exchange_2"}, GROUPS)


def test_documents_outside_any_chain_are_each_their_own_group():
    """체인에 없는 문서 둘을 같은 그룹(None)으로 묶으면 안 된다."""
    assert not evidence_hit({"zzz_a"}, {"zzz_b"}, GROUPS)


def test_no_chain_map_falls_back_to_document_ids():
    assert evidence_hit({"exchange_1"}, {"exchange_1"}, {})
    assert not evidence_hit({"exchange_1"}, {"exchange_1r"}, {})


def test_empty_gold_is_never_a_hit():
    assert not evidence_hit(set(), {"exchange_1"}, GROUPS)


# ------------------------------------------------------------- 단위 환산 (A-9b)

def test_percent_notation_equals_the_decimal_form():
    """정답지 `0.0430`, 답변 `4.30%` — 같은 값이다(G0146 실측)."""
    assert _answer_hit("지분율은 4.30% 입니다", ["0.0430"])
    assert _answer_hit("지분율은 0.0430 입니다", ["4.30%"])


def test_a_hundredfold_number_without_a_percent_sign_is_not_a_hit():
    """`%` 가 없으면 100배를 인정하지 않는다 — `4.30` 과 `430` 은 다른 값이다."""
    assert not _answer_hit("430 건입니다", ["4.30"])


@pytest.mark.parametrize("text,won", [
    ("3조 1,128억원", 3_112_800_000_000),
    ("3천억원", 300_000_000_000),
    ("1조원", 1_000_000_000_000),
    ("255,698,325천원", 255_698_325_000),
    ("5만 5,000원", 55_000),
])
def test_korean_amounts_are_read_in_won(text, won):
    assert korean_amounts(text) == {won}


def test_plain_numbers_are_not_read_as_korean_amounts():
    assert korean_amounts("224,787,773,988,054원") == set()
    assert korean_amounts("2024년 3월 15일") == set()


def test_korean_amount_matches_the_gold_in_table_units():
    """정답지 `3,112,850`(백만원 표) vs 답변 `3조 1,128억원`."""
    assert _answer_hit("영업비용은 3조 1,128억원입니다", ["3,112,850"])


def test_a_thousandfold_korean_amount_is_not_a_hit():
    """`255,698,325천원` 은 백만원 표의 `3,112,850` 이 아니다."""
    assert not _answer_hit("영업비용은 255,698,325천원입니다", ["3,112,850"])


# ------------------------------------------------------------- 단위 표기 (A-9c)
#
# 대회는 사람이 채점한다. 숫자만 적고 단위를 빼면 감점이다. 지금까지는
# 숫자만 비교해서 이게 지표에 전혀 안 보였다.

def test_a_number_without_a_unit_is_flagged():
    assert unit_stated("3,112,850") == 0


def test_a_number_with_a_unit_passes():
    assert unit_stated("3,112,850백만원입니다") == 1
    assert unit_stated("지분율은 4.30% 입니다") == 1


def test_an_answer_with_no_number_is_not_counted():
    assert unit_stated("제공된 근거로는 확인할 수 없습니다") is None


# --------------------------------------------- 2026-09-01 교차 검수에서 잡은 것
#
# 아래는 전부 "느슨해서 틀린 답을 정답으로 세던" 규칙이다. 채점기가 후하면
# 개선이 없는데도 있는 것처럼 보인다.

from score_answers import _strip_ordinals, _renderings  # noqa: E402
from decimal import Decimal  # noqa: E402


def test_a_short_gold_does_not_borrow_a_unit_it_was_not_given():
    """정답지의 `4.0` 에 답변 `3,999,455 백만원` 이 걸리면 안 된다.

    네 자리 미만 정답에는 단위 환산을 적용하지 않는다. 다만 답변에 적힌
    숫자 자체가 정답과 같으면(`150` <-> `150억원`) 그건 예전부터 정답으로
    센다 — 정답지에 단위가 안 적혀 있어 구분할 방법이 없다. 한 자리 정답이
    남의 숫자에 걸리는 경우는 이 규칙의 알려진 한계다.
    """
    assert not _answer_hit("금액은 3,999,455 백만원입니다", ["4.0"])
    assert _answer_hit("영업이익은 150억원입니다.", ["150"])


def test_a_number_inside_an_article_reference_is_not_a_value():
    assert not _answer_hit("정관 제3조 제1항에 따라 배당합니다.", ["3"])
    assert not _answer_hit("제55기 정기주주총회입니다.", ["55"])


def test_a_plain_count_still_matches():
    assert _answer_hit("총 3건입니다.", ["3"])


def test_ordinals_are_stripped_but_amounts_survive():
    assert "3" not in _strip_ordinals("제3조 제1항")
    assert "3,112,850" in _strip_ordinals("금액은 3,112,850백만원")


def test_a_near_miss_amount_is_not_a_hit():
    """예전엔 상대오차 5e-4 창으로 견줘 서로 다른 두 값이 같아졌다."""
    assert not _answer_hit("금액은 3,999,455 백만원입니다", ["4.0"])
    assert not _answer_hit("금액은 255,698,325천 원입니다", ["25,575,912"])


def test_renderings_are_the_korean_readings_of_one_amount():
    got = _renderings(Decimal("3112850000000"))
    assert Decimal("3112850000000") in got      # 그대로
    assert Decimal("3112800000000") in got      # "3조 1,128억원"
    assert Decimal("3000000000000") in got      # "3조원"


@pytest.mark.parametrize("text", [
    "정관 제3조 제1항에 따라",
    "발행주식수는 10만 주입니다",
    "임직원은 약 3만 명입니다",
    "제55기 정기주주총회",
])
def test_counting_words_and_ordinals_are_not_amounts(text):
    assert korean_amounts(text) == set()


def test_a_sentence_boundary_does_not_join_two_amounts():
    assert korean_amounts("총자산은 5조. 3천억은 부채다") == {
        5_000_000_000_000, 300_000_000_000}


def test_an_enumeration_keeps_each_amount_separate():
    assert korean_amounts("매출은 1,234억, 5,678억, 9,012억원이다") == {
        123_400_000_000, 567_800_000_000, 901_200_000_000}


def test_a_year_after_an_amount_is_not_part_of_it():
    assert korean_amounts("2024년 1,000억 2025년 2,000억 증가") == {
        100_000_000_000, 200_000_000_000}


# ------------------------------------------------------- 단위 표기 지표 (재정의)

def test_a_unit_must_be_attached_to_the_value():
    """본문 아무 데나 있는 글자를 세면 지표가 아무것도 못 거른다 —
    실측 272건 중 266건(97.8%)이 통과했다."""
    assert unit_stated("2024년 매출은 3,112,850 입니다.") == 0
    assert unit_stated("주주총회에서 1,234,567 를 의결했습니다.") == 0
    assert unit_stated("천안 공장의 생산능력은 1,234,567 입니다.") == 0


def test_a_date_is_skipped_before_judging_the_first_real_value():
    assert unit_stated("2024년 매출은 3조 1,128억원입니다.") == 1
    assert unit_stated("제55기 정기주주총회입니다.") is None
