"""멀티기업 귀속 채점기 테스트."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from score_attribution import cited_ids, grade_one  # noqa: E402

OWNER = {
    "periodic_20250311000001": "삼성전자",
    "periodic_20250311000002": "삼성SDI",
    "periodic_20250311000003": "SK하이닉스",
}


def test_doc_ids_are_pulled_out_of_the_citation_line():
    answer = ("삼성전자 300조, 삼성SDI 20조입니다.\n\n"
              "근거: report_id(periodic_20250311000001), "
              "report_id(periodic_20250311000002)")
    assert cited_ids(answer) == ["periodic_20250311000001", "periodic_20250311000002"]


def test_same_id_is_not_counted_twice():
    answer = "근거: report_id(periodic_20250311000001), report_id(periodic_20250311000001)"
    assert cited_ids(answer) == ["periodic_20250311000001"]


def test_all_companies_cited_from_their_own_documents_is_clean():
    answer = ("근거: report_id(periodic_20250311000001), "
              "report_id(periodic_20250311000002)")
    r = grade_one(answer, ["삼성전자", "삼성SDI"], OWNER)
    assert r["coverage"] == 1.0 and r["clean"] == 1 and r["n_foreign"] == 0


def test_the_reported_failure_is_caught():
    """삼성전자 값을 삼성SDI 문서에서만 가져오면 잡혀야 한다.

    숫자만 보는 채점기는 이걸 통과시킨다 — 그 표에 적힌 숫자 그대로이기
    때문이다. 출처를 봐야 잡힌다.
    """
    answer = "삼성전자 300조, 삼성SDI 20조.\n근거: report_id(periodic_20250311000002)"
    r = grade_one(answer, ["삼성전자", "삼성SDI"], OWNER)
    assert r["clean"] == 0
    assert r["missing"] == "삼성전자"
    assert r["coverage"] == 0.5


def test_documents_of_a_company_nobody_asked_about_are_flagged():
    answer = "근거: report_id(periodic_20250311000001), report_id(periodic_20250311000003)"
    r = grade_one(answer, ["삼성전자"], OWNER)
    assert r["foreign"] == "SK하이닉스" and r["clean"] == 0


def test_a_made_up_document_id_is_separated_from_foreign_companies():
    """지어낸 ID 와 '남의 회사 문서'는 원인이 다르다 — 섞으면 진단이 흐려진다."""
    answer = "근거: report_id(periodic_19990101000009)"
    r = grade_one(answer, ["삼성전자"], OWNER)
    assert r["unknown_ids"] == "periodic_19990101000009"
    assert r["n_foreign"] == 0 and r["clean"] == 0


def test_no_citation_at_all_is_zero_coverage():
    r = grade_one("확인할 수 없습니다.", ["삼성전자", "삼성SDI"], OWNER)
    assert r["coverage"] == 0.0 and r["n_cited_docs"] == 0


def test_pipeline_citations_win_over_the_answer_text():
    """모델이 인용 형식을 안 지켜도 채점은 정확해야 한다.

    첫 측정에서 8문항이 '근거 0건'으로 나왔는데 실제로는 제대로 답한
    문항이었다. 모델이 "근거: [EVIDENCE 1]" 이라고만 썼기 때문이다.
    채점기가 본문 형식에 매달리면 시스템이 아니라 채점기를 측정하게 된다.
    """
    answer = "삼성전자 300조, 삼성SDI 20조.\n근거: [EVIDENCE 1], [EVIDENCE 2]"
    assert cited_ids(answer) == []          # 본문만 보면 0건
    r = grade_one(answer, ["삼성전자", "삼성SDI"], OWNER,
                  pipeline_ids=["periodic_20250311000001", "periodic_20250311000002"])
    assert r["coverage"] == 1.0 and r["clean"] == 1
