"""ⓑ 공시유형별 표준항목 명세.

이 표가 틀리면 Sufficiency Check 가 엉뚱한 항목을 요구하고, 그 결과 답할 수
있는 질문이 '근거 부족'으로 거부된다. 그래서 (1) 이름 정규화 (2) 임계값 분류
(3) 질문→유형 매칭 세 갈래를 각각 본다.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_field_schema import build  # noqa: E402
from disclosure_rag.agent.field_schema import (  # noqa: E402
    FieldSchema,
    normalize_field_key,
    normalize_report_kind,
)

SCHEMA_PATH = ROOT / "config" / "field_schema.json"


# --------------------------------------------------------------------------- 정규화

def test_normalize_unifies_middle_dot():
    """DART 는 같은 공시를 'ㆍ' 와 '·' 로 섞어 쓴다."""
    assert (normalize_report_kind("단일판매ㆍ공급계약체결")
            == normalize_report_kind("단일판매·공급계약 체결")
            == "단일판매공급계약체결")


def test_normalize_strips_correction_prefix():
    """'[기재정정]주요사항보고서(...)' 도 같은 유형으로 봐야 한다."""
    assert (normalize_report_kind("[기재정정]주요사항보고서(자기주식 취득 결정)")
            == "주요사항보고서(자기주식취득결정)")


def test_normalize_matches_gold_question_wording():
    """정답셋 S001 의 표기와 facts 의 표기가 같은 키로 모여야 한다."""
    from_question = normalize_report_kind("주요사항보고서(자기주식취득결정)")
    from_facts = normalize_report_kind("주요사항보고서(자기주식 취득 결정)")
    assert from_question == from_facts


def test_normalize_field_key_is_space_insensitive():
    assert normalize_field_key("계약 금액") == normalize_field_key("계약금액") == "계약금액"


# --------------------------------------------------------------------------- 조회 API

def _toy() -> FieldSchema:
    return FieldSchema({"kinds": {
        "신규시설투자등": {
            "doc_group": "exchange", "n_docs": 43, "sufficient_data": True,
            "search_terms": ["신규시설투자등"],
            "required": ["투자금액", "투자목적"],
            "conditional": ["정정사유"],
            "optional": ["참석"],
            "ratios": {"투자금액": 1.0},
        },
        "주요사항보고서(자기주식취득결정)": {
            "doc_group": "major", "n_docs": 91, "sufficient_data": True,
            "search_terms": ["주요사항보고서(자기주식취득결정)", "자기주식취득결정"],
            "required": ["순자산액"], "conditional": [], "optional": [], "ratios": {},
        },
        "단일판매공급계약해지": {
            "doc_group": "exchange", "n_docs": 20, "sufficient_data": True,
            "search_terms": ["단일판매공급계약해지"],
            "required": ["계약금액"], "conditional": [], "optional": [], "ratios": {},
        },
    }})


def test_unknown_kind_returns_no_constraint():
    """모르는 유형에 required 를 걸면 안 된다 — 정답 가능한 질문이 거부된다."""
    s = _toy()
    assert s.required("사업보고서") == []
    assert s.required(None) == []
    assert s.classify("사업보고서", "매출액") == "unknown"


def test_empty_schema_is_harmless():
    s = FieldSchema.empty()
    assert s.kinds() == []
    assert s.required("신규시설투자등") == []
    assert s.match_kinds("아무 질문") == []


def test_classify_three_ways():
    s = _toy()
    assert s.classify("신규시설투자등", "투자금액") == "required"
    assert s.classify("신규시설투자등", "정정사유") == "conditional"
    assert s.classify("신규시설투자등", "참석") == "optional"
    assert s.classify("신규시설투자등", "없는항목") == "unknown"


def test_classify_normalizes_the_probe():
    assert _toy().classify("신규시설투자등", "투자 금액") == "required"


# --------------------------------------------------------------------------- 질문 매칭

def test_match_finds_kind_in_question():
    s = _toy()
    assert s.match_one("SK하이닉스의 2024년 신규시설투자등 공시를 정리해줘") == "신규시설투자등"


def test_match_prefers_longer_expression():
    """'단일판매공급계약해지' 가 '…체결' 과 헷갈리면 안 된다."""
    s = _toy()
    assert s.match_one("한전기술의 단일판매ㆍ공급계약 해지 내역") == "단일판매공급계약해지"


def test_match_does_not_double_count_inner_term():
    """'주요사항보고서(자기주식취득결정)' 이 걸렸으면 안쪽 '자기주식취득결정' 을
    따로 세지 않는다."""
    s = _toy()
    got = s.match_kinds("삼성전자의 주요사항보고서(자기주식취득결정)에 기재된 순자산액")
    assert got == ["주요사항보고서(자기주식취득결정)"]


def test_match_accepts_inner_term_alone():
    """질문이 괄호 안만 언급해도 잡아야 한다."""
    s = _toy()
    assert s.match_one("자기주식취득결정 공시 내용 알려줘") == "주요사항보고서(자기주식취득결정)"


def test_match_returns_empty_when_absent():
    assert _toy().match_kinds("삼성전자 매출액은 얼마인가?") == []


# --------------------------------------------------------------------------- 빌더 임계값

def _fake_facts(tmp_path: Path, rows: list[tuple]) -> Path:
    """rows = [(doc_id, doc_group, doc_subtype, report_name, key_norm), ...]"""
    p = tmp_path / "facts.sqlite"
    db = sqlite3.connect(p)
    db.execute("""CREATE TABLE facts (doc_id TEXT, doc_group TEXT, doc_subtype TEXT,
                  report_name TEXT, key_norm TEXT)""")
    db.executemany("INSERT INTO facts VALUES (?,?,?,?,?)", rows)
    db.commit()
    db.close()
    return p


def test_builder_splits_required_and_conditional(tmp_path):
    """10개 문서 중 9개에 있으면 required, 5개면 conditional, 1개면 optional."""
    rows = []
    for i in range(10):
        rows.append((f"d{i}", "exchange", "테스트공시", None, "항상"))
        if i < 9:
            rows.append((f"d{i}", "exchange", "테스트공시", None, "거의항상"))
        if i < 5:
            rows.append((f"d{i}", "exchange", "테스트공시", None, "가끔"))
        if i < 1:
            rows.append((f"d{i}", "exchange", "테스트공시", None, "드물게"))
    schema = build(_fake_facts(tmp_path, rows))["kinds"]["테스트공시"]
    assert schema["required"] == ["항상", "거의항상"]
    assert schema["conditional"] == ["가끔"]
    assert schema["optional"] == ["드물게"]


def test_builder_leaves_required_empty_for_small_samples(tmp_path):
    """문서 3건 전부에 있어도 '항상 있다'의 근거가 못 된다 — fail open."""
    rows = [(f"d{i}", "exchange", "소표본공시", None, "무언가") for i in range(3)]
    schema = build(_fake_facts(tmp_path, rows))["kinds"]["소표본공시"]
    assert schema["sufficient_data"] is False
    assert schema["required"] == []
    assert schema["conditional"] == ["무언가"]


def test_builder_drops_values_misparsed_as_keys(tmp_path):
    """facts 추출기가 값을 key 로 잡은 것(금액·주소·법인명)은 명세에 넣지 않는다."""
    rows = []
    for i in range(10):
        rows += [
            (f"d{i}", "major", None, "주요사항보고서(합병결정)", "합병비율"),
            (f"d{i}", "major", None, "주요사항보고서(합병결정)", "80,472원"),
            (f"d{i}", "major", None, "주요사항보고서(합병결정)", "경기도성남시분당구정자일로155두산타워26층"),
            (f"d{i}", "major", None, "주요사항보고서(합병결정)", "두산로보틱스주식회사"),
        ]
    out = build(_fake_facts(tmp_path, rows))
    schema = out["kinds"]["주요사항보고서(합병결정)"]
    assert schema["required"] == ["합병비율"]
    assert schema["dropped_artifact_facts"] == 30


def test_builder_demotes_table_structure_labels(tmp_path):
    """'항목'·'소계'·'참석' 은 100% 나와도 답변 필수 항목이 아니다."""
    rows = []
    for i in range(10):
        rows += [(f"d{i}", "major", None, "테스트", "순자산액"),
                 (f"d{i}", "major", None, "테스트", "항목"),
                 (f"d{i}", "major", None, "테스트", "소계")]
    schema = build(_fake_facts(tmp_path, rows))["kinds"]["테스트"]
    assert schema["required"] == ["순자산액"]
    assert set(schema["optional"]) == {"항목", "소계"}


def test_builder_records_the_gap_for_human_review(tmp_path):
    """임계값이 덩어리를 갈랐는지 골짜기에 떨어졌는지 사람이 볼 수 있어야 한다."""
    rows = []
    for i in range(10):
        rows.append((f"d{i}", "exchange", "테스트공시", None, "항상"))
        if i < 2:
            rows.append((f"d{i}", "exchange", "테스트공시", None, "드물게"))
    gap = build(_fake_facts(tmp_path, rows))["kinds"]["테스트공시"]["max_gap"]
    assert gap["above"] == 1.0 and gap["below"] == 0.2
    assert gap["gap"] == pytest.approx(0.8)


# --------------------------------------------------------------------------- 실제 산출물

@pytest.mark.skipif(not SCHEMA_PATH.exists(), reason="config/field_schema.json 미생성")
class TestBuiltSchema:
    @pytest.fixture(scope="class")
    @classmethod
    def schema(cls):
        return FieldSchema.load(SCHEMA_PATH)

    def test_competition_types_have_the_fields_questions_ask_about(self, schema):
        """정답셋이 실제로 묻는 항목이 required 에 들어 있어야 한다."""
        req = set(schema.required("신규시설투자등"))
        assert {"투자금액", "투자목적", "자기자본대비"} <= req      # S033~S036

        req = set(schema.required("단일판매공급계약체결"))
        assert {"계약금액", "계약상대", "체결계약명"} <= req        # S007~S014, S023~S024

        req = set(schema.required("주요사항보고서(자기주식취득결정)"))
        assert {"순자산액", "자본금의액", "자기주식취득금액한도"} <= req   # S001~S006

    def test_correction_fields_are_conditional_not_required(self, schema):
        """정정 항목을 required 로 두면 원본 공시 질문이 영원히 '부족'이 된다."""
        for kind in ("단일판매공급계약체결", "신규시설투자등"):
            assert "정정사유" not in schema.required(kind)
            assert "정정사유" in schema.conditional(kind)

    def test_no_parse_artifacts_survived(self, schema):
        import re
        bad = re.compile(r"\d{3,}|주식회사")
        for kind in schema.kinds():
            ks = schema.get(kind)
            for key in ks.required + ks.conditional + ks.optional:
                assert not bad.search(key), f"{kind}: {key}"

    def test_no_structure_labels_in_required(self, schema):
        for kind in schema.kinds():
            assert not ({"항목", "소계", "합계", "구분", "참석"}
                        & set(schema.required(kind)))

    def test_periodic_limitation_is_recorded(self, schema):
        """정기공시 미포함은 알고 써야 하는 한계다 — 문서에 남아 있어야 한다."""
        assert "periodic" in schema.coverage_note()
        assert "periodic" not in json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))["coverage"]["doc_groups"]

    def test_gold_questions_resolve_to_a_kind(self, schema):
        """정답셋 문항이 실제로 유형에 매칭되는지 — 안 되면 expected_fields 가 빈다."""
        cases = {
            "삼성전자의 [기재정정]주요사항보고서(자기주식취득결정)에 기재된 순자산액은 얼마인가?":
                "주요사항보고서(자기주식취득결정)",
            "SK하이닉스의 신규시설투자등에 기재된 투자금액(원)은 얼마인가?": "신규시설투자등",
            "한전기술이 체결한 단일판매·공급계약 중 이후 해지된 계약이 존재하는가?":
                "단일판매공급계약해지",
            "두산로보틱스의 2025년 투자판단관련주요경영사항 공시를 기준으로 주요 내용을 정리해줘.":
                "투자판단관련주요경영사항",
        }
        for q, expected in cases.items():
            assert expected in schema.match_kinds(q), q

    def test_summary_report_questions_get_a_field_list(self, schema):
        """open 요약 질문의 expected_fields 가 비면 충분성 검사가 무력해진다."""
        kind = schema.match_one("한미반도체의 2025년 신규시설투자등 공시를 기준으로 주요 내용을 정리해줘.")
        assert kind and len(schema.required(kind)) >= 5


# --------------------------------------------------------------------------- 흩어진 이름 매칭

def _toy_split() -> FieldSchema:
    return FieldSchema({"kinds": {
        "단일판매공급계약체결": {
            "doc_group": "exchange", "n_docs": 1106, "sufficient_data": True,
            "search_terms": ["단일판매공급계약체결"],
            "core_terms": ["단일판매공급계약"], "action_terms": ["체결"],
            "required": ["계약금액"], "conditional": [], "optional": [], "ratios": {},
        },
        "단일판매공급계약해지": {
            "doc_group": "exchange", "n_docs": 20, "sufficient_data": True,
            "search_terms": ["단일판매공급계약해지"],
            "core_terms": ["단일판매공급계약"], "action_terms": ["해지"],
            "required": ["계약금액"], "conditional": [], "optional": [], "ratios": {},
        },
    }})


def test_core_term_matching_when_name_is_split():
    """S015~S022 형태 — 이름이 통째로 안 나온다."""
    got = _toy_split().match_kinds("한전기술이 체결한 단일판매·공급계약 중 이후 해지된 계약이 존재하는가?")
    assert set(got) == {"단일판매공급계약체결", "단일판매공급계약해지"}


def test_action_word_narrows_the_siblings():
    got = _toy_split().match_kinds("현대건설의 단일판매ㆍ공급계약 해지 내역을 알려줘")
    assert got == ["단일판매공급계약해지"]


def test_core_without_any_action_returns_all_siblings():
    """어느 쪽인지 모를 때 좁히면 근거를 놓친다 — 전부 돌려준다."""
    got = _toy_split().match_kinds("삼성E&A의 단일판매ㆍ공급계약 내역")
    assert set(got) == {"단일판매공급계약체결", "단일판매공급계약해지"}


def test_core_matching_does_not_fire_without_the_stem():
    assert _toy_split().match_kinds("삼성전자 매출액은 얼마인가?") == []


@pytest.mark.skipif(not SCHEMA_PATH.exists(), reason="config/field_schema.json 미생성")
def test_built_schema_derives_core_terms_from_data():
    """어간·동작어는 손으로 쓴 게 아니라 유형 이름들의 공통 접두사에서 나온다."""
    schema = FieldSchema.load(SCHEMA_PATH)
    ks = schema.get("단일판매공급계약해지")
    assert ks is not None
    assert "단일판매공급계약" in ks.core_terms
    assert "해지" in ks.action_terms


# --------------------------------------------------------------------------- expected_fields

def test_expected_fields_single_kind_returns_required():
    assert _toy().expected_fields("SK하이닉스의 신규시설투자등 공시를 정리해줘") == ["투자금액", "투자목적"]


def test_expected_fields_two_kinds_returns_intersection():
    """어느 유형인지 확정 못 하면 '둘 중 무엇이든 반드시 필요한 것'만 요구한다."""
    s = FieldSchema({"kinds": {
        "가체결": {"n_docs": 100, "sufficient_data": True, "doc_group": "exchange",
                  "search_terms": ["가체결"], "core_terms": ["가"], "action_terms": ["체결"],
                  "required": ["공통", "체결전용"], "conditional": [], "optional": [], "ratios": {}},
        "가해지": {"n_docs": 100, "sufficient_data": True, "doc_group": "exchange",
                  "search_terms": ["가해지"], "core_terms": ["가"], "action_terms": ["해지"],
                  "required": ["공통", "해지전용"], "conditional": [], "optional": [], "ratios": {}},
    }})
    assert s.expected_fields("가 계약을 체결하고 해지한 내역") == ["공통"]


def test_expected_fields_empty_when_too_many_kinds():
    """S025 형태 — '주요사항보고서'만 말하면 19종에 걸린다. 그 required 를 전부
    요구하면 어떤 답변도 통과하지 못하므로 제약을 걸지 않는다."""
    kinds = {}
    for i in range(5):
        kinds[f"보고서(유형{i})"] = {
            "n_docs": 50, "sufficient_data": True, "doc_group": "major",
            "search_terms": [f"보고서(유형{i})"], "core_terms": ["보고서"],
            "action_terms": [f"(유형{i})"],
            "required": [f"항목{i}"], "conditional": [], "optional": [], "ratios": {},
        }
    s = FieldSchema({"kinds": kinds})
    assert len(s.match_kinds("보고서가 정정된 내역이 있는가?")) == 5
    assert s.expected_fields("보고서가 정정된 내역이 있는가?") == []


def test_expected_fields_empty_for_unknown_query():
    assert _toy().expected_fields("삼성전자 매출액은?") == []


@pytest.mark.skipif(not SCHEMA_PATH.exists(), reason="config/field_schema.json 미생성")
class TestExpectedFieldsOnGoldSet:
    @pytest.fixture(scope="class")
    @classmethod
    def schema(cls):
        return FieldSchema.load(SCHEMA_PATH)

    def test_summary_questions_get_a_usable_field_list(self, schema):
        for q in ("SK하이닉스의 2024년 신규시설투자등 공시를 기준으로 주요 내용을 정리해줘.",
                  "삼성E&A의 단일판매ㆍ공급계약체결 공시가 정정된 내역이 있는가?"):
            assert len(schema.expected_fields(q)) >= 5, q

    def test_periodic_questions_get_no_constraint(self, schema):
        """정기공시는 facts 미포함 — 잘못된 required 를 거는 것보다 낫다."""
        q = "삼성전자의 2024년 사업보고서와 2026년 사업보고서를 비교했을 때"
        assert schema.match_kinds(q) == []
        assert schema.expected_fields(q) == []

    def test_underspecified_major_question_gets_no_constraint(self, schema):
        q = "두산로보틱스의 주요사항보고서 공시가 정정된 내역이 있는가?"
        assert len(schema.match_kinds(q)) > 2      # 유형을 특정하지 못함
        assert schema.expected_fields(q) == []     # 그래서 제약 없음

    def test_closed_question_field_is_validated_not_listed(self, schema):
        """closed 질문의 항목은 질문에서 나온다 — 이 표는 검증만 한다."""
        assert schema.classify("단일판매공급계약체결", "계약금액") == "required"
        assert schema.classify("단일판매공급계약해지", "해지금액") == "required"
        assert schema.classify("단일판매공급계약체결", "없는항목") == "unknown"


# --------------------------------------------------------------------------- 회귀: 유형명 누출

@pytest.mark.skipif(not SCHEMA_PATH.exists(), reason="config/field_schema.json 미생성")
def test_disclosure_type_name_never_becomes_a_required_field():
    """실측 실패(v2 스모크 5문항 중 4문항 차단):

        "주요사항보고서(자기주식취득결정)에 기재된 순자산액은 얼마인가?"
        -> expected_fields 에 '주요사항보고서' 가 들어가 거부 게이트가
           답할 수 있는 질문을 막았다.

    '주요사항보고서' 는 kind 이름도 search_term 도 아닌 **core_term** 이라
    그 둘만 거르면 새어 나온다.
    """
    schema = FieldSchema.load(SCHEMA_PATH)
    cases = [
        ("삼성전자의 [기재정정]주요사항보고서(자기주식취득결정)에 기재된 순자산액은 얼마인가?",
         "순자산액"),
        ("POSCO홀딩스의 주요사항보고서(자기주식취득결정)에 기재된 자기주식 취득금액 한도는 얼마인가?",
         "자기주식취득금액한도"),
        ("SK하이닉스의 신규시설투자등에 기재된 투자금액(원)은 얼마인가?", "투자금액"),
        ("삼성전자의 단일판매ㆍ공급계약체결 공시의 계약금액은 얼마인가?", "계약금액"),
    ]
    for query, expected in cases:
        got = schema.fields_mentioned(query, schema.match_kinds(query))
        assert got == [expected], f"{query} -> {got}"


@pytest.mark.skipif(not SCHEMA_PATH.exists(), reason="config/field_schema.json 미생성")
def test_core_and_action_terms_are_excluded_from_fields():
    schema = FieldSchema.load(SCHEMA_PATH)
    for term in ("주요사항보고서", "단일판매공급계약", "체결", "해지"):
        assert term not in schema.fields_mentioned(f"{term} 관련 질문입니다", None)
