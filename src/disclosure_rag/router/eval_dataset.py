"""§48: Router 평가셋. 등록한 utterance(routes.py)와 절대 겹치지 않는 별도 phrasing 으로
구성한다. 초기 baseline 이라 route 당 10~15개 규모(§48 은 최종 목표로 route당 ~50개를
제시함) — 실제 서비스 전 반드시 확장해야 한다. 복합/애매한 질문은 별도 섹션(§47)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalExample:
    query: str
    expected_route: str | None  # None = 이 질문은 fallback 되는 게 맞다고 라벨링된 경우


EVAL_SET: list[EvalExample] = [
    # single_lookup
    EvalExample("[COMPANY] 재무제표 좀 보여줘", "single_lookup"),
    EvalExample("[COMPANY] 올해 매출 규모가 어느 정도야?", "single_lookup"),
    EvalExample("[COMPANY] 감사의견이 뭐였어?", "single_lookup"),
    EvalExample("[COMPANY] 임원 수가 몇 명이야?", "single_lookup"),
    EvalExample("[COMPANY] 본사가 어디 있어?", "single_lookup"),
    EvalExample("[COMPANY]의 총 발행주식수는?", "single_lookup"),
    EvalExample("[COMPANY] 계열회사가 몇 개야?", "single_lookup"),
    EvalExample("[COMPANY] 최근 분기 순이익 알려줘", "single_lookup"),
    EvalExample("[COMPANY] 설립일이 언제야", "single_lookup"),
    EvalExample("[COMPANY] 종업원 평균 근속연수는?", "single_lookup"),
    # correction_analysis
    EvalExample("[COMPANY] 공시 수정된 이유 설명해줘", "correction_analysis"),
    EvalExample("[COMPANY] 재무제표 수치가 바뀐 부분이 있어?", "correction_analysis"),
    EvalExample("[COMPANY] 정정신고 몇 차례나 했어?", "correction_analysis"),
    EvalExample("[COMPANY] 어느 부분이 잘못 기재됐던 거야?", "correction_analysis"),
    EvalExample("[COMPANY] 가장 최근 정정본 기준으로 알려줘", "correction_analysis"),
    EvalExample("[COMPANY] 기재정정 사유 알려줘", "correction_analysis"),
    EvalExample("[COMPANY] 정정 전 수치와 정정 후 수치 둘 다 보여줘", "correction_analysis"),
    EvalExample("[COMPANY] 공시 오류 정정한 적 있어?", "correction_analysis"),
    EvalExample("[COMPANY] 최초 제출본이랑 최종본 차이점", "correction_analysis"),
    EvalExample("[COMPANY] 정정 히스토리 알려줘", "correction_analysis"),
    # multi_compare
    EvalExample("[COMPANY_1]하고 [COMPANY_2] 중에 어디가 이익이 더 나?", "multi_compare"),
    EvalExample("[COMPANY] 지난 분기랑 이번 분기 실적 어떻게 달라졌어", "multi_compare"),
    EvalExample("[COMPANY_1], [COMPANY_2], [COMPANY_3] 매출 규모 순서대로 알려줘", "multi_compare"),
    EvalExample("[COMPANY] 연도별 매출 흐름 정리해줘", "multi_compare"),
    EvalExample("[COMPANY_1] vs [COMPANY_2] 부채비율 어디가 낮아?", "multi_compare"),
    EvalExample("[COMPANY]의 [YEAR_1]년이랑 [YEAR_2]년 영업이익 차이 보여줘", "multi_compare"),
    EvalExample("[COMPANY_1]과 [COMPANY_2] 시가총액 격차가 얼마나 나?", "multi_compare"),
    EvalExample("[COMPANY] 최근 몇 년간 실적 트렌드 알려줘", "multi_compare"),
    EvalExample("[COMPANY_1]과 [COMPANY_2] 둘 다 실적 알려주고 비교해줘", "multi_compare"),
    # calculation
    EvalExample("[COMPANY] 매출 증가율 몇 %야?", "calculation"),
    EvalExample("[COMPANY] 부채비율이 몇 퍼센트인지 계산해줘", "calculation"),
    EvalExample("[COMPANY] 최근 3개년 연평균성장률 알려줘", "calculation"),
    EvalExample("[COMPANY] 영업이익이 작년보다 얼마나 늘었어?", "calculation"),
    EvalExample("[COMPANY] 순이익률 몇 %야", "calculation"),
    EvalExample("[COMPANY] R&D 비중이 매출의 몇 %인지 계산해줘", "calculation"),
    EvalExample("[COMPANY] 매출 감소폭이 얼마야?", "calculation"),
    EvalExample("[COMPANY] ROE 얼마인지 구해줘", "calculation"),
    EvalExample("[COMPANY] 부채가 전년 대비 몇 % 늘었어", "calculation"),
    # ownership_analysis
    EvalExample("[COMPANY] 대주주가 누구야?", "ownership_analysis"),
    EvalExample("[COMPANY] 지분 5% 이상 가진 사람 있어?", "ownership_analysis"),
    EvalExample("[COMPANY] 최대주주 지분이 줄었어 늘었어?", "ownership_analysis"),
    EvalExample("[COMPANY] 특수관계인 보유 현황 알려줘", "ownership_analysis"),
    EvalExample("[COMPANY] 대량보유 신고 있었어?", "ownership_analysis"),
    EvalExample("[COMPANY] 최대주주 바뀐 적 있어?", "ownership_analysis"),
    EvalExample("[COMPANY] 주식 보유 목적이 경영참여야 단순투자야?", "ownership_analysis"),
    EvalExample("[COMPANY] 자사주 비율이 어떻게 돼?", "ownership_analysis"),
    # event_analysis
    EvalExample("[COMPANY] 최근에 큰 계약 따낸 거 있어?", "event_analysis"),
    EvalExample("[COMPANY] 채권 발행한 적 있어?", "event_analysis"),
    EvalExample("[COMPANY] 공장 증설 계획 있어?", "event_analysis"),
    EvalExample("[COMPANY] 다른 회사 인수했어?", "event_analysis"),
    EvalExample("[COMPANY] 자사주 매입 결정했어?", "event_analysis"),
    EvalExample("[COMPANY] 최근 수주 소식 있어?", "event_analysis"),
    EvalExample("[COMPANY] 합병 관련 이슈 있었어?", "event_analysis"),
    EvalExample("[COMPANY] 최근 공시 중에 이벤트성 뉴스 있어?", "event_analysis"),
    EvalExample("[COMPANY] 시설투자 규모 얼마야?", "event_analysis"),
]

# §47: 복합/애매한 질문 — Top-1 Route 로 명확히 판정하기 어려운 케이스.
# expected_route 를 두 후보 중 하나로 느슨하게 두거나, 실제로는 사람이 봐도
# 둘 다 맞는 경우가 많아 별도 집계(정답 후보 목록)로 다룬다.
AMBIGUOUS_SET: list[tuple[str, list[str]]] = [
    ("정정된 영업이익이 몇 % 줄었어?", ["correction_analysis", "calculation"]),
    ("[COMPANY] 정정 전후 부채비율 계산해서 비교해줘", ["correction_analysis", "calculation", "multi_compare"]),
    ("[COMPANY_1]과 [COMPANY_2] 중 어디가 최대주주 지분율이 더 높아?", ["multi_compare", "ownership_analysis"]),
    ("유상증자 이후 [COMPANY] 최대주주 지분이 몇 % 줄었어?", ["ownership_analysis", "calculation", "event_analysis"]),
]
