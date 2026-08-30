"""§38~46: 6개 baseline Route 와 대표 utterance.

utterance 는 [COMPANY]/[COMPANY_1]/[COMPANY_2] placeholder 로 정규화된 형태로
등록한다 — 실제 서비스에서도 Query Normalize(§36) 를 거친 뒤 Router 에 넣기
때문에, 등록 데이터와 입력 형태를 일치시켜야 한다.

각 Route 안에서도 서로 다른 하위 intent 를 커버한다 (§45: 단순 paraphrase 반복 금지).
"""

from __future__ import annotations

ROUTE_UTTERANCES: dict[str, list[str]] = {
    # §39: 단일 사실/수치 조회
    "single_lookup": [
        "[COMPANY]의 [YEAR] 매출액은 얼마야?",
        "[COMPANY] 영업이익 알려줘",
        "[COMPANY] 연구개발비 얼마야?",
        "[COMPANY] 당기순이익은?",
        "[COMPANY] 부채총액 알려줘",
        "[COMPANY] 사업보고서에서 매출액 찾아줘",
        "[COMPANY] 자산총계가 얼마야",
        "[COMPANY] 배당금 얼마 줬어?",
        "[COMPANY] 자본총계 알려줘",
        "[COMPANY] 최근 분기 영업이익률은?",
        "[COMPANY]의 [YEAR] 반기보고서 매출 얼마야",
        "[COMPANY] 유동비율이 어떻게 돼?",
        "[COMPANY] 총 직원 수 알려줘",
        "[COMPANY] 대표이사 누구야?",
        "[COMPANY] 상장일이 언제야?",
        "[COMPANY] 결산월이 몇 월이야",
        "[COMPANY]의 EBITDA 얼마야",
        "[COMPANY] 재고자산 규모 알려줘",
        "[COMPANY] 매출채권 얼마나 돼?",
        "[COMPANY] 최근 분기보고서 요약해줘",
    ],
    # §40: 정정 이유/항목/전후비교/최종본/다중 정정 이력
    "correction_analysis": [
        "[COMPANY] 정정공시에서 뭐가 바뀌었어?",
        "[COMPANY] 사업보고서가 왜 정정됐어?",
        "[COMPANY] 정정 전후 매출액 비교해줘",
        "[COMPANY] 기존 공시와 수정 공시의 차이가 뭐야?",
        "[COMPANY] 최종 정정본에서 바뀐 내용 알려줘",
        "[COMPANY] 정정 사유가 뭐야?",
        "[COMPANY] 어떤 항목이 정정됐어?",
        "[COMPANY] 정정된 숫자가 얼마야?",
        "[COMPANY] 최초 공시랑 정정본이랑 뭐가 달라?",
        "[COMPANY] 이 공시 몇 번 정정됐어?",
        "[COMPANY] 정정 이력 전부 보여줘",
        "[COMPANY] 최종 유효본이 어느 거야?",
        "[COMPANY] 왜 기재정정했는지 알려줘",
        "[COMPANY] 정정신고서 내용 요약해줘",
        "[COMPANY]의 [YEAR] 사업보고서 정정 이유가 뭐야",
        "[COMPANY] 반기보고서 정정된 부분만 알려줘",
        "[COMPANY] 계약금액이 정정으로 얼마나 바뀌었어?",
        "[COMPANY] 최근 정정공시 있었어?",
    ],
    # §41: 다중 비교 (기업간/기간간)
    "multi_compare": [
        "[COMPANY_1]와 [COMPANY_2] 매출 비교해줘",
        "[COMPANY_1]과 [COMPANY_2] 영업이익률 비교해줘",
        "[COMPANY]의 [YEAR_1]과 [YEAR_2] 실적 비교해줘",
        "[COMPANY] 최근 3년 매출 추이 비교해줘",
        "[COMPANY_1]과 [COMPANY_2] 중 어디가 더 커?",
        "[COMPANY_1], [COMPANY_2] 부채비율 비교해줘",
        "[COMPANY] 분기별 영업이익 추이 알려줘",
        "[COMPANY_1]과 [COMPANY_2] 연구개발비 비교",
        "[COMPANY]의 작년과 올해 순이익 비교해줘",
        "[COMPANY_1] 대비 [COMPANY_2] 시가총액이 얼마나 커?",
        "[COMPANY] 최근 5개 분기 실적 흐름 보여줘",
        "[COMPANY_1]과 [COMPANY_2]와 [COMPANY_3] 매출 순위 매겨줘",
        "[COMPANY]의 상반기와 하반기 매출 비교",
        "[COMPANY_1]과 [COMPANY_2] 중 배당수익률이 더 높은 곳은?",
    ],
    # §42: 계산 (증가율/CAGR/비율)
    "calculation": [
        "[COMPANY] 매출이 전년 대비 몇 % 증가했어?",
        "[COMPANY] 최근 3년 CAGR 계산해줘",
        "[COMPANY] 영업이익률 계산해줘",
        "[COMPANY] 부채비율 계산해줘",
        "정정 전보다 영업이익이 몇 % 감소했어?",
        "[COMPANY] 매출 성장률 알려줘",
        "[COMPANY] 자기자본이익률(ROE) 계산해줘",
        "[COMPANY] 연구개발비가 매출의 몇 %야?",
        "[COMPANY] 순이익률 계산해줘",
        "[COMPANY]의 [YEAR_1] 대비 [YEAR_2] 증감액 알려줘",
        "[COMPANY] 부채가 몇 % 늘었어?",
        "[COMPANY] 유상증자로 지분이 몇 % 희석됐어?",
        "[COMPANY] 3개년 평균 영업이익 계산해줘",
    ],
    # §43: 지분/최대주주 분석
    "ownership_analysis": [
        "[COMPANY] 최대주주 지분율은 얼마야?",
        "[COMPANY] 최대주주가 변경됐어?",
        "[COMPANY] 주요주주 지분이 어떻게 변했어?",
        "[COMPANY] 최근 지분 변동 알려줘",
        "유상증자 이후 [COMPANY] 최대주주 지분이 어떻게 변했어?",
        "[COMPANY] 대량보유상황보고서 내용 알려줘",
        "[COMPANY] 5% 이상 보유자가 누구야?",
        "[COMPANY] 특별관계자가 누구야?",
        "[COMPANY] 최대주주 보유 목적이 뭐야?",
        "[COMPANY] 지분 매도했어?",
        "[COMPANY] 자기주식 보유 비율 알려줘",
        "[COMPANY] 경영권 변경 있었어?",
        "[COMPANY] 주주 구성 알려줘",
    ],
    # §44: 이벤트(계약/투자/증자 등)
    "event_analysis": [
        "[COMPANY] 최근 유상증자했어?",
        "[COMPANY] 전환사채 발행했어?",
        "[COMPANY] 대규모 공급계약 체결했어?",
        "[COMPANY] 최근 인수합병 관련 공시 있어?",
        "[COMPANY] 신규 시설투자 결정했어?",
        "[COMPANY] 자기주식 취득했어?",
        "[COMPANY] 회사분할 있었어?",
        "[COMPANY] 최근 계약 해지된 거 있어?",
        "[COMPANY] 영업양수도 있었어?",
        "[COMPANY] 최근 공급계약 규모가 얼마야?",
        "[COMPANY] 신규 투자 계획 알려줘",
        "[COMPANY] 최근 주요사항보고서 뭐가 있었어?",
        "[COMPANY] 자사주 소각했어?",
        "[COMPANY] 최근 어떤 이벤트가 있었어?",
        "[COMPANY] 해외 상장 관련 공시 있었어?",
    ],
}
