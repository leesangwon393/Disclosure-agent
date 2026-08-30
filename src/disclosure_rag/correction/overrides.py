"""Deterministic rule 로 처리되지 않는 known edge case 의 명시적 override table.

사용자 결정 #5: "전체 정정 regex 를 과도하게 느슨하게 만들지 않는다. deterministic
rule 로 처리되지 않는 known edge case 는 explicit override table 로 처리한다."

각 entry 는 doc_id -> target_date(YYYYMMDD) 강제 지정이며, 왜 필요한지 주석으로
근거를 남긴다. 새 edge case 발견 시 여기에 "발견 경위 + 검증 근거"와 함께 추가한다.
"""

from __future__ import annotations

MANUAL_TARGET_DATE_OVERRIDES: dict[str, str] = {
    # 원문 자체 오타: "2. 정정대상 공시서류의 최초제출일 : 2025년 08년 28일"
    # (두 번째 구분자가 "월" 이어야 하는데 "년" 으로 잘못 기재됨 — 미래에셋증권 제출인 실수)
    # 검증: 같은 회사(미래에셋증권)가 같은 날(2025-08-28) 자기주식취득결정 원본
    # (major_20250828001211)을 제출한 사실을 확인 — 당일 정정 케이스로 판단.
    "major_20250828001452": "20250828",
}

# 이 doc_id 들은 regex/override 로도 원본을 특정할 수 없어 resolution_source="unresolved"
# 로 표시된다. 발견 시 사람이 직접 검토 후 위 dict 에 추가하거나 여기서 사유를 남긴다.
KNOWN_UNRESOLVABLE: dict[str, str] = {}
