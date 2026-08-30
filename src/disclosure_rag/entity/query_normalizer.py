"""Query Normalize (§36): Semantic Router 가 회사명 자체에 과도하게 좌우되지
않도록 회사명을 [COMPANY] / [COMPANY_1] / [COMPANY_2] placeholder 로 치환한다.
회사명을 삭제하지 않고, 기업 수와 질의 구조를 그대로 보존한다."""

from __future__ import annotations

from disclosure_rag.common.unicode_utils import normalize_nfc
from disclosure_rag.entity.entity_extractor import ExtractedEntities


def normalize_query(entities: ExtractedEntities) -> str:
    # span 은 normalize_nfc(raw_query) 기준으로 계산됐으므로, 동일하게 정규화된
    # 문자열 위에서만 슬라이싱해야 offset 이 어긋나지 않는다.
    text = normalize_nfc(entities.raw_query)
    spans = sorted(entities.company_spans, key=lambda s: s[0])

    if not spans:
        return text

    multi = len(spans) > 1
    # 같은 회사가 여러 번 언급돼도 등장 순서대로 번호를 매긴다 (같은 회사 반복 언급 시 같은 번호 재사용)
    order: list[str] = []
    for _s, _e, corp in spans:
        if corp not in order:
            order.append(corp)
    index_of = {corp: i + 1 for i, corp in enumerate(order)}

    pieces = []
    cursor = 0
    for start, end, corp in spans:
        pieces.append(text[cursor:start])
        placeholder = f"[COMPANY_{index_of[corp]}]" if multi else "[COMPANY]"
        pieces.append(placeholder)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)
