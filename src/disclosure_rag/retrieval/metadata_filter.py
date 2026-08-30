"""Metadata Filtering (§51): 전체 corpus 를 Coarse 하게 줄인 뒤 BM25/Dense 를 돌린다.

## period 매칭 규칙 (2026-08-30 수정 — 그 전까지 이 필터는 사실상 동작하지 않았다)

실측(artifacts_v2/l1/chunks.jsonl.gz 40만 chunk 스캔):

| report_type | 건수    | `period` 결측 |
|-------------|--------:|--------------:|
| periodic    | 374,619 |   0 (  0.0%)  |
| holding     |  22,904 | 전건 (100.0%) |
| major       |   2,379 | 전건 (100.0%) |
| exchange    |      98 | 전건 (100.0%) |

즉 `period` 는 **정기공시에만** 있고 값은 항상 `"YYYY-MM"` 13종이다
(`2023-03` ~ `2026-03`, 월은 03/06/09/12 — 70개사 전부 12월 결산).

옛 구현은 `chunk.period not in self.periods` 정확일치라서 두 가지로 깨졌다.

1. `"2024"` 로 거르면 chunk 값이 `"2024-12"` 라 **영구 0건**.
2. `"2024-12"` 로 거르면 period 가 없는 major/exchange/holding 이 **통째로 0건**.

게다가 `EntityExtractor` 가 뽑는 토큰은 `"2024년"` `"1분기"` `"상반기"`
`"최근 3년"` 형태라 지원 포맷과 애초에 겹치지 않았다. 그래서 실제로는
`tools.py` 의 coarse-to-fine 완화 단계가 매번 period 를 통째로 버리고 있었고,
"기간 필터가 있다" 는 것은 문서상의 주장일 뿐이었다.

수정 후:

- 요청 `"YYYY"` → chunk `"YYYY-MM"` 전부 매칭 (연도 단위)
- 요청 `"YYYY-MM"` → 같은 값 매칭
- chunk 에 period 가 없으면(major/exchange/holding) `filing_date` 의 **연도**로
  대체 판정한다. 요청이 `"YYYY-MM"` 이어도 연도만 비교한다 — 이 문서들에는
  보고 기준기간 자체가 없으므로 월 단위로 거르면 무조건 0건이 되기 때문이다.
- `filing_date` 도 없으면 판단 불가로 보고 제외한다(실측 결측 0건).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from disclosure_rag.chunking.chunk_schema import ChunkSchema
from disclosure_rag.common.unicode_utils import normalize_nfc

# "2024년" "2024" "2024.3" "2024-03" "2024. 3월"
_YEAR_RE = re.compile(r"(20\d{2})")
_YM_RE = re.compile(r"(20\d{2})\s*[.\-/]\s*(\d{1,2})")
# 70개사 전부 12월 결산이므로 분기 → 기준월이 결정된다.
_QUARTER_MONTH = {"1": "03", "2": "06", "3": "09", "4": "12"}


def normalize_period_tokens(tokens: list[str] | str | None) -> list[str] | None:
    """자연어 기간 토큰을 chunk 의 period 포맷(`YYYY-MM` / `YYYY`)으로 바꾼다.

    `EntityExtractor` 와 HCX tool 인자가 내놓는 표현을 그대로 필터에 넣으면
    영구 0건이 되므로 반드시 이 함수를 거친다.

    - `"2024년"`            -> `["2024"]`
    - `"2024년 1분기"`      -> `["2024-03"]`
    - `"2024.3월"`          -> `["2024-03"]`
    - `"2024년 상반기"`     -> `["2024-06"]`
    - `"2024년 하반기"`     -> `["2024"]`   (기준월이 하나로 안 정해져 연도 단위)
    - `"최근 3년"` `"1분기"` -> `[]`         (연도가 없어 필터로 못 씀)

    반환이 빈 리스트면 **기간 필터를 걸지 않는다**는 뜻이다. `None` 을 돌려주어
    호출부가 그대로 `periods=None` 으로 쓰게 한다.
    """
    if tokens is None:
        return None
    if isinstance(tokens, str):
        tokens = [tokens]
    joined = " ".join(t for t in tokens if t)
    if not joined.strip():
        return None

    out: list[str] = []

    for m in _YM_RE.finditer(joined):
        month = int(m.group(2))
        if 1 <= month <= 12:
            out.append(f"{m.group(1)}-{month:02d}")

    years = [m.group(1) for m in _YEAR_RE.finditer(joined)]
    # 이미 YYYY-MM 으로 잡힌 연도는 중복으로 넣지 않는다.
    ym_years = {v[:4] for v in out}
    years = [y for y in years if y not in ym_years]

    if years:
        qm = re.search(r"([1-4])\s*분기", joined)
        if qm:
            month = _QUARTER_MONTH[qm.group(1)]
            out.extend(f"{y}-{month}" for y in years)
        elif "상반기" in joined:
            out.extend(f"{y}-06" for y in years)
        else:
            # "하반기" 도 여기로 온다 — 기준월이 09/12 로 갈려 연도 단위가 안전하다.
            out.extend(years)

    # 순서 유지 중복 제거
    seen: set[str] = set()
    dedup = [x for x in out if not (x in seen or seen.add(x))]
    return dedup or None


@dataclass
class RetrievalFilter:
    companies: list[str] | None = None       # NFC 정규화된 corp_name 리스트
    report_ids: list[str] | None = None       # 특정 문서(report_id/doc_id)로 정확히 좁힐 때 사용
    doc_groups: list[str] | None = None       # periodic|major|exchange|holding
    doc_subtypes: list[str] | None = None
    periods: list[str] | None = None          # "YYYY-MM" 또는 "YYYY" (normalize_period_tokens 통과분)
    filing_date_from: str | None = None       # YYYYMMDD
    filing_date_to: str | None = None
    latest_only: bool = False                 # 일반 조회: 최신 유효본만 (§30)
    include_corrections: bool = True          # 정정 분석: original+정정 체인 유지 (§30)

    def __post_init__(self):
        if self.companies:
            self.companies = [normalize_nfc(c) for c in self.companies]

    @property
    def is_selective(self) -> bool:
        """후보를 아주 좁게 자르는 필터인가.

        `report_ids`(문서 1~수개) 나 `companies`(70분의 1~몇) 가 걸리면 통과
        대상이 전체의 1% 미만이 된다. 이때 리트리버가 "전체에서 상위 N개를 먼저
        뽑고 그 다음에 거른다" 는 순서로 동작하면 **통과 대상이 상위 N개 안에
        하나도 없어 빈 결과가 나온다.**

        2026-08-30 실측으로 확인된 문제다. 60문항 full 실행에서 에이전트 경로의
        정답문서 회수율이 66.7% 였는데, 같은 질의를 필터 없이 그냥 검색하면
        94.4% 였다. 2단계 검색(문서 확정 -> 그 문서 안에서 재검색)을 붙이려면
        이 순서가 먼저 고쳐져야 한다 — 안 그러면 2단계가 통째로 헛돈다.
        """
        return bool(self.report_ids or self.companies)

    def _period_matches(self, chunk: ChunkSchema) -> bool:
        if not self.periods:
            return True

        chunk_period = chunk.period
        if chunk_period:
            candidates = {chunk_period, chunk_period[:4]}
        else:
            filing_date = chunk.filing_date or ""
            if len(filing_date) < 4:
                return False
            candidates = {filing_date[:4]}

        for want in self.periods:
            want = (want or "").strip()
            if not want:
                continue
            if want in candidates:
                return True
            # 요청이 "YYYY-MM" 인데 기준기간이 없는 문서(major/exchange/holding)
            # 는 연도까지만 비교한다. 월로 거르면 이 3종이 전부 사라진다.
            if not chunk_period and len(want) >= 7 and want[:4] in candidates:
                return True
        return False

    def matches(self, chunk: ChunkSchema) -> bool:
        if self.report_ids and chunk.report_id not in self.report_ids:
            return False
        if self.companies and chunk.company not in self.companies:
            return False
        if self.doc_groups and chunk.report_type not in self.doc_groups:
            return False
        if self.doc_subtypes and chunk.report_subtype not in self.doc_subtypes:
            return False
        if not self._period_matches(chunk):
            return False
        if self.filing_date_from and (chunk.filing_date or "") < self.filing_date_from:
            return False
        if self.filing_date_to and (chunk.filing_date or "") > self.filing_date_to:
            return False
        if self.latest_only and chunk.is_latest is False:
            return False
        if not self.include_corrections and chunk.is_correction:
            return False
        return True


def filter_chunks(chunks: list[ChunkSchema], flt: RetrievalFilter | None) -> list[ChunkSchema]:
    if flt is None:
        return chunks
    return [c for c in chunks if flt.matches(c)]
