"""§37, §46, §49: Semantic Router 배선. Router 는 "어느 문서를 검색할지" 를
정하는 Retriever 가 아니라 "이 질문이 어떤 작업 유형인지" 만 판단한다.

Router interface 를 분리해 Semantic Router / HCX Router / Router 없음(§49) 을
교체 가능하게 한다."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel

from disclosure_rag.retrieval.embeddings import EmbeddingProvider

if TYPE_CHECKING:  # pragma: no cover
    from semantic_router import SemanticRouter

# `semantic_router` 는 선택 의존성이다. 현재 프로덕션 경로는 router=None 으로
# 돌고(HCX Agent 가 직접 판단), 이 패키지는 라우터 실험에서만 필요하다.
# 모듈 최상단에서 import 하면 미설치 환경에서 `agent_loop` -> `evidence` ->
# `validator` 까지 줄줄이 import 에러가 나서 **코어 테스트 3개 모듈이 통째로
# 수집 실패**했다(실측). 그래서 실제로 쓰는 함수 안에서만 import 한다.
from disclosure_rag.router.routes import ROUTE_UTTERANCES

DEFAULT_THRESHOLD = 0.5


class RouteResult(BaseModel):
    route: str | None  # None = fallback (§46) -> HCX Agent 가 직접 판단
    score: float | None


class Router(Protocol):
    name: str

    def route(self, normalized_query: str) -> RouteResult: ...


def build_semantic_router(
    provider: EmbeddingProvider, *, threshold: float = DEFAULT_THRESHOLD,
) -> "SemanticRouter":
    from semantic_router import Route, SemanticRouter

    # encoder_adapter 는 DenseEncoder 를 상속하므로 이것도 지연 import 해야 한다.
    from disclosure_rag.router.encoder_adapter import ProviderBackedEncoder

    encoder = ProviderBackedEncoder(provider)
    routes = [
        Route(name=name, utterances=utterances, score_threshold=threshold)
        for name, utterances in ROUTE_UTTERANCES.items()
    ]
    return SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")


def set_all_thresholds(router: "SemanticRouter", threshold: float) -> None:
    """모든 route 의 threshold 를 일괄 변경한다 (재임베딩 없이 threshold sweep 가능, §48)."""
    for route in router.routes:
        route.score_threshold = threshold


class SemanticRouterAdapter:
    name = "semantic_router"

    def __init__(self, provider: EmbeddingProvider, *, threshold: float = DEFAULT_THRESHOLD):
        self._router = build_semantic_router(provider, threshold=threshold)

    def route(self, normalized_query: str) -> RouteResult:
        choice = self._router(normalized_query)
        return RouteResult(route=choice.name, score=choice.similarity_score)

    def set_threshold(self, threshold: float) -> None:
        set_all_thresholds(self._router, threshold)


class NoRouter:
    """§49 비교축: Router 없이 HCX Agent 가 직접 판단하는 baseline."""

    name = "none"

    def route(self, normalized_query: str) -> RouteResult:
        return RouteResult(route=None, score=None)
