"""Dense Embedding Provider 추상화 (사용자 결정 #2).

baseline: BAAI/bge-m3. provider interface 를 분리해 추후 multilingual-e5-large-instruct,
HCX Embedding 과 Recall@K 비교가 가능하게 한다 (§74 평가 가능 모듈화).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Protocol

from disclosure_rag.common.device import pick_device, use_fp16_for


class EmbeddingProvider(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class BgeM3EmbeddingProvider:
    """baseline. sentence-transformers 로 BAAI/bge-m3 dense embedding 만 사용한다
    (bge-m3 는 dense/sparse/colbert 3 종을 지원하지만, baseline 은 dense 만 사용하고
    sparse/multi-vector 는 향후 비교 대상으로 남겨둔다)."""

    name = "bge-m3"
    dim = 1024

    def __init__(self, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("BAAI/bge-m3", device=pick_device(device))

    def embed(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        vecs = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class E5InstructEmbeddingProvider:
    """비교 대상 #1 (사용자 결정 #2). multilingual-e5-large-instruct 는 query 앞에
    instruction prefix 를 붙여야 성능이 나온다는 점이 bge-m3 와 다르다."""

    name = "e5-instruct"
    dim = 1024
    _QUERY_PREFIX = "Instruct: Given a financial disclosure question, retrieve relevant passages\nQuery: "

    def __init__(self, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            "intfloat/multilingual-e5-large-instruct", device=pick_device(device))

    def embed(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        vecs = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([self._QUERY_PREFIX + text])[0]


class Qwen3EmbeddingProvider:
    """비교 대상 (Stage 3 실험). Qwen3-Embedding-0.6B — e5 처럼 query 에 instruction
    prefix 를 붙여야 한다 (document 쪽은 prefix 없음)."""

    name = "qwen3-embedding-0.6b"
    dim = 1024
    _QUERY_PREFIX = "Instruct: Given a financial disclosure question, retrieve relevant passages\nQuery: "

    def __init__(self, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=pick_device(device))

    def embed(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        vecs = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([self._QUERY_PREFIX + text])[0]


class HCXEmbeddingProvider:
    """비교 대상 #2 (사용자 결정 #2). HCX API 연결이 별도 dependency 로 확정되기
    전까지는 placeholder — 호출 시 명시적으로 NotImplementedError 를 낸다
    (silent 하게 dummy 벡터를 반환하지 않는다, §7 원칙을 embedding 에도 적용)."""

    name = "hcx-embedding"
    dim = 1024  # TODO: 실제 API 연결 후 확정

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "HCX Embedding API 미연결 — .env 의 HCX_API_KEY 로 실제 API 클라이언트를 "
            "구현해야 사용 가능. Phase 15(HCX Agent) 연결 시 함께 배선 예정."
        )

    def embed(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


def build_embedding_provider(name: str, **kwargs) -> EmbeddingProvider:
    if name == "bge-m3":
        return BgeM3EmbeddingProvider(**kwargs)
    if name in ("bge-m3-multi", "bge-m3-3mode"):
        return BgeM3MultiProvider(**kwargs)
    if name == "e5-instruct":
        return E5InstructEmbeddingProvider(**kwargs)
    if name == "qwen3-embedding-0.6b":
        return Qwen3EmbeddingProvider(**kwargs)
    if name == "hcx-embedding":
        return HCXEmbeddingProvider(**kwargs)
    raise ValueError(f"알 수 없는 embedding provider: {name}")


# =====================================================================
# KIM 브랜치 추가 (2026-08-23): BGE-M3 3모드 동시 추출
# =====================================================================
class BgeM3MultiProvider:
    """BGE-M3 의 dense / sparse(learned lexical) / multi-vector(ColBERT) 를 **한 번에** 뽑는다.

    왜 중요한가
    -----------
    기존 코드는 sentence-transformers 로 로드해 dense 만 받았고, 주석에도
    "sparse/multi-vector 는 향후 비교 대상으로 남겨둔다"고 적혀 있었다.
    그런데 이 셋은 **같은 forward pass 의 서로 다른 출력 헤드**다. 즉 dense 만
    뽑으나 셋 다 뽑으나 **모델 연산 비용이 사실상 같다.**

    전체 코퍼스 임베딩은 M5 Pro/MPS 실측 61.4ms/chunk 기준 8~13시간짜리 작업이다.
    그 한 번으로 세 가지 검색 방식을 전부 확보해두면, 이후
    `BM25 / dense / sparse / colbert` 조합을 **재임베딩 없이** 실험할 수 있다.
    dense 만 뽑아두면 나중에 sparse 가 필요해질 때 또 10시간이다.

    도메인 관점
    -----------
    공시는 Dense 에 불리하다(표 조각의 숫자 비중 20%, 정정공시 43%가 원본과 거의
    동일 텍스트, 법정 통제어휘, leaf 의 절반 이상이 표). 특히 **표·숫자를 벡터
    하나로 뭉개는 문제의 정면 해결책이 late interaction(ColBERT)** 이고,
    그게 이미 이 모델 안에 들어 있다.
    """

    name = "bge-m3-multi"
    dim = 1024

    def __init__(self, device: str | None = None, *, use_fp16: bool | None = None):
        from FlagEmbedding import BGEM3FlagModel  # 별도 패키지: pip install FlagEmbedding

        # 장치는 절대 라이브러리 자동감지에 맡기지 않는다 — FlagEmbedding 은 버전에 따라
        # cuda 를 기본값으로 잡는데 이 환경(Apple Silicon)에는 CUDA 가 없다.
        dev = pick_device(device)
        fp16 = use_fp16_for(dev) if use_fp16 is None else use_fp16
        try:  # 신버전 시그니처
            self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=fp16, devices=dev)
        except TypeError:  # 구버전
            self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=fp16, device=dev)
        self.device = dev

    def encode_all(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        max_length: int = 1024,
        dense: bool = True,
        sparse: bool = True,
        colbert: bool = False,
    ) -> dict:
        """{'dense_vecs':..., 'lexical_weights':..., 'colbert_vecs':...} 를 돌려준다.

        max_length: 우리 청크는 p90 이 1,905자(~950토큰)라 1024 면 잘리지 않는다.
        8192 로 두면 패딩 비용만 커진다.
        """
        return self._model.encode(
            texts, batch_size=batch_size, max_length=max_length,
            return_dense=dense, return_sparse=sparse, return_colbert_vecs=colbert,
        )

    # --- 기존 EmbeddingProvider 인터페이스 호환 (router 등이 그대로 쓴다) ---
    def embed(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        out = self.encode_all(texts, batch_size=batch_size, sparse=False, colbert=False)
        return [v.tolist() for v in out["dense_vecs"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class SharedQueryEncoder:
    """질의 인코딩을 dense/sparse 가 **한 forward pass 로 나눠 쓰게** 한다.

    왜 필요한가
    -----------
    dense 검색기는 `embed_query()` 로 BGE-M3 를 한 번 돌리고, sparse 검색기는
    같은 질의로 lexical weights 를 받으려고 **또 한 번** 돌린다. 두 출력은
    같은 forward pass 의 서로 다른 헤드라서 원래 한 번이면 된다. 질의마다
    모델을 두 번 태우던 것을 한 번으로 줄인다 — 점수는 한 글자도 안 바뀐다
    (같은 모델, 같은 입력, 같은 출력).

    재검색(nudge)이나 하위 질의가 같은 문장을 다시 물을 때를 대비해 최근
    질의를 조금 캐시한다. 캐시는 프로세스 안에서만 살고 인덱스와 무관하다.

    max_length 를 넘기지 않는 이유: 원래 두 경로 모두 `encode_all` 기본값을
    썼다. 여기서 다른 값을 주면 '결과가 같다'는 보장이 깨진다.
    """

    def __init__(self, provider, *, cache_size: int = 128):
        self._provider = provider
        self._cache: "OrderedDict[str, tuple[list[float], dict[str, float]]]" = OrderedDict()
        self._cache_size = max(1, int(cache_size))
        self.hits = 0
        self.misses = 0

    # 원 provider 의 나머지 속성(name, dim, encode_all ...)은 그대로 통한다.
    def __getattr__(self, item):
        return getattr(self._provider, item)

    def _encode(self, text: str) -> tuple[list[float], dict[str, float]]:
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            self.hits += 1
            return cached
        self.misses += 1
        out = self._provider.encode_all(
            [text], batch_size=1, dense=True, sparse=True, colbert=False,
        )
        dense = [float(x) for x in out["dense_vecs"][0]]
        lexical = {str(k): float(v) for k, v in out["lexical_weights"][0].items()}
        self._cache[text] = (dense, lexical)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return dense, lexical

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)[0]

    def lexical_query(self, text: str) -> dict[str, float]:
        return self._encode(text)[1]

    def embed(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        """여러 건은 캐시하지 않는다 — 코퍼스 임베딩까지 들고 있으면 안 된다."""
        return self._provider.embed(texts, batch_size=batch_size)
