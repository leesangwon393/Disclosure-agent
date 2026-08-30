"""Dense Embedding Provider 추상화 (사용자 결정 #2).

baseline: BAAI/bge-m3. provider interface 를 분리해 추후 multilingual-e5-large-instruct,
HCX Embedding 과 Recall@K 비교가 가능하게 한다 (§74 평가 가능 모듈화).
"""

from __future__ import annotations

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
