"""BM25 용 Tokenizer 추상화 (§32, §74: whitespace / Kiwi / char n-gram 비교 가능해야 함).

baseline 은 Kiwi(kiwipiepy) 를 사용한다. 금융 전문용어가 형태소 분석기 기본
사전에 없어 엉뚱하게 쪼개지는 문제를 막기 위해 사용자 사전(config/financial_terms.txt)
을 로딩해 확장 가능하게 한다 (§32).
"""

from __future__ import annotations

from sys import intern

from pathlib import Path
from typing import Protocol

# BM25 키워드로 남길 품사: 체언(명사류) + 어근 + 외국어/한자/숫자. 조사/어미/부호는 제외.
_DEFAULT_KEEP_TAGS = {"NNG", "NNP", "NNB", "NR", "NP", "SL", "SH", "SN", "XR"}


class Tokenizer(Protocol):
    name: str

    def tokenize(self, text: str) -> list[str]: ...


class WhitespaceTokenizer:
    """가장 단순한 baseline. 비교 실험용."""

    name = "whitespace"

    def tokenize(self, text: str) -> list[str]:
        return text.split()


class CharNgramTokenizer:
    """형태소 분석기 없이도 동작하는 대안. 한글 조사 변화에 어느 정도 강건하다."""

    def __init__(self, n: int = 2):
        self.n = n
        self.name = f"char_{n}gram"

    def tokenize(self, text: str) -> list[str]:
        cleaned = "".join(text.split())
        if len(cleaned) < self.n:
            return [cleaned] if cleaned else []
        return [intern(cleaned[i:i + self.n]) for i in range(len(cleaned) - self.n + 1)]


class KiwiTokenizer:
    """baseline tokenizer. kiwipiepy 형태소 분석 + 금융 용어 사용자 사전."""

    name = "kiwi"

    def __init__(
        self,
        *,
        user_dict_path: str | Path | None = None,
        keep_tags: set[str] | None = None,
    ):
        from kiwipiepy import Kiwi  # 무거운 import 는 실제 사용 시점에

        self._kiwi = Kiwi()
        self._keep_tags = keep_tags or _DEFAULT_KEEP_TAGS
        if user_dict_path is not None:
            self._load_user_dict(Path(user_dict_path))

    def _load_user_dict(self, path: Path) -> None:
        if not path.is_file():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            word = parts[0].strip()
            tag = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "NNG"
            score = float(parts[2]) if len(parts) > 2 and parts[2].strip() else 0.0
            if word:
                self._kiwi.add_user_word(word, tag, score)

    def tokenize(self, text: str) -> list[str]:
        # sys.intern: 형태소는 코퍼스 전체에서 극도로 반복된다(55만 조각 / 고유 형태소 수십만).
        # intern 없이 조각마다 새 str 객체를 만들면 BM25 인덱싱용 corpus_tokens 가
        # **13.5GB** 가 된다(실측). intern 하면 2.0GB 다 — 리스트의 포인터만 남는다.
        # 2026-08-23: ColBERT / sparse 역색인에 이어 세 번째로 발견한 같은 유형의 문제.
        tokens = self._kiwi.tokenize(text)
        return [intern(t.form) for t in tokens if t.tag in self._keep_tags]


def build_tokenizer(name: str, *, user_dict_path: str | Path | None = None) -> Tokenizer:
    if name == "whitespace":
        return WhitespaceTokenizer()
    if name == "kiwi":
        return KiwiTokenizer(user_dict_path=user_dict_path)
    if name.startswith("char_") and name.endswith("gram"):
        n = int(name[len("char_"):-len("gram")])
        return CharNgramTokenizer(n=n)
    raise ValueError(f"알 수 없는 tokenizer: {name}")
