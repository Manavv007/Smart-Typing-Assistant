"""Suggestion engine: turns a model into a phone-keyboard style predictor.

Two modes, decided by the text itself:
  * next-word   "i am going to "  -> words likely to come next
  * completion  "i am going to sc" -> words starting with "sc" that fit
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Sequence

from .ngram import LanguageModel
from .text import SPECIAL, split_for_prediction

CANDIDATES_PER_ORDER = 30
UNIGRAM_CANDIDATES = 30


@dataclass
class Suggestion:
    word: str
    score: float
    order: int  # longest n-gram order in which (context, word) was seen; 0 = never


class Predictor:
    def __init__(self, model: LanguageModel):
        self.model = model
        self.counts = model.counts
        uni = self.counts.unigram
        self.by_freq = sorted((w for w in uni if w not in SPECIAL), key=lambda w: -uni[w])
        self.sorted_vocab = sorted(self.by_freq)
        self._top_cache: dict = {}
        self._prefix_cache: dict = {}

    # ---- candidate generation ------------------------------------------------
    def _top(self, k: int, h) -> list[str]:
        key = (k, h)
        top = self._top_cache.get(key)
        if top is None:
            ws = self.counts.raw[k].data.get(h, {})
            top = sorted((w for w in ws if w not in SPECIAL), key=lambda w: -ws[w])
            top = top[:CANDIDATES_PER_ORDER]
            if len(self._top_cache) > 200_000:
                self._top_cache.clear()
            self._top_cache[key] = top
        return top

    def _prefix_words(self, prefix: str) -> list[str]:
        """Most frequent vocabulary words starting with prefix."""
        top = self._prefix_cache.get(prefix)
        if top is None:
            uni = self.counts.unigram
            i = bisect_left(self.sorted_vocab, prefix)
            matches = []
            while i < len(self.sorted_vocab) and self.sorted_vocab[i].startswith(prefix):
                matches.append(self.sorted_vocab[i])
                i += 1
            matches.sort(key=lambda w: -uni[w])
            top = matches[:UNIGRAM_CANDIDATES]
            self._prefix_cache[prefix] = top
        return top

    def candidates(self, words: Sequence[str], prefix: str = "") -> set[str]:
        cands: set[str] = set()
        for k in self.model.candidate_orders():
            if k == 1:
                cands.update(self._prefix_words(prefix) if prefix else self.by_freq[:UNIGRAM_CANDIDATES])
                continue
            h = self.counts.context(words, k)
            if prefix:
                ws = self.counts.raw[k].data.get(h, {})
                cands.update(w for w in ws if w.startswith(prefix) and w not in SPECIAL)
            else:
                cands.update(self._top(k, h))
        return cands

    # ---- ranking ---------------------------------------------------------------
    def _seen_order(self, word: str, words: Sequence[str]) -> int:
        for k in range(self.model.order, 0, -1):
            if self.counts.raw[k].data.get(self.counts.context(words, k), {}).get(word):
                return k
        return 0

    def predict(self, words: Sequence[str], prefix: str = "", k: int = 3,
                explain: bool = False) -> list[Suggestion]:
        """Top-k next words after the sentence-so-far `words`."""
        ctx = self.counts.context(words, self.model.order)
        uni = self.counts.unigram
        scored = []
        for w in self.candidates(words, prefix):
            s = self.model.score(w, ctx)
            if s > 0:
                scored.append((s, uni.get(w, 0), w))  # ties -> more frequent word
        scored.sort(reverse=True)
        return [Suggestion(w, s, self._seen_order(w, words) if explain else -1)
                for s, _, w in scored[:k]]

    def suggest(self, text: str, k: int = 3, explain: bool = True) -> dict:
        """Suggestions for raw typed text (what the UI calls)."""
        words, prefix = split_for_prediction(text)
        sugg = self.predict(words, prefix, k, explain)
        return {
            "context": words[-(self.model.order - 1):] if self.model.order > 1 else [],
            "prefix": prefix,
            "mode": "complete" if prefix else "next",
            "sentence_start": not words,
            "suggestions": [s.__dict__ for s in sugg],
        }
