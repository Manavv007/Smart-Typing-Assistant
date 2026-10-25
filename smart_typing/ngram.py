"""N-gram counting and language models, implemented from scratch.

Counts
------
For every order k (1..max_order) we store, per context h (the k-1 previous
words), how often each next word w followed it:  raw[k][h][w] = C(h, w).
Sentences are padded with k-1 <s> tokens and one </s> token, so the model
never learns word pairs that cross a sentence boundary.

Models
------
MLE            P(w|h) = C(h,w) / C(h). Optional naive backoff to shorter
               contexts. No smoothing -> unseen sequences get 0.
Laplace        add-one smoothing at the top order. Normalized, so it has a
               perplexity, but it moves far too much mass to unseen words.
StupidBackoff  Brants et al. (2007). Ranking score, not a probability:
               S(w|h) = C(h,w)/C(h) if seen else 0.4 * S(w|shorter h).
KneserNey      interpolated Kneser-Ney (Chen & Goodman 1999). Proper
               probability distribution; lower orders use *continuation*
               counts ("in how many different contexts does w appear").
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence

from .text import BOS, EOS, UNK

Context = tuple[str, ...]


class Table:
    """context -> {word: count}, with cached totals and type counts."""

    __slots__ = ("data", "total", "types")

    def __init__(self, data: dict[Context, dict[str, int]]):
        self.data = data
        self.total = {h: sum(ws.values()) for h, ws in data.items()}
        self.types = {h: len(ws) for h, ws in data.items()}

    def discount(self) -> float:
        """Absolute discount D = n1 / (n1 + 2 n2) (Ney et al. 1994)."""
        coc = Counter(c for ws in self.data.values() for c in ws.values())
        n1, n2 = coc.get(1, 0), coc.get(2, 0)
        return n1 / (n1 + 2 * n2) if n1 + n2 else 0.75


class NGramCounts:
    def __init__(self, sentences: Iterable[Sequence[str]], max_order: int = 3,
                 min_count: int = 2):
        sentences = list(sentences)
        freq = Counter(w for s in sentences for w in s)
        self.vocab = {w for w, c in freq.items() if c >= min_count}
        self.max_order = max_order
        self.min_count = min_count
        # vocabulary the model predicts over: words + </s> + <unk> (never <s>)
        self.vocab_size = len(self.vocab) + 2

        raw: dict[int, dict] = {k: defaultdict(Counter) for k in range(1, max_order + 1)}
        for s in sentences:
            toks = self.map_unk(s) + [EOS]
            for k in range(1, max_order + 1):
                padded = [BOS] * (k - 1) + toks
                table = raw[k]
                for i in range(k - 1, len(padded)):
                    table[tuple(padded[i - k + 1:i])][padded[i]] += 1
        self.raw = {k: Table({h: dict(ws) for h, ws in t.items()}) for k, t in raw.items()}

        # Continuation counts for Kneser-Ney lower orders:
        # cont[k][h][w] = number of distinct words v with C(v, h, w) > 0
        self.cont: dict[int, Table] = {}
        for k in range(1, max_order):
            c: dict = defaultdict(Counter)
            for h, ws in self.raw[k + 1].data.items():
                shorter = h[1:]
                for w in ws:
                    c[shorter][w] += 1
            self.cont[k] = Table({h: dict(ws) for h, ws in c.items()})

        self.unigram = self.raw[1].data[()]
        self.n_tokens = self.raw[1].total[()]

    def map_unk(self, words: Sequence[str]) -> list[str]:
        return [w if w in self.vocab else UNK for w in words]

    def context(self, words: Sequence[str], order: int) -> Context:
        """Last order-1 tokens of the <s>-padded, <unk>-mapped sentence."""
        if order <= 1:
            return ()
        padded = [BOS] * (order - 1) + self.map_unk(words)
        return tuple(padded[len(padded) - (order - 1):])


class LanguageModel:
    name = "base"
    normalized = False  # True if score() is a real probability

    def __init__(self, counts: NGramCounts, order: int):
        if not 1 <= order <= counts.max_order:
            raise ValueError(f"order must be in 1..{counts.max_order}")
        self.counts = counts
        self.order = order

    def score(self, word: str, ctx: Context) -> float:
        """Ranking score for word after ctx (ctx has length order-1)."""
        raise NotImplementedError

    def candidate_orders(self) -> range:
        """Which n-gram orders may propose candidate words."""
        return range(self.order, 0, -1)

    @property
    def label(self) -> str:
        return f"{self.name} (n={self.order})"


class MLE(LanguageModel):
    name = "MLE"

    def __init__(self, counts, order, backoff=True):
        super().__init__(counts, order)
        self.backoff = backoff
        if not backoff:
            self.name = "MLE no-backoff"

    def candidate_orders(self):
        return super().candidate_orders() if self.backoff else range(self.order, self.order - 1, -1)

    def score(self, word, ctx):
        # Encodes "longest matching context wins": order k + relative freq.
        lowest = 1 if self.backoff else self.order
        for k in range(self.order, lowest - 1, -1):
            t = self.counts.raw[k]
            h = ctx[len(ctx) - (k - 1):] if k > 1 else ()
            c = t.data.get(h, {}).get(word, 0)
            if c:
                return k + c / t.total[h]
        return 0.0


class Laplace(LanguageModel):
    name = "Laplace"
    normalized = True

    def score(self, word, ctx):
        t = self.counts.raw[self.order]
        ws = t.data.get(ctx)
        c = ws.get(word, 0) if ws else 0
        total = t.total.get(ctx, 0)
        return (c + 1) / (total + self.counts.vocab_size)


class StupidBackoff(LanguageModel):
    name = "Stupid Backoff"

    def __init__(self, counts, order, alpha=0.4):
        super().__init__(counts, order)
        self.alpha = alpha

    def score(self, word, ctx):
        weight = 1.0
        for k in range(self.order, 0, -1):
            t = self.counts.raw[k]
            h = ctx[len(ctx) - (k - 1):] if k > 1 else ()
            c = t.data.get(h, {}).get(word, 0)
            if c:
                return weight * c / t.total[h]
            weight *= self.alpha
        return 0.0


class KneserNey(LanguageModel):
    name = "Kneser-Ney"
    normalized = True

    def __init__(self, counts, order):
        super().__init__(counts, order)
        # top order uses raw counts, lower orders use continuation counts
        self.tables = {k: counts.raw[k] if k == order else counts.cont[k]
                       for k in range(1, order + 1)}
        self.D = {k: t.discount() for k, t in self.tables.items()}

    def score(self, word, ctx):
        # Bottom-up interpolation: start from uniform, add each order.
        p = 1.0 / self.counts.vocab_size
        for k in range(1, self.order + 1):
            t = self.tables[k]
            h = ctx[len(ctx) - (k - 1):] if k > 1 else ()
            ws = t.data.get(h)
            if not ws:
                continue  # unseen context: P_k = P_{k-1}
            total, d = t.total[h], self.D[k]
            p = max(ws.get(word, 0) - d, 0.0) / total + d * t.types[h] / total * p
        return p


MODELS = {"mle": MLE, "laplace": Laplace, "sb": StupidBackoff, "kn": KneserNey}


def build_model(counts: NGramCounts, kind: str = "kn", order: int | None = None,
                **kwargs) -> LanguageModel:
    return MODELS[kind](counts, order or counts.max_order, **kwargs)
