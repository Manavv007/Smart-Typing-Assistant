"""Evaluate model variants on held-out text.

    python -m smart_typing.evaluate --split validation   # choose settings
    python -m smart_typing.evaluate --split test         # final numbers

Metrics
  top-1 / top-3 accuracy  next word is the 1st / among the 3 suggestions
  MRR@3                   mean reciprocal rank of the true word (0 if not shown)
  coverage                % of positions where the model suggested anything
  KSR                     keystroke savings rate: fraction of keystrokes saved
                          by a user who taps a suggestion as soon as the
                          intended word appears (tap = 1 key, incl. space)
  perplexity              only for normalized models (Laplace, Kneser-Ney)
  latency                 mean / p95 time per prediction call
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

from .data import load_split
from .ngram import KneserNey, Laplace, LanguageModel, MLE, NGramCounts, StupidBackoff
from .predictor import Predictor
from .text import EOS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def next_word_metrics(pred: Predictor, sents, k: int = 3) -> dict:
    top1 = topk = rr = covered = n = 0
    times = []
    for s in sents:
        for i, target in enumerate(s):
            t0 = time.perf_counter()
            sugg = [x.word for x in pred.predict(s[:i], "", k)]
            times.append(time.perf_counter() - t0)
            n += 1
            covered += bool(sugg)
            if target in sugg:
                r = sugg.index(target) + 1
                top1 += r == 1
                topk += 1
                rr += 1 / r
    times.sort()
    return {
        "top1": top1 / n, f"top{k}": topk / n, f"mrr@{k}": rr / n,
        "coverage": covered / n, "positions": n,
        "latency_ms_mean": 1000 * statistics.fmean(times),
        "latency_ms_p95": 1000 * times[int(0.95 * (len(times) - 1))],
    }


def keystroke_savings(pred: Predictor, sents, k: int = 3) -> float:
    base = cost = 0
    for s in sents:
        for i, w in enumerate(s):
            base += len(w) + 1  # letters + space
            for j in range(len(w) + 1):
                if w in (x.word for x in pred.predict(s[:i], w[:j], k)):
                    cost += 1  # tap the suggestion (inserts word + space)
                    break
                cost += 1  # type the next letter, or the space after the word
    return 1 - cost / base


def perplexity(model: LanguageModel, sents) -> float:
    counts = model.counts
    log_sum, n = 0.0, 0
    for s in sents:
        toks = counts.map_unk(s) + [EOS]
        for i, w in enumerate(toks):
            p = model.score(w, counts.context(s[:i], model.order))
            log_sum += math.log(p)
            n += 1
    return math.exp(-log_sum / n)


def ngram_entries(counts: NGramCounts, order: int) -> int:
    return sum(sum(counts.raw[k].types.values()) for k in range(1, order + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--max-order", type=int, default=4)
    ap.add_argument("--sentences", type=int, default=4000, help="sample for accuracy metrics")
    ap.add_argument("--ksr-sentences", type=int, default=600, help="sample for KSR (slow)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    train = load_split("train")
    evals = load_split(a.split)
    print(f"train {len(train):,} sentences | {a.split} {len(evals):,} sentences")
    counts = NGramCounts(train, max_order=a.max_order)
    oov = sum(w not in counts.vocab for s in evals for w in s) / sum(map(len, evals))
    print(f"vocab {counts.vocab_size:,}, {a.split} OOV rate {oov:.2%}")

    rng = random.Random(a.seed)
    sample = rng.sample(evals, min(a.sentences, len(evals)))
    ksr_sample = sample[: a.ksr_sentences]

    ranking = [
        ("Unigram", MLE(counts, 1)),
        ("Bigram, no backoff", MLE(counts, 2, backoff=False)),
        ("Trigram, no backoff", MLE(counts, 3, backoff=False)),
        ("Bigram + backoff", MLE(counts, 2)),
        ("Trigram + backoff", MLE(counts, 3)),
        ("Trigram Stupid Backoff", StupidBackoff(counts, 3)),
        ("Bigram Kneser-Ney", KneserNey(counts, 2)),
        ("Trigram Kneser-Ney", KneserNey(counts, 3)),
    ]
    if a.max_order >= 4:
        ranking += [("4-gram Stupid Backoff", StupidBackoff(counts, 4)),
                    ("4-gram Kneser-Ney", KneserNey(counts, 4))]

    results = {"split": a.split, "oov_rate": oov, "vocab_size": counts.vocab_size,
               "train_tokens": counts.n_tokens, "ranking": [], "perplexity": []}
    for name, model in ranking:
        t0 = time.perf_counter()
        pred = Predictor(model)
        r = next_word_metrics(pred, sample)
        r["ksr"] = keystroke_savings(pred, ksr_sample)
        r.update(name=name, ngrams=ngram_entries(counts, model.order))
        results["ranking"].append(r)
        print(f"{name:24s} top1 {r['top1']:.3f}  top3 {r['top3']:.3f}  mrr {r['mrr@3']:.3f}  "
              f"cov {r['coverage']:.3f}  ksr {r['ksr']:.3f}  "
              f"{r['latency_ms_mean']:.2f}ms  ({time.perf_counter() - t0:.0f}s)")

    for order in range(1, a.max_order + 1):
        for cls in (Laplace, KneserNey):
            m = cls(counts, order)
            ppl = perplexity(m, evals)
            results["perplexity"].append({"model": m.name, "order": order, "perplexity": ppl})
            print(f"perplexity {m.name:10s} n={order}: {ppl:8.1f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{a.split}.json").write_text(json.dumps(results, indent=2))
    (RESULTS_DIR / f"{a.split}.md").write_text(to_markdown(results, len(sample), len(ksr_sample)))
    print(f"wrote {RESULTS_DIR / (a.split + '.md')}")


def to_markdown(r: dict, n_sample: int, n_ksr: int) -> str:
    lines = [
        f"# Evaluation on `{r['split']}`", "",
        f"Train tokens: {r['train_tokens']:,} | vocabulary: {r['vocab_size']:,} | "
        f"OOV rate: {r['oov_rate']:.2%}", "",
        f"Next-word metrics on {n_sample:,} random sentences; KSR on {n_ksr:,} of them.", "",
        "| Model | Top-1 | Top-3 | MRR@3 | Coverage | KSR | Latency mean (ms) | p95 (ms) | Stored n-grams |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in r["ranking"]:
        lines.append(f"| {x['name']} | {x['top1']:.1%} | {x['top3']:.1%} | {x['mrr@3']:.3f} | "
                     f"{x['coverage']:.1%} | {x['ksr']:.1%} | {x['latency_ms_mean']:.2f} | "
                     f"{x['latency_ms_p95']:.2f} | {x['ngrams']:,} |")
    lines += ["", "## Perplexity (lower is better, full split)", "",
              "| Order | Laplace | Kneser-Ney |", "|---:|---:|---:|"]
    by = {(p["model"], p["order"]): p["perplexity"] for p in r["perplexity"]}
    for o in sorted({p["order"] for p in r["perplexity"]}):
        lines.append(f"| {o} | {by[('Laplace', o)]:.1f} | {by[('Kneser-Ney', o)]:.1f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
