"""Train (= count) the n-gram model and save it.

    python -m smart_typing.train                # trigram, Kneser-Ney (default)
    python -m smart_typing.train --order 4 --model sb
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

from .data import load_split
from .ngram import MODELS, NGramCounts, build_model

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_PATH = MODEL_DIR / "ngram.pkl"


def train(order: int = 3, min_count: int = 2) -> NGramCounts:
    sents = load_split("train")
    t0 = time.perf_counter()
    counts = NGramCounts(sents, max_order=order, min_count=min_count)
    print(f"{len(sents):,} sentences, {counts.n_tokens:,} tokens, "
          f"vocab {counts.vocab_size:,}, counted in {time.perf_counter() - t0:.1f}s")
    for k in range(1, order + 1):
        n = sum(counts.raw[k].types.values())
        print(f"  {k}-grams: {n:,} distinct")
    return counts


def save(counts: NGramCounts, kind: str, path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"counts": counts, "kind": kind, "order": counts.max_order}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load(path: Path = DEFAULT_PATH):
    """Load a saved model. Only load pickle files you created yourself:
    unpickling untrusted files can execute arbitrary code."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return build_model(obj["counts"], obj["kind"], obj["order"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--model", choices=sorted(MODELS), default="kn")
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    a = ap.parse_args()
    save(train(a.order, a.min_count), a.model, a.out)


if __name__ == "__main__":
    main()
