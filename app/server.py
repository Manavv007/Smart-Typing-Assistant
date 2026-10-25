"""FastAPI backend for the typing demo.

    python -m app.server            # http://127.0.0.1:8000

Local demo only: binds to 127.0.0.1 and has no authentication.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from smart_typing.ngram import build_model
from smart_typing.predictor import Predictor
from smart_typing.train import DEFAULT_PATH, load, save, train

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
KINDS = {"kn": "Kneser-Ney", "sb": "Stupid Backoff", "mle": "MLE + backoff"}

if not DEFAULT_PATH.exists():
    print("No trained model found - training one now (first run only)...")
    save(train(order=3), "kn")
_default = load(DEFAULT_PATH)
COUNTS = _default.counts

app = FastAPI(title="Smart Typing Assistant")


@lru_cache(maxsize=None)
def predictor(kind: str, order: int) -> Predictor:
    return Predictor(build_model(COUNTS, kind, order))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    return {
        "default": {"kind": _default.name, "order": _default.order},
        "max_order": COUNTS.max_order,
        "kinds": KINDS,
        "vocab_size": COUNTS.vocab_size,
        "train_tokens": COUNTS.n_tokens,
    }


@app.get("/api/suggest")
def suggest(text: str = Query("", max_length=5000), kind: str = "kn",
            order: int | None = None, k: int = Query(3, ge=1, le=10)):
    order = order or COUNTS.max_order
    if kind not in KINDS or not 1 <= order <= COUNTS.max_order:
        raise HTTPException(400, "unknown model kind or order")
    t0 = time.perf_counter()
    out = predictor(kind, order).suggest(text, k=k)
    out["latency_ms"] = round(1000 * (time.perf_counter() - t0), 3)
    return out


@app.get("/api/results")
def results():
    out = {}
    for split in ("validation", "test"):
        p = ROOT / "results" / f"{split}.json"
        if p.exists():
            out[split] = json.loads(p.read_text())
    return out


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
