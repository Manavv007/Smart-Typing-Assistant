"""Dataset download and loading.

Corpus = two casual, conversational English datasets (better than Wikipedia
for a phone-keyboard feel):

  * DailyDialog        - ~13k everyday-life dialogues (CC BY-NC-SA 4.0)
  * Blended Skill Talk - ~7k chit-chat dialogues + personas (ParlAI;
                         see its dataset card for terms)

Both come with official train / validation / test splits, which we keep.
Files are the Hugging Face auto-converted parquet exports.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from .text import sentences

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SPLITS = ("train", "validation", "test")

_HF = "https://huggingface.co/datasets"
SOURCES = {
    "dd": f"{_HF}/roskoN/dailydialog/resolve/refs%2Fconvert%2Fparquet/full/{{split}}/0000.parquet",
    "bst": f"{_HF}/ParlAI/blended_skill_talk/resolve/refs%2Fconvert%2Fparquet/default/{{split}}/0000.parquet",
}


def download(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        for split in SPLITS:
            path = RAW_DIR / f"{name}_{split}.parquet"
            if path.exists() and not force:
                continue
            print(f"downloading {name}/{split} ...")
            resp = requests.get(url.format(split=split), timeout=120)
            resp.raise_for_status()
            path.write_bytes(resp.content)


def _dialogues(split: str) -> list[list[str]]:
    """Each dialogue as a list of utterance strings."""
    dd = pd.read_parquet(RAW_DIR / f"dd_{split}.parquet")
    dialogues = [list(u) for u in dd["utterances"]]

    bst = pd.read_parquet(RAW_DIR / f"bst_{split}.parquet")
    for row in bst.itertuples(index=False):
        # Personas are short first-person facts ("i love flowers.") - great
        # for casual "I am / I like / I have" predictions. They come from one
        # shared pool across splits (~80% overlap), so they are train-only.
        if split == "train":
            dialogues.append(list(row.personas))
        turns = list(row.previous_utterance)
        for a, b in zip(row.free_messages, row.guided_messages):
            turns += [a, b]
        dialogues.append(turns)
    return dialogues


# Eval sentences at least this long that also occur verbatim in train are
# treated as leaked duplicates (DailyDialog repeats ~27% of them). Shorter
# ones ("thank you") are naturally common and are kept.
DEDUP_MIN_WORDS = 6


def load_split(split: str) -> list[list[str]]:
    """Tokenized sentences for a split. Eval splits are de-duplicated against
    train so that scores are not inflated by memorized sentences."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    download()
    sents = [s for d in _dialogues(split) for utt in d for s in sentences(utt)]
    if split != "train":
        train = {tuple(s) for s in load_split("train") if len(s) >= DEDUP_MIN_WORDS}
        sents = [s for s in sents if len(s) < DEDUP_MIN_WORDS or tuple(s) not in train]
    return sents
