"""Text normalization and tokenization.

The same rules are used for training data and for whatever the user types,
so the model always sees text in one consistent form.

Design choices (casual typing):
  * lowercase everything (display capitalization is handled by the UI),
  * keep contractions as single tokens ("don't", "i'm") like a phone keyboard,
  * sentence boundaries are . ! ? and newlines; other punctuation is dropped.
"""
from __future__ import annotations

import re

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"
SPECIAL = {BOS, EOS, UNK}

_APOSTROPHES = re.compile(r"[\u2019\u2018`\u00b4]")
# DailyDialog often writes "I ' m" / "don ’ t" -> re-attach common clitics.
_SPLIT_CLITIC = re.compile(r"(\w)\s*'\s*(m|s|t|ll|ve|re|d)\b")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.!?\n]")
_SENT_END = {".", "!", "?", "\n"}
# A word the user is still typing at the end of the text.
_PARTIAL = re.compile(r"[a-z0-9']+$")


def normalize(text: str) -> str:
    text = _APOSTROPHES.sub("'", text.lower())
    return _SPLIT_CLITIC.sub(r"\1'\2", text)


def sentences(text: str) -> list[list[str]]:
    """Split text into tokenized sentences (lists of word tokens)."""
    out: list[list[str]] = []
    cur: list[str] = []
    for tok in _TOKEN.findall(normalize(text)):
        if tok in _SENT_END:
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(tok.strip("'"))
    if cur:
        out.append(cur)
    return [[t for t in s if t] for s in out if any(s)]


def split_for_prediction(text: str) -> tuple[list[str], str]:
    """Return (context words of the current sentence, partial word prefix).

    "I am going to"   -> (["i", "am", "going"], "to")   # still typing "to"
    "I am going to "  -> (["i", "am", "going", "to"], "")
    "Thanks. "        -> ([], "")                         # new sentence
    """
    norm = normalize(text)
    prefix = ""
    if norm and not norm[-1].isspace():
        m = _PARTIAL.search(norm)
        if m:
            prefix = m.group(0)
            norm = norm[: m.start()]
    tokens = _TOKEN.findall(norm)
    # keep only the words after the last sentence boundary
    last = max((i for i, t in enumerate(tokens) if t in _SENT_END), default=-1)
    ctx = [t.strip("'") for t in tokens[last + 1:]]
    return [t for t in ctx if t], prefix.lstrip("'")
