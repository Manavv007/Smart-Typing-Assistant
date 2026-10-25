import math

import pytest

from smart_typing.evaluate import keystroke_savings, perplexity
from smart_typing.ngram import MLE, KneserNey, Laplace, NGramCounts, StupidBackoff
from smart_typing.predictor import Predictor
from smart_typing.text import BOS, EOS, UNK, sentences, split_for_prediction

CORPUS = [
    "I am going to school.",
    "I am going to college.",
    "I am going to work.",
    "I am going to school.",
    "She is going home. We are going to the park!",
    "I don ' t know. I can’t stay.",
]


@pytest.fixture(scope="module")
def counts():
    sents = [s for line in CORPUS for s in sentences(line)]
    return NGramCounts(sents, max_order=3, min_count=1)


# ---- text ------------------------------------------------------------------
def test_sentences_split_and_contractions():
    assert sentences("I don ' t know. I can’t stay!") == [["i", "don't", "know"], ["i", "can't", "stay"]]
    assert sentences("Hi , there ? ok") == [["hi", "there"], ["ok"]]


@pytest.mark.parametrize("text, expected", [
    ("I am going to", (["i", "am", "going"], "to")),
    ("I am going to ", (["i", "am", "going", "to"], "")),
    ("Thanks. ", ([], "")),
    ("Thanks. I", ([], "i")),
    ("", ([], "")),
    ("I don't", (["i"], "don't")),
])
def test_split_for_prediction(text, expected):
    assert split_for_prediction(text) == expected


# ---- counts ----------------------------------------------------------------
def test_counts_match_worked_example(counts):
    """The 'going to' example from the project spec: 2/4, 1/4, 1/4."""
    after = counts.raw[3].data[("going", "to")]
    assert after == {"school": 2, "college": 1, "work": 1, "the": 1}
    mle = MLE(counts, 3, backoff=False)
    assert mle.score("school", ("going", "to")) == pytest.approx(3 + 2 / 5)


def test_sentence_boundaries_not_crossed(counts):
    # "home" ends a sentence; "home we" must not be a bigram
    assert "we" not in counts.raw[2].data.get(("home",), {})
    assert counts.raw[2].data[("home",)] == {EOS: 1}
    assert counts.raw[3].data[(BOS, BOS)]["i"] >= 5


def test_unk_mapping():
    c = NGramCounts([["a", "b"], ["a", "c"]], max_order=2, min_count=2)
    assert c.vocab == {"a"}
    assert c.raw[2].data[("a",)] == {UNK: 2}
    assert c.context(["zzz"], 3) == (BOS, UNK)


# ---- models ----------------------------------------------------------------
@pytest.mark.parametrize("cls", [Laplace, KneserNey])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("ctx_words", [[], ["going", "to"], ["unseen", "words"], ["i"]])
def test_normalized_models_sum_to_one(counts, cls, order, ctx_words):
    m = cls(counts, order)
    ctx = counts.context(ctx_words, order)
    vocab = list(counts.vocab) + [EOS, UNK]
    assert sum(m.score(w, ctx) for w in vocab) == pytest.approx(1.0)


def test_kneser_ney_gives_unseen_nonzero(counts):
    m = KneserNey(counts, 3)
    assert m.score("park", ("going", "to")) > 0          # seen elsewhere, not here
    assert m.score("school", ("going", "to")) > m.score("park", ("going", "to"))


def test_stupid_backoff_penalizes_shorter_context(counts):
    sb = StupidBackoff(counts, 3)
    assert sb.score("school", ("going", "to")) == pytest.approx(2 / 5)
    # "home" never follows "am going" but follows "going" (1 of 6 times)
    assert sb.score("home", ("am", "going")) == pytest.approx(0.4 * 1 / 6)


def test_perplexity_kn_beats_laplace(counts):
    sents = [["i", "am", "going", "to", "work"]]
    assert perplexity(KneserNey(counts, 3), sents) < perplexity(Laplace(counts, 3), sents)
    assert math.isfinite(perplexity(KneserNey(counts, 3), [["never", "seen"]]))


# ---- predictor ---------------------------------------------------------------
def test_top3_after_going_to(counts):
    p = Predictor(KneserNey(counts, 3))
    words = [s.word for s in p.predict(["i", "am", "going", "to"])]
    # tiny corpus -> large KN discount, so only check the clear winner
    assert words[0] == "school" and {"college", "work"} & set(words)


def test_completion_uses_prefix(counts):
    p = Predictor(KneserNey(counts, 3))
    out = p.suggest("I am going to co")
    assert out["mode"] == "complete"
    assert [s["word"] for s in out["suggestions"]][0] == "college"


def test_never_suggests_special_tokens(counts):
    p = Predictor(KneserNey(counts, 3))
    for text in ["", "I am going home ", "zzz qqq "]:
        assert not {s["word"] for s in p.suggest(text)["suggestions"]} & {BOS, EOS, UNK}


def test_no_backoff_has_no_suggestions_for_unseen_context(counts):
    p = Predictor(MLE(counts, 3, backoff=False))
    assert p.predict(["zzz", "qqq"]) == []
    assert Predictor(MLE(counts, 3)).predict(["zzz", "qqq"])  # backoff fills in


def test_keystroke_savings(counts):
    p = Predictor(KneserNey(counts, 3))
    ksr = keystroke_savings(p, [["i", "am", "going", "to", "school"]])
    assert 0.5 < ksr < 1
