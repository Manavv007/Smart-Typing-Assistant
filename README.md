# Smart Typing Assistant (N-gram)

Phone-keyboard style next-word prediction built on Unigram, Bigram and Trigram language models. The counting, smoothing and backoff are implemented from scratch in `smart_typing/ngram.py`.

- Type `I am going to ` and it suggests what comes next (`be`, `the`, `do`).
- Type `I am going to sc` and it completes the current word (`school`, `schedule`, ...).
- The default model is an interpolated Kneser–Ney trigram. Stupid Backoff and MLE+backoff can be selected in the UI.

## Run

```bash
pip install -r requirements.txt
python -m smart_typing.train            # downloads data, counts n-grams -> models/ngram.pkl (~20 s)
python -m app.server                    # open http://127.0.0.1:8000
python -m smart_typing.evaluate --split validation   # compare settings
python -m smart_typing.evaluate --split test         # final numbers -> results/test.md
python -m pytest -q
```

The server trains a model automatically if `models/ngram.pkl` is missing. It is a local demo: it binds to 127.0.0.1 and has no authentication.

## Dataset

The training data is casual dialogue, which makes the suggestions feel like phone typing. Wikipedia text would give formal suggestions instead.

| Dataset | Content | Licence |
|---|---|---|
| [DailyDialog](https://huggingface.co/datasets/roskoN/dailydialog) | ~13k everyday conversations | CC BY-NC-SA 4.0 |
| [Blended Skill Talk](https://huggingface.co/datasets/ParlAI/blended_skill_talk) | ~7k chit-chat dialogues + personas | see dataset card |

- The official train / validation / test splits are kept.
- Size: 2.1M training tokens and 17,968 vocabulary words. Words seen fewer than 2 times become `<unk>`.
- Leakage control:
  - BST personas come from one shared pool (about 80% overlap across splits), so they are used for training only.
  - Eval sentences of 6 or more words that also appear verbatim in train are removed (DailyDialog repeats about 27% of them).

## How it works

1. **Preprocessing** (`text.py`):
   - lowercase the text and split it into sentences on `. ! ?`;
   - keep contractions as single tokens (`don't`, `i'm`) and drop other punctuation;
   - pad each sentence with `<s> <s> … </s>` so n-grams never cross sentence boundaries.
2. **Counting** (`ngram.py`): `C(h, w)` is counted for every context `h` of length 0–2. The continuation counts needed by Kneser–Ney are also counted.
3. **Models**:
   - **MLE**: `P(w|h) = C(h,w)/C(h)`, with or without naive backoff.
   - **Laplace**: add-one smoothing.
   - **Stupid Backoff**: unseen n-grams back off to a shorter context, scaled by 0.4.
   - **Interpolated Kneser–Ney**: absolute discount `D = n1/(n1+2·n2)` and continuation-count lower orders. It is a proper probability distribution, which a unit test checks.
4. **Prediction** (`predictor.py`):
   - candidates come from the trigram, bigram and unigram contexts (prefix-filtered while the user is mid-word);
   - each candidate is scored with the model;
   - the top 3 are returned, and `<s>`, `</s>` and `<unk>` are never suggested.
5. **UI** (`app/`): a FastAPI backend with a phone-mockup frontend. Suggestions update on every keystroke. The UI has a clickable on-screen keyboard, accepts the best suggestion with Tab, and shows a panel of candidates, scores and the longest n-gram that matched.

## Results (test split)

Full table: `results/test.md`. Settings were chosen on validation (`results/validation.md`).

| Model | Top-1 | Top-3 | Keystroke savings | Coverage | Perplexity |
|---|---:|---:|---:|---:|---:|
| Unigram | 4.9% | 10.3% | 41.4% | 100% | 408 |
| Bigram + backoff | 16.5% | 27.3% | 51.7% | 100% | 111 (KN) |
| Trigram, no backoff | 18.6% | 28.6% | 35.8% | 89.2% | n/a |
| **Trigram Kneser–Ney** | **20.5%** | **32.5%** | **53.8%** | 100% | **84.7** |
| 4-gram Kneser–Ney | 21.7% | 33.7% | 54.3% | 100% | 80.1 |

Main findings:

- **More context helps.** Moving from unigram to bigram to trigram raises top-3 accuracy from 10% to 27% to 32%.
- **Backoff is essential.** Without it, about 11% of trigram contexts are unseen and get no suggestion, and keystroke savings fall from 54% to 36%.
- **Laplace smoothing is a poor fit for language modelling.** Its perplexity gets worse with higher order (456 → 1933 → 3358) because it moves too much probability to unseen words. Kneser–Ney improves with order (111 → 85 → 80). Laplace also never changes the ranking within a context, so it doesn't help suggestions.
- **4-gram gives diminishing returns.** It adds about 1 point of accuracy but needs twice as many stored n-grams, so the trigram stays the default. Train with `--order 4` to use it.
- **Speed is real-time.** A prediction takes under 0.5 ms (p95), and the model file is 23 MB.

Keystroke savings rate (KSR) is the share of key presses saved when the user taps a suggestion as soon as the intended word appears. A tap counts as one key press and inserts the trailing space.

## Project layout

```
smart_typing/  text.py  data.py  ngram.py  predictor.py  train.py  evaluate.py
app/           server.py  static/index.html
tests/         test_ngram.py
results/       validation.md/json  test.md/json
```

Saved models are pickle files. Only load ones you created yourself, because unpickling an untrusted file can run arbitrary code.
