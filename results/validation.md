# Evaluation on `validation`

Train tokens: 2,148,960 | vocabulary: 17,968 | OOV rate: 1.32%

Next-word metrics on 4,000 random sentences; KSR on 600 of them.

| Model | Top-1 | Top-3 | MRR@3 | Coverage | KSR | Latency mean (ms) | p95 (ms) | Stored n-grams |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unigram | 4.9% | 10.6% | 0.073 | 100.0% | 41.2% | 0.03 | 0.03 | 17,968 |
| Bigram, no backoff | 16.5% | 27.8% | 0.214 | 99.9% | 47.5% | 0.04 | 0.06 | 342,727 |
| Trigram, no backoff | 18.8% | 29.4% | 0.234 | 89.5% | 36.8% | 0.03 | 0.06 | 1,160,752 |
| Bigram + backoff | 16.5% | 27.8% | 0.214 | 100.0% | 52.4% | 0.07 | 0.09 | 342,727 |
| Trigram + backoff | 20.4% | 33.0% | 0.259 | 100.0% | 54.3% | 0.10 | 0.15 | 1,160,752 |
| Trigram Stupid Backoff | 20.3% | 32.8% | 0.258 | 100.0% | 54.2% | 0.10 | 0.15 | 1,160,752 |
| Bigram Kneser-Ney | 16.5% | 27.9% | 0.214 | 100.0% | 52.5% | 0.11 | 0.14 | 342,727 |
| Trigram Kneser-Ney | 20.6% | 33.4% | 0.262 | 100.0% | 54.7% | 0.18 | 0.27 | 1,160,752 |
| 4-gram Stupid Backoff | 21.2% | 33.7% | 0.267 | 100.0% | 54.6% | 0.13 | 0.20 | 2,271,157 |
| 4-gram Kneser-Ney | 21.8% | 34.6% | 0.274 | 100.0% | 55.2% | 0.23 | 0.36 | 2,271,157 |

## Perplexity (lower is better, full split)

| Order | Laplace | Kneser-Ney |
|---:|---:|---:|
| 1 | 409.1 | 408.6 |
| 2 | 457.9 | 111.1 |
| 3 | 1939.8 | 83.8 |
| 4 | 3383.3 | 79.4 |
