# Evaluation on `test`

Train tokens: 2,148,960 | vocabulary: 17,968 | OOV rate: 1.38%

Next-word metrics on 4,000 random sentences; KSR on 600 of them.

| Model | Top-1 | Top-3 | MRR@3 | Coverage | KSR | Latency mean (ms) | p95 (ms) | Stored n-grams |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unigram | 4.9% | 10.3% | 0.071 | 100.0% | 41.4% | 0.04 | 0.06 | 17,968 |
| Bigram, no backoff | 16.5% | 27.2% | 0.211 | 100.0% | 46.3% | 0.06 | 0.12 | 342,727 |
| Trigram, no backoff | 18.6% | 28.6% | 0.229 | 89.2% | 35.8% | 0.05 | 0.10 | 1,160,752 |
| Bigram + backoff | 16.5% | 27.3% | 0.211 | 100.0% | 51.7% | 0.10 | 0.17 | 342,727 |
| Trigram + backoff | 20.2% | 32.1% | 0.254 | 100.0% | 53.5% | 0.15 | 0.26 | 1,160,752 |
| Trigram Stupid Backoff | 20.2% | 32.0% | 0.253 | 100.0% | 53.3% | 0.15 | 0.26 | 1,160,752 |
| Bigram Kneser-Ney | 16.4% | 27.2% | 0.211 | 100.0% | 51.8% | 0.15 | 0.27 | 342,727 |
| Trigram Kneser-Ney | 20.5% | 32.5% | 0.257 | 100.0% | 53.8% | 0.28 | 0.52 | 1,160,752 |
| 4-gram Stupid Backoff | 21.2% | 33.1% | 0.263 | 100.0% | 53.8% | 0.20 | 0.34 | 2,271,157 |
| 4-gram Kneser-Ney | 21.7% | 33.7% | 0.269 | 100.0% | 54.3% | 0.36 | 0.74 | 2,271,157 |

## Perplexity (lower is better, full split)

| Order | Laplace | Kneser-Ney |
|---:|---:|---:|
| 1 | 408.8 | 408.3 |
| 2 | 456.2 | 111.2 |
| 3 | 1932.5 | 84.7 |
| 4 | 3357.5 | 80.1 |
