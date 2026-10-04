| Retrieval mode | Recall@4 | Complete recall@4 | MRR@4 | Recall@1 | Recall@8 | Missed at k=4 |
|---|---|---|---|---|---|---|
| dense | 0.825 | 0.775 | 0.613 | 0.450 | 0.900 | q01, q09, q10, q16, q22, p07, p08 |
| bm25 | 0.850 | 0.800 | 0.756 | 0.675 | 0.875 | p01, p02, p03, p04, p07, m02 |
| hybrid | 0.825 | 0.800 | 0.665 | 0.550 | 0.900 | q01, q09, q22, p02, p03, p04, p07 |
| hybrid_rerank | 0.900 | 0.875 | 0.781 | 0.675 | 0.975 | p02, p06, p08, m02 |

| Retrieval mode | lookup (Recall@4 / MRR@4) | multi_step (Recall@4 / MRR@4) | paraphrase (Recall@4 / MRR@4) |
|---|---|---|---|
| dense | 0.77 / 0.59 | 1.00 / 0.71 | 0.75 / 0.54 |
| bm25 | 1.00 / 0.91 | 0.90 / 0.90 | 0.38 / 0.16 |
| hybrid | 0.86 / 0.76 | 1.00 / 0.83 | 0.50 / 0.20 |
| hybrid_rerank | 1.00 / 0.91 | 0.90 / 0.78 | 0.62 / 0.44 |
