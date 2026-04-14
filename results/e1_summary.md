# E1 — Wyniki eksperymentów (podsumowanie)

## 1. Pretrening na SQuAD v1.1

| Model | Params | Eval EM | Eval F1 |
|-------|--------|---------|---------|
| BERT | 108,893,186 | 81.11 | 88.35 |
| BioBERT | 107,721,218 | 81.37 | 88.75 |
| PubMedBERT | 108,893,186 | 82.24 | 89.60 |
| RoBERTa | 124,056,578 | 86.15 | 92.27 |
| DistilBERT | 66,364,418 | 77.41 | 85.55 |
| ClinicalBERT | 107,721,218 | 76.76 | 85.41 |

## 2. BioASQ (factoid)

| Model | Wariant | Eval EM | Eval F1 | Test EM | Test F1 |
|-------|---------|---------|---------|---------|---------|
| BERT | bez SQuAD | 30.14 | 47.21 | 34.15 | 50.24 |
| BERT | z SQuAD | 45.21 | 63.36 | 50.00 | 65.65 |
| BioBERT | bez SQuAD | 41.10 | 59.87 | 42.68 | 61.79 |
| BioBERT | z SQuAD | 56.16 | 74.07 | 45.12 | 69.24 |
| PubMedBERT | bez SQuAD | 42.47 | 64.99 | 53.66 | 71.39 |
| PubMedBERT | z SQuAD | 57.53 | 71.10 | 59.76 | 73.52 |
| RoBERTa | bez SQuAD | 42.47 | 59.64 | 42.68 | 62.30 |
| RoBERTa | z SQuAD | 56.16 | 71.94 | 50.00 | 69.03 |
| DistilBERT | bez SQuAD | 32.88 | 50.02 | 24.39 | 38.91 |
| DistilBERT | z SQuAD | 45.21 | 66.01 | 42.68 | 60.83 |
| ClinicalBERT | bez SQuAD | 35.62 | 50.35 | 35.37 | 50.70 |
| ClinicalBERT | z SQuAD | 45.21 | 65.45 | 46.34 | 65.90 |

### Poprawa dzięki pretreningowi SQuAD (BioASQ)

| Model | Test F1 bez SQuAD | Test F1 z SQuAD | Δ F1 |
|-------|-------------------|-----------------|------|
| BERT | 50.24 | 65.65 | +15.41 |
| BioBERT | 61.79 | 69.24 | +7.45 |
| PubMedBERT | 71.39 | 73.52 | +2.13 |
| RoBERTa | 62.30 | 69.03 | +6.73 |
| DistilBERT | 38.91 | 60.83 | +21.92 |
| ClinicalBERT | 50.70 | 65.90 | +15.20 |

## 3. COVID-QA (5-fold CV)

| Model | Wariant | Eval EM | Eval F1 | Test EM | Test F1 |
|-------|---------|---------|---------|---------|---------|
| BERT | bez SQuAD | 21.85 ± 3.23 | 42.79 ± 3.72 | 19.54 ± 0.85 | 39.66 ± 1.58 |
| BERT | z SQuAD | 32.81 ± 4.14 | 57.13 ± 3.17 | 29.83 ± 1.76 | 55.53 ± 0.92 |
| BioBERT | bez SQuAD | 25.58 ± 1.92 | 47.30 ± 3.55 | 22.90 ± 2.20 | 44.95 ± 1.97 |
| BioBERT | z SQuAD | 34.90 ± 3.34 | 60.80 ± 4.45 | 31.75 ± 1.52 | 57.06 ± 0.71 |
| PubMedBERT | bez SQuAD | 31.12 ± 3.74 | 55.71 ± 4.80 | 32.01 ± 0.81 | 54.66 ± 0.71 |
| PubMedBERT | z SQuAD | 38.69 ± 3.08 | 65.64 ± 3.34 | 37.03 ± 1.00 | 62.81 ± 1.49 |
| RoBERTa | bez SQuAD | 29.37 ± 1.19 | 55.28 ± 2.61 | 29.37 ± 1.48 | 53.43 ± 1.27 |
| RoBERTa | z SQuAD | 37.88 ± 3.70 | 62.81 ± 2.81 | 36.37 ± 1.87 | 61.45 ± 1.17 |
| DistilBERT | bez SQuAD | 18.12 ± 1.46 | 37.86 ± 3.03 | 15.71 ± 3.09 | 34.99 ± 3.24 |
| DistilBERT | z SQuAD | 30.59 ± 2.72 | 55.15 ± 2.84 | 27.06 ± 1.57 | 52.02 ± 1.04 |
| ClinicalBERT | bez SQuAD | 21.62 ± 1.23 | 43.33 ± 3.89 | 19.01 ± 2.56 | 39.44 ± 2.22 |
| ClinicalBERT | z SQuAD | 33.16 ± 3.20 | 58.28 ± 3.47 | 32.74 ± 1.94 | 56.76 ± 1.29 |

### Poprawa dzięki pretreningowi SQuAD (COVID-QA)

| Model | Test F1 bez SQuAD | Test F1 z SQuAD | Δ F1 |
|-------|-------------------|-----------------|------|
| BERT | 39.66 ± 1.58 | 55.53 ± 0.92 | +15.87 |
| BioBERT | 44.95 ± 1.97 | 57.06 ± 0.71 | +12.11 |
| PubMedBERT | 54.66 ± 0.71 | 62.81 ± 1.49 | +8.16 |
| RoBERTa | 53.43 ± 1.27 | 61.45 ± 1.17 | +8.02 |
| DistilBERT | 34.99 ± 3.24 | 52.02 ± 1.04 | +17.03 |
| ClinicalBERT | 39.44 ± 2.22 | 56.76 ± 1.29 | +17.31 |

## 4. Ranking końcowy (Test F1, z pretreningiem SQuAD)

| # | Model | BioASQ F1 | COVID-QA F1 | Średnia F1 |
|---|-------|-----------|-------------|------------|
| 1 | PubMedBERT | 73.52 | 62.81 | 68.17 |
| 2 | RoBERTa | 69.03 | 61.45 | 65.24 |
| 3 | BioBERT | 69.24 | 57.06 | 63.15 |
| 4 | ClinicalBERT | 65.90 | 56.76 | 61.33 |
| 5 | BERT | 65.65 | 55.53 | 60.59 |
| 6 | DistilBERT | 60.83 | 52.02 | 56.42 |
