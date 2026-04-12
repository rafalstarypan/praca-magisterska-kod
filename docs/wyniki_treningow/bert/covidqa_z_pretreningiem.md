# Wynik treningu: BERT — COVID-QA 5-fold CV (Z pretreningiem SQuAD)

Data: 2026-04-08
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | BERT-base-uncased (google-bert/bert-base-uncased) |
| Parametry (total) | 108,893,186 |
| Parametry (trainable) | 108,893,186 (100% — full fine-tuning) |
| Źródło wag | `results/squad/bert/best_model/` (po pretreningu na SQuAD v1.1) |

## Wyniki per fold

| Fold | Eval EM | Eval F1 | Test EM | Test F1 | Runtime |
|------|---------|---------|---------|---------|---------|
| 0 | 36.63% | 58.67% | 27.72% | 56.31% | 34 min |
| 1 | 26.82% | 52.39% | 31.35% | 56.69% | 34 min |
| 2 | 32.65% | 55.44% | 28.38% | 55.24% | 34 min |
| 3 | 36.73% | 59.04% | 31.68% | 54.81% | 34 min |
| 4 | 31.20% | 60.10% | 30.03% | 54.62% | 34 min |

## Wyniki zagregowane (test set)

| Metryka | Mean | Std |
|---------|------|-----|
| **Exact Match** | **29.83%** | **±1.62** |
| **F1** | **55.53%** | **±0.89** |

## Porównanie: z pretreningiem SQuAD vs bez

| Metryka | Bez SQuAD | Z SQuAD | Poprawa |
|---------|-----------|---------|---------|
| Test EM | 19.54 ± 0.87 | 29.83 ± 1.62 | +10.29pp |
| Test F1 | 39.66 ± 1.52 | 55.53 ± 0.89 | +15.87pp |

## Artefakty

- Wyniki per fold: `results/covidqa/bert/fold_{0-4}/results.json`
- Predykcje: `results/covidqa/bert/fold_{0-4}/{eval,test}_predictions.json`
