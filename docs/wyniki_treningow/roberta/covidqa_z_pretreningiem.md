# Wynik treningu: RoBERTa — COVID-QA 5-fold CV (Z pretreningiem SQuAD)

Data: 2026-04-10
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | RoBERTa-base (FacebookAI/roberta-base) |
| Parametry (total) | 124,056,578 |
| Parametry (trainable) | 124,056,578 (100% — full fine-tuning) |
| Źródło wag | `results/squad/roberta/best_model/` (po pretreningu na SQuAD v1.1) |

## Wyniki per fold

| Fold | Test EM | Test F1 | Peak VRAM | Czas |
|------|---------|---------|-----------|------|
| 0 | 37.62% | 62.99% | 4080 MB | 34 min |
| 1 | 37.95% | 62.08% | 4080 MB | 34 min |
| 2 | 34.32% | 59.90% | 4080 MB | 34 min |
| 3 | 34.32% | 60.90% | 4080 MB | 34 min |
| 4 | 37.62% | 61.38% | 4080 MB | 34 min |

## Wyniki zagregowane (test set)

| Metryka | Mean | Std |
|---------|------|-----|
| **Exact Match** | **36.37%** | **±1.68** |
| **F1** | **61.45%** | **±1.21** |

## Porównanie: z pretreningiem SQuAD vs bez

| Metryka | Bez SQuAD | Z SQuAD | Poprawa |
|---------|-----------|---------|---------|
| Test EM | 29.37 ± 1.42 | 36.37 ± 1.68 | +7.00pp |
| Test F1 | 53.43 ± 1.11 | 61.45 ± 1.21 | +8.02pp |

## Artefakty

- Wyniki per fold: `results/covidqa/roberta/fold_{0-4}/results.json`
- Predykcje: `results/covidqa/roberta/fold_{0-4}/{eval,test}_predictions.json`
