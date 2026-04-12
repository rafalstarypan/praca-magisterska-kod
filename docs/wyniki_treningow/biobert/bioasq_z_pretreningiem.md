# Wynik treningu: BioBERT — BioASQ (Z pretreningiem SQuAD)

Data: 2026-04-06
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 72 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | BioBERT v1.1 (dmis-lab/biobert-v1.1) |
| Parametry (total) | 107,721,218 |
| Parametry (trainable) | 107,721,218 (100% — full fine-tuning) |
| Źródło wag | `results/squad/biobert/best_model/` (po pretreningu na SQuAD v1.1) |

## Wyniki

| Zbiór | Exact Match | F1 |
|-------|-------------|-----|
| Dev | 56.16% | 74.07% |
| **Test** | **45.12%** | **69.24%** |

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| train_runtime | 72.0 s |
| train_loss (średni) | 0.704 |
| train_samples_per_second | 50.87 |

## Porównanie: z pretreningiem SQuAD vs bez

| Metryka | Bez SQuAD | Z SQuAD | Poprawa |
|---------|-----------|---------|---------|
| Dev EM | 41.10% | 56.16% | +15.07pp |
| Dev F1 | 59.87% | 74.07% | +14.20pp |
| Test EM | 42.68% | 45.12% | +2.44pp |
| Test F1 | 61.79% | 69.24% | +7.45pp |

### Uwaga

Poprawa na test set (+7.45pp F1) jest mniejsza niż na dev (+14.2pp).
Prawdopodobna przyczyna: mały test set (82 przykłady) + różnice w dystrybucji
pytań między dev (golden 12b) a test (golden 13b).

## Artefakty

- Model: `results/bioasq/biobert/best_model/`
- Metryki: `results/bioasq/biobert/results.json`
- Predykcje: `results/bioasq/biobert/{eval,test}_predictions.json`
