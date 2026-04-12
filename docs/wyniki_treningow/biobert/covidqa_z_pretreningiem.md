# Wynik treningu: BioBERT — COVID-QA 5-fold CV (Z pretreningiem SQuAD)

Data: 2026-04-09
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | BioBERT v1.1 (dmis-lab/biobert-v1.1) |
| Parametry (total) | 107,721,218 |
| Parametry (trainable) | 107,721,218 (100% — full fine-tuning) |
| Źródło wag | `results/squad/biobert/best_model/` (po pretreningu na SQuAD v1.1) |

## Wyniki per fold

| Fold | Test EM | Test F1 | Peak VRAM | Czas |
|------|---------|---------|-----------|------|
| 0 | 32.01% | 57.93% | 3890 MB | 36 min |
| 1 | 33.33% | 56.76% | 3890 MB | 36 min |
| 2 | 31.35% | 56.82% | 3891 MB | 35 min |
| 3 | 29.37% | 56.16% | 3892 MB | 35 min |
| 4 | 32.67% | 57.62% | 3890 MB | 36 min |

## Wyniki zagregowane (test set)

| Metryka | Mean | Std |
|---------|------|-----|
| **Exact Match** | **31.75%** | **±1.63** |
| **F1** | **57.06%** | **±0.72** |

## Porównanie: z pretreningiem SQuAD vs bez

| Metryka | Bez SQuAD | Z SQuAD | Poprawa |
|---------|-----------|---------|---------|
| Test EM | 22.90 ± 2.08 | 31.75 ± 1.63 | +8.85pp |
| Test F1 | 44.95 ± 1.88 | 57.06 ± 0.72 | +12.11pp |

## Artefakty

- Wyniki per fold: `results/covidqa/biobert/fold_{0-4}/results.json`
- Predykcje: `results/covidqa/biobert/fold_{0-4}/{eval,test}_predictions.json`
