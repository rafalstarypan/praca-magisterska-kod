# Wynik treningu: DistilBERT — COVID-QA 5-fold CV (Z pretreningiem SQuAD)

Data: 2026-04-07
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | DistilBERT (distilbert/distilbert-base-uncased) |
| Parametry (total) | 66,364,418 |
| Parametry (trainable) | 66,364,418 (100% — full fine-tuning) |
| Źródło wag | `results/squad/distilbert/best_model/` (po pretreningu na SQuAD v1.1) |

## Wyniki per fold

| Fold | Eval EM | Eval F1 | Test EM | Test F1 |
|------|---------|---------|---------|---------|
| 0 | 32.85% | 56.95% | 27.06% | 51.65% |
| 1 | 26.24% | 51.47% | 26.73% | 51.52% |
| 2 | 30.32% | 52.70% | 29.70% | 53.59% |
| 3 | 32.94% | 57.40% | 26.07% | 52.44% |
| 4 | 30.61% | 57.24% | 25.74% | 50.88% |

## Wyniki zagregowane (test set)

| Metryka | Mean | Std |
|---------|------|-----|
| **Exact Match** | **27.06%** | **±1.49** |
| **F1** | **51.98%** | **±1.16** |

## Porównanie: z pretreningiem SQuAD vs bez

| Metryka | Bez SQuAD | Z SQuAD | Poprawa |
|---------|-----------|---------|---------|
| Test EM | 15.71 ± 2.98 | 27.06 ± 1.49 | +11.35pp |
| Test F1 | 34.99 ± 3.25 | 51.98 ± 1.16 | +16.99pp |

SQuAD pretrening nie tylko poprawia wyniki, ale też stabilizuje model
(niższa wariancja między foldami).

## Artefakty

- Wyniki per fold: `results/covidqa/distilbert/fold_{0-4}/results.json`
- Predykcje: `results/covidqa/distilbert/fold_{0-4}/{eval,test}_predictions.json`
