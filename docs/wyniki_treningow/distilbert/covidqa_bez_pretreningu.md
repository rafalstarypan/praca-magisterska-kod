# Wynik treningu: DistilBERT — COVID-QA 5-fold CV (Bez pretreningu SQuAD)

Data: 2026-04-07
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | DistilBERT (distilbert/distilbert-base-uncased) |
| Parametry (total) | 66,364,418 |
| Parametry (trainable) | 66,364,418 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (pretrained LM, bez SQuAD) |

## Wyniki per fold

| Fold | Eval EM | Eval F1 | Test EM | Test F1 |
|------|---------|---------|---------|---------|
| 0 | 18.60% | 38.27% | 16.17% | 34.68% |
| 1 | 16.62% | 35.95% | 18.81% | 39.48% |
| 2 | 17.49% | 34.50% | 18.15% | 35.96% |
| 3 | 17.49% | 38.05% | 11.22% | 30.48% |
| 4 | 20.41% | 42.51% | 14.19% | 34.35% |

## Wyniki zagregowane (test set)

| Metryka | Mean | Std |
|---------|------|-----|
| **Exact Match** | **15.71%** | **±2.98** |
| **F1** | **34.99%** | **±3.25** |

## Artefakty

- Wyniki per fold: `results/covidqa/distilbert_no_squad/fold_{0-4}/results.json`
- Predykcje: `results/covidqa/distilbert_no_squad/fold_{0-4}/{eval,test}_predictions.json`
