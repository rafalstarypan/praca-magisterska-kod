# Wynik treningu: RoBERTa — COVID-QA 5-fold CV (Bez pretreningu SQuAD)

Data: 2026-04-09
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | RoBERTa-base (FacebookAI/roberta-base) |
| Parametry (total) | 124,056,578 |
| Parametry (trainable) | 124,056,578 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (pretrained LM, bez SQuAD) |

## Wyniki per fold

| Fold | Test EM | Test F1 | Peak VRAM | Czas |
|------|---------|---------|-----------|------|
| 0 | 27.39% | 52.04% | 4080 MB | 34 min |
| 1 | 29.70% | 53.60% | 4080 MB | 34 min |
| 2 | 30.36% | 54.66% | 4080 MB | 34 min |
| 3 | 28.38% | 52.20% | 4080 MB | 34 min |
| 4 | 31.02% | 54.63% | 4080 MB | 34 min |

## Wyniki zagregowane (test set)

| Metryka | Mean | Std |
|---------|------|-----|
| **Exact Match** | **29.37%** | **±1.42** |
| **F1** | **53.43%** | **±1.11** |

## Hardware

| Parametr | Wartość |
|----------|---------|
| GPU | NVIDIA RTX 4070 Laptop |
| Peak VRAM (allocated) | 4080 MB |
| Czas per fold | ~34 min |
| Batch size (train) | 16 |

## Artefakty

- Wyniki per fold: `results/covidqa/roberta_no_squad/fold_{0-4}/results.json`
- Predykcje: `results/covidqa/roberta_no_squad/fold_{0-4}/{eval,test}_predictions.json`
