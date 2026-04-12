# Wynik treningu: PubMedBERT — COVID-QA 5-fold CV (Bez pretreningu SQuAD)

Data: 2026-04-09
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) |
| Parametry (total) | 108,893,186 |
| Parametry (trainable) | 108,893,186 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (pretrained LM from scratch na PubMed, bez SQuAD) |

## Wyniki per fold

| Fold | Test EM | Test F1 | Peak VRAM | Czas |
|------|---------|---------|-----------|------|
| 0 | 31.35% | 53.88% | 3906 MB | 30 min |
| 1 | 32.34% | 54.05% | 3906 MB | 30 min |
| 2 | 33.00% | 54.89% | 3906 MB | 30 min |
| 3 | 31.02% | 54.83% | 3906 MB | 30 min |
| 4 | 32.34% | 55.63% | 3906 MB | 30 min |

## Wyniki zagregowane (test set)

| Metryka | Mean | Std |
|---------|------|-----|
| **Exact Match** | **32.01%** | **±0.81** |
| **F1** | **54.65%** | **±0.72** |

## Hardware

| Parametr | Wartość |
|----------|---------|
| GPU | NVIDIA RTX 4070 Laptop |
| Peak VRAM (allocated) | 3906 MB |
| Czas per fold | ~30 min |
| Batch size (train) | 16 |

## Artefakty

- Wyniki per fold: `results/covidqa/pubmedbert_no_squad/fold_{0-4}/results.json`
- Predykcje: `results/covidqa/pubmedbert_no_squad/fold_{0-4}/{eval,test}_predictions.json`
