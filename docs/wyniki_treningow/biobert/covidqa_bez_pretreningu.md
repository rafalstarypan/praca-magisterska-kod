# Wynik treningu: BioBERT — COVID-QA 5-fold CV (Bez pretreningu SQuAD)

Data: 2026-04-08
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | BioBERT v1.1 (dmis-lab/biobert-v1.1) |
| Parametry (total) | 107,721,218 |
| Parametry (trainable) | 107,721,218 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (pretrained LM + PubMed, bez SQuAD) |

## Wyniki per fold

| Fold | Eval EM | Eval F1 | Test EM | Test F1 | Peak VRAM | Czas |
|------|---------|---------|---------|---------|-----------|------|
| 0 | 26.45% | 47.14% | 21.78% | 44.89% | 3890 MB | 35 min |
| 1 | 26.07% | 47.83% | 26.07% | 47.83% | 3890 MB | 35 min |
| 2 | 22.11% | 43.73% | 22.11% | 43.73% | 3891 MB | 35 min |
| 3 | 20.46% | 42.67% | 20.46% | 42.67% | 3891 MB | 35 min |
| 4 | 24.09% | 45.65% | 24.09% | 45.65% | 3890 MB | 36 min |

## Wyniki zagregowane (test set)

| Metryka | Mean | Std |
|---------|------|-----|
| **Exact Match** | **22.90%** | **±2.08** |
| **F1** | **44.95%** | **±1.88** |

## Hardware

| Parametr | Wartość |
|----------|---------|
| GPU | NVIDIA RTX 4070 Laptop |
| Peak VRAM (allocated) | 3891 MB |
| Peak VRAM (reserved) | 4092 MB |
| Czas per fold | ~35 min |
| Batch size (train) | 16 |

## Artefakty

- Wyniki per fold: `results/covidqa/biobert_no_squad/fold_{0-4}/results.json`
- Predykcje: `results/covidqa/biobert_no_squad/fold_{0-4}/{eval,test}_predictions.json`
