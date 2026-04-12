# Wynik treningu: ClinicalBERT — COVID-QA 5-fold CV (Bez pretreningu SQuAD)

Data: 2026-04-10
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT) |
| Parametry (total) | 107,721,218 |
| Parametry (trainable) | 107,721,218 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (BioBERT + MIMIC-III, bez SQuAD) |

## Wyniki per fold

| Fold | Test EM | Test F1 | Peak VRAM | Czas |
|------|---------|---------|-----------|------|
| 0 | 21.78% | 40.30% | 3891 MB | 36 min |
| 1 | 19.80% | 41.13% | 3891 MB | 36 min |
| 2 | 14.85% | 35.83% | 3891 MB | 35 min |
| 3 | 18.81% | 38.85% | 3891 MB | 35 min |
| 4 | 19.80% | 41.11% | 3891 MB | 36 min |

## Wyniki zagregowane (test set)

| Metryka | Mean | Std |
|---------|------|-----|
| **Exact Match** | **19.21%** | **±2.51** |
| **F1** | **39.44%** | **±2.16** |

### Uwaga

ClinicalBERT bez SQuAD (F1=39.44) wypada gorzej niż BERT bez SQuAD (39.66)
i znacznie gorzej niż BioBERT/PubMedBERT bez SQuAD (44.95/54.65). Wąski
pretrening na notatkach klinicznych (MIMIC-III) nie pomaga na COVID-QA,
który zawiera teksty naukowe (artykuły), nie notatki kliniczne.

## Hardware

| Parametr | Wartość |
|----------|---------|
| GPU | NVIDIA RTX 4070 Laptop |
| Peak VRAM (allocated) | 3891 MB |
| Czas per fold | ~36 min |
| Batch size (train) | 16 |

## Artefakty

- Wyniki per fold: `results/covidqa/clinicalbert_no_squad/fold_{0-4}/results.json`
- Predykcje: `results/covidqa/clinicalbert_no_squad/fold_{0-4}/{eval,test}_predictions.json`
