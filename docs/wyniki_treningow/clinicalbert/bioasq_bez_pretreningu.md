# Wynik treningu: ClinicalBERT — BioASQ (Bez pretreningu SQuAD)

Data: 2026-04-06
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 77 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT) |
| Parametry (total) | 107,721,218 |
| Parametry (trainable) | 107,721,218 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (BioBERT + MIMIC-III, bez SQuAD) |

## Wyniki

| Zbiór | Exact Match | F1 |
|-------|-------------|-----|
| Dev | 35.62% | 50.35% |
| **Test** | **35.37%** | **50.70%** |

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| train_runtime | 77.4 s |
| train_loss (średni) | 2.212 |
| train_samples_per_second | 47.32 |

### Uwaga

ClinicalBERT bez SQuAD (Test F1=50.7%) jest na poziomie BERT bez SQuAD (50.2%),
mimo że bazuje na BioBERT. Dodatkowy pretrening na wąskiej domenie klinicznej
(MIMIC-III) nie poprawia wyników na ogólnej biomedycynie (BioASQ).

## Artefakty

- Model: `results/bioasq/clinicalbert_no_squad/best_model/`
- Metryki: `results/bioasq/clinicalbert_no_squad/results.json`
- Predykcje: `results/bioasq/clinicalbert_no_squad/{eval,test}_predictions.json`
