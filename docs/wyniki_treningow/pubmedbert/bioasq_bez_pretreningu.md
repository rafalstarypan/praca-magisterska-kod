# Wynik treningu: PubMedBERT — BioASQ (Bez pretreningu SQuAD)

Data: 2026-04-06
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 70 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) |
| Parametry (total) | 108,893,186 |
| Parametry (trainable) | 108,893,186 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (pretrained LM from scratch na PubMed, bez SQuAD) |

## Wyniki

| Zbiór | Exact Match | F1 |
|-------|-------------|-----|
| Dev | 42.47% | 64.99% |
| **Test** | **53.66%** | **71.39%** |

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| train_runtime | 70.0 s |
| train_loss (średni) | 1.887 |
| train_samples_per_second | 52.16 |

### Uwaga

PubMedBERT bez SQuAD (Test F1=71.39%) przewyższa BioBERT z SQuAD (69.24%).
Potwierdza to tezę Gu et al. (2022) o przewadze domain-specific pretreningu
from scratch nad continual pretreningiem.

## Artefakty

- Model: `results/bioasq/pubmedbert_no_squad/best_model/`
- Metryki: `results/bioasq/pubmedbert_no_squad/results.json`
- Predykcje: `results/bioasq/pubmedbert_no_squad/{eval,test}_predictions.json`
