# Wynik treningu: RoBERTa — BioASQ (Bez pretreningu SQuAD)

Data: 2026-04-06
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 80 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | RoBERTa-base (FacebookAI/roberta-base) |
| Parametry (total) | 124,056,578 |
| Parametry (trainable) | 124,056,578 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (pretrained LM, bez SQuAD) |

## Wyniki

| Zbiór | Exact Match | F1 |
|-------|-------------|-----|
| Dev | 42.47% | 59.64% |
| **Test** | **42.68%** | **62.30%** |

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| train_runtime | 80.4 s |
| train_loss (średni) | 1.994 |
| train_samples_per_second | 45.43 |

### Uwaga

RoBERTa bez SQuAD (Test F1=62.3%) jest słabszy niż PubMedBERT bez SQuAD (71.4%),
mimo że na ogólnym SQuAD RoBERTa był najlepszy (F1=92.3%). Potwierdza to,
że domenowy pretrening jest ważniejszy niż ogólna jakość modelu
dla zadań biomedycznych.

## Artefakty

- Model: `results/bioasq/roberta_no_squad/best_model/`
- Metryki: `results/bioasq/roberta_no_squad/results.json`
- Predykcje: `results/bioasq/roberta_no_squad/{eval,test}_predictions.json`
