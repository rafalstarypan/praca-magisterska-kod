# Wynik treningu: BioBERT — BioASQ (Bez pretreningu SQuAD)

Data: 2026-04-06
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 72 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | BioBERT v1.1 (dmis-lab/biobert-v1.1) |
| Parametry (total) | 107,721,218 |
| Parametry (trainable) | 107,721,218 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (pretrained LM + PubMed, bez SQuAD) |

## Wyniki

| Zbiór | Exact Match | F1 |
|-------|-------------|-----|
| Dev | 41.10% | 59.87% |
| **Test** | **42.68%** | **61.79%** |

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| train_runtime | 71.6 s |
| train_loss (średni) | 2.084 |
| train_samples_per_second | 51.14 |

## Artefakty

- Model: `results/bioasq/biobert_no_squad/best_model/`
- Metryki: `results/bioasq/biobert_no_squad/results.json`
- Predykcje: `results/bioasq/biobert_no_squad/{eval,test}_predictions.json`
