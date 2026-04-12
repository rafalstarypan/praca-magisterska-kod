# Wynik treningu: BERT — BioASQ (Bez pretreningu SQuAD)

Data: 2026-04-06
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 62 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | BERT-base-uncased (google-bert/bert-base-uncased) |
| Parametry (total) | 108,893,186 |
| Parametry (trainable) | 108,893,186 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (pretrained LM, bez SQuAD) |

## Wyniki

| Zbiór | Exact Match | F1 |
|-------|-------------|-----|
| Dev | 30.14% | 47.21% |
| **Test** | **34.15%** | **50.24%** |

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| train_runtime | 62.1 s |
| train_loss (średni) | 2.208 |
| train_samples_per_second | 58.83 |

## Artefakty

- Model: `results/bioasq/bert_no_squad/best_model/`
- Metryki: `results/bioasq/bert_no_squad/results.json`
- Predykcje: `results/bioasq/bert_no_squad/{eval,test}_predictions.json`
