# Wynik treningu: BERT-base — pretrening na SQuAD v1.1

Data: 2026-04-06
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 44 min 23 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | BERT-base-uncased (google-bert/bert-base-uncased) |
| Parametry (total) | 108,893,186 |
| Parametry (trainable) | 108,893,186 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (pretrained LM) |

## Dataset

| Parametr | Wartość |
|----------|---------|
| Dataset | SQuAD v1.1 (rajpurkar/squad) |
| Train | 87,599 przykładów |
| Dev | 10,570 przykładów |
| Test | Brak (SQuAD test set nie jest publiczny) |
| Typ pytań | Ekstrakcyjne QA (answer span w kontekście) |

## Hiperparametry

| Parametr | Wartość | Źródło |
|----------|---------|--------|
| learning_rate | 3e-5 | Devlin et al. (2019) |
| batch_size | 16 | Devlin et al. (2019), zakres {16, 32} |
| epochs | 2 | Devlin et al. (2019), Appendix A.3 |
| max_seq_length | 384 | Devlin et al. (2019), Section 4.2 |
| doc_stride | 128 | Devlin et al. (2019), Section 4.2 |
| weight_decay | 0.01 | Devlin et al. (2019) |
| warmup_ratio | 0.1 | ~10% warmup, standard BERT |
| fp16 | True | Mixed precision (Micikevicius et al., 2018) |
| optimizer | AdamW | HuggingFace Trainer default |
| lr_scheduler | linear decay with warmup | HuggingFace Trainer default |

## Wyniki

### Metryki na SQuAD v1.1 dev

| Metryka | Wynik | Oczekiwane (Devlin et al., 2019) |
|---------|-------|-------------------------------|
| **Exact Match** | 81.11% | ~80.8% |
| **F1** | 88.35% | ~88.5% |

Wyniki zgodne z literaturą — pipeline działa poprawnie.

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| Train loss (średni) | 1.130 |
| train_runtime | 2663.3 s |
| train_samples_per_second | 66.48 |
| train_steps_per_second | 4.16 |
| total_flos | 3.47e+16 |

## Porównanie z literaturą

| Model | EM | F1 | Źródło |
|-------|-----|-----|--------|
| BERT-base (Devlin et al.) | 80.8 | 88.5 | Devlin et al. (2019) |
| **BERT-base (nasz wynik)** | **81.1** | **88.3** | Niniejsza praca |

Różnica <0.2pp F1 względem oryginalnego BERT paper. EM lekko wyższy (+0.3pp),
co mieści się w normalnej wariancji.

## Artefakty

- Model: `results/squad/bert/best_model/`
- Metryki: `results/squad/bert/results.json`
- Predykcje: `results/squad/bert/eval_predictions.json`
- TensorBoard: `results/squad/bert/runs/`
