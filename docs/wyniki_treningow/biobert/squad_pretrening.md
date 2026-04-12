# Wynik treningu: BioBERT — pretrening na SQuAD v1.1

Data: 2026-04-06
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 44 min 18 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | BioBERT v1.1 (dmis-lab/biobert-v1.1) |
| Parametry (total) | 107,721,218 |
| Parametry (trainable) | 107,721,218 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (pretrained LM + PubMed) |

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

| Metryka | Wynik | Oczekiwane (Lee et al., 2020) |
|---------|-------|-------------------------------|
| **Exact Match** | 81.37% | ~82.1% |
| **F1** | 88.75% | ~89.1% |

Wyniki zgodne z literaturą — różnica <0.4pp F1.

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| Train loss (średni) | 1.108 |
| train_runtime | 2658.2 s |
| train_samples_per_second | 66.76 |
| train_steps_per_second | 4.17 |
| total_flos | 3.48e+16 |

## Porównanie z literaturą

| Model | EM | F1 | Źródło |
|-------|-----|-----|--------|
| BioBERT v1.1 (Lee et al.) | 82.1 | 89.1 | Lee et al. (2020) |
| **BioBERT v1.1 (nasz wynik)** | **81.4** | **88.7** | Niniejsza praca |

Różnica <0.4pp F1 względem oryginalnego BioBERT paper.

## Artefakty

- Model: `results/squad/biobert/best_model/`
- Metryki: `results/squad/biobert/results.json`
- Predykcje: `results/squad/biobert/eval_predictions.json`
- TensorBoard: `results/squad/biobert/runs/`
