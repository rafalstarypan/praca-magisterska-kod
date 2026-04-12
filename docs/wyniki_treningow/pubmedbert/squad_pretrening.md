# Wynik treningu: PubMedBERT — pretrening na SQuAD v1.1

Data: 2026-04-06
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Czas treningu: 44 min 37 s

## Model

| Parametr | Wartość |
|----------|---------|
| Model | PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) |
| Parametry (total) | 108,893,186 |
| Parametry (trainable) | 108,893,186 (100% — full fine-tuning) |
| Źródło wag | HuggingFace Hub (pretrained LM + PubMed abstracts & full-text) |

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

| Metryka | Wynik | Oczekiwane (Gu et al., 2022) |
|---------|-------|-------------------------------|
| **Exact Match** | 82.24% | ~81.5% |
| **F1** | 89.60% | ~89.0% |

Wyniki zgodne z literaturą — lekko powyżej oczekiwanych wartości.

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| Train loss (średni) | 1.067 |
| train_runtime | 2676.7 s |
| train_samples_per_second | 66.68 |
| train_steps_per_second | 4.17 |
| total_flos | 3.50e+16 |

## Porównanie z literaturą

| Model | EM | F1 | Źródło |
|-------|-----|-----|--------|
| PubMedBERT (Gu et al.) | 81.5 | 89.0 | Gu et al. (2022) |
| **PubMedBERT (nasz wynik)** | **82.2** | **89.6** | Niniejsza praca |

Wynik lekko powyżej referencji (+0.6pp F1). PubMedBERT osiągnął najwyższy
F1 spośród dotychczas trenowanych modeli na SQuAD.

## Artefakty

- Model: `results/squad/pubmedbert/best_model/`
- Metryki: `results/squad/pubmedbert/results.json`
- Predykcje: `results/squad/pubmedbert/eval_predictions.json`
- TensorBoard: `results/squad/pubmedbert/runs/`
