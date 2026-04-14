# Wynik treningu: RoBERTa — BioASQ LoRA r=8 (Z pretreningiem SQuAD)

Data: 2026-04-13
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Eksperyment: E2 (Full FT vs LoRA)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | RoBERTa (FacebookAI/roberta-base) |
| Parametry (total) | 124,353,028 |
| Parametry (trainable) | 296,450 (0.24% — LoRA) |
| Źródło wag | `results/squad/roberta/best_model/` (po pretreningu na SQuAD v1.1) |

## Konfiguracja LoRA

| Parametr | Wartość |
|----------|---------|
| Rank (r) | 8 |
| Alpha (α) | 16 |
| Dropout | 0.1 |
| Target modules | query, value |
| Learning rate | 1e-4 |

## Wyniki

| Zbiór | Exact Match | F1 |
|-------|-------------|-----|
| **Test** | **46.34%** | **65.70%** |

### Metryki końcowe treningu

| Metryka | Wartość |
|---------|---------|
| train_runtime | 42 s |
| train_loss (średni) | 1.178 |

## Porównanie z Full Fine-Tuning (E1)

| Metryka | Full FT | LoRA r=8 | Δ |
|---------|---------|----------|---|
| Test EM | 50.00 | 46.34 | -3.66pp |
| Test F1 | 69.03 | 65.70 | -3.33pp |
| Trainable params | 124,056,578 | 296,450 | -99.76% |
| Peak VRAM | ~4080 MB | 2580 MB | -37% |
| Czas treningu | ~68 s | 42 s | -38% |

### Komentarz

RoBERTa LoRA traci 3.33pp F1 na BioASQ vs Full FT — więcej niż PubMedBERT
(który stracił tylko 0.14pp). Na małym zbiorze BioASQ (82 test examples)
ta różnica jest na granicy istotności statystycznej. Wyższy spadek RoBERTa
może wynikać z tego, że model ogólnodomenowy potrzebuje więcej pojemności
adaptacyjnej do przystosowania się do terminologii biomedycznej.

## Hardware

| Parametr | Wartość |
|----------|---------|
| GPU | NVIDIA RTX 4070 Laptop |
| Peak VRAM (allocated) | 2580 MB |
| Czas treningu | 42 s |

## Artefakty

- Wyniki: `results/bioasq/roberta_lora_r8/results.json`
- Predykcje: `results/bioasq/roberta_lora_r8/{eval,test}_predictions.json`
- Adapter LoRA: `results/bioasq/roberta_lora_r8/best_model/` (~1.2 MB)
