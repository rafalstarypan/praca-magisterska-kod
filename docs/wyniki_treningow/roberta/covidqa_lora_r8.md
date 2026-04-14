# Wynik treningu: RoBERTa — COVID-QA 5-fold CV LoRA r=8 (Z pretreningiem SQuAD)

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

## Wyniki per fold

| Fold | Test EM | Test F1 | Peak VRAM | Czas |
|------|---------|---------|-----------|------|
| 0 | 37.95% | 64.21% | 2580 MB | 26 min |
| 1 | 38.94% | 65.08% | 2580 MB | 26 min |
| 2 | 37.62% | 64.70% | 2577 MB | 26 min |
| 3 | 39.27% | 65.26% | 2580 MB | 26 min |
| 4 | 38.28% | 64.64% | 2577 MB | 26 min |

## Wyniki zagregowane (test set)

| Metryka | Mean | Std |
|---------|------|-----|
| **Exact Match** | **38.42%** | **±0.68** |
| **F1** | **64.78%** | **±0.41** |

## Porównanie z Full Fine-Tuning (E1)

| Metryka | Full FT | LoRA r=8 | Δ |
|---------|---------|----------|---|
| Test EM | 36.37 ± 1.87 | 38.42 ± 0.68 | **+2.05pp** |
| Test F1 | 61.45 ± 1.17 | 64.78 ± 0.41 | **+3.33pp** |
| Trainable params | 124,056,578 | 296,450 | -99.76% |
| Peak VRAM | ~4080 MB | 2579 MB | -37% |
| Czas per fold | ~35 min | ~26 min | -26% |
| Std F1 | ±1.17 | ±0.41 | Niższa (stabilniejszy) |

### Komentarz

Identyczny wzorzec jak PubMedBERT: LoRA **przewyższa** Full FT na COVID-QA
(+3.33pp F1) i jest bardziej stabilna (std 0.41 vs 1.17). Efekt
regularyzacyjny LoRA jest jeszcze wyraźniejszy niż dla PubMedBERT
(+3.33pp vs +2.72pp).

Interesująca asymetria: na BioASQ LoRA jest gorsza od Full FT (-3.33pp),
a na COVID-QA lepsza (+3.33pp). Prawdopodobne wyjaśnienie: BioASQ ma krótkie
konteksty (~29 słów) i 82 przykłady testowe — Full FT nie overfittuje tak
bardzo. COVID-QA ma długie konteksty (~4900 słów) i wymaga lepszej
generalizacji, co faworyzuje LoRA.

## Hardware

| Parametr | Wartość |
|----------|---------|
| GPU | NVIDIA RTX 4070 Laptop |
| Peak VRAM (allocated) | 2579 MB (średnia) |
| Czas per fold | ~26 min (średnia) |
| Łączny czas | ~2h 10min |

## Artefakty

- Wyniki per fold: `results/covidqa/roberta_lora_r8/fold_{0-4}/results.json`
- Predykcje: `results/covidqa/roberta_lora_r8/fold_{0-4}/{eval,test}_predictions.json`
- Adaptery LoRA: `results/covidqa/roberta_lora_r8/fold_{0-4}/best_model/` (~1.2 MB each)
