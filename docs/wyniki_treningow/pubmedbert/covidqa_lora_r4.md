# Wynik treningu: PubMedBERT — COVID-QA 5-fold CV LoRA r=4 (ablacja ranku)

Data: 2026-04-19
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Eksperyment: E2 — ablacja ranku LoRA (r=4) na COVID-QA

## Model

| Parametr | Wartość |
|----------|---------|
| Model | PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) |
| Parametry (total) | 109,042,180 |
| Parametry (trainable) | 148,994 (0.14% — LoRA) |
| Źródło wag | `results/squad/pubmedbert/best_model/` (po pretreningu na SQuAD v1.1) |

## Konfiguracja LoRA

| Parametr | Wartość |
|----------|---------|
| Rank (r) | **4** (ablacja) |
| Alpha (α) | 16 (α/r = 4.0) |
| Dropout | 0.1 |
| Target modules | query, value |
| Learning rate | 1e-4 |

## Wyniki per fold

| Fold | Test EM | Test F1 | Peak VRAM | Czas |
|------|---------|---------|-----------|------|
| 0 | 39.93% | 65.70% | 2518 MB | 24 min |
| 1 | 39.60% | 65.50% | 2518 MB | 24 min |
| 2 | 39.93% | 65.57% | 2518 MB | 24 min |
| 3 | 37.62% | 64.99% | 2518 MB | 24 min |
| 4 | 38.61% | 65.17% | 2518 MB | 23 min |

## Wyniki zagregowane (test set)

| Metryka | Mean | Std |
|---------|------|-----|
| **Exact Match** | **39.14%** | **±1.01** |
| **F1** | **65.39%** | **±0.35** |

## Ablacja ranku — COVID-QA

| Strategia | Trainable | Test EM | Test F1 | Std F1 |
|-----------|-----------|---------|---------|--------|
| Full FT | 108,893,186 | 37.03 ± 1.00 | 62.81 ± 1.49 | ±1.49 |
| LoRA r=4 | 148,994 | 39.14 ± 1.01 | 65.39 ± 0.35 | **±0.35** |
| LoRA r=8 | 296,450 | 39.21 ± 0.72 | 65.53 ± 0.83 | ±0.83 |
| LoRA r=16 | 591,362 | 39.41 ± 1.33 | 65.25 ± 0.99 | ±0.99 |

### Komentarz

LoRA r=4 na COVID-QA potwierdza obserwacje z BioASQ: ranga nie wpływa
istotnie na wynik (F1 65.39 vs 65.53 vs 65.25 — różnice <0.3pp).
Wszystkie warianty LoRA przewyższają Full FT o ~2.5pp.

r=4 wyróżnia się **najniższym odchyleniem standardowym** (±0.35 vs ±0.83
dla r=8 i ±0.99 dla r=16), co sugeruje najstabilniejszy trening.

## Hardware

| Parametr | Wartość |
|----------|---------|
| Peak VRAM (allocated) | 2518 MB |
| Czas per fold | ~24 min |
| Łączny czas | ~2h |

## Artefakty

- Wyniki: `results/covidqa/pubmedbert_lora_r4/fold_{0-4}/results.json`
- Adaptery: `results/covidqa/pubmedbert_lora_r4/fold_{0-4}/best_model/`
