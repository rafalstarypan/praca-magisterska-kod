# Wynik treningu: PubMedBERT — COVID-QA 5-fold CV LoRA r=16 (ablacja ranku)

Data: 2026-04-19
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Eksperyment: E2 — ablacja ranku LoRA (r=16) na COVID-QA

## Model

| Parametr | Wartość |
|----------|---------|
| Model | PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) |
| Parametry (total) | 109,484,548 |
| Parametry (trainable) | 591,362 (0.54% — LoRA) |
| Źródło wag | `results/squad/pubmedbert/best_model/` (po pretreningu na SQuAD v1.1) |

## Konfiguracja LoRA

| Parametr | Wartość |
|----------|---------|
| Rank (r) | **16** (ablacja) |
| Alpha (α) | 16 (α/r = 1.0) |
| Dropout | 0.1 |
| Target modules | query, value |
| Learning rate | 1e-4 |

## Wyniki zagregowane (test set)

| Metryka | Mean | Std |
|---------|------|-----|
| **Exact Match** | **39.41%** | **±1.33** |
| **F1** | **65.25%** | **±0.99** |

## Ablacja ranku — COVID-QA (kompletna)

| Strategia | Trainable | % total | Test F1 | Std F1 | VRAM |
|-----------|-----------|---------|---------|--------|------|
| Full FT | 108,893,186 | 100% | 62.81 | ±1.49 | ~3900 MB |
| LoRA r=4 | 148,994 | 0.14% | 65.39 | ±0.35 | 2518 MB |
| LoRA r=8 | 296,450 | 0.27% | **65.53** | ±0.83 | 2522 MB |
| LoRA r=16 | 591,362 | 0.54% | 65.25 | ±0.99 | 2527 MB |

### Komentarz — wnioski z ablacji na COVID-QA

Wyniki potwierdzają obserwacje z BioASQ i wzmacniają tezę pracy:

1. **Ranga LoRA nie wpływa istotnie na jakość**: r=4/8/16 dają
   F1 65.25-65.53 (rozrzut <0.3pp). Identyczny wniosek jak na BioASQ.

2. **LoRA > Full FT niezależnie od rangi**: Wszystkie warianty LoRA
   przewyższają Full FT o 2.4-2.7pp F1. Efekt regularyzacji jest
   konsystentny na obu datasetach.

3. **Wyższa ranga = wyższe std**: r=4 (±0.35) < r=8 (±0.83) < r=16 (±0.99).
   Mniejsza pojemność → mniejsza wariancja między foldami.

4. **VRAM nie zależy od rangi**: 2518-2527 MB (~0.4% różnicy), bo
   adaptery stanowią <0.5% parametrów modelu.

## Hardware

| Parametr | Wartość |
|----------|---------|
| Peak VRAM (allocated) | 2527 MB |
| Czas per fold | ~24 min |
| Łączny czas | ~2h |

## Artefakty

- Wyniki: `results/covidqa/pubmedbert_lora_r16/fold_{0-4}/results.json`
- Adaptery: `results/covidqa/pubmedbert_lora_r16/fold_{0-4}/best_model/`
