# Wynik treningu: PubMedBERT — COVID-QA 5-fold CV LoRA r=8 (Z pretreningiem SQuAD)

Data: 2026-04-13
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Eksperyment: E2 (Full FT vs LoRA)

## Model

| Parametr | Wartość |
|----------|---------|
| Model | PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) |
| Parametry (total) | 109,189,636 |
| Parametry (trainable) | 296,450 (0.27% — LoRA) |
| Źródło wag | `results/squad/pubmedbert/best_model/` (po pretreningu na SQuAD v1.1) |

## Konfiguracja LoRA

| Parametr | Wartość |
|----------|---------|
| Rank (r) | 8 |
| Alpha (α) | 16 |
| Dropout | 0.1 |
| Target modules | query, value |
| Learning rate | 1e-4 (override vs 3e-5 dla full FT) |

## Wyniki per fold

| Fold | Test EM | Test F1 | Peak VRAM | Czas |
|------|---------|---------|-----------|------|
| 0 | 39.93% | 65.91% | 2522 MB | 25 min |
| 1 | 39.60% | 64.69% | 2520 MB | 26 min |
| 2 | 39.60% | 66.61% | 2522 MB | 28 min |
| 3 | 38.61% | 65.75% | 2522 MB | 28 min |
| 4 | 38.28% | 64.70% | 2520 MB | 28 min |

## Wyniki zagregowane (test set)

| Metryka | Mean | Std |
|---------|------|-----|
| **Exact Match** | **39.21%** | **±0.72** |
| **F1** | **65.53%** | **±0.83** |

## Porównanie z Full Fine-Tuning (E1)

| Metryka | Full FT | LoRA r=8 | Δ |
|---------|---------|----------|---|
| Test EM | 37.03 ± 1.00 | 39.21 ± 0.72 | **+2.18pp** |
| Test F1 | 62.81 ± 1.49 | 65.53 ± 0.83 | **+2.72pp** |
| Trainable params | 108,893,186 | 296,450 | -99.73% |
| Peak VRAM | ~3900 MB | 2521 MB | -35% |
| Czas per fold | ~35 min | ~27 min | -23% |
| Std F1 | ±1.49 | ±0.83 | Niższa (stabilniejszy) |

### Komentarz

Zaskakujący wynik: LoRA r=8 **przewyższa** Full Fine-Tuning na COVID-QA
zarówno w EM (+2.18pp) jak i F1 (+2.72pp). Ponadto ma niższe odchylenie
standardowe (0.83 vs 1.49), co wskazuje na bardziej stabilny trening.

Prawdopodobne wyjaśnienie: COVID-QA to mały dataset (~1,400 przykładów
treningowych per fold), na którym Full FT z 110M parametrami może lekko
overfittować. LoRA z 296K parametrami działa jako regularyzator —
ogranicza przestrzeń poszukiwań do niskowymiarowej podprzestrzeni,
co zapobiega overfittingowi i daje lepszą generalizację.

Ten wynik jest zgodny z obserwacjami Hu et al. (2022): "LoRA can
match or even exceed full fine-tuning quality" oraz z hipotezą
niskorangowej adaptacji (Aghajanyan et al. 2021).

## Hardware

| Parametr | Wartość |
|----------|---------|
| GPU | NVIDIA RTX 4070 Laptop |
| Peak VRAM (allocated) | 2521 MB (średnia) |
| Czas per fold | ~27 min (średnia) |
| Łączny czas | ~2h 15min |
| Platform | Windows 11 |
| PyTorch | 2.6.0+cu124 |

## Artefakty

- Wyniki per fold: `results/covidqa/pubmedbert_lora_r8/fold_{0-4}/results.json`
- Predykcje: `results/covidqa/pubmedbert_lora_r8/fold_{0-4}/{eval,test}_predictions.json`
- Adaptery LoRA: `results/covidqa/pubmedbert_lora_r8/fold_{0-4}/best_model/` (~1.9 MB each)
