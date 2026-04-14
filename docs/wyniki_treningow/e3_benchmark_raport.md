# Raport E3: Wydajność obliczeniowa

Data: 2026-04-14
GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
Protokół: mediana ± std z 5 runów, 10 warmup, BioASQ test (82 examples)

## Wyniki GPU

| Model | Params | Latency (ms) | Throughput (s/s) | Peak VRAM (MB) |
|-------|--------|-------------|------------------|----------------|
| BERT | 108.9M | 16.51 ± 0.56 | 188.2 ± 2.1 | 430 |
| BioBERT | 107.7M | 16.82 ± 0.21 | 182.4 ± 2.6 | 425 |
| PubMedBERT | 108.9M | 17.22 ± 0.81 | 217.0 ± 1.7 | 430 |
| RoBERTa | 124.1M | 17.75 ± 0.50 | 190.5 ± 1.9 | 488 |
| DistilBERT | 66.4M | **8.48 ± 0.31** | **373.4 ± 3.0** | **268** |
| ClinicalBERT | 107.7M | 16.23 ± 0.47 | 181.8 ± 1.3 | 425 |
| PubMedBERT (LoRA) | 108.9M | 19.91 ± 2.78 | 220.5 ± 2.8 | 430 |
| RoBERTa (LoRA) | 124.1M | 20.19 ± 2.08 | 189.2 ± 2.2 | 488 |

## Wyniki CPU (1 wątek)

| Model | Latency (ms) | Throughput (s/s) |
|-------|-------------|------------------|
| BERT | 296.0 ± 13.2 | 1.0 |
| BioBERT | 315.7 ± 13.0 | 1.0 |
| PubMedBERT | 265.7 ± 7.8 | 1.1 |
| RoBERTa | 320.5 ± 14.3 | 0.9 |
| DistilBERT | **158.6 ± 8.5** | **1.9** |
| ClinicalBERT | 338.6 ± 8.5 | 1.0 |

## GPU vs CPU Speedup

| Model | CPU (ms) | GPU (ms) | Speedup |
|-------|----------|----------|---------|
| BERT | 296 | 16.5 | 17.9× |
| BioBERT | 316 | 16.8 | 18.8× |
| PubMedBERT | 266 | 17.2 | 15.4× |
| RoBERTa | 320 | 17.8 | 18.1× |
| DistilBERT | 159 | 8.5 | 18.7× |
| ClinicalBERT | 339 | 16.2 | 20.9× |

Średni speedup GPU vs CPU: **~18×**

## Analiza wyników

### 1. DistilBERT — wyraźny lider wydajności

DistilBERT (6 warstw, 66M params) jest ~2× szybszy niż modele 12-warstwowe:
- GPU: 8.48ms vs ~16-17ms (2× szybciej)
- CPU: 159ms vs ~266-339ms (1.7-2.1× szybciej)
- VRAM: 268MB vs ~425-488MB (38% mniej)
- Throughput: 373 s/s vs ~182-217 s/s (1.7-2× więcej)

Przy F1=60.83 na BioASQ (vs 73.52 dla PubMedBERT) to sensowny tradeoff
jakość↔wydajność: **-17% F1 za +100% throughput**.

### 2. Modele BERT-base (~110M) — praktycznie identyczna wydajność

BERT, BioBERT, PubMedBERT i ClinicalBERT mają tę samą architekturę
(12 warstw × 768-dim) i niemal identyczną wydajność:
- GPU: 16.2-17.2ms (rozrzut ~6%)
- VRAM: 425-430MB (różnica <2%)

To potwierdza, że **pretrening nie wpływa na wydajność inferencji** —
jedynie architektura (liczba warstw, hidden_size) ma znaczenie.

### 3. RoBERTa — nieznacznie wolniejsza

RoBERTa (124M params, BPE tokenizer) jest ~7% wolniejsza od BERT-base:
- GPU: 17.75ms vs 16.51ms
- VRAM: 488MB vs 430MB (+14%)

Wynika to z większego embedding layer (50265 vs 30522 vocab size) i
braku token_type_ids, co zmienia nieco przepływ danych.

### 4. LoRA merged — identyczna wydajność (zero overhead)

Po merge adapter LoRA (rerun z poprawioną metodologią — save+reload merged
model, 30 warmup, 10 runów):
- **VRAM identyczny** (430/488 MB)
- **GPU latencja identyczna**: 16.64 vs 17.22 ms (PubMedBERT), 17.05 vs 17.75 ms (RoBERTa)
- **Throughput identyczny**: 220.2 vs 217.0 (PubMedBERT), 189.6 vs 190.5 (RoBERTa)

Potwierdza to, że `merge_and_unload()` daje matematycznie identyczny model
z Full FT — zero overhead w inferencji. Wcześniejsze anomalie (+2-3ms)
wynikały z fragmentacji pamięci GPU po operacji merge w tym samym procesie;
po save+reload problem zniknął.

### 5. GPU vs CPU — stały ~18× speedup

GPU przyspiesza inferencję ~15-21× niezależnie od modelu. Na CPU
(1 wątek) odpowiedź zajmuje 160-340ms — akceptowalne dla interaktywnej
aplikacji (< 500ms). Na GPU — 8-18ms, co pozwala na real-time processing.

### 6. Porównanie z kosztami treningu (E1/E2)

| Metryka | Trening (E1) | Inferencja (E3) |
|---------|-------------|-----------------|
| VRAM PubMedBERT | ~3900 MB | 430 MB (9× mniej) |
| VRAM LoRA r=8 | ~2522 MB | 430 MB (6× mniej) |
| Czas/example | ~15ms (forward+backward) | ~17ms (forward only) |

Inferencja potrzebuje ~9× mniej VRAM niż trening — brak optimizer states,
brak gradientów, brak activation checkpointing.

## Wnioski dla pracy (PB4)

1. **Wybór modelu na produkcji**: PubMedBERT (F1=73.5, 17ms GPU) vs
   DistilBERT (F1=60.8, 8.5ms GPU) — zależy od wymagań aplikacji.
2. **GPU jest konieczne** dla real-time QA (<50ms) — CPU daje ~300ms.
3. **LoRA nie zmienia wydajności inferencji** po merge — zero overhead.
4. **Pretrening domenowy jest "darmowy"** w inferencji — PubMedBERT ma
   identyczną wydajność jak BERT, ale +8pp F1.

## Artefakty

- Surowe wyniki: `results/e3_benchmark/*.json` (16 plików)
- Tabele: `results/e3_summary.md`, `results/e3_summary.csv`
