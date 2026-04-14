# E3 — Wydajność obliczeniowa (podsumowanie)

Protokół: mediana ± std z 5 runów, 10 warmup, BioASQ test (82 examples).

## GPU (NVIDIA RTX 4070 Laptop)

| Model | Params | Latency (ms) | Throughput (samples/s) | Peak VRAM (MB) |
|-------|--------|-------------|----------------------|----------------|
| BERT | 108,893,186 | 16.51 ± 0.56 | 188.2 ± 2.1 | 430 |
| BioBERT | 107,721,218 | 16.82 ± 0.21 | 182.4 ± 2.6 | 425 |
| PubMedBERT | 108,893,186 | 17.22 ± 0.81 | 217.0 ± 1.7 | 430 |
| RoBERTa | 124,056,578 | 17.75 ± 0.50 | 190.5 ± 1.9 | 488 |
| DistilBERT | 66,364,418 | 8.48 ± 0.31 | 373.4 ± 3.0 | 268 |
| ClinicalBERT | 107,721,218 | 16.23 ± 0.47 | 181.8 ± 1.3 | 425 |
| PubMedBERT (LoRA r=8) | 108,893,186 | 16.64 ± 0.50 | 220.2 ± 2.3 | 430 |
| RoBERTa (LoRA r=8) | 124,056,578 | 17.05 ± 0.35 | 189.6 ± 1.0 | 488 |

## CPU (1 thread)

| Model | Params | Latency (ms) | Throughput (samples/s) |
|-------|--------|-------------|----------------------|
| BERT | 108,893,186 | 295.99 ± 13.23 | 1.0 ± 0.0 |
| BioBERT | 107,721,218 | 315.74 ± 12.95 | 1.0 ± 0.0 |
| PubMedBERT | 108,893,186 | 265.74 ± 7.82 | 1.1 ± 0.1 |
| RoBERTa | 124,056,578 | 320.45 ± 14.30 | 0.9 ± 0.0 |
| DistilBERT | 66,364,418 | 158.62 ± 8.52 | 1.9 ± 0.0 |
| ClinicalBERT | 107,721,218 | 338.58 ± 8.50 | 1.0 ± 0.1 |
| PubMedBERT (LoRA r=8) | 108,893,186 | 230.18 ± 3.44 | 1.4 ± 0.0 |
| RoBERTa (LoRA r=8) | 124,056,578 | 259.69 ± 5.35 | 1.2 ± 0.0 |

## GPU vs CPU Speedup

| Model | CPU Latency (ms) | GPU Latency (ms) | Speedup |
|-------|-----------------|-----------------|---------|
| BERT | 295.99 | 16.51 | 17.9× |
| BioBERT | 315.74 | 16.82 | 18.8× |
| PubMedBERT | 265.74 | 17.22 | 15.4× |
| RoBERTa | 320.45 | 17.75 | 18.1× |
| DistilBERT | 158.62 | 8.48 | 18.7× |
| ClinicalBERT | 338.58 | 16.23 | 20.9× |
| PubMedBERT (LoRA r=8) | 230.18 | 16.64 | 13.8× |
| RoBERTa (LoRA r=8) | 259.69 | 17.05 | 15.2× |
