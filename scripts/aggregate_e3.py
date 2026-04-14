"""
Aggregate E3 benchmark results into summary table.

Usage:
    python scripts/aggregate_e3.py

Outputs:
    results/e3_summary.md   — Markdown table for thesis
    results/e3_summary.csv  — CSV for further analysis
"""

import csv
import json
from pathlib import Path

RESULTS_DIR = Path("results")
BENCH_DIR = RESULTS_DIR / "e3_benchmark"

MODEL_DISPLAY = {
    "bert": "BERT",
    "biobert": "BioBERT",
    "pubmedbert": "PubMedBERT",
    "roberta": "RoBERTa",
    "distilbert": "DistilBERT",
    "clinicalbert": "ClinicalBERT",
    "pubmedbert_lora_r8": "PubMedBERT (LoRA r=8)",
    "roberta_lora_r8": "RoBERTa (LoRA r=8)",
}

MODEL_ORDER = [
    "bert", "biobert", "pubmedbert", "roberta", "distilbert", "clinicalbert",
    "pubmedbert_lora_r8", "roberta_lora_r8",
]


def load_benchmark(label, device):
    path = BENCH_DIR / f"{label}_{device}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(val, prec=2):
    if val is None:
        return "—"
    return f"{val:.{prec}f}"


def fmt_pm(median, std, prec=2):
    if median is None:
        return "—"
    return f"{median:.{prec}f} ± {std:.{prec}f}"


def main():
    lines = []
    csv_rows = []

    def add(text=""):
        lines.append(text)

    add("# E3 — Wydajność obliczeniowa (podsumowanie)")
    add()
    add("Protokół: mediana ± std z 5 runów, 10 warmup, BioASQ test (82 examples).")
    add()

    # GPU table
    add("## GPU (NVIDIA RTX 4070 Laptop)")
    add()
    add("| Model | Params | Latency (ms) | Throughput (samples/s) | Peak VRAM (MB) |")
    add("|-------|--------|-------------|----------------------|----------------|")

    for label in MODEL_ORDER:
        r = load_benchmark(label, "cuda")
        if r is None:
            continue
        name = MODEL_DISPLAY.get(label, label)
        params = f"{r['total_params']:,}"
        lat = fmt_pm(r["latency_ms"]["median"], r["latency_ms"]["std"])
        tp = fmt_pm(r["throughput_samples_per_sec"]["median"],
                    r["throughput_samples_per_sec"]["std"], prec=1)
        vram = str(r["peak_vram_mb"]) if r["peak_vram_mb"] else "—"
        add(f"| {name} | {params} | {lat} | {tp} | {vram} |")
        csv_rows.append({
            "model": name,
            "device": "cuda",
            "params": r["total_params"],
            "latency_ms": r["latency_ms"]["median"],
            "latency_std": r["latency_ms"]["std"],
            "throughput": r["throughput_samples_per_sec"]["median"],
            "throughput_std": r["throughput_samples_per_sec"]["std"],
            "peak_vram_mb": r["peak_vram_mb"],
        })

    # CPU table
    add()
    add("## CPU (1 thread)")
    add()
    add("| Model | Params | Latency (ms) | Throughput (samples/s) |")
    add("|-------|--------|-------------|----------------------|")

    for label in MODEL_ORDER:
        r = load_benchmark(label, "cpu")
        if r is None:
            continue
        name = MODEL_DISPLAY.get(label, label)
        params = f"{r['total_params']:,}"
        lat = fmt_pm(r["latency_ms"]["median"], r["latency_ms"]["std"])
        tp = fmt_pm(r["throughput_samples_per_sec"]["median"],
                    r["throughput_samples_per_sec"]["std"], prec=1)
        add(f"| {name} | {params} | {lat} | {tp} |")
        csv_rows.append({
            "model": name,
            "device": "cpu",
            "params": r["total_params"],
            "latency_ms": r["latency_ms"]["median"],
            "latency_std": r["latency_ms"]["std"],
            "throughput": r["throughput_samples_per_sec"]["median"],
            "throughput_std": r["throughput_samples_per_sec"]["std"],
            "peak_vram_mb": None,
        })

    # GPU vs CPU speedup
    add()
    add("## GPU vs CPU Speedup")
    add()
    add("| Model | CPU Latency (ms) | GPU Latency (ms) | Speedup |")
    add("|-------|-----------------|-----------------|---------|")

    for label in MODEL_ORDER:
        r_gpu = load_benchmark(label, "cuda")
        r_cpu = load_benchmark(label, "cpu")
        if r_gpu is None or r_cpu is None:
            continue
        name = MODEL_DISPLAY.get(label, label)
        cpu_lat = r_cpu["latency_ms"]["median"]
        gpu_lat = r_gpu["latency_ms"]["median"]
        speedup = cpu_lat / gpu_lat if gpu_lat > 0 else 0
        add(f"| {name} | {fmt(cpu_lat)} | {fmt(gpu_lat)} | {fmt(speedup, 1)}× |")

    # Write outputs
    md_path = RESULTS_DIR / "e3_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {md_path}")

    csv_path = RESULTS_DIR / "e3_summary.csv"
    fieldnames = [
        "model", "device", "params",
        "latency_ms", "latency_std",
        "throughput", "throughput_std",
        "peak_vram_mb",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
