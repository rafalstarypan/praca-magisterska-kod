"""Inference benchmarking for E3: computational efficiency.

Measures latency (GPU/CPU), throughput, and peak VRAM for all models.

Usage:
    # Single model on GPU:
    python src/benchmark.py --model_path results/bioasq/pubmedbert/best_model --tokenizer pubmedbert --device cuda

    # Single model on CPU:
    python src/benchmark.py --model_path results/bioasq/pubmedbert/best_model --tokenizer pubmedbert --device cpu

    # LoRA model (merge adapter before benchmarking):
    python src/benchmark.py --model_path results/bioasq/pubmedbert_lora_r8/best_model --tokenizer pubmedbert --device cuda --lora --base_model_path results/squad/pubmedbert/best_model

    # All models:
    python scripts/run_e3.bat
"""

import argparse
import json
import logging
import os
import platform
import statistics
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from data import load_squad_examples

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_REGISTRY = {
    "bert": "google-bert/bert-base-uncased",
    "biobert": "dmis-lab/biobert-v1.1",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "roberta": "FacebookAI/roberta-base",
    "distilbert": "distilbert/distilbert-base-uncased",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
}


def load_model(model_path, tokenizer_name, device, lora=False, base_model_path=None):
    """Load model and tokenizer onto device. Merge LoRA if needed."""
    import tempfile
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REGISTRY[tokenizer_name])

    if lora:
        from peft import PeftModel
        # Merge on CPU, save to temp dir, reload cleanly to avoid
        # GPU memory fragmentation from base+adapter+merged coexisting.
        base_model = AutoModelForQuestionAnswering.from_pretrained(base_model_path)
        peft_model = PeftModel.from_pretrained(base_model, model_path)
        merged = peft_model.merge_and_unload()
        tmp_dir = tempfile.mkdtemp(prefix="lora_merged_")
        merged.save_pretrained(tmp_dir)
        del merged, peft_model, base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model = AutoModelForQuestionAnswering.from_pretrained(tmp_dir)
        logger.info("LoRA adapter merged, saved to %s, reloaded cleanly", tmp_dir)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    model = model.to(device)
    model.eval()
    return model, tokenizer


def prepare_examples(tokenizer, examples, max_seq_length=384):
    """Tokenize examples into individual model inputs (no batching)."""
    inputs_list = []
    for ex in examples:
        inputs = tokenizer(
            ex["question"],
            ex["context"],
            max_length=max_seq_length,
            truncation="only_second",
            return_tensors="pt",
        )
        inputs_list.append(inputs)
    return inputs_list


def prepare_batch(tokenizer, examples, max_seq_length=384, batch_size=32):
    """Tokenize examples into padded batches."""
    batches = []
    for i in range(0, len(examples), batch_size):
        batch_ex = examples[i:i + batch_size]
        questions = [ex["question"] for ex in batch_ex]
        contexts = [ex["context"] for ex in batch_ex]
        inputs = tokenizer(
            questions,
            contexts,
            max_length=max_seq_length,
            truncation="only_second",
            padding=True,
            return_tensors="pt",
        )
        batches.append((inputs, len(batch_ex)))
    return batches


def _warmup(model, inputs_list, device, n_warmup):
    """Run warmup passes to stabilize CUDA kernels and caches."""
    for idx in range(min(n_warmup, len(inputs_list))):
        inp = {k: v.to(device) for k, v in inputs_list[idx].items()}
        with torch.no_grad():
            model(**inp)
        if device.type == "cuda":
            torch.cuda.synchronize()


def benchmark_latency(model, inputs_list, device, n_warmup=10, n_runs=5):
    """Measure single-example latency (ms). Returns list of per-run medians."""
    # Initial warmup (critical after merge_and_unload)
    _warmup(model, inputs_list, device, n_warmup)

    run_medians = []

    for run in range(n_runs):
        # Per-run warmup (shorter)
        _warmup(model, inputs_list, device, min(5, n_warmup))

        # Measure
        latencies = []
        for inp_cpu in inputs_list:
            inp = {k: v.to(device) for k, v in inp_cpu.items()}

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                model(**inp)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)  # ms

        run_medians.append(statistics.median(latencies))

    return run_medians


def benchmark_throughput(model, batches, device, n_warmup=2, n_runs=5):
    """Measure batch throughput (samples/sec). Returns list of per-run values."""
    # Initial warmup
    for _ in range(max(n_warmup, 5)):
        inp = {k: v.to(device) for k, v in batches[0][0].items()}
        with torch.no_grad():
            model(**inp)
        if device.type == "cuda":
            torch.cuda.synchronize()

    run_throughputs = []

    for run in range(n_runs):

        # Measure all batches
        total_samples = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for batch_inputs, batch_size in batches:
            inp = {k: v.to(device) for k, v in batch_inputs.items()}
            with torch.no_grad():
                model(**inp)
            total_samples += batch_size

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        run_throughputs.append(total_samples / (t1 - t0))

    return run_throughputs


def benchmark_vram(model, inputs_list, device):
    """Measure peak VRAM during inference. Returns MB."""
    if device.type != "cuda":
        return None

    torch.cuda.reset_peak_memory_stats(device)

    # Run a few examples to get peak allocation
    for inp_cpu in inputs_list[:20]:
        inp = {k: v.to(device) for k, v in inp_cpu.items()}
        with torch.no_grad():
            model(**inp)
        torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return round(peak_mb)


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference performance")
    parser.add_argument("--model_path", required=True, help="Path to model dir")
    parser.add_argument(
        "--tokenizer", required=True, choices=list(MODEL_REGISTRY.keys()),
        help="Tokenizer key (for HF name lookup)",
    )
    parser.add_argument("--device", required=True, choices=["cuda", "cpu"])
    parser.add_argument("--lora", action="store_true", help="Load as LoRA adapter and merge")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Base model path for LoRA merging")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--label", type=str, default=None, help="Label for this config")
    args = parser.parse_args()

    if args.lora and not args.base_model_path:
        parser.error("--base_model_path is required when --lora is set")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = torch.device(args.device)
    label = args.label or Path(args.model_path).parent.name

    # Load model
    logger.info("Loading model from %s onto %s...", args.model_path, args.device)
    if args.device == "cpu":
        torch.set_num_threads(1)
        logger.info("CPU threads set to 1 for fair benchmarking")

    model, tokenizer = load_model(
        args.model_path, args.tokenizer, device,
        lora=args.lora, base_model_path=args.base_model_path,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %s params", f"{total_params:,}")

    # Load test examples
    test_path = DATA_DIR / "bioasq" / "test.json"
    examples = load_squad_examples(test_path)
    logger.info("Loaded %d test examples", len(examples))

    # Prepare inputs
    logger.info("Tokenizing...")
    inputs_list = prepare_examples(tokenizer, examples)
    batches = prepare_batch(tokenizer, examples, batch_size=args.batch_size)

    # Benchmark latency
    logger.info("Benchmarking latency (%d runs, %d warmup)...", args.n_runs, args.n_warmup)
    latency_medians = benchmark_latency(model, inputs_list, device, args.n_warmup, args.n_runs)
    lat_median = statistics.median(latency_medians)
    lat_std = statistics.stdev(latency_medians) if len(latency_medians) > 1 else 0.0
    logger.info("  Latency: %.2f ± %.2f ms", lat_median, lat_std)

    # Benchmark throughput
    logger.info("Benchmarking throughput (batch=%d, %d runs)...", args.batch_size, args.n_runs)
    throughputs = benchmark_throughput(model, batches, device, n_runs=args.n_runs)
    tp_median = statistics.median(throughputs)
    tp_std = statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
    logger.info("  Throughput: %.1f ± %.1f samples/s", tp_median, tp_std)

    # Benchmark VRAM
    vram_mb = benchmark_vram(model, inputs_list, device)
    if vram_mb is not None:
        logger.info("  Peak VRAM: %d MB", vram_mb)

    # Build results
    results = {
        "label": label,
        "model_path": args.model_path,
        "device": args.device,
        "lora_merged": args.lora,
        "total_params": total_params,
        "n_examples": len(examples),
        "n_warmup": args.n_warmup,
        "n_runs": args.n_runs,
        "batch_size": args.batch_size,
        "latency_ms": {
            "median": round(lat_median, 2),
            "std": round(lat_std, 2),
            "per_run_medians": [round(x, 2) for x in latency_medians],
        },
        "throughput_samples_per_sec": {
            "median": round(tp_median, 1),
            "std": round(tp_std, 1),
            "per_run": [round(x, 1) for x in throughputs],
        },
        "peak_vram_mb": vram_mb,
        "hardware": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
    }

    if device.type == "cuda":
        results["hardware"]["gpu"] = torch.cuda.get_device_name(0)
        results["hardware"]["cuda_version"] = torch.version.cuda

    if device.type == "cpu":
        results["hardware"]["num_threads"] = 1

    # Save
    output_path = args.output
    if output_path is None:
        output_dir = RESULTS_DIR / "e3_benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{label}_{args.device}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
