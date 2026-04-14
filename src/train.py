"""Training script for extractive QA experiments.

Usage:
    # Phase 1 — SQuAD pretraining (once per model):
    python src/train.py --model bert --dataset squad --epochs 2

    # Phase 2 — fine-tuning with SQuAD-pretrained model:
    python src/train.py --model bert --dataset bioasq --pretrained_from results/squad/bert/best_model
    python src/train.py --model bert --dataset covidqa --fold 0 --pretrained_from results/squad/bert/best_model

    # Direct fine-tuning (no SQuAD pretraining):
    python src/train.py --model bert --dataset bioasq
    python src/train.py --model biobert --dataset covidqa --fold 0 --no_tb

    # E2 — LoRA fine-tuning:
    python src/train.py --model pubmedbert --dataset bioasq --lora --pretrained_from results/squad/pubmedbert/best_model
    python src/train.py --model pubmedbert --dataset bioasq --lora --lora_r 4 --pretrained_from ...
"""

import os

# Fix Python 3.12 + multiprocess cleanup error (_recursion_count)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import platform
import sys
import time
from functools import partial
from pathlib import Path

import yaml
from datasets import Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data import load_squad_examples, tokenize_examples
from evaluate import evaluate_predictions, postprocess_qa_predictions

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"

MODEL_REGISTRY = {
    "bert": "google-bert/bert-base-uncased",
    "biobert": "dmis-lab/biobert-v1.1",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "roberta": "FacebookAI/roberta-base",
    "distilbert": "distilbert/distilbert-base-uncased",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
}

# Columns added by tokenize_examples that the model doesn't need as input
EXTRA_COLUMNS = ["example_id", "offset_mapping"]
LABEL_COLUMNS = ["start_positions", "end_positions"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(lora=False):
    """Load base training config from YAML, optionally merging LoRA overrides."""
    with open(CONFIGS_DIR / "base.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if lora:
        with open(CONFIGS_DIR / "lora.yaml", encoding="utf-8") as f:
            lora_cfg = yaml.safe_load(f)
        config.update(lora_cfg)
    return config


def get_data_paths(dataset_name, fold=None):
    """Return dict with train/eval file paths (and test if available)."""
    if dataset_name == "squad":
        base = DATA_DIR / "squad"
        return {
            "train": base / "train.json",
            "eval": base / "dev.json",
        }
    if dataset_name == "bioasq":
        base = DATA_DIR / "bioasq"
        return {
            "train": base / "train.json",
            "eval": base / "dev.json",
            "test": base / "test.json",
        }
    if dataset_name == "covidqa":
        if fold is None:
            raise ValueError("COVID-QA requires --fold (0-4)")
        base = DATA_DIR / "covidqa"
        return {
            "train": base / f"fold_{fold}" / "train.json",
            "eval": base / f"fold_{fold}" / "val.json",
            "test": base / "test.json",
        }
    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_output_dir(model_key, dataset_name, fold=None, pretrained_from=None,
                   lora=False, lora_r=None):
    """Build output directory path.

    When pretrained_from is None (direct fine-tuning), appends '_no_squad'
    to distinguish from the default two-stage pipeline.
    For squad dataset itself, no suffix is added.
    LoRA runs get a '_lora_rN' suffix.
    """
    dir_name = model_key
    if dataset_name != "squad" and pretrained_from is None:
        dir_name = f"{model_key}_no_squad"
    if lora:
        dir_name = f"{dir_name}_lora_r{lora_r}"
    path = RESULTS_DIR / dataset_name / dir_name
    if fold is not None:
        path = path / f"fold_{fold}"
    return path


def count_parameters(model):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def to_serializable(obj):
    """Recursively convert numpy/torch types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train extractive QA model")
    parser.add_argument(
        "--model", required=True, choices=list(MODEL_REGISTRY.keys()),
        help="Model key from registry",
    )
    parser.add_argument(
        "--dataset", required=True, choices=["bioasq", "covidqa", "squad"],
        help="Dataset to train on",
    )
    parser.add_argument("--fold", type=int, default=None, help="CV fold for COVID-QA (0-4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_tb", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument(
        "--pretrained_from", type=str, default=None,
        help="Path to pretrained model dir (e.g., results/squad/bert/best_model)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override num_train_epochs from config",
    )
    parser.add_argument(
        "--lora", action="store_true",
        help="Use LoRA (PEFT) instead of full fine-tuning",
    )
    parser.add_argument(
        "--lora_r", type=int, default=None,
        help="Override LoRA rank (default from lora.yaml)",
    )
    args = parser.parse_args()

    if args.dataset == "covidqa" and args.fold is None:
        parser.error("COVID-QA requires --fold (0-4)")
    if args.dataset == "squad" and args.fold is not None:
        parser.error("SQuAD does not use folds")

    # ---- config & seed ----
    config = load_config(lora=args.lora)
    if args.epochs is not None:
        config["num_train_epochs"] = args.epochs
    # Allow CLI override of LoRA rank
    if args.lora_r is not None:
        config["lora_r"] = args.lora_r
    lora_r = config.get("lora_r", 8) if args.lora else None
    set_seed(args.seed)

    model_hf_name = MODEL_REGISTRY[args.model]
    model_load_path = args.pretrained_from or model_hf_name
    data_paths = get_data_paths(args.dataset, args.fold)
    output_dir = Path(args.output_dir) if args.output_dir else get_output_dir(
        args.model, args.dataset, args.fold, args.pretrained_from,
        lora=args.lora, lora_r=lora_r,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    run_name = f"{args.model}_{args.dataset}"
    if args.fold is not None:
        run_name += f"_fold{args.fold}"

    report_to = "none"
    tb_log_dir = None
    if not args.no_tb:
        tb_log_dir = str(output_dir / "tensorboard")
        report_to = "tensorboard"
        logger.info("TensorBoard logs: %s", tb_log_dir)

    # ---- model & tokenizer ----
    logger.info("Loading model: %s", model_load_path)
    tokenizer = AutoTokenizer.from_pretrained(model_hf_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_load_path)

    # ---- LoRA wrapping (E2) ----
    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_config = LoraConfig(
            task_type=TaskType.QUESTION_ANS,
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["lora_target_modules"],
        )
        model = get_peft_model(model, lora_config)
        logger.info("LoRA enabled: r=%d, alpha=%d, targets=%s",
                     config["lora_r"], config["lora_alpha"], config["lora_target_modules"])
        model.print_trainable_parameters()

    total_params, trainable_params = count_parameters(model)
    logger.info("Parameters: %s total, %s trainable", f"{total_params:,}", f"{trainable_params:,}")

    # ---- load examples ----
    logger.info("Loading data...")
    train_examples = load_squad_examples(data_paths["train"])
    eval_examples = load_squad_examples(data_paths["eval"])
    logger.info("  Train: %d examples, Eval: %d examples", len(train_examples), len(eval_examples))

    # ---- tokenize ----
    tokenize_fn = partial(
        tokenize_examples,
        tokenizer=tokenizer,
        max_seq_length=config["max_seq_length"],
        doc_stride=config["doc_stride"],
    )
    orig_columns = ["id", "question", "context", "answers"]

    logger.info("Tokenizing...")
    train_all = Dataset.from_list(train_examples).map(
        tokenize_fn, batched=True, remove_columns=orig_columns,
    )
    eval_all = Dataset.from_list(eval_examples).map(
        tokenize_fn, batched=True, remove_columns=orig_columns,
    )
    logger.info("  Train features: %d, Eval features: %d", len(train_all), len(eval_all))

    # Views: Trainer gets model-input + labels; post-processing keeps metadata
    train_dataset = train_all.remove_columns(EXTRA_COLUMNS)
    eval_for_trainer = eval_all.remove_columns(EXTRA_COLUMNS)
    eval_for_postprocess = eval_all

    # ---- compute_metrics closure (used by Trainer during eval) ----
    def compute_metrics(eval_pred):
        preds = postprocess_qa_predictions(
            eval_examples, eval_for_postprocess, eval_pred.predictions,
            n_best_size=config["n_best_size"],
            max_answer_length=config["max_answer_length"],
        )
        return evaluate_predictions(preds, eval_examples)

    # ---- training arguments ----
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        fp16=config["fp16"],
        seed=args.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=config["logging_steps"],
        report_to=report_to,
        logging_dir=tb_log_dir,
        save_total_limit=1,
    )

    # ---- train ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_for_trainer,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ---- hardware info ----
    import torch
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info["device"] = torch.cuda.get_device_name(0)
        gpu_info["cuda_version"] = torch.version.cuda
        torch.cuda.reset_peak_memory_stats(0)

    logger.info("Starting training...")
    wall_start = time.perf_counter()
    train_result = trainer.train()
    wall_elapsed = time.perf_counter() - wall_start
    train_metrics = train_result.metrics
    logger.info("Training done. Loss: %.4f", train_metrics.get("train_loss", float("nan")))

    if torch.cuda.is_available():
        gpu_info["peak_vram_mb"] = round(torch.cuda.max_memory_allocated(0) / 1024 / 1024)
        gpu_info["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved(0) / 1024 / 1024)
        logger.info("Peak VRAM: %d MB allocated, %d MB reserved",
                     gpu_info["peak_vram_mb"], gpu_info["peak_vram_reserved_mb"])

    # Save best model
    if args.lora:
        # Save only LoRA adapter weights (~1-3 MB)
        model.save_pretrained(str(output_dir / "best_model"))
        tokenizer.save_pretrained(str(output_dir / "best_model"))
    else:
        trainer.save_model(str(output_dir / "best_model"))

    # ---- final eval on eval set (best model is already loaded) ----
    logger.info("Final evaluation on eval set...")
    eval_result = trainer.predict(eval_for_trainer)
    eval_preds = postprocess_qa_predictions(
        eval_examples, eval_for_postprocess, eval_result.predictions,
        config["n_best_size"], config["max_answer_length"],
    )
    eval_metrics = evaluate_predictions(eval_preds, eval_examples)
    logger.info("  Eval  EM=%.2f  F1=%.2f", eval_metrics["exact_match"], eval_metrics["f1"])

    # ---- eval on test set (if available) ----
    test_metrics = {}
    test_preds = {}
    if "test" in data_paths:
        logger.info("Evaluating on test set...")
        test_examples = load_squad_examples(data_paths["test"])
        test_all = Dataset.from_list(test_examples).map(
            tokenize_fn, batched=True, remove_columns=orig_columns,
        )
        # No labels -> Trainer won't call compute_metrics (avoids closure mismatch)
        test_for_trainer = test_all.remove_columns(EXTRA_COLUMNS + LABEL_COLUMNS)
        test_for_postprocess = test_all

        test_result = trainer.predict(test_for_trainer)
        test_preds = postprocess_qa_predictions(
            test_examples, test_for_postprocess, test_result.predictions,
            config["n_best_size"], config["max_answer_length"],
        )
        test_metrics = evaluate_predictions(test_preds, test_examples)
        logger.info("  Test  EM=%.2f  F1=%.2f", test_metrics["exact_match"], test_metrics["f1"])

    # ---- save results ----
    hardware_info = {
        "wall_time_seconds": round(wall_elapsed, 1),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        **gpu_info,
    }

    lora_info = None
    if args.lora:
        lora_info = {
            "r": config["lora_r"],
            "alpha": config["lora_alpha"],
            "dropout": config["lora_dropout"],
            "target_modules": config["lora_target_modules"],
        }

    results = to_serializable({
        "model": args.model,
        "model_hf_name": model_hf_name,
        "pretrained_from": args.pretrained_from,
        "dataset": args.dataset,
        "fold": args.fold,
        "seed": args.seed,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "lora": lora_info,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "test_metrics": test_metrics,
        "hardware": hardware_info,
        "config": config,
    })

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_dir / "eval_predictions.json", "w", encoding="utf-8") as f:
        json.dump(eval_preds, f, indent=2, ensure_ascii=False)

    if test_preds:
        with open(output_dir / "test_predictions.json", "w", encoding="utf-8") as f:
            json.dump(test_preds, f, indent=2, ensure_ascii=False)

    logger.info("All results saved to %s", output_dir)
    if tb_log_dir:
        logger.info("View TensorBoard: tensorboard --logdir %s", tb_log_dir)


if __name__ == "__main__":
    main()
