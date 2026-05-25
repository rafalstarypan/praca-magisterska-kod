"""E4 experiment runner: Interpretability (Attention + LIME + Faithfulness).

Runs attention analysis and LIME on BioASQ test examples for BERT (baseline)
and PubMedBERT (best domain model), as specified in the binding plan.

Usage:
    python src/run_e4.py                     # full run (default)
    python src/run_e4.py --n_attention 5     # quick test with fewer examples
    python src/run_e4.py --device cpu        # run on CPU
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_squad_examples
from interpret import (
    load_model,
    predict_answer,
    get_attention_weights,
    attention_to_context,
    attention_to_context_per_layer,
    attention_plausibility,
    get_lime_explanation,
    lime_plausibility,
    compute_comprehensiveness,
    compute_sufficiency,
    cross_method_agreement,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "e4_interpretability_v2"

# All 6 models from E1 (with SQuAD pretraining — best variants)
MODELS = {
    "bert": {
        "model_path": "results/bioasq/bert/best_model",
        "tokenizer": "bert",
        "label": "BERT (general)",
    },
    "biobert": {
        "model_path": "results/bioasq/biobert/best_model",
        "tokenizer": "biobert",
        "label": "BioBERT (domain)",
    },
    "pubmedbert": {
        "model_path": "results/bioasq/pubmedbert/best_model",
        "tokenizer": "pubmedbert",
        "label": "PubMedBERT (domain)",
    },
    "roberta": {
        "model_path": "results/bioasq/roberta/best_model",
        "tokenizer": "roberta",
        "label": "RoBERTa (general, BPE)",
    },
    "distilbert": {
        "model_path": "results/bioasq/distilbert/best_model",
        "tokenizer": "distilbert",
        "label": "DistilBERT (distilled)",
    },
    "clinicalbert": {
        "model_path": "results/bioasq/clinicalbert/best_model",
        "tokenizer": "clinicalbert",
        "label": "ClinicalBERT (clinical)",
    },
}


def run_attention_analysis(model, tokenizer, examples, model_name, device,
                            n_examples=30):
    """Run attention analysis on n_examples from BioASQ test set."""
    logger.info("Attention analysis: %s (%d examples)", model_name, n_examples)
    results = []

    for i, ex in enumerate(examples[:n_examples]):
        question = ex["question"]
        context = ex["context"]
        gold_answers = ex["answers"]["text"]

        if not gold_answers:
            continue

        try:
            attn = get_attention_weights(model, tokenizer, question, context,
                                         device)
            pred = attn["prediction"]

            ctx_scores = attention_to_context(
                attn["attentions"],
                pred["start_token"], pred["end_token"],
                attn["ctx_start"], attn["ctx_end"],
            )

            plaus = attention_plausibility(
                ctx_scores, attn["tokens"], attn["ctx_start"],
                gold_answers[0], tokenizer, k=5,
            )

            # Per-layer plausibility (Extension A)
            per_layer_scores = attention_to_context_per_layer(
                attn["attentions"],
                pred["start_token"], pred["end_token"],
                attn["ctx_start"], attn["ctx_end"],
            )
            n_layers = per_layer_scores.shape[0]
            per_layer_plaus = []
            for layer_idx in range(n_layers):
                lp = attention_plausibility(
                    per_layer_scores[layer_idx], attn["tokens"],
                    attn["ctx_start"], gold_answers[0], tokenizer, k=5,
                )
                per_layer_plaus.append(lp["plausibility"])

            results.append({
                "example_id": ex["id"],
                "question": question,
                "context": context[:200],
                "gold_answer": gold_answers[0],
                "predicted_answer": pred["answer"],
                "plausibility": plaus["plausibility"],
                "top_k_tokens": plaus["top_k_tokens"],
                "gold_tokens": plaus["gold_tokens"],
                "overlap": plaus.get("overlap", []),
                "per_layer_plausibility": per_layer_plaus,
            })

            if (i + 1) % 10 == 0:
                logger.info("  [%d/%d] done", i + 1, n_examples)

        except Exception as e:
            logger.warning("  Skipping example %s: %s", ex["id"], e)
            continue

    return results


def run_lime_analysis(model, tokenizer, examples, model_name, device,
                       n_examples=50, num_samples=2000):
    """Run LIME analysis on n_examples from BioASQ test set."""
    logger.info("LIME analysis: %s (%d examples, %d perturbations each)",
                model_name, n_examples, num_samples)
    results = []

    for i, ex in enumerate(examples[:n_examples]):
        question = ex["question"]
        context = ex["context"]
        gold_answers = ex["answers"]["text"]

        if not gold_answers:
            continue

        try:
            t0 = time.perf_counter()
            lime_result = get_lime_explanation(
                model, tokenizer, question, context, device,
                num_samples=num_samples,
            )
            elapsed = time.perf_counter() - t0

            plaus = lime_plausibility(lime_result["importances"],
                                       gold_answers[0], k=5)

            results.append({
                "example_id": ex["id"],
                "question": question,
                "context": context[:200],
                "gold_answer": gold_answers[0],
                "predicted_answer": lime_result["prediction"]["answer"],
                "plausibility": plaus["plausibility"],
                "top_k_words": plaus["top_k_words"],
                "gold_words": plaus["gold_words"],
                "overlap": plaus.get("overlap", []),
                "top_importances": lime_result["importances"][:10],
                "time_seconds": round(elapsed, 1),
            })

            if (i + 1) % 5 == 0:
                logger.info("  [%d/%d] done (%.1fs)", i + 1, n_examples,
                            elapsed)

        except Exception as e:
            logger.warning("  Skipping example %s: %s", ex["id"], e)
            continue

    return results


def run_faithfulness(model, tokenizer, examples, lime_results, model_name,
                      device, n_examples=50, k_values=(3, 5, 10)):
    """Compute comprehensiveness and sufficiency for multiple k values.

    Extension B: evaluates faithfulness at k=3, k=5, k=10 to show
    sensitivity of metrics to the choice of k.
    """
    logger.info("Faithfulness metrics: %s (%d examples, k=%s)",
                model_name, n_examples, k_values)

    lime_by_id = {r["example_id"]: r for r in lime_results}

    # Collect per-example scores for each k
    per_k = {k: {"comp": [], "suff": []} for k in k_values}

    for ex in examples[:n_examples]:
        if ex["id"] not in lime_by_id:
            continue

        lime_r = lime_by_id[ex["id"]]
        importances = lime_r.get("top_importances", [])
        if not importances:
            continue

        try:
            for k in k_values:
                comp = compute_comprehensiveness(
                    model, tokenizer, ex["question"], ex["context"],
                    importances, device, k=k,
                )
                suff = compute_sufficiency(
                    model, tokenizer, ex["question"], ex["context"],
                    importances, device, k=k,
                )
                per_k[k]["comp"].append(comp["comprehensiveness"])
                per_k[k]["suff"].append(suff["sufficiency"])

        except Exception as e:
            logger.warning("  Skipping %s: %s", ex["id"], e)
            continue

    result = {}
    for k in k_values:
        cs = per_k[k]["comp"]
        ss = per_k[k]["suff"]
        result[f"k={k}"] = {
            "comprehensiveness_mean": float(np.mean(cs)) if cs else 0,
            "comprehensiveness_std": float(np.std(cs)) if cs else 0,
            "sufficiency_mean": float(np.mean(ss)) if ss else 0,
            "sufficiency_std": float(np.std(ss)) if ss else 0,
            "n_examples": len(cs),
        }
        logger.info("  k=%d: comp=%.4f +/- %.4f, suff=%.4f +/- %.4f (n=%d)",
                    k,
                    result[f"k={k}"]["comprehensiveness_mean"],
                    result[f"k={k}"]["comprehensiveness_std"],
                    result[f"k={k}"]["sufficiency_mean"],
                    result[f"k={k}"]["sufficiency_std"],
                    len(cs))

    # Keep backwards compatibility: top-level keys from k=5
    k5 = result.get("k=5", {})
    result["comprehensiveness_mean"] = k5.get("comprehensiveness_mean", 0)
    result["comprehensiveness_std"] = k5.get("comprehensiveness_std", 0)
    result["sufficiency_mean"] = k5.get("sufficiency_mean", 0)
    result["sufficiency_std"] = k5.get("sufficiency_std", 0)
    result["n_examples"] = k5.get("n_examples", 0)

    return result


def run_cross_agreement(attn_results, lime_results, model, tokenizer,
                         examples, model_name, device):
    """Compute Spearman correlation between attention and LIME for overlapping examples."""
    logger.info("Cross-method agreement: %s", model_name)

    attn_by_id = {r["example_id"]: r for r in attn_results}
    lime_by_id = {r["example_id"]: r for r in lime_results}

    common_ids = set(attn_by_id.keys()) & set(lime_by_id.keys())
    examples_by_id = {ex["id"]: ex for ex in examples}

    correlations = []
    for eid in common_ids:
        ex = examples_by_id.get(eid)
        if ex is None:
            continue

        # Re-compute attention scores for this example
        try:
            attn = get_attention_weights(model, tokenizer, ex["question"],
                                         ex["context"], device)
            pred = attn["prediction"]
            ctx_scores = attention_to_context(
                attn["attentions"],
                pred["start_token"], pred["end_token"],
                attn["ctx_start"], attn["ctx_end"],
            )

            lime_imp = lime_by_id[eid].get("top_importances", [])
            context_words = ex["context"].split()

            agreement = cross_method_agreement(ctx_scores, lime_imp,
                                                context_words)
            correlations.append(agreement["spearman_r"])
        except Exception as e:
            logger.warning("  Skipping %s: %s", eid, e)

    return {
        "mean_spearman_r": float(np.mean(correlations)) if correlations else 0,
        "std_spearman_r": float(np.std(correlations)) if correlations else 0,
        "n_examples": len(correlations),
        "per_example": correlations,
    }


def main():
    parser = argparse.ArgumentParser(description="E4: Interpretability experiments")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_attention", type=int, default=82,
                        help="Number of examples for attention analysis")
    parser.add_argument("--n_lime", type=int, default=82,
                        help="Number of examples for LIME analysis")
    parser.add_argument("--lime_samples", type=int, default=2000,
                        help="Number of LIME perturbation samples")
    parser.add_argument("--k", type=int, default=5,
                        help="Top-k for plausibility and faithfulness")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load BioASQ test data
    data_path = PROJECT_ROOT / "data" / "processed" / "bioasq" / "test.json"
    examples = load_squad_examples(str(data_path))
    logger.info("Loaded %d BioASQ test examples", len(examples))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_key, model_cfg in MODELS.items():
        logger.info("=" * 60)
        logger.info("Processing model: %s (%s)", model_key, model_cfg["label"])
        logger.info("=" * 60)

        model_path = str(PROJECT_ROOT / model_cfg["model_path"])
        model, tokenizer = load_model(model_path, model_cfg["tokenizer"],
                                       args.device)

        model_dir = RESULTS_DIR / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        # 1. Attention analysis
        attn_results = run_attention_analysis(
            model, tokenizer, examples, model_key, args.device,
            n_examples=args.n_attention,
        )
        attn_plausibility_scores = [r["plausibility"] for r in attn_results]
        logger.info("Attention plausibility: %.3f +/- %.3f (n=%d)",
                    np.mean(attn_plausibility_scores),
                    np.std(attn_plausibility_scores),
                    len(attn_plausibility_scores))

        with open(model_dir / "attention_results.json", "w",
                  encoding="utf-8") as f:
            json.dump(attn_results, f, indent=2, ensure_ascii=False)

        # 2. LIME analysis
        lime_results = run_lime_analysis(
            model, tokenizer, examples, model_key, args.device,
            n_examples=args.n_lime,
            num_samples=args.lime_samples,
        )
        lime_plausibility_scores = [r["plausibility"] for r in lime_results]
        logger.info("LIME plausibility: %.3f +/- %.3f (n=%d)",
                    np.mean(lime_plausibility_scores),
                    np.std(lime_plausibility_scores),
                    len(lime_plausibility_scores))

        with open(model_dir / "lime_results.json", "w",
                  encoding="utf-8") as f:
            json.dump(lime_results, f, indent=2, ensure_ascii=False)

        # 3. Faithfulness (uses LIME importances) — multi-k (Extension B)
        faith = run_faithfulness(
            model, tokenizer, examples, lime_results, model_key,
            args.device, n_examples=args.n_lime,
        )

        with open(model_dir / "faithfulness.json", "w",
                  encoding="utf-8") as f:
            json.dump(faith, f, indent=2, ensure_ascii=False)

        # 4. Cross-method agreement (attention vs LIME)
        agreement = run_cross_agreement(
            attn_results, lime_results, model, tokenizer, examples,
            model_key, args.device,
        )
        logger.info("Cross-method Spearman: %.3f +/- %.3f (n=%d)",
                    agreement["mean_spearman_r"],
                    agreement["std_spearman_r"],
                    agreement["n_examples"])

        with open(model_dir / "cross_agreement.json", "w",
                  encoding="utf-8") as f:
            json.dump(agreement, f, indent=2, ensure_ascii=False)

        # Per-layer plausibility aggregation (Extension A)
        per_layer_all = [r["per_layer_plausibility"] for r in attn_results
                         if "per_layer_plausibility" in r]
        if per_layer_all:
            per_layer_array = np.array(per_layer_all)  # (n_examples, n_layers)
            per_layer_mean = per_layer_array.mean(axis=0).tolist()
            per_layer_std = per_layer_array.std(axis=0).tolist()
            logger.info("Per-layer plausibility (mean): %s",
                        [f"{v:.3f}" for v in per_layer_mean])
        else:
            per_layer_mean = []
            per_layer_std = []

        # Save per-layer results
        with open(model_dir / "per_layer_plausibility.json", "w",
                  encoding="utf-8") as f:
            json.dump({"mean": per_layer_mean, "std": per_layer_std,
                        "n_examples": len(per_layer_all)},
                      f, indent=2, ensure_ascii=False)

        # Model summary
        model_summary = {
            "model": model_key,
            "label": model_cfg["label"],
            "attention_plausibility_mean": float(np.mean(attn_plausibility_scores)),
            "attention_plausibility_std": float(np.std(attn_plausibility_scores)),
            "per_layer_plausibility_mean": per_layer_mean,
            "lime_plausibility_mean": float(np.mean(lime_plausibility_scores)),
            "lime_plausibility_std": float(np.std(lime_plausibility_scores)),
            **faith,
            "cross_agreement_spearman": agreement["mean_spearman_r"],
            "cross_agreement_std": agreement["std_spearman_r"],
            "n_attention": len(attn_results),
            "n_lime": len(lime_results),
        }
        all_results[model_key] = model_summary

        with open(model_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(model_summary, f, indent=2, ensure_ascii=False)

        # Free GPU memory before loading next model
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Global summary
    with open(RESULTS_DIR / "e4_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary table
    model_order = ["bert", "biobert", "pubmedbert", "roberta", "distilbert",
                   "clinicalbert"]
    present = [m for m in model_order if m in all_results]
    short_names = {
        "bert": "BERT", "biobert": "BioBERT", "pubmedbert": "PubMedBERT",
        "roberta": "RoBERTa", "distilbert": "DistilBERT",
        "clinicalbert": "ClinicalBERT",
    }

    col_w = 13
    print("\n" + "=" * (30 + col_w * len(present)))
    print("E4 INTERPRETABILITY - SUMMARY")
    print("=" * (30 + col_w * len(present)))
    header = f"{'Metric':<30}" + "".join(f"{short_names[m]:>{col_w}}" for m in present)
    print(header)
    print("-" * (30 + col_w * len(present)))

    rows = [
        ("Attn Plausibility", "attention_plausibility_mean"),
        ("LIME Plausibility", "lime_plausibility_mean"),
        ("Comprehensiveness", "comprehensiveness_mean"),
        ("Sufficiency", "sufficiency_mean"),
        ("Spearman (Attn/LIME)", "cross_agreement_spearman"),
    ]
    for label, key in rows:
        vals = "".join(f"{all_results[m].get(key, 0):>{col_w}.4f}" for m in present)
        print(f"{label:<30}{vals}")
    print("=" * (30 + col_w * len(present)))

    logger.info("All results saved to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
