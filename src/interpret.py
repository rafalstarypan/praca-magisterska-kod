"""Interpretability module for E4: Attention + LIME + Faithfulness metrics.

Provides:
- Attention weight extraction and plausibility scoring
- LIME token-level explanations
- Faithfulness metrics: comprehensiveness and sufficiency (ERASER framework)
- Cross-method agreement: Spearman correlation between attention and LIME

Usage (standalone test):
    python src/interpret.py --model_path results/bioasq/pubmedbert/best_model \
        --tokenizer pubmedbert --question "What is the target of rapamycin?" \
        --context "The target of rapamycin is mTOR."
"""

import logging
import re
import string
from collections import defaultdict

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent

MODEL_REGISTRY = {
    "bert": "google-bert/bert-base-uncased",
    "biobert": "dmis-lab/biobert-v1.1",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "roberta": "FacebookAI/roberta-base",
    "distilbert": "distilbert/distilbert-base-uncased",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
}


# ---------------------------------------------------------------------------
# Model loading and prediction
# ---------------------------------------------------------------------------

def load_model(model_path, tokenizer_name, device="cpu"):
    """Load fine-tuned QA model and tokenizer.

    Uses attn_implementation='eager' so that output_attentions=True works
    (SDPA backend does not support returning attention weights).
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REGISTRY[tokenizer_name])
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path, attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def predict_answer(model, tokenizer, question, context, device="cpu",
                   return_logits=False):
    """Run QA inference. Returns predicted answer text and span indices.

    Returns:
        dict with keys: answer, start, end, score, (start_logits, end_logits)
    """
    inputs = tokenizer(
        question, context,
        return_tensors="pt",
        truncation=True,
        max_length=384,
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop("offset_mapping")[0]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0].cpu().numpy()
    end_logits = outputs.end_logits[0].cpu().numpy()

    # Find context token range (skip question + special tokens)
    input_ids = inputs["input_ids"][0].cpu()
    sep_positions = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]

    # For BERT-family: context starts after first SEP
    # For RoBERTa: context starts after second SEP (</s> </s>)
    if tokenizer.cls_token == "<s>":  # RoBERTa
        ctx_start = sep_positions[1].item() + 1 if len(sep_positions) > 1 else sep_positions[0].item() + 1
    else:  # BERT-family
        ctx_start = sep_positions[0].item() + 1

    ctx_end = len(input_ids) - 1  # before final SEP/PAD

    # Mask non-context positions
    masked_start = start_logits.copy()
    masked_end = end_logits.copy()
    masked_start[:ctx_start] = -1e10
    masked_start[ctx_end:] = -1e10
    masked_end[:ctx_start] = -1e10
    masked_end[ctx_end:] = -1e10

    best_start = int(np.argmax(masked_start))
    best_end = int(np.argmax(masked_end))
    if best_end < best_start:
        best_end = best_start

    # Map to character offsets
    char_start = offset_mapping[best_start][0].item()
    char_end = offset_mapping[best_end][1].item()
    answer = context[char_start:char_end]

    result = {
        "answer": answer,
        "start_token": best_start,
        "end_token": best_end,
        "ctx_start": ctx_start,
        "ctx_end": ctx_end,
        "score": float(start_logits[best_start] + end_logits[best_end]),
    }
    if return_logits:
        result["start_logits"] = start_logits
        result["end_logits"] = end_logits
    return result


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def get_attention_weights(model, tokenizer, question, context, device="cpu"):
    """Extract attention weights from all layers and heads.

    Returns:
        dict with:
        - attentions: np.array [n_layers, n_heads, seq_len, seq_len]
        - tokens: list of token strings
        - ctx_start, ctx_end: indices of context tokens
        - prediction: predicted answer dict
    """
    inputs = tokenizer(
        question, context,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Stack attentions: tuple of (batch, heads, seq, seq) -> (layers, heads, seq, seq)
    attentions = np.stack([a[0].cpu().numpy() for a in outputs.attentions])

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

    # Get prediction info
    pred = predict_answer(model, tokenizer, question, context, device)

    return {
        "attentions": attentions,
        "tokens": tokens,
        "ctx_start": pred["ctx_start"],
        "ctx_end": pred["ctx_end"],
        "prediction": pred,
    }


def attention_to_context(attentions, pred_start, pred_end, ctx_start, ctx_end):
    """Compute attention from predicted answer span to context tokens.

    Averages attention from predicted answer tokens across all heads in the
    last layer, returning a score per context token.

    Returns:
        np.array of shape (ctx_len,) — attention score per context token.
    """
    # Use last layer, average over heads
    last_layer = attentions[-1]  # (heads, seq, seq)
    avg_heads = last_layer.mean(axis=0)  # (seq, seq)

    # Attention FROM answer span tokens TO all context tokens
    answer_attn = avg_heads[pred_start:pred_end + 1, ctx_start:ctx_end]
    # Average over answer tokens
    ctx_scores = answer_attn.mean(axis=0)

    return ctx_scores


def attention_plausibility(ctx_attention_scores, tokens, ctx_start,
                           gold_answer, tokenizer, k=5):
    """Compute plausibility: overlap of top-k attention tokens with gold answer.

    Returns:
        dict with plausibility score (0-1), top_k_tokens, gold_tokens.
    """
    ctx_tokens = tokens[ctx_start:ctx_start + len(ctx_attention_scores)]

    # Top-k context tokens by attention
    top_k_idx = np.argsort(ctx_attention_scores)[-k:]
    top_k_tokens = [ctx_tokens[i] for i in top_k_idx]

    # Tokenize gold answer
    gold_tokens = tokenizer.tokenize(gold_answer.lower())

    # Normalize for comparison (strip ## prefix for BERT-family)
    def normalize(tok):
        tok = tok.lower().replace("##", "").replace("\u0120", "").strip()
        return tok

    top_k_norm = set(normalize(t) for t in top_k_tokens)
    gold_norm = set(normalize(t) for t in gold_tokens)

    if not gold_norm:
        return {"plausibility": 0.0, "top_k_tokens": top_k_tokens,
                "gold_tokens": gold_tokens}

    overlap = top_k_norm & gold_norm
    plausibility = len(overlap) / len(gold_norm)

    return {
        "plausibility": min(plausibility, 1.0),
        "top_k_tokens": top_k_tokens,
        "gold_tokens": gold_tokens,
        "overlap": list(overlap),
    }


# ---------------------------------------------------------------------------
# LIME explanations
# ---------------------------------------------------------------------------

def get_lime_explanation(model, tokenizer, question, context, device="cpu",
                         num_samples=2000):
    """Generate LIME explanation for context tokens.

    Perturbs context words and measures impact on prediction confidence.

    Returns:
        dict with:
        - importances: list of (word, score) sorted by importance
        - context_words: list of context words
        - prediction: predicted answer
    """
    from lime.lime_text import LimeTextExplainer

    pred = predict_answer(model, tokenizer, question, context, device,
                          return_logits=True)

    def predict_fn(perturbed_texts):
        """Score function for LIME: returns answer confidence for each text."""
        scores = []
        for text in perturbed_texts:
            inputs = tokenizer(
                question, text,
                return_tensors="pt",
                truncation=True,
                max_length=384,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)

            start_probs = torch.softmax(outputs.start_logits[0], dim=0)
            end_probs = torch.softmax(outputs.end_logits[0], dim=0)

            # Confidence = max start prob * max end prob
            confidence = float(start_probs.max() * end_probs.max())
            # Return as 2-class: [1-confidence, confidence]
            scores.append([1 - confidence, confidence])
        return np.array(scores)

    explainer = LimeTextExplainer(class_names=["low_conf", "high_conf"],
                                  split_expression=r'\s+')

    exp = explainer.explain_instance(
        context,
        predict_fn,
        num_features=len(context.split()),
        num_samples=num_samples,
        labels=(1,),
    )

    # Extract importances for label=1 (high confidence)
    importances = exp.as_list(label=1)

    return {
        "importances": importances,
        "context_words": context.split(),
        "prediction": pred,
    }


def lime_plausibility(importances, gold_answer, k=5):
    """Compute plausibility: overlap of top-k LIME words with gold answer.

    Returns:
        dict with plausibility score (0-1), top_k_words, gold_words.
    """
    # Sort by absolute importance, take top-k
    sorted_imp = sorted(importances, key=lambda x: abs(x[1]), reverse=True)
    top_k = sorted_imp[:k]
    top_k_words = [w for w, _ in top_k]

    def normalize(s):
        s = s.lower()
        s = "".join(ch for ch in s if ch not in string.punctuation)
        return s.strip()

    top_k_norm = set(normalize(w) for w in top_k_words if normalize(w))
    gold_words = gold_answer.lower().split()
    gold_norm = set(normalize(w) for w in gold_words if normalize(w))

    if not gold_norm:
        return {"plausibility": 0.0, "top_k_words": top_k_words,
                "gold_words": gold_words}

    overlap = top_k_norm & gold_norm
    plausibility = len(overlap) / len(gold_norm)

    return {
        "plausibility": min(plausibility, 1.0),
        "top_k_words": top_k_words,
        "gold_words": list(gold_norm),
        "overlap": list(overlap),
    }


# ---------------------------------------------------------------------------
# Faithfulness metrics (ERASER framework)
# ---------------------------------------------------------------------------

def _get_answer_confidence(model, tokenizer, question, context, device="cpu"):
    """Get answer confidence as max(softmax start) * max(softmax end)."""
    inputs = tokenizer(
        question, context,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    start_probs = torch.softmax(outputs.start_logits[0], dim=0)
    end_probs = torch.softmax(outputs.end_logits[0], dim=0)
    return float(start_probs.max() * end_probs.max())


def compute_comprehensiveness(model, tokenizer, question, context,
                               important_words, device="cpu", k=5):
    """Comprehensiveness: remove top-k important words, measure confidence drop.

    comp = P(answer|original) - P(answer|masked)
    Higher = more faithful (important words truly matter).
    """
    original_conf = _get_answer_confidence(model, tokenizer, question, context,
                                           device)

    # Remove top-k words from context
    words = context.split()
    top_k_words = set(w.lower() for w, _ in
                      sorted(important_words, key=lambda x: abs(x[1]),
                             reverse=True)[:k])

    masked_words = [w for w in words if w.lower() not in top_k_words]
    masked_context = " ".join(masked_words) if masked_words else "[EMPTY]"

    masked_conf = _get_answer_confidence(model, tokenizer, question,
                                         masked_context, device)

    return {
        "comprehensiveness": original_conf - masked_conf,
        "original_confidence": original_conf,
        "masked_confidence": masked_conf,
        "removed_words": list(top_k_words),
    }


def compute_sufficiency(model, tokenizer, question, context,
                         important_words, device="cpu", k=5):
    """Sufficiency: keep only top-k important words, measure confidence retention.

    suff = P(answer|original) - P(answer|only_important)
    Lower = more faithful (important words alone are sufficient).
    """
    original_conf = _get_answer_confidence(model, tokenizer, question, context,
                                           device)

    # Keep only top-k words
    words = context.split()
    top_k_words = set(w.lower() for w, _ in
                      sorted(important_words, key=lambda x: abs(x[1]),
                             reverse=True)[:k])

    kept_words = [w for w in words if w.lower() in top_k_words]
    kept_context = " ".join(kept_words) if kept_words else "[EMPTY]"

    kept_conf = _get_answer_confidence(model, tokenizer, question,
                                       kept_context, device)

    return {
        "sufficiency": original_conf - kept_conf,
        "original_confidence": original_conf,
        "kept_confidence": kept_conf,
        "kept_words": list(top_k_words),
    }


# ---------------------------------------------------------------------------
# Cross-method agreement
# ---------------------------------------------------------------------------

def cross_method_agreement(attention_scores, lime_importances, context_words):
    """Spearman correlation between attention and LIME scores per context token.

    Maps LIME word-level scores to token-level by matching words to context
    positions, then computes rank correlation with attention scores.

    Returns:
        dict with spearman_r, p_value, n_matched.
    """
    # Build LIME word -> score mapping
    lime_map = {w.lower(): s for w, s in lime_importances}

    # Match context words to LIME scores
    n_ctx = len(attention_scores)
    lime_scores = []
    attn_scores = []

    for i, word in enumerate(context_words[:n_ctx]):
        score = lime_map.get(word.lower())
        if score is not None and i < len(attention_scores):
            lime_scores.append(abs(score))
            attn_scores.append(attention_scores[i])

    if len(lime_scores) < 3:
        return {"spearman_r": 0.0, "p_value": 1.0, "n_matched": len(lime_scores)}

    r, p = stats.spearmanr(attn_scores, lime_scores)

    return {
        "spearman_r": float(r) if not np.isnan(r) else 0.0,
        "p_value": float(p) if not np.isnan(p) else 1.0,
        "n_matched": len(lime_scores),
    }


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick interpretability test")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model, tokenizer = load_model(args.model_path, args.tokenizer, args.device)

    # Prediction
    pred = predict_answer(model, tokenizer, args.question, args.context,
                          args.device)
    print(f"\nPrediction: '{pred['answer']}' (score={pred['score']:.2f})")

    # Attention
    attn = get_attention_weights(model, tokenizer, args.question, args.context,
                                 args.device)
    ctx_scores = attention_to_context(
        attn["attentions"], pred["start_token"], pred["end_token"],
        attn["ctx_start"], attn["ctx_end"],
    )
    print(f"\nTop-5 attention tokens: ", end="")
    top5 = np.argsort(ctx_scores)[-5:][::-1]
    ctx_tokens = attn["tokens"][attn["ctx_start"]:attn["ctx_end"]]
    for i in top5:
        if i < len(ctx_tokens):
            print(f"{ctx_tokens[i]}({ctx_scores[i]:.3f}) ", end="")
    print()

    # LIME
    lime_result = get_lime_explanation(model, tokenizer, args.question,
                                       args.context, args.device,
                                       num_samples=500)
    print(f"\nTop-5 LIME words:")
    for w, s in sorted(lime_result["importances"],
                       key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {w}: {s:.4f}")
