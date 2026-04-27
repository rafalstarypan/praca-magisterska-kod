"""Gradio demo for biomedical extractive QA.

Usage:
    python app/demo.py
    # Opens browser at http://localhost:7860
"""

import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Available models (lazy-loaded)
MODELS = {
    "PubMedBERT (best)": ("results/bioasq/pubmedbert/best_model", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"),
    "RoBERTa": ("results/bioasq/roberta/best_model", "FacebookAI/roberta-base"),
    "BioBERT": ("results/bioasq/biobert/best_model", "dmis-lab/biobert-v1.1"),
    "BERT (baseline)": ("results/bioasq/bert/best_model", "google-bert/bert-base-uncased"),
    "DistilBERT (fast)": ("results/bioasq/distilbert/best_model", "distilbert/distilbert-base-uncased"),
    "ClinicalBERT": ("results/bioasq/clinicalbert/best_model", "emilyalsentzer/Bio_ClinicalBERT"),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_cache = {}  # model_name -> (model, tokenizer)


def get_model(model_name):
    """Load model and tokenizer (cached after first call)."""
    if model_name not in _cache:
        model_path, tok_name = MODELS[model_name]
        full_path = str(PROJECT_ROOT / model_path)
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        model = AutoModelForQuestionAnswering.from_pretrained(full_path)
        model.to(DEVICE).eval()
        _cache[model_name] = (model, tokenizer)
    return _cache[model_name]


def answer_question(model_name, context, question):
    """Run extractive QA and return answer, highlighted context, confidence."""
    if not context.strip() or not question.strip():
        return "", [], 0.0

    model, tokenizer = get_model(model_name)

    inputs = tokenizer(
        question, context,
        return_tensors="pt",
        truncation=True,
        max_length=384,
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop("offset_mapping")[0]
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # Find context token range
    input_ids = inputs["input_ids"][0].cpu()
    sep_positions = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
    if tokenizer.cls_token == "<s>":  # RoBERTa
        ctx_start = sep_positions[1].item() + 1 if len(sep_positions) > 1 else sep_positions[0].item() + 1
    else:
        ctx_start = sep_positions[0].item() + 1
    ctx_end = len(input_ids) - 1

    # Mask non-context positions
    mask = torch.full_like(start_logits, -1e10)
    mask[ctx_start:ctx_end] = 0
    masked_start = start_logits + mask
    masked_end = end_logits + mask

    best_start = int(torch.argmax(masked_start))
    best_end = int(torch.argmax(masked_end))
    if best_end < best_start:
        best_end = best_start

    # Confidence
    start_probs = torch.softmax(masked_start, dim=0)
    end_probs = torch.softmax(masked_end, dim=0)
    confidence = float(start_probs[best_start] * end_probs[best_end])

    # Extract answer text
    char_start = offset_mapping[best_start][0].item()
    char_end = offset_mapping[best_end][1].item()
    answer = context[char_start:char_end]

    # Build highlighted text
    highlighted = []
    if char_start > 0:
        highlighted.append((context[:char_start], None))
    highlighted.append((context[char_start:char_end], "Answer"))
    if char_end < len(context):
        highlighted.append((context[char_end:], None))

    return answer, highlighted, round(confidence, 4)


# Preloaded examples: [question, context, model]
EXAMPLES = [
    # BioASQ examples
    [
        "Which protein is silencing MIR139 to promote AML oncogenesis?",
        "The tumor suppressor MIR139 is silenced by POLR2M to promote AML oncogenesis.",
        "PubMedBERT (best)",
    ],
    [
        "What is the primary vector of zika virus?",
        "Aedes aegypti is the primary vector of these pathogens and Culex quinquefasciatus may be a potential ZIKV vector.",
        "PubMedBERT (best)",
    ],
    [
        "What is the cause of African trypanosomiasis?",
        "Human African trypanosomiasis (HAT) or sleeping sickness is caused by the protozoan parasites Trypanosoma brucei (T.b.) gambiense (West African form) and T.b. rhodesiense (East African form), transmitted by tsetse flies.",
        "PubMedBERT (best)",
    ],
    # COVID-QA examples
    [
        "What proportion of health workers reported distress?",
        "A considerable proportion of participants reported symptoms of depression (634, 50.4%), anxiety (560, 44.6%), insomnia (427, 34.0%), and distress (899, 71.5%). Nurses, women, frontline health care workers, and those working in Wuhan reported more severe degrees of all measurements of mental health symptoms.",
        "PubMedBERT (best)",
    ],
    [
        "How does PEDV spread?",
        "PEDV spreads primarily through fecal-oral contact. Once the virus is internalized, it destroys the lining of piglets' intestines, making them incapable of digesting and deriving nutrition from milk and feed.",
        "PubMedBERT (best)",
    ],
    [
        "Which Human Coronavirus showed species specific clinical characteristics?",
        "The most common species were HCoV-OC43 (35%), HCoV-NL63 (27%), HCoV-229E (22%), and HCoV-HKU1 (16%). We did not observe species-specific differences in the clinical characteristics of HCoV infection, with the exception of HCoV-HKU1, for which the severity of gastrointestinal symptoms trended higher on the fourth day of illness.",
        "PubMedBERT (best)",
    ],
]

# Build Gradio UI
with gr.Blocks(title="Biomedical QA — BERT Models") as demo:
    gr.Markdown("# Biomedical Extractive Question Answering")
    gr.Markdown("Select a model, paste a biomedical context, and ask a question.")

    with gr.Row():
        model_dd = gr.Dropdown(
            choices=list(MODELS.keys()),
            value="PubMedBERT (best)",
            label="Model",
        )

    with gr.Row():
        with gr.Column():
            context_tb = gr.Textbox(
                label="Context (biomedical text)",
                lines=6,
                placeholder="Paste a biomedical text here...",
            )
            question_tb = gr.Textbox(
                label="Question",
                lines=1,
                placeholder="Ask a question about the context...",
            )
            submit_btn = gr.Button("Answer", variant="primary")

        with gr.Column():
            answer_tb = gr.Textbox(label="Predicted Answer", interactive=False)
            highlighted = gr.HighlightedText(
                label="Context with highlighted answer",
                color_map={"Answer": "green"},
            )
            confidence_nb = gr.Number(label="Confidence", precision=4)

    submit_btn.click(
        fn=answer_question,
        inputs=[model_dd, context_tb, question_tb],
        outputs=[answer_tb, highlighted, confidence_nb],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[question_tb, context_tb, model_dd],
        label="Example questions (BioASQ + COVID-QA)",
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
