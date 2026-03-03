import evaluate
import nltk
import numpy as np
from transformers import T5Tokenizer

nltk.download("punkt", quiet=True)
metric = evaluate.load("rouge")
tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    # Convert logits to token IDs
    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    # REPLACING -100: Ensure labels are handled before decoding
    # We replace -100 with the pad_token_id so the decoder ignores them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode and ensure we handle the tensors correctly
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE expectations: newline after every sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    # Extract a few results and convert to percentages
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}
