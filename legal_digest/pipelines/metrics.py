import evaluate
import nltk
import numpy as np
from transformers import T5Tokenizer

from utils.load_yaml_config import load_config

config = load_config("config/training.yaml")
MODEL_NAME = config["model"]["name"]

nltk.download("punkt", quiet=True)

rouge_metric = evaluate.load("rouge")
tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    em = np.mean([p == l for p, l in zip(decoded_preds, decoded_labels)])

    def compute_f1(pred, label):
        pred_tokens = pred.split()
        label_tokens = label.split()

        common = set(pred_tokens) & set(label_tokens)
        if len(common) == 0:
            return 0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(label_tokens)

        return 2 * precision * recall / (precision + recall)

    f1 = np.mean([compute_f1(p, l) for p, l in zip(decoded_preds, decoded_labels)])

    rouge_result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )

    return {
        "exact_match": em,
        "f1": f1,
        "rougeL": rouge_result["rougeL"] if rouge_result else None,
    }
    }
