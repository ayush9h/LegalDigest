from datasets import load_dataset
from transformers import AutoTokenizer

from src.config import BASE_MODEL, DATA_FILE, MAX_LENGTH

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def format_train_data(data):
    if data["input"]:
        prompt = f"Instruction: {data['instruction']}\nInput:{data['input']}\nAnswer:"
    else:
        prompt = f"Instruction: {data['instruction']}\nAnswer:"

    return {
        "input_ids": tokenizer(
            prompt, padding="max_length", truncation=True, max_length=MAX_LENGTH
        ).input_ids,
        "attention_mask": tokenizer(
            prompt, padding="max_length", truncation=True, max_length=MAX_LENGTH
        ).attention_mask,
        "labels": tokenizer(
            data["output"], padding="max_length", truncation=True, max_length=MAX_LENGTH
        ).input_ids,
    }


def load_and_tokenize():
    dataset = load_dataset("json", data_files={"train": DATA_FILE})
    tokenized_dataset = dataset["train"].map(format_train_data)
    return tokenized_dataset
