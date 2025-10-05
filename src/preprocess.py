from datasets import load_dataset
from transformers import AutoTokenizer

from src.config import BASE_MODEL, DATA_FILE, MAX_LENGTH, MODEL_TYPE

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if MODEL_TYPE == "causal" and tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

prefix = "summarize: "

def format_train_data(examples):
    if MODEL_TYPE == "causal":
        texts = [
            f"<|system|> You are a helpful assistant.<|user|> {i}\n<|assistant|> {o}"
            for i, o in zip(examples["instruction"], examples["output"])
        ]
        tokenized = tokenizer(
            texts, truncation=True, max_length=MAX_LENGTH, padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    else:
        inputs = [prefix + doc for doc in examples["instruction"]]
        model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
        labels = tokenizer(
            text_target=examples["output"], max_length=MAX_LENGTH, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

def load_and_tokenize():
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset = dataset.train_test_split(test_size=0.3)
    tokenized_ds = dataset.map(format_train_data, batched=True)
    return tokenized_ds
