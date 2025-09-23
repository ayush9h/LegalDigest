from datasets import load_dataset
from transformers import AutoTokenizer

from src.config import BASE_MODEL, DATA_FILE, MAX_LENGTH

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

prefix = "summarize: "
def format_train_data(examples):
    inputs = [prefix + doc for doc in examples["instruction"]]
    model_inputs = tokenizer(inputs, max_length = MAX_LENGTH, truncation=True)


    labels = tokenizer(text_target = examples["output"], max_length = MAX_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def load_and_tokenize():
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset = dataset.train_test_split(test_size=0.3)
    tokenized_ds =  dataset.map(format_train_data, batched=True)
    return tokenized_ds