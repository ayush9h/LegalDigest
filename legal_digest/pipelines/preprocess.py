from datasets import DatasetDict
from transformers import T5Tokenizer

prefix = "Please answer the question in detail format: "


def tokenize_prefix(examples: DatasetDict, tokenizer: T5Tokenizer):
    """
    Tokenization of the question and answer along with the prefix attached.

    Args:
     - examples: DatasetDict

    Return:
     - tokenized_data: DatasetDict
    """

    inputs = [prefix + doc for doc in examples["question"]]  # type: ignore

    model_inputs = tokenizer(
        inputs, max_length=128, truncation=True, padding="max_length"
    )

    targets = [str(ans) for ans in examples["answer"]]

    labels = tokenizer(
        text_target=targets,
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_fn(dataset: DatasetDict, tokenizer: T5Tokenizer) -> DatasetDict:
    """
    Preprocessing of the raw data

    Args:
     - examples: DatasetDict

    Return:
     - tokenized_data: DatasetDict
    """

    tokenized_dataset: DatasetDict = dataset.map(
        lambda examples: tokenize_prefix(examples=examples, tokenizer=tokenizer),
        batched=True,
    )
    return tokenized_dataset
