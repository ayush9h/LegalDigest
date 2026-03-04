from huggingface_hub import login
from transformers import T5ForConditionalGeneration, T5Tokenizer


def push_to_hub(REPO_ID: str, CHECKPOINT_PATH: str):

    model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT_PATH)
    tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT_PATH)

    model.push_to_hub(REPO_ID)
    tokenizer.push_to_hub(REPO_ID)
