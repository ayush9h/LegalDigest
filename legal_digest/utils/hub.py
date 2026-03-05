from pathlib import Path

from peft import PeftModel
from transformers import T5ForConditionalGeneration, T5Tokenizer


def push_hub(REPO_ID: str):

    BASE_DIR = Path(__file__).resolve().parents[1]

    CHECKPOINT_PATH = BASE_DIR / "inferences" / "checkpoint-383"
    base_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

    model = model.merge_and_unload()

    model.push_to_hub(REPO_ID)
    tokenizer.push_to_hub(REPO_ID)
