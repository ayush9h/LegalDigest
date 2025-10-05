import os

import pandas as pd
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

from src.config import (
    BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    LOGGING_DIR,
    MODEL_TYPE,
    NUM_EPOCHS,
    OUTPUT_DIR,
)
from src.model import get_model
from src.preprocess import load_and_tokenize, tokenizer

os.makedirs(LOGGING_DIR, exist_ok=True)

def train():
    model = get_model()
    dataset = load_and_tokenize()

    if MODEL_TYPE == "causal":
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        trainer_cls = Trainer
        args_cls = TrainingArguments
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        trainer_cls = Seq2SeqTrainer
        args_cls = Seq2SeqTrainingArguments

    training_args = args_cls(
        output_dir=OUTPUT_DIR,
        logging_dir=LOGGING_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
    )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()

    pd.DataFrame(trainer.state.log_history).to_csv(
        f"{LOGGING_DIR}/training_log_{MODEL_TYPE}.csv", index=False
    )
    model.save_pretrained(OUTPUT_DIR)
