from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from config import (
    BATCH_SIZE,
    DEVICE,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    LOGGING_DIR,
    NUM_EPOCHS,
    OUTPUT_DIR,
)
from src.model import get_model
from src.preprocess import load_and_tokenize, tokenizer


def train():
    model = get_model()
    dataset = load_and_tokenize()
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOGGING_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        report_to=["tensorboard", "csv"],
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
