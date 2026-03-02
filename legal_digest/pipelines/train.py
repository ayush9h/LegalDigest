from core.load_yaml_config import config
from datasets import load_dataset
from pipelines.metrics import compute_metrics
from pipelines.preprocess import preprocess_fn
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

MODEL_NAME = config["model"]["name"]
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def finetune_pipeline():

    # Load dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/constitution_qa.json",
        },
    )
    TEST_SPLIT = config["data"]["test_split"]
    dataset = dataset["train"].train_test_split(test_size=TEST_SPLIT)

    # Preprocess function
    tokenized_dataset = preprocess_fn(
        dataset=dataset,
        tokenizer=tokenizer,
    )

    L_RATE = config["training"]["learning_rate"]
    BATCH_SIZE = config["training"]["train_batch_size"]
    PER_DEVICE_EVAL_BATCH = config["training"]["eval_batch_size"]
    WEIGHT_DECAY = config["training"]["weight_decay"]
    NUM_EPOCHS = config["training"]["num_epochs"]
    SAVE_TOTAL_LIM = config["training"]["save_total_limit"]
    OUTPUT_DIR = config["output"]["dir"]

    training_args = Seq2SeqTrainingArguments(
        eval_strategy="epoch",
        output_dir=OUTPUT_DIR,
        learning_rate=L_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=SAVE_TOTAL_LIM,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        push_to_hub=False,
    )

    print("Finetuning started")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
