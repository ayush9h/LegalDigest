import torch
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
from utils.load_yaml_config import load_config
from utils.logger import logger

# =================== Model and tokenizer ==========================
config = load_config("config/training.yaml")
MODEL_NAME = config["model"]["name"]
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def finetune_pipeline():
    """
    runs the finetuning pipeline:

    **Steps involved**
     - Load dataset
     - Tokenization
     - Preprocess dataset
     - Defining Training arguments
     - Finetuning
    """
    TEST_SPLIT = config["data"]["test_split"]
    try:
        logger.info("Loading the dataset")
        dataset = load_dataset(
            "json",
            data_files={
                "train": "data/constitution_qa.json",
            },
        )
        dataset = dataset["train"].train_test_split(test_size=TEST_SPLIT)
        logger.info("Preprocessing of dataset started")
        tokenized_dataset = preprocess_fn(
            dataset=dataset,
            tokenizer=tokenizer,
        )
        logger.info("Preprocessing of the dataset completed")

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
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
            hub_model_id="ayush9h/legal-digest-flan-t5-small",
        )

        logger.info("Finetuning started")
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        logger.info("Finetuning completed")

        logger.info("Pushing to HUB")
        model.push_to_hub()
        tokenizer.push_to_hub("ayush9h/legal-digest-flan-t5-small")
        logger.info("Pushing to HUB successful")

    except Exception as e:
        logger.error(f"Error occured during finetuning due to :{e}")
