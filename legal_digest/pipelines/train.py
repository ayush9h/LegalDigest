from pathlib import Path

import torch
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from legal_digest.config import load_config
from legal_digest.core.model import ModelWrapper


def train(model: str):
    cfg = load_config(model)

    wrapper = ModelWrapper(cfg)
    model_obj = wrapper.model
    tokenizer = wrapper.tokenizer

    examples = [
        {
            "input": "Explain IPC Section 420",
            "target": "Section 420 deals with cheating and dishonestly inducing delivery of property.",
        }
    ]

    class LegalDatasetLoader(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer, max_len):
            self.data = data
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]

            model_inputs = self.tokenizer(
                item["input"],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
            )

            labels = self.tokenizer(
                item["target"],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
            )["input_ids"]

            model_inputs["labels"] = labels
            return model_inputs

    train_dataset = LegalDatasetLoader(
        examples,
        tokenizer,
        cfg.training.max_length,
    )

    args = TrainingArguments(
        output_dir=cfg.model.active.output_dir,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.grad_accum_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        logging_dir=cfg.logging.dir,
        logging_steps=cfg.logging.steps,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        save_strategy=cfg.training.save_strategy,
        save_total_limit=cfg.training.save_total_limit,
        report_to="none",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model_obj,
    )

    trainer = Trainer(
        model=model_obj,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()

    save_dir = Path(cfg.model.active.output_dir) / cfg.model.version
    save_dir.mkdir(parents=True, exist_ok=True)
    wrapper.save(save_dir)
