from pathlib import Path

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

from legal_digest.config import load_config
from legal_digest.core.dataset import LegalDataset
from legal_digest.core.model import ModelWrapper


def train(model: str):
    cfg = load_config(model)

    wrapper = ModelWrapper(cfg)
    model_obj = wrapper.model
    tokenizer = wrapper.tokenizer

    dataset = LegalDataset(cfg.data.processed_train)

    class LegalTrainDataLoader(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer, max_len):
            self.data = data
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]

            model_inputs = self.tokenizer(
                item["prompt"],
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

    train_dataset = LegalTrainDataLoader(
        dataset,
        tokenizer,
        cfg.training.max_length,
    )
    if cfg.model.active.model_type == "seq2seq":
        collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model_obj,
        )
        args_cls = Seq2SeqTrainingArguments
        trainer_cls = Seq2SeqTrainer
    else:
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        args_cls = TrainingArguments
        trainer_cls = Trainer

    args = args_cls(
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
    )

    trainer = trainer_cls(
        model=model_obj,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()

    save_dir = Path(cfg.model.active.output_dir) / cfg.model.version
    save_dir.mkdir(parents=True, exist_ok=True)
    wrapper.save(save_dir)


import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: legal-digest-train <model>\n"
            "Example: legal-digest-train flan_t5\n"
            "         legal-digest-train llama"
        )

    model = sys.argv[1]
    train(model)


if __name__ == "__main__":
    main()
