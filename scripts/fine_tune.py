import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

model_name = "google/flan-t5-small"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_config = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q", "v"],
    task_type="SEQ_2_SEQ_LM",
)


def format_train_data(data):
    if data["input"]:
        prompt = f"Instruction: {data['instruction']}\nInput:{data['input']}\nAnswer:"
    else:
        prompt = f"Instruction: {data['instruction']}\nAnswer:"

    return {
        "input_ids": tokenizer(
            prompt, padding="max_length", truncation=True, max_length=512
        ).input_ids,
        "attention_mask": tokenizer(
            prompt, padding="max_length", truncation=True, max_length=512
        ).attention_mask,
        "labels": tokenizer(
            data["output"], padding="max_length", truncation=True, max_length=512
        ).input_ids,
    }


def finetune_flan():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    dataset = load_dataset("json", data_files={"train": "data/ipc.jsonl"})

    tokenized_dataset = dataset["train"].map(format_train_data)

    model = get_peft_model(model, peft_config)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./flan-t5-ipc",
        logging_dir="./logs",
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=20,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained("./flan-t5-ipc")
