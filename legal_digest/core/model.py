from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer)


class ModelWrapper:
    def __init__(self, cfg):
        self.cfg = cfg
        mcfg = cfg.model.active

        self.device = cfg.runtime.device

        self.tokenizer = AutoTokenizer.from_pretrained(
            mcfg.base_model,
            trust_remote_code=True,
        )

        if mcfg.model_type == "seq2seq":
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                mcfg.base_model,
            )
            task_type = "SEQ_2_SEQ_LM"

        elif mcfg.model_type == "causal":
            base_model = AutoModelForCausalLM.from_pretrained(
                mcfg.base_model,
            )
            task_type = "CAUSAL_LM"

        else:
            raise ValueError(f"Unknown model_type: {mcfg.model_type}")

        lora_cfg = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.alpha,
            lora_dropout=cfg.model.lora.dropout,
            bias="none",
            target_modules=mcfg.target_modules,
            task_type=task_type,
        )

        self.model = get_peft_model(base_model, lora_cfg)
        self.model.to(self.device)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def generate(self, text: str, max_length: int = 128) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_length=max_length,
        )

        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )
