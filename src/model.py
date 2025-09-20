from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig

from config import BASE_MODEL, DEVICE, LORA_ALPHA, LORA_DROPOUT, LORA_R, TARGET_MODULES


def get_model():
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        task_type="SEQ_2_SEQ_LM",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = get_peft_model(
        model,
        peft_config,
    )

    model.to(DEVICE)

    return model
