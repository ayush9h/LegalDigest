import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="flan-t5", choices=["flan-t5", "llama"])
args, _ = parser.parse_known_args()

if args.model == "llama":
    BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
    MODEL_TYPE = "causal"
    OUTPUT_DIR = "checkpoints/llama-1b-ipc"
else:
    BASE_MODEL = "google/flan-t5-small"
    MODEL_TYPE = "seq2seq"
    OUTPUT_DIR = "checkpoints/flan-t5-ipc"

DATA_FILE = "data/ipc.jsonl"
LOGGING_DIR = "logs"

LEARNING_RATE = 1e-5
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 1
MAX_LENGTH = 512

LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"] if args.model == "llama" else ["q", "v"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
