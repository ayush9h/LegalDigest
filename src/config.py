import torch

BASE_MODEL = "google/flan-t5-small"


DATA_FILE = "data/ipc.jsonl"
OUTPUT_DIR = "checkpoints/flan-t5-ipc"
LOGGING_DIR = "logs"

LEARNING_RATE = 1e-4
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 5
MAX_LENGTH = 512

LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q", "v"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
