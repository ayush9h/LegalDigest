from fastapi import APIRouter, Request
from pipelines.train import finetune_pipeline

train_router = APIRouter(prefix="/train")
from utils.logger import logger


@train_router.post("/flan-t5-small")
async def train_model(request: Request):

    try:
        logger.info("Starting the Finetuning Pipeline for google/flan-t5-small")
        finetune_pipeline()
        logger.info("Finetuning process completed.")
        return {
            "message": "Training of the model successful.",
            "status": 200,
        }

    except Exception as e:
        logger.error(f"Error occurred due to {e}")
        return {
            "message": f"Error occurred due to {e}",
            "status": 400,
        }
