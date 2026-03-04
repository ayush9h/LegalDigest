import os
from contextlib import asynccontextmanager

from api.inference import inference_router
from api.metrics import metric_router
from api.train import train_router
from dotenv import load_dotenv
from fastapi import FastAPI
from huggingface_hub import login
from utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    access_huggingface_token = os.environ.get("HF_TOKEN", "")

    if not access_huggingface_token:
        logger.error("Huggingface token not provided")
    else:
        logger.info("Logging into Huggingface HUB")
        login(token=access_huggingface_token)

        logger.info("Logged into Huggingface HUB")
    yield


app = FastAPI(
    title="Legal Digest -  Train and Inference", version="0.0.1", lifespan=lifespan
)

app.include_router(inference_router, prefix="/api/v1")
app.include_router(train_router, prefix="/api/v1")
app.include_router(metric_router)
