from api.inference import inference_router
from api.train import train_router
from fastapi import FastAPI

app = FastAPI(
    title="Legal Digest -  Train and Inference",
    version="0.0.1",
)

app.include_router(inference_router, prefix="/api/v1")
app.include_router(train_router, prefix="/api/v1")
