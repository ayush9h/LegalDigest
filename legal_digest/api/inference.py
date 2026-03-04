from dataclasses import dataclass

from fastapi import APIRouter, Request
from pipelines.inference import inference
from pydantic import Field
from utils.monitoring import ERROR_COUNT, REQUEST_COUNT, REQUEST_LATENCY

inference_router = APIRouter(prefix="/inference")
import time

from utils.logger import logger


@dataclass
class QueryRequest:
    query: str = Field(..., description="User query")


@inference_router.post("/flan-t5-small")
async def inference_model(request: QueryRequest):
    start = time.time()
    REQUEST_COUNT.labels(endpoint="/flan-t5-small").inc()

    try:

        logger.info("Starting the Inferencing Pipeline for google/flan-t5-small")

        user_query = request.query
        answer = inference("results/checkpoint-383", user_query=user_query)

        REQUEST_LATENCY.labels(endpoint="/flan-t5-small").observe(
            time.time() - start,
        )

        logger.info("Inferencing process completed.")

        return {
            "message": answer,
            "status": 200,
        }

    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Error occurred due to {e}")
        return {
            "message": f"Error occurred due to {e}",
            "status": 400,
        }
