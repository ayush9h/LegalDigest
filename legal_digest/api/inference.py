from fastapi import APIRouter, Request
from pipelines.inference import inference

inference_router = APIRouter(prefix="/inference")
from utils.logger import logger


@inference_router.post("/flan-t5-small")
async def inference_model(request: Request):

    try:

        logger.info("Starting the Inferencing Pipeline for google/flan-t5-small")

        user_query = request.query_params.get("query", "")
        answer = inference("results/checkpoint-383", user_query=user_query)

        logger.info("Inferencing process completed.")

        return {
            "message": answer,
            "status": 200,
        }

    except Exception as e:
        logger.error(f"Error occurred due to {e}")
        return {
            "message": f"Error occurred due to {e}",
            "status": 400,
        }
