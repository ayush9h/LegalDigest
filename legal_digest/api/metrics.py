from fastapi import APIRouter, Response
from prometheus_client import generate_latest
from utils.monitoring import collect_metrics

metric_router = APIRouter()


@metric_router.get("/metrics")
def metrics():
    collect_metrics()
    return Response(
        generate_latest(),
        media_type="text/plain",
    )
