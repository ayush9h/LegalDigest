import psutil
import pynvml
from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint"],
)

REQUEST_LATENCY = Histogram(
    "api_latency_seconds",
    "Request latency",
    ["endpoint"],
)

ERROR_COUNT = Counter(
    "api_errors_total",
    "Toal API Errors",
)

CPU_USAGE = Gauge(
    "cpu_usage_percent",
    "CPU Usage percent",
)


MEMORY_USAGE = Gauge(
    "memory_usage_percent",
    "Memory Usage percent",
)

GPU_UTIL = Gauge(
    "gpu_utilization_percent",
    "GPU utilization percent",
)


def collect_metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle=handle)
        GPU_UTIL.set(util.gpu)
    except:
        # No GPU found
        GPU_UTIL.set(0)
