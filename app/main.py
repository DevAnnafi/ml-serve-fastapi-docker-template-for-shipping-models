"""FastAPI application — exposes /predict, /health, and /metrics."""

from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.predictor import SentimentPredictor
from app.schemas import HealthResponse, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("MODEL_ID", "distilbert-base-uncased-finetuned-sst-2-english")

app = FastAPI(
    title="ml-serve",
    description="FastAPI template for shipping HuggingFace sentiment models.",
    version="0.1.0",
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "predict_requests_total",
    "Total number of /predict requests",
    ["status"],
)
REQUEST_LATENCY = Histogram(
    "predict_request_latency_seconds",
    "Latency of /predict requests in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

predictor: SentimentPredictor | None = None


@app.on_event("startup")
async def startup_event() -> None:
    """Load the model pipeline on startup.

    Runs once when the server starts. Stores the loaded predictor in the
    module-level ``predictor`` variable so all requests share the same
    instance.
    """
    global predictor
    predictor = SentimentPredictor(model_id=MODEL_ID)
    logger.info("Predictor ready.")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Return service health and the currently loaded model version.

    Returns:
        HealthResponse with ``status="ok"`` and the model ID.

    Raises:
        HTTPException 503: If the model is not yet loaded.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return HealthResponse(status="ok", model_version=predictor.model_id)


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(body: PredictRequest) -> PredictResponse:
    """Run sentiment classification on the supplied text.

    Args:
        body: JSON payload matching ``PredictRequest`` (a ``text`` field).

    Returns:
        ``PredictResponse`` with ``label``, ``score``, and ``model``.

    Raises:
        HTTPException 503: If the model is not loaded.
        HTTPException 422: If ``text`` is empty (handled by Pydantic validation).
        HTTPException 500: On unexpected predictor errors.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    start = time.perf_counter()
    try:
        result = predictor.predict(body.text)
        REQUEST_COUNT.labels(status="success").inc()
        return PredictResponse(
            label=result["label"],
            score=result["score"],
            model=predictor.model_id,
        )
    except ValueError as exc:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        REQUEST_COUNT.labels(status="error").inc()
        logger.exception("Predictor error: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start)


@app.get("/metrics", tags=["ops"])
async def metrics() -> Response:
    """Expose Prometheus metrics in the standard text format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
