"""Pydantic schemas for request and response validation."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Input payload for the /predict endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Text to run sentiment analysis on.",
        examples=["This movie was absolutely fantastic!"],
    )


class PredictResponse(BaseModel):
    """Prediction result returned by the /predict endpoint."""

    label: str = Field(..., description="Predicted sentiment label (POSITIVE or NEGATIVE).")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the predicted label.")
    model: str = Field(..., description="HuggingFace model ID used to produce this prediction.")


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str = Field(..., description="Service status. Always 'ok' when the model is loaded.")
    model_version: str = Field(..., description="HuggingFace model ID currently loaded.")
