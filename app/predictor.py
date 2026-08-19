"""Sentiment predictor — wraps a HuggingFace pipeline for inference."""

from __future__ import annotations

import logging
from typing import Any

from transformers import pipeline

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


class SentimentPredictor:
    """Loads a HuggingFace sentiment pipeline and runs predictions.

    The model is loaded once at startup and reused for every request.
    Supports any HF model compatible with the ``text-classification`` task.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        """Load the model pipeline.

        Args:
            model_id: HuggingFace Hub model identifier. Defaults to
                ``distilbert-base-uncased-finetuned-sst-2-english``.
        """
        self.model_id = model_id
        logger.info("Loading model: %s", model_id)
        self._pipeline = pipeline("text-classification", model=model_id)
        logger.info("Model loaded.")

    def predict(self, text: str) -> dict[str, Any]:
        """Run sentiment classification on a single input string.

        Args:
            text: The input text to classify. Must be non-empty and fit
                within the model's max token length.

        Returns:
            A dict with keys ``label`` (str) and ``score`` (float).

        Raises:
            ValueError: If ``text`` is empty.
        """
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        results = self._pipeline(text)
        # HF pipelines return a list of dicts; we send single inputs.
        result = results[0]
        return {"label": result["label"], "score": float(result["score"])}
