"""Sentiment predictor — wraps a HuggingFace pipeline for inference."""

from __future__ import annotations

import logging
from typing import Any

from transformers import pipeline

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
# Most BERT-family models cap at 512 tokens. The pipeline handles tokenization,
# so we pass truncation=True and set a safe ceiling here.
MAX_LENGTH = 512


class SentimentPredictor:
    """Loads a HuggingFace sentiment pipeline and runs predictions.

    The model is loaded once at startup and reused for every request.
    Works with any HF model that supports the ``text-classification`` task.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        max_length: int = MAX_LENGTH,
    ) -> None:
        """Load the model pipeline.

        Args:
            model_id: HuggingFace Hub model identifier. Defaults to
                ``distilbert-base-uncased-finetuned-sst-2-english``.
            max_length: Token ceiling passed to the pipeline for truncation.
                Defaults to 512.
        """
        self.model_id = model_id
        self.max_length = max_length
        logger.info("Loading model: %s", model_id)
        self._pipeline = pipeline(
            "text-classification",
            model=model_id,
            truncation=True,
            max_length=max_length,
        )
        logger.info("Model loaded.")

    def predict(self, text: str) -> dict[str, Any]:
        """Run sentiment classification on a single input string.

        Args:
            text: The input text to classify. Must be non-empty. Text longer
                than ``max_length`` tokens is truncated automatically.

        Returns:
            A dict with keys ``label`` (str) and ``score`` (float).

        Raises:
            ValueError: If ``text`` is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        results = self._pipeline(text)
        # HF pipelines return a list of dicts for single inputs.
        result = results[0]
        return {"label": result["label"], "score": float(result["score"])}
