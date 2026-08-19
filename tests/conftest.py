"""Shared pytest fixtures for ml-serve tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"


@pytest.fixture()
def mock_predictor():
    """A MagicMock that stands in for SentimentPredictor.

    Returns a fixed POSITIVE prediction so tests don't download any model.
    """
    m = MagicMock()
    m.model_id = MODEL_ID
    m.predict.return_value = {"label": "POSITIVE", "score": 0.9998}
    return m


@pytest.fixture()
def client(mock_predictor):
    """TestClient with the model predictor swapped for a mock.

    Patches ``app.main.SentimentPredictor`` so the startup event never
    downloads a real model. Each test gets a fresh client with the
    predictor pre-loaded.
    """
    with patch("app.main.SentimentPredictor", return_value=mock_predictor):
        with TestClient(app) as c:
            yield c


@pytest.fixture()
def negative_client(mock_predictor):
    """TestClient whose predictor returns a NEGATIVE prediction."""
    mock_predictor.predict.return_value = {"label": "NEGATIVE", "score": 0.9876}
    with patch("app.main.SentimentPredictor", return_value=mock_predictor):
        with TestClient(app) as c:
            yield c
