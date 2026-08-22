"""Unit tests for SentimentPredictor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.predictor import DEFAULT_MODEL, SentimentPredictor


def _make_pipeline_mock(label: str = "POSITIVE", score: float = 0.9998):
    """Return a callable mock that mimics the HF pipeline return value."""
    mock_pipe = MagicMock(return_value=[{"label": label, "score": score}])
    return mock_pipe


class TestSentimentPredictorInit:
    def test_loads_default_model(self):
        with patch("app.predictor.pipeline") as mock_pipe_fn:
            mock_pipe_fn.return_value = _make_pipeline_mock()
            p = SentimentPredictor()
        call_kwargs = mock_pipe_fn.call_args
        assert call_kwargs.args[0] == "text-classification"
        assert call_kwargs.kwargs["model"] == DEFAULT_MODEL
        assert p.model_id == DEFAULT_MODEL

    def test_loads_custom_model(self):
        custom = "textattack/bert-base-uncased-SST-2"
        with patch("app.predictor.pipeline") as mock_pipe_fn:
            mock_pipe_fn.return_value = _make_pipeline_mock()
            p = SentimentPredictor(model_id=custom)
        call_kwargs = mock_pipe_fn.call_args
        assert call_kwargs.args[0] == "text-classification"
        assert call_kwargs.kwargs["model"] == custom
        assert p.model_id == custom


class TestSentimentPredictorPredict:
    @pytest.fixture()
    def predictor(self):
        with patch("app.predictor.pipeline") as mock_pipe_fn:
            mock_pipe_fn.return_value = _make_pipeline_mock("POSITIVE", 0.9998)
            yield SentimentPredictor()

    def test_returns_label_and_score(self, predictor):
        result = predictor.predict("I love this")
        assert result["label"] == "POSITIVE"
        assert abs(result["score"] - 0.9998) < 1e-6

    def test_score_is_float(self, predictor):
        result = predictor.predict("Great product")
        assert isinstance(result["score"], float)

    def test_empty_string_raises_value_error(self, predictor):
        with pytest.raises(ValueError, match="non-empty"):
            predictor.predict("")

    def test_whitespace_only_raises_value_error(self, predictor):
        with pytest.raises(ValueError, match="non-empty"):
            predictor.predict("   ")

    def test_negative_prediction(self):
        with patch("app.predictor.pipeline") as mock_pipe_fn:
            mock_pipe_fn.return_value = _make_pipeline_mock("NEGATIVE", 0.9876)
            p = SentimentPredictor()
        result = p.predict("This was a terrible experience")
        assert result["label"] == "NEGATIVE"
        assert abs(result["score"] - 0.9876) < 1e-6

    def test_pipeline_called_with_text(self, predictor):
        text = "The service is fast and reliable"
        predictor.predict(text)
        predictor._pipeline.assert_called_once_with(text)
