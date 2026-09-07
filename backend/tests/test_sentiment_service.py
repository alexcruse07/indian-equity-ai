"""Unit tests for `SentimentService`.

All Hugging Face calls (`InferenceClient.text_classification` and the local
`transformers.pipeline`) are mocked. No real network or model calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from app.config import Settings
from app.models.sentiment import Sentiment
from app.services.sentiment_service import (
    AuthenticationError,
    InferenceError,
    InferenceTimeoutError,
    InvalidInputError,
    MalformedModelResponseError,
    SentimentService,
)


def _settings() -> Settings:
    return Settings(
        hf_token="fake-token",
        hf_model_id="alexcruse07/indian-equity-sentiment-model",
        hf_inference_timeout_seconds=5.0,
    )


def _http_error(status_code: int) -> HfHubHTTPError:
    request = httpx.Request("POST", "https://router.huggingface.co/fake")
    response = httpx.Response(status_code=status_code, request=request)
    return HfHubHTTPError("boom", response=response)


@pytest.fixture
def service() -> SentimentService:
    with patch("app.services.sentiment_service.InferenceClient"):
        return SentimentService(_settings())


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_input_raises_invalid_input_error(service: SentimentService):
    with pytest.raises(InvalidInputError):
        service.analyze_sentiment("")


def test_whitespace_only_input_raises_invalid_input_error(service: SentimentService):
    with pytest.raises(InvalidInputError):
        service.analyze_sentiment("   \n\t  ")


def test_none_input_raises_invalid_input_error(service: SentimentService):
    with pytest.raises(InvalidInputError):
        service.analyze_sentiment(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Successful remote inference
# ---------------------------------------------------------------------------


def test_successful_remote_inference_returns_domain_result(
    service: SentimentService,
):
    service._client.text_classification.return_value = [
        {"label": "BULLISH", "score": 0.87}
    ]

    result = service.analyze_sentiment("Reliance profits surge.")

    assert result.sentiment == Sentiment.BULLISH
    assert result.confidence == pytest.approx(0.87)
    assert result.raw_label == "BULLISH"
    assert result.model_name == "alexcruse07/indian-equity-sentiment-model"
    service._client.text_classification.assert_called_once_with(
        "Reliance profits surge."
    )


def test_successful_remote_inference_accepts_dict_result(
    service: SentimentService,
):
    # Some SDK versions may return a single dict rather than a list.
    service._client.text_classification.return_value = {
        "label": "NEUTRAL",
        "score": 0.5,
    }

    result = service.analyze_sentiment("Markets closed flat today.")

    assert result.sentiment == Sentiment.NEUTRAL
    assert result.confidence == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Hugging Face authentication failure
# ---------------------------------------------------------------------------


def test_authentication_failure_raises_authentication_error(
    service: SentimentService,
):
    service._client.text_classification.side_effect = _http_error(401)

    with pytest.raises(AuthenticationError):
        service.analyze_sentiment("Some market text.")


# ---------------------------------------------------------------------------
# Hugging Face timeout
# ---------------------------------------------------------------------------


def test_timeout_raises_inference_timeout_error(service: SentimentService):
    service._client.text_classification.side_effect = TimeoutError("timed out")

    with pytest.raises(InferenceTimeoutError):
        service.analyze_sentiment("Some market text.")


# ---------------------------------------------------------------------------
# Remote unavailable -> local fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [400, 403, 404])
def test_remote_unavailable_falls_back_to_local_pipeline(
    service: SentimentService, status_code: int
):
    service._client.text_classification.side_effect = _http_error(status_code)

    fake_pipeline = MagicMock(return_value=[{"label": "BEARISH", "score": 0.62}])
    with patch(
        "transformers.pipelines.pipeline", return_value=fake_pipeline
    ) as mock_pipeline_factory:
        result = service.analyze_sentiment("Company reports heavy losses.")

    mock_pipeline_factory.assert_called_once_with(
        "text-classification",
        model="alexcruse07/indian-equity-sentiment-model",
        token="fake-token",
    )
    fake_pipeline.assert_called_once_with("Company reports heavy losses.")
    assert result.sentiment == Sentiment.BEARISH
    assert result.confidence == pytest.approx(0.62)


def test_local_pipeline_is_loaded_only_once(service: SentimentService):
    service._client.text_classification.side_effect = _http_error(404)
    fake_pipeline = MagicMock(return_value=[{"label": "NEUTRAL", "score": 0.4}])

    with patch(
        "transformers.pipelines.pipeline", return_value=fake_pipeline
    ) as mock_pipeline_factory:
        service.analyze_sentiment("First call.")
        service.analyze_sentiment("Second call.")

    mock_pipeline_factory.assert_called_once()
    assert fake_pipeline.call_count == 2


def test_unexpected_remote_error_falls_back_to_local(service: SentimentService):
    service._client.text_classification.side_effect = RuntimeError(
        "unexpected transport failure"
    )
    fake_pipeline = MagicMock(return_value=[{"label": "BULLISH", "score": 0.7}])

    with patch("transformers.pipelines.pipeline", return_value=fake_pipeline):
        result = service.analyze_sentiment("Some market text.")

    assert result.sentiment == Sentiment.BULLISH


# ---------------------------------------------------------------------------
# Other Hugging Face inference errors (non-auth, non-timeout, non-fallback)
# ---------------------------------------------------------------------------


def test_unexpected_http_error_raises_inference_error(service: SentimentService):
    service._client.text_classification.side_effect = _http_error(500)

    with pytest.raises(InferenceError):
        service.analyze_sentiment("Some market text.")


def test_local_pipeline_failure_raises_inference_error(service: SentimentService):
    service._client.text_classification.side_effect = _http_error(404)

    with patch("transformers.pipelines.pipeline", side_effect=RuntimeError("boom")):
        with pytest.raises(InferenceError):
            service.analyze_sentiment("Some market text.")


# ---------------------------------------------------------------------------
# Malformed model response
# ---------------------------------------------------------------------------


def test_malformed_remote_response_missing_score_raises(
    service: SentimentService,
):
    service._client.text_classification.return_value = [{"label": "BULLISH"}]

    with pytest.raises(MalformedModelResponseError):
        service.analyze_sentiment("Some market text.")


def test_malformed_remote_response_empty_list_raises(service: SentimentService):
    service._client.text_classification.return_value = []

    with pytest.raises(MalformedModelResponseError):
        service.analyze_sentiment("Some market text.")


def test_unrecognized_label_raises_malformed_response_error(
    service: SentimentService,
):
    service._client.text_classification.return_value = [
        {"label": "SOMETHING_UNKNOWN", "score": 0.9}
    ]

    with pytest.raises(MalformedModelResponseError):
        service.analyze_sentiment("Some market text.")


def test_non_numeric_score_raises_malformed_response_error(
    service: SentimentService,
):
    service._client.text_classification.return_value = [
        {"label": "BULLISH", "score": "not-a-number"}
    ]

    with pytest.raises(MalformedModelResponseError):
        service.analyze_sentiment("Some market text.")


def test_malformed_local_pipeline_response_raises(service: SentimentService):
    service._client.text_classification.side_effect = _http_error(404)

    with patch("transformers.pipelines.pipeline", return_value=MagicMock(return_value=[])):
        with pytest.raises(MalformedModelResponseError):
            service.analyze_sentiment("Some market text.")
