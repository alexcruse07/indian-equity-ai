"""Unit tests for the sentiment analysis API endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.sentiment import MAX_TEXT_LENGTH
from app.main import app
from app.models.sentiment import Sentiment, SentimentResult
from app.services.sentiment_service import (
    AuthenticationError,
    InferenceError,
    InferenceTimeoutError,
    InvalidInputError,
    MalformedModelResponseError,
    SentimentService,
    get_sentiment_service,
)


class FakeSentimentService:
    """Stand-in for `SentimentService` that returns/raises configured values."""

    def __init__(self, result=None, error=None) -> None:
        self._result = result
        self._error = error
        self.calls: list[str] = []

    def analyze_sentiment(self, text: str) -> SentimentResult:
        self.calls.append(text)
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def _use_fake_service(fake: FakeSentimentService) -> None:
    app.dependency_overrides[get_sentiment_service] = lambda: fake


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_analyze_sentiment_returns_domain_response(client):
    fake = FakeSentimentService(
        result=SentimentResult(
            sentiment=Sentiment.BULLISH,
            confidence=0.9123,
            raw_label="BULLISH",
            model_name="alexcruse07/indian-equity-sentiment-model",
        )
    )
    _use_fake_service(fake)

    response = client.post("/api/v1/sentiment", json={"text": "Nifty rallies to new high."})

    assert response.status_code == 200
    assert response.json() == {
        "sentiment": "BULLISH",
        "confidence": 0.9123,
        "model": "alexcruse07/indian-equity-sentiment-model",
    }
    assert fake.calls == ["Nifty rallies to new high."]


def test_rejects_empty_text(client):
    fake = FakeSentimentService(
        result=SentimentResult(
            sentiment=Sentiment.NEUTRAL,
            confidence=0.5,
            raw_label="NEUTRAL",
            model_name="m",
        )
    )
    _use_fake_service(fake)

    response = client.post("/api/v1/sentiment", json={"text": ""})

    assert response.status_code == 422
    assert fake.calls == []


def test_rejects_missing_text(client):
    fake = FakeSentimentService(
        result=SentimentResult(
            sentiment=Sentiment.NEUTRAL,
            confidence=0.5,
            raw_label="NEUTRAL",
            model_name="m",
        )
    )
    _use_fake_service(fake)

    response = client.post("/api/v1/sentiment", json={})

    assert response.status_code == 422
    assert fake.calls == []


def test_rejects_text_exceeding_max_length(client):
    fake = FakeSentimentService(
        result=SentimentResult(
            sentiment=Sentiment.NEUTRAL,
            confidence=0.5,
            raw_label="NEUTRAL",
            model_name="m",
        )
    )
    _use_fake_service(fake)

    oversized = "a" * (MAX_TEXT_LENGTH + 1)
    response = client.post("/api/v1/sentiment", json={"text": oversized})

    assert response.status_code == 422
    assert fake.calls == []


def test_invalid_input_from_service_maps_to_400(client):
    fake = FakeSentimentService(error=InvalidInputError("Input text must not be empty."))
    _use_fake_service(fake)

    response = client.post("/api/v1/sentiment", json={"text": "   ignored by fake"})

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_authentication_error_maps_to_502(client):
    fake = FakeSentimentService(error=AuthenticationError("bad token"))
    _use_fake_service(fake)

    response = client.post("/api/v1/sentiment", json={"text": "sample"})

    assert response.status_code == 502
    assert response.json()["detail"] == "Sentiment provider authentication failed."


def test_timeout_error_maps_to_504(client):
    fake = FakeSentimentService(error=InferenceTimeoutError("timed out"))
    _use_fake_service(fake)

    response = client.post("/api/v1/sentiment", json={"text": "sample"})

    assert response.status_code == 504
    assert response.json()["detail"] == "Sentiment inference timed out."


def test_malformed_response_maps_to_502(client):
    fake = FakeSentimentService(
        error=MalformedModelResponseError("unrecognized label 'FOO'")
    )
    _use_fake_service(fake)

    response = client.post("/api/v1/sentiment", json={"text": "sample"})

    assert response.status_code == 502
    assert response.json()["detail"] == "Sentiment provider returned an unexpected response."


def test_inference_error_maps_to_502(client):
    fake = FakeSentimentService(error=InferenceError("boom"))
    _use_fake_service(fake)

    response = client.post("/api/v1/sentiment", json={"text": "sample"})

    assert response.status_code == 502
    assert response.json()["detail"] == "Sentiment inference failed."


def test_health_endpoint_still_works(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_get_sentiment_service_returns_real_service():
    # Ensure the DI factory returns the concrete service class when not overridden.
    app.dependency_overrides.clear()
    service = get_sentiment_service()
    assert isinstance(service, SentimentService)
