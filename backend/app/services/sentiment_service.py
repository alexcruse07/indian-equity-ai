"""Sentiment analysis service backed by the Hugging Face sentiment model.

Inference strategy:
    1. Try the Hugging Face Inference Providers API (`InferenceClient`).
    2. If that is unavailable for this model (no provider deployment, as is
       currently the case for `alexcruse07/indian-equity-sentiment-model`) or
       otherwise fails, fall back to running the model locally via
       `transformers.pipeline`. The local pipeline is loaded once and reused.

This module contains no FastAPI route logic - it is a plain service class
that can be constructed with settings and invoked directly or injected as a
FastAPI dependency.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from app.config import Settings
from app.models.sentiment import Sentiment, SentimentResult

logger = logging.getLogger(__name__)

_KNOWN_LABELS = {label.value for label in Sentiment}


class SentimentServiceError(Exception):
    """Base class for all sentiment service errors."""


class InvalidInputError(SentimentServiceError):
    """Raised when the input text fails validation."""


class AuthenticationError(SentimentServiceError):
    """Raised when Hugging Face authentication fails."""


class InferenceTimeoutError(SentimentServiceError):
    """Raised when the inference request times out."""


class InferenceError(SentimentServiceError):
    """Raised when inference fails for reasons other than auth/timeout."""


class MalformedModelResponseError(SentimentServiceError):
    """Raised when the model response cannot be interpreted."""


class SentimentService:
    """Analyzes text sentiment using the configured Hugging Face model."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = InferenceClient(
            model=settings.hf_model_id,
            token=settings.hf_token,
            timeout=settings.hf_inference_timeout_seconds,
        )
        self._local_pipeline: Any | None = None

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze the sentiment of `text` and return a domain result.

        Raises:
            InvalidInputError: if the input is empty/blank.
            AuthenticationError: if the Hugging Face token is invalid.
            InferenceTimeoutError: if the request times out.
            InferenceError: if inference fails for another reason.
            MalformedModelResponseError: if the model response cannot be parsed.
        """
        cleaned_text = self._validate_input(text)

        raw_result = self._run_remote_inference(cleaned_text)
        if raw_result is None:
            raw_result = self._run_local_inference(cleaned_text)

        return self._to_domain_result(raw_result)

    def _validate_input(self, text: str) -> str:
        if text is None:
            raise InvalidInputError("Input text must not be None.")
        cleaned = text.strip()
        if not cleaned:
            raise InvalidInputError("Input text must not be empty.")
        return cleaned

    def _run_remote_inference(self, text: str) -> dict[str, Any] | None:
        """Attempt inference via the Hugging Face Inference Providers API.

        Returns None (rather than raising) when the remote API is not usable
        for this model, so the caller can fall back to local inference.
        Raises AuthenticationError/InferenceTimeoutError for errors that
        should not be silently masked by a fallback.
        """
        try:
            result = self._client.text_classification(text)
        except HfHubHTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 401:
                raise AuthenticationError(
                    "Authentication with Hugging Face failed. Check HF_TOKEN."
                ) from exc
            if status_code in (400, 403, 404):
                # Model has no remote provider deployment, or the token lacks
                # Inference Providers access. Fall back to local inference.
                logger.warning(
                    "Remote inference unavailable for model '%s' (status=%s); "
                    "falling back to local inference.",
                    self._settings.hf_model_id,
                    status_code,
                )
                return None
            raise InferenceError(
                f"Hugging Face inference request failed (status={status_code})."
            ) from exc
        except TimeoutError as exc:
            raise InferenceTimeoutError(
                "Hugging Face inference request timed out."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - unknown transport/SDK errors
            logger.warning(
                "Remote inference failed unexpectedly for model '%s': %s. "
                "Falling back to local inference.",
                self._settings.hf_model_id,
                exc,
            )
            return None

        return self._normalize_client_result(result)

    def _normalize_client_result(self, result: Any) -> dict[str, Any]:
        """Normalize the `InferenceClient.text_classification` result shape."""
        if isinstance(result, list) and result:
            top = result[0]
        else:
            top = result
        if not isinstance(top, dict) or "label" not in top or "score" not in top:
            raise MalformedModelResponseError(
                f"Unexpected remote model response shape: {result!r}"
            )
        return {"label": top["label"], "score": top["score"]}

    def _run_local_inference(self, text: str) -> dict[str, Any]:
        """Run inference locally using a lazily-loaded `transformers` pipeline."""
        try:
            pipeline = self._get_local_pipeline()
            result = pipeline(text)
        except (AuthenticationError, InferenceError, MalformedModelResponseError):
            raise
        except Exception as exc:  # noqa: BLE001 - transformers/torch errors
            raise InferenceError(f"Local model inference failed: {exc}") from exc

        if not isinstance(result, list) or not result:
            raise MalformedModelResponseError(
                f"Unexpected local model response shape: {result!r}"
            )
        top = result[0]
        if not isinstance(top, dict) or "label" not in top or "score" not in top:
            raise MalformedModelResponseError(
                f"Unexpected local model response shape: {result!r}"
            )
        return {"label": top["label"], "score": top["score"]}

    def _get_local_pipeline(self) -> Any:
        if self._local_pipeline is None:
            logger.info(
                "Loading local sentiment model '%s' (first use only)...",
                self._settings.hf_model_id,
            )
            # Import the `transformers.pipelines` submodule (not the
            # top-level `transformers` package, which is a lazily-loaded
            # module) and call its `pipeline` attribute directly. This keeps
            # the import lazy (avoiding the torch/transformers import cost
            # unless local inference is actually needed) while remaining
            # reliably patchable in tests via
            # `patch("transformers.pipelines.pipeline", ...)`.
            import transformers.pipelines as hf_pipelines

            self._local_pipeline = hf_pipelines.pipeline(
                "text-classification",
                model=self._settings.hf_model_id,
                token=self._settings.hf_token,
            )
        return self._local_pipeline

    def _to_domain_result(self, raw_result: dict[str, Any]) -> SentimentResult:
        raw_label = str(raw_result["label"])
        score = raw_result["score"]

        try:
            confidence = float(score)
        except (TypeError, ValueError) as exc:
            raise MalformedModelResponseError(
                f"Model score is not numeric: {score!r}"
            ) from exc

        normalized_label = raw_label.strip().upper()
        if normalized_label not in _KNOWN_LABELS:
            raise MalformedModelResponseError(
                f"Model returned an unrecognized label: {raw_label!r}"
            )

        return SentimentResult(
            sentiment=Sentiment(normalized_label),
            confidence=confidence,
            raw_label=raw_label,
            model_name=self._settings.hf_model_id,
        )


@lru_cache
def get_sentiment_service() -> SentimentService:
    """Return a cached, shared `SentimentService` instance."""
    from app.config import get_settings

    return SentimentService(get_settings())
