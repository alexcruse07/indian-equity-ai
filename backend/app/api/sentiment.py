"""Sentiment analysis API endpoint."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.models.sentiment import Sentiment
from app.services.sentiment_service import (
    AuthenticationError,
    InferenceError,
    InferenceTimeoutError,
    InvalidInputError,
    MalformedModelResponseError,
    SentimentService,
    get_sentiment_service,
)

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 1000

router = APIRouter(prefix="/api/v1", tags=["sentiment"])


class SentimentRequest(BaseModel):
    """Request body for sentiment analysis."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Text to analyze for market sentiment.",
    )


class SentimentResponse(BaseModel):
    """Response returned to API clients."""

    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    model: str


SentimentServiceDependency = Annotated[SentimentService, Depends(get_sentiment_service)]


@router.post(
    "/sentiment",
    response_model=SentimentResponse,
    status_code=status.HTTP_200_OK,
)
def analyze_sentiment(
    payload: SentimentRequest,
    service: SentimentServiceDependency,
) -> SentimentResponse:
    """Analyze the sentiment of a piece of text."""
    logger.info("Sentiment request received (text_length=%d)", len(payload.text))

    try:
        result = service.analyze_sentiment(payload.text)
    except InvalidInputError as exc:
        logger.info("Sentiment request rejected: invalid input")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except AuthenticationError as exc:
        logger.error("Sentiment service authentication failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Sentiment provider authentication failed.",
        ) from exc
    except InferenceTimeoutError as exc:
        logger.warning("Sentiment inference timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Sentiment inference timed out.",
        ) from exc
    except MalformedModelResponseError as exc:
        logger.error("Sentiment service returned a malformed response: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Sentiment provider returned an unexpected response.",
        ) from exc
    except InferenceError as exc:
        logger.error("Sentiment inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Sentiment inference failed.",
        ) from exc

    logger.info(
        "Sentiment request completed (sentiment=%s, confidence=%.4f)",
        result.sentiment.value,
        result.confidence,
    )

    return SentimentResponse(
        sentiment=result.sentiment,
        confidence=result.confidence,
        model=result.model_name,
    )
