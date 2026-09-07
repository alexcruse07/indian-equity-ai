"""Domain models for sentiment analysis."""

from enum import Enum

from pydantic import BaseModel, Field


class Sentiment(str, Enum):
    """Normalized sentiment classes produced by the model."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SentimentResult(BaseModel):
    """Application-level result of a sentiment analysis request."""

    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    raw_label: str
    model_name: str
