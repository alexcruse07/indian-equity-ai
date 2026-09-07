"""Application configuration loaded from environment variables."""

from functools import lru_cache
from os import getenv
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_APP_NAME = "Indian Equity Sentiment Analyzer"
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
DEFAULT_HF_MODEL_ID = "alexcruse07/indian-equity-sentiment-model"
DEFAULT_HF_INFERENCE_TIMEOUT_SECONDS = 30.0


class Settings(BaseModel):
    """Runtime settings for the API."""

    app_name: str = Field(default=DEFAULT_APP_NAME)
    allowed_origins: list[str] = Field(
        default_factory=lambda: DEFAULT_ALLOWED_ORIGINS.copy()
    )
    hf_token: str | None = Field(default=None, repr=False)
    hf_model_id: str = Field(default=DEFAULT_HF_MODEL_ID)
    hf_inference_timeout_seconds: float = Field(
        default=DEFAULT_HF_INFERENCE_TIMEOUT_SECONDS
    )

    @classmethod
    def from_environment(cls) -> "Settings":
        """Build settings from environment variables."""
        origins = getenv("ALLOWED_ORIGINS")
        timeout = getenv("HF_INFERENCE_TIMEOUT_SECONDS")
        return cls(
            app_name=getenv("APP_NAME", DEFAULT_APP_NAME),
            allowed_origins=(
                [origin.strip() for origin in origins.split(",") if origin.strip()]
                if origins
                else DEFAULT_ALLOWED_ORIGINS.copy()
            ),
            hf_token=getenv("HF_TOKEN"),
            hf_model_id=getenv("HF_MODEL_ID", DEFAULT_HF_MODEL_ID),
            hf_inference_timeout_seconds=(
                float(timeout) if timeout else DEFAULT_HF_INFERENCE_TIMEOUT_SECONDS
            ),
        )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings.from_environment()
