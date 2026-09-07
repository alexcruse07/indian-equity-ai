"""Standalone script to test the Indian Equity Sentiment Hugging Face model.

Usage:
    HF_TOKEN=... HF_MODEL_ID=... python test_model.py

Reads HF_TOKEN and HF_MODEL_ID from the environment (or a .env file) and runs
a sample sentence through the model.

Note on inference method:
    This model (`alexcruse07/indian-equity-sentiment-model`) has no active
    Hugging Face serverless Inference Providers deployment (confirmed via the
    Hub API: `inferenceProviderMapping` is empty for this repo), and the
    legacy `api-inference.huggingface.co` endpoint has been decommissioned.
    Custom/private fine-tuned models like this one are not available via
    hosted inference unless a paid Inference Endpoint is created for them.

    Therefore this script downloads the model weights from the Hub (using
    HF_TOKEN for authentication, via huggingface_hub under the hood) and runs
    inference locally with `transformers.pipeline`. This still uses the
    Hugging Face Python SDK for model/token resolution and requires no
    separate hosted inference infrastructure.
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.utils import GatedRepoError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_TEXT = "Reliance Industries reported strong quarterly profits, boosting investor confidence."


def load_config() -> tuple[str, str]:
    """Load and validate required configuration from the environment.

    Returns the (token, model_id) tuple. Never logs the token value.
    """
    load_dotenv()

    token = os.getenv("HF_TOKEN")
    model_id = os.getenv("HF_MODEL_ID")

    if not token:
        logger.error("HF_TOKEN is not set. Please set it in the environment or .env file.")
        sys.exit(1)

    if not model_id:
        logger.error("HF_MODEL_ID is not set. Please set it in the environment or .env file.")
        sys.exit(1)

    logger.info("Loaded HF_MODEL_ID=%s (HF_TOKEN is set, value hidden)", model_id)
    return token, model_id


def run_inference(token: str, model_id: str, text: str) -> None:
    """Download (if needed) and run the model locally, then print the raw result."""
    # Imported lazily so that a missing/invalid config fails fast, before
    # paying the cost of importing torch/transformers.
    from transformers import pipeline
    from transformers.pipelines.base import PipelineException

    try:
        classifier = pipeline("text-classification", model=model_id, token=token)
        result = classifier(text)
    except GatedRepoError:
        logger.error("Access denied: model '%s' is gated and HF_TOKEN lacks access.", model_id)
        sys.exit(1)
    except RepositoryNotFoundError:
        logger.error("Model '%s' was not found or is not accessible.", model_id)
        sys.exit(1)
    except HfHubHTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code == 401:
            logger.error("Authentication failed: invalid or missing HF_TOKEN.")
        elif status_code == 403:
            logger.error(
                "Access denied: HF_TOKEN lacks permission to download model '%s'.",
                model_id,
            )
        else:
            logger.error("Hub request failed (status=%s): %s", status_code, exc)
        sys.exit(1)
    except PipelineException as exc:
        logger.error("Model inference failed: %s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - top-level script guard
        logger.error("Unexpected error during inference: %s", exc)
        sys.exit(1)

    logger.info("Sample input: %s", text)
    print("Raw model response:")
    print(result)


def main() -> None:
    token, model_id = load_config()
    run_inference(token, model_id, SAMPLE_TEXT)


if __name__ == "__main__":
    main()
