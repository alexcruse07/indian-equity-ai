import type {
  ApiErrorResponse,
  SentimentRequest,
  SentimentResponse,
} from "../types/sentiment";

/**
 * Base URL of our FastAPI backend, centralized here so every API call uses
 * the same source of truth. Read from Vite env config
 * (`VITE_API_BASE_URL`, e.g. via `.env`/`.env.production`) rather than
 * hardcoded, so each environment (local dev, staging, production) can point
 * at a different backend without code changes.
 *
 * Falls back to the local backend URL only for local development
 * convenience; no production URL is ever hardcoded here.
 */
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

/** Raised for both HTTP error responses and network-level failures. */
export class ApiError extends Error {
  /** HTTP status code, or 0 for network/connection failures. */
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Calls our backend's `POST /api/v1/sentiment` endpoint.
 *
 * Throws `ApiError` for:
 * - network errors (backend unreachable, DNS failure, CORS block, etc.)
 * - non-2xx HTTP responses
 * - a 2xx response whose body is not valid JSON
 */
export async function analyzeSentiment(
  text: string
): Promise<SentimentResponse> {
  const requestBody: SentimentRequest = { text };

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/v1/sentiment`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });
  } catch {
    // fetch() rejects only on network-level failures (offline, DNS,
    // connection refused, CORS block) - never on HTTP error statuses.
    throw new ApiError(
      "Unable to reach the sentiment analysis service. Please check your connection and try again.",
      0
    );
  }

  if (!response.ok) {
    let detail: string | undefined;
    try {
      const errorBody = (await response.json()) as ApiErrorResponse;
      detail = errorBody.detail;
    } catch {
      detail = undefined;
    }
    throw new ApiError(
      detail ?? `Request failed with status ${response.status}.`,
      response.status
    );
  }

  try {
    return (await response.json()) as SentimentResponse;
  } catch {
    throw new ApiError(
      "Received an invalid response from the sentiment analysis service.",
      response.status
    );
  }
}

