/**
 * Types describing the FastAPI backend's sentiment analysis contract.
 * Kept in sync with `backend/app/api/sentiment.py`.
 */

export type SentimentLabel = "BULLISH" | "BEARISH" | "NEUTRAL";

export interface SentimentRequest {
  text: string;
}

export interface SentimentResponse {
  sentiment: SentimentLabel;
  confidence: number;
  model: string;
}

export interface ApiErrorResponse {
  detail?: string;
}
