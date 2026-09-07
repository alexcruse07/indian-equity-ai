import { useCallback, useState } from "react";
import { analyzeSentiment, ApiError } from "../services/api";
import type { SentimentResponse } from "../types/sentiment";

interface UseSentimentAnalysisResult {
  result: SentimentResponse | null;
  error: string | null;
  isLoading: boolean;
  analyze: (text: string) => Promise<void>;
  reset: () => void;
}

/**
 * Encapsulates the request lifecycle (loading / result / error) for
 * analyzing a piece of text against our backend's sentiment endpoint.
 */
export function useSentimentAnalysis(): UseSentimentAnalysisResult {
  const [result, setResult] = useState<SentimentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const analyze = useCallback(async (text: string) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await analyzeSentiment(text);
      setResult(response);
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : "Something went wrong while analyzing the text. Please try again.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, error, isLoading, analyze, reset };
}
