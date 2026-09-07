import type { SentimentResponse } from "../types/sentiment";
import { SentimentBadge } from "./SentimentBadge";

interface ResultCardProps {
  result: SentimentResponse;
}

export function ResultCard({ result }: ResultCardProps) {
  const confidencePercent = (result.confidence * 100).toFixed(1);

  return (
    <section className="result-card" aria-live="polite">
      <div className="result-card__row">
        <span className="result-card__label">Sentiment</span>
        <SentimentBadge sentiment={result.sentiment} />
      </div>

      <div className="result-card__row">
        <span className="result-card__label">Confidence</span>
        <div className="result-card__confidence">
          <div className="result-card__confidence-bar">
            <div
              className="result-card__confidence-fill"
              style={{ width: `${confidencePercent}%` }}
            />
          </div>
          <span className="result-card__confidence-value">
            {confidencePercent}%
          </span>
        </div>
      </div>

      <div className="result-card__row">
        <span className="result-card__label">Model</span>
        <span className="result-card__model">{result.model}</span>
      </div>
    </section>
  );
}
