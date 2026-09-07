import type { SentimentLabel } from "../types/sentiment";

interface SentimentDisplayConfig {
  displayLabel: string;
  icon: string;
  className: string;
}

/**
 * Maps the backend's financial sentiment labels (BULLISH / BEARISH / NEUTRAL)
 * to the positive / negative / neutral visual language requested for the UI.
 */
const SENTIMENT_DISPLAY: Record<SentimentLabel, SentimentDisplayConfig> = {
  BULLISH: {
    displayLabel: "Positive (Bullish)",
    icon: "▲",
    className: "sentiment-badge--positive",
  },
  BEARISH: {
    displayLabel: "Negative (Bearish)",
    icon: "▼",
    className: "sentiment-badge--negative",
  },
  NEUTRAL: {
    displayLabel: "Neutral",
    icon: "●",
    className: "sentiment-badge--neutral",
  },
};

interface SentimentBadgeProps {
  sentiment: SentimentLabel;
}

export function SentimentBadge({ sentiment }: SentimentBadgeProps) {
  const config = SENTIMENT_DISPLAY[sentiment];

  return (
    <span className={`sentiment-badge ${config.className}`}>
      <span className="sentiment-badge__icon" aria-hidden="true">
        {config.icon}
      </span>
      {config.displayLabel}
    </span>
  );
}
