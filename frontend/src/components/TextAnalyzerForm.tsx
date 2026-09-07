import { useState } from "react";
import type { FormEvent } from "react";
import { ExamplePrompts } from "./ExamplePrompts";
import { LoadingSpinner } from "./LoadingSpinner";

const MAX_TEXT_LENGTH = 1000;

interface TextAnalyzerFormProps {
  isLoading: boolean;
  onSubmit: (text: string) => void;
}

/**
 * Textarea + example prompts + submit button for entering text to analyze.
 * Owns only the input value; analysis state lives in the parent.
 */
export function TextAnalyzerForm({
  isLoading,
  onSubmit,
}: TextAnalyzerFormProps) {
  const [text, setText] = useState("");

  const trimmedText = text.trim();
  const isSubmitDisabled = isLoading || trimmedText.length === 0;

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (isSubmitDisabled) {
      return;
    }
    onSubmit(trimmedText);
  };

  return (
    <form className="analyzer-form" onSubmit={handleSubmit}>
      <label className="analyzer-form__label" htmlFor="market-text">
        Market text
      </label>
      <textarea
        id="market-text"
        className="analyzer-form__textarea"
        placeholder="Enter a stock market statement, news headline, analyst comment, or financial text..."
        value={text}
        maxLength={MAX_TEXT_LENGTH}
        rows={6}
        disabled={isLoading}
        onChange={(event) => setText(event.target.value)}
      />
      <div className="analyzer-form__meta">
        <span className="analyzer-form__count">
          {text.length} / {MAX_TEXT_LENGTH}
        </span>
      </div>

      <ExamplePrompts onSelect={setText} disabled={isLoading} />

      <button
        type="submit"
        className="analyzer-form__submit"
        disabled={isSubmitDisabled}
      >
        {isLoading ? (
          <>
            <LoadingSpinner />
            Analyzing…
          </>
        ) : (
          "Analyze"
        )}
      </button>
    </form>
  );
}
