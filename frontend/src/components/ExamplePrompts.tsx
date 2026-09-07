import { EXAMPLE_PROMPTS } from "../constants/examplePrompts";

interface ExamplePromptsProps {
  onSelect: (text: string) => void;
  disabled: boolean;
}

/**
 * Quick-fill buttons that let users try the analyzer without typing.
 */
export function ExamplePrompts({ onSelect, disabled }: ExamplePromptsProps) {
  return (
    <div className="example-prompts">
      <span className="example-prompts__label">Try an example:</span>
      <div className="example-prompts__list">
        {EXAMPLE_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            type="button"
            className="example-prompts__chip"
            onClick={() => onSelect(prompt)}
            disabled={disabled}
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
