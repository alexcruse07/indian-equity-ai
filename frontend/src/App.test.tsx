import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { ApiError } from "./services/api";

const { analyzeSentimentMock } = vi.hoisted(() => ({
  analyzeSentimentMock: vi.fn(),
}));

vi.mock("./services/api", async () => {
  const actual =
    await vi.importActual<typeof import("./services/api")>(
      "./services/api"
    );
  return {
    ...actual,
    analyzeSentiment: analyzeSentimentMock,
  };
});

const SAMPLE_TEXT = "Reliance reported strong quarterly earnings.";

function typeAndSubmit() {
  const textarea = screen.getByPlaceholderText(
    /Enter a stock market statement/i
  );
  const button = screen.getByRole("button", { name: /analyze/i });
  return { textarea, button };
}

describe("App", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders the application header, subtitle, textarea, and analyze button", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", { name: /Indian Equity Sentiment Analyzer/i })
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        /Analyze sentiment in Indian equity market text using AI\./i
      )
    ).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText(/Enter a stock market statement/i)
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /analyze/i })
    ).toBeInTheDocument();
  });

  it("allows typing into the textarea", async () => {
    const user = userEvent.setup();
    render(<App />);

    const textarea = screen.getByPlaceholderText(
      /Enter a stock market statement/i
    ) as HTMLTextAreaElement;

    await user.type(textarea, SAMPLE_TEXT);

    expect(textarea.value).toBe(SAMPLE_TEXT);
  });

  it("disables the analyze button while text is empty", () => {
    render(<App />);

    expect(screen.getByRole("button", { name: /analyze/i })).toBeDisabled();
  });

  it("shows a loading state and disables the button while the request is running", async () => {
    const user = userEvent.setup();
    let resolveRequest: (value: unknown) => void = () => {};
    analyzeSentimentMock.mockReturnValue(
      new Promise((resolve) => {
        resolveRequest = resolve;
      })
    );

    render(<App />);
    const { textarea, button } = typeAndSubmit();
    await user.type(textarea, SAMPLE_TEXT);
    await user.click(button);

    expect(screen.getByRole("button", { name: /analyzing/i })).toBeDisabled();

    resolveRequest({
      sentiment: "BULLISH",
      confidence: 0.58,
      model: "alexcruse07/indian-equity-sentiment-model",
    });

    await waitFor(() =>
      expect(screen.getByText(/Bullish/i)).toBeInTheDocument()
    );
  });

  it("renders a successful sentiment response in the result card", async () => {
    const user = userEvent.setup();
    analyzeSentimentMock.mockResolvedValue({
      sentiment: "BULLISH",
      confidence: 0.581,
      model: "alexcruse07/indian-equity-sentiment-model",
    });

    render(<App />);
    const { textarea, button } = typeAndSubmit();
    await user.type(textarea, SAMPLE_TEXT);
    await user.click(button);

    await waitFor(() =>
      expect(screen.getByText(/Bullish/i)).toBeInTheDocument()
    );
    expect(screen.getByText(/58\.1%/)).toBeInTheDocument();
    expect(
      screen.getByText(/alexcruse07\/indian-equity-sentiment-model/)
    ).toBeInTheDocument();
  });

  it("shows an error message when the API call fails", async () => {
    const user = userEvent.setup();
    analyzeSentimentMock.mockRejectedValue(
      new ApiError("Unable to reach the sentiment analysis service.", 0)
    );

    render(<App />);
    const { textarea, button } = typeAndSubmit();
    await user.type(textarea, SAMPLE_TEXT);
    await user.click(button);

    await waitFor(() =>
      expect(
        screen.getByText(/Unable to reach the sentiment analysis service\./i)
      ).toBeInTheDocument()
    );
    expect(
      screen.getByRole("button", { name: /^analyze$/i })
    ).not.toBeDisabled();
  });
});
