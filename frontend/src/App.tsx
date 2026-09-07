import { ErrorBanner } from "./components/ErrorBanner";
import { Header } from "./components/Header";
import { ResultCard } from "./components/ResultCard";
import { TextAnalyzerForm } from "./components/TextAnalyzerForm";
import { useSentimentAnalysis } from "./hooks/useSentimentAnalysis";
import "./App.css";

function App() {
  const { result, error, isLoading, analyze } = useSentimentAnalysis();

  return (
    <div className="app-shell">
      <main className="app-card">
        <Header />

        <TextAnalyzerForm isLoading={isLoading} onSubmit={analyze} />

        {error && <ErrorBanner message={error} />}

        {result && <ResultCard result={result} />}
      </main>
    </div>
  );
}

export default App;
