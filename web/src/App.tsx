import { useCallback, useEffect, useState } from "react";
import { fetchAllEvents, fetchPayouts, fetchSummary } from "./api";
import type { EpisodeEvent, Payout, Summary } from "./types";
import DecisionFeed from "./components/DecisionFeed";
import EvolutionChart from "./components/EvolutionChart";
import ResultsTable from "./components/ResultsTable";
import Receipts from "./components/Receipts";
import ModeToggle from "./components/ModeToggle";
import ThemeToggle from "./components/ThemeToggle";

interface Data {
  summary: Summary;
  events: EpisodeEvent[];
  payouts: Payout[];
}

export default function App() {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setError(null);
    Promise.all([fetchSummary(), fetchAllEvents(), fetchPayouts()])
      .then(([summary, events, p]) => setData({ summary, events, payouts: p.payouts }))
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(load, [load]);

  return (
    <>
      <a className="skip" href="#main">Skip to content</a>
      <div className="sim-banner" role="status">
        <strong>SIMULATED replay - synthetic data.</strong>
        <span className="banner-more"> Nothing here is real-world fraud data. Only sampled payouts touch the PayPal sandbox.</span>
        {data?.summary.synthetic_fixture && <span className="badge badge-sample">sample data</span>}
      </div>
      <header className="top">
        <div>
          <h1>PayGuard Arena</h1>
          <p className="sub">
            Defenders decide RELEASE / HOLD / STEP_UP on instant gig payouts while attackers mutate.
          </p>
        </div>
        <div className="top-controls">
          <ModeToggle />
          <ThemeToggle />
        </div>
      </header>
      <main id="main">
        {error && (
          <div className="card error" role="alert">
            <p>Could not load the replay: {error}</p>
            <button type="button" className="btn" onClick={load}>Retry</button>
          </div>
        )}
        {!data && !error && <p className="muted" role="status">Loading replay...</p>}
        {data && (
          <>
            <p className="meta muted">
              Replay generated {data.summary.generated_at} - schema v{data.summary.schema_version}
              {data.summary.synthetic_fixture && " - synthetic sample fixture"}
            </p>
            <div className="grid2">
              <DecisionFeed events={data.events} episode={data.summary.episode} />
              <EvolutionChart rows={data.summary.evolution} synthetic={data.summary.synthetic_fixture} />
            </div>
            <ResultsTable rows={data.summary.results} objective={data.summary.objective} />
            <Receipts payouts={data.payouts} />
          </>
        )}
      </main>
      <footer className="foot muted">
        PayGuard Arena - PayPal AI Hackathon. Ground truth labels are shown in Replay only.
      </footer>
    </>
  );
}
