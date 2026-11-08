import { useMemo } from "react";
import type { ReactNode } from "react";
import { fetchAllEvents, fetchPayouts, fetchSummary, fetchWebhooks } from "./api";
import { useAsync } from "./hooks";
import type { AsyncState } from "./hooks";
import TopBlock from "./components/TopBlock";
import Hero from "./components/Hero";
import type { LiveState } from "./components/Hero";
import DecisionFeed from "./components/DecisionFeed";
import EvolutionChart from "./components/EvolutionChart";
import ResultsTable from "./components/ResultsTable";
import Receipts from "./components/Receipts";
import WebhookLog from "./components/WebhookLog";
import { ChartSkeleton, FeedSkeleton, HeroSkeleton, TableSkeleton } from "./components/Skeletons";
import { ErrorCard } from "./components/ui";

/** Skeleton while the first load runs, a per-section error card on failure, otherwise the content. */
function Async<T>({
  state,
  what,
  skeleton,
  children,
}: {
  state: AsyncState<T>;
  what: string;
  skeleton: ReactNode;
  children: (data: T) => ReactNode;
}) {
  if (state.status === "loading" || (state.status === "ready" && state.data === null)) return <>{skeleton}</>;
  if (state.status === "error") return <ErrorCard what={what} message={state.error} onRetry={() => void state.reload()} />;
  return <>{children(state.data as T)}</>;
}

export default function App() {
  const summary = useAsync(fetchSummary);
  const events = useAsync(fetchAllEvents);
  const payouts = useAsync(fetchPayouts);
  const webhooks = useAsync(fetchWebhooks);

  // Batch ids that have a signature-verified webhook (null until webhooks are available).
  const verifiedIds = useMemo(() => {
    if (!webhooks.data) return null;
    const ids = new Set<string>();
    for (const w of webhooks.data.events) if (w.payout_batch_id) ids.add(w.payout_batch_id);
    return ids;
  }, [webhooks.data]);

  const live: LiveState = useMemo(() => {
    if (!payouts.data) return payouts.status === "error" ? "error" : "loading";
    const liveIds = new Set(payouts.data.payouts.filter((p) => p.mode === "live").map((p) => p.payout_batch_id));
    const k = webhooks.data
      ? webhooks.data.events.filter((w) => w.payout_batch_id && liveIds.has(w.payout_batch_id)).length
      : null;
    return { n: payouts.data.payouts.filter((p) => p.mode === "live").length, k };
  }, [payouts.data, payouts.status, webhooks.data]);

  return (
    <>
      <a className="skip" href="#main">Skip to content</a>
      <TopBlock summary={summary.data} />
      <main id="main" className="container">
        <section id="overview" aria-label="Overview">
          <Async state={summary} what="the overview" skeleton={<HeroSkeleton />}>
            {(s) => <Hero results={s.results} live={live} />}
          </Async>
        </section>

        <div className="grid">
          <div id="replay" className="col-7">
            <Async state={events} what="the replay" skeleton={<FeedSkeleton />}>
              {(ev) =>
                summary.data ? (
                  <DecisionFeed events={ev} episode={summary.data.episode} />
                ) : summary.status === "error" ? (
                  <ErrorCard what="the replay" message={summary.error} onRetry={() => void summary.reload()} />
                ) : (
                  <FeedSkeleton />
                )
              }
            </Async>
          </div>
          <div className="col-5">
            <Async state={summary} what="the evolution chart" skeleton={<ChartSkeleton />}>
              {(s) => <EvolutionChart rows={s.evolution} />}
            </Async>
          </div>
        </div>

        <div className="grid">
          <div className="col-12">
            <Async state={summary} what="the results" skeleton={<TableSkeleton what="results" />}>
              {(s) => <ResultsTable rows={s.results} />}
            </Async>
          </div>
        </div>

        <div id="paypal" className="grid">
          <div className="col-7">
            <Async state={payouts} what="the receipts" skeleton={<TableSkeleton what="receipts" rows={4} />}>
              {(p) => <Receipts payouts={p.payouts} verified={verifiedIds} />}
            </Async>
          </div>
          <div className="col-5">
            <WebhookLog state={webhooks} />
          </div>
        </div>
      </main>
      <footer className="foot">PayGuard Arena · PayPal AI Hackathon 2026</footer>
    </>
  );
}
