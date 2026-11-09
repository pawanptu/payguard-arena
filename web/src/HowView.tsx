import { useMemo } from "react";
import type { ReactNode } from "react";
import { fetchAllEvents, fetchPayouts, fetchSummary, fetchWebhooks } from "./api";
import { useAsync } from "./hooks";
import type { AsyncState } from "./hooks";
import { plural, SEP } from "./format";
import TestCards from "./components/Hero";
import DecisionFeed from "./components/DecisionFeed";
import EvolutionChart from "./components/EvolutionChart";
import ResultsTable from "./components/ResultsTable";
import Receipts from "./components/Receipts";
import WebhookLog from "./components/WebhookLog";
import { ChartSkeleton, FeedSkeleton, TableSkeleton } from "./components/Skeletons";
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

const STEPS: { title: string; text: string }[] = [
  { title: "Score", text: "Four specialists score each payout: velocity, graph, drift and policy." },
  { title: "Combine", text: "Their trust-weighted vote sets the risk. An anomaly layer can only raise it." },
  {
    title: "Review",
    text: "Risk above the step-up or hold line queues the payout for a reviewer, with an AI explanation grounded in the evidence.",
  },
  { title: "Verify", text: "The ops copilot checks the PayPal sandbox through the official Agent Toolkit." },
];

/** "How it works": the four steps, then the learning curve, the replay and the PayPal sandbox proof. */
export default function HowView() {
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

  // "N LIVE payouts, K signature-verified webhooks" for the proof section (null until both are known).
  const proofLine = useMemo(() => {
    if (!payouts.data) return null;
    const liveIds = new Set(payouts.data.payouts.filter((p) => p.mode === "live").map((p) => p.payout_batch_id));
    const k = webhooks.data
      ? webhooks.data.events.filter((w) => w.payout_batch_id && liveIds.has(w.payout_batch_id)).length
      : null;
    return `${plural(liveIds.size, "LIVE payout")}${k !== null ? `${SEP}${plural(k, "signature-verified webhook")}` : ""}`;
  }, [payouts.data, webhooks.data]);

  return (
    <>
      <div className="page-head">
        <h1>How PayGuard works</h1>
        <p className="page-sub">From a payout to a reviewer&apos;s decision, in four steps.</p>
      </div>
      <ol className="steps" aria-label="The four steps">
        {STEPS.map((st, i) => (
          <li key={st.title} className="step">
            <span className="step-n" aria-hidden="true">{i + 1}</span>
            <div>
              <p className="step-title">{st.title}</p>
              <p className="step-text">{st.text}</p>
            </div>
          </li>
        ))}
      </ol>

      <div className="grid">
        <div className="col-5">
          <Async state={summary} what="the learning curve" skeleton={<ChartSkeleton />}>
            {(sum) => <EvolutionChart rows={sum.evolution} />}
          </Async>
        </div>
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
      </div>

      <section id="paypal" className="proof-section" aria-labelledby="proof-h">
        <div className="section-head">
          <h2 id="proof-h">PayPal sandbox proof</h2>
          <p className="page-sub">
            Test-money payouts sent to the PayPal sandbox, and the webhooks PayPal sent back.
            {proofLine && <> {proofLine}.</>}
          </p>
        </div>
        <div className="grid">
          <div className="col-7">
            <Async state={payouts} what="the receipts" skeleton={<TableSkeleton what="receipts" rows={4} />}>
              {(p) => <Receipts payouts={p.payouts} verified={verifiedIds} />}
            </Async>
          </div>
          <div className="col-5">
            <WebhookLog state={webhooks} />
          </div>
        </div>
      </section>

      <details className="earlier">
        <summary>Earlier test cards</summary>
        <p className="page-sub">The first dashboard&apos;s held-out tests. Rounds 1 and 2 are also on the Results tab.</p>
        <Async state={summary} what="the earlier tests" skeleton={<TableSkeleton what="earlier tests" rows={4} />}>
          {(sum) => (
            <>
              <TestCards results={sum.results} />
              <div className="earlier-table">
                <ResultsTable rows={sum.results} />
              </div>
            </>
          )}
        </Async>
      </details>
    </>
  );
}
