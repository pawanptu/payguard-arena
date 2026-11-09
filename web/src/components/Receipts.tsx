import { useEffect, useRef, useState } from "react";
import type { Payout } from "../types";
import { plural, shortId, shortTime, usd } from "../format";
import { CheckIcon, CopyIcon, InboxIcon } from "./Icons";
import { TIPS } from "../tips";
import { CardHeader, EmptyState, LivePill, Pill, SimPill } from "./ui";

function PayoutTable({
  rows,
  verified,
  onCopy,
  caption,
}: {
  rows: Payout[];
  verified: Set<string> | null;
  onCopy: (id: string) => void;
  caption: string;
}) {
  return (
    <div className="table-wrap" tabIndex={0} role="region" aria-label={caption}>
      <table className="stack-table" role="table">
        <caption className="visually-hidden">{caption}</caption>
        <thead role="rowgroup">
          <tr role="row">
            <th role="columnheader" scope="col">Mode</th>
            <th role="columnheader" scope="col" className="num">Amount</th>
            <th role="columnheader" scope="col">Status</th>
            <th role="columnheader" scope="col">Payout batch</th>
            <th role="columnheader" scope="col">Created</th>
            <th role="columnheader" scope="col">Webhook</th>
          </tr>
        </thead>
        <tbody role="rowgroup">
          {rows.map((p) => (
            <tr role="row" key={`${p.sender_batch_id}|${p.event_id}`}>
              <td role="cell" data-label="Mode">{p.mode === "live" ? <LivePill /> : <SimPill />}</td>
              <td role="cell" data-label="Amount" className="num">{usd(p.amount_usd)}</td>
              <td role="cell" data-label="Status"><Pill variant="chip">{p.item_status}</Pill></td>
              <td role="cell" data-label="Payout batch">
                <span className="batch">
                  <code title={`Payout batch ${p.payout_batch_id}. Sender batch ${p.sender_batch_id}. Event #${p.event_id}.`}>
                    {shortId(p.payout_batch_id)}
                  </code>
                  <button
                    type="button"
                    className="icon-btn icon-btn-sm"
                    aria-label={`Copy payout batch id ${shortId(p.payout_batch_id)}`}
                    onClick={() => onCopy(p.payout_batch_id)}
                  >
                    <CopyIcon size={14} />
                  </button>
                </span>
              </td>
              <td role="cell" data-label="Created">{shortTime(p.created_at)}</td>
              <td role="cell" data-label="Webhook">
                {verified?.has(p.payout_batch_id) ? (
                  <Pill variant="verified">
                    <CheckIcon size={12} />
                    Verified
                  </Pill>
                ) : (
                  <span className="muted">-</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Receipts({ payouts, verified }: { payouts: Payout[]; verified: Set<string> | null }) {
  const [showSim, setShowSim] = useState(false);
  const [msg, setMsg] = useState("");
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);

  const live = payouts.filter((p) => p.mode === "live");
  const sim = payouts.filter((p) => p.mode !== "live");

  const copy = async (id: string) => {
    let text = "Copy failed";
    try {
      await navigator.clipboard.writeText(id);
      text = "Copied";
    } catch {
      /* clipboard unavailable (insecure context or denied) */
    }
    setMsg(text);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setMsg(""), 2000);
  };

  return (
    <section className="card" aria-labelledby="rec-h">
      <CardHeader
        id="rec-h"
        level={3}
        title="PayPal sandbox receipts"
        tipLabel="LIVE and SIM rows"
        tip={TIPS.liveSim}
        subtitle="Test-money payouts sent to the PayPal sandbox."
      />
      <p className="visually-hidden" role="status" aria-live="polite">{msg}</p>
      {msg && <span className="toast" aria-hidden="true">{msg}</span>}

      {payouts.length === 0 ? (
        <EmptyState icon={<InboxIcon size={22} />} title="No sandbox payouts yet" cause="LIVE receipts appear here after a payout run." />
      ) : (
        <>
          {live.length > 0 ? (
            <PayoutTable rows={live} verified={verified} onCopy={copy} caption="Live PayPal sandbox payouts" />
          ) : (
            <p className="muted">No LIVE payouts in this replay.</p>
          )}
          {sim.length > 0 && (
            <div className="sim-block">
              <button type="button" className="btn" aria-expanded={showSim} onClick={() => setShowSim((s) => !s)}>
                {showSim ? "Hide" : "Show"} {plural(sim.length, "simulated payout")}
              </button>
              {showSim && <PayoutTable rows={sim} verified={verified} onCopy={copy} caption="Simulated payouts" />}
            </div>
          )}
        </>
      )}
    </section>
  );
}
