import { useState } from "react";
import type { WebhookEvent, WebhooksResponse } from "../types";
import type { AsyncState } from "../hooks";
import { plural, shortId, shortTime } from "../format";
import InfoTip from "./InfoTip";
import { CheckIcon, InboxIcon, RefreshIcon } from "./Icons";
import { CardHeader, EmptyState, ErrorCard, Pill } from "./ui";
import { TableSkeleton } from "./Skeletons";

/** Item status wins (it is the money state); otherwise the batch status. */
const statusOf = (e: WebhookEvent) => e.item_status ?? e.batch_status ?? "-";

export default function WebhookLog({ state }: { state: AsyncState<WebhooksResponse> }) {
  const [msg, setMsg] = useState("");

  if (state.status === "loading") return <TableSkeleton what="webhook events" rows={4} />;
  if (state.status === "error") return <ErrorCard what="webhook events" message={state.error} onRetry={() => void state.reload()} />;
  const data = state.data;
  if (!data) return null;

  const refresh = async () => {
    setMsg("Refreshing");
    await state.reload();
    setMsg("Webhook list refreshed");
  };

  const mixed = new Set(data.events.map((e) => e.source)).size > 1;

  return (
    <section className="card" aria-labelledby="wh-h">
      <CardHeader
        id="wh-h"
        title="Webhooks"
        actions={
          <button
            type="button"
            className="icon-btn"
            aria-label="Refresh webhook events"
            title="Refresh"
            disabled={state.busy}
            onClick={() => void refresh()}
          >
            <RefreshIcon size={16} />
          </button>
        }
      />
      <p className="visually-hidden" role="status" aria-live="polite">{msg}</p>

      {!data.configured ? (
        <EmptyState icon={<InboxIcon size={22} />} title="Webhooks are off on this server." />
      ) : data.events.length === 0 ? (
        <EmptyState icon={<InboxIcon size={22} />} title="No verified webhooks yet." />
      ) : (
        <>
          <p className="wh-summary">
            <CheckIcon size={14} />
            All {data.events.length} {data.events.length === 1 ? "event" : "events"} signature-verified
            <InfoTip label="signature verification">
              Each event passed PayPal&apos;s signature verification before it was recorded. Only ids and statuses are kept.
            </InfoTip>
          </p>
          <div className="table-wrap" tabIndex={0} role="region" aria-label={`${plural(data.events.length, "verified webhook event")}, newest first`}>
            <table>
              <caption className="visually-hidden">Verified PayPal webhook events, newest first</caption>
              <thead>
                <tr>
                  <th scope="col">Event</th>
                  <th scope="col">Batch</th>
                  <th scope="col">Status</th>
                  <th scope="col">Received</th>
                  {mixed && <th scope="col">Source</th>}
                  <th scope="col"><span className="visually-hidden">Signature</span></th>
                </tr>
              </thead>
              <tbody>
                {data.events.map((e) => (
                  <tr key={e.event_id}>
                    <td><code>{e.event_type}</code></td>
                    <td>{e.payout_batch_id ? <code title={e.payout_batch_id}>{shortId(e.payout_batch_id)}</code> : "-"}</td>
                    <td>{statusOf(e)}</td>
                    <td>{shortTime(e.received_at)}</td>
                    {mixed && <td><Pill variant="chip">{e.source}</Pill></td>}
                    <td className="check-cell">
                      <span className="verified-icon"><CheckIcon size={14} /></span>
                      <span className="visually-hidden">Signature verified</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}
