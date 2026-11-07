import { useCallback, useEffect, useState } from "react";
import { fetchWebhooks } from "../api";
import { shortId, shortTime } from "../format";
import type { WebhookEvent, WebhooksResponse } from "../types";

/** Item status wins (it is the money state); otherwise the batch status. */
const statusOf = (e: WebhookEvent) => e.item_status ?? e.batch_status ?? "-";

export default function WebhookLog() {
  const [data, setData] = useState<WebhooksResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setError(null);
    fetchWebhooks()
      .then(setData)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(load, [load]);

  return (
    <section className="card" aria-labelledby="wh-h">
      <div className="card-head">
        <h2 id="wh-h">Verified PayPal webhooks</h2>
        <button type="button" className="btn" onClick={load}>Refresh</button>
      </div>
      <p className="muted small">
        Each event was checked with PayPal&apos;s signature verification before it was recorded. Only ids and
        statuses are kept.
      </p>
      {error && <p role="alert">Could not load webhook events: {error}</p>}
      {!data && !error && <p className="muted" role="status">Loading webhook events...</p>}
      {data && !data.configured && <p className="muted" role="status">Webhook not configured on this server</p>}
      {data && data.configured && data.events.length === 0 && (
        <p className="muted" role="status">No verified webhook yet</p>
      )}
      {data && data.events.length > 0 && (
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Verified webhook events, scrolls sideways on narrow screens">
          <table>
            <caption className="visually-hidden">Verified PayPal webhook events, newest first</caption>
            <thead>
              <tr>
                <th scope="col">Event</th>
                <th scope="col">Batch</th>
                <th scope="col">Status</th>
                <th scope="col">Received</th>
                <th scope="col">Check</th>
                <th scope="col">Source</th>
              </tr>
            </thead>
            <tbody>
              {data.events.map((e) => (
                <tr key={e.event_id}>
                  <td><code>{e.event_type}</code></td>
                  <td>{e.payout_batch_id ? <code title={e.payout_batch_id}>{shortId(e.payout_batch_id)}</code> : "-"}</td>
                  <td>{statusOf(e)}</td>
                  <td>{shortTime(e.received_at)}</td>
                  <td><span className="badge badge-verified"><span aria-hidden="true">✓ </span>signature verified</span></td>
                  <td><span className="badge badge-sim">{e.source}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
