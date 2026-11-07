import type { Payout } from "../types";
import { usd } from "../format";

function ModeBadge({ mode }: { mode: Payout["mode"] }) {
  return mode === "live" ? (
    <span className="badge badge-live">LIVE (PayPal sandbox)</span>
  ) : (
    <span className="badge badge-sim">SIM (no PayPal call)</span>
  );
}

export default function Receipts({ payouts }: { payouts: Payout[] }) {
  return (
    <section className="card" aria-labelledby="rec-h">
      <h2 id="rec-h">PayPal receipts</h2>
      <p className="muted small">
        SIM rows are simulated payouts. LIVE rows are real payouts made through the PayPal <strong>sandbox</strong>{" "}
        (test money only).
      </p>
      {payouts.length === 0 ? (
        <p className="muted">No payouts recorded in this replay.</p>
      ) : (
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Payout receipts, scrolls sideways on narrow screens">
          <table>
            <caption className="visually-hidden">Payout receipts</caption>
            <thead>
              <tr>
                <th scope="col">Mode</th>
                <th scope="col">Event</th>
                <th scope="col" className="num">Amount</th>
                <th scope="col">Sender batch</th>
                <th scope="col">Payout batch</th>
                <th scope="col">Batch status</th>
                <th scope="col">Item status</th>
                <th scope="col">Created</th>
              </tr>
            </thead>
            <tbody>
              {payouts.map((p) => (
                <tr key={`${p.sender_batch_id}|${p.event_id}`}>
                  <td><ModeBadge mode={p.mode} /></td>
                  <td>#{p.event_id}</td>
                  <td className="num">{usd(p.amount_usd)}</td>
                  <td><code>{p.sender_batch_id}</code></td>
                  <td><code>{p.payout_batch_id}</code></td>
                  <td>{p.batch_status}</td>
                  <td>{p.item_status}</td>
                  <td>{p.created_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
