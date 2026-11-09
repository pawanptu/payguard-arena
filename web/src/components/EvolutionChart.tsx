import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { EvolutionRow } from "../types";
import { useMediaQuery } from "../hooks";
import { num1, num2, pct, SEP } from "../format";
import { CardHeader, EmptyState } from "./ui";
import { InboxIcon } from "./Icons";

interface DotProps {
  cx?: number;
  cy?: number;
  payload?: EvolutionRow;
}

const TICK = { fontSize: 12, fill: "var(--muted)" };
const TICK_ATK = { fontSize: 12, fill: "var(--c-atk)" };

/** Promotion diamonds on the champion line; other generations get no marker. */
function PromotionDot({ cx, cy, payload }: DotProps) {
  if (cx == null || cy == null || !payload?.promoted) return <g />;
  return (
    <path
      d={`M${cx} ${cy - 7} L${cx + 7} ${cy} L${cx} ${cy + 7} L${cx - 7} ${cy} Z`}
      fill="var(--surface)"
      stroke="var(--c-champ)"
      strokeWidth={2.5}
    />
  );
}

interface TipProps {
  active?: boolean;
  payload?: { payload?: EvolutionRow }[];
}

function EvoTooltip({ active, payload }: TipProps) {
  const r = payload?.[0]?.payload;
  if (!active || !r) return null;
  return (
    <div className="chart-tip">
      <div className="chart-tip-title">Generation {r.generation}{r.promoted ? " · challenger promoted" : ""}</div>
      <dl>
        <dt>Champion</dt><dd>{num2(r.champion_objective)}</dd>
        <dt>Challenger</dt><dd>{num2(r.challenger_objective)}</dd>
        <dt>Fraud let through</dt>
        <dd>{r.attacker_leakage_pct == null ? "-" : pct(r.attacker_leakage_pct)}{r.attacker_family ? ` (${r.attacker_family})` : ""}</dd>
      </dl>
    </div>
  );
}

export default function EvolutionChart({ rows }: { rows: EvolutionRow[] }) {
  const narrow = useMediaQuery("(max-width: 639px)");

  if (rows.length === 0) {
    return (
      <section className="card" aria-labelledby="evo-h">
        <CardHeader id="evo-h" title="How the swarm improved" />
        <EmptyState icon={<InboxIcon size={22} />} title="No evolution data" cause="This replay has no evolution generations." />
      </section>
    );
  }

  const promotions = rows.filter((r) => r.promoted).length;
  const first = rows[0].champion_objective;
  const last = rows[rows.length - 1].champion_objective;

  return (
    <section className="card evo-card" aria-labelledby="evo-h">
      <CardHeader
        id="evo-h"
        title="How the swarm improved"
        subtitle={`Training score ${num1(first)} → ${num1(last)} over ${rows.length} generations${SEP}${promotions} ${promotions === 1 ? "promotion" : "promotions"}${SEP}not a test result`}
        tipLabel="the evolution chart"
        tip="Lower score is better. Diamond = a challenger was promoted. Dashed line = fraud let through (%)."
      />

      <ul className="legend" aria-hidden="true">
        <li>
          <svg width="30" height="12" viewBox="0 0 30 12"><line x1="0" y1="6" x2="30" y2="6" stroke="var(--c-champ)" strokeWidth="2.5" /><path d="M15 1 L20 6 L15 11 L10 6 Z" fill="var(--surface)" stroke="var(--c-champ)" strokeWidth="2" /></svg>
          Champion
        </li>
        <li>
          <svg width="30" height="12" viewBox="0 0 30 12"><line x1="0" y1="6" x2="30" y2="6" stroke="var(--c-chal)" strokeWidth="1.5" /><circle cx="15" cy="6" r="3" fill="var(--c-chal)" /></svg>
          Challenger
        </li>
        <li>
          <svg width="30" height="12" viewBox="0 0 30 12"><line x1="0" y1="6" x2="30" y2="6" stroke="var(--c-atk)" strokeWidth="2" strokeDasharray="6 4" /></svg>
          Fraud let through
        </li>
      </ul>

      <div className="chart-axes" aria-hidden="true">
        <span>Score</span>
        {!narrow && <span className="axis-atk">Fraud let through %</span>}
      </div>
      <div
        className="chart"
        role="img"
        aria-label={`Line chart of the champion score over ${rows.length} generations, from ${num1(first)} to ${num1(last)}, with ${promotions} promotions. Lower is better. A table follows.`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: narrow ? 8 : 0, bottom: 0, left: 0 }} accessibilityLayer={false}>
            <CartesianGrid vertical={false} stroke="var(--border)" />
            <XAxis dataKey="generation" axisLine={false} tickLine={false} tick={TICK} tickMargin={8} interval="equidistantPreserveStart" minTickGap={14} />
            <YAxis yAxisId="obj" width={40} axisLine={false} tickLine={false} tick={TICK} />
            <YAxis
              yAxisId="leak"
              orientation="right"
              width={44}
              hide={narrow}
              axisLine={false}
              tickLine={false}
              tick={TICK_ATK}
              tickFormatter={(v: number) => `${v}%`}
            />
            <Tooltip content={<EvoTooltip />} cursor={{ stroke: "var(--border-strong)", strokeDasharray: "3 3" }} />
            <Line
              yAxisId="obj"
              type="stepAfter"
              dataKey="challenger_objective"
              name="Challenger score"
              stroke="var(--c-chal)"
              strokeWidth={1.5}
              dot={{ r: 3, fill: "var(--c-chal)", stroke: "var(--c-chal)" }}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
            />
            <Line
              yAxisId="leak"
              type="monotone"
              dataKey="attacker_leakage_pct"
              name="Fraud let through"
              stroke="var(--c-atk)"
              strokeWidth={2}
              strokeDasharray="6 4"
              connectNulls
              dot={false}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
            />
            <Line
              yAxisId="obj"
              type="stepAfter"
              dataKey="champion_objective"
              name="Champion score"
              stroke="var(--c-champ)"
              strokeWidth={2.5}
              dot={<PromotionDot />}
              activeDot={{ r: 5 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <details className="data-details">
        <summary>Show as a table</summary>
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Evolution data table">
          <table>
            <caption className="visually-hidden">Evolution by generation</caption>
            <thead>
              <tr>
                <th scope="col">Gen</th>
                <th scope="col" className="num">Champion score</th>
                <th scope="col" className="num">Challenger score</th>
                <th scope="col">Promoted</th>
                <th scope="col">Attacker family</th>
                <th scope="col" className="num">Fraud let through</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.generation}>
                  <th scope="row">{r.generation}</th>
                  <td className="num">{num2(r.champion_objective)}</td>
                  <td className="num">{num2(r.challenger_objective)}</td>
                  <td>{r.promoted ? "yes" : "no"}</td>
                  <td>{r.attacker_family ?? "-"}</td>
                  <td className="num">{r.attacker_leakage_pct == null ? "-" : pct(r.attacker_leakage_pct)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
}
