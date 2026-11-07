import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EvolutionRow } from "../types";
import { num2, pct } from "../format";

interface DotProps {
  cx?: number;
  cy?: number;
  payload?: EvolutionRow;
}

/** Promoted generations get a diamond; others a small circle (shape, not just color, carries the meaning). */
function ChampionDot({ cx, cy, payload }: DotProps) {
  if (cx == null || cy == null || !payload) return null;
  if (payload.promoted) {
    return <path className="dot-promoted" d={`M${cx} ${cy - 8} L${cx + 8} ${cy} L${cx} ${cy + 8} L${cx - 8} ${cy} Z`} />;
  }
  return <circle className="dot-champ" cx={cx} cy={cy} r={3} />;
}

export default function EvolutionChart({ rows, synthetic }: { rows: EvolutionRow[]; synthetic: boolean }) {
  const promotions = rows.filter((r) => r.promoted).length;
  return (
    <section className="card" aria-labelledby="evo-h">
      <h2 id="evo-h">Evolution</h2>
      <p className="muted small">
        Champion objective per generation (<strong>lower is better</strong>). Diamonds mark generations where a
        challenger was promoted. Dashed line: leakage of the attacker family tried that generation.
        {synthetic && " In this sample file the evolution numbers are placeholders, not a run."}
      </p>
      {rows.length === 0 ? (
        <p className="muted">This replay has no evolution data.</p>
      ) : (
        <>
          <ul className="legend" aria-hidden="true">
            <li>
              <svg width="34" height="14" viewBox="0 0 34 14"><line className="lg-champ" x1="0" y1="7" x2="34" y2="7" strokeWidth="2.5" /><path className="dot-promoted" d="M17 2 L22 7 L17 12 L12 7 Z" /></svg>
              Champion objective (diamond = promotion)
            </li>
            <li>
              <svg width="34" height="14" viewBox="0 0 34 14"><line className="lg-atk" x1="0" y1="7" x2="34" y2="7" strokeWidth="2" strokeDasharray="6 4" /></svg>
              Attacker leakage (right axis)
            </li>
          </ul>
          <div className="chart" role="img" aria-label={`Line chart of champion objective over ${rows.length} generations with ${promotions} promotions. Lower is better. A data table follows.`}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rows} margin={{ top: 12, right: 12, bottom: 24, left: 4 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="generation" label={{ value: "Generation", position: "insideBottom", offset: -12 }} />
                <YAxis
                  yAxisId="obj"
                  width={56}
                  label={{ value: "Objective (lower is better)", angle: -90, position: "insideLeft", offset: 10, style: { textAnchor: "middle" } }}
                />
                <YAxis
                  yAxisId="leak"
                  orientation="right"
                  width={44}
                  tickFormatter={(v: number) => `${v}%`}
                />
                <Tooltip
                  formatter={(v, name) => (name === "Attacker leakage" ? pct(Number(v)) : num2(Number(v)))}
                  labelFormatter={(g) => `Generation ${g}`}
                />
                <Line
                  yAxisId="obj"
                  className="line-champ"
                  type="stepAfter"
                  dataKey="champion_objective"
                  name="Champion objective"
                  strokeWidth={2.5}
                  dot={<ChampionDot />}
                  activeDot={{ r: 6 }}
                  isAnimationActive={false}
                />
                <Line
                  yAxisId="leak"
                  className="line-atk"
                  type="monotone"
                  dataKey="attacker_leakage_pct"
                  name="Attacker leakage"
                  strokeWidth={2}
                  strokeDasharray="6 4"
                  connectNulls
                  dot={{ r: 3 }}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <details>
            <summary>Show chart data as a table</summary>
            <div className="table-wrap" tabIndex={0} role="region" aria-label="Evolution data table">
              <table>
                <caption className="visually-hidden">Evolution by generation</caption>
                <thead>
                  <tr>
                    <th scope="col">Gen</th>
                    <th scope="col" className="num">Champion obj.</th>
                    <th scope="col" className="num">Challenger obj.</th>
                    <th scope="col">Promoted</th>
                    <th scope="col">Attacker family</th>
                    <th scope="col" className="num">Attacker leakage</th>
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
        </>
      )}
    </section>
  );
}
