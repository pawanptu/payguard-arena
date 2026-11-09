import type { ReactElement } from "react";
import { Bar, BarChart, Cell, CartesianGrid, LabelList, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ErrorBar } from "recharts";
import { fetchRounds } from "../api";
import { useAsync, useMediaQuery } from "../hooks";
import type { Hypothesis, PairedMargin, RoundInfo, RoundRow, RoundTone } from "../types";
import { defenderLabel, num1, num2, pct, SEP, SPLIT_LABEL, signed1 } from "../format";
import InfoTip from "../components/InfoTip";
import { CheckIcon, CrossIcon, InboxIcon, WarnIcon } from "../components/Icons";
import { EmptyState, ErrorCard, Pill, SimPill, Skeleton } from "../components/ui";
import type { Split } from "../types";

const TICK = { fontSize: 11, fill: "var(--muted)" };
const SIM_NOTE = "Simulated, synthetic data";

const TONE_ICON: Record<RoundTone, ReactElement> = {
  fail: <CrossIcon size={18} />,
  pass: <CheckIcon size={18} />,
  mixed: <WarnIcon size={18} />,
};
const TONE_NAME: Record<RoundTone, string> = { fail: "Not met", pass: "Met", mixed: "Mixed" };

function barColor(role: string): string {
  if (role === "primary") return "var(--accent)";
  if (role === "baseline") return "var(--bar-base)";
  return "var(--bar-other)";
}

const sortByObjective = (rows: RoundRow[]) => [...rows].sort((a, b) => a.objective - b.objective);

/* ---------- Bars: objective per defender ---------- */

function ObjectiveBars({ rows, title }: { rows: RoundRow[]; title: string }) {
  const sorted = sortByObjective(rows);
  const data = sorted.map((r) => ({ ...r, label: defenderLabel(r.defender) }));
  const h = data.length * 24 + 52;
  return (
    <figure className="fig" aria-label={`${title}: objective per defender`}>
      <ul className="legend" aria-hidden="true">
        <li><span className="swatch" style={{ background: "var(--accent)" }} />Primary</li>
        <li><span className="swatch" style={{ background: "var(--bar-base)" }} />Baselines</li>
        <li><span className="swatch" style={{ background: "var(--bar-other)" }} />Other defenders</li>
      </ul>
      <div className="rbars" style={{ height: h }}>
        <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 320, height: h }}>
          <BarChart data={data} layout="vertical" margin={{ top: 2, right: 34, bottom: 22, left: 0 }} barCategoryGap={4}>
            <CartesianGrid horizontal={false} stroke="var(--border)" />
            <XAxis
              type="number"
              domain={[0, (max: number) => Math.max(20, Math.ceil(max / 20) * 20)]}
              allowDecimals={false}
              tickCount={4}
              tick={TICK}
              tickLine={false}
              stroke="var(--border-strong)"
              label={{ value: "Objective (lower is better)", position: "insideBottom", offset: -12, fontSize: 11, fill: "var(--muted)" }}
            />
            <YAxis type="category" dataKey="label" width={132} tick={TICK} tickLine={false} interval={0} stroke="var(--border-strong)" />
            <Bar dataKey="objective" radius={[0, 3, 3, 0]} isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.defender} fill={barColor(d.role)} stroke={d.role === "primary" ? "var(--text)" : "none"} strokeWidth={1.5} />
              ))}
              <LabelList dataKey="objective" position="right" formatter={(v: unknown) => num1(Number(v))} style={{ fontSize: 11, fill: "var(--text)" }} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="fig-cap">
        <SimPill /> {SIM_NOTE}
      </figcaption>
      <details className="data-details">
        <summary>Show as a table</summary>
        <RowsTable rows={sorted} />
      </details>
    </figure>
  );
}

function RowsTable({ rows }: { rows: RoundRow[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th scope="col">Defender</th>
            <th scope="col" className="num">Objective</th>
            <th scope="col" className="num">Leakage</th>
            <th scope="col" className="num">Honest blocked</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.defender} className={r.role === "primary" ? "row-primary" : undefined}>
              <th scope="row">
                <span className="def">
                  {defenderLabel(r.defender)}
                  {r.role === "primary" && <Pill variant="headline">primary</Pill>}
                </span>
              </th>
              <td className="num">{num1(r.objective)}</td>
              <td className="num">{pct(r.leakage_pct)}</td>
              <td className="num">{pct(r.honest_block_rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------- Round cards ---------- */

function RoundCard({ r }: { r: RoundInfo }) {
  const tone = r.tone;
  const id = `round-${r.round}`;
  return (
    <section className="card round" aria-labelledby={id}>
      <h2 id={id}>{r.title}</h2>
      <p className={`verdict-badge tone-${tone}`}>
        {TONE_ICON[tone]}
        <span className="visually-hidden">{TONE_NAME[tone]}: </span>
        <span>{r.verdict}</span>
      </p>
      <p className="round-meta">
        {r.split && <>{SPLIT_LABEL[r.split as Split] ?? r.split}{SEP}</>}
        {r.family && <>family {r.family}{SEP}</>}
        {r.decision_ref && <>decision {r.decision_ref}</>}
      </p>
      {r.rule && (
        <p className="round-rule">
          <strong>Rule, fixed before the run:</strong> {r.rule}
        </p>
      )}
      {r.rows && r.rows.length > 0 && <ObjectiveBars rows={r.rows} title={r.title} />}
      {r.caveats && r.caveats.length > 0 && (
        <div className="round-cav">
          <h3>Caveats</h3>
          <ul className="test-caveats" aria-label={`${r.title}: caveats`}>
            {r.caveats.map((c) => (
              <li key={c} className="test-caveat">{c}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

/* ---------- Round 3: dreamer panel ---------- */

function hypTone(v: string): "release" | "hold" | "step" {
  const u = v.toUpperCase();
  if (u.startsWith("GO") || u === "PASS") return "release";
  if (u.startsWith("FAIL") || u.startsWith("NO")) return "hold";
  return "step";
}

function Hyp({ h }: { h: Hypothesis }) {
  const tone = hypTone(h.verdict);
  return (
    <li className="hyp">
      <div className="hyp-head">
        <span className="hyp-id">{h.id}</span>
        <Pill variant={tone}>
          {tone === "release" ? <CheckIcon size={12} /> : tone === "hold" ? <CrossIcon size={12} /> : <WarnIcon size={12} />}
          {h.verdict}
        </Pill>
      </div>
      <p>{h.text}</p>
    </li>
  );
}

/** Round axis bounds and tick marks (zero is always a tick). */
function niceScale(lo: number, hi: number): { min: number; max: number; ticks: number[] } {
  const raw = Math.max(hi - lo, 1) / 6;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((x) => x >= raw) ?? 10 * mag;
  const min = Math.floor(lo / step) * step;
  const max = Math.ceil(hi / step) * step;
  const ticks: number[] = [];
  for (let t = min; t <= max + step / 2; t += step) ticks.push(Math.round(t * 1e6) / 1e6);
  return { min, max, ticks };
}

interface ForestPoint {
  y: number;
  label: string;
  mean: number;
  lo: number;
  hi: number;
  err: [number, number];
}

function ForestTip({ active, payload }: { active?: boolean; payload?: { payload?: ForestPoint }[] }) {
  const p = payload?.[0]?.payload;
  if (!active || !p) return null;
  return (
    <div className="chart-tip">
      <div className="chart-tip-title">{p.label}</div>
      <dl>
        <dt>Mean</dt><dd>{signed1(p.mean)}</dd>
        <dt>95% CI</dt><dd>{signed1(p.lo)} to {signed1(p.hi)}</dd>
      </dl>
    </div>
  );
}

function ForestPlot({ paired }: { paired: PairedMargin[] }) {
  const narrow = useMediaQuery("(max-width: 639px)");
  const pts: ForestPoint[] = [...paired]
    .sort((a, b) => a.mean - b.mean)
    .map((p, i) => ({
      y: i,
      label: defenderLabel(p.arm),
      mean: p.mean,
      lo: p.ci95[0],
      hi: p.ci95[1],
      err: [p.mean - p.ci95[0], p.ci95[1] - p.mean],
    }));
  const { min, max, ticks } = niceScale(Math.min(0, ...pts.map((p) => p.lo)), Math.max(0, ...pts.map((p) => p.hi)));
  const h = pts.length * 44 + 64;
  return (
    <figure className="fig forest">
      <div className="forest-plot" style={{ height: h }}>
        <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 480, height: h }}>
          <ScatterChart margin={{ top: 8, right: 20, bottom: 22, left: 4 }}>
            <CartesianGrid horizontal={false} stroke="var(--border)" />
            <XAxis
              type="number"
              dataKey="mean"
              domain={[min, max]}
              ticks={ticks}
              tick={TICK}
              tickLine={false}
              stroke="var(--border-strong)"
              label={{ value: "Objective points, arm minus full dreamer", position: "insideBottom", offset: -12, fontSize: 11, fill: "var(--muted)" }}
            />
            <YAxis
              type="number"
              dataKey="y"
              reversed
              domain={[-0.6, pts.length - 0.4]}
              ticks={pts.map((p) => p.y)}
              tickFormatter={(i: number) => pts[i]?.label ?? ""}
              width={narrow ? 112 : 150}
              tick={TICK}
              tickLine={false}
              interval={0}
              stroke="var(--border-strong)"
            />
            <ReferenceLine x={0} stroke="var(--text)" strokeDasharray="4 3" strokeWidth={1.5} />
            <Tooltip content={<ForestTip />} cursor={false} />
            <Scatter data={pts} fill="var(--accent)" isAnimationActive={false}>
              <ErrorBar dataKey="err" direction="x" width={7} stroke="var(--accent)" strokeWidth={2} />
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div className="forest-ends" aria-hidden="true">
        <span>← leaked less than the full dreamer</span>
        <span>leaked more →</span>
      </div>
      <figcaption className="fig-cap">
        <SimPill /> {SIM_NOTE}{SEP}dot = mean over episodes, bar = 95% bootstrap interval
      </figcaption>
      <details className="data-details">
        <summary>Show as a table</summary>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Arm</th>
                <th scope="col" className="num">Mean</th>
                <th scope="col" className="num">95% CI</th>
              </tr>
            </thead>
            <tbody>
              {pts.map((p) => (
                <tr key={p.label}>
                  <th scope="row">{p.label}</th>
                  <td className="num">{signed1(p.mean)}</td>
                  <td className="num">{signed1(p.lo)} to {signed1(p.hi)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

function ReportOnly({ rows, ro }: { rows: RoundRow[]; ro: NonNullable<RoundInfo["report_only"]> }) {
  const imm = ro.immunity_attempts ?? {};
  const nev = ro.never_contained_rate ?? {};
  const arms = rows.length > 0 ? rows.map((r) => r.defender) : Object.keys(imm);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th scope="col">Arm</th>
            <th scope="col" className="num">Immunity attempts</th>
            <th scope="col" className="num">Never contained</th>
          </tr>
        </thead>
        <tbody>
          {arms.map((a) => (
            <tr key={a}>
              <th scope="row">{defenderLabel(a)}</th>
              <td className="num">{imm[a] !== undefined ? num1(imm[a]) : "-"}</td>
              <td className="num">{nev[a] !== undefined ? pct(nev[a]) : "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DreamerPanel({ r }: { r: RoundInfo }) {
  const rows = r.rows ?? [];
  const sorted = sortByObjective(rows);
  const spent = Object.entries(r.spent_splits ?? {}).map(([k, v]) => ({ split: k, objective: typeof v === "number" ? v : v.objective }));
  return (
    <section className="dreamer" aria-labelledby="dreamer-h">
      <h2 id="dreamer-h">Round 3: the dreamer panel</h2>
      <p className="page-sub">
        One full dreamer against three ablations, the swarm from round 2 and two baselines
        {r.split && <>{SEP}{r.split}</>}
        {r.family && <>{SEP}family {r.family}</>}.
      </p>

      {r.hypotheses && r.hypotheses.length > 0 && (
        <ul className="hyps" aria-label="Pre-registered hypotheses">
          {r.hypotheses.map((h) => (
            <Hyp key={h.id} h={h} />
          ))}
        </ul>
      )}

      <div className="dreamer-grid">
        {r.paired && r.paired.length > 0 && (
          <section className="card" aria-labelledby="forest-h">
            <header className="card-head">
              <div className="card-head-main">
                <h3 id="forest-h" className="card-h3">
                  Each arm against the full dreamer
                  <InfoTip label="the forest plot">
                    Per episode, arm minus full dreamer on the objective, with a 95% bootstrap interval (2,000 resamples). Negative means the
                    arm leaked less than the full dreamer; positive means it leaked more.
                  </InfoTip>
                </h3>
                <p className="card-sub">Left of the zero line: the arm did better than the full dreamer.</p>
              </div>
            </header>
            <ForestPlot paired={r.paired} />
          </section>
        )}

        {sorted.length > 0 && (
          <section className="card" aria-labelledby="arms-h">
            <header className="card-head">
              <div className="card-head-main">
                <h3 id="arms-h" className="card-h3">The seven arms</h3>
                <p className="card-sub">Pooled over all episodes{SEP}lower objective is better</p>
              </div>
            </header>
            <RowsTable rows={sorted} />
            <p className="fig-cap"><SimPill /> {SIM_NOTE}</p>
          </section>
        )}

        {r.report_only && (r.report_only.immunity_attempts || r.report_only.never_contained_rate) && (
          <section className="card" aria-labelledby="ro-h">
            <header className="card-head">
              <div className="card-head-main">
                <h3 id="ro-h" className="card-h3">
                  Report only
                  <InfoTip label="report-only measures">
                    Reported for context, not part of the decision rule. Immunity attempts: attack payouts that succeed before a campaign is
                    contained, on average. Never contained: share of campaigns that were never contained.
                  </InfoTip>
                </h3>
                <p className="card-sub">Not part of the decision rule</p>
              </div>
            </header>
            <ReportOnly rows={sorted} ro={r.report_only} />
            <p className="fig-cap"><SimPill /> {SIM_NOTE}</p>
          </section>
        )}

        {spent.length > 0 && (
          <section className="card" aria-labelledby="spent-h">
            <header className="card-head">
              <div className="card-head-main">
                <h3 id="spent-h" className="card-h3">The dreamer&apos;s spent splits</h3>
                <p className="card-sub">
                  Already used for training or in earlier rounds, so these are for reference, not clean tests.
                </p>
              </div>
            </header>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Split</th>
                    <th scope="col" className="num">Objective</th>
                  </tr>
                </thead>
                <tbody>
                  {spent.map((s) => (
                    <tr key={s.split}>
                      <th scope="row">{SPLIT_LABEL[s.split as Split] ?? s.split}</th>
                      <td className="num">{num2(s.objective)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="fig-cap"><SimPill /> {SIM_NOTE}</p>
          </section>
        )}
      </div>
    </section>
  );
}

/* ---------- View ---------- */

export default function ResultsView() {
  const state = useAsync(fetchRounds);
  const data = state.data;
  const round3 = data?.rounds.find((r) => r.round === 3);

  return (
    <>
      <div className="page-head">
        <h1>Three pre-registered rounds</h1>
        <p className="page-sub">
          Each round was scored on attack families the defender had never seen, against rules written down before the run.
          {data && (
            <>
              {" "}Objective: <code>{data.objective}</code>.
            </>
          )}
        </p>
      </div>
      {state.status === "error" ? (
        <ErrorCard what="the results" message={state.error} onRetry={() => void state.reload()} />
      ) : data === null ? (
        <div className="round-grid" aria-busy="true">
          <span className="visually-hidden">Loading the results</span>
          {[0, 1, 2].map((i) => (
            <div key={i} className="card"><Skeleton className="skel-title" /><Skeleton className="skel-chart" /></div>
          ))}
        </div>
      ) : data.rounds.length === 0 ? (
        <div className="card">
          <EmptyState icon={<InboxIcon size={22} />} title="No rounds in the results yet" />
        </div>
      ) : (
        <>
          <div className="round-grid">
            {data.rounds.map((r) => (
              <RoundCard key={r.round} r={r} />
            ))}
          </div>
          {round3 && <DreamerPanel r={round3} />}
        </>
      )}
    </>
  );
}
