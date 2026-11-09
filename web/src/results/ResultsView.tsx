import type { ReactElement } from "react";
import { Bar, BarChart, Cell, CartesianGrid, LabelList, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ErrorBar } from "recharts";
import { fetchRounds } from "../api";
import { useAsync, useMediaQuery } from "../hooks";
import type { Hypothesis, PairedMargin, RoundInfo, RoundRow, RoundTone, Split } from "../types";
import { defenderLabel, num1, num2, pct, SEP, SPLIT_LABEL, signed1 } from "../format";
import { TIPS } from "../tips";
import InfoTip from "../components/InfoTip";
import { CheckIcon, CrossIcon, InboxIcon, WarnIcon } from "../components/Icons";
import { CardHeader, EmptyState, ErrorCard, Pill, SimPill, Skeleton } from "../components/ui";

/** Friendly split names; the round-3 sealed split is not in the replay contract's Split type. */
function splitName(split: string): string {
  if (split.endsWith("3")) return "Round 3 family";
  return SPLIT_LABEL[split as Split] ?? split;
}

const TICK = { fontSize: 12, fill: "var(--muted)" };
const SIM_NOTE = "Simulated data";

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

const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
const word = (n: number) => WORDS[n] ?? String(n);

function hypTone(v: string): "release" | "hold" | "step" {
  const u = v.toUpperCase();
  if (u.startsWith("GO") || u === "PASS") return "release";
  if (u.startsWith("FAIL") || u.startsWith("NO")) return "hold";
  return "step";
}

/** "failed", "passed with caveats", "passed one test and failed one": the outcome of a round, from its data. */
function outcomeWords(r: RoundInfo): string {
  if (r.hypotheses && r.hypotheses.length > 0) {
    const go = r.hypotheses.filter((h) => hypTone(h.verdict) === "release").length;
    const bad = r.hypotheses.length - go;
    if (go > 0 && bad > 0) return `passed ${word(go)} ${go === 1 ? "test" : "tests"} and failed ${word(bad)}`;
    return bad === 0 ? "passed" : "failed";
  }
  if (r.tone === "fail") return "failed";
  if (r.tone === "pass") return /caveat/i.test(r.verdict) ? "passed with caveats" : "passed";
  return r.verdict.toLowerCase();
}

/** "Failed: 80.6 against static rules' 28.5." for rounds scored by one pass rule. */
function plainLine(r: RoundInfo): string | null {
  const p = r.rows?.find((x) => x.role === "primary");
  const s = r.rows?.find((x) => x.defender === "static_rules");
  if (!p || !s) return null;
  const w = outcomeWords(r);
  return `${w.charAt(0).toUpperCase()}${w.slice(1)}: ${num1(p.objective)} against static rules' ${num1(s.objective)}.`;
}

/** The pass rule in plain words when the server's rule string has the usual shape; otherwise null. */
function ruleSentence(r: RoundInfo): string | null {
  if (!r.rule || (r.hypotheses && r.hypotheses.length > 0)) return null;
  const m = /(\d+(?:\.\d+)?) below static_rules and single_policy/.exec(r.rule);
  if (!m) return null;
  return `The tested defender must score at least ${m[1]} points better than static rules and the policy-only agent, and hold back no more honest payouts than static rules.`;
}

function hypSentence(h: Hypothesis): string {
  const n = h.numbers;
  if (h.id === "H1" && n && Number.isFinite(n.margin))
    return `The full dreamer beats static rules and the best single agent by at least ${n.margin.toFixed(1)} points, and holds back no more honest payouts than static rules.`;
  if (h.id === "H2" && n && Number.isFinite(n.margin))
    return `The full dreamer beats its version without a world model by at least ${n.margin.toFixed(1)} points.`;
  return h.text;
}

/** The measured numbers behind a hypothesis, in words. */
function hypNumbers(h: Hypothesis): string | null {
  const n = h.numbers;
  if (!n || !Number.isFinite(n.objective)) return null;
  if (h.id === "H1" && Number.isFinite(n.static_rules) && Number.isFinite(n.single_policy))
    return `Full dreamer ${num2(n.objective)} vs static rules ${num2(n.static_rules)} and policy-only ${num2(n.single_policy)}.`;
  if (h.id === "H2" && Number.isFinite(n.dreamer_nowm) && Number.isFinite(n.margin))
    return `Full dreamer ${num2(n.objective)} vs no world model ${num2(n.dreamer_nowm)}, needed ${n.margin.toFixed(1)} better.`;
  return null;
}

/* ---------- Bars: score per defender ---------- */

function barSummary(sorted: RoundRow[]): string {
  const best = sorted[0];
  const primary = sorted.find((r) => r.role === "primary");
  const stat = sorted.find((r) => r.defender === "static_rules");
  let t = `Bar chart of the score of ${sorted.length} defenders, lower is better. Best: ${defenderLabel(best.defender)} ${num1(best.objective)}.`;
  if (primary) t += ` Tested defender ${defenderLabel(primary.defender)}: ${num1(primary.objective)}.`;
  if (stat) t += ` Static rules: ${num1(stat.objective)}.`;
  return `${t} A table follows.`;
}

function ObjectiveBars({ rows, title }: { rows: RoundRow[]; title: string }) {
  const sorted = sortByObjective(rows);
  const data = sorted.map((r) => ({ ...r, label: defenderLabel(r.defender) }));
  const h = data.length * 24 + 52;
  return (
    <figure className="fig" aria-label={`${title}: score per defender`}>
      <ul className="legend" aria-hidden="true">
        <li><span className="swatch" style={{ background: "var(--accent)" }} />Tested defender</li>
        <li><span className="swatch swatch-base" style={{ background: "var(--bar-base)" }} />Baselines</li>
        <li><span className="swatch" style={{ background: "var(--bar-other)" }} />Other defenders</li>
      </ul>
      <div className="rbars" style={{ height: h }} role="img" aria-label={barSummary(sorted)}>
        <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 320, height: h }}>
          <BarChart data={data} layout="vertical" margin={{ top: 2, right: 34, bottom: 22, left: 0 }} barCategoryGap={4} accessibilityLayer={false}>
            <CartesianGrid horizontal={false} stroke="var(--border)" />
            <XAxis
              type="number"
              domain={[0, (max: number) => Math.max(20, Math.ceil(max / 20) * 20)]}
              allowDecimals={false}
              tickCount={4}
              tick={TICK}
              tickLine={false}
              stroke="var(--border-strong)"
              label={{ value: "Score (lower is better)", position: "insideBottom", offset: -12, fontSize: 12, fill: "var(--muted)" }}
            />
            <YAxis type="category" dataKey="label" width={140} tick={TICK} tickLine={false} interval={0} stroke="var(--border-strong)" />
            <Bar dataKey="objective" radius={[0, 3, 3, 0]} isAnimationActive={false}>
              {data.map((d) => (
                <Cell
                  key={d.defender}
                  fill={barColor(d.role)}
                  stroke={d.role === "primary" ? "var(--text)" : d.role === "baseline" ? "var(--border-strong)" : "none"}
                  strokeWidth={d.role === "primary" ? 1.5 : 1}
                />
              ))}
              <LabelList dataKey="objective" position="right" formatter={(v: unknown) => num1(Number(v))} style={{ fontSize: 12, fill: "var(--text)" }} />
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

function Th({ children, tip, tipLabel }: { children: string; tip: string; tipLabel: string }) {
  return (
    <th scope="col" className="num">
      <span className="th-in">
        {children}
        <InfoTip label={tipLabel}>{tip}</InfoTip>
      </span>
    </th>
  );
}

function RowsTable({ rows }: { rows: RoundRow[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th scope="col">Defender</th>
            <Th tip={TIPS.score} tipLabel="the score">Score</Th>
            <Th tip={TIPS.leakage} tipLabel="fraud let through">Fraud let through</Th>
            <Th tip={TIPS.honestHeld} tipLabel="honest held">Honest held</Th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.defender} className={r.role === "primary" ? "row-primary" : undefined}>
              <th scope="row">
                <span className="def">
                  {defenderLabel(r.defender)}
                  {r.role === "primary" && <Pill variant="headline">Tested</Pill>}
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

function Hyp({ h }: { h: Hypothesis }) {
  const tone = hypTone(h.verdict);
  const numbers = hypNumbers(h);
  return (
    <li className="hyp">
      <Pill variant={tone} className="pill-lg">
        {tone === "release" ? <CheckIcon size={14} /> : tone === "hold" ? <CrossIcon size={14} /> : <WarnIcon size={14} />}
        {h.id} {h.verdict}
      </Pill>
      {numbers && <p className="hyp-num">{numbers}</p>}
      <p className="hyp-text">{hypSentence(h)}</p>
    </li>
  );
}

/** First caveat, plus any caveat that says the defender still loses somewhere; the rest sit behind a count. */
function Caveats({ r }: { r: RoundInfo }) {
  const all = r.caveats ?? [];
  if (all.length === 0) return null;
  const lead = all.filter((c, i) => i === 0 || /still loses/i.test(c));
  const rest = all.filter((c) => !lead.includes(c));
  const hasK1 = all.some((c) => /\bK1\b|\bfolds?\b/i.test(c));
  return (
    <div className="round-cav">
      <div className="cav-head">
        <h3>What to keep in mind</h3>
        {hasK1 && <InfoTip label="the caveats">{TIPS.caveats}</InfoTip>}
      </div>
      <ul className="test-caveats" aria-label={`${r.title}: caveats`}>
        {lead.map((c) => (
          <li key={c} className="test-caveat">{c}</li>
        ))}
      </ul>
      {rest.length > 0 && (
        <details className="more-cav">
          <summary>More caveats ({rest.length})</summary>
          <ul className="test-caveats" aria-label={`${r.title}: more caveats`}>
            {rest.map((c) => (
              <li key={c} className="test-caveat">{c}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function RoundCard({ r }: { r: RoundInfo }) {
  const tone = r.tone;
  const plain = plainLine(r);
  const rule = ruleSentence(r);
  const hyps = r.hypotheses ?? [];
  return (
    <section id={`round-${r.round}`} className="card round" aria-labelledby={`round-${r.round}-h`}>
      <h2 id={`round-${r.round}-h`}>{r.title}</h2>
      <div className="badge-row">
        <p className={`verdict-badge tone-${tone}`}>
          {TONE_ICON[tone]}
          <span className="visually-hidden">{TONE_NAME[tone]}: </span>
          <span>{r.verdict}</span>
        </p>
        <InfoTip label="GO and NO-GO">{hyps.length > 0 ? TIPS.h1h2 : TIPS.verdictWords}</InfoTip>
      </div>
      {hyps.length > 0 ? (
        <ul className="hyps" aria-label="Hypotheses fixed before the run">
          {hyps.map((h) => (
            <Hyp key={h.id} h={h} />
          ))}
        </ul>
      ) : (
        plain && <p className="round-plain">{plain}</p>
      )}
      <p className="round-meta">
        {r.family ? `Never-seen family${SEP}${r.family}` : r.split ? splitName(r.split) : null}
        {r.decision_ref && <InfoTip label="the decision record">Decision record {r.decision_ref}.</InfoTip>}
      </p>
      {r.rule && !hyps.length && (
        <div className="round-rule">
          <p>
            <strong>Pass rule, written before the run</strong>
            <InfoTip label="the pass rule">{TIPS.preRegistered}</InfoTip>
          </p>
          {rule && <p>{rule}</p>}
          <details className="rule-raw">
            <summary>Rule fixed before the run</summary>
            <code>{r.rule}</code>
          </details>
        </div>
      )}
      {r.rows && r.rows.length > 0 && <ObjectiveBars rows={r.rows} title={r.title} />}
      <Caveats r={r} />
    </section>
  );
}

/* ---------- Round 3: more numbers ---------- */

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
        <dt>Average</dt><dd>{signed1(p.mean)}</dd>
        <dt>95% range</dt><dd>{signed1(p.lo)} to {signed1(p.hi)}</dd>
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
  const summary = `Range chart of the score difference of ${pts.length} versions against the full dreamer. ${pts
    .map((p) => `${p.label} ${signed1(p.mean)}, range ${signed1(p.lo)} to ${signed1(p.hi)}`)
    .join("; ")}. A table follows.`;
  return (
    <figure className="fig forest">
      <div className="forest-plot" style={{ height: h }} role="img" aria-label={summary}>
        <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 480, height: h }}>
          <ScatterChart margin={{ top: 8, right: 20, bottom: 22, left: 4 }} accessibilityLayer={false}>
            <CartesianGrid horizontal={false} stroke="var(--border)" />
            <XAxis
              type="number"
              dataKey="mean"
              domain={[min, max]}
              ticks={ticks}
              tick={TICK}
              tickLine={false}
              stroke="var(--border-strong)"
              label={{ value: "Score points: version minus full dreamer", position: "insideBottom", offset: -12, fontSize: 12, fill: "var(--muted)" }}
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
        <span>← scored better than the full dreamer</span>
        <span>scored worse →</span>
      </div>
      <figcaption className="fig-cap">
        <SimPill /> {SIM_NOTE}{SEP}dot = average, bar = 95% range
        <InfoTip label="the 95% range">{TIPS.range95}</InfoTip>
      </figcaption>
      <details className="data-details">
        <summary>Show as a table</summary>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Version</th>
                <th scope="col" className="num">Average</th>
                <th scope="col" className="num">95% range</th>
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
            <th scope="col">Defender</th>
            <th scope="col" className="num">Attacks before containment</th>
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
  const spent = Object.entries(r.spent_splits ?? {}).map(([k, v]) => ({
    split: k,
    objective: typeof v === "number" ? v : v.objective,
    why: typeof v === "number" ? undefined : v.why_not_clean,
  }));
  // "Every cut-down version and the round 2 swarm scored better" only when the paired data says so.
  const arms = (r.paired ?? []).filter((p) => p.arm.startsWith("dreamer_") || p.arm.startsWith("swarm_"));
  const allBetter = arms.length > 0 && arms.every((p) => p.mean < 0);
  const forestSub = allBetter
    ? "Every cut-down version and the round 2 swarm scored better than the full dreamer."
    : "Left of the zero line: the version scored better than the full dreamer.";
  return (
    <div className="dreamer">
      <p className="page-sub">The full dreamer against three cut-down versions, the round 2 swarm and two baselines.</p>
      <div className="dreamer-grid">
        {r.paired && r.paired.length > 0 && (
          <section className="card" aria-labelledby="forest-h">
            <CardHeader
              id="forest-h"
              level={3}
              title="Each version against the full dreamer"
              tipLabel="the range chart"
              tip={`Per episode, version minus full dreamer on the score. ${TIPS.range95} Negative means the version scored better than the full dreamer.`}
              subtitle={forestSub}
            />
            <ForestPlot paired={r.paired} />
          </section>
        )}

        {sorted.length > 0 && (
          <section className="card" aria-labelledby="arms-h">
            <CardHeader
              id="arms-h"
              level={3}
              title={`All ${sorted.length} defenders`}
              tipLabel="combined numbers"
              tip={TIPS.pooled}
              subtitle={`Combined over all episodes${SEP}lower is better`}
            />
            <RowsTable rows={sorted} />
            <p className="fig-cap"><SimPill /> {SIM_NOTE}</p>
          </section>
        )}

        {r.report_only && (r.report_only.immunity_attempts || r.report_only.never_contained_rate) && (
          <section className="card" aria-labelledby="ro-h">
            <CardHeader
              id="ro-h"
              level={3}
              title="For context"
              tipLabel="the context table"
              tip="Attacks before containment: attack payouts that succeed before a campaign is contained, on average. Never contained: share of campaigns that were never contained."
              subtitle="Not part of the pass rule"
            />
            <ReportOnly rows={sorted} ro={r.report_only} />
            <p className="fig-cap"><SimPill /> {SIM_NOTE}</p>
          </section>
        )}

        {spent.length > 0 && (
          <section className="card" aria-labelledby="spent-h">
            <CardHeader
              id="spent-h"
              level={3}
              title="Earlier test sets (already used)"
              tipLabel="earlier test sets"
              tip={TIPS.spent}
              subtitle="Used for training or in earlier rounds, so these are for reference, not clean tests."
            />
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Test set</th>
                    <th scope="col" className="num">Score</th>
                    {spent.some((s) => s.why) && <th scope="col">Why not clean</th>}
                  </tr>
                </thead>
                <tbody>
                  {spent.map((s) => (
                    <tr key={s.split}>
                      <th scope="row">{splitName(s.split)}</th>
                      <td className="num">{num2(s.objective)}</td>
                      {spent.some((x) => x.why) && <td>{s.why ? `${s.why.charAt(0).toUpperCase()}${s.why.slice(1)}` : "-"}</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="fig-cap"><SimPill /> {SIM_NOTE}</p>
          </section>
        )}
      </div>
    </div>
  );
}

/* ---------- View ---------- */

/** "Round 3 · H1 GO · H2 FAIL": the verdict text from the API, one separator style. */
const stripText = (r: RoundInfo) => `Round ${r.round}${SEP}${r.verdict.replace(/,\s*/g, SEP)}`;

function RoundStrip({ rounds }: { rounds: RoundInfo[] }) {
  return (
    <nav className="rstrip" aria-label="Round verdicts">
      {rounds.map((r) => (
        <a key={r.round} href={`#round-${r.round}`} className={`verdict-badge tone-${r.tone}`}>
          {TONE_ICON[r.tone]}
          <span className="visually-hidden">{TONE_NAME[r.tone]}: </span>
          <span>{stripText(r)}</span>
        </a>
      ))}
    </nav>
  );
}

export default function ResultsView() {
  const state = useAsync(fetchRounds);
  const data = state.data;
  const round3 = data?.rounds.find((r) => r.round === 3);
  const summary =
    data && data.rounds.length > 0
      ? (() => {
          const s = data.rounds.map((r) => `round ${r.round} ${outcomeWords(r)}`).join(", ");
          return `${s.charAt(0).toUpperCase()}${s.slice(1)}.`;
        })()
      : null;

  return (
    <>
      <div className="page-head">
        <h1>Three rounds, each judged by a rule written first</h1>
        <p className="page-sub">
          Each round tested defenders on an attack family they had never seen.
          {summary && <> {summary}</>}
          {data && (
            <>
              {" "}Lower score is better.
              <InfoTip label="the score">
                {TIPS.score} Exact formula: {data.objective}
              </InfoTip>
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
          <RoundStrip rounds={data.rounds} />
          <div className="round-grid">
            {data.rounds.map((r) => (
              <RoundCard key={r.round} r={r} />
            ))}
          </div>
          {round3 && (
            <details className="more-r3">
              <summary>More round-3 numbers</summary>
              <DreamerPanel r={round3} />
            </details>
          )}
        </>
      )}
    </>
  );
}
