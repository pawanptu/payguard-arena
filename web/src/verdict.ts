// Pure module: the pre-registered D-022 test, computed from replay result rows.
// Mirrors scripts/final_eval.py::check (on the held-out split):
//   mStatic = static.objective - primary.objective
//   mSingle = single.objective - primary.objective
//   met     = mStatic >= 3.0 && mSingle >= 3.0 && primary.honest_block_rate <= static.honest_block_rate
// "single" is the best single specialist, chosen on TRAIN (lowest TRAIN objective among single_* defenders).
import type { ResultRow, Split } from "./types";
import { num1, signed1 } from "./format";

export const MARGIN = 3.0;
export const PRIMARY_DEFENDER = "swarm_evolved_memory";
export const STATIC_DEFENDER = "static_rules";
export const SINGLE_PREFIX = "single_";
export const DECISIVE_SPLIT: Split = "TEST_HELDOUT";

/** The hero's result cards (U5), in display order. Adding a round-2 split is one new line here. */
export interface TestSpec {
  split: Split;
  title: string;
  /** Short noun phrase used in the headline sentence. */
  short: string;
  /** Pill prefix, e.g. "Pre-registered test". */
  pillLabel: string;
  /** The pre-registered go/no-go test; the others are reported alongside it. */
  decisive: boolean;
  /** Defender under test; defaults to PRIMARY_DEFENDER. */
  primary?: string;
}
export const TESTS: TestSpec[] = [
  { split: "TEST_SEEN", title: "Known attack families, new seeds", short: "known attacks", pillLabel: "Secondary check", decisive: false },
  { split: "TEST_HELDOUT", title: "Never-seen attack family (round 1)", short: "a never-seen attack family", pillLabel: "Pre-registered test", decisive: true },
  {
    split: "TEST_HELDOUT2",
    title: "Second sealed attack family (round 2)",
    short: "a second never-seen family",
    pillLabel: "Pre-registered test",
    decisive: true,
    primary: "swarm_r2",
  },
];

/** "a" / "a and b" / "a, b and c". */
const joinList = (items: string[]) =>
  items.length <= 2 ? items.join(" and ") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;

export interface VerdictResult {
  primary: ResultRow;
  static: ResultRow;
  single: ResultRow;
  /** static.objective - primary.objective (positive = primary is better). */
  mStatic: number;
  /** single.objective - primary.objective (positive = primary is better). */
  mSingle: number;
  /** min(mStatic, mSingle): the margin over the strongest baseline. */
  m: number;
  /** The baseline with the lower objective (static wins ties). */
  baseline: ResultRow;
  /** Report-only (D-038): the single_* row with the lowest objective ON this split. Not part of the rule. */
  splitBest: ResultRow;
  passStatic: boolean;
  passSingle: boolean;
  passHonest: boolean;
  met: boolean;
}

/** The single_* defender picked on TRAIN: lowest TRAIN objective; ties and missing TRAIN rows fall back to the first. */
export function pickBestSingle(rows: ResultRow[], candidates: ResultRow[]): ResultRow | undefined {
  let best: ResultRow | undefined;
  let bestTrain = Infinity;
  for (const c of candidates) {
    const train = rows.find((r) => r.split === "TRAIN" && r.defender === c.defender);
    const score = train ? train.objective : Infinity;
    if (best === undefined || score < bestTrain) {
      best = c;
      bestTrain = score;
    }
  }
  return best;
}

/** Returns null when the held-out rows (or any of the three needed rows) are missing. Never throws. */
export function computeVerdict(rows: ResultRow[], split: Split = DECISIVE_SPLIT, primaryName: string = PRIMARY_DEFENDER): VerdictResult | null {
  const held = rows.filter((r) => r.split === split);
  const primary = held.find((r) => r.defender === primaryName);
  const stat = held.find((r) => r.defender === STATIC_DEFENDER);
  const singles = held.filter((r) => r.defender.startsWith(SINGLE_PREFIX));
  const single = pickBestSingle(rows, singles);
  if (!primary || !stat || !single) return null;
  const splitBest = singles.reduce((a, b) => (b.objective < a.objective ? b : a));

  const nums = [primary.objective, stat.objective, single.objective, primary.honest_block_rate, stat.honest_block_rate];
  if (!nums.every(Number.isFinite)) return null;

  const mStatic = stat.objective - primary.objective;
  const mSingle = single.objective - primary.objective;
  const passStatic = mStatic >= MARGIN;
  const passSingle = mSingle >= MARGIN;
  const passHonest = primary.honest_block_rate <= stat.honest_block_rate;
  return {
    primary,
    static: stat,
    single,
    mStatic,
    mSingle,
    m: Math.min(mStatic, mSingle),
    baseline: stat.objective <= single.objective ? stat : single,
    splitBest,
    passStatic,
    passSingle,
    passHonest,
    met: passStatic && passSingle && passHonest,
  };
}

/**
 * One factual sentence for the result card when the pre-registered comparison hides something a reader should know
 * about the singles on THIS split (D-038), or null when there is nothing to add. Never changes `met`.
 * Example data, round 2: primary swarm_r2 26.30; rule single (TRAIN pick) single_policy 100.09;
 * v.splitBest = single_graph 28.87, so the primary leads it by only 2.57 (< MARGIN).
 * Round 1: v.splitBest === v.single on every split, so the card must stay unchanged there.
 */
export function caveatFor(v: VerdictResult): string | null {
  const best = v.splitBest;
  // Nothing is hidden when the TRAIN pick is also the strongest single here (ties included).
  if (best.defender === v.single.defender || best.objective >= v.single.objective) return null;
  const lead = best.objective - v.primary.objective; // positive = primary is better
  const under = lead < MARGIN ? `, under +${MARGIN.toFixed(1)}` : "";
  const name = best.defender.slice(SINGLE_PREFIX.length);
  return `Best single on this split: ${name} ${num1(best.objective)} (lead ${signed1(lead)}${under}); the rule's single was fixed on TRAIN.`;
}

export interface TestResult {
  test: TestSpec;
  v: VerdictResult;
}

/** One entry per configured test whose rows exist (missing splits are simply skipped). */
export function evaluateTests(rows: ResultRow[]): TestResult[] {
  const out: TestResult[] = [];
  for (const test of TESTS) {
    const v = computeVerdict(rows, test.split, test.primary);
    if (v) out.push({ test, v });
  }
  return out;
}

/** Factual one-sentence headline built from the met and not-met tests (works for any number of cards). */
export function headlineFor(results: TestResult[]): string {
  const met = results.filter((r) => r.v.met).map((r) => r.test.short);
  const unmet = results.filter((r) => !r.v.met).map((r) => r.test.short);
  if (met.length === 0) return "Did not beat the baselines on this benchmark.";
  if (unmet.length === 0) return `Beat the baselines on ${joinList(met)}.`;
  return `Beat the baselines on ${joinList(met)}; did not on ${joinList(unmet)}.`;
}
