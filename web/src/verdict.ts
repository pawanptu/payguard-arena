// Pure module: the pre-registered D-022 test, computed from replay result rows.
// Mirrors scripts/final_eval.py::check (on the held-out split):
//   mStatic = static.objective - primary.objective
//   mSingle = single.objective - primary.objective
//   met     = mStatic >= 3.0 && mSingle >= 3.0 && primary.honest_block_rate <= static.honest_block_rate
// "single" is the best single specialist, chosen on TRAIN (lowest TRAIN objective among single_* defenders).
import type { ResultRow, Split } from "./types";

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
  { split: "TEST_HELDOUT", title: "Never-seen attack family", short: "a never-seen attack family", pillLabel: "Pre-registered test", decisive: true },
];

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
  const single = pickBestSingle(
    rows,
    held.filter((r) => r.defender.startsWith(SINGLE_PREFIX)),
  );
  if (!primary || !stat || !single) return null;

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
    passStatic,
    passSingle,
    passHonest,
    met: passStatic && passSingle && passHonest,
  };
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
  if (unmet.length === 0) return `Beat the baselines on ${met.join(" and ")}.`;
  return `Beat the baselines on ${met.join(" and ")}; did not on ${unmet.join(" and ")}.`;
}
