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
  /** Splits whose result was already known while this primary was designed: its scores there are not clean tests. */
  seenWhileBuilding?: Split[];
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
    seenWhileBuilding: ["TEST_HELDOUT"],
  },
];

export const SWARM_PREFIX = "swarm_";

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
 * Caveat (a): one factual sentence for the result card when the pre-registered comparison hides something a reader
 * should know about the singles on THIS split (D-038), or null when there is nothing to add. Never changes `met`.
 * Example data, round 2: primary swarm_r2 26.30; rule single (TRAIN pick) single_policy 100.09;
 * v.splitBest = single_graph 28.87, so the primary leads it by only 2.57 (< MARGIN).
 * Round 1: v.splitBest === v.single on every split, so the card must stay unchanged there.
 */
export function singleCaveat(v: VerdictResult): string | null {
  const best = v.splitBest;
  // Nothing is hidden when the TRAIN pick is also the strongest single here (ties included).
  if (best.defender === v.single.defender || best.objective >= v.single.objective) return null;
  const lead = best.objective - v.primary.objective; // positive = primary is better
  const under = lead < MARGIN ? `, under +${MARGIN.toFixed(1)}` : "";
  const name = best.defender.slice(SINGLE_PREFIX.length);
  return `Best single on this split: ${name} ${num1(best.objective)} (lead ${signed1(lead)}${under}); the rule's single was fixed on TRAIN.`;
}

/**
 * Caveat (b), D-039: do OTHER swarm_* defenders on this split also pass the same rule (same static row, same
 * TRAIN-picked single, same honest-block check)? If so, the card's GO is mostly about the swarm design, not this
 * primary. Return one sentence, or null when there is nothing to add. Never changes `met`.
 * Data, round 2 (TEST_HELDOUT2): swarm_r2 26.30 (primary); other swarms 28.19-30.13, all pass vs static 52.21.
 * Round 1 (TEST_HELDOUT): every swarm scores >= 55.10 vs static 28.54, so none pass.
 * TEST_SEEN is a secondary check (test.decisive === false).
 */
export function otherSwarmsCaveat(rows: ResultRow[], test: TestSpec, v: VerdictResult): string | null {
  // Only a passed, pre-registered card can overstate what its primary adds.
  if (!test.decisive || !v.met) return null;
  const others = rows.filter(
    (r) => r.split === test.split && r.defender.startsWith(SWARM_PREFIX) && r.defender !== v.primary.defender,
  );
  const passing = others.filter((r) => computeVerdict(rows, test.split, r.defender)?.met).map((r) => r.objective);
  if (passing.length === 0) return null;
  const range = passing.length === 1 ? num1(passing[0]) : `${num1(Math.min(...passing))}-${num1(Math.max(...passing))}`;
  const who = passing.length === others.length ? `All ${others.length} other swarms` : `${passing.length} of ${others.length} other swarms`;
  return `${who} also pass this rule (${range}), so the result mostly reflects the swarm design.`;
}

/**
 * Caveat (c), D-039: the same primary on the OTHER decisive splits. If it does not meet the rule there, say so,
 * and flag the result as not clean when that split was known while the primary was built (TestSpec.seenWhileBuilding).
 * Round 2: swarm_r2 on TEST_HELDOUT 45.10 vs static 28.54 (not met, known while built). Round 1: swarm_evolved_memory
 * on TEST_HELDOUT2 28.85 meets the rule, so its card gets nothing.
 */
export function otherFamilyCaveat(rows: ResultRow[], test: TestSpec, v: VerdictResult): string | null {
  // Secondary cards would only repeat the decisive card next to them.
  if (!test.decisive) return null;
  for (const other of TESTS) {
    if (!other.decisive || other.split === test.split) continue;
    const ov = computeVerdict(rows, other.split, v.primary.defender);
    if (!ov || ov.met) continue;
    const known = test.seenWhileBuilding?.includes(other.split) ? " (known while it was built, so not a clean test)" : "";
    return `The same defender on the ${other.title.toLowerCase()}: ${num1(ov.primary.objective)} vs static rules ${num1(ov.static.objective)}, not met${known}.`;
  }
  return null;
}

/** All caveat lines for one result card, in order (a), (b), (c); empty when there is nothing to add. */
export function caveatsFor(rows: ResultRow[], r: TestResult): string[] {
  return [singleCaveat(r.v), otherSwarmsCaveat(rows, r.test, r.v), otherFamilyCaveat(rows, r.test, r.v)].filter(
    (c): c is string => c !== null,
  );
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
