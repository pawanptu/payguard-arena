import type { Split } from "./types";

/** Separator for metadata lines. */
export const SEP = " · ";
export const pct = (n: number) => `${n.toFixed(1)}%`;
export const num1 = (n: number) => n.toFixed(1);
export const num2 = (n: number) => n.toFixed(2);
/** Python-style "{n:+.1f}": always signed, with a real minus sign. */
export const signed1 = (n: number) => `${n < 0 ? "−" : "+"}${Math.abs(n).toFixed(1)}`;
export const usd = (n: number) =>
  `$${n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
/** Hours -> days with one decimal: 295.2 -> "12.3". */
export const days1 = (h: number) => (h / 24).toFixed(1);
/** Hours -> whole days when round, else one decimal: 720 -> "30". */
export const daysTotal = (h: number) => {
  const d = h / 24;
  return Number.isInteger(d) ? String(d) : d.toFixed(1);
};
/** 1 -> "1st", 10 -> "10th", 12 -> "12th", 22 -> "22nd". */
export const ordinal = (n: number) => {
  const v = n % 100;
  const suffix = v >= 11 && v <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th";
  return `${n}${suffix}`;
};
export const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
/** "PB-1234567890ABCDEF" -> "PB-1234…CDEF"; short ids are left alone. */
export const shortId = (id: string) => (id.length > 14 ? `${id.slice(0, 7)}…${id.slice(-4)}` : id);
/** "2026-11-07T12:00:01.123456Z" -> "2026-11-07 12:00:01 UTC"; anything else is shown as given. */
export const shortTime = (iso: string) => {
  const m = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})/.exec(iso);
  return m && /(Z|[+-]00:?00)$/.test(iso) ? `${m[1]} ${m[2]} UTC` : iso;
};

const DEFENDER_LABELS: Record<string, string> = {
  always_release: "Always release",
  always_hold: "Always hold",
  static_rules: "Static rules",
  llm_judge_sarvam: "LLM judge (Sarvam)",
  llm_judge_gemini: "LLM judge (Gemini)",
  swarm_frozen: "Swarm, fixed weights",
  swarm_l1: "Swarm, learned weights",
  swarm_evolved: "Swarm, evolved",
  swarm_memory: "Swarm + memory",
  swarm_evolved_memory: "Swarm, evolved + memory",
  swarm_r2: "Swarm, robust (round 2)",
  dreamer_full: "Dreamer, full",
  dreamer_nowm: "Dreamer, no world model",
  dreamer_noskeptic: "Dreamer, no skeptic",
  dreamer_nocalib: "Dreamer, no calibration",
};

/** Human label for a raw defender id; unknown ids are shown as given. */
export function defenderLabel(id: string): string {
  const known = DEFENDER_LABELS[id];
  if (known) return known;
  // "Single", not "Best single": a split can show the rule's TRAIN pick and its own best single (D-038).
  if (id.startsWith("single_") && id.length > "single_".length) return `Single: ${id.slice("single_".length)}`;
  return id;
}

export const SPLIT_ORDER: Split[] = ["TEST_HELDOUT2", "TEST_HELDOUT", "TEST_SEEN", "TRAIN"];
export const SPLIT_LABEL: Record<Split, string> = {
  TEST_HELDOUT: "Round 1 family",
  TEST_SEEN: "Known families",
  TRAIN: "Training data",
  TEST_HELDOUT2: "Round 2 family",
};
