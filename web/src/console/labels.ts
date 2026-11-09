import type { ConsoleCase, ConsoleQueue, DecidedBy, SpecialistInfo } from "../types";
import { SPECIALIST_KEYS } from "../types";

/** Human names for the anomaly layer's nine statistics (contract A). Unknown names are made readable. */
const STAT_LABELS: Record<string, string> = {
  window_usd: "24 h payout volume",
  window_n: "24 h payout count",
  amount: "payout amount",
  amount_ratio: "amount vs. this account's usual",
  amount_over_cap: "amount vs. limit for account age",
  window_over_cap: "24 h volume vs. limit for account age",
  rate_z: "payout rate vs. this account's usual",
  graph_cluster: "tight cluster of linked accounts",
  sharing: "shared device or payout destination",
};

export const statLabel = (name: string) => STAT_LABELS[name] ?? name.replace(/_/g, " ");

export const DECIDED_BY: Record<DecidedBy, string> = {
  swarm: "Specialist swarm",
  layer: "Anomaly layer",
  both: "Swarm and anomaly layer",
};

const FALLBACK_SPECIALISTS: SpecialistInfo[] = [
  { key: "velocity", label: "Velocity", what: "Payout count and USD volume in the account's last 24 h" },
  { key: "graph", label: "Graph", what: "Collusion structure: accounts tied to the same requester, device or destination" },
  { key: "drift", label: "Drift", what: "Payout rate against the account's own normal rate" },
  { key: "policy", label: "Policy", what: "Amount against the limit for the account's age" },
  { key: "layer", label: "Anomaly layer", what: "Nine statistics with cut-offs set on honest traffic only" },
];

export function specialistInfo(q: ConsoleQueue): Map<string, SpecialistInfo> {
  const m = new Map<string, SpecialistInfo>();
  for (const s of FALLBACK_SPECIALISTS) m.set(s.key, s);
  for (const s of q.specialists ?? []) m.set(s.key, s);
  return m;
}

/** The specialist with the biggest share of the vote; the anomaly layer when nobody spoke up. */
export function topSpecialistKey(c: ConsoleCase): string {
  let best = "layer";
  let share = 0;
  for (const k of SPECIALIST_KEYS) {
    const v = c.vote_share?.[k] ?? 0;
    if (v > share) {
      share = v;
      best = k;
    }
  }
  return best;
}

export const clamp01 = (n: number) => Math.min(1, Math.max(0, Number.isFinite(n) ? n : 0));
export const num = (n: number, d = 2) => (Number.isFinite(n) ? n.toFixed(d) : "-");
export const pct0 = (share: number) => `${Math.round(clamp01(share) * 100)}%`;
