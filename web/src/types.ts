// Mirrors the replay contract (schema_version 1). See payguard/api/replay.py.
export type Split = "TRAIN" | "TEST_SEEN" | "TEST_HELDOUT";
export type VerdictName = "RELEASE" | "STEP_UP" | "HOLD";

export interface ResultRow {
  defender: string;
  split: Split;
  leakage_pct: number;
  honest_block_rate: number;
  honest_unpaid_rate: number;
  honest_median_ttp_h: number;
  immunity_attempts: number;
  never_contained_rate: number;
  objective: number;
}

export interface EvolutionRow {
  generation: number;
  champion_objective: number;
  challenger_objective: number;
  promoted: boolean;
  attacker_family: string | null;
  attacker_leakage_pct: number | null;
  weights: Record<string, number>;
}

export interface EpisodeEvent {
  id: number;
  t: number;
  worker_id: number;
  requester_id: number;
  amount: number;
  verdict: VerdictName;
  paid: boolean;
  paid_at: number | null;
  is_attack: boolean;
  family: string;
  signals: Record<string, number> | null;
  weights: Record<string, number> | null;
}

export interface Payout {
  event_id: number;
  mode: "sim" | "live";
  sender_batch_id: string;
  payout_batch_id: string;
  batch_status: string;
  item_status: string;
  amount_usd: number;
  created_at: string;
}

/** A PayPal payouts webhook whose signature was verified server-side (ids and statuses only). */
export interface VerifiedWebhook {
  event_id: string;
  event_type: string;
  received_at: string;
  payout_batch_id?: string | null;
  sender_batch_id?: string | null;
  batch_status?: string | null;
  item_status?: string | null;
}

/** A row of GET /api/webhooks/recent: the replay's fields plus where it came from. */
export interface WebhookEvent extends VerifiedWebhook {
  outcome: "processed" | "pending_receipt" | null;
  source: "runtime" | "replay";
}

export interface WebhooksResponse {
  configured: boolean;
  events: WebhookEvent[];
}

export interface EpisodeMeta {
  seed: number;
  split: string;
  defender: string;
  n_events: number;
}

export interface Summary {
  schema_version: number;
  simulated: boolean;
  synthetic_fixture: boolean;
  generated_at: string;
  objective: string;
  results: ResultRow[];
  evolution: EvolutionRow[];
  episode: EpisodeMeta;
  payouts: Payout[];
  verified_webhooks?: VerifiedWebhook[];
}

export interface EventsPage {
  total: number;
  offset: number;
  limit: number;
  events: EpisodeEvent[];
}
