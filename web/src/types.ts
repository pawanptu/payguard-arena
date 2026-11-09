// Mirrors the replay contract (schema_version 1). See payguard/api/replay.py.
export type Split = "TRAIN" | "TEST_SEEN" | "TEST_HELDOUT" | "TEST_HELDOUT2";
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

/* ---------- Ops console contracts A-D (docs/CONSOLE_CONTRACT.md) ---------- */

export type SpecialistKey = "velocity" | "graph" | "drift" | "policy";
export const SPECIALIST_KEYS: SpecialistKey[] = ["velocity", "graph", "drift", "policy"];
export type DecidedBy = "swarm" | "layer" | "both";

// A. Review queue: GET /api/console/queue
export interface SpecialistInfo {
  key: string;
  label: string;
  what: string;
}

export interface CaseFeatures {
  age_days: number;
  n_prior: number;
  window_n: number;
  window_usd: number;
  amount_ratio: number;
  device_workers: number;
  destination_workers: number;
  requester_workers: number;
  worker_requesters: number;
}

export interface CaseOutcome {
  paid: boolean;
  paid_at: number | null;
  delay_h: number | null;
}

export interface CaseTruth {
  is_attack: boolean;
  family: string;
}

export interface ConsoleCase {
  event_id: number;
  /** Position in the reviewer's queue (1 = look first). Falls back to array order when absent. */
  rank?: number;
  t: number;
  worker_id: number;
  requester_id: number;
  amount: number;
  verdict: VerdictName;
  risk: number;
  decided_by: DecidedBy;
  scores: Record<string, number>;
  weights: Record<string, number>;
  vote_share: Record<string, number>;
  layer: { verdict: VerdictName; fired: string[] };
  features: CaseFeatures;
  outcome?: CaseOutcome | null;
  truth?: CaseTruth | null;
}

export interface QueueCounts {
  events: number;
  RELEASE: number;
  STEP_UP: number;
  HOLD: number;
  cases: number;
  cases_attack: number;
  cases_honest: number;
  attack_usd_total: number;
  attack_usd_stopped: number;
  attack_usd_leaked: number;
}

export interface ConsoleQueue {
  schema_version: number;
  simulated: boolean;
  generated_at: string;
  source: { split: string; seed: number; defender: string; replay: string; parity_checked_events: number };
  thresholds: { step_up: number; hold: number };
  counts: QueueCounts;
  specialists: SpecialistInfo[];
  cases: ConsoleCase[];
}

// B. AI explanations: GET /api/console/explanations (404 {"error": "not_built"} until built)
export interface CaseExplanation {
  source: "llm" | "template";
  summary: string;
  evidence: string[];
  check_next: string;
  suggested_action: VerdictName;
}

export interface ConsoleExplanations {
  schema_version: number;
  provider: string;
  model: string;
  prompt_version: string;
  generated_at: string;
  grounding: string;
  counts: { llm: number; template: number };
  explanations: Record<string, CaseExplanation>;
}

/** The explanations file may not exist yet; that is a state, not a failure. */
export type ExplanationsResult = { built: true; data: ConsoleExplanations } | { built: false };

// C. Copilot sessions: GET /api/copilot/sessions
export type ToolSource = "paypal-agent-toolkit" | "payguard-rest" | "payguard-console";

export type CopilotTurn =
  | { role: "user"; text: string }
  | { role: "tool_call"; step: number; tool: string; source: ToolSource | string; args: Record<string, unknown> }
  | { role: "tool_result"; step: number; tool: string; ok: boolean; ms: number; data?: unknown; error?: string }
  | { role: "assistant"; text: string };

export interface CopilotSession {
  schema_version: number;
  id: string;
  order: number;
  title: string;
  recorded_at: string;
  environment: string;
  llm: { provider: string; model: string };
  toolkit: { package: string; version: string; read_only: boolean; enabled_tools: string[] };
  redaction: string;
  turns: CopilotTurn[];
}

export interface CopilotSessions {
  sessions: CopilotSession[];
}

// D. Three rounds: GET /api/results/rounds
export type RoundTone = "fail" | "pass" | "mixed";
export type RowRole = "primary" | "baseline" | "ablation" | "reference" | "llm";

export interface RoundRow {
  defender: string;
  role: RowRole | string;
  objective: number;
  leakage_pct: number;
  honest_block_rate: number;
}

export interface Hypothesis {
  id: string;
  verdict: string;
  text: string;
}

export interface PairedMargin {
  arm: string;
  mean: number;
  ci95: [number, number];
}

export interface RoundInfo {
  round: number;
  title: string;
  split?: string;
  family?: string;
  verdict: string;
  tone: RoundTone;
  decision_ref?: string;
  rule?: string;
  primary?: string;
  rows?: RoundRow[];
  caveats?: string[];
  source?: string;
  hypotheses?: Hypothesis[];
  paired?: PairedMargin[];
  report_only?: {
    immunity_attempts?: Record<string, number>;
    never_contained_rate?: Record<string, number>;
  };
  spent_splits?: Record<string, number | { objective: number }>;
}

export interface RoundsResponse {
  schema_version: number;
  simulated: boolean;
  objective: string;
  rounds: RoundInfo[];
}
