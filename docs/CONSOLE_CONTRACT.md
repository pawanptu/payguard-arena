# Ops console contract (D-049)

Director-owned. The three build tracks (console API, copilot, console UI) code against this file. Change it only
through the director. All data is a SIMULATED replay on synthetic data. The public site never sends money.

## Product in one line

An ops console for PayPal payout fraud. A self-evolving defender swarm holds risky payouts. A reviewer works the
queue with per-specialist evidence and an AI explanation. An AI ops copilot built on the official PayPal Agent
Toolkit answers questions against the PayPal sandbox. On the public site everything is replayed: the queue comes
from the published round-2 replay episode, and the copilot sessions were recorded live on the sandbox.

## Site map (hash routes, no router dependency)

| Route | View | Content |
|---|---|---|
| `#/console` (default) | Console | Review queue, case detail, AI explanation, evidence, replay-mode decisions |
| `#/copilot` | Copilot | Recorded Agent Toolkit sessions, tool calls with source badges |
| `#/results` | Results | Three pre-registered rounds (1 NO-GO, 2 GO with caveats, 3 H1 GO / H2 FAIL) and the dreamer panel |
| `#/how` | How it works | The existing dashboard: hero, decision feed, evolution chart, results table, receipts, webhooks |

A permanent banner on every view says: "Replay mode: simulated synthetic data. This public site never sends
payouts."

## A. Review queue: `eval/console/queue.json`, `GET /api/console/queue`

Built by `python -m scripts.build_console` from the frozen swarm_r2 (eval/final_r2/defender_freeze.json must
match) on the published replay episode (TEST_HELDOUT2, seed 4000, the same episode as eval/replay_live.json).
The builder ASSERTS that every one of the episode's verdicts equals the verdict in eval/replay_live.json and
refuses to write on any mismatch. Nothing is tuned; the sealed set stays spent. Cases are every event whose
verdict is HOLD or STEP_UP.

```json
{
  "schema_version": 1,
  "simulated": true,
  "generated_at": "2026-11-09T10:00:00+00:00",
  "source": {"split": "TEST_HELDOUT2", "seed": 4000, "defender": "swarm_r2",
             "replay": "eval/replay_live.json", "parity_checked_events": 4323},
  "thresholds": {"step_up": 0.3, "hold": 0.6},
  "counts": {"events": 4323, "RELEASE": 3971, "STEP_UP": 137, "HOLD": 215,
             "cases": 352, "cases_attack": 97, "cases_honest": 255,
             "attack_usd_total": 0.0, "attack_usd_stopped": 0.0, "attack_usd_leaked": 0.0},
  "specialists": [
    {"key": "velocity", "label": "Velocity", "what": "Payout count and USD volume in the account's last 24 h"},
    {"key": "graph", "label": "Graph", "what": "Collusion structure: accounts tied to the same requester, device or destination"},
    {"key": "drift", "label": "Drift", "what": "Payout rate against the account's own normal rate"},
    {"key": "policy", "label": "Policy", "what": "Amount against the limit for the account's age"},
    {"key": "layer", "label": "Anomaly layer", "what": "Nine statistics with cut-offs set on honest traffic only"}
  ],
  "cases": [
    {
      "event_id": 24, "t": 1.69, "worker_id": 160, "requester_id": 0, "amount": 99.47,
      "verdict": "HOLD", "risk": 0.71, "decided_by": "swarm | layer | both",
      "scores":  {"velocity": 0.0, "graph": 1.0, "drift": 0.0, "policy": 0.0},
      "weights": {"velocity": 0.25, "graph": 0.25, "drift": 0.25, "policy": 0.25},
      "vote_share": {"velocity": 0.0, "graph": 1.0, "drift": 0.0, "policy": 0.0},
      "layer": {"verdict": "HOLD", "fired": ["graph_cluster"]},
      "features": {"age_days": 3.2, "n_prior": 4, "window_n": 2, "window_usd": 140.1, "amount_ratio": 1.4,
                   "device_workers": 1, "destination_workers": 1, "requester_workers": 6, "worker_requesters": 1},
      "outcome": {"paid": true, "paid_at": 25.69, "delay_h": 24.0},
      "truth": {"is_attack": false, "family": "honest"}
    }
  ]
}
```

- `risk` is the swarm's aggregated risk at decision time; `decided_by` says which part set the final verdict
  (the layer can only raise severity).
- `vote_share[i] = w_i * s_i / sum_j w_j * s_j` (0 when nobody speaks up): each specialist's share of the vote.
- `truth` is ground truth from the simulator. The UI hides it until the reviewer decides (or clicks "reveal").
- Floats rounded to 3 decimals. `counts.attack_usd_*` cover the whole episode, not only the cases.

## B. AI explanations: `eval/console/explanations.json`, `GET /api/console/explanations`

Built offline by `python -m scripts.explain_holds` (director runs it; it needs an LLM key). The deployed site only
reads the file. The LLM sees ONLY decision-time evidence (scores, weights, vote share, layer statistics, features,
amount, verdict, thresholds), never `truth` or `outcome`.

```json
{
  "schema_version": 1,
  "provider": "gemini", "model": "gemini-3.5-flash-lite", "prompt_version": "explain-v1",
  "generated_at": "...",
  "grounding": "Every number in summary and evidence must appear in the case's decision-time evidence (after rounding); otherwise the template text is used and source is 'template'.",
  "counts": {"llm": 350, "template": 2},
  "explanations": {
    "24": {
      "source": "llm | template",
      "summary": "Held because ... (at most 60 words, plain language, no jargon)",
      "evidence": ["Graph: requester 0 is tied to 6 accounts", "..."],
      "check_next": "One sentence: what a human reviewer should verify before releasing.",
      "suggested_action": "HOLD | STEP_UP | RELEASE"
    }
  }
}
```

## C. Copilot sessions: `eval/copilot/sessions/*.json`, `GET /api/copilot/sessions`

Recorded locally by `python -m scripts.copilot ask "..." --record <id>` against the PayPal sandbox. The endpoint
returns `{"sessions": [...]}` with every session file (sorted by `order`, then `id`), or an empty list.

```json
{
  "schema_version": 1,
  "id": "why-held-24", "order": 1,
  "title": "Why was payout #24 held, and is it safe to release?",
  "recorded_at": "2026-11-10T09:00:00+00:00",
  "environment": "paypal-sandbox",
  "llm": {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
  "toolkit": {"package": "paypal-agent-toolkit", "version": "1.11.0", "read_only": true,
              "enabled_tools": ["list_transactions", "list_disputes", "get_dispute", "list_invoices", "get_invoice", "get_order_details"]},
  "redaction": "Emails, names, addresses, phone numbers and payer ids are removed; ids, statuses, amounts and dates are kept.",
  "turns": [
    {"role": "user", "text": "Why was payout #24 held, and is it safe to release?"},
    {"role": "tool_call", "step": 1, "tool": "get_hold_case", "source": "payguard-console", "args": {"event_id": 24}},
    {"role": "tool_result", "step": 1, "tool": "get_hold_case", "ok": true, "ms": 3, "data": {}},
    {"role": "tool_call", "step": 2, "tool": "get_payout_batch", "source": "payguard-rest", "args": {"payout_batch_id": "U2UPGB9BXLFRY"}},
    {"role": "tool_result", "step": 2, "tool": "get_payout_batch", "ok": true, "ms": 412, "data": {}},
    {"role": "tool_call", "step": 3, "tool": "list_transactions", "source": "paypal-agent-toolkit", "args": {}},
    {"role": "tool_result", "step": 3, "tool": "list_transactions", "ok": false, "ms": 380, "error": "short error text"},
    {"role": "assistant", "text": "Final answer in plain language."}
  ]
}
```

Tool `source` values:
- `paypal-agent-toolkit`: a tool from the official PayPal Agent Toolkit (`paypal_agent_toolkit.shared`), read-only
  actions only, sandbox context.
- `payguard-rest`: our own read-only REST call. The Agent Toolkit has no Payouts tools, so payout batch status
  comes from `GET /v1/payments/payouts/{id}` through `payguard/paypal_client.py`. Say this wherever it shows.
- `payguard-console`: local console data (queue, cases, explanations).

## D. Three rounds: `GET /api/results/rounds`

Computed at request time from `eval/final/results.json`, `eval/final_r2/results.json`,
`eval/final_r2/gemini_row.json` and `eval/final_r3/results.json` (read-only). Caveat strings are fixed text,
written by the director in `payguard/console/rounds.py`, never generated.

```json
{
  "schema_version": 1, "simulated": true,
  "objective": "leakage % + 0.5 x honest-block % + honest-unpaid % (lower is better)",
  "rounds": [
    {
      "round": 1, "title": "Round 1: sleeper family", "split": "TEST_HELDOUT", "family": "sleeper",
      "verdict": "NO-GO", "tone": "fail", "decision_ref": "D-033",
      "rule": "primary objective >= 3.0 below static_rules and single_policy, honest block <= static_rules'",
      "primary": "swarm_evolved_memory",
      "rows": [{"defender": "static_rules", "role": "baseline", "objective": 28.54, "leakage_pct": 0, "honest_block_rate": 0}],
      "caveats": ["..."], "source": "eval/final/results.json"
    },
    {"round": 2, "verdict": "GO with caveats", "tone": "pass", "...": "..."},
    {"round": 3, "verdict": "H1 GO, H2 FAIL", "tone": "mixed",
     "hypotheses": [{"id": "H1", "verdict": "GO", "text": "..."}, {"id": "H2", "verdict": "FAIL", "text": "..."}],
     "paired": [{"arm": "swarm_r2", "mean": -24.871, "ci95": [-29.883, -19.672]}],
     "report_only": {"immunity_attempts": {"dreamer_full": 9.5}},
     "spent_splits": {"TRAIN": 18.69},
     "...": "..."}
  ]
}
```

`role` is one of `primary`, `baseline`, `ablation`, `reference`, `llm`. Round 3's `paired` is arm minus
dreamer_full, per episode, bootstrap 95% CI (2000 resamples, seed 0); negative means the arm leaked less.

## Rules for every track

- Never print or read `.env` values in tests or logs. Tests make no network calls (fake transports only).
- Do not change frozen files (eval/FROZEN*.json and the files they hash, eval/final*/, eval/r2/champion.json,
  payguard/swarm/*.py, payguard/robust/*.py, payguard/heldout*/), and never import `payguard.heldout3`.
- `requirements.txt` stays as it is (Render installs it). The copilot's extra dependency goes in
  `requirements-copilot.txt` only.
- Write "no direct match found" if novelty comes up, never "first" or "novel".
