# PayGuard Arena

A co-evolving fraud arms race on instant gig payouts. A swarm of defender agents decides whether to **release, hold or step-up** each payout through **PayPal sandbox Payouts**, while a population of attackers mutates its tactics. The defenders self-evolve through outcome-driven reputation updates and a champion-challenger gate.

> **Status: work in progress** (built for the PayPal AI Hackathon, Nov 2026). Nothing below claims results yet. Every number shown by this project is from a **simulated replay**, never real-world fraud data, and only sampled payouts go through the real PayPal sandbox.

## Simulator and benchmark protocol

The simulator is a deterministic synthetic gig-payout platform. For a given seed it builds the same episode every time: 14 simulated days, 300 honest workers, 60 requesters and 6 small shops (`SimConfig` in `payguard/sim/engine.py`), plus 2-4 attack campaigns.

Run the offline checks and the baselines from the repo root:

```bash
python -m pytest -q
python -m scripts.run_baselines
```

### What a defender sees

- Each payout request is an `Event`: time in hours, worker, requester, device, payout destination and USD amount.
- A defender can also compute features from the platform's own prior history (account age, amounts in the last 24 h, how many workers share a device, destination or requester). See `payguard/sim/features.py`.
- Ground truth (`Truth`: is_attack, family, campaign id) is never shown to a defender. It reaches the defender only as a delayed label through `learn()`, after the outcome is known.
- Resolution randomness is keyed by (episode seed, event id), not by call order. A defender that decides differently does not change the draws seen by later events.

### Honest population and hard negatives

Honest workers come in four personas (share of workers in brackets):

- `regular` (60%): about 0.6 payouts a day.
- `new` (15%): accounts 0-10 days old, about 1 payout a day.
- `power` (8%): about 3 payouts a day, larger amounts.
- `bursty` (17%): quiet most days; on about a quarter of days, 3-6 payouts land within that day.

Amounts are lognormal and clipped to USD 5-1,500, so some honest payouts are above the USD 500 review threshold.

Hard negatives, so that flagging anything unusual is not enough:

- Small shops: groups of 3 workers who are all paid by one shop requester. A graph rule may see them as a ring.
- Family sharing: some pairs of workers share a device (pairs numbering 2.5% of workers) or a payout destination (1.5%).

### Attack families

Attackers are parameterized generators. They are not an LLM. `sample_params` draws parameters uniformly within bounds, and `mutate` in `payguard/sim/attacks.py` jitters them within the same bounds.

- **ring:** 4-30 accounts (default 12), each with its own device and destination, normal amounts and rate, and aged accounts. Each account is clean on its own. The only signal is shared structure: the ring's workers are paid by the same small set of 1-5 requesters.
- **structuring:** 1-4 young accounts make bursts of 2-6 payouts within 1-12 hours. Each payout is just under the USD 500 review threshold (by 1-60 USD).
- **sleeper (held out):** 1-4 accounts behave honestly for 2-9 days, then make 2-6 cash-outs spaced 2-36 hours apart. Each cash-out is 3-12 times the account's own baseline, capped at USD 450, so it stays under the threshold. Only the cash-outs count as attack (loss); the trust-building payouts before them are labelled honest, because in a real bust-out they produce no chargebacks, so no fraud label can reach a defender before the cash-out starts. This family is used only for the held-out split.

### Verdicts and their costs

Every event gets one verdict. Costs are defined in `payguard/sim/engine.py`:

- **RELEASE:** paid immediately. The label arrives 72 h after the request.
- **STEP_UP:** the requester must pass extra verification, which takes 4 h. The pass rate depends on the true type: honest 95%, ring 50%, sleeper 35%, structuring 20%. A pass is paid at 4 h. A fail is never paid. The label arrives 72 h after a pass and 4 h after a fail.
- **HOLD:** human review with a 24 h turnaround. The reviewer is correct 90% of the time. A correct review pays honest payouts and blocks attacks. A wrong review does the reverse. Every HOLD label arrives 24 h after the request.

For block-rate metrics, HOLD and STEP_UP both count as "blocked". Only RELEASE does not.

### Metrics

Defined in `payguard/sim/metrics.py`. All are computed over replayed episodes:

- `leakage_pct`: attack USD that was actually paid, as a share of all attack USD. Also reported per family, with `leaked_usd`.
- `honest_block_rate`: share of honest payouts that were not released instantly.
- `honest_unpaid_rate`: share of honest payouts that were never paid (failed step-up or wrong review).
- `honest_median_ttp_h` and `honest_p95_ttp_h`: hours from request to payment, for honest payouts that were paid.
- `immunity_attempts`: mean number of attack payouts that succeed in a campaign before it is contained.
- `never_contained_rate`: share of campaigns that are never contained.

Containment: a campaign is contained at the first point after which 3 consecutive payouts of that campaign are all blocked (HOLD or STEP_UP, even if a STEP_UP payout is later paid). A campaign with fewer than 3 payouts needs all of them blocked. Payouts paid before that point count as leaks. A campaign that is never contained counts all of its paid payouts as leaks.

### Protocol splits

Defined in `payguard/protocol.py`. Each episode has 2-4 campaigns. They are picked round-robin from the split's families, starting at a random family, with parameters drawn uniformly within bounds.

| Split | Seeds | Families | Intended use (from the module docstring) |
|---|---|---|---|
| TRAIN | 1000-1019 | ring, structuring | may be used for tuning |
| TEST_SEEN | 2000-2019 | ring, structuring | same families, unseen seeds |
| TEST_HELDOUT | 3000-3019 | sleeper only | unseen family, never used for tuning |

### Baselines

Two trivial bounds frame every result:

- `always_release`: RELEASE every payout. Zero friction, maximum leakage.
- `always_hold`: HOLD every payout. Maximum friction; leakage limited only by review accuracy.

Pre-committed baselines, fixed before any swarm code was written:

- **(a) `static_rules`** (implemented, final): six candidate per-account rules (big payout, 24 h total, device sharing, young account, velocity, drift from the worker's own median). It uses no cross-account graph features and does not learn. To make it a fair opponent, its thresholds and HOLD/STEP_UP actions were tuned once on TRAIN seeds only, with the same pre-registered objective the swarm is scored with (`python -m scripts.tune_static`: 20,000 random configurations plus coordinate descent; the report is in `eval/static_rules_tuning.json`). The tuned rules keep three: HOLD a payout of USD 300 or more, STEP_UP a payout of USD 250 or more from an account under one day old, and HOLD a payout at least 2x the worker's own median.
- **(b)** best single defender, **(c)** single cached LLM judge, **(d)** frozen swarm, **(e)** evolved swarm, **(f)** evolved swarm + shared memory: not built yet.

`scripts/run_baselines.py` runs the two bounds and `static_rules` on each of the three splits and prints a header per split, then one line per defender.

### Freeze

`scripts/freeze.py` records SHA-256 hashes of the benchmark files in `eval/FROZEN.json`. These are the simulator (`payguard/sim/*.py`), `payguard/protocol.py` and the two baseline files.

- `python -m scripts.freeze --write` writes the hashes. It was run once, after the static baseline was final.
- `python -m scripts.freeze` checks the files against the hashes. It lists anything changed, added or missing, and exits with 1 if anything moved. If `eval/FROZEN.json` does not exist, it prints "not frozen yet" and exits with 0.

The freeze exists so that no defender is tuned against a moving benchmark. Changing a frozen file after the freeze needs a logged decision and a new freeze. The benchmark is frozen; `tests/test_frozen.py` fails if any frozen file changes.

### Honesty note

Everything here is a simulated replay on synthetic data. The parameters are design choices, not calibrated to real-world fraud statistics, and this README makes no claim about real fraud rates. The simulator never calls PayPal. Real PayPal Sandbox Payouts are used only by the separate live adapter (in progress), never by the benchmark.

## What is in the repo today

| Path | Purpose |
|---|---|
| `payguard/paypal_client.py` | Sandbox-only PayPal client: OAuth, create payout batch, get batch |
| `payguard/config.py` | Reads secrets from a local `.env` (never committed) |
| `scripts/smoke_paypal_payouts.py` | Day-0 gate: one real sandbox payout batch |
| `scripts/smoke_sarvam.py` | Day-0 gate: schema-valid JSON from the LLM defender |
| `payguard/sim/` | Deterministic payout simulator, attack generators, features, runner, metrics |
| `payguard/protocol.py` | Seed splits (TRAIN / TEST_SEEN / TEST_HELDOUT) and episode construction |
| `payguard/baselines/` | Trivial bounds and the static-rules baseline (a) |
| `scripts/run_baselines.py` | Prints the baseline table on all three splits |
| `scripts/tune_static.py` | One-time TRAIN-only tuning of the static baseline (`eval/static_rules_tuning.json`) |
| `scripts/freeze.py` | Freezes / verifies the benchmark files (`eval/FROZEN.json`) |
| `tests/` | Offline tests (no network, no keys) |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -e ".[dev]"
pytest                           # offline, needs no keys
cp .env.example .env             # then fill in your own sandbox credentials
python -m scripts.smoke_paypal_payouts
python -m scripts.smoke_sarvam
```

## How PayPal and AI are used

To be completed as features land. Planned:

- **PayPal:** Payouts API (the action the swarm takes), batch polling, signed webhooks feeding the learning loop. Sandbox only.
- **AI:** one LLM defender (Sarvam, structured JSON output, used only on borderline cases and cached) plus the evolution loop that tunes the deterministic defenders.

## License

MIT, see [LICENSE](LICENSE).
