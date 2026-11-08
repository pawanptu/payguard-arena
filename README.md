# PayGuard Arena

A co-evolving fraud arms race on instant gig payouts. A swarm of defender agents decides whether to **release, hold or step-up** each payout through **PayPal sandbox Payouts**, while a population of attackers mutates its tactics. The defenders self-evolve through outcome-driven reputation updates and a champion-challenger gate.

> **Status:** built for the PayPal AI Hackathon (Nov 2026). Every number shown by this project is from a **simulated replay** on synthetic data, never real-world fraud data, and only sampled payouts go through the real PayPal sandbox (test money). Two pre-registered tests were run, each once: **round 1 failed (NO-GO)** on a held-out sleeper family; **round 2 passed (GO)** on a second sealed family, with caveats stated next to it. See [Results](#results).

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
- **h2, account takeover (sealed, round 2):** a crew takes over aged worker accounts (60-600 days) in one wave and drains their pending earnings. Amounts stay ordinary for the victim (about 1-4x its own median) and under the review threshold by a margin. The payouts go to 1-3 shared cash-out destinations from 1-3 attacker devices, while requesters look normal. It was written by a separate worker who never saw the round-2 defender, frozen by hash, and never built before that defender was frozen (`payguard/heldout2/`, design note `docs/HELDOUT2_FAMILY.md`). It is used only for TEST_HELDOUT2.

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
| TEST_HELDOUT2 | 4000-4019 | h2 only | round 2: second sealed family (`payguard/heldout2/protocol.py`); a test tripwire refuses these seeds |

### Baselines

Two trivial bounds frame every result:

- `always_release`: RELEASE every payout. Zero friction, maximum leakage.
- `always_hold`: HOLD every payout. Maximum friction; leakage limited only by review accuracy.

Pre-committed baselines, fixed before any swarm code was written:

- **(a) `static_rules`** (implemented, final): six candidate per-account rules (big payout, 24 h total, device sharing, young account, velocity, drift from the worker's own median). It uses no cross-account graph features and does not learn. To make it a fair opponent, its thresholds and HOLD/STEP_UP actions were tuned once on TRAIN seeds only, with the same pre-registered objective the swarm is scored with (`python -m scripts.tune_static`: 20,000 random configurations plus coordinate descent; the report is in `eval/static_rules_tuning.json`). The tuned rules keep three: HOLD a payout of USD 300 or more, STEP_UP a payout of USD 250 or more from an account under one day old, and HOLD a payout at least 2x the worker's own median.
- **(b) best single defender** (implemented): each of the four swarm specialists on its own (`payguard/swarm/single.py`); the best one is chosen on TRAIN.
- **(c) single cached LLM judge** (implemented, `payguard/llm/judge.py`): one model sees a coarse, truth-free summary of each payout and decides it. Two judges are reported: Sarvam `sarvam-105b-conversations` (JSON-schema output) and Google `gemini-3.5-flash-lite`. Replies are cached by payload (`data/llm_cache.jsonl`, `data/llm_cache_gemini.jsonl`), so a warmed cache replays offline and deterministically. Paid calls are opt-in: `PAYGUARD_LLM_MAX_CALLS` defaults to 0, and past the cap a fixed fallback rule decides. The judges are baselines, not part of the swarm.
- **(d) frozen swarm** (implemented): the four specialists (Velocity, Graph, BaselineDrift, Policy) and the coordinator, with weights that never change (`payguard/swarm/`). `swarm_l1` adds online reputation learning of the weights.
- **(e) evolved swarm** (implemented, `payguard/evolve/`): weights and thresholds evolved by population search against co-evolving attackers, with a champion-challenger gate on held-out TRAIN seeds. Champion `g17-7`.
- **(f) evolved swarm + shared memory** (implemented, `payguard/swarm/memory.py`): the evolved swarm plus a shared, decaying memory of the workers, devices, destinations and requesters touched by confirmed fraud, aimed at faster time-to-immunity.
- **(g) `swarm_r2`** (round 2, `payguard/robust/`): the L1 swarm plus an anomaly layer of 9 history-only statistics whose cut-offs are honest-only quantiles from TRAIN (budgets: 2% of honest events may be held, 5% stepped up). The verdict is the more severe of the swarm's and the layer's. It was selected on TRAIN only by leave-one-family-out ([Round 2](#round-2-second-sealed-family)).

`scripts/run_baselines.py` runs the two bounds and `static_rules` on each split and prints a header per split, then one line per defender. `scripts/run_swarm.py` runs (a), (b), (d) and the L1 swarm on TRAIN only. The test splits were run once each, by `scripts/final_eval.py` (round 1) and `scripts/final_eval_r2.py` (round 2); see [Results](#results). Two known limits:
- The Graph specialist's default cluster-size threshold (4 workers) sits just above the simulator's honest small-shop size (3 workers). That is a design choice that knows the simulator.
- Online reputation learning (L1) adds almost nothing on TRAIN. The gain comes from the specialists complementing each other.

### Freeze

`scripts/freeze.py` records SHA-256 hashes of the benchmark files in `eval/FROZEN.json`. These are the simulator (`payguard/sim/*.py`), `payguard/protocol.py` and the two baseline files.

- `python -m scripts.freeze --write` writes the hashes. It was run once, after the static baseline was final.
- `python -m scripts.freeze` checks the files against the hashes. It lists anything changed, added or missing, and exits with 1 if anything moved. If `eval/FROZEN.json` does not exist, it prints "not frozen yet" and exits with 0.

The freeze exists so that no defender is tuned against a moving benchmark. Changing a frozen file after the freeze needs a logged decision and a new freeze. The benchmark is frozen; `tests/test_frozen.py` fails if any frozen file changes.

Round 2 adds two more freezes. `eval/FROZEN_R2.json` covers the sealed family (`python -m scripts.freeze --round 2`). `eval/final_r2/defender_freeze.json` covers the `swarm_r2` champion and the code it runs (`payguard/robust/`, `payguard/swarm/`); every round-2 command except the freeze itself refuses to run if any of those files changed.

### Honesty note

Everything here is a simulated replay on synthetic data. The parameters are design choices, not calibrated to real-world fraud statistics, and this README makes no claim about real fraud rates. The simulator never calls PayPal. Real PayPal Sandbox Payouts are used only by the separate live rail (`payguard/rail/live_rail.py`), never by the benchmark.

## Results

**Every number in this section comes from a simulated replay on synthetic data.** Nothing is calibrated to real fraud rates. Only rows marked **LIVE** used the PayPal sandbox, with test money.

### Summary

- **Round 1 failed its pre-registered test (NO-GO).** On the held-out sleeper family, the evolved swarm scored **80.56**, static rules **28.54**, and the best single specialist (`single_policy`) **64.15** (lower is better). The swarm was worse than both and also worse than `always_hold` (71.45).
- On the seen families (TEST_SEEN) the same swarm scored 17.14 against 65.64 for static rules. That pass is secondary: those families were used to build and tune the defenders.
- **Round 2 passed its pre-registered test (GO).** On a second sealed family nobody built against (account takeover with a shared cash-out channel), the new defender `swarm_r2` scored **26.30** against **52.21** for static rules and **100.09** for `single_policy`, with honest block 7.4% (static rules 9.7%).
- **Read round 2 with these caveats, from the same table.**
  - On that split the Graph specialist alone (`single_graph`) scored 28.87, only 2.57 behind `swarm_r2`, which is under the 3.0 margin. The rule compares against the single chosen on TRAIN, as pre-registered.
  - Every round-1 swarm also clears the rule there, so the result mostly shows that the swarm design handles a shared-channel family that static rules miss. The round-2 changes add about 2-4 points.
  - Round 1 still failed; its numbers stand.

### Objective and pass rule

Fixed before any swarm result existed (D-019):

```
objective = leakage_pct + 0.5 * honest_block_rate + 1.0 * honest_unpaid_rate     (lower is better)
```

The metrics are defined in [Metrics](#metrics).

Go/no-go rule, written before any TEST number existed (D-022). On the held-out split, the primary defender passes only if all three hold:

1. Its objective is at least **3.0 lower** than `static_rules`.
2. Its objective is at least **3.0 lower** than the best single specialist. That specialist is chosen on TRAIN (it is `single_policy`), not on the test.
3. Its honest block rate is no higher than `static_rules`'.

Each round's test splits were run once, by one script, after the defender was fixed. The scripts refuse to run again while their results file exists (`eval/final/results.json`, `eval/final_r2/results.json`).

### Round 1: held-out sleeper family

Primary defender: `swarm_evolved_memory` (champion `g17-7`, SHA-256 `0fefe00f54942385c6a9a6a86e5ccaa14264665713c7bf9bf2e77c4ebb4dfe82`). Chosen on TRAIN only. 20 episodes per split, about 80,000 honest payouts per split. Values are copied from `eval/final/results.json` (git rev `121c8ba`). SIMULATED.

**Verdict: NO-GO.**

| Check | Split | Swarm | Static rules | Best single | Margin vs static | Margin vs single | Honest block (swarm / static) | Pass |
|---|---|---|---|---|---|---|---|---|
| Decisive | TEST_HELDOUT (sleeper) | 80.557 | 28.539 | 64.153 | -52.019 | -16.404 | 2.835 / 9.595 | no |
| Secondary | TEST_SEEN (ring, structuring) | 17.142 | 65.639 | 70.371 | +48.496 | +53.229 | 3.085 / 9.849 | yes |

A margin is the other defender's objective minus the swarm's. A pass needs +3.0 on both.

#### TEST_HELDOUT (seeds 3000-3019, sleeper only). Decisive.

| Defender | Objective | Leakage % | Honest block % | Honest unpaid % |
|---|---|---|---|---|
| always_release | 100.000 | 100.000 | 0.000 | 0.000 |
| always_hold | 71.453 | 11.513 | 100.000 | 9.940 |
| static_rules | **28.539** | 22.783 | 9.595 | 0.958 |
| single_policy (best single, chosen on TRAIN) | 64.153 | 63.026 | 1.989 | 0.133 |
| llm_judge_sarvam | 102.605 | 96.256 | 11.443 | 0.628 |
| llm_judge_gemini | 65.202 | 50.445 | 26.757 | 1.379 |
| swarm_frozen | 55.107 | 51.775 | 5.878 | 0.393 |
| swarm_l1 | 55.097 | 51.775 | 5.858 | 0.393 |
| swarm_evolved | 80.232 | 78.545 | 2.834 | 0.270 |
| swarm_memory | 60.017 | 56.690 | 5.866 | 0.394 |
| swarm_evolved_memory (primary) | 80.557 | 78.870 | 2.835 | 0.270 |

#### TEST_SEEN (seeds 2000-2019, ring and structuring). Secondary.

| Defender | Objective | Leakage % | Honest block % | Honest unpaid % |
|---|---|---|---|---|
| always_release | 100.000 | 100.000 | 0.000 | 0.000 |
| always_hold | 70.040 | 10.130 | 100.000 | 9.910 |
| static_rules | 65.639 | 59.831 | 9.849 | 0.883 |
| single_policy (best single, chosen on TRAIN) | 70.371 | 69.291 | 1.871 | 0.144 |
| llm_judge_sarvam | 98.149 | 91.185 | 12.426 | 0.751 |
| llm_judge_gemini | 83.478 | 68.651 | 26.978 | 1.338 |
| swarm_frozen | 20.459 | 17.158 | 5.836 | 0.382 |
| swarm_l1 | 20.451 | 17.158 | 5.825 | 0.380 |
| swarm_evolved | 18.280 | 16.505 | 3.003 | 0.273 |
| swarm_memory | 18.493 | 15.133 | 5.946 | 0.387 |
| swarm_evolved_memory (primary) | **17.142** | 15.322 | 3.085 | 0.278 |

TRAIN rows are in `eval/final/results.json` as well. They are not evidence: the defenders were tuned on TRAIN.

#### Why round 1 failed (descriptive, no tuning on the test)

This reading comes from the table above; we did not run a new experiment on TEST data to confirm it.

- Evolution lowered honest friction (honest block 5.9% to 2.8% on the held-out split) by loosening thresholds. Sleepers pass those looser thresholds: leakage went from 51.8% for the fixed-weight swarm to 78.5% for the evolved one.
- Shared memory did not help against sleepers, because a sleeper builds a clean history first. Adding memory to the fixed-weight swarm moved it from 55.11 to 60.02 (worse), and the evolved swarm with memory (80.56) is no better than without it (80.23).
- The tuned static rules include a rule on payouts at least 2x the worker's own median. That rule happens to suit the sleeper family, which is why static rules are strong here.
- The two LLM judges are baselines, not part of the swarm. Both scored worse than static rules on the held-out split. Replayed from cache, the Sarvam judge fell back to its fixed rule on 0 payouts and the Gemini judge on at most 0.007% of payouts (its free daily quota ran out).
- The TEST data of round 1 is spent. Any new claim needs a new, never-built held-out family. That is round 2.

### Round 2: second sealed family

Decided before the round-2 defender existed (D-034, amended by D-035):

- **Family.** A second attack family (h2, see [Attack families](#attack-families)) was written by a separate worker who did not see the defender. Its code is frozen by hash in `eval/FROZEN_R2.json`, and its seeds (4000-4019, split TEST_HELDOUT2) were not built during development. A test tripwire refuses them, and a scanner test fails if any other test file names them.
- **Defender.** `swarm_r2` was designed on TRAIN only and selected by leave-one-family-out (4 folds: fit on one TRAIN family and half of the TRAIN seeds, score on the other family and the other half), with a rule written before scoring (`eval/r2/SELECTION_RULE.md`). Champion candidate `C2-bh2-bs5`, `eval/r2/champion.json` SHA-256 `fe38d6c8408c21ebdf485045c7b746b1ad9b9261a8aee1d90c4967b3b502a17e`. It was frozen (`eval/final_r2/defender_freeze.json`, git rev `03e2da4`) before any TEST_HELDOUT2 episode was built.
- **Disclosed amendment.** The selection rule was amended once, after a TRAIN-only smoke run and before the full grid (Amendment 1: an honest-block margin of 0.75 points and a 1.5-point noise band, then the most sensitive candidate). Under the original rule the pick would have been the default swarm, which is reported below as `swarm_l1`. It also passes, so the amendment did not decide the verdict.
- **Rule.** Same rule as round 1, same margins, applied to `swarm_r2` on TEST_HELDOUT2. One run, no re-runs, no tuning afterwards.
- **Gemini.** The Gemini judge row is added in a second, separate run (its free quota was spent). It cannot change any other row or the verdict: the rule reads only `swarm_r2`, `static_rules` and `single_policy`.

**Verdict: GO** (`eval/final_r2/results.json`).

| Check | Split | swarm_r2 | Static rules | Best single (TRAIN pick) | Margin vs static | Margin vs single | Honest block (swarm_r2 / static) | Pass |
|---|---|---|---|---|---|---|---|---|
| Decisive | TEST_HELDOUT2 (h2) | 26.298 | 52.206 | 100.086 | +25.908 | +73.788 | 7.385 / 9.745 | yes |

#### TEST_HELDOUT2 (seeds 4000-4019, h2 only; 20 episodes, 83,173 payouts). Decisive for round 2.

All rows, sorted by objective:

| Defender | Objective | Leakage % | Honest block % | Honest unpaid % |
|---|---|---|---|---|
| swarm_r2 (primary) | **26.298** | 22.085 | 7.385 | 0.521 |
| swarm_memory | 28.190 | 24.800 | 5.947 | 0.417 |
| swarm_evolved_memory (round-1 primary) | 28.849 | 27.007 | 3.085 | 0.300 |
| single_graph (best single on this split; not the rule's single) | 28.866 | 26.875 | 3.490 | 0.246 |
| swarm_l1 (strict-rule pick, see above) | 29.061 | 25.669 | 5.952 | 0.416 |
| swarm_frozen | 29.700 | 26.305 | 5.961 | 0.414 |
| swarm_evolved | 30.127 | 28.285 | 3.084 | 0.300 |
| static_rules | 52.206 | 46.349 | 9.745 | 0.984 |
| llm_judge_sarvam | 68.449 | 61.856 | 11.705 | 0.740 |
| always_hold | 69.139 | 9.032 | 100.000 | 10.107 |
| single_drift | 95.386 | 94.796 | 1.026 | 0.077 |
| always_release | 100.000 | 100.000 | 0.000 | 0.000 |
| single_policy (best single, chosen on TRAIN) | 100.086 | 99.091 | 1.751 | 0.120 |
| single_velocity | 100.851 | 100.000 | 1.501 | 0.100 |
| llm_judge_gemini (added in a later run) | pending<!-- TODO-R2GEMINI --> | | | |

The Sarvam judge replayed all 83,173 payouts from cache with 0 fallbacks.

`swarm_r2` on the other splits: TRAIN 20.823, TEST_SEEN 21.049, TEST_HELDOUT 45.101. The TEST_HELDOUT value is **not a clean test**, because round 1's failure on that family was known while `swarm_r2` was being designed. It is also worse than static rules there (28.539): the sleeper family is still lost.

#### Reading round 2

- `swarm_r2` passed the pre-registered rule on a second family it had never seen. This does not undo round 1. It is one family, one run, on synthetic data.
- The Graph specialist alone is close (28.87, +2.57). This family's footprint is in the payout channel (shared destinations and devices), which is the kind of structure the Graph specialist was built to see. Against the best single *on the test*, `swarm_r2` would not clear the 3.0 margin. The rule fixed the single on TRAIN to stop exactly this kind of after-the-fact choice, so the verdict stands, and we show both.
- Every round-1 swarm scores 28.2-30.1 here, against 52.2 for static rules. The round-2 layer adds 1.9-3.8 points over them, at a higher honest block rate (7.4% vs 3.1-6.0%).
- We did not analyse why static rules miss this family, and we will not tune anything on it. The two facts on record: the drains are about 1-4x the victim's own median and stay under the review threshold (design note), while the tuned static rules act at 2x the worker's median or USD 300.

### LIVE rows (PayPal sandbox)

The benchmark never calls PayPal. The LIVE rail is separate (`payguard/rail/live_rail.py`): `scripts/live_run.py` sends a small sample of the payouts the swarm released as real sandbox Payouts batches, guarded by a per-payout cap, a session budget and a receiver allowlist, then waits for PayPal's signed webhooks.

- First live run: 5 sandbox payouts of USD 1.00 each (test money), all batches SUCCESS, and 15 signature-verified webhooks received for them. They paid released events of the round-1 episode (TEST_HELDOUT seed 3000); the receipts are kept in `eval/live/round1_replay_live.json`.
- <!-- TODO-R2LIVE: replace this bullet with the round-2 live run's numbers --> A second live run on the round-2 episode (TEST_HELDOUT2 seed 4000, the episode the dashboard shows) is pending.
- These rows show that the rail works end to end. They are not a fraud result and are not in the tables above.

### How to reproduce

Offline, no keys, no network:

```bash
python -m pytest -q
python -m scripts.freeze                  # prints "FROZEN files match" if the benchmark is untouched
python -m scripts.freeze --round 2        # prints "FROZEN_R2 files match" if the sealed family is untouched
python -m scripts.run_baselines           # trivial bounds + static rules on every split
python -m scripts.run_swarm               # TRAIN only: baselines, single specialists, swarm
python -m scripts.final_eval --plan       # counts LLM payloads per split; makes no call and runs no defender
python -m scripts.final_eval_r2 --plan    # the same for round 2 (checks the defender freeze first)
```

The round-1 table is stored in `eval/final/results.json` and the round-2 table in `eval/final_r2/results.json` (champion hash, git rev, every row). `python -m scripts.final_eval --run` and `python -m scripts.final_eval_r2 --run` are the one-shots; each refuses to run while its results file exists. The LLM judges replay from `data/llm_cache.jsonl` (Sarvam) and `data/llm_cache_gemini.jsonl` (Gemini), so a rerun needs no API key.

### Limits

- Simulator and attackers are our own design, and we wrote every attack family. Beating them says little about real fraud.
- Round 1 used one decisive family and 20 episodes; round 2 used one more family and 20 episodes. Two families are not a general result, and the defender lost one of them.
- The Graph specialist's default cluster-size threshold (4 workers) sits just above the simulator's honest small-shop size (3 workers). That is a design choice that knows the simulator.
- Online reputation learning (L1) added almost nothing measurable; the gains come from complementary specialists, from evolved thresholds (round 1) and from the honest-calibrated anomaly layer (round 2).

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
| `payguard/swarm/` | Four specialist defenders, the reputation-weighted coordinator (L1), single-specialist baselines |
| `scripts/run_swarm.py` | TRAIN-only table: baselines vs single specialists vs swarm |
| `payguard/rail/` | Payout rails: `SimRail` (simulated) and `LiveRail` (PayPal sandbox, guarded by a per-payout cap, a session budget and a receiver allowlist), a receipt store, and the signature-verified webhook `POST /webhooks/paypal` |
| `payguard/llm/` | Cached Sarvam LLM judge (baseline c) and LLM quality specialist; truth-free payload, opt-in paid calls |
| `scripts/estimate_llm_cost.py` | Offline count of the LLM calls a cold cache would need (makes no API call) |
| `payguard/api/` | FastAPI app: replay API, payout receipts, PayPal webhook, and the built dashboard |
| `web/` | React + Vite + Recharts dashboard; `web/dist` is the committed build that the API serves |
| `scripts/make_sample_replay.py` | Builds the sample replay fixture (TRAIN seed 1000, marked `synthetic_fixture`) |
| `payguard/evolve/`, `scripts/evolve.py` | Population search with co-evolving attackers and a champion-challenger gate (baseline e); lineage in `eval/evolution/` |
| `payguard/swarm/memory.py`, `scripts/run_memory.py` | Shared attack memory (baseline f) |
| `scripts/final_eval.py` | Round-1 one-shot on the test splits (`eval/final/results.json`) |
| `payguard/heldout2/` | Round 2: the sealed attack family h2 and split TEST_HELDOUT2 (frozen in `eval/FROZEN_R2.json`) |
| `payguard/robust/`, `scripts/r2_select.py` | Round 2: `swarm_r2` (honest-calibrated anomaly layer) and its TRAIN-only leave-one-family-out selection (`eval/r2/`) |
| `scripts/final_eval_r2.py` | Round-2 one-shot (`eval/final_r2/results.json`) and the replay the dashboard shows (`eval/replay.json`) |
| `scripts/llm_warm.py` | Fills the LLM caches on TRAIN under an explicit call cap (the one-shot scripts warm their own test splits, also capped) |
| `scripts/live_run.py` | Recorded live run: pays a few released replay events through the PayPal sandbox and records receipts and verified webhooks (dry run unless `--send`) |
| `eval/live/` | Kept live-run receipts |
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

Dashboard (no keys needed). It replays the file in `PAYGUARD_REPLAY_PATH`, else `eval/replay_live.json`, else `eval/replay.json`, else the sample fixture:

```bash
uvicorn payguard.api.app:app --port 8000        # serves the API and the built UI at http://127.0.0.1:8000
cd web && npm ci && npm run dev                 # UI development server, proxies /api to port 8000
cd web && npm run build                         # rebuild web/dist (commit it; the deploy has no Node)
```

Deploy (Render, Python only, no secrets needed for Replay mode): build `pip install -r requirements.txt`, start `uvicorn payguard.api.app:app --host 0.0.0.0 --port $PORT`, health check `/api/health`.

The PayPal webhook (`POST /webhooks/paypal`) accepts nothing until `PAYPAL_WEBHOOK_ID` is set: it returns 503, and every event is checked with PayPal's verify-webhook-signature API before it changes a receipt.

## How PayPal and AI are used

**PayPal (sandbox only):**

- **Payouts API** is the action the defenders take. A released payout becomes a sandbox batch (`POST /v1/payments/payouts`), and batch and item status are polled (`GET /v1/payments/payouts/{id}`). OAuth client credentials authenticate every call, and the client refuses any host other than `api-m.sandbox.paypal.com`.
- **Webhooks:** `POST /webhooks/paypal` checks every event with PayPal's `verify-webhook-signature` API before it changes a payout receipt. The 11 payout events the handler maps are registered.
- **Guards:** the live rail has a per-payout cap, a session budget and a receiver allowlist. The hosted dashboard runs in replay mode and has no endpoint that sends payouts.

**AI:**

- **Self-evolving defenders:** the swarm's weights and thresholds are evolved by population search against attackers that mutate their parameters, with a champion-challenger gate. This is search, not an LLM.
- **Learned calibration (round 2):** `swarm_r2` learns its anomaly cut-offs from honest TRAIN traffic only, under fixed friction budgets.
- **LLM judges (baselines):** Sarvam `sarvam-105b-conversations` and Google `gemini-3.5-flash-lite` each decide payouts from a truth-free summary with structured output. They are baselines, not part of the swarm. Both scored worse than tuned static rules on the round-1 held-out family, and the Sarvam judge did on round 2 as well (<!-- TODO-R2GEMINI -->the Gemini round-2 row is pending).
- The project was built with Claude Code (Anthropic models) for planning, coding and review.

## License

MIT, see [LICENSE](LICENSE).
