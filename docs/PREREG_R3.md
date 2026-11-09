# Round-3 pre-registration (DRAFT for user approval): "the swarm that dreams and doubts"

> Repo copy, added 2026-11-09, text unchanged from the version the user approved on 2026-11-08 (D-041), so the "DRAFT" status line below is historical. Results: `eval/final_r3/results.json` (H1 GO, H2 FAIL, D-046).

Written 2026-11-08, **before any round-3 code exists**. Nothing here is built. Status: waiting for the user's approval (milestone M0). Prior art and its limits: `docs/PRIOR_ART.md`. Conventions follow D-034 (round 2): sealed held-out family built blind, freeze by hash, one-shot run, publish whatever happens.

## 1. What changes and what stays

- **Stays (frozen, reused unchanged):** simulator, workers, TRAIN families (ring, structuring), the four specialist defenders, coordinator, shared memory, objective (`leakage% + 0.5 x honest_block% + honest_unpaid%`), the champion-challenger gate (`MIN_GAIN` 0.5, `MAX_BLOCK_INCREASE` 1.0 pp), the PayPal Sim and Live rails, the dashboard, and every round-1 and round-2 number (the NO-GO and the GO-with-caveats stay visible in every write-up).
- **New:** four components on top of the existing evolution loop (`payguard/evolve/`): a world model, a dream loop with an Imaginer, a Skeptic, and a calibration rule for how much the swarm trusts its dreams.
- **Not new, not claimed as novel:** the genetic algorithm and the gate (stage-6 self-evolution is well published, see PRIOR_ART section 2).

## 2. Design

Per generation `g` of the existing loop:

1. **World model `WM_g`.** For each TRAIN attack family, a posterior over the parameters of its generator (ring size/overlap, structuring epsilon, burst rate, amount distribution), fitted by approximate Bayesian computation from summary statistics of attacks the swarm *confirmed* (labels arrive 4-72 h late, as today). Plus a "recombination" component that samples tactic mixes (parameters of different TRAIN families combined). Parameter ranges and recombination operators are defined **from TRAIN families only**. Family #3 may lie outside the model's reach; that is part of the test.
2. **Imaginer.** Samples `N` dream attackers per generation: a fixed share from the posterior, a share from regions the Skeptic marks uncertain, and a share from recombinations. The shares are fixed at freeze (section 5).
3. **Dream evaluation.** Each candidate defender (a challenger from the GA) is simulated against the dream attackers, giving a predicted objective.
4. **Skeptic.** Estimates where `WM_g` is unreliable: fit `WM` on part of TRAIN, predict the summary statistics of the held-back part, record the error per region. A dream in a high-error region counts less (weight `1 / (1 + err / tau)`).
5. **Calibrated trust.** After each generation, `dream_gap_g = | predicted gain of the promoted challenger - realized gain on the real gate set |`. The weight of dreamed fitness in the GA, `lambda_g = lambda_0 / (1 + EMA(dream_gap) / sigma)`, shrinks when the swarm has been dreaming badly. **The promotion gate itself stays real and unchanged** (real TRAIN_B gate episodes, existing rule), so imagination can propose but never promote alone.
6. **Reality anchor.** Labels and outcomes come from the simulator's ground truth with delayed labels. The PayPal sandbox rail is used for the live demo (real receipts and verified webhooks); it does **not** supply the labels.

Core design uses **no LLM** (statistical world model), so it costs no paid calls and is reproducible offline. An LLM Imaginer, or defenders written as rules by an LLM, is a stretch outside every hypothesis below; it would need its own pre-registration and, per PRIOR_ART, would not be a novelty claim.

## 3. Arms evaluated on the sealed set

| Arm | Meaning |
|---|---|
| `static_rules` | frozen baseline (a), unchanged |
| `single_policy` | best single specialist chosen on TRAIN (round-1 rule), unchanged |
| `swarm_r2` | the round-2 champion, unchanged (evolved without dreaming) |
| `dreamer_full` | all four new components |
| `dreamer_nowm` | same pipeline, dreams drawn uniformly from the same parameter ranges (no world model) |
| `dreamer_noskeptic` | no Skeptic weighting |
| `dreamer_nocalib` | `lambda` fixed (no calibrated trust) |

All arms are scored on TEST_HELDOUT3 once. TRAIN, TEST_SEEN, TEST_HELDOUT and TEST_HELDOUT2 numbers are shown for `dreamer_full` and marked **not clean** (their results were known while designing).

## 4. Hypotheses and decision rule (fixed before any score exists)

- **H1 (usefulness, D-022 applied unchanged):** on TEST_HELDOUT3, `dreamer_full` objective is at least **3.0 lower** than BOTH `static_rules` and `single_policy`, with honest block no higher than `static_rules`'. If H1 fails: **NO-GO**, reported exactly like round 1.
- **H2 (dreaming adds value):** `dreamer_full` is at least **2.0 lower** than `dreamer_nowm` on TEST_HELDOUT3. If H1 passes but H2 fails, the honest statement is "the swarm works, the world model did not matter".
- **H3 (calibration, report only):** the EMA of `dream_gap` over the last third of generations vs the first third on TRAIN. No pass/fail.
- **H4 (immunity, report only):** time-to-immunity (attack attempts that succeed before the whole swarm blocks a new tactic) on TEST_HELDOUT3.
- Ablations `nosceptic` and `nocalib` are reported in the table whatever they show (including "no difference").
- Statistics: means over the 20 episodes, with bootstrap 95% intervals over episodes; the rule uses the means, as in D-022. No re-runs, no tuning after the run.

## 5. Protocol and seal

- **Sealed family #3:** package `payguard/heldout3/`, built by a **separate worker blind to the defender, the world model and the Imaginer** (as D-034). Split TEST_HELDOUT3 = seeds **5000-5019** (no collision with 1000-1019, 2000-2019, 3000-3019, 4000-4019). A pytest tripwire raises if any episode on these seeds is built during development. Hashes go in `eval/FROZEN_R3.json`; `python -m scripts.freeze --round 3` must print "FROZEN files match". The family's design note stays unread by the director until `dreamer_full` is frozen.
- **Selection:** all dreamer hyper-parameters (shares, `N`, `tau`, `sigma`, `lambda_0`, recombination operators) are chosen on TRAIN only, by leave-one-family-out (ring <-> structuring), using a rule written in `eval/r3/SELECTION_RULE.md` **before any scoring**. The chosen values and the champion hash are appended to the decision log at freeze, before any TEST_HELDOUT3 episode is built.
- **Dev check K1 (before spending the sealed set):** on TRAIN leave-one-family-out, `dreamer_full` must beat `dreamer_nowm` by at least **1.0**. If not, the design has failed its own development test: stop, publish the negative finding, and **do not** spend TEST_HELDOUT3 on it.

## 6. Kill criteria and honesty rules

- **K1** above. **K2:** if the world model cannot be fitted stably on TRAIN (posteriors collapse or the ABC does not converge) at milestone M2, abort round 3 and keep the current submission as is. **K3:** any information about family #3 reaching the dreamer's design voids the run.
- The round-1 NO-GO and the round-2 caveats stay in every write-up. A failed H1 or H2 ships as a failure; a pass ships with its caveats.
- Prior rounds' test sets stay spent. The simulator is ours: every result is "simulated, synthetic data".

## 7. Milestones (gates, no dates) and approvals

| # | Gate | Needs the user |
|---|---|---|
| M0 | Approve this document (or amend it) | **yes** |
| M1 | Family #3 built blind, sealed, hashes in `FROZEN_R3.json` | no |
| M2 | World model + dream loop + Skeptic + calibration on TRAIN, tests green, K2 check | no |
| M3 | Selection on TRAIN, K1 dev check | no |
| M4 | Freeze: hashes, commit, decision-log entry | no |
| M5 | One-shot TEST_HELDOUT3 run (`scripts/final_eval_r3.py`) | no (rule already fixed) |
| M6 | Write-up, dashboard card, video, Devpost and README edits, any push to `main` | **yes, each** |

Scheduling note, not a constraint on design: the hackathon's submit-by is Fri Nov 13 01:30 IST; whether round-3 results are entered in this hackathon is the user's decision at M6. The current submission stays untouched until then.
