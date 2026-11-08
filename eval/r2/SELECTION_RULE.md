# swarm_r2 selection rule (written BEFORE any candidate was scored)

Goal: a defender that generalizes to an attack family it has never seen. Data: TRAIN seeds 1000-1019, families
`ring` and `structuring` only. Nothing else is ever built (no TEST_SEEN, TEST_HELDOUT, seeds 4000-4019, or the
third family). Everything below is fixed in advance; the numbers in `eval/r2/selection.json` are produced by
`python -m scripts.r2_select` and nothing is edited by hand.

## 1. Protocol: leave-one-family-out (LOFO), and out-of-seed on top

The honest traffic of an episode is a function of its seed only, so "fit on ring seed 1005, test on structuring
seed 1005" would share the exact same honest payouts between fit and test. That is leakage on the honest side
(thresholds tuned to those payouts), and the sealed family will come with new honest traffic. So every fold is
out-of-family AND out-of-seed (stricter than the literal rule):

    halves:  H0 = seeds 1000-1009, H1 = seeds 1010-1019
    fold (A -> B, h):  fit on episodes `episode_for(seed, (A,))` for seeds in H_h,
                       evaluate on episodes `episode_for(seed, (B,))` for seeds in H_(1-h)
    directions:        ring -> structuring and structuring -> ring, h in {0, 1}   (4 folds, 10 + 10 episodes each)

Direction objective = mean of its two folds' objectives. **Selection score = mean of the two direction
objectives** (equivalently the mean of the 4 fold objectives). Objective = `leakage_pct + 0.5*honest_block_rate
+ 1.0*honest_unpaid_rate` (payguard.swarm.objective), computed per fold with the frozen metrics.

"Fitting" means everything that looks at episodes: the GA, the honest-quantile calibration, nothing else.
Hand-set design constants below were fixed before scoring and are not tuned afterwards.

## 2. Reference rows (same LOFO evaluation episodes, same aggregation)

`static_rules`, `single_policy` (default params), `swarm_l1` (default params), `swarm_evolved` (champion g17-7) and
`swarm_evolved_memory` (g17-7 + default memory). Not fitted per fold. Caveat printed with the table: g17-7 was
evolved on ALL TRAIN seeds and both families, so its LOFO rows are in-sample and optimistic for it.
Also reported for every candidate and reference: full-TRAIN objective (20 seeds x 2 families, the protocol
episodes `episode_for(seed, ("ring","structuring"))`) with its three components.

## 3. Candidates (at most three mechanisms; hyper-parameter grids are part of the mechanism and fixed here)

* **C1 Prior-anchored evolution.** GA over `SwarmParams` (payguard.evolve.ga operators, pop 20, 10 generations,
  elites 3, tournament 3, mutation scale 0.25 -> 0.05, seed 0, learn_weights=True). Fitness on the FIT episodes:
  `objective + lam * d(theta, default) + 100 * max(0, block - block_static)`, where `d` is the RMS over genes of
  the deviation from the default swarm in units of the gene's BOUNDS range, and `block_static` is static_rules'
  honest block on the same fit episodes. Principle: a parameter the fit family gives no evidence about must stay
  at its prior (MAP estimate with a quadratic prior), instead of being loosened to save honest friction.
  Grid: `lam` in {0, 20, 80}  (lam = 0 is the unregularized ablation).
* **C2 Honest-calibrated anomaly union layer on the default swarm.** Layer statistics (history only, computed
  before the event is added, no labels): window_usd, window_n, amount, amount_ratio, amount/cap(age),
  window_usd/(3*cap(age)), 24 h rate z-score, graph cluster score (the Graph lens with open ramps), max device or
  destination sharing. For each statistic the HOLD and STEP_UP cut-offs are the honest empirical upper quantiles
  (honest fit events only; no attack event is used), with one common tail level per verdict, found by bisection so
  that the UNION of the statistics flags a target share of the honest fit events: `beta_h` for HOLD, `beta_s` (>=
  beta_h) for STEP_UP or worse. The defender is `swarm_l1` with DEFAULT params; its verdict is the more severe of the
  swarm's and the layer's. Principle: an unseen attack has to deviate from the (stable, known) honest behaviour on
  some lens; allocating a fixed false-positive budget to every lens is independent of which attack was seen.
  Attack data only picks the budget. Grid `(beta_h, beta_s)` in {(0.5,2), (1,3), (1,5), (2,5), (2,8)} (percent).
* **C3 Stack.** C1 at the `lam` selected inside C1 (by the rule in section 4), plus the C2 layer on top, with
  `(beta_h, beta_s)` selected inside C3 over the same C2 grid. Only run if C1 and C2 each produced an eligible
  candidate.

## 4. Eligibility and selection

A candidate (a mechanism with one grid point) is ELIGIBLE iff all hold:
1. honest block rate <= static_rules' honest block rate MINUS 0.75 points on the same evaluation episodes, in BOTH
   directions (0.75 = safety margin: the sealed run has its own, new honest sample; Amendment 1);
2. LOFO selection score <= min(static_rules, single_policy) LOFO selection score - 3.0 (the sealed-family pass
   rule transplanted to LOFO);
3. full-TRAIN objective <= full-TRAIN objective of default `swarm_l1` + 3.0 (do not sacrifice the seen families).

The champion is chosen among the eligible candidates in two steps (Amendment 1):
(a) the ACCEPTABLE set = eligible candidates whose LOFO selection score is within 1.5 points of the lowest eligible
score (1.5 = the noise band of the proxy: D-022 measured 20.3-22.4 on TRAIN across reasonable settings, and two
GA seeds or one fold half differ by more than that);
(b) inside the acceptable set take the candidate with the HIGHEST mean LOFO honest block rate (the most sensitive
one: with the pre-registered objective a missed attack dollar costs 1.0 and a false alarm costs 0.5 per point, so
when the proxy cannot tell candidates apart the more sensitive one is the better insurance against a family the
proxy never showed); remaining ties (block rates within 0.25 points): lower LOFO score, then lower
worst-direction objective, then the simpler mechanism (C2 < C1 < C3).
The same two steps are used inside C1 (to pick the C1 `lam` that C3 stacks on) and inside C3. The strict
"lowest LOFO score" pick is also printed and stored in selection.json (`strict_min_pick`) for transparency. If NO candidate is eligible, the champion is the lowest selection score among the
candidates meeting condition 1, and the report states that the LOFO analogue of the pass rule failed.

After selection the champion's parameters are REFIT with the same procedure on all of TRAIN (both families, all 20
seeds, GA seed 0; honest quantiles from all TRAIN honest events) and frozen into `eval/r2/champion.json`. The
refit is not re-scored for selection; its full-TRAIN objective is reported, and if its honest block exceeds
static_rules' on the full-TRAIN episodes the failure is reported as is (no second refit, no substitution).

No candidate is added, no grid point is changed and no rule is edited after the full-grid run has started.
Any exploration of honest-only score distributions before scoring is not a candidate evaluation.

## Amendment 1 (disclosed; made before the full-grid run, after a smoke run)

A smoke run (`--quick`: one grid point per mechanism, GA pop 4 x 2 generations) was executed to test the pipeline.
It scored the five reference rows and C1 lam=20, C2 (0.5,2), C3 on that C2 point, nothing else. It showed (i) that
the seen-family LOFO rows of the swarm family are close to each other (default swarm 19.1, evolved g17-7 17.4, both
far below static_rules 54.5 and single_policy 57.2), so a rule that takes the strict minimum would be decided by
differences smaller than the proxy's noise, and (ii) that the honest-friction margin of the rule had no safety
margin for a new honest sample. Amendment: the 0.75 margin in condition 1 and the two-step choice in section 4
(acceptable set within 1.5 points, then most sensitive). The reason is risk preference under a proxy that
cannot rank defenders finely, not a result on a particular grid point. Nothing else changed; the grids are as above.
