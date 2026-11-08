# Round-3 selection rule (written 2026-11-08, before any dreamer code or score exists)

Pre-registration: `docs/PREREG_R3.md` (project docs folder), approved by the user (M0). This file fixes HOW the dreamer's settings are chosen, on TRAIN only. It may be shrunk (never grown or changed) before the first scoring run, and any shrink is logged in the decision log before that run.

## 1. Candidate grid (fixed)

All candidates use `DreamConfig` defaults except the two columns below: `n_dreams` 32, `tau` 1.0, `sigma` 1.0, `ema_beta` 0.5, `n_bins` 4.

| id | shares (posterior / uncertain / recombined) | lambda0 |
|---|---|---|
| c1 | 0.50 / 0.25 / 0.25 | 0.3 |
| c2 | 0.50 / 0.25 / 0.25 | 0.6 |
| c3 | 0.34 / 0.33 / 0.33 | 0.3 |
| c4 | 0.34 / 0.33 / 0.33 | 0.6 |

Evolution settings: the D-028 full `EvolveConfig` (same pop, gens, elites, pools, arena episodes, budget guard) with evolution seed 0, unless the decision log records a smaller budget before scoring (applied to every arm alike).

## 2. Leave-one-family-out score (TRAIN only)

For each candidate `c` and each TRAIN family `f` in {ring, structuring}:
1. Evolve with dreams on, training families = the OTHER TRAIN family only (fitness TRAIN_A, gate TRAIN_B, as today).
2. Score the final champion on the 10 TRAIN_B seeds built with family `f` ONLY (`payguard.protocol.episode_for(seed, (f,))`): objective as in D-019.

`LOFO(c)` = mean of the two objectives. The winner is the lowest `LOFO(c)`; ties go to the lower id.

## 3. Dev check K1 (before the sealed set is touched)

Re-run the winner and its `dreamer_nowm` twin (identical except `use_wm=False`) through section 2 with evolution seeds 0 and 1. K1 passes only if the mean LOFO objective of the winner is at least **1.0 lower** than the twin's. If K1 fails: stop, publish the negative finding, do not run TEST_HELDOUT3.

## 4. Freeze

The frozen `dreamer_full` = the winner's `DreamConfig` + the champion of a full-TRAIN evolution (both TRAIN families, seed 0). Ablation arms (`nowm`, `noskeptic`, `nocalib`) are evolved the same way from the same config with their one switch flipped. All champions, the config and the hashes of `payguard/dream/*` are written to `eval/r3/defender_freeze.json` and logged before any TEST_HELDOUT3 episode is built.

## 5. Not allowed

No setting outside the grid; no change to this file after the first scoring run; no use of TEST_SEEN, TEST_HELDOUT, TEST_HELDOUT2 or TEST_HELDOUT3 in any choice above.
