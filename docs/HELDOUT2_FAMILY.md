# TEST_HELDOUT2 attack family "h2" (SEALED design note)

Sealed: do not read until the round-2 defender is frozen. Code: `payguard/heldout2/`. Seeds 4000-4019, never built
before the freeze. Difficulty was NOT tuned against any defender; no defender was run on this family on any seed.
Only structural sanity checks (event counts, label counts, amount ranges, timing, feature distributions of the
generator output versus honest events) were run, on dev seeds 9000-9004.

## 1. Mechanism: account takeover with a shared cash-out channel

A credential-stuffing crew takes over established, aged gig-worker accounts in one wave and redirects their
payouts to attacker-controlled payout destinations ("hubs") using attacker-controlled devices.

* `n_victims` aged accounts (age 60-600 days) have an ordinary history on their OWN device and destination:
  normal amounts, normal rate, two home requesters (70/30 split). This is the real owner's activity.
* At a takeover time (inside a wave starting at `start_day`, lasting `wave_span_hours`) the crew drains the
  account. The real owner stops generating payouts. The drain payouts go to one of `n_hubs` shared destinations
  (victim i -> hub i mod n_hubs) from one of `n_devices` attacker devices (fixed per victim), and are triggered
  against the victim's own pending earnings, so the requester is one of the victim's usual requesters.
* `drain_count` payouts per victim, `drain_gap_hours` apart, each `victim_median * drain_mult * lognormal(0.25)`
  and capped at `REVIEW_THRESHOLD_USD - cap_margin` (attackers know the threshold and stay under it with a margin
  that is itself a parameter).

### How it differs from the existing families

| family | core trick | this family |
|---|---|---|
| ring | clean accounts, collective signal in requester overlap, each account has its own device and destination | the footprint is in the payout channel (destination and device), requesters look completely normal; accounts are hijacked, not fabricated |
| structuring | repeated amounts just under the threshold, bursts, young accounts | amounts are ordinary (roughly 1-4x the victim's own median), accounts are old, no amount clustering |
| sleeper | one fresh account builds honest-looking history, then cashes out at a large multiple of its own baseline with spacing | no multiple-of-baseline jump and no waiting by the attacker: the history is somebody else's (the real owner) and the change is WHERE the money goes, not how much |

## 2. Realism

Account takeover (ATO) against payout platforms is one of the most common real losses: stolen or stuffed
credentials, payout destination swapped, earnings withdrawn. Crews operate from a handful of devices and a few
cash-out accounts (mules or burner wallets) shared across many victims, and they work in waves because stuffed
credentials go stale. They drain pending earnings, which looks like the owner's normal payout size or a small multiple of it.
Honest look-alikes already exist in the sim: a few honest workers share a device or destination with one other
worker (family members), small shops and power workers produce large honest amounts.

## 3. Labels

* The owner's own payouts before the takeover are honest: real owner, real work, no loss, no fraud label.
* Only the drain payouts are `is_attack=True` (a genuine loss: the owner disputes, the platform eats it).
  `family="h2"`, `campaign_id` = the campaign index.
* No fraud label can reach a defender about a victim before its drain starts (same principle as the sleeper
  trust-building phase). Honest events never carry the attack family.
* Resolution under STEP_UP: the party answering the challenge is the attacker, who holds the password but not the
  owner's second factor; some pass via phished OTP or SIM swap. `STEPUP_PASS["h2"] = 0.25` is set by reasoning
  about the actor (between structuring 0.2 and sleeper 0.35), and was not tuned.

## 4. Footprint (reasoning, and a distribution sanity check on dev seeds only)

Detectable in principle through features that `features.py` exposes:

* `destination_workers` and `device_workers`: a hub or an attacker device accumulates many distinct workers
  (the n-th victim sees roughly n-1 earlier workers on the same destination/device). Honest values are 1, or 2 for
  the legitimate family-sharing pairs, so counts of 3 and up on an aged account are far outside the honest
  distribution. The footprint grows as the wave progresses, and the first victim(s) of each hub are
  necessarily invisible to these two features (cold start), which is what makes it a real test of memory and
  graph reasoning rather than a trivially separable one.
* `age_days` is old and `n_prior` modest, so the aged-account prior makes the victims look MORE trustworthy to a
  naive "new account" rule. `amount` and `amount_ratio` are inside the honest range (power workers, bursty
  workers), so single-event amount rules do not help.
* Beyond features: the worker's destination/device on drain events differs from the one in its own history
  (derivable from the event log), and the owner's activity stops after the takeover.

Dev-seed sanity check (9000-9004, structural only): about two thirds of attack events show
`destination_workers >= 3`, versus none of the honest events. Remaining attack events are the early victims of each hub.

## 5. Parameters and bounds (uniform; integers marked)

| field | bounds | meaning |
|---|---|---|
| n_victims (int) | 3-14 | hijacked accounts |
| n_hubs (int) | 1-3 | distinct cash-out destinations |
| n_devices (int) | 1-3 | attacker devices |
| victim_rate_per_day | 0.3-1.2 | owner's pre-takeover payout rate |
| victim_median | 30-120 | owner's median payout, USD |
| account_age_days | 60-600 | victim account age |
| start_day | 2-8 | start of the takeover wave |
| wave_span_hours | 6-72 | length of the wave |
| drain_count (int) | 1-4 | drain payouts per victim |
| drain_gap_hours | 1-30 | spacing between drains on one account |
| drain_mult | 1-4 | drain amount relative to the owner's median |
| cap_margin | 20-150 | drains stay this far below `REVIEW_THRESHOLD_USD` |

`sample_params` and `mutate` are the round-1 functions, operating on the registered `PARAMS`/`BOUNDS`/`INT_FIELDS`.

## 6. Implementation and freeze

* `payguard/heldout2/__init__.py` registers the generator, params, bounds, int fields, `engine.STEPUP_PASS["h2"]`
  and `metrics.FAMILIES` at import time (runtime registration, idempotent). No file under `payguard/sim/` or any
  other round-1 frozen file is edited.
* `eval/FROZEN_R2.json` holds the sha256 (CRLF to LF) of every file in `payguard/heldout2/`;
  `python -m scripts.freeze --round 2` verifies it.
* Victim ids are fresh worker ids from `IdAlloc` (they are attacker-campaign entities in the simulator, not members
  of the 300-worker honest population), which is the same way ring and sleeper allocate theirs.
