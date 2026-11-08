# Round-3 sealed family "h3" (TEST_HELDOUT3, seeds 5000-5019)

## Mechanism
A synthetic-identity account farm. An operator opens a stream of fresh worker accounts over a few days. Each
account waits a short seasoning period (more than one day, so "brand new account asking for a lot" rules do not
fire), then collects a few flat, moderate payouts approved by a small pool of ordinary-looking requesters, and
is abandoned. Every payout of every farm account is a loss (`is_attack=True`); there is no honest trust-building
phase, so no label leaks early. Accounts are disposable, so per-account history never accumulates evidence.

The reuse of infrastructure is the only structure: devices are shared by `accounts_per_device` consecutive
accounts and payout destinations by `accounts_per_dest` consecutive accounts, with different strides, so the
device groups and destination groups interlock into one connected component. Account creation times are
clustered in a window.

## Parameters (frozen dataclass `H3Params`, bounds in `BOUNDS_H3`, ints in `INT_FIELDS_H3`)
| field | bounds | meaning |
|---|---|---|
| n_accounts (int) | 4-20 | accounts in the farm |
| accounts_per_device (int) | 2-5 | accounts per shared device |
| accounts_per_dest (int) | 2-6 | accounts per shared destination |
| spawn_span_days | 2-9 | window over which accounts are created |
| start_day | 0-4 | first account creation |
| age_at_first_hours | 26-200 | account age at first payout |
| payouts_per_account (int) | 2-6 | payouts before abandonment |
| gap_hours | 1-20 | spacing of an account's payouts |
| amount_median | 40-200 | median payout USD |
| amount_sigma | 0.1-0.4 | lognormal spread (small: flat amounts, no drift vs own median) |
| n_requesters (int) | 1-4 | size of the requester pool the farm draws from (honest popular requesters) |

Amounts are capped at REVIEW_THRESHOLD_USD - 50. STEPUP_PASS_PROB = 0.3: operators hold synthetic identities
and can answer scripted challenges, but live verification of a made-up person fails more often than not
(between structuring 0.2 and sleeper 0.35); chosen by reasoning, not tuned.

## Why realistic
Fake-account farms exploiting instant payouts are the standard marketplace payout abuse: cheap identities,
shared hardware and cash-out rails, burn after cash-out.

## Why hard
Per account: age over a day, amounts moderate and flat, ordinary requesters, only 2-6 payouts. Static per-account
rules (big payout, young+large, drift vs own median) rarely fire. Honest hard negatives (new workers, shared
device pairs, shared destination pairs, small shops) mean raw "young" or "shared device" flags carry false
positives; delayed labels arrive after 72 h and each device or destination is used by only a few accounts.

## What a perfect oracle sees
The set of farm accounts: a connected component through shared devices and destinations (sizes above honest
pairs), creation times bunched together, requesters drawn from a tiny pool, near-identical flat amounts, and no
payouts beyond the short post-seasoning burst. Linking accounts by device/destination and group-level rate
catches the later accounts of a component before they cash out.

## DEV measurement (seeds 9100-9119, 2-4 campaigns, params uniform in bounds)
StaticRules leakage_pct 96.1 (honest_block_rate 9.82); AlwaysRelease leakage_pct 100.0.
