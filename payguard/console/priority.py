"""Default order of the review queue in the ops console (docs/CONSOLE_CONTRACT.md, section A).

The queue builder sorts every case with `priority_key` and the console lists them in that order. A reviewer works
from the top, so this order decides which held payouts a human looks at first when there is no time for all.
Director-owned: build tracks import it and never edit it.
"""
from __future__ import annotations


def priority_key(case: dict) -> tuple:
    """Sort key for one queue case; smaller sorts first.

    Use decision-time fields only: verdict, risk, amount, scores, weights, vote_share, layer, features.
    Never `truth` or `outcome`: a reviewer does not know them when the case arrives (a test enforces this).
    """
    # 1. HOLD before STEP_UP: a step-up already resolves through extra verification, a hold waits on a human.
    # 2. Within a group, expected dollars at risk (risk x amount), largest first: the reviewer's first minutes go
    #    where a wrong release would cost the most.
    # 3. event_id last, so ties always sort the same way.
    group = 0 if case["verdict"] == "HOLD" else 1
    at_risk = round(case["risk"] * case["amount"], 3)
    return (group, -at_risk, case["event_id"])
