"""The pre-registered scalar fitness (director's decision; do not change).

    objective = leakage_pct + 0.5 * honest_block_rate + 1.0 * honest_unpaid_rate        (lower is better)

Leakage is the share of attack USD actually paid; the honest terms price the friction put on honest
workers (any non-instant payout, and payouts that never arrive). The scalar hides trade-offs, so it must
never be reported alone: always print the three components next to it (`describe`).
"""
from __future__ import annotations

from ..sim.metrics import Metrics

W_BLOCK = 0.5
W_UNPAID = 1.0


def objective(m: Metrics) -> float:
    return m.leakage_pct + W_BLOCK * m.honest_block_rate + W_UNPAID * m.honest_unpaid_rate


def components(m: Metrics) -> dict[str, float]:
    return {"leakage_pct": m.leakage_pct, "honest_block_rate": m.honest_block_rate,
            "honest_unpaid_rate": m.honest_unpaid_rate}


def describe(m: Metrics) -> str:
    return (f"obj={objective(m):6.2f} = leak {m.leakage_pct:5.1f} + {W_BLOCK}*block {m.honest_block_rate:4.1f}"
            f" + {W_UNPAID}*unpaid {m.honest_unpaid_rate:4.1f}")
