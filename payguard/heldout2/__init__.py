"""Round-2 sealed attack family, registered into the round-1 simulator AT IMPORT TIME.

Nothing under payguard/sim/ is edited (those files are frozen in eval/FROZEN.json). Instead, importing this
package adds the family to the live dicts of `payguard.sim.attacks` (GENERATORS, PARAMS, BOUNDS, INT_FIELDS),
to `payguard.sim.engine.STEPUP_PASS` (needed so a STEP_UP verdict on this family's payouts resolves) and to
`payguard.sim.metrics.FAMILIES` (so per-family leakage is reported). Registration is idempotent. Processes that
never import this package see the round-1 simulator unchanged. Frozen by eval/FROZEN_R2.json.
"""
from payguard.sim import attacks as _attacks
from payguard.sim import engine as _engine
from payguard.sim import metrics as _metrics

from .family import BOUNDS_H2, INT_FIELDS_H2, NAME, STEPUP_PASS_PROB, H2Params, gen_h2


def register() -> None:
    _attacks.GENERATORS[NAME] = gen_h2
    _attacks.PARAMS[NAME] = H2Params
    _attacks.BOUNDS[NAME] = BOUNDS_H2
    _attacks.INT_FIELDS.update(INT_FIELDS_H2)
    _engine.STEPUP_PASS[NAME] = STEPUP_PASS_PROB
    if NAME not in _metrics.FAMILIES:
        _metrics.FAMILIES = (*_metrics.FAMILIES, NAME)


register()

from .protocol import HELDOUT2_FAMILIES, TEST_HELDOUT2_SEEDS, episode_for, episodes  # noqa: E402

__all__ = ["register", "NAME", "H2Params", "gen_h2", "BOUNDS_H2", "INT_FIELDS_H2",
           "HELDOUT2_FAMILIES", "TEST_HELDOUT2_SEEDS", "episode_for", "episodes"]
