"""EmaCalibrator (P3): protocol, math, switches."""
import math

import pytest

from payguard.dream.calibrate import EmaCalibrator
from payguard.dream.contract import Calibrator


def test_protocol_and_start():
    c = EmaCalibrator(0.3, 1.0, 0.5)
    assert isinstance(c, Calibrator)
    assert c.gap_ema == 0.0 and c.history == [] and c.lam() == 0.3


def test_math_and_monotone():
    c = EmaCalibrator(0.6, 2.0, 0.5)
    c.update(3.0, 1.0)  # gap 2 -> ema 1
    assert math.isclose(c.gap_ema, 1.0) and math.isclose(c.lam(), 0.6 / 1.5)
    c.update(-1.0, 3.0)  # gap 4 -> 0.5*4 + 0.5*1 = 2.5
    assert math.isclose(c.gap_ema, 2.5) and math.isclose(c.lam(), 0.6 / (1 + 2.5 / 2))
    assert [h["gap"] for h in c.history] == [2.0, 4.0]
    cs = [EmaCalibrator(0.3, 1.0, 0.5) for _ in range(3)]
    for c1, gap in zip(cs, (0.0, 1.0, 5.0)):
        c1.update(gap, 0.0)
    assert cs[0].lam() > cs[1].lam() > cs[2].lam() > 0
    assert math.isclose(cs[0].lam(), 0.3)


def test_lam_recovers_when_predictions_improve():
    c = EmaCalibrator(0.3, 1.0, 0.5)
    c.update(5.0, 0.0)
    low = c.lam()
    for _ in range(10):
        c.update(1.0, 1.0)
    assert low < c.lam() <= 0.3


def test_none_does_not_update():
    c = EmaCalibrator(0.3, 1.0, 0.5)
    c.update(2.0, 0.0)
    before = (c.gap_ema, list(c.history), c.lam())
    c.update(None, None)
    c.update(1.0, None)
    assert (c.gap_ema, c.history, c.lam()) == before


def test_disabled_stays_at_lambda0_but_logs():
    c = EmaCalibrator(0.6, 1.0, 0.5, enabled=False)
    c.update(10.0, 0.0)
    assert c.lam() == 0.6 and len(c.history) == 1 and c.gap_ema > 0


def test_validation():
    for bad in ({"lambda0": 0.0}, {"lambda0": 1.5}, {"sigma": 0.0}, {"ema_beta": 0.0}):
        with pytest.raises(ValueError):
            EmaCalibrator(**(dict(lambda0=0.3, sigma=1.0, ema_beta=0.5) | bad))
