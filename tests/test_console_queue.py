"""The ops console review queue (payguard/console/queue.py, scripts/build_console.py). No network, no .env."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from payguard.console.priority import priority_key
from payguard.console.queue import (ParityError, assemble_queue, check_parity, decided_by, record_episode,
                                    vote_share)
from payguard.protocol import episode_for
from payguard.robust.layer import R2Defender, calibrate, collect_stats
from payguard.sim.engine import SimConfig
from payguard.swarm.params import default_params

ROOT = Path(__file__).resolve().parents[1]
SMALL = SimConfig(days=5, n_workers=90, n_requesters=20, n_small_shops=3)


# ---------------------------------------------------------------- pure helpers ---------------------------
def test_vote_share_math():
    scores = {"velocity": 0.0, "graph": 1.0, "drift": 0.5, "policy": 0.0}
    weights = {"velocity": 0.25, "graph": 0.25, "drift": 0.25, "policy": 0.25}
    vs = vote_share(scores, weights)
    assert vs == {"velocity": 0.0, "graph": 0.667, "drift": 0.333, "policy": 0.0}
    assert abs(sum(vs.values()) - 1.0) < 0.01
    # a specialist's say is weight x score: a heavier weight takes a bigger share
    heavier = vote_share(scores, {**weights, "drift": 0.75})
    assert heavier["drift"] > vs["drift"]


def test_vote_share_is_zero_when_nobody_speaks_up():
    zero = {k: 0.0 for k in ("velocity", "graph", "drift", "policy")}
    assert vote_share(zero, {k: 0.25 for k in zero}) == zero


@pytest.mark.parametrize("swarm,layer,final,expected", [
    ("HOLD", "RELEASE", "HOLD", "swarm"),
    ("HOLD", "STEP_UP", "HOLD", "swarm"),
    ("STEP_UP", "RELEASE", "STEP_UP", "swarm"),
    ("RELEASE", "STEP_UP", "STEP_UP", "layer"),
    ("RELEASE", "HOLD", "HOLD", "layer"),
    ("STEP_UP", "HOLD", "HOLD", "layer"),
    ("HOLD", "HOLD", "HOLD", "both"),
    ("STEP_UP", "STEP_UP", "STEP_UP", "both"),
])
def test_decided_by(swarm, layer, final, expected):
    assert decided_by(swarm, layer, final) == expected


# ---------------------------------------------------------------- a small real episode --------------------
@pytest.fixture(scope="module")
def played():
    ep = episode_for(1000, ("ring", "structuring"), SMALL)  # a TRAIN seed: nothing sealed
    x, flags = collect_stats(ep)
    spec = calibrate(x[~flags], 0.01, 0.04)
    defender = R2Defender(default_params(), spec)
    result, at = record_episode(ep, defender)
    replay = {"episode": {"events": [{"id": i, "verdict": v.value} for i, v in result.verdicts.items()]}}
    return ep, result, at, replay, defender


def queue_of(played, **kw):
    ep, result, at, replay, d = played
    p = d.params
    return assemble_queue(ep, result, at, replay, thresholds={"step_up": p.step_up_threshold,
                                                              "hold": p.hold_threshold},
                          source={"split": "TRAIN", "seed": 1000, "defender": "swarm_r2", "replay": "x"},
                          generated_at="2026-01-01T00:00:00+00:00", **kw)


def test_queue_shape_and_counts(played):
    ep, result, at, replay, d = played
    q = queue_of(played)
    c = q["counts"]
    assert q["schema_version"] == 1 and q["simulated"] is True
    assert c["events"] == len(ep.events) == c["RELEASE"] + c["STEP_UP"] + c["HOLD"]
    assert c["cases"] == c["STEP_UP"] + c["HOLD"] == len(q["cases"]) == c["cases_attack"] + c["cases_honest"]
    assert q["source"]["parity_checked_events"] == len(ep.events)
    assert abs(c["attack_usd_total"] - c["attack_usd_stopped"] - c["attack_usd_leaked"]) < 0.02
    attack_usd = sum(e.amount for e in ep.events if ep.truth[e.id].is_attack)
    assert abs(c["attack_usd_total"] - attack_usd) < 0.02
    paid = sum(e.amount for e in ep.events if ep.truth[e.id].is_attack and result.outcomes[e.id].paid)
    assert abs(c["attack_usd_leaked"] - paid) < 0.02
    assert {s["key"] for s in q["specialists"]} == {"velocity", "graph", "drift", "policy", "layer"}
    assert q["thresholds"] == {"step_up": 0.3, "hold": 0.6}
    assert c["cases"] > 0 and c["cases_attack"] > 0


def test_cases_are_exactly_the_hold_and_step_up_events(played):
    ep, result, *_ = played
    q = queue_of(played)
    want = {i for i, v in result.verdicts.items() if v.value != "RELEASE"}
    assert {c["event_id"] for c in q["cases"]} == want
    assert all(c["verdict"] == result.verdicts[c["event_id"]].value for c in q["cases"])


def test_case_fields_follow_the_contract(played):
    q = queue_of(played)
    for c in q["cases"]:
        assert set(c["scores"]) == set(c["weights"]) == set(c["vote_share"]) == {"velocity", "graph", "drift", "policy"}
        assert set(c["features"]) == {"age_days", "n_prior", "window_n", "window_usd", "amount_ratio",
                                      "device_workers", "destination_workers", "requester_workers",
                                      "worker_requesters"}
        assert c["layer"]["verdict"] in ("RELEASE", "STEP_UP", "HOLD") and isinstance(c["layer"]["fired"], list)
        assert c["decided_by"] in ("swarm", "layer", "both")
        assert set(c["outcome"]) == {"paid", "paid_at", "delay_h"} and set(c["truth"]) == {"is_attack", "family"}
        # shares sum to 1 (or are all 0) and each value has at most 3 decimals
        assert abs(sum(c["vote_share"].values()) - 1.0) < 0.01 or sum(c["vote_share"].values()) == 0
        for v in (c["risk"], c["t"], *c["scores"].values(), *c["vote_share"].values()):
            assert v == round(v, 3)
        if c["outcome"]["paid"]:
            assert c["outcome"]["delay_h"] >= 0


def test_decided_by_agrees_with_the_recorded_layer_and_swarm(played):
    _, _, at, *_ = played
    q = queue_of(played)
    seen = set()
    for c in q["cases"]:
        rec = at[c["event_id"]]
        seen.add(c["decided_by"])
        if c["decided_by"] == "layer":
            assert rec["swarm_verdict"] != c["verdict"] and rec["layer"]["verdict"] == c["verdict"]
        elif c["decided_by"] == "swarm":
            assert rec["swarm_verdict"] == c["verdict"] != rec["layer"]["verdict"]
        else:
            assert rec["swarm_verdict"] == rec["layer"]["verdict"] == c["verdict"]
    assert "swarm" in seen or "both" in seen


def test_rank_order_follows_priority_key(played):
    q = queue_of(played)
    cases = q["cases"]
    assert [c["rank"] for c in cases] == list(range(1, len(cases) + 1))
    keys = [priority_key(c) for c in cases]
    assert keys == sorted(keys)
    # a different sort key reorders the queue and the ranks follow it
    other = queue_of(played, sort_key=lambda c: (c["amount"], c["event_id"]))["cases"]
    assert [c["amount"] for c in other] == sorted(c["amount"] for c in other)
    assert [c["rank"] for c in other] == list(range(1, len(other) + 1))


def test_priority_key_never_reads_truth_or_outcome(played):
    class Blind(dict):
        def __getitem__(self, k):
            assert k not in ("truth", "outcome"), f"priority_key read {k!r}"
            return super().__getitem__(k)

        def get(self, k, default=None):
            assert k not in ("truth", "outcome"), f"priority_key read {k!r}"
            return super().get(k, default)

    for c in queue_of(played)["cases"][:25]:
        stripped = {k: v for k, v in c.items() if k not in ("truth", "outcome")}
        assert priority_key(stripped) == priority_key(c) == priority_key(Blind(c))


# ---------------------------------------------------------------- parity ---------------------------------
def test_parity_ok_returns_the_number_of_events(played):
    _, result, _, replay, _ = played
    assert check_parity(result.verdicts, replay["episode"]["events"]) == len(result.verdicts)


def test_parity_refuses_a_doctored_replay(played, tmp_path):
    ep, result, at, replay, d = played
    doctored = copy.deepcopy(replay)
    first = doctored["episode"]["events"][0]
    first["verdict"] = "HOLD" if first["verdict"] != "HOLD" else "RELEASE"
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(doctored), encoding="utf-8")
    with pytest.raises(ParityError, match="nothing written") as exc:
        assemble_queue(ep, result, at, json.loads(path.read_text(encoding="utf-8")),
                       thresholds={"step_up": 0.3, "hold": 0.6}, source={})
    assert "1 verdicts differ" in str(exc.value) and f"#{first['id']}" in str(exc.value)


def test_parity_refuses_missing_and_extra_events(played):
    _, result, _, replay, _ = played
    events = replay["episode"]["events"]
    with pytest.raises(ParityError, match="event ids differ"):
        check_parity(result.verdicts, events[1:])
    with pytest.raises(ParityError, match="event ids differ"):
        check_parity(result.verdicts, [*events, {"id": 10**9, "verdict": "RELEASE"}])


def test_build_script_writes_nothing_on_a_mismatch(tmp_path, monkeypatch, capsys):
    """scripts.build_console returns non-zero and leaves no file when the parity check fails."""
    from scripts import build_console

    def refuse(*a, **kw):
        raise ParityError("parity with the published replay failed, nothing written: test")

    monkeypatch.setattr(build_console, "build", refuse)
    out = tmp_path / "queue.json"
    assert build_console.main(["--out", str(out)]) == 3
    assert not out.exists() and "REFUSED" in capsys.readouterr().err


def test_build_refuses_a_replay_of_another_episode(tmp_path):
    """build() reads the replay first: an episode other than the published one is refused before any play."""
    from scripts import build_console

    doc = json.loads(build_console.REPLAY_PATH.read_text(encoding="utf-8"))
    doc["episode"]["seed"] += 1
    bad = tmp_path / "replay.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit, match="nothing written"):
        build_console.build(bad)


# ---------------------------------------------------------------- the real build --------------------------
COMMITTED = ROOT / "eval" / "console" / "queue.json"


@pytest.mark.skipif(not COMMITTED.is_file(), reason="eval/console/queue.json has not been built")
def test_real_build_matches_the_committed_queue():
    """Re-play the published episode with the frozen swarm_r2 (about 2 s): same queue as the committed file."""
    from scripts import build_console

    built = json.loads(json.dumps(build_console.build()))
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    for d in (built, committed):
        d.pop("generated_at")
    assert built == committed
    c = committed["counts"]
    assert c["events"] == committed["source"]["parity_checked_events"]
    assert c["cases"] == c["HOLD"] + c["STEP_UP"] == len(committed["cases"])
    assert committed["source"]["replay"] == "eval/replay_live.json"
