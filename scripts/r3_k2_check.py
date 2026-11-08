"""K2 stability check (PREREG_R3 M2), TRAIN seeds only. Usage: python -m scripts.r3_k2_check > eval/r3/k2_check.json. Criteria fixed BEFORE the first run (director, 2026-11-08):
  (1) effective sample size (1/sum w^2) of the posterior particles >= 20 for every family with mass;
  (2) both TRAIN families keep 20-80% of the posterior mass (TRAIN episodes contain both);
  (3) posterior means across 3 reference tables (pool_seed 0,1,2; fit ignores its rng, so rng spread is trivially 0) differ by < 0.05 of each field's range (max over fields).
Run on (a) the 20 TRAIN protocol observations and (b) the 10 TRAIN_A observations alone."""
import json
from collections import defaultdict
from dataclasses import fields

import numpy as np

from payguard.dream import observe
from payguard.dream.worldmodel import AbcWorldModel
from payguard.protocol import TRAIN_FAMILIES, episode_for
from payguard.sim.attacks import BOUNDS


def summarize(model):
    parts = model.posterior_particles()
    by = defaultdict(list)
    for fam, p, w in parts:
        by[fam].append((p, w))
    tot = sum(w for _, _, w in parts)
    out = {}
    for fam, lst in by.items():
        w = np.array([x[1] for x in lst]) / sum(x[1] for x in lst)
        mean = {f.name: float(sum(wi * getattr(p, f.name) for (p, _), wi in zip(lst, w))) for f in fields(lst[0][0])}
        out[fam] = {"mass": float(sum(x[1] for x in lst) / tot), "ess": float(1 / np.sum(w ** 2)), "mean": mean}
    return out


def main():
    sets = {"TRAIN_1000-1019": range(1000, 1020), "TRAIN_A_1000-1009": range(1000, 1010)}
    report, ok = {}, True
    for name, seeds in sets.items():
        obs = [observe(episode_for(s, TRAIN_FAMILIES)) for s in seeds]
        runs = [summarize(AbcWorldModel(pool_seed=k).fit(obs, np.random.default_rng(k))) for k in (0, 1, 2)]
        r = {"per_rng": [{f: {"mass": round(v["mass"], 3), "ess": round(v["ess"], 1)} for f, v in s.items()} for s in runs]}
        c1 = all(v["ess"] >= 20 for s in runs for v in s.values())
        c2 = all(0.2 <= s.get(f, {"mass": 0})["mass"] <= 0.8 for s in runs for f in TRAIN_FAMILIES)
        spread = 0.0
        for fam in TRAIN_FAMILIES:
            for fld, (lo, hi) in BOUNDS[fam].items():
                vals = [s[fam]["mean"][fld] for s in runs if fam in s]
                if len(vals) == 3 and hi > lo:
                    spread = max(spread, (max(vals) - min(vals)) / (hi - lo))
        c3 = spread < 0.05
        r.update({"c1_ess>=20": c1, "c2_mass_0.2-0.8": c2, "c3_max_mean_spread": round(spread, 4), "c3_ok": c3})
        ok &= c1 and c2 and c3
        report[name] = r
    report["K2"] = "PASS" if ok else "FAIL"
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
