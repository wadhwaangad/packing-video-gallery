"""Retry a failed action and list, per stage, what the hands touch (development aid).
Usage: python dbg_contacts.py RUN_DIR"""
import pickle, sys
from pathlib import Path
import numpy as np, mujoco
import executor as X, build_model as B

blob = pickle.loads((Path(sys.argv[1]) / "failure.pkl").read_bytes())
ex, act = blob["ex"], blob["action"]
ex.rewind(len(ex.records)); r = ex.r
fresh, _ = B.build()
for g in range(fresh.ngeom):
    nm = fresh.geom(g).name
    if nm:
        gi = r.m.geom(nm).id
        for attr in ("geom_pos", "geom_size", "geom_friction", "geom_priority", "geom_solref", "geom_solimp", "geom_solmix"):
            getattr(r.m, attr)[gi] = getattr(fresh, attr)[g]
mujoco.mj_forward(r.m, r.d)
ex.__class__ = X.Executor
hb = set(r.hand_bodies["r"]) | set(r.hand_bodies["l"])
seen = {}
orig = r.step
def step(n=1):
    for _ in range(n):
        orig(1)
        if r.step_count % 25 == 0:
            for c in r.d.contact[: r.d.ncon]:
                b1, b2 = r.m.geom_bodyid[c.geom1], r.m.geom_bodyid[c.geom2]
                if (b1 in hb) != (b2 in hb):
                    other = c.geom2 if b1 in hb else c.geom1
                    seen.setdefault(r.label, set()).add(r.m.geom(other).name)
r.step = step
try:
    ex.execute(act, blob["label"], blob["event"]); print("OK")
except Exception as e:
    print("FAILED", e)
for k, v in seen.items():
    print(f"{k:40s} {sorted(v)}")
