"""Retry a failed action from its checkpoint, then list what the lid touches (development aid).
Usage: python dbg_lid_contacts.py RUN_DIR"""
import pickle, sys
from pathlib import Path
import numpy as np, mujoco
import executor as X, scene as S

blob = pickle.loads((Path(sys.argv[1]) / "failure.pkl").read_bytes())
ex, act = blob["ex"], blob["action"]
ex.rewind(len(ex.records)); r = ex.r; ex.__class__ = X.Executor
try:
    ex.execute(act); print("OK")
except Exception as e:
    print("FAILED", e)
m, d = r.m, r.d
lid = m.body("suitcase_lid").id
f = np.zeros(6)
for i, c in enumerate(d.contact[: d.ncon]):
    b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
    if lid in (b1, b2) and "mattress" not in m.geom(c.geom1).name + m.geom(c.geom2).name:
        mujoco.mj_contactForce(m, d, i, f)
        print("lid contact", m.geom(c.geom1).name, m.geom(c.geom2).name, round(f[0], 1), "N")
sp, sR = r.suitcase_frame()
for n, loc in ex.sim.state.loc.items():
    if loc.startswith("case:"):
        p, R = r.body_pose(n); l = sR.T @ (p - sp)
        print(f"{n:14s} {loc:22s} {l.round(3)} top {l[2] + S.item(n).size[2] / 2:.3f}")
