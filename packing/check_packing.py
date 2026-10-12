"""Report each item's final pose in suitcase-base coordinates vs its planned slot."""
import json, sys, pickle, numpy as np
from pathlib import Path
import scene as S
r = pickle.loads(Path(sys.argv[1]).read_bytes())
sp, sR = r.suitcase_frame()
W, L = S.SUITCASE["outer"]; D, DL, t = S.SUITCASE["base_depth"], S.SUITCASE["lid_depth"], S.SUITCASE["shell"]
for it in S.ITEMS:
    p, R = r.body_pose(it.name)
    loc = sR.T @ (p - sp)
    slot = S.PACK_SLOTS.get("tshirts" if it.name.startswith("tshirt") else it.name)
    err = np.linalg.norm(loc[:2] - np.array(slot["pos"][:2])) if slot else float("nan")
    inside = abs(loc[0]) < W / 2 - t and abs(loc[1]) < L / 2 - t and t < loc[2] < D + DL - t
    print(f"{it.name:13s} local {loc.round(3)}  xy-err {err*1000:6.1f} mm  inside {inside}  speed {r.item_speed(it.name)[0]:.1e}")
print("suitcase", sp.round(3), "lid deg", np.degrees(r.d.qpos[r.m.joint('lid_hinge').qposadr[0]]).round(1))
