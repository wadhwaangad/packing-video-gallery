import pickle, numpy as np, mujoco
from pathlib import Path
import tasks as T, skills_case as C, run_full as RF
r = pickle.loads(Path("build/run_e/checkpoint_8.pkl").read_bytes())
T.to_rest(r, T=1.0)
r.walk([(2.45, 0.62), (2.46, 1.21)], final_yaw=np.pi)
orig = r.move; k = [0]
def move(*a, **kw):
    orig(*a, **kw); k[0] += 1
    sp, sR = r.suitcase_frame()
    c = r.contacts_between(r.hand_bodies["r"], r.suit_ids)
    print(k[0], r.label, "palm", r.hand_pose("r")[0].round(3), "tgt", r.hand["r"][0].round(3), "case z", round(sp[2], 3), "hand-case contacts", c[0], round(c[1], 1))
    pass
    if k[0] == 6:
        from scipy.spatial.transform import Rotation as RR
        sp_, sR_ = r.suitcase_frame(); print("   up-tilt", np.degrees(np.arccos(np.clip(sR_[2,1],-1,1))).round(1), "case", sp_.round(3))
r.move = move
try:
    C.lift_from_bed_foot(r)
except Exception as e:
    print("ERR", e)
r.hold(2.0)
sp, sR = r.suitcase_frame()
print("FINAL case", sp.round(3), "up-tilt", np.degrees(np.arccos(np.clip(sR[2, 1], -1, 1))).round(1), "hand contacts", r.contacts_between(r.hand_bodies["r"], r.suit_ids)[0])
m, d = r.m, r.d
ids = set(r.suit_ids)
for i, c in enumerate(d.contact[:d.ncon]):
    b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
    if (b1 in ids) != (b2 in ids):
        f = np.zeros(6); mujoco.mj_contactForce(m, d, i, f)
        print("  case contact", m.geom(c.geom1).name, m.geom(c.geom2).name, round(f[0], 1))
print("case axes: up(base y)", sR[:, 1].round(2), "depth(base z)", sR[:, 2].round(2))
