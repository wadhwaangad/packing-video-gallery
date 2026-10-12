import pickle, numpy as np, mujoco
from pathlib import Path
import tasks as T, skills_case as C, run_full as RF
r = pickle.loads(Path("build/run_c/checkpoint_7.pkl").read_bytes())
T.to_rest(r, T=1.0)
r.walk([(2.45, 0.62), (2.48, 1.62)], final_yaw=np.pi)
C.close_lid(r)
T.to_rest(r, T=1.0)
r.walk([(2.45, 0.62), RF.SUIT], final_yaw=np.pi / 2)
orig = r.step; cnt = [0]
def step(n=1):
    for _ in range(n):
        orig(1); cnt[0] += 1
        if cnt[0] % 400 == 0:
            m, d = r.m, r.d; lid = m.body('suitcase_lid').id; cs = []
            for i, c in enumerate(d.contact[:d.ncon]):
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if lid in (b1, b2) and (b1 in r.hand_bodies['r'] + r.hand_bodies['l'] or b2 in r.hand_bodies['r'] + r.hand_bodies['l']):
                    f = np.zeros(6); mujoco.mj_contactForce(m, d, i, f); cs.append(f"{m.geom(c.geom1).name[:14]}/{m.geom(c.geom2).name[:14]}:{f[0]:.0f}")
            print(f"{d.time:.1f} {r.label[:26]:26s} lid {np.degrees(C.lid_angle(r)):5.1f} zip {r.zip_progress:.2f}", cs[:4])
r.step = step
try:
    C.press_and_zip(r)
except Exception as e:
    print(e)
