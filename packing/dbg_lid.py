import pickle, numpy as np, mujoco
from pathlib import Path
import tasks as T, skills_case as C, run_full as RF
r = pickle.loads(Path("build/run_c/checkpoint_7.pkl").read_bytes())
import build_model as B
fresh, _ = B.build(); r.m.dof_frictionloss[:] = fresh.dof_frictionloss; r.m.dof_damping[:] = fresh.dof_damping
T.to_rest(r, T=1.0)
r.walk([(2.45, 0.62), (2.48, 1.62)], final_yaw=np.pi)
C.close_lid(r)
for _ in range(3):
    r.hold(0.5); print('lid', round(np.degrees(C.lid_angle(r)), 2))
m, d = r.m, r.d
lid = m.body('suitcase_lid').id
for i, c in enumerate(d.contact[:d.ncon]):
    b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
    if lid in (b1, b2):
        f = np.zeros(6); mujoco.mj_contactForce(m, d, i, f)
        print('  lid contact', m.geom(c.geom1).name, m.geom(c.geom2).name, round(f[0], 1))
r.debug_snapshot('lid')
