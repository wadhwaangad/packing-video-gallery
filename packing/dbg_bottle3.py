import pickle, numpy as np, mujoco
from pathlib import Path
import tasks as T, skills_case as C, run_full as RF
r = pickle.loads(Path('build/run_a/checkpoint_4.pkl').read_bytes())
T.to_rest(r, T=1.0)
r.walk(RF.BED_TO_DOOR + [(3.90, 3.95), (6.10, 2.70), (6.12, 0.75)], final_yaw=0.0)
orig = r.move; k = [0]
def move(*a, **kw):
    orig(*a, **kw); k[0] += 1
    if k[0] in (2, 3):
        m, d = r.m, r.d
        bp, bR = r.body_pose('bottle'); ax = bR[:, 2]
        print('--- after move', k[0], 'palm', r.hand_pose('r')[0].round(3), 'target', r.hand['r'][0].round(3))
        print('    wrist q', round(float(d.qpos[m.joint('flexion_r').qposadr[0]]), 3), 'cmd', r.finger['r']['flexion_r'])
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in r.hand_bodies['r']:
                v = d.geom_xpos[g] - bp; v -= ax * (v @ ax)
                dist = np.linalg.norm(v) - 0.036 - m.geom_size[g][0]
                if dist < 0.01: print('   ', m.geom(g).name, 'radial clearance %.1f mm' % (dist * 1000), 'height above bottle centre %.3f' % ((d.geom_xpos[g] - bp) @ ax))
    if k[0] == 3: raise SystemExit
r.move = move
C.wrap_pick(r, 'bottle', lift=0.12)
