import pickle, numpy as np
from pathlib import Path
import tasks as T, skills_case as C, run_full as RF
r = pickle.loads(Path('build/run_a/checkpoint_4.pkl').read_bytes())
T.to_rest(r, T=1.0)
r.walk(RF.BED_TO_DOOR + [(3.90, 3.95), (6.10, 2.70), (6.12, 0.75)], final_yaw=0.0)
orig = r.move; k = [0]
def rep():
    m, d = r.m, r.d
    bp, bR = r.body_pose('bottle'); ax = bR[:, 2]
    pp, pR = r.hand_pose('r')
    n = -pR[:, 2]; fwd = -pR[:, 1]
    print('palm->bottle centre: along normal %.3f along fingers %.3f' % ((bp - pp) @ n, (bp - pp) @ fwd))
    for b in ['distph2_r', 'distph3_r', 'distph4_r', 'distph5_r', 'distal_thumb_r', '2proxph_r']:
        q = d.xpos[m.body(b).id]; v = q - bp; v -= ax * (v @ ax)
        print(f"   {b:15s} radial {np.linalg.norm(v)*1000:6.1f} mm  normal-coord {(q-pp)@n*1000:6.1f}  fwd-coord {(q-pp)@fwd*1000:6.1f}")
def move(*a, **kw):
    if k[0] == 2:
        pass
    orig(*a, **kw); k[0] += 1
    if k[0] in (3, 4):
        print('== after move', k[0]); rep()
    if k[0] == 4: raise SystemExit
r.move = move
C.wrap_pick(r, 'bottle', lift=0.12)
