import pickle, numpy as np
from pathlib import Path
import tasks as T, skills_case as C, run_full as RF
from scipy.spatial.transform import Rotation as R
r = pickle.loads(Path('build/run_a/checkpoint_4.pkl').read_bytes())
T.to_rest(r, T=1.0)
r.walk(RF.BED_TO_DOOR + [(4.00, 3.75), (6.10, 2.70), (6.12, 0.75)], final_yaw=0.0)
orig_move = r.move
k = [0]
def tilt():
    q = r.d.xquat[r.m.body('bottle').id]; z = R.from_quat([q[1], q[2], q[3], q[0]]).apply([0, 0, 1])
    return round(float(np.degrees(np.arccos(np.clip(z[2], -1, 1)))), 1)
def move(*a, **kw):
    orig_move(*a, **kw)
    k[0] += 1
    print('move', k[0], r.label, 'tilt', tilt(), 'palm', r.hand_pose('r')[0].round(3))
    r.debug_snapshot(f"bottle_{k[0]}")
r.move = move
print('start tilt', tilt(), 'bottle', r.body_pose('bottle')[0].round(3), 'root', {k_: round(v, 3) for k_, v in r.root.items()})
try:
    C.wrap_pick(r, 'bottle')
except Exception as e:
    print(e)
