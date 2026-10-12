import pickle, numpy as np
from pathlib import Path
import tasks as T, skills_case as C, run_full as RF
r = pickle.loads(Path('build/run_a/checkpoint_4.pkl').read_bytes())
T.to_rest(r, T=1.0)
r.walk(RF.BED_TO_DOOR + [(3.90, 3.95), (6.10, 2.70), (6.12, 0.75)], final_yaw=0.0)
C.wrap_pick(r, 'bottle', lift=0.12)
r.debug_snapshot('bottle_grip')
m = r.m
for c in r.d.contact[: r.d.ncon]:
    n1, n2 = m.geom(c.geom1).name, m.geom(c.geom2).name
    if 'bottle' in n1 or 'bottle' in n2: print(n1, n2)
