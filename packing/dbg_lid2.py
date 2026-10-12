import pickle, numpy as np, mujoco
from pathlib import Path
import tasks as T, skills_case as C, run_full as RF, build_model as B
r = pickle.loads(Path("build/run_c/checkpoint_7.pkl").read_bytes())
fresh, _ = B.build(); r.m.dof_frictionloss[:] = fresh.dof_frictionloss; r.m.dof_damping[:] = fresh.dof_damping
T.to_rest(r, T=1.0)
r.walk([(2.45, 0.62), (2.48, 1.62)], final_yaw=np.pi)
orig = r.move
def move(*a, **kw):
    if r.label == "Let the lid fall closed":
        m, d = r.m, r.d
        pl, Rl = r.body_pose("suitcase_lid")
        loc = lambda x: (Rl.T @ (x - pl)).round(3)
        print("lid deg", round(np.degrees(C.lid_angle(r)), 1), "palm(lid)", loc(r.hand_pose("r")[0]),
              "tip2(lid)", loc(d.xpos[m.body("distph2_r").id]), "tip3", loc(d.xpos[m.body("distph3_r").id]),
              "palm normal(lid)", (Rl.T @ -r.hand_pose("r")[1][:, 2]).round(2), "fingers(lid)", (Rl.T @ -r.hand_pose("r")[1][:, 1]).round(2))
    orig(*a, **kw)
r.move = move
C.close_lid(r)
