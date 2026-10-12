import pickle, numpy as np, mujoco
from pathlib import Path
import tasks as T, skills_case as C, run_full as RF, build_model as B
r = pickle.loads(Path('build/run_a/checkpoint_4.pkl').read_bytes())
fresh, _ = B.build()
r.m.opt.noslip_iterations = fresh.opt.noslip_iterations
for attr in ["actuator_forcerange", "actuator_gainprm", "actuator_biasprm", "jnt_range", "jnt_solref"]:
    getattr(r.m, attr)[:] = getattr(fresh, attr)
T.to_rest(r, T=1.0)
r.walk(RF.BED_TO_DOOR + [(3.90, 3.95), (6.10, 2.70), (6.12, 0.75)], final_yaw=0.0)
C.wrap_pick(r, "bottle", side="r", lift=0.12)
r.move(1.2, hands=C.carry_one(r, "bottle", "r", height=1.10, forward=0.30, lateral=0.06), root=dict(h=0.95))
orig = r.step; cnt = [0]
def step(n=1):
    for _ in range(n):
        orig(1); cnt[0] += 1
        if cnt[0] % 20 == 0 and r.body_pose('bottle')[0][2] > 0.9 and cnt[0] > 1000:
            m, d = r.m, r.d
            cs = []
            for i, c in enumerate(d.contact[:d.ncon]):
                n1, n2 = m.geom(c.geom1).name, m.geom(c.geom2).name
                if 'bottle' in n1 or 'bottle' in n2:
                    f = np.zeros(6); mujoco.mj_contactForce(m, d, i, f); cs.append(f"{(n1 if 'bottle' not in n1 else n2)[:12]}:{f[0]:.0f}")
            hp, hR = r.hand_pose("r"); tp, tR = r.hand["r"]
            print(f"{d.time:.3f} palm {hp.round(3)} tgt {tp.round(3)} wrist {d.qpos[m.joint('flexion_r').qposadr[0]]:.2f} cmdw {r.finger['r']['flexion_r']:.2f} | z {r.body_pose('bottle')[0][2]:.3f} root {r.root['x']:.2f},{r.root['y']:.2f} yaw {r.root['yaw']:.2f}", cs)
r.step = step
r.walk([(6.10, 2.70), (3.90, 3.95), (3.20, 3.95)] + RF.DOOR_TO_BED, final_yaw=np.pi / 2, speed=0.7, yaw_rate=1.4)

r.step = orig
r.walk([(3.20, 3.95)] + RF.DOOR_TO_BED, final_yaw=np.pi / 2, speed=0.7, yaw_rate=1.4)
print("END bottle z", r.body_pose("bottle")[0].round(3), "t", round(r.d.time, 2))
