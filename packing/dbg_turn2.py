import numpy as np, run_full as RF, tasks as T, mujoco
from runner import Runner
from scipy.spatial.transform import Rotation as Rot
r = Runner(); T.to_rest(r, T=1.0); r.hold(0.5)
r.walk(RF.TO_DRESSER + [(2.95, 0.98)], final_yaw=0.0)
T.squeeze_pick(r, ["jeans"])
def rep(tag):
    m, d = r.m, r.d
    print("==", tag, round(d.time, 2))
    for s in "rl":
        p, R = r.hand_pose(s); pt, Rt = r.hand[s]
        f = [round(d.actuator_force[r.aid[f'w{s}_{a}']], 1) for a in 'xyz']
        tq = [round(d.ctrl[i], 1) for i in r.rot_aid[s]]
        print(f"  {s} pos err {np.linalg.norm(p-pt)*1000:5.1f}mm rot err {np.degrees(np.linalg.norm(Rot.from_matrix(R.T@Rt).as_rotvec())):5.1f}  slide F {f}  rot tau {tq}")
        hb = set(r.hand_bodies[s]); ff = np.zeros(6)
        for i, c in enumerate(d.contact[:d.ncon]):
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in hb) != (b2 in hb):
                mujoco.mj_contactForce(m, d, i, ff)
                print(f"     {m.geom(c.geom1).name:22s} {m.geom(c.geom2).name:22s} N={ff[0]:6.1f}")
rep("after lift")
r.move(1.2, hands=T.carry_pose(r, ["jeans"]), root=dict(h=0.95))
rep("carry pose")
r.hold(0.5); rep("carry hold")
