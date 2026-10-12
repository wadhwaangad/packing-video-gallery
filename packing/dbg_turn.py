import numpy as np, run_full as RF, tasks as T
from runner import Runner
from scipy.spatial.transform import Rotation as Rot
r = Runner(); T.to_rest(r, T=1.0); r.hold(0.5)
r.walk(RF.TO_DRESSER + [(2.95, 0.98)], final_yaw=0.0)
T.squeeze_pick(r, ["jeans"])
r.move(1.2, hands=T.carry_pose(r, ["jeans"]), root=dict(h=0.95))
orig = r.step
cnt = [0]
def step(n=1):
    for _ in range(n):
        orig(1); cnt[0] += 1
        if cnt[0] % 100 == 0:
            out = []
            for s in "rl":
                p, R = r.hand_pose(s); pt, Rt = r.hand[s]
                ang = np.degrees(np.linalg.norm(Rot.from_matrix(R.T @ Rt).as_rotvec()))
                out.append(f"{s}: pos err {np.linalg.norm(p-pt)*1000:5.1f}mm rot err {ang:5.1f}deg")
            print(f"{r.d.time:7.2f} yaw {r.root['yaw']:.2f} jeans z {r.body_pose('jeans')[0][2]:.3f} ", *out)
r.step = step
r.walk(RF.TO_DRESSER + [RF.SUIT], final_yaw=np.pi / 2)
