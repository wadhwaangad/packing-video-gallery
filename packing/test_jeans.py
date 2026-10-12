"""Primitive test: walk to the dresser, squeeze-pick the jeans, carry, place in the suitcase."""
import json, sys, time
import numpy as np
import mujoco
from PIL import Image
from runner import Runner
import tasks as T
import build_model as B

t0 = time.time()
r = Runner()
r.stage("Settle")
T.to_rest(r, T=1.0)
r.hold(0.5)
r.stage("Walk to the dresser")
r.walk([(2.45, 0.70), (2.95, 0.98)], final_yaw=0.0)
info = T.squeeze_pick(r, ["jeans"])
r.stage("Carry jeans")
r.move(1.2, hands=T.carry_pose(r, ["jeans"]), root=dict(h=0.95))
r.walk([(2.45, 0.70), (1.42, 0.78)], final_yaw=np.pi / 2)
T.place_item(r, ["jeans"], "jeans")
T.to_rest(r)
r.hold(1.0)
print(json.dumps(r.events, indent=1))
print(json.dumps({k: v for k, v in r.diag.items()}, indent=1, default=str))
print("frames", len(r.frames), "wall", round(time.time() - t0, 1), "sim", round(r.d.time, 2))
# quick previews from recorded frames
rend = mujoco.Renderer(r.m, 480, 640)
opt = mujoco.MjvOption(); opt.geomgroup[:] = 1
cam = mujoco.MjvCamera(); cam.lookat[:] = (2.0, 1.0, 0.6); cam.distance = 2.6; cam.azimuth = 250; cam.elevation = -40
mujoco.mj_forward(r.m, r.d)
rend.update_scene(r.d, cam, opt); Image.fromarray(rend.render()).save(B.BUILD / "test_jeans_end.png")
run = B.BUILD / "run_test"; run.mkdir(exist_ok=True)
import shutil; shutil.copy(r.xml, run / "model.xml")
json.dump(dict(frames=r.frames), open(run / "frames.json", "w"))
