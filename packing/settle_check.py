"""Settle the initial scene under gravity and save quick MuJoCo previews."""
import numpy as np, mujoco
from PIL import Image
import build_model as B
import scene as S

m, _ = B.build()
d = mujoco.MjData(m)
lid = m.joint("lid_hinge")
d.qpos[lid.qposadr[0]] = np.pi
mujoco.mj_forward(m, d)
start = {it.name: d.xpos[m.body(it.name).id].copy() for it in S.ITEMS}
for _ in range(2000):
    mujoco.mj_step(m, d)
print("lid angle", np.degrees(d.qpos[lid.qposadr[0]]).round(2))
for it in S.ITEMS:
    b = m.body(it.name).id
    print(f"{it.name:14s} drift {np.linalg.norm(d.xpos[b]-start[it.name])*1000:6.2f} mm  speed {np.linalg.norm(d.cvel[b][3:]):.2e}")
sb = m.body("suitcase").id
print("suitcase z", d.xpos[sb].round(4), "max pen", max([-c.dist for c in d.contact] + [0]))
r = mujoco.Renderer(m, 720, 1280)
opt = mujoco.MjvOption(); opt.geomgroup[:] = 1
for name, (lookat, dist, az, el) in {
    "top": ((3.5, 3.2, 0), 9.5, 90, -89), "bedroom": ((1.9, 1.3, 0.5), 3.2, 230, -35),
}.items():
    cam = mujoco.MjvCamera(); cam.lookat[:] = lookat; cam.distance = dist; cam.azimuth = az; cam.elevation = el
    r.update_scene(d, cam, opt)
    Image.fromarray(r.render()).save(B.BUILD / f"preview_{name}.png")
np.save(B.BUILD / "settled_qpos.npy", d.qpos)
