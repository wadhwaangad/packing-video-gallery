"""Find a finger pre-shape whose plan-view curl clears a cylinder lowered from above."""
import numpy as np, mujoco, xml.etree.ElementTree as E
import hands as H, runner as RN
root = E.Element("mujoco")
for n in H.source_defaults(): root.append(n)
root.find("compiler").attrib.pop("meshdir", None); root.find("compiler").attrib.pop("texturedir", None)
w = E.SubElement(root, "worldbody"); a = E.SubElement(root, "actuator")
H.add_hand(w, a, "r", (0, 0, 0))
m = mujoco.MjModel.from_xml_string(E.tostring(root, encoding="unicode")); d = mujoco.MjData(m)
hb = [i for i in range(m.ngeom) if "coll" in m.geom(i).name and m.geom(i).name.startswith(("proxph", "midph", "distph", "5"))]
def geoms(shape, c):
    d.qpos[:] = 0
    tg = RN.shape_targets(shape, "r"); op = RN.shape_targets("open", "r")
    for j, v in tg.items():
        d.qpos[m.joint(j).qposadr[0]] = op[j] + (v - op[j]) * c
    mujoco.mj_forward(m, d)
    out = []
    for g in hb:
        # capsule segment endpoints in the radius frame (identity at origin)
        R = d.geom_xmat[g].reshape(3, 3); p = d.geom_xpos[g]; s = m.geom_size[g]
        ax = R[:, 2] * (s[1] if m.geom_type[g] == 3 else 0)
        out.append((m.geom(g).name, p - ax, p + ax, s[0]))
    return out
palm = RN.PALM["r"]
def clearance(shape, c, Rc, d_fwd, z_lo=-0.0, z_hi=0.0):
    """Cylinder axis along local x (thumb axis), centre at palm + (-z)*(Rc+gap) + (-y)*d_fwd."""
    best = 1.0; wrap = -1.0
    ctr = palm + np.array([0, -d_fwd, -(Rc + 0.004)])
    for name, a, b, r in geoms(shape, c):
        for t in np.linspace(0, 1, 7):
            q = a + (b - a) * t
            v = q - ctr; v[0] = 0
            dist = np.linalg.norm(v) - Rc - r
            best = min(best, dist)
            if name.startswith(("distph", "5distph")):
                wrap = max(wrap, -(q[2] - ctr[2]))   # how far tips reach past the centre (towards -z = palm normal side)
    return best, wrap
for shape in ["wrap", "squeeze"]:
    for c in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for dfw in [0.02, 0.03, 0.04, 0.05]:
            cl, wr = clearance(shape, c, 0.036, dfw)
            if cl > 0.003:
                print(shape, c, dfw, "clear %.1f mm" % (cl * 1000), "tips past centre %.1f mm" % (wr * 1000))
