"""Cup/wrap search in the true palm frame (capitate-based, wrist extended)."""
import numpy as np, mujoco, xml.etree.ElementTree as E
import hands as H, runner as RN
root = E.Element("mujoco")
for n in H.source_defaults(): root.append(n)
root.find("compiler").attrib.pop("meshdir", None); root.find("compiler").attrib.pop("texturedir", None)
w = E.SubElement(root, "worldbody"); a = E.SubElement(root, "actuator")
H.add_hand(w, a, "r", (0, 0, 0))
m = mujoco.MjModel.from_xml_string(E.tostring(root, encoding="unicode")); d = mujoco.MjData(m)
cb = m.body("capitate_r").id
d.qpos[:] = 0; mujoco.mj_forward(m, d)
R0 = d.xmat[cb].reshape(3, 3).copy(); c0 = d.xpos[cb].copy(); off = RN.PALM["r"] - c0; off_c = R0.T @ off
hand_geoms = [g for g in range(m.ngeom) if "coll" in m.geom(g).name and m.geom_bodyid[g] > 2 and "forearm" not in m.geom(g).name and "palm_pad" not in m.geom(g).name]
def pose(q):
    d.qpos[:] = 0
    for j, v in q.items(): d.qpos[m.joint(j).qposadr[0]] = v
    mujoco.mj_forward(m, d)
    Rc = d.xmat[cb].reshape(3, 3); pal = d.xpos[cb] + Rc @ off_c; Rp = Rc @ R0.T
    return pal, Rp
def segs():
    out = []
    for g in hand_geoms:
        R = d.geom_xmat[g].reshape(3, 3); p = d.geom_xpos[g]; s = m.geom_size[g]
        ax = R[:, 2] * (s[1] if m.geom_type[g] == 3 else 0)
        out.append((m.geom(g).name, p - ax, p + ax, s[0] if m.geom_type[g] != 6 else min(s)))
    return out
def metric(q, Rc, dfw):
    pal, Rp = pose(q)
    n = -Rp[:, 2]; fwd = -Rp[:, 1]; axis = Rp[:, 0]
    ctr = pal + n * (Rc + 0.004) + fwd * dfw
    clear = 1.0; touching = 0; who = ""
    for name, a_, b_, r in segs():
        best = 1.0
        for t in np.linspace(0, 1, 9):
            v = a_ + (b_ - a_) * t - ctr; v -= axis * (v @ axis)
            best = min(best, np.linalg.norm(v) - Rc - r)
        if best < clear: clear, who = best, name
        touching += best < 0.002
    return clear, touching, who
shape = lambda nm, c: {j: o + (v - o) * c for (j, o), v in zip(RN.shape_targets("open", "r").items(), RN.shape_targets(nm, "r").values())}
cup_thumb = dict(cmc_flexion_r=0.70, cmc_abduction_r=-0.50, mp_flexion_r=-0.78, ip_flexion_r=-1.30)
for dfw in [0.0, 0.01, 0.02, 0.03, 0.04]:
    full = shape("wrap", 1.0); full["flexion_r"] = -0.4
    fc, ft, fw = metric(full, 0.036, dfw)
    row = []
    for c in [0.2, 0.3, 0.4, 0.5]:
        q = shape("wrap", c); q.update(cup_thumb); q["flexion_r"] = -0.4
        cc, _, cw = metric(q, 0.036, dfw)
        row.append(f"c{c}:{cc*1000:5.1f}({cw[:8]})")
    print(f"dfw {dfw:.2f}  full-wrap penetration {fc*1000:6.1f}mm touching {ft:2d} |", *row)

print("--- thumb search")
import itertools
full = shape("wrap", 1.0); full["flexion_r"] = -0.4
best = []
for cf, ca, mp, ip in itertools.product(np.linspace(-0.78, 0.7, 9), np.linspace(-0.5, 0.78, 9), np.linspace(-0.78, 0.69, 6), np.linspace(-1.3, 0.43, 6)):
    q = dict(full, cmc_flexion_r=cf, cmc_abduction_r=ca, mp_flexion_r=mp, ip_flexion_r=ip)
    pal, Rp = pose(q); n = -Rp[:, 2]; fwd = -Rp[:, 1]; axis = Rp[:, 0]
    ctr = pal + n * 0.040 + fwd * 0.010
    tip = None; minr = 1
    for g in hand_geoms:
        nm = m.geom(g).name
        if nm.startswith(("distal_thumb", "proximal_thumb")):
            v = d.geom_xpos[g] - ctr; v -= axis * (v @ axis)
            rr = np.linalg.norm(v) - 0.036 - m.geom_size[g][0]
            minr = min(minr, rr)
            if nm == "distal_thumb_coll_r":
                tip = (d.geom_xpos[g] - pal)
    f_, n_ = tip @ fwd, tip @ n
    # want the thumb tip on the wrist side of the bottle, touching it (radial ~0)
    score = abs(minr - 0.0) * 3 + max(0, f_ - (-0.02)) + abs(n_ - 0.04)
    best.append((score, round(minr * 1000, 1), round(f_ * 1000, 1), round(n_ * 1000, 1), (round(cf, 2), round(ca, 2), round(mp, 2), round(ip, 2))))
best.sort()
for b in best[:6]: print(b)
