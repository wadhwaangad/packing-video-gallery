"""Compose the MuJoCo model: apartment, suitcase, items and two anatomical hands.

Contact groups (contype/conaffinity): hands 1/6, movable items and suitcase 2/2,
static apartment 4/2. Hands therefore touch items and the apartment but not each
other; items touch everything. No weld constraint connects a hand to anything.
The model has no equality constraints; the zipper is idealised by narrowing the lid
hinge range at runtime after a closure check.
"""
import xml.etree.ElementTree as E
from pathlib import Path
import numpy as np
import mujoco
import scene as S
import hands as H

BUILD = Path(__file__).resolve().parent / "build"


def fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def quat_yaw(yaw):
    return (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))


def static_geom(world, b):
    if not b.collide:
        return
    attrs = dict(name=b.name, type="box", size=fmt(np.array(b.size) / 2), pos=fmt(b.center),
                 contype="4", conaffinity="2", group="3", rgba=".7 .7 .7 .3", friction=".7 .01 .001")
    if b.yaw:
        attrs["quat"] = fmt(quat_yaw(b.yaw))
    if b.name == "mattress":
        # Compliant mattress: ~1 cm sink under a 9 kg suitcase.
        attrs.update(solref=".03 .9", solimp=".85 .95 .004", friction=".9 .01 .001")
    E.SubElement(world, "geom", **attrs)


def suitcase(world, equality):
    c = S.SUITCASE
    W, L = c["outer"]; D, Dl, t = c["base_depth"], c["lid_depth"], c["shell"]
    x0, y0, z0 = c["pos"]
    base = E.SubElement(world, "body", name="suitcase", pos=fmt((x0, y0, z0)), quat=fmt(quat_yaw(c["yaw"])))
    E.SubElement(base, "freejoint", name="suitcase_free")
    common = dict(contype="2", conaffinity="2", friction=".45 .01 .001", group="2", rgba=fmt((*c["color"], 1)),
                  solref=".004 1", solimp=".95 .99 .001")
    def box(parent, name, center, size, mass, **kw):
        E.SubElement(parent, "geom", name=name, type="box", pos=fmt(center), size=fmt(np.array(size) / 2),
                     mass=str(mass), **{**common, **kw})
    m = c["base_mass"]
    box(base, "base_back", (0, 0, t / 2), (W, L, t), m * 0.40)
    box(base, "base_wall_px", (W / 2 - t / 2, 0, D / 2), (t, L, D), m * 0.12)
    box(base, "base_wall_nx", (-W / 2 + t / 2, 0, D / 2), (t, L, D), m * 0.12)
    box(base, "base_wall_py", (0, L / 2 - t / 2, D / 2), (W - 2 * t, t, D), m * 0.08)
    box(base, "base_wall_ny", (0, -L / 2 + t / 2, D / 2), (W - 2 * t, t, D), m * 0.08)
    # Carry handle on the top end (+y) and side handle on the -x long side.
    hc = dict(contype="2", conaffinity="2", friction=".8 .01 .001", group="2", rgba=".05 .05 .05 1", mass=".06")
    E.SubElement(base, "geom", name="top_handle", type="capsule", size=".011",
                 fromto=fmt((-0.065, L / 2 + 0.045, D * 0.55, 0.065, L / 2 + 0.045, D * 0.55)), **hc)
    for sx in (-0.075, 0.075):
        E.SubElement(base, "geom", name=f"top_handle_post_{sx:+.2f}", type="box", size=".012 .018 .012",
                     pos=fmt((sx, L / 2 + 0.023, D * 0.55)), **hc)
    E.SubElement(base, "geom", name="side_handle", type="capsule", size=".011",
                 fromto=fmt((W / 2 + 0.045, -0.065, D * 0.55, W / 2 + 0.045, 0.065, D * 0.55)), **hc)
    for sy in (-0.075, 0.075):
        E.SubElement(base, "geom", name=f"side_handle_post_{sy:+.2f}", type="box", size=".018 .012 .012",
                     pos=fmt((W / 2 + 0.023, sy, D * 0.55)), **hc)
    # Telescopic handle housing on the back (inside the back panel line, visual + mass).
    E.SubElement(base, "geom", name="trolley_housing", type="box", size=".12 .30 .008",
                 pos=fmt((0, 0.0, -0.002)), mass="0.7", contype="0", conaffinity="0", group="2", rgba=".1 .1 .1 1")
    # Spinner wheels on the -y end (base side pair).
    for sx in (-W / 2 + 0.045, W / 2 - 0.045):
        E.SubElement(base, "geom", name=f"wheel_base_{'p' if sx > 0 else 'n'}", type="sphere", size=".028",
                     pos=fmt((sx, -L / 2 - 0.03, 0.045)), mass=".09", contype="2", conaffinity="2",
                     friction=".9 .01 .001", group="2", rgba=".05 .05 .05 1")
    # Lid: hinge along the -x long edge at the rim (opens over towards -x).
    lid = E.SubElement(base, "body", name="suitcase_lid", pos=fmt((-W / 2, 0, D)))
    E.SubElement(lid, "joint", name="lid_hinge", type="hinge", axis="0 -1 0", range=f"0 {np.pi + 0.35}", solreflimit=".004 1",
                 damping=".5", frictionloss=".08", armature=".005", limited="true")
    ml = c["lid_mass"]
    box(lid, "lid_panel", (W / 2, 0, Dl - t / 2), (W, L, t), ml * 0.45)
    box(lid, "lid_wall_px", (W - t / 2, 0, Dl / 2), (t, L, Dl), ml * 0.15)
    box(lid, "lid_wall_nx", (t / 2, 0, Dl / 2), (t, L, Dl), ml * 0.15)
    box(lid, "lid_wall_py", (W / 2, L / 2 - t / 2, Dl / 2), (W - 2 * t, t, Dl), ml * 0.1)
    box(lid, "lid_wall_ny", (W / 2, -L / 2 + t / 2, Dl / 2), (W - 2 * t, t, Dl), ml * 0.1)
    for sx in (W - 0.045, 0.045):
        E.SubElement(lid, "geom", name=f"wheel_lid_{'p' if sx > W / 2 else 'n'}", type="sphere", size=".028",
                     pos=fmt((sx, -L / 2 - 0.03, Dl - 0.045)), mass=".09", contype="2", conaffinity="2",
                     friction=".9 .01 .001", group="2", rgba=".05 .05 .05 1")
    # Idealised zipper: once the closure test passes, the run narrows the lid hinge range
    # to [0, 0.003] rad (no equality constraint; see skills_case.press_and_zip).
    return base


def item_body(world, it):
    q = quat_yaw(it.start_yaw)
    b = E.SubElement(world, "body", name=it.name, pos=fmt(it.start), quat=fmt(q))
    E.SubElement(b, "freejoint", name=f"{it.name}_free")
    common = dict(contype="2", conaffinity="2", group="2", condim="4")
    if it.soft:
        # Folded fabric: compliant contact (about 2-3 mm of give under a grip).
        common.update(friction=".9 .02 .001", solref=".012 1", solimp=".8 .95 .003")
    else:
        common.update(friction=".7 .01 .001", solref=".005 1", solimp=".95 .99 .001")
    sx, sy, sz = it.size
    if it.name == "bottle":
        # Tall and light: rolling friction and over-damped contact keep it from rocking.
        common.update(condim="6", friction=".8 .02 .002", solref=".005 1.4")
        # Insulated steel bottle: body, flat base ring, narrower cap.
        E.SubElement(b, "geom", name="bottle_body", type="cylinder", size=fmt((0.036, sz / 2 - 0.017)),
                     pos=fmt((0, 0, -0.017)), mass=str(it.mass * 0.85), rgba=".36 .45 .38 1", **common)
        E.SubElement(b, "geom", name="bottle_base", type="box", size=fmt((0.0255, 0.0255, 0.003)),
                     pos=fmt((0, 0, -sz / 2 + 0.003)), mass=str(it.mass * 0.05), rgba=".3 .3 .3 1", **common)
        E.SubElement(b, "geom", name="bottle_cap", type="cylinder", size=fmt((0.026, 0.017)),
                     pos=fmt((0, 0, sz / 2 - 0.017)), mass=str(it.mass * 0.10), rgba=".7 .7 .7 1", **common)
    elif it.name == "camera":
        E.SubElement(b, "geom", name="camera_body", type="box", size=fmt((sx / 2, sy / 2, sz / 2)),
                     mass=str(it.mass * 0.7), rgba=".1 .1 .1 1", **common)
        E.SubElement(b, "geom", name="camera_lens", type="cylinder", size=fmt((0.027, 0.027)),
                     pos=fmt((0.012, -0.0445, -0.004)), quat=fmt((0.7071068, 0.7071068, 0, 0)),
                     mass=str(it.mass * 0.3), rgba=".2 .2 .2 1", **common)
    else:
        E.SubElement(b, "geom", name=f"{it.name}_geom", type="box", size=fmt((sx / 2, sy / 2, sz / 2)),
                     mass=str(it.mass), rgba=fmt((*it.tint, 1)), **common)
    return b


def build(start_hands=None):
    root = E.Element("mujoco", model="Apartment suitcase packing")
    for node in H.source_defaults():
        root.append(node)
    root.find("compiler").attrib.pop("meshdir", None)
    root.find("compiler").attrib.pop("texturedir", None)
    root.find("compiler").set("angle", "radian")
    E.SubElement(root, "option", timestep="0.001", integrator="implicitfast", cone="elliptic", impratio="10",
                 iterations="80", ls_iterations="20", noslip_iterations="3", gravity="0 0 -9.81")
    size = E.SubElement(root, "size", memory="256M")
    vis = E.SubElement(root, "visual"); E.SubElement(vis, "global", offwidth="1280", offheight="720")
    world = E.SubElement(root, "worldbody")
    E.SubElement(world, "light", pos="2 2 2.5", dir="0 0 -1", diffuse=".8 .8 .8")
    E.SubElement(world, "geom", name="floor", type="plane", size="10 10 .1", pos="3.5 3.2 0",
                 contype="4", conaffinity="2", rgba=".55 .45 .35 1", friction=".8 .01 .001")
    for b in S.all_static_boxes():
        static_geom(world, b)
    equality = None
    suitcase(world, equality)
    for it in S.ITEMS:
        item_body(world, it)
    actuator = E.SubElement(root, "actuator")
    hp = start_hands or {"r": (1.7, 0.75, 0.95), "l": (1.4, 0.75, 0.95)}
    for side in "rl":
        H.add_hand(world, actuator, side, hp[side])
    BUILD.mkdir(exist_ok=True)
    path = BUILD / "packing_model.xml"
    E.indent(root)
    E.ElementTree(root).write(path)
    return mujoco.MjModel.from_xml_path(str(path)), path


if __name__ == "__main__":
    m, p = build()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(p, "nq", m.nq, "nv", m.nv, "nu", m.nu, "nbody", m.nbody, "ngeom", m.ngeom)
    print("suitcase mass", sum(m.body_mass[i] for i in range(m.nbody) if m.body(i).name.startswith("suitcase")))
