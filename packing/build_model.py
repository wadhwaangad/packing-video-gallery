"""Compose the MuJoCo model: apartment, suitcase, items and two anatomical hands.

Contact groups (contype/conaffinity): hands 1/6, movable items and suitcase 2/2,
static apartment 4/2. Hands therefore touch items and the apartment but not each
other; items touch everything. No weld constraint connects a hand to anything.
The model has no equality constraints; the zipper is idealised by narrowing the lid
hinge range at runtime after a closure check.
"""
import os
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


def suitcase(world):
    """Scanned spinner case as contact geometry: liner walls and floors, the retracted
    trolley bar, four casters and a zip-track lip on the lid's free edges."""
    c = S.SUITCASE
    W, L = c["outer"]; D, Dl, t = c["base_depth"], c["lid_depth"], c["shell"]
    fl, lfl = c["floor"], c["lid_floor"]
    x0, y0, z0 = c["pos"]
    base = E.SubElement(world, "body", name="suitcase", pos=fmt((x0, y0, z0)), quat=fmt(quat_yaw(c["yaw"])))
    E.SubElement(base, "freejoint", name="suitcase_free")
    common = dict(contype="2", conaffinity="2", friction=".45 .01 .001", group="2", rgba=fmt((*c["color"], 1)),
                  solref=".004 1", solimp=".95 .99 .001", solmix="50")   # stiff bar dominates the soft skin under load
    def box(parent, name, center, size, mass, **kw):
        E.SubElement(parent, "geom", name=name, type="box", pos=fmt(center), size=fmt(np.array(size) / 2),
                     mass=str(mass), **{**common, **kw})
    m = c["base_mass"]
    box(base, "base_back", (0, 0, fl / 2), (W, L, fl), m * 0.40)
    # Thick shadow floors that only the packed items collide with (not the mattress or the
    # hands): a dropped item cannot tunnel through the thin liner floor.
    shadow = dict(contype="0", conaffinity="2", group="3", rgba=".5 .5 .5 .2", solref=".004 1", solimp=".95 .99 .001",
                  friction=".45 .01 .001")
    E.SubElement(base, "geom", name="base_floor_shadow", type="box", pos=fmt((0, 0, fl - 0.03)),
                 size=fmt((W / 2 - t, L / 2 - t, 0.03)), mass="0.001", **shadow)
    box(base, "base_wall_px", (W / 2 - t / 2, 0, D / 2), (t, L, D), m * 0.12)
    # The hinge-side wall: the base's goes up to the hinge axis and the lid's starts there,
    # so neither half swings into the other's packing space (a fabric hinge at the seam).
    hh = c["hinge_h"]
    box(base, "base_wall_nx", (-W / 2 + t / 2, 0, hh / 2), (t, L, hh), m * 0.12)
    box(base, "base_wall_py", (0, L / 2 - t / 2, D / 2), (W - 2 * t, t, D), m * 0.08)
    box(base, "base_wall_ny", (0, -L / 2 + t / 2, D / 2), (W - 2 * t, t, D), m * 0.08)
    trolley(base)
    # Spinner casters: two on each half.
    wc = dict(contype="2", conaffinity="2", friction=".9 .01 .001", group="2", rgba=".05 .05 .05 1", mass=".12")
    for sx in (-c["wheel_x"], c["wheel_x"]):
        caster(base, f"wheel_base_{'p' if sx > 0 else 'n'}", (sx, c["wheel_y"], c["wheel_z_base"]), wc)
    # Lid: hinge along the -x long edge at the rim (opens over towards -x).
    hh = c["hinge_h"]; zr = D - hh             # lid rim height in the lid frame
    lid = E.SubElement(base, "body", name="suitcase_lid", pos=fmt((-W / 2, 0, hh)))
    E.SubElement(lid, "joint", name="lid_hinge", type="hinge", axis="0 -1 0", range=f"0 {np.pi + 0.35}", solreflimit=".004 1",
                 damping=".5", frictionloss=".08", armature=".005", limited="true")
    ml = c["lid_mass"]
    box(lid, "lid_panel", (W / 2, 0, zr + Dl - lfl / 2), (W, L, lfl), ml * 0.45)
    E.SubElement(lid, "geom", name="lid_panel_shadow", type="box", pos=fmt((W / 2, 0, zr + Dl - lfl + 0.03)),
                 size=fmt((W / 2 - t, L / 2 - t, 0.03)), mass="0.001", **shadow)
    lid_front_wall(lid, box, ml * 0.15)
    box(lid, "lid_wall_nx", (t / 2, 0, (zr + Dl) / 2), (t, L, zr + Dl), ml * 0.15)
    box(lid, "lid_wall_py", (W / 2, L / 2 - t / 2, zr + Dl / 2), (W - 2 * t, t, Dl), ml * 0.1)
    box(lid, "lid_wall_ny", (W / 2, -L / 2 + t / 2, zr + Dl / 2), (W - 2 * t, t, Dl), ml * 0.1)
    # Zip-track lip along the free long edge: what fingertips catch to lift the lid open.
    box(lid, "lid_lip_px", (W + 0.006, 0, zr + 0.007), (0.012, L, 0.012), 0.02)
    for sx in (W / 2 - c["wheel_x"], W / 2 + c["wheel_x"]):
        caster(lid, f"wheel_lid_{'p' if sx > W / 2 else 'n'}", (sx, c["wheel_y"], zr + c["wheel_z_lid"]), wc)
    lock_hardware(lid)
    zip_hardware(base)
    strap_hardware(lid)
    # Idealised zipper: once the closure test passes, the run narrows the lid hinge range
    # to [0, 0.003] rad (no equality constraint; see runner._mechanisms).
    return base


HW_TOUCH = dict(contype="0", conaffinity="1", group="2")      # hardware the hands touch (nothing else)


def caster(parent, name, pos, wc):
    """A 360-degree spinner wheel: a sphere on a ball joint, so it rolls in any direction.
    Joint friction is the wheel's rolling resistance (about 1 N per wheel at the floor),
    so a standing case stays put until it is pushed."""
    c = S.SUITCASE
    b = E.SubElement(parent, "body", name=f"{name}_body", pos=fmt(pos))
    E.SubElement(b, "joint", name=f"{name}_ball", type="ball", damping=".002", frictionloss=".035", armature="2e-4")
    E.SubElement(b, "geom", name=name, type="sphere", size=str(c["wheel_r"]), **wc)


def trolley(base):
    """Telescopic handle: the bar and its posts slide out along +y (the case length) on a
    slide joint whose range the runner narrows to a stop (the latch) unless the button in
    the grip is held in."""
    c = S.SUITCASE; hb = c["bar"]; tr = S.HARDWARE["trolley"]
    L = c["outer"][1]
    hc = dict(contype="2", conaffinity="2", friction=".8 .01 .001", group="2", rgba=".05 .05 .05 1", mass=".10",
              solref=".004 1", solimp=".95 .99 .001", solmix="50")   # stiff bar dominates the soft skin under load
    body = E.SubElement(base, "body", name="trolley", pos=fmt((hb["x"], hb["y"], hb["z"])))
    E.SubElement(body, "joint", name="trolley_slide", type="slide", axis="0 1 0", range=f"0 {tr['travel']}",
                 limited="true", damping="15", frictionloss=str(tr["friction"]), armature=".01", solreflimit=".004 1")
    E.SubElement(body, "geom", name="top_handle", type="capsule", size=str(hb["r"]),
                 fromto=fmt((-hb["half"], 0, 0, hb["half"], 0, 0)), **hc)
    post_len = hb["y"] - L / 2 + tr["travel"] + 0.02
    for k, sx in enumerate((-hb["half"] - 0.016, hb["half"] + 0.016)):
        E.SubElement(body, "geom", name=f"trolley_post_{k}", type="box", size=fmt((0.011, post_len / 2, 0.011)),
                     pos=fmt((sx, -post_len / 2 + 0.011, 0)), contype="0", conaffinity="0", group="2",
                     rgba=".05 .05 .05 1", mass=".08")
        # the top of each post (always outside the case) stops a hand sliding off the bar's ends
        E.SubElement(body, "geom", name=f"trolley_post_top_{k}", type="box", size=fmt((0.011, 0.03, 0.011)),
                     pos=fmt((sx, -0.03 + 0.011, 0)), mass=".001", rgba=".05 .05 .05 1", friction=".8 .01 .001",
                     **HW_TOUCH)
    bt = tr["button"]
    btn = E.SubElement(body, "body", name="trolley_btn", pos=fmt((0, hb["r"] - bt["half"][1] + bt["proud"], 0)))
    E.SubElement(btn, "joint", name="trolley_btn_slide", type="slide", axis="0 -1 0", range=f"0 {bt['travel']}",
                 limited="true", stiffness=str(bt["stiffness"]), springref="0", damping="2", armature=".001")
    E.SubElement(btn, "geom", name="trolley_button", type="box", size=fmt(bt["half"]), mass=".005",
                 rgba=".25 .25 .27 1", friction="1 .01 .001", **HW_TOUCH)


def lid_front_wall(lid, box, mass):
    """The free-edge wall of the lid, recessed to the scan surface where the lock sits."""
    c = S.SUITCASE; hw = S.HARDWARE
    W, L = c["outer"]; D, Dl, t = c["base_depth"], c["lid_depth"], c["shell"]
    zr = D - c["hinge_h"]
    y0, y1 = hw["lock_window"]
    zl0, zl1 = hw["dial_z"] - c["hinge_h"] - 0.040, hw["dial_z"] - c["hinge_h"] + 0.045   # recessed band round the lock (lid frame)

    def seg(name, ya, yb, za, zb, xa=W - t, xb=W):
        # mass by face area, so the segmented wall weighs what the plain wall did
        box(lid, name, ((xa + xb) / 2, (ya + yb) / 2, (za + zb) / 2), (xb - xa, yb - ya, zb - za),
            mass * (yb - ya) * (zb - za) / (L * Dl))
    seg("lid_wall_px", -L / 2, y0, zr, zr + Dl)
    seg("lid_wall_px_b", y1, L / 2, zr, zr + Dl)
    seg("lid_wall_px_lo", y0, y1, zr, zl0)
    seg("lid_wall_px_hi", y0, y1, zl1, zr + Dl)
    xs = hw["wall_px_visual"] + W / 2           # lock plate surface (lid frame)
    seg("lid_wall_px_lock", y0, y1, zl0, zl1, xa=xs - 0.004, xb=xs)


def lock_hardware(lid):
    """Three-dial combination lock with a release button, in the front wall of the lid."""
    c = S.SUITCASE; hw = S.HARDWARE
    W = c["outer"][0]
    xs = hw["wall_px_visual"] + W / 2
    zl = hw["dial_z"] - c["hinge_h"]
    for k, y in enumerate(hw["dial_y"]):
        b = E.SubElement(lid, "body", name=f"dial_{k + 1}", pos=fmt((xs - hw["dial_r"] + 0.0045, y, zl)))
        E.SubElement(b, "joint", name=f"dial_{k + 1}_hinge", type="hinge", axis="0 1 0", damping="2e-4",
                     frictionloss="3e-4", armature="2e-5")
        E.SubElement(b, "geom", name=f"dial_{k + 1}_geom", type="cylinder", size=fmt((hw["dial_r"], hw["dial_w"] / 2)),
                     quat=fmt((0.7071068, 0.7071068, 0, 0)), mass=".002", rgba=".55 .55 .57 1", friction="1.4 .01 .001",
                     **HW_TOUCH)
    bt = hw["button"]
    b = E.SubElement(lid, "body", name="lock_btn", pos=fmt((xs + bt["half"][0] - 0.001, bt["y"], zl)))
    E.SubElement(b, "joint", name="lock_btn_slide", type="slide", axis="-1 0 0", range=f"0 {bt['travel']}", limited="true",
                 stiffness=str(bt["stiffness"]), springref="0", damping="1", armature=".001")
    E.SubElement(b, "geom", name="lock_button", type="box", size=fmt(bt["half"]), mass=".004", rgba=".6 .1 .1 1",
                 friction="1 .01 .001", **HW_TOUCH)


def zip_hardware(base):
    """Two zip sliders on the front side of the seam; each carries a pull tab that flips
    up into the dock of the lock. Slider A closes from the wheel end, B from the handle end."""
    hw = S.HARDWARE; sl = hw["slider"]; tb = hw["tab"]
    ends = {"a": (sl["open_a"], hw["dock_y"] - sl["gap"]), "b": (sl["open_b"], hw["dock_y"] + sl["gap"])}
    for k, (y_open, y_shut) in ends.items():
        b = E.SubElement(base, "body", name=f"slider_{k}", pos=fmt((sl["x"], y_open, sl["z"])))
        lo, hi = sorted((0.0, y_shut - y_open))
        E.SubElement(b, "joint", name=f"slider_{k}_slide", type="slide", axis="0 1 0", range=f"{lo} {hi}", limited="true",
                     damping="3", frictionloss=str(sl["friction"]), armature=".002")
        E.SubElement(b, "geom", name=f"slider_{k}_geom", type="box", size=fmt(sl["half"]), mass=".008",
                     rgba=".7 .72 .74 1", friction="1.2 .01 .001", **HW_TOUCH)
        # The pull hangs flat against the side and swings up in that plane (about the wall
        # normal): A towards the wheel end, B towards the handle end, so they pass clear.
        t = E.SubElement(b, "body", name=f"tab_{k}", pos=fmt((sl["half"][0] + tb["r"] + 0.004, 0, -sl["half"][2] + 0.002)))
        E.SubElement(t, "joint", name=f"tab_{k}_hinge", type="hinge", axis="-1 0 0" if k == "a" else "1 0 0",
                     range=f"0 {np.pi:.5f}", limited="true", damping="5e-4", armature="2e-5")
        E.SubElement(t, "geom", name=f"tab_{k}_geom", type="capsule", size=str(tb["r"]),
                     fromto=fmt((0, 0, -0.004, 0, 0, -tb["len"])), mass=".005", rgba=".1 .1 .1 1",
                     friction="1.2 .01 .001", **HW_TOUCH)


def strap_hardware(lid):
    """Side-release buckle of the strap along the lid front: the male half slides out
    along the strap (+y) once its two side tabs are squeezed."""
    c = S.SUITCASE; st = S.HARDWARE["strap"]
    W = c["outer"][0]
    zl = st["z"] - c["hinge_h"]
    E.SubElement(lid, "geom", name="buckle_female", type="box", size=fmt((st["width"] / 2 + 0.003, 0.017, 0.006)),
                 pos=fmt((W / 2 + st["x"], st["y_female"], zl + 0.006)), mass=".01", rgba=".08 .08 .08 1",
                 friction="1 .01 .001", **HW_TOUCH)
    b = E.SubElement(lid, "body", name="buckle_male", pos=fmt((W / 2 + st["x"], st["y_male"], zl + 0.005)))
    E.SubElement(b, "joint", name="buckle_slide", type="slide", axis="0 1 0", range=f"0 {st['travel']}", limited="true",
                 damping="2", frictionloss=".3", armature=".002")
    E.SubElement(b, "geom", name="buckle_male_geom", type="box", size=fmt((st["width"] / 2 - 0.004, 0.020, 0.0045)),
                 mass=".008", rgba=".08 .08 .08 1", friction="1.2 .01 .001", **HW_TOUCH)
    for k, sx in enumerate((-1, 1)):
        # side-release prongs: plastic springs that flex a few millimetres when squeezed
        E.SubElement(b, "geom", name=f"buckle_tab_{k}", type="box", size=fmt((0.003, 0.008, 0.004)),
                     pos=fmt((sx * (st["width"] / 2 - 0.001), 0.004, 0)), mass=".001", rgba=".08 .08 .08 1",
                     friction="1.4 .01 .001", solref=".012 1", solimp=".8 .95 .004", **HW_TOUCH)


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


def cloth(world):
    """The loose towel: a flex grid lying flat on its station (x across the counter
    depth, y along it)."""
    c = S.CLOTH
    nx, ny = c["count"]
    sx, sy = c["size"][0] / (nx - 1), c["size"][1] / (ny - 1)
    top = S.STATIONS[c["station"]].top
    fc = E.SubElement(world, "flexcomp", name=c["name"], type="grid", count=f"{nx} {ny} 1",
                      spacing=fmt((sx, sy, 0.05)), pos=fmt((c["centre"][0], c["centre"][1], top + c["radius"] + 0.001)),
                      radius=str(c["radius"]), dim="2", mass=str(c["mass"]), rgba=".85 .55 .35 1")
    E.SubElement(fc, "edge", equality="false", damping=".002")
    E.SubElement(fc, "elasticity", young=str(c["young"]), poisson=str(c["poisson"]), thickness=str(c["thickness"]),
                 damping=".0005", elastic2d="both")
    E.SubElement(fc, "contact", condim="3", friction=".9", solref=".003 1", selfcollide="auto", internal="false",
                 contype="2", conaffinity="3")


def build(start_hands=None, with_cloth=False):
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
    suitcase(world)
    for it in S.ITEMS:
        item_body(world, it)
    if with_cloth:
        cloth(world)
    actuator = E.SubElement(root, "actuator")
    hp = start_hands or {"r": (1.7, 0.75, 0.95), "l": (1.4, 0.75, 0.95)}
    for side in "rl":
        H.add_hand(world, actuator, side, hp[side])
    BUILD.mkdir(exist_ok=True)
    path = BUILD / f"packing_model_{os.getpid()}.xml"     # per process: generators run in parallel
    E.indent(root)
    E.ElementTree(root).write(path)
    return mujoco.MjModel.from_xml_path(str(path)), path


def build_web():
    """The model the 3D page runs live (play/pack/physics.xml, MuJoCo WebAssembly): the same
    apartment, case (shells, hinge, casters, handle, lock, sliders) and items with the same
    contact parameters, without the anatomical hands, which cannot run at interactive rates.
    The page carries an item with a mocap 'grip' and a soft weld per item (off until the
    item is picked up): soft enough that walls and furniture stop what is carried."""
    root = E.Element("mujoco", model="Apartment suitcase packing (web)")
    E.SubElement(root, "compiler", angle="radian", autolimits="true")
    # 2 ms steps and fewer solver iterations than the 1 ms offline runs: the page steps in real time
    # (impratio 10 keeps the hands' grasps from slipping offline; in the WebAssembly build (3.15)
    # it sends the case and the bottle flying, and without hands nothing needs it)
    E.SubElement(root, "option", timestep="0.002", integrator="implicitfast", cone="elliptic",
                 iterations="30", ls_iterations="10", gravity="0 0 -9.81")
    E.SubElement(root, "size", memory="64M")
    world = E.SubElement(root, "worldbody")
    E.SubElement(world, "geom", name="floor", type="plane", size="10 10 .1", pos="3.5 3.2 0",
                 contype="4", conaffinity="2", rgba=".55 .45 .35 1", friction=".8 .01 .001")
    for b in S.all_static_boxes():
        static_geom(world, b)
    suitcase(world)
    # Gram-scale hardware (zip sliders and pulls, dials, buttons, buckle) is unstable at 2 ms
    # steps and is moved by the page from the game state: fixed in place here, shapes kept.
    small = ("slider_", "tab_", "dial_", "lock_btn", "buckle_male", "trolley_btn")
    for b in world.iter("body"):
        if b.get("name", "").startswith(small):
            for j in b.findall("joint"):
                b.remove(j)
    for it in S.ITEMS:
        item_body(world, it)
    E.SubElement(world, "body", name="grip", mocap="true", pos="1.55 0.64 1.2")
    eq = E.SubElement(root, "equality")
    for it in S.ITEMS:
        E.SubElement(eq, "weld", name=f"hold_{it.name}", body1="grip", body2=it.name, relpose="0 0 0 1 0 0 0",
                     active="false", solref=".01 1", solimp=".9 .95 .002")
    E.indent(root)
    return E.tostring(root, encoding="unicode")


if __name__ == "__main__":
    m, p = build()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(p, "nq", m.nq, "nv", m.nv, "nu", m.nu, "nbody", m.nbody, "ngeom", m.ngeom)
    print("suitcase mass", sum(m.body_mass[i] for i in range(m.nbody) if m.body(i).name.startswith("suitcase")))
