"""Two anatomical MyoHands (right + mirrored left) on force-limited wrist carriages.

Source: MyoHub/myo_sim via this repository's hand experiment (Apache-2.0,
packing/hand/LICENSE-myosim.txt). Muscles are replaced by torque-limited joint
position servos. Each forearm hangs from a 6-DoF carriage (3 slides with position
servos + a ball joint with a rotation-vector PD on three motors), all with finite
force and torque caps; no hand-
object welds exist anywhere in the model.

Hand frame convention (radius body, right hand): local -y points from elbow to
fingers, local -z is the palm normal, local +x is the thumb side. The left hand
is the mirror image through the local x=0 plane, so its thumb side is -x.
"""
import copy, re
import xml.etree.ElementTree as E
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "hand" / "anatomical_source.xml"

# Finger flexion torque caps (N m). Assumed adult values, still well below
# reported maximal voluntary flexion torques (adult power grip ~300 N).
FINGER_TORQUE = {"mcp": 3.0, "pm": 1.8, "md": 0.9, "thumb": 2.4, "abd": 0.5, "wrist": 8.0}
CARRIAGE_FORCE = {"x": 120.0, "y": 120.0, "z": 160.0}   # N
CARRIAGE_TORQUE = 40.0                                 # N m, well below adult elbow/shoulder strength
CARRIAGE_KP, CARRIAGE_KV = 6000.0, 260.0
ROT_KP, ROT_KV = 450.0, 8.0   # rotation-vector PD (Nm/rad), joint damping (Nm s/rad)


def _mirror_vec(s):
    v = [float(x) for x in s.split()]
    return " ".join(f"{x:.6g}" for x in ([-v[0], v[1], v[2]]))


def _mirror_quat(s):
    w, x, y, z = (float(t) for t in s.split())
    return f"{w:.6g} {x:.6g} {-y:.6g} {-z:.6g}"


def _mirror_axis(s):
    x, y, z = (float(t) for t in s.split())
    return f"{x:.6g} {-y:.6g} {-z:.6g}"


def _mirror_fromto(s):
    v = [float(x) for x in s.split()]
    return " ".join(f"{t:.6g}" for t in [-v[0], v[1], v[2], -v[3], v[4], v[5]])


def source_defaults():
    src = E.parse(SRC).getroot()
    out = []
    for tag in ["compiler", "default"]:
        node = copy.deepcopy(src.find(tag))
        if tag == "default":
            for parent in list(node.iter("default")):
                for child in list(parent):
                    if child.tag == "default" and not child.get("class"):
                        parent.remove(child)
                        for n in list(child):
                            parent.append(n)
        out.append(node)
    return out


def hand_body(side):
    """Return (radius body element, finger joint names) for side 'r' or 'l'."""
    src = E.parse(SRC).getroot()
    body = copy.deepcopy(src.find('.//body[@name="radius_r"]'))
    body.set("pos", "0 0 0")
    body.set("childclass", "myoarm")
    for parent in list(body.iter()):
        for x in list(parent):
            if x.tag in ("camera", "site") or (x.tag == "geom" and "coll" not in x.get("name", "")):
                parent.remove(x)
    # Forearm rotation is supplied by the carriage roll hinge.
    pro = body.find('joint[@name="pro_sup_r"]')
    body.remove(pro)
    # The source forearm capsules bulge 3-4 cm past the palm on the palmar side at the
    # wrist; replace them with two capsules sized like a forearm near the wrist
    # (about 6 cm wide, 4 cm thick, palmar surface flush with the heel of the palm).
    for g in list(body.findall("geom")):
        if g.get("name", "").startswith("radius_coll"):
            body.remove(g)
    E.SubElement(body, "geom", name="forearm_a_coll_r", type="capsule", size=".022",
                 fromto=".020 -.030 .030 .020 -.205 .028", mass=".12")
    E.SubElement(body, "geom", name="forearm_b_coll_r", type="capsule", size=".022",
                 fromto=".000 -.030 .032 .002 -.205 .030", mass=".11")
    # A palm pad fills the gap between the metacarpal skin capsules so the palm
    # itself can carry load (palm soft tissue is otherwise missing from the source).
    cap = body.find('.//body[@name="capitate_r"]')
    E.SubElement(cap, "geom", name="palm_pad_coll_r", type="box", size=".028 .030 .009",
                 pos="-.001 -.026 -.015", quat="1 0 0 0", mass=".02")
    E.SubElement(body.find('.//body[@name="firstmc_r"]'), "geom", name="thenar_pad_coll_r",
                 type="ellipsoid", size=".013 .020 .009", pos=".002 -.010 -.008", mass=".008")
    joints = []
    for g in body.iter("geom"):
        g.attrib.pop("class", None)
        g.attrib.pop("material", None)
        if g.get("type") is None:
            g.set("type", "capsule")
        g.set("contype", "1"); g.set("conaffinity", "6"); g.set("group", "1")
        g.set("friction", "1.1 .02 .001"); g.set("solref", ".006 1"); g.set("solimp", ".95 .99 .001")
        g.set("condim", "4"); g.set("margin", "0"); g.set("rgba", ".63 .36 .23 1")
    for j in body.iter("joint"):
        j.set("damping", ".05"); j.set("armature", ".0002")
        j.set("solreflimit", ".004 1"); j.set("solimplimit", ".99 .999 .001")
        joints.append(j.get("name"))
    if side == "l":
        for node in body.iter():
            if node.get("name"):
                node.set("name", re.sub(r"_r$", "_l", node.get("name")))
            for key, fn in [("pos", _mirror_vec), ("quat", _mirror_quat), ("axis", _mirror_axis), ("fromto", _mirror_fromto)]:
                if node.get(key):
                    node.set(key, fn(node.get(key)))
        joints = [re.sub(r"_r$", "_l", n) for n in joints]
    return body, joints


def finger_cap(name):
    if name.startswith(("flexion", "deviation")):
        return FINGER_TORQUE["wrist"]
    if "abduction" in name:
        return FINGER_TORQUE["abd"]
    if name.startswith(("cmc", "mp_", "ip_")):
        return FINGER_TORQUE["thumb"]
    if name.startswith("mcp"):
        return FINGER_TORQUE["mcp"]
    if name.startswith("pm"):
        return FINGER_TORQUE["pm"]
    return FINGER_TORQUE["md"]


def add_hand(world, actuator, side, start_pos, start_euler=(0, 0, 0)):
    """Attach carriage + hand to world. Returns dict of actuator names."""
    names = {}
    parent = E.SubElement(world, "body", name=f"carriage_{side}", pos=" ".join(map(str, start_pos)))
    E.SubElement(parent, "inertial", pos="0 0 0", mass=".05", diaginertia="1e-4 1e-4 1e-4")
    for ax, vec in zip("xyz", ["1 0 0", "0 1 0", "0 0 1"]):
        E.SubElement(parent, "joint", name=f"w{side}_{ax}", type="slide", axis=vec, damping="4", armature=".05")
    # Orientation: one ball joint (no gimbal singularity) driven by three torque motors.
    cur = E.SubElement(parent, "body", name=f"carriage_{side}_rot")
    E.SubElement(cur, "inertial", pos="0 0 0", mass=".02", diaginertia="1e-5 1e-5 1e-5")
    E.SubElement(cur, "joint", name=f"w{side}_rot", type="ball", damping=str(ROT_KV), armature=".02")
    body, joints = hand_body(side)
    cur.append(body)
    for ax in "xyz":
        n = f"w{side}_{ax}"
        E.SubElement(actuator, "position", name=n, joint=n, kp=str(CARRIAGE_KP), kv=str(CARRIAGE_KV),
                     forcerange=f"-{CARRIAGE_FORCE[ax]} {CARRIAGE_FORCE[ax]}", ctrllimited="false")
    for k, g in enumerate(["1 0 0", "0 1 0", "0 0 1"]):
        E.SubElement(actuator, "motor", name=f"w{side}_t{k}", joint=f"w{side}_rot", gear=g,
                     ctrlrange=f"-{CARRIAGE_TORQUE} {CARRIAGE_TORQUE}", ctrllimited="true")
    for j in joints:
        jn = body.find(f'.//joint[@name="{j}"]')
        cap = finger_cap(j)
        kp = 30.0 if cap >= 1 else 12.0
        if j.startswith(("flexion", "deviation")):
            kp = 120.0
        E.SubElement(actuator, "position", name=f"{j}_servo", joint=j, kp=str(kp), kv=str(kp * 0.02),
                     ctrlrange=jn.get("range"), forcerange=f"-{cap} {cap}")
    names["fingers"] = joints
    return names


# Right-hand frame helpers -------------------------------------------------
def hand_rotation(forward, palm, side):
    """Rotation of the radius body: local -y -> forward, local -z -> palm normal."""
    f = np.asarray(forward, float); f /= np.linalg.norm(f)
    p = np.asarray(palm, float); p -= f * (p @ f); p /= np.linalg.norm(p)
    y = -f; z = -p; x = np.cross(y, z)
    R = np.column_stack([x, y, z])
    return R  # the mirrored left hand uses the same local axes; its thumb is at local -x


def euler_zyx(R, prev=None):
    """yaw, pitch, roll with R = Rz(yaw) Ry(pitch) Rx(roll); yaw unwrapped near prev."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    out = np.array([yaw, pitch, roll])
    if prev is not None:
        out[0] = prev[0] + (out[0] - prev[0] + np.pi) % (2 * np.pi) - np.pi
        out[2] = prev[2] + (out[2] - prev[2] + np.pi) % (2 * np.pi) - np.pi
    return out
