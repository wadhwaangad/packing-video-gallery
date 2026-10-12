"""Online action runner: targets come from the live simulation state (what an
operator can see), motion is min-jerk, contact is the only thing that moves items.

Physics 1 kHz; carriage and finger targets updated at 500 Hz; frames recorded at 25 Hz.
"""
import json, hashlib, time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot, Slerp
import build_model as B
import scene as S
import hands as H

PHYS_DT = 0.001
CTRL_EVERY = 2          # 500 Hz
FRAME_EVERY = 40        # 25 Hz
PALM = {"r": np.array([0.021, -0.283, 0.003]), "l": np.array([-0.021, -0.283, 0.003])}

THUMB_OPEN = dict(cmc_flexion=0.6, cmc_abduction=0.7, mp_flexion=0.4, ip_flexion=0.2)
SHAPES = {
    "open": dict(mcp=0.12, pm=0.12, md=0.06, abd=(0.15, 0.05, -0.05, -0.15), **THUMB_OPEN),
    "flat": dict(mcp=0.20, pm=0.25, md=0.15, abd=(0.0, 0.0, 0.0, 0.0),
                 cmc_flexion=0.35, cmc_abduction=0.10, mp_flexion=-0.15, ip_flexion=-0.25),
    # Straight fingers: the palm pad (which protrudes past the finger pads) touches first.
    "press": dict(mcp=0.02, pm=0.04, md=0.03, abd=(0.0, 0.0, 0.0, 0.0),
                  cmc_flexion=0.70, cmc_abduction=-0.50, mp_flexion=-0.78, ip_flexion=0.43),
    "squeeze": dict(mcp=0.55, pm=0.75, md=0.45, abd=(0.0, 0.0, 0.0, 0.0),
                    cmc_flexion=0.45, cmc_abduction=0.0, mp_flexion=-0.35, ip_flexion=-0.6),
    "wrap": dict(mcp=1.386, pm=1.518, md=0.678, abd=(0.0, 0.0, 0.0, 0.0), wrist=-0.40,
                 cmc_flexion=0.379, cmc_abduction=-0.388, mp_flexion=-0.565, ip_flexion=-1.257),
    "hook": dict(mcp=1.10, pm=1.50, md=1.10, abd=(0.0, 0.0, 0.0, 0.0),
                 cmc_flexion=0.5, cmc_abduction=0.65, mp_flexion=0.1, ip_flexion=0.0),
    "relaxed": dict(mcp=0.35, pm=0.45, md=0.25, abd=(0.05, 0.0, -0.03, -0.08),
                    cmc_flexion=0.45, cmc_abduction=0.35, mp_flexion=0.0, ip_flexion=-0.2),
}


def _blend(a, b, c):
    out = {}
    for k in a:
        if k == "abd":
            out[k] = tuple(x + (y - x) * c for x, y in zip(a[k], b[k]))
        else:
            out[k] = a[k] + (b[k] - a[k]) * c
    return out


# Cup: 30% of a full power wrap with the thumb tucked. Searched in the true palm
# frame (dbg_cshape2.py) so a 72 mm cylinder 1 cm distal of the palm centre clears
# the hand by ~3 mm while it is lowered over the bottle.
SHAPES["cup"] = _blend(SHAPES["open"], SHAPES["wrap"], 0.3)
SHAPES["cup"]["wrist"] = -0.40
# Thumb tucked beside the index metacarpal: searched to clear a 72 mm cylinder by ~9 mm.
SHAPES["cup"].update(cmc_flexion=0.70, cmc_abduction=-0.50, mp_flexion=-0.78, ip_flexion=-1.30)
# Large-cylinder power grip: thumb abducted to oppose the fingers on the wrist side of
# the object (searched in dbg_cshape2.py); 'cup_thumb' closes the thumb first.
_OPP = dict(cmc_flexion=0.70, cmc_abduction=0.78, mp_flexion=0.69, ip_flexion=-0.61)
SHAPES["cup_thumb"] = dict(SHAPES["cup"], **_OPP)
SHAPES["wrap_big"] = dict(_blend(SHAPES["open"], SHAPES["wrap"], 0.88), wrist=-0.40, **_OPP)   # stops just past contact: moderate grip force
# Handle hook from above: fingers hang straight down from the knuckles (to drop into the
# gap behind a bar), then 'hook' curls them under it.
SHAPES["hook_open"] = dict(SHAPES["press"], mcp=1.45, pm=0.15, md=0.10)
SHAPES["cup_open"] = dict(SHAPES["open"], wrist=-0.40)   # release a wrapped object without swinging the forearm


def minjerk(u):
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def shape_targets(shape, side):
    sh = SHAPES[shape]
    out = {}
    for k in "2345":
        out[f"mcp{k}_flexion_{side}"] = sh["mcp"]
        out[f"pm{k}_flexion_{side}"] = sh["pm"]
        out[f"md{k}_flexion_{side}"] = sh["md"]
        out[f"mcp{k}_abduction_{side}"] = sh["abd"]["2345".index(k)]
    for k in ["cmc_flexion", "cmc_abduction", "mp_flexion", "ip_flexion"]:
        out[f"{k}_{side}"] = sh[k]
    out[f"flexion_{side}"] = sh.get("wrist", 0.0)
    out[f"deviation_{side}"] = 0.0
    return out


class Runner:
    def __init__(self, seed=7):
        self.m, self.xml = B.build()
        m = self.m
        self.d = mujoco.MjData(m)
        d = self.d
        d.qpos[m.joint("lid_hinge").qposadr[0]] = np.pi + 0.12
        mujoco.mj_forward(m, d)
        self.aid = {m.actuator(i).name: i for i in range(m.nu)}
        self.carriage0 = {s: m.body(f"carriage_{s}").pos.copy() for s in "rl"}
        self._build_wrist_table()
        self.radius = {s: m.body(f"radius_{s}").id for s in "rl"}
        self.hand_bodies = {s: self._subtree(self.radius[s]) for s in "rl"}
        self.item_ids = {it.name: m.body(it.name).id for it in S.ITEMS}
        self.suit_ids = [m.body("suitcase").id, m.body("suitcase_lid").id]
        self.frames, self.events, self.stages = [], [], []
        self.diag = dict(max_penetration=0.0, worst_penetration=None, max_joint_excursion=0.0, worst_joint=None,
                         max_carriage_force=0.0, max_finger_torque_ratio=0.0, hand_item_welds=0)
        self.label = "Start"
        self.step_count = 0
        # Commanded state
        self.root = dict(x=1.45, y=0.64, yaw=np.pi / 2, h=0.95)
        self.hand = {}
        for s in "rl":
            p, R = self.hand_pose(s)
            self.hand[s] = (p, R)
        self.finger = {s: shape_targets("relaxed", s) for s in "rl"}
        self.gaze = np.array([1.55, 1.42, 0.6])
        self.rot_qadr = {s: int(m.jnt_qposadr[m.joint(f"w{s}_rot").id]) for s in "rl"}
        self.rot_target = {s: np.eye(3) for s in "rl"}
        self.rot_target_q = {s: (1.0, 0.0, 0.0, 0.0) for s in "rl"}
        self.rot_aid = {s: [self.aid[f"w{s}_t{k}"] for k in range(3)] for s in "rl"}
        self.root_log = []
        self.zip_progress = 0.0
        self.zip_locked = False

    # ------------------------------------------------------------ wrist kinematics
    WRIST_GRID = np.linspace(-0.75, 0.75, 151)

    def _build_wrist_table(self):
        """Palm frame relative to the forearm (radius) frame as a function of wrist flexion.

        Targets are palm poses; the carriage drives the forearm, so the commanded wrist
        flexion is undone here to obtain the forearm pose."""
        m = self.m
        d = mujoco.MjData(m)
        self.wrist = {}
        for s in "rl":
            rb, cb = m.body(f"radius_{s}").id, m.body(f"capitate_{s}").id
            j = m.joint(f"flexion_{s}").qposadr[0]
            pos, rot = [], []
            d.qpos[:] = m.qpos0
            mujoco.mj_forward(m, d)
            R0 = d.xmat[rb].reshape(3, 3).T @ d.xmat[cb].reshape(3, 3)
            c0 = d.xmat[rb].reshape(3, 3).T @ (d.xpos[cb] - d.xpos[rb])
            off = PALM[s] - c0                       # palm point relative to the capitate at neutral
            for phi in self.WRIST_GRID:
                d.qpos[:] = m.qpos0
                d.qpos[j] = phi
                mujoco.mj_forward(m, d)
                Rr = d.xmat[rb].reshape(3, 3)
                Rc = Rr.T @ d.xmat[cb].reshape(3, 3)
                c = Rr.T @ (d.xpos[cb] - d.xpos[rb])
                rel = Rc @ R0.T                      # palm rotation relative to the forearm
                pos.append(c + Rc @ (R0.T @ off)); rot.append(rel)
            self.wrist[s] = (np.array(pos), np.array(rot))

    def wrist_rel(self, s, phi):
        g = self.WRIST_GRID
        i = int(np.clip(np.searchsorted(g, phi) - 1, 0, len(g) - 2))
        u = (phi - g[i]) / (g[i + 1] - g[i])
        P, Rt = self.wrist[s]
        pos = P[i] + (P[i + 1] - P[i]) * u
        R = Rt[i] if u < 0.5 else Rt[i + 1]
        return pos, R

    # ------------------------------------------------------------ state access
    def _subtree(self, bid):
        m = self.m
        return [i for i in range(m.nbody) if self._is_desc(i, bid)]

    def _is_desc(self, i, root):
        m = self.m
        while i > 0:
            if i == root:
                return True
            i = m.body_parentid[i]
        return False

    def hand_pose(self, s):
        """Measured palm point and palm frame (via the capitate, so wrist flexion is included)."""
        m, d = self.m, self.d
        b = self.radius[s]
        R = d.xmat[b].reshape(3, 3)
        phi = d.qpos[m.joint(f"flexion_{s}").qposadr[0]]
        prel, Rrel = self.wrist_rel(s, phi)
        return d.xpos[b] + R @ prel, R @ Rrel

    def body_pose(self, name):
        b = self.m.body(name).id
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def suitcase_frame(self):
        return self.body_pose("suitcase")

    def root_frame(self, root=None):
        r = root or self.root
        return np.array([r["x"], r["y"], 0.0]), rz(r["yaw"])

    # ------------------------------------------------------------ control
    def _apply(self):
        m, d = self.m, self.d
        for s in "rl":
            p, Rp = self.hand[s]
            prel, Rrel = self.wrist_rel(s, self.finger[s][f"flexion_{s}"])
            R = Rp @ Rrel.T
            elbow = p - R @ prel
            off = elbow - self.carriage0[s]
            for k, ax in enumerate("xyz"):
                d.ctrl[self.aid[f"w{s}_{ax}"]] = off[k]
            self.rot_target[s] = R
            xyzw = Rot.from_matrix(R).as_quat()
            self.rot_target_q[s] = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
            for j, v in self.finger[s].items():
                a = self.aid[f"{j}_servo"]
                lo, hi = m.actuator_ctrlrange[a]
                d.ctrl[a] = np.clip(v, lo, hi)

    def _monitor(self):
        m, d = self.m, self.d
        for c in d.contact[: d.ncon]:
            if -c.dist > self.diag["max_penetration"]:
                self.diag["max_penetration"] = float(-c.dist)
                self.diag["worst_penetration"] = dict(t=round(d.time, 4), stage=self.label,
                                                      geoms=[m.geom(c.geom1).name, m.geom(c.geom2).name])
        for j in range(m.njnt):
            if m.jnt_limited[j] and m.jnt_type[j] == 3:
                q = d.qpos[m.jnt_qposadr[j]]
                ex = max(m.jnt_range[j, 0] - q, q - m.jnt_range[j, 1], 0.0)
                if ex > self.diag["max_joint_excursion"]:
                    self.diag["max_joint_excursion"] = float(ex)
                    self.diag["worst_joint"] = dict(t=round(d.time, 4), stage=self.label, joint=m.joint(j).name)

    def _rot_torque(self):
        """Rotation-vector PD on the ball joints (torques in the joint frame), every physics step."""
        d = self.d
        for s in "rl":
            qa = self.rot_qadr[s]
            qc = d.qpos[qa:qa + 4]
            qt = self.rot_target_q[s]
            # error quaternion q_e = conj(q_c) * q_t, expressed in the joint frame
            w1, x1, y1, z1 = qc[0], -qc[1], -qc[2], -qc[3]
            w2, x2, y2, z2 = qt
            ew = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            ex = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            ey = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            ez = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            if ew < 0:
                ew, ex, ey, ez = -ew, -ex, -ey, -ez
            sn = (ex * ex + ey * ey + ez * ez) ** 0.5
            if sn < 1e-12:
                tx = ty = tz = 0.0
            else:
                ang = 2.0 * np.arctan2(sn, ew)
                k = H.ROT_KP * ang / sn
                tx, ty, tz = k * ex, k * ey, k * ez
                n = (tx * tx + ty * ty + tz * tz) ** 0.5
                if n > H.CARRIAGE_TORQUE:
                    f = H.CARRIAGE_TORQUE / n
                    tx, ty, tz = tx * f, ty * f, tz * f
            ids = self.rot_aid[s]
            d.ctrl[ids[0]] = tx; d.ctrl[ids[1]] = ty; d.ctrl[ids[2]] = tz

    def step(self, n=1):
        m, d = self.m, self.d
        for _ in range(n):
            if self.step_count % CTRL_EVERY == 0:
                self._apply()
            self._rot_torque()
            mujoco.mj_step(m, d)
            self.step_count += 1
            if self.step_count % 20 == 0:
                self._monitor()
            if self.step_count % FRAME_EVERY == 0:
                self._record()

    def _record(self):
        d = self.d
        self.frames.append(dict(t=round(d.time, 4), stage=self.label, xpos=d.xpos.astype(np.float32).tolist(),
                                xquat=d.xquat.astype(np.float32).tolist(), root=dict(self.root),
                                gaze=self.gaze.tolist(), lid=float(d.qpos[self.m.joint("lid_hinge").qposadr[0]]), zip_progress=float(self.zip_progress),
                                zip=bool(getattr(self, "zip_locked", False))))

    # ------------------------------------------------------------ primitives
    def stage(self, label):
        self.label = label
        self.stages.append(dict(t=round(self.d.time, 3), title=label))
        print(f"{self.d.time:8.2f}s  {label}", flush=True)

    def event(self, kind, **kw):
        self.events.append(dict(t=round(self.d.time, 4), stage=self.label, event=kind, **kw))

    def move(self, T, hands=None, root=None, fingers=None, gaze=None, frame="world", follow=()):
        """Min-jerk move of any subset of targets over T seconds.
        hands: {side: (palm_pos, R)} in world (or root frame if frame='root').
        follow: sides without a target that keep their pose relative to the moving root."""
        h0 = {s: (self.hand[s][0].copy(), self.hand[s][1].copy()) for s in "rl"}
        r0 = dict(self.root)
        o0, R0r = self.root_frame()
        rel0 = {s: (R0r.T @ (h0[s][0] - o0), R0r.T @ h0[s][1]) for s in follow}
        f0 = {s: dict(self.finger[s]) for s in "rl"}
        g0 = self.gaze.copy()
        r1 = dict(r0, **(root or {}))
        if "yaw" in (root or {}):
            r1["yaw"] = r0["yaw"] + (r1["yaw"] - r0["yaw"] + np.pi) % (2 * np.pi) - np.pi
        f1 = {s: (shape_targets(fingers[s], s) if fingers and s in fingers else f0[s]) for s in "rl"}
        slerps = {}
        hands = hands or {}
        targets = {}
        for s in "rl":
            if s in hands:
                p, R = hands[s]
                if frame == "root":
                    o, Rr = self.root_frame(r1)
                    p, R = o + Rr @ p, Rr @ R
                targets[s] = (np.asarray(p, float), np.asarray(R, float))
                slerps[s] = Slerp([0, 1], Rot.from_matrix([h0[s][1], targets[s][1]]))
        n = max(1, int(round(T / PHYS_DT)))
        for k in range(n):
            u = minjerk((k + 1) / n)
            for key in ["x", "y", "yaw", "h"]:
                self.root[key] = r0[key] + (r1[key] - r0[key]) * u
            for s in targets:
                self.hand[s] = (h0[s][0] + (targets[s][0] - h0[s][0]) * u, slerps[s](u).as_matrix())
            if rel0:
                o, Rr = self.root_frame()
                for s, (lp, lR) in rel0.items():
                    self.hand[s] = (o + Rr @ lp, Rr @ lR)
            for s in "rl":
                self.finger[s] = {j: f0[s][j] + (f1[s][j] - f0[s][j]) * u for j in f0[s]}
            if gaze is not None:
                self.gaze = g0 + (np.asarray(gaze) - g0) * u
            self.step()

    def hold(self, T):
        self.step(int(round(T / PHYS_DT)))

    def walk(self, waypoints, speed=0.85, carry=True, gaze_ahead=True, final_yaw=None, yaw_rate=3.0, yaw_acc=8.0):
        """Walk the root along a polyline; hands keep their pose in the root frame."""
        pts = [np.array([self.root["x"], self.root["y"]])] + [np.array(p, float) for p in waypoints]
        seg = [np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])]
        L = sum(seg)
        if L < 1e-3:
            return
        o, Rr = self.root_frame()
        local = {s: (Rr.T @ (self.hand[s][0] - o), Rr.T @ self.hand[s][1]) for s in "rl"}
        T = L / speed + 1.0
        n = int(round(T / PHYS_DT))
        cum = np.concatenate([[0], np.cumsum(seg)])
        yaw0 = self.root["yaw"]
        def at(sv):
            i = min(np.searchsorted(cum, sv, side="right") - 1, len(seg) - 1)
            u = (sv - cum[i]) / max(seg[i], 1e-9)
            return pts[i] + (pts[i + 1] - pts[i]) * u, pts[i + 1] - pts[i]
        yaw_prev = yaw0
        w = 0.0
        for k in range(n):
            sv = L * minjerk((k + 1) / n)
            p, dvec = at(sv)
            # Heading looks 0.45 m ahead along the path so turns start before corners.
            ahead, _ = at(min(sv + 0.45, L))
            dvec = ahead - p if np.linalg.norm(ahead - p) > 0.05 else dvec
            heading = np.arctan2(dvec[1], dvec[0])
            # Turn smoothly into the walking direction, then towards the final yaw.
            if final_yaw is not None and sv > L - 0.5:
                w = minjerk((sv - (L - 0.5)) / 0.5)
                dh = (final_yaw - heading + np.pi) % (2 * np.pi) - np.pi
                heading = heading + dh * w
            dyaw = (heading - yaw_prev + np.pi) % (2 * np.pi) - np.pi
            # Heading follows with bounded angular rate and acceleration (no jerky turns).
            w_des = np.clip(dyaw * 6.0, -yaw_rate, yaw_rate)
            w = w + np.clip(w_des - w, -yaw_acc * PHYS_DT, yaw_acc * PHYS_DT)
            yaw = yaw_prev + w * PHYS_DT
            yaw_prev = yaw
            self.root.update(x=p[0], y=p[1], yaw=yaw)
            o, Rr = self.root_frame()
            for s in "rl":
                self.hand[s] = (o + Rr @ local[s][0], Rr @ local[s][1])
            if gaze_ahead:
                self.gaze = np.array([p[0] + 2.0 * np.cos(yaw), p[1] + 2.0 * np.sin(yaw), 0.9])
            self.step()
        if final_yaw is not None:
            self.turn(final_yaw)

    def turn(self, yaw, T=None):
        dy = abs((yaw - self.root["yaw"] + np.pi) % (2 * np.pi) - np.pi)
        if dy < 1e-3:
            return
        o, Rr = self.root_frame()
        local = {s: (Rr.T @ (self.hand[s][0] - o), Rr.T @ self.hand[s][1]) for s in "rl"}
        T = T or max(0.6, dy / 1.6)
        y0 = self.root["yaw"]
        y1 = y0 + (yaw - y0 + np.pi) % (2 * np.pi) - np.pi
        n = int(round(T / PHYS_DT))
        for k in range(n):
            self.root["yaw"] = y0 + (y1 - y0) * minjerk((k + 1) / n)
            o, Rr = self.root_frame()
            for s in "rl":
                self.hand[s] = (o + Rr @ local[s][0], Rr @ local[s][1])
            self.step()

    # ------------------------------------------------------------ queries
    def contacts_between(self, bodies_a, bodies_b):
        m, d = self.m, self.d
        sa, sb = set(bodies_a), set(bodies_b)
        n, force = 0, 0.0
        f = np.zeros(6)
        for i, c in enumerate(d.contact[: d.ncon]):
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in sa and b2 in sb) or (b2 in sa and b1 in sb):
                mujoco.mj_contactForce(m, d, i, f)
                n += 1
                force += abs(f[0])
        return n, force

    def hand_item_contact(self, item):
        b = [self.item_ids[item]] if isinstance(item, str) else item
        return {s: self.contacts_between(self.hand_bodies[s], b) for s in "rl"}

    def item_speed(self, name):
        b = self.m.body(name).id
        return float(np.linalg.norm(self.d.cvel[b][3:])), float(np.linalg.norm(self.d.cvel[b][:3]))

    def debug_snapshot(self, tag):
        """Print every hand contact and save MuJoCo views around the hands."""
        m, d = self.m, self.d
        for s_ in "rl":
            hb = set(self.hand_bodies[s_])
            for c in d.contact[: d.ncon]:
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if (b1 in hb) != (b2 in hb):
                    print("  contact", s_, m.geom(c.geom1).name, m.geom(c.geom2).name, round(-c.dist * 1000, 2), "mm")
            p, _ = self.hand_pose(s_)
            print("  hand", s_, "palm", p.round(3), "target", self.hand[s_][0].round(3))
        from PIL import Image
        rend = mujoco.Renderer(m, 480, 640)
        opt = mujoco.MjvOption(); opt.geomgroup[:] = 1
        mid = (self.hand_pose("r")[0] + self.hand_pose("l")[0]) / 2
        for k, (az, el) in enumerate([(self.root["yaw"] * 180 / np.pi + 180, -35), (self.root["yaw"] * 180 / np.pi + 90, -15)]):
            cam = mujoco.MjvCamera(); cam.lookat[:] = mid; cam.distance = 0.9; cam.azimuth = az; cam.elevation = el
            rend.update_scene(d, cam, opt)
            Image.fromarray(rend.render()).save(B.BUILD / f"debug_{tag}_{k}.png")
