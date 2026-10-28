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
TIP_R = 0.007           # fingertip capsule radius
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
# Letting go of a bar: fingers hang straight down behind it and the thumb opens away, so
# nothing is left under the bar when the hand lifts.
SHAPES["bar_release"] = dict(SHAPES["hook_open"], **THUMB_OPEN)
# Two-hand hook on the 32 mm trolley bar: a softer wrap than 'hook' (sized for a thinner
# handle) with the thumb held out of the way, so the bar is not driven into the palm.
SHAPES["bar_hook"] = dict(SHAPES["hook"], mcp=0.95, pm=1.25, md=0.85,
                          cmc_flexion=0.2, cmc_abduction=0.9, mp_flexion=0.0, ip_flexion=0.0)
# Overhand grip on the bar: fingers round the far side and under, thumb under it from the
# near side, so the loop is closed and the bar can turn in the hand but not roll out.
SHAPES["bar_fist"] = dict(SHAPES["hook"], mcp=1.25, pm=1.45, md=1.0,
                          cmc_flexion=0.70, cmc_abduction=0.78, mp_flexion=0.69, ip_flexion=-0.61)
SHAPES["cup_open"] = dict(SHAPES["open"], wrist=-0.40)   # release a wrapped object without swinging the forearm


def minjerk(u):
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Index finger straight, the others curled into the palm, thumb tucked: a fingertip for
# buttons, dials and zip sliders. 'point_curl' bends the index slightly (pad forward).
SHAPES["point"] = dict(mcp=(0.10, 1.35, 1.40, 1.40), pm=(0.10, 1.50, 1.55, 1.55), md=(0.05, 1.10, 1.10, 1.10),
                       abd=(0.0, 0.0, 0.0, 0.0), cmc_flexion=0.70, cmc_abduction=-0.50, mp_flexion=-0.78, ip_flexion=0.43)
SHAPES["point_curl"] = dict(SHAPES["point"], mcp=(0.45, 1.35, 1.40, 1.40), pm=(0.45, 1.50, 1.55, 1.55),
                            md=(0.25, 1.10, 1.10, 1.10))
# Thumb-index pinch, the other fingers curled. Joint values solved on the hand model for
# the distance between the thumb's and the index finger's distal collision capsules
# (solve: mj_geomDistance): pinch -2 mm (squeezes whatever is between the pads),
# pinch_mid 12 mm, pinch_open 30 mm; the pads meet 34 cm along the forearm.
_CURL = dict(mcp=(None, 1.35, 1.40, 1.40), pm=(None, 1.50, 1.55, 1.55), md=(None, 1.10, 1.10, 1.10))


def _pinch(cmc_f, cmc_a, mp, ip, mcp2, pm2, md2):
    return dict(mcp=(mcp2, 1.35, 1.40, 1.40), pm=(pm2, 1.50, 1.55, 1.55), md=(md2, 1.10, 1.10, 1.10),
                abd=(0.0, 0.0, 0.0, 0.0), cmc_flexion=cmc_f, cmc_abduction=cmc_a, mp_flexion=mp, ip_flexion=ip)


SHAPES["pinch"] = _pinch(-0.193, 0.104, -0.139, -0.471, 0.822, 0.795, 0.787)
SHAPES["pinch_mid"] = _pinch(-0.081, 0.129, -0.067, -0.445, 0.792, 0.784, 0.786)
SHAPES["pinch_open"] = _pinch(0.068, 0.157, 0.019, -0.410, 0.774, 0.778, 0.778)
SHAPES["pinch_wide"] = _pinch(0.341, 0.223, 0.179, -0.360, 0.652, 0.704, 0.734)    # 65 mm: round a buckle
for _k in range(1, 5):                     # wide -> squeezed, for closing on something wide
    SHAPES[f"pinch_w{_k}"] = {kk: (tuple(a + (b - a) * _k / 5 for a, b in zip(SHAPES["pinch_wide"][kk], SHAPES["pinch"][kk]))
                                    if isinstance(SHAPES["pinch"][kk], tuple) else
                                    SHAPES["pinch_wide"][kk] + (SHAPES["pinch"][kk] - SHAPES["pinch_wide"][kk]) * _k / 5)
                               for kk in SHAPES["pinch"]}


def shape_targets(shape, side):
    sh = SHAPES[shape]
    out = {}
    per = lambda v, i: v[i] if isinstance(v, (tuple, list)) else v
    for i, k in enumerate("2345"):
        out[f"mcp{k}_flexion_{side}"] = per(sh["mcp"], i)
        out[f"pm{k}_flexion_{side}"] = per(sh["pm"], i)
        out[f"md{k}_flexion_{side}"] = per(sh["md"], i)
        out[f"mcp{k}_abduction_{side}"] = sh["abd"]["2345".index(k)]
    for k in ["cmc_flexion", "cmc_abduction", "mp_flexion", "ip_flexion"]:
        out[f"{k}_{side}"] = sh[k]
    out[f"flexion_{side}"] = sh.get("wrist", 0.0)
    out[f"deviation_{side}"] = 0.0
    return out


class Runner:
    def __init__(self, seed=7, with_cloth=False, xml=None):
        # xml: a model already built (the 3D page ships build()'s output as play/pack/physics_hands.xml)
        self.m, self.xml = (B.build(with_cloth=with_cloth) if xml is None else (mujoco.MjModel.from_xml_string(xml), None))
        self.with_cloth = with_cloth
        m = self.m
        self.d = mujoco.MjData(m)
        d = self.d
        # Open lid lying flat on the mattress beside the base.
        d.qpos[m.joint("lid_hinge").qposadr[0]] = np.pi
        mujoco.mj_forward(m, d)
        self.aid = {m.actuator(i).name: i for i in range(m.nu)}
        self.carriage0 = {s: m.body(f"carriage_{s}").pos.copy() for s in "rl"}
        self._build_wrist_table()
        self.radius = {s: m.body(f"radius_{s}").id for s in "rl"}
        self.hand_bodies = {s: self._subtree(self.radius[s]) for s in "rl"}
        self.item_ids = {it.name: m.body(it.name).id for it in S.ITEMS}
        # compliant contacts (penetration is deflection): folded piles and the buckle's sprung prongs
        self.soft_ids = {m.body(it.name).id for it in S.ITEMS if it.soft} | {m.body("buckle_male").id}
        self.suit_ids = [m.body("suitcase").id, m.body("suitcase_lid").id]
        self.frames, self.events, self.stages = [], [], []
        self.diag = dict(max_penetration=0.0, worst_penetration=None, max_joint_excursion=0.0, worst_joint=None,
                         max_carriage_force=0.0, max_finger_torque_ratio=0.0, hand_item_welds=0)
        self.label = "Start"
        self.step_count = 0
        # Commanded state
        self.root = dict(x=S.NAV_NODES["suitcase"][0], y=S.NAV_NODES["suitcase"][1], yaw=np.pi / 2, h=0.95)
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
        self._init_mechanisms()
        # The 3D page runs this module in Pyodide over MuJoCo's WebAssembly build, whose
        # stand-in mujoco module offers `batch`: open-loop motions (move, hold, walk, turn)
        # collect their per-step commands and run as one batch there (play/runner_batch.js
        # does what step() does), instead of crossing between Python and JavaScript every ms.
        self._batch = getattr(mujoco, "batch", None)
        self._plan = None

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
    def _apply_values(self):
        """The carriage, rotation and finger commands for the current targets."""
        m = self.m
        ids, vals, quats = [], [], {}
        for s in "rl":
            p, Rp = self.hand[s]
            prel, Rrel = self.wrist_rel(s, self.finger[s][f"flexion_{s}"])
            R = Rp @ Rrel.T
            elbow = p - R @ prel
            off = elbow - self.carriage0[s]
            for k, ax in enumerate("xyz"):
                ids.append(self.aid[f"w{s}_{ax}"]); vals.append(off[k])
            self.rot_target[s] = R
            xyzw = Rot.from_matrix(R).as_quat()
            quats[s] = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
            for j, v in self.finger[s].items():
                a = self.aid[f"{j}_servo"]
                lo, hi = m.actuator_ctrlrange[a]
                ids.append(a); vals.append(np.clip(v, lo, hi))
        return ids, vals, quats

    def _apply(self):
        ids, vals, quats = self._apply_values()
        for a, v in zip(ids, vals):
            self.d.ctrl[a] = v
        self.rot_target_q.update(quats)

    def _monitor(self):
        m, d = self.m, self.d
        for c in d.contact[: d.ncon]:
            # Folded garment piles are deliberately compliant: their contacts are tracked apart.
            b1, b2 = self.contact_bodies(c)
            soft = b1 in self.soft_ids or b2 in self.soft_ids or c.geom1 < 0 or c.geom2 < 0   # cloth is soft too
            key = "max_penetration_soft" if soft else "max_penetration"
            if -c.dist > self.diag.get(key, 0.0):
                self.diag[key] = float(-c.dist)
                self.diag["worst_penetration_soft" if soft else "worst_penetration"] = dict(
                    t=round(d.time, 4), stage=self.label,
                    geoms=[m.geom(g).name if g >= 0 else "cloth" for g in (c.geom1, c.geom2)])
        for j in self._anat_joints:
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
        if self._plan is not None:                     # batched (3D page): collect, run later in one go
            for _ in range(n):
                k = self.step_count + len(self._plan)
                ctrl = self._apply_values() if k % CTRL_EVERY == 0 else None
                rec = (dict(self.root), self.gaze.tolist(), self.label) if (k + 1) % FRAME_EVERY == 0 else None
                self._plan.append((ctrl, rec))
            return
        m, d = self.m, self.d
        for _ in range(n):
            if self.step_count % CTRL_EVERY == 0:
                self._apply()
                self._mechanisms()
            self._detents()
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
                                zip=bool(getattr(self, "zip_locked", False)), hw=self.hardware_reading(),
                                **({"cloth": np.round(d.flexvert_xpos, 4).tolist()} if self.with_cloth else {})))

    # ------------------------------------------------------------ suitcase mechanisms
    def _init_mechanisms(self):
        m = self.m
        hand = set(self.hand_bodies["r"]) | set(self.hand_bodies["l"])
        self._anat_joints = [j for j in range(m.njnt) if m.jnt_bodyid[j] in hand or m.joint(j).name == "lid_hinge"]
        jid = lambda n: m.joint(n).id
        self.hw = dict(
            dial=[(jid(f"dial_{k}_hinge"), int(m.jnt_dofadr[jid(f"dial_{k}_hinge")]), int(m.jnt_qposadr[jid(f"dial_{k}_hinge")]))
                  for k in (1, 2, 3)],
            trolley=(jid("trolley_slide"), int(m.jnt_qposadr[jid("trolley_slide")])),
            trolley_btn=int(m.jnt_qposadr[jid("trolley_btn_slide")]),
            lock_btn=int(m.jnt_qposadr[jid("lock_btn_slide")]),
            slider={k: (jid(f"slider_{k}_slide"), int(m.jnt_qposadr[jid(f"slider_{k}_slide")])) for k in "ab"},
            tab={k: (jid(f"tab_{k}_hinge"), int(m.jnt_qposadr[jid(f"tab_{k}_hinge")])) for k in "ab"},
            buckle=(jid("buckle_slide"), int(m.jnt_qposadr[jid("buckle_slide")])),
            buckle_tabs=[m.geom(f"buckle_tab_{k}").id for k in (0, 1)],
            lid=jid("lid_hinge"),
        )
        self.hw_range0 = m.jnt_range.copy()
        tr = S.HARDWARE["trolley"]
        self.mech = dict(trolley_level=0, tabs={"a": False, "b": False}, tab_block={}, buckle=True, code="000", lock_events=0)
        self._latch_trolley(0)
        self._latch_buckle(True)

    def _latch_trolley(self, level):
        j, _ = self.hw["trolley"]
        if level is None:
            self.m.jnt_range[j] = self.hw_range0[j]
        else:
            q = S.HARDWARE["trolley"]["levels"][level]
            self.m.jnt_range[j] = [q - 0.0005, q + 0.0005]
        self.mech["trolley_level"] = level

    def _latch_tab(self, k, on):
        jt, _ = self.hw["tab"][k]
        js, qs = self.hw["slider"][k]
        if on:
            self.m.jnt_range[jt] = [np.pi - 0.04, np.pi]
            q = self.d.qpos[qs]
            self.m.jnt_range[js] = [q - 0.0005, q + 0.0005]
        else:
            self.m.jnt_range[jt] = self.hw_range0[jt]
            self.m.jnt_range[js] = self.hw_range0[js]
        self.mech["tabs"] = dict(self.mech["tabs"], **{k: on})

    def _latch_buckle(self, on):
        j, _ = self.hw["buckle"]
        self.m.jnt_range[j] = [0.0, 0.0008] if on else self.hw_range0[j]
        self.mech["buckle"] = on

    def dial_digits(self):
        return [int(round(self.d.qpos[qa] / (2 * np.pi / 10))) % 10 for _, _, qa in self.hw["dial"]]

    def zip_closed_fraction(self):
        """Fraction of the reachable zip track closed, from the two slider positions."""
        out = []
        for k in "ab":
            j, qa = self.hw["slider"][k]
            lo, hi = self.hw_range0[j]
            span = hi - lo
            q = self.d.qpos[qa]
            out.append((q - lo) / span if k == "a" else (hi - q) / span)
        return float(np.clip(np.mean(out), 0, 1))

    def hardware_reading(self):
        d = self.d
        return dict(dials=self.dial_digits(), trolley=round(float(d.qpos[self.hw["trolley"][1]]), 4),
                    trolley_level=self.mech["trolley_level"], zip_closed=round(self.zip_closed_fraction(), 3),
                    tabs={k: round(float(d.qpos[self.hw["tab"][k][1]]), 3) for k in "ab"},
                    docked=dict(self.mech["tabs"]), buckle=round(float(d.qpos[self.hw["buckle"][1]]), 4),
                    buckled=self.mech["buckle"])

    def _detents(self):
        """Ten detents per turn on each lock dial (a spring-loaded ball in a notched wheel)."""
        K = S.HARDWARE["dial_detent"]
        for _, dof, qa in self.hw["dial"]:
            self.d.qfrc_applied[dof] = -K * np.sin(10 * self.d.qpos[qa])

    def _mechanisms(self):
        m, d, hw = self.m, self.d, self.hw
        # trolley latch: free while the grip button is held in, catches at the next stop
        tr = S.HARDWARE["trolley"]
        q = d.qpos[hw["trolley"][1]]
        bq = d.qpos[hw["trolley_btn"]]
        if bq > 0.75 * tr["button"]["travel"]:
            if self.mech["trolley_level"] is not None:
                self._latch_trolley(None)
                self.event("trolley_unlatched", q=round(float(q), 4))
        elif self.mech["trolley_level"] is None and bq < 0.3 * tr["button"]["travel"]:
            # the latch pin drops back only once the button is let out (hysteresis)
            for i, lv in enumerate(tr["levels"]):
                if abs(q - lv) < 0.003:
                    self._latch_trolley(i)
                    self.event("trolley_latched", level=i, q=round(float(q), 4))
                    break
        # pull tabs: latch when flipped up into the dock with the slider at the dock
        for k in "ab":
            jt, qt = hw["tab"][k]
            js, qs = hw["slider"][k]
            lo, hi = self.hw_range0[js]
            at_dock = abs(d.qpos[qs] - (hi if k == "a" else lo)) < 0.004
            if self.mech["tab_block"].get(k) and d.qpos[qt] < np.pi - 0.4:
                self.mech["tab_block"] = dict(self.mech["tab_block"], **{k: False})   # it has dropped out of the slot
            if not self.mech["tabs"][k] and not self.mech["tab_block"].get(k) and at_dock and d.qpos[qt] > np.pi - 0.10:
                self._latch_tab(k, True)
                self.event("pull_docked", tab=k)
        # release button: frees the pulls only if the dials show the code
        if d.qpos[hw["lock_btn"]] > 0.7 * S.HARDWARE["button"]["travel"]:
            if any(self.mech["tabs"].values()):
                if "".join(map(str, self.dial_digits())) == self.mech["code"]:
                    for k in "ab":
                        self._latch_tab(k, False)
                    self.mech["tab_block"] = {"a": True, "b": True}
                    self.event("pulls_released", dials=self.dial_digits())
                elif self.mech["lock_events"] < 1:
                    self.mech["lock_events"] += 1
                    self.event("lock_button_blocked", dials=self.dial_digits())
        else:
            self.mech["lock_events"] = 0
        # buckle: clicks shut when pushed home; opens when both side tabs are squeezed
        jb, qb = hw["buckle"]
        if not self.mech["buckle"] and d.qpos[qb] < 0.0015:
            self._latch_buckle(True)
            self.event("buckle_latched")
        elif self.mech["buckle"]:
            squeezed = 0
            hand = set(self.hand_bodies["r"]) | set(self.hand_bodies["l"])
            f = np.zeros(6)
            for i, c in enumerate(d.contact[: d.ncon]):
                g = c.geom1 if c.geom1 in hw["buckle_tabs"] else c.geom2 if c.geom2 in hw["buckle_tabs"] else -1
                other = c.geom2 if g == c.geom1 else c.geom1
                if g >= 0 and m.geom_bodyid[other] in hand:
                    mujoco.mj_contactForce(m, d, i, f)
                    if abs(f[0]) > 1.5:
                        squeezed |= 1 << hw["buckle_tabs"].index(g)
            if squeezed == 3:
                self._latch_buckle(False)
                self.event("buckle_released")
        # idealised zipper: the lid is held shut once part of the zip is closed and the
        # lid is down (closure check); it is freed when the zip is opened again
        zc = self.zip_closed_fraction()
        lid_q = d.qpos[m.jnt_qposadr[hw["lid"]]]
        if not self.zip_locked and zc > 0.05 and lid_q < np.radians(2.5):
            m.jnt_range[hw["lid"]] = [0.0, 0.003]
            self.zip_locked = True
            self.event("zipper_engaged", lid_angle_deg=round(float(np.degrees(lid_q)), 2), closed_fraction=round(zc, 3))
        elif self.zip_locked and zc < 0.04:
            m.jnt_range[hw["lid"]] = self.hw_range0[hw["lid"]]
            self.zip_locked = False
            self.event("zipper_released", lid_angle_deg=round(float(np.degrees(lid_q)), 2))
        self.zip_progress = zc

    def contact_bodies(self, c):
        """Bodies of a contact's two sides (flex contacts name a vertex, not a geom)."""
        m = self.m
        out = []
        for k in (0, 1):
            g = c.geom[k]
            if g >= 0:
                out.append(int(m.geom_bodyid[g]))
            else:
                f, v = c.flex[k], c.vert[k]
                out.append(int(m.flex_vertbodyid[m.flex_vertadr[f] + v]) if f >= 0 and v >= 0 else -1)
        return out[0], out[1]

    def cloth_vertices(self):
        return self.d.flexvert_xpos.reshape(-1, 3).copy()

    def fingertip(self, side="r", finger=2):
        """Centre of the end sphere of a finger's distal collision capsule (radius
        TIP_R): the fingertip that touches buttons, dials and sliders."""
        g = self.m.geom(f"distph{finger}_coll_{side}" if finger != 1 else f"distal_thumb_coll_{side}").id
        c = self.d.geom_xpos[g]
        ax = self.d.geom_xmat[g].reshape(3, 3)[:, 2] * self.m.geom_size[g][1]
        b = self.m.body(f"midph{finger}_{side}" if finger != 1 else f"proximal_thumb_{side}").id
        # the distal end is the one away from the middle phalanx
        return c + ax if np.linalg.norm(c + ax - self.d.xpos[b]) > np.linalg.norm(c - ax - self.d.xpos[b]) else c - ax

    # ------------------------------------------------------------ primitives
    def stage(self, label):
        self.label = label
        self.stages.append(dict(t=round(self.d.time, 3), title=label))
        print(f"{self.d.time:8.2f}s  {label}", flush=True)

    def event(self, kind, **kw):
        self.events.append(dict(t=round(self.d.time, 4), stage=self.label, event=kind, **kw))

    def _batched(self, fn, *a, **kw):
        """Run a motion: stepped here, or collected and run as one batch (see __init__)."""
        if self._batch is None or self._plan is not None:
            return fn(*a, **kw)
        self._plan = []
        try:
            out = fn(*a, **kw)
        finally:
            plan, self._plan = self._plan, None
        if plan:
            self._batch(self, plan)
        return out

    def move(self, *a, **kw):
        return self._batched(self._move, *a, **kw)

    def hold(self, T):
        return self._batched(self.step, int(round(T / PHYS_DT)))

    def walk(self, *a, **kw):
        return self._batched(self._walk, *a, **kw)

    def turn(self, *a, **kw):
        return self._batched(self._turn, *a, **kw)

    def _move(self, T, hands=None, root=None, fingers=None, gaze=None, frame="world", follow=()):
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

    def _walk(self, waypoints, speed=0.85, carry=True, gaze_ahead=True, final_yaw=None, yaw_rate=3.0, yaw_acc=8.0):
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
            self._turn(final_yaw)

    def _turn(self, yaw, T=None, direction=None):
        """Turn in place, carrying the hands along. direction=+1/-1 forces the way round."""
        y0 = self.root["yaw"]
        dyw = (yaw - y0 + np.pi) % (2 * np.pi) - np.pi
        if direction is not None and np.sign(dyw) != direction and abs(dyw) > 1e-3:
            dyw += 2 * np.pi * direction
        dy = abs(dyw)
        if dy < 1e-3:
            return
        o, Rr = self.root_frame()
        local = {s: (Rr.T @ (self.hand[s][0] - o), Rr.T @ self.hand[s][1]) for s in "rl"}
        T = T or max(0.6, dy / 1.6)
        y1 = y0 + dyw
        n = int(round(T / PHYS_DT))
        for k in range(n):
            self.root["yaw"] = y0 + (y1 - y0) * minjerk((k + 1) / n)
            o, Rr = self.root_frame()
            for s in "rl":
                self.hand[s] = (o + Rr @ local[s][0], Rr @ local[s][1])
            self.step()

    # ------------------------------------------------------------ rewind
    def snapshot(self):
        """Light checkpoint (physics + commanded state + log lengths) for rewinding an action."""
        d, m = self.d, self.m
        return dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), act=d.act.copy(), ctrl=d.ctrl.copy(), time=d.time,
                    qacc_warmstart=d.qacc_warmstart.copy(), jnt_range=m.jnt_range.copy(), mech=dict(self.mech),
                    root=dict(self.root), hand={s: (self.hand[s][0].copy(), self.hand[s][1].copy()) for s in "rl"},
                    finger={s: dict(self.finger[s]) for s in "rl"}, gaze=self.gaze.copy(), zip=self.zip_progress,
                    zip_locked=self.zip_locked, label=self.label, step_count=self.step_count,
                    n=(len(self.frames), len(self.events), len(self.stages)))

    def restore(self, snap):
        d, m = self.d, self.m
        d.qpos[:] = snap["qpos"]; d.qvel[:] = snap["qvel"]; d.act[:] = snap["act"]; d.ctrl[:] = snap["ctrl"]
        d.time = snap["time"]; d.qacc_warmstart[:] = snap["qacc_warmstart"]
        m.jnt_range[:] = snap["jnt_range"]
        self.mech = dict(snap["mech"])
        self.root = dict(snap["root"]); self.hand = {s: (v[0].copy(), v[1].copy()) for s, v in snap["hand"].items()}
        self.finger = {s: dict(v) for s, v in snap["finger"].items()}; self.gaze = snap["gaze"].copy()
        self.zip_progress, self.zip_locked = snap["zip"], snap["zip_locked"]
        self.label, self.step_count = snap["label"], snap["step_count"]
        nf, ne, ns = snap["n"]
        del self.frames[nf:]; del self.events[ne:]; del self.stages[ns:]
        mujoco.mj_forward(m, d)

    # ------------------------------------------------------------ queries
    def contacts_between(self, bodies_a, bodies_b):
        m, d = self.m, self.d
        sa, sb = set(bodies_a), set(bodies_b)
        n, force = 0, 0.0
        f = np.zeros(6)
        for i, c in enumerate(d.contact[: d.ncon]):
            b1, b2 = self.contact_bodies(c)
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
                b1, b2 = self.contact_bodies(c)
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
