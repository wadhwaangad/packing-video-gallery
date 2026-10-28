"""The part of the `mujoco` Python API the hand pipeline uses (runner.py, tasks.py,
skills_case.py, skills_hw.py, executor.py), over MuJoCo's official WebAssembly build,
for running the videos' physics unchanged inside Pyodide (the 3D page's worker).

play/worker.js loads @mujoco/mujoco, registers it as the module `mjjs` and installs this
file as `mujoco`. State crosses by copy: what Python writes (qpos, qvel, act, ctrl,
qfrc_applied, qacc_warmstart, time, jnt_range) is pushed before every mj_step/mj_forward
and read back after it; what it only reads (poses, velocities, contacts) is fetched on
first use after each step.
"""
import numpy as np
import js, mjjs
from pyodide.ffi import to_js

MJ = mjjs.mj


class mjtObj:
    mjOBJ_BODY = 1
    mjOBJ_JOINT = 3
    mjOBJ_GEOM = 5
    mjOBJ_EQUALITY = 17
    mjOBJ_ACTUATOR = 19


_ENUM = {1: "mjOBJ_BODY", 3: "mjOBJ_JOINT", 5: "mjOBJ_GEOM", 17: "mjOBJ_EQUALITY", 19: "mjOBJ_ACTUATOR"}


def _arr(js, dtype=float, shape=None):
    a = np.array(js.to_py(), dtype=dtype)
    return a.reshape(shape) if shape else a


def _push(dst, src):
    dst.set(to_js(np.ascontiguousarray(src, dtype=np.float64).ravel()))


class _Named:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class MjModel:
    def __init__(self, js):
        self.js = js
        self.nbody, self.njnt, self.ngeom, self.nu = js.nbody, js.njnt, js.ngeom, js.nu
        self.nq, self.nv, self.nflex = js.nq, js.nv, js.nflex
        i = lambda k: _arr(getattr(js, k), np.int64)
        self.jnt_qposadr, self.jnt_dofadr, self.jnt_bodyid = i("jnt_qposadr"), i("jnt_dofadr"), i("jnt_bodyid")
        self.jnt_type, self.jnt_limited = i("jnt_type"), i("jnt_limited")
        self.jnt_range = _arr(js.jnt_range, shape=(self.njnt, 2))
        self.geom_bodyid, self.geom_size = i("geom_bodyid"), _arr(js.geom_size, shape=(self.ngeom, 3))
        self.body_parentid, self.body_mocapid, self.body_weldid = i("body_parentid"), i("body_mocapid"), i("body_weldid")
        self.body_mass, self.body_pos = _arr(js.body_mass), _arr(js.body_pos, shape=(self.nbody, 3))
        self.actuator_ctrlrange = _arr(js.actuator_ctrlrange, shape=(self.nu, 2))
        self.flex_vertadr = i("flex_vertadr") if self.nflex else np.zeros(0, np.int64)
        self.flex_vertbodyid = i("flex_vertbodyid") if self.nflex else np.zeros(0, np.int64)
        self.qpos0 = _arr(js.qpos0)
        self.opt = _Named(timestep=js.opt.timestep)
        self._names = {}
        for t, n in ((1, self.nbody), (3, self.njnt), (5, self.ngeom), (19, self.nu)):
            names = [MJ.mj_id2name(js, t, k) for k in range(n)]
            self._names[t] = (names, {nm: k for k, nm in enumerate(names) if nm})

    @staticmethod
    def from_xml_string(xml):
        return MjModel(MJ.MjModel.from_xml_string(xml))

    @staticmethod
    def from_xml_path(path):
        return MjModel.from_xml_string(open(path).read())

    def _id(self, t, key):
        names, ids = self._names[t]
        k = key if isinstance(key, (int, np.integer)) else ids[key]
        return int(k), names[int(k)]

    def body(self, key):
        k, nm = self._id(1, key)
        return _Named(id=k, name=nm, pos=self.body_pos[k], mass=self.body_mass[k:k + 1])

    def joint(self, key):
        k, nm = self._id(3, key)
        return _Named(id=k, name=nm, qposadr=self.jnt_qposadr[k:k + 1], dofadr=self.jnt_dofadr[k:k + 1])

    def geom(self, key):
        k, nm = self._id(5, key)
        return _Named(id=k, name=nm)

    def actuator(self, key):
        k, nm = self._id(19, key)
        return _Named(id=k, name=nm)


class _Contact:
    __slots__ = ("geom1", "geom2", "dist", "geom", "flex", "vert")


# read after each step on first use: name -> (columns or None)
_LAZY = dict(xpos=3, xmat=9, xquat=4, cvel=6, geom_xpos=3, geom_xmat=9, flexvert_xpos=3)
_STATE = ("qpos", "qvel", "act", "ctrl", "qfrc_applied", "qacc_warmstart")


class MjData:
    def __init__(self, m):
        self.m = m
        self.js = MJ.MjData.new(m.js)
        for k in _STATE:
            setattr(self, k, _arr(getattr(self.js, k)))
        self.time = float(self.js.time)
        self._stamp = 0
        self._cache = {}

    def _pull(self):
        for k in _STATE:
            getattr(self, k)[:] = _arr(getattr(self.js, k))
        self.time = float(self.js.time)
        self._stamp += 1
        self._cache = {}

    def _push(self):
        for k in _STATE:
            _push(getattr(self.js, k), getattr(self, k))
        self.js.time = self.time
        _push(self.m.js.jnt_range, self.m.jnt_range)

    def __getattr__(self, k):
        if k in _LAZY:
            c = self.__dict__["_cache"]
            if k not in c:
                if k == "flexvert_xpos" and not self.m.nflex:
                    c[k] = np.zeros((0, 3))
                else:
                    c[k] = _arr(getattr(self.js, k), shape=(-1, _LAZY[k]))
            return c[k]
        if k in ("ncon", "contact"):
            c = self.__dict__["_cache"]
            if "contact" not in c:
                n = int(self.js.ncon)
                vec, out = self.js.contact, []
                for i in range(n):
                    j = vec.get(i)
                    o = _Contact()
                    o.geom1, o.geom2, o.dist = int(j.geom1), int(j.geom2), float(j.dist)
                    o.geom = (o.geom1, o.geom2)
                    fl, vt = j.flex, j.vert
                    o.flex, o.vert = (int(fl[0]), int(fl[1])), (int(vt[0]), int(vt[1]))
                    out.append(o)
                    j.delete()
                c["contact"], c["ncon"] = out, n
            return c[k]
        raise AttributeError(k)


def mj_step(m, d):
    d._push()
    MJ.mj_step(m.js, d.js)
    d._pull()


def mj_forward(m, d):
    d._push()
    MJ.mj_forward(m.js, d.js)
    d._pull()


def mj_resetData(m, d):
    MJ.mj_resetData(m.js, d.js)
    d._pull()


def mj_contactForce(m, d, i, f):
    out = to_js(np.zeros(6))
    MJ.mj_contactForce(m.js, d.js, int(i), out)
    f[:] = np.array(out.to_py())


def mj_geomDistance(m, d, g1, g2, distmax, fromto=None):
    out = to_js(np.zeros(6))
    r = MJ.mj_geomDistance(m.js, d.js, int(g1), int(g2), float(distmax), out)
    if fromto is not None:
        fromto[:] = np.array(out.to_py())
    return float(r)


def mj_ray(m, d, pnt, vec, geomgroup, flg_static, bodyexclude, geomid):
    gid = to_js(np.zeros(1, np.int32))
    group = to_js(np.ones(6, np.uint8) if geomgroup is None else np.asarray(geomgroup, np.uint8))
    r = MJ.mj_ray(m.js, d.js, to_js(np.asarray(pnt, float)), to_js(np.asarray(vec, float)), group,
                  bool(flg_static), int(bodyexclude), gid, None)
    geomid[0] = int(gid.to_py()[0])
    return float(r)


def mj_id2name(m, t, i):
    return MJ.mj_id2name(m.js, int(t), int(i)) or None


# ---------------------------------------------------------------- Runner motions in one go
def _cfg(r):
    """What play/runner_batch.js needs from the Runner, fixed for the run."""
    import hands as H, scene as S
    hw = r.hw
    return dict(
        rotQadr=np.array([r.rot_qadr[s] for s in "rl"], np.int32),
        rotAid=np.array([a for s in "rl" for a in r.rot_aid[s]], np.int32),
        rotKp=float(H.ROT_KP), torqueCap=float(H.CARRIAGE_TORQUE),
        detentDof=np.array([dof for _, dof, _ in hw["dial"]], np.int32),
        detentQadr=np.array([qa for _, _, qa in hw["dial"]], np.int32),
        detentK=float(S.HARDWARE["dial_detent"]),
        hw=dict(trolley=[int(v) for v in hw["trolley"]], trolley_btn=int(hw["trolley_btn"]), lock_btn=int(hw["lock_btn"]),
                slider={k: [int(v) for v in hw["slider"][k]] for k in "ab"}, tab={k: [int(v) for v in hw["tab"][k]] for k in "ab"},
                buckle=[int(v) for v in hw["buckle"]], dial=[[int(v) for v in t] for t in hw["dial"]], lid=int(hw["lid"])),
        range0=np.ascontiguousarray(r.hw_range0, np.float64).ravel(),
        trolleyLevels=[float(v) for v in S.HARDWARE["trolley"]["levels"]],
        trolleyBtnTravel=float(S.HARDWARE["trolley"]["button"]["travel"]),
        lockBtnTravel=float(S.HARDWARE["button"]["travel"]),
        buckleTabs=[int(g) for g in hw["buckle_tabs"]],
        handBodies=[int(b) for b in r.hand_bodies["r"] + r.hand_bodies["l"]])


def batch(r, plan):
    """Run the steps a Runner motion collected (see Runner._batched) in JavaScript."""
    m, d = r.m, r.d
    if not hasattr(r, "_batch_cfg"):
        r._batch_cfg = _cfg(r)
    cfg = dict(r._batch_cfg)
    cfg.update(mech=dict(r.mech, tabs=dict(r.mech["tabs"]), tab_block=dict(r.mech["tab_block"])),
               zipLocked=bool(r.zip_locked), zipProgress=float(r.zip_progress),
               rotQ=np.array([v for s in "rl" for v in r.rot_target_q[s]], np.float64))
    ctrl_at = [k for k, (c, _) in enumerate(plan) if c is not None]
    rec_at = [k for k, (_, rc) in enumerate(plan) if rc is not None]
    ids = plan[ctrl_at[0]][0][0] if ctrl_at else []
    vals = np.array([plan[k][0][1] for k in ctrl_at], np.float64).ravel() if ctrl_at else np.zeros(0)
    quats = np.array([[*plan[k][0][2]["r"], *plan[k][0][2]["l"]] for k in ctrl_at], np.float64).ravel() if ctrl_at else np.zeros(0)
    spec = dict(n=len(plan), ctrlAt=np.array(ctrl_at, np.int32), ctrlIds=np.array(ids, np.int32), ctrlVals=vals,
                rotQs=quats, recAt=np.array(rec_at, np.int32), cfg=cfg)
    t0 = d.time
    d._push()
    res = mjjs.batch.run(MJ, m.js, d.js, to_js(spec, dict_converter=js.Object.fromEntries))
    d._pull()
    m.jnt_range[:] = _arr(m.js.jnt_range, shape=(m.njnt, 2))
    out = res.to_py()
    r.step_count += len(plan)
    for i, f in enumerate(out["frames"]):
        root, gaze, label = plan[rec_at[i]][1]
        r.frames.append(dict(t=f["t"], stage=label, xpos=np.array(out["xpos"][i], np.float32).reshape(-1, 3).tolist(),
                             xquat=np.array(out["xquat"][i], np.float32).reshape(-1, 4).tolist(), root=root, gaze=gaze,
                             lid=float(f["lid"]), zip_progress=float(f["zip_progress"]), zip=bool(f["zip"]), hw=f["hw"]))
    dt = m.opt.timestep
    for k, name, kw in out["events"]:
        r.events.append(dict(t=round(t0 + (k + 1) * dt, 4), stage=r.label, event=name, **kw))
    mech = out["mech"]
    r.mech = dict(mech, tabs=dict(mech["tabs"]), tab_block=dict(mech["tab_block"]))
    r.zip_locked, r.zip_progress = bool(out["zipLocked"]), float(out["zipProgress"])
    q = out["rotQ"]
    r.rot_target_q = {"r": tuple(q[:4]), "l": tuple(q[4:])}
