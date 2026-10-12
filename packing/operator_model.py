"""Kinematic operator body solved around the simulated forearms.

The forearm + hand of each side are the physical MuJoCo bodies. For every
recorded frame we solve the torso (pelvis position, yaw, forward lean, side tilt,
spine twist) so each shoulder lies exactly one upper-arm length from the simulated
elbow, while staying close to the planned stance. Legs are two-bone IK to feet
that are planted and stepped reactively. Segment lengths are fixed by construction
and audited afterwards. No balance, foot force or whole-body collision is simulated.
"""
import numpy as np
from scipy.optimize import least_squares

UPPER_ARM = 0.30
SHOULDER_HALF = 0.18
TORSO = 0.47          # hip-joint centre to shoulder line
HIP_HALF = 0.095
THIGH, SHIN = 0.43, 0.43
ANKLE_H = 0.075
NECK = 0.10
HEAD_UP = 0.13
EYE_FWD, EYE_UP = 0.085, 0.03


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def torso_R(q):
    return rz(q[3] + q[6]) @ ry(q[4]) @ rx(q[5])


def shoulders(q):
    R = torso_R(q)
    p = q[:3]
    return p + R @ np.array([0, -SHOULDER_HALF, TORSO]), p + R @ np.array([0, SHOULDER_HALF, TORSO])


class TorsoSolver:
    def __init__(self):
        self.q = None

    def solve(self, elbow_r, elbow_l, root, ankles=None):
        """root: dict(x, y, yaw, h). Returns q and arm-length residuals (m)."""
        yaw = root["yaw"]
        F = np.array([np.cos(yaw), np.sin(yaw), 0])
        h = root["h"]
        # Bending at the hips moves the pelvis back over the feet.
        mid = (elbow_r + elbow_l) / 2
        if self.q is None:
            self.q = np.array([root["x"], root["y"], h, yaw, 0.1, 0.0, 0.0])
        q0 = self.q.copy()
        q0[3] = yaw + (q0[3] - yaw + np.pi) % (2 * np.pi) - np.pi

        def res(q):
            sr, sl = shoulders(q)
            lean = q[4]
            back = -F * 0.30 * np.sin(max(lean, 0)) * TORSO
            prior_xy = np.array([root["x"], root["y"]]) + back[:2]
            # Standing pelvis height falls as the knees bend; keep within leg length.
            r = [
                60 * (np.linalg.norm(sr - elbow_r) - UPPER_ARM),
                60 * (np.linalg.norm(sl - elbow_l) - UPPER_ARM),
                1.5 * (q[0] - prior_xy[0]), 1.5 * (q[1] - prior_xy[1]),
                1.2 * (q[2] - h),
                1.0 * ((q[3] - yaw + np.pi) % (2 * np.pi) - np.pi),
                0.35 * q[4],
                1.5 * q[5],
                0.8 * q[6],
                # keep shoulders behind/above the elbows in a natural way
                0.4 * max(0.0, (sr + sl)[2] / 2 - mid[2] - 0.55),
            ]
            if ankles is not None:
                for i, sy in enumerate((-1, 1)):
                    hip = q[:3] + rz(q[3]) @ np.array([0, sy * HIP_HALF, -0.02])
                    r.append(80 * max(0.0, np.linalg.norm(hip - ankles[i]) - (THIGH + SHIN - 0.004)))
            return np.array(r)

        lo = [-np.inf, -np.inf, 0.45, -np.inf, -0.25, -0.35, -0.6]
        hi = [np.inf, np.inf, ANKLE_H + THIGH + SHIN + 0.02, np.inf, 1.35, 0.35, 0.6]
        q0 = np.clip(q0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        sol = least_squares(res, q0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=200)
        self.q = sol.x
        sr, sl = shoulders(sol.x)
        return sol.x, (np.linalg.norm(sr - elbow_r) - UPPER_ARM, np.linalg.norm(sl - elbow_l) - UPPER_ARM)


def two_bone(hip, foot, forward, l1=THIGH, l2=SHIN):
    """Knee position for a planar two-bone chain; knee bends towards `forward`."""
    d = foot - hip
    L = np.linalg.norm(d)
    L = min(L, l1 + l2 - 1e-6)
    u = d / np.linalg.norm(d)
    a = (l1 * l1 - l2 * l2 + L * L) / (2 * L)
    hgt = np.sqrt(max(l1 * l1 - a * a, 0.0))
    side = forward - u * (forward @ u)
    side /= np.linalg.norm(side) + 1e-12
    return hip + u * a + side * hgt


class Feet:
    """Footsteps planned with look-ahead on the known root trajectory.

    A foot steps (one at a time) when its planted position is far from where its
    home under the hip will be one swing duration later."""
    SWING = 0.34

    def __init__(self):
        self.pos = None
        self.yaw = None
        self.swing = {0: None, 1: None}

    @staticmethod
    def home(root, side):
        yaw = root["yaw"]
        lat = np.array([-np.sin(yaw), np.cos(yaw), 0]) * (0.115 if side else -0.115)
        return np.array([root["x"], root["y"], 0.0]) + lat

    def update(self, t, root, future):
        if self.pos is None:
            self.pos = {s: self.home(root, s) for s in (0, 1)}
            self.yaw = {s: root["yaw"] for s in (0, 1)}
        out = {}
        for s in (0, 1):
            sw = self.swing[s]
            if sw is not None:
                u = (t - sw["t0"]) / self.SWING
                if u >= 1:
                    self.pos[s] = sw["p1"]; self.yaw[s] = sw["y1"]; self.swing[s] = None
                else:
                    e = u * u * (3 - 2 * u)
                    p = sw["p0"] + (sw["p1"] - sw["p0"]) * e + np.array([0, 0, 0.08 * np.sin(np.pi * u)])
                    out[s] = (p, sw["y0"] + (sw["y1"] - sw["y0"]) * e)
                    continue
            out[s] = (self.pos[s], self.yaw[s])
        if self.swing[0] is None and self.swing[1] is None:
            errs = []
            for s in (0, 1):
                hm = self.home(future, s)
                err = np.linalg.norm((hm - self.pos[s])[:2])
                dyaw = abs((future["yaw"] - self.yaw[s] + np.pi) % (2 * np.pi) - np.pi)
                errs.append((err + 0.25 * dyaw, s, hm))
            e, s, hm = max(errs)
            if e > 0.07:
                self.swing[s] = dict(t0=t, p0=self.pos[s].copy(), p1=hm, y0=self.yaw[s],
                                     y1=self.yaw[s] + (future["yaw"] - self.yaw[s] + np.pi) % (2 * np.pi) - np.pi)
        return out


def body_frame(q, feet, elbows, gaze):
    """All joint centres needed to draw the body, plus the head camera pose."""
    R = torso_R(q)
    pelvis = q[:3]
    F = R[:, 0]
    hips = [pelvis + rz(q[3]) @ np.array([0, -HIP_HALF, -0.02]), pelvis + rz(q[3]) @ np.array([0, HIP_HALF, -0.02])]
    sr, sl = shoulders(q)
    neck = pelvis + R @ np.array([0, 0, TORSO + NECK * 0.5])
    head_base = pelvis + R @ np.array([0, 0, TORSO + NECK])
    # Head looks at the gaze point within neck limits.
    g = np.asarray(gaze) - head_base
    yaw_t = np.arctan2(g[1], g[0])
    tor_yaw = q[3] + q[6]
    hy = np.clip((yaw_t - tor_yaw + np.pi) % (2 * np.pi) - np.pi, -1.1, 1.1)
    pitch_t = np.arctan2(-g[2], np.linalg.norm(g[:2]))
    lean = q[4]
    hp = np.clip(pitch_t - lean, -0.6, 0.95)   # head pitch relative to the torso
    Rh = rz(tor_yaw + hy) @ ry(hp + lean)
    head = head_base + Rh @ np.array([0.0, 0, HEAD_UP * 0.5]) + R @ np.array([0, 0, HEAD_UP * 0.5]) * 0
    eye = head + Rh @ np.array([EYE_FWD, 0, EYE_UP])
    knees, ankles = [], []
    for i, s in enumerate((0, 1)):
        fp, fy = feet[s]
        ankle = fp + np.array([0, 0, ANKLE_H])
        fwd = np.array([np.cos(fy), np.sin(fy), 0])
        knees.append(two_bone(hips[i], ankle, fwd))
        ankles.append(ankle)
    return dict(pelvis=pelvis, R=R, hips=hips, shoulders=[sr, sl], elbows=elbows, neck=neck, head=head,
                head_R=Rh, eye=eye, knees=knees, ankles=ankles, feet=[feet[0], feet[1]])
