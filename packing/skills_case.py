"""Thermos wrap grasp, lid opening and closing, and the final suitcase lift.
(The zip, lock, strap and trolley-handle controls are in skills_hw.py.)"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import scene as S
from runner import minjerk
from tasks import unit, hand_R, facing, clear_hands, approach

W, L = S.SUITCASE["outer"]
D, DL, T_SHELL = S.SUITCASE["base_depth"], S.SUITCASE["lid_depth"], S.SUITCASE["shell"]
HH = S.SUITCASE["hinge_h"]
ZR = D - HH                       # lid rim height in the lid frame


# ---------------------------------------------------------------- thermos
def wrap_pick(r, name, side="r", lift=0.06, push=0.006):
    p, R = r.body_pose(name)
    F, right = facing(r)
    rad = S.item(name).size[0] / 2
    lat = -right if side == "l" else right          # hand sits on this side of the item
    f = F.copy()                                    # thumb axis exactly along the bottle axis
    Rh = hand_R(f, -lat)                            # palm faces the item
    zc = p[2] + 0.012         # grip just above the centre of mass so the bottle does not swing in the hand

    def palm(off, up=0.0):
        # Cylinder axis 1 cm distal of the palm centre, under the knuckles (dbg_cshape2.py).
        return np.array([p[0], p[1], zc + up]) + lat * (rad + off) - F * 0.010

    label = S.item(name).label
    r.stage(f"Reach for the {label}")
    approach(r, {side: (palm(0.006, up=0.20), Rh)}, fingers={side: "cup"}, gaze=p)
    r.move(1.0, hands={side: (palm(0.004), Rh)})
    r.stage(f"Wrap fingers around the {label}")
    r.move(0.6, fingers={side: "cup_thumb"})
    r.move(1.2, hands={side: (palm(0.004 - push), Rh)}, fingers={side: "wrap_big"})
    r.hold(0.3)
    z0 = r.body_pose(name)[0][2]
    r.stage(f"Lift the {label}")
    r.move(0.8, hands={side: (palm(0.004 - push, up=lift), Rh)})
    r.hold(0.3)
    rise = float(r.body_pose(name)[0][2] - z0)
    c = r.hand_item_contact(name)
    r.event("lift_check", items=[name], rise_m={name: rise}, contacts={s: c[s][0] for s in "rl"},
            normal_force_N={s: round(c[s][1], 2) for s in "rl"})
    if rise < lift * 0.6:
        r.debug_snapshot(f"liftfail_{name}")
        raise RuntimeError(f"wrap lift failed: {rise}")


def carry_one(r, name, side, height=1.0, forward=0.33, lateral=0.12):
    """Hold a one-hand item upright in front of the body, on the hand's side."""
    p, R = r.body_pose(name)
    hp, hR = r.hand[side]
    relp, relR = R.T @ (hp - p), R.T @ hR
    o, Rr = r.root_frame()
    pd = o + Rr @ np.array([forward, -lateral if side == "r" else lateral, height])
    return {side: (pd + R @ relp, R @ relR)}


def place_upright(r, name, p_slot, side="r", raise_h=0.08):
    """Lower the upright bottle into its slot, open the fingers and lift the hand straight up."""
    p, R = r.body_pose(name)
    hp, hR = r.hand[side]
    relp, relR = R.T @ (hp - p), R.T @ hR
    p_slot = np.asarray(p_slot, float)
    # Straighten the bottle (it may have tilted in the grip); it is round, so yaw is free.
    axis = R[:, 2]
    v = np.cross(axis, [0, 0, 1.0]); sn = np.linalg.norm(v)
    Rd = (Rot.from_rotvec(v / sn * np.arcsin(min(sn, 1.0))).as_matrix() @ R) if sn > 1e-6 else R
    r.event("bottle_tilt_before_place_deg", value=float(np.degrees(np.arcsin(min(sn, 1.0)))))
    label = S.item(name).label
    r.stage(f"Bring the {label} over the suitcase")
    r.move(1.6, hands={side: (p_slot + np.array([0, 0, raise_h]) + Rd @ relp, Rd @ relR)}, root=dict(h=0.84), gaze=p_slot)
    # re-measure and straighten once more right above the slot
    p, R = r.body_pose(name); hp, hR = r.hand[side]; relp, relR = R.T @ (hp - p), R.T @ hR
    axis = R[:, 2]; v = np.cross(axis, [0, 0, 1.0]); sn = np.linalg.norm(v)
    Rd = (Rot.from_rotvec(v / sn * np.arcsin(min(sn, 1.0))).as_matrix() @ R) if sn > 1e-6 else R
    r.move(0.6, hands={side: (p_slot + np.array([0, 0, raise_h]) + Rd @ relp, Rd @ relR)})
    r.stage(f"Stand the {label} in the suitcase")
    r.move(1.2, hands={side: (p_slot + np.array([0, 0, 0.003]) + Rd @ relp, Rd @ relR)})
    r.hold(0.2)
    # With the base on the floor, square it up: rotate the grip about the base until the
    # bottle stands vertical (it sits a little tilted in a one-hand grip).
    h = S.item(name).size[2]
    for _ in range(2):
        p, R = r.body_pose(name)
        axis = R[:, 2]; v = np.cross(axis, [0, 0, 1.0]); sn = np.linalg.norm(v)
        if sn < np.sin(np.radians(1.5)):
            break
        Rc = Rot.from_rotvec(v / sn * np.arcsin(min(sn, 1.0))).as_matrix()
        pivot = p - axis * h / 2
        hp, hR = r.hand[side]
        r.move(0.5, hands={side: (pivot + Rc @ (hp - pivot), Rc @ hR)})
        r.hold(0.15)
    r.event("bottle_tilt_at_release_deg", value=float(np.degrees(np.arccos(np.clip(r.body_pose(name)[1][2, 2], -1, 1)))))
    r.stage(f"Release the {label}")
    # Open slowly while the palm backs away from the bottle, then lift clear.
    hp, hR = r.hand[side]
    r.move(1.0, hands={side: (hp + hR[:, 2] * 0.015, hR)}, fingers={side: "cup_open"})
    hp, hR = r.hand[side]
    r.move(1.0, hands={side: (hp + np.array([0, 0, 0.22]), hR)})
    clear_hands(r)
    pp, _ = r.body_pose(name)
    r.event("placed", item=name, pos=pp.tolist(), speed=r.item_speed(name))


def place_lying(r, name, p_slot, side="r", raise_h=0.10, axis=None):
    """Rotate a wrapped cylinder to lie along `axis` (default: the suitcase length), palm on top, and set it down."""
    p, R = r.body_pose(name)
    hp, hR = r.hand[side]
    relp, relR = R.T @ (hp - p), R.T @ hR
    z_i = np.array([0, 0, 1.0])                   # cylinder axis in the item frame
    n_i = relR @ np.array([0, 0, -1.0])          # palm normal in the item frame
    n_i = unit(n_i - z_i * (n_i @ z_i))
    p_slot = np.asarray(p_slot, float)
    sp, sR = r.suitcase_frame()
    F, _ = facing(r)
    best = None
    for sgn in (1, -1):
        a = (sR[:, 1] if axis is None else np.asarray(axis, float)) * sgn   # lying direction
        A = np.column_stack([z_i, n_i, np.cross(z_i, n_i)])
        B = np.column_stack([a, [0, 0, -1.0], np.cross(a, [0, 0, -1.0])])
        Rd = B @ A.T
        f = Rd @ relR @ np.array([0, -1.0, 0])
        if best is None or f @ F > best[0]:
            best = (f @ F, Rd)
    Rd = best[1]
    label = S.item(name).label
    r.stage(f"Turn the {label} to lie down")
    o, Rr = r.root_frame()
    mid = o + Rr @ np.array([0.42, -0.05, 1.0])
    r.move(1.4, hands={side: (mid + Rd @ relp, Rd @ relR)}, gaze=p_slot)
    r.stage(f"Lay the {label} in the suitcase")
    r.move(1.5, hands={side: (p_slot + np.array([0, 0, raise_h]) + Rd @ relp, Rd @ relR)}, root=dict(h=0.80))
    r.move(1.1, hands={side: (p_slot + np.array([0, 0, 0.004]) + Rd @ relp, Rd @ relR)})
    r.hold(0.2)
    r.stage(f"Release the {label}")
    hp, hR = r.hand[side]
    r.move(0.7, hands={side: (hp + np.array([0, 0, 0.012]), hR)}, fingers={side: "cup_open"})
    r.move(0.8, hands={side: (hp + np.array([0, 0, 0.16]), hR)})
    clear_hands(r)
    pp, _ = r.body_pose(name)
    r.event("placed", item=name, pos=pp.tolist(), speed=r.item_speed(name))


# ---------------------------------------------------------------- lid
def lid_angle(r):
    return float(r.d.qpos[r.m.joint("lid_hinge").qposadr[0]])


def hinge_axis(r):
    sp, sR = r.suitcase_frame()
    return sp + sR @ np.array([-W / 2, 0, HH]), sR @ np.array([0, -1.0, 0])


# Lid-edge hooks tried in turn: (grip inset from the free edge, palm offset over the wall, palm height).
LID_HOOKS = [(0.075, 0.022, 0.004), (0.10, 0.018, 0.0), (0.055, 0.026, 0.008)]
LID_SWING_T = 2.3     # seconds for the swing over


def close_lid(r, side="r"):
    """Hook the free corner of the lid's handle-end wall and swing the lid shut.

    In the lid frame the handle-end wall lies at y=+L/2 (towards the foot of the bed);
    its rim is at z=ZR and the free edge at x=+W. The palm rests on the rim with the
    fingers reaching over into the lid; curling them makes a hook, so lifting rotates
    the lid by contact alone. The hook is tested with a short lift, and the swing is
    watched: if the lid falls behind the hand, the person lets go and hooks it again."""
    for attempt, hook in enumerate(LID_HOOKS):
        if _hook_lid(r, side, hook, attempt) and _swing_lid(r, side):
            break
        r.move(0.4, fingers={side: "open"})
        hp, hR = r.hand[side]
        r.move(0.6, hands={side: (hp + np.array([0, 0, 0.08]), hR)})
        r.hold(1.0)
        if lid_angle(r) < np.radians(150):
            # Left part-way: hooking it again from here would press it onto the contents.
            raise RuntimeError(f"close_lid: lid left at {np.degrees(lid_angle(r)):.0f} deg")
    else:
        raise RuntimeError("close_lid: could not swing the lid over")
    hw, _ = hinge_axis(r)
    r.stage("Let the lid fall closed")
    # Open the fingers and draw the hand straight back out over the wall (against the
    # finger direction) before lifting it, so nothing props the lid.
    hp, hR = r.hand[side]
    _, R_l = r.body_pose("suitcase_lid")
    out_wall = R_l @ np.array([0, 1.0, 0])          # back out over the handle-end wall
    r.move(0.35, fingers={side: "open"})
    # Out past the handle end (fingers slide out under the wall's rim edge), clear of the
    # arc the falling lid sweeps, then up.
    r.move(0.7, hands={side: (hp + out_wall * 0.20, hR)})
    hp, hR = r.hand[side]
    r.move(0.8, hands={side: (hp + np.array([0, 0, 0.10]), hR)}, gaze=hw)
    r.hold(1.5)
    r.event("lid_released", lid_angle_deg=float(np.degrees(lid_angle(r))))


def _hook_lid(r, side, hook, attempt):
    inset, over, h = hook
    p_lid, R_lid = r.body_pose("suitcase_lid")
    down_l = R_lid.T @ np.array([0, 0, -1.0])     # world-down expressed in the lid frame
    Rh_l = hand_R(np.array([0, -1.0, 0]), down_l)  # fingers along lid -y, over the wall into the lid
    grip = np.array([W - inset, L / 2 - T_SHELL / 2, ZR])
    palm_l = grip + np.array([0, over, 0])

    def tgt(up=0.0, back=0.0):
        p, R = r.body_pose("suitcase_lid")
        return {side: (p + R @ (palm_l + np.array([0, back, 0])) + np.array([0, 0, up]), R @ Rh_l)}

    r.stage("Hook the lid edge" if attempt == 0 else "Re-hook the lid edge")
    r.move(1.6 if attempt == 0 else 1.0, hands=tgt(0.07, 0.06), fingers={side: "open"}, root=dict(h=0.86),
           gaze=p_lid + R_lid @ grip)
    r.move(0.9, hands=tgt(h + 0.008))
    r.move(0.9, hands=tgt(h), fingers={side: "hook"})
    r.hold(0.25)
    th_a = lid_angle(r)
    r.move(0.5, hands=tgt(h + 0.03))                  # test lift: the hooked lid must come up
    lifted = th_a - lid_angle(r)
    r.event("lid_hook_check", attempt=attempt, lid_rise_deg=float(np.degrees(lifted)))
    return lifted > np.radians(1.5)


def _swing_lid(r, side):
    hp, hR = r.hand[side]
    p_lid, R_lid = r.body_pose("suitcase_lid")
    relp, relR = R_lid.T @ (hp - p_lid), R_lid.T @ hR
    hw, ax = hinge_axis(r)
    th0 = lid_angle(r)

    def pose_at(th):
        dR = Rot.from_rotvec(ax * (th - th0)).as_matrix()
        Rl = dR @ R_lid
        pl = hw + dR @ (p_lid - hw)
        return pl + Rl @ relp, Rl @ relR

    r.stage("Swing the lid over")
    # This deep lid balances near 64 deg (it must be carried past vertical), so it is swung
    # briskly: its momentum carries it over the balance point as the hand slows.
    th_rel = np.radians(35.0)
    n = int(LID_SWING_T * 1000)
    for k in range(n):
        th = th0 + (th_rel - th0) * minjerk((k + 1) / n)
        r.hand[side] = pose_at(th)
        r.gaze = hw
        r.step()
        # A lid left behind by the hand high up has slipped off the hook; near its balance
        # point (about 64 deg) it is only stalling, so the swing carries on.
        if k % 100 == 99 and lid_angle(r) - th > np.radians(10) and lid_angle(r) > np.radians(76):
            r.event("lid_slipped", lid_angle_deg=float(np.degrees(lid_angle(r))), hand_deg=float(np.degrees(th)))
            return False
    r.event("lid_swung", lid_angle_deg=float(np.degrees(lid_angle(r))))
    return lid_angle(r) < np.radians(62)


def open_lid(r, side="r"):
    """From the foot of the bed: hook the fingertips under the zip-track lip on the lid's
    free long edge near the handle end, swing the lid up past its balance point and let it fall open."""
    if r.zip_locked:
        raise RuntimeError("open_lid: the case is zipped")
    p_lid, R_lid = r.body_pose("suitcase_lid")
    # Lid frame: free-edge wall outer face at x = W, lip under-side at z = ZR + 0.001.
    f_l = np.array([0, 0, -1.0])                  # fingers hang down along the outside of the wall
    n_l = np.array([-1.0, 0, 0])                  # palm faces the wall
    Rh_l = hand_R(f_l, n_l)
    corner = np.array([W, L / 2 - 0.09, ZR])

    def tgt(up=0.0, out=0.0):
        p, R = r.body_pose("suitcase_lid")
        return {side: (p + R @ (corner + np.array([0.034 + out, 0, 0.055])) + np.array([0, 0, up]), R @ Rh_l)}

    r.stage("Hook the lid edge")
    r.move(1.6, hands=tgt(0.08, 0.05), fingers={side: "open"}, root=dict(h=0.86), gaze=p_lid + R_lid @ corner)
    r.move(0.9, hands=tgt(0.0, 0.02))
    r.move(0.6, hands=tgt(0.0))
    r.move(0.9, fingers={side: "hook"})
    r.hold(0.25)
    hp, hR = r.hand[side]
    p_lid, R_lid = r.body_pose("suitcase_lid")
    relp, relR = R_lid.T @ (hp - p_lid), R_lid.T @ hR
    hw, ax = hinge_axis(r)
    th0 = lid_angle(r)

    def pose_at(th):
        # The fingertips follow the lip round the hinge; the hand itself turns only half as
        # far (the wrist and fingers take the rest), so the forearm stays within reach.
        dR = Rot.from_rotvec(ax * (th - th0)).as_matrix()
        dRh = Rot.from_rotvec(ax * 0.5 * (th - th0)).as_matrix()
        return hw + dR @ (p_lid - hw) + dR @ R_lid @ relp, dRh @ R_lid @ relR

    r.stage("Swing the lid open")
    th_rel = np.radians(82.0)                     # past its balance point (~64 deg): gravity opens it from here
    n = 4200
    for k in range(n):
        th = th0 + (th_rel - th0) * minjerk((k + 1) / n)
        r.hand[side] = pose_at(th)
        r.gaze = hw
        r.step()
    r.event("lid_lifted", lid_angle_deg=float(np.degrees(lid_angle(r))))
    if lid_angle(r) < np.radians(68):
        raise RuntimeError(f"open_lid: lid only reached {np.degrees(lid_angle(r)):.1f} deg")
    r.stage("Let the lid fall open")
    hp, hR = r.hand[side]
    _, R_l = r.body_pose("suitcase_lid")
    r.move(0.35, fingers={side: "open"})
    r.move(0.7, hands={side: (hp + R_l @ np.array([0.12, 0, 0]), hR)})
    hp, hR = r.hand[side]
    r.move(0.8, hands={side: (hp + np.array([0, 0, 0.10]), hR)}, gaze=hw)
    r.hold(1.5)
    r.event("lid_opened", lid_angle_deg=float(np.degrees(lid_angle(r))))


# ---------------------------------------------------------------- final lift
WRIST_FOLLOW = 0.8      # fraction of the case's tip the wrists follow while standing it up
STAND_T = 3.2           # seconds to stand the case up
RELEASE_RISE = 0.07     # hands rise this much while the fingers open
CARRY_REACH = 0.32      # pelvis distance from the bar while carrying the upright case
# Hand placements tried in turn (forearm pitch deg, knuckle height above the bar, knuckle
# depth into the gap as a fraction of it); a slipping grip is released and the next tried.
GRIP_ATTEMPTS = [(15, 0.010, 0.25), (0, 0.010, 0.15), (25, 0.008, 0.35), (10, 0.012, 0.20)]


def lift_from_bed_foot(r, floor_x=2.52):
    """From the foot of the bed: hook the trolley bar with both hands, stand the case up
    on its wheels, lift it clear, carry it past the bed end and set it down. A packed
    large case weighs about 11 kg, so both hands share it. A grip that does not hold on the
    test pull is released and retried with the hands placed differently (a re-grip).

    Every pull (up, or towards the body) seats the bar deeper in the curled fingers."""
    sp, sR = r.suitcase_frame()
    hb = S.SUITCASE["bar"]
    bar = lambda: r.body_pose("trolley")[0]          # the bar on its (latched) telescopic slide
    bar0 = bar()
    away = bar0 - np.array([r.root["x"], r.root["y"], bar0[2]]); away[2] = 0; away = unit(away)
    up = np.array([0, 0, 1.0])
    right = np.array([away[1], -away[0], 0.0])
    lat = {"r": 0.046, "l": -0.046}                 # side by side along the bar
    gap = hb["y"] - S.SUIT_L / 2 - hb["r"]          # free space between the bar and the case end
    # Knuckle line in the palm frame (MCP joints of fingers 2-5), measured on this hand.
    knuck = {}
    for s in "rl":
        hp, hR = r.hand_pose(s)
        k = np.mean([r.d.xpos[r.m.body(f"{i}proxph_{s}").id] for i in (2, 3, 4, 5)], axis=0)
        knuck[s] = hR.T @ (k - hp)

    def shifted(base, delta):
        return {s: (base[s][0] + delta, base[s][1]) for s in "rl"}

    def grip(pitch_deg, knuckle_up, depth, first):
        """Hook the bar: forearms pitched down towards the case, palms facing down and
        away, knuckles over the gap behind the bar so the hanging fingers drop into it and
        curl under. Then test it with a short pull: True if the bar comes up."""
        pitch = np.radians(pitch_deg)
        Rh = hand_R(away * np.cos(pitch) - up * np.sin(pitch), -up * np.cos(pitch) - away * np.sin(pitch))

        def hook(lift=0.0, back=0.0):
            b = bar()
            return {s: (b + right * lat[s] + away * (hb["r"] + depth * gap - back)
                        + up * (hb["r"] + knuckle_up + lift) - Rh @ knuck[s], Rh) for s in "rl"}

        r.stage("Hook the trolley handle" if first else "Re-grip the trolley handle")
        if first:
            approach(r, hook(0.10, 0.04), fingers={"r": "hook_open", "l": "hook_open"}, gaze=bar0, root=dict(h=0.72))
        else:
            r.move(0.8, hands=hook(0.10, 0.04), fingers={"r": "hook_open", "l": "hook_open"})
        r.move(1.0, hands=hook(0.06))
        r.move(0.9, hands=hook(0.0))
        r.move(0.9, fingers={"r": "bar_fist", "l": "bar_fist"})
        r.hold(0.3)
        z0 = bar()[2]
        base = {s: r.hand[s] for s in "rl"}
        r.move(0.8, hands=shifted(base, up * 0.07))
        rise = bar()[2] - z0
        r.event("hook_check", pitch_deg=pitch_deg, bar_rise_m=float(rise),
                contacts={s: r.contacts_between(r.hand_bodies[s], [r.m.body("suitcase").id])[0] for s in "rl"})
        if rise > 0.035:
            return True
        # Slipped: let go and lift the hands clear before trying again.
        r.move(0.6, fingers={"r": "hook_open", "l": "hook_open"})
        base = {s: r.hand[s] for s in "rl"}
        r.move(0.6, hands=shifted(base, up * 0.06))
        return False

    for k, (pd, ku, dp) in enumerate(GRIP_ATTEMPTS):
        if grip(pd, ku, dp, first=k == 0):
            break
    else:
        raise RuntimeError("suitcase lift failed: no grip on the trolley bar")
    p0 = {s: (r.hand[s][0].copy(), r.hand[s][1].copy()) for s in "rl"}
    bar1 = bar()
    wheel_drop = hb["y"] - S.SUITCASE["wheel_y"] + S.SUITCASE["wheel_r"]   # bar axis to wheel bottoms, case upright
    r.stage("Stand it up by the handle")
    stand_z = S.BED["mattress_top"] + wheel_drop + 0.01
    delta = -away * 0.10 + up * (stand_z - bar1[2])
    h0 = r.root["h"]
    n = int(STAND_T * 1000)
    trace = []
    pitch0 = np.arcsin(np.clip(r.suitcase_frame()[1][2, 1], -1, 1))     # handle-end elevation
    for k in range(n):
        u = minjerk((k + 1) / n)
        # The wrists turn part of the way with the case as it tips up, so the bar stays
        # seated in the curled fingers instead of rolling out of them.
        tip = np.arcsin(np.clip(r.suitcase_frame()[1][2, 1], -1, 1)) - pitch0
        Rw = Rot.from_rotvec(-right * WRIST_FOLLOW * tip).as_matrix()
        for s in "rl":
            off = p0[s][0] - bar1
            r.hand[s] = (bar1 + delta * u + Rw @ off, Rw @ p0[s][1])
            # the wrist joint takes up to 40 degrees of that turn, so the forearm stays low
            r.finger[s][f"flexion_{s}"] = float(min(0.7, WRIST_FOLLOW * max(tip, 0.0)))
        r.root["h"] = h0 + (0.95 - h0) * u
        r.gaze = bar0 + up * 0.3
        r.step()
        if k % (n // 8) == n // 8 - 1:
            trace.append(dict(u=round(u, 2), bar=np.round(bar(), 3).tolist(),
                              contacts={s: r.contacts_between(r.hand_bodies[s], [r.m.body("suitcase").id])[0] for s in "rl"}))
    r.event("stand_trace", trace=trace)
    r.hold(0.8)
    sp1, sR1 = r.suitcase_frame()
    r.event("suitcase_lift_check", rise_m=float(sp1[2] - sp[2]),
            handle_end_up_deg=float(np.degrees(np.arccos(np.clip(sR1[2, 1], -1, 1)))),
            contacts={s: r.contacts_between(r.hand_bodies[s], r.suit_ids)[0] for s in "rl"})
    if sp1[2] - sp[2] < 0.05:
        raise RuntimeError("suitcase lift failed")

    def at_bar(target):
        off = {s: r.hand[s][0] - bar() for s in "rl"}
        return {s: (target + off[s], r.hand[s][1]) for s in "rl"}

    r.stage("Lift it off the bed")
    b1 = bar()
    r.move(1.2, hands=at_bar(b1 + up * 0.08))
    r.stage("Carry it past the end of the bed")
    r.move(3.0, hands=at_bar(np.array([floor_x, b1[1], b1[2] + 0.08])), root=dict(x=floor_x + CARRY_REACH),
           gaze=np.array([floor_x, b1[1], 0.4]))
    r.hold(0.8)
    r.event("hanging_check", handle_end_up_deg=float(np.degrees(np.arccos(np.clip(r.suitcase_frame()[1][2, 1], -1, 1)))))
    r.stage("Set it down on its wheels")
    r.move(2.4, hands=at_bar(np.array([floor_x, b1[1], wheel_drop + 0.03])), root=dict(h=0.70))
    # Square it up: bring the handle directly above the wheel footprint as it touches down.
    foot_local = np.array([0, S.SUITCASE["wheel_y"], (S.SUITCASE["wheel_z_base"] + HH + ZR + S.SUITCASE["wheel_z_lid"]) / 2])
    for _ in range(3):
        cp, cR = r.suitcase_frame()
        foot = cp + cR @ foot_local
        r.move(0.8, hands=at_bar(np.array([foot[0], foot[1], wheel_drop - 0.004])))
    r.hold(0.5)
    r.event("touchdown_tilt_deg", value=float(np.degrees(np.arccos(np.clip(r.suitcase_frame()[1][2, 1], -1, 1)))))
    r.stage("Let go of the handle")
    # A spinner case standing on its wheels tips over at about 9 degrees front to back, so
    # the release must not push it: unload the grip by a centimetre, open the fingers
    # slowly, then lift the hands straight up and only then draw them back.
    base = {s: r.hand[s] for s in "rl"}
    r.move(0.4, hands=shifted(base, up * 0.01))
    # open while rising, so the fingertips swing out and up rather than into the case
    base = {s: r.hand[s] for s in "rl"}
    r.move(1.4, hands=shifted(base, up * RELEASE_RISE), fingers={"r": "open", "l": "open"})
    r.hold(0.4)
    base = {s: r.hand[s] for s in "rl"}
    r.move(1.2, hands=shifted(base, up * 0.15))
    r.event("release_contacts", contacts={s: r.contacts_between(r.hand_bodies[s], r.suit_ids)[0] for s in "rl"})
    base = {s: r.hand[s] for s in "rl"}
    r.move(0.8, hands=shifted(base, -away * 0.12), fingers={"r": "relaxed", "l": "relaxed"}, root=dict(h=0.95))
    # Step back from the standing case before lowering the arms past its top.
    back = np.array([r.root["x"], r.root["y"]]) - away[:2] * 0.22
    r.move(1.2, root=dict(x=float(back[0]), y=float(back[1])), follow=("r", "l"))
    r.hold(0.6)

