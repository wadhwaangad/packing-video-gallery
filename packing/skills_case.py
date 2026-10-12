"""Thermos wrap grasp, lid closing, zipper pass and the final suitcase lift."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import scene as S
from runner import minjerk
from tasks import unit, hand_R, facing, slot_pose, clear_hands, approach

W, L = S.SUITCASE["outer"]
D, DL, T_SHELL = S.SUITCASE["base_depth"], S.SUITCASE["lid_depth"], S.SUITCASE["shell"]


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


def place_upright(r, name, side="r", raise_h=0.08):
    """Lower the upright bottle into its slot, open the fingers and lift the hand straight up."""
    p, R = r.body_pose(name)
    hp, hR = r.hand[side]
    relp, relR = R.T @ (hp - p), R.T @ hR
    p_slot, R_slot = slot_pose(r, name)
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
    r.stage(f"Release the {label}")
    hp, hR = r.hand[side]
    r.move(0.7, hands={side: (hp - hR[:, 2] * -0.004, hR)}, fingers={side: "cup_open"})
    r.move(1.0, hands={side: (hp + np.array([0, 0, 0.22]), hR)})
    clear_hands(r)
    pp, _ = r.body_pose(name)
    r.event("placed", item=name, pos=pp.tolist(), speed=r.item_speed(name))


def place_lying(r, name, side="r", raise_h=0.10):
    """Rotate a wrapped cylinder to lie along the suitcase length, palm on top, and set it down."""
    p, R = r.body_pose(name)
    hp, hR = r.hand[side]
    relp, relR = R.T @ (hp - p), R.T @ hR
    z_i = np.array([0, 0, 1.0])                   # cylinder axis in the item frame
    n_i = relR @ np.array([0, 0, -1.0])          # palm normal in the item frame
    n_i = unit(n_i - z_i * (n_i @ z_i))
    p_slot, _ = slot_pose(r, name)
    sp, sR = r.suitcase_frame()
    F, _ = facing(r)
    best = None
    for sgn in (1, -1):
        a = sR[:, 1] * sgn                        # suitcase length direction
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
    return sp + sR @ np.array([-W / 2, 0, D]), sR @ np.array([0, -1.0, 0])


def close_lid(r, side="r"):
    """Hook the free corner of the lid's handle-end wall and swing the lid shut.

    In the lid frame the handle-end wall lies at y=+L/2 (towards the foot of the bed);
    its rim is at z=0 and the free edge at x=+W. The palm rests on the rim with the
    fingers reaching over into the lid; curling them makes a hook, so lifting rotates
    the lid by contact alone."""
    grip = np.array([W - 0.075, L / 2 - T_SHELL / 2, 0.0])
    p_lid, R_lid = r.body_pose("suitcase_lid")
    down_l = R_lid.T @ np.array([0, 0, -1.0])     # world-down expressed in the lid frame (open lid)
    f_l = np.array([0, -1.0, 0])                  # fingers point along lid -y, over the wall into the lid
    Rh_l = hand_R(f_l, down_l)
    palm_l = grip + np.array([0, 0.022, 0])

    def tgt(up=0.0, back=0.0):
        p, R = r.body_pose("suitcase_lid")
        return {side: (p + R @ (palm_l + np.array([0, back, 0])) + np.array([0, 0, up]), R @ Rh_l)}

    r.stage("Hook the lid edge")
    r.move(1.6, hands=tgt(0.07, 0.06), fingers={side: "open"}, root=dict(h=0.86), gaze=p_lid + R_lid @ grip)
    r.move(0.9, hands=tgt(0.012))
    r.move(0.9, hands=tgt(0.004), fingers={side: "hook"})
    r.hold(0.25)
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
    th_rel = np.radians(50.0)                       # well past vertical: gravity closes it from here
    n = 4200
    for k in range(n):
        th = th0 + (th_rel - th0) * minjerk((k + 1) / n)
        r.hand[side] = pose_at(th)
        r.gaze = hw
        r.step()
    r.event("lid_swung", lid_angle_deg=float(np.degrees(lid_angle(r))))
    r.stage("Let the lid fall closed")
    # Open the fingers and draw the hand straight back out over the wall (against the
    # finger direction) before lifting it, so nothing props the lid.
    hp, hR = r.hand[side]
    _, R_l = r.body_pose("suitcase_lid")
    clear_rim = R_l @ np.array([0, 0, -1.0])        # fingers off the rim, away from the lid panel
    out_wall = R_l @ np.array([0, 1.0, 0])          # back out over the handle-end wall
    r.move(0.35, fingers={side: "open"})
    # Out past the handle end (fingers slide out under the wall's rim edge), clear of the
    # arc the falling lid sweeps, then up.
    r.move(0.7, hands={side: (hp + out_wall * 0.20, hR)})
    hp, hR = r.hand[side]
    r.move(0.8, hands={side: (hp + np.array([0, 0, 0.10]), hR)}, gaze=hw)
    r.hold(1.5)
    r.event("lid_released", lid_angle_deg=float(np.degrees(lid_angle(r))))


def press_and_zip(r):
    """Both palms press the lid; the right hand then runs the zipper round three sides."""
    sp, sR = r.suitcase_frame()
    top = sp + sR @ np.array([0, 0, D + DL])
    F, right = facing(r)
    down = np.array([0, 0, -1.0])
    f = F.copy()                       # flat hands: forearms level so the palms land first
    R_press = {"r": hand_R(f, down), "l": hand_R(f, down)}
    pr, pl = top + right * 0.16 - F * 0.03, top - right * 0.16 - F * 0.03
    r.stage("Press the lid down")
    approach(r, {"r": (pr + [0, 0, 0.06], R_press["r"]), "l": (pl + [0, 0, 0.06], R_press["l"])},
             fingers={"r": "press", "l": "press"}, root=dict(h=0.86), gaze=top)
    # Gentle, purely vertical press: ~4 mm past first contact.
    r.move(1.0, hands={"r": (pr - [0, 0, 0.004], R_press["r"]), "l": (pl - [0, 0, 0.004], R_press["l"])})
    r.hold(0.4)
    c = r.contacts_between(r.hand_bodies["r"] + r.hand_bodies["l"], r.suit_ids)
    r.event("lid_press", contacts=c[0], force_N=round(c[1], 1), lid_angle_deg=float(np.degrees(lid_angle(r))))
    # The left hand keeps pressing while the right hand runs the zipper.
    # The hand runs the zip along the near long side (the reachable part of the track);
    # the weld idealises closure of the full perimeter.
    path_local = [(W / 2, -L / 2 + 0.04, D), (W / 2, 0.0, D), (W / 2, L / 2 - 0.05, D)]   # near side; stops short of the top handle
    pts = [sp + sR @ np.array(p) for p in path_local]
    seg = [np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])]
    cum = np.concatenate([[0], np.cumsum(seg)]); Ltot = cum[-1]

    def at(sv):
        i = min(np.searchsorted(cum, sv, side="right") - 1, len(seg) - 1)
        return pts[i] + (pts[i + 1] - pts[i]) * ((sv - cum[i]) / seg[i])

    centre = sp + sR @ np.array([0, 0, D])
    # Fingers down beside the track, palm towards the case: nothing points into the shell.
    out0 = unit((at(0.0) - centre) * np.array([1, 1, 0]))
    pinch = hand_R(down, -out0)
    r.stage("Run the zipper along the case")

    def zp(sv):
        p = at(sv)
        out = unit((p - centre) * np.array([1, 1, 0]))
        return p + out * 0.085 + np.array([0, 0, 0.03])       # traces the track just clear of the shell

    approach(r, {"r": (zp(0.0) + [0, 0, 0.04], pinch)}, fingers={"r": "squeeze"}, gaze=at(0.0))
    r.move(0.5, hands={"r": (zp(0.0), pinch)})
    n = 9000
    for k in range(n):
        sv = Ltot * minjerk((k + 1) / n)
        r.hand["r"] = (zp(sv), pinch)
        r.zip_progress = sv / Ltot
        r.gaze = at(sv)
        r.step()
    ok = lid_angle(r) < np.radians(2.5)
    r.event("zip_check", lid_angle_deg=float(np.degrees(lid_angle(r))), closed=bool(ok))
    if not ok:
        raise RuntimeError(f"lid not closed: {np.degrees(lid_angle(r)):.2f} deg")
    # Idealised zipper: lock the lid by narrowing its hinge range to the closed pose.
    j = r.m.joint("lid_hinge").id
    r.m.jnt_range[j] = [0.0, 0.003]
    r.zip_locked = True
    r.event("zipper_engaged", lid_angle_deg=float(np.degrees(lid_angle(r))))
    r.stage("Zipped shut")
    # Back the zip hand straight out from the case (fingers open) before anything rises.
    hp, hR = r.hand["r"]
    out = unit((hp - centre) * np.array([1, 1, 0]))
    r.move(0.5, hands={"r": (hp + [0, 0, 0.04], hR)}, fingers={"r": "open"})
    r.move(0.6, hands={"r": (hp + [0, 0, 0.04] + out * 0.07, hR)})
    r.move(0.9, hands={s: (r.hand[s][0] + [0, 0, 0.14], r.hand[s][1]) for s in "rl"},
           fingers={"r": "relaxed", "l": "relaxed"})


# ---------------------------------------------------------------- final lift
def lift_suitcase(r, floor_xy=(2.02, 0.52), stand=(1.50, 0.50)):
    """Hook the top and side handles, lift the case off the bed, turn it upright, set it on its wheels."""
    sp, sR = r.suitcase_frame()
    zc = D * 0.55
    top_bar = np.array([0, L / 2 + 0.045, zc])     # bar axis along base x
    side_bar = np.array([-W / 2 - 0.045, 0, zc])   # bar axis along base y
    down = np.array([0, 0, -1.0])
    # Both hands come from above (forearm pointing down), palm facing the case, fingers
    # hanging outside the bar; curling them hooks under the bar towards the case.
    down_b = sR.T @ down
    f_top_b, n_top_b = down_b, np.array([0, -1.0, 0])     # top handle on the +y end: palm faces -y
    f_side_b, n_side_b = down_b, np.array([1.0, 0, 0])    # side handle on the -x side: palm faces +x
    Rl_b = hand_R(f_top_b, n_top_b); Rr_b = hand_R(f_side_b, n_side_b)

    def hook_targets(up=0.0, out=0.0):
        sp, sR = r.suitcase_frame()
        # palm 4.5 cm above the bar axis (knuckles level with it), just outside it
        zl = sp + sR @ (top_bar + np.array([0, 0.019 + out, 0]) - down_b * 0.045) + [0, 0, up]
        zr = sp + sR @ (side_bar + np.array([-0.019 - out, 0, 0]) - down_b * 0.045) + [0, 0, up]
        return {"l": (zl, sR @ Rl_b), "r": (zr, sR @ Rr_b)}

    r.stage("Grip both handles")
    approach(r, hook_targets(0.10, out=0.02), fingers={"r": "press", "l": "press"}, gaze=sp, root=dict(h=0.86))
    r.move(0.9, hands=hook_targets(0.0, out=0.02))
    r.move(0.5, hands=hook_targets(0.0))
    r.move(0.9, fingers={"r": "hook", "l": "hook"})
    r.hold(0.3)
    rel = {}
    sp, sR = r.suitcase_frame()
    for s in "rl":
        hp, hR = r.hand[s]
        rel[s] = (sR.T @ (hp - sp), sR.T @ hR)
    z0 = sp[2]

    def hands_for(p, R):
        return {s: (p + R @ rel[s][0], R @ rel[s][1]) for s in "rl"}

    r.stage("Lift the suitcase")
    r.move(1.4, hands=hands_for(sp + [0, 0, 0.12], sR), root=dict(h=0.92))
    r.hold(0.4)
    rise = r.suitcase_frame()[0][2] - z0
    r.event("suitcase_lift_check", rise_m=float(rise),
            contacts={s: r.contacts_between(r.hand_bodies[s], r.suit_ids)[0] for s in "rl"})
    if rise < 0.06:
        raise RuntimeError(f"suitcase lift failed: {rise}")
    # Upright: base +y (handle end) points up; wheels (-y) down; base +z (lid side) faces away from the bed.
    centre_local = np.array([0, 0, (D + DL) / 2])
    R_up = np.column_stack([np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, -1.0, 0])])
    # keep the lid facing the room (+y) or (-y) according to the current lid direction
    if (sR[:, 2] @ np.array([0, 1.0, 0])) > 0:
        R_up = np.column_stack([np.array([-1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, 1.0, 0])])
    wheel_drop = L / 2 + 0.03 + 0.028
    p_floor_centre = np.array([floor_xy[0], floor_xy[1], wheel_drop])
    sp1, sR1 = r.suitcase_frame()
    c1 = sp1 + sR1 @ centre_local
    mid_c = np.array([(c1[0] + floor_xy[0]) / 2, (c1[1] + floor_xy[1]) / 2 - 0.15, 0.85])
    slerp_R = Rot.from_matrix([sR1, R_up])
    from scipy.spatial.transform import Slerp
    sl = Slerp([0, 1], slerp_R)
    r.stage("Swing it off the bed and upright")
    root0 = dict(r.root)
    n = 4200
    for k in range(n):
        u = minjerk((k + 1) / n)
        Ru = sl(u).as_matrix()
        # quadratic path through mid_c
        c = (1 - u) ** 2 * c1 + 2 * u * (1 - u) * mid_c + u * u * (p_floor_centre + [0, 0, 0.06])
        p = c - Ru @ centre_local
        r.hand.update(hands_for(p, Ru))
        r.root["x"] = root0["x"] + (stand[0] - root0["x"]) * u
        r.root["y"] = root0["y"] + (stand[1] - root0["y"]) * u
        r.root["h"] = root0["h"] + (0.80 - root0["h"]) * np.sin(np.pi * u) * 0.5
        r.gaze = c
        r.step()
    r.stage("Set it down on its wheels")
    p_end = p_floor_centre - R_up @ centre_local + [0, 0, -0.004]
    r.move(1.4, hands=hands_for(p_end, R_up), root=dict(h=0.84))
    r.hold(0.4)
    r.stage("Let go of the handles")
    out = {}
    for s in "rl":
        hp, hR = r.hand[s]
        out[s] = (hp + hR[:, 1] * 0.04 + np.array([0, 0, 0.03]), hR)
    r.move(0.8, fingers={"r": "open", "l": "open"})
    r.move(0.9, hands=out)
    r.hold(0.3)


def lift_by_top_handle(r, floor_xy=(1.12, 0.64)):
    """Hook the top handle with the left hand and lift: gravity swings the case to hang
    upright from the handle; carry it off the bed and lower it onto its wheels."""
    sp, sR = r.suitcase_frame()
    zc = D * 0.55
    top_bar = np.array([0, L / 2 + 0.045, zc])
    down = np.array([0, 0, -1.0])
    down_b = sR.T @ down
    Rl_b = hand_R(down_b, np.array([0, -1.0, 0]))       # forearm down, palm facing the case end

    def hook_target(up=0.0, out=0.0):
        sp, sR = r.suitcase_frame()
        return sp + sR @ (top_bar + np.array([0, 0.019 + out, 0]) - down_b * 0.045) + [0, 0, up], sR @ Rl_b

    r.stage("Hook the top handle")
    approach(r, {"l": hook_target(0.10, out=0.02)}, fingers={"l": "press"}, gaze=sp, root=dict(h=0.86))
    r.move(0.9, hands={"l": hook_target(0.0, out=0.02)})
    r.move(0.5, hands={"l": hook_target(0.0)})
    r.move(0.9, fingers={"l": "hook"})
    r.hold(0.3)
    p0, R0 = r.hand["l"]
    bar0 = sp + sR @ top_bar
    r.stage("Lift it by the handle")
    wheel_drop = L / 2 + 0.045 + L / 2 + 0.03 + 0.028    # bar axis to wheel bottoms when hanging
    # Raise the handle end to about chest height: the case pivots on its wheel end.
    lift_z = 1.08
    r.move(2.4, hands={"l": (np.array([p0[0], p0[1] - 0.03, p0[2] + (lift_z - bar0[2])]), R0)},
           root=dict(h=0.95), gaze=bar0 + [0, 0, 0.3])
    r.hold(0.5)
    sp1, sR1 = r.suitcase_frame()
    r.event("suitcase_lift_check", rise_m=float(sp1[2] - sp[2]),
            handle_end_up_deg=float(np.degrees(np.arccos(np.clip(sR1[2, 1], -1, 1)))),
            contacts={"l": r.contacts_between(r.hand_bodies["l"], r.suit_ids)[0]})
    if sp1[2] - sp[2] < 0.05:
        raise RuntimeError("suitcase lift failed")
    bar = lambda: r.suitcase_frame()[0] + r.suitcase_frame()[1] @ top_bar
    off = r.hand["l"][0] - bar()
    # Draw it towards the body: the wheel end slides to the bed edge and swings down,
    # leaving the case hanging upright from the handle.
    r.stage("Draw it off the bed")
    p1 = np.array([floor_xy[0] + 0.02, floor_xy[1] - 0.14, lift_z + 0.08]) + off
    r.move(3.0, hands={"l": (p1, R0)}, root=dict(x=1.62, y=0.52, yaw=np.pi / 2 + 0.25),
           gaze=np.array([floor_xy[0], floor_xy[1], 0.5]))
    r.hold(1.5)                                           # let the hanging case stop swinging
    r.event("hanging_check", handle_end_up_deg=float(np.degrees(np.arccos(np.clip(r.suitcase_frame()[1][2, 1], -1, 1)))))
    p2 = np.array([floor_xy[0], floor_xy[1], wheel_drop + 0.05]) + off
    r.move(2.0, hands={"l": (p2, R0)})
    r.stage("Set it down on its wheels")
    p3 = np.array([floor_xy[0], floor_xy[1], wheel_drop - 0.006]) + off
    r.move(1.6, hands={"l": (p3, R0)}, root=dict(h=0.90))
    r.hold(0.5)
    r.stage("Let go of the handle")
    r.move(0.8, fingers={"l": "open"})
    hp, hR = r.hand["l"]
    r.move(1.0, hands={"l": (hp + [0, -0.06, 0.12], hR)})
    r.hold(0.4)


def lift_from_bed_foot(r, side="r", floor_x=2.52):
    """From the foot of the bed: hook the top handle palm-down with a level forearm, stand
    the case up on its wheels, lift it clear, carry it past the bed end and set it down.

    Every pull (up, or towards the body) seats the bar deeper in the curled fingers."""
    sp, sR = r.suitcase_frame()
    zc = D * 0.55
    top_bar = np.array([0, L / 2 + 0.045, zc])
    bar = lambda: r.suitcase_frame()[0] + r.suitcase_frame()[1] @ top_bar
    bar0 = bar()
    away = bar0 - np.array([r.root["x"], r.root["y"], bar0[2]]); away[2] = 0; away = unit(away)
    up = np.array([0, 0, 1.0])
    idle = "l" if side == "r" else "r"
    Rh = hand_R(away, -up)                          # forearm level towards the case, palm down

    def hook(lift=0.0, back=0.0):
        b = bar()
        # knuckles over the gap behind the bar; palm pad resting on the bar top
        return b - away * (0.022 + back) + up * (0.011 + 0.002 + lift), Rh

    r.stage("Hook the top handle")
    approach(r, {side: hook(0.10, 0.04)}, fingers={side: "hook_open"}, gaze=bar0, root=dict(h=0.72))
    r.move(1.0, hands={side: hook(0.06)})
    r.move(0.9, hands={side: hook(0.0)})
    r.move(0.9, fingers={side: "hook"})
    r.hold(0.3)
    p0, R0 = r.hand[side]
    wheel_drop = L / 2 + 0.045 + L / 2 + 0.03 + 0.028    # bar axis to wheel bottoms, case upright
    r.stage("Stand it up by the handle")
    stand_z = S.BED["mattress_top"] + wheel_drop + 0.01
    r.move(3.2, hands={side: (p0 - away * 0.10 + up * (stand_z - bar0[2]), R0)}, root=dict(h=0.95), follow=(idle,),
           gaze=bar0 + up * 0.3)
    r.hold(0.8)
    sp1, sR1 = r.suitcase_frame()
    r.event("suitcase_lift_check", rise_m=float(sp1[2] - sp[2]),
            handle_end_up_deg=float(np.degrees(np.arccos(np.clip(sR1[2, 1], -1, 1)))),
            contacts={side: r.contacts_between(r.hand_bodies[side], r.suit_ids)[0]})
    if sp1[2] - sp[2] < 0.05:
        raise RuntimeError("suitcase lift failed")
    off = r.hand[side][0] - bar()
    r.stage("Lift it off the bed")
    b1 = bar()
    r.move(1.2, hands={side: (b1 + up * 0.08 + off, R0)}, follow=(idle,))
    r.stage("Carry it past the end of the bed")
    r.move(3.0, hands={side: (np.array([floor_x, b1[1], b1[2] + 0.08]) + off, R0)},
           root=dict(x=floor_x + 0.50), gaze=np.array([floor_x, b1[1], 0.4]), follow=(idle,))
    r.hold(0.8)
    r.event("hanging_check", handle_end_up_deg=float(np.degrees(np.arccos(np.clip(r.suitcase_frame()[1][2, 1], -1, 1)))))
    off = r.hand[side][0] - bar()
    r.stage("Set it down on its wheels")
    r.move(2.4, hands={side: (np.array([floor_x, b1[1], wheel_drop + 0.03]) + off, R0)}, root=dict(h=0.70), follow=(idle,))
    # Square it up: bring the handle directly above the wheel footprint as it touches down.
    foot_local = np.array([0, -L / 2 - 0.03, (D + DL) / 2])
    for _ in range(3):
        cp, cR = r.suitcase_frame()
        foot = cp + cR @ foot_local
        b_now = bar()
        tgt = np.array([foot[0], foot[1], wheel_drop - 0.004])
        r.move(0.8, hands={side: (r.hand[side][0] + (tgt - b_now), R0)})
    r.hold(0.5)
    r.event("touchdown_tilt_deg", value=float(np.degrees(np.arccos(np.clip(r.suitcase_frame()[1][2, 1], -1, 1)))))
    r.stage("Let go of the handle")
    # Straighten the fingers out from under the bar (they point back over the case top),
    # slide the hand back off the bar, then lift it away.
    r.move(0.9, fingers={side: "press"})
    hp, hR = r.hand[side]
    r.move(0.7, hands={side: (hp - away * 0.09 + up * 0.01, hR)})
    hp, hR = r.hand[side]
    r.move(0.8, hands={side: (hp + up * 0.10, hR)}, root=dict(h=0.95))
    r.hold(0.4)
