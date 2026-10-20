"""Skills for the suitcase's controls: zip sliders, pull tabs, lock dials and buttons, the
trolley handle and the strap buckle.

Every one acts by contact only. Targets are fingertip (or pinch-point) positions computed
from the live pose of the part; the fingertip offset of each hand shape is measured on
the hand model itself (FK in a throwaway rewind), so the palm target that puts the pad
on the part is exact for that shape. Each skill then checks the part's own state (slider
position, dial reading, latch) rather than assuming the motion worked; dials are turned
closed-loop, one detent at a time, re-reading after every flick.
"""
import numpy as np
import scene as S
from runner import minjerk, TIP_R
from tasks import unit, hand_R, approach, clear_hands

HW = S.HARDWARE
_TIP = {}


def tip_offset(r, side, shape, finger=2):
    """Fingertip pad position in the palm frame for a finger shape (measured, cached)."""
    key = (side, shape, finger)
    if key not in _TIP:
        snap = r.snapshot()
        r.move(0.5, fingers={side: shape})
        p, R = r.hand_pose(side)
        _TIP[key] = R.T @ (r.fingertip(side, finger) - p)
        r.restore(snap)
    return _TIP[key]


def palm_for_tip(r, side, shape, tip, R, finger=2):
    """Palm pose that puts the fingertip of `shape` at `tip` with hand rotation R."""
    return tip - R @ tip_offset(r, side, shape, finger), R


def away_from_bar(r, side, back=0.28):
    """After letting go of the handle: up, then back towards the body, so the hand never
    comes down onto the grip (and its button) on its way to rest."""
    hp, hR = r.hand[side]
    r.move(0.5, hands={side: (hp + np.array([0, 0, 0.06]), hR)})
    yaw = r.root["yaw"]
    F = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    o = np.array([r.root["x"], r.root["y"], 0.0])
    hp, hR = r.hand[side]
    # pull the hand in to within 25 cm in front of the body, at the same height
    loc = hp - o
    loc[2] = 0.0                                   # horizontal only: never down onto the grip
    fwd = loc @ F
    r.move(0.8, hands={side: (hp - F * max(0.0, fwd - 0.25) - (loc - F * fwd) * 0.5, hR)})


def thumb_away(R, side, away):
    """Turn the hand 180 deg about the finger axis if its thumb points against `away`
    (the thumb must not stick out towards the case wall)."""
    thumb = R[:, 0] if side == "r" else -R[:, 0]
    if thumb @ away < 0:
        f = -R[:, 1]
        return hand_R(f, R[:, 2])                  # palm normal reversed: rotated about the finger axis
    return R


def case_axes(r):
    """Suitcase axes in the world: u along the length (+y, handle end), n out of the
    front side (+x), w out of the lid (+z)."""
    _, sR = r.suitcase_frame()
    return sR[:, 1], sR[:, 0], sR[:, 2]


def _tip_path(r, side, shape, R, path, T, stop=None):
    """Drive a fingertip along path(u), u in 0..1 (min-jerk), optionally stopping early."""
    n = max(1, int(T * 1000))
    for k in range(n):
        u = minjerk((k + 1) / n)
        r.hand[side] = palm_for_tip(r, side, shape, path(u), R)
        r.step()
        if stop is not None and k % 20 == 0 and stop():
            return True
    return False


def _withdraw(r, side, out, T=0.6):
    hp, hR = r.hand[side]
    r.move(T, hands={side: (hp + out * 0.05 + np.array([0, 0, 0.04]), hR)})


# ---------------------------------------------------------------- zip sliders
def push_slider(r, k, q_goal, side=None):
    """Push zip slider k ('a' from the wheel end, 'b' from the handle end) along the
    track to joint position q_goal with the index fingertip on its trailing face."""
    side = side or ("l" if k == "a" else "r")
    j, qa = r.hw["slider"][k]
    q0 = r.d.qpos[qa]
    if abs(q_goal - q0) < 0.003:
        return
    u, n, w = case_axes(r)
    s = np.sign(q_goal - q0)
    sl = HW["slider"]
    tip_r = TIP_R
    c0, _ = r.body_pose(f"slider_{k}")
    # fingertip just behind the slider (trailing face), a little outward of its centre
    back = -u * s * (sl["half"][1] + tip_r + 0.003)
    f = unit(-w * 0.8 - n * 0.45)                    # index points down, slightly towards the case
    R = thumb_away(hand_R(f, u * s), side, n)
    shape = "point"
    start = c0 + back + n * 0.002
    r.stage("Push the zip slider along" if s > 0 else "Pull the zip slider back")
    pre = palm_for_tip(r, side, shape, start + w * 0.05 + n * 0.03, R)
    approach(r, {side: pre}, fingers={side: shape}, root=dict(h=0.86), gaze=c0)
    r.move(0.6, hands={side: palm_for_tip(r, side, shape, start, R)})
    dist = abs(q_goal - q0) + 0.004
    done = lambda: abs(r.d.qpos[qa] - q_goal) < 0.002 or (s > 0 and r.d.qpos[qa] >= q_goal) or (s < 0 and r.d.qpos[qa] <= q_goal)
    T = 0.6 + dist / 0.12
    _tip_path(r, side, shape, R, lambda x: start + u * s * dist * x, T, stop=done)
    r.hold(0.15)
    reached = float(r.d.qpos[qa])
    r.event("slider_pushed", slider=k, goal=round(float(q_goal), 4), reached=round(reached, 4))
    _withdraw(r, side, n)
    return reached


def slider_goal(k, opening):
    """Slider joint position for a zip opening fraction (1 = open, 0 = closed)."""
    sl = HW["slider"]
    if k == "a":
        span = (HW["dock_y"] - sl["gap"]) - sl["open_a"]
        return span * (1 - opening)
    span = sl["open_b"] - (HW["dock_y"] + sl["gap"])
    return -span * (1 - opening)


def run_zip(r, frm, to):
    """Zip (to < frm) or unzip (to > frm) to an opening fraction by pushing both sliders.
    The left hand presses the lid down while the right closes from the handle end; then
    the hands swap for the wheel-end slider."""
    closing = to < frm
    if closing and frm > 0.999:
        press_lid(r)
    order = ("b", "a") if closing else ("a", "b")
    for k in order:
        push_slider(r, k, slider_goal(k, to))
    zc = r.zip_closed_fraction()
    r.event("zip_check", closed_fraction=round(zc, 3), goal_closed=round(1 - to, 3),
            lid_angle_deg=float(np.degrees(r.d.qpos[r.m.joint("lid_hinge").qposadr[0]])), lid_held=bool(r.zip_locked))
    clear_hands(r)
    if abs(zc - (1 - to)) > 0.04:
        raise RuntimeError(f"zip at {zc:.2f} closed, wanted {1 - to:.2f}")
    if closing and (1 - to) > 0.05 and not r.zip_locked:
        raise RuntimeError("zip closed but the lid is not held: lid was not down")


def press_lid(r):
    """Both palms press the closed lid down before the zip is run."""
    sp, sR = r.suitcase_frame()
    c = S.SUITCASE
    top = sp + sR @ np.array([0, 0, c["base_depth"] + c["lid_depth"]])
    yaw = r.root["yaw"]
    F = np.array([np.cos(yaw), np.sin(yaw), 0.0]); right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.array([0, 0, -1.0])
    Rp = hand_R(F, down)
    pr, pl = top + right * 0.16 - F * 0.03, top - right * 0.16 - F * 0.03
    r.stage("Press the lid down")
    approach(r, {"r": (pr + [0, 0, 0.06], Rp), "l": (pl + [0, 0, 0.06], Rp)}, fingers={"r": "press", "l": "press"},
             root=dict(h=0.86), gaze=top)
    r.move(1.0, hands={"r": (pr - [0, 0, 0.004], Rp), "l": (pl - [0, 0, 0.004], Rp)})
    r.hold(0.4)
    cn = r.contacts_between(r.hand_bodies["r"] + r.hand_bodies["l"], r.suit_ids)
    r.event("lid_press", contacts=cn[0], force_N=round(cn[1], 1))
    for s in "rl":
        hp, hR = r.hand[s]
        r.move(0.4, hands={s: (hp + [0, 0, 0.08], hR)})


# ---------------------------------------------------------------- pull tabs
def flip_tab(r, k, side="r"):
    """Swing pull tab k up into the lock's dock: it hangs flat against the side and turns
    in that plane; the index fingertip pushes its edge round the pivot from where it
    hangs to upright (A over the wheel-end side, B over the handle-end side), where its
    end clicks into the slot under the lock."""
    jt, qt = r.hw["tab"][k]
    u, n, w = case_axes(r)
    P, _ = r.body_pose(f"tab_{k}")
    sgn = -1 if k == "a" else 1
    rho = 0.060
    gap = HW["tab"]["r"] + TIP_R + 0.001
    q0 = float(r.d.qpos[qt])
    q1 = np.pi + 0.05
    d = lambda q: -w * np.cos(q) + sgn * u * np.sin(q)
    m = lambda q: w * np.sin(q) + sgn * u * np.cos(q)        # direction the tab moves as q grows
    tip = lambda q: P + rho * d(q) - gap * m(q)
    f = unit(-n * 0.8 - w * 0.6)
    R = thumb_away(hand_R(f, -u * sgn), side, n)
    shape = "point"
    r.stage("Swing the zip pull up into the lock")
    start = tip(max(q0 - 0.1, 0.0))
    approach(r, {side: palm_for_tip(r, side, shape, start + n * 0.04, R)}, fingers={side: shape}, root=dict(h=0.84), gaze=P)
    r.move(0.6, hands={side: palm_for_tip(r, side, shape, start, R)})
    _tip_path(r, side, shape, R, lambda x: tip(max(q0 - 0.1, 0.0) + (q1 - max(q0 - 0.1, 0.0)) * x), 2.2)
    r.hold(0.3)
    ok = r.mech["tabs"][k]
    r.event("tab_flip", tab=k, angle_deg=round(float(np.degrees(r.d.qpos[qt])), 1), docked=bool(ok))
    _withdraw(r, side, n)
    if not ok:
        raise RuntimeError(f"pull {k} did not latch ({np.degrees(r.d.qpos[qt]):.0f} deg)")


def insert_pulls(r):
    for k in ("a", "b"):
        if not r.mech["tabs"][k]:
            flip_tab(r, k)
    clear_hands(r)


# ---------------------------------------------------------------- buttons
def press_button(r, body, joint_q, travel, hold=0.3, times=1, side="r", face=None, label="the button"):
    """Index fingertip on the button face, pushed in until the button bottoms out
    (measured from its joint), held, released; `times` times."""
    u, n, w = case_axes(r)
    face = n if face is None else face
    c, Rb = r.body_pose(body)
    bid = r.m.body(body).id
    half = r.m.geom_size[[g for g in range(r.m.ngeom) if r.m.geom_bodyid[g] == bid][0]]
    depth = float(abs(Rb.T @ face) @ half)              # half extent along the push axis
    tip_r = TIP_R
    f = unit(-face * 0.95 - w * 0.30)
    R = thumb_away(hand_R(f, u), side, face)
    shape = "point"
    surf = c + face * (depth + tip_r)
    r.stage(f"Press {label}")
    approach(r, {side: palm_for_tip(r, side, shape, surf + face * 0.03 + w * 0.02, R)}, fingers={side: shape},
             root=dict(h=0.86), gaze=c)
    pressed_for = []
    for i in range(times):
        r.move(0.5, hands={side: palm_for_tip(r, side, shape, surf + face * 0.004, R)})
        r.move(0.5, hands={side: palm_for_tip(r, side, shape, surf - face * (travel + 0.0025), R)})
        t_in = None
        n_hold = int(hold * 1000)
        for kk in range(n_hold):
            r.step()
            if r.d.qpos[joint_q] > 0.7 * travel and t_in is None:
                t_in = kk
        pressed_for.append(0.0 if t_in is None else (n_hold - t_in) / 1000)
        r.move(0.4, hands={side: palm_for_tip(r, side, shape, surf + face * 0.01, R)})
    r.event("button_press", button=body, times=times, held_s=[round(x, 2) for x in pressed_for])
    _withdraw(r, side, face)
    if min(pressed_for) < 0.5 * hold:
        raise RuntimeError(f"{label} did not go in (held {pressed_for})")


def press_lock_button(r, hold=0.3, times=1):
    press_button(r, "lock_btn", r.hw["lock_btn"], HW["button"]["travel"], hold, times, label="the lock's release button")
    clear_hands(r)


# ---------------------------------------------------------------- dials
def flick_dial(r, k, step, side="r"):
    """Drag the exposed rim of dial k one detent (step = +1 / -1)."""
    u, n, w = case_axes(r)
    c, _ = r.body_pose(f"dial_{k}")
    rd = HW["dial_r"]
    tip_r = TIP_R
    # rim point facing out (+n); increasing angle moves that point towards -w
    t = -w * step
    f = unit(-n * 0.95 - w * 0.30)
    R = thumb_away(hand_R(f, u), side, n)
    shape = "point"
    drag = 2 * np.pi * rd / 10 * 1.15
    contact = c + n * (rd + tip_r - 0.0009)
    a = contact - t * drag / 2
    b = contact + t * drag / 2
    r.move(0.35, hands={side: palm_for_tip(r, side, shape, a + n * 0.004, R)})
    r.move(0.25, hands={side: palm_for_tip(r, side, shape, a, R)})
    _tip_path(r, side, shape, R, lambda x: a + (b - a) * x, 0.45)
    r.move(0.25, hands={side: palm_for_tip(r, side, shape, b + n * 0.005, R)})
    r.hold(0.1)


def turn_dial(r, k, to=None, by=None, side="r"):
    """Turn lock dial k (1-3) to a digit, or by a number of clicks: read, flick the
    shorter way round, re-read, until the dial shows the goal."""
    read = lambda: r.dial_digits()[k - 1]
    start = read()
    goal = to if to is not None else (start + by) % 10
    c, _ = r.body_pose(f"dial_{k}")
    u, n, w = case_axes(r)
    f = unit(-n * 0.95 - w * 0.30)
    R = thumb_away(hand_R(f, u), side, n)
    r.stage(f"Turn the {['left', 'middle', 'right'][k - 1]} dial to {goal}")
    approach(r, {side: palm_for_tip(r, side, "point", c + n * 0.04 + w * 0.02, R)}, fingers={side: "point"},
             root=dict(h=0.86), gaze=c)
    flicks = 0
    while read() != goal and flicks < 14:
        diff = (goal - read()) % 10
        step = 1 if (diff <= 5 if by is None else (by > 0)) else -1
        flick_dial(r, k, step, side)
        flicks += 1
    r.event("dial_turned", dial=k, start=start, goal=goal, reading=read(), flicks=flicks)
    _withdraw(r, side, n)
    if read() != goal:
        raise RuntimeError(f"dial {k} shows {read()}, wanted {goal}")


def set_code(r, code):
    for k, ch in enumerate(code):
        if r.dial_digits()[k] != int(ch):
            turn_dial(r, k + 1, to=int(ch))
    clear_hands(r)


# ---------------------------------------------------------------- trolley handle
def bar_hook_pose(r, side, lat, pitch_deg=15, knuckle_up=0.010, depth=0.25, lift=0.0, back=0.0):
    """The trolley-bar hook used to stand the case up (skills_case.lift_from_bed_foot):
    forearm pitched down towards the bar, palm facing down and away, knuckles over the gap
    behind the bar so the hanging fingers drop into it and curl under ('bar_fist').
    `lat` is the offset along the person's right."""
    hb = S.SUITCASE["bar"]
    u, n, w = case_axes(r)
    b = r.body_pose("trolley")[0]
    away = b - np.array([r.root["x"], r.root["y"], b[2]]); away[2] = 0; away = unit(away)
    up = np.array([0, 0, 1.0])
    right = np.array([away[1], -away[0], 0.0])
    # free space behind the bar: towards the case end (lying) or down to the case top (upright)
    gap = r.body_pose("trolley")[0] - r.suitcase_frame()[0]
    gap = abs(gap @ u) - S.SUIT_L / 2 - hb["r"]
    hp, hR = r.hand_pose(side)
    k = np.mean([r.d.xpos[r.m.body(f"{i}proxph_{side}").id] for i in (2, 3, 4, 5)], axis=0)
    knuck = hR.T @ (k - hp)
    pitch = np.radians(pitch_deg)
    Rh = hand_R(away * np.cos(pitch) - up * np.sin(pitch), -up * np.cos(pitch) - away * np.sin(pitch))
    return (b + right * lat + away * (hb["r"] + depth * min(gap, 0.07) - back) + up * (hb["r"] + knuckle_up + lift)
            - Rh @ knuck, Rh)


def hook_bar(r, side, lat, T=0.8):
    """Hook one hand over the trolley bar: raise it beside the body above the bar, over it,
    down onto it, and close the fist (the stand-up hook)."""
    hook = bar_hook_pose(r, side, lat, pitch_deg=15, knuckle_up=0.010, depth=0.25)
    hp, hR = r.hand[side]
    top = max(hook[0][2], hp[2]) + 0.16
    r.move(T, hands={side: (np.array([hp[0], hp[1], top]), hook[1])}, fingers={side: "hook_open"})
    r.move(T, hands={side: (hook[0] + np.array([0, 0, 0.10]), hook[1])})
    r.move(T, hands={side: (hook[0] + np.array([0, 0, 0.03]), hook[1])})
    r.move(0.7, hands={side: hook})
    r.move(0.8, fingers={side: "bar_fist"})
    r.hold(0.2)


def set_trolley(r, level):
    """Extend or retract the telescopic handle to a stop: one hand hooks the bar (the
    stand-up hook) while the index finger of the other holds the button in the grip;
    both move with the bar to the stop; the button is let go and the latch catches there.
    Case lying on the bed (from its foot): right hand hooks, left finger on the button.
    Case standing (from beside it): left hand hooks, right finger on the button."""
    tr = HW["trolley"]
    j, qa = r.hw["trolley"]
    q0 = float(r.d.qpos[qa]); q1 = tr["levels"][level]
    u, n, w = case_axes(r)
    upright = u[2] > 0.7
    if upright:
        return _set_trolley_upright(r, level)
    hk, bt = ("l", "r") if upright else ("r", "l")
    b0 = r.body_pose("trolley")[0]
    s = np.sign(q1 - q0)
    shape_b = "point"
    away = b0 - np.array([r.root["x"], r.root["y"], b0[2]]); away[2] = 0; away = unit(away)
    btn_face = lambda: r.body_pose("trolley_btn")[0] + u * (tr["button"]["half"][1] + TIP_R)
    if upright:
        f_b = unit(-u * 0.9 + away * 0.3)
        Rbtn = thumb_away(hand_R(f_b, away), bt, -away)
    else:
        f_b = unit(-u * 0.85 - np.array([0, 0, 1.0]) * 0.5)
        Rbtn = thumb_away(hand_R(f_b, unit(np.cross(f_b, away))), bt, -np.cross(u, away))
    r.stage("Grip the handle and hold its button in")
    r.move(0.6, root=dict(h=0.95 if upright else 0.74), gaze=b0)
    hook_bar(r, hk, 0.06 if hk == "l" else -0.06 if upright else 0.060)
    approach(r, {bt: palm_for_tip(r, bt, shape_b, btn_face() + u * 0.04, Rbtn)}, fingers={bt: shape_b})
    # pressed well past the button's travel: the fingertip keeps it in even when the bar
    # lags the hands a few millimetres while it slides
    r.move(0.6, hands={bt: palm_for_tip(r, bt, shape_b, btn_face() - u * (tr["button"]["travel"] + 0.006), Rbtn)})
    r.hold(0.3)
    if r.mech["trolley_level"] is not None and q1 != q0:
        raise RuntimeError("trolley button not held in")
    r.stage("Pull the handle out" if s > 0 else "Push the handle in")
    h0 = {x: r.hand[x] for x in "rl"}
    dist = q1 - q0
    n_st = int((1.0 + abs(dist) / 0.15) * 1000)
    for kk in range(n_st):
        x = minjerk((kk + 1) / n_st) * dist
        for sd in "rl":
            r.hand[sd] = (h0[sd][0] + u * x, h0[sd][1])
        r.step()
    r.hold(0.2)
    for _ in range(4):                      # feel for the stop: correct what the grip lost or overshot
        err = q1 - float(r.d.qpos[qa])
        if abs(err) < 0.002:
            break
        hs = {sd: r.hand[sd] for sd in "rl"}
        r.move(0.4, hands={sd: (hs[sd][0] + u * err, hs[sd][1]) for sd in "rl"})
        r.hold(0.15)
    # let the button out: the latch catches at the stop
    hb_ = r.hand[bt]
    r.move(0.5, hands={bt: (hb_[0] + u * 0.025, hb_[1])})
    r.hold(0.3)
    lv = r.mech["trolley_level"]
    r.event("trolley_set", goal=level, latched=lv, q=round(float(r.d.qpos[qa]), 4))
    # let go of the bar rising while the fingers open (nothing presses the button on the way)
    hh = r.hand[hk]
    r.move(0.3, hands={hk: (hh[0] + np.array([0, 0, 0.01]), hh[1])})
    hh = r.hand[hk]
    r.move(1.0, hands={hk: (hh[0] + np.array([0, 0, 0.06]), hh[1])}, fingers={hk: "open"})
    hh = r.hand[hk]
    r.move(0.6, hands={hk: (hh[0] + np.array([0, 0, 0.08]), hh[1])})
    clear_hands(r)
    lv = r.mech["trolley_level"]
    if lv != level:
        raise RuntimeError(f"handle latched at {lv}, wanted {level} (q={r.d.qpos[qa]:.3f})")


def _set_trolley_upright(r, level, side="l"):
    """Standing case: the button is on top of the grip, so an overhand fist round the
    middle of the bar squeezes it in; the hand lifts (or pushes down) the handle to the
    stop, feels for it, and opens, letting the button out so the latch catches there."""
    tr = HW["trolley"]
    j, qa = r.hw["trolley"]
    q0 = float(r.d.qpos[qa]); q1 = tr["levels"][level]
    u, n, w = case_axes(r)
    r.stage("Grip the handle over its button")
    r.move(0.5, root=dict(h=0.95))
    hook_bar(r, side, 0.0)
    # squeeze: press the palm onto the button until the latch lets go
    for k in range(6):
        if r.mech["trolley_level"] is None:
            break
        hp, hR = r.hand[side]
        r.move(0.3, hands={side: (hp - u * 0.0015, hR)})
    if r.mech["trolley_level"] is not None and q1 != q0:
        raise RuntimeError("trolley button not pressed by the grip")
    r.stage("Pull the handle out" if q1 > q0 else "Push the handle in")
    h0 = r.hand[side]
    dist = q1 - q0
    n_st = int((1.0 + abs(dist) / 0.15) * 1000)
    for kk in range(n_st):
        x = minjerk((kk + 1) / n_st) * dist
        r.hand[side] = (h0[0] + u * x, h0[1])
        r.step()
    r.hold(0.2)
    for _ in range(4):                      # feel for the stop: top up what the grip lost
        err = q1 - float(r.d.qpos[qa])
        if abs(err) < 0.002:
            break
        hp, hR = r.hand[side]
        r.move(0.4, hands={side: (hp + u * err * 1.2, hR)})
        r.hold(0.15)
    r.stage("Let go of the handle")
    hp, hR = r.hand[side]
    r.move(1.0, hands={side: (hp + u * 0.05, hR)}, fingers={side: "open"})
    r.hold(0.4)
    lv = r.mech["trolley_level"]
    r.event("trolley_set", goal=level, latched=lv, q=round(float(r.d.qpos[qa]), 4))
    away_from_bar(r, side)
    if lv != level:
        raise RuntimeError(f"handle latched at {lv}, wanted {level} (q={r.d.qpos[qa]:.3f})")


# ---------------------------------------------------------------- strap buckle
_PINCH = {}


def pinch_geometry(r, side, shape):
    """Midpoint and thumb->index direction of the pinch (closest points between the two
    distal collision capsules), in the palm frame; measured on the model, cached."""
    import mujoco
    key = (side, shape)
    if key not in _PINCH:
        snap = r.snapshot()
        r.move(0.5, fingers={side: shape})
        ft = np.zeros(6)
        gt, gi = r.m.geom(f"distal_thumb_coll_{side}").id, r.m.geom(f"distph2_coll_{side}").id
        mujoco.mj_geomDistance(r.m, r.d, gt, gi, 0.2, ft)
        p, R = r.hand_pose(side)
        a, b = ft[:3], ft[3:]
        _PINCH[key] = (R.T @ ((a + b) / 2 - p), unit(R.T @ (b - a)))
        r.restore(snap)
    return _PINCH[key]


def _pinch_pose(r, side, mid, axis, palm_dir, shape="pinch_open", fwd=None):
    """Hand pose whose pinch midpoint (in `shape`) is at `mid` with the thumb->index
    axis exactly along `axis` (either sign), spun about that axis so the palm faces as
    close to palm_dir as possible."""
    from scipy.spatial.transform import Rotation as Rot
    m_h, a_h = pinch_geometry(r, side, shape)
    best = None
    for sgn in (1, -1):
        tgt = unit(axis) * sgn
        v = np.cross(a_h, tgt); c = float(a_h @ tgt)
        if np.linalg.norm(v) < 1e-9:
            R0 = np.eye(3) if c > 0 else Rot.from_rotvec(np.pi * unit(np.cross(a_h, [1, 0, 0]))).as_matrix()
        else:
            R0 = Rot.from_rotvec(unit(v) * np.arctan2(np.linalg.norm(v), c)).as_matrix()
        for phi in np.linspace(-np.pi, np.pi, 360, endpoint=False):
            R = Rot.from_rotvec(tgt * phi).as_matrix() @ R0
            sc = -(-R[:, 2]) @ unit(palm_dir) - (0 if fwd is None else 2.0 * ((-R[:, 1]) @ unit(fwd)))
            if best is None or sc < best[0]:
                best = (sc, R)
    R = best[1]
    return mid - R @ m_h, R


def _pad_forces(r, side, body):
    """Normal force (N) of the thumb and of the index finger on `body`."""
    import mujoco
    b = r.m.body(body).id
    th = {r.m.body(f"distal_thumb_{side}").id, r.m.body(f"proximal_thumb_{side}").id}
    ix = {r.m.body(f"distph2_{side}").id, r.m.body(f"midph2_{side}").id}
    f = np.zeros(6); out = [0.0, 0.0]
    for i, c in enumerate(r.d.contact[: r.d.ncon]):
        b1, b2 = r.contact_bodies(c)
        if b in (b1, b2):
            o = b2 if b1 == b else b1
            mujoco.mj_contactForce(r.m, r.d, i, f)
            if o in th:
                out[0] += abs(f[0])
            elif o in ix:
                out[1] += abs(f[0])
    return out


def squeeze_on(r, side, mid, axis, palm_dir, until=None, T=1.2, fwd=None, body=None):
    """Close a wide pinch on something at `mid`: the fingers go from pinch_wide to pinch
    in steps while the hand shifts so the pads stay centred on `mid` (the pinch midpoint
    moves in the hand as the thumb and index close). With `body`, the pads' contact forces
    re-centre the hand: if one pad presses while the other has not arrived, the hand moves
    towards the free side before closing further. Stops early when until() holds."""
    shapes = ["pinch_wide"] + [f"pinch_w{k}" for k in range(1, 5)] + ["pinch"]
    shift = np.zeros(3)
    ax = unit(axis)
    for sh in shapes[1:]:
        pose = _pinch_pose(r, side, mid + shift, axis, palm_dir, shape=sh, fwd=fwd)
        r.move(T / (len(shapes) - 1), hands={side: pose}, fingers={side: sh})
        if body is not None:
            for _ in range(4):
                ft, fi = _pad_forces(r, side, body)
                if (ft > 2.0) == (fi > 2.0):
                    break
                # the thumb sits at -axis of the pinch midpoint in hand terms: find which pad
                # presses and move the hand away from it along the axis by 2 mm
                tpos = r.fingertip(side, 1)
                towards_thumb = unit(tpos - (mid + shift))
                sgn = 1.0 if ft > fi else -1.0
                d = -sgn * ax * np.sign(towards_thumb @ ax) * 0.002
                shift = shift + d
                hp, hR = r.hand[side]
                r.move(0.15, hands={side: (hp + d, hR)})
        if until is not None and until():
            return sh
    return shapes[-1]


def _held(r, side, body, force=2.0):
    """Thumb and index both pressing on `body` with at least `force` N each."""
    import mujoco
    b = r.m.body(body).id
    th = {r.m.body(f"distal_thumb_{side}").id, r.m.body(f"proximal_thumb_{side}").id}
    ix = {r.m.body(f"distph2_{side}").id, r.m.body(f"midph2_{side}").id}
    f = np.zeros(6); got = {"t": 0.0, "i": 0.0}
    for i, c in enumerate(r.d.contact[: r.d.ncon]):
        b1, b2 = r.contact_bodies(c)
        if b in (b1, b2):
            o = b2 if b1 == b else b1
            mujoco.mj_contactForce(r.m, r.d, i, f)
            if o in th:
                got["t"] += abs(f[0])
            elif o in ix:
                got["i"] += abs(f[0])
    return got["t"] > force and got["i"] > force


def unfasten_strap(r, side="r"):
    """Squeeze the side-release tabs of the buckle between thumb and index, then pull the
    male half out along the strap."""
    u, n, w = case_axes(r)
    c, Rb = r.body_pose("buckle_male")
    tabs_mid = c + u * 0.004 + w * 0.001
    r.stage("Squeeze the buckle's side tabs")
    pose = _pinch_pose(r, side, tabs_mid, n, -w, shape="pinch_wide", fwd=u)
    approach(r, {side: (pose[0] + w * 0.05, pose[1])}, fingers={side: "pinch_wide"}, root=dict(h=0.86), gaze=c)
    r.move(0.6, hands={side: pose})
    sh = squeeze_on(r, side, tabs_mid, n, -w, until=lambda: not r.mech["buckle"], fwd=u)
    r.hold(0.2)
    released = not r.mech["buckle"]
    r.stage("Pull the buckle apart")
    hp, hR = r.hand[side]
    r.move(0.9, hands={side: (hp + u * HW["strap"]["travel"] * 0.8, hR)})
    r.move(0.4, fingers={side: "pinch_wide"})
    _withdraw(r, side, w)
    q = float(r.d.qpos[r.hw["buckle"][1]])
    r.event("strap_unfastened", released=released, squeeze_shape=sh, open_m=round(q, 4))
    clear_hands(r)
    if r.mech["buckle"] or q < 0.02:
        raise RuntimeError(f"buckle still closed (latched {r.mech['buckle']}, {q:.3f} m)")


def fasten_strap(r, side="r"):
    """Pinch the loose male half and push it into the female until it clicks."""
    u, n, w = case_axes(r)
    c, Rb = r.body_pose("buckle_male")
    grip = c + u * 0.006 + w * 0.001
    r.stage("Pinch the buckle")
    pose = _pinch_pose(r, side, grip, n, -w, shape="pinch_wide", fwd=u)
    approach(r, {side: (pose[0] + w * 0.05, pose[1])}, fingers={side: "pinch_wide"}, root=dict(h=0.86), gaze=c)
    r.move(0.6, hands={side: pose})
    squeeze_on(r, side, grip, n, -w, fwd=u, until=lambda: _held(r, side, "buckle_male"))
    r.hold(0.2)
    r.stage("Push the buckle home")
    q = float(r.d.qpos[r.hw["buckle"][1]])
    hp, hR = r.hand[side]
    r.move(1.0, hands={side: (hp - u * (q + 0.005), hR)})
    r.hold(0.3)
    r.move(0.4, fingers={side: "pinch_wide"})
    _withdraw(r, side, w)
    r.event("strap_fastened", latched=bool(r.mech["buckle"]))
    clear_hands(r)
    if not r.mech["buckle"]:
        raise RuntimeError("buckle did not click shut")


# ---------------------------------------------------------------- rolling the case
def grip_handle_top(r, side="l"):
    """Overhand grip on the bar of the extended handle of the standing case, beside the
    button in the middle of the grip (pressing it would let the handle collapse)."""
    r.stage("Grip the trolley handle")
    r.move(0.5, root=dict(h=0.95))
    hook_bar(r, side, 0.062 if side == "l" else -0.062)
    r.hold(0.1)


def _upright_bar_xy(r):
    """Where the bar must be (horizontally) for the standing case to be upright: above the
    point of the footprint it sits over when the case is vertical."""
    sp, sR = r.suitcase_frame()
    c = S.SUITCASE; hb = c["bar"]
    wz = (c["wheel_z_base"] + c["hinge_h"] + c["base_depth"] - c["hinge_h"] + c["wheel_z_lid"]) / 2
    foot = sp + sR @ np.array([0.0, c["wheel_y"], wz])            # centre of the four wheels
    off = sR @ np.array([hb["x"], 0.0, hb["z"] - wz])              # bar offset across the footprint
    off[2] = 0
    return (foot + off)[:2]


def roll_case(r, stand, yaw, route, case_goal, side="l", speed=0.35):
    """Roll the standing case alongside on its spinner wheels: stand beside it, take the
    handle with the near hand and walk the route; the hand keeps its place by the body
    and, as a person does, steers the bar back over the wheels whenever the case leans
    (a slow correction on the measured lean). Then let go while rising and step away."""
    r.stage("Step beside the case")
    r.walk([stand], final_yaw=yaw, speed=0.5)
    grip_handle_top(r, side)
    r.stage("Roll the case")
    pts = [np.array([r.root["x"], r.root["y"]])] + [np.array(p, float) for p in route]
    seg = [np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])]
    cum = np.concatenate([[0], np.cumsum(seg)]); Ltot = cum[-1]
    o, Rr = r.root_frame()
    local = {s_: (Rr.T @ (r.hand[s_][0] - o), Rr.T @ r.hand[s_][1]) for s_ in "rl"}
    corr = np.zeros(2)
    n = int((Ltot / speed + 1.5) * 1000)
    yaw_prev = r.root["yaw"]
    for k in range(n):
        sv = Ltot * minjerk((k + 1) / n)
        i = min(np.searchsorted(cum, sv, side="right") - 1, len(seg) - 1)
        p = pts[i] + (pts[i + 1] - pts[i]) * ((sv - cum[i]) / max(seg[i], 1e-9))
        ahead_i = min(np.searchsorted(cum, min(sv + 0.4, Ltot), side="right") - 1, len(seg) - 1)
        d = pts[ahead_i + 1] - pts[ahead_i]
        heading = np.arctan2(d[1], d[0])
        dyaw = (heading - yaw_prev + np.pi) % (2 * np.pi) - np.pi
        yaw_prev = yaw_prev + np.clip(dyaw * 3.0, -1.0, 1.0) * 0.001
        r.root.update(x=float(p[0]), y=float(p[1]), yaw=float(yaw_prev))
        o, Rr = r.root_frame()
        if k % 10 == 0:
            err = _upright_bar_xy(r) - r.body_pose("trolley")[0][:2]
            corr = np.clip(corr + 0.04 * err, -0.15, 0.15)
        for s_ in "rl":
            hp = o + Rr @ local[s_][0]
            if s_ == side:
                hp = hp + np.array([corr[0], corr[1], 0.0])
            r.hand[s_] = (hp, Rr @ local[s_][1])
        r.step()
    r.hold(0.6)
    for _ in range(10):                                  # settle it upright before letting go
        err = _upright_bar_xy(r) - r.body_pose("trolley")[0][:2]
        hp, hR = r.hand[side]
        r.move(0.15, hands={side: (hp + np.array([*(0.5 * err), 0.0]), hR)})
    r.stage("Let go of the handle")
    hp, hR = r.hand[side]
    r.move(1.2, hands={side: (hp + np.array([0, 0, 0.06]), hR)}, fingers={side: "open"})
    hp, hR = r.hand[side]
    r.move(0.8, hands={side: (hp + np.array([0, 0, 0.12]), hR)})
    yaw = r.root["yaw"]
    away = np.array([np.sin(yaw), -np.cos(yaw)]) * (1 if side == "l" else -1)     # away from the case
    q = np.array([r.root["x"], r.root["y"]]) + 0.20 * away
    away_from_bar(r, side)
    r.move(1.0, root=dict(x=float(q[0]), y=float(q[1])), follow=("r", "l"))
    r.hold(1.0)
    sp, sR = r.suitcase_frame()
    tilt = float(np.degrees(np.arccos(np.clip(sR[2, 1], -1, 1))))
    err = float(np.linalg.norm(sp[:2] - np.asarray(case_goal)))
    r.event("roll_check", case_xy=sp[:2].round(3).tolist(), goal=list(case_goal), error_m=round(err, 3), tilt_deg=round(tilt, 1))
    if tilt > 10:
        raise RuntimeError(f"case tipped while rolling ({tilt:.0f} deg)")
    if err > 0.30:
        raise RuntimeError(f"case {err:.2f} m from where it should be")


# ---------------------------------------------------------------- folding cloth
def cloth_grid(r):
    """Vertex positions of the towel as an (nx, ny, 3) grid (x across the counter, y along)."""
    nx, ny = S.CLOTH["count"]
    return r.cloth_vertices().reshape(nx, ny, 3)


def cloth_extent(r):
    v = r.cloth_vertices()
    return float(v[:, 0].max() - v[:, 0].min()), float(v[:, 1].max() - v[:, 1].min())


def fold_cloth(r, frac=0.5):
    """Fold the towel away from the person: slide both flat hands, palms up, under its
    near edge where it hangs over the counter's edge, lift it clear of the counter's lip,
    carry it up and over `frac` of the depth beyond the fold line (half: to the far
    edge), then tip the fingertips down so the edge slides off them onto the towel, and
    lift the hands away (no dragging back under the cloth)."""
    g = cloth_grid(r)
    nx, ny = g.shape[:2]
    near = 0 if g[0, :, 0].mean() < g[-1, :, 0].mean() else nx - 1
    far = nx - 1 - near
    edge = g[near]
    toward = unit(np.append(g[far, ny // 2, :2] - g[near, ny // 2, :2], 0.0))
    top = S.STATIONS[S.CLOTH["station"]].top
    span = float((g[far, :, :2].mean(0) - g[near, :, :2].mean(0)) @ toward[:2])
    up = np.array([0, 0, 1.0])
    R = hand_R(toward, up)                                    # fingers towards the counter, palm up
    spots = {"r": edge[0] + (edge[-1] - edge[0]) * 0.18, "l": edge[0] + (edge[-1] - edge[0]) * 0.82}
    low = min(float(v[2]) for v in edge)
    palm = {s_: np.array([spots[s_][0], spots[s_][1], low - 0.03]) - toward * 0.10 for s_ in "rl"}
    r.stage(f"Slide the hands under the {S.CLOTH['label']}'s edge")
    approach(r, {s_: (palm[s_] - toward * 0.06, R) for s_ in "rl"}, fingers={"r": "flat", "l": "flat"},
             root=dict(h=0.86), gaze=(spots["l"] + spots["r"]) / 2)
    r.move(0.8, hands={s_: (palm[s_], R) for s_ in "rl"})
    r.stage(f"Lift the {S.CLOTH['label']}'s edge")
    # catch the edge on the fingers, draw back clear of the counter's front lip while
    # rising past it, then up above the towel
    r.move(0.7, hands={s_: (palm[s_] + up * (low + 0.035 - palm[s_][2]), R) for s_ in "rl"})
    r.move(0.8, hands={s_: (palm[s_] - toward * 0.045 + up * (top + 0.03 - palm[s_][2]), R) for s_ in "rl"})
    lifted = {s_: (palm[s_] - toward * 0.03 + up * (top + 0.10 - palm[s_][2])) for s_ in "rl"}
    r.move(0.8, hands={s_: (lifted[s_], R) for s_ in "rl"})
    r.stage(f"Fold the {S.CLOTH['label']} over")
    over = toward * (span * (2 * frac) + 0.02)
    r.move(2.0, hands={s_: (lifted[s_] + over, R) for s_ in "rl"})
    r.move(0.8, hands={s_: (lifted[s_] + over - up * 0.04, R) for s_ in "rl"})
    r.stage("Let the edge slide off the fingers")
    # tip the fingertips down to about 55 deg, raising the wrist so the tips stay just above
    # the towel: the edge slides off forward under its own weight
    Rd = hand_R(unit(toward * 0.57 - up * 0.82), unit(up * 0.57 + toward * 0.82))
    tip_drop = 0.09
    r.move(0.9, hands={s_: (lifted[s_] + over - up * 0.04 + up * tip_drop * 0.6 - toward * 0.03, Rd) for s_ in "rl"})
    r.hold(0.6)
    r.move(0.8, hands={s_: (r.hand[s_][0] + up * 0.12, Rd) for s_ in "rl"})
    r.move(0.6, hands={s_: (r.hand[s_][0] - toward * 0.10, R) for s_ in "rl"})
    clear_hands(r)
    r.hold(0.8)


def fold(r, style="half"):
    d0, w0 = cloth_extent(r)
    if style != "half":
        raise RuntimeError("only folding in half is simulated")
    fold_cloth(r, 0.5)
    want = 0.5
    d1, w1 = cloth_extent(r)
    ratio = d1 / d0
    r.event("fold_check", style=style, depth_before=round(d0, 3), depth_after=round(d1, 3), ratio=round(ratio, 2),
            width_after=round(w1, 3))
    if abs(ratio - want) > 0.2:
        raise RuntimeError(f"fold ended at {ratio:.2f} of the depth, wanted {want:.2f}")
