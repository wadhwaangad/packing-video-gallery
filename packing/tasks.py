"""Task-level skills: bimanual squeeze, one-hand wrap, placement in the suitcase.

Grasp targets are computed from the item's observed pose. After a grasp the palm
poses are re-expressed in the item frame, so a desired item pose maps directly to
palm targets; contact friction is what actually carries the item.
"""
import numpy as np
import scene as S
from runner import rz, minjerk

BETA = np.radians(28)          # forearm downward tilt during squeeze grasps


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def hand_R(forward, palm_normal):
    f = unit(forward)
    n = np.asarray(palm_normal, float)
    n = unit(n - f * (n @ f))
    y, z = -f, -n
    return np.column_stack([np.cross(y, z), y, z])


def facing(r):
    yaw = r.root["yaw"]
    return np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([np.sin(yaw), -np.cos(yaw), 0.0])


def rest_targets(r):
    """Arms relaxed at the sides, in the root frame."""
    out = {}
    for s, sy in (("r", -1), ("l", 1)):
        f = unit([0.35, 0.0, -1.0])
        n = np.array([0.0, -sy, 0.0])
        out[s] = (np.array([0.06, 0.225 * sy, 0.86]), hand_R(f, n))
    return out


def approach(r, pre, fingers=None, gaze=None, root=None, T1=0.9, T2=1.0):
    """Reach a pre-grasp pose by first rising in front of the support, then moving forward over it."""
    o, Rr = r.root_frame()
    raised = {}
    for s, (p, R) in pre.items():
        cur = Rr.T @ (r.hand[s][0] - o)
        tgt = Rr.T @ (p - o)
        raised[s] = (o + Rr @ np.array([min(cur[0], tgt[0] - 0.18), tgt[1], max(tgt[2], cur[2])]), R)
    r.move(T1, hands=raised, fingers=fingers, gaze=gaze)
    r.move(T2, hands=pre, root=root)


def clear_hands(r, height=1.05, T=0.9):
    """Raise both hands well above the suitcase rim and draw them towards the body."""
    o, Rr = r.root_frame()
    out = {}
    for s in "rl":
        hp, hR = r.hand[s]
        loc = Rr.T @ (hp - o)
        loc[0] = min(loc[0], 0.42); loc[2] = max(loc[2], height)
        out[s] = (o + Rr @ loc, hR)
    r.move(T, hands=out)


def to_rest(r, T=1.2, gaze=None):
    r.move(T, hands=rest_targets(r), fingers={"r": "relaxed", "l": "relaxed"}, frame="root",
           root=dict(h=0.95), gaze=gaze)


def item_stack(r, names):
    """Combined centre, long axis, half extents for one or more stacked items."""
    ps, Rs = zip(*[r.body_pose(n) for n in names])
    sizes = [np.array(S.item(n).size) for n in names]
    zb = min(p[2] - s[2] / 2 for p, s in zip(ps, sizes))
    zt = max(p[2] + s[2] / 2 for p, s in zip(ps, sizes))
    c = np.mean(ps, axis=0)
    c[2] = (zb + zt) / 2
    a = Rs[0][:, 0].copy(); a[2] = 0; a = unit(a)
    hl = min(s[0] for s in sizes) / 2
    return c, a, hl, (zt - zb) / 2, zb, Rs[0]


def squeeze_palms(c, a, hl, zb, F, out=0.0, up=0.0, d=0.0, palm_z=None):
    z = palm_z if palm_z is not None else max(c[2], zb + 0.036)
    f = unit(F * np.cos(BETA) - np.array([0, 0, 1.0]) * np.sin(BETA))
    pr = np.array([c[0], c[1], z + up]) + a * (hl - d + out)
    pl = np.array([c[0], c[1], z + up]) - a * (hl - d + out)
    return {"r": (pr, hand_R(f, -a)), "l": (pl, hand_R(f, a))}


def squeeze_pick(r, names, crouch_h=0.80, lift=0.10, d=0.007, palm_up=0.0):
    c, a, hl, hz, zb, _ = item_stack(r, names)
    zb = zb + palm_up
    F, right = facing(r)
    if a @ right < 0:
        a = -a
    label = " and ".join(S.item(n).label for n in names)
    r.stage(f"Reach for {label}")
    pre = squeeze_palms(c, a, hl, zb, F, out=0.06, up=0.09)
    approach(r, pre, fingers={"r": "press", "l": "press"}, gaze=c, root=dict(h=crouch_h))
    r.move(0.8, hands=squeeze_palms(c, a, hl, zb, F, out=0.045))
    r.stage(f"Grip {label}")
    r.move(0.7, hands=squeeze_palms(c, a, hl, zb, F, d=d), fingers={"r": "squeeze", "l": "squeeze"})
    r.hold(0.25)
    z0 = {n: r.body_pose(n)[0][2] for n in names}
    hz0 = {s: r.hand_pose(s)[0][2] for s in "rl"}
    r.stage(f"Lift {label}")
    r.move(0.9, hands=squeeze_palms(c, a, hl, zb, F, d=d, up=lift))
    r.hold(0.3)
    rise = {n: float(r.body_pose(n)[0][2] - z0[n]) for n in names}
    contact = r.hand_item_contact([r.item_ids[n] for n in names])
    hand_rise = {s: float(r.hand_pose(s)[0][2] - hz0[s]) for s in "rl"}
    r.event("lift_check", items=names, rise_m=rise, hand_rise_m=hand_rise, contacts={s: contact[s][0] for s in "rl"},
            normal_force_N={s: round(contact[s][1], 2) for s in "rl"})
    print("   lift", {k: round(v, 4) for k, v in rise.items()}, "hands", {k: round(v, 4) for k, v in hand_rise.items()},
          "N", {s: round(contact[s][1], 1) for s in "rl"}, flush=True)
    ok = all(v > lift * 0.7 for v in rise.values())
    if not ok:
        r.debug_snapshot(f"liftfail_{names[0]}")
        raise RuntimeError(f"lift failed for {names}: rise {rise}")
    return dict(names=names, a=a, hl=hl, d=d)


def grasp_relative(r, names):
    """Palm poses expressed in the frame of the first item."""
    p, R = r.body_pose(names[0])
    rel = {}
    for s in "rl":
        hp, hR = r.hand[s]
        rel[s] = (R.T @ (hp - p), R.T @ hR)
    return rel


def carry_pose(r, names, height=1.0, forward=0.36):
    """Bring the held item in front of the chest, long axis lateral."""
    rel = grasp_relative(r, names)
    p, R = r.body_pose(names[0])
    o, Rr = r.root_frame()
    # desired item yaw: long axis lateral (root y axis), keep the current sign
    ax = R[:, 0]
    want = Rr[:, 1] if ax @ Rr[:, 1] >= 0 else -Rr[:, 1]
    yaw_now = np.arctan2(ax[1], ax[0]); yaw_want = np.arctan2(want[1], want[0])
    dR = rz((yaw_want - yaw_now + np.pi) % (2 * np.pi) - np.pi)
    Rd = dR @ R
    pd = o + Rr @ np.array([forward, 0.0, height])
    return {s: (pd + Rd @ rel[s][0], Rd @ rel[s][1]) for s in "rl"}


def slot_pose(r, name, slot=None):
    slot = slot or S.PACK_SLOTS[name]
    sp, sR = r.suitcase_frame()
    p = sp + sR @ np.array(slot["pos"])
    R = sR @ rz(slot["yaw"])
    return p, R


def place_item(r, names, slot_name, drop=0.004, raise_h=0.12, out=0.035, crouch_h=0.80):
    """Lower the held item into its slot, open the grip and withdraw."""
    rel = grasp_relative(r, names)
    p_slot, R_slot = slot_pose(r, slot_name)
    p_item, R_item = r.body_pose(names[0])
    # Choose the slot yaw (mod pi) closest to the current item yaw.
    ax_now, ax_slot = R_item[:, 0], R_slot[:, 0]
    if ax_now @ ax_slot < 0:
        R_slot = R_slot @ rz(np.pi)
    # Stack offset: slot gives the centre of the first item; for a pair, lift accordingly.
    label = " and ".join(S.item(n).label for n in names)
    def palms(pc):
        return {s: (pc + R_slot @ rel[s][0], R_slot @ rel[s][1]) for s in "rl"}
    r.stage(f"Bring {label} over the suitcase")
    r.move(1.6, hands=palms(p_slot + np.array([0, 0, raise_h])), root=dict(h=crouch_h), gaze=p_slot)
    r.stage(f"Lower {label} into place")
    r.move(1.2, hands=palms(p_slot + np.array([0, 0, drop])))
    r.hold(0.2)
    r.stage(f"Release {label}")
    # Open: each palm backs away along its own palm normal, then lifts.
    opened = {}
    for s in "rl":
        hp, hR = r.hand[s]
        n = -hR[:, 2]
        opened[s] = (hp - n * out, hR)
    r.move(0.6, hands=opened, fingers={"r": "flat", "l": "flat"})
    for n_ in names:
        c = r.hand_item_contact(n_)
        r.event("release_contacts", item=n_, contacts={s: c[s][0] for s in "rl"})
    up = {s: (opened[s][0] + np.array([0, 0, 0.14]), opened[s][1]) for s in "rl"}
    r.move(0.8, hands=up)
    clear_hands(r)
    r.hold(0.2)
    for n_ in names:
        p, R = r.body_pose(n_)
        r.event("placed", item=n_, pos=p.tolist(), speed=r.item_speed(n_))
