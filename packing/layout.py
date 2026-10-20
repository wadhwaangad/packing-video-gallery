"""Geometry of the inside of the open case: where an item can go, what it rests on.

Placements are continuous: an item in the case has a centre (x, y) in the suitcase-base
frame (x across the width, +x = the free long side the person packs from, y along the
length, +y = handle end), a yaw (direction of the item's local x axis, which is the axis
the two palms squeeze along) and an orientation. A candidate pose is checked against a
5 mm height map of everything already packed:

  inside      the footprint lies inside the liner walls
  support     the item rests on the highest surface under it; at least 60 % of its
              footprint and its centre must be within 15 mm of that height (soft piles
              conform a little), and nothing under it may be fragile or unstackable
  height      its top stays below the closed lid's liner
  lid sweep   no top edge is inside the arc the closing lid's walls and panel sweep
  hands       the grip needs room: two 35 mm strips beside the squeezed faces (two-hand
              items) or a 40 mm strip on one side (the one-hand bottle grip) must be free
              of anything taller than the surface the item rests on

The same rules are used by the planner (to choose a spot) and by the action validator
(to accept or reject a requested spot with a reason).
"""
from dataclasses import dataclass
import numpy as np
import scene as S

W, L = S.SUITCASE["outer"]
T = S.SUITCASE["shell"]
D, DL = S.SUITCASE["base_depth"], S.SUITCASE["lid_depth"]
FLOOR = S.SUITCASE["floor"]
TOP = D + DL - S.SUITCASE["lid_floor"]          # underside of the closed lid's liner (base frame)
HH = S.SUITCASE["hinge_h"]
X0, X1 = -W / 2 + T, W / 2 - T
Y0, Y1 = -L / 2 + T, L / 2 - T
GRID = 0.005
NX, NY = int(round((X1 - X0) / GRID)), int(round((Y1 - Y0) / GRID))
XS = X0 + GRID * (np.arange(NX) + 0.5)
YS = Y0 + GRID * (np.arange(NY) + 0.5)
GX, GY = np.meshgrid(XS, YS, indexing="ij")

SUPPORT_TOL = 0.015
SUPPORT_FRAC = 0.6
HEADROOM = 0.008
LID_MARGIN = 0.005
PALM_STRIP = 0.035
BOTTLE_STRIP = 0.040
WALL_MARGIN = 0.004

# Named spots inside the case (what "the wheel end" etc. refer to), with the yaw the
# planner uses there. These are the places the reference trajectory packs into.
ANCHORS = {
    "wheel_end": dict(xy=(-0.068, -0.155), label="the wheel-end garment column"),
    "handle_end": dict(xy=(-0.068, 0.155), label="the handle-end garment column"),
    "front_strip": dict(xy=(0.170, 0.120), label="the front strip"),
    "bottle_corner": dict(xy=(0.140, -0.205), label="the front wheel-end corner"),
    "centre": dict(xy=(0.0, 0.0), label="the middle of the case"),
}


def dims(item, orient=""):
    """(along local x, along local y, height) of an item in a placement orientation."""
    it = S.item(item)
    sx, sy, sz = it.size
    if item == "bottle" and orient == "lying":
        return sz, sx, sx           # axis along the placement x direction
    return sx, sy, sz


def default_yaw(item, orient=""):
    """Long side along the case length, short (squeeze) side across it, as packed by
    the reference planner: palms squeeze along the case length."""
    if item == "bottle":
        return np.pi / 2 if orient == "lying" else 0.0
    return np.pi / 2


@dataclass
class Fit:
    ok: bool
    code: str = ""
    reason: str = ""
    z: float = 0.0              # bottom of the item (base frame)
    on: tuple = ()              # items it rests on (empty: the case floor)
    detail: dict = None


def _mask(x, y, yaw, hx, hy, grow=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    u = (GX - x) * c + (GY - y) * s
    v = -(GX - x) * s + (GY - y) * c
    return (np.abs(u) <= hx + grow) & (np.abs(v) <= hy + grow)


def _strip(x, y, yaw, hx, hy, side, width):
    """Cells of a strip `width` wide outside face `side` (+x, -x, +y, -y of the item frame)."""
    c, s = np.cos(yaw), np.sin(yaw)
    u = (GX - x) * c + (GY - y) * s
    v = -(GX - x) * s + (GY - y) * c
    if side in ("+x", "-x"):
        sg = 1 if side == "+x" else -1
        return (sg * u > hx) & (sg * u <= hx + width) & (np.abs(v) <= hy)
    sg = 1 if side == "+y" else -1
    return (sg * v > hy) & (sg * v <= hy + width) & (np.abs(u) <= hx)


def _corners(x, y, yaw, hx, hy):
    c, s = np.cos(yaw), np.sin(yaw)
    return [(x + c * a - s * b, y + s * a + c * b) for a in (-hx, hx) for b in (-hy, hy)]


def _strip_inside(x, y, yaw, hx, hy, side, width):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = []
    if side in ("+x", "-x"):
        sg = 1 if side == "+x" else -1
        for b in (-hy, hy):
            a = sg * (hx + width)
            pts.append((x + c * a - s * b, y + s * a + c * b))
    else:
        sg = 1 if side == "+y" else -1
        for a in (-hx, hx):
            b = sg * (hy + width)
            pts.append((x + c * a - s * b, y + s * a + c * b))
    return all(X0 <= px <= X1 and Y0 <= py <= Y1 for px, py in pts)


def heightmap(placements, exclude=()):
    """Top height and top owner per cell (owner '' = case floor)."""
    h = np.full((NX, NY), FLOOR)
    owner = np.full((NX, NY), "", dtype=object)
    for name, p in sorted(placements.items(), key=lambda kv: kv[1]["z"]):
        if name in exclude:
            continue
        sx, sy, sz = dims(name, p.get("orient", ""))
        m = _mask(p["x"], p["y"], p["yaw"], sx / 2, sy / 2)
        top = p["z"] + sz
        upd = m & (top >= h)
        h[upd] = top; owner[upd] = name
    return h, owner


# Lid geometry in the lid frame's x-z plane (the lid hinges about the base's -x edge).
_ZR = D - HH
_LID_RECTS = [  # (x0, x1, z0, z1)
    (0.0, W, _ZR + DL - S.SUITCASE["lid_floor"], _ZR + DL),          # panel with its liner
    (W - T, W, _ZR, _ZR + DL),                                       # free-edge wall
    (0.0, T, 0.0, _ZR + DL),                                         # hinge-side wall (from the hinge axis up)
]
_THETA = np.linspace(0.0, np.pi, 181)


def lid_hit(points_xz, margin=LID_MARGIN):
    """First (x, z) point of the base frame that the lid passes through while it swings
    from open to shut, or None."""
    for (px, pz) in points_xz:
        rx, rz = px + W / 2, pz - HH
        c, s = np.cos(_THETA), np.sin(_THETA)
        xl = rx * c + rz * s
        zl = -rx * s + rz * c
        for (a0, a1, b0, b1) in _LID_RECTS:
            if np.any((xl > a0 - margin) & (xl < a1 + margin) & (zl > b0 - margin) & (zl < b1 + margin)):
                return (px, pz)
    return None


def evaluate(placements, item, x, y, yaw, orient="", exclude=()):
    """Can `item` rest at (x, y, yaw) in the case, given what is packed already?"""
    it = S.item(item)
    fam = S.FAMILIES[it.family]
    sx, sy, sz = dims(item, orient)
    hx, hy = sx / 2, sy / 2
    det = dict(x=round(x, 4), y=round(y, 4), yaw_deg=round(float(np.degrees(yaw)), 1))
    for (px, py) in _corners(x, y, yaw, hx, hy):
        if not (X0 + WALL_MARGIN <= px <= X1 - WALL_MARGIN and Y0 + WALL_MARGIN <= py <= Y1 - WALL_MARGIN):
            return Fit(False, "out_of_range", f"the {it.label} would not fit inside the case walls there", detail=det)
    h, owner = heightmap(placements, exclude=set(exclude) | {item})
    m = _mask(x, y, yaw, hx, hy)
    if not m.any():
        return Fit(False, "out_of_range", "that spot is outside the case", detail=det)
    z = float(h[m].max())
    sup = m & (h >= z - SUPPORT_TOL)
    frac = float(sup.sum() / m.sum())
    ci, cj = int(np.clip((x - X0) / GRID, 0, NX - 1)), int(np.clip((y - Y0) / GRID, 0, NY - 1))
    on = tuple(sorted({o for o in owner[sup] if o}))
    det.update(rests_at_m=round(z - FLOOR, 4), supported_fraction=round(frac, 2), on=list(on))
    for o in on:
        of = S.FAMILIES[S.item(o).family]
        if not of["stackable"]:
            return Fit(False, "unsupported_surface", f"nothing can go on top of the {S.item(o).label}", z, on, det)
    if frac < SUPPORT_FRAC or h[ci, cj] < z - SUPPORT_TOL:
        return Fit(False, "unstable", f"the {it.label} would overhang what is under it "
                                      f"({frac:.0%} supported) and tip over", z, on, det)
    if z + sz > TOP - HEADROOM:
        return Fit(False, "too_tall", f"the {it.label} would stand {100 * (z + sz - TOP + HEADROOM):.1f} cm above "
                                      f"the closed lid", z, on, det)
    # Top edge cross-section (x, z) as the lid sweeps over it.
    xs = [px for px, _ in _corners(x, y, yaw, hx, hy)]
    pts = [(px, z + sz) for px in np.linspace(min(xs), max(xs), 9)]
    hit = lid_hit(pts)
    if hit is not None:
        return Fit(False, "lid_sweep", f"the closing lid would hit the {it.label} "
                                       f"({100 * (hit[0] + W / 2):.0f} cm from the hinge side, "
                                       f"{100 * (hit[1] - FLOOR):.0f} cm up)", z, on, det)
    # Room for the hands.
    if fam["hands"] == 2:
        for side in ("+x", "-x"):
            st = _strip(x, y, yaw, hx, hy, side, PALM_STRIP)
            if not _strip_inside(x, y, yaw, hx, hy, side, PALM_STRIP):
                return Fit(False, "no_hand_room", f"no room for a hand between the {it.label} and the case wall",
                           z, on, det)
            if st.any() and h[st].max() > z + 0.02:
                blocker = S.item(owner[st][np.argmax(h[st])]).label
                return Fit(False, "no_hand_room", f"no room for a hand beside the {it.label}: "
                                                  f"the {blocker} is in the way", z, on, det)
    else:
        free = []
        for side in ("+x", "-x", "+y", "-y"):
            st = _strip(x, y, yaw, hx, hy, side, BOTTLE_STRIP)
            if _strip_inside(x, y, yaw, hx, hy, side, BOTTLE_STRIP) and (not st.any() or h[st].max() <= z + 0.02):
                free.append(side)
        if not free:
            return Fit(False, "no_hand_room", f"no room for a hand around the {it.label}", z, on, det)
        det["hand_sides_free"] = free
    return Fit(True, "", "", z, on, det)


def resolve_spot(placements, item, at=None, rel="in", ref=None, side=None, yaw=None, orient=""):
    """Turn a placement request into a pose.

    at:    None (choose), anchor name, or (x, y) in metres in the base frame
    rel:   'in' (the case), 'on' (on top of `ref`), 'beside' (`side` of `ref`)
    Returns (x, y, yaw) or raises ValueError(code, reason)."""
    yaw = default_yaw(item, orient) if yaw is None else yaw
    if rel == "on":
        p = placements[ref]
        return p["x"], p["y"], yaw
    if rel == "beside":
        p = placements[ref]
        osx, osy, _ = dims(ref, p.get("orient", ""))
        sx, sy, _ = dims(item, orient)
        # extent of each footprint along the case axes
        ext = lambda a, b, yw: (abs(np.cos(yw)) * a + abs(np.sin(yw)) * b, abs(np.sin(yw)) * a + abs(np.cos(yw)) * b)
        oex, oey = ext(osx, osy, p["yaw"]); ex, ey = ext(sx, sy, yaw)
        gap = PALM_STRIP + 0.005 if S.FAMILIES[S.item(item).family]["hands"] == 2 else 0.01
        off = {"wheel_end": (0, -(oey + ey) / 2 - gap), "handle_end": (0, (oey + ey) / 2 + gap),
               "front_side": ((oex + ex) / 2 + gap, 0), "hinge_side": (-(oex + ex) / 2 - gap, 0)}[side]
        return p["x"] + off[0], p["y"] + off[1], yaw
    if at is None:
        raise ValueError("missing_argument", "where in the case?")
    if isinstance(at, str):
        a = ANCHORS[at]["xy"]
        return a[0], a[1], yaw
    return float(at[0]), float(at[1]), yaw


def auto_place(placements, item, orient="", prefer=()):
    """Planner's choice of spot: the preferred anchors first (in order), then a scan of
    the case floor and stacks; heavy items prefer the floor, the bottle a corner."""
    it = S.item(item)
    yaw0 = default_yaw(item, orient)
    cands = []
    for a in prefer:
        xy = ANCHORS[a]["xy"]
        cands.append((xy[0], xy[1], yaw0, a))
    for yw in (yaw0, yaw0 + np.pi / 2):
        for xx in np.arange(X0 + 0.04, X1 - 0.04, 0.02):
            for yy in np.arange(Y0 + 0.04, Y1 - 0.04, 0.02):
                cands.append((float(xx), float(yy), float(yw), ""))
    best = None
    for (x, y, yw, a) in cands:
        f = evaluate(placements, item, x, y, yw, orient)
        if not f.ok:
            continue
        if a:
            return x, y, yw, f
        # lower is better; heavy items want the floor; then well supported, then towards
        # the hinge side (keeps the free side clear for the hands)
        score = f.z * (10 if it.heavy else 1) + 0.2 * (1 - f.detail["supported_fraction"]) + 0.01 * (x - X0)
        if best is None or score < best[0]:
            best = (score, x, y, yw, f)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


def nothing_on(placements, item):
    """Items resting directly on `item`."""
    return [n for n, p in placements.items() if item in p.get("on", ())]


def nudge(placements, item, x, y, yaw, orient="", keep_on=None, max_d=0.08, step=0.01):
    """The nearest pose to (x, y) within max_d that fits (and still rests on `keep_on`):
    what a person does when 'on the trousers' or 'at the wheel end' does not fit exactly."""
    offs = [(dx, dy) for dx in np.arange(-max_d, max_d + 1e-9, step) for dy in np.arange(-max_d, max_d + 1e-9, step)
            if dx * dx + dy * dy <= max_d * max_d + 1e-12]
    offs.sort(key=lambda o: o[0] ** 2 + o[1] ** 2)
    rank = ["unsupported_surface", "too_tall", "unstable", "lid_sweep", "no_hand_room", "out_of_range"]
    worst = None
    for dx, dy in offs:
        f = evaluate(placements, item, x + dx, y + dy, yaw, orient)
        if f.ok and (keep_on is None or keep_on in f.on):
            return x + dx, y + dy, f
        if f.ok:
            f = Fit(False, "unstable", f"it would not rest on the {S.item(keep_on).label} there", f.z, f.on, f.detail)
        # report the most telling reason seen near the requested spot
        if worst is None or rank.index(f.code) < rank.index(worst.code) if f.code in rank else False:
            worst = f
    return None, None, worst


COLUMNS = ("wheel_end", "handle_end")


def rest_z(placements, item, anchor):
    x, y = ANCHORS[anchor]["xy"]
    return evaluate(placements, item, x, y, default_yaw(item)).z


def choose_spot(placements, item):
    """The planner's spot: a named spot if it fits there (columns lowest first, the bag
    and fragile items in the front strip, heavy items on an empty column, the bottle in
    its corner), else the best measured spot in the case."""
    it = S.item(item)
    if it.family == "bottle":
        return "bottle_corner"
    cols = sorted(COLUMNS, key=lambda a: rest_z(placements, item, a))
    if item == "toiletry_bag" or it.family == "fragile_box":
        prefs = ["front_strip"] + cols
    elif it.heavy:
        prefs = [a for a in cols if rest_z(placements, item, a) <= FLOOR + 1e-6] + cols + ["front_strip"]
    else:
        prefs = cols + ["front_strip"]
    for a in prefs:
        x, y = ANCHORS[a]["xy"]
        if evaluate(placements, item, x, y, default_yaw(item)).ok:
            return a
    r = auto_place(placements, item)
    return (round(r[0], 4), round(r[1], 4)) if r else "front_strip"
