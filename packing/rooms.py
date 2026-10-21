"""Rooms, the walking graph and line of sight.

The person knows where everything is, but a single go_to only takes them to something
in the room they are in, or in view from where they stand (an unobstructed line from
eye height to it). Anything else is reached room by room: enter(room) walks through to
the first point of the walking graph inside the next room on the way, and the target
is looked for again from there. Navigation is pure geometry on scene.py, so it works
the same in symbolic dry runs and in physics.
"""
import heapq
import numpy as np
import scene as S

EYE_H = 1.55                      # eye height of the person, m
LABEL = {"bedroom": "the bedroom", "bathroom": "the bathroom", "kitchen": "the kitchen", "hall": "the hall",
         "living": "the living room"}
WORDS = {"bedroom": ["bedroom", "bed room"], "bathroom": ["bathroom", "bath room", "washroom"],
         "kitchen": ["kitchen"], "hall": ["hall", "hallway", "corridor"],
         "living": ["living room", "lounge", "sitting room"]}
# a walking-graph node well inside each room (where "go to the kitchen" ends)
CENTRE = {"bedroom": "bed_east", "bathroom": "bath_in", "kitchen": "kitchen_in", "hall": "kitchen_hall",
          "living": "living_mid"}


def room_at(xy):
    x, y = float(xy[0]), float(xy[1])
    for name, r in S.ROOMS.items():
        if r["x"][0] <= x <= r["x"][1] and r["y"][0] <= y <= r["y"][1]:
            return name
    return ""


NODE_ROOM = {n: room_at(p) for n, p in S.NAV_NODES.items()}


def nav_path(a, b):
    """Shortest node path on the walking graph."""
    adj = {n: [] for n in S.NAV_NODES}
    for u, v in S.NAV_EDGES:
        d = float(np.hypot(*np.subtract(S.NAV_NODES[u], S.NAV_NODES[v])))
        adj[u].append((v, d)); adj[v].append((u, d))
    dist, prev, pq = {a: 0.0}, {}, [(0.0, a)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == b:
            break
        for v, w in adj[u]:
            if d + w < dist.get(v, 1e9):
                dist[v] = d + w; prev[v] = u; heapq.heappush(pq, (d + w, v))
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return path[::-1]


def nearest_node(xy):
    return min(S.NAV_NODES, key=lambda n: np.hypot(*np.subtract(S.NAV_NODES[n], xy)))


def adjacent(r1, r2):
    """True if the walking graph has a step from room r1 straight into room r2."""
    return any({NODE_ROOM[u], NODE_ROOM[v]} == {r1, r2} for u, v in S.NAV_EDGES)


# ---------------------------------------------------------------- line of sight
_OCC = None


def occluders():
    """Opaque static boxes (walls with their door gaps, furniture) as (lo, hi, name)."""
    global _OCC
    if _OCC is None:
        _OCC = [(np.array(b.center) - np.array(b.size) / 2, np.array(b.center) + np.array(b.size) / 2, b.name)
                for b in S.all_static_boxes() if b.collide]
    return _OCC


def _hits(p, q, lo, hi, t0, t1):
    d = q - p
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if p[k] < lo[k] or p[k] > hi[k]:
                return False
            continue
        a, b = (lo[k] - p[k]) / d[k], (hi[k] - p[k]) / d[k]
        if a > b:
            a, b = b, a
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return False
    return True


def line_of_sight(eye, target, end_clear=0.30):
    """(visible, name of the first blocking box). The last `end_clear` metres are not
    tested: the target rests on (or in) the furniture there."""
    p, q = np.asarray(eye, float), np.asarray(target, float)
    L = float(np.linalg.norm(q - p))
    if L < 1e-6:
        return True, ""
    t1 = max(0.0, 1.0 - end_clear / L)
    for lo, hi, name in occluders():
        if _hits(p, q, lo, hi, 0.0, t1):
            return False, name
    return True, ""
