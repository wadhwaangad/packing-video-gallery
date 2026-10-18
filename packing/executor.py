"""Physical grounding of PackSim actions in the MuJoCo run.

Each semantic action is checked symbolically first (a refused action returns feedback
and leaves the world unchanged), then executed with the contact-only skills in tasks.py
and skills_case.py, then verified against the physical state (lift height, item inside
its region, lid angle, case upright). Only a verified action is committed to PackSim.

Targets are computed from the live simulation: stand poses from where the item is now,
the drop height in the case from the piles already in that region.
"""
import copy, heapq
import numpy as np
import scene as S
import tasks as T
import skills_case as C
from runner import rz
from packsim import Action, CASE_STATION, ITEM_NAMES


class ActionFailed(RuntimeError):
    pass


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


class Executor:
    def __init__(self, runner, sim, persona=None):
        self.r = runner
        self.sim = sim
        self.persona = persona or dict(walk=0.9, pause=0.3)
        self.at_rest = True
        self.records = []
        self.snaps = []
        # Small variation applied when an action is retried after a physical failure.
        self.jitter = dict(stand=(0.0, 0.0), time=1.0, place=(0.0, 0.0), release=0.0)

    # ------------------------------------------------------------ helpers
    def label(self, n):
        return S.item(n).label

    def yaw_of(self, station):
        n = S.STATIONS[station].normal
        return float(np.arctan2(n[1], n[0]))

    def stand_for(self, target):
        """(station, stand xy, yaw) for a go_to target."""
        st = self.sim.state
        if target in ITEM_NAMES:
            if st.in_case(target):
                target = "suitcase"
            else:
                station = S.item(target).station
                p, _ = self.r.body_pose(target)
                stn = S.STATIONS[station]
                xy = np.array(p[:2]) - np.array(stn.normal) * stn.standoff
                return station, tuple(xy), self.yaw_of(station)
        stn = S.STATIONS[target]
        if st.holding and S.item(st.holding).station == target:
            # carrying something back: stand in front of its home spot
            it = S.item(st.holding)
            xy = np.array(it.xy) - np.array(stn.normal) * stn.standoff
            return target, tuple(xy), self.yaw_of(target)
        return target, tuple(stn.stand), self.yaw_of(target)

    def route(self, frm, to, stand):
        if frm == to:
            return [stand]
        a, b = S.STATIONS[frm], S.STATIONS[to]
        pts = list(reversed(a.via))
        pts += [S.NAV_NODES[n] for n in nav_path(a.access, b.access)]
        pts += list(b.via) + [stand]
        here = np.array([self.r.root["x"], self.r.root["y"]])
        seq = [here]
        for p in pts:
            p = np.array(p, float)
            if np.linalg.norm(p - seq[-1]) > 0.05:
                seq.append(p)
        # Shortcut waypoints whose bypass stays clear: no detours or U-turns while carrying.
        out, i = [], 0
        while i < len(seq) - 1:
            j = len(seq) - 1
            while j > i + 1 and not self.segment_clear(seq[i], seq[j]):
                j -= 1
            out.append(seq[j]); i = j
        return [tuple(p) for p in out]

    _walk_boxes = None

    def segment_clear(self, a, b, radius=0.30, skip=0.35):
        """True if the straight walk a->b keeps `radius` from furniture and walls (the first
        and last `skip` metres are exempt: stand points are close to what they face)."""
        if Executor._walk_boxes is None:
            Executor._walk_boxes = [x for x in S.all_static_boxes() if x.collide and x.center[2] - x.size[2] / 2 < 0.5
                                    and x.name != "mattress"]
        L = float(np.linalg.norm(b - a))
        if L < 2 * skip:
            return True
        for sv in np.arange(skip, L - skip + 1e-9, 0.05):
            p = a + (b - a) * (sv / L)
            for x in Executor._walk_boxes:
                lo = np.array(x.center[:2]) - np.array(x.size[:2]) / 2
                hi = np.array(x.center[:2]) + np.array(x.size[:2]) / 2
                if np.linalg.norm(np.maximum(0, np.maximum(lo - p, p - hi))) < radius:
                    return False
        return True

    def rest(self):
        if not self.at_rest:
            T.to_rest(self.r, T=1.0)
            self.at_rest = True

    def pause(self):
        if self.persona["pause"] > 0:
            self.r.hold(self.persona["pause"])

    def case_target(self, region, item, lying=False):
        """World pose for an item placed on top of whatever is already in `region`."""
        sp, sR = self.r.suitcase_frame()
        rg = S.REGIONS[region]
        top = S.SUITCASE["floor"]
        for n in self.sim.state.stacks[region]:
            p, R = self.r.body_pose(n)
            loc = sR.T @ (p - sp)
            h = S.item(n).size[2] if n != "bottle" or self.sim.state.orient["bottle"] == "upright" else S.item(n).size[0]
            top = max(top, loc[2] + h / 2)
        it = S.item(item)
        if item == "bottle":
            if lying:
                return sp + sR @ np.array([rg.xy[0], -0.20, top + it.size[0] / 2]), None
            return sp + sR @ np.array([rg.xy[0], rg.xy[1], top + it.size[2] / 2]), None
        return sp + sR @ np.array([rg.xy[0], rg.xy[1], top + it.size[2] / 2]), sR @ rz(np.pi / 2)

    # ------------------------------------------------------------ verbs
    def go_to(self, a):
        st = self.sim.state
        station, stand, yaw = self.stand_for(a.target)
        stand = (stand[0] + self.jitter["stand"][0], stand[1] + self.jitter["stand"][1])
        holding = st.holding
        if not holding:
            self.rest()
        dest = S.item(a.target).label if a.target in ITEM_NAMES and not st.in_case(a.target) else S.STATIONS[station].label
        self.r.stage(f"Walk to {dest}" if a.target not in ITEM_NAMES else f"Walk to the {dest}")
        speed = 0.85 * self.persona["walk"] * (0.82 if holding == "bottle" else 1.0)
        pts = self.route(st.station, station, stand)
        if pts:
            self.face(pts[0])
        self.r.walk(pts, final_yaw=yaw, speed=speed, yaw_rate=1.4 if holding == "bottle" else 3.0)
        if holding:
            self.check_in_hand(holding)

    def check_in_hand(self, item):
        mid = (self.r.hand_pose("r")[0] + self.r.hand_pose("l")[0]) / 2
        d = float(np.linalg.norm(self.r.body_pose(item)[0] - mid))
        if d > 0.30:
            raise ActionFailed(f"dropped the {S.item(item).label} on the way ({d:.2f} m from the hands)")

    def face(self, xy):
        """Turn in place towards the first waypoint, the way round whose hand sweep keeps
        clear of walls and tall furniture (a carried pile must not hit anything)."""
        r = self.r
        here = np.array([r.root["x"], r.root["y"]])
        d = np.array(xy) - here
        if np.linalg.norm(d) < 0.05:
            return
        yaw1 = float(np.arctan2(d[1], d[0]))
        dyw = (yaw1 - r.root["yaw"] + np.pi) % (2 * np.pi) - np.pi
        if abs(dyw) < np.radians(50):
            return
        # Obstacles that reach carry height; the first part of the sweep is skipped (the
        # hands are still leaving the surface they just worked over).
        boxes = [b for b in S.all_static_boxes() if b.collide and b.center[2] + b.size[2] / 2 > 0.95]
        def clearance(direction):
            sweep = dyw if np.sign(dyw) == direction else dyw + 2 * np.pi * direction
            best = 1e9
            for u in np.linspace(0.2, 1, 24):
                y = r.root["yaw"] + sweep * u
                c, s_ = np.cos(y), np.sin(y)
                for fx, fy in ((0.50, 0.0), (0.42, 0.20), (0.42, -0.20)):
                    p = here + np.array([c * fx - s_ * fy, s_ * fx + c * fy])
                    for b in boxes:
                        lo = np.array(b.center[:2]) - np.array(b.size[:2]) / 2
                        hi = np.array(b.center[:2]) + np.array(b.size[:2]) / 2
                        best = min(best, float(np.linalg.norm(np.maximum(0, np.maximum(lo - p, p - hi)))))
            return best
        short = 1 if dyw > 0 else -1
        c_short, c_long = clearance(short), clearance(-short)
        direction = short if c_short >= min(c_long, 0.12) else -short
        r.turn(yaw1, direction=direction)

    def pick_up(self, a, from_case=False):
        st = self.sim.state
        it = S.item(a.target)
        stn = S.STATIONS["suitcase" if from_case else it.station]
        z0 = self.r.body_pose(a.target)[0][2]
        self.at_rest = False
        if it.family == "bottle":
            if from_case and st.orient["bottle"] != "upright":
                raise ActionFailed("a lying bottle cannot be gripped from above")
            C.wrap_pick(self.r, a.target, side="r", lift=0.12)
            self.r.stage(f"Carry the {it.label}")
            self.r.move(1.2, hands=C.carry_one(self.r, a.target, "r", height=max(1.10, z0 + 0.10), forward=0.30, lateral=0.06),
                        root=dict(h=0.95))
        else:
            soft_base = stn.name in ("bed", "bed_far")
            T.squeeze_pick(self.r, [a.target], crouch_h=stn.crouch, d=0.010 if soft_base else 0.007,
                           palm_up=0.004 if soft_base else 0.0)
            self.r.stage(f"Carry the {it.label}")
            self.r.move(1.2, hands=T.carry_pose(self.r, [a.target], height=max(1.0, z0 + 0.10)), root=dict(h=0.95))
        z = self.r.body_pose(a.target)[0][2]
        if z < z0 + 0.07:
            raise ActionFailed(f"{a.target} not lifted (rose {z - z0:.3f} m)")

    def put_in(self, a):
        st = self.sim.state
        item = st.holding
        lying = a.arg == "lying"
        p, R = self.case_target(a.target, item, lying)
        sp, sR = self.r.suitcase_frame()
        p = p + sR @ np.array([*self.jitter["place"], 0.0])
        if item == "bottle":
            (C.place_lying if lying else C.place_upright)(self.r, item, p, side="r")
        else:
            T.place_item(self.r, [item], (p, R), crouch_h=0.80, out=0.035 + self.jitter.get("release", 0.0))
        self.at_rest = False
        # verify: resting in the region, inside the walls
        sp, sR = self.r.suitcase_frame()
        loc = sR.T @ (self.r.body_pose(item)[0] - sp)
        rg = S.REGIONS[a.target]
        W, L = S.SUITCASE["outer"]
        ok = abs(loc[0] - rg.xy[0]) < 0.07 and abs(loc[1] - (-0.20 if lying else rg.xy[1])) < 0.07 \
            and abs(loc[0]) < W / 2 and abs(loc[1]) < L / 2
        if item == "bottle":
            axis = sR.T @ self.r.body_pose(item)[1][:, 2]
            tilt = float(np.degrees(np.arccos(np.clip(abs(axis[2]), -1, 1))))
            ok = ok and (tilt < 10 if not lying else tilt > 70)
        self.r.event("placed_check", item=item, region=a.target, local_m=loc.round(4).tolist(), ok=bool(ok))
        if not ok:
            raise ActionFailed(f"{item} did not land in {a.target}: {loc.round(3)}")

    def put_back(self, a):
        st = self.sim.state
        item = st.holding
        it = S.item(item)
        stn = S.STATIONS[it.station]
        p = np.array(it.start)
        if item == "bottle":
            C.place_upright(self.r, item, p, side="r")
        else:
            T.place_item(self.r, [item], (p, rz(it.start_yaw)), crouch_h=stn.crouch, where=stn.label)
        self.at_rest = False
        d = np.linalg.norm(self.r.body_pose(item)[0][:2] - np.array(it.xy))
        self.r.event("put_back_check", item=item, offset_m=float(d))
        if d > 0.08:
            raise ActionFailed(f"{item} not back in place ({d:.3f} m)")

    def case_action(self, a):
        self.rest()
        r = self.r
        if a.verb in ("close_lid", "open_lid", "stand_up"):
            # side-step along the foot of the bed to where this action is done from
            if a.verb == "close_lid":
                p_lid, R_lid = r.body_pose("suitcase_lid")
                W, L = S.SUITCASE["outer"]
                grip = p_lid + R_lid @ np.array([W - 0.075, L / 2, C.ZR])
                stand = (2.48, float(grip[1]) - 0.15)
            elif a.verb == "open_lid":
                # in line with the lid's free edge, just outside it
                p_lid, R_lid = r.body_pose("suitcase_lid")
                W, L = S.SUITCASE["outer"]
                edge = p_lid + R_lid @ np.array([W, L / 2, C.ZR])
                stand = (2.46, float(edge[1]) - 0.04)
            else:
                sp, _ = r.suitcase_frame()
                stand = (2.46, float(sp[1]))
            stand = (stand[0] + self.jitter["stand"][0], stand[1] + self.jitter["stand"][1])
            if np.hypot(r.root["x"] - stand[0], r.root["y"] - stand[1]) > 0.03:
                r.walk([stand], final_yaw=np.pi, speed=0.5)
        if a.verb == "close_lid":
            T0 = C.LID_SWING_T
            C.LID_SWING_T = T0 * self.jitter["time"]
            try:
                C.close_lid(r, side="r")
            finally:
                C.LID_SWING_T = T0
            if C.lid_angle(r) > np.radians(3.0):
                raise ActionFailed(f"lid not shut: {np.degrees(C.lid_angle(r)):.1f} deg")
        elif a.verb == "open_lid":
            C.open_lid(r, side="r")
            if C.lid_angle(r) < np.radians(150):
                raise ActionFailed(f"lid not open: {np.degrees(C.lid_angle(r)):.1f} deg")
        elif a.verb == "zip":
            C.press_and_zip(r)
        elif a.verb == "unzip":
            C.unzip(r)
        elif a.verb == "stand_up":
            C.lift_from_bed_foot(r)
            R = r.suitcase_frame()[1]
            tilt = float(np.degrees(np.arccos(np.clip(R[2, 1], -1, 1))))
            if tilt > 10:
                raise ActionFailed(f"case not upright: {tilt:.1f} deg")
        self.at_rest = False

    # ------------------------------------------------------------ dispatch
    def execute(self, a, label="nominal", event=""):
        """Check, run, verify and commit one action. Returns the action record."""
        r = self.r
        self.snaps.append((r.snapshot(), copy.deepcopy(self.sim.state)))
        ok, msg = self.sim.check(a)
        rec = dict(index=len(self.records), action=a.to_dict(), label=label, event=event, t_start=round(r.d.time, 3),
                   frame_start=len(r.frames), ok=ok, feedback=msg)
        if not ok:
            r.event("refused", action=a.to_dict(), feedback=msg)
            rec.update(t_end=rec["t_start"], frame_end=len(r.frames), state=self.sim.snapshot())
            self.records.append(rec)
            return rec
        if a.verb == "go_to":
            self.go_to(a)
        elif a.verb == "pick_up":
            self.pick_up(a)
        elif a.verb == "take_out":
            self.pick_up(a, from_case=True)
        elif a.verb == "put_in":
            self.put_in(a)
        elif a.verb == "put_back":
            self.put_back(a)
        elif a.verb in CASE_STATION:
            self.case_action(a)
        elif a.verb == "wait":
            self.rest()
            r.stage("Pause")
            r.hold(float(a.arg))
        elif a.verb == "say":
            self.rest()
            r.stage(f'"{a.arg}"')
            r.event("utterance", text=a.arg)
            r.hold(2.5)
        if a.verb in ("pick_up", "take_out", "put_in", "put_back"):
            self.pause()
        self.sim.step(a)
        rec.update(t_end=round(r.d.time, 3), frame_end=len(r.frames), state=self.sim.snapshot())
        self.records.append(rec)
        return rec

    def rewind(self, k):
        """Undo actions k.. (physics, recordings and symbolic state are restored)."""
        snap, state = self.snaps[k]
        self.r.restore(snap)
        self.sim.state = state
        del self.snaps[k:]
        del self.records[k:]
        self.at_rest = False
