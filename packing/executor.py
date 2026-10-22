"""Physical grounding of PackSim actions in the MuJoCo run.

Each semantic action is checked symbolically first (a refused action returns feedback
and leaves the world unchanged), then executed with the contact-only skills in tasks.py
and skills_case.py, then verified against the physical state (lift height, item inside
its region, lid angle, case upright). Only a verified action is committed to PackSim.

Targets are computed from the live simulation: stand poses from where the item is now,
the drop height in the case from the piles already in that region.
"""
import copy
import mujoco
import numpy as np
import scene as S
import layout as LO
import tasks as T
import skills_case as C
import skills_hw as H
import rooms as R
from rooms import nav_path
from runner import rz
from packsim import Action, ITEM_NAMES, FLOOR_SPOTS, access_node


class ActionFailed(RuntimeError):
    pass


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
        if target in LO.ANCHORS:
            target = "suitcase"
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
        """Waypoints from where the person is (`frm`: a station, node:<n> or case_floor) to
        `stand` at `to` (a station, or node:<n> for a point of the walking graph)."""
        if frm == to:
            return [stand]
        via_a = list(reversed(S.STATIONS[frm].via)) if frm in S.STATIONS else []
        b_access, via_b = (to[5:], []) if to.startswith("node:") else (S.STATIONS[to].access, list(S.STATIONS[to].via))
        pts = via_a + [S.NAV_NODES[n] for n in nav_path(access_node(self.sim.state), b_access)]
        pts += via_b + [stand]
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

    def case_target(self, item, x, y, yaw, orient, on):
        """World pose for an item placed at (x, y, yaw) in the case, resting on the
        measured tops of the items under it (live poses, not nominal heights)."""
        sp, sR = self.r.suitcase_frame()
        top = S.SUITCASE["floor"]
        for n in on:
            p, R = self.r.body_pose(n)
            loc = sR.T @ (p - sp)
            o = self.sim.state.place.get(n, {}).get("orient", "")
            top = max(top, loc[2] + LO.dims(n, o)[2] / 2)
        h = LO.dims(item, orient)[2]
        pos = sp + sR @ np.array([x, y, top + h / 2])
        if item == "bottle":
            return pos, None
        return pos, sR @ rz(yaw)

    # ------------------------------------------------------------ verbs
    def go_to(self, a):
        st = self.sim.state
        if st.case_pose == "upright_floor" and (a.target == "suitcase" or a.target in LO.ANCHORS):
            # the standing case: step beside it, where its handle is taken
            fs = FLOOR_SPOTS[st.case_at]
            self.rest()
            self.r.stage("Walk to the suitcase")
            self.r.walk([fs["beside"]], final_yaw=fs["beside_yaw"], speed=0.6)
            return
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

    def enter(self, a):
        """Walk through to the entry point of the next room (on the way to a target)."""
        st = self.sim.state
        node = __import__("packsim").entry_node(st, a.target)
        holding = st.holding
        if not holding:
            self.rest()
        self.r.stage(f"Walk into {R.LABEL[a.target]}")
        speed = 0.85 * self.persona["walk"] * (0.82 if holding == "bottle" else 1.0)
        pts = self.route(st.station, "node:" + node, tuple(S.NAV_NODES[node]))
        if pts:
            self.face(pts[0])
        self.r.walk(pts, speed=speed, yaw_rate=1.4 if holding == "bottle" else 3.0)
        here = (self.r.root["x"], self.r.root["y"])
        if R.room_at(here) != a.target:
            raise ActionFailed(f"ended at {here[0]:.2f}, {here[1]:.2f}, not in {R.LABEL[a.target]}")
        self.r.event("entered_room", room=a.target, node=node, xy=list(map(float, here)))
        if holding:
            self.check_in_hand(holding)

    def in_sight(self, p):
        """(visible, what blocks): a ray in the physics scene from the eyes to p. Anything
        welded to the world (walls, door frames, furniture) blocks it; hands, items and the
        case do not (they are passed through)."""
        m, d = self.r.m, self.r.d
        eye = np.array([self.r.root["x"], self.r.root["y"], R.EYE_H])
        p = np.asarray(p, float)
        L = float(np.linalg.norm(p - eye))
        start, gid = eye.copy(), np.array([-1], dtype=np.int32)
        u = (p - eye) / L
        travelled = 0.0
        for _ in range(40):
            dist = mujoco.mj_ray(m, d, start, u, None, 1, -1, gid)
            if gid[0] < 0 or dist < 0 or travelled + dist >= L - 0.30:
                return True, ""
            b = int(m.geom_bodyid[gid[0]])
            if int(m.body_weldid[b]) == 0 and int(m.body_mocapid[b]) < 0:
                return False, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid[0]) or                     mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"geom {gid[0]}"
            travelled += dist + 1e-3
            start = eye + u * travelled
        return True, ""

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

    def place(self, a):
        """Put the held item at a resolved spot in the case (or back on a station)."""
        st = self.sim.state
        item = st.holding
        dest = a.get("dest", "suitcase")
        if a.get("rel", "in") == "in" and dest in S.STATIONS and dest != "suitcase":
            self.put_back(a)
            return None
        r, msg, code = self.sim.place_pose(a)
        if r is None:
            raise ActionFailed(f"place: {msg}")
        x, y, yaw, orient, fit = r
        x += self.jitter["place"][0]; y += self.jitter["place"][1]
        lying = orient == "lying"
        p, R = self.case_target(item, x, y, yaw, orient, fit.on)
        sp, sR = self.r.suitcase_frame()
        self.r.event("place_target", item=item, local_m=[round(x, 4), round(y, 4)], yaw_deg=round(float(np.degrees(yaw)), 1),
                     orient=orient, on=list(fit.on), fit=fit.detail)
        if item == "bottle":
            if lying:
                C.place_lying(self.r, item, p, side="r", axis=sR @ np.array([np.cos(yaw), np.sin(yaw), 0.0]))
            else:
                C.place_upright(self.r, item, p, side="r")
        else:
            T.place_item(self.r, [item], (p, R), crouch_h=0.80, out=0.035 + self.jitter.get("release", 0.0))
        self.at_rest = False
        # verify: resting near the requested centre, inside the walls
        sp, sR = self.r.suitcase_frame()
        pb, Rb = self.r.body_pose(item)
        loc = sR.T @ (pb - sp)
        W, L = S.SUITCASE["outer"]
        ok = abs(loc[0] - x) < 0.07 and abs(loc[1] - y) < 0.07 and abs(loc[0]) < W / 2 and abs(loc[1]) < L / 2
        axis_l = sR.T @ Rb[:, 0]
        yaw_m = float(np.arctan2(axis_l[1], axis_l[0]))
        if item == "bottle":
            ax = sR.T @ Rb[:, 2]
            tilt = float(np.degrees(np.arccos(np.clip(abs(ax[2]), -1, 1))))
            ok = ok and (tilt < 10 if not lying else tilt > 70)
            yaw_m = float(np.arctan2(ax[1], ax[0])) if lying else yaw
        self.r.event("placed_check", item=item, target_m=[round(x, 4), round(y, 4)], local_m=loc.round(4).tolist(), ok=bool(ok))
        if not ok:
            raise ActionFailed(f"{item} did not land at ({x:.3f}, {y:.3f}): {loc.round(3)}")
        # the symbolic state keeps the measured centre and yaw, the planned support
        return dict(x=float(loc[0]), y=float(loc[1]), yaw=yaw_m % np.pi if item != "bottle" or lying else 0.0,
                    orient=orient, z=fit.z, on=fit.on)

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

    def side_step(self, stand):
        r = self.r
        stand = (stand[0] + self.jitter["stand"][0], stand[1] + self.jitter["stand"][1])
        if np.hypot(r.root["x"] - stand[0], r.root["y"] - stand[1]) > 0.03:
            r.walk([stand], final_yaw=np.pi, speed=0.5)

    def lid(self, a):
        """open / close the lid from the foot of the bed."""
        self.rest()
        r = self.r
        p_lid, R_lid = r.body_pose("suitcase_lid")
        W, L = S.SUITCASE["outer"]
        if a.verb == "close":
            grip = p_lid + R_lid @ np.array([W - 0.075, L / 2, C.ZR])
            self.side_step((2.48, float(grip[1]) - 0.15))
            T0 = C.LID_SWING_T
            C.LID_SWING_T = T0 * self.jitter["time"]
            try:
                C.close_lid(r, side="r")
            finally:
                C.LID_SWING_T = T0
            if C.lid_angle(r) > np.radians(3.0):
                raise ActionFailed(f"lid not shut: {np.degrees(C.lid_angle(r)):.1f} deg")
        else:
            # in line with the lid's free edge, just outside it
            edge = p_lid + R_lid @ np.array([W, L / 2, C.ZR])
            self.side_step((2.46, float(edge[1]) - 0.04))
            C.open_lid(r, side="r")
            if C.lid_angle(r) < np.radians(150):
                raise ActionFailed(f"lid not open: {np.degrees(C.lid_angle(r)):.1f} deg")
        self.at_rest = False

    def zip(self, a):
        """open / close the main zip to a fraction (1 = fully open): both sliders pushed
        along the track by a fingertip; the measured fraction must match."""
        self.rest()
        H.run_zip(self.r, frm=self.sim.state.zip_open, to=float(a.get("to")))
        self.at_rest = False

    def lock_ops(self, a):
        """insert / turn / set (lock) / press (lock button) at the front of the case."""
        self.rest()
        v, t = a.verb, a.target
        if v == "insert":
            H.insert_pulls(self.r)
        elif v == "turn":
            H.turn_dial(self.r, int(t[-1]), to=a.get("to"), by=a.get("by"))
            H.clear_hands(self.r)
        elif v == "set":
            H.set_code(self.r, a.get("value"))
        elif v == "press":
            H.press_lock_button(self.r, hold=a.get("hold", 0.3), times=a.get("times", 1))
        self.at_rest = False

    def trolley_ops(self, a):
        self.rest()
        st = self.sim.state
        if st.case_pose == "on_bed":
            sp, _ = self.r.suitcase_frame()
            self.side_step((2.46, float(sp[1])))
        else:
            # beside the standing case, where its handle is taken
            fs = FLOOR_SPOTS[st.case_at]
            if np.hypot(self.r.root["x"] - fs["beside"][0], self.r.root["y"] - fs["beside"][1]) > 0.05:
                self.r.stage("Step beside the case")
                self.r.walk([fs["beside"]], final_yaw=fs["beside_yaw"], speed=0.5)
        if a.verb == "set":
            H.set_trolley(self.r, int(a.get("value")))
        else:
            H.press_button(self.r, "trolley_btn", self.r.hw["trolley_btn"], H.HW["trolley"]["button"]["travel"],
                           hold=a.get("hold", 0.3), times=a.get("times", 1), face=H.case_axes(self.r)[0],
                           label="the trolley-handle button")
            H.clear_hands(self.r)
        self.at_rest = False

    def roll(self, a):
        fs = FLOOR_SPOTS[a.get("to")]
        here = FLOOR_SPOTS[self.sim.state.case_at]
        self.rest()
        H.roll_case(self.r, here["beside"], here["beside_yaw"], fs["route"], fs["case"])
        self.at_rest = False

    def strap_ops(self, a):
        self.rest()
        (H.fasten_strap if a.verb == "fasten" else H.unfasten_strap)(self.r)
        self.at_rest = False

    def stand_up(self, a):
        self.rest()
        sp, _ = self.r.suitcase_frame()
        self.side_step((2.46, float(sp[1])))
        C.lift_from_bed_foot(self.r)
        R = self.r.suitcase_frame()[1]
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 1], -1, 1))))
        if tilt > 10:
            raise ActionFailed(f"case not upright: {tilt:.1f} deg")
        self.at_rest = False

    def not_simulated(self, a):
        raise ActionFailed(f"'{a.text()}' has no physical skill yet")

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
        v, t = a.verb, a.target
        pose = None
        if v == "go_to":
            self.go_to(a)
        elif v == "enter":
            self.enter(a)
        elif v == "look_at":
            self.look_at(a)
        elif v == "pick_up":
            self.pick_up(a, from_case=self.sim.state.in_case(t))
        elif v == "place":
            pose = self.place(a)
        elif v == "put_back":
            self.put_back(a)
        elif v in ("open", "close") and t == "lid":
            self.lid(a)
        elif v in ("open", "close") and t == "main_zip":
            self.zip(a)
        elif v == "stand_up":
            self.stand_up(a)
        elif v in ("insert", "turn") or (v == "set" and t == "lock") or (v == "press" and t == "lock_button"):
            self.lock_ops(a)
        elif (v == "set" and t == "trolley_handle") or (v == "press" and t == "trolley_button"):
            self.trolley_ops(a)
        elif v in ("fasten", "unfasten"):
            self.strap_ops(a)
        elif v == "roll":
            self.roll(a)
        elif v == "fold":
            self.rest()
            stn = S.STATIONS[S.CLOTH["station"]]
            self.r.walk([(stn.stand[0], S.CLOTH["centre"][1])], final_yaw=self.yaw_of(S.CLOTH["station"]), speed=0.4)
            H.fold(self.r, a.get("style", "half"))
            self.at_rest = False
        elif v in SKILLS:
            SKILLS[v](self, a)
        elif v == "wait":
            self.rest()
            r.stage("Pause")
            r.hold(float(a.get("seconds")))
        elif v == "say":
            self.rest()
            r.stage(f'"{a.get("text")}"')
            r.event("utterance", text=a.get("text"))
            r.hold(2.5)
        else:
            self.not_simulated(a)
        if v in ("pick_up", "place", "put_back"):
            self.pause()
        note = self.sim.feedback(a)
        self.sim.step(a, pose)
        rec.update(t_end=round(r.d.time, 3), frame_end=len(r.frames), state=self.sim.snapshot())
        if note:
            rec["feedback"] = note
        self.records.append(rec)
        return rec

    def look_at(self, a):
        t = a.target
        if t in ITEM_NAMES:
            p = self.r.body_pose(t)[0]
        elif t in S.STATIONS:
            stn = S.STATIONS[t]
            p = np.array([*stn.stand, 0.0]) + np.array([*stn.normal, 0.0]) * 0.6 + np.array([0, 0, stn.top])
        else:
            p = self.r.suitcase_frame()[0]
        self.r.stage(f"Look at {a.text()[8:]}" if not a.parent else f"Look for {a.text()[8:]}")
        self.r.move(0.8, gaze=np.asarray(p, float))
        self.r.hold(0.6)
        vis, by = self.in_sight(p)
        self.r.event("look_check", target=t, visible=vis, blocked_by=by,
                     distance_m=float(np.linalg.norm(np.asarray(p)[:2] - [self.r.root["x"], self.r.root["y"]])))

    def rewind(self, k):
        """Undo actions k.. (physics, recordings and symbolic state are restored)."""
        snap, state = self.snaps[k]
        self.r.restore(snap)
        self.sim.state = state
        del self.snaps[k:]
        del self.records[k:]
        self.at_rest = False


# Skills for suitcase controls, registered by the modules that implement them.
SKILLS = {}
