"""You are the user: pack the case yourself in the 3D page (play/pack.html).

The page owns walking (free movement with the keyboard, collision with walls and
furniture) and drawing. This module owns everything else, with the same rules as every
generated episode:

  options(x, y, heading)  what can be done from where the person stands: catalogue ids
                          (discrete.py) whose preconditions hold (PackSim.check) once the
                          person is at the station the action is done from;
  act(id)                 runs it: the walk there is recorded as go_to (the page walked
                          it), then the action itself; returns the new state;
  the checklist           the task's goal predicates (PackSim.goals) in words;
  the assistant slot      after every action (and every time you speak) the assistant
                          is asked for guidance with what an assistant may see
                          (usersim.public_view's fields). The slot is
                          assistant_template.respond, which is empty: it never speaks.
  the record              task, actions with times, conversation, trajectory, goals.
"""
import copy, math
import scene as S
import layout as LO
import discrete as D
import dialogue as DL
from packsim import PackSim, TASKS, ITEM_NAMES, FLOOR_SPOTS
import assistant_template

REACH_ITEM = 1.0        # m from the person's centre to an item's centre
REACH_CASE = 0.8        # m from the person's centre to the edge of the case's footprint
FRONT = math.radians(40)    # what you act on must be within this angle of where you face (no turning)
MAX_OPTIONS = 9


def _r(v, n=3):
    return round(float(v), n)


def case_footprint(st):
    """(cx, cy, half_x, half_y) of the closed or open base in world x/y."""
    W, L = S.SUITCASE["outer"]
    if st.case_pose == "upright_floor":
        cx, cy = FLOOR_SPOTS[st.case_at]["case"]
        return cx, cy, S.SUITCASE["base_depth"] / 2 + 0.12, W / 2
    cx, cy = S.SUITCASE["pos"][:2]
    return cx, cy, L / 2, W / 2           # yaw -pi/2: the case length runs along world x


def dist_to_rect(x, y, cx, cy, hx, hy):
    dx = max(abs(x - cx) - hx, 0.0)
    dy = max(abs(y - cy) - hy, 0.0)
    return math.hypot(dx, dy)


# walls that block reaching (full height, or reaching above the hands)
BLOCKING = [((w.center[0] - w.size[0] / 2, w.center[0] + w.size[0] / 2), (w.center[1] - w.size[1] / 2, w.center[1] + w.size[1] / 2))
            for w in S.WALLS if w.center[2] - w.size[2] / 2 < 0.5 and w.center[2] + w.size[2] / 2 > 1.0]


def clear_line(x0, y0, x1, y1):
    """No blocking wall between two floor points (slab test per wall box)."""
    for (ax, bx), (ay, by) in BLOCKING:
        t0, t1 = 0.0, 1.0
        for p, d, lo, hi in ((x0, x1 - x0, ax, bx), (y0, y1 - y0, ay, by)):
            if abs(d) < 1e-9:
                if p < lo or p > hi:
                    break
                continue
            ta, tb = sorted(((lo - p) / d, (hi - p) / d))
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                break
        else:
            return False
    return True


def angle_to(x, y, heading, tx, ty):
    a = math.atan2(ty - y, tx - x) - heading
    return abs((a + math.pi) % (2 * math.pi) - math.pi)


def goal_text(key):
    lab = lambda n: S.item(n).label
    if key.startswith("packed("):
        return f"Pack the {lab(key[7:-1])}"
    if key.startswith("heavy_on_floor("):
        return f"Heavy: the {lab(key[15:-1])} on the case floor"
    if key.startswith("nothing_on("):
        return f"Nothing on top of the {lab(key[11:-1])} (fragile)"
    return {"no_extra_items": "Nothing that isn't on the list", "bottle_upright": "The water bottle stands upright",
            "lid_closed": "Close the lid", "zipped": "Zip the case shut", "case_upright": "Stand the case up",
            "locked": "Lock it: pulls in the lock, dials scrambled"}.get(key, key)


ORDER = ("packed", "no_extra", "bottle", "heavy", "nothing", "lid", "zipped", "locked", "case")


def checklist(sim):
    g = sim.goals()
    keys = sorted(g, key=lambda k: next(i for i, p in enumerate(ORDER) if k.startswith(p)))
    task = TASKS[sim.task]
    pack_order = list(task.pack)
    keys.sort(key=lambda k: (next(i for i, p in enumerate(ORDER) if k.startswith(p)),
                             pack_order.index(k[7:-1]) if k.startswith("packed(") else 0))
    out, before = [], True
    for k in keys:
        rule = not k.startswith("packed(") and k not in ("lid_closed", "zipped", "case_upright", "locked")
        done = bool(g[k])
        if not rule and not k.startswith("packed("):
            # a step counts only after the steps before it: closing an empty case is not "Close the lid"
            done = done and before
        if not rule:
            before = before and done
        out.append(dict(key=k, text=goal_text(k), done=done, rule=rule))
    return out


class Game:
    def __init__(self, task="weekend_trip"):
        self.task = task
        self.sim = PackSim(task)
        self.actions = []          # the record: one entry per catalogue action
        self.conv = []
        self.path = []             # (t, x, y, heading) samples from the page
        self.t = 0.0
        self.pos = (S.STATIONS["suitcase"].stand[0], S.STATIONS["suitcase"].stand[1], math.pi / 2)
        self.say_assistant(self.ask_assistant())

    # ------------------------------------------------------------ what can be done here
    def _contexts(self, x, y, heading):
        """(station, facing, [ids]) for every place within reach of (x, y)."""
        st = self.sim.state
        out = []
        upright = st.case_pose == "upright_floor"
        fp = case_footprint(st)
        cx, cy, hx_, hy_ = fp
        # the case counts as faced if the nearest point of it, or its middle, is in front
        nx_, ny_ = min(max(x, cx - hx_), cx + hx_), min(max(y, cy - hy_), cy + hy_)
        case_ang = min(angle_to(x, y, heading, nx_, ny_) if (nx_, ny_) != (x, y) else 0.0, angle_to(x, y, heading, cx, cy))
        near_case = dist_to_rect(x, y, *fp) <= REACH_CASE and case_ang <= FRONT and clear_line(x, y, cx, cy)
        h = st.holding
        if h:
            if near_case and not upright:
                best = LO.choose_spot(st.place, h)
                spots = [best] + [s for s in D.SPOTS if s != best] if isinstance(best, str) else list(D.SPOTS)
                out.append((case_ang, "suitcase", "", [f"place({h}, {s})" for s in spots]))
            hx, hy = S.item(h).xy
            a = angle_to(x, y, heading, hx, hy)
            if math.hypot(hx - x, hy - y) <= REACH_ITEM and a <= FRONT and clear_line(x, y, hx, hy):
                out.append((a, S.item(h).station, "", [f"put_back({h})"]))
            return [c[1:] for c in sorted(out, key=lambda c: c[0])]
        # items lying where they belong (or were put back), nearest first
        near = []
        for n in ITEM_NAMES:
            loc = st.loc[n]
            if not loc.startswith("station:"):
                continue
            ix, iy = S.item(n).xy
            d, a = math.hypot(ix - x, iy - y), angle_to(x, y, heading, ix, iy)
            if d <= REACH_ITEM and a <= FRONT and clear_line(x, y, ix, iy):
                near.append((a, n, loc.split(":", 1)[1]))
        for a, n, station in near:
            out.append((a, station, n, [f"pick_up({n})"]))
        if near_case:
            if upright:
                ids = ["dock_zip_pulls", "press_lock_button"] + self._dials() + self._handle() + \
                      [f"roll_case({f})" for f in D.FLOOR]
                out.append((case_ang, "case_floor", "", ids))
            else:
                out.append((case_ang, "bed_foot", "", ["close_lid", "open_lid"]))
                out.append((case_ang, "suitcase", "", ["zip_to(100)", "zip_to(0)", "dock_zip_pulls", "press_lock_button"]
                            + self._dials()))
                out.append((case_ang, "bed_foot", "", ["stand_up_case"] + self._handle()))
                # what is packed can come out again (topmost items only pass the check)
                out.append((case_ang, "suitcase", "", [f"pick_up({n})" for n in ITEM_NAMES if st.in_case(n)]))
        return [c[1:] for c in sorted(out, key=lambda c: c[0])]          # stable: case actions keep their order

    def _dials(self):
        st = self.sim.state
        return [f"turn_dial({d}, {(st.dials[k] + 1) % 10})" for k, d in enumerate(D.DIALS)]

    def _handle(self):
        st = self.sim.state
        return [f"set_handle({k})" for k in D.HANDLE_STOPS if k != st.trolley]

    def _check(self, station, facing, aid):
        probe = PackSim(self.task)
        probe.state = copy.deepcopy(self.sim.state)
        probe.state.station, probe.state.facing = station, facing
        return probe.check(D.action(aid))

    def options(self, x, y, heading):
        self.pos = (x, y, heading)
        opts, seen, blocked = [], set(), []
        for station, facing, ids in self._contexts(x, y, heading):
            for aid in ids:
                if aid in seen or aid not in D.BY_ID:
                    continue
                ok, why = self._check(station, facing, aid)
                if ok:
                    seen.add(aid)
                    opts.append(dict(id=aid, text=D.BY_ID[aid]["does"], station=station, facing=facing))
                elif aid in ("close_lid", "zip_to(100)", "stand_up_case") and why:
                    blocked.append(dict(id=aid, text=D.BY_ID[aid]["does"], why=why))
        return dict(options=opts[:MAX_OPTIONS], blocked=blocked[:2], where=self.where(x, y))

    def where(self, x, y):
        import rooms as R
        return R.LABEL[R.room_at((x, y))]

    # ------------------------------------------------------------ doing it
    def act(self, aid, station, facing, t, x=None, y=None):
        """Run catalogue action `aid` from `station` (facing `facing`) at page time t."""
        st = self.sim.state
        self.t = float(t)
        if x is not None:
            self.pos = (x, y, self.pos[2])
        if aid not in D.BY_ID:
            return dict(ok=False, feedback=f"'{aid}' is not in the catalogue")
        if (st.station, st.facing) != (station, facing):
            go = f"go_to({facing or station})"
            self.actions.append(dict(id=go, text=D.BY_ID[go]["does"] if go in D.BY_ID else go, t=_r(t, 2),
                                     effect="move", ok=True, walked=True))
            st.station, st.facing = station, facing
        a = D.action(aid)
        ok, why = self.sim.check(a)
        note = self.sim.feedback(a) if ok else ""
        if ok:
            self.sim.step(a)
        rec = dict(id=aid, text=D.BY_ID[aid]["does"], t=_r(t, 2), effect=D.BY_ID[aid]["effect"], ok=ok,
                   feedback=why if not ok else note)
        self.actions.append(rec)
        said = self.ask_assistant()
        self.say_assistant(said)
        return dict(ok=ok, feedback=rec["feedback"], record=rec, state=self.view(), said=said)

    def set_time(self, t):
        self.t = float(t)

    # ------------------------------------------------------------ conversation and the assistant slot
    def user_says(self, text, t):
        self.t = float(t)
        text = text.strip()
        if not text:
            return dict(said=[])
        self.conv.append(dict(t=_r(t, 2), speaker="user", text=text))
        said = self.ask_assistant()
        self.say_assistant(said)
        return dict(said=said, conv=self.conv)

    def public_view(self):
        """What an assistant may see (the fields of usersim.public_view): the task, the
        conversation, and where the person is and what they hold. Not the goals' state."""
        st = self.sim.state
        task = TASKS[self.task]
        x, y, hd = self.pos
        import rooms as R
        now = dict(t=_r(self.t, 2), step=len(self.actions), position=[_r(x, 2), _r(y, 2)], room=R.room_at((x, y)),
                   holding=S.item(st.holding).label if st.holding else "", conversation=len(self.conv))
        if self.actions:
            now["last_action"] = self.actions[-1]["text"]
        return dict(task=dict(title=task.title, pack=[S.item(n).label for n in task.pack]), preferences=[],
                    conversation=[dict(c) for c in self.conv], now=now, history=[])

    def ask_assistant(self):
        out = assistant_template.respond(self.public_view())
        if not out:
            return []
        return [s.strip() for s in ([out] if isinstance(out, str) else out) if s and s.strip()]

    def say_assistant(self, texts):
        for s in texts:
            self.conv.append(dict(t=_r(self.t, 2), speaker="assistant", text=s))

    # ------------------------------------------------------------ state for the page
    def view(self):
        st = self.sim.state
        items = {}
        for n in ITEM_NAMES:
            d = dict(loc=st.loc[n].split(":")[0], station=st.loc[n].split(":", 1)[1] if ":" in st.loc[n] else "")
            if n in st.place:
                p = st.place[n]
                dx, dy, dz = LO.dims(n, p.get("orient", ""))
                d["place"] = dict(x=_r(p["x"]), y=_r(p["y"]), z=_r(p["z"]), yaw=_r(p["yaw"]),
                                  orient=p.get("orient", ""), dims=[_r(dx), _r(dy), _r(dz)], on=p.get("on"))
            items[n] = d
        cl = checklist(self.sim)
        return dict(items=items, holding=st.holding, station=st.station, facing=st.facing, lid=st.lid,
                    zip_open=_r(st.zip_open), pulls=st.pulls, dials=list(st.dials), trolley=st.trolley,
                    strap=st.strap, case_pose=st.case_pose, case_at=st.case_at, locked=st.locked,
                    checklist=cl, done=all(c["done"] for c in cl), conv=self.conv)

    def record(self, path=None):
        """The episode as played: the same catalogue ids a generated episode uses."""
        task = TASKS[self.task]
        return dict(kind="played", task=self.task, title=task.title, pack=list(task.pack),
                    actions=self.actions, conversation=self.conv, trajectory=path or self.path,
                    goals=self.sim.goals(), done=self.sim.done(), seconds=_r(self.t, 1),
                    assistant="assistant_template.respond (empty)")


# ---------------------------------------------------------------- the worker's entry point
GAME = None


def handle(cmd, a):
    """game_new {task} | game_options {x, y, heading} | game_act {id, station, facing, t}
    | game_say {text, t} | game_record {path}"""
    global GAME
    if cmd == "game_new":
        GAME = Game(a.get("task") or "weekend_trip")
        return dict(state=GAME.view(), task=GAME.task)
    if GAME is None:
        raise RuntimeError("no game: start one first")
    if cmd == "game_options":
        return GAME.options(float(a["x"]), float(a["y"]), float(a.get("heading", 0.0)))
    if cmd == "game_act":
        return GAME.act(a["id"], a.get("station", ""), a.get("facing", ""), a.get("t", 0.0), a.get("x"), a.get("y"))
    if cmd == "game_say":
        return GAME.user_says(a.get("text", ""), a.get("t", 0.0))
    if cmd == "game_record":
        return GAME.record(a.get("path"))
    raise ValueError(f"unknown command {cmd!r}")
