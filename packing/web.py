"""The browser playground's side of the simulator, and the episode files it replays.

The same code runs in three places:
  * in the browser (Pyodide): play/worker.js loads the symbolic modules (no MuJoCo, no
    Blender) and drives a Session one step at a time while you type as the assistant;
  * export_web.py: runs sampled cases with a silent and an oracle assistant and writes
    them as replays (play/episodes/*.json);
  * export_web.py --physics RUN: turns a physics run (actions.json + episode.json +
    goals.json) into the same frame format, next to its rendered videos.

A frame is one executed action: its time span, the walking path, the state after it, the
conversation turns spoken since the last frame and, in a `private` block, what an
assistant is not shown (labels, events, the user's beliefs).
"""
import json, random
import numpy as np
import scene as S
import rooms as R
import layout as LO
import packsim as P
from packsim import TASKS, ITEM_NAMES, place_xy, place_room, access_node
import usersim as U
import dialogue as DL
import episodes as EP


def _r(v, n=3):
    return round(float(v), n)


def _jsonable(o):
    """For the conversation's understood acts (they can hold Actions)."""
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def clean(o):
    return json.loads(json.dumps(o, default=_jsonable))


# ---------------------------------------------------------------- static world
def world():
    """Everything the map draws that never moves: rooms, walls, furniture, stations, the
    walking graph, items and where they start, the case and its regions."""
    walls = [dict(c=[_r(b.center[0]), _r(b.center[1])], s=[_r(b.size[0]), _r(b.size[1])])
             for b in S.WALLS if b.center[2] - b.size[2] / 2 < 0.01]
    furn = [dict(name=b.name, c=[_r(b.center[0]), _r(b.center[1])], s=[_r(b.size[0]), _r(b.size[1])],
                 h=_r(b.center[2] + b.size[2] / 2), kind=b.material, solid=b.collide) for b in S.PROCEDURAL]
    for a in S.ASSETS:
        for b in a.collision:
            furn.append(dict(name=a.name, c=[_r(b.center[0]), _r(b.center[1])], s=[_r(b.size[0]), _r(b.size[1])],
                             h=_r(b.size[2]), kind="asset", solid=True))
    items = {}
    for it in S.ITEMS:
        items[it.name] = dict(label=it.label, family=it.family, size=[_r(v) for v in it.size], heavy=it.heavy,
                              station=it.station, room=it.room, xy=[_r(v) for v in it.xy], yaw=_r(it.start_yaw),
                              fragile=S.FAMILIES[it.family]["fragile"])
    stations = {s.name: dict(label=s.label, room=s.room, stand=[_r(v) for v in s.stand], normal=list(s.normal))
                for s in S.STATIONS.values()}
    sc = S.SUITCASE
    return dict(
        rooms={k: dict(x=list(v["x"]), y=list(v["y"]), label=R.LABEL.get(k, k)) for k, v in S.ROOMS.items()},
        walls=walls, furniture=furn, stations=stations, items=items,
        nodes={k: [_r(v[0]), _r(v[1])] for k, v in S.NAV_NODES.items()}, edges=[list(e) for e in S.NAV_EDGES],
        case=dict(pos=[_r(sc["pos"][0]), _r(sc["pos"][1])], yaw=_r(sc["yaw"]), outer=[_r(v) for v in sc["outer"]],
                  inner=[_r(LO.X0), _r(LO.X1), _r(LO.Y0), _r(LO.Y1)], height=_r(LO.TOP - LO.FLOOR),
                  floor_spots={k: dict(label=v["label"], case=[_r(x) for x in v["case"]],
                                       beside=[_r(x) for x in v["beside"]]) for k, v in P.FLOOR_SPOTS.items()},
                  anchors={k: dict(xy=[_r(x) for x in v["xy"]], label=v["label"]) for k, v in LO.ANCHORS.items()}),
        tasks={k: dict(title=t.title, pack=list(t.pack), lock=bool(t.lock)) for k, t in TASKS.items()},
        personas={k: dict(label=v["label"], style=v["style"], compliance=v["compliance"],
                          gullibility=v["gullibility"], patience=v["patience"]) for k, v in U.PERSONAS.items()},
        event_types={k: dict(channel=v[0], description=v[1], sampled=k in EP.SAMPLED)
                     for k, v in U.EVENT_TYPES.items()},
        guidance=DL.GUIDANCE if hasattr(DL, "GUIDANCE") else {},
    )


# ---------------------------------------------------------------- frames
def stand_point(state):
    st = P.State(**{k: v for k, v in state.items() if k in P.State.__dataclass_fields__})
    return place_xy(st), access_node(st), st.station


def walk_path(prev, nxt):
    """The walking route between two states' stand points along the walking graph (the
    executor follows the same graph, with shortcuts where they stay clear)."""
    (a, na, sa), (b, nb, sb) = stand_point(prev), stand_point(nxt)
    if np.hypot(a[0] - b[0], a[1] - b[1]) < 0.02:
        return [[_r(a[0], 2), _r(a[1], 2)]]
    via_a = list(reversed(S.STATIONS[sa].via)) if sa in S.STATIONS else []
    via_b = list(S.STATIONS[sb].via) if sb in S.STATIONS else []
    pts = [a] + via_a + [S.NAV_NODES[n] for n in R.nav_path(na, nb)] + via_b + [b]
    out = []
    for p in pts:
        if not out or np.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 0.05:
            out.append([_r(p[0], 2), _r(p[1], 2)])
    # drop graph detours that step back past the start or the end
    if len(out) > 2 and np.hypot(out[1][0] - out[0][0], out[1][1] - out[0][1]) > \
            np.hypot(out[2][0] - out[0][0], out[2][1] - out[0][1]) + 0.3:
        out.pop(1)
    return out


def state_view(state):
    keep = ("station", "facing", "holding", "loc", "place", "orient", "lid", "zip_open", "pulls", "dials", "trolley",
            "strap", "case_pose", "case_at", "pack", "zipped", "locked")
    out = {k: state[k] for k in keep if k in state}
    out["place"] = {n: {k: (_r(v) if isinstance(v, float) else v) for k, v in p.items()}
                    for n, p in state.get("place", {}).items()}
    out["zip_open"] = _r(state.get("zip_open", 1.0))
    st = P.State(**{k: v for k, v in state.items() if k in P.State.__dataclass_fields__})
    out["xy"] = [_r(v, 2) for v in place_xy(st)]
    out["room"] = place_room(st)
    return out


def goals_of(state, task):
    sim = P.PackSim(task)
    sim.state = P.State(**{k: v for k, v in state.items() if k in P.State.__dataclass_fields__})
    if "code" not in state:
        sim.state.code = "000"
    return sim.goals()


def make_frame(i, rec, prev_state, turns, step=None, events=None, beliefs=None, plan=None):
    a = rec["action"]
    f = dict(i=i, t0=_r(rec["t_start"], 2), t1=_r(rec["t_end"], 2), text=a.get("text", ""), verb=a.get("verb", ""),
             target=a.get("target", ""), ok=bool(rec["ok"]), feedback=rec.get("feedback", "") or "",
             path=walk_path(prev_state, rec["state"]), state=state_view(rec["state"]), conv=turns)
    if "frame_start" in rec:
        f["frames"] = [rec["frame_start"], rec["frame_end"]]
    f["private"] = dict(label=rec.get("label", ""), event=rec.get("event", ""), source=rec.get("source", ""),
                        controller=rec.get("controller", ""), commit=bool(rec.get("commit")))
    if events is not None:
        f["private"]["events"] = events
    if beliefs is not None:
        f["private"]["beliefs"] = beliefs
    if plan is not None:
        f["private"]["plan"] = plan
    return f


def turn_view(c):
    t = dict(t=_r(c["t"], 2), speaker=c["speaker"], text=c["text"])
    if c.get("event"):
        t["event"] = c["event"]
    if c.get("act"):
        t["act"] = c["act"]
    if "understood" in c:
        t["understood"] = clean([{k: v for k, v in g.items() if k not in ("requests", "first")} for g in c["understood"]])
        t["responses"] = clean(c.get("responses", []))
    return t


def events_view(h):
    return [dict(id=e.id, type=e.type, channel=e.channel, target=e.target, other=e.other, fired=e.fired,
                 done=e.done, committed=e.committed, guided=e.guided, outcome=e.outcome,
                 description=U.EVENT_TYPES[e.type][1], answer=clean(e.answer), given=clean(e.given),
                 t=clean(e.t), steps=clean(e.steps)) for e in h.events]


def beliefs_view(u):
    return dict(omitted=sorted(u.omitted), believed=sorted(u.believed), misplaced=sorted(u.misplaced),
                thinks_zipped=bool(u.thinks_zipped))


# ---------------------------------------------------------------- live session
class Inbox:
    """The playground's assistant: whatever was typed since the last step. With
    `autopilot`, the oracle speaks when nothing was typed."""

    def __init__(self, autopilot=None):
        self.queue = []
        self.autopilot = autopilot

    def respond(self, obs, h):
        out, self.queue = self.queue, []
        if not out and self.autopilot:
            out = self.autopilot.respond(obs, h)
        return out


class Session:
    def __init__(self, case, assistant="human", max_steps=220):
        self.case = case
        auto = {"human": None, "none": None, "oracle": U.Oracle()}[assistant]
        self.inbox = Inbox(auto)
        self.h = U.Harness(case, U.SymbolicBackend(case["task"], U.PERSONAS[case["persona"]]), self.inbox,
                           max_steps=max_steps)
        self.prev = self.h.b.sim.snapshot()
        self.seen = 0
        self.done = False

    def _turns(self):
        out = [turn_view(c) for c in self.h.conv[self.seen:]]
        self.seen = len(self.h.conv)
        return out

    def start(self):
        self.h.begin()
        return dict(case=public_case(self.case), private_case=self.case, state=state_view(self.prev),
                    conv=self._turns(), events=events_view(self.h), t=_r(self.h.b.time, 2),
                    goals=self.h.b.sim.goals(), plan=self.h.user.plan(self.h.st))

    def step(self, text=""):
        """Say `text` (if any) before the next action, then run it. Returns the frame, or
        the summary when the user is done."""
        if self.done:
            return dict(done=True, summary=self.summary())
        if text.strip():
            self.inbox.queue.append(text.strip())
        n = len(self.h.b.records)
        more = self.h.tick()
        if not more:
            self.h.end()
            self.done = True
            return dict(done=True, conv=self._turns(), summary=self.summary(), events=events_view(self.h),
                        t=_r(self.h.b.time, 2))
        rec = self.h.b.records[n]
        f = make_frame(n, rec, self.prev, self._turns(), events=events_view(self.h),
                       beliefs=beliefs_view(self.h.user), plan=self.h.user.plan(self.h.st))
        f["goals"] = self.h.b.sim.goals()
        f["t"] = _r(self.h.b.time, 2)
        self.prev = rec["state"]
        return dict(done=False, frame=f)

    def say_only(self, text):
        """Speak without letting the user act (the reply comes at once)."""
        self.h.hear(text.strip())
        return dict(conv=self._turns(), events=events_view(self.h), t=_r(self.h.b.time, 2))

    def hint(self):
        """What the oracle (which reads the hidden schedule) would say now, without saying it."""
        o = U.Oracle(persistence=99)
        msg = o.respond(self.h.observation(), self.h)
        return msg[0] if msg else ""

    def summary(self):
        h = self.h
        return clean(dict(outcomes={e.id: e.outcome for e in h.events}, goals=h.b.sim.goals(),
                          goals_satisfied=h.b.sim.done(), preferences=h.preference_goals(), actions=len(h.steps),
                          seconds=_r(h.b.time, 1), events=events_view(h), beliefs=beliefs_view(h.user)))


def public_case(case):
    task = TASKS[case["task"]]
    return dict(task=case["task"], title=task.title, pack=list(task.pack), persona=case["persona"],
                preferences=case.get("preferences", []))


def new_case(task=None, seed=0, persona=None, events=None, n_events=None):
    """A validated case (silent dry run + oracle check), as episodes.sample_case."""
    allow = events or None
    k = n_events if n_events is not None else (len(events) if events else None)
    c = EP.sample_case(task=task or None, seed=int(seed), persona=persona or None, allow=allow, n_events=k)
    return {k: v for k, v in c.items() if k not in ("dry_run", "oracle")}


# ---------------------------------------------------------------- whole episodes
def replay(case, assistant):
    """A full symbolic episode as frames (for the replay list)."""
    s = Session(case, assistant)
    head = s.start()
    frames = []
    while True:
        r = s.step()
        if r["done"]:
            tail = r.get("conv", [])
            break
        frames.append(r["frame"])
    return clean(dict(kind="symbolic", assistant=assistant, case=head["case"], private_case=case, start=head,
                      frames=frames, end_conv=tail, summary=s.summary()))


def from_physics(run):
    """A physics run (generate.py output) in the frame format, aligned with its videos."""
    from pathlib import Path
    run = Path(run)
    acts = json.loads((run / "actions.json").read_text())
    ep = json.loads((run / "episode.json").read_text())
    goals = json.loads((run / "goals.json").read_text())
    case = json.loads((run / "case.json").read_text())
    conv = ep["private"]["conversation"]
    steps = ep["private"]["steps"]
    sim = P.PackSim(case["task"])
    prev = sim.snapshot()
    frames, k = [], 0
    # turns go with the action they were spoken before or during (speech holds the body)
    pre = [turn_view(c) for c in conv if acts and c["t"] < acts[0]["t_start"] - 1e-6]
    k = len(pre)
    for i, rec in enumerate(acts):
        turns = []
        while k < len(conv) and conv[k]["t"] <= rec["t_end"] + 1e-6:
            turns.append(turn_view(conv[k])); k += 1
        if i < len(steps):
            rec = dict(rec, label=steps[i]["label"], event=steps[i]["event"], source=steps[i]["source"],
                       commit=steps[i]["commit"])
        f = make_frame(i, rec, prev, turns)
        f["goals"] = goals_of(rec["state"], case["task"])
        frames.append(f)
        prev = rec["state"]
    tail = [turn_view(c) for c in conv[k:]]
    sched = goals["schedule"]
    summary = dict(outcomes={e["id"]: e.get("outcome", "") for e in sched}, goals=goals["goals"],
                   goals_satisfied=goals["satisfied"], preferences=goals.get("preferences", {}), actions=len(acts),
                   seconds=_r(acts[-1]["t_end"], 1) if acts else 0, events=sched,
                   beliefs=ep["private"].get("beliefs", {}))
    head = dict(case=public_case(case), state=state_view(P.PackSim(case["task"]).snapshot()), conv=pre, t=0.0)
    return dict(kind="physics", case=head["case"], private_case={k2: v for k2, v in case.items()
                                                                 if k2 not in ("dry_run", "oracle")},
                start=head, frames=frames, end_conv=tail, summary=clean(summary))


# ---------------------------------------------------------------- the worker's entry point
SESSION = None


def handle(msg):
    """One request from the page (play/worker.js passes JSON through): returns JSON-able.
      new   {task, persona, seed, events, assistant}  -> start payload
      step  {text}                                    -> frame or summary
      say   {text}                                    -> the user's reply (no action runs)
      hint  {}                                        -> what the oracle would say now
      world {}                                        -> world()
      game_* {...}                                    -> game.handle (play/pack.html)
    """
    global SESSION
    cmd, a = msg.get("cmd"), msg.get("args") or {}
    if cmd and cmd.startswith("game_"):         # the 3D page, where you are the user (game.py)
        import game
        return clean(game.handle(cmd, a))
    if cmd == "world":
        return world()
    if cmd == "new":
        events = [e for e in (a.get("events") or []) if e in EP.SAMPLED] or None
        case = new_case(task=a.get("task") or None, seed=int(a.get("seed") or 0), persona=a.get("persona") or None,
                        events=events, n_events=None if events else a.get("n_events"))
        SESSION = Session(case, a.get("assistant") or "human")
        return clean(SESSION.start())
    if SESSION is None:
        raise RuntimeError("no session: start one first")
    if cmd == "step":
        return clean(SESSION.step(a.get("text") or ""))
    if cmd == "say":
        if SESSION.done:
            return dict(conv=[], t=_r(SESSION.h.b.time, 2), done=True)
        return clean(SESSION.say_only(a.get("text") or ""))
    if cmd == "hint":
        return dict(text=SESSION.hint())
    raise ValueError(f"unknown command {cmd!r}")
