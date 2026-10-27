"""User simulation: a simulated person who packs, talks, and reacts to guidance.

An episode is a case (task, seed, persona, preferences, private event schedule) run in a
closed loop between three parties:

  user model   a replanning policy over its own beliefs (which items belong in the case,
               whether the zip is shut, where the heavy bag goes), plus a conversational
               policy: what it says, and whether it takes a piece of guidance (persona
               traits: compliance, gullibility, patience, style). Its plan is correct by
               construction; every error comes from the schedule.
  harness      owns the schedule, which the assistant never sees. When an event's
               progress trigger holds, the harness edits the user's beliefs or takes over
               the user's action queue with the deviation script. It marks the commit
               point (the first step that changes the world wrongly), watches the
               resolution condition, and hands control back when the event is resolved,
               cancelled by accepted guidance, or past its window. Meanwhile the user
               model keeps talking.
  assistant    anything that turns public observations into messages. Model is the
               slot for one under test (a respond(view) function; empty = silent).
               The rest are test fixtures: Silent, Scripted, Interactive (type as the
               assistant), and Oracle (reads the hidden schedule; it checks that every
               event can be prevented or answered).

Backends execute catalogue actions: SymbolicBackend (PackSim plus a clock from action
durations measured on the physics runs) and PhysicsBackend (executor.Executor in MuJoCo).
The same harness drives both, so a case validated symbolically runs the same way in
physics.

Outcomes per event: prevented (accepted guidance before the commit), corrected (after
it), self_recovered, unresolved; answered (correct or not) / unanswered; reminded /
missed; notified / not_notified; acknowledged / unacknowledged.
"""
from dataclasses import dataclass, field, asdict
import copy, json, random
import scene as S
import layout as LO
from packsim import PackSim, Action, TASKS, ITEM_NAMES, navigation, place_xy, place_room
import discrete as D
import dialogue as DL
import instruct as IN
import rooms as R

# ---------------------------------------------------------------- personas
# walk / pause drive the body (executor); the rest drive the conversation.
#   compliance   P(taking a correct correction the first time); each repeat raises it
#   gullibility  P(taking guidance that contradicts the task)
#   patience     2 s waits given to an unanswered question or an awaited notice
#   narrate      says so when it notices its own slip;  clarify: asks when it cannot parse
PERSONAS = {
    "brisk": dict(walk=1.05, pause=0.0, label="hurried: walks quickly, no pauses, terse, often needs telling twice",
                  style="terse", compliance=0.6, gullibility=0.15, patience=1, narrate=False, clarify=False),
    "steady": dict(walk=0.90, pause=0.4, label="cooperative: even pace, chatty, takes advice readily",
                   style="chatty", compliance=1.0, gullibility=0.35, patience=3, narrate=True, clarify=True),
    "deliberate": dict(walk=0.75, pause=1.0, label="careful: slow, checks advice against the list, asks when unsure",
                       style="chatty", compliance=0.85, gullibility=0.0, patience=4, narrate=True, clarify=True),
}

# ---------------------------------------------------------------- events
EVENT_TYPES = {
    # type: (channel, description)
    "wrong_object": ("error", "picks up a pile that is not on the list instead of the intended one"),
    "omission": ("error", "skips an item on the list"),
    "repetition": ("error", "walks back for an item that is already packed"),
    "nav_deviation": ("error", "heads to the wrong place first"),
    "heavy_on_top": ("error", "puts the heavy toiletry bag on top of a clothes pile"),
    "premature_close": ("error", "closes the lid before everything is packed"),
    "incomplete_zip": ("error", "stops zipping three quarters of the way and moves on"),
    "stall": ("error", "stops making progress for several seconds"),
    "bottle_lying": ("error", "lays the water bottle down instead of standing it up (not sampled: the lying "
                              "placement is not reliable with the scanned case)"),
    "ask_should_pack": ("question", "asks whether an item belongs in the case"),
    "ask_packed": ("question", "asks whether an item is already packed"),
    "ask_left": ("question", "asks what is left to pack"),
    "ask_since": ("question", "asks how long ago an item was packed (temporal)"),
    "ask_elapsed": ("question", "asks how long the packing has taken so far (temporal)"),
    "request_remind": ("request", "asks to be reminded about an item before the lid is closed, then forgets it"),
    "request_notify_done": ("request", "asks to be told when the list is done, and waits for it before closing"),
    "plan_change": ("plan_change", "announces a legitimate change to the packing list"),
}
ALIASES = {"question": "ask_should_pack"}          # earlier case files

# What a helpful assistant does, and by when (for evaluation; never shown to it).
EXPECTED = {
    "wrong_object": dict(act="not_on_list", about="other", by="commit"),
    "omission": dict(act="not_packed", about="target", by="commit"),
    "repetition": dict(act="packed", about="target", by="commit"),
    "nav_deviation": dict(act="where", about="target", by="commit"),
    "heavy_on_top": dict(act="on_floor", about="target", by="commit"),
    "premature_close": dict(act="not_packed", about="remaining", by="commit"),
    "incomplete_zip": dict(act="zip_open", by="window"),
    "stall": dict(act="next", by="window"),
    "bottle_lying": dict(act="do", about="target", by="commit"),
    "ask_should_pack": dict(act="answer", by="window"), "ask_packed": dict(act="answer", by="window"),
    "ask_left": dict(act="answer", by="window"), "ask_since": dict(act="answer", by="window"),
    "ask_elapsed": dict(act="answer", by="window"),
    "request_remind": dict(act="not_packed", about="target", by="commit", due="closing"),
    "request_notify_done": dict(act="all_packed", by="window", due="all packed"),
    "plan_change": dict(act="ack", by="window"),
}
WINDOW = dict(error=14, question=0, request=0, plan_change=4)   # steps; 0 = set by the script / condition
WAIT = D.WAIT_S


@dataclass
class Event:
    id: str
    type: str
    target: str = ""            # the intended item the event is about
    other: str = ""             # distractor, wrong station, add:/remove: item
    after: int = -1             # count trigger: fire once this many items are packed
    recover: bool = True        # the user corrects it unaided (errors only)
    text: str = ""
    trigger: dict = field(default_factory=dict)
    window: int = 0
    # run-time
    fired: bool = False
    done: bool = False
    committed: bool = False
    cancelled: bool = False
    guided: bool = False
    outcome: str = ""
    answer: object = None       # ground truth (questions)
    given: object = None        # the assistant's answer as understood
    steps: dict = field(default_factory=dict)   # fired / commit / resolved / released step indices
    t: dict = field(default_factory=dict)       # the same, in seconds

    def __post_init__(self):
        self.type = ALIASES.get(self.type, self.type)
        if not self.trigger:
            self.trigger = default_trigger(self)
        if not self.window:
            self.window = WINDOW[self.channel]

    @property
    def channel(self):
        return EVENT_TYPES[self.type][0]

    @staticmethod
    def from_dict(d):
        return Event(**{k: v for k, v in d.items() if k in Event.__dataclass_fields__ and k not in (
            "fired", "done", "committed", "cancelled", "guided", "outcome", "given", "steps", "t")})

    def spec(self):
        """The schedule entry as written to case.json (no run-time fields)."""
        return dict(id=self.id, type=self.type, channel=self.channel, target=self.target, other=self.other,
                    after=self.after, recover=self.recover, text=self.text, trigger=self.trigger, window=self.window)

    def to_dict(self):
        d = self.spec()
        d.update(description=EVENT_TYPES[self.type][1], expected=EXPECTED[self.type], fired=self.fired, done=self.done,
                 committed=self.committed, guided=self.guided, outcome=self.outcome, answer=self.answer,
                 given=self.given, steps=self.steps, t=self.t)
        return d


def default_trigger(ev):
    """A task-progress condition over the state and the user's next intended action."""
    t = ev.type
    if t in ("wrong_object", "nav_deviation"):
        return dict(next=["go_to", ev.target], hands="free")
    if t == "omission":
        return dict(next=["go_to", ev.target])
    if t == "repetition":
        return dict(in_case=ev.target, next=["go_to", "*"], hands="free", at="suitcase")
    if t == "heavy_on_top":
        return dict(hands=ev.target, next=["go_to", "suitcase"], column_above_floor=ev.target)
    if t == "bottle_lying":
        return dict(hands="bottle", next=["go_to", "suitcase"])
    if t == "incomplete_zip":
        return dict(next=["close", "main_zip"])
    trig = dict(packed_at_least=max(ev.after, 0), next=["go_to", "*"], hands="free", lid="open")
    if t == "ask_since":
        trig["in_case"] = ev.target
    if t == "request_remind":
        trig["not_in_case"] = ev.target
    return trig


def holds(trig, st, nxt, n_packed):
    if nxt is None:
        return False
    nv = trig.get("next")
    if nv and (nxt.verb != nv[0] or (nv[1] != "*" and nxt.target != nv[1])):
        return False
    if "hands" in trig and st.holding != ("" if trig["hands"] == "free" else trig["hands"]):
        return False
    if n_packed < trig.get("packed_at_least", 0):
        return False
    if "in_case" in trig and not st.in_case(trig["in_case"]):
        return False
    if "not_in_case" in trig and st.in_case(trig["not_in_case"]):
        return False
    if "lid" in trig and st.lid != trig["lid"]:
        return False
    if "at" in trig and st.station != trig["at"]:
        return False
    if "column_above_floor" in trig and not any(_rest_z(st, trig["column_above_floor"], a) > LO.FLOOR + 1e-6
                                                for a in LO.COLUMNS):
        return False
    return True


# ---------------------------------------------------------------- the user's plan
def pack_order(task, rng):
    """Heavy items first (they go on the case floor), then the bottle (stood in its corner
    while the hand still has room beside it), then garments, then the notebook, the fragile
    camera last. Order within a phase is random."""
    items = list(TASKS[task].pack)

    def phase(n):
        it = S.item(n)
        if it.heavy:
            return 0
        if it.family == "bottle":
            return 1
        if it.family == "folded_pile":
            return 2
        if S.FAMILIES[it.family]["fragile"]:
            return 4
        return 3
    rng.shuffle(items)
    return sorted(items, key=phase)


def _rest_z(st, item, anchor):
    return LO.rest_z(st.place, item, anchor)


def choose_spot(st, item):
    """A named spot (the action space has no measured ones); front_strip if none fits,
    which the dry run then refuses, so such a case is not sampled."""
    return LO.choose_named(st.place, item) or "front_strip"


def spot_fits(st, item, spot):
    orient = "upright" if item == "bottle" else ""
    x, y = LO.ANCHORS[spot]["xy"]
    return LO.nudge(st.place, item, x, y, LO.default_yaw(item, orient), orient)[0] is not None


def topmost_over(st, n):
    """The item at the top of whatever rests on n (n itself if nothing does)."""
    top = n
    while True:
        above = st.on_top_of(top)
        if not above:
            return top
        top = above[0]


def closing(st, a):
    """The action starts closing the case: the lid, or the walk to the foot of the bed for it."""
    return a is not None and ((a.verb == "close" and a.target == "lid") or
                              (a.verb == "go_to" and a.target == "bed_foot" and st.lid == "open"))


class UserModel:
    """The person: a plan that is right by construction, beliefs the schedule can corrupt,
    and a policy for answering guidance."""

    def __init__(self, task, rng, persona, prefs=()):
        self.task = task
        self.rng = rng
        self.p = persona
        self.order = pack_order(task, rng)
        self.prefs = {i: s for i, s in prefs}
        self.omitted = set()        # items the user has (for now) forgotten
        self.believed = set()       # items the user wrongly thinks belong in the case
        self.misplaced = set()      # items the user has been told sit wrongly in the case
        self.thinks_zipped = False  # believes the zip is shut when it is not
        self.await_notice = False   # waits to be told the list is done before closing
        self.notice_waits = 0
        self.next_pref = ""         # an item suggested as the next one to fetch
        self.repeats = {}           # how often each piece of guidance has been given
        self._spots = {}

    def say(self, act, **kw):
        return DL.say(act, self.rng, self.p["style"], **kw)

    # ------------------------------------------------------------ plan
    def intended(self, st):
        return [n for n in self.order + sorted(self.believed) if (n in st.pack or n in self.believed)
                and n not in self.omitted]

    def todo(self, st):
        todo = [n for n in self.intended(st) if not st.in_case(n)]
        if self.next_pref in todo:
            todo.remove(self.next_pref)
            todo.insert(0, self.next_pref)
        return todo

    def plan(self, st):
        """The plan as the person sees it now (from their beliefs, so a forgotten item is
        missing from it and a wrong one is on it): [{"step", "item", "status"}], status
        done | next | todo. Private, like the beliefs."""
        task = TASKS[self.task]
        steps = [(f"pack the {S.item(n).label}", n, st.in_case(n)) for n in self.intended(st)]
        shut = st.lid == "closed" and not self.todo(st)     # the lid starts closed: count it once packed
        steps += [("close the lid", "", shut),
                  ("zip it shut", "", shut and (st.zipped or self.thinks_zipped))]
        if task.lock:
            steps.append(("lock it", "", shut and st.locked))
        steps.append(("stand the case up", "", shut and st.case_pose == "upright_floor"))
        out, nxt = [], (self.todo(st) or [None])[0]
        for text, item, done in steps:
            status = "done" if done else "todo"
            if not done and not any(s["status"] == "next" for s in out) and (item == nxt or nxt is None):
                status = "next"
            out.append(dict(step=text, item=item, status=status))
        return out

    def place_args(self, st, h):
        # the spot search is costly and nominal() is asked often: memoised on what is packed where
        key = (h, h in self.misplaced, repr(sorted(st.place.items())))
        if key not in self._spots:
            at = None
            if h in self.misplaced:
                at = next((a for a in LO.ANCHORS if a != "bottle_corner" and _rest_z(st, h, a) <= LO.FLOOR + 1e-6
                           and spot_fits(st, h, a)), None)
            if at is None and self.prefs.get(h) and spot_fits(st, h, self.prefs[h]):
                at = self.prefs[h]
            self._spots[key] = at or choose_spot(st, h)
        args = dict(rel="in", dest="suitcase", at=self._spots[key])
        if h == "bottle":
            args["orient"] = "upright"
        return args

    def nominal(self, st):
        """The next action of the plan, from the user's beliefs (no side effects)."""
        h = st.holding
        go = lambda where: Action("go_to", where)
        intended = self.intended(st)
        if h:
            if h in intended:
                if st.station != "suitcase":
                    return go("suitcase")
                return Action("place", h, self.place_args(st, h))
            home = S.item(h).station
            return go(home) if st.station != home else Action("put_back", h)
        if st.lid == "open":
            extra = [n for n in ITEM_NAMES if st.in_case(n) and n not in intended]
            fix = [n for n in sorted(self.misplaced) if st.in_case(n) and st.place[n].get("on")]
            if extra or fix:
                if st.station != "suitcase":
                    return go("suitcase")
                return Action("pick_up", topmost_over(st, (extra + fix)[0]))
        todo = self.todo(st)
        zipped = st.zipped or self.thinks_zipped
        if todo:
            if st.lid == "closed":
                if st.pulls == "docked":
                    if st.station != "suitcase":
                        return go("suitcase")
                    if st.locked:
                        k = next(i for i in range(3) if st.dials[i] != int(st.code[i]))
                        return Action("turn", D.DIALS[k], dict(to=int(st.code[k])))
                    return Action("press", "lock_button", dict(hold=0.3, times=1))
                if st.zip_open < 0.95:
                    return go("suitcase") if st.station != "suitcase" else Action("open", "main_zip", dict(to=1.0))
                return go("bed_foot") if st.station != "bed_foot" else Action("open", "lid", dict(to=1.0))
            n = todo[0]
            return go(n) if st.facing != n else Action("pick_up", n)
        if st.lid == "open":
            return go("bed_foot") if st.station != "bed_foot" else Action("close", "lid", dict(to=0.0))
        if not zipped:
            return go("suitcase") if st.station != "suitcase" else Action("close", "main_zip", dict(to=0.0))
        if TASKS[self.task].lock and st.pulls == "free":
            return go("suitcase") if st.station != "suitcase" else Action("insert", "zip_pulls", dict(into="lock"))
        if TASKS[self.task].lock and not st.locked:
            return Action("turn", "dial_1", dict(to=(st.dials[0] + 3) % 10))
        if st.case_pose == "on_bed":
            return go("bed_foot") if st.station != "bed_foot" else Action("stand_up", "suitcase")
        return None

    # ------------------------------------------------------------ guidance
    def truthful(self, g, st, h):
        """Whether a guidance act agrees with the real task and world."""
        a, i = g["act"], g.get("item", "")
        if a == "not_on_list":
            return i not in st.pack
        if a == "on_list":
            return i in st.pack
        if a == "not_packed":
            return i in st.pack and not st.in_case(i)
        if a == "packed":
            return st.in_case(i)
        if a == "on_floor":
            return S.item(i).heavy
        if a == "zip_open":
            return not st.zipped
        if a == "all_packed":
            return all(st.in_case(n) for n in st.pack)
        if a == "next":
            return i in st.pack and not st.in_case(i)
        if a == "where":
            return not g.get("place") or g["place"] in (S.item(i).station, S.item(i).room)
        if a == "dont":
            return h.queued_matches(g)          # stopping a step of an error script
        if a == "do":
            return not harmful(g.get("first"), st)
        return True

    def needed(self, g, st, h):
        """Whether the guidance asks the user to change anything (else it is old news)."""
        a, i = g["act"], g.get("item", "")
        aim = set(self.intended(st)) | ({st.holding} if st.holding else set()) | h.queued_items()
        if a == "not_on_list":
            return i in aim
        if a in ("on_list", "not_packed"):
            return i not in self.intended(st) and not st.in_case(i)
        if a == "packed":
            return i in h.queued_items() or (i in self.todo(st))
        if a == "on_floor":
            return i in aim or (st.in_case(i) and bool(st.place[i].get("on")))
        if a == "zip_open":
            return self.thinks_zipped or h.queued_verb("close", "main_zip")
        if a == "all_packed":
            return self.await_notice
        if a == "next":
            return (bool(self.todo(st)) and self.todo(st)[0] != i) or (h.active is not None and h.active.type == "stall")
        if a == "where":
            return i in h.queued_items() or h.active is not None
        if a == "dont":
            return h.queued_matches(g) or h.nominal_matches(g)
        return a in ("do", "stop")

    def apply(self, g, st, truthful):
        a, i = g["act"], g.get("item", "")
        if a == "not_on_list":
            self.believed.discard(i)
            if not truthful:
                self.omitted.add(i)
        elif a in ("on_list", "not_packed"):
            self.omitted.discard(i)
            if not truthful and i not in st.pack:
                self.believed.add(i)
        elif a == "packed" and not truthful:
            self.omitted.add(i)
        elif a == "on_floor":
            self.misplaced.add(i)
        elif a == "zip_open":
            self.thinks_zipped = False
        elif a == "all_packed":
            self.await_notice = False
        elif a == "next":
            self.next_pref = i

    def decide(self, g, truthful):
        key = (g["act"], g.get("item", ""), g.get("verb", ""))
        n = self.repeats[key] = self.repeats.get(key, 0) + 1
        p = self.p["compliance"] if truthful else self.p["gullibility"]
        return self.rng.random() < 1 - (1 - p) ** n

    def on_refusal(self, feedback):
        if "zip" in feedback:
            self.thinks_zipped = False


def harmful(a, st):
    """A guided action that works against the task: an item that is not on the list into
    the hands or the case, the lid shut with items left, a heavy item on top of another."""
    if a is None:
        return False
    if a.verb == "pick_up" and a.target not in st.pack and not st.in_case(a.target):
        return True
    item = a.target or st.holding
    if a.verb == "place" and item and (item not in st.pack or (S.item(item).heavy and a.get("rel") == "on")):
        return True
    if a.verb == "close" and a.target == "lid" and any(not st.in_case(n) for n in st.pack):
        return True
    return False


# ---------------------------------------------------------------- backends
# Seconds per action at the steady persona's pace (medians of the physics runs in trip/).
DURATION = {"go_to": 6.3, "enter": 5.4, "look_at": 1.4, "pick_up": 6.5, "place": 5.9, "put_back": 6.2,
            "lid": 17.0, "zip": 16.7, "stand_up": 29.3, "insert": 8.0, "turn": 3.0, "press": 2.5, "set": 6.0,
            "fasten": 8.0, "unfasten": 6.0, "roll": 12.0, "fold": 25.0, "refused": 0.5}


def speech_seconds(text):
    return max(1.2, 0.36 * len(text.split()))


class SymbolicBackend:
    """PackSim with a clock: each action takes its measured duration (walking scaled by the
    persona's speed, plus its pause after handling an item)."""

    def __init__(self, task, persona, cloth=False):
        self.sim = PackSim(task, cloth=cloth)
        self.persona = persona
        self.t = 0.0
        self.records = []

    @property
    def time(self):
        return self.t

    def duration(self, a):
        v, t = a.verb, a.target
        if v in ("go_to", "enter"):
            return DURATION[v] * 0.9 / self.persona["walk"]
        if v == "wait":
            return float(a.get("seconds", WAIT))
        if v in ("open", "close") and t == "main_zip":
            return 2.0 + DURATION["zip"] * abs(float(a.get("to")) - self.sim.state.zip_open)
        if v in ("open", "close"):
            return DURATION["lid"]
        d = DURATION.get(v, 3.0)
        return d + (self.persona["pause"] if v in ("pick_up", "place", "put_back") else 0.0)

    def execute(self, a, label, eid):
        ok, msg = self.sim.check(a)
        rec = dict(index=len(self.records), action=a.to_dict(), label=label, event=eid, t_start=round(self.t, 3), ok=ok,
                   feedback=msg)
        if ok:
            note = self.sim.feedback(a)
            self.t += self.duration(a)
            self.sim.step(a)
            if note:
                rec["feedback"] = note
        else:
            self.t += DURATION["refused"]
        rec.update(t_end=round(self.t, 3), state=self.sim.snapshot())
        self.records.append(rec)
        return rec

    def speak(self, speaker, text):
        self.t += speech_seconds(text)


class PhysicsBackend:
    """The MuJoCo executor; speech holds the body still for its duration and is captioned."""

    def __init__(self, runner, executor, seed):
        self.r, self.ex, self.seed = runner, executor, seed

    @property
    def sim(self):
        return self.ex.sim

    @property
    def records(self):
        return self.ex.records

    @property
    def time(self):
        return float(self.r.d.time)

    def execute(self, a, label, eid):
        return execute_with_retries(self.r, self.ex, a, label, eid, self.seed)

    def speak(self, speaker, text):
        self.r.stage(f'{"You" if speaker == "user" else "Assistant"}: "{text}"')
        self.r.event("utterance", speaker=speaker, text=text)
        self.r.hold(speech_seconds(text))


MAX_RETRIES = 3     # physical retries of a failed action (rewound, slightly varied)


def execute_with_retries(r, ex, act, label, eid, seed):
    """Run one action; on a physical failure rewind to just before it and try again with a
    small variation (a person adjusting)."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            rec = ex.execute(act, label, eid)
            break
        except Exception as err:
            if attempt == MAX_RETRIES:
                raise
            print(f"   retry {attempt + 1} of '{act.text()}' after: {err}", flush=True)
            ex.rewind(len(ex.records))
            jr = random.Random(seed * 100 + len(ex.records) * 7 + attempt)
            ex.jitter = dict(stand=(jr.uniform(-0.04, 0.04), jr.uniform(-0.04, 0.04)),
                             time=jr.uniform(0.75, 1.15), place=(jr.uniform(-0.01, 0.01), jr.uniform(-0.01, 0.01)),
                             release=jr.uniform(0.01, 0.03))
            r.event("action_retry", action=act.to_dict(), attempt=attempt + 1, reason=str(err), jitter=ex.jitter)
    ex.jitter = dict(stand=(0.0, 0.0), time=1.0, place=(0.0, 0.0), release=0.0)
    return rec


# ---------------------------------------------------------------- harness
@dataclass
class Step:
    action: Action
    label: str = "nominal"      # nominal | error | recovery | guided | listen | question | request | plan_change
    event: str = ""
    source: str = "plan"        # plan (the user's own) | event (the harness's script) | guidance
    commit: bool = False        # the step at which the event's error becomes real
    say: str = ""               # user act uttered before the step (e.g. noticing a slip)


class Harness:
    def __init__(self, case, backend, assistant=None, max_steps=200, verbose=False):
        self.case = case
        self.rng = random.Random(case["seed"])
        self.persona = PERSONAS[case["persona"]]
        self.events = [e if isinstance(e, Event) else Event.from_dict(e) for e in case["events"]]
        self.user = UserModel(case["task"], self.rng, self.persona, case.get("preferences", []))
        self.b = backend
        self.assistant = assistant or Silent()
        self.max_steps = max_steps
        self.verbose = verbose
        self.queue = []             # harness-owned steps of the active event
        self.active = None          # the event owning the queue
        self.nav = []               # remaining steps of a go_to taken room by room
        self.directed = None        # instructions the user took on from guidance (instruct.Instructions)
        self.pauses = 0
        self.pending_q = None       # the user's open question (an Event)
        self.conv = []              # the conversation: one turn per utterance
        self.steps = []             # Step per executed action, aligned with the backend's records
        self.observations = []
        self.placed_at = {}
        self.belief_edits = {}
        self.current = None
        self.t0 = None
        self.ended = False

    @property
    def st(self):
        return self.b.sim.state

    def n_packed(self, st=None):
        st = st or self.st
        return sum(st.in_case(n) for n in ITEM_NAMES)

    def ev(self, eid):
        return next(e for e in self.events if e.id == eid)

    # ------------------------------------------------------------ talk
    def user_says(self, act, event="", **kw):
        text = self.user.say(act, **kw)
        self.conv.append(dict(t=round(self.b.time, 2), speaker="user", act=DL.act_id(act, kw.get("item", "")),
                              text=text, event=event, step=len(self.steps)))
        if self.verbose:
            print(f"{self.b.time:8.1f}s  USER: {text}")
        self.b.speak("user", text)

    def hear(self, text):
        """An assistant message: understood as guidance acts, answered by the user, and
        credited to the events it addresses."""
        st = self.st
        acts = DL.interpret(text, st, self.pending_q.type if self.pending_q else None)
        turn = dict(t=round(self.b.time, 2), speaker="assistant", text=text, step=len(self.steps), understood=acts,
                    responses=[])
        self.conv.append(turn)
        if self.verbose:
            print(f"{self.b.time:8.1f}s  ASSISTANT: {text}   -> {[a['act'] for a in acts]}")
        self.b.speak("assistant", text)
        reply, taken = None, False
        for g in acts:
            if g["act"] == "do":
                g["first"] = self._ground_first(g["requests"])
            if g["act"] == "answer" and self.pending_q:
                reply = self._answered(g["value"]) or reply
                continue
            if g["act"] in ("ack", "unclear"):
                for e in self._open(channel="plan_change"):
                    self._acknowledge(e)
                if g["act"] == "unclear" and not reply:
                    reply = "clarify" if self.persona["clarify"] else None
                continue
            if g["act"] == "stop":
                self.pauses = 1
                turn["responses"].append(dict(act="stop", accepted=True))
                reply = reply or "accept"
                continue
            truthful = self.user.truthful(g, st, self)
            needed = self.user.needed(g, st, self)
            if not needed:
                turn["responses"].append(dict(act=g["act"], item=g.get("item", ""), needed=False, truthful=truthful))
                if g["act"] == "packed" or (g["act"] == "not_packed" and st.in_case(g.get("item", ""))):
                    reply = reply or "already_done"
                elif g["act"] == "all_packed" and not truthful:
                    reply = reply or "not_yet"
                else:
                    reply = reply or "accept"
                for e in self._open(channel="plan_change"):
                    self._acknowledge(e)
                continue
            ok = self.user.decide(g, truthful)
            turn["responses"].append(dict(act=g["act"], item=g.get("item", ""), needed=True, truthful=truthful,
                                          accepted=ok))
            if not ok:
                if not taken:
                    reply = "decline" if reply in (None, "accept", "already_done") else reply
                continue
            self.user.apply(g, st, truthful)
            reply, taken = "accept", True       # a message taken in part is answered as taken
            if g["act"] == "do":
                self.directed = IN.Instructions(g["requests"])
            if g["act"] == "dont" and not self.queued_matches(g) and self.nominal_matches(g):
                self.pauses = 1             # told not to do what was planned: hold off one step
            for e in self._open():
                if addresses(e, g, st):
                    self._guided(e)
        for e in self._open(channel="plan_change"):
            self._acknowledge(e)
        if reply:
            self.user_says(reply)

    def _ground_first(self, requests):
        try:
            n = IN.Instructions(copy.deepcopy(requests)).next(copy.deepcopy(self.st))
        except Exception:
            return None
        return n[0] if n and n[0] != "rejected" else None

    def _answered(self, value):
        e = self.pending_q
        e.given = value
        e.outcome = "answered_correct" if answer_ok(e, value) else "answered_wrong"
        e.guided = True
        self._resolve(e)
        if e.type == "ask_should_pack" and value == "yes" and e.other not in self.st.pack:
            self.user.believed.add(e.other)         # a wrong answer has consequences
        if e.type == "ask_should_pack" and value == "yes" and e.other in self.st.pack:
            self.user.omitted.discard(e.other)
        self.pending_q = None
        self._release(e)
        return "thanks"

    def _acknowledge(self, e):
        if not e.done:
            e.guided = True
            e.outcome = "acknowledged"
            self._resolve(e)

    # ------------------------------------------------------------ queue predicates (for the user model)
    def _event_steps(self):
        return [s for s in self.queue + self.nav if s.source == "event"]

    def queued_items(self):
        out = {self.active.target} if self.active and self.active.type in ("repetition", "nav_deviation") else set()
        for s in self._event_steps():
            a = s.action
            if a.target in ITEM_NAMES:
                out.add(a.target)
            if a.target in S.STATIONS and self.active and self.active.type in ("repetition", "nav_deviation"):
                out.add(self.active.target)
        return out

    def queued_verb(self, verb, target):
        return any(s.action.verb == verb and s.action.target == target for s in self._event_steps())

    def queued_matches(self, g):
        return any(s.action.verb == g["verb"] and (not g.get("target") or g["target"] == s.action.target)
                   for s in self._event_steps())

    def nominal_matches(self, g):
        a = self.user.nominal(self.st)
        return a is not None and a.verb == g["verb"] and (not g.get("target") or a.target == g["target"])

    # ------------------------------------------------------------ events
    def _open(self, channel=None):
        return [e for e in self.events if e.fired and not e.done and (channel is None or e.channel == channel)]

    def _guided(self, e):
        if not e.guided:
            e.guided = True
            e.steps["guided"] = len(self.steps)
            e.t["guided"] = round(self.b.time, 2)
        if e.type == "premature_close":
            self.user.omitted -= self.belief_edits.get(e.id, set())    # 'not done yet': re-checks the list
        if e is self.active:
            if not e.committed:
                e.cancelled = True
                e.outcome = "prevented"
                self._resolve(e)
            self._release(e)
        if e.type == "request_notify_done":
            e.outcome = "notified" if all(self.st.in_case(n) for n in self.st.pack) else "notified_early"
            self._resolve(e)

    def _release(self, e):
        """Control of the action queue goes back to the user."""
        if self.active is e:
            self.queue = []
            self.nav = [s for s in self.nav if s.event != e.id]
            self.active = None
            e.steps.setdefault("released", len(self.steps))
            e.t.setdefault("released", round(self.b.time, 2))

    def _resolve(self, e):
        if not e.done:
            e.done = True
            e.steps["resolved"] = len(self.steps)
            e.t["resolved"] = round(self.b.time, 2)

    def _fire(self, e):
        """The event happens: the user says something, its beliefs change, or the harness
        takes over the action queue. Returns the scripted steps (may be empty)."""
        st, u = self.st, self.user
        e.fired = True
        e.steps["fired"] = len(self.steps)
        e.t["fired"] = round(self.b.time, 2)
        S_ = lambda a, commit=False, label="error", say="": Step(a, label, e.id, "event", commit, say)
        wait = lambda label: Step(D.action("wait"), label, e.id, "event")
        ty = e.type
        if ty == "wrong_object":
            out = [S_(Action("go_to", e.other)), S_(Action("pick_up", e.other), commit=True)]
            if e.recover:
                out.append(S_(Action("put_back", e.other), label="recovery", say="self_correct"))
            else:
                u.believed.add(e.other)
            return out
        if ty == "omission":
            u.omitted.add(e.target)
            return []
        if ty in ("repetition", "nav_deviation"):
            where = S.item(e.target).station if ty == "repetition" else e.other
            return [S_(Action("go_to", where), commit=True), wait("error")]
        if ty == "heavy_on_top":
            col = max(LO.COLUMNS, key=lambda a: _rest_z(st, e.target, a))
            out = [S_(Action("go_to", "suitcase")),
                   S_(Action("place", e.target, dict(rel="in", dest="suitcase", at=col)), commit=True)]
            if e.recover:
                out += [S_(Action("pick_up", e.target), label="recovery", say="self_correct"),
                        S_(Action("place", e.target, dict(rel="in", dest="suitcase", at="front_strip")), label="recovery")]
            return out
        if ty == "bottle_lying":
            return [S_(Action("go_to", "suitcase")),
                    S_(Action("place", "bottle", dict(rel="in", dest="suitcase", at="bottle_corner", orient="lying")),
                       commit=True)]
        if ty == "premature_close":
            if not e.recover:
                self.belief_edits[e.id] = {n for n in u.intended(st) if not st.in_case(n)}
                u.omitted |= self.belief_edits[e.id]
            return [S_(Action("go_to", "bed_foot")), S_(Action("close", "lid", dict(to=0.0)), commit=True)]
        if ty == "incomplete_zip":
            u.thinks_zipped = True
            return [S_(Action("close", "main_zip", dict(to=0.25)), commit=True)]
        if ty == "stall":
            out = [wait("error") for _ in range(3)]
            out[0].commit = True
            return out
        if e.channel == "question":
            e.answer = self.truth(e)
            kw = dict(item=e.other if ty == "ask_should_pack" else e.target) if ty in (
                "ask_should_pack", "ask_packed", "ask_since") else {}
            self.user_says(ty, event=e.id, **kw)
            self.pending_q = e
            return [wait("question") for _ in range(self.persona["patience"])]
        if ty == "request_remind":
            self.user_says("request_remind", event=e.id, item=e.target)
            u.omitted.add(e.target)
            return []
        if ty == "request_notify_done":
            self.user_says("request_notify_done", event=e.id)
            u.await_notice = True
            u.notice_waits = self.persona["patience"]
            return []
        if ty == "plan_change":
            add, item = e.other.split(":")
            st.pack = tuple(n for n in st.pack if n != item) + ((item,) if add == "add" else ())
            if add == "add" and item not in u.order:
                u.order.append(item)
            u.omitted.discard(item)
            self.user_says(f"announce_{add}", event=e.id, item=item)
            return []
        return []

    def truth(self, e):
        st = self.st
        if e.type == "ask_should_pack":
            return "yes" if e.other in st.pack else "no"
        if e.type == "ask_packed":
            return "yes" if st.in_case(e.target) else "no"
        if e.type == "ask_left":
            return [n for n in st.pack if not st.in_case(n)]
        if e.type == "ask_since":
            t = max(r["t_end"] for r in self.b.records if r["ok"] and r["action"]["verb"] == "place"
                    and r["action"]["target"] == e.target)
            return round(self.b.time - t, 1)
        if e.type == "ask_elapsed":
            return round(self.b.time - (self.t0 or 0.0), 1)
        return None

    def check_resolution(self):
        st = self.st
        for e in self._open():
            ty, done = e.type, False
            if ty == "wrong_object":
                done = e.committed and self.active is not e and st.holding != e.other and not st.in_case(e.other)
            elif ty in ("omission", "request_remind"):
                done = st.in_case(e.target)
            elif ty in ("repetition", "nav_deviation", "stall"):
                done = self.active is not e
            elif ty == "heavy_on_top":
                done = e.committed and st.in_case(e.target) and not st.place[e.target].get("on")
            elif ty == "bottle_lying":
                done = e.committed and st.orient["bottle"] == "upright"
            elif ty == "premature_close":
                done = e.committed and (st.lid == "open" or all(st.in_case(n) for n in st.pack))
            elif ty == "incomplete_zip":
                done = st.zipped
            if done:
                before = e.guided and ("commit" not in e.steps or e.steps["guided"] <= e.steps["commit"])
                if ty == "request_remind":
                    e.outcome = ("reminded" if before else "reminded_late") if e.guided else "self_recovered"
                else:
                    e.outcome = e.outcome or (("prevented" if before else "corrected") if e.guided else "self_recovered")
                self._resolve(e)
        for e in self._open(channel="plan_change"):
            if len(self.steps) - e.steps["fired"] > e.window:
                e.outcome = "unacknowledged"
                self._resolve(e)
        # windows: the harness gives control back after an event's maximum duration
        a = self.active
        if a and a.channel == "error" and len(self.steps) - a.steps["fired"] >= a.window:
            self._release(a)

    def finish(self):
        for e in self.events:
            if not e.fired or e.done:
                continue
            e.outcome = {"question": "unanswered", "plan_change": "unacknowledged"}.get(e.channel) or \
                {"request_remind": "missed", "request_notify_done": "not_notified"}.get(e.type, "unresolved")

    # ------------------------------------------------------------ choosing the next step
    def next_step(self):
        if self.nav:
            return self.nav.pop(0)
        s = self._next()
        if s is None or s.action.verb not in ("go_to", "enter") or s.source == "guidance":
            return s
        steps = navigation(self.st, s.action)
        if not steps:
            return self.next_step()
        out = [Step(a, s.label, s.event, s.source, False, s.say if k == 0 else "") for k, a in enumerate(steps)]
        out[-1].commit = s.commit
        self.nav = out[1:]
        return out[0]

    def _next(self):
        st, u = self.st, self.user
        if self.pauses:
            self.pauses -= 1
            return Step(D.action("wait"), "listen", "", "guidance")
        while self.directed:
            n = self.directed.next(st)
            if n is None:
                self.directed = None
                break
            if n[0] == "rejected":
                self.directed = None
                self.user_says("report_problem", feedback=n[1].message)
                break
            return Step(n[0], "guided", self.active.id if self.active else "", "guidance")
        if self.queue:
            s = self.queue.pop(0)
            if s.label == "question" and self.pending_q is None:
                self.queue = []                 # answered: no more waiting
                return self._next()
            return s
        if self.active and not self.nav:
            self._release(self.active)
        if self.pending_q:                      # gave up waiting for an answer
            self.pending_q.outcome = "unanswered"
            self._resolve(self.pending_q)
            self.pending_q = None
        nxt = u.nominal(st)
        # the user waits for the notice it asked for before closing (up to its patience)
        if closing(st, nxt) and u.await_notice:
            if u.notice_waits > 0:
                u.notice_waits -= 1
                e = next(e for e in self.events if e.type == "request_notify_done")
                return Step(D.action("wait"), "request", e.id, "plan")
            u.await_notice = False
            e = next(e for e in self.events if e.type == "request_notify_done")
            e.outcome = "not_notified"
            self._resolve(e)
        # a forgotten item is remembered just before closing (self-recovery)
        if closing(st, nxt):
            for e in self._open():
                if e.type == "omission" and e.recover and e.target in u.omitted:
                    u.omitted.discard(e.target)
                    if self.persona["narrate"]:
                        self.user_says("remember", event=e.id, item=e.target)
                    return Step(u.nominal(st), "recovery", e.id)
        for e in self.events:
            if not e.fired and holds(e.trigger, st, nxt, self.n_packed()):
                if self.active:
                    continue                    # the harness runs one script at a time
                acts = self._fire(e)
                if acts:
                    self.active = e
                    self.queue = acts[1:]
                    return acts[0]
                nxt = u.nominal(st)
                break                           # one event per step
        if nxt is None:
            return None
        # steps of the plan that undo an open error are recoveries
        for e in self._open(channel="error"):
            if corrective(e, nxt, st, u):
                return Step(nxt, "recovery", e.id)
        if u.thinks_zipped and nxt.verb in ("stand_up", "insert"):
            e = next(e for e in self.events if e.type == "incomplete_zip")
            return Step(nxt, "error", e.id)
        return Step(nxt)

    # ------------------------------------------------------------ the loop
    def run(self):
        self.begin()
        while self.tick():
            pass
        return self.end()

    # run() split into begin / tick / end so a caller (the browser playground) can step it
    def begin(self):
        task = TASKS[self.case["task"]]
        self.t0 = self.b.time
        self.ended = False
        self.user_says("state_task", title="a " + task.title.lower(), items=list(task.pack))
        for item, spot in self.case.get("preferences", []):
            self.user_says("state_pref", item=item, spot=spot)
        return self

    def tick(self):
        """One step: the assistant may speak, then the next action runs. False when the
        user is done (or the step budget is used up)."""
        if self.ended or len(self.steps) >= self.max_steps:
            return False
        for text in self.assistant.respond(self.observation(), self) or []:
            self.hear(text)
        s = self.next_step()
        if s is None:
            return False
        if s.say and self.persona["narrate"]:
            self.user_says(s.say, event=s.event)
        self.current = s
        rec = self.b.execute(s.action, s.label, s.event)
        rec.update(source=s.source, controller="harness" if s.source == "event" else "user")
        if s.commit and rec["ok"] and s.event:
            e = self.ev(s.event)
            e.committed = True
            e.steps["commit"] = len(self.steps)
            e.t["commit"] = rec["t_end"]
            rec["commit"] = True
        self.steps.append(s)
        if self.verbose:
            print(f"{rec['t_start']:8.1f}s  [{s.label}{(' ' + s.event) if s.event else ''}] {s.action.text()}"
                  f"{'' if rec['ok'] else '  -> refused: ' + rec['feedback']}", flush=True)
        self.after(s, rec)
        return True

    def end(self):
        if self.ended:
            return self
        self.ended = True
        # the assistant may still speak at the end (e.g. a late notice)
        for text in self.assistant.respond(self.observation(), self) or []:
            self.hear(text)
        self.check_resolution()
        self.finish()
        return self

    def after(self, s, rec):
        st, u, a = self.st, self.user, s.action
        if rec["ok"]:
            if a.verb == "place":
                self.placed_at[a.target or ""] = a.get("at")
                u.misplaced.discard(a.target)       # one try: if the floor is full, it stays where it fits
            if u.thinks_zipped and st.zipped:
                u.thinks_zipped = False
            if a.verb == "close" and a.target == "lid":
                for e in self._open():
                    if e.type in ("omission", "request_remind") and not st.in_case(e.target) and not e.committed:
                        e.committed = True
                        e.steps["commit"] = len(self.steps) - 1
                        e.t["commit"] = rec["t_end"]
        else:
            u.on_refusal(rec["feedback"])
            if s.source == "guidance":
                self.directed = None
            if s.source != "plan" or s.label == "error":
                self.user_says("report_problem", feedback=rec["feedback"])
        self.check_resolution()
        self.observations.append(self.observation(rec))

    # ------------------------------------------------------------ what an assistant may see
    def observation(self, rec=None):
        """Public: time, the frames of the last action (physics), where the person is
        (a point of the top-down trajectory), what is in their hands, and the conversation
        so far. Labels, events, beliefs and the plan are private."""
        st = self.st
        x, y = place_xy(st)
        o = dict(t=round(self.b.time, 2), step=len(self.steps), position=[round(x, 2), round(y, 2)],
                 room=place_room(st), holding=S.item(st.holding).label if st.holding else "",
                 conversation=len(self.conv))
        if rec is not None:
            o.update(t_start=rec["t_start"], t_end=rec["t_end"])
            if "frame_start" in rec:
                o.update(frames=[rec["frame_start"], rec["frame_end"]])
        return o

    def record(self):
        """The episode: what an assistant may be given (public) and the annotations (private)."""
        task = TASKS[self.case["task"]]
        prefs = self.case.get("preferences", [])
        return dict(
            public=dict(task=dict(title=task.title, pack=[S.item(n).label for n in task.pack]),
                        instruction=self.conv[0]["text"] if self.conv else "",
                        preferences=[dict(item=S.item(i).label, spot=s) for i, s in prefs],
                        conversation=[dict(t=c["t"], speaker=c["speaker"], text=c["text"]) for c in self.conv],
                        observations=self.observations),
            private=dict(persona=dict(name=self.case["persona"], **self.persona),
                         schedule=[e.to_dict() for e in self.events],
                         conversation=self.conv,
                         steps=[dict(index=k, label=s.label, event=s.event, source=s.source, commit=s.commit)
                                for k, s in enumerate(self.steps)],
                         beliefs=dict(omitted=sorted(self.user.omitted), believed=sorted(self.user.believed),
                                      misplaced=sorted(self.user.misplaced), thinks_zipped=self.user.thinks_zipped),
                         preferences=self.preference_goals(),
                         outcomes={e.id: e.outcome for e in self.events}))

    def preference_goals(self):
        return {f"pref({i}, {s})": bool(self.st.in_case(i) and self.placed_at.get(i) == s)
                for i, s in self.case.get("preferences", [])}


def corrective(e, a, st, u):
    """A step of the user's own plan that undoes an open error."""
    if e.type == "wrong_object":
        return st.holding == e.other or (a.verb == "pick_up" and a.target == e.other) or \
            (a.verb == "go_to" and st.in_case(e.other))
    if e.type == "premature_close":
        return a.target in ("lid", "main_zip") and a.verb == "open" or (a.verb == "go_to" and st.lid == "closed")
    if e.type == "heavy_on_top":
        return e.committed and (st.holding == e.target or a.target == e.target)
    if e.type == "incomplete_zip":
        return e.committed and not u.thinks_zipped and (a.target == "main_zip" or a.verb == "go_to")
    return False


def addresses(e, g, st):
    """Whether an accepted guidance act is about this event."""
    a, i = g["act"], g.get("item", "")
    ty = e.type
    if a == "do" and e.channel == "error":
        return True
    if ty == "wrong_object":
        return (a in ("not_on_list",) and i == e.other) or (a in ("on_list", "next", "where") and i == e.target) or \
            (a == "dont" and g["verb"] in ("pick_up", "go_to"))
    if ty in ("omission", "request_remind"):
        return a in ("not_packed", "on_list", "next") and i == e.target
    if ty == "repetition":
        return (a == "packed" and i == e.target) or a == "next" or (a == "dont" and g["verb"] == "go_to")
    if ty == "nav_deviation":
        return (a in ("where", "next", "on_list") and i == e.target) or (a == "dont" and g["verb"] == "go_to")
    if ty == "heavy_on_top":
        return (a == "on_floor" and i == e.target) or (a == "dont" and g["verb"] == "place")
    if ty == "premature_close":
        return (a in ("not_packed", "on_list") and not st.in_case(i)) or (a == "dont" and g["verb"] == "close")
    if ty == "incomplete_zip":
        return a == "zip_open"
    if ty == "stall":
        return a in ("next", "not_packed", "on_list", "where")
    if ty == "request_notify_done":
        return a == "all_packed"
    return False


def answer_ok(e, v):
    if e.type in ("ask_should_pack", "ask_packed"):
        return v == e.answer
    if e.type == "ask_left":
        return isinstance(v, list) and set(v) == set(e.answer)
    if e.type in ("ask_since", "ask_elapsed"):
        return v is not None and abs(v - e.answer) <= max(15.0, 0.25 * e.answer)
    return False


# ---------------------------------------------------------------- assistants (test fixtures)
class Silent:
    """Never speaks: the user's own behaviour, errors and self-recoveries only."""

    def respond(self, obs, h):
        return []


class Scripted:
    """Says fixed lines at given steps: [{"step": 12, "text": "..."}, ...]."""

    def __init__(self, lines):
        self.lines = sorted(lines, key=lambda x: x["step"])

    def respond(self, obs, h):
        out = [x["text"] for x in self.lines if x["step"] == obs["step"]]
        return out


class Interactive:
    """You are the assistant: you see what an assistant would (and the user's words), and
    type a message or press Enter to stay silent."""

    def __init__(self):
        self.seen = 0

    def respond(self, obs, h):
        for c in h.conv[self.seen:]:
            if c["speaker"] == "user":
                print(f'   user: "{c["text"]}"')
        self.seen = len(h.conv)
        if h.steps:
            print(f"   [{obs['t']:.0f}s] {h.steps[-1].action.text()}  (in {R.LABEL[obs['room']]}"
                  f"{', holding the ' + obs['holding'] if obs['holding'] else ''})")
        try:
            text = input("assistant> ").strip()
        except EOFError:
            return []
        self.seen = len(h.conv) + (1 if text else 0)
        return [text] if text else []


class Oracle:
    """Privileged test fixture: reads the hidden schedule and says the right thing at the
    right time, in plain words that go through the same interpretation as any assistant's.
    Used to check that every scheduled event can be prevented, corrected or answered."""

    def __init__(self, persistence=3):
        self.told = {}
        self.persistence = persistence

    def respond(self, obs, h):
        st, u = h.st, h.user
        lab = DL.lab
        for e in h.events:
            if not e.fired or e.done or self.told.get(e.id, 0) >= self.persistence:
                continue
            msg = self.message(e, h, st, u, lab)
            if msg:
                self.told[e.id] = self.told.get(e.id, 0) + 1
                return [msg]
        return []

    def message(self, e, h, st, u, lab):
        ty = e.type
        if e.channel == "question" and h.pending_q is e:
            v = e.answer
            if ty == "ask_should_pack":
                return "Yes, it's on your list." if v == "yes" else "No, it's not on your list."
            if ty == "ask_packed":
                return "Yes, it's already in." if v == "yes" else "No, not yet."
            if ty == "ask_left":
                return f"You still need {DL.items_text(v)}." if v else "Nothing, that's everything."
            if ty == "ask_since" and v < 8:
                return "Just now."
            if ty in ("ask_since", "ask_elapsed"):
                return f"About {round(v)} seconds{' ago' if ty == 'ask_since' else ' so far'}."
        if ty == "plan_change":
            return "Okay, noted."
        if ty == "request_notify_done":
            return "That's everything on your list." if all(st.in_case(n) for n in st.pack) else None
        if e.guided:
            return None                 # it was taken; the user is on it
        nxt = u.nominal(st)
        if ty in ("omission", "request_remind"):
            if e.target in u.omitted and (closing(st, nxt) or ty == "omission"):
                return f"Don't forget the {lab(e.target)}." if ty == "omission" else \
                    f"Remember the {lab(e.target)} before you close the lid."
            return None
        if h.active is not e and not e.committed:
            return None
        if ty == "wrong_object":
            if st.holding == e.other or h.active is e or st.in_case(e.other):
                return f"Careful, the {lab(e.other)} {DL.be(e.other)}n't on your list. You want the {lab(e.target)}."
        if ty == "repetition":
            return f"You already packed the {lab(e.target)}."
        if ty == "nav_deviation":
            return f"The {lab(e.target)} {DL.be(e.target)} in {R.LABEL[S.item(e.target).room]}."
        if ty == "heavy_on_top":
            return f"Put the {lab(e.target)} on the bottom of the case, not on top of the clothes."
        if ty == "premature_close":
            left = [n for n in st.pack if not st.in_case(n)]
            return f"Don't close the lid yet, you still need the {lab(left[0])}." if left else None
        if ty == "incomplete_zip":
            return "The zip isn't closed all the way." if e.committed else None
        if ty == "stall":
            todo = [n for n in u.intended(st) if not st.in_case(n)]
            return f"Next you could get the {lab(todo[0])}." if todo else "Next you can close the lid."
        return None


# ---------------------------------------------------------------- the slot for a model
def public_view(h, obs):
    """Everything an assistant under test is given at one step, and nothing private: the
    task and stated preferences, the conversation so far, and the observations (latest
    last). No labels, events, beliefs or plan."""
    task = TASKS[h.case["task"]]
    return dict(task=dict(title=task.title, pack=[S.item(n).label for n in task.pack]),
                preferences=[dict(item=S.item(i).label, spot=s) for i, s in h.case.get("preferences", [])],
                conversation=[dict(t=c["t"], speaker=c["speaker"], text=c["text"]) for c in h.conv],
                now=obs, history=list(h.observations))


class Model:
    """The assistant under test: a function respond(view) -> text, a list of texts, or
    None (stay quiet), called once before every action of the user and once at the end.
    `view` is public_view(). A function that always returns None is the empty slot: the
    episode runs as with Silent."""

    def __init__(self, fn):
        self.fn = fn

    def respond(self, obs, h):
        out = self.fn(public_view(h, obs))
        if not out:
            return []
        return [t.strip() for t in ([out] if isinstance(out, str) else out) if t and t.strip()]


def load_model(path):
    """A Python file defining respond(view) (see assistant_template.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("assistant_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return Model(mod.respond)


ASSISTANTS = dict(none=Silent, oracle=Oracle, interactive=Interactive)


def make_assistant(spec):
    """'none' | 'oracle' | 'interactive' | a .py file defining respond(view) (the model
    slot) | path to a JSON list of {"step", "text"}."""
    if spec in ASSISTANTS:
        return ASSISTANTS[spec]()
    if spec.endswith(".py"):
        return load_model(spec)
    from pathlib import Path
    return Scripted(json.loads(Path(spec).read_text()))
