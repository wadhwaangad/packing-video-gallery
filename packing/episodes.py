"""Episode construction: a scripted user, progress-conditioned events and personas.

A case = task + seed + persona + event schedule. The user is a replanning policy over
the symbolic state (PackSim): it does not replay a fixed action list, so an event that
changes the world (a wrong pile packed, the lid shut early) changes what comes next.

An event has a semantic trigger, its targets, the deviating actions it injects, and an
optional self-recovery. While an event is active it owns the action queue; the
harness releases it on completion. Questions and plan changes are separate,
non-error channels. Every case is dry-run symbolically before any physics: schedules
whose events cannot fire, or whose actions would be refused, are rejected.
"""
from dataclasses import dataclass, field
import json, random
import scene as S
from packsim import PackSim, Action, TASKS, ITEM_NAMES, initial_state

PERSONAS = {
    "brisk": dict(walk=1.05, pause=0.0, label="walks quickly, no pauses"),
    "steady": dict(walk=0.90, pause=0.4, label="even pace, short pause after each step"),
    "deliberate": dict(walk=0.75, pause=1.0, label="slow walker, pauses to look after each step"),
}

EVENT_TYPES = {
    # type: (channel, description)
    "wrong_object": ("error", "picks up a pile that is not on the list instead of the intended one"),
    "omission": ("error", "skips an item on the list"),
    "repetition": ("error", "walks back for an item that is already packed"),
    "nav_deviation": ("error", "heads to the wrong room first"),
    "heavy_on_top": ("error", "puts the heavy toiletry bag on top of a clothes pile"),
    "bottle_lying": ("error", "lays the water bottle down instead of standing it up (not sampled by default yet: the lying placement is not reliable with the scanned case)"),
    "premature_close": ("error", "closes the lid before everything is packed"),
    "stall": ("error", "stops making progress for several seconds"),
    "question": ("question", "asks a question about the task"),
    "plan_change": ("plan_change", "announces a legitimate change to the packing list"),
}


@dataclass
class Event:
    id: str
    type: str
    target: str = ""            # the intended item / station the event is about
    other: str = ""             # distractor, wrong station, item added or removed
    after: int = -1             # fire once this many items are packed (count triggers)
    recover: bool = True
    text: str = ""
    fired: bool = False
    done: bool = False
    t: dict = field(default_factory=dict)

    @property
    def channel(self):
        return EVENT_TYPES[self.type][0]

    def to_dict(self):
        return dict(id=self.id, type=self.type, channel=self.channel, target=self.target, other=self.other,
                    after=self.after, recover=self.recover, text=self.text, fired=self.fired,
                    description=EVENT_TYPES[self.type][1])


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


def choose_region(st, item):
    it = S.item(item)
    if it.family == "bottle":
        return "bottle_corner"
    cols = ["wheel_end", "handle_end"]
    fits = lambda r: st.height(r) + it.size[2] <= S.INNER_H - 0.008 and \
        (not st.stacks[r] or S.FAMILIES[S.item(st.stacks[r][-1]).family]["stackable"])
    if item == "toiletry_bag" or it.family == "fragile_box":
        return "front_strip" if fits("front_strip") else min(cols, key=st.height)
    if it.heavy:
        empty = [r for r in cols if not st.stacks[r]]
        if empty:
            return empty[0]
    ok = [r for r in cols if fits(r)]
    if ok:
        return min(ok, key=st.height)
    return "front_strip"


class User:
    """Nominal policy plus event handling. `intended` is what the user currently means to pack."""

    def __init__(self, task, rng, events=()):
        self.task = task
        self.order = pack_order(task, rng)
        self.events = list(events)
        self.queue = []             # (Action, label, event id) injected by an active event
        self.omitted = set()        # items the user has (for now) forgotten
        self.believed = set()       # items the user wrongly thinks belong in the case
        self.packed_count = 0
        self.last_error = ""

    def intended(self, st):
        return [n for n in self.order + sorted(self.believed) if (n in st.pack or n in self.believed)
                and n not in self.omitted]

    def nominal(self, st):
        h = st.holding
        if h:
            if h in self.intended(st):
                if st.station != "suitcase":
                    return Action("go_to", "suitcase")
                return Action("put_in", choose_region(st, h), "upright" if h == "bottle" else "")
            home = S.item(h).station
            if st.station != home:
                return Action("go_to", home)
            return Action("put_back")
        extra = [n for n in ITEM_NAMES if st.in_case(n) and n not in self.intended(st)]
        if extra and st.lid == "open":
            n = extra[0]
            stack = st.stacks[st.region_of(n)]
            top = stack[-1]
            if st.station != "suitcase":
                return Action("go_to", "suitcase")
            return Action("take_out", top)
        todo = [n for n in self.intended(st) if not st.in_case(n)]
        if todo:
            if st.lid == "closed":
                if st.zipped:
                    return Action("go_to", "suitcase") if st.station != "suitcase" else Action("unzip")
                return Action("go_to", "bed_foot") if st.station != "bed_foot" else Action("open_lid")
            n = todo[0]
            if st.facing != n:
                return Action("go_to", n)
            return Action("pick_up", n)
        if st.lid == "open":
            return Action("go_to", "bed_foot") if st.station != "bed_foot" else Action("close_lid")
        if not st.zipped:
            return Action("go_to", "suitcase") if st.station != "suitcase" else Action("zip")
        if st.case_pose == "on_bed":
            return Action("go_to", "bed_foot") if st.station != "bed_foot" else Action("stand_up")
        return None

    # ------------------------------------------------------------ events
    def _fire(self, ev, st, nxt):
        """Return the actions an event injects (or [] if it only edits the user's beliefs)."""
        E = lambda a, label="error": (a, label, ev.id)
        if ev.type == "wrong_object":
            acts = [E(Action("go_to", ev.other)), E(Action("pick_up", ev.other))]
            if ev.recover:
                acts += [E(Action("put_back"), "recovery")]
            else:
                self.believed.add(ev.other)
            return acts
        if ev.type == "omission":
            self.omitted.add(ev.target)
            return []
        if ev.type == "repetition":
            return [E(Action("go_to", S.item(ev.target).station)), E(Action("wait", "", "2"))]
        if ev.type == "nav_deviation":
            return [E(Action("go_to", ev.other)), E(Action("wait", "", "1.5"))]
        if ev.type == "heavy_on_top":
            col = max(["wheel_end", "handle_end"], key=st.height)
            acts = [E(Action("put_in", col))]
            if ev.recover:
                acts += [E(Action("take_out", ev.target), "recovery"), E(Action("put_in", "front_strip"), "recovery")]
            return acts
        if ev.type == "bottle_lying":
            return [E(Action("put_in", "bottle_corner", "lying"))]
        if ev.type == "premature_close":
            acts = [E(Action("go_to", "bed_foot")), E(Action("close_lid"))]
            if not ev.recover:
                self.omitted |= {n for n in self.intended(st) if not st.in_case(n)}
            return acts
        if ev.type == "stall":
            return [E(Action("wait", "", "6"))]
        if ev.type == "question":
            return [E(Action("say", "", ev.text), "question"), E(Action("wait", "", "1.5"), "question")]
        if ev.type == "plan_change":
            return [E(Action("say", "", ev.text), "plan_change")]
        return []

    def _triggered(self, ev, st, nxt):
        if ev.fired or nxt is None:
            return False
        if ev.type == "wrong_object":
            return nxt.verb == "go_to" and nxt.target == ev.target and not st.holding
        if ev.type == "omission":
            return nxt.verb == "go_to" and nxt.target == ev.target
        if ev.type == "nav_deviation":
            return nxt.verb == "go_to" and nxt.target == ev.target and not st.holding
        if ev.type == "repetition":
            return st.in_case(ev.target) and nxt.verb == "go_to" and not st.holding and st.station == "suitcase"
        if ev.type == "heavy_on_top":
            return nxt.verb == "put_in" and st.holding == ev.target and \
                any(st.stacks[r] for r in ("wheel_end", "handle_end"))
        if ev.type == "bottle_lying":
            return nxt.verb == "put_in" and st.holding == "bottle"
        if ev.type in ("premature_close", "stall", "question", "plan_change"):
            return self.n_packed(st) >= ev.after and nxt.verb == "go_to" and not st.holding and st.lid == "open"
        return False

    def n_packed(self, st):
        return sum(st.in_case(n) for n in ITEM_NAMES)

    def next(self, st):
        """Next (action, label, event id). Recovery for omissions happens before closing."""
        if self.queue:
            return self.queue.pop(0)
        nxt = self.nominal(st)
        # A forgotten item is remembered just before the lid is closed (self-recovery).
        closing = nxt is not None and (nxt.verb == "close_lid" or (nxt.verb == "go_to" and nxt.target == "bed_foot"
                                                                   and st.lid == "open"))
        for ev in self.events:
            if ev.type == "omission" and ev.fired and ev.recover and not ev.done and closing:
                self.omitted.discard(ev.target); ev.done = True
                return (self.nominal(st), "recovery", ev.id)
        for ev in self.events:
            if self._triggered(ev, st, nxt):
                ev.fired = True
                if ev.type == "plan_change":
                    add, item = ev.other.split(":")
                    st.pack = tuple(n for n in st.pack if n != item) + ((item,) if add == "add" else ())
                    if add == "add" and item not in self.order:
                        self.order.append(item)
                acts = self._fire(ev, st, nxt)
                if not acts:
                    nxt = self.nominal(st)
                    continue
                self.queue = acts[1:]
                if ev.type not in ("omission",):
                    ev.done = True
                if ev.channel == "error":
                    self.last_error = ev.id
                return acts[0]
        if nxt is None:
            return None
        # Corrective steps the policy takes on its own after an error are recoveries.
        if self.last_error and (nxt.verb in ("open_lid", "unzip", "put_back") or
                                (nxt.verb == "take_out" and nxt.target not in self.intended(st))):
            return (nxt, "recovery", self.last_error)
        return (nxt, "nominal", "")


# ---------------------------------------------------------------- case generation
def dry_run(case, max_steps=200):
    """Run a case symbolically. Returns (ok, trace, sim)."""
    rng = random.Random(case["seed"])
    sim = PackSim(case["task"])
    events = [Event(**{k: v for k, v in e.items() if k in Event.__dataclass_fields__}) for e in case["events"]]
    user = User(case["task"], rng, events)
    trace = []
    for _ in range(max_steps):
        nxt = user.next(sim.state)
        if nxt is None:
            break
        a, label, eid = nxt
        ok, msg = sim.step(a)
        trace.append(dict(action=a.to_dict(), label=label, event=eid, ok=ok, feedback=msg))
        if not ok:
            return False, trace, sim, user
    fired = all(e.fired for e in events)
    return fired and user.next(sim.state) is None, trace, sim, user


def _distractor_near(item, task, rng):
    pack = set(TASKS[task].pack)
    it = S.item(item)
    cands = [o for o in S.ITEMS if o.name not in pack and o.family == it.family]
    same_room = [o for o in cands if o.room == it.room]
    pool = same_room or cands
    return rng.choice(pool).name if pool else ""


def sample_case(task=None, seed=0, n_events=None, persona=None, allow=None):
    """Sample a case whose dry run succeeds (events fire, every action is legal)."""
    for attempt in range(200):
        rng = random.Random(seed * 1000 + attempt)
        tk = task or rng.choice(sorted(TASKS))
        pack = list(TASKS[tk].pack)
        k = n_events if n_events is not None else rng.choice([1, 2, 2, 3])
        # bottle_lying is in the taxonomy but its placement skill is not yet reliable with the
        # scanned case, so it is only used when asked for explicitly.
        types = [t for t in EVENT_TYPES if (allow is None and t != "bottle_lying") or (allow and t in allow)]
        evs, used_targets = [], set()
        for i in range(k):
            ty = rng.choice([t for t in types if t not in {x['type'] for x in evs}] or types)
            e = dict(id=f"e{i + 1}", type=ty, target="", other="", after=-1, recover=rng.random() < 0.7, text="")
            if ty in ("wrong_object", "omission", "nav_deviation", "repetition"):
                cand = [n for n in pack if n not in used_targets]
                if not cand:
                    continue
                e["target"] = rng.choice(cand)
                if ty == "wrong_object":
                    e["other"] = _distractor_near(e["target"], tk, rng)
                    if not e["other"]:
                        continue
                if ty == "nav_deviation":
                    rooms = [s for s in S.STATIONS if S.STATIONS[s].room != S.item(e["target"]).room
                             and s not in ("suitcase", "bed_foot")]
                    e["other"] = rng.choice(rooms)
                used_targets.add(e["target"])
            elif ty == "heavy_on_top":
                if "toiletry_bag" not in pack or "toiletry_bag" in used_targets:
                    continue
                e["target"] = "toiletry_bag"; used_targets.add("toiletry_bag")
            elif ty == "bottle_lying":
                if "bottle" not in pack or "bottle" in used_targets:
                    continue
                e["target"] = "bottle"; e["recover"] = False; used_targets.add("bottle")
            elif ty in ("premature_close", "stall", "question", "plan_change"):
                e["after"] = rng.randint(1, max(1, len(pack) - 2))
                if ty == "question":
                    extra = [o.label for o in S.ITEMS if o.name not in pack]
                    e["text"] = f"Should the {rng.choice(extra)} go in too?" if extra else "Is the bottle closed?"
                if ty == "plan_change":
                    removable = [n for n in pack if n not in used_targets and not S.item(n).heavy]
                    addable = [o.name for o in S.ITEMS if o.name not in pack and o.family == "folded_pile"]
                    if rng.random() < 0.5 and removable:
                        it = rng.choice(removable); e["other"] = f"remove:{it}"
                        e["text"] = f"Actually, I'll leave the {S.item(it).label} at home."
                    elif addable:
                        it = rng.choice(addable); e["other"] = f"add:{it}"
                        e["text"] = f"I'll take the {S.item(it).label} as well."
                    else:
                        continue
                    used_targets.add(it)
            evs.append(e)
        case = dict(task=tk, seed=seed, attempt=attempt, persona=persona or rng.choice(sorted(PERSONAS)), events=evs)
        ok, trace, sim, user = dry_run(case)
        if ok and _windows_separated(trace):
            case["dry_run"] = dict(actions=len(trace), final_goals=sim.goals())
            return case
    raise RuntimeError("could not sample a valid case")


def _windows_separated(trace):
    """Distinct events must not overlap: at least one nominal action between them."""
    last, gap = None, 1
    for t in trace:
        if t["event"] and t["event"] != last and last is not None and gap == 0:
            return False
        if t["event"]:
            last, gap = t["event"], 0
        else:
            gap += 1
    return True


def reference_case():
    """The case shown on the page: weekend trip, one wrong pile picked up and put back."""
    return dict(task="weekend_trip", seed=3, persona="steady",
                events=[dict(id="e1", type="wrong_object", target="sweatshirts", other="striped_shirts", recover=True)])


if __name__ == "__main__":
    import sys
    case = reference_case() if len(sys.argv) < 2 else sample_case(seed=int(sys.argv[1]))
    ok, trace, sim, _ = dry_run(case)
    print(json.dumps({k: v for k, v in case.items() if k != "dry_run"}, indent=1))
    for t in trace:
        print(f"{t['label']:>11} {t['event']:>3} {'ok ' if t['ok'] else 'NO '} {t['action']['text']}  {t['feedback']}")
    print("valid:", ok, "goals:", sim.goals())
