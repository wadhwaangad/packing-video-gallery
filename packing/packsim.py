"""PackSim: the symbolic layer of the packing simulator.

It holds the task-level world state, defines the semantic action space, checks each
action's preconditions (returning feedback instead of silently changing the world),
applies effects, and evaluates goal predicates. It has no physics: the executor
(executor.py) grounds each accepted action in the MuJoCo simulation and commits the
effect here only after the physical outcome has been verified.

Action space (verbs and arguments; angle brackets are resolved by the grounder):
  navigate     go_to <station | item>
  manipulate   pick_up <item>      put_in <region> [upright|lying]     put_back
               take_out <item>
  case         close_lid   zip   unzip   open_lid   stand_up
  other        wait <seconds>      say <text>
"""
from dataclasses import dataclass, field, asdict
import copy, re
import scene as S


# ---------------------------------------------------------------- actions
@dataclass
class Action:
    verb: str
    target: str = ""        # station, item or region
    arg: str = ""           # orientation for put_in, text for say, seconds for wait

    def text(self):
        lab = lambda n: S.item(n).label if n in ITEM_NAMES else S.STATIONS[n].label if n in S.STATIONS else \
            S.REGIONS[n].label if n in S.REGIONS else n
        return {
            "go_to": lambda: f"go to {lab(self.target)}",
            "pick_up": lambda: f"pick up the {lab(self.target)}",
            "put_in": lambda: f"put it in {lab(self.target)}" + (f", {self.arg}" if self.arg else ""),
            "put_back": lambda: "put it back where it was",
            "take_out": lambda: f"take the {lab(self.target)} out of the suitcase",
            "close_lid": lambda: "close the lid", "zip": lambda: "zip the case", "unzip": lambda: "unzip the case",
            "open_lid": lambda: "open the lid", "stand_up": lambda: "stand the case up on its wheels",
            "wait": lambda: f"pause for {self.arg} s", "say": lambda: f'say "{self.arg}"',
        }[self.verb]()

    def to_dict(self):
        return dict(verb=self.verb, target=self.target, arg=self.arg, text=self.text())


VERBS = {
    "go_to": "navigate", "pick_up": "manipulate", "put_in": "manipulate", "put_back": "manipulate",
    "take_out": "manipulate", "close_lid": "case", "zip": "case", "unzip": "case", "open_lid": "case",
    "stand_up": "case", "wait": "other", "say": "other",
}
ITEM_NAMES = [it.name for it in S.ITEMS]

# Station each case action is performed from.
CASE_STATION = {"close_lid": "bed_foot", "open_lid": "bed_foot", "stand_up": "bed_foot",
                "zip": "suitcase", "unzip": "suitcase"}


# ---------------------------------------------------------------- tasks
@dataclass
class Task:
    name: str
    title: str
    pack: tuple                      # items that must end up in the case
    note: str = ""


TASKS = {t.name: t for t in [
    Task("weekend_trip", "Weekend away", ("trousers", "tshirts", "sweatshirts", "towel", "toiletry_bag", "bottle",
                                         "notebook", "camera"), "the reference task shown on the page"),
    Task("gym_weekend", "Gym weekend", ("gym_clothes", "tshirts", "towel", "toiletry_bag", "bottle")),
    Task("work_trip", "Work trip", ("trousers", "striped_shirts", "toiletry_bag", "notebook", "bottle")),
    Task("beach_holiday", "Beach holiday", ("tshirts", "striped_shirts", "towel_stack", "toiletry_bag", "bottle", "camera")),
    Task("winter_visit", "Winter visit", ("sweater_pile", "sweatshirts", "trousers", "toiletry_bag", "bottle", "notebook")),
]}


# ---------------------------------------------------------------- state
@dataclass
class State:
    station: str = "suitcase"
    facing: str = ""                                  # item the person stands in front of
    holding: str = ""
    loc: dict = field(default_factory=dict)           # item -> "station:<name>" | "held" | "case:<region>"
    stacks: dict = field(default_factory=dict)        # region -> [items bottom..top]
    orient: dict = field(default_factory=dict)        # bottle -> upright | lying
    lid: str = "open"
    zipped: bool = False
    case_pose: str = "on_bed"                         # on_bed | upright_floor
    pack: tuple = ()                                  # current goal list (plan changes may edit it)

    def where(self, item):
        return self.loc[item]

    def in_case(self, item):
        return self.loc[item].startswith("case:")

    def region_of(self, item):
        return self.loc[item].split(":", 1)[1] if self.in_case(item) else ""

    def height(self, region):
        return sum(S.item(n).size[2] for n in self.stacks[region])


def initial_state(task):
    st = State(pack=tuple(TASKS[task].pack))
    st.loc = {it.name: f"station:{it.station}" for it in S.ITEMS}
    st.stacks = {r: [] for r in S.REGIONS}
    st.orient = {"bottle": "upright"}
    return st


class PackSim:
    """Legality, effects and goals over State. `check` never changes the world."""

    def __init__(self, task="weekend_trip"):
        self.task = task
        self.state = initial_state(task)
        self.log = []

    # ------------------------------------------------------------ preconditions
    def check(self, a):
        st = self.state
        it = lambda n: S.item(n)
        if a.verb not in VERBS:
            return False, f"unknown action '{a.verb}'"
        if a.verb in ("pick_up", "take_out") and a.target not in ITEM_NAMES:
            return False, f"there is no '{a.target}' here"
        if st.case_pose != "on_bed" and a.verb not in ("go_to", "wait", "say"):
            return False, "the case is already standing on the floor"
        if a.verb == "go_to":
            if a.target not in S.STATIONS and a.target not in ITEM_NAMES:
                return False, f"cannot find '{a.target}'"
            if a.target in ITEM_NAMES and st.loc[a.target] == "held":
                return False, f"you are already holding the {it(a.target).label}"
            return True, ""
        if a.verb == "pick_up":
            if st.holding:
                return False, f"your hands are full (holding the {it(st.holding).label})"
            if st.in_case(a.target):
                return False, f"the {it(a.target).label} is in the suitcase: take it out instead"
            home = st.loc[a.target].split(":", 1)[1]
            if st.station != home or st.facing != a.target:
                return False, f"the {it(a.target).label} is not within reach (you are at {S.STATIONS[st.station].label})"
            return True, ""
        if a.verb == "take_out":
            if st.holding:
                return False, f"your hands are full (holding the {it(st.holding).label})"
            if not st.in_case(a.target):
                return False, f"the {it(a.target).label} is not in the suitcase"
            if st.station != "suitcase":
                return False, "you are not at the suitcase"
            if st.lid != "open":
                return False, "the lid is closed"
            stack = st.stacks[st.region_of(a.target)]
            if stack[-1] != a.target:
                return False, f"the {it(stack[stack.index(a.target) + 1]).label} is on top of it"
            return True, ""
        if a.verb == "put_in":
            if not st.holding:
                return False, "you are not holding anything"
            if st.station != "suitcase":
                return False, "you are not at the suitcase"
            if st.lid != "open":
                return False, "the lid is closed"
            if a.target not in S.REGIONS:
                return False, f"there is no place called '{a.target}' in the case"
            reg, item = S.REGIONS[a.target], it(st.holding)
            if item.family not in reg.families:
                return False, f"the {item.label} does not fit in {reg.label}"
            if st.stacks[a.target] and not S.FAMILIES[it(st.stacks[a.target][-1]).family]["stackable"]:
                return False, f"nothing can go on top of the {it(st.stacks[a.target][-1]).label}"
            if st.height(a.target) + item.size[2] > S.INNER_H - 0.008:
                return False, f"{reg.label} is full"
            if a.arg and item.family != "bottle":
                return False, "only the bottle can be stood upright or laid down"
            return True, ""
        if a.verb == "put_back":
            if not st.holding:
                return False, "you are not holding anything"
            home = it(st.holding).station
            if st.station != home:
                return False, f"the {it(st.holding).label} belongs at {S.STATIONS[home].label}"
            return True, ""
        if a.verb in CASE_STATION:
            if st.holding:
                return False, "put down what you are holding first"
            if st.station != CASE_STATION[a.verb]:
                return False, f"do that from {S.STATIONS[CASE_STATION[a.verb]].label}"
            need = {"close_lid": (st.lid == "open", "the lid is already closed"),
                    "open_lid": (st.lid == "closed" and not st.zipped, "unzip the case first" if st.zipped else "the lid is already open"),
                    "zip": (st.lid == "closed" and not st.zipped, "close the lid first" if st.lid == "open" else "already zipped"),
                    "unzip": (st.zipped, "the case is not zipped"),
                    "stand_up": (st.zipped, "zip the case before standing it up")}[a.verb]
            if not need[0]:
                return False, need[1]
            if a.verb == "close_lid" and max(st.height(r) for r in S.REGIONS) > S.INNER_H:
                return False, "the case is too full to close"
            return True, ""
        return True, ""

    # ------------------------------------------------------------ effects
    def apply(self, a):
        st = self.state
        if a.verb == "go_to":
            if a.target in ITEM_NAMES:
                where = st.loc[a.target]
                st.station = "suitcase" if where.startswith("case:") else where.split(":", 1)[1]
                st.facing = "" if where.startswith("case:") else a.target
            else:
                st.station, st.facing = a.target, ""
        elif a.verb in ("pick_up", "take_out"):
            if st.in_case(a.target):
                st.stacks[st.region_of(a.target)].remove(a.target)
            st.loc[a.target] = "held"; st.holding = a.target
        elif a.verb == "put_in":
            st.loc[st.holding] = f"case:{a.target}"; st.stacks[a.target].append(st.holding)
            if st.holding == "bottle":
                st.orient["bottle"] = a.arg or "upright"
            st.holding = ""
        elif a.verb == "put_back":
            st.loc[st.holding] = f"station:{S.item(st.holding).station}"
            if st.holding == "bottle":
                st.orient["bottle"] = "upright"
            st.facing = st.holding; st.holding = ""
        elif a.verb == "close_lid":
            st.lid = "closed"
        elif a.verb == "open_lid":
            st.lid = "open"
        elif a.verb == "zip":
            st.zipped = True
        elif a.verb == "unzip":
            st.zipped = False
        elif a.verb == "stand_up":
            st.case_pose = "upright_floor"

    def step(self, a):
        ok, msg = self.check(a)
        self.log.append(dict(action=a.to_dict(), ok=ok, feedback=msg))
        if ok:
            self.apply(a)
        return ok, msg

    # ------------------------------------------------------------ goals
    def goals(self):
        st = self.state
        g = {}
        for n in st.pack:
            g[f"packed({n})"] = st.in_case(n)
        extras = [n for n in ITEM_NAMES if n not in st.pack and st.in_case(n)]
        g["no_extra_items"] = not extras
        if "bottle" in st.pack:
            g["bottle_upright"] = st.orient["bottle"] == "upright"
        for n in ITEM_NAMES:
            it = S.item(n)
            if st.in_case(n) and it.heavy:
                g[f"heavy_on_floor({n})"] = st.stacks[st.region_of(n)][0] == n
            if st.in_case(n) and S.FAMILIES[it.family]["fragile"]:
                g[f"nothing_on({n})"] = st.stacks[st.region_of(n)][-1] == n
        g["lid_closed"] = st.lid == "closed"
        g["zipped"] = st.zipped
        g["case_upright"] = st.case_pose == "upright_floor"
        return g

    def done(self):
        return all(self.goals().values())

    def snapshot(self):
        return copy.deepcopy(asdict(self.state))


# ---------------------------------------------------------------- text grounder
_SYN = {
    "t-shirts": "tshirts", "t-shirt": "tshirts", "tees": "tshirts", "shirts": "striped_shirts", "pants": "trousers",
    "jeans": "trousers", "jumpers": "sweatshirts", "hoodies": "sweatshirts", "sweater": "sweater_pile",
    "towels": "towel_stack", "washbag": "toiletry_bag", "wash bag": "toiletry_bag", "water bottle": "bottle",
    "journal": "notebook", "sweatpants": "gym_clothes", "socks": "gym_clothes",
}


def resolve(phrase, kinds=("item", "station", "region")):
    """Map a noun phrase to an item, station or region name (None if ambiguous or unknown)."""
    p = re.sub(r"^(the|a|an|my)\s+", "", phrase.strip().lower())
    cands = []
    if "item" in kinds:
        cands += [(it.name, [it.name.replace("_", " "), it.label.lower()]) for it in S.ITEMS]
    if "station" in kinds:
        cands += [(st.name, [st.name.replace("_", " "), st.label.lower().replace("the ", "")]) for st in S.STATIONS.values()]
    if "region" in kinds:
        cands += [(rg.name, [rg.name.replace("_", " "), rg.label.lower().replace("the ", "")]) for rg in S.REGIONS.values()]
    exact = [n for n, names in cands if p in names]
    if len(exact) == 1:
        return exact[0]
    if p in _SYN and "item" in kinds:
        return _SYN[p]
    part = [n for n, names in cands if any(p in x or x in p for x in names)]
    return part[0] if len(set(part)) == 1 else None


def parse(text):
    """'pick up the T-shirt pile' -> Action. Returns (Action | None, feedback)."""
    t = text.strip().lower().rstrip(".")
    pats = [
        (r"^(go|walk) to (.+)$", lambda m: ("go_to", m.group(2), ("item", "station"))),
        (r"^(pick up|grab|take) (.+?)( from .+)?$", lambda m: ("pick_up", m.group(2), ("item",))),
        (r"^take (.+) out( of the (suit)?case)?$", lambda m: ("take_out", m.group(1), ("item",))),
        (r"^(put|place) (it|.+) (in|into) (.+?)(, (upright|lying))?$", lambda m: ("put_in", m.group(4), ("region",), m.group(6) or "")),
        (r"^(put|place) (it|.+) back.*$", lambda m: ("put_back", "", ())),
        (r"^close (the )?lid$", lambda m: ("close_lid", "", ())), (r"^open (the )?lid$", lambda m: ("open_lid", "", ())),
        (r"^zip.*$", lambda m: ("zip", "", ())), (r"^unzip.*$", lambda m: ("unzip", "", ())),
        (r"^stand (the case|it) up.*$", lambda m: ("stand_up", "", ())),
        (r"^(wait|pause)( for)? ([0-9.]+).*$", lambda m: ("wait", "", (), m.group(3))),
    ]
    for pat, f in pats:
        m = re.match(pat, t)
        if not m:
            continue
        r = f(m)
        verb, phrase, kinds = r[0], r[1], r[2]
        arg = r[3] if len(r) > 3 else ""
        if not kinds:
            return Action(verb, "", arg), ""
        name = resolve(phrase, kinds)
        if name is None:
            return None, f"cannot tell what '{phrase}' refers to"
        return Action(verb, name, arg), ""
    return None, f"no action matches '{text}'"
