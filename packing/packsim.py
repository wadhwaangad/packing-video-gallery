"""PackSim: the symbolic layer of the packing simulator.

It holds the task-level world state, checks each grounded action's preconditions
(returning feedback instead of silently changing the world), applies effects, and
evaluates goal predicates. It has no physics: the executor (executor.py) grounds each
accepted action in the MuJoCo simulation and commits the effect here only after the
physical outcome has been verified.

The action space itself (verbs, typed arguments, asset parts, reference resolution)
is defined in actionspace.py; where things fit in the case in layout.py.
"""
from dataclasses import dataclass, field, asdict
import copy
import numpy as np
import scene as S
import layout as LO
import rooms as R
from actionspace import Action, VERBS, SUITCASE_PARTS, ground, Rejection, describe

ITEM_NAMES = [it.name for it in S.ITEMS]

# Station each case action is performed from (while the case lies on the bed).
CASE_STATION = {"open_lid": "bed_foot", "close_lid": "bed_foot", "stand_up": "bed_foot", "set_trolley": "bed_foot",
                "zip": "suitcase", "lock": "suitcase", "strap": "suitcase"}


# ---------------------------------------------------------------- tasks
@dataclass
class Task:
    name: str
    title: str
    pack: tuple                      # items that must end up in the case
    note: str = ""
    lock: str = ""                   # if set: finish by docking the zip pulls and scrambling the dials


TASKS = {t.name: t for t in [
    Task("weekend_trip", "Weekend away", ("trousers", "tshirts", "sweatshirts", "towel", "toiletry_bag", "bottle",
                                         "notebook", "camera"), "the reference task shown on the page"),
    Task("gym_weekend", "Gym weekend", ("gym_clothes", "tshirts", "towel", "toiletry_bag", "bottle")),
    Task("work_trip", "Work trip", ("trousers", "striped_shirts", "toiletry_bag", "notebook", "bottle"),
         "locked before it leaves: pulls docked, dials scrambled", lock="yes"),
    Task("beach_holiday", "Beach holiday", ("tshirts", "striped_shirts", "towel_stack", "toiletry_bag", "bottle", "camera")),
    Task("winter_visit", "Winter visit", ("sweater_pile", "sweatshirts", "trousers", "toiletry_bag", "bottle", "notebook")),
]}


# ---------------------------------------------------------------- state
@dataclass
class State:
    station: str = "suitcase"
    facing: str = ""                                  # item the person stands in front of
    holding: str = ""
    loc: dict = field(default_factory=dict)           # item -> "station:<name>" | "held" | "case"
    spot: dict = field(default_factory=dict)          # item set down away from home -> its spot (scene.SET_DOWN)
    place: dict = field(default_factory=dict)         # item in the case -> x, y, yaw, z, orient, on
    orient: dict = field(default_factory=dict)        # bottle -> upright | lying
    lid: str = "open"
    zip_open: float = 1.0                             # opening of the main zip: 1 fully open, 0 zipped shut
    pulls: str = "free"                               # zip pulls: free | docked (in the lock)
    dials: list = field(default_factory=lambda: [0, 0, 0])
    code: str = "000"                                 # the lock's combination (factory setting)
    trolley: int = 0                                  # telescopic handle stop
    strap: str = "fastened"                           # lid strap buckle: fastened | open
    case_pose: str = "on_bed"                         # on_bed | upright_floor
    case_at: str = "bed"                              # where the case is: bed | bed_foot | bedroom_door
    pack: tuple = ()                                  # current goal list (plan changes may edit it)
    cloth: dict = field(default_factory=dict)         # loose cloth in the scene -> flat | folded_half

    # derived
    @property
    def zipped(self):
        return self.zip_open <= 1e-6

    @property
    def locked(self):
        return self.pulls == "docked" and "".join(map(str, self.dials)) != self.code

    def where(self, item):
        return self.loc[item]

    def in_case(self, item):
        return self.loc[item] == "case"

    def on_top_of(self, item):
        return LO.nothing_on(self.place, item)

    def top_height(self):
        return max([p["z"] + LO.dims(n, p.get("orient", ""))[2] for n, p in self.place.items()] + [LO.FLOOR])


def initial_state(task):
    st = State(pack=tuple(TASKS[task].pack))
    st.loc = {it.name: f"station:{it.station}" for it in S.ITEMS}
    st.orient = {"bottle": "upright"}
    return st


def _home(n):
    return S.item(n).station


def item_station(st, n):
    """The station an item lies on (None if held or in the case)."""
    return st.loc[n].split(":", 1)[1] if st.loc[n].startswith("station:") else None


def item_pose(st, n):
    """(x, y, z of its centre, yaw) of an item lying on a station: its home spot, or the
    spot it was set down on."""
    stn, it = item_station(st, n), S.item(n)
    if n not in st.spot:
        return (*it.start, it.start_yaw)
    x, y = S.SET_DOWN[stn][st.spot[n]]
    return (x, y, S.STATIONS[stn].top + it.size[2] / 2, S.set_down_yaw(n, stn))


def lying_on(st, station, but=""):
    """(name, xy, yaw) of every item lying on `station` now."""
    out = []
    for o in ITEM_NAMES:
        if o != but and item_station(st, o) == station:
            x, y, _, yaw = item_pose(st, o)
            out.append((o, (x, y), yaw))
    return out


def set_down_spot(st, n, station):
    """Where on `station` the item would be set down now (None: no room)."""
    return S.set_down_spot(n, station, lying_on(st, station, but=n))


def home_blocked_by(st, n):
    """What was set down where `n` goes back (or beside it, in the way of the hands): None if nothing."""
    it = S.item(n)
    for o, xy, yaw in lying_on(st, it.station, but=n):
        if o in st.spot and not S.room_at(n, it.xy, it.start_yaw, it.station, [(o, xy, yaw)], keep_out=False):
            return o
    return None


class PackSim:
    """Legality, effects and goals over State. `check` never changes the world."""

    def __init__(self, task="weekend_trip", cloth=False):
        self.task = task
        self.state = initial_state(task)
        if cloth:
            self.state.cloth = {S.CLOTH["name"]: "flat"}
        self.log = []

    # ------------------------------------------------------------ placement
    def place_pose(self, a):
        """(x, y, yaw, orient, Fit) for a place-in-case action, or (None, reason, code)."""
        st = self.state
        item = a.target or st.holding
        orient = a.get("orient", "") or ("upright" if item == "bottle" else "")
        if orient == "flat":
            orient = ""
        rel, dest = a.get("rel", "in"), a.get("dest", "suitcase")
        yaw = np.radians(a.get("yaw")) if a.get("yaw") is not None else None
        at = a.get("at")
        if rel == "in" and at is None:
            at = LO.choose_spot(st.place, item)
        try:
            if rel in ("on", "beside"):
                if dest not in st.place:
                    return None, f"the {S.item(dest).label} is not in the case", "precondition"
                x, y, yw = LO.resolve_spot(st.place, item, rel=rel, ref=dest, side=a.get("side"), yaw=yaw, orient=orient)
            else:
                x, y, yw = LO.resolve_spot(st.place, item, at=at, yaw=yaw, orient=orient)
        except ValueError as e:
            return None, e.args[1], e.args[0]
        if rel in ("on", "beside") or isinstance(at, str):
            # a relation or a named spot is approximate: take the nearest pose that fits
            nx, ny, fit = LO.nudge(st.place, item, x, y, yw, orient, keep_on=dest if rel == "on" else None)
            if nx is None:
                return None, fit.reason, fit.code
            x, y = nx, ny
        else:
            fit = LO.evaluate(st.place, item, x, y, yw, orient)
            if not fit.ok:
                return None, fit.reason, fit.code
        if rel == "on" and dest not in fit.on:
            return None, f"it would not rest on the {S.item(dest).label} there (it would rest on " \
                         f"{', '.join(S.item(o).label for o in fit.on) or 'the case floor'})", "unstable"
        return (x, y, yw, orient, fit), "", ""

    # ------------------------------------------------------------ preconditions
    def check(self, a):
        st = self.state
        lab = lambda n: S.item(n).label
        v = a.verb
        if v not in VERBS:
            return False, f"unknown action '{v}'"
        upright = st.case_pose == "upright_floor"
        near_case = st.station in ("suitcase", "bed_foot") if not upright else st.station == "case_floor"
        if v == "go_to":
            t = a.target
            if t in LO.ANCHORS:
                return True, ""
            if t in R.LABEL:
                if place_room(st) == t:
                    return False, f"you are already in {R.LABEL[t]}"
                return False, f"{R.LABEL[t]} is reached room by room (enter each room on the way)"
            if t not in S.STATIONS and t not in ITEM_NAMES:
                return False, f"cannot find '{t}'"
            if t in ITEM_NAMES and st.loc[t] == "held":
                return False, f"you are already holding the {lab(t)}"
            here_t = ("case_floor" if upright else "suitcase") if t == "suitcase" else t
            if (t in ITEM_NAMES and st.facing == t) or (t not in ITEM_NAMES and st.station == here_t and not st.facing):
                return False, f"you are already at {_the_any(t)}"
            ok, _ = in_view(st, t)
            if not ok:
                there = R.LABEL[R.room_at(target_point(st, t)[:2])]
                name = _the_any(t)
                return False, f"{name} is in {there}, out of view from {place_label(st)} " \
                              f"(you are in {R.LABEL[place_room(st)]}): go there room by room"
            return True, ""
        if v == "enter":
            if a.target not in R.LABEL:
                return False, f"'{a.target}' is not a room"
            here = place_room(st)
            if a.target == here:
                return False, f"you are already in {R.LABEL[here]}"
            if not R.adjacent(here, a.target):
                return False, f"{R.LABEL[a.target]} does not open off {R.LABEL[here]}"
            return True, ""
        if v == "look_at":
            if a.target in ITEM_NAMES and st.loc[a.target] == "held":
                return False, f"you are holding the {lab(a.target)}"
            ok, _ = in_view(st, a.target)
            if not ok:
                return False, f"{_the_any(a.target)} is out of view from {place_label(st)}"
            return True, ""
        if v == "pick_up":
            t = a.target
            if st.holding:
                return False, f"your hands are full (holding the {lab(st.holding)})"
            if st.in_case(t):
                if upright or st.lid != "open":
                    return False, "the case is closed"
                if st.station != "suitcase":
                    return False, "you are not at the suitcase"
                top = st.on_top_of(t)
                if top:
                    return False, f"the {lab(top[0])} is on top of it"
                if t == "bottle" and st.orient["bottle"] != "upright":
                    return False, "a lying bottle cannot be gripped from above in the case: it is wedged against the wall"
                return True, ""
            home = st.loc[t].split(":", 1)[1]
            if st.station != home or st.facing != t:
                return False, f"the {lab(t)} is not within reach (you are at {place_label(st)})"
            return True, ""
        if v == "place":
            item = a.target or st.holding
            if not st.holding:
                return False, "you are not holding anything"
            if item != st.holding:
                return False, f"you are holding the {lab(st.holding)}, not the {lab(item)}"
            dest = a.get("dest", "suitcase")
            if a.get("rel", "in") == "in" and dest in S.STATIONS and dest != "suitcase":
                if st.station != dest:
                    return False, f"you are not at {S.STATIONS[dest].label}"
                if dest == _home(item):
                    return True, ""
                if dest not in S.SURFACES:
                    return False, f"nothing can be set down at {S.STATIONS[dest].label}"
                if set_down_spot(st, item, dest) is None:
                    return False, f"there is no room for the {lab(item)} on {S.STATIONS[dest].label}"
                return True, ""
            if st.station != "suitcase":
                return False, "you are not at the suitcase"
            if upright or st.lid != "open":
                return False, "the lid is closed"
            if a.get("orient") in ("upright", "lying") and item != "bottle":
                return False, "only the bottle can be stood upright or laid down"
            r, msg, code = self.place_pose(a)
            if r is None:
                return False, msg
            return True, ""
        if v == "put_back":
            if not st.holding:
                return False, "you are not holding anything"
            home = _home(st.holding)
            if st.station != home:
                return False, f"the {lab(st.holding)} belongs at {S.STATIONS[home].label}"
            o = home_blocked_by(st, st.holding)
            if o:
                return False, f"the {lab(o)} is in the way where the {lab(st.holding)} goes back: move it first"
            return True, ""
        if v in ("wait", "say"):
            return True, ""
        if v in ("fold", "unfold"):
            if a.target not in st.cloth:
                return False, f"there is no loose {S.CLOTH['label']} in this scene"
            if st.holding:
                return False, "put down what you are holding first"
            if st.station != S.CLOTH["station"]:
                return False, f"go to {S.STATIONS[S.CLOTH['station']].label} first"
            if v == "fold" and st.cloth[a.target] != "flat":
                return False, f"the {S.CLOTH['label']} is already folded"
            return True, ""
        # -------- suitcase parts
        if st.holding and v not in ("roll",):
            return False, "put down what you are holding first"
        if v == "roll":
            if st.holding:
                return False, "put down what you are holding first"
            if not upright:
                return False, "stand the case up on its wheels first"
            if st.trolley < 1:
                return False, "extend the trolley handle first"
            if st.station != "case_floor":
                return False, "go to the suitcase first"
            if a.get("to") == st.case_at:
                return False, "the case is already there"
            if a.get("to") not in FLOOR_SPOTS:
                return False, f"the case can be rolled to: {', '.join(FLOOR_SPOTS)}"
            return True, ""
        t = a.target
        if v in ("stand_up", "lay_down"):
            if v == "stand_up":
                if upright:
                    return False, "the case is already standing"
                if st.station != "bed_foot":
                    return False, "do that from the foot of the bed"
                if not st.zipped:
                    return False, "zip the case before standing it up"
                return True, ""
            if not upright:
                return False, "the case is already lying down"
            if st.case_at != "bed_foot":
                return False, "roll the case back to the foot of the bed first"
            if st.station != "case_floor":
                return False, "go to the suitcase first"
            if st.trolley != 0:
                return False, "retract the trolley handle first"
            return True, ""
        if v in ("open", "close") and t == "lid":
            if upright:
                return False, "lay the case down on the bed first"
            if st.station != "bed_foot":
                return False, "do that from the foot of the bed"
            if v == "open":
                if st.lid == "open":
                    return False, "the lid is already open"
                if st.zip_open < 0.95:
                    return False, "unzip the case first" if st.zip_open < 0.05 else f"the zip is still {1 - st.zip_open:.0%} closed"
                return True, ""
            if st.lid == "closed":
                return False, "the lid is already closed"
            if st.top_height() > LO.TOP:
                return False, "the case is too full to close"
            return True, ""
        if v in ("open", "close") and t == "main_zip":
            to = a.get("to")
            if upright:
                return False, "lay the case down on the bed first"
            if st.station != "suitcase":
                return False, "go to the suitcase first"
            if st.lid != "closed":
                return False, "close the lid first"
            if st.pulls == "docked":
                return False, "the zip pulls are locked in the lock" if st.locked else \
                    "the zip pulls are in the lock: press the lock's release button"
            # a zip runs to a position either way, as a real one does
            if abs(to - st.zip_open) < 1e-6:
                return False, f"the zip is already {1 - st.zip_open:.0%} closed"
            if st.strap != "fastened" and False:
                return False, ""
            return True, ""
        if v == "insert":
            if st.station not in ("suitcase", "case_floor"):
                return False, "go to the suitcase first"
            if not st.zipped:
                return False, "zip the case shut first: the pulls meet at the lock only when it is closed"
            if st.pulls == "docked":
                return False, "the zip pulls are already in the lock"
            return True, ""
        if v in ("turn", "set") and t.startswith(("dial_", "lock")):
            if st.station not in ("suitcase", "case_floor"):
                return False, "go to the suitcase first"
            if st.lid != "closed":
                return False, "the lock is on the lid's side wall, which lies face down while the case is open"
            return True, ""
        if v == "press" and t == "lock_button":
            if st.station not in ("suitcase", "case_floor"):
                return False, "go to the suitcase first"
            if st.lid != "closed":
                return False, "the lock is on the lid's side wall, which lies face down while the case is open"
            return True, ""
        if (v == "set" and t == "trolley_handle") or (v == "press" and t == "trolley_button"):
            if upright:
                if st.station != "case_floor":
                    return False, "go to the suitcase first"
            elif st.station != "bed_foot":
                return False, "do that from the foot of the bed"
            if v == "set" and a.get("value") == st.trolley:
                return False, f"the handle is already at stop {st.trolley}"
            if v == "set" and not upright and a.get("value") > 1:
                return False, "with the case lying on the bed the handle can only be pulled out to stop 1 " \
                              "(it would hit the end of the bed frame)"
            return True, ""
        if v in ("fasten", "unfasten"):
            if st.lid != "closed":
                return False, "the strap is on the lid's front, which lies face down while the case is open"
            if upright:
                return False, "lay the case down on the bed first"
            if st.station != "suitcase":
                return False, "go to the suitcase first"
            if v == "fasten" and st.strap == "fastened":
                return False, "the strap is already fastened"
            if v == "unfasten" and st.strap != "fastened":
                return False, "the strap is already open"
            return True, ""
        if v in ("fold", "unfold"):
            if a.target not in st.cloth:
                return False, f"there is no loose {S.CLOTH['label']} in this scene"
            if st.holding:
                return False, "put down what you are holding first"
            if st.station != S.CLOTH["station"]:
                return False, f"go to {S.STATIONS[S.CLOTH['station']].label} first"
            if v == "fold" and st.cloth[a.target] != "flat":
                return False, f"the {S.CLOTH['label']} is already folded"
            return True, ""
        return False, f"{describe(a)}: not possible here"

    # ------------------------------------------------------------ effects
    def apply(self, a, pose=None):
        """Commit an action. `pose` (x, y, yaw, orient, z, on) is the placement as measured
        by the executor; without it the planned pose is used."""
        st = self.state
        v, t = a.verb, a.target
        if v == "go_to":
            if t in LO.ANCHORS or t == "suitcase":
                st.station = "case_floor" if st.case_pose == "upright_floor" else "suitcase"
                st.facing = ""
            elif t in ITEM_NAMES:
                where = st.loc[t]
                st.station = "suitcase" if where == "case" else where.split(":", 1)[1]
                st.facing = "" if where == "case" else t
            else:
                st.station, st.facing = t, ""
        elif v == "enter":
            st.station, st.facing = f"node:{entry_node(st, t)}", ""
        elif v == "pick_up":
            if st.in_case(t):
                st.place.pop(t, None)
            st.spot.pop(t, None)
            st.loc[t] = "held"; st.holding = t
        elif v == "place":
            item = st.holding
            dest = a.get("dest", "suitcase")
            if a.get("rel", "in") == "in" and dest in S.STATIONS and dest != "suitcase":
                if dest != _home(item):
                    st.spot[item] = set_down_spot(st, item, dest)
                st.loc[item] = f"station:{dest}"
                if item == "bottle":
                    st.orient["bottle"] = "upright"
                st.facing = item
            else:
                if pose is None:
                    r, _, _ = self.place_pose(a)
                    x, y, yw, orient, fit = r
                    pose = dict(x=x, y=y, yaw=yw, orient=orient, z=fit.z, on=fit.on)
                st.place[item] = dict(pose)
                st.loc[item] = "case"
                if item == "bottle":
                    st.orient["bottle"] = pose.get("orient") or "upright"
            st.holding = ""
        elif v == "put_back":
            st.loc[st.holding] = f"station:{_home(st.holding)}"
            st.spot.pop(st.holding, None)
            if st.holding == "bottle":
                st.orient["bottle"] = "upright"
            st.facing = st.holding; st.holding = ""
        elif v in ("open", "close") and t == "lid":
            st.lid = "open" if v == "open" else "closed"
        elif v in ("open", "close") and t == "main_zip":
            st.zip_open = float(a.get("to"))
        elif v == "insert":
            st.pulls = "docked"
        elif v == "turn":
            k = int(t[-1]) - 1
            st.dials[k] = a.get("to") if a.get("to") is not None else (st.dials[k] + a.get("by")) % 10
        elif v == "set" and t == "lock":
            st.dials = [int(c) for c in a.get("value")]
        elif v == "press" and t == "lock_button":
            if st.pulls == "docked" and not st.locked:
                st.pulls = "free"
        elif v == "set" and t == "trolley_handle":
            st.trolley = int(a.get("value"))
        elif v == "fasten":
            st.strap = "fastened"
        elif v == "unfasten":
            st.strap = "open"
        elif v == "stand_up":
            st.case_pose = "upright_floor"; st.case_at = "bed_foot"; st.station = "case_floor"
        elif v == "lay_down":
            st.case_pose = "on_bed"; st.case_at = "bed"; st.station = "bed_foot"
        elif v == "roll":
            st.case_at = a.get("to"); st.station = "case_floor"
        elif v == "fold":
            st.cloth[t] = "folded_" + a.get("style", "half")

    def feedback(self, a):
        """What the person notices after an accepted action that changes nothing."""
        st = self.state
        if a.verb == "press" and a.target == "lock_button":
            if st.pulls != "docked":
                return "nothing is in the lock"
            if st.locked:
                return "the button does not move: the dials do not show the code"
        if a.verb == "press" and a.target == "trolley_button":
            return "the handle is free to move only while the button is held"
        return ""

    def step(self, a, pose=None):
        ok, msg = self.check(a)
        self.log.append(dict(action=a.to_dict(), ok=ok, feedback=msg))
        if ok:
            self.apply(a, pose)
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
                g[f"heavy_on_floor({n})"] = not st.place[n].get("on")
            if st.in_case(n) and S.FAMILIES[it.family]["fragile"]:
                g[f"nothing_on({n})"] = not st.on_top_of(n)
        g["lid_closed"] = st.lid == "closed"
        g["zipped"] = st.zipped
        g["case_upright"] = st.case_pose == "upright_floor"
        task = TASKS[self.task]
        if task.lock:
            g["locked"] = st.locked
        return g

    def done(self):
        return all(self.goals().values())

    def snapshot(self):
        d = copy.deepcopy(asdict(self.state))
        d["zipped"] = self.state.zipped
        d["locked"] = self.state.locked
        d.pop("code", None)                  # the combination is private to the person
        return d


# Floor spots the upright case can be rolled to. The person walks beside the case with
# the left hand on the handle (the case rolls alongside, as spinner manuals recommend):
# `stand` is where they take the handle, `route` their path, `case` where the case ends.
FLOOR_SPOTS = {
    # case: where the standing case is; beside / beside_yaw: where the person takes its
    # handle (facing the way out); route: the person's path when rolling the case here
    "bed_foot": dict(label="the foot of the bed", case=(2.55, 1.26), beside=(2.86, 1.30), beside_yaw=np.pi / 2,
                     route=((3.0, 2.2), (2.86, 1.30))),
    "bedroom_door": dict(label="the bedroom door", case=(3.05, 2.95), beside=(3.35, 2.95), beside_yaw=-np.pi / 2,
                         route=((2.98, 2.05), (3.35, 2.95))),
}


def _the_any(t):
    if t in ITEM_NAMES:
        return f"the {S.item(t).label}"
    if t in S.STATIONS:
        return S.STATIONS[t].label
    return "the suitcase"


# ---------------------------------------------------------------- navigation
# A go_to reaches only what is in the person's room or in view; anything else is
# reached room by room (enter), then looked for, then walked to. See rooms.py.
def place_xy(st):
    """Where the person stands now (x, y)."""
    s = st.station
    if s.startswith("node:"):
        return tuple(S.NAV_NODES[s[5:]])
    if s == "case_floor":
        return FLOOR_SPOTS[st.case_at]["beside"]
    return tuple(S.STATIONS[s].stand)


def place_room(st):
    return R.room_at(place_xy(st))


def place_label(st):
    s = st.station
    if s.startswith("node:"):
        return R.LABEL[place_room(st)]
    if s == "case_floor":
        return f"the standing case at {FLOOR_SPOTS[st.case_at]['label']}"
    return S.STATIONS[s].label


def access_node(st):
    s = st.station
    if s.startswith("node:"):
        return s[5:]
    if s == "case_floor":
        return R.nearest_node(FLOOR_SPOTS[st.case_at]["beside"])
    return S.STATIONS[s].access


def _case_point(st):
    if st.case_pose == "upright_floor":
        x, y = FLOOR_SPOTS[st.case_at]["case"]
        return np.array([x, y, 0.40])
    stn = S.STATIONS["suitcase"]
    return np.array([*stn.stand, 0.0]) + np.array([*stn.normal, 0.0]) * 0.6 + np.array([0, 0, stn.top + 0.1])


def target_point(st, t):
    """A point on the target, for line of sight (None for rooms and held items)."""
    if t in R.LABEL:
        return None
    if t == "suitcase" or t in LO.ANCHORS or (t in ITEM_NAMES and st.in_case(t)):
        return _case_point(st)
    if t in ITEM_NAMES:
        if st.loc[t] == "held":
            return None
        return np.array(item_pose(st, t)[:3], float)
    stn = S.STATIONS[t]
    return np.array([*stn.stand, 0.0]) + np.array([*stn.normal, 0.0]) * 0.6 + np.array([0, 0, max(stn.top, 0.5)])


def target_access(st, t):
    if t in R.LABEL:
        return R.CENTRE[t]
    if t == "suitcase" or t in LO.ANCHORS or (t in ITEM_NAMES and st.in_case(t)):
        if st.case_pose == "upright_floor":
            return R.nearest_node(FLOOR_SPOTS[st.case_at]["beside"])
        return S.STATIONS["suitcase"].access
    if t in ITEM_NAMES:
        return S.STATIONS[item_station(st, t) or _home(t)].access
    return S.STATIONS[t].access


def in_view(st, t, frm=None):
    """(ok, why): the target is in the person's room, or visible from where they stand."""
    p = target_point(st, t)
    if p is None:
        return True, ""
    xy = place_xy(st) if frm is None else frm
    if R.room_at(xy) == R.room_at(p[:2]):
        return True, "same room"
    vis, _ = R.line_of_sight((xy[0], xy[1], R.EYE_H), p)
    return vis, ("in view" if vis else "")


def entry_node(st, room):
    """Where enter(room) ends: the point of that room nearest along the walking graph."""
    return R.entry_node(access_node(st), room)


def navigation(st, a):
    """A go_to / enter request as the steps a person can take from here: [a] if its target
    is in this room or in view; otherwise one enter(room) per room on the way until the
    target is in view (or its room is reached), then look_at(item) for an item that has to
    be found, then go_to(target). A go_to(room) is the enters alone ([] if already there).
    Each step carries the request it came from (`parent`)."""
    t = a.target
    here = place_room(st)
    if a.verb == "enter" or t in R.LABEL:
        if here == t:
            return []
        goal_room, item = t, None
    else:
        if in_view(st, t)[0]:
            return [a]
        goal_room, item = R.room_at(target_point(st, t)[:2]), t
    # rooms on the way: shortest path over the rooms' doorways (adjacency), so a step never
    # skips a room; enter(room) walks to entry_node(room) from wherever the person is
    steps, node = [], access_node(st)
    for rn in R.room_path(here, goal_room)[1:]:
        node = R.entry_node(node, rn)
        steps.append(Action("enter", rn))
        if rn == goal_room or (item and in_view(st, item, frm=S.NAV_NODES[node])[0]):
            break
    if item:
        if item in ITEM_NAMES and not st.in_case(item):
            steps.append(Action("look_at", item))
        steps.append(Action("go_to", item))
    parent = a.text()
    for s in steps:
        s.parent = parent
    return steps


# ---------------------------------------------------------------- compatibility helpers
def act(verb, target="", **args):
    return Action(verb, target, args)
