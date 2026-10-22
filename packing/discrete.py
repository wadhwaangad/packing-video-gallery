"""The discrete action space: a closed, finite catalogue of fully specified actions.

Every action the simulator runs is one entry of CATALOGUE, named by an id such as
`pick_up(trousers)`, `place_on(tshirts, trousers)`, `zip_to(50)` or `turn_dial(dial_2, 7)`.
An id has no free words, no numbers outside an enumerated set, no optional arguments
and no defaults, so an action is never ambiguous and never out of range. What is in
the catalogue is what a person can really do with these things and what the
simulation carries out; what is real but not (yet) reliable in the simulation is
left out with the reason (EXCLUDED). The rules:

  * one id, one skill, one effect: no verb covers two parts (open_lid / open_zip), and
    no effect can be reached by two different ids (a lock code is three turn_dial ids;
    there is no set-code action);
  * every argument is drawn from a finite list fixed by the scene: rooms, stations,
    items, the five named spots in the case, dial digits, handle stops, floor spots;
  * a continuous quantity a person really varies is sampled at steps a person would
    use: the zip at 0, 25, 50, 75, 100 % closed;
  * nothing that is not simulated reliably is in the catalogue (EXCLUDED);
  * legal(sim) lists the ids whose preconditions hold now (same room or in view for
    go_to, adjacent for enter, hands free, lid open ...). An agent that picks from
    that list cannot make a grounding or precondition error; only the physical
    outcome can still fail, and that is measured.

Language and the event generator produce actions through `canonical(a, state)`,
which turns a deterministic paraphrase into catalogue entries (a code into three dial
turns, clicks into a digit, a long pause into 2 s waits, "beside" with no side into
the first side that fits) or rejects it with the reason (code `not_in_action_space`):
a zip position between the steps, a spot given in centimetres, an item turned by an
angle. Ambiguous words are resolved (or returned with their candidates) by the
grounder in actionspace.py before this step.
"""
import math
import scene as S
import layout as LO
import rooms as R
from actionspace import Action, Rejection

ITEMS = [it.name for it in S.ITEMS]
SPOTS = list(LO.ANCHORS)                         # named spots in the case
STATIONS = list(S.STATIONS)
DIALS = ["dial_1", "dial_2", "dial_3"]
HANDLE_STOPS = [0, 1, 2]
FLOOR = ["bed_foot", "bedroom_door"]             # packsim.FLOOR_SPOTS
SIDES = ["wheel_end", "handle_end", "front_side", "hinge_side"]
ZIP_CLOSED = [0, 25, 50, 75, 100]                # % of the track zipped shut

# Real actions that are not in the catalogue, and why.
EXCLUDED = {
    "lay the bottle down": "the lying placement is not reliable with the scanned case yet",
    "lay the case down": "not simulated yet",
    "unfold the towel": "neither the scoop nor a pinch gets under the folded edge with this hand model",
    "fold in thirds": "only folding in half is simulated",
    "carry the case by its side handle": "the handle has a grip but no skill yet",
    "change the lock's combination": "needs a pen in the reset hole; there is no pen in the scene",
}
WAIT_S = 2.0
CLOTH = S.CLOTH["name"]


def ask_text(item):
    return f"Should the {S.item(item).label} go in too?"


def announce_text(change, item):
    return f"I'll take the {S.item(item).label} as well." if change == "add" else \
        f"Actually, I'll leave the {S.item(item).label} at home."


def _entries():
    """(id, group, Action, what it does) for every action."""
    E = []
    add = lambda i, g, a, d: E.append((i, g, a, d))
    for r in R.LABEL:
        add(f"enter({r})", "navigate", Action("enter", r), f"walk into {R.LABEL[r]} from a room it opens off")
    for s in STATIONS:
        add(f"go_to({s})", "navigate", Action("go_to", s), f"walk to {S.STATIONS[s].label} (in this room or in view)")
    for n in ITEMS:
        add(f"go_to({n})", "navigate", Action("go_to", n), f"walk to the {S.item(n).label} where it lies, and face it")
    for n in ITEMS:
        add(f"look_at({n})", "navigate", Action("look_at", n), f"turn the head to the {S.item(n).label}")
    for n in ITEMS:
        add(f"pick_up({n})", "manipulate", Action("pick_up", n), f"pick up the {S.item(n).label}")
    for n in ITEMS:
        for p in SPOTS:
            args = dict(rel="in", dest="suitcase", at=p)
            if n == "bottle":
                args["orient"] = "upright"
            add(f"place({n}, {p})", "manipulate", Action("place", n, args),
                f"put the {S.item(n).label} in {LO.ANCHORS[p]['label']}" + (", upright" if n == "bottle" else ""))
    for n in ITEMS:
        for o in ITEMS:
            if o != n:
                args = dict(rel="on", dest=o, **({"orient": "upright"} if n == "bottle" else {}))
                add(f"place_on({n}, {o})", "manipulate", Action("place", n, args),
                    f"put the {S.item(n).label} on top of the {S.item(o).label} in the case")
    for n in ITEMS:
        for o in ITEMS:
            if o != n:
                for sd in SIDES:
                    args = dict(rel="beside", dest=o, side=sd, **({"orient": "upright"} if n == "bottle" else {}))
                    add(f"place_beside({n}, {o}, {sd})", "manipulate", Action("place", n, args),
                        f"put the {S.item(n).label} beside the {S.item(o).label}, on its {sd.replace('_', ' ')} side")
    for n in ITEMS:
        add(f"put_back({n})", "manipulate", Action("put_back", n), f"put the {S.item(n).label} back where it was")
    add("open_lid", "case", Action("open", "lid", dict(to=1.0)), "swing the lid fully open")
    add("close_lid", "case", Action("close", "lid", dict(to=0.0)), "swing the lid shut")
    for c in ZIP_CLOSED:
        add(f"zip_to({c})", "case", Action("close", "main_zip", dict(to=round(1 - c / 100, 2))),
            "unzip the case all the way" if c == 0 else "zip the case shut" if c == 100 else
            f"run the zip to {c}% closed (either way)")
    add("dock_zip_pulls", "case", Action("insert", "zip_pulls", dict(into="lock")), "push both zip pulls into the lock")
    add("press_lock_button", "case", Action("press", "lock_button", dict(hold=0.3, times=1)),
        "press the lock's release button once")
    for d in DIALS:
        for k in range(10):
            add(f"turn_dial({d}, {k})", "case", Action("turn", d, dict(to=k)), f"turn {d.replace('_', ' ')} to {k}")
    add("press_handle_button", "case", Action("press", "trolley_button", dict(hold=0.3, times=1)),
        "press the button in the trolley handle's grip (frees the handle while held)")
    add("fasten_strap", "case", Action("fasten", "strap"), "click the strap buckle shut")
    add("unfasten_strap", "case", Action("unfasten", "strap"), "release the strap buckle")
    for k in HANDLE_STOPS:
        add(f"set_handle({k})", "case", Action("set", "trolley_handle", dict(value=k)),
            "retract the trolley handle" if k == 0 else f"pull the trolley handle out to stop {k}")
    add("stand_up_case", "case", Action("stand_up", "suitcase"), "stand the case up on its wheels")
    for f in FLOOR:
        add(f"roll_case({f})", "case", Action("roll", "suitcase", dict(to=f)), f"roll the standing case to {f.replace('_', ' ')}")
    add(f"fold({CLOTH})", "cloth", Action("fold", CLOTH, dict(style="half")), "fold the loose towel in half")
    add("wait", "other", Action("wait", "", dict(seconds=WAIT_S)), f"pause for {WAIT_S:g} s")
    for n in ITEMS:
        add(f"ask({n})", "other", Action("say", "", dict(text=ask_text(n))), f'ask "{ask_text(n)}"')
    for c in ("add", "remove"):
        for n in ITEMS:
            add(f"announce({c}, {n})", "other", Action("say", "", dict(text=announce_text(c, n))),
                f'say "{announce_text(c, n)}"')
    return E


CATALOGUE = [dict(id=i, group=g, action=a, does=d) for i, g, a, d in _entries()]
BY_ID = {e["id"]: e for e in CATALOGUE}
assert len(BY_ID) == len(CATALOGUE)


def action(aid):
    """A fresh executor Action for a catalogue id."""
    a = BY_ID[aid]["action"]
    return Action(a.verb, a.target, dict(a.args))


def _key(a):
    args = {k: v for k, v in a.args.items() if k != "via"}
    return (a.verb, a.target, tuple(sorted((k, repr(v)) for k, v in args.items())))


_BY_KEY = {_key(e["action"]): e["id"] for e in CATALOGUE}


def to_id(a):
    """The catalogue id of an Action, or None if it is not exactly a catalogue entry."""
    if a.verb == "place" and a.target == "bottle" and "orient" not in a.args:
        a = Action(a.verb, a.target, dict(a.args, orient="upright"))
    if a.verb == "open" and a.target == "main_zip":      # the zip goes to a position, either way
        a = Action("close", "main_zip", dict(a.args))
    if a.target == "main_zip" and a.get("to") is not None:
        a = Action(a.verb, a.target, dict(a.args, to=round(float(a.get("to")), 2)))
    return _BY_KEY.get(_key(a))


def canonical(a, st):
    """A grounded Action as catalogue Actions (a list), or a Rejection saying why it has
    no place in the action space. Only deterministic rewrites are made."""
    v, t = a.verb, a.target
    # unset optional arguments are not arguments
    a = Action(v, t, {k: x for k, x in a.args.items() if x is not None and x != "" and k != "via"}, a.parent)
    if v == "place":
        a.args.setdefault("rel", "in"); a.args.setdefault("dest", "suitcase")
        if a.args.get("orient") == "flat":
            a.args.pop("orient")
    out = lambda *ids: [Action(**{**action(i).__dict__, "parent": a.parent}) for i in ids]
    if to_id(a):
        b = Action(a.verb, a.target, dict(a.args), a.parent)
        b.args.pop("via", None)
        if v == "place" and t == "bottle":
            b.args.setdefault("orient", "upright")
        return [b]
    no = lambda msg: Rejection("not_in_action_space", msg)
    if v == "go_to" and t in R.LABEL:
        return out(f"enter({t})")
    if v == "go_to" and t in LO.ANCHORS:
        return out("go_to(suitcase)")
    if v == "turn" and t in DIALS and a.get("by") is not None:
        return out(f"turn_dial({t}, {(st.dials[int(t[-1]) - 1] + int(a.get('by'))) % 10})")
    if v == "set" and t == "lock":
        code = str(a.get("value"))
        return out(*[f"turn_dial({d}, {c})" for d, c, now in zip(DIALS, code, st.dials) if int(c) != now]) or no(
            f"the dials already read {code}")
    if v == "wait":
        return out(*["wait"] * max(1, math.ceil(float(a.get("seconds", WAIT_S)) / WAIT_S - 1e-9)))
    if v in ("open", "close") and t == "main_zip":
        return no(f"the zip goes to {', '.join(map(str, ZIP_CLOSED))} % closed; "
                  f"{100 * (1 - float(a.get('to'))):.0f} % is between them")
    if v == "place":
        rel, dest, at = a.get("rel", "in"), a.get("dest", "suitcase"), a.get("at")
        if rel == "in" and dest in S.STATIONS and dest != "suitcase":
            return out(f"put_back({t or st.holding})") if dest == S.item(t or st.holding).station else \
                no(f"an item goes back only to where it came from (put_back)")
        if a.get("orient") == "lying":
            return no(f"laying the bottle down: {EXCLUDED['lay the bottle down']}")
        if a.get("yaw") is not None:
            return no("an item goes in the way it packs (squeezed side along the case); turning it by an angle "
                      "is not an action")
        if rel == "beside" and not a.get("side"):
            # no side given: the first side (in a fixed order) where it fits
            from packsim import PackSim
            sim = PackSim.__new__(PackSim); sim.state = st
            for sd in SIDES:
                b = Action("place", t or st.holding, dict(a.args, side=sd), a.parent)
                if sim.place_pose(b)[0] is not None:
                    return canonical(b, st)
            return no(f"there is no room beside the {S.item(dest).label} on any side")
        if rel in ("on", "beside"):
            return no(f"{rel} {dest}: only another item in the case")
        if at is None:
            # no spot named: the planner's rule picks one (deterministic, logged as the id)
            item = t or st.holding
            spot = LO.choose_named(st.place, item, a.get("orient", ""))
            if spot is None:
                return no(f"none of the named spots ({', '.join(SPOTS)}) has room for the {S.item(item).label}")
            b = Action("place", item, {**{k: v for k, v in a.args.items() if v is not None}, "at": spot}, a.parent)
            return canonical(b, st)
        if not isinstance(at, str):
            return no(f"a spot is named, not measured: {', '.join(SPOTS)}")
        return no("a spot is placed with the item's own orientation; turning or siding it is not an action")
    if v == "press" and t == "trolley_button":
        return out("press_handle_button")
    if v == "press":
        return out("press_lock_button")
    if v == "say":
        return no("speech is one of ask(item) / announce(add|remove, item)")
    if v == "lay_down":
        return no(f"laying the case down: {EXCLUDED['lay the case down']}")
    if v == "unfold":
        return no(f"unfolding: {EXCLUDED['unfold the towel']}")
    return no(f"no catalogue action for {a.text()}")


def legal(sim):
    """Ids whose preconditions hold in sim's state (sim.check never changes the world)."""
    st = sim.state
    ok = []
    for e in CATALOGUE:
        a = e["action"]
        # cheap filters before the full check
        if a.verb == "place" and a.target != st.holding:
            continue
        if a.verb in ("pick_up", "look_at", "go_to") and a.target in ITEMS and st.loc[a.target] == "held":
            continue
        if a.verb == "fold" and a.target not in st.cloth:
            continue
        if a.verb == "say":
            continue                 # speech is always possible; listed separately
        if sim.check(action(e["id"]))[0]:
            ok.append(e["id"])
    return ok


def spec():
    groups = {}
    for e in CATALOGUE:
        groups.setdefault(e["group"], []).append(e["id"])
    return dict(size=len(CATALOGUE), groups={g: len(v) for g, v in groups.items()},
                templates=_templates(), wait_seconds=WAIT_S, excluded=EXCLUDED)


def _templates():
    """The catalogue written compactly: one row per action family with its argument sets."""
    rows, seen = [], {}
    for e in CATALOGUE:
        fam = e["id"].split("(")[0]
        if fam not in seen:
            seen[fam] = dict(family=fam, group=e["group"], ids=[], example=e["does"])
            rows.append(seen[fam])
        seen[fam]["ids"].append(e["id"])
    return [dict(family=r["family"], group=r["group"], count=len(r["ids"]), ids=r["ids"], example=r["example"])
            for r in rows]


if __name__ == "__main__":
    from packsim import PackSim
    print(len(CATALOGUE), spec()["groups"])
    sim = PackSim("weekend_trip")
    print("legal at the start:", legal(sim))
