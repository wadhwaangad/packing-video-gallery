"""The action space: verbs with typed arguments, the operable parts of every asset, and
how a request is grounded (references resolved, values checked) before it is run.

An action is a verb applied to arguments, a = f(theta). No object, part, position or
amount is an action of its own: `set(trolley_handle, level=2)` and
`place(tshirts, in suitcase, at=(12 cm, 30 cm))` are one verb each with arguments.

Grounding a request (a dict {"verb": ..., "args": {...}} whose references are words)
returns either a resolved Action or a Rejection with a code:

  unknown_verb          no such verb
  missing_argument      a required argument is absent
  bad_value             a value does not parse as its type
  unknown_ref           nothing in the scene matches the words
  ambiguous_ref         several things match (the candidates are listed)
  wrong_kind            the reference resolves, but not to the kind the slot needs
  unsupported_operation the verb does not apply to that part of that asset
  unsupported_value     the value is the right type but this part cannot take it
  out_of_range          the value is outside the part's range

Preconditions that depend on the current state (hands full, lid closed, zip pulls
locked ...) are checked afterwards by PackSim.check and return feedback.
"""
from dataclasses import dataclass, field, asdict
import re
import numpy as np
import scene as S
import layout as LO
import rooms as R


# ---------------------------------------------------------------- action
@dataclass
class Action:
    verb: str
    target: str = ""              # resolved id of the main reference
    args: dict = field(default_factory=dict)
    parent: str = ""              # the request this step was expanded from (navigation)

    def get(self, k, default=None):
        return self.args.get(k, default)

    def text(self):
        return describe(self)

    def to_dict(self):
        d = dict(verb=self.verb, target=self.target, args=dict(self.args), text=self.text())
        if self.parent:
            d["parent"] = self.parent
        return d

    @staticmethod
    def from_dict(d):
        return Action(d["verb"], d.get("target", ""), dict(d.get("args", {})), d.get("parent", ""))


@dataclass
class Rejection:
    code: str
    message: str
    candidates: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- argument types
TYPES = {
    "Ref": "words naming something in the scene (an item, a place, a part or control of an asset), "
           "resolved against what is there now; 'it' is the item in hand",
    "Frac": "fraction of a part's travel, 0 to 1 ('halfway', '25%', 'fully')",
    "Level": "a stop of a telescopic handle, 0 (retracted) to its top stop",
    "Digit": "one lock-dial digit, 0 to 9",
    "Code": "a three-digit lock combination ('3-7-2')",
    "Extent": "number of detent clicks to turn a dial (positive = up)",
    "Dur": "seconds",
    "Count": "a positive whole number of repetitions",
    "Rel": "in | on | beside",
    "Side": "wheel_end | handle_end | front_side | hinge_side",
    "Spot": "where in the case: a named spot, or the item centre in cm from the inside of the wheel-end wall "
            "(along) and from the hinge-side wall (across)",
    "Angle": "degrees about the vertical; 0 = the item's long side across the case",
    "Orient": "upright | lying (the bottle); flat for everything else",
    "Style": "half (folding in thirds is not simulated: see the fold verb)",
    "Text": "free text",
}


# ---------------------------------------------------------------- assets
# Every asset lists its parts and, for each part, the operations it supports and their
# value ranges. The suitcase follows its scan (BlenderKit "Black suitcase") and the
# operations given in spinner-case manuals (Samsonite lock instructions, Samsonite and
# Delsey spinner manuals): telescopic handle with push-button stops, carry handle,
# four 360-degree spinner wheels, zipper whose two pulls dock in a 3-dial TSA
# combination lock, and the lengthwise compression strap with a side-release buckle.
SUITCASE_PARTS = {
    "lid": dict(kind="part", label="the lid", nouns=["lid", "top", "cover"], ops={
        "open": dict(to="Frac", values=[1.0], why="the lid only stays put closed or lying fully open"),
        "close": dict(to="Frac", values=[0.0], why="the lid only stays put closed or lying fully open")}),
    "main_zip": dict(kind="part", label="the main zip", nouns=["zip", "zipper", "zips", "zippers"], adjs=["main"], ops={
        "open": dict(to="Frac", range=(0.0, 1.0)), "close": dict(to="Frac", range=(0.0, 1.0))}),
    "zip_pulls": dict(kind="part", label="the zip pulls", nouns=["pulls", "pull", "tabs", "sliders"], adjs=["zip", "zipper"],
                      ops={"insert": dict(into="Ref")}),
    "lock": dict(kind="device", label="the combination lock", nouns=["lock", "padlock", "combination"],
                 adjs=["combination", "tsa", "number"], ops={"set": dict(value="Code")}),
    "dial_1": dict(kind="control", label="the left lock dial", nouns=["dial", "wheel", "digit"], adjs=["left", "first", "1st", "lock"],
                   ops={"turn": dict(to="Digit", by="Extent")}),
    "dial_2": dict(kind="control", label="the middle lock dial", nouns=["dial", "wheel", "digit"], adjs=["middle", "second", "2nd", "lock"],
                   ops={"turn": dict(to="Digit", by="Extent")}),
    "dial_3": dict(kind="control", label="the right lock dial", nouns=["dial", "wheel", "digit"], adjs=["right", "third", "3rd", "lock"],
                   ops={"turn": dict(to="Digit", by="Extent")}),
    "lock_button": dict(kind="control", label="the lock's release button", nouns=["button", "release", "slider"],
                        adjs=["lock", "release", "tsa", "open"], ops={"press": dict(hold="Dur", times="Count")}),
    "trolley_handle": dict(kind="control", label="the trolley handle", nouns=["handle"],
                           adjs=["trolley", "telescopic", "telescoping", "pull", "extendable", "retractable", "long"],
                           ops={"set": dict(value="Level")}, levels=(0, 1, 2)),
    "trolley_button": dict(kind="control", label="the trolley-handle button", nouns=["button"], adjs=["handle", "trolley", "grip"],
                           ops={"press": dict(hold="Dur", times="Count")}),
    "side_handle": dict(kind="handle", label="the side carry handle", nouns=["handle"], adjs=["side", "carry", "carrying"],
                        ops={}),
    "strap": dict(kind="part", label="the strap buckle", nouns=["strap", "buckle", "clip"], adjs=["front", "compression", "lid"],
                  ops={"fasten": {}, "unfasten": {}}),
    "wheels": dict(kind="part", label="the spinner wheels", nouns=["wheels", "wheel", "casters"], adjs=["spinner"], ops={}),
}

# Things the person can pick up. `nouns` are head nouns that can name the item,
# `adjs` modifiers; family decides the grasp and how it is set down.
ITEM_WORDS = {
    "trousers": dict(nouns=["trousers", "pants", "jeans", "slacks"], adjs=["blue", "folded", "denim"]),
    "tshirts": dict(nouns=["t-shirts", "tshirts", "tees", "t-shirt", "tshirt", "tee", "shirts"], adjs=["white", "folded", "t-shirt"]),
    "sweater_pile": dict(nouns=["sweaters", "sweater", "jumpers", "knits", "knitwear"], adjs=["folded", "wool", "knit"]),
    "striped_shirts": dict(nouns=["shirts", "shirt"], adjs=["striped", "folded", "button"]),
    "sweatshirts": dict(nouns=["sweatshirts", "sweatshirt", "hoodies", "hoodie", "jumpers"], adjs=["grey", "gray", "folded"]),
    "towel": dict(nouns=["towel"], adjs=["folded", "hand", "small", "white"]),
    "towel_stack": dict(nouns=["towels", "stack"], adjs=["bath", "stacked"]),
    "gym_clothes": dict(nouns=["sweatpants", "socks", "gym", "workout", "kit"], adjs=["gym", "sports"]),
    "toiletry_bag": dict(nouns=["bag", "washbag", "toiletries", "kit"], adjs=["toiletry", "wash", "green", "dopp"]),
    "bottle": dict(nouns=["bottle", "flask", "thermos"], adjs=["water", "steel", "metal", "green"]),
    "notebook": dict(nouns=["notebook", "journal", "book", "binder", "notepad"], adjs=["leather", "brown"]),
    "camera": dict(nouns=["camera"], adjs=["black", "digital"]),
}
GENERIC_ITEM_NOUNS = {"pile", "stack", "clothes", "clothing", "garments", "item", "thing", "things", "stuff"}
GARMENTS = {"trousers", "tshirts", "sweater_pile", "striped_shirts", "sweatshirts", "towel", "towel_stack", "gym_clothes"}

STATION_WORDS = {
    "dresser": ["dresser", "drawers", "chest"], "bed": ["bed", "mattress"], "bed_far": ["far side of the bed", "other side of the bed"],
    "nightstand": ["nightstand", "bedside table", "night table"], "vanity": ["vanity", "sink", "bathroom counter"],
    "towel_shelf": ["towel shelf", "shelf", "bathroom shelf"], "kitchen_counter": ["kitchen counter", "counter"],
    "coffee_table": ["coffee table", "living room table", "table"], "suitcase": ["suitcase", "case", "luggage"],
    "bed_foot": ["foot of the bed", "end of the bed", "bed foot"],
    "bedroom_door": ["bedroom door", "door", "doorway", "door of the bedroom"],
}

ANCHOR_WORDS = {
    "wheel_end": ["wheel end", "bottom end", "wheel-end", "bottom"], "handle_end": ["handle end", "top end", "handle-end"],
    "front_strip": ["front", "front side", "near side", "front strip"], "bottle_corner": ["corner", "front corner", "wheel-end corner"],
    "centre": ["middle", "centre", "center"],
}

ASSETS = {
    "suitcase": dict(kind="device", label="the suitcase", parts=SUITCASE_PARTS,
                     ops=["stand_up", "lay_down", "roll"],
                     source=["scan: BlenderKit 'Black suitcase' (L'ANE)", "Samsonite TSA lock instructions",
                             "Samsonite / Delsey spinner manuals"]),
}


# ---------------------------------------------------------------- verbs
# signature: ordered (name, type, required, default); `ref` lists the kinds the main
# reference may resolve to. `steps` is the grounding and execution algorithm, shown on
# the page; `check` is the physical success test; `effect` the symbolic change.
VERBS = {
    "go_to": dict(group="navigate", sig=[("target", "Ref", True, None)], ref=("station", "item", "anchor_case", "room"),
                  steps=["Resolve the place, or the item (then its current place).",
                         "Reachable in one step only if it is in the person's room, or in view: an unobstructed line "
                         "from eye height (1.55 m) to it past walls, door frames and furniture. Otherwise the request "
                         "is expanded: enter(room) for each room on the way until the target is in view, then "
                         "look_at(item) for an item that has to be found, then go_to (each step logged with its parent). "
                         "go_to(room) is the enters alone.",
                         "Stand point: a station's default spot, or in front of the item at the station's standoff.",
                         "Route: shortest path on the walking graph, shortcut wherever the straight line keeps 30 cm "
                         "from furniture; a carried item is turned the way round that keeps it clear of walls.",
                         "Walk at the persona's speed and turn to face the station."],
                  check="if carrying, the item is still between the hands (< 30 cm)", effect="station := place"),
    "enter": dict(group="navigate", sig=[("target", "Ref", True, None)], ref=("room",),
                  steps=["Resolve the room; it must open off the person's room (a step of the walking graph crosses "
                         "into it).",
                         "Entry point: the first walking-graph point inside that room on the way to it (or the "
                         "given one, when expanded from a go_to).",
                         "Walk there through the doorway (or across the open-plan boundary) at the persona's speed; "
                         "a carried item is turned the way round that keeps it clear of walls."],
                  check="if carrying, the item is still between the hands (< 30 cm); the person ends inside the room",
                  effect="room := room (standing at its entry point)"),
    "look_at": dict(group="navigate", sig=[("target", "Ref", True, None)], ref=("station", "item", "part"),
                    steps=["Resolve the reference to a point (item centre, part centre or station).",
                           "Turn the head (gaze target) towards it; the body does not move.",
                           "Cast a ray in the physics scene from eye height to it; anything static (walls, door "
                           "frames, furniture) hit first means it is not in sight."],
                    check="recorded: the target is in sight (look_check event)", effect="none"),
    "pick_up": dict(group="manipulate", sig=[("target", "Ref", True, None)], ref=("item",),
                    steps=["Resolve the item; it must be within reach (at this station, or in the open case with nothing on it).",
                           "Grasp by family: folded piles and boxes with a two-hand side squeeze across the short side, "
                           "the bottle with a one-hand power wrap just above its centre of mass.",
                           "Targets from the item's observed pose: palms outside the faces, then in by 7-10 mm.",
                           "Lift 10-12 cm and test: the item must rise with the hands by contact alone.",
                           "Carry in front of the chest at max(1.0 m, start + 10 cm)."],
                    check="item rose >= 70 % of the hand lift by contact (no attachment)", effect="holding := item"),
    "place": dict(group="manipulate",
                  sig=[("target", "Ref", False, "it"), ("rel", "Rel", False, "in"), ("dest", "Ref", False, "suitcase"),
                       ("at", "Spot", False, None), ("side", "Side", False, None), ("yaw", "Angle", False, None),
                       ("orient", "Orient", False, "")],
                  ref=("item",),
                  steps=["The item must be the one in hand.",
                         "Destination: 'in' the case at a Spot (named or measured), 'on' another packed item, "
                         "'beside' one (on a given side), or on a station's surface.",
                         "Pose: Spot -> (x, y) in the case frame; yaw default puts the squeezed side along the case "
                         "length; the bottle 'upright' or 'lying'.",
                         "Fit (5 mm height map of what is packed): inside the liner walls, resting on the highest "
                         "surface under it with >= 60 % support, nothing fragile underneath, top below the closed lid, "
                         "clear of the lid's swing, room for the hands beside the squeezed faces.",
                         "Lower to the measured top of the supports (live poses), open the grip along the palm "
                         "normals, lift clear."],
                  check="item rests within 7 cm of the requested centre inside the walls; bottle within 10 deg of "
                        "vertical (upright) or > 70 deg (lying)",
                  effect="item := in case at (x, y, yaw), on its supports"),
    "put_back": dict(group="manipulate", sig=[("target", "Ref", False, "it")], ref=("item",),
                     steps=["The item must be the one in hand and the person at its home station.",
                            "Lower it to its starting pose on that surface and release."],
                     check="item within 8 cm of where it started", effect="item := at home"),
    "open": dict(group="articulate", sig=[("target", "Ref", True, None), ("to", "Frac", False, 1.0)], ref=("part",),
                 steps=["Resolve the part; check the asset supports 'open' on it and the value (the lid only "
                        "fully, a zip to any fraction).",
                        "Lid: from the foot of the bed, hook the fingertips under the free edge's zip-track lip, swing "
                        "it past its balance point (~64 deg) and let it fall open.",
                        "Zip: pinch the slider and run it back along the track to the requested fraction."],
                 check="lid > 150 deg open; zip slider at the fraction", effect="part opening := to"),
    "close": dict(group="articulate", sig=[("target", "Ref", True, None), ("to", "Frac", False, 0.0)], ref=("part",),
                  steps=["Resolve the part and the value as for open.",
                         "Lid: hook the free edge, test-lift it, swing it over briskly past the balance point; if it "
                         "slips the hook is released and tried again in a different place.",
                         "Zip: press the lid down with one palm, pinch the slider and run it along the track to the "
                         "fraction; fully closed engages only after a closure check (lid < 2.5 deg)."],
                  check="lid < 3 deg; zip at the fraction", effect="part opening := to"),
    "press": dict(group="control", sig=[("target", "Ref", True, None), ("hold", "Dur", False, 0.3), ("times", "Count", False, 1)],
                  ref=("control",),
                  steps=["Resolve the control (a button).",
                         "Fingertip to 1 cm above the button along its axis, then in until the measured contact force "
                         "exceeds the button's spring (detected from contacts, not assumed).",
                         "Hold for `hold` seconds; release; repeat `times` times."],
                  check="button travel reached its stop for >= hold", effect="device reacts (lock releases the zip pulls; "
                                                                             "the handle latch frees)"),
    "turn": dict(group="control", sig=[("target", "Ref", True, None), ("to", "Digit", False, None), ("by", "Extent", False, None)],
                 ref=("control",),
                 steps=["Resolve the dial; exactly one of `to` / `by`.",
                        "Read the dial; the shorter way round gives the clicks to go.",
                        "Fingertip on the exposed rim, drag tangentially one detent at a time; re-read after each "
                        "flick (closed loop) until the reading equals the target."],
                 check="dial reads the target digit (joint angle within half a detent)", effect="dial := digit"),
    "set": dict(group="control", sig=[("target", "Ref", True, None), ("value", "Setting", True, None)], ref=("control", "device"),
                steps=["Resolve the control or device and parse the setting for it (Code for the lock, Level for "
                       "the trolley handle).",
                       "Lock: turn each dial to its digit (as `turn`).",
                       "Trolley handle: grip the bar, press the button in the grip, pull or push to the stop, "
                       "release the button so the latch engages."],
                check="lock dials read the code; handle latched within 5 mm of the stop", effect="setting := value"),
    "insert": dict(group="control", sig=[("target", "Ref", True, None), ("into", "Ref", False, "lock")], ref=("part",),
                   steps=["Resolve the zip pulls; the zip must be fully closed.",
                          "Pinch both pulls and push them into the lock's slots until they latch."],
                   check="both pulls latched", effect="zip pulls := docked"),
    "fasten": dict(group="articulate", sig=[("target", "Ref", True, None)], ref=("part",),
                   steps=["Resolve the strap; the lid must be closed.",
                          "Pinch the male buckle, align it with the female and push until it clicks."],
                   check="buckle latched", effect="strap := fastened"),
    "unfasten": dict(group="articulate", sig=[("target", "Ref", True, None)], ref=("part",),
                     steps=["Resolve the strap.", "Squeeze the two side-release tabs and pull the male half out."],
                     check="buckle free", effect="strap := open"),
    "stand_up": dict(group="case", sig=[("target", "Ref", False, "suitcase")], ref=("device",),
                     steps=["The case must be closed and zipped; from the foot of the bed.",
                            "Two-hand overhand grip on the trolley bar (test pull, re-grip if it slips).",
                            "Tip it up onto its wheels with the wrists following the case, lift it off the bed, carry "
                            "it past the bed end and set it down; release while rising."],
                     check="case within 10 deg of upright on four wheels", effect="case := upright on the floor"),
    "lay_down": dict(group="case", sig=[("target", "Ref", False, "suitcase")], ref=("device",),
                     steps=["Not simulated yet: requests are rejected (unsupported_operation)."],
                     check="-", effect="-"),
    "roll": dict(group="case", sig=[("target", "Ref", False, "suitcase"), ("to", "Ref", True, None)], ref=("device",),
                 steps=["The case must stand on its wheels with the trolley handle at a stop above 0.",
                        "Grip the handle, walk the route to the place pushing the case alongside on its "
                        "360-degree wheels (the way the manuals recommend), let go."],
                 check="case upright within 20 cm of the destination", effect="case place := destination"),
    "fold": dict(group="cloth", sig=[("target", "Ref", True, None), ("style", "Style", False, "half")], ref=("cloth",),
                 steps=["The loose towel lies flat on the kitchen counter, its near edge hanging over the counter's "
                        "front edge.",
                        "Both flat hands, palms up, slide under the hanging edge (a third in from each corner), lift "
                        "it, draw back past the counter's lip and rise above the towel.",
                        "Carry the edge over to just past the far edge, lower it, tip the fingertips down so it slides "
                        "off them onto the towel, lift the hands away."],
                 check="the towel's depth shrank to 0.5 of flat (+-0.2), measured on the simulated cloth",
                 effect="cloth := folded in half"),
    "unfold": dict(group="cloth", sig=[("target", "Ref", True, None)], ref=("cloth",),
                   steps=["Not simulated: the folded edge lies flat on the towel, where neither the scoop nor a "
                          "pinch can get under it with this hand model (the cloth is too coarse to bunch)."],
                   check="-", effect="-"),
    "wait": dict(group="other", sig=[("seconds", "Dur", True, None)], ref=(),
                 steps=["Arms at rest; hold for the duration."], check="none", effect="none"),
    "say": dict(group="other", sig=[("text", "Text", True, None)], ref=(),
                steps=["Arms at rest; the utterance is logged as an event (question / plan-change channel)."],
                check="none", effect="none"),
}

# Composite requests expanded into primitives by the grounder.
MACROS = {
    "pack": "go_to(item) -> pick_up(item) -> go_to(suitcase) -> place(item, in, suitcase, at)",
    "unpack": "go_to(suitcase) -> pick_up(item) -> go_to(home) -> put_back(item)",
    "lock": "close(main_zip) -> insert(zip_pulls, into=lock) -> turn dials away from the code",
    "unlock": "set(lock, code) -> press(lock_button)",
}


# ---------------------------------------------------------------- reference resolution
STOP = {"the", "a", "an", "my", "your", "this", "that", "these", "those", "of", "to", "please", "some", "its", "their"}
PRONOUNS = {"it", "them", "this", "that", "these", "those", "one"}


def _tok(s):
    return [w for w in re.findall(r"[a-z0-9\-]+", s.lower()) if w not in STOP]


def candidates(state):
    """Everything a reference can name now, with its kind and vocabulary."""
    out = []
    for it in S.ITEMS:
        w = ITEM_WORDS.get(it.name, dict(nouns=[it.name], adjs=[]))
        where = state.loc.get(it.name, "") if state else ""
        loc_words = []
        if where.startswith("station:"):
            stn = where.split(":", 1)[1]
            loc_words = STATION_WORDS.get(stn, []) + R.WORDS.get(S.STATIONS[stn].room, [])
        elif where.startswith("case:"):
            loc_words = ["suitcase", "case", "packed"]
        elif where == "held":
            loc_words = ["hand", "hands", "held", "holding"]
        nouns = list(w["nouns"]) + (sorted(GENERIC_ITEM_NOUNS) if it.name in GARMENTS else ["item", "thing"])
        out.append(dict(id=it.name, kind="item", label=it.label, nouns=nouns, adjs=list(w["adjs"]),
                        loc=[x for p in loc_words for x in _tok(p)], phrases=[it.label.lower(), it.name.replace("_", " ")]))
    for name, words in STATION_WORDS.items():
        if name in S.STATIONS or name == "bedroom_door":
            out.append(dict(id=name, kind="station", label=words[0], nouns=[], adjs=[], loc=[], phrases=words))
    for pid, p in SUITCASE_PARTS.items():
        kind = p["kind"]
        out.append(dict(id=pid, kind="part" if kind in ("part", "handle") else kind, label=p["label"], nouns=p["nouns"],
                        adjs=p.get("adjs", []) + ["suitcase", "case", "luggage"], loc=[], phrases=[p["label"][4:]]))
    out.append(dict(id=S.CLOTH["name"], kind="cloth", label=S.CLOTH["label"], nouns=["towel", "cloth"],
                    adjs=["beach", "loose", "big", "flat", "orange"], loc=["kitchen", "counter"],
                    phrases=[S.CLOTH["label"], "loose towel"]))
    for rm, words in R.WORDS.items():
        out.append(dict(id=rm, kind="room", label=R.LABEL[rm], nouns=[], adjs=[], loc=[], phrases=words))
    for a, words in ANCHOR_WORDS.items():
        out.append(dict(id=a, kind="anchor_case", label=LO.ANCHORS[a]["label"], nouns=[], adjs=[], loc=[], phrases=words))
    return out


def resolve_ref(phrase, kinds, state=None, held="", last=""):
    """Words -> (id, kind) or Rejection. Algorithm:
      1. pronouns ('it', 'them') name the item in hand, else the last item mentioned;
      2. an exact label or phrase match of an allowed kind wins;
      3. otherwise the phrase's content words must all belong to a candidate's
         vocabulary (nouns, modifiers, where it is now) and its head word (the last
         word) must be one of the candidate's nouns;
      4. one match -> resolved; none -> unknown_ref; several -> ambiguous_ref, unless
         exactly one of them has the most modifiers matched."""
    p = phrase.strip().lower()
    toks = _tok(p)
    if not toks or (len(toks) == 1 and toks[0] in PRONOUNS) or p in ("it", "them"):
        tgt = held or last
        if not tgt:
            return Rejection("unknown_ref", f"'{phrase}' does not refer to anything: nothing is in hand")
        return tgt, "item"
    cands = [c for c in candidates(state) if c["kind"] in kinds or (c["kind"] == "device" and "device" in kinds)]
    pp = " ".join(toks)
    exact = [c for c in cands if pp in [" ".join(_tok(x)) for x in c["phrases"]] or pp == c["id"].replace("_", " ")]
    if len(exact) == 1:
        return exact[0]["id"], exact[0]["kind"]
    # "the pile on the dresser": the head noun comes before a locative preposition, and
    # the words after it must say where the thing is now
    LOC = {"on", "in", "at", "from", "near", "by", "inside", "beside"}
    cut = next((k for k, t in enumerate(toks) if t in LOC and k > 0), len(toks))
    main, where = toks[:cut], [t for t in toks[cut + 1:] if t not in LOC]
    head = main[-1]
    scored = []
    for c in cands:
        vocab = set(c["nouns"]) | set(c["adjs"])
        phr = set(t for x in c["phrases"] for t in _tok(x))
        if c["kind"] in ("station", "anchor_case", "room"):
            if all(t in phr for t in toks):
                scored.append((len(toks), c))
            continue
        if head not in c["nouns"]:
            continue
        if all(t in vocab or t in phr for t in main) and all(t in c["loc"] for t in where):
            mods = sum(1 for t in main if t in c["adjs"]) + len(where)
            scored.append((mods, c))
    if not scored and where:
        return Rejection("unknown_ref", f"nothing matching '{' '.join(main)}' is {' '.join(toks[cut:])} now")
    if not scored:
        other = [c for c in candidates(state) if c["kind"] not in kinds]
        for c in other:
            vocab = set(c["nouns"]) | set(c["adjs"]) | set(t for x in c["phrases"] for t in _tok(x))
            if head in vocab and all(t in vocab | set(c["loc"]) for t in toks):
                if c["kind"] == "cloth":
                    return Rejection("wrong_kind", f"'{phrase}' is the loose {c['label']}: it can be folded, not "
                                                   f"picked up or packed in this simulator")
                art = lambda k: ("an " if k[0] in "aeiou" else "a ") + {"anchor_case": "spot in the case",
                                                                        "cloth": "loose cloth"}.get(k, k)
                lab = c["label"] if c["label"].startswith("the ") else "the " + c["label"]
                return Rejection("wrong_kind", f"'{phrase}' is {lab}, which is not {' or '.join(art(k) for k in kinds)}")
        return Rejection("unknown_ref", f"nothing here matches '{phrase}'")
    best = max(s for s, _ in scored)
    top = [c for s, c in scored if s == best]
    if len(top) == 1:
        return top[0]["id"], top[0]["kind"]
    return Rejection("ambiguous_ref", f"'{phrase}' could mean " + ", ".join(c["label"] for c in top),
                     candidates=[c["id"] for c in top])


# ---------------------------------------------------------------- values
WORD_NUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "once": 1, "twice": 2, "thrice": 3}


def parse_value(typ, v):
    """Parse a value of a type; raises ValueError(message)."""
    if v is None:
        return None
    if typ == "Frac":
        if isinstance(v, (int, float)):
            f = float(v)
            return f / 100 if f > 1 else f
        s = str(v).strip().lower()
        named = {"fully": 1.0, "all the way": 1.0, "full": 1.0, "completely": 1.0, "halfway": 0.5, "half way": 0.5, "half": 0.5,
                 "a quarter": 0.25, "quarter": 0.25, "a third": 1 / 3, "three quarters": 0.75, "partly": 0.5, "a bit": 0.2,
                 "shut": 0.0, "closed": 0.0, "open": 1.0}
        if s in named:
            return named[s]
        m = re.match(r"^([0-9.]+)\s*%$", s)
        if m:
            return float(m.group(1)) / 100
        try:
            f = float(s)
        except ValueError:
            raise ValueError(f"'{v}' is not a fraction")
        return f / 100 if f > 1 else f
    if typ in ("Digit", "Count", "Level", "Extent"):
        if isinstance(v, (int, float)) and float(v) == int(v):
            return int(v)
        s = str(v).strip().lower()
        if typ == "Level":
            named = {"retracted": 0, "down": 0, "in": 0, "first": 1, "first stop": 1, "1st": 1, "middle": 1, "half": 1,
                     "second": 2, "second stop": 2, "2nd": 2, "top": 2, "full": 2, "fully": 2, "all the way": 2, "up": 2}
            if s in named:
                return named[s]
        if s in WORD_NUM:
            return WORD_NUM[s]
        try:
            f = float(s)
        except ValueError:
            raise ValueError(f"'{v}' is not a whole number")
        if f != int(f):
            raise ValueError(f"{v} is not a whole number")
        return int(f)
    if typ == "Code":
        digits = re.sub(r"[^0-9]", "", str(v))
        if len(digits) != 3:
            raise ValueError(f"'{v}' is not a three-digit code")
        return digits
    if typ == "Dur":
        if isinstance(v, (int, float)):
            return float(v)
        m = re.match(r"^([0-9.]+)\s*(s|sec|secs|second|seconds|ms)?$", str(v).strip().lower())
        if not m:
            if str(v).strip().lower() in ("a second", "one second"):
                return 1.0
            raise ValueError(f"'{v}' is not a duration")
        f = float(m.group(1))
        return f / 1000 if m.group(2) == "ms" else f
    if typ == "Angle":
        return float(re.sub(r"[^0-9.\-]", "", str(v)) or "nan")
    if typ == "Rel":
        s = str(v).strip().lower()
        s = {"into": "in", "inside": "in", "on top of": "on", "onto": "on", "next to": "beside", "by": "beside"}.get(s, s)
        if s not in ("in", "on", "beside"):
            raise ValueError(f"'{v}' is not one of in / on / beside")
        return s
    if typ == "Side":
        s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        s = {"wheel": "wheel_end", "bottom": "wheel_end", "handle": "handle_end", "top": "handle_end", "front": "front_side",
             "near": "front_side", "hinge": "hinge_side", "back": "hinge_side", "far": "hinge_side"}.get(s, s)
        if s not in ("wheel_end", "handle_end", "front_side", "hinge_side"):
            raise ValueError(f"'{v}' is not a side of an item in the case")
        return s
    if typ == "Orient":
        s = str(v).strip().lower()
        s = {"standing": "upright", "vertical": "upright", "on its side": "lying", "flat": "lying" if False else "flat",
             "horizontal": "lying", "laid down": "lying", "lying down": "lying"}.get(s, s)
        if s not in ("upright", "lying", "flat", ""):
            raise ValueError(f"'{v}' is not upright / lying / flat")
        return s
    if typ == "Style":
        s = str(v).strip().lower()
        s = {"in half": "half", "halves": "half", "in thirds": "thirds", "third": "thirds"}.get(s, s)
        if s not in ("half", "thirds"):
            raise ValueError(f"'{v}' is not a fold style (half / thirds)")
        return s
    if typ == "Spot":
        if isinstance(v, dict):
            if "anchor" in v:
                return v["anchor"]
            a = float(v.get("from_wheel_end_cm", v.get("along_cm")))
            b = float(v.get("from_hinge_side_cm", v.get("across_cm")))
            return (LO.X0 + b / 100, LO.Y0 + a / 100)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (LO.X0 + float(v[1]) / 100, LO.Y0 + float(v[0]) / 100)
        return str(v)
    return v


PAST = {"open": "opened", "close": "closed", "press": "pressed", "turn": "turned", "set": "set", "insert": "inserted",
        "fasten": "fastened", "unfasten": "unfastened"}


# ---------------------------------------------------------------- grounding
def ground(req, state, last=""):
    """{'verb': v, 'args': {...}} with words -> Action, or Rejection."""
    verb = str(req.get("verb", "")).strip().lower().replace(" ", "_")
    args = dict(req.get("args", {}))
    if "target" in req and "target" not in args:
        args["target"] = req["target"]
    if verb not in VERBS:
        return Rejection("unknown_verb", f"there is no action '{verb}' (verbs: {', '.join(VERBS)})")
    spec = VERBS[verb]
    held = state.holding if state else ""
    out = {}
    target = ""
    for (name, typ, req_, default) in spec["sig"]:
        v = args.get(name, default)
        if v is None:
            if req_:
                return Rejection("missing_argument", f"{verb} needs '{name}' ({TYPES.get(typ, typ)})")
            continue
        if typ == "Ref":
            if name == "target":
                kinds = spec["ref"]
            elif name == "dest":
                kinds = ("station", "item", "device")
            elif name == "into":
                kinds = ("device", "part")
            else:
                kinds = ("station",)
            if verb == "roll" and name == "to":
                kinds = ("station",)
            if name == "target" and isinstance(v, str) and v in [it.name for it in S.ITEMS] + list(SUITCASE_PARTS) + ["suitcase", S.CLOTH["name"]]:
                r = (v, "item" if v in [it.name for it in S.ITEMS] else "cloth" if v == S.CLOTH["name"] else
                     "device" if v in ("suitcase", "lock") else "part")
            elif name == "dest" and v == "suitcase":
                r = ("suitcase", "device")
            else:
                r = resolve_ref(str(v), kinds + (("device",) if v in ("suitcase", "case") else ()), state, held, last)
                if isinstance(r, Rejection) and str(v).lower() in ("suitcase", "the suitcase", "case", "the case", "luggage"):
                    r = ("suitcase", "device")
            if isinstance(r, Rejection):
                return r
            if name == "target":
                target = r[0]
            else:
                out[name] = r[0]
            continue
        if typ == "Setting":
            out[name] = v
            continue
        try:
            out[name] = parse_value(typ, v)
        except ValueError as e:
            return Rejection("bad_value", f"{name}: {e}")
    a = Action(verb, target, out)
    rej = check_operation(a)
    return rej or a


def check_operation(a):
    """Does the asset support this verb on this part, with this value?"""
    v, t = a.verb, a.target
    if v in ("open", "close", "press", "turn", "set", "insert", "fasten", "unfasten"):
        part = SUITCASE_PARTS.get(t) if t != "suitcase" else None
        if t == "suitcase":
            if v in ("open", "close"):
                return Rejection("unsupported_operation", "say which part: the lid or the zip "
                                                          f"(e.g. '{v} the lid', '{v} the zip')")
            return Rejection("unsupported_operation", f"the suitcase itself cannot be {v}ed; name one of its parts")
        if part is None:
            return Rejection("unsupported_operation", f"{v} applies to parts and controls of the suitcase, not '{t}'")
        if v not in part["ops"]:
            ok = [k for k, p in SUITCASE_PARTS.items() if v in p["ops"]]
            return Rejection("unsupported_operation", f"{part['label']} cannot be {PAST.get(v, v + 'ed')}"
                                                      f" (it supports: {', '.join(part['ops']) or 'no operations'}); "
                                                      f"'{v}' applies to: {', '.join(SUITCASE_PARTS[k]['label'] for k in ok)}")
        op = part["ops"][v]
        if v in ("open", "close"):
            to = a.get("to")
            if "values" in op and to not in op["values"]:
                return Rejection("unsupported_value", f"{part['label']} cannot be {'opened' if v == 'open' else 'closed'} to "
                                                          f"{to:.0%}: {op['why']}")
            if "range" in op and not (op["range"][0] <= to <= op["range"][1]):
                return Rejection("out_of_range", f"{to} is outside 0..1")
        if v == "turn":
            if (a.get("to") is None) == (a.get("by") is None):
                return Rejection("missing_argument", "turn needs exactly one of 'to' (a digit) or 'by' (clicks)")
            if a.get("to") is not None and not (0 <= a.get("to") <= 9):
                return Rejection("out_of_range", f"a dial shows 0-9, not {a.get('to')}")
        if v == "set":
            val = a.get("value")
            if t == "lock":
                try:
                    a.args["value"] = parse_value("Code", val)
                except ValueError as e:
                    return Rejection("bad_value", str(e))
            elif t == "trolley_handle":
                try:
                    lv = parse_value("Level", val)
                except ValueError as e:
                    return Rejection("bad_value", str(e))
                if lv not in part["levels"]:
                    return Rejection("unsupported_value", f"the trolley handle locks at stops "
                                                          f"{', '.join(map(str, part['levels']))} (0 = retracted), not {val}")
                a.args["value"] = lv
        if v == "press":
            if not (0.05 <= a.get("hold", 0.3) <= 10):
                return Rejection("out_of_range", "hold must be 0.05-10 s")
            if not (1 <= a.get("times", 1) <= 5):
                return Rejection("out_of_range", "times must be 1-5")
    if v in ("pick_up", "put_back") and t not in [it.name for it in S.ITEMS]:
        if t == S.CLOTH["name"]:
            return Rejection("unsupported_operation", f"the loose {S.CLOTH['label']} cannot be picked up and packed "
                                                      f"in this simulator (only folded)")
        return Rejection("wrong_kind", f"{v} needs an item")
    if v in ("fold", "unfold"):
        if t != S.CLOTH["name"]:
            return Rejection("unsupported_operation", f"{_the(t)} is a scanned folded pile; only the loose "
                                                      f"{S.CLOTH['label']} can be folded")
        if v == "unfold":
            return Rejection("unsupported_operation", "unfolding is not simulated: the hands cannot get under the "
                                                      "folded edge lying on the towel")
        if a.get("style", "half") != "half":
            return Rejection("unsupported_value", "only folding in half is simulated: a second fold needs the folded "
                                                  "edge to hang over the counter's edge, and it does not")
    if v == "place":
        o = a.get("orient", "")
        if o and t and t != "bottle" and o not in ("flat", ""):
            return Rejection("unsupported_value", f"only the bottle can stand upright or lie down; the "
                                                  f"{S.item(t).label} can only be laid flat")
        if a.get("rel") in ("on", "beside") and a.get("dest") in (None, "suitcase"):
            return Rejection("missing_argument", f"place {a.get('rel')} what?")
        if a.get("rel") == "beside" and not a.get("side"):
            return Rejection("missing_argument", "beside which side (wheel_end / handle_end / front_side / hinge_side)?")
        y = a.get("yaw")
        if y is not None:
            if not np.isfinite(y):
                return Rejection("bad_value", "yaw must be degrees")
            a.args["yaw"] = float(y) % 180.0
    if v in ("stand_up", "lay_down", "roll") and t not in ("suitcase", ""):
        return Rejection("wrong_kind", f"{v} applies to the suitcase")
    if v == "lay_down":
        return Rejection("unsupported_operation", "laying the standing case back down on the bed is not simulated yet")
    if v == "wait" and not (0 < a.get("seconds", 0) <= 60):
        return Rejection("out_of_range", "wait 0-60 s")
    return None


# ---------------------------------------------------------------- text
def _lab(n):
    if n in [it.name for it in S.ITEMS]:
        return S.item(n).label
    if n in S.STATIONS:
        return S.STATIONS[n].label
    if n in SUITCASE_PARTS:
        return SUITCASE_PARTS[n]["label"]
    if n in LO.ANCHORS:
        return LO.ANCHORS[n]["label"]
    if n == S.CLOTH["name"]:
        return "the " + S.CLOTH["label"]
    if n in R.LABEL:
        return R.LABEL[n]
    return {"suitcase": "the suitcase", "bedroom_door": "the bedroom door"}.get(n, n)


def _the(n):
    s = _lab(n)
    return s if s.startswith("the ") else "the " + s


def describe(a):
    v, t = a.verb, a.target
    if v == "go_to":
        return f"go to {_the(t) if t in [it.name for it in S.ITEMS] else _lab(t)}"
    if v == "enter":
        return f"walk into {R.LABEL.get(t, t)}"
    if v == "look_at":
        return f"look at {_the(t)}"
    if v == "pick_up":
        return f"pick up {_the(t)}"
    if v == "place":
        rel, dest = a.get("rel", "in"), a.get("dest", "suitcase")
        s = f"put {_the(t) if t else 'it'} "
        if rel == "in" and dest == "suitcase":
            at = a.get("at")
            if isinstance(at, str):
                s += f"in {_lab(at)}"
            elif at is not None:
                s += f"in the case {100 * (at[1] - LO.Y0):.0f} cm from the wheel end, {100 * (at[0] - LO.X0):.0f} cm from the hinge side"
            else:
                s += "in the suitcase"
        elif rel == "on":
            s += f"on {_the(dest)}"
        elif rel == "beside":
            s += f"beside {_the(dest)} ({str(a.get('side', '')).replace('_', ' ')})"
        else:
            s += f"on {_lab(dest)}"
        if a.get("orient") in ("upright", "lying"):
            s += f", {a.get('orient')}"
        if a.get("yaw") is not None:
            s += f", turned {a.get('yaw'):.0f} deg"
        return s
    if v == "put_back":
        return "put it back where it was"
    if v in ("open", "close"):
        to = a.get("to")
        part = _lab(t)
        if t == "main_zip":
            if v == "close":
                return "zip the case" if to == 0 else f"zip the case to {1 - to:.0%} closed"
            return "unzip the case" if to == 1 else f"unzip the case to {to:.0%} open"
        return f"{v} {part}"
    if v == "press":
        s = f"press {_lab(t)}"
        if a.get("times", 1) > 1:
            s += f" {a.get('times')} times"
        if a.get("hold", 0.3) > 0.5:
            s += f" for {a.get('hold'):.1f} s"
        return s
    if v == "turn":
        return f"turn {_lab(t)} to {a.get('to')}" if a.get("to") is not None else f"turn {_lab(t)} by {a.get('by')}"
    if v == "set":
        if t == "trolley_handle":
            return "retract the trolley handle" if a.get("value") == 0 else f"extend the trolley handle to stop {a.get('value')}"
        return f"set {_lab(t)} to {a.get('value')}"
    if v == "insert":
        return "lock the zip pulls into the lock"
    if v in ("fasten", "unfasten"):
        return f"{v} {_lab(t)}"
    if v == "stand_up":
        return "stand the case up on its wheels"
    if v == "lay_down":
        return "lay the case down on the bed"
    if v == "roll":
        return f"roll the case to {_lab(a.get('to'))}"
    if v == "fold":
        return f"fold {_the(t)} in {a.get('style', 'half')}"
    if v == "unfold":
        return f"unfold {_the(t)}"
    if v == "wait":
        return f"pause for {a.get('seconds'):g} s"
    if v == "say":
        return f'say "{a.get("text")}"'
    return v


def spec():
    """JSON description of the action space for the page and for an LLM parser."""
    return dict(types=TYPES, verbs={k: dict(group=v["group"], signature=[dict(name=n, type=t, required=r, default=d)
                                                                        for n, t, r, d in v["sig"]],
                                            ref_kinds=list(v["ref"]), steps=v["steps"], check=v["check"], effect=v["effect"])
                                    for k, v in VERBS.items()},
                macros=MACROS,
                assets=dict(suitcase=dict(label="the suitcase", source=ASSETS["suitcase"]["source"],
                                          case_ops=ASSETS["suitcase"]["ops"],
                                          parts={k: dict(kind=p["kind"], label=p["label"], words=p["nouns"] + p.get("adjs", []),
                                                         ops={o: {kk: vv for kk, vv in d.items()} for o, d in p["ops"].items()},
                                                         **({"levels": list(p["levels"])} if "levels" in p else {}))
                                                     for k, p in SUITCASE_PARTS.items()})),
                anchors={k: dict(label=v["label"], words=ANCHOR_WORDS[k], centre_cm=dict(
                    from_wheel_end=round(100 * (v["xy"][1] - LO.Y0), 1), from_hinge_side=round(100 * (v["xy"][0] - LO.X0), 1)))
                         for k, v in LO.ANCHORS.items()},
                case_interior_cm=dict(along=round(100 * (LO.Y1 - LO.Y0), 1), across=round(100 * (LO.X1 - LO.X0), 1),
                                      height_closed=round(100 * (LO.TOP - LO.FLOOR), 1)),
                rejection_codes=["unknown_verb", "missing_argument", "bad_value", "unknown_ref", "ambiguous_ref", "wrong_kind",
                                 "unsupported_operation", "unsupported_value", "out_of_range"])
