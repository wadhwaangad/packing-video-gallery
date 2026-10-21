"""Instructions -> grounded actions.

  text ──parse──> requests ──expand macros──> ──ground (refs, values)──> ──reach──> Action
                  {"verb", "args"}            pack / unpack / lock /     actionspace.ground   go_to inserted when the
                  with words for refs         unlock -> primitives                            target is out of reach

Two parsers produce the same request format:
  * rules  (parse_rules): patterns for every verb, numbers and named values; no network.
  * gemini (parse_gemini): an LLM fills the verb and argument slots of a fixed JSON schema
    (it never picks object ids: it copies the user's words, and the deterministic
    resolver decides what they refer to, so ambiguity is detected the same way).
Set GEMINI_API_KEY (and optionally GEMINI_MODEL) to use it.

Navigation: a go_to reaches only what is in the person's room or in view; otherwise it is
expanded (packsim.navigation) into enter(room) per room on the way, look_at(item) and the
final go_to, each logged with its parent request. With auto_reach off nothing is inserted
or expanded, so an out-of-view go_to is refused (strict mode for evaluating agents).

Grounding happens lazily, against the state at the moment the request is reached, so
'the pile on the dresser' or 'it' resolve against what is there then.
"""
import json, os, re, urllib.request
import scene as S
import layout as LO
from actionspace import Action, Rejection, VERBS, ground, spec, SUITCASE_PARTS
from packsim import ITEM_NAMES, navigation

# ---------------------------------------------------------------- rule parser
FRAC_WORDS = r"(halfway|half way|half|fully|all the way|a quarter|quarter|three quarters|a third|partly|a bit|[0-9.]+\s*%|0?\.[0-9]+)"
VERB_START = re.compile(r"^(pack|enter|unpack|pick|grab|take|lift|put|place|set|stand|lay|go|walk|head|move|look|open|close|shut|zip|unzip|"
                        r"lock|unlock|turn|rotate|spin|press|push|extend|raise|pull|retract|lower|collapse|fasten|buckle|"
                        r"unfasten|unbuckle|undo|roll|wheel|fold|unfold|wait|pause|say|tell|ask|insert|dock|clip)\b")


def split_steps(text):
    parts = re.split(r"\s*(?:;|\n|\.\s+|,?\s+then\s+|,?\s+and then\s+|,\s+(?=[a-z]))\s*", text.strip().rstrip("."), flags=re.I)
    out = []
    for p in parts:
        # 'and' joins two steps only when a verb follows ('sweatpants and socks' stays one name)
        sub = re.split(r"\s+and\s+(?=(?:" + VERB_START.pattern[2:-3] + r")\b)", p, flags=re.I)
        out += [s.strip() for s in sub if s and s.strip()]
    return out


def _spot(s):
    """'12 cm from the wheel end and 5 cm from the hinge side' -> Spot dict; anchor words -> anchor."""
    s = s.strip().lower()
    a = re.search(r"([0-9.]+)\s*cm\s+from\s+the\s+(wheel|bottom)[- ]end", s)
    b = re.search(r"([0-9.]+)\s*cm\s+from\s+the\s+(hinge|back)[- ]side", s)
    if a or b:
        return dict(from_wheel_end_cm=float(a.group(1)) if a else 100 * (0 - LO.Y0),
                    from_hinge_side_cm=float(b.group(1)) if b else 100 * (0.0 - LO.X0))
    from actionspace import ANCHOR_WORDS
    for k, words in ANCHOR_WORDS.items():
        for w in sorted(words, key=len, reverse=True):
            if re.search(r"\b" + re.escape(w) + r"\b", s):
                return dict(anchor=k)
    return None


def parse_one(t):
    """One step of text -> request dict, or None."""
    s = t.strip().lower().rstrip(".!")
    s = re.sub(r"^(?:(?:please|now|next|first|finally|ok|okay|then|and|after that|lastly)\s*,?\s+)+", "", s)
    s = re.sub(r"\s+please$", "", s)
    m = re.match(r"^(?:say|tell (?:them|him|her)|ask)\s*[:,]?\s*[\"“']?(.+?)[\"”']?$", s)
    if m:
        return dict(verb="say", args=dict(text=t.strip().split(None, 1)[1].strip(" \"'“”:")))
    m = re.match(r"^(?:wait|pause)(?: for)? ([0-9.]+|a|one|two|three|five|ten)\s*(?:s|sec|secs|seconds?)?$", s)
    if m:
        n = {"a": 1, "one": 1, "two": 2, "three": 3, "five": 5, "ten": 10}.get(m.group(1), m.group(1))
        return dict(verb="wait", args=dict(seconds=float(n)))
    m = re.match(r"^(?:enter|(?:go|walk|head|step)(?: back)? (?:into|in to))\s+(.+)$", s)
    if m:
        return dict(verb="enter", args=dict(target=m.group(1)))
    m = re.match(r"^(?:go|walk|head)(?: over| back)? to (.+)$", s)
    if m:
        return dict(verb="go_to", args=dict(target=m.group(1)))
    m = re.match(r"^look at (.+)$", s)
    if m:
        return dict(verb="look_at", args=dict(target=m.group(1)))
    m = re.match(r"^unpack (.+)$", s)
    if m:
        return dict(verb="unpack", args=dict(target=m.group(1)))
    m = re.match(r"^pack (.+)$", s)
    if m:
        rest = m.group(1)
        args = {}
        mo = re.search(r",?\s*(upright|standing|lying(?: down)?|on its side)$", rest)
        if mo:
            args["orient"] = "upright" if mo.group(1) in ("upright", "standing") else "lying"
            rest = rest[:mo.start()]
        mr = re.match(r"^(.+?) (on top of|on|onto|beside|next to) (.+)$", rest)
        if mr and not re.match(r"^(?:the )?(?:suit)?case", mr.group(3)):
            what, rel, where = mr.groups()
            rel = "on" if rel in ("on", "onto", "on top of") else "beside"
            args.update(target=what, rel=rel, dest=where)
            if rel == "beside":
                ms = re.match(r"^(.+?),?\s*(?:on|at|towards?) the (wheel[- ]end|handle[- ]end|front|hinge|back|far|near).*$", where)
                if ms:
                    args["dest"], args["side"] = ms.group(1), ms.group(2).replace("-", " ").split()[0]
            return dict(verb="pack", args=args)
        mi = re.match(r"^(.+?)(?:\s+(?:in|into) (?:the )?(?:suit)?case)?(?:(?:,\s*|\s+)(?:(?:in|at|on) )?([0-9][0-9.]*\s*cm from.*|the (?:wheel|handle|bottom|top)[- ]end|the front(?: strip| side)?|the (?:front )?corner|the middle|the cent(?:re|er)))?$", rest)
        args["target"] = mi.group(1) if mi else rest
        if mi and mi.group(2):
            sp = _spot(mi.group(2))
            if sp:
                args["at"] = sp
        return dict(verb="pack", args=args)
    m = re.match(r"^(?:pick up|pick|grab|take|lift)(?: up)? (.+?)(?: up)?(?: (?:from|off|out of) .+)?$", s)
    if m and not re.match(r"^(?:take|lift) (?:it|them)? ?out", s):
        return dict(verb="pick_up", args=dict(target=m.group(1)))
    m = re.match(r"^take (.+?) out(?: of the (?:suit)?case)?$", s)
    if m:
        return dict(verb="pick_up", args=dict(target=m.group(1)))
    m = re.match(r"^(?:put|place|set|lay) (.+?) back(?: .*)?$", s)
    if m:
        return dict(verb="put_back", args=dict(target=m.group(1)))
    m = re.match(r"^(?:put|place|set|lay|stand) (it|them|.+?) (in|into|inside|on|onto|on top of|beside|next to|by) (.+?)"
                 r"(?:,?\s*(upright|standing|lying(?: down)?|on its side|flat))?(?:,?\s*(?:turned|rotated) ([0-9.]+)\s*(?:deg|degrees)?)?$", s)
    if m:
        what, rel, where, orient, yaw = m.groups()
        rel = {"into": "in", "inside": "in", "onto": "on", "on top of": "on", "next to": "beside", "by": "beside"}.get(rel, rel)
        args = dict(target=what, rel=rel)
        if orient:
            args["orient"] = "upright" if orient in ("upright", "standing") else "lying" if orient != "flat" else "flat"
        if yaw:
            args["yaw"] = float(yaw)
        if rel == "in":
            mm = re.match(r"^(?:the )?(?:suit)?case(?:,?\s*(?:in|at|on) (.+))?$", where)
            if mm:
                args["dest"] = "suitcase"
                if mm.group(1):
                    sp = _spot(mm.group(1))
                    if sp is None:
                        return dict(verb="place", args=dict(args, at=mm.group(1)))
                    args["at"] = sp
            else:
                sp = _spot(where)
                if sp:
                    args["dest"] = "suitcase"; args["at"] = sp
                else:
                    args["dest"] = where
        elif rel == "on":
            args["dest"] = where
        else:
            mm = re.match(r"^(.+?)(?:,?\s*(?:on|at|towards?) the (wheel[- ]end|handle[- ]end|front|hinge|back|far|near)(?: side| end)?(?: side)?)$", where)
            if mm:
                args["dest"], args["side"] = mm.group(1), mm.group(2).replace("-", " ").split()[0]
            else:
                args["dest"] = where
        return dict(verb="place", args=args)
    m = re.match(r"^(?:un-?zip)(?: (?:the|it|the case|the suitcase|the bag|the zip(?:per)?))?(?:\s+(?:to )?(" + FRAC_WORDS + r")(?: open| of the way)?)?$", s)
    if m:
        return dict(verb="open", args=dict(target="main zip", to=m.group(1) or 1.0))
    m = re.match(r"^zip(?: up)?(?: (?:the|it|the case|the suitcase|it up|the zip(?:per)?))?(?: up| shut| closed)?(?:\s+(?:to )?(" + FRAC_WORDS + r")(?: closed| shut| of the way)?)?$", s)
    if m:
        f = m.group(1)
        if f:
            from actionspace import parse_value
            return dict(verb="close", args=dict(target="main zip", to=1 - parse_value("Frac", f)))
        return dict(verb="close", args=dict(target="main zip", to=0.0))
    m = re.match(r"^(open|close|shut)(?: up)? (.+?)(?:\s+(?:to )?(" + FRAC_WORDS + r"))?$", s)
    if m:
        v = "open" if m.group(1) == "open" else "close"
        tgt = m.group(2)
        to = m.group(3)
        if tgt in ("the case", "the suitcase", "it", "the luggage", "case", "suitcase"):
            tgt = "the lid"
        args = dict(target=tgt)
        if to is not None:
            args["to"] = to if v == "open" else to
        return dict(verb=v, args=args)
    m = re.match(r"^(?:unlock)(?: the (?:case|suitcase|lock))?(?: (?:with|using)(?: the)?(?: code| combination)? ([0-9\- ]+))?$", s)
    if m:
        return dict(verb="unlock", args=dict(code=(m.group(1) or "").strip() or None))
    m = re.match(r"^lock(?: it| the (?:case|suitcase|bag|zip(?:per)?s?|pulls))?(?: up)?$", s)
    if m:
        return dict(verb="lock", args={})
    m = re.match(r"^(?:insert|dock|clip|put|push) (?:the )?(?:zip(?:per)? )?pulls?(?: (?:in|into) (?:the )?lock)?$", s)
    if m:
        return dict(verb="insert", args=dict(target="zip pulls", into="lock"))
    m = re.match(r"^(?:set|dial)(?: in)? (?:the )?(?:lock|combination|code)(?: to)? ([0-9\- ]+)$", s) or \
        re.match(r"^(?:dial|enter)(?: in)? (?:the )?(?:code |combination )?([0-9\- ]+)$", s)
    if m:
        return dict(verb="set", args=dict(target="lock", value=m.group(1).strip()))
    m = re.match(r"^(?:turn|rotate|spin|set) (.+?dial|.+?wheel|.+?digit) (to|by|up|down) ?([0-9a-z]+)?(?: clicks?| steps?| notches?)?$", s)
    if m:
        tgt, how, n = m.groups()
        if how == "to":
            return dict(verb="turn", args=dict(target=tgt, to=n))
        k = int(n) if n and n.isdigit() else 1
        return dict(verb="turn", args=dict(target=tgt, by=-k if how == "down" else k))
    m = re.match(r"^(?:press|push|hold)(?: down)? (.+?)(?: (once|twice|thrice|[0-9]+ times))?(?: for ([0-9.]+)\s*(?:s|sec|seconds?))?$", s)
    if m:
        args = dict(target=m.group(1))
        if m.group(2):
            args["times"] = {"once": 1, "twice": 2, "thrice": 3}.get(m.group(2), (m.group(2) or "1").split()[0])
        if m.group(3):
            args["hold"] = float(m.group(3))
        return dict(verb="press", args=args)
    m = re.match(r"^(extend|raise|pull up|pull out|retract|lower|collapse|push down|push in) (?:the )?((?:trolley |telescopic |pull )?handle)"
                 r"(?:\s+(?:to|up to)\s+(?:the\s+)?(.+?))?$", s)
    if m:
        v, tgt, lvl = m.groups()
        if v in ("retract", "lower", "collapse", "push down", "push in"):
            return dict(verb="set", args=dict(target="trolley handle", value=lvl or 0))
        lvl = (lvl or "").replace(" stop", "").replace("stop ", "").strip() or "1"
        return dict(verb="set", args=dict(target="trolley handle", value=lvl))
    m = re.match(r"^set (?:the )?(trolley handle|handle)(?: to)? (?:stop |level )?(.+)$", s)
    if m:
        return dict(verb="set", args=dict(target="trolley handle", value=m.group(2)))
    m = re.match(r"^(fasten|buckle|do up|clip) (.+)$", s)
    if m:
        return dict(verb="fasten", args=dict(target=m.group(2)))
    m = re.match(r"^(unfasten|unbuckle|undo|unclip|release) (.+)$", s)
    if m:
        return dict(verb="unfasten", args=dict(target=m.group(2)))
    if re.match(r"^stand (?:the (?:suit)?case|it)(?: up)?(?: on its wheels)?$", s) or s in ("stand it up", "stand up the case"):
        return dict(verb="stand_up", args=dict(target="suitcase"))
    if re.match(r"^lay (?:the (?:suit)?case|it) (?:down|flat)(?: on the bed)?$", s):
        return dict(verb="lay_down", args=dict(target="suitcase"))
    m = re.match(r"^(?:roll|wheel|push|take) (?:the (?:suit)?case|it|the luggage) to (.+)$", s)
    if m:
        return dict(verb="roll", args=dict(target="suitcase", to=m.group(1)))
    m = re.match(r"^fold (.+?)(?: (?:in|into) (half|halves|thirds))?$", s)
    if m:
        return dict(verb="fold", args=dict(target=m.group(1), style=m.group(2) or "half"))
    m = re.match(r"^unfold (.+)$", s)
    if m:
        return dict(verb="unfold", args=dict(target=m.group(1)))
    return None


def parse_rules(text):
    out = []
    for step in split_steps(text):
        r = parse_one(step)
        out.append(r if r else dict(verb="?", args=dict(text=step)))
    return out


# ---------------------------------------------------------------- Gemini parser
ARG_NAMES = ["target", "rel", "dest", "side", "anchor", "from_wheel_end_cm", "from_hinge_side_cm", "yaw", "orient",
             "to", "by", "value", "hold", "times", "into", "style", "seconds", "text", "code"]


def _schema():
    props = {k: {"type": "STRING", "nullable": True} for k in ARG_NAMES}
    for k in ("from_wheel_end_cm", "from_hinge_side_cm", "yaw", "hold", "seconds"):
        props[k] = {"type": "NUMBER", "nullable": True}
    for k in ("times", "by"):
        props[k] = {"type": "INTEGER", "nullable": True}
    verbs = sorted(VERBS) + ["pack", "unpack", "lock", "unlock"]
    return {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "verb": {"type": "STRING", "enum": verbs},
        "args": {"type": "OBJECT", "properties": props}}, "required": ["verb", "args"]}}


def _prompt(text, state):
    sp = spec()
    verbs = "\n".join(f"- {v}({', '.join(a['name'] + ':' + a['type'] + ('' if a['required'] else '?') for a in d['signature'])})"
                      for v, d in sp["verbs"].items())
    things = [it.label for it in S.ITEMS] + [p["label"] for p in SUITCASE_PARTS.values()] + \
        [s.label for s in S.STATIONS.values()] + ["the bedroom door"]
    return f"""You convert a person's instructions for packing a suitcase into a list of actions.
Allowed verbs and arguments (Ref arguments are words naming something):
{verbs}
Composite verbs: pack(target, at?, orient?) = fetch the item and put it in the case; unpack(target);
lock() = dock the zip pulls in the lock and scramble the dials; unlock(code).
Rooms: the bedroom, the bathroom, the kitchen, the hall, the living room ("go into the kitchen" = enter).
Argument conventions:
- Copy the user's own words for every Ref argument (e.g. "the shirts", "it"). Do NOT choose between objects
  that the words could mean; never invent object names. Resolve pronouns only to the user's earlier words.
- place: rel is in|on|beside; dest is the suitcase or the item/place it goes on/beside; side is wheel_end|handle_end|front_side|hinge_side;
  a spot inside the case is either anchor (one of wheel_end, handle_end, front_strip, bottle_corner, centre) or
  from_wheel_end_cm / from_hinge_side_cm (item centre measured from the inside of those walls).
- open/close: target is the lid or the zip; 'to' is the opening fraction (0 shut, 1 fully open): "zip it halfway" = close(zip, to=0.5).
- set: lock value is a 3-digit code; trolley handle value is a stop 0 (retracted), 1 or 2.
- turn: a lock dial, either to (a digit) or by (clicks, + up).
- press: hold (seconds), times.
Things in the scene: {', '.join(things)}.
If a step cannot be expressed with these verbs, output verb "say" with args.text = "CANNOT: <the step>".
Instructions: {text}"""


def parse_gemini(text, state=None, key=None, model=None):
    key = key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    body = {"contents": [{"role": "user", "parts": [{"text": _prompt(text, state)}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": _schema()}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                                                                          "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=60) as f:
        out = json.loads(f.read())
    raw = json.loads(out["candidates"][0]["content"]["parts"][0]["text"])
    reqs = []
    for r in raw:
        args = {k: v for k, v in (r.get("args") or {}).items() if v not in (None, "")}
        if "anchor" in args or "from_wheel_end_cm" in args or "from_hinge_side_cm" in args:
            if "anchor" in args:
                args["at"] = dict(anchor=args.pop("anchor"))
            else:
                args["at"] = dict(from_wheel_end_cm=args.pop("from_wheel_end_cm", 100 * (0 - LO.Y0)),
                                  from_hinge_side_cm=args.pop("from_hinge_side_cm", 100 * (0 - LO.X0)))
        if r["verb"] == "say" and str(args.get("text", "")).startswith("CANNOT:"):
            reqs.append(dict(verb="?", args=dict(text=args["text"][7:].strip())))
            continue
        reqs.append(dict(verb=r["verb"], args=args))
    return reqs


# ---------------------------------------------------------------- execution-time expansion
def reach(a, st):
    """The go_to needed before `a` (None if already in reach)."""
    v, t = a.verb, a.target
    upright = st.case_pose == "upright_floor"
    case_station = "case_floor" if upright else "suitcase"
    if v == "pick_up":
        if st.loc.get(t) == "case":
            return None if st.station == "suitcase" else "suitcase"
        if st.loc.get(t, "").startswith("station:"):
            return None if st.facing == t else t
        return None
    if v == "place":
        dest = a.get("dest", "suitcase")
        if a.get("rel", "in") == "in" and dest in S.STATIONS and dest != "suitcase":
            return None if st.station == dest else dest
        return None if st.station == "suitcase" else "suitcase"
    if v == "put_back" and st.holding:
        home = S.item(st.holding).station
        return None if st.station == home else home
    if v in ("fold", "unfold"):
        return None if st.station == S.CLOTH["station"] else S.CLOTH["station"]
    if (v in ("open", "close") and t == "lid") or v == "stand_up" or (v in ("set", "press") and t.startswith("trolley") and not upright):
        return None if st.station == "bed_foot" else "bed_foot"
    if v in ("open", "close", "insert", "turn", "set", "press", "fasten", "unfasten", "roll", "lay_down"):
        return None if st.station == case_station else "suitcase"
    return None


class Instructions:
    """A list of requests turned into actions one at a time against the live state.
    next(state) -> (Action, label, '') | ('rejected', Rejection, request) | None."""

    def __init__(self, requests, auto_reach=True):
        self.todo = list(requests)
        self.pending = []           # primitives of an expanded macro / inserted go_to
        self.auto_reach = auto_reach
        self.last = ""
        self.current = None

    def _expand(self, req, st):
        v, args = req["verb"], dict(req.get("args", {}))
        if v == "pack":
            g = ground(dict(verb="pick_up", args=dict(target=args["target"])), st, self.last)
            if isinstance(g, Rejection):
                return g
            item = g.target
            pa = dict(target=item, rel=args.get("rel", "in"), dest=args.get("dest", "suitcase"))
            for k in ("at", "orient", "yaw", "side"):
                if k in args:
                    pa[k] = args[k]
            if item == "bottle" and "orient" not in pa:
                pa["orient"] = "upright"
            if st.holding == item:
                return [dict(verb="place", args=pa)]
            return [dict(verb="pick_up", args=dict(target=item)), dict(verb="place", args=pa)]
        if v == "unpack":
            g = ground(dict(verb="pick_up", args=dict(target=args["target"])), st, self.last)
            if isinstance(g, Rejection):
                return g
            return [dict(verb="pick_up", args=dict(target=g.target)), dict(verb="put_back", args=dict(target=g.target))]
        if v == "lock":
            out = []
            if not st.zipped:
                out.append(dict(verb="close", args=dict(target="main_zip", to=0.0)))
            out.append(dict(verb="insert", args=dict(target="zip_pulls", into="lock")))
            out.append(dict(verb="turn", args=dict(target="dial_1", by=3)))
            return out
        if v == "unlock":
            code = args.get("code")
            out = []
            if code:
                out.append(dict(verb="set", args=dict(target="lock", value=code)))
            out.append(dict(verb="press", args=dict(target="lock_button")))
            return out
        if v == "?":
            return Rejection("unknown_verb", f"no action matches '{args.get('text')}'")
        return [req]

    def next(self, st):
        while True:
            if not self.pending:
                if not self.todo:
                    return None
                self.current = self.todo.pop(0)
                ex = self._expand(self.current, st)
                if isinstance(ex, Rejection):
                    return ("rejected", ex, self.current)
                self.pending = ex
            req = self.pending.pop(0)
            if isinstance(req, Action):
                return (req, "nominal", "")
            g = ground(req, st, self.last)
            if isinstance(g, Rejection):
                self.pending = []
                return ("rejected", g, req)
            if g.verb == "place" and not g.target:
                g.target = st.holding
            if g.target in ITEM_NAMES:
                self.last = g.target
            if self.auto_reach:
                where = reach(g, st)
                if where:
                    self.pending.insert(0, g)
                    g = Action("go_to", where)
                if g.verb in ("go_to", "enter"):
                    # out of view: room by room, then look for it, then walk to it
                    steps = navigation(st, g)
                    if not steps:
                        continue
                    self.pending[0:0] = steps[1:]
                    g = steps[0]
            return (g, "nominal", "")
