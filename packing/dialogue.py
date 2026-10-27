"""Dialogue: the closed set of speech acts on the conversation channel, and how the
simulated user understands what an assistant says.

The conversation runs beside the action stream. An utterance is a speech act, an id from
a closed catalogue like the physical actions in discrete.py, plus its surface text. The
id carries the meaning and is what evaluation reads. The text is sampled from templates
in the persona's style, so the same act is not always worded the same way.

  user -> assistant (USER_ACTS): the task and preferences, questions (each with an answer
      computed from the trajectory: is it on the list, is it packed, what is left, how
      long ago), proactive requests (a reminder, a notice when the list is done), plan
      changes, and replies to guidance (accept, decline, already done, clarify, problem).
  assistant -> user (GUIDANCE): the acts the user can understand. Assistants speak free
      text; interpret() grounds it into these acts with the instruction grounder's
      resolver (actionspace.resolve_ref for items, instruct.parse_rules for commands),
      so anything an assistant says is either a list of guidance acts or `unclear`.
"""
import re
import scene as S
import rooms as R
from actionspace import resolve_ref, Rejection, PRONOUNS, STOP, _tok
import instruct as IN

ITEMS = [it.name for it in S.ITEMS]


def lab(n):
    return S.item(n).label


PLURAL = {"trousers", "striped_shirts", "sweatshirts", "gym_clothes"}    # labels that take a plural verb


def be(n):
    return "are" if n in PLURAL else "is"


def items_text(names):
    labs = [f"the {lab(n)}" for n in names]
    return labs[0] if len(labs) == 1 else ", ".join(labs[:-1]) + " and " + labs[-1] if labs else "nothing"


# ---------------------------------------------------------------- user -> assistant
# id template -> {style: [surface templates]}. Arguments are enumerated (items, tasks,
# spots, add|remove); {feedback} is the simulator's own refusal message.
USER_ACTS = {
    "state_task": dict(terse=["Packing for {title}: {items}."],
                       chatty=["Hi! I'm packing for {title}. I need {items}.",
                               "Okay, {title}. On my list: {items}."]),
    "state_pref": dict(terse=["{Item} {go} at the {spot}."],
                       chatty=["I like the {item} at the {spot} of the case.", "I'd like the {item} at the {spot}."]),
    "ask_should_pack": dict(terse=["{Do} the {item} go in?"],
                            chatty=["Should the {item} go in too?", "Do I need the {item} as well?"]),
    "ask_packed": dict(terse=["Did I pack the {item}?"],
                       chatty=["Did I already put the {item} in?", "{Be} the {item} in the case already?"]),
    "ask_left": dict(terse=["What's left?"], chatty=["What do I still need to pack?", "What's left on my list?"]),
    "ask_since": dict(terse=["When did the {item} go in?"],
                      chatty=["How long ago did I put the {item} in?", "When did I pack the {item}?"]),
    "ask_elapsed": dict(terse=["How long have I been at this?"],
                        chatty=["How long have I been packing now?", "How long has this taken so far?"]),
    "request_remind": dict(terse=["Remind me about the {item} before I close the lid."],
                           chatty=["Can you remind me to pack the {item} before I close the lid? I'll grab it last.",
                                   "Remind me about the {item} before I shut the case, okay?"]),
    "request_notify_done": dict(terse=["Tell me when everything's in."],
                                chatty=["Let me know when everything on my list is in, then I'll close up.",
                                        "Can you tell me once everything is packed?"]),
    "announce_add": dict(terse=["Taking the {item} too."], chatty=["I'll take the {item} as well.",
                                                                   "Actually, I want the {item} too."]),
    "announce_remove": dict(terse=["Not taking the {item}."], chatty=["Actually, I'll leave the {item} at home.",
                                                                      "Change of plan, the {item} stays here."]),
    "accept": dict(terse=["Right.", "Okay."], chatty=["Oh, right, thanks!", "Good catch, thanks.", "Ah, okay, will do."]),
    "decline": dict(terse=["No, I'm fine.", "I'll keep going."],
                    chatty=["Hmm, I don't think so. I'll do it my way.", "No, I'm pretty sure that's right."]),
    "already_done": dict(terse=["Already did."], chatty=["I already did that.", "That's done already."]),
    "not_yet": dict(terse=["Not yet, there's more."], chatty=["Not yet, I think there's still something left."]),
    "clarify": dict(terse=["What?"], chatty=["Sorry, what do you mean?", "Sorry, I didn't get that."]),
    "report_problem": dict(terse=["Can't: {feedback}."], chatty=["Hmm, that didn't work: {feedback}.",
                                                               "I can't, {feedback}."]),
    "remember": dict(terse=["The {item}!"], chatty=["Oh wait, I forgot the {item}!", "Oh, the {item}, nearly forgot."]),
    "self_correct": dict(terse=["Oops."], chatty=["Oops, not that one.", "Wait, that's not right."]),
    "thanks": dict(terse=["Thanks."], chatty=["Thanks!", "Great, thanks."]),
}

SPOT_WORDS = {"wheel_end": "wheel end", "handle_end": "handle end", "front_strip": "front", "bottle_corner": "corner",
              "centre": "middle"}


def say(act, rng, style="chatty", **kw):
    """Surface text for a user act (sampled by style)."""
    opts = USER_ACTS[act].get(style) or USER_ACTS[act]["chatty"]
    t = rng.choice(opts)
    item = kw.get("item")
    f = dict(kw)
    if item:
        f["item"] = lab(item)
        f["Item"] = "The " + lab(item)
        plural = item in PLURAL
        f.update(go="go" if plural else "goes", Do="Do" if plural else "Does", Be="Are" if plural else "Is")
    if "items" in kw and not isinstance(kw["items"], str):
        f["items"] = items_text(kw["items"])
    if "spot" in kw:
        f["spot"] = SPOT_WORDS.get(kw["spot"], kw["spot"])
    return t.format(**f)


def act_id(act, item="", **_):
    return f"{act}({item})" if item else act


# The ids of the old physical catalogue's speech entries, kept so earlier records still map.
def ask_text(item):
    return f"Should the {lab(item)} go in too?"


def announce_text(change, item):
    return f"I'll take the {lab(item)} as well." if change == "add" else \
        f"Actually, I'll leave the {lab(item)} at home."


def user_act_of_text(text):
    """The act id of a known user utterance (exact template match), else None."""
    for n in ITEMS:
        if text == ask_text(n):
            return f"ask_should_pack({n})"
        for c in ("add", "remove"):
            if text == announce_text(c, n):
                return f"announce_{c}({n})"
    return None


def catalogue():
    """Every user act id (arguments enumerated)."""
    per_item = ["state_pref", "ask_should_pack", "ask_packed", "ask_since", "request_remind", "announce_add",
                "announce_remove"]
    out = []
    for a in USER_ACTS:
        if a in per_item:
            out += [f"{a}({n})" for n in ITEMS]
        else:
            out.append(a)
    return out


# ---------------------------------------------------------------- assistant -> user
# What the simulated user can take from an assistant's words. Each act is a dict
# {"act": kind, ...}; `item` is an item id, `requests` are instruction requests that the
# user grounds against the live state when acting on them.
GUIDANCE = {
    "stop": "stop or wait: the user pauses for one step and listens",
    "dont": "do not do <verb> (to <item>): cancels a queued action of that kind",
    "do": "an instruction (requests grounded by the instruction grounder and canonical())",
    "next": "suggests the next item to fetch",
    "not_on_list": "<item> is not on the packing list",
    "on_list": "<item> is on the packing list",
    "packed": "<item> is already in the case",
    "not_packed": "<item> still has to be packed (also: a reminder)",
    "on_floor": "<item> belongs on the floor of the case, not on top of others",
    "where": "where <item> is (a room or station)",
    "zip_open": "the zip is not fully closed",
    "all_packed": "everything on the list is in the case",
    "answer": "an answer to the user's pending question: yes | no | items | seconds",
    "ack": "an acknowledgement",
    "unclear": "nothing the user can act on",
}

NEG = r"(?:not|n't|no longer|never)"
CUES = [  # (act, pattern) in priority order; matched per clause
    ("not_packed", r"\b(?:forg[eo]t|forgotten|still need|still have to|missing|have(?:n't| not) (?:packed|put)|"
                   r"remember|remind|left to pack|don't leave out|need to pack)\b"),
    ("packed", r"\balready\b(?! *" + NEG + ")"),
    ("on_floor", r"\b(?:bottom|floor|underneath|beneath|lowest|base)\b"),
    ("zip_open", r"\bzip\w*\b.*\b(?:not|isn't|all the way|fully|finish|completely|open|rest of the way)\b|"
                 r"\b(?:not|isn't)\b.*\bzip\w*\b"),
    ("all_packed", r"\b(?:that's|that is|thats) (?:everything|all)\b|\b(?:everything|all)\b.*\b(?:packed|in(?: the case)?|done)\b"),
    ("not_on_list", r"\b(?:isn't|is not|aren't|are not|not|isnt|arent)\b.*\b(?:list|need(?:ed)?|for this trip)\b|"
                    r"\b(?:don't|do not|dont) (?:need|pack|take|bring)\b|\bleave\b|\bnot (?:the|that|those|this)\b|"
                    r"\bwrong (?:one|pile|item|thing)\b|\bno need\b"),
    ("on_list", r"\b(?:is|are) on (?:your|the) list\b|\byou (?:want|need)\b"),
]
DONT_VERBS = [("close", "lid", r"\b(?:close|shut)\b"), ("pick_up", "", r"\b(?:pick|grab|take|lift)\b"),
              ("place", "", r"\b(?:put|place|set|stack)\b"), ("go_to", "", r"\b(?:go|walk|head)\b"),
              ("close", "main_zip", r"\bzip\b")]
YES = r"^(?:yes|yeah|yep|yup|correct|right|sure|it is|it does|they are|you did|you have)\b"
NO = r"^(?:no|nope|not|nah)\b"
NEXT = r"\b(?:get|grab|fetch|next|pack|go for|then)\b"


def mentions(text, st):
    """Items named in the text, in order (longest unambiguous spans; pronouns skipped)."""
    words = re.findall(r"[a-z0-9\-']+", text.lower())
    out, i = [], 0
    LOC = {"on", "in", "at", "from", "near", "by", "inside", "beside", "into", "before", "after", "not", "is", "are"}
    while i < len(words):
        hit = None
        for L in (5, 4, 3, 2, 1):
            span = words[i:i + L]
            if len(span) < L or any(w in LOC for w in span[1:]):
                continue
            toks = _tok(" ".join(span))
            if not toks or any(t in PRONOUNS for t in toks) or toks[-1] in ("list", "case", "suitcase", "lid", "zip"):
                continue
            r = resolve_ref(" ".join(span), ("item",), st)
            if not isinstance(r, Rejection):
                hit = (r[0], L)
                break
        if hit:
            if hit[0] not in out:
                out.append(hit[0])
            i += hit[1]
        else:
            i += 1
    return out


DEICTIC = r"\b(?:it|that|this|those|these|them|they|that's|thats|it's|they're)\b"


def deictic(c, st):
    """'That's not on your list': a pronoun with no item named is the item in the person's
    hands, or else the one they stand in front of."""
    if not re.search(DEICTIC, c):
        return []
    return [n for n in (st.holding or st.facing,) if n]


def _seconds(t):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m)\b", t)
    if m:
        v = float(m.group(1))
        return v * 60 if m.group(2).startswith("m") else v
    for w, v in (("half a minute", 30), ("a minute and a half", 90), ("a couple of minutes", 120), ("a minute", 60),
                 ("a few minutes", 180), ("just now", 5), ("a moment ago", 10)):
        if w in t:
            return float(v)
    return None


def clauses(text):
    return [c.strip() for c in re.split(r"[.;!?]+|,|\s+(?:-|--|—)\s+|\s+but\s+", text.lower()) if c.strip()]


def interpret(text, st, pending=None):
    """Assistant text -> list of guidance acts. `pending` is the user's open question
    (an event type such as 'ask_left'), if any: then the text is first read as its answer."""
    t = text.strip().lower()
    out = []
    if pending:
        a = _answer(t, st, pending)
        if a is not None:
            return [dict(act="answer", value=a)]
    for c in clauses(t):
        names = mentions(c, st) or deictic(c, st)
        done = False
        for act, pat in CUES:
            if re.search(pat, c):
                if act in ("zip_open", "all_packed"):
                    out.append(dict(act=act))
                elif names:
                    out += [dict(act=act, item=n) for n in names]
                else:
                    continue
                done = True
                break
        if done:
            continue
        neg = re.search(r"\b(?:don't|do not|dont|stop|wait|hold on|hang on|not yet|never)\b", c)
        if neg:
            for verb, part, pat in DONT_VERBS:
                if re.search(pat, c):
                    out.append(dict(act="dont", verb=verb, target=part or (names[0] if names else "")))
                    break
            else:
                out.append(dict(act="stop"))
            continue
        if IN.VERB_START.match(re.sub(r"^(?:please|now|next|first|then|so|ok|okay)\s*,?\s*", "", c)):
            reqs = [r for r in IN.parse_rules(c) if r.get("verb") != "?"]
            if reqs:
                out.append(dict(act="do", requests=reqs, text=c))
                continue
        if names and re.search(r"\b(?:is|are|it's|they're)\s+(?:in|on|at)\b", c):
            where = _place_word(c)
            out += [dict(act="where", item=n, place=where) for n in names]
            continue
        if names and re.search(NEXT, c):
            out.append(dict(act="next", item=names[0]))
            continue
        if re.search(r"\b(?:ok|okay|got it|noted|sure|sounds good|will do|understood|great|nice)\b", c):
            out.append(dict(act="ack"))
            continue
    if not out:
        out.append(dict(act="unclear", text=text))
    seen, uniq = set(), []
    for g in out:
        k = (g["act"], g.get("item", ""), g.get("verb", ""), g.get("target", ""))
        if k not in seen:
            seen.add(k)
            uniq.append(g)
    return uniq


def _place_word(c):
    for rm, words in R.WORDS.items():
        if any(w in c for w in words):
            return rm
    for s in S.STATIONS:
        if S.STATIONS[s].label.replace("the ", "") in c:
            return s
    return ""


def _answer(t, st, pending):
    if pending in ("ask_should_pack", "ask_packed"):
        if re.search(YES, t):
            return "yes"
        if re.search(NO, t):
            return "no"
        if re.search(r"\b(?:isn't|aren't|not|haven't|hasn't|didn't|don't|doesn't)\b", t):
            return "no"
        if re.search(r"\b(?:yes|yeah|already|it is|it's in|they're in|it does|you did)\b", t):
            return "yes"
        return None
    if pending in ("ask_since", "ask_elapsed"):
        return _seconds(t)
    if pending == "ask_left":
        if re.search(r"\b(?:nothing|that's everything|that's all|all done|you're done)\b", t):
            return []
        names = [n for n in mentions(t, st)]
        return names or None
    return None
