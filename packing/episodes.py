"""Episode construction: sampling and validating cases for the user simulation.

A case is a task, a seed, a persona, optional stated preferences, and a private event
schedule. The simulated user, the events, and the harness that runs them are in
usersim.py. This module samples schedules and keeps only those that work as a test:

  * with a silent assistant (dry_run) every event fires, the user finishes, every action
    is in the catalogue, and nothing is refused except the steps an error is meant to
    make fail (an incomplete zip found out when the case will not stand up);
  * with the oracle assistant every event can be dealt with: errors prevented or
    corrected, questions answered correctly, reminders and notices delivered, plan
    changes acknowledged, and the goals reached. So a trigger that can never be acted
    on, or a window too short to speak in, is not sampled;
  * distinct events do not overlap (at least one ordinary step between their windows).
"""
import json, random
import scene as S
from packsim import TASKS
import discrete as D
import usersim as U
from usersim import PERSONAS, EVENT_TYPES, Event, pack_order, choose_spot, topmost_over   # noqa: F401 (re-exported)

GOOD = {"prevented", "corrected", "self_recovered", "answered_correct", "reminded", "reminded_late", "notified",
        "acknowledged"}
SAMPLED = [t for t in EVENT_TYPES if t != "bottle_lying"]   # bottle_lying: not reliable in physics yet


def run(case, assistant=None, max_steps=220, verbose=False):
    """The case run symbolically (PackSim with the duration clock). Returns the harness."""
    b = U.SymbolicBackend(case["task"], PERSONAS[case["persona"]])
    return U.Harness(case, b, assistant, max_steps=max_steps, verbose=verbose).run()


def trace_of(h):
    return [dict(action=r["action"], label=r["label"], event=r["event"], source=r.get("source", ""), ok=r["ok"],
                 feedback=r["feedback"], t=r["t_start"]) for r in h.b.records]


def problems(h):
    """Why a run is not a valid episode ([] if it is)."""
    out = []
    if len(h.steps) >= h.max_steps:
        out.append("did not finish")
    out += [f"event {e.id} ({e.type}) never fired" for e in h.events if not e.fired]
    for r in h.b.records:
        if not D.to_id(U.Action.from_dict(r["action"])):
            out.append(f"not in the catalogue: {r['action']['text']}")
        if not r["ok"] and r.get("source") == "plan" and r["label"] != "error":
            out.append(f"refused: {r['action']['text']} ({r['feedback']})")
    return out


def dry_run(case, max_steps=220, assistant=None):
    """Run a case symbolically with a silent assistant. Returns (ok, trace, sim, harness)."""
    h = run(case, assistant, max_steps)
    return not problems(h) and _windows_separated(trace_of(h)), trace_of(h), h.b.sim, h


def oracle_check(case):
    """(ok, harness): with an assistant that knows the schedule, every event is dealt with."""
    h = run(case, U.Oracle())
    ok = not problems(h) and all(e.outcome in GOOD for e in h.events) and h.b.sim.done() and \
        all(h.preference_goals().values())
    return ok, h


def _distractor_near(item, task, rng):
    pack = set(TASKS[task].pack)
    it = S.item(item)
    cands = [o for o in S.ITEMS if o.name not in pack and o.family == it.family]
    same_room = [o for o in cands if o.room == it.room]
    pool = same_room or cands
    return rng.choice(pool).name if pool else ""


def _event(i, ty, tk, pack, order, used, rng):
    """One scheduled event of type ty, or None if this task has nothing for it."""
    e = dict(id=f"e{i + 1}", type=ty, target="", other="", after=-1, recover=rng.random() < 0.7, text="")
    free = [n for n in pack if n not in used]
    if ty in ("wrong_object", "omission", "nav_deviation", "repetition"):
        if not free:
            return None
        e["target"] = rng.choice(free)
        if ty == "wrong_object":
            e["other"] = _distractor_near(e["target"], tk, rng)
            if not e["other"]:
                return None
        if ty == "nav_deviation":
            e["other"] = rng.choice([s for s in S.STATIONS if S.STATIONS[s].room != S.item(e["target"]).room
                                     and s not in ("suitcase", "bed_foot")])
        used.add(e["target"])
    elif ty == "heavy_on_top":
        if "toiletry_bag" not in free:
            return None
        e["target"] = "toiletry_bag"; used.add("toiletry_bag")
    elif ty == "incomplete_zip":
        e["recover"] = True                 # found out when the case will not stand up or lock
    else:
        e["after"] = rng.randint(1, max(1, len(pack) - 2))
        e["recover"] = ty in ("stall",) or (ty == "premature_close" and e["recover"])
        if ty == "ask_should_pack":
            extra = [o.name for o in S.ITEMS if o.name not in pack]
            e["other"] = rng.choice(extra)
        elif ty == "ask_packed":
            e["target"] = rng.choice(pack)
        elif ty == "ask_since":
            e["target"] = rng.choice(order[:e["after"]])
        elif ty == "request_remind":
            cand = [n for n in order[e["after"]:] if n in free and not S.item(n).heavy]
            if not cand:
                return None
            e["target"] = rng.choice(cand); used.add(e["target"])
            e["recover"] = False            # it is what the reminder is for
        elif ty == "plan_change":
            removable = [n for n in free if not S.item(n).heavy]
            addable = [o.name for o in S.ITEMS if o.name not in pack and o.family == "folded_pile"]
            if rng.random() < 0.5 and removable:
                it = rng.choice(removable); e["other"] = f"remove:{it}"
            elif addable:
                it = rng.choice(addable); e["other"] = f"add:{it}"
            else:
                return None
            used.add(it)
    return e


def sample_case(task=None, seed=0, n_events=None, persona=None, allow=None, prefs=None, check_oracle=True):
    """Sample a case that passes the silent dry run and, by default, the oracle check."""
    for attempt in range(300):
        rng = random.Random(seed * 1000 + attempt)
        tk = task or rng.choice(sorted(TASKS))
        pack = list(TASKS[tk].pack)
        order = pack_order(tk, random.Random(seed * 1000 + attempt))
        k = n_events if n_events is not None else rng.choice([2, 2, 3, 3, 4])
        types = [t for t in SAMPLED if allow is None or t in allow]
        evs, used = [], set()
        for i in range(k):
            fresh = [t for t in types if t not in {x["type"] for x in evs}] or types
            e = _event(len(evs), rng.choice(fresh), tk, pack, order, used, rng)
            if e:
                evs.append(e)
        if len(evs) < k:
            continue                        # this task has nothing for one of the events
        n_pref = prefs if prefs is not None else rng.choice([0, 0, 1])
        pref = []
        if n_pref:
            piles = [n for n in pack if S.item(n).family == "folded_pile" and n not in used]
            if piles:
                pref = [(rng.choice(piles), rng.choice(["wheel_end", "handle_end"]))]
        case = dict(task=tk, seed=seed * 1000 + attempt, persona=persona or rng.choice(sorted(PERSONAS)),
                    preferences=pref, events=[Event.from_dict(e).spec() for e in evs])
        ok, trace, sim, h = dry_run(case)
        if not ok:
            continue
        if check_oracle:
            ok_o, ho = oracle_check(case)
            if not ok_o:
                continue
            case["oracle"] = dict(outcomes={e.id: e.outcome for e in ho.events}, goals_satisfied=ho.b.sim.done(),
                                  actions=len(ho.steps), seconds=round(ho.b.time, 1))
        case["dry_run"] = dict(actions=len(trace), seconds=round(h.b.time, 1), final_goals=sim.goals(),
                               outcomes={e.id: e.outcome for e in h.events})
        return case
    raise RuntimeError("could not sample a valid case")


def persona_variants(case):
    """The case once per persona (as in a benchmark that pairs each case with every
    persona); each variant is validated separately, since patience and compliance change
    the run. Returns {persona: case or None if it does not validate}."""
    out = {}
    for p in PERSONAS:
        c = dict({k: v for k, v in case.items() if k not in ("dry_run", "oracle")}, persona=p)
        ok, _, _, _ = dry_run(c)
        out[p] = c if ok and oracle_check(c)[0] else None
    return out


def _windows_separated(trace):
    """Distinct events must not overlap: at least one plan step between them."""
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
    a = sys.argv[1:]
    case = reference_case() if not a else sample_case(seed=int(a[0]))
    who = a[1] if len(a) > 1 else "none"
    h = run(case, U.make_assistant(who), verbose=True)
    print(json.dumps({k: v for k, v in case.items() if k not in ("dry_run", "oracle")}, indent=1))
    print("problems:", problems(h))
    print("outcomes:", {e.id: (e.type, e.outcome) for e in h.events})
    print("goals:", {k: v for k, v in h.b.sim.goals().items() if not v} or "all satisfied", h.preference_goals())
