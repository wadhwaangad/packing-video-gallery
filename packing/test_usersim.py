"""Checks for the user simulation (symbolic only, a few seconds).

  python packing/test_usersim.py

1. Guidance text -> guidance acts (dialogue.interpret), including paraphrases.
2. Every sampled event type: the silent run is valid and the oracle deals with it.
3. Consequences: a wrong answer packs an extra item; a gullible user follows bad advice;
   an instruction the user cannot carry out comes back as a reported problem.
4. Every step a generated episode takes is a catalogue id that legal() listed (or a
   refusal the event is meant to produce).
5. The model slot: the empty template runs exactly as no assistant, a model sees only
   public things, and what it says is heard like any assistant's words.
6. The user's plan: it starts with every item to pack and ends with every step done.
"""
import random
import dialogue as DL
import episodes as E
import usersim as U
import discrete as D
from packsim import PackSim

INTERPRET = [
    ("Careful, the striped shirts aren't on your list. You want the sweatshirts.", None,
     [("not_on_list", "striped_shirts"), ("on_list", "sweatshirts")]),
    ("those striped shirts don't need to come", None, [("not_on_list", "striped_shirts")]),
    ("Don't forget the camera.", None, [("not_packed", "camera")]),
    ("you haven't packed the water bottle yet", None, [("not_packed", "bottle")]),
    ("Remember the leather notebook before you close the lid.", None, [("not_packed", "notebook")]),
    ("You already packed the sweatshirts.", None, [("packed", "sweatshirts")]),
    ("Put the toiletry bag on the bottom of the case, not on top of the clothes.", None, [("on_floor", "toiletry_bag")]),
    ("Don't close the lid yet, you still need the camera.", None, [("dont", "lid"), ("not_packed", "camera")]),
    ("wait!", None, [("stop", "")]),
    ("The zip isn't closed all the way.", None, [("zip_open", "")]),
    ("That's everything on your list.", None, [("all_packed", "")]),
    ("The folded towel is in the bathroom.", None, [("where", "towel")]),
    ("Next you could get the T-shirt pile.", None, [("next", "tshirts")]),
    ("pack the camera in the front strip", None, [("do", "")]),
    ("Okay, noted.", None, [("ack", "")]),
    ("purple elephants", None, [("unclear", "")]),
    ("No, it's not on your list.", "ask_should_pack", [("answer", "no")]),
    ("Yes, it's already in.", "ask_packed", [("answer", "yes")]),
    ("About 40 seconds ago.", "ask_since", [("answer", 40.0)]),
    ("roughly 3 minutes so far", "ask_elapsed", [("answer", 180.0)]),
    ("You still need the camera and the leather notebook.", "ask_left", [("answer", ["camera", "notebook"])]),
]


def check_interpret():
    st = PackSim("weekend_trip").state
    bad = []
    for text, pending, want in INTERPRET:
        got = DL.interpret(text, st, pending)
        key = [(g["act"], g.get("value", g.get("item", g.get("target", "")))) for g in got]
        exp = [(a, v) for a, v in want]
        norm = [(a, "" if a in ("do", "stop", "ack", "unclear", "zip_open", "all_packed") else v) for a, v in key]
        if norm != exp:
            bad.append((text, norm, exp))
    for b in bad:
        print("   interpret MISMATCH", b)
    print(f"interpret: {len(INTERPRET) - len(bad)}/{len(INTERPRET)}")
    return not bad


def check_events(seeds=range(3)):
    ok = True
    for t in E.SAMPLED:
        res = {}
        for seed in seeds:
            for p in U.PERSONAS:
                c = E.sample_case(seed=seed, allow=[t], n_events=1, persona=p)
                e = c["events"][0]["id"]
                k = (c["dry_run"]["outcomes"][e], c["oracle"]["outcomes"][e])
                res[k] = res.get(k, 0) + 1
        print(f"   {t:20s} silent -> oracle: {res}")
        ok &= all(o in E.GOOD for _, o in res)
    return ok


def check_consequences():
    ok = True
    # a wrong answer to 'should the X go in?' makes the user pack it
    case = E.sample_case(seed=1, allow=["ask_should_pack"], n_events=1, persona="steady")
    ev = case["events"][0]

    class Liar(U.Silent):
        def respond(self, obs, h):
            return ["Yes, take it."] if h.pending_q else []
    h = E.run(case, Liar())
    extra = not h.b.sim.goals()["no_extra_items"] and h.events[0].outcome == "answered_wrong"
    print(f"   wrong answer packs {ev['other']}: {extra}")
    ok &= extra
    # bad advice: a gullible user drops an item it was told is not on the list
    U.PERSONAS["_gullible"] = dict(U.PERSONAS["steady"], gullibility=1.0)
    case = dict(task="weekend_trip", seed=5, persona="_gullible", events=[])
    lines = [dict(step=2, text="The camera isn't on your list.")]
    h = E.run(case, U.Scripted(lines))
    del U.PERSONAS["_gullible"]
    took = h.user.omitted == {"camera"} and not h.b.sim.goals()["packed(camera)"]
    said = lambda h: next(c for c in h.conv if c["speaker"] == "assistant")["responses"][0]
    acc = said(h)
    ok &= took
    print(f"   gullible user told a lie about the camera: accepted={acc.get('accepted')} -> camera left out={took}")
    h2 = E.run(dict(case, persona="deliberate"), U.Scripted(lines))
    print(f"   deliberate user, same lie: accepted={said(h2).get('accepted')} "
          f"goals met={h2.b.sim.done()}")
    ok &= h2.b.sim.done()
    # an instruction that cannot be carried out comes back as a reported problem
    h = E.run(dict(case, persona="steady"), U.Scripted([dict(step=3, text="zip the case")]))
    rep = [c["text"] for c in h.conv if c["speaker"] == "user" and c["act"] == "report_problem"]
    print(f"   impossible instruction -> user says: {rep[:1]}")
    ok &= bool(rep) and h.b.sim.done()
    return ok


def check_legal(seeds=range(4)):
    class Checked(U.SymbolicBackend):
        def execute(self, a, label, eid):
            aid = D.to_id(a)
            assert aid, a
            ok = aid in D.legal(self.sim)
            rec = super().execute(a, label, eid)
            assert ok == rec["ok"], (aid, rec["feedback"])
            return rec
    n = 0
    for seed in seeds:
        case = E.sample_case(seed=seed)
        for who in (U.Silent(), U.Oracle()):
            h = U.Harness(case, Checked(case["task"], U.PERSONAS[case["persona"]]), who).run()
            assert not E.problems(h), E.problems(h)
            n += len(h.steps)
    print(f"legal: {n} steps, each a catalogue id, refused exactly when legal() did not list it")
    return True


def check_model_slot(seed=3):
    from pathlib import Path
    case = E.sample_case(seed=seed)
    empty = U.make_assistant(str(Path(__file__).with_name("assistant_template.py")))
    a, b = E.run(case, empty), E.run(case, U.Silent())
    same = [r["action"]["text"] for r in a.b.records] == [r["action"]["text"] for r in b.b.records] and         [c["text"] for c in a.conv] == [c["text"] for c in b.conv]
    views, oracle = [], U.Oracle()
    def fn(view):                       # a "model" that copies the oracle, to see its words land
        views.append(view)
        return oracle.respond(view["now"], h)
    h = U.Harness(case, U.SymbolicBackend(case["task"], U.PERSONAS[case["persona"]]), U.Model(fn))
    h.run()
    want = {"task", "preferences", "conversation", "now", "history"}
    private = {"label", "event", "source", "controller", "commit", "beliefs", "plan", "schedule", "events"}
    public = all(set(v) == want and not private & set(v["now"]) and
                 all(set(c) == {"t", "speaker", "text"} for c in v["conversation"]) for v in views)
    heard = [e.outcome for e in h.events] == [e.outcome for e in E.run(case, U.Oracle()).events]
    ok = same and public and heard and len(views) >= len(h.steps) + 1
    print(f"model slot: empty == silent {same}, public view only {public}, words heard {heard}, "
          f"called {len(views)}x for {len(h.steps)} actions")
    return ok


def check_plan(seed=3):
    case = E.sample_case(seed=seed)
    h = U.Harness(case, U.SymbolicBackend(case["task"], U.PERSONAS[case["persona"]]), U.Silent()).begin()
    first = h.user.plan(h.st)
    items = [p["item"] for p in first if p["item"]]
    ok = sorted(items) == sorted(h.st.pack) and first[0]["status"] == "next" and         all(p["status"] == "todo" for p in first[1:])
    while h.tick():
        plan = h.user.plan(h.st)
        ok &= sum(p["status"] == "next" for p in plan) <= 1
    h.end()
    ok &= all(p["status"] == "done" for p in h.user.plan(h.st))
    print(f"plan: {len(first)} steps, {len(items)} items, all done at the end: {ok}")
    return ok


if __name__ == "__main__":
    results = [check_interpret()]
    print("events:")
    results.append(check_events())
    print("consequences:")
    results.append(check_consequences())
    results.append(check_legal())
    results.append(check_model_slot())
    results.append(check_plan())
    print("ALL OK" if all(results) else "FAILURES")
