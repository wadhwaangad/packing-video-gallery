"""Grounding tests: instructions -> actions (symbolic only, no physics).

Each case is run against a fresh PackSim; the expected outcome is either the list of
action texts or a rejection code / precondition feedback.
Usage: python test_grounding.py
"""
from packsim import PackSim
from actionspace import Rejection
import instruct as I
import discrete as D
import episodes as E


def run(text, task="weekend_trip", setup=(), auto_reach=True):
    sim = PackSim(task)
    for s in setup:
        sim.step(s)
    ins = I.Instructions(I.parse_rules(text), auto_reach=auto_reach)
    out = []
    for _ in range(60):
        n = ins.next(sim.state)
        if n is None:
            break
        if n[0] == "rejected":
            out.append(f"REJECT {n[1].code}: {n[1].message}")
            continue
        a = n[0]
        assert D.to_id(a), f"not a catalogue action: {a}"
        ok, msg = sim.step(a)
        out.append(a.text() if ok else f"REFUSED {a.text()}: {msg}")
    return out


CASES = [
    "pack the t-shirts",
    "pick up the tees and put them in the case at the handle end",
    "pick up the shirts",
    "pack the striped shirts in the suitcase, 20 cm from the wheel end and 10 cm from the hinge side",
    "pack the trousers. pack the sweatshirts on top of the trousers",
    "pick up the trousers then put them on the sweatshirts",
    "pack the bottle lying down",
    "pack the camera, then pack the towel on the camera",
    "pack the trousers in the suitcase at the front",
    "pack the sweater",
    "grab the clothes",
    "pack the sweatpants and socks",
    "zip the case",
    "close the lid halfway",
    "close the lid then zip it halfway then unzip it",
    "close the lid. zip it up. lock the case. unlock with code 000",
    "close the lid. zip it. lock it. unzip it",
    "close the lid. zip it. insert the zip pulls. set the lock to 3-7-2. press the lock button",
    "close the lid. turn the left dial to 7. turn the middle dial up 2. turn the lock dial to 3",
    "extend the trolley handle to the top stop",
    "extend the handle",
    "set the trolley handle to 5",
    "press the handle button twice",
    "fold the towel",
    "close the lid. zip it. stand it up. extend the trolley handle to stop 2. roll it to the bedroom door",
    "pick up the bottle on the counter",
    "pick up the pile on the dresser",
    "unfasten the strap",
    "close the lid then unbuckle the strap then buckle the strap",
    "dance around",
    "press the lid",
    "open the suitcase",
    # navigation: one go_to reaches only what is in this room or in view
    "go to the water bottle",
    "go to the kitchen. pick up the bottle on the counter",
    "enter the kitchen",
    "walk into the living room then go to the camera",
    "go to the bedroom",
    "go into the bathroom then pick up the toiletry bag then pack it",
]
STRICT = [  # auto_reach off: nothing is inserted or expanded (evaluating an agent)
    "go to the water bottle",
    "walk into the living room. walk into the hall. go to the water bottle",
    "enter the kitchen",
]

def check_catalogue(seeds=range(6)):
    """Every id round-trips; every generated action is a catalogue id that legal() listed."""
    for e in D.CATALOGUE:
        assert D.to_id(D.action(e["id"])) == e["id"], e["id"]
    n = 0
    for task in sorted(E.TASKS):
        for seed in seeds:
            case = E.sample_case(task=task, seed=seed)
            sim = PackSim(case["task"])
            events = [E.Event(**{k: v for k, v in x.items() if k in E.Event.__dataclass_fields__}) for x in case["events"]]
            user = E.User(case["task"], __import__("random").Random(case["seed"]), events)
            while (nxt := user.next(sim.state)) is not None:
                a = nxt[0]
                aid = D.to_id(a)
                assert aid, a
                assert aid.startswith(("ask(", "announce(")) or aid in D.legal(sim), (task, seed, aid)
                assert sim.step(a)[0]
                n += 1
    print(f"catalogue: {len(D.CATALOGUE)} actions, all round-trip; {n} generated actions all listed by legal()")


if __name__ == "__main__":
    check_catalogue()
    for c in CASES:
        print(">>", c)
        for line in run(c):
            print("     ", line)
    for c in STRICT:
        print(">> [strict]", c)
        for line in run(c, auto_reach=False):
            print("     ", line)
