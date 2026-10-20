"""Grounding tests: instructions -> actions (symbolic only, no physics).

Each case is run against a fresh PackSim; the expected outcome is either the list of
action texts or a rejection code / precondition feedback.
Usage: python test_grounding.py
"""
from packsim import PackSim
from actionspace import Rejection
import instruct as I


def run(text, task="weekend_trip", setup=()):
    sim = PackSim(task)
    for s in setup:
        sim.step(s)
    ins = I.Instructions(I.parse_rules(text))
    out = []
    for _ in range(60):
        n = ins.next(sim.state)
        if n is None:
            break
        if n[0] == "rejected":
            out.append(f"REJECT {n[1].code}: {n[1].message}")
            continue
        a = n[0]
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
]

if __name__ == "__main__":
    for c in CASES:
        print(">>", c)
        for line in run(c):
            print("     ", line)
