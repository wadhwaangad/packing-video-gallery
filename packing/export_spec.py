"""Write the simulator's specification to trip/spec.json for the site, straight from the code:
the action space (verbs, typed arguments, how each is grounded and executed), every
asset's operable parts and their value ranges, the placement rules, rejection codes with
live grounding examples, items, families, tasks, events and personas.

Usage: python packing/export_spec.py
"""
import json
from pathlib import Path
import numpy as np
import scene as S
import layout as LO
import packsim as P
import episodes as EP
import actionspace as AS
import instruct as I

HERE = Path(__file__).resolve().parent

# Instructions run through the parser, grounder and symbolic preconditions (no physics):
# what each turns into, or why it is rejected / refused.
EXAMPLES = [
    ("pack the t-shirts", {}),
    ("pick up the tees and put them in the case at the handle end", {}),
    ("pack the striped shirts in the suitcase, 30 cm from the wheel end and 15 cm from the hinge side", {}),
    ("pack the trousers. pack the sweatshirts on top of the trousers", {}),
    ("pack the camera, then pack the folded towel on the camera", {}),
    ("pack the bottle lying down", {}),
    ("pick up the shirts", {}),
    ("grab the clothes", {}),
    ("close the lid halfway", {}),
    ("close the lid. zip it halfway", {}),
    ("close the lid. zip it. insert the zip pulls. set the lock to 3-7-2. press the lock button", {}),
    ("close the lid. turn the left dial to 7. turn the lock dial to 3", {}),
    ("set the trolley handle to 5", {}),
    ("extend the handle to the top stop", {}),
    ("close the lid. zip it. stand the case up. extend the handle to the top stop. roll the case to the bedroom door", {}),
    ("unfasten the strap", {}),
    ("fold the beach towel in half", {"cloth": True}),
    ("fold the beach towel in thirds", {"cloth": True}),
    ("fold the t-shirts", {"cloth": True}),
    ("press the lid", {}),
    ("dance around", {}),
]


def run_example(text, cloth=False):
    sim = P.PackSim("weekend_trip", cloth=cloth)
    reqs = I.parse_rules(text)
    ins = I.Instructions(reqs)
    out = []
    for _ in range(60):
        n = ins.next(sim.state)
        if n is None:
            break
        if n[0] == "rejected":
            out.append(dict(kind="rejected", code=n[1].code, message=n[1].message))
            continue
        a = n[0]
        ok, msg = sim.step(a)
        d = a.to_dict()
        if a.verb == "place" and ok and a.target in sim.state.place:
            pl = sim.state.place[a.target]
            d["resolved"] = dict(from_wheel_end_cm=round(100 * (pl["y"] - LO.Y0), 1),
                                 from_hinge_side_cm=round(100 * (pl["x"] - LO.X0), 1),
                                 yaw_deg=round(float(np.degrees(pl["yaw"])), 0),
                                 rests_on=[S.item(o).label for o in pl.get("on", ())] or ["case floor"])
        out.append(dict(kind="action" if ok else "refused", action=d, feedback=msg))
    return dict(text=text, requests=reqs, cloth=cloth, result=out)


def main():
    scans = {**S.GARMENT_SCANS, "suitcase": S.SUITCASE_SCAN}
    sp = AS.spec()
    spec = dict(
        action_space=sp,
        examples=[run_example(t, **kw) for t, kw in EXAMPLES],
        families={k: v for k, v in S.FAMILIES.items()},
        items=[dict(name=it.name, label=it.label, family=it.family, station=it.station, room=it.room,
                    size_cm=[round(x * 100, 1) for x in it.size], mass_kg=it.mass, heavy=it.heavy,
                    words=AS.ITEM_WORDS.get(it.name, {}),
                    source=(f"BlenderKit scan: {scans[it.name]['name']} ({scans[it.name]['author']})" if it.name in scans
                            else "Poly Haven model" if it.kind == "asset" else "procedural"))
               for it in S.ITEMS],
        cloth=dict(name=S.CLOTH["name"], label=S.CLOTH["label"], station=S.CLOTH["station"],
                   size_cm=[round(100 * x) for x in S.CLOTH["size"]], vertices=list(S.CLOTH["count"]),
                   mass_kg=S.CLOTH["mass"]),
        stations=[dict(name=s.name, label=s.label, room=s.room, crouch=s.crouch) for s in S.STATIONS.values()],
        floor_spots={k: dict(label=v["label"]) for k, v in P.FLOOR_SPOTS.items()},
        placement=dict(grid_mm=round(LO.GRID * 1000), support_fraction=LO.SUPPORT_FRAC, support_tol_mm=round(LO.SUPPORT_TOL * 1000),
                       headroom_mm=round(LO.HEADROOM * 1000), lid_margin_mm=round(LO.LID_MARGIN * 1000),
                       palm_strip_mm=round(LO.PALM_STRIP * 1000), bottle_strip_mm=round(LO.BOTTLE_STRIP * 1000)),
        tasks=[dict(name=t.name, title=t.title, pack=[S.item(n).label for n in t.pack]) for t in P.TASKS.values()],
        events=[dict(type=k, channel=v[0], description=v[1]) for k, v in EP.EVENT_TYPES.items()],
        personas=[dict(name=k, **v) for k, v in EP.PERSONAS.items()],
        suitcase=dict(outer_cm=[round(x * 100, 1) for x in S.SUITCASE["outer"]],
                      depth_cm=round((S.SUITCASE["base_depth"] + S.SUITCASE["lid_depth"]) * 100, 1),
                      inner_height_cm=round(S.INNER_H * 100, 1), source=f"BlenderKit scan: {S.SUITCASE_SCAN['name']} "
                      f"({S.SUITCASE_SCAN['author']}), scaled {S.SUITCASE_SCAN['scale']}x",
                      trolley_stops_cm=[round(100 * q) for q in S.HARDWARE["trolley"]["levels"]]),
    )
    out = HERE.parent / "trip" / "spec.json"
    out.write_text(json.dumps(spec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("wrote", out, len(spec["examples"]), "examples")


if __name__ == "__main__":
    main()
