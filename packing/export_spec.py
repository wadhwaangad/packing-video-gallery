"""Write the simulator's specification (actions, families, items, stations, regions, tasks,
events, personas) to trip/spec.json for the site, straight from the code.

Usage: python packing/export_spec.py
"""
import json
from pathlib import Path
import scene as S
import packsim as P
import episodes as EP

HERE = Path(__file__).resolve().parent

ACTIONS = [
    # verb, class, argument, grounding, preconditions (refusal feedback), physical skill
    ("go_to", "navigate", "<station | item>", "a station, or the stand point in front of an item's current position",
     "target exists; an item already in your hands cannot be a destination",
     "turn in place (wall-aware), walk the room graph, face the station"),
    ("pick_up", "manipulate", "<item>", "a visible item on the station you face",
     "hands free; item is on that station and you are in front of it; items in the case need take_out",
     "two-hand side squeeze, or one-hand power grip for the bottle; lift checked by contact"),
    ("put_in", "manipulate", "<region> [upright|lying]", "a region of the open case; drop height from the piles already there",
     "holding something; at the suitcase; lid open; family accepted by the region; nothing fragile underneath; height fits",
     "lower onto the stack, open the grip, withdraw; landing checked in the region"),
    ("put_back", "manipulate", "-", "the held item's home spot",
     "holding something; standing at its home station", "lower onto the surface, open, withdraw"),
    ("take_out", "manipulate", "<item>", "an item in the case",
     "hands free; at the suitcase; lid open; the item is on top of its stack", "squeeze pick inside the case"),
    ("close_lid", "case", "-", "the lid (from the foot of the bed)", "hands free; lid open; nothing above the rim",
     "hook the lid's end wall, swing past vertical, let gravity shut it"),
    ("zip", "case", "-", "the zip track (from beside the bed)", "lid closed; not zipped",
     "press the lid with both palms, run the zip along the near side; closure checked before locking"),
    ("unzip", "case", "-", "the zip track", "zipped", "run the zip back; hinge released"),
    ("open_lid", "case", "-", "the lid (from the foot of the bed)", "lid closed; not zipped",
     "fingertips under the zip-track lip on the free edge, swing past vertical, let it fall open"),
    ("stand_up", "case", "-", "the trolley handle (from the foot of the bed)", "zipped",
     "two-hand overhand grip on the bar, tested with a short pull (re-grip if it slips); stand the case up with the wrists turning, lift it off the bed, set it on its wheels, let go"),
    ("wait", "other", "<seconds>", "-", "-", "hold still"),
    ("say", "other", "<text>", "-", "-", "utterance logged with its time (questions, plan changes)"),
]


def main():
    scans = {**S.GARMENT_SCANS, "suitcase": S.SUITCASE_SCAN}
    spec = dict(
        actions=[dict(verb=v, cls=c, arg=a, grounds=g, pre=p, skill=k) for v, c, a, g, p, k in ACTIONS],
        families={k: v for k, v in S.FAMILIES.items()},
        items=[dict(name=it.name, label=it.label, family=it.family, station=it.station, room=it.room,
                    size_cm=[round(x * 100, 1) for x in it.size], mass_kg=it.mass, heavy=it.heavy,
                    source=(f"BlenderKit scan: {scans[it.name]['name']} ({scans[it.name]['author']})" if it.name in scans
                            else "Poly Haven model" if it.kind == "asset" else "procedural"))
               for it in S.ITEMS],
        stations=[dict(name=s.name, label=s.label, room=s.room, crouch=s.crouch) for s in S.STATIONS.values()],
        regions=[dict(name=r.name, label=r.label, families=list(r.families)) for r in S.REGIONS.values()],
        tasks=[dict(name=t.name, title=t.title, pack=[S.item(n).label for n in t.pack]) for t in P.TASKS.values()],
        events=[dict(type=k, channel=v[0], description=v[1]) for k, v in EP.EVENT_TYPES.items()],
        personas=[dict(name=k, **v) for k, v in EP.PERSONAS.items()],
        suitcase=dict(outer_cm=[round(x * 100, 1) for x in S.SUITCASE["outer"]],
                      depth_cm=round((S.SUITCASE["base_depth"] + S.SUITCASE["lid_depth"]) * 100, 1),
                      inner_height_cm=round(S.INNER_H * 100, 1), source=f"BlenderKit scan: {S.SUITCASE_SCAN['name']} "
                      f"({S.SUITCASE_SCAN['author']}), scaled {S.SUITCASE_SCAN['scale']}x"),
    )
    out = HERE.parent / "trip" / "spec.json"
    out.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
