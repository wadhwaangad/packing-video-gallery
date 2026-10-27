"""Write the browser playground's data (play/).

Usage:
  python packing/export_web.py                       # world.json, modules.json, symbolic replays
  python packing/export_web.py --physics RUN NAME    # a rendered physics run (videos + frames)

Replays: sampled cases (every sampled event type at least once), each run twice, with a
silent assistant and with the oracle, so the page can show what the user does on its own
and what changes when someone speaks up. Rendered episodes: a physics run from generate.py
whose videos were made with render_all.py; its frames carry the video time of each action.
"""
import argparse, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import web
import episodes as EP

SITE = HERE.parent / "play"
# The modules the browser loads (the symbolic simulator; no MuJoCo, no Blender). test_web.py
# checks that web.py imports from these files alone.
MODULES = ["scene.py", "rooms.py", "layout.py", "actionspace.py", "packsim.py", "discrete.py", "instruct.py",
           "dialogue.py", "usersim.py", "episodes.py", "web.py"]

# (task, persona, event types): pairs chosen so every sampled type appears once.
REPLAYS = [
    ("weekend_trip", "steady", ["wrong_object", "ask_since"]),
    ("gym_weekend", "brisk", ["request_remind", "premature_close"]),
    ("beach_holiday", "deliberate", ["omission", "ask_should_pack"]),
    ("work_trip", "brisk", ["heavy_on_top", "ask_left"]),
    ("winter_visit", "steady", ["nav_deviation", "plan_change"]),
    ("weekend_trip", "brisk", ["repetition", "request_notify_done"]),
    ("gym_weekend", "deliberate", ["incomplete_zip", "ask_packed"]),
    ("beach_holiday", "steady", ["stall", "ask_elapsed"]),
]
TITLES = {"none": "nobody speaks", "oracle": "an assistant that knows the schedule"}


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")))


def index_row(name, ep, extra=None):
    c = ep["private_case"]
    s = ep["summary"]
    row = dict(name=name, kind=ep["kind"], assistant=ep.get("assistant", ""), task=c["task"],
               title=ep["case"]["title"], persona=c["persona"], events=[e["type"] for e in c["events"]],
               outcomes=s["outcomes"], goals_satisfied=s["goals_satisfied"], actions=s["actions"],
               seconds=s["seconds"], preferences=bool(c.get("preferences")))
    row.update(extra or {})
    return row


def update_index(rows):
    f = SITE / "episodes" / "index.json"
    old = json.loads(f.read_text()) if f.exists() else []
    names = {r["name"] for r in rows}
    dump(f, [r for r in old if r["name"] not in names] + rows)


def export_static(seed0=0):
    SITE.mkdir(exist_ok=True)
    dump(SITE / "world.json", web.world())
    dump(SITE / "modules.json", MODULES)
    rows = []
    for k, (task, persona, types) in enumerate(REPLAYS):
        case = web.new_case(task=task, seed=seed0 + k + 1, persona=persona, events=types)
        assert sorted(e["type"] for e in case["events"]) == sorted(types), (types, case["events"])
        for who in ("none", "oracle"):
            ep = web.replay(case, who)
            name = f"{task.replace('_', '-')}-{'-'.join(t.replace('_', '-') for t in types)}-{who}"
            dump(SITE / "episodes" / f"{name}.json", ep)
            rows.append(index_row(name, ep, dict(caption=TITLES[who])))
            print(f"{name:70s} {len(ep['frames']):3d} actions  {ep['summary']['outcomes']}  "
                  f"goals {ep['summary']['goals_satisfied']}", flush=True)
    update_index(rows)
    covered = {t for _, _, ts in REPLAYS for t in ts}
    missing = [t for t in EP.SAMPLED if t not in covered]
    print("event types without a replay:", missing or "none")


def export_physics(run, name, title="", assistant=""):
    run = Path(run)
    ep = web.from_physics(run)
    vids = json.loads((run / "videos" / "videos.json").read_text())
    frames = json.loads((run / "frames.json").read_text())["frames"]
    dest = SITE / "rendered" / name
    dest.mkdir(parents=True, exist_ok=True)
    views = []
    for v in vids:
        stem = Path(v["file"]).stem
        for ext in (".mp4", ".jpg"):
            if (run / "videos" / (stem + ext)).exists():
                shutil.copy(run / "videos" / (stem + ext), dest / (stem + ext))
        # sim time of each encoded frame (frames rendered every `speed`-th, then a freeze)
        n_src = v["source_frames"]
        times = [round(frames[min(len(frames) - 1, i * v["playback_speed"])]["t"], 3) for i in range(n_src)]
        views.append(dict(view=v["view"], file=v["file"], poster=stem + ".jpg", speed=v["playback_speed"],
                          fps=v["fps"], duration=v["duration_seconds"], times=times))
    ep["videos"] = views
    ep["assistant"] = assistant
    dump(dest / "episode.json", ep)
    update_index([index_row(name, ep, dict(caption=title, folder=f"rendered/{name}",
                                           views=[v["view"] for v in views]))])
    print("published rendered episode", name, len(ep["frames"]), "actions,", [v["file"] for v in views])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics", nargs=2, metavar=("RUN", "NAME"))
    ap.add_argument("--title", default=""); ap.add_argument("--assistant", default="")
    a = ap.parse_args()
    if a.physics:
        export_physics(*a.physics, title=a.title, assistant=a.assistant)
    else:
        export_static()
