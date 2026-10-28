"""Write the browser playground's data (play/).

Usage:
  python packing/export_web.py                       # world.json, modules.json, symbolic replays
  python packing/export_web.py --physics RUN NAME    # a rendered physics run (videos + frames)
  python packing/export_web.py --pack3d              # only the 3D page (play/pack/): world and CC0 assets

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
           "dialogue.py", "usersim.py", "episodes.py", "web.py", "assistant_template.py", "game.py"]

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


# ---------------------------------------------------------------- the 3D page (play/pack.html)
PACK = SITE / "pack"
ASSETS = HERE / "assets"
# Overview cameras of the rendered videos (render_blender.OVERVIEW_CAMS): eye, target, lens (mm, 36 mm sensor).
OVERVIEW_CAMS = {
    "bedroom": ((3.60, 3.20, 2.25), (1.75, 1.20, 0.60), 16),
    "bathroom": ((3.95, 2.05, 2.25), (5.10, 0.70, 0.85), 16),
    "kitchen": ((5.75, 2.10, 2.25), (6.60, 0.80, 0.90), 16),
    "hall": ((6.85, 6.25, 2.35), (3.80, 3.20, 0.70), 20),
    "living": ((6.85, 6.25, 2.35), (2.40, 4.50, 0.60), 20),
}


def world3d():
    """The scene the 3D page draws, from scene.py (the same numbers as physics and the
    Blender renders). Scanned BlenderKit models are not in it: the page builds its own
    garment piles and case (the scans may not be redistributed)."""
    import scene as S, layout as LO, packsim as P, rooms as R
    r4 = lambda v: [round(float(x), 4) for x in v]
    box = lambda b: dict(name=b.name, c=r4(b.center), s=r4(b.size), m=b.material, solid=b.collide, yaw=round(float(b.yaw), 4))
    items = {}
    for it in S.ITEMS:
        st = S.STATIONS[it.station]
        items[it.name] = dict(label=it.label, kind=it.kind, family=it.family, size=r4(it.size), heavy=it.heavy,
                              station=it.station, xy=r4(it.xy), top=round(st.top, 4), yaw=round(float(it.start_yaw), 4),
                              tint=r4(it.tint), asset=it.asset, mesh_offset=r4(it.mesh_offset),
                              mesh_objects=list(it.mesh_objects), hands=S.FAMILIES[it.family]["hands"])
    sc = dict(S.SUITCASE)
    return dict(
        wall_h=S.WALL_H, wall_t=S.WALL_T,
        rooms={k: dict(x=list(v["x"]), y=list(v["y"]), floor=v["floor"], label=R.LABEL[k]) for k, v in S.ROOMS.items()},
        walls=[box(b) for b in S.WALLS], windows=[list(w) for w in S.WINDOWS],
        boxes=[box(b) for b in S.PROCEDURAL],
        assets=[dict(name=a.name, asset=a.asset, pos=r4(a.pos), yaw=round(float(a.yaw), 4), scale=a.scale,
                     collision=[box(b) for b in a.collision]) for a in S.ASSETS],
        bed=dict(S.BED), items=items,
        stations={k: dict(label=s.label, room=s.room, normal=list(s.normal), top=s.top, stand=r4(s.stand))
                  for k, s in S.STATIONS.items()},
        case=dict(outer=r4(sc["outer"]), base_depth=sc["base_depth"], lid_depth=sc["lid_depth"], shell=sc["shell"],
                  floor=sc["floor"], lid_floor=sc["lid_floor"], hinge_h=sc["hinge_h"], bar=sc["bar"],
                  wheel_r=sc["wheel_r"], wheel_x=sc["wheel_x"], wheel_y=sc["wheel_y"], wheel_z_base=sc["wheel_z_base"],
                  wheel_z_lid=sc["wheel_z_lid"], pos=r4(sc["pos"]), yaw=round(float(sc["yaw"]), 4),
                  inner=r4((LO.X0, LO.X1, LO.Y0, LO.Y1)), hardware=S.HARDWARE,
                  floor_spots={k: dict(case=r4(v["case"]), beside=r4(v["beside"])) for k, v in P.FLOOR_SPOTS.items()}),
        tasks={k: dict(title=t.title, pack=list(t.pack), lock=bool(t.lock), note=t.note) for k, t in P.TASKS.items()},
        overview=OVERVIEW_CAMS,
        hdri="assets/hdri/kloofendal_48d_partly_cloudy_puresky_2k.hdr",
    )


def export_3d():
    """play/pack/world3d.json and the CC0 Poly Haven files the page loads (from the
    asset manifest, Poly Haven entries only; 2k textures are reduced to 1k)."""
    from PIL import Image
    man = json.loads((ASSETS / "asset-manifest.json").read_text())
    out = PACK / "assets"
    n = 0
    for a in man["assets"]:
        if "polyhaven.com" not in a.get("page", ""):
            continue                                   # BlenderKit scans are never copied
        if a["kind"] == "model":
            src, dst = ASSETS / "models" / a["id"], out / "models" / a["id"]
        elif a["kind"] == "texture":
            name = a["id"] if (ASSETS / "textures" / a["id"]).exists() else a["id"] + "_2k"
            src, dst = ASSETS / "textures" / name, out / "textures" / a["id"]
        else:
            src, dst = ASSETS / "hdri", out / "hdri"
        for f in a["files"]:
            d = dst / f
            d.parent.mkdir(parents=True, exist_ok=True)
            if f.endswith(".jpg"):
                im = Image.open(src / f)
                if max(im.size) > 1024:
                    im = im.resize((1024, round(1024 * im.size[1] / im.size[0])), Image.LANCZOS)
                    im.save(d, quality=90)
                    n += 1
                    continue
            shutil.copy(src / f, d)
            n += 1
    (out / "LICENSE.txt").write_text("These files are Poly Haven assets (https://polyhaven.com), CC0 1.0: "
                                     "https://polyhaven.com/license\n")
    dump(PACK / "world3d.json", world3d())
    try:                                               # the live physics (MuJoCo WebAssembly); needs mujoco (.venv)
        import build_model
        (PACK / "physics.xml").write_text(build_model.build_web())
    except ImportError as e:
        print("physics.xml not written:", e)
    dump(SITE / "modules.json", MODULES)
    print(f"3D page: {n} asset files, world3d.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics", nargs=2, metavar=("RUN", "NAME"))
    ap.add_argument("--title", default=""); ap.add_argument("--assistant", default="")
    ap.add_argument("--pack3d", action="store_true", help="only the 3D page's world and assets")
    a = ap.parse_args()
    if a.physics:
        export_physics(*a.physics, title=a.title, assistant=a.assistant)
    elif a.pack3d:
        export_3d()
    else:
        export_static()
        export_3d()
