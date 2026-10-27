"""Checks for the browser playground's Python side (web.py), symbolic only.

  * the modules the page loads (export_web.MODULES) import on their own, without
    MuJoCo or Blender, as they will in Pyodide;
  * a Session stepped one action at a time gives the same episode as Harness.run;
  * handle() covers a played episode: guidance said between actions is heard, the
    oracle hint is what the oracle says, and the episode ends with a summary;
  * the exported files (play/) are consistent with the code;
  * the 3D page's game (game.py): every task can be played to the end through options()
    and act() alone, options only offer what PackSim allows, mistakes show on the
    checklist, the assistant slot is empty, and the record is plain JSON;
  * the 3D page's files (play/pack/) are current and hold no BlenderKit scan.
Usage: python packing/test_web.py
"""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import web
import episodes as EP
import usersim as U
from export_web import MODULES, SITE

fails = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        fails.append(what)


def check_closure():
    with tempfile.TemporaryDirectory() as d:
        for m in MODULES:
            shutil.copy(HERE / m, Path(d) / m)
        code = ("import sys; sys.modules['mujoco'] = None; sys.modules['bpy'] = None; sys.path.insert(0, sys.argv[1]); "
                "import web, json; s = web.handle(dict(cmd='new', args=dict(task='gym_weekend', seed=2))); "
                "r = web.handle(dict(cmd='step', args={})); "
                "g = web.handle(dict(cmd='game_new', args=dict(task='work_trip'))); "
                "o = web.handle(dict(cmd='game_options', args=dict(x=1.55, y=0.64, heading=1.57))); print(json.dumps(o)[:40])")
        p = subprocess.run([sys.executable, "-I", "-c", code, d], capture_output=True, text=True, cwd=d)
        check(p.returncode == 0, f"the {len(MODULES)} browser modules run alone" + ("" if p.returncode == 0 else ": " + p.stderr[-400:]))


def check_stepping():
    for seed in (1, 2, 3):
        case = web.new_case(seed=seed)
        for who in ("none", "oracle"):
            ep = web.replay(case, who)
            h = EP.run(case, U.Oracle() if who == "oracle" else None)
            same = [f["text"] for f in ep["frames"]] == [r["action"]["text"] for r in h.b.records] and \
                ep["summary"]["outcomes"] == {e.id: e.outcome for e in h.events}
            check(same, f"seed {seed} {who}: stepped session == Harness.run ({len(ep['frames'])} actions)")
            check(json.loads(json.dumps(ep)) == ep, f"seed {seed} {who}: frames are plain JSON")


def check_handle():
    s = web.handle(dict(cmd="new", args=dict(task="weekend_trip", persona="steady", seed=3,
                                             events=["wrong_object", "ask_since"])))
    check(sorted(e["type"] for e in s["private_case"]["events"]) == ["ask_since", "wrong_object"], "new: the asked-for events")
    said = 0
    for _ in range(300):
        hint = web.handle(dict(cmd="hint"))["text"]
        if hint:
            r = web.handle(dict(cmd="say", args=dict(text=hint)))
            check(r["conv"][0]["speaker"] == "assistant" and r["conv"][-1]["speaker"] == "user",
                  f"say: heard and answered ({hint[:40]}...)")
            said += 1
        r = web.handle(dict(cmd="step", args={}))
        if r["done"]:
            break
    out = r["summary"]["outcomes"]
    check(said >= 2 and set(out.values()) <= EP.GOOD and r["summary"]["goals_satisfied"],
          f"played with the hints: {out}, goals {r['summary']['goals_satisfied']}")


def check_files():
    w = json.loads((SITE / "world.json").read_text())
    check(w == json.loads(json.dumps(web.world())), "play/world.json is up to date")
    check(json.loads((SITE / "modules.json").read_text()) == MODULES, "play/modules.json lists the modules")
    idx = json.loads((SITE / "episodes" / "index.json").read_text())
    for row in idx:
        f = SITE / (row["folder"] + "/episode.json" if row["kind"] == "physics" else f"episodes/{row['name']}.json")
        ok = f.exists()
        if ok and row["kind"] == "physics":
            ep = json.loads(f.read_text())
            ok = all((f.parent / v["file"]).exists() and len(v["times"]) > 10 for v in ep["videos"])
        check(ok, f"episode {row['name']} has its files")
    covered = {t for r in idx if r["kind"] == "symbolic" for t in r["events"]}
    check(set(EP.SAMPLED) <= covered, "every sampled event type has a replay")


def play(task, order=None, mistakes=()):
    """Play `task` through game.Game like the page does: stand in front of each item, take
    the first option that picks it up, stand at the case, take the first place option
    (or the spot named in `mistakes`), then close, zip, lock, stand up."""
    import math, packsim as P, scene as S, discrete as D, game
    g = game.Game(task)
    t = [0.0]
    case = (1.55, 0.64, math.pi / 2)

    def do(x, y, hd, prefix):
        o = g.options(x, y, hd)
        for q in o["options"]:             # everything offered is allowed from where it is done
            probe = P.PackSim(task); probe.state = __import__("copy").deepcopy(g.sim.state)
            probe.state.station, probe.state.facing = q["station"], q["facing"]
            assert probe.check(D.action(q["id"]))[0], q
        q = next((q for q in o["options"] if q["id"].startswith(prefix)), None)
        assert q, (task, prefix, [q["id"] for q in o["options"]])
        t[0] += 3
        r = g.act(q["id"], q["station"], q["facing"], t[0], x, y)
        assert r["ok"], r
        return r
    for n in order or P.TASKS[task].pack:
        it = S.item(n); nx, ny = S.STATIONS[it.station].normal
        do(it.xy[0] - nx * 0.6, it.xy[1] - ny * 0.6, math.atan2(ny, nx), f"pick_up({n})")
        do(*case, dict(mistakes).get(n, f"place({n}"))
    do(*case, "close_lid"); do(*case, "zip_to(100)")
    if P.TASKS[task].lock:
        do(*case, "dock_zip_pulls"); do(*case, "turn_dial(dial_1")
    r = do(*case, "stand_up_case")
    return g, r["state"]


def check_game():
    import random, packsim as P, usersim as U, game
    for task in P.TASKS:
        order = U.pack_order(task, random.Random(1))         # heavy first, as the user model packs
        g, s = play(task, order)
        rec = g.record()
        check(s["done"] and rec["done"] and json.loads(json.dumps(rec)) == rec,
              f"{task}: played to the end through options/act ({len(rec['actions'])} actions, plain JSON)")
        check(not [c for c in g.conv if c["speaker"] == "assistant"], f"{task}: the empty assistant slot never spoke")
    # a heavy item on top of the clothes breaks a rule on the checklist, and the game is not done
    g, s = play("weekend_trip", ["tshirts", "trousers", "sweatshirts", "towel", "toiletry_bag", "bottle", "notebook", "camera"],
                mistakes={"trousers": "place(trousers, wheel_end)"})
    broken = [c["key"] for c in s["checklist"] if not c["done"]]
    check(not s["done"] and broken == ["heavy_on_floor(trousers)"], f"a mistake stays on the checklist: {broken}")
    r = g.user_says("is that everything?", 99.0)
    check(r["said"] == [] and g.conv[-1]["speaker"] == "user", "speaking to the empty slot: heard, no reply")


def check_pack_files():
    from export_web import PACK, world3d
    w = json.loads((PACK / "world3d.json").read_text())
    check(w == json.loads(json.dumps(world3d())), "play/pack/world3d.json is up to date")
    need = [f"assets/models/{a['asset']}/{a['asset']}.gltf" for a in w["assets"]] +            [f"assets/models/{it['asset']}/{it['asset']}.gltf" for it in w["items"].values() if it["asset"]] + [w["hdri"]] +            [f"assets/textures/{t}/{f}.jpg" for t in ("oak_wood_planks", "long_white_tiles", "cotton_jersey", "wool_boucle",
                                                     "denim_fabric", "scuba_suede", "brown_leather", "painted_plaster_wall")
            for f in ("diff", "nor_gl", "rough")]
    missing = [f for f in need if not (PACK / f).exists()]
    check(not missing, f"the 3D page's {len(need)} model, texture and sky files are there" + (f": missing {missing}" if missing else ""))
    scans = [f.name for f in PACK.rglob("*") if f.suffix == ".blend" or "blenderkit" in str(f).lower()]
    check(not scans, "no BlenderKit scan is published")
    # the baked light: one entry for every box pack.js gives an `lm` name, and nothing else
    lm = json.loads((PACK / "lightmap.json").read_text())
    want = {f"floor_{r}" for r in w["rooms"]} | {"ceiling", "basin", "backsplash", "pillow0", "pillow1"}
    want |= {x["name"] for x in w["walls"]} | {x["name"] + "_base" for x in w["walls"] if x["c"][2] - x["s"][2] / 2 < 1e-6 and x["s"][2] > 0.5}
    want |= {f"win{i}_frame{k}" for i in range(len(w["windows"])) for k in range(5)} | {f"win{i}_sill" for i in range(len(w["windows"]))}
    want |= {f"door{i}_{p}" for i in range(2) for p in ("side0", "side1", "top", "leaf")} | {b["name"] for b in w["boxes"]}
    check(set(lm["objects"]) == want and (PACK / "lightmap.png").exists(),
          f"lightmap.png and lightmap.json cover the page's {len(want)} baked boxes"
          + ("" if set(lm["objects"]) == want else f": extra {set(lm['objects']) - want}, missing {want - set(lm['objects'])}"))
    page = (SITE / "pack.js").read_text(encoding="utf-8")
    check("@0.165.0" in (SITE / "pack.html").read_text(encoding="utf-8") and "game_act" in page and "game_options" in page,
          "pack.html pins three.js and pack.js uses the game commands")


if __name__ == "__main__":
    check_closure()
    check_stepping()
    check_handle()
    check_files()
    check_game()
    check_pack_files()
    print("ALL OK" if not fails else f"{len(fails)} FAILED")
    sys.exit(1 if fails else 0)
