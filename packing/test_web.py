"""Checks for the browser playground's Python side (web.py), symbolic only.

  * the modules the page loads (export_web.MODULES) import on their own, without
    MuJoCo or Blender, as they will in Pyodide;
  * a Session stepped one action at a time gives the same episode as Harness.run;
  * handle() covers a played episode: guidance said between actions is heard, the
    oracle hint is what the oracle says, and the episode ends with a summary;
  * the exported files (play/) are consistent with the code.
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
                "r = web.handle(dict(cmd='step', args={})); print(json.dumps(r)[:40])")
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


if __name__ == "__main__":
    check_closure()
    check_stepping()
    check_handle()
    check_files()
    print("ALL OK" if not fails else f"{len(fails)} FAILED")
    sys.exit(1 if fails else 0)
