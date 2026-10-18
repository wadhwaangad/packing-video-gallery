"""Copy a verified trajectory's videos and records into the site folder.

Usage: python publish.py RUN_DIR [--site trip]
Expects RUN_DIR/videos/*.mp4 (render_all.py) and RUN_DIR/validation.json (verify.py).
Chapters come from the action log: one per item fetched, one per event, one per case step.
"""
import argparse, json, shutil
from pathlib import Path
import scene as S

HERE = Path(__file__).resolve().parent
CASE_TITLES = {"close_lid": "Close the lid", "zip": "Zip", "unzip": "Unzip", "open_lid": "Open the lid",
               "stand_up": "Stand it up"}


def chapters(actions, schedule, speed):
    out, seen_events = [], set()
    desc = {e["id"]: e for e in schedule}
    for i, a in enumerate(actions):
        act, t = a["action"], a["t_start"]
        if a["event"] and a["event"] not in seen_events:
            seen_events.add(a["event"])
            e = desc[a["event"]]
            kind = {"error": "Error", "question": "Question", "plan_change": "Plan change"}[e["channel"]]
            out.append(dict(title=f"{kind}: {e['description']}", time=round(t / speed, 2), sim_time=t, event=a["event"],
                            channel=e["channel"]))
            continue
        if a["label"] != "nominal":
            continue
        if act["verb"] == "go_to" and i + 1 < len(actions) and actions[i + 1]["action"]["verb"] == "pick_up" \
                and not actions[i + 1]["event"]:
            label = S.item(actions[i + 1]["action"]["target"]).label
            out.append(dict(title=label[0].upper() + label[1:], time=round(t / speed, 2), sim_time=t))
        elif act["verb"] in CASE_TITLES:
            prev = actions[i - 1] if i else None
            t0 = prev["t_start"] if prev and prev["action"]["verb"] == "go_to" else t
            out.append(dict(title=CASE_TITLES[act["verb"]], time=round(t0 / speed, 2), sim_time=t0))
    return out


def publish_sample(run, site, name):
    """Records of a generated trajectory (no video) for the page's sample table."""
    dest = site / "samples" / name
    dest.mkdir(parents=True, exist_ok=True)
    for f in ["case.json", "actions.json", "goals.json", "validation.json", "result.json"]:
        if (run / f).exists():
            shutil.copy(run / f, dest / f)
    case = json.loads((run / "case.json").read_text())
    res = json.loads((run / "result.json").read_text())
    goals = json.loads((run / "goals.json").read_text())
    val = json.loads((run / "validation.json").read_text()) if (run / "validation.json").exists() else {}
    acts = json.loads((run / "actions.json").read_text())
    row = dict(name=name, task=case["task"], persona=case["persona"], seed=case["seed"],
               events=[dict(type=e["type"], recover=e["recover"], channel=e["channel"]) for e in goals["schedule"]],
               actions=len(acts), labels={k: sum(a["label"] == k for a in acts) for k in
                                          ("nominal", "error", "recovery", "question", "plan_change")},
               sim_seconds=round(res["sim_seconds"], 1), status=res["status"], goals_satisfied=goals["satisfied"],
               failed_goals=[k for k, v in goals["goals"].items() if not v], verified=bool(val.get("passed")),
               failed_checks=[k for k, v in val.get("checks", {}).items() if not v])
    index = site / "samples.json"
    rows = [x for x in (json.loads(index.read_text()) if index.exists() else []) if x["name"] != name]
    index.write_text(json.dumps(rows + [row], indent=1))
    print("published sample", name, row["status"], "verified", row["verified"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--site", default="trip"); ap.add_argument("--sample", default="")
    a = ap.parse_args()
    run, site = Path(a.run), HERE.parent / a.site
    if a.sample:
        return publish_sample(run, site, a.sample)
    site.mkdir(exist_ok=True)
    val = json.loads((run / "validation.json").read_text())
    actions = json.loads((run / "actions.json").read_text())
    goals = json.loads((run / "goals.json").read_text())
    vids = json.loads((run / "videos" / "videos.json").read_text())
    for v in vids:
        for ext in (".mp4", ".jpg"):
            src = run / "videos" / (Path(v["file"]).stem + ext)
            if src.exists():
                shutil.copy(src, site / src.name)
    speed = next((v["playback_speed"] for v in vids if v["playback_speed"] != 1), 4)
    ch = chapters(actions, goals["schedule"], speed)
    (site / "chapters.json").write_text(json.dumps(ch, indent=1))
    manifest = dict(views=sorted({v["view"] for v in vids}), source_frames=val["frames"], video_fps=25,
                    sim_seconds=round(val["sim_seconds"], 2), chapters=ch, artifacts=vids)
    (site / "video-manifest.json").write_text(json.dumps(manifest, indent=1))
    for name in ["validation.json", "events.json", "stages.json", "case.json", "actions.json", "goals.json"]:
        shutil.copy(run / name, site / name)
    shutil.copy(HERE / "assets" / "asset-manifest.json", site / "asset-manifest.json")
    shutil.copy(HERE / "assets" / "scans" / "scans.json", site / "scans.json")
    print("published", len(vids), "videos,", len(ch), "chapters; validation passed:", val["passed"])


if __name__ == "__main__":
    main()
