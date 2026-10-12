"""Copy a verified run's videos and evidence into trip/ for the gallery page.

Usage: python publish.py RUN_DIR
Expects RUN_DIR/videos/*.mp4 (from encode.py) and RUN_DIR/validation.json (from verify.py).
"""
import hashlib, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent / "trip"

CHAPTER_RULES = [  # first stage containing the key opens a chapter
    ("folded jeans", "Jeans"), ("T-shirt", "T-shirts"), ("knit sweater", "Sweater"),
    ("folded towel", "Towel"), ("toiletry bag", "Toiletry bag"), ("water bottle", "Water bottle"),
    ("leather notebook", "Notebook"), ("camera", "Camera"), ("lid edge", "Close the lid"),
    ("zipper", "Zip"), ("top handle", "Lift and stand"), ("Packed and ready", "Ready"),
]


def chapters(stages, speed):
    out, seen = [], set()
    for st in stages:
        for key, title in CHAPTER_RULES:
            if title not in seen and key.lower() in st["title"].lower():
                # chapters start at the walk that precedes the first matching stage
                t = st["t"]
                prev = [s for s in stages if s["t"] <= t and s["title"].startswith("Walk")]
                if prev and t - prev[-1]["t"] < 20 and title not in ("Zip", "Ready"):
                    t = prev[-1]["t"]
                out.append(dict(title=title, time=round(t / speed, 2), sim_time=round(t, 2)))
                seen.add(title)
    return sorted(out, key=lambda c: c["time"])


def main(run):
    run = Path(run)
    SITE.mkdir(exist_ok=True)
    val = json.loads((run / "validation.json").read_text())
    stages = json.loads((run / "stages.json").read_text())
    vids = json.loads((run / "videos" / "videos.json").read_text())
    for v in vids:
        for ext in (".mp4", ".jpg"):
            src = run / "videos" / (Path(v["file"]).stem + ext)
            if src.exists():
                shutil.copy(src, SITE / src.name)
    speed = next((v["playback_speed"] for v in vids if v["playback_speed"] != 1), 4)
    ch = chapters(stages, speed)
    (SITE / "chapters.json").write_text(json.dumps(ch, indent=1))
    manifest = dict(views=sorted({v["view"] for v in vids}), source_frames=val["frames"], video_fps=25,
                    sim_seconds=round(val["sim_seconds"], 2), chapters=ch, artifacts=vids)
    (SITE / "video-manifest.json").write_text(json.dumps(manifest, indent=1))
    for name in ["validation.json", "events.json", "stages.json"]:
        shutil.copy(run / name, SITE / name)
    shutil.copy(run / "operator_poses.json", SITE / "operator_poses.json") if False else None
    shutil.copy(HERE / "assets" / "asset-manifest.json", SITE / "asset-manifest.json")
    print("published", len(vids), "videos,", len(ch), "chapters; validation passed:", val["passed"])


if __name__ == "__main__":
    main(sys.argv[1])
