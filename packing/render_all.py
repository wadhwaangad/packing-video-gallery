"""Render both views of a verified run, encode the videos and write videos.json.

Usage: python render_all.py RUN_DIR [--stride 4] [--samples 48] [--jobs 2]
Rendering is split into interleaved chunks across `jobs` Blender processes per view.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--samples", type=int, default=48); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--views", default="ego,overview"); ap.add_argument("--realtime", action="store_true")
    a = ap.parse_args()
    run = Path(a.run_dir).resolve()
    n = len(json.loads((run / "frames.json").read_text())["frames"])
    views = a.views.split(",")
    procs = []
    t0 = time.time()
    for view in views:
        stride = 1 if (a.realtime and view == "ego") else a.stride
        idx = list(range(0, n, stride))
        chunks = [idx[k::a.jobs] for k in range(a.jobs)]
        out = run / (f"frames_{view}" + ("_rt" if stride == 1 else ""))
        out.mkdir(exist_ok=True)
        for k, ch in enumerate(chunks):
            todo = [i for i in ch if not (out / f"{i:06d}.jpg").exists()]
            if not todo:
                continue
            lst = run / f"render_{view}_{k}.txt"
            log = open(run / f"render_{view}_{k}.log", "w")
            procs.append(subprocess.Popen([PY, str(HERE / "render_blender.py"), str(run), view, "--samples", str(a.samples),
                                           "--out", str(out), "--indices", ",".join(map(str, todo))],
                                          stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE)))
    for p in procs:
        p.wait()
    print(f"rendered in {time.time() - t0:.0f}s")
    vids = []
    vdir = run / "videos"; vdir.mkdir(exist_ok=True)
    tags = {"ego": "OPERATOR VIEW", "overview": "ROOM CAMERAS"}
    for view in views:
        stride = 1 if (a.realtime and view == "ego") else a.stride
        name = f"packing-{view}" + ("-realtime" if stride == 1 else "")
        src = run / (f"frames_{view}" + ("_rt" if stride == 1 else ""))
        res = subprocess.run([PY, str(HERE / "encode.py"), str(run), view, str(vdir / f"{name}.mp4"), "--speed", str(stride),
                              "--tag", tags[view], "--frames_dir", str(src)], capture_output=True, text=True, cwd=str(HERE))
        info = json.loads(res.stdout.strip().splitlines()[-1])
        info["view"] = view
        vids.append(info)
        print(info)
    old = vdir / "videos.json"
    prev = json.loads(old.read_text()) if old.exists() else []
    keep = [v for v in prev if v["file"] not in {x["file"] for x in vids}]
    old.write_text(json.dumps(keep + vids, indent=1))


if __name__ == "__main__":
    main()
