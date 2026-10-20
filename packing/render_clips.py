"""Close-up clips of individual actions from simulated runs (the 'detail' camera follows the
part each action works on), each captioned with the action and its measured outcome.

Usage: python render_clips.py RUN_DIR [RUN_DIR ...] [--verbs close,insert,...] [--samples 32] [--stride 2]
       [--publish trip]
For every successful action whose verb is selected (one clip per distinct verb+target per
run), renders its frames in one Blender pass per run, encodes RUN_DIR/clips/<n>_<verb>.mp4
at 2x (stride 2 of the 25 Hz recording) and writes RUN_DIR/clips/clips.json. With
--publish, copies the clips to <site>/clips/ and merges them into <site>/clips.json.
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw
import encode as EN

HERE = Path(__file__).resolve().parent
PY = sys.executable
DEFAULT = "open,close,insert,turn,set,press,fasten,unfasten,stand_up,roll,fold,place,pick_up"


def outcome(rec, events):
    """One line: what the simulation measured for this action."""
    t0, t1 = rec["t_start"], rec["t_end"]
    ev = [e for e in events if t0 - 1e-6 <= e["t"] <= t1 + 1e-6]
    pick = lambda name: [e for e in ev if e["event"] == name]
    v, t = rec["action"]["verb"], rec["action"].get("target", "")
    if v in ("open", "close") and t == "main_zip":
        z = pick("zip_check")
        return f"zip measured {z[-1]['closed_fraction']:.0%} closed" + (", lid held shut" if z and z[-1]["lid_held"] else "") if z else ""
    if v in ("open", "close") and t == "lid":
        a = pick("lid_released") or pick("lid_opened")
        return f"lid at {a[-1]['lid_angle_deg']:.1f} deg after letting go" if a else ""
    if v == "insert":
        d = pick("tab_flip")
        return "; ".join(f"pull {e['tab']} at {e['angle_deg']:.0f} deg, {'latched' if e['docked'] else 'not latched'}" for e in d)
    if v == "turn" or (v == "set" and t == "lock"):
        d = pick("dial_turned")
        return "; ".join(f"dial {e['dial']}: {e['start']} -> {e['reading']} in {e['flicks']} flicks" for e in d)
    if v == "press" and t == "lock_button":
        if pick("pulls_released"):
            return "dials show the code: the pulls are released"
        if pick("lock_button_blocked"):
            return f"dials show {''.join(map(str, pick('lock_button_blocked')[0]['dials']))}, not the code: nothing released"
        return "nothing in the lock"
    if v == "set" and t == "trolley_handle":
        d = pick("trolley_set")
        return f"handle latched at stop {d[-1]['latched']} ({100 * d[-1]['q']:.1f} cm out)" if d else ""
    if v in ("fasten", "unfasten"):
        d = pick("strap_fastened") or pick("strap_unfastened")
        if not d:
            return ""
        e = d[-1]
        return "buckle clicked shut" if v == "fasten" and e.get("latched") else f"buckle released, {100 * e.get('open_m', 0):.1f} cm apart"
    if v == "stand_up":
        d = pick("touchdown_tilt_deg")
        return f"standing on its wheels, {d[-1]['value']:.1f} deg from vertical at touchdown" if d else ""
    if v == "roll":
        d = pick("roll_check")
        return f"rolled to {d[-1]['error_m'] * 100:.0f} cm from the spot, {d[-1]['tilt_deg']:.1f} deg tilt" if d else ""
    if v == "fold":
        d = pick("fold_check")
        return f"towel depth {100 * d[-1]['depth_before']:.0f} -> {100 * d[-1]['depth_after']:.0f} cm ({d[-1]['ratio']:.2f})" if d else ""
    if v == "place":
        d = pick("placed_check")
        if d:
            e = d[-1]
            tm, lm = e.get("target_m"), e.get("local_m")
            if tm and lm:
                err = ((tm[0] - lm[0]) ** 2 + (tm[1] - lm[1]) ** 2) ** 0.5
                return f"landed {100 * err:.1f} cm from the resolved spot"
    if v == "pick_up":
        d = pick("lift_check")
        if d:
            return "lifted by contact: " + ", ".join(f"{k} rose {100 * x:.1f} cm" for k, x in d[-1]["rise_m"].items())
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--verbs", default=DEFAULT)
    ap.add_argument("--samples", type=int, default=32); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--publish", default=""); ap.add_argument("--max-per-run", type=int, default=12)
    a = ap.parse_args()
    verbs = set(a.verbs.split(","))
    published = []
    for run_s in a.runs:
        run = Path(run_s).resolve()
        acts = json.loads((run / "actions.json").read_text())
        events = json.loads((run / "events.json").read_text())
        frames = json.loads((run / "frames.json").read_text())["frames"]
        seen, chosen = set(), []
        for rec in acts:
            v, t = rec["action"]["verb"], rec["action"].get("target", "")
            if not rec.get("ok") or v not in verbs or (v, t) in seen or rec["frame_end"] <= rec["frame_start"]:
                continue
            seen.add((v, t)); chosen.append(rec)
        chosen = chosen[:a.max_per_run]
        if not chosen:
            continue
        idx = sorted({i for rec in chosen for i in range(rec["frame_start"], min(rec["frame_end"], len(frames)), a.stride)})
        out = run / "frames_detail"; out.mkdir(exist_ok=True)
        todo = [i for i in idx if not (out / f"{i:06d}.jpg").exists()]
        if todo:
            print(f"{run.name}: rendering {len(todo)} detail frames for {len(chosen)} clips", flush=True)
            subprocess.run([PY, str(HERE / "render_blender.py"), str(run), "detail", "--samples", str(a.samples),
                            "--out", str(out), "--indices", ",".join(map(str, todo))], cwd=str(HERE),
                           stdout=open(run / "render_detail.log", "w"), stderr=subprocess.STDOUT, check=True)
        cdir = run / "clips"; cdir.mkdir(exist_ok=True)
        f_small, f_cap = EN.font(19), EN.font(24)
        rows = []
        for k, rec in enumerate(chosen):
            name = f"{run.name}_{k:02d}_{rec['action']['verb']}"
            mp4 = cdir / f"{name}.mp4"
            text = rec["action"]["text"]
            res = outcome(rec, events)
            cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", "25", "-i", "-",
                   "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            files = [out / f"{i:06d}.jpg" for i in range(rec["frame_start"], min(rec["frame_end"], len(frames)), a.stride)]
            files = [f for f in files if f.exists()]
            last = None
            for fp in files:
                im = Image.open(fp).convert("RGB")
                dr = ImageDraw.Draw(im, "RGBA")
                tag = f"CLOSE-UP / {a.stride}x / {rec['action']['verb'].upper()}"
                bb = dr.textbbox((0, 0), tag, font=f_small)
                dr.rectangle((14, 14, 14 + bb[2] + 20, 14 + bb[3] + 14), fill=(20, 20, 20, 150))
                dr.text((24, 20), tag, font=f_small, fill=(240, 240, 240))
                cap = frames[int(fp.stem)]["stage"]
                bb = dr.textbbox((0, 0), cap, font=f_cap)
                x0 = (im.width - bb[2] - 28) // 2
                dr.rectangle((x0, im.height - 60, x0 + bb[2] + 28, im.height - 22), fill=(20, 20, 20, 160))
                dr.text((x0 + 14, im.height - 57), cap, font=f_cap, fill=(245, 245, 245))
                im.save(proc.stdin, "JPEG", quality=94)
                last = im
            for _ in range(25):
                last.save(proc.stdin, "JPEG", quality=94)
            proc.stdin.close(); proc.wait()
            Image.open(files[len(files) * 2 // 3]).convert("RGB").save(mp4.with_suffix(".jpg"), quality=86)
            rows.append(dict(run=run.name, verb=rec["action"]["verb"], target=rec["action"].get("target", ""), action=text,
                             outcome=res, file=f"{name}.mp4", poster=f"{name}.jpg", t_start=rec["t_start"],
                             t_end=rec["t_end"], seconds=round((rec["t_end"] - rec["t_start"]) / a.stride, 1)))
            print(" ", name, "|", text, "|", res, flush=True)
        (cdir / "clips.json").write_text(json.dumps(rows, indent=1))
        published += [(cdir, r) for r in rows]
    if a.publish:
        site = HERE.parent / a.publish
        (site / "clips").mkdir(parents=True, exist_ok=True)
        index = site / "clips.json"
        old = json.loads(index.read_text()) if index.exists() else []
        keep = [r for r in old if r["file"].split("/")[-1] not in {x[1]["file"] for x in published}]
        new = []
        for cdir, r in published:
            for ext in (".mp4", ".jpg"):
                shutil.copy(cdir / (Path(r["file"]).stem + ext), site / "clips" / (Path(r["file"]).stem + ext))
            new.append(dict(r, file=f"{a.publish}/clips/{r['file']}", poster=f"{a.publish}/clips/{r['poster']}"))
        index.write_text(json.dumps(keep + new, indent=1))
        print("published", len(new), "clips")


if __name__ == "__main__":
    main()
