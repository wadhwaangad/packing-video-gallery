"""Label rendered frames and encode MP4s (H.264, 25 fps) plus posters and a manifest entry.

Usage: python encode.py RUN_DIR VIEW OUT.mp4 --speed 4 [--tag "OPERATOR VIEW"] [--freeze 3]
Frames rendered every `speed`-th recorded frame play back at `speed`x.
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def font(size):
    for f in ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("view"); ap.add_argument("out")
    ap.add_argument("--speed", type=int, default=4); ap.add_argument("--tag", default="")
    ap.add_argument("--freeze", type=float, default=3.0); ap.add_argument("--frames_dir", default="")
    a = ap.parse_args()
    run = Path(a.run_dir)
    frames = json.loads((run / "frames.json").read_text())["frames"]
    src = Path(a.frames_dir) if a.frames_dir else run / f"frames_{a.view}"
    files = sorted(src.glob("*.jpg"))
    f_small, f_cap = font(19), font(25)
    tag = f"PACKING / SIMULATION / {a.tag or a.view.upper()} / {a.speed}x" if a.speed != 1 else \
          f"PACKING / SIMULATION / {a.tag or a.view.upper()} / NORMAL SPEED"
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", "25", "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    last = None
    n = 0
    for fp in files:
        i = int(fp.stem)
        im = Image.open(fp).convert("RGB")
        dr = ImageDraw.Draw(im, "RGBA")
        bb = dr.textbbox((0, 0), tag, font=f_small)
        dr.rectangle((14, 14, 14 + bb[2] + 20, 14 + bb[3] + 14), fill=(20, 20, 20, 150))
        dr.text((24, 20), tag, font=f_small, fill=(240, 240, 240))
        cap = frames[i]["stage"]
        bb = dr.textbbox((0, 0), cap, font=f_cap)
        w = bb[2] + 28
        x0 = (im.width - w) // 2
        dr.rectangle((x0, im.height - 62, x0 + w, im.height - 22), fill=(20, 20, 20, 160))
        dr.text((x0 + 14, im.height - 58), cap, font=f_cap, fill=(245, 245, 245))
        im.save(proc.stdin, "JPEG", quality=95)
        last = im
        n += 1
    for _ in range(int(a.freeze * 25)):
        last.save(proc.stdin, "JPEG", quality=95); n += 1
    proc.stdin.close(); proc.wait()
    poster = out.with_suffix(".jpg")
    Image.open(files[min(len(files) - 1, len(files) // 3)]).convert("RGB").save(poster, quality=88)
    info = dict(file=out.name, playback_speed=a.speed, encoded_frames=n, fps=25, duration_seconds=round(n / 25, 2),
                dimensions=[last.width, last.height], bytes=out.stat().st_size,
                sha256=hashlib.sha256(out.read_bytes()).hexdigest(), source_frames=len(files))
    print(json.dumps(info))
    return info


if __name__ == "__main__":
    main()
