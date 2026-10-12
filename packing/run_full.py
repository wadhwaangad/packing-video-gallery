"""One continuous packing run: eight trips through the apartment, close, zip, lift.

Usage: python run_full.py OUT_DIR [--until N]
"""
import argparse, json, pickle, shutil, time, hashlib
from pathlib import Path
import numpy as np
from runner import Runner
import tasks as T
import skills_case as C
import scene as S

PI = np.pi
SUIT = (1.45, 0.64)          # stand beside the bed, facing the suitcase
TO_DRESSER = [(2.45, 0.62)]
# Doorways are crossed on straight segments: bedroom door centre x=3.20, bathroom door x=4.55.
BED_TO_DOOR = [(2.45, 0.62), (2.62, 2.20), (3.20, 2.75), (3.20, 3.95)]
DOOR_TO_BED = list(reversed(BED_TO_DOOR))[1:] + [SUIT]
DOOR_TO_BATH = [(3.90, 3.95), (4.55, 3.30), (4.55, 1.85)]
BATH_TO_DOOR = [(4.55, 1.85), (4.55, 3.30), (3.90, 3.95), (3.20, 3.95)]


def squeeze_trip(r, names, route_out, stand, route_back, crouch=0.80, carry_h=1.0, **grip):
    T.to_rest(r, T=1.0)
    r.stage(f"Walk to the {S.item(names[0]).room}")
    r.walk(route_out + [stand[:2]], final_yaw=stand[2])
    T.squeeze_pick(r, names, crouch_h=crouch, **grip)
    r.stage("Carry " + " and ".join(S.item(n).label for n in names))
    r.move(1.2, hands=T.carry_pose(r, names, height=carry_h), root=dict(h=0.95))
    r.walk(route_back, final_yaw=PI / 2)
    slot = "tshirts" if names[0].startswith("tshirt") else names[0]
    T.place_item(r, names, slot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--until", type=int, default=99)
    ap.add_argument("--resume", default=""); ap.add_argument("--from-step", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.resume:
        r = pickle.loads(Path(a.resume).read_bytes())
        # Development aid: carry over actuator/joint parameters edited since the checkpoint.
        import build_model as B
        fresh, _ = B.build()
        for attr in ["actuator_forcerange", "actuator_gainprm", "actuator_biasprm", "jnt_range", "jnt_solref"]:
            if getattr(fresh, attr).shape == getattr(r.m, attr).shape:
                getattr(r.m, attr)[:] = getattr(fresh, attr)
        r.resumed_from = a.resume
    else:
        r = Runner()
        r.stage("Start beside the open suitcase")
        T.to_rest(r, T=1.0)
        r.hold(1.0)
    steps = [
        lambda: squeeze_trip(r, ["jeans"], TO_DRESSER, (2.95, 0.98, 0.0), TO_DRESSER + [SUIT]),
        lambda: squeeze_trip(r, ["tshirt_white", "tshirt_navy"], TO_DRESSER, (2.95, 1.42, 0.0), TO_DRESSER + [SUIT]),
        lambda: squeeze_trip(r, ["sweater"], [], (0.72, 0.64, PI / 2), [SUIT], d=0.010, palm_up=0.004),
        lambda: squeeze_trip(r, ["towel"], BED_TO_DOOR + DOOR_TO_BATH + [(4.70, 1.15)], (4.92, 1.15, 0.0),
                             BATH_TO_DOOR + DOOR_TO_BED, crouch=0.90),
        lambda: squeeze_trip(r, ["toiletry_bag"], BED_TO_DOOR + DOOR_TO_BATH + [(4.55, 1.20)], (4.55, 0.86, -PI / 2),
                             BATH_TO_DOOR + DOOR_TO_BED, crouch=0.90),
        lambda: bottle_trip(r),
        lambda: squeeze_trip(r, ["notebook"], BED_TO_DOOR + [(3.00, 4.25)], (1.88, 4.22, PI / 2),
                             [(3.00, 4.25), (3.20, 3.95)] + DOOR_TO_BED, crouch=0.62),
        lambda: squeeze_trip(r, ["camera"], BED_TO_DOOR + [(3.00, 4.25)], (2.32, 4.22, PI / 2),
                             [(3.00, 4.25), (3.20, 3.95)] + DOOR_TO_BED, crouch=0.62),
        lambda: close_and_zip(r),
        lambda: final_lift(r),
    ]
    status = "complete"
    try:
        for i, f in enumerate(steps[: a.until]):
            if i < a.from_step:
                continue
            f()
            (out / f"checkpoint_{i}.pkl").write_bytes(pickle.dumps(r))
            print(f"--- step {i} done at sim {r.d.time:.1f}s wall {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        status = f"failed: {e}"
        print("FAILED", e, flush=True)
    save(r, out, status, time.time() - t0)


def bottle_trip(r):
    T.to_rest(r, T=1.0)
    r.stage("Walk to the kitchen")
    # Keep the resting hands clear of the counter edge (x 6.35) and the bathroom wall (x 5.6).
    r.walk(BED_TO_DOOR + [(3.90, 3.95), (5.95, 2.70), (5.95, 0.80), (6.12, 0.75)], final_yaw=0.0)
    C.wrap_pick(r, "bottle", side="r", lift=0.12)
    r.stage("Carry the water bottle")
    r.move(1.2, hands=C.carry_one(r, "bottle", "r", height=1.10, forward=0.30, lateral=0.06), root=dict(h=0.95))
    # One-handed friction grip: walk a little slower and turn gently.
    r.walk([(5.95, 0.85), (5.95, 2.70), (3.90, 3.95), (3.20, 3.95)] + DOOR_TO_BED, final_yaw=PI / 2, speed=0.7, yaw_rate=1.4)
    C.place_upright(r, "bottle", side="r")


def close_and_zip(r):
    T.to_rest(r, T=1.0)
    r.stage("Walk to the foot of the bed")
    r.walk([(2.45, 0.62), (2.48, 1.62)], final_yaw=PI)
    C.close_lid(r, side="r")
    T.to_rest(r, T=1.0)
    r.stage("Walk back beside the suitcase")
    r.walk([(2.45, 0.62), SUIT], final_yaw=PI / 2)
    C.press_and_zip(r)


def final_lift(r):
    T.to_rest(r, T=1.0)
    r.stage("Walk to the foot of the bed")
    r.walk([(2.45, 0.62), (2.46, 1.21)], final_yaw=PI)
    C.lift_from_bed_foot(r)
    T.to_rest(r, T=1.2)
    r.stage("Packed and ready")
    r.hold(5.0)


def save(r, out, status, wall):
    shutil.copy(r.xml, out / "model.xml")
    (out / "frames.json").write_text(json.dumps(dict(frames=r.frames)))
    (out / "events.json").write_text(json.dumps(r.events, indent=1))
    (out / "stages.json").write_text(json.dumps(r.stages, indent=1))
    res = dict(status=status, sim_seconds=r.d.time, wall_seconds=wall, frames=len(r.frames), diagnostics=r.diag)
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
