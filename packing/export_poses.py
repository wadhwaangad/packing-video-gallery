"""Solve the kinematic operator body for every recorded frame and audit it."""
import json, sys
from pathlib import Path
import numpy as np
import mujoco
import operator_model as O

BUILD = Path(__file__).resolve().parent / "build"


def export(run_dir):
    run_dir = Path(run_dir)
    frames = json.loads((run_dir / "frames.json").read_text())["frames"]
    m = mujoco.MjModel.from_xml_path(str(run_dir / "model.xml"))
    rid = {s: m.body(f"radius_{s}").id for s in "rl"}
    solver, feet = O.TorsoSolver(), O.Feet()
    out, worst = [], dict(arm=0.0, leg=0.0)
    unreachable_arm = unreachable_leg = 0
    bad_arm, bad_leg = [], []
    look = int(round(O.Feet.SWING * 25))
    for i, fr in enumerate(frames):
        P = np.array(fr["xpos"])
        er, el = P[rid["r"]], P[rid["l"]]
        future = frames[min(i + look, len(frames) - 1)]["root"]
        ft = feet.update(fr["t"], fr["root"], future)
        ankles = [ft[s][0] + np.array([0, 0, O.ANKLE_H]) for s in (0, 1)]
        q, res = solver.solve(er, el, fr["root"], ankles)
        b = O.body_frame(q, ft, [er, el], fr["gaze"])
        arm_err = max(abs(res[0]), abs(res[1]))
        worst["arm"] = max(worst["arm"], arm_err)
        unreachable_arm += arm_err > 0.003
        if arm_err > 0.003:
            bad_arm.append((fr["t"], fr["stage"]))
        legs = [np.linalg.norm(b["ankles"][i] - b["hips"][i]) for i in range(2)]
        if max(legs) > O.THIGH + O.SHIN + 1e-4:
            unreachable_leg += 1
            bad_leg.append((fr["t"], fr["stage"]))
        worst["leg"] = max(worst["leg"], max(legs))
        out.append(dict(t=fr["t"], q=q.tolist(),
                        pts={k: (np.asarray(v).tolist() if k not in ("feet", "R", "head_R") else None)
                             for k, v in b.items() if k not in ("feet", "R", "head_R")},
                        R=b["R"].tolist(), head_R=b["head_R"].tolist(),
                        feet=[[ft[s][0].tolist(), float(ft[s][1])] for s in (0, 1)]))
    audit = dict(frames=len(out), max_arm_length_error_m=worst["arm"], unreachable_arm_frames=int(unreachable_arm),
                 max_hip_to_ankle_m=worst["leg"], leg_length_m=O.THIGH + O.SHIN, unreachable_leg_frames=int(unreachable_leg),
                 upper_arm_m=O.UPPER_ARM, shoulder_width_m=2 * O.SHOULDER_HALF,
                 unreachable_arm_stages=sorted({st for _, st in bad_arm}), unreachable_leg_stages=sorted({st for _, st in bad_leg}),
                 unreachable_arm_times=[round(t, 2) for t, _ in bad_arm[::10]], unreachable_leg_times=[round(t, 2) for t, _ in bad_leg[::10]])
    (run_dir / "operator_poses.json").write_text(json.dumps(dict(poses=out, audit=audit)))
    print(json.dumps(audit, indent=1))
    return audit


if __name__ == "__main__":
    export(sys.argv[1])
