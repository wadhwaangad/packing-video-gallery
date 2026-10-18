"""Retry the failed action of a generated run from the snapshot taken just before it.

Usage: python dbg_standup.py RUN_DIR
Loads RUN_DIR/failure.pkl (written by generate.py), rewinds to before the failed action,
re-executes it with the current skill code and prints the physical checks.
"""
import pickle, sys, importlib
from pathlib import Path
import numpy as np
import skills_case as C
import executor as X


def main(run):
    blob = pickle.loads((Path(run) / "failure.pkl").read_bytes())
    ex, act = blob["ex"], blob["action"]
    ex.rewind(len(ex.records))
    r = ex.r
    print("retrying", act.text(), "at", round(r.d.time, 2), "s")
    importlib.reload(C); importlib.reload(X)
    # carry over geometry edited since the run (same names, same bodies)
    import build_model as B
    fresh, _ = B.build()
    for g in range(fresh.ngeom):
        nm = fresh.geom(g).name
        if nm:
            gi = r.m.geom(nm).id
            r.m.geom_pos[gi] = fresh.geom_pos[g]; r.m.geom_size[gi] = fresh.geom_size[g]
            r.m.geom_friction[gi] = fresh.geom_friction[g]; r.m.geom_priority[gi] = fresh.geom_priority[g]
            r.m.geom_solref[gi] = fresh.geom_solref[g]; r.m.geom_solimp[gi] = fresh.geom_solimp[g]; r.m.geom_solmix[gi] = fresh.geom_solmix[g]
    import mujoco; mujoco.mj_forward(r.m, r.d)
    ex.__class__ = X.Executor
    n0 = len(r.frames)
    try:
        ex.execute(act, blob["label"], blob["event"])
        print("OK")
    except Exception as e:
        print("FAILED", e)
    for e in r.events[-6:]:
        print(" ", {k: v for k, v in e.items() if k != 'trace'})
    sp, sR = r.suitcase_frame()
    print("case pos", sp.round(3), "handle-end up deg", np.degrees(np.arccos(np.clip(sR[2, 1], -1, 1))).round(1))
    from scipy.spatial.transform import Rotation as Rot
    b = r.m.body("suitcase").id
    for f in r.frames[n0::12]:
        q = f["xquat"][b]; R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        print(f"  {f['t']:7.2f} {f['stage'][:34]:34s} case {np.round(f['xpos'][b], 3)} tilt {np.degrees(np.arccos(np.clip(R[2, 1], -1, 1))):5.1f}")
    r.debug_snapshot("retry")


if __name__ == "__main__":
    main(sys.argv[1])
