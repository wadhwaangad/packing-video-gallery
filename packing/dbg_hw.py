"""Try the suitcase-control skills on a closed case (no packing first).

Usage: python dbg_hw.py zip|tabs|dials|button|strap|trolley ...
Saves MuJoCo views to build/debug_hw_*.png at the end of each step.
"""
import sys, time
import numpy as np
import mujoco
from runner import Runner
import tasks as T
import skills_hw as H


def closed_case():
    r = Runner()
    r.d.qpos[r.m.joint("lid_hinge").qposadr[0]] = 0.0
    mujoco.mj_forward(r.m, r.d)
    T.to_rest(r, T=0.8)
    r.hold(1.0)
    return r


def at_suitcase(r):
    r.walk([(1.55, 0.64)], final_yaw=np.pi / 2, speed=0.6)


def at_bed_foot(r):
    r.walk([(2.45, 0.62), (2.48, 1.80)], final_yaw=np.pi, speed=0.8)


if __name__ == "__main__":
    r = closed_case()
    t0 = time.time()
    for step in sys.argv[1:]:
        if step == "zip":
            at_suitcase(r); H.run_zip(r, 1.0, 0.0)
        elif step == "halfzip":
            at_suitcase(r); H.run_zip(r, 1.0, 0.5)
        elif step == "unzip":
            H.run_zip(r, 0.0, 1.0)
        elif step == "tabs":
            H.insert_pulls(r)
        elif step == "dials":
            H.set_code(r, "372")
        elif step == "dial1":
            H.turn_dial(r, 1, by=1)
        elif step == "button":
            H.press_lock_button(r)
        elif step == "code0":
            H.set_code(r, "000")
        elif step == "strap":
            at_suitcase(r); H.unfasten_strap(r)
        elif step == "fasten":
            H.fasten_strap(r)
        elif step == "trolley":
            at_bed_foot(r); H.set_trolley(r, 1)
        elif step == "retract":
            H.set_trolley(r, 0)
        print(f"{step}: {r.hardware_reading()}  t={r.d.time:.1f}s wall={time.time() - t0:.0f}s", flush=True)
        r.debug_snapshot(f"hw_{step}")
    print([e for e in r.events if e["event"] not in ("stand_trace",)][-12:])
