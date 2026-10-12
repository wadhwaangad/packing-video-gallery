"""Checks on a recorded packing run. Writes RUN_DIR/validation.json.

Usage: python verify.py RUN_DIR
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot
import scene as S
import operator_model as O

HERE = Path(__file__).resolve().parent
CODE = ["scene.py", "hands.py", "build_model.py", "runner.py", "tasks.py", "skills_case.py", "run_full.py",
        "operator_model.py", "export_poses.py", "render_blender.py", "encode.py", "verify.py", "fetch_assets.py"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def quat_R(q):
    return Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def main(run):
    run = Path(run)
    frames = json.loads((run / "frames.json").read_text())["frames"]
    events = json.loads((run / "events.json").read_text())
    result = json.loads((run / "result.json").read_text())
    poses = json.loads((run / "operator_poses.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(run / "model.xml"))
    bid = {m.body(i).name: i for i in range(m.nbody)}
    W, L = S.SUITCASE["outer"]; D, DL, t = S.SUITCASE["base_depth"], S.SUITCASE["lid_depth"], S.SUITCASE["shell"]
    last = frames[-1]
    P = np.array(last["xpos"]); Q = np.array(last["xquat"])
    sp, sR = P[bid["suitcase"]], quat_R(Q[bid["suitcase"]])
    checks, detail = {}, {}

    # -- run integrity
    ts = np.array([f["t"] for f in frames])
    checks["single_continuous_run"] = result["status"] == "complete"
    checks["strictly_increasing_time"] = bool(np.all(np.diff(ts) > 0))
    checks["gravity_enabled"] = bool(np.allclose(m.opt.gravity, [0, 0, -9.81]))
    checks["no_equality_constraints_in_model"] = m.neq == 0
    detail["equality_constraints"] = int(m.neq)
    checks["finite_poses"] = bool(all(np.isfinite(np.array(f["xpos"])).all() for f in frames[::25]))

    # -- packing result
    inside = {}
    for it in S.ITEMS:
        loc = sR.T @ (P[bid[it.name]] - sp)
        inside[it.name] = dict(local_m=loc.round(4).tolist(),
                               inside=bool(abs(loc[0]) < W / 2 - t and abs(loc[1]) < L / 2 - t and t < loc[2] < D + DL - t))
    detail["items_in_suitcase"] = inside
    checks["all_items_inside_closed_suitcase"] = all(v["inside"] for v in inside.values())
    lid_deg = float(np.degrees(last["lid"]))
    detail["final_lid_angle_deg"] = lid_deg
    checks["lid_closed_under_2deg"] = abs(lid_deg) < 2.0
    zip_ev = [e for e in events if e["event"] == "zipper_engaged"]
    zcheck = [e for e in events if e["event"] == "zip_check"]
    checks["zipper_engaged_only_after_closure_check"] = bool(zip_ev and zcheck and zcheck[0]["closed"] and zcheck[0]["t"] <= zip_ev[0]["t"])
    first_zip = next((f["t"] for f in frames if f["zip"]), None)
    detail["zipper_first_active_t"] = first_zip

    # -- suitcase upright on its wheels
    up = sR[:, 1]
    tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
    detail["final_suitcase_tilt_deg"] = tilt
    checks["suitcase_upright_under_5deg"] = tilt < 5.0
    wheel_z = []
    for g in range(m.ngeom):
        nm = m.geom(g).name
        if nm.startswith("wheel_"):
            b = m.geom_bodyid[g]
            pw = P[b] + quat_R(Q[b]) @ m.geom_pos[g]
            wheel_z.append(float(pw[2] - m.geom_size[g][0]))
    detail["wheel_bottom_heights_m"] = wheel_z
    checks["four_wheels_on_floor"] = len(wheel_z) == 4 and max(abs(z) for z in wheel_z) < 0.006
    # settled: compare the last two seconds
    k = min(50, len(frames) - 1)
    drift = {n: float(np.linalg.norm(np.array(frames[-1]["xpos"][bid[n]]) - np.array(frames[-1 - k]["xpos"][bid[n]])))
             for n in ["suitcase"] + [it.name for it in S.ITEMS]}
    detail["last_2s_drift_m"] = drift
    checks["settled_last_2s_under_2mm"] = max(drift.values()) < 0.002

    # -- the suitcase was not knocked before the final lift
    # Window: until the operator deliberately takes hold of the handle for the final lift.
    t_lift = next((f["t"] for f in frames if f["stage"].startswith("Hook the top handle")), ts[-1])
    s0 = np.array(frames[0]["xpos"][bid["suitcase"]])
    disp = max(float(np.linalg.norm(np.array(f["xpos"][bid["suitcase"]]) - s0)) for f in frames if f["t"] < t_lift)
    detail["suitcase_max_displacement_before_lift_m"] = disp
    checks["suitcase_undisturbed_before_lift_under_15mm"] = disp < 0.015

    # -- every grasp lifted and nothing was dropped en route
    lifts = [e for e in events if e["event"] == "lift_check"]
    detail["lift_checks"] = [dict(t=e["t"], items=e["items"], rise_m=e["rise_m"], normal_force_N=e["normal_force_N"]) for e in lifts]
    checks["every_item_lifted_by_contact"] = len(lifts) == 8
    drops = []
    t_close = next((e["t"] for e in events if e["event"] == "lid_swung"), ts[-1])
    keep = ts < t_close           # after the lid closes, items move with the case itself
    for it in S.ITEMS:
        b = bid[it.name]
        z = np.array([f["xpos"][b][2] for f in frames])[keep]
        vz = np.diff(z) / np.diff(ts[keep])
        if np.any(vz < -1.2):     # free fall faster than any planned lowering
            i = int(np.argmax(vz < -1.2))
            drops.append(dict(item=it.name, t=float(ts[i]), stage=frames[i]["stage"]))
    detail["drops"] = drops
    checks["no_item_dropped"] = not drops

    # -- physics diagnostics recorded at 50 Hz during the run
    dg = result["diagnostics"]
    detail["diagnostics"] = dg
    checks["contact_penetration_under_5mm"] = dg["max_penetration"] < 0.005
    checks["joint_limits_within_2deg"] = np.degrees(dg["max_joint_excursion"]) < 2.0

    # -- operator body
    au = poses["audit"]
    detail["operator_audit"] = au
    checks["upper_arm_length_fixed"] = au["max_arm_length_error_m"] < 0.003
    checks["no_unreachable_arm_frames"] = au["unreachable_arm_frames"] == 0
    checks["no_unreachable_leg_frames"] = au["unreachable_leg_frames"] == 0
    # pelvis / torso clearance from walls and furniture (2D, body radius 0.16 m)
    boxes = [b for b in S.all_static_boxes() if b.collide and b.size[2] > 0.3 and b.name != "mattress"
             and b.center[2] - b.size[2] / 2 < 1.0]           # ignore wall sections above door frames
    worst = (1e9, None, None)
    for po in poses["poses"][::5]:
        c = np.array(po["pts"]["pelvis"])[:2]
        for b in boxes:
            lo = np.array(b.center[:2]) - np.array(b.size[:2]) / 2
            hi = np.array(b.center[:2]) + np.array(b.size[:2]) / 2
            dd = np.linalg.norm(np.maximum(0, np.maximum(lo - c, c - hi)))
            if dd < worst[0]:
                worst = (dd, b.name, po["t"])
    detail["min_pelvis_to_obstacle_m"] = dict(distance=worst[0], obstacle=worst[1], t=worst[2])
    checks["pelvis_clear_of_walls_and_furniture"] = worst[0] > 0.12

    passed = all(checks.values())
    traj = hashlib.sha256((run / "frames.json").read_bytes()).hexdigest()
    out = dict(passed=passed, checks=checks, detail=detail, sim_seconds=result["sim_seconds"], frames=len(frames),
               frame_interval_s=0.04, physics_hz=1000, control_hz=500, trajectory_sha256=traj,
               mujoco_version=mujoco.__version__, code_sha256={c: sha(HERE / c) for c in CODE if (HERE / c).exists()},
               scope=("Contact-only grasps (no hand-object welds) with anatomical MyoHand fingers on force-limited wrist "
                      "carriages; kinematic body solved around the simulated forearms; idealised zipper weld after a "
                      "closure check. Not full-body dynamics and not a measured sim-to-real validation."))
    (run / "validation.json").write_text(json.dumps(out, indent=1, default=float))
    for k_, v in checks.items():
        print(("PASS " if v else "FAIL ") + k_)
    print("passed" if passed else "FAILED")
    return out


if __name__ == "__main__":
    main(sys.argv[1])
