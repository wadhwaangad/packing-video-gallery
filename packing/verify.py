"""Checks on a generated packing trajectory. Writes RUN_DIR/validation.json.

Two kinds of check: the simulation is physically sound (contact-only, no drops,
penetration and joint limits, reachable body), and the recorded symbolic outcome
agrees with the physical one (every item the action log says is packed is inside the
closed case, and nothing else is).

Usage: python verify.py RUN_DIR
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot
import scene as S
from actionspace import Action

HERE = Path(__file__).resolve().parent
CODE = ["scene.py", "hands.py", "build_model.py", "runner.py", "tasks.py", "skills_case.py", "skills_hw.py", "packsim.py", "discrete.py",
        "actionspace.py", "layout.py", "rooms.py", "instruct.py", "episodes.py", "executor.py", "generate.py", "operator_model.py",
        "export_poses.py", "render_blender.py", "item_assets.py", "prepare_assets.py", "encode.py", "verify.py",
        "fetch_assets.py"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def quat_R(q):
    return Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def main(run):
    run = Path(run)
    frames = json.loads((run / "frames.json").read_text())["frames"]
    events = json.loads((run / "events.json").read_text())
    result = json.loads((run / "result.json").read_text())
    actions = json.loads((run / "actions.json").read_text())
    goals = json.loads((run / "goals.json").read_text())
    poses = json.loads((run / "operator_poses.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(run / "model.xml"))
    bid = {m.body(i).name: i for i in range(m.nbody)}
    W, L = S.SUITCASE["outer"]; t = S.SUITCASE["shell"]
    depth = S.SUITCASE["base_depth"] + S.SUITCASE["lid_depth"]
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
    checks["finite_poses"] = bool(all(np.isfinite(np.array(f["xpos"])).all() for f in frames[::25]))

    # -- symbolic log agrees with physics
    final_state = actions[-1]["state"]
    inside = {}
    for it in S.ITEMS:
        loc = sR.T @ (P[bid[it.name]] - sp)
        phys = bool(abs(loc[0]) < W / 2 - t and abs(loc[1]) < L / 2 - t and 0 < loc[2] < depth)
        logged = final_state["loc"][it.name] == "case"
        inside[it.name] = dict(local_m=loc.round(4).tolist(), inside_physically=phys, packed_in_log=logged)
    detail["items"] = inside
    checks["log_matches_physics_for_every_item"] = all(v["inside_physically"] == v["packed_in_log"] for v in inside.values())
    detail["goals"] = goals["goals"]
    detail["goals_satisfied"] = goals["satisfied"]
    expected = {e["id"]: e for e in goals["schedule"]}
    checks["every_scheduled_event_fired"] = all(e["fired"] for e in expected.values())
    case = json.loads((run / "case.json").read_text()) if (run / "case.json").exists() else {}
    instructed = case.get("mode") == "instructions"
    detail["refused_or_rejected"] = [dict(text=a["action"]["text"], feedback=a["feedback"]) for a in actions if not a["ok"]]
    if not instructed:
        # a sampled case is legal by construction; with instructions, refusals are outputs
        checks["no_refused_actions"] = all(a["ok"] for a in actions)
    detail["action_retries"] = [dict(t=e["t"], action=e["action"]["text"], reason=e["reason"]) for e in events
                                if e["event"] == "action_retry"]
    places = [e for e in events if e["event"] == "placed_check"]
    checks["every_placement_landed_in_its_region"] = all(e["ok"] for e in places)

    # -- navigation: one go_to only reaches what is in the person's room or in view
    import rooms as R
    nav = []
    for a in actions:
        if not a["ok"] or a["action"]["verb"] != "go_to" or a["action"]["target"] not in bid or \
                a["action"]["target"] == "suitcase":
            continue
        f = frames[min(a["frame_start"], len(frames) - 1)]
        eye = (f["root"]["x"], f["root"]["y"], R.EYE_H)
        p = np.array(f["xpos"][bid[a["action"]["target"]]])
        same = R.room_at(eye[:2]) == R.room_at(p[:2])
        vis = same or R.line_of_sight(eye, p)[0]
        nav.append(dict(action=a["action"]["text"], from_room=R.room_at(eye[:2]), to_room=R.room_at(p[:2]), ok=bool(vis)))
    detail["go_to_reach"] = nav
    checks["every_go_to_target_in_room_or_view"] = all(n["ok"] for n in nav)
    looks = [e for e in events if e["event"] == "look_check"]
    detail["look_checks"] = looks
    checks["every_looked_for_item_in_sight"] = all(e["visible"] for e in looks)
    enters = [a for a in actions if a["ok"] and a["action"]["verb"] == "enter"]
    checks["every_enter_ends_in_its_room"] = all(
        R.room_at((frames[min(a["frame_end"], len(frames) - 1)]["root"]["x"],
                   frames[min(a["frame_end"], len(frames) - 1)]["root"]["y"])) == a["action"]["target"] for a in enters)
    detail["placements"] = places
    import discrete as D
    checks["every_action_in_catalogue"] = all(D.to_id(Action.from_dict(a["action"])) for a in actions)

    # -- the case at the end: every part where the log says it is
    lid_deg = float(np.degrees(last["lid"]))
    detail["final_lid_angle_deg"] = lid_deg
    if final_state["lid"] == "closed":
        checks["lid_closed_under_2deg"] = abs(lid_deg) < 2.0
    else:
        checks["lid_open_over_150deg"] = lid_deg > 150
    zip_ev = [e for e in events if e["event"] == "zipper_engaged"]
    checks["zipper_engaged_only_with_lid_down"] = all(abs(e["lid_angle_deg"]) < 2.5 for e in zip_ev)
    hw = last.get("hw")
    if hw:
        detail["final_hardware"] = hw
        zc_log = 1 - final_state["zip_open"]
        checks["zip_matches_log"] = abs(hw["zip_closed"] - zc_log) < 0.05
        checks["lock_dials_match_log"] = hw["dials"] == final_state["dials"]
        checks["zip_pulls_match_log"] = all(hw["docked"].values()) == (final_state["pulls"] == "docked") and             (any(hw["docked"].values()) == (final_state["pulls"] == "docked"))
        checks["trolley_stop_matches_log"] = hw["trolley_level"] == final_state["trolley"]
        checks["strap_matches_log"] = hw["buckled"] == (final_state["strap"] == "fastened")
    tilt = float(np.degrees(np.arccos(np.clip(sR[2, 1], -1, 1))))
    detail["final_suitcase_tilt_deg"] = tilt
    wheel_z = []
    for g in range(m.ngeom):
        if m.geom(g).name.startswith("wheel_"):
            b = m.geom_bodyid[g]
            wheel_z.append(float((P[b] + quat_R(Q[b]) @ m.geom_pos[g])[2] - m.geom_size[g][0]))
    detail["wheel_bottom_heights_m"] = wheel_z
    if final_state["case_pose"] == "upright_floor":
        checks["suitcase_upright_under_5deg"] = tilt < 5.0
        checks["four_wheels_on_floor"] = len(wheel_z) == 4 and max(abs(z) for z in wheel_z) < 0.008
    else:
        lie = float(np.degrees(np.arccos(np.clip(sR[2, 2], -1, 1))))
        detail["final_suitcase_lie_deg"] = lie
        checks["suitcase_lying_on_its_back_under_10deg"] = lie < 10.0
    k = min(50, len(frames) - 1)
    drift = {n: float(np.linalg.norm(np.array(frames[-1]["xpos"][bid[n]]) - np.array(frames[-1 - k]["xpos"][bid[n]])))
             for n in ["suitcase"] + [it.name for it in S.ITEMS]}
    detail["last_2s_drift_m"] = drift
    checks["settled_last_2s_under_2mm"] = max(drift.values()) < 0.002

    # -- the case stayed put while it was being packed
    # While packing the case must stay put; flipping the heavy lid over may shift it a little
    # on the soft mattress (as a real case does), up to the final lift.
    t_lift = next((f["t"] for f in frames if f["stage"].startswith(("Hook the trolley handle", "Hook the top handle"))), ts[-1])
    t_lid = next((a["t_start"] for a in actions if a["action"]["verb"] == "close" and a["action"].get("target") == "lid"
                  and a["ok"]), t_lift)
    s0 = np.array(frames[0]["xpos"][bid["suitcase"]])
    dist = lambda f: float(np.linalg.norm(np.array(f["xpos"][bid["suitcase"]]) - s0))
    disp_pack = max([dist(f) for f in frames if f["t"] < t_lid] or [0.0])
    disp_lid = max([dist(f) for f in frames if t_lid <= f["t"] < t_lift] or [0.0])
    detail["suitcase_max_displacement_while_packing_m"] = disp_pack
    detail["suitcase_max_displacement_lid_to_lift_m"] = disp_lid
    checks["suitcase_undisturbed_while_packing_under_15mm"] = disp_pack < 0.015
    checks["suitcase_shift_while_closing_under_60mm"] = disp_lid < 0.06

    # -- grasps
    lifts = [e for e in events if e["event"] == "lift_check"]
    picks = [a for a in actions if a["action"]["verb"] == "pick_up" and a["ok"]]
    detail["lift_checks"] = [dict(t=e["t"], items=e["items"], rise_m=e["rise_m"], normal_force_N=e["normal_force_N"]) for e in lifts]
    checks["every_pick_lifted_by_contact"] = len(lifts) == len(picks)
    drops = []
    t_close = [e["t"] for e in events if e["event"] in ("lid_swung",)]
    t_close = t_close[-1] if t_close else ts[-1]
    keep = ts < t_close
    for it in S.ITEMS:
        z = np.array([f["xpos"][bid[it.name]][2] for f in frames])[keep]
        vz = np.diff(z) / np.diff(ts[keep])
        if np.any(vz < -1.2):
            i = int(np.argmax(vz < -1.2))
            drops.append(dict(item=it.name, t=float(ts[i]), stage=frames[i]["stage"]))
    detail["drops"] = drops
    checks["no_item_dropped"] = not drops

    # -- physics diagnostics recorded at 50 Hz
    dg = result["diagnostics"]
    detail["diagnostics"] = dg
    checks["contact_penetration_under_5mm"] = dg["max_penetration"] < 0.005
    if "max_penetration_soft" in dg:     # compliant folded piles: fingers sink into the fabric
        checks["soft_pile_penetration_under_10mm"] = dg["max_penetration_soft"] < 0.010
    checks["joint_limits_within_2_5deg"] = np.degrees(dg["max_joint_excursion"]) < 2.5

    # -- operator body
    au = poses["audit"]
    detail["operator_audit"] = au
    checks["upper_arm_length_fixed"] = au["max_arm_length_error_m"] < 0.003
    checks["no_unreachable_arm_frames"] = au["unreachable_arm_frames"] == 0
    checks["no_unreachable_leg_frames"] = au["unreachable_leg_frames"] == 0
    boxes = [b for b in S.all_static_boxes() if b.collide and b.size[2] > 0.3 and b.name != "mattress"
             and b.center[2] - b.size[2] / 2 < 1.0]
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
    out = dict(passed=passed, checks=checks, detail=detail, sim_seconds=result["sim_seconds"], frames=len(frames),
               actions=len(actions), frame_interval_s=0.04, physics_hz=1000, control_hz=500,
               trajectory_sha256=hashlib.sha256((run / "frames.json").read_bytes()).hexdigest(),
               mujoco_version=mujoco.__version__, code_sha256={c: sha(HERE / c) for c in CODE if (HERE / c).exists()},
               scope=("Contact-only grasps (no hand-object welds) with anatomical MyoHand fingers on force-limited wrist "
                      "carriages; kinematic body solved around the simulated forearms; idealised zipper (hinge range "
                      "narrowed after a closure check). Goal predicates are evaluated on the symbolic log and "
                      "cross-checked against final body poses. Not a measured sim-to-real validation."))
    (run / "validation.json").write_text(json.dumps(out, indent=1, default=float))
    for k_, v in checks.items():
        print(("PASS " if v else "FAIL ") + k_)
    print("passed" if passed else "FAILED")
    return out


if __name__ == "__main__":
    main(sys.argv[1])
