"""Generate one packing trajectory: sample (or load) a case, run it in physics, save it.

Usage:
  python generate.py OUT_DIR --reference                 # the case shown on the page
  python generate.py OUT_DIR --seed 12 [--task work_trip] [--events 2] [--persona brisk]
  python generate.py OUT_DIR --case case.json
  python generate.py OUT_DIR --seed 12 --dry-run         # symbolic plan only, no physics

Outputs in OUT_DIR:
  case.json      task, seed, persona and the private event schedule
  actions.json   one record per semantic action: text, label (nominal / error / recovery /
                 question / plan_change), event id, start/end time and frame, feedback,
                 symbolic state after the action
  goals.json     goal predicates at the end (what an assistant should have prevented)
  frames.json    25 Hz recording of every body pose (input to rendering)
  events.json    physical checks (lift heights, contacts, placements, lid, zip)
  stages.json    caption timeline;  model.xml, result.json
Then: export_poses.py, verify.py, render_all.py (see README).
"""
import argparse, json, random, shutil, time
from pathlib import Path
import numpy as np
import tasks as T
import episodes as EP
from packsim import PackSim, Action
from runner import Runner
from executor import Executor

MAX_RETRIES = 3     # physical retries of a failed action (rewound, slightly varied)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--reference", action="store_true"); ap.add_argument("--case", default="")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--task", default=None)
    ap.add_argument("--events", type=int, default=None); ap.add_argument("--persona", default=None)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--max-actions", type=int, default=120)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.reference:
        case = EP.reference_case()
    elif a.case:
        case = json.loads(Path(a.case).read_text())
    else:
        case = EP.sample_case(task=a.task, seed=a.seed, n_events=a.events, persona=a.persona)
    ok, trace, sim, _ = EP.dry_run(case)
    case["dry_run"] = dict(valid=ok, actions=len(trace), final_goals=sim.goals())
    (out / "case.json").write_text(json.dumps(case, indent=1))
    print(f"case: {case['task']} / {case['persona']} / {[e['type'] for e in case['events']]} -> {len(trace)} actions")
    if a.dry_run:
        for t in trace:
            print(f"{t['label']:>11} {t['event']:>3} {t['action']['text']}")
        return

    t0 = time.time()
    rng = random.Random(case["seed"])
    events = [EP.Event(**{k: v for k, v in e.items() if k in EP.Event.__dataclass_fields__}) for e in case["events"]]
    user = EP.User(case["task"], rng, events)
    sim = PackSim(case["task"])
    r = Runner()
    ex = Executor(r, sim, EP.PERSONAS[case["persona"]])
    r.stage("Start beside the open suitcase")
    T.to_rest(r, T=1.0)
    r.hold(1.0)
    status = "complete"
    try:
        for _ in range(a.max_actions):
            nxt = user.next(sim.state)
            if nxt is None:
                break
            act, label, eid = nxt
            rec = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    rec = ex.execute(act, label, eid)
                    break
                except Exception as err:
                    if attempt == MAX_RETRIES:
                        raise
                    # Rewind physics, recordings and symbolic state to just before this
                    # action and try again with a small variation (a person adjusting).
                    print(f"   retry {attempt + 1} of '{act.text()}' after: {err}", flush=True)
                    ex.rewind(len(ex.records))
                    jr = random.Random(case["seed"] * 100 + len(ex.records) * 7 + attempt)
                    ex.jitter = dict(stand=(jr.uniform(-0.04, 0.04), jr.uniform(-0.04, 0.04)),
                                     time=jr.uniform(0.75, 1.15), place=(jr.uniform(-0.01, 0.01), jr.uniform(-0.01, 0.01)),
                                     release=jr.uniform(0.01, 0.03))
                    r.event("action_retry", action=act.to_dict(), attempt=attempt + 1, reason=str(err), jitter=ex.jitter)
            ex.jitter = dict(stand=(0.0, 0.0), time=1.0, place=(0.0, 0.0), release=0.0)
            print(f"{r.d.time:8.1f}s  [{label}{(' ' + eid) if eid else ''}] {act.text()}"
                  f"{'' if rec['ok'] else '  -> refused: ' + rec['feedback']}   (wall {time.time() - t0:.0f}s)", flush=True)
        r.stage("Packed and ready" if sim.done() else "Finished")
        T.to_rest(r, T=1.2)
        r.hold(4.0)
    except Exception as e:
        status = f"failed: {e}"
        print("FAILED", e, flush=True)
        r.debug_snapshot("failure")
        # Development aid: the executor (physics + symbolic state) with the snapshot taken
        # just before the failed action, to retry that action alone.
        import pickle
        (out / "failure.pkl").write_bytes(pickle.dumps(dict(ex=ex, user=user, action=act, label=label, event=eid)))
    save(r, ex, sim, user, out, status, time.time() - t0)


def save(r, ex, sim, user, out, status, wall):
    shutil.copy(r.xml, out / "model.xml")
    (out / "frames.json").write_text(json.dumps(dict(frames=r.frames)))
    (out / "events.json").write_text(json.dumps(r.events, indent=1, default=float))
    (out / "stages.json").write_text(json.dumps(r.stages, indent=1))
    (out / "actions.json").write_text(json.dumps(ex.records, indent=1, default=float))
    (out / "goals.json").write_text(json.dumps(dict(goals=sim.goals(), satisfied=sim.done(), pack=list(sim.state.pack),
                                                    schedule=[e.to_dict() for e in user.events]), indent=1))
    res = dict(status=status, sim_seconds=r.d.time, wall_seconds=wall, frames=len(r.frames), actions=len(ex.records),
               goals_satisfied=sim.done(), diagnostics=r.diag)
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "diagnostics"}, indent=1, default=str))


if __name__ == "__main__":
    main()
