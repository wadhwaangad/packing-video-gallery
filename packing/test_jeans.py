"""Primitive test (name kept from the first version): walk to the dresser, squeeze-pick the
folded trousers, carry them to the case and put them in the wheel-end column, through the
same PackSim + executor path a generated trajectory uses."""
import json, time
from runner import Runner
import tasks as T
from packsim import PackSim, Action
from executor import Executor

t0 = time.time()
r = Runner()
sim = PackSim("weekend_trip")
ex = Executor(r, sim)
r.stage("Settle")
T.to_rest(r, T=1.0)
r.hold(0.5)
for a in [Action("go_to", "trousers"), Action("pick_up", "trousers"), Action("go_to", "suitcase"),
          Action("put_in", "wheel_end")]:
    rec = ex.execute(a)
    print(f"{rec['t_start']:7.1f}-{rec['t_end']:7.1f}s {a.text()}  ok={rec['ok']} {rec['feedback']}")
T.to_rest(r)
r.hold(1.0)
print(json.dumps([e for e in r.events if e["event"] in ("lift_check", "placed_check")], indent=1, default=float))
print("frames", len(r.frames), "wall", round(time.time() - t0, 1), "sim", round(r.d.time, 2))
