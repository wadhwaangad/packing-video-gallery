# Packing simulator: source

A person packs a large spinner suitcase lying open on the bed of a two-room apartment, then operates it: closes and zips it, docks the zip pulls in the combination lock, dials a code, fastens the lid strap, extends the telescopic handle, stands the case on its wheels and rolls it to the door. A loose towel can be folded on the kitchen counter. Runs come from a sampled case (task, seed, persona, private event schedule) or from instructions in plain language or JSON. Every request is grounded into a typed action, refused with a reason if it cannot be, and executed by contact in MuJoCo with anatomical hands. Runs are rendered with Blender Cycles.

## Pipeline

```
python packing/fetch_assets.py        # Poly Haven (CC0) + BlenderKit scans (royalty-free; not committed)
python packing/prepare_assets.py      # clean / orient / scale the garment scans; cut the suitcase scan and split its moving parts
python packing/generate.py RUN --reference                 # the trajectory on the page
python packing/generate.py RUN --seed 12                   # a sampled case (add --task / --events / --persona)
python packing/generate.py RUN --seed 12 --dry-run         # symbolic run only, no physics
python packing/generate.py RUN --seed 12 --assistant oracle    # someone talks to the user: none | oracle | lines.json
python packing/generate.py RUN --seed 12 --dry-run --assistant interactive   # you type as the assistant
python packing/generate.py RUN --seed 12 --dry-run --assistant packing/assistant_template.py   # the model slot (empty: never speaks)
python packing/generate.py RUN --say "pack the t-shirts at the handle end; close the lid; zip it; lock it"
python packing/generate.py RUN --plan plan.json            # list of {"verb", "args"} requests, or sentences
python packing/generate.py RUN --say "..." --parser gemini # LLM fills the request schema (GEMINI_API_KEY)
python packing/export_poses.py RUN    # body solved around the simulated forearms
python packing/verify.py RUN          # checks -> RUN/validation.json
python packing/render_all.py RUN      # Blender Cycles: head camera + room cameras, encoded
python packing/render_clips.py RUN --publish trip   # close-up clip per operation, captioned with its measured outcome
python packing/publish.py RUN         # copy videos and records to trip/
python packing/export_spec.py         # the action space, assets, placement rules and live grounding examples -> trip/spec.json
python packing/test_grounding.py      # instructions -> actions / rejections, symbolic only
python packing/test_usersim.py        # guidance -> acts, every event type, consequences, legality
python packing/export_web.py          # the browser playground's data: play/world.json, replays of every event type
python packing/export_web.py --physics RUN NAME --assistant oracle --title "..."   # a rendered run -> play/rendered/NAME
python packing/export_web.py --pack3d                # the 3D page's world and CC0 assets -> play/pack/
python packing/test_web.py            # the playground's Python side: module closure, stepping == Harness.run, files
```

Install: Python 3.11, then `pip install mujoco==3.3.* bpy==4.5.* numpy scipy pillow trimesh` and `ffmpeg`. The `bpy` wheel is Blender 4.5 as a Python module. Cycles renders on OptiX or CUDA when available.

## Layers

| File | Layer | What it does |
|---|---|---|
| `actionspace.py` | action space | Verbs with typed signatures, per-asset parts and operations (with value ranges), reference resolution, value parsing, rejection codes, action text, `spec()` |
| `instruct.py` | instructions | Rule parser and Gemini parser (same request format), macro expansion (pack / unpack / lock / unlock), lazy grounding, auto-inserted `go_to` when out of reach |
| `discrete.py` | action catalogue | The closed list of fully specified actions (ids), `legal(sim)`, `canonical()` from grounded language |
| `rooms.py` | navigation | Rooms, the walking graph, line of sight from eye height past walls, door frames and furniture |
| `layout.py` | placement | Continuous placement in the case on a 5 mm height map: walls, support, height, the lid's swing, room for the hands; named spots; nudging to the nearest pose that fits |
| `packsim.py` | symbolic | State (items, case hardware, cloth), preconditions with feedback, effects, goal predicates, tasks, floor spots |
| `usersim.py` | user simulation | Personas, progress-conditioned events, the user model (plan, beliefs, replies to guidance), the harness, symbolic and physics backends, test assistants |
| `dialogue.py` | conversation | Closed speech-act catalogue for the user, guidance acts the user understands, `interpret()` from assistant text |
| `episodes.py` | cases | Case sampling and validation (silent dry run, oracle check, separated windows), persona variants |
| `web.py`, `export_web.py` | playground | A steppable session (`Harness.begin / tick / end`), frames (time span, walking path, state, turns, private labels) for the browser, `handle()` for the Pyodide worker, replays and physics runs exported for `play/` |
| `game.py`, `assistant_template.py` | you as the user | The 3D page's simulator side: what can be done from where you stand, running it through `PackSim`, the checklist, the assistant slot (`respond(view)`, empty), the played record |
| `executor.py` | grounding | Runs each accepted action with the physical skills, measures the outcome, commits it; per-action snapshots allow rewinding |
| `runner.py`, `hands.py`, `build_model.py` | physics | MuJoCo model (no equality constraints): the hands, the case with its hardware, the items, the optional cloth; latches and detents (`_mechanisms`), 25 Hz recording with hardware readings |
| `tasks.py`, `skills_case.py`, `skills_hw.py` | skills | Squeeze pick / carry / place, bottle grip, lid close and open, stand-up; fingertip and pinch skills for the zip sliders, pull tabs, dials, buttons, buckle, handle; rolling; folding |
| `operator_model.py`, `export_poses.py` | body | Torso least-squares solve, two-bone legs, look-ahead footsteps, head camera |
| `prepare_assets.py`, `item_assets.py`, `render_blender.py`, `encode.py`, `render_all.py`, `render_clips.py` | rendering | Scans, hardware overlays and cloth, path-traced head / room / close-up cameras, labelled videos |
| `verify.py`, `publish.py`, `export_spec.py` | evidence | Checks, publishing, the spec on the page |
| `dbg_hw.py`, `dbg_*.py`, `test_*.py` | development | `dbg_hw.py` runs the hardware skills on a closed case; `dbg_standup.py` retries a failed action from the checkpoint `generate.py` saves |

## Action space

The action space is a closed catalogue (`discrete.py`, 833 physical actions) of fully specified ids, plus a closed set of speech acts on the conversation channel (`dialogue.py`, see User simulation):
- navigation: `enter(room)`, `go_to(station | item)`, `look_at(item)`
- handling items: `pick_up(item)`, `place(item, spot)` with spot one of `wheel_end, handle_end, front_strip, bottle_corner, centre`, `place_on(item, other)`, `place_beside(item, other, side)`, `put_back(item)`
- the case: `open_lid`, `close_lid`, `zip_to(0|25|50|75|100)` (% closed, either way), `dock_zip_pulls`, `press_lock_button`, `turn_dial(dial_k, digit)`, `press_handle_button`, `fasten_strap`, `unfasten_strap`, `set_handle(0|1|2)`, `stand_up_case`, `roll_case(bed_foot | bedroom_door)`
- other: `fold(beach_towel)`, `wait` (2 s)

Each id also has an effect kind: `move` (`go_to`, `enter`, `look_at`), `wait`, or `state` (everything that changes the world). An error event commits at its first `state` step. Before that, guidance can still prevent it.

- The catalogue holds what a person can really do and the simulation carries out. Real actions that are not yet reliable in simulation are listed in `EXCLUDED` with the reason (the bottle laid down, the case laid down, unfolding, thirds, the side handle, changing the combination).
- Every argument comes from a finite list fixed by the scene. A quantity a person varies is taken in the steps a person would use (the zip in quarters). There are no coordinates, angles, durations, free text, optional arguments or defaults.
- One id is one skill with one effect. No effect has two ids: a code is set with `turn_dial`, one dial at a time.
- `legal(sim)` lists the ids whose preconditions hold now. An agent choosing from it cannot make a grounding or precondition error.
- Language is still grounded by the resolver: ambiguous words are resolved or returned with their candidates. The grounded action then passes through `canonical(action, state)`, which makes only deterministic rewrites:
  - a code becomes dial turns, and clicks become a digit;
  - `go_to(room)` becomes `enter(room)`;
  - beside with no side takes the first side that fits;
  - a pack request without a spot takes `layout.choose_named`.

  Anything else is rejected as `not_in_action_space`.
- Each record logs the action's `id`, and `verify.py` checks every one (`every_action_in_catalogue`).

The verbs below are the general form the grounder reads from language and the executor runs; only their catalogue instances are executed. The page lists each verb's resolution steps, physical check and effect (`trip/spec.json`).

| Group | Verbs |
|---|---|
| navigate | `go_to(target: Ref)`, `enter(room: Ref)`, `look_at(target: Ref)` |
| manipulate | `pick_up(target: Ref)`, `place(target = it, rel: in\|on\|beside = in, dest: Ref = suitcase, at: Spot?, side: Side?, yaw: Angle?, orient: Orient?)`, `put_back(target = it)` |
| articulate | `open(target: Ref, to: Frac = 1)`, `close(target: Ref, to: Frac = 0)`, `fasten(target)`, `unfasten(target)` |
| control | `press(target, hold: Dur = 0.3, times: Count = 1)`, `turn(target, to: Digit \| by: Extent)`, `set(target, value: Setting)`, `insert(target, into = lock)` |
| case | `stand_up`, `roll(to: Ref)` (`lay_down` is rejected: not simulated yet) |
| cloth | `fold(target, style = half)` (`unfold` and `thirds` are rejected with the reason) |
| other | `wait(seconds)`, `say(text)` |

Navigation: one `go_to` reaches only what is in the person's room or in view (an unobstructed line from eye height). Anything else is expanded (`packsim.navigation`) into `enter(room)` for each room on the way until the target is in view, `look_at(item)` for an item that has to be found (a physics ray cast records whether it was in sight), then `go_to`. Each step is its own logged action with a `parent` naming the request it came from. `--no-reach` turns off all insertion and expansion, so an out-of-view `go_to` is refused (strict mode for evaluating agents).

Suitcase parts (from the scan and spinner-case manuals): `lid` (open / close, fully only), `main_zip` (open / close to any fraction), `zip_pulls` (insert into the lock), `lock` (set a code), `dial_1..3` (turn to a digit or by clicks), `lock_button` (press: frees the pulls only when the dials show the code), `trolley_handle` (set to stop 0, 1, 2; only stop 1 with the case lying on the bed), `trolley_button`, `strap` (fasten / unfasten), `side_handle` (grip only), `wheels`.

Rejection codes: `unknown_verb`, `missing_argument`, `bad_value`, `unknown_ref`, `ambiguous_ref` (with candidates), `wrong_kind`, `unsupported_operation`, `unsupported_value`, `out_of_range`, `not_in_action_space`. State-dependent refusals ("the zip pulls are locked in the lock") come from `PackSim.check`.

## User simulation

An episode runs in a closed loop between the simulated user, the harness and an assistant (`usersim.py`). It follows the IVG-style simulators: a hidden schedule of progress-conditioned events, a harness that takes over the action queue during an error while the user keeps talking, guidance that prevents an error before it commits or helps recovery after it, and execution feedback when an action cannot be done.

- **User model.** The user's plan is right by construction: heavy items first, the camera last, the lid, zip, lock and stand-up. It replans after every action from its beliefs: what belongs in the case, what it has forgotten, whether the zip is shut, where the heavy bag goes. Events corrupt those beliefs, and guidance fixes them. It states the task and any preferences (e.g. "I like the sweatshirts at the handle end"), asks its questions, and answers every message.
- **Personas** set both the body and the conversation: `brisk` (fast, terse, takes a correction 60 % of the time and more on repeats, waits 2 s for answers), `steady` (cooperative, chatty, takes advice readily, 35 % gullible), `deliberate` (slow, never takes advice that contradicts the task, waits 8 s).
- **Events** (`EVENT_TYPES`) each have a declarative trigger over the state and the user's next intended action (`trigger` in `case.json`, e.g. `{"next": ["go_to", "towel"], "hands": "free"}` or `{"packed_at_least": 3, "lid": "open"}`), targets, a deviation script with a marked commit step, a resolution condition, a window (maximum steps under harness control) and the expected intervention.
  - errors: `wrong_object`, `omission`, `repetition`, `nav_deviation`, `heavy_on_top`, `premature_close`, `incomplete_zip` (found out when the case will not stand up or lock), `stall`
  - questions, each with a ground-truth answer from the trajectory: `ask_should_pack`, `ask_packed`, `ask_left`, `ask_since` and `ask_elapsed` (temporal)
  - proactive requests: `request_remind` (the user then forgets the item unless reminded), `request_notify_done` (the user waits for the notice before closing)
  - `plan_change`
- **Guidance.** An assistant speaks free text. `dialogue.interpret` turns it into guidance acts: `stop`, `dont(verb, item)`, `do(instruction)`, `next(item)`, `not_on_list`, `on_list`, `packed`, `not_packed`, `on_floor`, `where`, `zip_open`, `all_packed`, `answer`, `ack`, or `unclear`. Items are resolved with the instruction grounder's resolver, and instructions go through `instruct` and `discrete.canonical`, so an assistant can never make the user run an action outside the catalogue. The user takes an act if it is needed (it changes something) and passes a persona draw: compliance if it agrees with the task, gullibility if it does not. A wrong answer or bad advice has consequences, such as an extra item packed or an item left out.
- **Outcomes** per event: `prevented`, `corrected`, `self_recovered`, `unresolved`; `answered_correct` / `answered_wrong` / `unanswered`; `reminded` / `missed`; `notified` / `not_notified`; `acknowledged`.
- **Clock.** The symbolic backend times each action with the medians measured on the physics runs (walking scaled by persona). The physics backend uses simulation time, and speech holds the body still for its duration with a caption.
- **Records.** `episode.json` splits what an assistant may see (task, preferences, conversation, per-step time, frames, position for a top-down trajectory, what is in the hands) from what it may not: the schedule with outcomes, step labels, source (plan / event / guidance), controller (user / harness), the user's beliefs and preference goals.
- **Validation.** A sampled case must pass a silent dry run (every event fires, every action in the catalogue, no refusals except where an error is meant to be found out) and an oracle run (an assistant that reads the schedule prevents, corrects or answers every event and the goals are met). This rejects triggers that can never be acted on.
- **Test assistants** (`--assistant`): `none`, `oracle`, `interactive` (you type), or a JSON list of `{"step", "text"}`.
- **The model slot.** `--assistant my_model.py` loads a file that defines `respond(view)` and wraps it in `usersim.Model`. It is called before every action of the user and once at the end. `view` (`usersim.public_view`) holds only the task, the stated preferences, the conversation and the observations; it never holds labels, events, beliefs or the plan. Return a sentence, a list of sentences, or `None` to stay quiet. `assistant_template.py` is the empty slot: it returns `None`, so the episode is the same as `--assistant none` (checked in `test_usersim.py`). Copy it to plug in a model.
- **The user's plan.** `UserModel.plan(state)` lists what the person means to do next from their own beliefs: each item to pack in order (heavy first, fragile last), then close the lid, zip it, lock it if the task has a lock, and stand the case up. Each step is done, next or todo. A forgotten item drops off the plan and a wrongly believed one gets on it, so the plan shows how an event has led the person astray. It is private and appears in the playground's hidden panel.

Limitations: guidance takes effect between actions, not mid-action (navigation is already split into room-by-room steps, so a walk can be redirected). The guidance reader is rule-based, and anything it cannot read is `unclear`, which the user asks about.

## Browser playground (`play/`)

[`play/index.html`](../play/index.html) runs the symbolic simulator in the browser: `play/worker.js` loads Pyodide and numpy, fetches the modules listed in `play/modules.json` from `packing/`, and calls `web.handle()`. No GPU, MuJoCo or Blender is involved, and the same seed gives the same episode as `episodes.run` (checked by `test_web.py` and in the browser).

- **Play live**: pick a task, persona, seed and up to three event types (or none for a random schedule). The case is sampled and validated like any other (silent run plus oracle run). You type as the assistant. A message sent while the person is between actions is heard at once. One sent during an action is heard before the next one. Suggested replies are built from what an assistant could see. *Peek* asks the oracle what it would say now, which uses the hidden schedule.
- **Replays**: every sampled event type, each case run with nobody speaking and with the oracle (`export_web.py`).
- **Rendered episodes**: physics runs rendered with `render_all.py`, published with `export_web.py --physics`. The map, conversation and annotations follow the video's simulated time (`times` maps each encoded frame to it).
- One renderer draws all three: the top-down map (rooms, walls, furniture, items, the case with its lid, the person walking the graph's route), the inside of the case, the conversation, the goals, the action log and a timeline. *Show what the assistant can't see* adds the schedule with outcomes, step labels and commit points, how each message was read, and the user's beliefs.

The map draws walking along the walking graph between stand points. Physics runs shortcut corners where the way is clear, so the map's route is close to the real one but not identical.

## Pack it yourself (`play/pack.html`)

You are the user. The page is a real-time 3D version of the apartment in the rendered videos (three.js, WebGL), and the simulator decides everything that is not walking:

- **Scene.** `export_web.py --pack3d` writes `play/pack/world3d.json` from `scene.py` (rooms, walls, windows, furniture, items, stations, the case and its hardware, the overview cameras) and copies the CC0 Poly Haven models, textures and sky the renders use. The materials, sun, room panels, sky and AgX tone mapping follow `render_blender.py`. The scanned garment piles and the scanned case are BlenderKit assets that may not be redistributed, so the page builds stand-ins of the same sizes (folded layers in fabric scans; a hard-shell case with lining, zip, lock, wheels and handle). The mannequin is the renders' `Body`, posed by procedural walking and two-bone reaching.
- **Playing.** Walk with WASD or the arrow keys (collision with walls and furniture). `game.options(x, y, heading)` lists the catalogue actions whose preconditions hold once you are at the station they are done from: pick up what is in front of you, put what you hold in any of the case's named spots, put it back, close the lid, zip, dock the pulls, turn a dial, stand the case up. `E` does the first, `1`–`9` the others. `game.act` records the walk as `go_to` and runs the action through `PackSim`, so the rules are the generator's (a heavy item on top of the clothes shows as a broken rule on the checklist; the lid will not close over a full case).
- **Checklist.** The task's goal predicates (`PackSim.goals`) in words, ticked as they hold.
- **Assistant slot.** After every action, and whenever you say something (`T`), `assistant_template.respond(view)` is called with what an assistant may see. It is empty, so it never speaks; filling it in makes the page give guidance.
- **Record.** *Download the episode* gives the actions as catalogue ids with times, the conversation, the walked trajectory and the goals: the same action format as a generated episode.
- **Cameras.** Follow (keeps clear of walls), the room cameras of the rendered videos, and first person. Quality drops from high (ambient occlusion, 4k shadows) to medium on its own if frames are slow.

`test_web.py` plays every task through `game.options`/`game.act` and checks the page's files (and that no BlenderKit file is published). It has also been played in headless Chrome by key presses and by its test hooks (`window.PACK`).

## Checks (`validation.json`)

- one continuous run with gravity on and no equality constraints
- every scheduled event fired; in sampled cases, refusals only where an event or guidance produced them (instructions may be refused: they are reported)
- the action log agrees with physics for every item, the zip fraction, the dials, the pulls, the handle stop and the buckle
- the lid shut or open as logged; the zipper engaged only with the lid down; the case upright on four wheels or lying on its back, as logged; everything settled
- every `go_to` target in the person's room or in view at its start; every looked-for item in sight (ray cast); every `enter` ends in its room
- every pick lifted by contact, no drops; the case undisturbed while it was being packed
- contact penetration under 5 mm (10 mm on soft piles and cloth), finger joints within 2.5° of their limits
- fixed segment lengths, no unreachable arm or leg frames, the pelvis clear of walls and furniture

## Limitations

See the page's limitations section: the idealised mechanisms (the zip holding the lid, the latches, the dial detents, the mid-depth hinge), what is rejected rather than simulated (combination change, side-handle carry, laying the case down, unfolding, folding in thirds), the kinematic body, and the assumed physical parameters.

## Licences

- MyoHand model: Apache-2.0 (`hand/LICENSE-myosim.txt`).
- Poly Haven assets: CC0, listed with hashes in `assets/asset-manifest.json`.
- BlenderKit scans: the garment piles (antoine f, Paweł Wałasiewicz, Rex Hans, GL3D Gabriela Linhares) and the "Black suitcase" (L'ANE). Royalty-free licence: they may be used in renders but not redistributed, so `fetch_assets.py` downloads them and the repository never includes them.
- Toiletry bag, bottle, liner, hardware overlays, towel, bed, cabinets and apartment shell: procedural.
