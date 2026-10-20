# Packing simulator: source

A person packs a large spinner suitcase lying open on the bed of a two-room apartment, then operates it: closes and zips it, docks the zip pulls in the combination lock, dials a code, fastens the lid strap, extends the telescopic handle, stands the case on its wheels and rolls it to the door. A loose towel can be folded on the kitchen counter. Runs come from a sampled case (task, seed, persona, private event schedule) or from instructions in plain language or JSON. Every request is grounded into a typed action, refused with a reason if it cannot be, and executed by contact in MuJoCo with anatomical hands. Runs are rendered with Blender Cycles.

## Pipeline

```
python packing/fetch_assets.py        # Poly Haven (CC0) + BlenderKit scans (royalty-free; not committed)
python packing/prepare_assets.py      # clean / orient / scale the garment scans; cut the suitcase scan and split its moving parts
python packing/generate.py RUN --reference                 # the trajectory on the page
python packing/generate.py RUN --seed 12                   # a sampled case (add --task / --events / --persona)
python packing/generate.py RUN --seed 12 --dry-run         # symbolic plan only, no physics
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
```

Install: Python 3.11, then `pip install mujoco==3.3.* bpy==4.5.* numpy scipy pillow trimesh` and `ffmpeg`. The `bpy` wheel is Blender 4.5 as a Python module. Cycles renders on OptiX or CUDA when available.

## Layers

| File | Layer | What it does |
|---|---|---|
| `actionspace.py` | action space | Verbs with typed signatures, per-asset parts and operations (with value ranges), reference resolution, value parsing, rejection codes, action text, `spec()` |
| `instruct.py` | instructions | Rule parser and Gemini parser (same request format), macro expansion (pack / unpack / lock / unlock), lazy grounding, auto-inserted `go_to` when out of reach |
| `layout.py` | placement | Continuous placement in the case on a 5 mm height map: walls, support, height, the lid's swing, room for the hands; named spots; nudging to the nearest pose that fits |
| `packsim.py` | symbolic | State (items, case hardware, cloth), preconditions with feedback, effects, goal predicates, tasks, floor spots |
| `episodes.py` | user + events | Replanning user policy, progress-conditioned events, personas, case sampling with a symbolic dry run |
| `executor.py` | grounding | Runs each accepted action with the physical skills, measures the outcome, commits it; per-action snapshots allow rewinding |
| `runner.py`, `hands.py`, `build_model.py` | physics | MuJoCo model (no equality constraints): the hands, the case with its hardware, the items, the optional cloth; latches and detents (`_mechanisms`), 25 Hz recording with hardware readings |
| `tasks.py`, `skills_case.py`, `skills_hw.py` | skills | Squeeze pick / carry / place, bottle grip, lid close and open, stand-up; fingertip and pinch skills for the zip sliders, pull tabs, dials, buttons, buckle, handle; rolling; folding |
| `operator_model.py`, `export_poses.py` | body | Torso least-squares solve, two-bone legs, look-ahead footsteps, head camera |
| `prepare_assets.py`, `item_assets.py`, `render_blender.py`, `encode.py`, `render_all.py`, `render_clips.py` | rendering | Scans, hardware overlays and cloth, path-traced head / room / close-up cameras, labelled videos |
| `verify.py`, `publish.py`, `export_spec.py` | evidence | Checks, publishing, the spec on the page |
| `dbg_hw.py`, `dbg_*.py`, `test_*.py` | development | `dbg_hw.py` runs the hardware skills on a closed case; `dbg_standup.py` retries a failed action from the checkpoint `generate.py` saves |

## Action space

An action is a verb with typed arguments; the page lists each verb's resolution steps, physical check and effect (`trip/spec.json`).

| Group | Verbs |
|---|---|
| navigate | `go_to(target: Ref)`, `look_at(target: Ref)` |
| manipulate | `pick_up(target: Ref)`, `place(target = it, rel: in\|on\|beside = in, dest: Ref = suitcase, at: Spot?, side: Side?, yaw: Angle?, orient: Orient?)`, `put_back(target = it)` |
| articulate | `open(target: Ref, to: Frac = 1)`, `close(target: Ref, to: Frac = 0)`, `fasten(target)`, `unfasten(target)` |
| control | `press(target, hold: Dur = 0.3, times: Count = 1)`, `turn(target, to: Digit \| by: Extent)`, `set(target, value: Setting)`, `insert(target, into = lock)` |
| case | `stand_up`, `roll(to: Ref)` (`lay_down` is rejected: not simulated yet) |
| cloth | `fold(target, style = half)` (`unfold` and `thirds` are rejected with the reason) |
| other | `wait(seconds)`, `say(text)` |

Suitcase parts (from the scan and spinner-case manuals): `lid` (open / close, fully only), `main_zip` (open / close to any fraction), `zip_pulls` (insert into the lock), `lock` (set a code), `dial_1..3` (turn to a digit or by clicks), `lock_button` (press: frees the pulls only when the dials show the code), `trolley_handle` (set to stop 0, 1, 2; only stop 1 with the case lying on the bed), `trolley_button`, `strap` (fasten / unfasten), `side_handle` (grip only), `wheels`.

Rejection codes: `unknown_verb`, `missing_argument`, `bad_value`, `unknown_ref`, `ambiguous_ref` (with candidates), `wrong_kind`, `unsupported_operation`, `unsupported_value`, `out_of_range`. State-dependent refusals ("the zip pulls are locked in the lock") come from `PackSim.check`.

## Checks (`validation.json`)

- one continuous run with gravity on and no equality constraints
- every scheduled event fired; in sampled cases, no refused actions (instructions may be refused: they are reported)
- the action log agrees with physics for every item, the zip fraction, the dials, the pulls, the handle stop and the buckle
- the lid shut or open as logged; the zipper engaged only with the lid down; the case upright on four wheels or lying on its back, as logged; everything settled
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
