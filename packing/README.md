# Packing simulator: source

A person packs a large hard-shell suitcase lying open on the bed of a two-room apartment, then closes, zips and stands it on its wheels. Every run is a new trajectory: the case (task, seed, persona, event schedule) decides what happens. Each semantic action runs in a contact-only MuJoCo simulation, and the run is rendered with Blender Cycles.

## Pipeline

```
python packing/fetch_assets.py        # Poly Haven (CC0) + BlenderKit scans (royalty-free; not committed)
python packing/prepare_assets.py      # clean / orient / scale the garment scans, cut the suitcase scan
python packing/generate.py RUN --reference          # the trajectory on the page
python packing/generate.py RUN --seed 12            # a sampled case (add --task / --events / --persona)
python packing/generate.py RUN --seed 12 --dry-run  # symbolic plan only, no physics
python packing/export_poses.py RUN    # body solved around the simulated forearms
python packing/verify.py RUN          # checks -> RUN/validation.json
python packing/render_all.py RUN      # Blender Cycles: head camera + room cameras, encoded
python packing/publish.py RUN         # copy videos and records to trip/
python packing/export_spec.py         # actions / families / items / events -> trip/spec.json for the page
```

Install: Python 3.11, then `pip install mujoco==3.3.* bpy==4.5.* numpy scipy pillow trimesh` and `ffmpeg`. The `bpy` wheel is Blender 4.5 as a Python module. Cycles renders on OptiX or CUDA when available. One reference trajectory takes about 6 minutes of physics on a laptop CPU. Rendering both views at stride 4 and 48 samples takes about 3 hours on an RTX 4060.

## Layers

| File | Layer | What it does |
|---|---|---|
| `scene.py` | world | Apartment, stations (where you stand to act), walking graph, items with families, case regions, the scanned suitcase's dimensions |
| `packsim.py` | symbolic | State, the action space, preconditions with feedback, effects, goal predicates, task library, text grounder (`parse("pick up the T-shirt pile")`) |
| `episodes.py` | user + events | Replanning user policy, progress-conditioned events, personas, case sampling with a symbolic dry run that rejects invalid schedules |
| `executor.py` | grounding | Runs each accepted action with the physical skills, verifies the outcome, commits it to PackSim; per-action snapshots allow rewinding |
| `runner.py`, `hands.py`, `build_model.py` | physics | MuJoCo model (no equality constraints), min-jerk targets, walking, 25 Hz recording, rewind snapshots |
| `tasks.py`, `skills_case.py` | skills | Squeeze pick / carry / place, bottle power grip, lid close and open, zip and unzip, stand the case up |
| `operator_model.py`, `export_poses.py` | body | Torso least-squares solve, two-bone legs, look-ahead footsteps, head camera |
| `prepare_assets.py`, `item_assets.py`, `render_blender.py`, `encode.py`, `render_all.py` | rendering | Scans and procedural items, path-traced frames, labelled videos |
| `verify.py`, `publish.py`, `export_spec.py` | evidence | Checks, publishing, the spec shown on the page |
| `dbg_*.py`, `test_jeans.py`, `check_packing.py`, `settle_check.py` | development | Diagnostic scripts used while developing the skills. The older `dbg_*` scripts were written against the first, fixed-trip scene and are kept for reference; `dbg_standup.py`, `dbg_contacts.py` and `dbg_lid_contacts.py` retry a failed action from the checkpoint `generate.py` saves |

## Action space

| Class | Action | Grounds to | Refused when |
|---|---|---|---|
| navigate | `go_to <station \| item>` | a station, or the stand point in front of an item's current position | unknown target; the item is in your hands |
| manipulate | `pick_up <item>` | a visible item on the station you face | hands full; not in front of it; item is in the case |
| | `put_in <region> [upright\|lying]` | a region of the open case (drop height from the piles already there) | not holding; not at the case; lid closed; family not accepted; something fragile underneath; full |
| | `put_back` | the held item's home spot | not at its home station |
| | `take_out <item>` | an item in the case | hands full; lid closed; something on top of it |
| case | `close_lid`, `open_lid`, `stand_up` | from the foot of the bed | hands full; wrong state (open / closed / zipped) |
| | `zip`, `unzip` | from beside the bed | lid open; not zipped |
| other | `wait <s>`, `say <text>` | | |

## Object families

| Family | Items | `pick_up` | `put_in` | Constraint |
|---|---|---|---|---|
| `folded_pile` | trousers, T-shirts, sweater pile, striped shirts, sweatshirts, towel, towel stack, sweatpants and socks | two-hand side squeeze | lay flat in a garment column | stackable |
| `rigid_box` | toiletry bag, notebook | two-hand side squeeze | lay flat (bag in the front strip) | the heavy bag must rest on the case floor |
| `fragile_box` | camera | two-hand side squeeze | lay flat in the front strip | nothing on top |
| `bottle` | water bottle | one-hand power grip, thumb opposing | stand upright in the front corner | must stay upright |

## Events

`wrong_object`, `omission`, `repetition`, `nav_deviation`, `heavy_on_top`, `bottle_lying`, `premature_close` and `stall` are errors; most have an optional self-recovery. `question` and `plan_change` are non-error channels; a plan change edits the packing list mid-episode. Triggers are semantic ("about to fetch X", "after k items are packed", "about to close the lid"). An active event owns the user's action queue until it completes.

## Outputs per trajectory

`case.json` (task, persona, private schedule, dry-run result), `actions.json` (one record per action: text, label, event id, start/end time and frame, feedback, symbolic state after), `goals.json`, `frames.json` (25 Hz body poses), `events.json` (lift, placement, lid and zip checks), `stages.json` (captions), `model.xml`, `result.json`, `validation.json`, then `operator_poses.json` and `videos/`.

## Checks (`validation.json`)

- one continuous run with gravity on and no equality constraints
- every scheduled event fired, no refused actions, every placement landed in its region
- the action log agrees with physics for every item (packed in the log ⇔ inside the closed case)
- the lid within 2° of shut, the zipper engaged only after the closure check, the case upright within 5° on four wheels and settled
- every pick lifted by contact, no drops
- the case undisturbed while it was being packed
- contact penetration under 5 mm, finger joints within 2° of their limits
- fixed segment lengths, no unreachable arm or leg frames, the pelvis clear of walls and furniture

Goal predicates (what an assistant should have prevented) are reported separately in `goals.json`. An unrecovered error is a valid trajectory with a failed goal.

## Limitations

- The body is kinematic: no balance or foot forces, and no whole-body collision.
- Garments are compliant rigid piles, not cloth; a pile moves as one folded unit.
- The zipper is idealised: after a closure check, the lid hinge range is narrowed to the closed pose. The hinge sits at mid-depth, standing in for the fabric hinge, so both halves lie flat. A thin zip-track lip is what the fingertips catch when opening the lid.
- Standing the case up uses a two-hand overhand grip on the trolley bar, tested with a short pull. A grip that
  slips is released and retried with the hands placed differently (a re-grip); attempts are logged as `hook_check` events.
- A physically failed action is rewound (physics, recordings, symbolic state) and retried with a small variation
  in stand position and timing, up to three times (`action_retry` events); the saved trajectory is one continuous
  run. A case that still fails is rejected (`result.json` status `failed: ...`, plus `failure.pkl` for debugging).
- The user is a scripted policy; the event taxonomy is controlled, not exhaustive.
- Masses, friction and finger strengths are assumed values. Wrist carriages stand in for the shoulder and elbow. Nothing here is calibrated sim-to-real.

## Licences

- MyoHand model: Apache-2.0 (`hand/LICENSE-myosim.txt`).
- Poly Haven assets: CC0, listed with hashes in `assets/asset-manifest.json`.
- BlenderKit scans: the garment piles (antoine f, Paweł Wałasiewicz, Rex Hans, GL3D Gabriela Linhares) and the "Black suitcase" (L'ANE). Royalty-free licence: they may be used in renders but not redistributed, so `fetch_assets.py` downloads them and the repository never includes them.
- Toiletry bag, bottle, liner, bed, cabinets and apartment shell: procedural.
