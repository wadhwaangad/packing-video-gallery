# Packing a suitcase: simulation source

This folder builds the packing videos on [`packing.html`](../packing.html). It is the
packing counterpart of the assembly journal in this repository: one continuous MuJoCo
run, anatomical hands, a kinematic body solved around the simulated forearms, and
path-traced renders with completion checks.

## The task

A person walks through a two-room apartment and packs a medium hard-shell suitcase
that lies open on the bed:

| Trip | Item | From | Grasp |
|---|---|---|---|
| 1 | folded jeans | bedroom dresser | two-hand side squeeze |
| 2 | white and navy T-shirts, together | bedroom dresser | two-hand side squeeze |
| 3 | knit sweater | bed | two-hand side squeeze |
| 4 | folded towel | bathroom shelf | two-hand side squeeze |
| 5 | toiletry bag | bathroom vanity | two-hand side squeeze |
| 6 | insulated water bottle | kitchen counter | one-hand power grip (thumb opposing) |
| 7 | leather notebook | living-room coffee table | two-hand side squeeze |
| 8 | camera | living-room coffee table | two-hand side squeeze |

After the trips, the person:

1. Hooks the lid from the foot of the bed, swings it past vertical and lets it fall shut.
2. Presses the lid down and runs the zipper.
3. Hooks the top handle and stands the case up.
4. Lifts it clear of the bed and sets it on its wheels.

## Pipeline

```
python packing/fetch_assets.py            # CC0 Poly Haven models, textures and HDRI (+ hashes)
python packing/run_full.py RUN            # one continuous simulation (about 12 min on a laptop CPU)
python packing/export_poses.py RUN        # body solved around the simulated forearms
python packing/verify.py RUN              # completion and quality checks -> RUN/validation.json
python packing/render_all.py RUN          # Blender Cycles: head camera + room cameras, encode
python packing/publish.py RUN             # copy videos and evidence to trip/
```

Install: Python 3.11, then `pip install mujoco==3.3.* bpy==4.5.* numpy scipy pillow trimesh`
and `ffmpeg`. The `bpy` wheel is Blender 4.5 as a Python module, so no separate Blender
install is needed. Cycles renders on OptiX or CUDA when available.

## Files

- `scene.py`: apartment layout, furniture, suitcase dimensions, items, packing slots, trips.
- `hands.py`: two MyoHand hands (right and mirrored left). Torque-limited joint servos
  replace the muscles. Each hand sits on a 6-DoF wrist carriage: slides with force caps,
  plus a ball joint with a torque-capped rotation-vector controller. The forearm colliders
  are reshaped to a realistic wrist cross-section, with palm and thenar soft-tissue pads.
- `build_model.py`: MJCF composer. There are no equality constraints anywhere.
- `runner.py`: online controller. Min-jerk moves, palm-frame targets through a wrist
  kinematics table, walking with bounded turn rate and acceleration, 25 Hz recording and
  50 Hz diagnostics.
- `tasks.py`, `skills_case.py`: grasp, carry, place, lid, zipper and lift skills. Targets
  are computed from the live simulation state.
- `operator_model.py`, `export_poses.py`: torso least-squares solve (upper arms reach the
  simulated elbows), two-bone legs, look-ahead footsteps, head camera.
- `render_blender.py`, `encode.py`, `render_all.py`, `publish.py`: rendering and publishing.
- `dbg_*.py`: the diagnostic scripts used while developing each skill (kept for reference).

## Checks (`validation.json`)

The run passes only if all of these hold:

- one continuous run, gravity on, no equality constraints
- all nine items inside the closed case
- the lid within 2° of shut, and the zipper engaged only after the closure check
- the case upright within 5° on four wheels and settled
- every item lifted by contact, with no drops
- the case undisturbed during packing
- contact penetration under 5 mm, finger joints within 2° of their limits
- fixed segment lengths with no unreachable arm or leg frames
- the pelvis clear of walls and furniture

## Limitations

- The body is kinematic: no balance or foot forces, and no whole-body collision.
- The zipper is idealised. After the closure check, the lid hinge range is narrowed to the
  closed pose. The hand traces only the near side of the track.
- Garments are compliant rigid bodies, not cloth.
- Masses, friction and finger strengths are assumed values.
- Wrist carriages stand in for the shoulder and elbow.
- Skin is visual geometry bound to rigid finger bodies.
- Nothing here is calibrated sim-to-real.

## Licences

- MyoHand model: Apache-2.0 (`hand/LICENSE-myosim.txt`).
- Poly Haven assets: CC0, listed with hashes in `assets/asset-manifest.json` (fetched, not
  committed).
- Suitcase, garments, toiletry bag, bottle, bed, cabinets and apartment shell: procedural.
