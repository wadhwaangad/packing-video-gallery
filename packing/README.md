# Packing simulation

The packing page presents one deterministic 90-second run through synchronized room and operator views. Seven items move through approach, reach, grip, lift, carry, align, lower, release, and settle phases. Held-item paths and grasps are authored; released items use the lightweight fixed-step contact solver. The clothing models are local CC0 assets, so they load without a remote asset service.

## Open the simulation

Run `packing/start.ps1` from the repository folder, or open `packing/index.html` through any static web server. The page does not need a separate interactive bedroom scene or external JavaScript modules.

## Controls

- **Play sequence** starts or pauses the run.
- **Restart** returns to the start.
- Use the timeline, chapter buttons, and playback speed menu to seek or replay the run.
- **First person** switches the operator camera between head-carried and third-person views.

## Simulation scope

The available run evidence identifies MuJoCo 3.14.0 and Blender in a separate assembly pipeline, but its named implementation scripts are not present here. This packing page therefore remains a browser-based task replay: grasp and carry motion are scripted, and a reduced AABB solver handles gravity, friction, contact, and yaw after release. The source evidence informs the explicit action phases and the separation between motion, contact, and rendering; it does not provide a reusable MuJoCo packing model. Full human biomechanics, cloth deformation, six-degree-of-freedom tumbling, and measured suitcase capacity are outside the model.

Clothing model provenance and license notes are in [assets/garments/ASSET_SOURCES.md](assets/garments/ASSET_SOURCES.md). Run checks and limitations are in [validation.json](validation.json).
