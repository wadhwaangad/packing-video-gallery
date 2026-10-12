# Assembly video journal

## New: packing a suitcase for a trip

[`packing.html`](packing.html) shows one continuous simulated run in a two-room apartment. A person collects eight items from the dresser, the bed, the bathroom, the kitchen and the living room, and packs them into a hard-shell suitcase lying on the bed. They then close and zip it and stand it on its wheels. All grasps are contact-only (no hand-object welds), using MyoHand hands in MuJoCo. The page has a moving head-camera view and room cameras rendered with Blender Cycles, plus completion checks in [`trip/validation.json`](trip/validation.json). Source and pipeline: [`packing/`](packing/README.md). Assets: Poly Haven (CC0). Hands: MyoHub/myo_sim (Apache-2.0).

Watch the videos at https://hellomuffin.github.io/assembly-video-gallery/ .

The newest pair shows one continuous BEKVÄM assembly run from loose parts to a released stool: eight wooden components, two dowels, and eleven fasteners. The operator view uses a moving head-position camera and connected arms with finite reach. The outside view shows the same recorded run. The previews play at four times simulation speed. A separate operator video preserves every recorded frame at 25 fps and original movement speed. The page includes operation chapters.

`run-evidence.json` records the completion checks, avatar reach checks, and source hashes. `assembly-events.json` contains the recorded assembly milestones. These are simulation checks, not evidence of real-world transfer.

Limitations: the human avatar is kinematic, without balance, foot-contact, or whole-body collision simulation. Grasps and bore/thread engagement use idealized connectors. Material and contact properties are assumed. The rebate collision profiles include explicit clearance for overlap in the source CAD; this has not been calibrated against real furniture. Fingers are visual geometry, not an independently controlled contact model. The footage is simulated, not photoreal human capture.

Furniture geometry and derived videos: [IKEA3DAssemblyDataset](https://github.com/IKEA/IKEA3DAssemblyDataset), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Older supported-human-lift media uses [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym), MIT. The new operator avatar is procedural.

## Assembly data explorer

[Browse the collected data samples](https://hellomuffin.github.io/assembly-video-gallery/data/index.html): 24 playable excerpts from 15 datasets, original annotation timelines and records, part-mask stills, a 36-source assembly registry, and the complete 124-row slide-linked training-catalog audit. Filter by dataset, assembly type, and annotation. Acquisition counts are a dated local snapshot; source licenses and original-video rights remain separate.
