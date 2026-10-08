# Assembly video journal

Live site: [wadhwaangad.github.io/packing-video-gallery](https://wadhwaangad.github.io/packing-video-gallery/). GitHub Pages deploys the static gallery from the `main` branch.

The newest pair shows one continuous BEKVÄM assembly run from loose parts to a released stool: eight wooden components, two dowels, and eleven fasteners. The operator view uses a moving head-position camera and connected arms with finite reach. The outside view shows the same recorded run. The previews play at four times simulation speed. A separate operator video preserves every recorded frame at 25 fps and original movement speed. The page includes operation chapters.

`run-evidence.json` records the completion checks, avatar reach checks, and source hashes. `assembly-events.json` contains the recorded assembly milestones. These are simulation checks, not evidence of real-world transfer.

Limitations: the human avatar is kinematic, without balance, foot-contact, or whole-body collision simulation. Grasps and bore/thread engagement use idealized connectors. Material and contact properties are assumed. The rebate collision profiles include explicit clearance for overlap in the source CAD; this has not been calibrated against real furniture. Fingers are visual geometry, not an independently controlled contact model. The footage is simulated, not photoreal human capture.

Furniture geometry and derived videos: [IKEA3DAssemblyDataset](https://github.com/IKEA/IKEA3DAssemblyDataset), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Older supported-human-lift media uses [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym), MIT. The new operator avatar is procedural.

## Assembly data explorer

[Browse the collected data samples](https://hellomuffin.github.io/assembly-video-gallery/data/index.html): 24 playable excerpts from 15 datasets, original annotation timelines and records, part-mask stills, a 36-source assembly registry, and the complete 124-row slide-linked training-catalog audit. Filter by dataset, assembly type, and annotation. Acquisition counts are a dated local snapshot; source licenses and original-video rights remain separate.

## Bedroom packing simulation

[Explore the room-scale suitcase-packing simulation](packing/index.html): a procedural 3D room, operator and overview cameras, synchronized chapter seeking, a seven-item packing sequence, and linked run evidence. The task motion and grasps are scripted; the page documents the physics and transfer limits.
