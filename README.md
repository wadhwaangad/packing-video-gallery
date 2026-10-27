# Packing simulator

[`index.html`](index.html) presents a simulator for packing a suitcase in a two-room apartment, with a reference trajectory rendered from the head camera and room cameras.

- **Executable tasks, not recordings.** A case is a task (the packing list), a seed, a persona and a private schedule of events. A scripted user replans after every action, so the same case can unfold differently when the world changes.
- **Semantic actions with a grounder.** A closed catalogue of 833 physical actions (navigation, handling items, every part of the case) plus speech acts on a separate conversation channel. Refused actions return feedback and change nothing.
- **Simulated user.** The user replans from its own beliefs, talks (task, preferences, questions, requests), and answers whatever an assistant says, by persona (compliance, gullibility, patience). A hidden schedule hands the action queue to a harness during errors. Accepted guidance prevents an error before it commits or corrects it after.
- **Object families.** Folded piles, rigid boxes, fragile boxes and the bottle each decide which contact skill grounds `pick_up` and `put_in`.
- **Goal predicates.** Success is a set of predicates (packed items, no extras, bottle upright, heavy items on the floor, nothing on the camera, closed, zipped, upright), not a reference route.
- **Progress-conditioned events.** These are wrong object, omission, repetition, navigation deviation, heavy item on top, bottle laid down, premature close and stall, each with optional self-recovery. Questions and plan changes use separate channels.
- **Physics.** Every action runs in MuJoCo with two anatomical MyoHand hands. Grasps are contact-only, with no welds or equality constraints, and the outcome is verified before it is committed.
- **Photoreal renders.** The garment piles and the suitcase are photo-scanned BlenderKit models. The apartment uses Poly Haven assets. Frames are path-traced with Blender Cycles.

**Try it:** [`play/`](play/index.html) runs the simulated user live in the browser (Pyodide, no GPU). You type as the assistant while they pack. It also holds replays of every event type, with and without an assistant, and rendered physics episodes with the map and conversation synced to the video.

Source, pipeline and file formats: [`packing/README.md`](packing/README.md). The reference trajectory's records are in [`trip/`](trip/). [`hands/`](hands/index.html) holds the earlier hand-only lift and turn physics tests.

Limitations: the body is kinematic, solved around the simulated forearms. Garments are compliant rigid piles, not cloth. The zipper is idealised, and the user is scripted. These are simulation checks, not evidence of real-world transfer.

Assets: BlenderKit (royalty-free licence; fetched by script, not redistributed), Poly Haven (CC0). Hands: MyoHub/myo_sim (Apache-2.0).
