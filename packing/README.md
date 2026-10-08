# Packing simulation

The editorial gallery in `index.html` keeps the synchronized overview/operator views and seekable 90-second sequence. The synchronized overview and operator views share one deterministic run. Each item trip is logged as approach, reach, grip, lift, carry, align, lower, release, and settle; the avatar bends and reaches with both arms as those stages play. The authored task paths and scripted grasp events remain kinematic. The **Enter the detailed 3D bedroom** link opens the interactive Three.js/Rapier scene with the bundled furniture and travel-item models.

## Open the detailed 3D room

From the repository folder, run:

```powershell
.\packing\start.ps1
```

The script serves the repository on `localhost:8011` and opens the gallery. The 3D scene imports Three.js and Rapier from `esm.sh`, so its first load needs internet access. Opening `immersive/index.html` directly as a `file://` URL prevents the browser from loading those modules.

## Controls

- Click **Enter the bedroom** to begin.
- **W A S D** walks; mouse looks; **Shift** moves faster; **Ctrl** crouches.
- Aim at an item and press **E** to pick it up. Carry it to the case, aim inside, scroll to lower the item, then press **E** to release it.
- **Planned sim** runs the guided bedroom route; **Reset room** restores the scene.

The detailed scene uses local GLB/glTF furniture and item models plus Rapier rigid-body contacts. Grasped objects follow the simulated hands; released items use gravity and contact collisions. The room-scale gallery remains a deterministic authored sequence with simplified geometry and a lightweight contact solver. Neither scene simulates deformable fabric or full human biomechanics. Asset provenance and licenses are in [immersive/assets/ASSET_SOURCES.md](immersive/assets/ASSET_SOURCES.md).

## Collect packing runs

The interactive room records completed, stopped, failed, and interrupted runs in browser storage. Use **Export runs** in the scene toolbar to download `packing-runs.json`. Each run contains the event timeline, actor and item poses sampled at 5 Hz, item source/mass/final position, model load status, and completion checks. The record follows the event-plus-validation pattern in [the assembly event log](../assembly-events.json) and [full-mechanics validation](evidence/full-mechanics-validation.json). Collection stays local to the browser until you export it.
