# Packing Field Notes

This branch contains a suitcase-packing simulation and video research site. The home page presents overview and operator packing videos, followed by the embedded interactive simulation. The packing data page remains a separate source-linked collection of published packing demonstrations.

## Pages

- [Packing home](index.html)
- [Interactive suitcase simulation](packing/index.html)
- [Packing video library](packing-data/index.html)

## Simulation and generated videos

The browser simulation replays a deterministic 90-second run. Seven items travel from room surfaces to an open suitcase through named reach, grip, lift, carry, align, lower, release, and settle phases. The room overview, operator follow camera, and first-person egocentric camera share the same timeline. The egocentric view adds enlarged, camera-relative 3D forearms and hands as a first-person view model so they remain visible while reaching across the scene. Carried motion is authored; released items use the reduced contact solver documented on the simulation page.

The MP4 previews in `packing/media/` are rendered from those simulation cameras at 2560×1440, not captured human footage. The home page plays the overview, operator, and egocentric recordings before the embedded interactive simulation. To regenerate all three, run `node render-packing-video.cjs 8`; to update only the egocentric clip, run `node render-packing-video.cjs 8 --egocentric-only`. Both commands require Chrome, Playwright, and FFmpeg locally.

## GitHub Pages

This static site is published directly from the docs branch at the repository root. In GitHub, open **Settings → Pages** and set **Build and deployment** to **Deploy from a branch**, branch **docs**, folder **/(root)**. There is no separate build command or GitHub Actions workflow required. Commit the site files and media to the root of docs to publish updates.

Clothing model sources and license notes are in [packing/assets/garments/ASSET_SOURCES.md](packing/assets/garments/ASSET_SOURCES.md). Run evidence and limitations are in [packing/validation.json](packing/validation.json).
