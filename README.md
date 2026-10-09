# Packing Field Notes — GitHub Pages

This `docs` branch contains the packing-only static website. It includes the synchronized room/operator packing simulation at [`/packing/`](packing/index.html), locally bundled clothing models, and the human packing video/data index at [`/packing-data/`](packing-data/index.html). The home page links to both.

## Enable GitHub Pages

In the repository on GitHub, open **Settings → Pages**. Under **Build and deployment**, choose **Deploy from a branch**, select branch **`docs`** and folder **`/(root)`**, then save. This branch has no Actions workflow; Pages serves its committed files directly. The `main` branch retains the source repository and no longer includes the Pages workflow.

The published URLs will be:

- `https://wadhwaangad.github.io/packing-video-gallery/`
- `https://wadhwaangad.github.io/packing-video-gallery/packing/`
- `https://wadhwaangad.github.io/packing-video-gallery/packing-data/`

After Pages is enabled, push future site updates to `docs` to publish them.
