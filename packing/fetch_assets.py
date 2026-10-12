"""Download the CC0 Poly Haven assets used by the packing simulation.

Every file is recorded in assets/asset-manifest.json with its source URL,
license and sha256, so renders can be traced back to exact inputs.
Usage: python packing/fetch_assets.py
"""
import hashlib, json, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
API = "https://api.polyhaven.com"
UA = {"User-Agent": "packing-video-gallery/1.0 (research simulation)"}

# Furniture and props: glTF at 1k texture resolution.
MODELS = {
    # bedroom
    "ClassicNightstand_01": "bedroom nightstand",
    "vintage_wooden_drawer_01": "bedroom dresser (clothes source)",
    "potted_plant_02": "bedroom plant",
    # living room
    "sofa_03": "living room sofa",
    "modern_coffee_table_01": "living room coffee table (book, camera source)",
    "wooden_bookshelf_worn": "living room bookshelf",
    "Television_01": "living room television",
    "throw_pillows_01": "sofa pillows",
    "potted_plant_04": "living room plant",
    "side_table_01": "hallway side table (decor)",
    # items that are packed
    "Camera_01": "packed item: camera",
    "binder_notebook": "packed item: notebook",
    "digital_wrist_watch": "packed item: watch (rendered only, too small to grasp)",
    "alarm_clock_01": "nightstand decoration",
}
TEXTURES = {
    "oak_wood_planks": "apartment floor",
    "painted_plaster_wall": "apartment walls",
    "long_white_tiles": "bathroom floor and wall",
    "denim_fabric": "folded jeans",
    "cotton_jersey": "folded t-shirts",
    "knitted_fleece": "folded sweater",
    "rough_linen": "bed linen",
    "waffle_pique_cotton": "towel",
    "fabric_pattern_07": "toiletry bag",
    "wool_boucle": "rug",
}
HDRIS = {"kloofendal_48d_partly_cloudy_puresky": "daylight outside the windows"}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
            dest.write_bytes(r.read())
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def main():
    manifest = {"license": "CC0 1.0 (Poly Haven)", "license_url": "https://polyhaven.com/license", "assets": []}
    for name, role in MODELS.items():
        files = get_json(f"{API}/files/{name}")["gltf"]["1k"]["gltf"]
        folder = ASSETS / "models" / name
        entry = dict(id=name, kind="model", role=role, page=f"https://polyhaven.com/a/{name}", files={})
        entry["files"][f"{name}.gltf"] = fetch(files["url"], folder / f"{name}.gltf")
        for rel, info in files["include"].items():
            entry["files"][rel] = fetch(info["url"], folder / rel)
        manifest["assets"].append(entry)
        print("model", name, flush=True)
    for name, role in TEXTURES.items():
        files = get_json(f"{API}/files/{name}")
        folder = ASSETS / "textures" / name
        entry = dict(id=name, kind="texture", role=role, page=f"https://polyhaven.com/a/{name}", files={})
        for channel, key in [("diff", "Diffuse"), ("nor_gl", "nor_gl"), ("rough", "Rough"), ("arm", "arm")]:
            block = files.get(key)
            if block and "1k" in block:
                fmt = "jpg" if "jpg" in block["1k"] else next(iter(block["1k"]))
                info = block["1k"][fmt]
                entry["files"][f"{channel}.{fmt}"] = fetch(info["url"], folder / f"{channel}.{fmt}")
        manifest["assets"].append(entry)
        print("texture", name, flush=True)
    for name, role in HDRIS.items():
        info = get_json(f"{API}/files/{name}")["hdri"]["2k"]["hdr"]
        entry = dict(id=name, kind="hdri", role=role, page=f"https://polyhaven.com/a/{name}",
                     files={f"{name}_2k.hdr": fetch(info["url"], ASSETS / "hdri" / f"{name}_2k.hdr")})
        manifest["assets"].append(entry)
        print("hdri", name, flush=True)
    (ASSETS / "asset-manifest.json").write_text(json.dumps(manifest, indent=1))
    print("assets:", len(manifest["assets"]))


if __name__ == "__main__":
    sys.exit(main())
