"""Download the assets used by the packing simulation.

- Poly Haven (CC0): furniture, props, textures and the sky HDRI.
- BlenderKit (royalty-free licence, free assets): the scanned folded-garment piles and
  the scanned suitcase listed in scene.GARMENT_SCANS / scene.SUITCASE_SCAN. Free
  BlenderKit assets download without an account key. The licence allows use in
  renders but not redistribution of the files, so they are fetched, never committed.

Every file is recorded in assets/asset-manifest.json with its source URL, licence and
sha256, so renders can be traced back to exact inputs. Run prepare_assets.py next.
Usage: python packing/fetch_assets.py
"""
import hashlib, json, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene as S

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
    "denim_fabric": "operator trousers",
    "cotton_jersey": "operator shirt, bed sheet, suitcase lining",
    "wool_boucle": "rug and headboard",
}
# Fabrics for the procedural toiletry bag, at 2k because the head camera sees it up close.
ITEM_TEXTURES = {
    "brown_leather": "toiletry bag pull and trim",
    "scuba_suede": "toiletry bag shell",
}
BK_API = "https://www.blenderkit.com/api/v1"
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
    jobs = [(n, r, "1k", n) for n, r in TEXTURES.items()] + [(n, r, "2k", f"{n}_2k") for n, r in ITEM_TEXTURES.items()]
    for name, role, res, dirname in jobs:
        files = get_json(f"{API}/files/{name}")
        folder = ASSETS / "textures" / dirname
        entry = dict(id=name, kind="texture", role=role, resolution=res, folder=dirname,
                     page=f"https://polyhaven.com/a/{name}", files={})
        for channel, key in [("diff", "Diffuse"), ("nor_gl", "nor_gl"), ("rough", "Rough"), ("arm", "arm")]:
            block = files.get(key)
            if block and res in block:
                fmt = "jpg" if "jpg" in block[res] else next(iter(block[res]))
                info = block[res][fmt]
                entry["files"][f"{channel}.{fmt}"] = fetch(info["url"], folder / f"{channel}.{fmt}")
        manifest["assets"].append(entry)
        print("texture", name, flush=True)
    for name, role in HDRIS.items():
        info = get_json(f"{API}/files/{name}")["hdri"]["2k"]["hdr"]
        entry = dict(id=name, kind="hdri", role=role, page=f"https://polyhaven.com/a/{name}",
                     files={f"{name}_2k.hdr": fetch(info["url"], ASSETS / "hdri" / f"{name}_2k.hdr")})
        manifest["assets"].append(entry)
        print("hdri", name, flush=True)
    for key, g in [(k, v) for k, v in S.GARMENT_SCANS.items()] + [("suitcase", S.SUITCASE_SCAN)]:
        hit = get_json(f"{BK_API}/search/?query=asset_base_id:{g['asset']}+asset_type:model")["results"][0]
        if not hit.get("isFree"):
            raise RuntimeError(f"{g['name']} is not a free BlenderKit asset")
        blend = next(f for f in hit["files"] if f["fileType"] == "blend")
        signed = get_json(blend["downloadUrl"] + "?scene_uuid=00000000-0000-0000-0000-000000000000")["filePath"]
        dest = ASSETS / "blenderkit" / f"{g['asset']}.blend"
        entry = dict(id=g["asset"], kind="blenderkit_model", role=f"scan: {key}", name=hit["name"],
                     author=hit["author"]["fullName"], license=hit["license"],
                     license_url="https://www.blenderkit.com/docs/licenses/",
                     page=f"https://www.blenderkit.com/get-blenderkit/{hit['id']}/",
                     files={dest.name: fetch(signed, dest)})
        manifest["assets"].append(entry)
        print("blenderkit", hit["name"], flush=True)
    manifest["license"] = "Poly Haven assets: CC0 1.0. BlenderKit assets: royalty-free licence (not redistributed)."
    (ASSETS / "asset-manifest.json").write_text(json.dumps(manifest, indent=1))
    print("assets:", len(manifest["assets"]))


if __name__ == "__main__":
    sys.exit(main())
