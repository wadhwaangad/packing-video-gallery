"""The videos' scanned assets for the 3D page, in a form players cannot simply download.

The garment piles and the suitcase are BlenderKit scans (royalty-free licence). The
licence FAQ allows them in games and apps when "the assets shouldn't be directly
extractable in the distribution", so the page never gets model files: this script builds
the assets exactly as render_blender.py does (the same scans, cut and lined the same way),
exports each one as a lightweight GLB, and seals them all in one AES-GCM encrypted bundle
that pack.js opens in memory. The repository and the site hold only the bundle.

  .venv/Scripts/python packing/export_scans.py export   # Blender: GLBs -> packing/assets/web_scans/ (not committed)
  python packing/export_scans.py bundle                 # encrypt them -> play/pack/scans.bin

Without the bundle the page falls back to its procedural stand-ins.
"""
import json, os, struct, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
GLB_DIR = HERE / "assets" / "web_scans"
BUNDLE = HERE.parent / "play" / "pack" / "scans.bin"
# The page holds the same key: it is there to keep files from being lifted, not to keep a secret.
KEY = bytes.fromhex("5f3a9c1e7b2d48a6c0f19e3b7d2a5c8e1f4b6d9a2c7e0b3f5a8d1c4e7b0a3f6d")
PILES = ["trousers", "tshirts", "sweater_pile", "striped_shirts", "sweatshirts", "towel", "towel_stack", "gym_clothes"]
MAX_FACES = 40000
MAX_TEX = 1024


def export():
    import bpy
    import render_blender as RB
    GLB_DIR.mkdir(parents=True, exist_ok=True)

    def slim(objs):
        for o in objs:
            if o.type != "MESH":
                continue
            bpy.context.view_layer.objects.active = o
            for m in list(o.modifiers):                      # solidify, liners' seams: as rendered
                bpy.ops.object.modifier_apply(modifier=m.name)
            n = len(o.data.polygons)
            if n > MAX_FACES:
                d = o.modifiers.new("slim", "DECIMATE"); d.ratio = MAX_FACES / n
                bpy.ops.object.modifier_apply(modifier="slim")
        for im in bpy.data.images:
            if im.size[0] > MAX_TEX or im.size[1] > MAX_TEX:
                f = MAX_TEX / max(im.size[:])
                im.scale(int(im.size[0] * f), int(im.size[1] * f))

    def save(name, objs):
        for o in bpy.context.view_layer.objects:
            o.select_set(o in objs)
        bpy.ops.export_scene.gltf(filepath=str(GLB_DIR / f"{name}.glb"), export_format="GLB", use_selection=True,
                                  export_apply=True, export_image_format="JPEG", export_jpeg_quality=85,
                                  export_yup=False, export_animations=False)
        print(name, sum(len(o.data.polygons) for o in objs if o.type == "MESH"), "faces,",
              f"{(GLB_DIR / f'{name}.glb').stat().st_size / 1e6:.1f} MB", flush=True)

    for n in PILES:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        objs = RB.append_scan(n)
        slim(objs)
        save(n, objs)
    # the case: shells cut at the zip seam, solidified, with their fabric liners (build_suitcase),
    # and the scanned telescopic bar and strap buckle (build_hardware's scan parts)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    base, lid = RB.build_suitcase()
    parts = RB.build_hardware(base)
    keep = {base, lid, parts.get("trolley"), parts.get("buckle_male")} | {o for o in bpy.data.objects if o.name.startswith("liner_")}
    for o in list(bpy.data.objects):
        if o not in keep:
            bpy.data.objects.remove(o, do_unlink=True)
    objs = [o for o in keep if o]
    slim(objs)
    save("suitcase", objs)


def bundle():
    """[magic 'PKS1'][12-byte nonce][AES-GCM( header length (u32) + JSON header + GLBs )]"""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    names = PILES + ["suitcase"]
    blobs = [(GLB_DIR / f"{n}.glb").read_bytes() for n in names]
    head, off = [], 0
    for n, b in zip(names, blobs):
        head.append(dict(name=n, offset=off, length=len(b))); off += len(b)
    hj = json.dumps(head).encode()
    plain = struct.pack("<I", len(hj)) + hj + b"".join(blobs)
    nonce = os.urandom(12)
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    BUNDLE.write_bytes(b"PKS1" + nonce + AESGCM(KEY).encrypt(nonce, plain, None))
    print(f"scans.bin: {len(names)} assets, {BUNDLE.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    {"export": export, "bundle": bundle}[sys.argv[1]]()
