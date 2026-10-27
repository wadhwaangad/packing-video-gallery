"""Bake the apartment's light for the 3D page (play/pack.html) with Cycles.

The page cannot path-trace, so the light that never changes is traced here once: what the
sky and the ceiling panels give the walls, floors and built-in furniture, directly and
after bouncing, and the sun's bounce light. The sun's direct light stays live in the page,
so the person, the items and the case still cast moving shadows; those moving things are
lit live as before. The scene is render_blender's (build_apartment, build_furniture,
setup_world): the same rooms, materials, lights and sky as the videos. The Poly Haven
furniture models are in the bake (they shade the walls and floor) but are lit live.

Every baked box gets six rectangles in one atlas, one per side; a point's texel follows
from its position in the box and the side it faces, so the page computes the same
lightmap UVs on its own geometry (pack.js lightmapUV) from lightmap.json.

Usage: .venv/Scripts/python packing/bake_lightmaps.py [--samples 1024] [--size 2048]
Writes play/pack/lightmap.png (sRGB-encoded light / scale) and play/pack/lightmap.json.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bpy
import scene as S
import render_blender as RB

OUT = HERE.parent / "play" / "pack"
PAD = 4                                   # texels around each rectangle (mipmaps, filtering)
# Not baked: see-through, mirrored or outside (the page draws these its own way).
SKIP = ("glass_", "mirror", "tap", "outside_ground")


def stable_names():
    """Names the page can rebuild: windows and doors by index, not by coordinate."""
    ob = bpy.data.objects
    for i, (wall, s0, *_r) in enumerate(S.WINDOWS):
        for k in range(5):
            ob[f"wframe_{wall}_{s0}_{k}"].name = f"win{i}_frame{k}"
        ob[f"sill_{wall}_{s0}"].name = f"win{i}_sill"
    for i, (wall, s0, s1) in enumerate([("bed_north", 2.75, 3.65), ("bath_north", 4.15, 4.95)]):
        for j, s in enumerate((s0, s1)):
            ob[f"casing_{wall}_{s}"].name = f"door{i}_side{j}"
        ob[f"casing_{wall}_top"].name = f"door{i}_top"
    ob["door_bedroom"].name = "door0_leaf"
    ob["door_bath"].name = "door1_leaf"
    for j, y in enumerate((1.35, 2.15)):
        ob[f"pillow_{y}"].name = f"pillow{j}"


def targets():
    out = []
    for o in bpy.data.objects:
        if o.type != "MESH" or o.parent is not None or o.name.startswith(SKIP):
            continue
        out.append(o)
    return sorted(out, key=lambda o: o.name)


def apply_modifiers(objs):
    dg = bpy.context.evaluated_depsgraph_get()
    for o in objs:
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
        o.modifiers.clear()
        o.data = me


def bounds(o):
    co = np.empty(len(o.data.vertices) * 3); o.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    return co.min(0), co.max(0)


# the six sides: +x -x +y -y +z -z; (u, v) are the two axes across the side
AXES = [(1, 2), (1, 2), (0, 2), (0, 2), (0, 1), (0, 1)]


def pack(boxes, size):
    """Shelf-pack six rectangles per box at the highest density (texels per metre) that fits."""
    def attempt(dens):
        rects, x, y, shelf = {}, 0, 0, 0
        todo = []
        for name, (lo, hi) in boxes.items():
            ext = hi - lo
            for k, (a, b) in enumerate(AXES):
                todo.append((name, k, max(4, math.ceil(ext[a] * dens)), max(4, math.ceil(ext[b] * dens))))
        todo.sort(key=lambda t: -t[3])
        for name, k, w, h in todo:
            W, H = w + 2 * PAD, h + 2 * PAD
            if x + W > size:
                x, y, shelf = 0, y + shelf, 0
            if y + H > size or W > size:
                return None
            rects[(name, k)] = (x + PAD, y + PAD, w, h)
            x += W; shelf = max(shelf, H)
        return rects
    lo_d, hi_d, best = 1.0, 400.0, None
    for _ in range(30):
        mid = (lo_d + hi_d) / 2
        r = attempt(mid)
        if r: best, lo_d = (mid, r), mid
        else: hi_d = mid
    return best


def write_uvs(o, lo, hi, rects, size):
    me = o.data
    uv = me.uv_layers.new(name="lightmap")
    me.uv_layers.active = uv; uv.active_render = True
    ext = np.maximum(hi - lo, 1e-6)
    for p in me.polygons:
        n = p.normal
        ax = int(np.argmax(np.abs(n)))
        k = 2 * ax + (0 if n[ax] >= 0 else 1)
        a, b = AXES[k]
        x, y, w, h = rects[(o.name, k)]
        for li in p.loop_indices:
            c = me.vertices[me.loops[li].vertex_index].co
            tu = min(max((c[a] - lo[a]) / ext[a], 0), 1); tv = min(max((c[b] - lo[b]) / ext[b], 0), 1)
            uv.data[li].uv = ((x + 0.5 + tu * (w - 1)) / size, (y + 0.5 + tv * (h - 1)) / size)


def bake(objs, img, passes, size):
    for o in bpy.context.view_layer.objects:
        o.select_set(o in objs)
    bpy.context.view_layer.objects.active = objs[0]
    t = time.time()
    bpy.ops.object.bake(type="DIFFUSE", pass_filter=passes, use_clear=True, margin=PAD, margin_type="EXTEND")
    px = np.empty(size * size * 4, np.float32); img.pixels.foreach_get(px)
    print(f"  baked {sorted(passes)} in {time.time() - t:.0f}s", flush=True)
    return px.reshape(size, size, 4)[..., :3].copy()


def smooth(L, rects, sigma):
    """Path-tracing noise: blur inside each rectangle (never across), then refill the padding."""
    from scipy.ndimage import gaussian_filter
    out = L.copy()
    for x, y, w, h in rects.values():
        blk = gaussian_filter(L[y:y + h, x:x + w], (sigma, sigma, 0), mode="nearest")
        out[y - PAD:y + h + PAD, x - PAD:x + w + PAD] = np.pad(blk, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="edge")
    return out


def srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--size", type=int, default=2048)
    a = ap.parse_args()
    size = a.size
    bpy.ops.wm.read_factory_settings(use_empty=True)
    RB.setup_world(a.samples)
    RB.build_apartment("ego")
    RB.build_furniture()
    stable_names()
    objs = targets()
    apply_modifiers(objs)
    boxes = {o.name: bounds(o) for o in objs}
    dens, rects = pack(boxes, size)
    print(f"{len(objs)} baked objects, {dens:.0f} texels per metre in a {size}^2 atlas", flush=True)
    for o in objs:
        write_uvs(o, *boxes[o.name], rects, size)
    img = bpy.data.images.new("lightmap", size, size, float_buffer=True, alpha=False)
    for m in {s.material for o in objs for s in o.material_slots if s.material}:
        n = m.node_tree.nodes.new("ShaderNodeTexImage"); n.image = img
        m.node_tree.nodes.active = n
    sc = bpy.context.scene
    sc.render.bake.margin = PAD
    sun = next(o for o in bpy.data.objects if o.type == "LIGHT" and o.data.type == "SUN")
    # all the bounce light (the sun's too) + the direct light of the sky and the panels, not the sun's
    indirect = bake(objs, img, {"INDIRECT"}, size)
    sun.hide_render = True
    direct = bake(objs, img, {"DIRECT"}, size)
    L = smooth(indirect, rects, 2.0) + smooth(direct, rects, 0.8)
    scale = float(np.percentile(L, 99.8)) * 1.15
    enc = np.clip(np.round(srgb(np.clip(L / scale, 0, 1)) * 255), 0, 255).astype(np.uint8)
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(enc[::-1]).save(OUT / "lightmap.png", optimize=True)   # Blender's rows run bottom-up
    r4 = lambda v: [round(float(x), 5) for x in v]
    meta = dict(size=size, scale=round(scale, 6), samples=a.samples, density=round(dens, 1),
                objects={o.name: dict(min=r4(boxes[o.name][0]), max=r4(boxes[o.name][1]),
                                      rects=[[round((x + 0.5) / size, 6), round((y + 0.5) / size, 6), round((w - 1) / size, 6), round((h - 1) / size, 6)]
                                             for k in range(6) for x, y, w, h in [rects[(o.name, k)]]])
                         for o in objs})
    (OUT / "lightmap.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"wrote lightmap.png ({(OUT / 'lightmap.png').stat().st_size / 1e6:.1f} MB), scale {scale:.3f}")


if __name__ == "__main__":
    main()
