"""Close-up preview of the procedural items (development aid).

Usage: python preview_items.py OUT.jpg [--samples 64] [--items a,b,c] [--view top|side|ego]
"""
import argparse, math, sys
from pathlib import Path
import bpy
from mathutils import Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene as S
import render_blender as RB
import item_assets as IA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--items", default="jeans,tshirt_white,tshirt_navy,sweater,towel,toiletry_bag,bottle")
    ap.add_argument("--view", default="ego"); ap.add_argument("--res", default="1600x900")
    a = ap.parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    RB.setup_world(a.samples)
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = map(int, a.res.split("x"))
    # indoor-like light: no direct sun, dim sky, one soft window-size area light
    for ob in list(bpy.data.objects):
        if ob.type == "LIGHT":
            bpy.data.objects.remove(ob)
    sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.25
    bpy.ops.object.light_add(type="AREA", location=(0.3, -0.6, 1.4)); L = bpy.context.object
    L.data.size = 1.0; L.data.energy = 30; L.rotation_euler = (math.radians(25), 0, 0)
    floor = RB.box_mesh("table", (3, 3, 0.04), material=RB.textured("oak", "oak_wood_planks", scale=1.2))
    floor.location = (0, 0, -0.02)
    names = a.items.split(",")
    x = 0.0
    xs = []
    for nm in names:
        it = S.item(nm)
        w = it.size[0] if nm != "bottle" else 0.08
        xs.append(x + w / 2); x += w + 0.05
    off = x / 2
    for nm, cx in zip(names, xs):
        it = S.item(nm)
        h = IA.build(it)
        h.location = (cx - off, 0, it.size[2] / 2)
        h.rotation_euler.z = 0.0 if nm != "toiletry_bag" else 0.3
    bpy.ops.object.camera_add(); cam = bpy.context.object; sc.camera = cam
    cam.data.lens = 35
    span = max(x, 0.55)
    if a.view == "top":
        RB.look_at(cam, (0, -0.05, span * 0.95), (0, 0, 0))
    elif a.view == "side":
        RB.look_at(cam, (0, -span * 0.75, 0.18), (0, 0, 0.03))
    else:
        RB.look_at(cam, (0, -span * 0.62, span * 0.52), (0, 0, 0.02))
    sc.render.filepath = str(Path(a.out).resolve())
    sc.render.image_settings.file_format = "JPEG"
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
