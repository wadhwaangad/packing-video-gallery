"""Prepare the scanned BlenderKit assets (garment piles and the suitcase) for the renders.

Garments (scene.GARMENT_SCANS): keep the named mesh objects, delete the flat ground
patches the scans were captured on, rotate so the short side lies on the item's x
axis (the two-hand squeeze axis), scale uniformly and centre on the collision box.

Suitcase (scene.SUITCASE_SCAN): push the telescopic handle down to its retracted
position, map the scan into the suitcase-base frame (x across, y along the length,
z out of the back panel) at the chosen scale, and cut it at the zip seam into a base
and a lid (lid frame: origin on the hinge line, z from the rim). The casters go whole
to the half they are mounted on. Inner lining and zip-tape rims are added at render time.

Textures are capped at 4K. Output: assets/scans/<name>.blend and scans.json.
Usage: python packing/prepare_assets.py [--measure]
"""
import json, math, sys
from pathlib import Path
import numpy as np
import bpy, bmesh
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene as S

SRC = HERE / "assets" / "blenderkit"
OUT = HERE / "assets" / "scans"
MAX_TEX = 4096


def components(bm):
    seen, out = set(), []
    for v in bm.verts:
        if v.index in seen:
            continue
        comp, st = [], [v]; seen.add(v.index)
        while st:
            u = st.pop(); comp.append(u)
            for e in u.link_edges:
                w = e.other_vert(u)
                if w.index not in seen:
                    seen.add(w.index); st.append(w)
        out.append(comp)
    return out


def drop_ground_patches(ob):
    """Delete connected components that are flat (< 2 mm tall) at the bottom of the scan."""
    bm = bmesh.new(); bm.from_mesh(ob.data); bm.verts.ensure_lookup_table()
    zmin = min(v.co.z for v in bm.verts)
    kill = []
    for comp in components(bm):
        zs = [c.co.z for c in comp]
        if max(zs) - min(zs) < 0.002 and min(zs) - zmin < 0.004:
            kill += comp
    if kill:
        bmesh.ops.delete(bm, geom=kill, context="VERTS")
    bm.to_mesh(ob.data); bm.free()
    return len(kill)


def load(blend, names=()):
    with bpy.data.libraries.load(str(blend)) as (src, dst):
        dst.objects = [n for n in src.objects]
    sc = bpy.context.scene
    keep = []
    for o in dst.objects:
        if o is None:
            continue
        sc.collection.objects.link(o)
        if o.type == "MESH" and (not names or o.name in names):
            keep.append(o)
    bpy.context.view_layer.update()
    for o in keep:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = mw
    for o in list(sc.collection.objects):
        if o not in keep:
            bpy.data.objects.remove(o)
    for o in keep:
        o.data = o.data.copy()
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return keep


def extents(objs):
    V = []
    for o in objs:
        V.append(np.array([v.co[:] for v in o.data.vertices]))
    V = np.concatenate(V)
    return V.min(0), V.max(0)


def cap_textures():
    for im in bpy.data.images:
        if im.size[0] > MAX_TEX or im.size[1] > MAX_TEX:
            f = MAX_TEX / max(im.size[:])
            im.scale(int(im.size[0] * f), int(im.size[1] * f))
            im.pack()


def garment(item, g, measure):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    objs = load(SRC / f"{g['asset']}.blend", g.get("objects", ()))
    dropped = sum(drop_ground_patches(o) for o in objs)
    lo, hi = extents(objs)
    c = (lo + hi) / 2
    M = (Matrix.Scale(g["scale"], 4) @ Matrix.Rotation(math.radians(g.get("yaw_deg", 0)), 4, "Z")
         @ Matrix.Translation((-c[0], -c[1], -c[2])))
    for o in objs:
        o.data.transform(M); o.data.update()
    lo2, hi2 = extents(objs)
    info = dict(kind="garment", asset=g["asset"], name=g["name"], author=g["author"], raw_extent_m=(hi - lo).round(4).tolist(),
                scale=g["scale"], yaw_deg=g.get("yaw_deg", 0), extent_m=(hi2 - lo2).round(4).tolist(),
                collision_box_m=list(S.scan_size(item)), ground_verts_removed=dropped)
    print(item, info["raw_extent_m"], "->", info["extent_m"], "box", info["collision_box_m"], flush=True)
    if not measure:
        cap_textures()
        for o in objs:
            o.name = f"scan_{item}"
        bpy.data.libraries.write(str(OUT / f"{item}.blend"), set(objs), fake_user=True, compress=True)
    return info


def suitcase(measure):
    c = S.SUITCASE_SCAN
    s = c["scale"]
    W, L = S.SUITCASE["outer"]; D = S.SUITCASE["base_depth"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ob = load(SRC / f"{c['asset']}.blend")[0]
    bm = bmesh.new(); bm.from_mesh(ob.data)
    # Retract the telescopic handle: everything above the shell top at the back moves
    # down; the tube ends that would then sit inside the shell are removed.
    top = c["z_top"] + 0.005
    moved = [v for v in bm.verts if v.co.z > top and v.co.y > 0.085]
    for v in moved:
        v.co.z -= c["trolley_drop"]
        v.co.y -= c["trolley_shift"]
    hidden = [v for v in moved if v.co.z < top]
    bmesh.ops.delete(bm, geom=hidden, context="VERTS")
    # Scan (x, y, z up) -> base frame: x' = s(x - xc), y' = s(z - zc), z' = s(back_y - y).
    zc = (c["z_top"] + c["z_bottom"]) / 2
    T = Matrix(((s, 0, 0, -s * c["x_centre"]), (0, 0, s, -s * zc), (0, -s, 0, s * c["back_y"]), (0, 0, 0, 1)))
    bm.transform(T)
    # Cut the shell exactly at the seam plane z' = D.
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(0, 0, D), plane_no=(0, 0, 1))
    wheel_top = -L / 2 + 0.002                       # caster zone: below the shell bottom
    wheel_seam = s * (c["back_y"] - c["wheel_seam_y"])
    bm.faces.index_update(); bm.faces.ensure_lookup_table()
    lid_faces = []
    for f in bm.faces:
        ctr = f.calc_center_median()
        seam = wheel_seam if ctr.y < wheel_top else D
        if ctr.z > seam:
            lid_faces.append(f.index)
    me_base = bpy.data.meshes.new("case_base"); bm.to_mesh(me_base)
    me_lid = bpy.data.meshes.new("case_lid"); bm.to_mesh(me_lid)
    bm.free()
    for me, keep_lid in ((me_base, False), (me_lid, True)):
        b2 = bmesh.new(); b2.from_mesh(me); b2.faces.index_update(); b2.faces.ensure_lookup_table()
        lid_set = set(lid_faces)
        kill = [f for f in b2.faces if (f.index in lid_set) != keep_lid]
        bmesh.ops.delete(b2, geom=kill, context="FACES")
        if keep_lid:
            # Stretched scan faces left by the handle retraction reach behind the seam: drop them.
            stray = [f for f in b2.faces if min(v.co.z for v in f.verts) < D - 0.003 and f.calc_center_median().y > wheel_top]
            bmesh.ops.delete(b2, geom=stray, context="FACES")
            # lid frame: origin on the hinge axis (-W/2, 0, hinge_h) of the closed case
            b2.transform(Matrix.Translation((W / 2, 0, -S.SUITCASE["hinge_h"])))
        b2.to_mesh(me); b2.free()
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    base = bpy.data.objects.new("scan_case_base", me_base)
    lid = bpy.data.objects.new("scan_case_lid", me_lid)
    for o in (base, lid):
        bpy.context.scene.collection.objects.link(o)
        for p in o.data.polygons:
            p.use_smooth = True
    lo, hi = extents([base]); lo2, hi2 = extents([lid])
    info = dict(kind="suitcase", asset=c["asset"], name=c["name"], author=c["author"], scale=s,
                base_extent=[lo.round(4).tolist(), hi.round(4).tolist()], lid_extent=[lo2.round(4).tolist(), hi2.round(4).tolist()],
                physics=dict(outer=list(S.SUITCASE["outer"]), base_depth=D, lid_depth=S.SUITCASE["lid_depth"]))
    print("suitcase base", lo.round(3), hi.round(3), "lid", lo2.round(3), hi2.round(3), flush=True)
    if not measure:
        cap_textures()
        bpy.data.libraries.write(str(OUT / "suitcase.blend"), {base, lid}, fake_user=True, compress=True)
    return info


def main():
    measure = "--measure" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "scans.json"
    info = json.loads(path.read_text()) if path.exists() else {}
    for item, g in S.GARMENT_SCANS.items():
        if not only or item in only:
            info[item] = garment(item, g, measure)
    if not only or "suitcase" in only:
        info["suitcase"] = suitcase(measure)
    if not measure:
        path.write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
