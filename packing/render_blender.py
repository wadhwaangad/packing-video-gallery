"""Render a recorded packing run with Blender Cycles (bpy module).

Usage: python render_blender.py RUN_DIR VIEW [--stride N] [--start I] [--stop J] [--samples S]
VIEW is 'ego' (moving head camera) or 'overview' (fixed room cameras, cut by room).
Everything placed here comes from the recorded MuJoCo frames, the solved operator
poses and scene.py; nothing is animated by hand.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import bpy, bmesh
from mathutils import Vector, Matrix, Quaternion
import mujoco

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene as S
import operator_model as O
import item_assets as IA

ASSETS = HERE / "assets"
TEX = ASSETS / "textures"

# ------------------------------------------------------------------ materials
_mat_cache = {}


def principled(name, color=(0.8, 0.8, 0.8), rough=0.5, metal=0.0, **extra):
    if name in _mat_cache:
        return _mat_cache[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bs = m.node_tree.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (*color, 1)
    bs.inputs["Roughness"].default_value = rough
    bs.inputs["Metallic"].default_value = metal
    for k, v in extra.items():
        bs.inputs[k].default_value = v
    _mat_cache[name] = m
    return m


def textured(name, tex, scale=1.0, tint=(1, 1, 1), rough_mul=1.0, bump=1.0, projection="BOX", coords="Object"):
    """PBR material from a Poly Haven texture folder with box projection in metres."""
    key = f"{name}"
    if key in _mat_cache:
        return _mat_cache[key]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bs = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / scale, 1 / scale, 1 / scale)
    nt.links.new(tc.outputs[coords], mp.inputs["Vector"])
    folder = TEX / tex

    def img(fname, non_color):
        p = folder / fname
        if not p.exists():
            return None
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(str(p), check_existing=True)
        if non_color:
            n.image.colorspace_settings.name = "Non-Color"
        n.projection = projection
        n.projection_blend = 0.25
        nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
        return n

    d = img("diff.jpg", False)
    if d is not None:
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        ca = [x for x in mix.inputs if x.type == "RGBA"]
        nt.links.new(d.outputs["Color"], ca[0])
        ca[1].default_value = (*tint, 1)
        nt.links.new([x for x in mix.outputs if x.type == "RGBA"][0], bs.inputs["Base Color"])
    else:
        bs.inputs["Base Color"].default_value = (*tint, 1)
    r = img("rough.jpg", True)
    if r is not None:
        mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = rough_mul
        nt.links.new(r.outputs["Color"], mul.inputs[0]); nt.links.new(mul.outputs[0], bs.inputs["Roughness"])
    n = img("nor_gl.jpg", True)
    if n is not None:
        nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = bump
        nt.links.new(n.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], bs.inputs["Normal"])
    _mat_cache[key] = m
    return m


def skin_material():
    if "skin" in _mat_cache:
        return _mat_cache["skin"]
    m = principled("skin", (0.56, 0.36, 0.26), 0.45)
    bs = m.node_tree.nodes["Principled BSDF"]
    bs.inputs["Subsurface Weight"].default_value = 0.12
    bs.inputs["Subsurface Radius"].default_value = (0.012, 0.006, 0.003)
    nt = m.node_tree
    noise = nt.nodes.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 900; noise.inputs["Detail"].default_value = 3
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.12; bump.inputs["Distance"].default_value = 0.0002
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], bs.inputs["Normal"])
    return m


def suitcase_shell_material():
    """Polycarbonate shell with moulded ribs along the length."""
    if "shell" in _mat_cache:
        return _mat_cache["shell"]
    c = S.SUITCASE["color"]
    m = principled("shell", c, 0.32, 0.0, **{"Coat Weight": 0.35, "Coat Roughness": 0.15})
    nt = m.node_tree; bs = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    wave = nt.nodes.new("ShaderNodeTexWave"); wave.wave_type = "BANDS"; wave.bands_direction = "X"
    wave.inputs["Scale"].default_value = 1.6; wave.inputs["Distortion"].default_value = 0.0
    nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeMapRange"); ramp.inputs["From Min"].default_value = 0.85; ramp.inputs["From Max"].default_value = 1.0
    nt.links.new(wave.outputs["Fac"], ramp.inputs["Value"])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.35; bump.inputs["Distance"].default_value = 0.002
    nt.links.new(ramp.outputs["Result"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], bs.inputs["Normal"])
    _mat_cache["shell"] = m
    return m


# ------------------------------------------------------------------ geometry helpers
def link(obj, coll=None):
    (coll or bpy.context.scene.collection).objects.link(obj)
    return obj


def box_mesh(name, size, bevel=0.0, segments=3, material=None, subdiv=0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2]))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    ob = link(bpy.data.objects.new(name, me))
    if bevel > 0:
        md = ob.modifiers.new("bevel", "BEVEL"); md.width = bevel; md.segments = segments; md.limit_method = "NONE"
    if subdiv:
        md = ob.modifiers.new("subd", "SUBSURF"); md.levels = subdiv; md.render_levels = subdiv
    if material:
        me.materials.append(material)
    for p in me.polygons:
        p.use_smooth = True
    return ob


def place(ob, center, yaw=0.0):
    ob.location = Vector(center)
    ob.rotation_euler = (0, 0, yaw)
    return ob


def set_pose(ob, pos, quat):
    ob.location = Vector(pos)
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = Quaternion(quat)


def import_gltf(asset, objects=()):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(ASSETS / "models" / asset / f"{asset}.gltf"))
    new = [o for o in bpy.data.objects if o not in before]
    if objects:
        for o in list(new):
            if o.type == "MESH" and o.name.split(".")[0] not in objects:
                bpy.data.objects.remove(o, do_unlink=True)
                new.remove(o)
    root = bpy.data.objects.new(f"{asset}_root", None)
    link(root)
    for o in new:
        if o.parent is None:
            o.parent = root
    return root, new


# ------------------------------------------------------------------ apartment
def build_apartment(view):
    floor_wood = textured("oak floor", "oak_wood_planks", scale=2.0, tint=(0.92, 0.85, 0.78), bump=0.6)
    tiles = textured("white tiles", "long_white_tiles", scale=1.2, bump=0.8)
    plaster = principled("wall paint", (0.86, 0.84, 0.80), 0.85)
    _nt = plaster.node_tree; _bs = _nt.nodes["Principled BSDF"]
    _n = _nt.nodes.new("ShaderNodeTexNoise"); _n.inputs["Scale"].default_value = 180; _n.inputs["Detail"].default_value = 6
    _b = _nt.nodes.new("ShaderNodeBump"); _b.inputs["Strength"].default_value = 0.08; _b.inputs["Distance"].default_value = 0.0005
    _nt.links.new(_n.outputs["Fac"], _b.inputs["Height"]); _nt.links.new(_b.outputs["Normal"], _bs.inputs["Normal"])
    for name, r in S.ROOMS.items():
        x0, x1 = r["x"]; y0, y1 = r["y"]
        mat = tiles if r["floor"] == "long_white_tiles" else floor_wood
        f = box_mesh(f"floor_{name}", (x1 - x0, y1 - y0, 0.02), material=mat)
        place(f, ((x0 + x1) / 2, (y0 + y1) / 2, -0.01))
    trim = principled("trim", (0.9, 0.9, 0.88), 0.35)
    for w in S.WALLS:
        ob = box_mesh(w.name, w.size, material=plaster)
        place(ob, w.center)
        z0 = w.center[2] - w.size[2] / 2
        if z0 < 1e-6 and w.size[2] > 0.5:
            sx, sy = w.size[0], w.size[1]
            bb = box_mesh(w.name + "_base", (sx + (0.024 if sx < sy else 0), sy + (0.024 if sy < sx else 0), 0.085),
                          bevel=0.003, material=trim)
            place(bb, (w.center[0], w.center[1], 0.0425))
    # Ceiling: visible to the head camera, hidden from the room cameras (cut-away).
    ceil = box_mesh("ceiling", (7.2, 6.6, 0.05), material=principled("ceiling", (0.93, 0.93, 0.92), 0.8))
    place(ceil, (3.5, 3.2, S.WALL_H + 0.025))
    ceil.visible_camera = view == "ego"
    # Windows: frames + glass
    glass = principled("glass", (1, 1, 1), 0.0, **{"Transmission Weight": 1.0, "IOR": 1.0})
    frame = principled("window frame", (0.95, 0.95, 0.94), 0.3)
    for wall, s0, s1, z0, z1 in S.WINDOWS:
        horizontal = wall in ("wall_south", "wall_north")
        y = 0.0 if wall == "wall_south" else 6.4
        cx = (s0 + s1) / 2
        g = box_mesh(f"glass_{wall}_{s0}", (s1 - s0, 0.01, z1 - z0), material=glass)
        place(g, (cx, y, (z0 + z1) / 2))
        g.visible_shadow = False
        for k, (sz, c) in enumerate([((s1 - s0 + 0.1, 0.12, 0.05), (cx, y, z0 - 0.02)),
                                     ((s1 - s0 + 0.1, 0.12, 0.05), (cx, y, z1 + 0.02)),
                                     ((0.05, 0.12, z1 - z0), (s0, y, (z0 + z1) / 2)),
                                     ((0.05, 0.12, z1 - z0), (s1, y, (z0 + z1) / 2)),
                                     ((0.04, 0.10, z1 - z0), (cx, y, (z0 + z1) / 2))]):
            place(box_mesh(f"wframe_{wall}_{s0}_{k}", sz, bevel=0.004, material=frame), c)
        sill = box_mesh(f"sill_{wall}_{s0}", (s1 - s0 + 0.2, 0.2, 0.025), bevel=0.003, material=trim)
        place(sill, (cx, y + (0.08 if wall == "wall_south" else -0.08), z0 - 0.05))
    ground = box_mesh("outside_ground", (60, 60, 0.02), material=principled("lawn", (0.16, 0.22, 0.10), 0.9))
    place(ground, (3.5, 3.2, -3.0))
    # Door casings
    casing = principled("door casing", (0.92, 0.92, 0.9), 0.35)
    for wall, s0, s1, ax, at in [("bed_north", 2.75, 3.65, "x", 3.4), ("bath_north", 4.15, 4.95, "x", 2.2)]:
        for s in (s0, s1):
            place(box_mesh(f"casing_{wall}_{s}", (0.07, 0.14, 2.08), bevel=0.004, material=casing), (s, at, 1.04))
        place(box_mesh(f"casing_{wall}_top", (s1 - s0 + 0.14, 0.14, 0.07), bevel=0.004, material=casing), ((s0 + s1) / 2, at, 2.085))
    # Open door leaves
    leaf = principled("door leaf", (0.88, 0.87, 0.84), 0.4)
    d1 = box_mesh("door_bedroom", (0.84, 0.04, 2.02), bevel=0.004, material=leaf)
    place(d1, (2.75 - 0.43, 3.4 - 0.06, 1.01), yaw=0.0)
    d2 = box_mesh("door_bath", (0.76, 0.04, 2.02), bevel=0.004, material=leaf)
    place(d2, (4.15 - 0.39 + 0.0, 2.2 - 0.06, 1.01), yaw=0.0)


def build_furniture():
    linen = textured("sheet", "cotton_jersey", scale=0.35, tint=(1.0, 0.97, 0.92), bump=0.35)
    duvet_m = textured("duvet", "rough_linen", scale=0.6, tint=(0.72, 0.52, 0.40), bump=0.6)
    upholstery = textured("headboard fabric", "wool_boucle", scale=0.4, tint=(0.55, 0.52, 0.47), bump=0.8)
    walnut = principled("walnut", (0.20, 0.11, 0.06), 0.45)
    oak = principled("light oak", (0.55, 0.40, 0.25), 0.5)
    lacquer = principled("white lacquer", (0.88, 0.88, 0.86), 0.25)
    sage = principled("sage lacquer", (0.43, 0.50, 0.42), 0.35)
    stone = principled("stone counter", (0.85, 0.84, 0.81), 0.25)
    rug_m = textured("rug", "wool_boucle", scale=0.5, tint=(0.75, 0.70, 0.62), bump=1.0)
    for b in S.PROCEDURAL:
        mats = {"bed_frame": walnut, "mattress": linen, "vanity": lacquer, "counter": stone, "rug": rug_m}
        mat = mats.get(b.material, oak)
        if b.name == "headboard":
            mat = upholstery
        if b.name.startswith("kitchen_base"):
            mat = sage
        if b.name == "kitchen_top":
            mat = oak
        bev = 0.04 if b.name == "mattress" else 0.03 if b.name == "headboard" else 0.006
        ob = box_mesh(b.name, b.size, bevel=bev, segments=5 if b.name == "mattress" else 2, material=mat)
        place(ob, b.center, b.yaw)
    # Duvet folded at the head end, pillows
    # Throw blanket folded over the foot end of the bed
    duvet = box_mesh("throw", (0.20, 1.60, 0.035), bevel=0.015, segments=4, material=duvet_m, subdiv=1)
    place(duvet, (2.05, 1.75, 0.575))
    drape = box_mesh("throw_drape", (0.03, 1.60, 0.22), bevel=0.012, segments=3, material=duvet_m, subdiv=1)
    place(drape, (2.165, 1.75, 0.47))
    for y in (1.35, 2.15):
        p = box_mesh(f"pillow_{y}", (0.36, 0.62, 0.14), bevel=0.06, segments=5, material=linen, subdiv=1)
        place(p, (0.30, y, 0.66), yaw=0.0)
    # Bathroom: sink basin, tap and mirror
    basin = box_mesh("basin", (0.42, 0.32, 0.02), bevel=0.008, material=principled("ceramic", (0.95, 0.95, 0.95), 0.08))
    place(basin, (4.95, 0.27, 0.872))
    tap = box_mesh("tap", (0.03, 0.03, 0.16), bevel=0.012, material=principled("chrome", (0.9, 0.9, 0.9), 0.08, 1.0))
    place(tap, (4.95, 0.09, 0.95))
    mirror = box_mesh("mirror", (0.8, 0.02, 0.7), bevel=0.004, material=principled("mirror", (0.95, 0.95, 0.95), 0.02, 1.0))
    place(mirror, (4.70, 0.06, 1.45))
    # Kitchen backsplash and a kettle-free counter
    back = box_mesh("backsplash", (0.02, 2.04, 0.55), material=textured("splash tiles", "long_white_tiles", scale=0.8))
    place(back, (6.94, 1.10, 1.17))
    for a in S.ASSETS:
        root, objs = import_gltf(a.asset)
        root.location = Vector(a.pos)
        root.rotation_euler = (0, 0, a.yaw)


# ------------------------------------------------------------------ suitcase
def shell_object(name, W, L, D, t, open_top=True):
    """Open rounded box: outer W x L x D (z 0..D), wall thickness t, lining inside."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * W, v.co.y * L, v.co.z * D + D / 2))
    top = [f for f in bm.faces if f.normal.z > 0.9]
    bmesh.ops.delete(bm, geom=top, context="FACES")
    edges = [e for e in bm.edges if not (abs(e.verts[0].co.z - D) < 1e-6 and abs(e.verts[1].co.z - D) < 1e-6)]
    bmesh.ops.bevel(bm, geom=edges, offset=0.045, segments=8, affect="EDGES", profile=0.5)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    ob = link(bpy.data.objects.new(name, me))
    for p in me.polygons:
        p.use_smooth = True
    sol = ob.modifiers.new("shell", "SOLIDIFY"); sol.thickness = t; sol.offset = -1; sol.material_offset = 1
    sol.use_rim = True; sol.material_offset_rim = 2
    me.materials.append(suitcase_shell_material())
    me.materials.append(textured("lining", "cotton_jersey", scale=0.25, tint=(0.18, 0.19, 0.21), bump=0.4))
    me.materials.append(principled("zip tape", (0.03, 0.03, 0.035), 0.6))
    ob.modifiers.new("wn", "WEIGHTED_NORMAL")
    return ob


def build_suitcase():
    c = S.SUITCASE
    W, L = c["outer"]; D, Dl, t = c["base_depth"], c["lid_depth"], c["shell"]
    rubber = principled("handle rubber", (0.025, 0.025, 0.027), 0.55)
    metal = principled("brushed alu", (0.7, 0.72, 0.74), 0.3, 1.0)
    base = shell_object("suitcase_base", W, L, D, t)
    lid_shell = shell_object("suitcase_lid_shell", W, L, Dl, t)
    lid = bpy.data.objects.new("suitcase_lid", None); link(lid)
    lid_shell.parent = lid
    lid_shell.location = (W / 2, 0, Dl)
    lid_shell.rotation_euler = (np.pi, 0, 0)
    # zipper band on the rims
    for parent, depth, nm in [(base, D, "b"), (lid_shell, Dl, "l")]:
        bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            v.co = Vector((v.co.x * (W + 0.003), v.co.y * (L + 0.003), v.co.z * 0.012))
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if abs(f.normal.z) > 0.9], context="FACES")
        bmesh.ops.bevel(bm, geom=[e for e in bm.edges if abs(e.verts[0].co.z - e.verts[1].co.z) > 1e-6],
                        offset=0.045, segments=8, affect="EDGES")
        me = bpy.data.meshes.new(f"zip_{nm}"); bm.to_mesh(me); bm.free()
        z = link(bpy.data.objects.new(f"zip_{nm}", me)); me.materials.append(principled("zip", (0.02, 0.02, 0.02), 0.5))
        for pg in me.polygons:
            pg.use_smooth = True
        z.parent = parent; z.location = (0, 0, depth - 0.006)
        sol = z.modifiers.new("hollow", "SOLIDIFY"); sol.thickness = 0.003; sol.offset = 1
    # handles (base)
    def bar(nm, a, b, r, parent, mat):
        a, b = Vector(a), Vector(b)
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=r, depth=(b - a).length)
        ob = bpy.context.object; ob.name = nm
        ob.location = (a + b) / 2
        ob.rotation_euler = (b - a).to_track_quat("Z", "Y").to_euler()
        ob.data.materials.append(mat)
        md = ob.modifiers.new("bevel", "BEVEL"); md.width = r * 0.6; md.segments = 3
        for p in ob.data.polygons:
            p.use_smooth = True
        ob.parent = parent
        return ob
    zc = D * 0.55
    bar("top_handle", (-0.065, L / 2 + 0.045, zc), (0.065, L / 2 + 0.045, zc), 0.012, base, rubber)
    for sx in (-0.075, 0.075):
        b = box_mesh(f"top_post_{sx}", (0.026, 0.036, 0.026), bevel=0.008, material=rubber); b.parent = base
        b.location = (sx, L / 2 + 0.023, zc)
    bar("side_handle", (W / 2 + 0.045, -0.065, zc), (W / 2 + 0.045, 0.065, zc), 0.012, base, rubber)
    for sy in (-0.075, 0.075):
        b = box_mesh(f"side_post_{sy}", (0.036, 0.026, 0.026), bevel=0.008, material=rubber); b.parent = base
        b.location = (W / 2 + 0.023, sy, zc)
    # trolley tube caps on the top end
    for sx in (-0.09, 0.09):
        b = box_mesh(f"trolley_cap_{sx}", (0.03, 0.02, 0.02), bevel=0.006, material=metal); b.parent = base
        b.location = (sx, L / 2 + 0.006, 0.03)
    # spinner wheels
    # The lid shell is flipped (pi about x) inside the lid frame, so its wheel end is at +y
    # and its outer panel at z=0 in shell coordinates.
    for parent, zs, ys in [(base, 0.045, -1), (lid_shell, 0.045, 1)]:
        for sx in (-W / 2 + 0.045, W / 2 - 0.045):
            h = box_mesh("wheel_housing", (0.06, 0.03, 0.05), bevel=0.012, material=rubber); h.parent = parent
            h.location = (sx, ys * (L / 2 + 0.012), zs)
            for dx in (-0.012, 0.012):
                bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.026, depth=0.016)
                w = bpy.context.object; w.parent = parent
                w.location = (sx + dx, ys * (L / 2 + 0.03), zs); w.rotation_euler = (0, np.pi / 2, 0)
                w.data.materials.append(rubber)
                md = w.modifiers.new("bevel", "BEVEL"); md.width = 0.006; md.segments = 3
    # zipper pull: driven by the recorded zipper progress along the same rim path the hand traced
    pull = box_mesh("zip_pull", (0.012, 0.03, 0.006), bevel=0.003, material=metal)
    pull.parent = base
    base["zip_path"] = [W / 2, -L / 2 + 0.04, D, W / 2, 0.0, D, W / 2, L / 2 - 0.05, D]
    # brand-free badge
    badge = box_mesh("badge", (0.07, 0.025, 0.003), bevel=0.004, material=metal); badge.parent = lid_shell
    badge.location = (0, -0.22, -0.001)
    return base, lid


def place_zip_pull(base, u):
    pts = np.array(base["zip_path"]).reshape(-1, 3)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
    sv = u * cum[-1]
    i = min(np.searchsorted(cum, sv, side="right") - 1, len(seg) - 1)
    p = pts[i] + (pts[i + 1] - pts[i]) * ((sv - cum[i]) / seg[i])
    d = pts[i + 1] - pts[i]
    out = np.array([d[1], -d[0], 0.0]); out /= np.linalg.norm(out) + 1e-9
    ob = bpy.data.objects["zip_pull"]
    ob.location = Vector(p + out * 0.012)
    ob.rotation_euler = (0, 0, math.atan2(d[1], d[0]) + math.pi / 2)


# ------------------------------------------------------------------ items
def build_items():
    objs = {}
    for it in S.ITEMS:
        if it.name in IA.BUILDERS:
            objs[it.name] = IA.build(it)
            continue
        if it.kind == "asset":
            root, new = import_gltf(it.asset, it.mesh_objects)
            holder = bpy.data.objects.new(f"item_{it.name}", None); link(holder)
            root.parent = holder
            root.location = Vector(it.mesh_offset)
            objs[it.name] = holder
            continue
        sx, sy, sz = it.size
        if it.kind == "bottle":
            holder = bpy.data.objects.new(f"item_{it.name}", None); link(holder)
            coat = principled("bottle powder coat", it.tint, 0.55)
            steel = principled("bottle steel", (0.78, 0.78, 0.80), 0.22, 1.0)
            bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.036, depth=sz - 0.034, location=(0, 0, -0.017))
            body_ = bpy.context.object; body_.data.materials.append(coat)
            md = body_.modifiers.new("bevel", "BEVEL"); md.width = 0.006; md.segments = 4
            bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.0362, depth=0.012, location=(0, 0, sz / 2 - 0.040))
            band = bpy.context.object; band.data.materials.append(steel)
            bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.026, depth=0.034, location=(0, 0, sz / 2 - 0.017))
            cap = bpy.context.object; cap.data.materials.append(principled("bottle cap", (0.06, 0.06, 0.06), 0.5))
            md = cap.modifiers.new("bevel", "BEVEL"); md.width = 0.004; md.segments = 3
            for o in (body_, band, cap):
                o.parent = holder
                for pg in o.data.polygons:
                    pg.use_smooth = True
            objs[it.name] = holder
            continue
        if it.kind == "bag":
            mat = textured("bag fabric", "denim_fabric", scale=0.15, tint=(0.25, 0.42, 0.36), bump=0.8)
            ob = box_mesh(f"item_{it.name}", (sx, sy, sz), bevel=0.035, segments=6, material=mat, subdiv=1)
            zip_ = box_mesh("bag_zip", (sx - 0.03, 0.008, 0.004), bevel=0.002, material=principled("bag zip", (0.05, 0.05, 0.05), 0.4))
            zip_.parent = ob; zip_.location = (0, 0, sz / 2 + 0.001)
            pull = box_mesh("bag_pull", (0.025, 0.012, 0.004), bevel=0.002, material=principled("pull", (0.75, 0.75, 0.75), 0.3, 1.0))
            pull.parent = ob; pull.location = (sx / 2 - 0.035, 0.008, sz / 2 + 0.003)
        else:
            mat = textured(f"{it.name} fabric", it.material, scale=0.12 if it.material != "denim_fabric" else 0.18,
                           tint=it.tint, bump=1.0)
            ob = box_mesh(f"item_{it.name}", (sx, sy, sz), bevel=min(0.016, sz * 0.32), segments=5, material=mat, subdiv=1)
            # soft irregular folds
            tex = bpy.data.textures.new(f"{it.name}_folds", "CLOUDS"); tex.noise_scale = 0.06
            dm = ob.modifiers.new("folds", "DISPLACE"); dm.texture = tex; dm.strength = 0.003; dm.texture_coords = "LOCAL"
            sub = ob.modifiers["subd"]; sub.levels = 2; sub.render_levels = 3
            # move displace after the subdivision
            bpy.context.view_layer.objects.active = ob
            while ob.modifiers[-1].name != "folds":
                bpy.ops.object.modifier_move_down(modifier="folds")
        objs[it.name] = ob
    return objs


# ------------------------------------------------------------------ hands (skin bound to MyoHand bodies)
def build_hands(model_path):
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    skin = skin_material()
    nail = principled("nail", (0.70, 0.50, 0.42), 0.3)
    out = {}
    for side in "rl":
        root = m.body(f"radius_{side}").id
        def desc(b):
            while b > 0:
                if b == root:
                    return True
                b = m.body_parentid[b]
            return False
        geoms = [g for g in range(m.ngeom) if desc(m.geom_bodyid[g]) and "coll" in m.geom(g).name]
        parts, binds = [], []
        for g in geoms:
            typ = m.geom_type[g]; s = m.geom_size[g]; pos = d.geom_xpos[g]; R = d.geom_xmat[g].reshape(3, 3)
            if typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
                bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=s[0], depth=2 * s[1])
                obs = [bpy.context.object]
                for z in (-s[1], s[1]):
                    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=s[0], location=(0, 0, z))
                    obs.append(bpy.context.object)
            elif typ == mujoco.mjtGeom.mjGEOM_BOX:
                bpy.ops.mesh.primitive_cube_add(size=2); obs = [bpy.context.object]; obs[0].scale = s
            else:
                bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10); obs = [bpy.context.object]
                obs[0].scale = s if typ == mujoco.mjtGeom.mjGEOM_ELLIPSOID else (s[0],) * 3
            bpy.ops.object.select_all(action="DESELECT")
            for o in obs:
                o.select_set(True)
            bpy.context.view_layer.objects.active = obs[0]
            bpy.ops.object.join(); o = obs[0]
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            o.data.transform(Matrix.Translation(Vector(pos)) @ Matrix(R.tolist()).to_4x4())
            parts.append(o)
            binds.append(dict(body=int(m.geom_bodyid[g]), typ=int(typ), size=s.copy(), pos=pos.copy(), R=R.copy()))
        bpy.ops.object.select_all(action="DESELECT")
        for o in parts:
            o.select_set(True)
        bpy.context.view_layer.objects.active = parts[0]
        bpy.ops.object.join(); hand = parts[0]; hand.name = f"hand_skin_{side}"
        md = hand.modifiers.new("remesh", "REMESH"); md.mode = "VOXEL"; md.voxel_size = 0.0016
        bpy.ops.object.modifier_apply(modifier=md.name)
        md = hand.modifiers.new("smooth", "SMOOTH"); md.factor = 1.2; md.iterations = 6
        bpy.ops.object.modifier_apply(modifier=md.name)
        hand.data.materials.append(skin)
        for p in hand.data.polygons:
            p.use_smooth = True
        V = np.array([v.co[:] for v in hand.data.vertices])
        dist = []
        for g in binds:
            local = (V - g["pos"]) @ g["R"]; s = g["size"]
            if g["typ"] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                local[:, 2] -= np.clip(local[:, 2], -s[1], s[1]); dd = np.linalg.norm(local, axis=1) - s[0]
            else:
                dd = (np.linalg.norm(local / np.maximum(s, 1e-3), axis=1) - 1) * min(s)
            dist.append(dd)
        D = np.array(dist).T
        k = min(3, D.shape[1])
        near = np.argsort(D, axis=1)[:, :k]
        vb = np.array([g["body"] for g in binds])[near]
        w = np.exp(-np.take_along_axis(D, near, axis=1) / 0.002); w /= w.sum(axis=1)[:, None]
        R0 = np.array([d.xmat[b].reshape(3, 3) for b in range(m.nbody)]); P0 = d.xpos.copy()
        local = np.einsum("nkji,nkj->nki", R0[vb], V[:, None, :] - P0[vb])
        out[side] = dict(obj=hand, vb=vb, w=w, local=local)
    return out


def update_hand(h, P, Rb):
    verts = (h["w"][:, :, None] * (np.einsum("nkij,nkj->nki", Rb[h["vb"]], h["local"]) + P[h["vb"]])).sum(axis=1)
    h["obj"].data.vertices.foreach_set("co", verts.ravel())
    h["obj"].data.update()


# ------------------------------------------------------------------ operator body
class Body:
    """Smooth mannequin made of tapered segments, posed every frame from the solved joints."""

    def __init__(self):
        self.shirt = textured("shirt", "cotton_jersey", scale=0.15, tint=(0.36, 0.45, 0.40), bump=0.6)
        self.pants = textured("trousers", "denim_fabric", scale=0.2, tint=(0.30, 0.30, 0.33), bump=0.6)
        self.skin = skin_material()
        self.segs = {}
        # (name, radius at start, radius at end, material)
        for nm, r0, r1, mat in [("upper_arm_r", 0.050, 0.040, self.shirt), ("upper_arm_l", 0.050, 0.040, self.shirt),
                                ("thigh_r", 0.078, 0.056, self.pants), ("thigh_l", 0.078, 0.056, self.pants),
                                ("shin_r", 0.054, 0.040, self.pants), ("shin_l", 0.054, 0.040, self.pants),
                                ("neck", 0.052, 0.048, self.skin), ("waist", 0.150, 0.150, self.shirt)]:
            bpy.ops.mesh.primitive_cone_add(vertices=40, radius1=r0, radius2=r1, depth=1.0, location=(0, 0, 0.5))
            ob = bpy.context.object; ob.name = nm
            bpy.context.scene.cursor.location = (0, 0, 0)
            bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
            ob.data.materials.append(mat)
            for pg in ob.data.polygons:
                pg.use_smooth = True
            self.segs[nm] = ob
        self.balls = {}
        for nm, r, mat in [("shoulder_r", 0.056, self.shirt), ("shoulder_l", 0.056, self.shirt),
                           ("elbow_r", 0.040, self.skin), ("elbow_l", 0.040, self.skin),
                           ("hip_r", 0.080, self.pants), ("hip_l", 0.080, self.pants),
                           ("knee_r", 0.057, self.pants), ("knee_l", 0.057, self.pants)]:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=r)
            ob = bpy.context.object; ob.name = nm; ob.data.materials.append(mat)
            for pg in ob.data.polygons:
                pg.use_smooth = True
            self.balls[nm] = ob
        self.chest = box_mesh("chest", (0.36, 0.22, 0.34), bevel=0.10, segments=8, material=self.shirt, subdiv=2)
        self.hips = box_mesh("hips", (0.33, 0.21, 0.20), bevel=0.08, segments=6, material=self.pants, subdiv=2)
        self.head = box_mesh("head", (0.165, 0.14, 0.205), bevel=0.068, segments=8, material=self.skin, subdiv=2)
        self.hair = box_mesh("hair", (0.175, 0.15, 0.10), bevel=0.05, segments=6,
                             material=principled("hair", (0.05, 0.035, 0.025), 0.6), subdiv=2)
        self.hair.parent = self.head; self.hair.location = (-0.012, 0, 0.058)
        shoe = principled("shoe", (0.85, 0.85, 0.83), 0.5)
        self.shoes = [box_mesh(f"shoe_{i}", (0.27, 0.105, 0.075), bevel=0.035, segments=5, material=shoe, subdiv=1) for i in (0, 1)]

    @staticmethod
    def _segment(ob, a, b):
        a, b = Vector(a), Vector(b)
        d = b - a
        ob.location = a
        ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = d.to_track_quat("Z", "Y")
        ob.scale = (1, 1, max(d.length, 1e-4))

    def update(self, pose, hide_head=False):
        p = {k: (np.array(v) if v is not None else None) for k, v in pose["pts"].items()}
        R = np.array(pose["R"])
        q = Matrix(R.tolist()).to_quaternion()
        pelvis = p["pelvis"]; up = R[:, 2]
        sr, sl = p["shoulders"]; er, el = p["elbows"]
        set_pose(self.chest, pelvis + up * 0.31, q)
        set_pose(self.hips, pelvis + up * 0.02, Quaternion((0, 0, 1), pose["q"][3]))
        self._segment(self.segs["waist"], pelvis + up * 0.06, pelvis + up * 0.20)
        for s_, e_, k in ((sr, er, "r"), (sl, el, "l")):
            self._segment(self.segs[f"upper_arm_{k}"], s_, e_)
            self.balls[f"shoulder_{k}"].location = Vector(s_)
            self.balls[f"elbow_{k}"].location = Vector(e_)
        for i, k in enumerate("rl"):
            hip, knee, ankle = p["hips"][i], p["knees"][i], p["ankles"][i]
            self._segment(self.segs[f"thigh_{k}"], hip, knee)
            self._segment(self.segs[f"shin_{k}"], knee, ankle + np.array([0, 0, 0.02]))
            self.balls[f"hip_{k}"].location = Vector(hip)
            self.balls[f"knee_{k}"].location = Vector(knee)
        self._segment(self.segs["neck"], p["neck"] - up * 0.05, p["neck"] + up * 0.08)
        hR = np.array(pose["head_R"])
        set_pose(self.head, p["head"] + hR[:, 2] * 0.03, Matrix(hR.tolist()).to_quaternion())
        for ob in (self.head, self.hair, self.segs["neck"]):
            ob.visible_camera = not hide_head
        for i, (fp, fy) in enumerate(pose["feet"]):
            fp = np.array(fp)
            f = np.array([np.cos(fy), np.sin(fy), 0])
            set_pose(self.shoes[i], fp + f * 0.05 + np.array([0, 0, 0.0375]), Quaternion((0, 0, 1), fy))


# ------------------------------------------------------------------ cameras / lights / world
OVERVIEW_CAMS = {
    "bedroom": ((3.60, 3.20, 2.25), (1.75, 1.20, 0.60), 16),
    "bathroom": ((3.95, 2.05, 2.25), (5.10, 0.70, 0.85), 16),
    "kitchen": ((5.75, 2.10, 2.25), (6.60, 0.80, 0.90), 16),
    "hall": ((6.85, 6.25, 2.35), (3.80, 3.20, 0.70), 20),
    "living": ((6.85, 6.25, 2.35), (2.40, 4.50, 0.60), 20),
}


def room_of(x, y):
    for name, r in S.ROOMS.items():
        if r["x"][0] <= x <= r["x"][1] and r["y"][0] <= y <= r["y"][1]:
            return name
    return "living"


def setup_world(samples):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for t in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = t; prefs.get_devices()
            for dv in prefs.devices:
                dv.use = dv.type == t
            sc.cycles.device = "GPU"
            break
        except Exception:
            continue
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.denoising_use_gpu = True
    sc.cycles.max_bounces = 8; sc.cycles.diffuse_bounces = 4; sc.cycles.glossy_bounces = 3
    sc.cycles.transmission_bounces = 8; sc.cycles.caustics_reflective = False; sc.cycles.caustics_refractive = False
    sc.cycles.blur_glossy = 1.0
    sc.render.use_persistent_data = True
    sc.render.resolution_x, sc.render.resolution_y = 1280, 720
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.view_settings.exposure = -0.3
    w = bpy.data.worlds.new("sky"); sc.world = w; w.use_nodes = True
    nt = w.node_tree; bg = nt.nodes["Background"]
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(ASSETS / "hdri" / "kloofendal_48d_partly_cloudy_puresky_2k.hdr"))
    mp = nt.nodes.new("ShaderNodeMapping"); tc = nt.nodes.new("ShaderNodeTexCoord")
    mp.inputs["Rotation"].default_value = (0, 0, math.radians(200))
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"]); bg.inputs["Strength"].default_value = 1.0
    # Late-afternoon sun entering through the south windows.
    bpy.ops.object.light_add(type="SUN", rotation=(math.radians(58), 0, math.radians(165)))
    sun = bpy.context.object; sun.data.energy = 3.2; sun.data.angle = math.radians(1.2)
    sun.data.color = (1.0, 0.93, 0.84)
    # Soft interior fill from the ceiling (warm LED panels), one per room.
    for name, r in S.ROOMS.items():
        cx, cy = sum(r["x"]) / 2, sum(r["y"]) / 2
        sx, sy = r["x"][1] - r["x"][0], r["y"][1] - r["y"][0]
        bpy.ops.object.light_add(type="AREA", location=(cx, cy, S.WALL_H - 0.02))
        L = bpy.context.object; L.data.shape = "RECTANGLE"; L.data.size = sx * 0.5; L.data.size_y = sy * 0.5
        L.data.energy = 7 * sx * sy; L.data.color = (1.0, 0.88, 0.74)
        L.visible_camera = False


def look_at(cam, eye, target):
    cam.location = Vector(eye)
    cam.rotation_euler = (Vector(target) - Vector(eye)).to_track_quat("-Z", "Y").to_euler()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("view", choices=["ego", "overview"])
    ap.add_argument("--stride", type=int, default=4); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=-1); ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--out", default=None); ap.add_argument("--indices", default="")
    a = ap.parse_args([x for x in sys.argv[1:]])
    run = Path(a.run_dir).resolve()
    frames = json.loads((run / "frames.json").read_text())["frames"]
    poses = json.loads((run / "operator_poses.json").read_text())["poses"]
    m = mujoco.MjModel.from_xml_path(str(run / "model.xml"))
    bid = {m.body(i).name: i for i in range(m.nbody)}
    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_world(a.samples)
    build_apartment(a.view)
    build_furniture()
    base, lid = build_suitcase()
    items = build_items()
    hands = build_hands(run / "model.xml")
    body = Body()
    cam_data = bpy.data.cameras.new("cam"); cam = link(bpy.data.objects.new("cam", cam_data))
    bpy.context.scene.camera = cam
    cam_data.clip_start = 0.02 if a.view == "ego" else 0.1
    cam_data.sensor_width = 36
    out = Path(a.out).resolve() if a.out else run / f"frames_{a.view}"
    out.mkdir(parents=True, exist_ok=True)
    idx = list(range(a.start, len(frames) if a.stop < 0 else min(a.stop, len(frames)), a.stride))
    if a.indices:
        idx = [int(x) for x in a.indices.split(",")]
    room_prev, room_since = None, -1e9
    sc = bpy.context.scene
    sc.render.image_settings.file_format = "JPEG"; sc.render.image_settings.quality = 93
    for i in idx:
        t0 = time.time()
        fr, po = frames[i], poses[i]
        P = np.array(fr["xpos"]); Q = np.array(fr["xquat"])
        Rb = np.array([np.array(Quaternion(q).to_matrix()) for q in Q])
        set_pose(base, P[bid["suitcase"]], Q[bid["suitcase"]])
        set_pose(lid, P[bid["suitcase_lid"]], Q[bid["suitcase_lid"]])
        place_zip_pull(base, fr.get("zip_progress", 0.0))
        for name, ob in items.items():
            set_pose(ob, P[bid[name]], Q[bid[name]])
        for s in "rl":
            update_hand(hands[s], P, Rb)
        body.update(po, hide_head=a.view == "ego")
        if a.view == "ego":
            eye = np.array(po["pts"]["eye"]); hR = np.array(po["head_R"])
            f, up = hR[:, 0], hR[:, 2]
            Rc = np.column_stack([np.cross(up, -f), up, -f])
            cam.location = Vector(eye); cam.rotation_mode = "QUATERNION"
            cam.rotation_quaternion = Matrix(Rc.tolist()).to_quaternion()
            cam_data.lens = 15
        else:
            rt = fr["root"]
            room = room_of(rt["x"], rt["y"])
            if room != room_prev and (fr["t"] - room_since > 1.5 or room_prev is None):
                room_prev, room_since = room, fr["t"]
            eye, tgt, lens = OVERVIEW_CAMS[room_prev]
            look_at(cam, eye, tgt); cam_data.lens = lens
        sc.render.filepath = str(out / f"{i:06d}.jpg")
        bpy.ops.render.render(write_still=True)
        print(f"FRAME {a.view} {i} {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
