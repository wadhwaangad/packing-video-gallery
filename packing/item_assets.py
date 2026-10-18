"""Procedural models of the toiletry bag and the water bottle for the Blender renders.

(The folded garments are scanned piles; see prepare_assets.py.) Each builder returns an
empty (the item's body frame, centred on its collision box) with the visual parts
parented to it: the bag's coil zip, slider, leather pull, piping and carry loop; the
bottle's lathe-turned steel foot, powder-coated body, lip ring, ribbed cap and carry
loop. Surface materials use Poly Haven CC0 scans.

Nothing here touches the physics: every model stays inside its item's collision box
(scene.ITEMS), so the recorded simulation is unchanged.
"""
import math
from pathlib import Path
import numpy as np
import bpy, bmesh
from mathutils import Vector, noise
from mathutils.bvhtree import BVHTree

TEX = Path(__file__).resolve().parent / "assets" / "textures"
_mats = {}


# ------------------------------------------------------------------ helpers
def _link(ob):
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _obj(name, verts, faces, mat=None, uvs=None, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.validate(); me.update()
    if uvs is not None:
        uv = me.uv_layers.new(name="UVMap")
        for poly in me.polygons:
            for li in poly.loop_indices:
                uv.data[li].uv = uvs[me.loops[li].vertex_index]
    ob = _link(bpy.data.objects.new(name, me))
    if mat is not None:
        me.materials.append(mat)
    for p in me.polygons:
        p.use_smooth = smooth
    return ob


def _subsurf(ob, levels=1, render=2):
    md = ob.modifiers.new("subd", "SUBSURF"); md.levels = levels; md.render_levels = render
    return md


def _lum_gain(path):
    """1 / mean linear luminance of an image, so a tint sets the actual albedo."""
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB").reduce(16), dtype=np.float64) / 255.0
    lin = np.where(im <= 0.04045, im / 12.92, ((im + 0.055) / 1.055) ** 2.4)
    return 1.0 / max(1e-3, float((lin @ [0.2126, 0.7152, 0.0722]).mean()))


class Surface:
    """Height lookup (top surface) over one or more objects, by ray casting down."""

    def __init__(self, *objs):
        self.trees = []
        self.add(*objs)

    def add(self, *objs):
        for ob in objs:
            dg = bpy.context.evaluated_depsgraph_get()
            bm = bmesh.new(); bm.from_object(ob, dg); bm.transform(ob.matrix_basis)
            self.trees.append(BVHTree.FromBMesh(bm)); bm.free()

    def z(self, x, y, default=0.0):
        best = None
        for t in self.trees:
            hit = t.ray_cast(Vector((x, y, 1.0)), Vector((0, 0, -1)))
            if hit[0] is not None and (best is None or hit[0].z > best):
                best = hit[0].z
        return default if best is None else best


# ------------------------------------------------------------------ materials
def fabric(name, tex, scale, albedo=(1, 1, 1), desat=1.0, normalize=True, sheen=0.35, sheen_rough=0.5,
           rough_mul=1.0, rough_add=0.0, bump=1.0, rot=0.0, coords="Object", projection="BOX",
           edge_fade=0.0, bands=(), micro_bump=0.0, loops=0.0):
    """Principled fabric from a scanned texture folder.

    desat: 1 removes the scan's own colour so `albedo` is the colour; normalize rescales
    the scan to unit mean luminance. edge_fade lightens convex fold edges (worn denim).
    bands: (axis, lo, hi, colour_mul, bump_mul) stripes in object space (towel border).
    """
    if name in _mats:
        return _mats[name]
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; N, L = nt.nodes, nt.links
    bs = N["Principled BSDF"]
    tc = N.new("ShaderNodeTexCoord"); mp = N.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / scale,) * 3
    mp.inputs["Rotation"].default_value = (0, 0, rot)
    L.new(tc.outputs[coords], mp.inputs["Vector"])
    folder = TEX / tex

    def img(fname, non_color):
        p = folder / fname
        n = N.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(str(p), check_existing=True)
        if non_color:
            n.image.colorspace_settings.name = "Non-Color"
        n.projection = projection
        if projection == "BOX":
            n.projection_blend = 0.3
        L.new(mp.outputs["Vector"], n.inputs["Vector"])
        return n

    d = img("diff.jpg", False)
    hs = N.new("ShaderNodeHueSaturation"); hs.inputs["Saturation"].default_value = 1 - desat
    L.new(d.outputs["Color"], hs.inputs["Color"])
    gain = _lum_gain(folder / "diff.jpg") if normalize else 1.0
    mix = N.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    ca = [x for x in mix.inputs if x.type == "RGBA"]
    L.new(hs.outputs["Color"], ca[0])
    ca[1].default_value = (*[c * gain for c in albedo], 1)
    col = [x for x in mix.outputs if x.type == "RGBA"][0]

    nstrength = N.new("ShaderNodeValue"); nstrength.outputs[0].default_value = bump
    nstr = nstrength.outputs[0]
    if bands:
        sep = N.new("ShaderNodeSeparateXYZ"); L.new(tc.outputs["Object"], sep.inputs[0])
        for axis, lo, hi, cmul, bmul in bands:
            g1 = N.new("ShaderNodeMath"); g1.operation = "GREATER_THAN"; g1.inputs[1].default_value = lo
            g2 = N.new("ShaderNodeMath"); g2.operation = "LESS_THAN"; g2.inputs[1].default_value = hi
            L.new(sep.outputs["XYZ".index(axis.upper())], g1.inputs[0]); L.new(sep.outputs["XYZ".index(axis.upper())], g2.inputs[0])
            mk = N.new("ShaderNodeMath"); mk.operation = "MULTIPLY"
            L.new(g1.outputs[0], mk.inputs[0]); L.new(g2.outputs[0], mk.inputs[1])
            cm = N.new("ShaderNodeMix"); cm.data_type = "RGBA"; cm.blend_type = "MULTIPLY"
            L.new(mk.outputs[0], cm.inputs["Factor"])
            cc = [x for x in cm.inputs if x.type == "RGBA"]
            L.new(col, cc[0]); cc[1].default_value = (*cmul, 1)
            col = [x for x in cm.outputs if x.type == "RGBA"][0]
            bm_ = N.new("ShaderNodeMapRange")
            L.new(mk.outputs[0], bm_.inputs["Value"])
            bm_.inputs["From Min"].default_value = 0; bm_.inputs["From Max"].default_value = 1
            bm_.inputs["To Min"].default_value = 1; bm_.inputs["To Max"].default_value = bmul
            ml = N.new("ShaderNodeMath"); ml.operation = "MULTIPLY"
            L.new(nstr, ml.inputs[0]); L.new(bm_.outputs["Result"], ml.inputs[1])
            nstr = ml.outputs[0]
    if edge_fade > 0:
        geo = N.new("ShaderNodeNewGeometry")
        mr = N.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value = 0.52; mr.inputs["From Max"].default_value = 0.66
        L.new(geo.outputs["Pointiness"], mr.inputs["Value"])
        fm = N.new("ShaderNodeMath"); fm.operation = "MULTIPLY"; fm.inputs[1].default_value = edge_fade
        L.new(mr.outputs["Result"], fm.inputs[0])
        sc = N.new("ShaderNodeMix"); sc.data_type = "RGBA"; sc.blend_type = "SCREEN"
        L.new(fm.outputs[0], sc.inputs["Factor"])
        cc = [x for x in sc.inputs if x.type == "RGBA"]
        L.new(col, cc[0]); cc[1].default_value = (0.55, 0.62, 0.72, 1)
        col = [x for x in sc.outputs if x.type == "RGBA"][0]
    loop_h = None
    if loops > 0:
        # terry pile: a dense field of rounded loops (cellular noise ~1.5 mm) with shadowed gaps
        vo = N.new("ShaderNodeTexVoronoi"); vo.feature = "SMOOTH_F1"; vo.inputs["Scale"].default_value = 620.0
        vo.inputs["Randomness"].default_value = 1.0
        L.new(tc.outputs["Object"], vo.inputs["Vector"])
        inv = N.new("ShaderNodeMath"); inv.operation = "SUBTRACT"; inv.inputs[0].default_value = 1.0
        L.new(vo.outputs["Distance"], inv.inputs[1])
        loop_h = inv.outputs[0]
        ao = N.new("ShaderNodeMapRange"); ao.inputs["From Min"].default_value = 0.2; ao.inputs["From Max"].default_value = 1.0
        ao.inputs["To Min"].default_value = 0.45; ao.inputs["To Max"].default_value = 1.08
        L.new(loop_h, ao.inputs["Value"])
        cm = N.new("ShaderNodeMix"); cm.data_type = "RGBA"; cm.blend_type = "MULTIPLY"; cm.inputs["Factor"].default_value = 1.0
        cc = [x for x in cm.inputs if x.type == "RGBA"]
        L.new(col, cc[0]); L.new(ao.outputs["Result"], cc[1])
        col = [x for x in cm.outputs if x.type == "RGBA"][0]
    L.new(col, bs.inputs["Base Color"])

    r = img("rough.jpg", True)
    mr = N.new("ShaderNodeMath"); mr.operation = "MULTIPLY_ADD"
    mr.inputs[1].default_value = rough_mul; mr.inputs[2].default_value = rough_add; mr.use_clamp = True
    L.new(r.outputs["Color"], mr.inputs[0]); L.new(mr.outputs[0], bs.inputs["Roughness"])
    n = img("nor_gl.jpg", True)
    nm = N.new("ShaderNodeNormalMap")
    L.new(nstr, nm.inputs["Strength"])
    L.new(n.outputs["Color"], nm.inputs["Color"])
    normal_out = nm.outputs["Normal"]
    if micro_bump > 0:
        nz = N.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 900.0; nz.inputs["Detail"].default_value = 4
        L.new(tc.outputs["Object"], nz.inputs["Vector"])
        bp = N.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = micro_bump; bp.inputs["Distance"].default_value = 0.0004
        L.new(nz.outputs["Fac"], bp.inputs["Height"]); L.new(normal_out, bp.inputs["Normal"])
        normal_out = bp.outputs["Normal"]
    if loop_h is not None:
        bp = N.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = loops; bp.inputs["Distance"].default_value = 0.0012
        L.new(loop_h, bp.inputs["Height"]); L.new(normal_out, bp.inputs["Normal"])
        normal_out = bp.outputs["Normal"]
    L.new(normal_out, bs.inputs["Normal"])
    bs.inputs["Sheen Weight"].default_value = sheen
    bs.inputs["Sheen Roughness"].default_value = sheen_rough
    _mats[name] = m
    return m


def plain(name, color, rough=0.5, metal=0.0, aniso=0.0, coat=0.0, speckle=0.0):
    if name in _mats:
        return _mats[name]
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; bs = nt.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (*color, 1)
    bs.inputs["Roughness"].default_value = rough
    bs.inputs["Metallic"].default_value = metal
    bs.inputs["Anisotropic"].default_value = aniso
    bs.inputs["Coat Weight"].default_value = coat
    if speckle > 0:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 1400.0; nz.inputs["Detail"].default_value = 2
        nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
        bp = nt.nodes.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = speckle; bp.inputs["Distance"].default_value = 0.0003
        nt.links.new(nz.outputs["Fac"], bp.inputs["Height"]); nt.links.new(bp.outputs["Normal"], bs.inputs["Normal"])
    _mats[name] = m
    return m


# ------------------------------------------------------------------ folded cloth
def sweep(name, pts, width, thick, mat, up=(0, 0, 1), closed=False, subdiv=1, uv_scale=1.0, taper=None):
    """Rectangular strip swept along a 3D polyline (bands, straps, piping, belt loops)."""
    P = np.asarray(pts, float); n = len(P)
    upv = np.asarray(up, float)
    verts = []; uvs = []; s = 0.0
    prof = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    for i in range(n):
        tng = P[(i + 1) % n if closed else min(i + 1, n - 1)] - P[(i - 1) % n if closed else max(i - 1, 0)]
        tng /= np.linalg.norm(tng) + 1e-12
        side = np.cross(tng, upv); side /= np.linalg.norm(side) + 1e-12
        u2 = np.cross(side, tng)
        if i:
            s += float(np.linalg.norm(P[i] - P[i - 1]))
        w = width * (taper(i / max(1, n - 1)) if taper else 1.0)
        for a_, b_ in prof:
            verts.append(P[i] + side * a_ * w + u2 * b_ * thick)
            uvs.append((s / uv_scale, (a_ + 0.5) * w / uv_scale))
    faces = []
    segs = n if closed else n - 1
    for i in range(segs):
        j = (i + 1) % n
        for k in range(4):
            k2 = (k + 1) % 4
            faces.append((i * 4 + k, i * 4 + k2, j * 4 + k2, j * 4 + k))
    if not closed:
        faces.append((3, 2, 1, 0)); faces.append(((n - 1) * 4, (n - 1) * 4 + 1, (n - 1) * 4 + 2, (n - 1) * 4 + 3))
    ob = _obj(name, verts, faces, mat, uvs=uvs)
    bm = bmesh.new(); bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces); bm.to_mesh(ob.data); bm.free()
    if subdiv:
        _subsurf(ob, 1, subdiv)
    return ob


def drape(path_xy, surf, lift, step=0.002):
    """Resample a 2D path and lift it onto the surface (for sweeps lying on cloth)."""
    P = np.asarray(path_xy, float)
    out = []
    for a_, b_ in zip(P[:-1], P[1:]):
        n = max(1, int(np.linalg.norm(b_ - a_) / step))
        for i in range(n):
            q = a_ + (b_ - a_) * i / n
            out.append((q[0], q[1], surf.z(q[0], q[1]) + lift))
    q = P[-1]; out.append((q[0], q[1], surf.z(q[0], q[1]) + lift))
    return out


def _edge_dist(poly, x, y):
    P = np.asarray(poly, float); Q = np.roll(P, -1, axis=0)
    d = Q - P; t = np.clip(((x - P[:, 0]) * d[:, 0] + (y - P[:, 1]) * d[:, 1]) / (d ** 2).sum(1).clip(1e-12), 0, 1)
    return float(np.min(np.hypot(P[:, 0] + t * d[:, 0] - x, P[:, 1] + t * d[:, 1] - y)))


def patch(name, poly_xy, thick, surf, mat, lift=0.0002, res=0.004, bevel=0.0007, uv_scale=0.1, puff=0.0007, seed=0.0):
    """Thin panel of cloth/leather draped on a surface (pockets, labels, patches)."""
    bm = bmesh.new()
    vs = [bm.verts.new((x, y, 0)) for x, y in poly_xy]
    f = bm.faces.new(vs)
    bmesh.ops.triangulate(bm, faces=[f], quad_method="BEAUTY", ngon_method="BEAUTY")
    for _ in range(8):
        long_ = [e for e in bm.edges if e.calc_length() > res]
        if not long_:
            break
        bmesh.ops.subdivide_edges(bm, edges=long_, cuts=1, use_grid_fill=True)
        bmesh.ops.triangulate(bm, faces=[f_ for f_ in bm.faces if len(f_.verts) > 3])
    bmesh.ops.beautify_fill(bm, faces=bm.faces[:], edges=bm.edges[:])
    for v in bm.verts:
        v.co.z = surf.z(v.co.x, v.co.y) + lift
    ret = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
    for v in [g for g in ret["geom"] if isinstance(g, bmesh.types.BMVert)]:
        dd = _edge_dist(poly_xy, v.co.x, v.co.y)
        v.co.z += thick + puff * min(1.0, dd / 0.007) + 0.0003 * noise.noise(Vector((v.co.x * 60, v.co.y * 60, seed)))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    uv = bm.loops.layers.uv.new("UVMap")
    for f_ in bm.faces:
        for l in f_.loops:
            l[uv].uv = (l.vert.co.x / uv_scale, l.vert.co.y / uv_scale)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    ob = _link(bpy.data.objects.new(name, me)); me.materials.append(mat)
    for p_ in me.polygons:
        p_.use_smooth = True
    md = ob.modifiers.new("bevel", "BEVEL"); md.width = bevel; md.segments = 2; md.limit_method = "ANGLE"
    md.harden_normals = False
    ob.modifiers.new("wn", "WEIGHTED_NORMAL")
    return ob


def stitches(name, path_xy, surf, mat, lift=0.0004, pitch=0.0032, dash=0.0021, r=0.00042, closed=False):
    """Dashed thread stitches lying on a surface."""
    P = np.asarray(path_xy, float)
    if closed:
        P = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)])
    verts = []; faces = []
    for s0 in np.arange(pitch * 0.3, s[-1] - dash, pitch):
        def at(sv):
            i = min(np.searchsorted(s, sv) - 1, len(seg) - 1); i = max(i, 0)
            f = (sv - s[i]) / max(seg[i], 1e-9)
            return P[i] + (P[i + 1] - P[i]) * f
        a_, b_ = at(s0), at(s0 + dash)
        d = b_ - a_; d /= np.linalg.norm(d) + 1e-12; sd = np.array([-d[1], d[0]])
        za = surf.z(*a_) + lift; zb = surf.z(*b_) + lift
        base = len(verts)
        for (pt, z) in ((a_, za), (b_, zb)):
            for (u, w) in ((-1, 0), (1, 0), (1, 1), (-1, 1)):
                q = pt + sd * u * r
                verts.append((q[0], q[1], z + w * r * 1.3 - r * 0.3))
        for k in range(4):
            k2 = (k + 1) % 4
            faces.append((base + k, base + k2, base + 4 + k2, base + 4 + k))
        faces.append((base + 3, base + 2, base + 1, base)); faces.append((base + 4, base + 5, base + 6, base + 7))
    ob = _obj(name, verts, faces, mat)
    return ob


def lathe(name, prof, mat, seg=72, radial_fn=None, z_lo=None, z_hi=None):
    """Surface of revolution from (r, z) profile points (bottom to top)."""
    verts = []; faces = []
    for i, (r, z) in enumerate(prof):
        for j in range(seg):
            th = 2 * math.pi * j / seg
            rr = r + (radial_fn(th, z) if radial_fn else 0.0)
            verts.append((rr * math.cos(th), rr * math.sin(th), z))
    for i in range(len(prof) - 1):
        for j in range(seg):
            j2 = (j + 1) % seg
            faces.append((i * seg + j, i * seg + j2, (i + 1) * seg + j2, (i + 1) * seg + j))
    ob = _obj(name, verts, faces, mat)
    bm = bmesh.new(); bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces); bm.to_mesh(ob.data); bm.free()
    for p_ in ob.data.polygons:
        p_.use_smooth = True
    return ob


def _holder(name, parts):
    h = _link(bpy.data.objects.new(name, None))
    for o in parts:
        o.parent = h
    return h


def _arc(cx, cy, rx, ry, a0, a1, n=24):
    return [(cx + rx * math.cos(a0 + (a1 - a0) * i / n), cy + ry * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


# ------------------------------------------------------------------ items
THREAD_GOLD = (0.62, 0.32, 0.07)


def toiletry_bag(name, size, albedo=(0.045, 0.08, 0.068), seed=5.0):
    L, W, H = size
    shell = fabric("bag canvas", "scuba_suede_2k", 0.10, albedo=albedo, sheen=0.3, bump=0.9, micro_bump=0.04)
    leather = fabric("bag leather", "brown_leather_2k", 0.08, albedo=(0.11, 0.055, 0.028), desat=0.5, sheen=0.1, bump=0.7)
    tape_m = plain("zip tape", (0.03, 0.03, 0.035), 0.75)
    metal = plain("zip metal", (0.78, 0.74, 0.62), 0.28, 1.0)
    hx, hy, hz = L / 2 - 0.003, W / 2 - 0.004, H / 2 - 0.002
    # body: subdivided cube pushed onto a rounded box, top seam pinched to the zip, soft sag
    bm = bmesh.new(); bmesh.ops.create_cube(bm, size=2.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=22, use_grid_fill=True)
    e = 5.0
    for v in bm.verts:
        x, y, z = v.co
        nrm = (abs(x) ** e + abs(y) ** e + abs(z) ** e) ** (1 / e)
        x, y, z = x / nrm, y / nrm, z / nrm
        X, Y, Z = x * hx, y * hy, z * hz
        # bottom: flat; top: rises to the zip ridge and slumps either side
        if Z > 0:
            Z = Z - 0.010 * (abs(y) ** 1.5) * (1 - abs(x) ** 6) * (z ** 2)
        if Z < -hz * 0.92:
            Z = -hz * 0.92 - (Z + hz * 0.92) * 0.15
        # side panels bow out slightly in the middle and pucker near the seams
        Y *= 1 + 0.035 * (1 - abs(x) ** 2) * (1 - z ** 2)
        wr = 0.0012 * noise.noise(Vector((X * 40, Y * 40, Z * 40 + seed)))
        wr += 0.0016 * noise.noise(Vector((X * 12, Y * 12, Z * 12 - seed)))
        pl = 0.0018 * math.sin(Y * 260) * max(0.0, abs(x) - 0.75) * 4 * max(0.0, z)   # end pleats near the zip
        n = Vector((X / hx ** 2, Y / hy ** 2, Z / hz ** 2)).normalized()
        v.co = Vector((X, Y, Z)) + n * (wr + pl)
    me = bpy.data.meshes.new(f"{name}_body"); bm.to_mesh(me); bm.free()
    body = _link(bpy.data.objects.new(f"{name}_body", me)); me.materials.append(shell)
    for p_ in me.polygons:
        p_.use_smooth = True
    _subsurf(body, 1, 1)
    surf = Surface(body)
    parts = [body]
    zx = 0.096
    tape = sweep(f"{name}_tape", drape([(-zx - 0.006, 0.0), (zx + 0.006, 0.0)], surf, 0.0006), 0.016, 0.0012, tape_m, subdiv=1)
    parts.append(tape)
    # coil teeth: alternating small blocks along the zip
    tv = []; tf = []
    for i, x in enumerate(np.arange(-zx, zx, 0.0024)):
        s_ = 1 if i % 2 else -1
        z0 = surf.z(x, 0.0) + 0.0012
        b = len(tv)
        for (dx, dy, dz) in [(-0.0008, 0, 0), (0.0008, 0, 0), (0.0008, s_ * 0.0042, 0), (-0.0008, s_ * 0.0042, 0)]:
            tv.append((x + dx, dy - s_ * 0.0008, z0 + dz))
        for (dx, dy, dz) in [(-0.0008, 0, 0), (0.0008, 0, 0), (0.0008, s_ * 0.0042, 0), (-0.0008, s_ * 0.0042, 0)]:
            tv.append((x + dx, dy - s_ * 0.0008, z0 + 0.0015))
        tf += [(b, b + 1, b + 2, b + 3), (b + 7, b + 6, b + 5, b + 4), (b, b + 4, b + 5, b + 1), (b + 1, b + 5, b + 6, b + 2),
               (b + 2, b + 6, b + 7, b + 3), (b + 3, b + 7, b + 4, b)]
    teeth = _obj(f"{name}_teeth", tv, tf, metal, smooth=False)
    parts.append(teeth)
    # slider and leather pull tab
    xs = 0.058; zs = surf.z(xs, 0) + 0.0028
    bpy.ops.mesh.primitive_cube_add(size=1, location=(xs, 0, zs)); sl = bpy.context.object
    sl.scale = (0.013, 0.0095, 0.0042); sl.data.materials.append(metal)
    md = sl.modifiers.new("bevel", "BEVEL"); md.width = 0.0015; md.segments = 3
    parts.append(sl)
    tab_pts = [(xs + 0.006, 0, zs + 0.001), (xs + 0.016, 0.002, zs + 0.0005), (xs + 0.026, 0.004, zs - 0.0015), (xs + 0.033, 0.005, zs - 0.004)]
    tab = sweep(f"{name}_tab", tab_pts, 0.0085, 0.0022, leather, subdiv=1, taper=lambda f: 1.0 - 0.15 * f)
    parts.append(tab)
    # piping around the top and bottom seams
    for zr, nm in ((0.55, "top"), (-0.80, "bot")):
        ring_pts = []
        for i in range(96):
            a_ = 2 * math.pi * i / 96
            dx, dy = math.cos(a_), math.sin(a_)
            # ray from the inside outward at this height
            o = Vector((0, 0, zr * hz)); d = Vector((dx, dy, 0))
            res_ = BVH_cast(body, o, d)
            if res_ is not None:
                ring_pts.append(tuple(res_ + d * 0.0012))
        parts.append(sweep(f"{name}_pipe_{nm}", ring_pts, 0.0042, 0.0042, leather, closed=True, subdiv=1))
    # end carry loop (webbing) at -x
    yl = 0.026; z0 = 0.004
    loop_pts = [(-hx + 0.002, -yl, z0)] + [(-hx - 0.012 * math.sin(math.pi * i / 12), -yl + 2 * yl * i / 12, z0 + 0.002 * math.sin(math.pi * i / 12))
                                         for i in range(1, 12)] + [(-hx + 0.002, yl, z0)]
    loop = sweep(f"{name}_loop", loop_pts, 0.020, 0.0028, leather, up=(0, 0, 1), subdiv=1)
    parts.append(loop)
    return _holder(f"item_{name}", parts)


def BVH_cast(ob, origin, direction):
    dg = bpy.context.evaluated_depsgraph_get()
    bm = bmesh.new(); bm.from_object(ob, dg); bm.transform(ob.matrix_basis)
    tree = BVHTree.FromBMesh(bm); bm.free()
    hit = tree.ray_cast(origin, direction)
    return hit[0]


def water_bottle(name, size, albedo=(0.30, 0.40, 0.33)):
    D, _, Hh = size
    R = D / 2 - 0.0005; zb, zt = -Hh / 2, Hh / 2
    coat = plain("bottle powder coat", albedo, 0.62, coat=0.15, speckle=0.25)
    steel = plain("bottle brushed steel", (0.80, 0.80, 0.80), 0.26, 1.0, aniso=0.6)
    cap_m = plain("bottle cap plastic", (0.045, 0.045, 0.05), 0.5, speckle=0.15)
    parts = []
    # steel foot (uncoated base) and the coated body
    foot = [(0.0, zb), (R - 0.006, zb), (R - 0.0025, zb + 0.0008), (R - 0.0006, zb + 0.003), (R, zb + 0.006)]
    parts.append(lathe(f"{name}_foot", foot, steel))
    zs = zt - 0.060                                  # shoulder start
    prof = [(R, zb + 0.006), (R, zs - 0.01)]
    for i in range(1, 13):
        f = i / 12
        prof.append((R - (R - 0.0255) * (1 - math.cos(f * math.pi / 2)), zs - 0.01 + 0.03 * math.sin(f * math.pi / 2)))
    zn = prof[-1][1]
    prof += [(0.0255, zn + 0.004)]
    parts.append(lathe(f"{name}_body", prof, coat))
    # steel lip ring
    ring = [(0.0255, zn + 0.004), (0.0262, zn + 0.0045), (0.0264, zn + 0.009), (0.0258, zn + 0.0095), (0.020, zn + 0.0095)]
    parts.append(lathe(f"{name}_lip", ring, steel))
    # cap with grip ribs and a carry loop
    zc0 = zn + 0.0095; zc1 = zt - 0.012
    capp = [(0.0, zc0), (0.0272, zc0), (0.0278, zc0 + 0.0015)]
    for i in range(1, 8):
        capp.append((0.0278, zc0 + 0.0015 + (zc1 - zc0 - 0.006) * i / 7))
    for i in range(1, 9):
        a_ = math.pi / 2 * i / 8
        capp.append((0.0278 - 0.0045 * (1 - math.cos(a_)), zc1 - 0.0045 + 0.0045 * math.sin(a_)))
    capp.append((0.0, zc1))
    rib_z = (zc0 + 0.002, zc1 - 0.006)
    parts.append(lathe(f"{name}_cap", capp, cap_m, seg=144,
                       radial_fn=lambda th, z: 0.0006 * max(0.0, math.cos(th * 36)) if rib_z[0] < z < rib_z[1] else 0.0))
    arch = [(0.0125 * math.cos(math.pi * i / 24), 0.0, zc1 - 0.002 + 0.0105 * math.sin(math.pi * i / 24)) for i in range(25)]
    parts.append(sweep(f"{name}_loop", arch, 0.0062, 0.0050, cap_m, up=(0, 1, 0), subdiv=1))
    return _holder(f"item_{name}", parts)


BUILDERS = {
    "toiletry_bag": lambda it: toiletry_bag(it.name, it.size),
    "bottle": lambda it: water_bottle(it.name, it.size),
}


def build(it):
    return BUILDERS[it.name](it)
