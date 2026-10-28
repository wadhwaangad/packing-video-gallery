"""The videos' hands for the 3D page: play/pack/hands.glb and play/pack/hands.json.

render_blender.build_hands makes each hand's skin the way the renders do (the MyoHand's
collision capsules joined, voxel-remeshed and smoothed, every vertex bound to its nearest
hand bodies). Here that skin is decimated for real time and written with its binding, and
the hand's kinematic tree (MyoHand bodies and hinge joints, from build_model) goes beside
it, with the grip shapes the video controllers use (runner.SHAPES) and the palm-to-forearm
table of the wrist (Runner._build_wrist_table). The page poses the hands from those: the
same skeleton and finger shapes as the videos, driven by the page's animation instead of
the offline physics (which the browser cannot run at interactive speed).

Usage: .venv/Scripts/python packing/export_hands.py
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bpy, mujoco
import render_blender as RB
import build_model as B
import runner as RN

OUT = HERE.parent / "play" / "pack"
MAX_FACES = 9000          # per hand


def subtree(m, root):
    def desc(b):
        while b > 0:
            if b == root:
                return True
            b = m.body_parentid[b]
        return False
    return [b for b in range(m.nbody) if desc(b)]


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    m, path = B.build()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    hands = RB.build_hands(path)
    r = RN.Runner.__new__(RN.Runner)               # only the wrist table: Runner._build_wrist_table on this model
    r.m = m
    r._build_wrist_table()
    out = dict(shapes={k: RN.shape_targets(k, "r") for k in RN.SHAPES}, palm={s: RN.PALM[s].tolist() for s in "rl"},
               wrist_grid=RN.Runner.WRIST_GRID.tolist(), sides={})
    r4 = lambda v: [round(float(x), 6) for x in np.ravel(v)]
    for s in "rl":
        h = hands[s]
        ob = h["obj"]
        bodies = subtree(m, m.body(f"radius_{s}").id)
        idx = {b: i for i, b in enumerate(bodies)}
        # decimate, then bind the kept vertices again as render_blender does (nearest bodies)
        n0 = len(ob.data.polygons)
        bpy.context.view_layer.objects.active = ob
        md = ob.modifiers.new("slim", "DECIMATE"); md.ratio = min(1.0, MAX_FACES / n0)
        bpy.ops.object.modifier_apply(modifier="slim")
        bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.quads_convert_to_tris(); bpy.ops.object.mode_set(mode="OBJECT")
        V = np.array([v.co[:] for v in ob.data.vertices])
        F = np.array([p.vertices[:] for p in ob.data.polygons])
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in idx and "coll" in m.geom(g).name]
        dist = []
        for g in geoms:
            pos, R, sz = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
            local = (V - pos) @ R
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                local[:, 2] -= np.clip(local[:, 2], -sz[1], sz[1]); dd = np.linalg.norm(local, axis=1) - sz[0]
            else:
                dd = (np.linalg.norm(local / np.maximum(sz, 1e-3), axis=1) - 1) * min(sz)
            dist.append(dd)
        D = np.array(dist).T
        near = np.argsort(D, axis=1)[:, :3]
        vb = np.array([idx[m.geom_bodyid[g]] for g in geoms])[near]
        w = np.exp(-np.take_along_axis(D, near, axis=1) / 0.002); w /= w.sum(axis=1)[:, None]
        # skeleton: per body its parent (in this list), rest pos/quat in the parent, and its hinges
        bl = []
        for b in bodies:
            js = [dict(name=m.joint(j).name, pos=r4(m.jnt_pos[j]), axis=r4(m.jnt_axis[j]))
                  for j in range(m.njnt) if m.jnt_bodyid[j] == b and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
            bl.append(dict(name=m.body(b).name, parent=idx.get(int(m.body_parentid[b]), -1), pos=r4(m.body_pos[b]),
                           quat=r4(m.body_quat[b]), joints=js,
                           bind_pos=r4(d.xpos[b]), bind_quat=r4(d.xquat[b])))
        P, Rt = r.wrist[s]
        out["sides"][s] = dict(bodies=bl, vertices=r4(V), faces=F.astype(int).ravel().tolist(), skin_body=vb.ravel().tolist(),
                               skin_weight=r4(w), wrist_pos=r4(P), wrist_rot=r4(Rt))
        print(s, len(V), "vertices,", len(F), "faces,", len(bodies), "bodies,", sum(len(b["joints"]) for b in bl), "hinges", flush=True)
    (OUT / "hands.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"hands.json {(OUT / 'hands.json').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
