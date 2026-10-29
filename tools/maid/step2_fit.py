"""Runs inside Blender: align the donor body to the target's, then morph one onto
the other and carry the outfit through with a Surface Deform.

    Blender -b step1.blend --python tools/maid/step2_fit.py -- <out.blend>

The fit is the standard one: bind the cloth to the body it was authored on, then
move that body onto ours. The cloth never sees the target directly, so it keeps its
own drape and its standoff instead of being pulled onto a surface.
"""
import sys, os
import bpy, bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index('--')+1:]
OUT = os.path.abspath(argv[0])

CLOTH, DBODY, TBODY = 'maid_cloth', 'donor_body', 'target_body'
REACH = 1.2          # a correspondence longer than this is not a correspondence


def co(ob):
    n = len(ob.data.vertices)
    a = np.empty(n * 3)
    ob.data.vertices.foreach_get('co', a)
    return a.reshape(n, 3) @ np.array(ob.matrix_world.to_3x3()).T + np.array(ob.matrix_world.translation)


def icp(A, target, iters=40):
    """Uniform scale + translation carrying point cloud A onto a mesh object."""
    tree = BVHTree.FromObject(target, bpy.context.evaluated_depsgraph_get())
    B0 = co(target)
    s, t = (B0[:, 2].max() - B0[:, 2].min()) / (A[:, 2].max() - A[:, 2].min()), np.zeros(3)
    t = B0.mean(0) - s * A.mean(0)
    for _ in range(iters):
        P = A * s + t
        M = np.array([tree.find_nearest(Vector(p))[0] for p in P])
        d = np.linalg.norm(M - P, axis=1)
        keep = d < max(np.percentile(d, 85), 1e-9)
        X, Y = A[keep], M[keep]
        cx, cy = X.mean(0), Y.mean(0)
        s = float(((X - cx) * (Y - cy)).sum() / ((X - cx) ** 2).sum())
        t = cy - s * cx
    P = A * s + t
    d = np.array([(tree.find_nearest(Vector(p))[0] - Vector(p)).length for p in P])
    return s, t, d


ob = bpy.data.objects
dbody, tbody, cloth = ob[DBODY], ob[TBODY], ob[CLOTH]

print('### align')
s, t, d = icp(co(dbody), tbody)
print(f'###   scale {s:.4f}  offset {np.round(t, 3)}  '
      f'residual median {np.median(d):.3f} p90 {np.percentile(d, 90):.3f} max {d.max():.3f}')

def move_verts(o, s, t):
    n = len(o.data.vertices)
    a = np.empty(n * 3)
    o.data.vertices.foreach_get('co', a)
    o.data.vertices.foreach_set('co', (a.reshape(n, 3) * s + t).ravel())
    o.data.update()


for o in (dbody, cloth):
    move_verts(o, s, t)

# The outfit's own bones -- the skirt chains, the sleeve chains, the bows -- have to
# arrive where their cloth arrives. They are carried as a point cloud through exactly
# the same two stages the mesh goes through, so a chain cannot end up outside the
# geometry hanging on it. Two vertices per bone: head, then tail.
arm = ob['donor_arm']
names, pts = [], []
for b in arm.data.bones:
    names.append(b.name)
    pts.append(np.array(arm.matrix_world @ b.head_local) * s + t)
    pts.append(np.array(arm.matrix_world @ b.tail_local) * s + t)
me = bpy.data.meshes.new('bone_points')
me.from_pydata([tuple(p) for p in pts], [], [])
me.update()
bp = bpy.data.objects.new('bone_points', me)
bpy.context.scene.collection.objects.link(bp)
bp['bone_names'] = names
print(f'###   carrying {len(names)} donor bones as {len(pts)} points')
bpy.context.view_layer.update()

# --- the morph, and whether its correspondence is honest
print('### shrinkwrap donor body -> target body')
tree = BVHTree.FromObject(tbody, bpy.context.evaluated_depsgraph_get())
A = co(dbody)
hit = [tree.find_nearest(Vector(p)) for p in A]
M = np.array([h[0] for h in hit])
N = np.array([h[1] for h in hit])
move = np.linalg.norm(M - A, axis=1)
# The A-pose trap: a thigh whose nearest surface is the hand hanging beside it. It
# shows up as a long move, and as a landing normal that disagrees with the one the
# donor had there.
dn = np.empty(len(A) * 3)
dbody.data.vertices.foreach_get('normal', dn)
dn = dn.reshape(-1, 3)
flip = (dn * N).sum(1) < 0.0
print(f'###   move median {np.median(move):.3f} p90 {np.percentile(move, 90):.3f} '
      f'max {move.max():.3f}')
print(f'###   over {REACH}: {int((move > REACH).sum())} of {len(A)}')
print(f'###   landing normal disagrees with the donor\'s: {int(flip.sum())} of {len(A)}')
bad = (move > REACH) | flip
if bad.any():
    P = A[bad]
    for lo in range(0, 22, 2):
        k = int(((P[:, 2] >= lo) & (P[:, 2] < lo + 2)).sum())
        if k:
            print(f'###     z[{lo:2d},{lo+2:2d}) {k}')
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print(f'### wrote {OUT}')
