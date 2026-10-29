"""Runs inside Blender: morph the donor's body onto the target's and carry the
outfit through with a Surface Deform.

    Blender -b step2.blend --python tools/maid/step3_deform.py -- <out.blend>

The cloth is bound to the body it was authored on, not to the target. So it keeps
its own drape and its own standoff, and what reaches it is only the difference
between the two figures.
"""
import sys, os
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index('--')+1:]
OUT = os.path.abspath(argv[0])
CLOTH, DBODY, TBODY = 'maid_cloth', 'donor_body', 'target_body'

ob = bpy.data.objects
cloth, dbody, tbody = ob[CLOTH], ob[DBODY], ob[TBODY]


def co(o):
    n = len(o.data.vertices)
    a = np.empty(n * 3)
    o.data.vertices.foreach_get('co', a)
    return a.reshape(n, 3)


def normals(o):
    n = len(o.data.vertices)
    a = np.empty(n * 3)
    o.data.vertices.foreach_get('normal', a)
    return a.reshape(n, 3)


def evaluated(o):
    dg = bpy.context.evaluated_depsgraph_get()
    me = o.evaluated_get(dg).to_mesh()
    n = len(me.vertices)
    a = np.empty(n * 3)
    me.vertices.foreach_get('co', a)
    out = a.reshape(n, 3).copy()
    o.evaluated_get(dg).to_mesh_clear()
    return out


def pierce(P, target, label):
    """How much of P is inside `target`, by the sign of the nearest surface."""
    tree = BVHTree.FromObject(target, bpy.context.evaluated_depsgraph_get())
    inside, depth = 0, []
    for p in P:
        loc, nor, _, dist = tree.find_nearest(Vector(p))
        if loc is None:
            continue
        s = (Vector(p) - loc).dot(nor)
        if s < 0 and dist < 0.5:
            inside += 1
            depth.append(-s)
    d = np.array(depth) if depth else np.zeros(1)
    print(f'###   {label}: {inside} of {len(P)} inside, '
          f'median {np.median(d):.3f} p90 {np.percentile(d, 90):.3f} max {d.max():.3f}')
    return inside


print('### before')
pierce(co(cloth), tbody, 'cloth in target body')

# --- the morph, as a shape key on the body the cloth is bound to
print('### morph')
tree = BVHTree.FromObject(tbody, bpy.context.evaluated_depsgraph_get())
A, DN = co(dbody), normals(dbody)
moved = A.copy()
held = 0
for i, p in enumerate(A):
    loc, nor, _, _ = tree.find_nearest(Vector(p))
    if loc is None:
        continue
    # A landing whose normal opposes the one the donor had there is not the same
    # piece of body -- it is the far wall of a crease, or the other side of a finger.
    # Those are left where the alignment put them.
    if np.dot(DN[i], np.array(nor)) < 0.0:
        held += 1
        continue
    moved[i] = np.array(loc)
print(f'###   held {held} vertices whose landing faced the wrong way')

if dbody.data.shape_keys is None:
    dbody.shape_key_add(name='Basis', from_mix=False)
key = dbody.shape_key_add(name='to_target', from_mix=False)
for i, p in enumerate(moved):
    key.data[i].co = Vector(p)
key.value = 0.0
bpy.context.view_layer.update()

# --- bind while the body is still hers, then raise it
print('### surface deform')
carried = [cloth] + ([ob['bone_points']] if 'bone_points' in ob else [])
mods = {}
for o in carried:
    m = o.modifiers.new('fit', 'SURFACE_DEFORM')
    m.target = dbody
    with bpy.context.temp_override(object=o, active_object=o, selected_objects=[o]):
        bpy.ops.object.surfacedeform_bind(modifier=m.name)
    mods[o.name] = m
    print(f'###   {o.name}: bound {m.is_bound}')
key.value = 1.0
bpy.context.view_layer.update()

for o in carried:
    d = np.linalg.norm(evaluated(o) - co(o), axis=1)
    print(f'###   {o.name} moved: median {np.median(d):.3f} '
          f'p90 {np.percentile(d, 90):.3f} max {d.max():.3f}')

for o in carried:
    with bpy.context.temp_override(object=o, active_object=o, selected_objects=[o]):
        bpy.ops.object.modifier_apply(modifier=mods[o.name].name)
key.value = 1.0

print('### after')
pierce(co(cloth), tbody, 'cloth in target body')

bpy.ops.wm.save_as_mainfile(filepath=OUT)
print(f'### wrote {OUT}')
