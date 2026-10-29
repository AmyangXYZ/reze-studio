"""Runs inside Blender: lift the outfit clear of the target body.

    Blender -b step3.blend --python tools/maid/step4_clear.py -- <out.blend> [offset]

Shrinkwrap in `OUTSIDE` mode is exactly this operation: it moves a vertex only if it
is inside the target, and leaves everything hanging free alone.

Corrective Smooth does not belong after it. It smooths the deformation a modifier
above it produced, and against an already-applied Shrinkwrap there is no deformation
left to read -- it smooths the mesh itself instead, which moved all 11028 vertices by
a median of 0.145 and took the drape with it.

The hosiery is the piece that needs it. Below z 9.17 the donor's legs ARE the
hosiery, so it is part of the body the fit reads, and the morph lands it exactly on
the target's bare leg. That is the right place for it and an unrenderable one.
"""
import sys, os
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index('--')+1:]
OUT = os.path.abspath(argv[0])
OFFSET = float(argv[1]) if len(argv) > 1 else 0.015

ob = bpy.data.objects
cloth, tbody = ob['maid_cloth'], ob['target_body']


def co(o):
    n = len(o.data.vertices)
    a = np.empty(n * 3)
    o.data.vertices.foreach_get('co', a)
    return a.reshape(n, 3)


def report(label):
    tree = BVHTree.FromObject(tbody, bpy.context.evaluated_depsgraph_get())
    depth = []
    for p in co(cloth):
        loc, nor, _, dist = tree.find_nearest(Vector(p))
        if loc is None or dist > 0.5:
            continue
        s = (Vector(p) - loc).dot(nor)
        if s < OFFSET * 0.5:
            depth.append(OFFSET * 0.5 - s)
    d = np.array(depth) if depth else np.zeros(1)
    print(f'###   {label}: {len(depth)} of {len(cloth.data.vertices)} short of clearance, '
          f'median {np.median(d):.4f} max {d.max():.4f}')


before = co(cloth).copy()
print('### clearance')
report('before')

sw = cloth.modifiers.new('clear', 'SHRINKWRAP')
sw.target = tbody
sw.wrap_method = 'NEAREST_SURFACEPOINT'
sw.wrap_mode = 'OUTSIDE'
sw.offset = OFFSET

with bpy.context.temp_override(object=cloth, active_object=cloth,
                               selected_objects=[cloth]):
    bpy.ops.object.modifier_apply(modifier=sw.name)

report('after')
d = np.linalg.norm(co(cloth) - before, axis=1)
print(f'###   outfit moved: {int((d > 1e-6).sum())} vertices, median '
      f'{np.median(d[d > 1e-6]) if (d > 1e-6).any() else 0:.4f} max {d.max():.4f}')

bpy.ops.wm.save_as_mainfile(filepath=OUT)
print(f'### wrote {OUT}')
