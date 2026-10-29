"""Runs inside Blender: import both models, cut the outfit out of the donor, and
report how the two BODIES stand against each other.

    Blender --background --python tools/maid/step1_extract.py -- <out.blend>

Nothing is fitted here. This is the measurement the fit will be judged against.

The target arrives whole -- root, armature and mesh, with its mmd_root intact -- so
that what comes out at the end is still an MMD model mmd_tools can write. The donor
is reduced to two things: the outfit, and the body surface the outfit was authored
against. `donor_arm` stays for its outfit bones.
"""
import sys, os
import bpy, bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index('--')+1:]
OUT = os.path.abspath(argv[0])

DONOR = 'public/models/女仆/家政小帮手.pmx'
TARGET = 'public/models/reze-midnight/reze-midnight.pmx'

# Her outfit. `skin_cloth` stays behind: it is her own collarbone, and the target has
# one there already, so it would arrive as a second skin a hundredth off the first.
CLOTH = ['cloth01', 'metal', 'stocking', 'cloth01_alpha', 'skin_stocking', 'skin_metal']

# The surface the outfit is fitted through. Below z 9.17 her legs ARE the hosiery, so
# it counts as body. `face01` is in it because the headdress sits on the head and
# there is nothing else up there to bind it to -- both models are 深空之眼 and their
# `face01` is the same base head, so it carries a correspondence. Neither hair is:
# hers is a waist-length braid and the target's is a short bob, and nothing about one
# says where the other goes.
DONOR_BODY = ['skin', 'stocking', 'face01']
TARGET_BODY = ['skin', 'face01']

# The target's own outfit, which the maid's replaces. Its body, head, hair and eyes stay.
TARGET_DROP = ['cloth01', 'metal01', 'cloth01_shoes', 'cloth01_alpha1',
               'cloth01_alpha2', 'metal01_2', 'gem01', 'cloth01_skin',
               'cloth01_metal01', 'cloth01_metal02']

bpy.ops.wm.read_factory_settings(use_empty=True)
addon_utils = __import__('addon_utils')
addon_utils.enable('bl_ext.blender_org.mmd_tools', default_set=True)


def load(path, tag):
    before = set(bpy.data.objects)
    bpy.ops.mmd_tools.import_model(
        filepath=os.path.abspath(path),
        types={'MESH', 'ARMATURE', 'MORPHS', 'PHYSICS', 'DISPLAY'},
        scale=1.0, clean_model=False)
    new = [o for o in bpy.data.objects if o not in before]
    arm = next(o for o in new if o.type == 'ARMATURE')
    # With PHYSICS imported, every rigid body is a MESH object too. The model's own
    # mesh is the one hanging off the armature.
    mesh = next(o for o in new if o.type == 'MESH' and o.parent is arm)
    root = next(o for o in new if o.type == 'EMPTY' and o.parent is None)
    mesh.name, arm.name, root.name = tag + '_mesh', tag + '_arm', tag + '_root'
    return mesh, arm, root


def base(m):
    """The material name as it was in the PMX, before Blender uniquified it."""
    return m.name.split('.')[0] if m.name.rsplit('.', 1)[-1].isdigit() else m.name


def strip(ob, drop):
    """Delete the faces on the named materials, in place, and the vertices left with
    nothing on them. Deleting faces cannot reach a vertex no face refers to, and the
    source carries 5.2k of those -- they are what put a 'body' bounding box up at the
    top of the hair."""
    idx = {i for i, m in enumerate(ob.data.materials) if base(m) in drop}
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.faces.ensure_lookup_table()
    gone = [f for f in bm.faces if f.material_index in idx]
    bmesh.ops.delete(bm, geom=gone, context='FACES')
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()


def cut(mesh, keep, name):
    """A free-standing copy of `mesh` holding only the named materials."""
    ob = mesh.copy()
    ob.data = mesh.data.copy()
    ob.name = name
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = None
    ob.modifiers.clear()
    # These three are working copies: a measurement surface, a bind surface, and the
    # cloth. None of them wants the donor's 72 expression shape keys, and carrying
    # them costs correctness -- `bmesh` round-trips leave the keys stale, the
    # evaluated mesh then disagrees with the vertices, and Surface Deform binds to
    # the disagreement. It moved the outfit's bones by a median of 1.114 instead of
    # 0.066.
    ob.shape_key_clear()
    have = {base(m) for m in ob.data.materials}
    missing = set(keep) - have
    if missing:
        raise SystemExit(f'{name}: {sorted(missing)} not among {sorted(have)}')
    strip(ob, have - set(keep))
    return ob


def bounds(ob):
    vs = [ob.matrix_world @ v.co for v in ob.data.vertices]
    lo = Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
    hi = Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
    return lo, hi


dm, da, dr = load(DONOR, 'donor')
tm, ta, tr = load(TARGET, 'target')

cloth = cut(dm, CLOTH, 'maid_cloth')
# Both models have a `cloth01.png` and a `cloth01_alpha.png`, and they are different
# sheets. Exported into one folder the second overwrites the first, and the target's
# skin ends up drawn with the maid's dress. The donor's materials are tagged here so
# the export can rename their sheets rather than guess by name later.
for m in cloth.data.materials:
    m['from_donor'] = True
dbody = cut(dm, DONOR_BODY, 'donor_body')
tbody = cut(tm, TARGET_BODY, 'target_body')

# The target's own outfit goes now, so that nothing measures the fit against it.
strip(tm, set(TARGET_DROP))
bpy.data.objects.remove(dm, do_unlink=True)
bpy.data.objects.remove(dr, do_unlink=True)
bpy.context.view_layer.update()

print('### pieces')
for ob in (cloth, dbody, tbody, tm):
    lo, hi = bounds(ob)
    print(f'###   {ob.name:14s} verts={len(ob.data.vertices):6d} polys={len(ob.data.polygons):6d} '
          f'z {lo.z:6.2f}..{hi.z:6.2f}  x +-{max(abs(lo.x), abs(hi.x)):.2f}')
print(f'###   target keeps: {sorted({base(m) for m in tm.data.materials} - set(TARGET_DROP))}')

dlo, dhi = bounds(dbody)
tlo, thi = bounds(tbody)
print(f'### donor body height {dhi.z-dlo.z:.3f}, target {thi.z-tlo.z:.3f}, '
      f'ratio {(thi.z-tlo.z)/(dhi.z-dlo.z):.4f}')

tree = BVHTree.FromObject(tbody, bpy.context.evaluated_depsgraph_get())
d = sorted((tree.find_nearest(dbody.matrix_world @ v.co)[0]
            - (dbody.matrix_world @ v.co)).length for v in dbody.data.vertices)
print(f'### donor body -> target body, unaligned: median {d[len(d)//2]:.3f} '
      f'p90 {d[int(.9*len(d))]:.3f} max {d[-1]:.3f}')

bpy.ops.wm.save_as_mainfile(filepath=OUT)
print(f'### wrote {OUT}')
