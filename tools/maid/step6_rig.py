"""Runs inside Blender: give the outfit the target's skeleton, then write the PMX.

    Blender -b step5.blend --python tools/maid/step6_rig.py -- <out.pmx> [out.blend]

A vertex group on the outfit means one of two things, and the fit itself tells them
apart. Carried through the same deform as the cloth, a bone that is really the same
bone lands on the target's own: `頭` at 0.058, `手首` at 0.068, `上半身` at 0.135, and
the whole body ends by 0.436 at `胸`. A bone that only shares a name -- both models
number a skirt `skirt_<panel>_<link>` -- lands 1.755 away at the nearest and 4.794 at
the worst. Nothing in between, so the split is read off the distance and not guessed.

Bones the target does not have arrive under a prefix rather than displacing anything,
and the outfit's vertex groups are renamed to match.
"""
import sys, os
import bpy, addon_utils
import numpy as np
from mathutils import Vector

# Opening a .blend restores its preferences, which turns the extension back off.
addon_utils.enable('bl_ext.blender_org.mmd_tools', default_set=True)

argv = sys.argv[sys.argv.index('--')+1:]
PMX = os.path.abspath(argv[0])
BLEND = os.path.abspath(argv[1]) if len(argv) > 1 else None

SAME = 1.0          # the gap is 0.436 .. 1.755; anything in this band is one bone
PREFIX = 'M_'
MIN_BONE = 0.01     # Blender drops a zero-length bone

ob = bpy.data.objects
cloth, bp, tarm, tmesh = ob['maid_cloth'], ob['bone_points'], ob['target_arm'], ob['target_mesh']
darm = ob['donor_arm']
root = ob['target_root']

names = list(bp['bone_names'])
P = np.array([v.co for v in bp.data.vertices]).reshape(-1, 2, 3)
head = {n: P[i, 0] for i, n in enumerate(names)}
tail = {n: P[i, 1] for i, n in enumerate(names)}
dparent = {b.name: (b.parent.name if b.parent else None) for b in darm.data.bones}

# `mmd_edge_scale` and `mmd_vertex_order` are mmd_tools' own bookkeeping groups, not
# bones. They carry weight on every vertex and have to survive the export.
KEEP_GROUPS = ('mmd_',)
used = set()
for v in cloth.data.vertices:
    for g in v.groups:
        nm = cloth.vertex_groups[g.group].name
        if g.weight > 0.001 and not nm.startswith(KEEP_GROUPS):
            used.add(nm)
have = {b.name: np.array(b.head_local) for b in tarm.data.bones}

shared, fresh = {}, []
for n in sorted(used):
    if n in have and n in head and np.linalg.norm(head[n] - have[n]) < SAME:
        shared[n] = n
    else:
        fresh.append(n)
print(f'### outfit leans on {len(used)} bones: {len(shared)} are the target\'s own, '
      f'{len(fresh)} arrive with it')
print(f'###   shared: {", ".join(sorted(shared)[:10])} ...')


def resolve(n):
    """The name this donor bone should hang from, on the target."""
    seen = 0
    while n is not None and seen < 40:
        if n in shared:
            return shared[n]
        if n in newname:
            return newname[n]
        n = dparent.get(n)
        seen += 1
    return None


# Every ancestor of a bone we carry has to come too, or the chain arrives cut.
need = set(fresh)
while True:
    grew = False
    for n in list(need):
        p = dparent.get(n)
        if p and p not in need and p not in shared and p in head:
            need.add(p)
            grew = True
    if not grew:
        break
order = [n for n in names if n in need]        # donor order keeps parents first
newname = {n: PREFIX + n for n in order}
print(f'### creating {len(order)} bones on the target armature')

bpy.context.view_layer.objects.active = tarm
bpy.ops.object.mode_set(mode='EDIT')
eb = tarm.data.edit_bones
made = {}
for n in order:
    h, t = Vector(head[n]), Vector(tail[n])
    if (t - h).length < MIN_BONE:
        t = h + Vector((0.0, 0.0, MIN_BONE))
    b = eb.new(newname[n])
    b.head, b.tail, b.use_connect = h, t, False
    made[n] = b
for n in order:
    p = resolve(dparent.get(n))
    if p and p in eb:
        made[n].parent = eb[p]
bpy.ops.object.mode_set(mode='OBJECT')

# This build of mmd_tools keeps the Japanese name as the bone name itself and only
# swaps 左/右 for a .L/.R suffix, so the prefix is the whole of the rename.
for n in order:
    g = cloth.vertex_groups.get(n)
    if g:
        g.name = newname[n]
dead = [g.name for g in cloth.vertex_groups
        if g.name not in used and not g.name.startswith(KEEP_GROUPS)]
for nm in dead:
    cloth.vertex_groups.remove(cloth.vertex_groups[nm])
print(f'###   dropped {len(dead)} vertex groups nothing leans on')

missing = [g.name for g in cloth.vertex_groups
           if g.name not in tarm.data.bones and not g.name.startswith(KEEP_GROUPS)]
if missing:
    raise SystemExit(f'### {len(missing)} groups still have no bone: {missing[:8]}')
print('### every vertex group now names a bone on the target armature')

cloth.modifiers.clear()
cloth.parent = tarm
m = cloth.modifiers.new('armature', 'ARMATURE')
m.object = tarm

for name in ('donor_body', 'target_body', 'bone_points'):
    if name in ob:
        bpy.data.objects.remove(ob[name], do_unlink=True)
bpy.data.objects.remove(darm, do_unlink=True)

bpy.ops.object.select_all(action='DESELECT')
cloth.select_set(True)
tmesh.select_set(True)
bpy.context.view_layer.objects.active = tmesh
bpy.ops.object.join()
# The outfit arrived carrying all 20 of the donor's material slots, 14 of which the
# cut left with no faces on them. Exported they become empty materials in the PMX.
counts = [0] * len(tmesh.data.materials)
for poly in tmesh.data.polygons:
    counts[poly.material_index] += 1
empty = [i for i, c in enumerate(counts) if c == 0]
for i in reversed(empty):
    tmesh.data.materials.pop(index=i)
print(f'### joined: {len(tmesh.data.vertices)} verts, {len(tmesh.data.polygons)} polys, '
      f'{len(tmesh.data.materials)} materials ({len(empty)} empty slots dropped)')
print(f'###   {[m.name for m in tmesh.data.materials]}')

# Rename the donor's sheets so they cannot collide with the target's on the way out.
import shutil, tempfile
tmp = tempfile.mkdtemp(prefix='maid-tex-')
done = {}
for m in tmesh.data.materials:
    if not m.get('from_donor') or not m.use_nodes:
        continue
    for node in m.node_tree.nodes:
        img = getattr(node, 'image', None)
        if img is None or not img.filepath:
            continue
        src = bpy.path.abspath(img.filepath)
        if img.name in done or not os.path.exists(src):
            continue
        dst = os.path.join(tmp, 'maid_' + os.path.basename(src))
        shutil.copyfile(src, dst)
        img.filepath = dst
        done[img.name] = os.path.basename(dst)
print(f'### renamed {len(done)} donor sheets: {sorted(done.values())}')

bpy.ops.object.select_all(action='DESELECT')
root.select_set(True)
bpy.context.view_layer.objects.active = root
bpy.ops.mmd_tools.export_pmx(filepath=PMX, scale=1.0, check_existing=False,
                             copy_textures_mode='OVERWRITE', sort_materials=False,
                             log_level='WARNING')
print(f'### wrote {PMX}')
if BLEND:
    bpy.ops.wm.save_as_mainfile(filepath=BLEND)
    print(f'### wrote {BLEND}')
