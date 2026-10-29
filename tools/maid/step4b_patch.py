"""Runs inside Blender: close the holes the target's own outfit used to hide.

    Blender -b step4.blend --python tools/maid/step4b_patch.py -- <out.blend>

reze-midnight's skin carries 219 boundary edges across the thigh band, and they are
in the source file, not made here -- the same count before and after the fit. Its own
outfit had straps and a garter running through them, so nothing ever showed. The maid
skirt ends above them and they read as black slots down both thighs.

Only small closed loops are filled. The skin's real openings -- the neck seam, the
foot, whatever the hosiery cut away -- are long loops, and filling one of those would
weld the body shut across a gap that is meant to be there.

The fill adds faces over vertices that already exist, so there are no new UVs to
invent: each new corner takes the UV that vertex already carries elsewhere.
"""
import sys, os
from collections import defaultdict
import bpy, bmesh
from mathutils import Vector

argv = sys.argv[sys.argv.index('--')+1:]
OUT = os.path.abspath(argv[0])
SKIN = 'skin'
MAX_EDGES = 40        # a longer loop is a real opening, not a slot
MAX_SPAN = 1.2        # and so is a wider one

tbody = bpy.data.objects['target_mesh']


def mat_base(m):
    return m.name.split('.')[0] if m.name.rsplit('.', 1)[-1].isdigit() else m.name


skin = {i for i, m in enumerate(tbody.data.materials) if mat_base(m) == SKIN}
bm = bmesh.new()
bm.from_mesh(tbody.data)
bm.verts.ensure_lookup_table()
bm.edges.ensure_lookup_table()
uv = bm.loops.layers.uv.active

border = [e for e in bm.edges
          if len(e.link_faces) == 1 and e.link_faces[0].material_index in skin]
print(f'### {len(border)} skin edges have only one face on them')

adj = defaultdict(list)
for e in border:
    adj[e.verts[0]].append(e)
    adj[e.verts[1]].append(e)
seen, loops = set(), []
for e0 in border:
    if e0 in seen:
        continue
    stack, edges = [e0], []
    seen.add(e0)
    while stack:
        e = stack.pop()
        edges.append(e)
        for v in e.verts:
            for n in adj[v]:
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
    loops.append(edges)
print(f'### they form {len(loops)} separate openings')

# a UV each vertex already carries, to hand to the corners of the new faces
known = {}
for f in bm.faces:
    if f.material_index not in skin:
        continue
    for lp in f.loops:
        known.setdefault(lp.vert.index, lp[uv].uv.copy())

filled, skipped = 0, []
for edges in loops:
    vs = {v for e in edges for v in e.verts}
    pts = [v.co for v in vs]
    span = max((a - b).length for a in pts for b in pts) if len(pts) > 1 else 0.0
    if len(edges) > MAX_EDGES or span > MAX_SPAN:
        skipped.append((len(edges), span))
        continue
    try:
        res = bmesh.ops.triangle_fill(bm, edges=edges, use_beauty=True)
    except Exception:
        continue
    for f in res.get('geom', []):
        if not isinstance(f, bmesh.types.BMFace):
            continue
        f.material_index = min(skin)
        f.smooth = True
        for lp in f.loops:
            if lp.vert.index in known:
                lp[uv].uv = known[lp.vert.index]
        filled += 1

print(f'### filled {filled} faces over {len(loops) - len(skipped)} slots')
print(f'### left {len(skipped)} real openings alone, the largest '
      f'{max(skipped)[0] if skipped else 0} edges across '
      f'{max((s for _, s in skipped), default=0):.2f}')

bm.normal_update()
bm.to_mesh(tbody.data)
bm.free()
tbody.data.update()
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print(f'### wrote {OUT}')
