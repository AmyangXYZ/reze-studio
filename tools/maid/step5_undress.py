"""Runs inside Blender: take the target's body out from under the opaque hosiery.

    Blender -b step4.blend --python tools/maid/step5_undress.py -- <out.blend>

Clearance alone does not settle this pair, because the two surfaces interleave rather
than one sitting inside the other. The hosiery is 974 vertices over a leg the target
draws with several thousand, so a hosiery triangle spans bumps the skin resolves: 610
of 1546 leg vertices sit outside the hosiery while 102 hosiery vertices sit inside the
skin, and lifting either one further only trades which of them shows.

The donor has no skin under her hosiery for the same reason, and neither does any
other model with opaque legwear. So the body goes.

Two guards keep the cut honest. A face goes only when all three of its corners are
covered, and a corner within `HEM` of the hosiery's open edge is never counted --
otherwise the ring of skin the top edge needs to hide behind is the first thing to go.
"""
import sys, os
import bpy, bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

argv = sys.argv[sys.argv.index('--')+1:]
OUT = os.path.abspath(argv[0])
OPAQUE = ['stocking']      # her material names, as they came in
REACH = 0.4                # further than this is a different limb
SLACK = 0.0                # only skin strictly inside is cut; the rest is tucked
HEM = 0.35                 # never cut or tuck this close to the hosiery's open edge
TUCK = 0.02                # how far under the hosiery the kept ring is pushed

ob = bpy.data.objects
# The real mesh, not the `target_body` helper -- that one is a measurement copy and
# step 6 deletes it, so anything cut there is thrown away.
cloth, tbody = ob['maid_cloth'], ob['target_mesh']
SKIN = 'skin'

slots = [m.name.split('.')[0] for m in cloth.data.materials]
want = {i for i, n in enumerate(slots) if n in OPAQUE}
if not want:
    raise SystemExit(f'no {OPAQUE} among {slots}')

bm = bmesh.new()
bm.from_mesh(cloth.data)
bm.faces.ensure_lookup_table()
sub = bmesh.new()
vmap = {}
for f in bm.faces:
    if f.material_index not in want:
        continue
    vs = []
    for v in f.verts:
        if v.index not in vmap:
            vmap[v.index] = sub.verts.new(v.co)
        vs.append(vmap[v.index])
    try:
        sub.faces.new(vs)
    except ValueError:
        pass
bm.free()
# Without this the face normals are zero and every signed test reads as outside.
sub.normal_update()
sub.verts.ensure_lookup_table()
sub.faces.ensure_lookup_table()
sub.edges.ensure_lookup_table()
tree = BVHTree.FromBMesh(sub)

# The hem guard measures to the open edge, so the edge is sampled along its length,
# not just at its vertices -- the loop's vertices sit further apart than the guard
# itself, and a point between two of them was reading as far from the hem and getting
# cut. That is where the holes in the front of both thighs came from.
open_edges = [e for e in sub.edges if len(e.link_faces) < 2]
samples = []
for e in open_edges:
    a, b = e.verts[0].co, e.verts[1].co
    for k in range(4):
        samples.append(a.lerp(b, k / 3.0))
kd = KDTree(len(samples))
for i, p in enumerate(samples):
    kd.insert(p, i)
kd.balance()
edge = samples
print(f'### hosiery: {len(sub.verts)} verts, {len(sub.faces)} faces, '
      f'{len(open_edges)} open edges sampled at {len(samples)} points')

def mat_base(m):
    return m.name.split('.')[0] if m.name.rsplit('.', 1)[-1].isdigit() else m.name


skin_slots = {i for i, m in enumerate(tbody.data.materials) if mat_base(m) == SKIN}
on_skin = np.zeros(len(tbody.data.vertices), bool)
for poly in tbody.data.polygons:
    if poly.material_index in skin_slots:
        for vi in poly.vertices:
            on_skin[vi] = True
print(f'### {int(on_skin.sum())} of {len(on_skin)} target vertices are skin')

n = len(tbody.data.vertices)
a = np.empty(n * 3)
tbody.data.vertices.foreach_get('co', a)
V = a.reshape(n, 3)
covered = np.zeros(n, bool)
inside = 0
for i, p in enumerate(V):
    if not on_skin[i]:
        continue
    loc, nor, _, dist = tree.find_nearest(Vector(p))
    if loc is None or dist > REACH:
        continue
    s = (Vector(p) - loc).dot(nor)
    inside += s < 0
    if s < SLACK and kd.find(Vector(p))[2] > HEM:
        covered[i] = True
print(f'### {inside} body vertices are strictly inside it; '
      f'{int(covered.sum())} of {n} count as covered')

bm = bmesh.new()
bm.from_mesh(tbody.data)
bm.verts.ensure_lookup_table()
gone = [f for f in bm.faces
        if f.material_index in skin_slots and all(covered[v.index] for v in f.verts)]
print(f'### dropping {len(gone)} of {len(bm.faces)} body faces')
bmesh.ops.delete(bm, geom=gone, context='FACES')
loose = [v for v in bm.verts if not v.link_faces]
if loose:
    bmesh.ops.delete(bm, geom=loose, context='VERTS')

# What is left is the ring the all-three-corners rule held back, and some of it still
# stands proud of the hosiery -- the two surfaces interleave, so a corner of a face
# that was kept can be outside while its neighbours are under. That ring is what
# showed as the white streaking down both thighs; the cloth on its own renders clean.
# It is tucked rather than cut, so the hem keeps something to hide behind. A vertex
# whose nearest hosiery point is on the open edge is left alone: there the hosiery has
# ended and the thigh above it is meant to be bare.
bm.verts.ensure_lookup_table()
tucked = 0
for v in bm.verts:
    if v.index >= len(on_skin) or not on_skin[v.index]:
        continue
    loc, nor, _, dist = tree.find_nearest(v.co)
    if loc is None or dist > REACH or kd.find(v.co)[2] <= HEM:
        continue
    s = (v.co - loc).dot(nor)
    if s > -TUCK:
        v.co = loc - nor * TUCK
        tucked += 1
print(f'### tucked {tucked} body vertices back under the hosiery')

bm.to_mesh(tbody.data)
bm.free()
tbody.data.update()

z = np.array([v.co.z for v in tbody.data.vertices])
low = z[z < 12]
print(f'### body now {len(tbody.data.vertices)} verts; below the hip it reaches '
      f'down to z {low.min():.2f}')
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print(f'### wrote {OUT}')
