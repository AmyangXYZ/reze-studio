"""Runs inside Blender: deform the chest with a lattice and smooth what needs it.

    Blender --background --python bust_blender.py -- in.npz out.npz scene.blend

in.npz carries the vertices, the faces of each material that becomes a mesh
object (`mesh_names`, `faces_<name>`), the lattice (box, dims, displacement per
control point) and per-vertex weights: `gate` says how much of the lattice each
vertex takes, `smooth` how much the Laplacian smoothing may move it. Every other
vertex is a loose point in one more object, the rig's points a last one; all
hang on the same lattice, so cup, skin, ruffle and bones move in one field. The
evaluated positions go to out.npz; the scene is saved so the cage can be
inspected or nudged in Blender.
"""
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index('--') + 1:]
src, dst, blend = argv[:3]
d = np.load(src)
V = d['V']
gate, smooth = d['gate'], d['smooth']
box_lo, box_hi, dims, disp = d['box_lo'], d['box_hi'], d['dims'], d['disp']
lam, iters = float(d['lambda']), int(d['iterations'])
mesh_names = [str(n) for n in d['mesh_names']]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def mesh_object(name, verts, faces=()):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(map(int, f)) for f in faces])
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    return ob


def group(ob, name, weights):
    vg = ob.vertex_groups.new(name=name)
    for i, w in enumerate(weights):
        if w > 0:
            vg.add([int(i)], float(w), 'REPLACE')
    return vg


# the lattice: rest points span (dims-1) local units, so scale maps them onto the box
lat = bpy.data.lattices.new('bust')
lat.points_u, lat.points_v, lat.points_w = (int(x) for x in dims)
lat.interpolation_type_u = lat.interpolation_type_v = lat.interpolation_type_w = 'KEY_BSPLINE'
cage = bpy.data.objects.new('bust', lat)
scene.collection.objects.link(cage)
scale = (box_hi - box_lo) / (dims - 1)
cage.location = tuple(float(x) for x in (box_lo + box_hi) / 2)
cage.scale = tuple(float(x) for x in scale)
for i, p in enumerate(lat.points):
    p.co_deform = tuple(float(x) for x in np.array(p.co) + disp[i] / scale)

out = V.copy()
taken = np.zeros(len(V), bool)
meshes = []
for name in mesh_names:
    faces = d['faces_' + name]
    idx = np.unique(faces)
    taken[idx] = True
    remap = -np.ones(len(V), dtype=int)
    remap[idx] = np.arange(len(idx))
    ob = mesh_object(name, V[idx], remap[faces])
    group(ob, 'gate', gate[idx])
    m = ob.modifiers.new('bust', 'LATTICE')
    m.object, m.vertex_group = cage, 'gate'
    if (smooth[idx] > 0).any():
        group(ob, 'smooth', smooth[idx])
        s = ob.modifiers.new('smooth', 'LAPLACIANSMOOTH')
        s.lambda_factor, s.iterations, s.use_volume_preserve, s.vertex_group = lam, iters, False, 'smooth'
    meshes.append((ob, idx))

rest_idx = np.where(~taken)[0]
rest = mesh_object('rest', V[rest_idx])
group(rest, 'gate', gate[rest_idx])
m = rest.modifiers.new('bust', 'LATTICE')
m.object, m.vertex_group = cage, 'gate'

rig = mesh_object('rig', d['rig'])
rig.modifiers.new('bust', 'LATTICE').object = cage

dg = bpy.context.evaluated_depsgraph_get()
for ob, idx in meshes:
    out[idx] = [v.co[:] for v in ob.evaluated_get(dg).data.vertices]
out[rest_idx] = [v.co[:] for v in rest.evaluated_get(dg).data.vertices]
rig_out = np.array([v.co[:] for v in rig.evaluated_get(dg).data.vertices]).reshape(-1, 3)
np.savez(dst, V=out, rig=rig_out)
bpy.ops.wm.save_as_mainfile(filepath=blend)
print(f'lattice {dims[0]}x{dims[1]}x{dims[2]} over {np.round(box_lo, 2)}..{np.round(box_hi, 2)}; '
      + ', '.join(f'{ob.name} {len(idx)} vertices' for ob, idx in meshes)
      + f', {len(rest_idx)} loose, {len(rig_out)} rig points; saved {blend}')
