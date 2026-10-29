# -*- coding: utf-8 -*-
"""Put the ZZZ mod's outfit on the hand-built Reze model.

The hand-built model is a 塞尔凯特2 body wearing a 塞勒涅 face, and its shirt is
20224 triangles of subdivided mesh over a photographic grey swatch. The mod's is
1940 triangles with the collar, placket and seams painted where the geometry
actually has them. The two bodies agree to a median 0.36 across the torso and hips,
so the mod's garment fits with only a low-frequency correction for the shape
difference — and none of the model's 111 morphs touch the garments it replaces.

    python3 tools/zzz2pmx/graft.py
"""
import argparse
import os
import shutil
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import align
import presentation
import garment
import surface
import zzzmod
from pmx import PMX
from pmxw import write_pmx

MOD = 'zzz-mod/EvelynMod'
ATLAS = 'EvelynBodyADiffuse'

# What each incoming garment replaces, and whose material presentation it takes.
PLAN = [
    ('shirt', 'shirt'),
    ('shorts', 'shorts'),
    ('ribbon', None),
    ('choker', None),
]
# The model's own body surface, used both to place the garments and to weight them.
BODY_MATS = ['skin', 'cloth01', 'cloth01.001', 'bozi']
# How far each piece has to clear that surface. The choker and ribbon sit on the
# neck rather than over it, so they get almost nothing.
STANDOFF = {'shirt': 0.07, 'shorts': 0.07, 'ribbon': 0.02, 'choker': 0.03}
# Cloth drapes to the body it is on; a buckle keeps its shape and moves as one.
RIGID = {'choker', 'ribbon'}
class _MatHolder:
    def __init__(self, materials):
        self.materials = materials


def material_groups(pmx):
    faces = np.array(pmx.faces).reshape(-1, 3)
    out, off = {}, 0
    for m in pmx.materials:
        n = m['fcount'] // 3
        out[m['name']] = faces[off:off + n]
        off += n
    return out


def smooth_field(anchors, values, points, radius, k=24):
    """Sample a displacement field defined on `anchors` at `points`, smoothly."""
    tree = cKDTree(anchors)
    k = min(k, len(anchors))
    d, j = tree.query(points, k=k)
    w = np.exp(-(d / radius) ** 2)
    w /= w.sum(1, keepdims=True) + 1e-12
    return (values[j] * w[..., None]).sum(1)


def fit_and_warp(gverts, mod_body, host_body, radius, verbose=''):
    """Place a garment on the model's body, keeping its standoff from the skin.

    The correction is read off the two bodies rather than off the two garments: the
    old shirt is cut differently, and warping onto it would drag the mod's collar
    down to where the frill used to be.
    """
    s, t, d = align.icp_similarity(mod_body, host_body, 1.0,
                                   host_body.mean(0) - mod_body.mean(0), iters=60)
    src = mod_body * s + t
    tree = cKDTree(host_body)
    dist, j = tree.query(src)
    keep = dist < np.percentile(dist, 95)
    D = np.zeros_like(src)
    D[keep] = host_body[j[keep]] - src[keep]
    moved = gverts * s + t
    moved = moved + smooth_field(src, D, moved, radius)
    if verbose:
        print(f'  {verbose}: body fit scale {s:.4f}, residual median {np.median(d):.3f}; '
              f'shape correction median {np.linalg.norm(D[keep], axis=1).mean():.3f}')
    return moved, s, t


def build(args):
    host = PMX(args.target)
    host_dir = os.path.dirname(args.target)
    groups = material_groups(host)
    HV = np.array([v[0] for v in host.vertices])

    body = zzzmod.load(MOD)['Body']
    P = align.to_mmd(body.pos) * 12.5055 + np.array([0.0, 0.3052, -0.0006])
    print('segmenting the mod outfit:')
    pieces = garment.identify(P, zzzmod.ib(MOD, 'BodyA'))

    # the two bodies, as surfaces
    mod_body_tris = garment.identify(
        P, zzzmod.ib(MOD, 'BodyA'),
        {'torso': dict(y=(11.0, 19.0), halfwidth=(5.0, 7.0), min_verts=5000)},
        verbose=False)['torso']
    mod_body = P[np.unique(mod_body_tris)]
    mod_body = mod_body[(mod_body[:, 1] > 11.5) & (abs(mod_body[:, 0]) < 3.2)]
    host_body_tris = np.concatenate([groups[k] for k in BODY_MATS])
    host_body = HV[np.unique(host_body_tris)]
    host_body = host_body[(host_body[:, 1] > 9.0) & (abs(host_body[:, 0]) < 3.2)]

    print('placing:')
    collide = surface.Mesh(HV, np.concatenate([groups[k] for k in BODY_MATS]))
    placed = {}
    for name, _ in PLAN:
        used = np.unique(pieces[name])
        moved, _, _ = fit_and_warp(P[used], mod_body, host_body, radius=1.2, verbose=name)
        moved, lifted = surface.push_out(collide, moved, offset=STANDOFF[name],
                                         rigid=name in RIGID)
        placed[name] = (used, moved)
        print(f'  {name:8s} lifted {lifted.sum()} of {len(used)} vertices clear of the body')

    # weights from the model's own clothed surface: the new garment lands within a
    # third of a unit of the old one, so the nearest source is the right source
    donor_tris = np.concatenate([groups[k] for k in BODY_MATS + ['shirt', 'shorts']])
    donor = surface.Mesh(HV, donor_tris)
    src_w = [dict((b, w) for b, w in zip(v[5], v[6]) if w > 1e-4) for v in host.vertices]

    # weights from the model's own clothed surface: the new garment lands within a
    # third of a unit of the old one, so the nearest source is the right source
    donor_tris = np.concatenate([groups[k] for k in BODY_MATS + ['shirt', 'shorts']])
    donor = surface.Mesh(HV, donor_tris)
    src_w = [dict((b, w) for b, w in zip(v[5], v[6]) if w > 1e-4) for v in host.vertices]

    # ---- assemble ------------------------------------------------------
    drop = set(np.unique(np.concatenate([groups['shirt'], groups['shorts']])).tolist())
    keep_old = [i for i in range(len(host.vertices)) if i not in drop]
    old2new = {o: i for i, o in enumerate(keep_old)}
    for mo in host.morphs:
        if mo['type'] in (1, 3, 4, 5, 6, 7):
            lost = [o for o in mo['offsets'] if o[0] in drop]
            if lost:
                raise SystemExit(f'graft: morph {mo["name"]} drives {len(lost)} of the '
                                 'vertices being replaced')
    verts = [host.vertices[i] for i in keep_old]

    tex_dir = os.path.join(host_dir, 'textures')
    atlas_rel = f'textures\\zzz_{ATLAS}.png'
    atlas_path = os.path.join(tex_dir, f'zzz_{ATLAS}.png')
    if not os.path.exists(atlas_path):
        im = Image.open(os.path.join(MOD, ATLAS + '.dds')).convert('RGBA')
        a = np.asarray(im)[:, :, 3]
        im = im.convert('RGB') if (a > 250).all() else im
        im.save(atlas_path)
    textures = list(host.textures)
    if atlas_rel not in textures:
        textures.append(atlas_rel)
    atlas_idx = textures.index(atlas_rel)

    new_faces_by_mat = {}
    for name, _ in PLAN:
        used, moved = placed[name]
        wts, dist = surface.transfer_weights(donor, src_w, moved)
        base = len(verts)
        remap = {int(o): base + i for i, o in enumerate(used)}
        nrm = align.to_mmd(body.normal)
        for k, o in enumerate(used):
            bs, ws = top4(wts[k])
            n = nrm[o] / (np.linalg.norm(nrm[o]) + 1e-9)
            verts.append((list(moved[k]), list(n), list(body.uv[o]),
                          [[0.0] * 4] * host.uv_add, 2, bs, ws, 1.0))
        new_faces_by_mat[name] = np.vectorize(remap.get)(pieces[name])
        print(f'  {name:8s} {len(used):5d} verts, donor distance median {np.median(dist):.3f}')

    materials, faces = [], []
    ref = {m['name']: m for m in host.materials}
    for m in host.materials:
        name = m['name']
        mm = dict(m)
        if name == 'shirt':
            tris = new_faces_by_mat['shirt']
            mm['tex'] = atlas_idx
            mm['fcount'] = len(tris) * 3
            faces.append(tris)
            materials.append(mm)
            for extra, presenter in (('ribbon', 'shorts'), ('choker', 'shorts')):
                em = dict(ref[presenter])
                em['name'] = em['name_en'] = extra
                em['tex'] = atlas_idx
                em['sph'] = -1
                em['sph_mode'] = 0
                t2 = new_faces_by_mat[extra]
                em['fcount'] = len(t2) * 3
                faces.append(t2)
                materials.append(em)
        elif name == 'shorts':
            tris = new_faces_by_mat['shorts']
            mm['tex'] = atlas_idx
            mm['fcount'] = len(tris) * 3
            faces.append(tris)
            materials.append(mm)
        else:
            tris = np.vectorize(old2new.get)(groups[name])
            mm['fcount'] = len(tris) * 3
            faces.append(tris)
            materials.append(mm)

    out = Grafted(host, verts, np.concatenate(faces).ravel().tolist(), materials,
                  textures, old2new)
    n = write_pmx(args.out, out)
    print(f'wrote {args.out}  {n/1e6:.2f} MB  {len(verts)} verts  '
          f'{len(out.faces)//3} tris  {len(materials)} materials')


def top4(d):
    items = sorted(d.items(), key=lambda kv: -kv[1])[:4]
    tot = sum(w for _, w in items)
    if tot <= 0:
        return [0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]
    bones = [b for b, _ in items]
    ws = [w / tot for _, w in items]
    while len(bones) < 4:
        bones.append(0)
        ws.append(0.0)
    return bones, ws


class Grafted:
    def __init__(self, host, verts, faces, materials, textures, old2new):
        self.version = 2.0
        self.uv_add = host.uv_add
        self.name = host.name
        self.name_en = host.name_en
        self.comment = host.comment
        self.comment_en = host.comment_en
        self.vertices = verts
        self.faces = faces
        self.materials = materials
        self.textures = textures
        self.bones = host.bones
        self.morphs = [remap_morph(m, old2new) for m in host.morphs]
        self.frames = host.frames
        self.rigids = host.rigids
        self.joints = host.joints


def remap_morph(mo, old2new):
    m = dict(mo)
    if m['type'] in (1, 3, 4, 5, 6, 7):
        m['offsets'] = [(old2new[o[0]],) + tuple(o[1:]) for o in m['offsets'] if o[0] in old2new]
    return m


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='public/models/reze/reze-base.pmx')
    ap.add_argument('--out', default='public/models/reze/reze-zzz.pmx')
    build(ap.parse_args())
