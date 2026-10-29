# -*- coding: utf-8 -*-
"""Put the mod's bomb form on the hand-built Reze model.

The bomb form is the mod's other body draw. It replaces the head outright, so the
face, the eyes, the mouth and the hair come off and the mask goes on in their place
-- which is what the model's own earlier bomb build did, and it means the 111 facial
morphs go with them. Everything below the neck stays the model's own.

    python3 tools/zzz2pmx/bombgraft.py
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import align
import garment
import surface
import zzzmod
from pmx import PMX
from pmxw import write_pmx

MOD = 'zzz-mod/EvelynMod'
ATLAS = 'EvelynBodyBDiffuse'
# The head is gone in this form, and so is everything that hangs off it.
# The head goes, and the bomb form wears its own skirt so the outfit's shorts go too.
DROP_MATS = ['face01', '齿', '口腔', '舌', '唇', 'eyebrow', 'eyelash', 'eyelash_crease',
             '目白', '瞳1', '瞳2', '头发', 'shorts']
BODY_MATS = ['skin', 'cloth01', 'cloth01.001', 'bozi']
# What each incoming piece rides, and whether it drapes or keeps its shape.
# The strip is taken as it stands in reze-zzz/reze-bomb -- fitting it slid it off the
# panel it belongs to. The gloves cannot be: left as they stand they cover her forearm
# where the reference covers a hand, and no choice of weights fixes geometry that is
# not over the bone driving it. They are fitted and then stretched to reach.
PIECES = {
    'head': dict(bone='頭', rigid=True, standoff=0.0),
    'pin': dict(bone='頭', rigid=True, standoff=0.0),
    'jacket': dict(donor=['shirt'] + BODY_MATS, rigid=False, standoff=0.09),
    'skirt': dict(donor=BODY_MATS, rigid=False, standoff=0.05),
    # Placed and then lengthened, nothing else. The body correction field is built from
    # nearest points on her torso and arms, and near the shoulder the nearest point is
    # the torso -- it pulls the top of the sleeve inward and splits the glove, which is
    # the bare upper arm between two dark cuffs. A sleeve wants its own axis, not a
    # field fitted to a body.
    'gloves': dict(donor=['skin'], rigid=True, standoff=0.0, stretch=True),
    'strip': dict(bone='下半身', rigid=True, standoff=0.0),
}


def material_groups(pmx):
    faces = np.array(pmx.faces).reshape(-1, 3)
    out, off = {}, 0
    for m in pmx.materials:
        n = m['fcount'] // 3
        out[m['name']] = faces[off:off + n]
        off += n
    return out


def pieces_from_mod(P, ib):
    found = garment.identify(P, ib, garment.BOMB_SIGNATURES)
    found['jacket'] = found.pop('top')
    found['strip'] = garment.collect(P, ib, lambda d: d['hw'] < 0.45 and d['z'] < -1.0)
    # The gloves are a slice off the arm shell, and cutting it at |X| > 2.6 threw the
    # upper arm away and left her shoulders bare. Cut by what the piece IS instead:
    # the glove is painted dark and the arm under it is not, so the diffuse separates
    # them where no plane through the arm can.
    shell = garment.collect(P, ib, lambda d: d['n'] > 8000)
    atlas = np.asarray(Image.open(os.path.join(MOD, ATLAS + '.dds')).convert('RGB'),
                       np.float32) / 255
    h, w, _ = atlas.shape
    uv = zzzmod.load(MOD)['Body'].uv
    lum = atlas[np.clip((np.clip(uv[:, 1], 0, 1) * (h - 1)).astype(int), 0, h - 1),
                np.clip((np.clip(uv[:, 0], 0, 1) * (w - 1)).astype(int), 0, w - 1)].mean(1)
    dark = lum[shell].mean(1) < 0.35
    arm = (np.abs(P[shell][:, :, 0]) > 1.5).all(1)
    found['gloves'] = shell[dark & arm]
    for k in ('strip', 'gloves'):
        print(f'  {k:8s} {len(found[k]):5d} tris  (gathered by rule)')
    return found


def build(args):
    host = PMX(args.target)
    groups = material_groups(host)
    HV = np.array([v[0] for v in host.vertices])
    body = zzzmod.load(MOD)['Body']
    P = align.to_mmd(body.pos) * 12.5055 + np.array([0.0, 0.3052, -0.0006])
    N = align.to_mmd(body.normal)
    print('segmenting the bomb form:')
    found = pieces_from_mod(P, zzzmod.ib(MOD, 'BodyB'))

    # place against the model's body, the same way the casual outfit does
    mod_body = garment.collect(P, zzzmod.ib(MOD, 'BodyB'), lambda d: d['n'] > 8000)
    mod_body = P[np.unique(mod_body)]
    # The similarity and the correction field want different samples, and conflating
    # them cost two rounds. Fit on the TORSO: the arms are what the two models agree
    # least about, and letting them into the fit dragged the mask 1.9 units up and off
    # centre. Then build the field over the WHOLE body, arms included, or the gloves
    # get no shape correction at all and land on the wrong part of the hand.
    all_host = HV[np.unique(np.concatenate([groups[k] for k in BODY_MATS]))]
    torso_mod = mod_body[(mod_body[:, 1] > 11.5) & (abs(mod_body[:, 0]) < 3.2)]
    torso_host = all_host[(all_host[:, 1] > 9.0) & (abs(all_host[:, 0]) < 3.2)]
    s, t, d = align.icp_similarity(torso_mod, torso_host, 1.0,
                                   torso_host.mean(0) - torso_mod.mean(0), iters=60)
    mod_body = mod_body[mod_body[:, 1] > 11.0]
    host_body = all_host[all_host[:, 1] > 9.0]
    tree = __import__('scipy.spatial', fromlist=['cKDTree']).cKDTree(host_body)
    src = mod_body * s + t
    dist, j = tree.query(src)
    keep = dist < np.percentile(dist, 95)
    D = np.zeros_like(src)
    D[keep] = host_body[j[keep]] - src[keep]
    print(f'placement: scale {s:.4f}, body residual {np.median(d):.3f}')

    bone_of = {b['name']: i for i, b in enumerate(host.bones)}
    collide = surface.Mesh(HV, np.concatenate([groups[k] for k in BODY_MATS]))
    placed = {}
    for name, cfg in PIECES.items():
        used = np.unique(found[name])
        moved = P[used] * s + t
        if not cfg['rigid']:
            moved = moved + _smooth(src, D, moved, 1.2)
            if cfg['standoff'] is None:
                print(f'  {name:8s} {len(used):5d} verts, warped onto the body')
            else:
                moved, lifted = surface.push_out(collide, moved, offset=cfg['standoff'])
                print(f'  {name:8s} {len(used):5d} verts, {lifted.sum()} lifted clear')
        else:
            print(f'  {name:8s} {len(used):5d} verts, placed rigidly')
        placed[name] = (used, moved)

    # The mask has to swallow the skull it replaces: Reze's head is not the head it
    # was moulded on. Grow it about the head bone -- not its own centroid, which is off
    # axis and drifts the mask sideways -- and solve the scale directly rather than
    # iterating, which ran away and swallowed the shoulders.
    head_idx = bone_of['頭']
    skull = np.array([HV[i] for i in np.unique(groups['skin'])
                      if max(zip(host.vertices[i][6], host.vertices[i][5]),
                             default=(0, -1))[1] == head_idx])
    used, moved = placed['head']
    if len(skull):
        skull = skull[skull[:, 1] > 18.3]
    if len(skull):
        c = np.array(host.bones[head_idx]['pos'])
        tri = np.vectorize({int(o): i for i, o in enumerate(used)}.get)(found['head'])
        mask = surface.Mesh(moved, tri)
        _, ti, ba = mask.closest(skull)
        f = mask.F[ti]
        q = (ba[:, 0:1] * mask.V[f[:, 0]] + ba[:, 1:2] * mask.V[f[:, 1]]
             + ba[:, 2:3] * mask.V[f[:, 2]])
        need = np.linalg.norm(skull - c, axis=1) / np.maximum(np.linalg.norm(q - c, axis=1), 1e-6)
        grow = float(np.clip(need.max() * 1.02, 1.0, 1.25))
        moved = c + (moved - c) * grow
        placed['head'] = (used, moved)
        print(f'  head grown {grow:.3f} about the head bone to clear {len(skull)} skull vertices')

    # Her arms are longer than the ones these gloves were cut for, and a nearest-point
    # field can only slide a garment, never lengthen one. Stretch each along its own arm
    # axis, anchored at the cuff -- anchoring at the shoulder drags the cuff outward and
    # opens a bare band across the upper arm.
    used, moved = placed['gloves']
    arm_skin = HV[np.unique(groups['skin'])]
    for sign in (-1, 1):
        side = (moved[:, 0] * sign) > 0.5
        if not side.any():
            continue
        shoulder = np.array(host.bones[bone_of['左腕' if sign > 0 else '右腕']]['pos'])
        wrist = np.array(host.bones[bone_of['左手首' if sign > 0 else '右手首']]['pos'])
        d_arm = wrist - shoulder
        d_arm /= np.linalg.norm(d_arm)
        hers = float(((arm_skin[(arm_skin[:, 0] * sign) > 2.0] - shoulder) @ d_arm).max())
        u = (moved[side] - shoulder) @ d_arm
        cuff, tip = float(u.min()), float(u.max())
        if tip - cuff <= 1e-6:
            continue
        k = (hers - cuff) / (tip - cuff)
        moved[side] += np.outer(cuff + (u - cuff) * k - u, d_arm)
        print(f'  gloves stretched {k:.3f} beyond the cuff along the '
              f'{"left" if sign > 0 else "right"} arm')
    placed['gloves'] = (used, moved)

    src_w = [dict((b, w) for b, w in zip(v[5], v[6]) if w > 1e-4) for v in host.vertices]

    drop_tris = [groups[k] for k in DROP_MATS if k in groups]
    drop = set(np.unique(np.concatenate(drop_tris)).tolist()) if drop_tris else set()
    keep_old = [i for i in range(len(host.vertices)) if i not in drop]
    old2new = {o: i for i, o in enumerate(keep_old)}
    verts = [host.vertices[i] for i in keep_old]

    tex_dir = os.path.join(os.path.dirname(args.target), 'textures')
    rel = 'textures\\zzz_' + ATLAS + '.png'
    path = os.path.join(tex_dir, 'zzz_' + ATLAS + '.png')
    if not os.path.exists(path):
        Image.open(os.path.join(MOD, ATLAS + '.dds')).convert('RGB').save(path)
    textures = list(host.textures)
    if rel not in textures:
        textures.append(rel)

    new_faces = {}
    for name, cfg in PIECES.items():
        used, moved = placed[name]
        if 'bone' in cfg:
            b = bone_of[cfg['bone']]
            wts = [{b: 1.0}] * len(used)
        else:
            donor = surface.Mesh(HV, np.concatenate([groups[k] for k in cfg['donor']]))
            wts, _ = surface.transfer_weights(donor, src_w, moved)
        base = len(verts)
        remap = {int(o): base + i for i, o in enumerate(used)}
        for k, o in enumerate(used):
            bs, ws = _top4(wts[k])
            n = N[o] / (np.linalg.norm(N[o]) + 1e-9)
            verts.append((list(moved[k]), list(n), list(body.uv[o]),
                          [[0.0] * 4] * host.uv_add, 2, bs, ws, 1.0))
        new_faces[name] = np.vectorize(remap.get)(found[name])

    ref = {m['name']: m for m in host.materials}
    materials, faces = [], []
    for m in host.materials:
        if m['name'] in DROP_MATS:
            continue
        mm = dict(m)
        f = np.vectorize(old2new.get)(groups[m['name']])
        mm['fcount'] = len(f) * 3
        faces.append(f)
        materials.append(mm)
    for name in PIECES:
        em = dict(ref['shirt'] if name in ('jacket', 'gloves') else ref['skin'])
        em['name'] = em['name_en'] = 'bomb_' + name
        em['tex'] = textures.index(rel)
        em['sph'] = -1
        em['sph_mode'] = 0
        f = new_faces[name]
        em['fcount'] = len(f) * 3
        faces.append(f)
        materials.append(em)

    out = Assembled(host, verts, np.concatenate(faces).ravel().tolist(), materials,
                    textures, old2new)
    n = write_pmx(args.out, out)
    print(f'wrote {args.out}  {n/1e6:.2f} MB  {len(verts)} verts  {len(out.faces)//3} tris  '
          f'{len(materials)} materials  {len(out.morphs)} morphs (the facial ones went '
          f'with the face)')


def _smooth(anchors, values, points, radius, k=24):
    from scipy.spatial import cKDTree
    tree = cKDTree(anchors)
    d, j = tree.query(points, k=min(k, len(anchors)))
    w = np.exp(-(d / radius) ** 2)
    w /= w.sum(1, keepdims=True) + 1e-12
    return (values[j] * w[..., None]).sum(1)


def _top4(d):
    items = sorted(d.items(), key=lambda kv: -kv[1])[:4]
    tot = sum(w for _, w in items)
    if tot <= 0:
        return [0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]
    bones = [b for b, _ in items] + [0] * (4 - len(items))
    ws = [w / tot for _, w in items] + [0.0] * (4 - len(items))
    return bones, ws


class Assembled:
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
        mat_names = [m['name'] for m in materials]
        mat_map = {i: mat_names.index(m['name']) for i, m in enumerate(host.materials)
                   if m['name'] in mat_names}
        morphs, keep = [], {}
        for i, mo in enumerate(host.morphs):
            m = dict(mo)
            if m['type'] in (1, 3, 4, 5, 6, 7):
                m['offsets'] = [(old2new[o[0]],) + tuple(o[1:]) for o in m['offsets']
                                if o[0] in old2new]
            elif m['type'] == 8:
                m['offsets'] = [o if o[0] < 0 else (mat_map[o[0]],) + tuple(o[1:])
                                for o in m['offsets'] if o[0] < 0 or o[0] in mat_map]
            if not m['offsets'] and m['type'] != 2:
                continue
            keep[i] = len(morphs)
            morphs.append(m)
        for m in morphs:
            if m['type'] in (0, 9):
                m['offsets'] = [(keep[o[0]], o[1]) for o in m['offsets'] if o[0] in keep]
        self.morphs = morphs
        self.frames = [dict(f, items=[(k, i) for k, i in f['items']
                                      if k == 0 or i in keep]) for f in host.frames]
        for f in self.frames:
            f['items'] = [(k, i if k == 0 else keep[i]) for k, i in f['items']]
        self.rigids = host.rigids
        self.joints = host.joints


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='public/models/reze/reze-zzz.pmx')
    ap.add_argument('--out', default='public/models/reze/reze-bomb.pmx')
    build(ap.parse_args())
