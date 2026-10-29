# -*- coding: utf-8 -*-
"""Rebuild a ZZZ Evelyn mod as an MMD model.

The mod ships the body and the hair. It leaves the game's own head in place and
only reskins the face, so the face plate comes from a PMX rip of the same
character, which also supplies the skeleton, the morphs and the physics.

The rip and the game asset share the head shell, and fitting one to the other
pins the game->MMD placement exactly; everything else follows from it.

    python3 tools/zzz2pmx/build.py --variant casual
    python3 tools/zzz2pmx/build.py --variant bomb
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import align
import bonemap
import zzzmod
from pmx import PMX
from pmxw import write_pmx

HOST = 'public/models/伊芙琳_by_绝区零/伊芙琳.pmx'
MOD = 'zzz-mod/EvelynMod'

# The face plate the mod leaves open, and everything that lives on it.
FACE_MATS = ['颜', '痣', '眉', '眉睫影', '白目', '目', '目光', '目光2', '睫', '口', '舌', '齿', '目影']
# Host surface the mod's weights are read off. Coat, hair, tie and weapon are
# Evelyn's own and would drag their physics bones onto Reze's skin.
DONOR_MATS = ['肌', '衣', '皮裤', '衬衣', '胸衣', '黑丝衣']
# Geometry the mod replaces; its physics goes with it.
DEAD_PHYSICS = ('外套', '发带', '发穗', '后发髻', '领带', '刘海', '侧发', '碎发', '发结',
                '刀', '耳坠', '背坠', '环珠', '腰环', '手套扣', '穗')

VARIANTS = {
    # The bomb form draws its own head, and the mod skips the face draw with it.
    'casual': dict(body='BodyA', tex='EvelynBodyADiffuse', hair=['HairA', 'HairB'],
                   face=True, name='Reze', name_en='Reze'),
    'bomb': dict(body='BodyB', tex='EvelynBodyBDiffuse', hair=[],
                 face=False, name='Reze (bomb)', name_en='Reze (bomb)'),
}

# Presentation for the mod's own materials, taken from the rip's nearest equivalent.
PRESENTATION = {'体': '衣', '髪1': '发', '髪2': '发'}


def material_groups(pmx):
    faces = np.array(pmx.faces).reshape(-1, 3)
    out, off = {}, 0
    for m in pmx.materials:
        n = m['fcount'] // 3
        out[m['name']] = faces[off:off + n]
        off += n
    return out


def fit_placement(host, hv, groups, body):
    """Pin game->MMD by fitting the mod's head shell onto the rip's own head skin.

    Both are the untouched game head, so the fit is exact and fixes the scale for
    the whole model. A bone-centroid match gives the starting scale. The head only
    survives in the jacket variant, so that is what the fit always reads, whichever
    variant is being built.
    """
    ib_body = zzzmod.ib(MOD, 'BodyA')
    donor = np.unique(np.concatenate([groups[k] for k in DONOR_MATS + ['颜']]))
    hc, hm = align.host_centroids(host, hv, donor)
    P0 = align.to_mmd(body.pos)
    va = np.unique(ib_body)
    mc, mmass = align.bone_centroids(P0, body.bones, body.weights, va)
    s0, t0, pairs, _ = align.match(mc, mmass, hc, hm, 13.6379, [0, 0.195, 0])

    skin = hv[np.unique(groups['肌'])]
    tgt = skin[skin[:, 1] > 18.5]
    src = P0[va][(P0[va] * s0 + t0)[:, 1] > (t0[1] + s0 * 1.52)]
    s, t, d = align.icp_similarity(src, tgt, s0, t0, iters=150)
    return s, np.array([0.0, t[1], t[2]]), np.median(d), pairs


def sample_alpha(path):
    im = Image.open(path)
    if im.mode != 'RGBA':
        return False
    a = np.asarray(im.convert('RGBA'))[:, :, 3]
    return bool((a < 250).mean() > 0.002)


def convert_texture(src, dst, flip=False):
    if os.path.exists(dst):
        return
    im = Image.open(src).convert('RGBA')
    if flip:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    a = np.asarray(im)[:, :, 3]
    if (a > 250).all():
        im = im.convert('RGB')
    im.save(dst)


def build(args):
    cfg = VARIANTS[args.variant]
    host = PMX(HOST)
    host_dir = os.path.dirname(HOST)
    hv = np.array([v[0] for v in host.vertices])
    groups = material_groups(host)
    comps = zzzmod.load(MOD)
    body = comps['Body']
    hair = comps.get('Hair')
    ib_body = zzzmod.ib(MOD, cfg['body'])

    s, t, resid, pairs = fit_placement(host, hv, groups, body)
    print(f'placement: scale {s:.4f} translate {np.round(t, 4)} head-shell residual {resid:.4f}')

    # ---- mod geometry, in MMD space -------------------------------------
    parts = []            # (name, positions, normals, uvs, faces, texture stem)
    P = align.to_mmd(body.pos) * s + t
    N = align.to_mmd(body.normal)
    parts.append(('体', P, N, body.uv, ib_body, cfg['tex']))
    if cfg['hair'] and hair is not None:
        HP = align.to_mmd(hair.pos) * s + t
        HN = align.to_mmd(hair.normal)
        for i, ibn in enumerate(cfg['hair']):
            parts.append((f'髪{i + 1}', HP, HN, hair.uv,
                          zzzmod.ib(MOD, ibn), 'EvelynHairADiffuse'))

    # ---- weights, carried over from the mod's own rig --------------------
    names = [b['name'] for b in host.bones]
    donor = np.unique(np.concatenate([groups[k] for k in DONOR_MATS + ['颜']]))
    hc, hm = align.host_centroids(host, hv, donor, min_mass=2.0)
    hc = {k: v for k, v in hc.items() if not any(p in names[k] for p in DEAD_PHYSICS)}
    hm = {k: v for k, v in hm.items() if k in hc}
    va = np.unique(ib_body)
    mc, mmass = align.bone_centroids(P, body.bones, body.weights, va, min_mass=2.0)
    mapping, unmatched, report = bonemap.build(mc, mmass, hc, hm, names)
    keys = sorted(mc)
    fb = bonemap.fallback(unmatched, mapping,
                          bonemap.weight_matrix(body.bones, body.weights, va, keys), keys)
    mapping.update(fb)
    worst = max(report, key=lambda r: r[2])
    print(f'  bone map: {len(report)} matched (worst {names[worst[1]]} at {worst[2]:.2f}), '
          f'{len(fb)} merged into a neighbour, {len(unmatched) - len(fb)} unresolved')

    head_idx = names.index('頭')

    new_verts, new_faces, materials, textures = [], [], [], []
    tex_dir = os.path.join(args.out_dir, 'tex')
    os.makedirs(tex_dir, exist_ok=True)

    def texture_slot(rel):
        if rel not in textures:
            textures.append(rel)
        return textures.index(rel)

    # face plate from the host, vertices first so their morphs stay addressable
    face_mats = FACE_MATS if cfg['face'] else []
    keep = (np.unique(np.concatenate([groups[k] for k in face_mats])) if face_mats
            else np.zeros(0, int))
    old2new = {int(o): i for i, o in enumerate(keep)}
    for o in keep:
        new_verts.append(host.vertices[int(o)])

    host_tex_used = {}
    for name in face_mats:
        m = dict(host.materials[[x['name'] for x in host.materials].index(name)])
        tris = groups[name]
        for tri in tris:
            new_faces.extend(old2new[int(i)] for i in tri)
        for key in ('tex', 'sph'):
            if m[key] >= 0:
                rel = host.textures[m[key]]
                if rel not in host_tex_used:
                    base = os.path.basename(rel.replace('\\', '/'))
                    if base == '颜.tga':
                        out = 'zzz_face.png'
                        convert_texture(os.path.join(MOD, 'EvelynFaceADiffuse.dds'),
                                        os.path.join(tex_dir, out), flip=True)
                    else:
                        out = base
                        src_path = os.path.join(host_dir, rel.replace('\\', '/'))
                        if not os.path.exists(os.path.join(tex_dir, out)):
                            Image.open(src_path).save(os.path.join(tex_dir, out))
                    host_tex_used[rel] = 'tex/' + out
                m[key] = texture_slot(host_tex_used[rel])
        if not m['toon_shared'] and m['toon'] >= 0:
            rel = host.textures[m['toon']]
            if rel not in host_tex_used:
                base = os.path.basename(rel.replace('\\', '/'))
                Image.open(os.path.join(host_dir, rel.replace('\\', '/'))).save(
                    os.path.join(tex_dir, base))
                host_tex_used[rel] = 'tex/' + base
            m['toon'] = texture_slot(host_tex_used[rel])
        m['fcount'] = len(tris) * 3
        materials.append(m)

    # mod parts
    host_names = [x['name'] for x in host.materials]
    for name, pos, nrm, uv, tris, stem in parts:
        ref = dict(host.materials[host_names.index(PRESENTATION[name])])
        used = np.unique(tris)
        remap = {int(o): len(new_verts) + i for i, o in enumerate(used)}
        if name.startswith('髪'):
            wts = [{head_idx: 1.0}] * len(used)
        else:
            wts = bonemap.remap(body.bones, body.weights, used, mapping)
        for k, o in enumerate(used):
            bs, ws = bonemap.top4(wts[k])
            new_verts.append((list(pos[o]), list(nrm[o] / (np.linalg.norm(nrm[o]) + 1e-9)),
                              list(uv[o]), [[0.0] * 4] * host.uv_add, 2, bs, ws, 1.0))
        for tri in tris:
            new_faces.extend(remap[int(i)] for i in tri)
        m = dict(ref)
        m['name'] = name
        m['name_en'] = name
        stem_png = stem + '.png'
        convert_texture(os.path.join(MOD, stem + '.dds'), os.path.join(tex_dir, stem_png))
        m['tex'] = texture_slot('tex/' + stem_png)
        m['sph'] = -1
        m['sph_mode'] = 0
        if not m['toon_shared'] and m['toon'] >= 0:
            rel = host.textures[m['toon']]
            if rel not in host_tex_used:
                base = os.path.basename(rel.replace('\\', '/'))
                Image.open(os.path.join(host_dir, rel.replace('\\', '/'))).save(
                    os.path.join(tex_dir, base))
                host_tex_used[rel] = 'tex/' + base
            m['toon'] = texture_slot(host_tex_used[rel])
        m['fcount'] = len(tris) * 3
        materials.append(m)

    out = Assembled(host, new_verts, new_faces, materials, textures, keep, old2new,
                    cfg, args)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, args.out or f'reze-{args.variant}.pmx')
    n = write_pmx(path, out)
    print(f'wrote {path}  {n/1e6:.2f} MB  {len(new_verts)} verts  '
          f'{len(new_faces)//3} tris  {len(materials)} materials  '
          f'{len(out.rigids)} rigid bodies')
    return path


class Assembled:
    """A PMX assembled from the host's rig and the mod's geometry."""

    def __init__(self, host, verts, faces, materials, textures, keep, old2new, cfg, args):
        self.version = 2.0
        self.uv_add = host.uv_add
        self.name = cfg['name']
        self.name_en = cfg['name_en']
        self.comment = f"{cfg['name']}\nZZZ mod geometry on the 伊芙琳 (by 绝区零) rig."
        self.comment_en = self.comment
        self.vertices = verts
        self.faces = faces
        self.materials = materials
        self.textures = textures
        self.bones = host.bones
        self.morphs, morph_map = remap_morphs(host, old2new, materials)
        self.frames = remap_frames(host, morph_map)
        self.rigids, self.joints = prune_physics(host)
        n, d = bonemap.retarget(self)
        print(f'  retargeted {n} bones onto the mod\'s own joints (median move {d:.2f})')
        prune_bones(self)


def remap_morphs(host, old2new, materials):
    """Carry the rip's morphs over, dropping what the mod's geometry left behind."""
    mat_names = [m['name'] for m in materials]
    mat_map = {i: mat_names.index(m['name']) for i, m in enumerate(host.materials)
               if m['name'] in mat_names}
    bone_names = {b['name'] for b in host.bones}
    keep_morph, out = {}, []
    for i, mo in enumerate(host.morphs):
        m = dict(mo)
        t = m['type']
        offs = []
        for o in m['offsets']:
            if t == 1 or t in (3, 4, 5, 6, 7):
                if o[0] in old2new:
                    offs.append((old2new[o[0]],) + tuple(o[1:]))
            elif t == 8:
                if o[0] < 0:
                    offs.append(o)
                elif o[0] in mat_map:
                    offs.append((mat_map[o[0]],) + tuple(o[1:]))
            elif t in (0, 9):
                offs.append(o)     # fixed up below
            else:
                offs.append(o)
        m['offsets'] = offs
        if not offs and t != 2:
            continue
        keep_morph[i] = len(out)
        out.append(m)
    for m in out:
        if m['type'] in (0, 9):
            m['offsets'] = [(keep_morph[o[0]], o[1]) for o in m['offsets'] if o[0] in keep_morph]
    return out, keep_morph


def remap_frames(host, morph_map):
    out = []
    for fr in host.frames:
        f = dict(fr)
        f['items'] = [(k, i if k == 0 else morph_map[i]) for k, i in fr['items']
                      if k == 0 or i in morph_map]
        out.append(f)
    return out


# Bones MMD drives directly rather than through weights, so nothing references them.
ALWAYS_KEEP = ('全ての親', '操作中心', 'センター', 'グルーブ', '腰', '両目', '両目先',
               '左ダミー', '右ダミー', '左手先', '右手先')


def prune_bones(model):
    """Drop the bones left behind by the geometry the mod replaced."""
    bones = model.bones
    n = len(bones)
    keep = set()
    for v in model.vertices:
        for b, w in zip(v[5], v[6]):
            if w > 0:
                keep.add(b)
    for r in model.rigids:
        if r['bone'] >= 0:
            keep.add(r['bone'])
    for i, b in enumerate(bones):
        if b['flag'] & 0x0020 or b['name'] in ALWAYS_KEEP or b['name'].endswith(('先', '指先')):
            keep.add(i)

    changed = True
    while changed:
        changed = False
        for i in sorted(keep):
            b = bones[i]
            refs = [b['parent']]
            if b['flag'] & 0x0001:
                refs.append(b['tail'])
            if b['flag'] & 0x0300 and 'grant' in b:
                refs.append(b['grant'][0])
            if b['flag'] & 0x0020:
                refs.append(b['ik_target'])
                refs.extend(l[0] for l in b['ik_links'])
            for r in refs:
                if 0 <= r < n and r not in keep:
                    keep.add(r)
                    changed = True

    order = sorted(keep)
    remap = {o: i for i, o in enumerate(order)}
    out = []
    for o in order:
        b = dict(bones[o])
        b['parent'] = remap.get(b['parent'], -1)
        if b['flag'] & 0x0001:
            b['tail'] = remap.get(b['tail'], -1)
        if b['flag'] & 0x0300 and 'grant' in b:
            src = remap.get(b['grant'][0], -1)
            if src < 0:
                b['flag'] &= ~0x0300
                b.pop('grant', None)
            else:
                b['grant'] = (src, b['grant'][1])
        if b['flag'] & 0x0020:
            b['ik_target'] = remap.get(b['ik_target'], -1)
            b['ik_links'] = [(remap[l[0]], l[1]) for l in b['ik_links'] if l[0] in remap]
        out.append(b)
    model.bones = out

    verts = []
    for v in model.vertices:
        v = list(v)
        v[5] = [remap.get(b, 0) for b in v[5]]
        verts.append(tuple(v))
    model.vertices = verts
    for r in model.rigids:
        r['bone'] = remap.get(r['bone'], -1)
    for mo in model.morphs:
        if mo['type'] == 2:
            mo['offsets'] = [(remap[o[0]],) + tuple(o[1:]) for o in mo['offsets'] if o[0] in remap]
    kept_morphs = {i: j for j, i in enumerate(
        i for i, mo in enumerate(model.morphs) if mo['offsets'])}
    model.morphs = [mo for mo in model.morphs if mo['offsets']]
    for mo in model.morphs:
        if mo['type'] in (0, 9):
            mo['offsets'] = [(kept_morphs[o[0]], o[1]) for o in mo['offsets']
                             if o[0] in kept_morphs]
    for fr in model.frames:
        fr['items'] = [(k, remap[i] if k == 0 else kept_morphs[i]) for k, i in fr['items']
                       if (i in remap if k == 0 else i in kept_morphs)]
    print(f'  bones {n} -> {len(out)}')


def prune_physics(host):
    names = [b['name'] for b in host.bones]
    keep, rmap = [], {}
    for i, r in enumerate(host.rigids):
        bone = names[r['bone']] if 0 <= r['bone'] < len(names) else ''
        if any(p in bone for p in DEAD_PHYSICS) or any(p in r['name'] for p in DEAD_PHYSICS):
            continue
        rmap[i] = len(keep)
        keep.append(r)
    joints = []
    for j in host.joints:
        if j['a'] in rmap and j['b'] in rmap:
            jj = dict(j)
            jj['a'] = rmap[j['a']]
            jj['b'] = rmap[j['b']]
            joints.append(jj)
    return keep, joints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=sorted(VARIANTS), default='casual')
    ap.add_argument('--out-dir', default='public/models/reze-zzz')
    ap.add_argument('--out', default=None)
    build(ap.parse_args())


if __name__ == '__main__':
    main()
