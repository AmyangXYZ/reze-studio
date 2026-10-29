# -*- coding: utf-8 -*-
"""Put our head on another model's body, and change nothing else about that body.

This is the operation the model was made by in the first place -- a 塞尔凯特 body
wearing a 塞勒涅 face -- run again against a different 塞尔凯特. It is a far smaller
problem than moving an outfit across, and better conditioned: the host's skin, cloth,
shoes, physics and foot pose all stay exactly as its author left them, internally
consistent, so there is nothing to push out of anything. One seam instead of nine
garments.

It also settles the heels for nothing. 午夜沉沦 is authored standing in them -- her
ankle sits at y 1.54 against our 0.99 -- with the foot pose, the IK and the shoe all
agreeing already.

What has to be right is the head:

**The fit** is read off the two faces. Both are the same base head -- ours spans
x +-0.96 and hers spans x +-0.96 -- so a uniform scale and an offset carry one onto
the other, and ICP finds them without being told. The bones are not used for it: a
`頭` sits at whatever depth in a skull its author chose, and the two here differ.

**The sway** comes with the hair. Our 15 hair-chain bones are absent from the host and
carry 18% of the head's skin weight; resolved up the parent chain they would collapse
onto `頭` and the hair would go rigid. `chaingraft.py` brings the chains over whole --
bones, rigid bodies, and the joints between.

**The expressions** come with the face. All 111 of them, 98 vertex morphs of which
every one but `抗穿模` lives entirely on head vertices, so the head is appended as a
single block and the morphs are renumbered onto it. The host's own morphs are dropped:
they drive a face that is no longer here.

    python3 tools/zzz2pmx/headgraft.py --host <body.pmx> --head <ours.pmx> \\
        --out <out.pmx> [--drop mat,mat] [--take mat,mat]
"""
import argparse
import os
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import align
import chaingraft
from pmx import PMX
from pmxw import write_pmx

# ours, in the order they are laid down
TAKE = ['头发', 'face01', '齿', '口腔', '舌', '唇', 'eyebrow', 'eyelash',
        'eyelash_crease', '目白', '瞳1', '瞳2']
# the host's, replaced by them
DROP = ['face01', '齿', '口腔', '舌', '唇', 'eyebrow', 'eyelash', '目白',
        '右瞳', '左瞳', 'hair_F', 'hair_B', 'hair_accessory', 'earring']
FACE = 'face01'          # what the fit is read from, on both sides


def groups(m):
    F = np.array(m.faces).reshape(-1, 3)
    out, at = {}, 0
    for mat in m.materials:
        n = mat['fcount'] // 3
        out[mat['name']] = F[at:at + n]
        at += n
    return out


class Fit:
    """A uniform scale and an offset carrying our head onto theirs."""

    def __init__(self, head, host, scale=None, offset=None):
        gh, gs = groups(head), groups(host)
        A = np.array([v[0] for v in head.vertices])[np.unique(gh[FACE])]
        B = np.array([v[0] for v in host.vertices])[np.unique(gs[FACE])]
        if scale is not None and offset is not None:
            # Given, when the caller already knows it. An ICP reads the host's face
            # off whatever is in its `face01`, and a host that had its face cut back
            # out of a welded shell has a remeshed scalp in there too: 0.9385 and
            # 1.97 up, against the 1.0 and 0.883 the two faces actually differ by.
            self.scale, self.shift = float(scale), np.asarray(offset, float)
            d = np.linalg.norm(A * self.scale + self.shift
                               - B[cKDTree(B).query(A * self.scale + self.shift)[1]],
                               axis=1)
        elif scale is not None:
            # The size is given -- every reze head the same size, hers -- and only
            # where it sits is read off the host's face.
            self.scale, self.shift, d = align.icp_similarity(
                A, B, float(scale), B.mean(0) - A.mean(0) * float(scale), iters=80,
                fixed_scale=True)
        else:
            self.scale, self.shift, d = align.icp_similarity(
                A, B, 1.0, B.mean(0) - A.mean(0), iters=80)
        print(f'head fit: scale {self.scale:.4f}, offset {np.round(self.shift, 3)}, '
              f'face residual median {np.median(d):.4f} max {d.max():.4f}')

    def __call__(self, P):
        return np.atleast_2d(np.asarray(P, float)) * self.scale + self.shift


def textures_in(head, mats, host, head_dir, out_dir):
    """Copy the sheets our head needs and register them on the host."""
    gh = groups(head)
    want = {}
    for mat in head.materials:
        if mat['name'] in mats and 0 <= mat['tex'] < len(head.textures):
            want[mat['name']] = head.textures[mat['tex']]
    os.makedirs(out_dir, exist_ok=True)
    index = {}
    for name, rel in want.items():
        src = os.path.join(head_dir, rel.replace('\\', '/'))
        dst = os.path.join(out_dir, os.path.basename(rel.replace('\\', '/')))
        if os.path.exists(src):
            shutil.copyfile(src, dst)
        key = 'textures\\' + os.path.basename(rel.replace('\\', '/'))
        if key not in host.textures:
            host.textures.append(key)
        index[name] = host.textures.index(key)
    return index


def main(a):
    host, head = PMX(a.host), PMX(a.head)
    take = a.take.split(',') if a.take else TAKE
    drop = set(a.drop.split(',') if a.drop else DROP)
    fit = Fit(head, host, a.scale, a.offset and [float(x) for x in a.offset.split(',')])

    out_dir = os.path.dirname(a.out)
    os.makedirs(out_dir, exist_ok=True)
    src_dir = os.path.dirname(a.host)
    for entry in os.listdir(src_dir):
        s, d = os.path.join(src_dir, entry), os.path.join(out_dir, entry)
        if os.path.isdir(s) and not os.path.exists(d):
            shutil.copytree(s, d)
    tmp = os.path.join(out_dir, '_host.pmx')
    write_pmx(tmp, host)

    # -- the hair chains first, so the head's weights find them by name
    chaingraft.main(a.head, tmp, tmp, take, frame='髪', place=fit)
    host = PMX(tmp)
    os.remove(tmp)

    tex = textures_in(head, take, host, os.path.dirname(a.head),
                      os.path.join(out_dir, 'textures'))

    gh, gs = groups(head), groups(host)
    hb = {b['name']: i for i, b in enumerate(host.bones)}
    names = [b['name'] for b in head.bones]

    # -- the head, as one block, so the morphs can be renumbered onto it
    used = sorted(set(int(i) for m in take for i in np.unique(gh[m])))
    remap = {}
    Vh = np.array([v[0] for v in head.vertices])
    moved = fit(Vh[np.array(used)])
    lost = set()
    for k, i in enumerate(used):
        v = head.vertices[i]
        bones = []
        for b in v[5]:
            nm = names[b] if 0 <= b < len(names) else ''
            if nm not in hb:
                lost.add(nm)
            bones.append(hb.get(nm, hb.get('頭', 0)))
        w = np.array(v[6], float)
        tot = w[:len(v[5])].sum()
        if tot > 0:
            w[:len(v[5])] /= tot
        add = [[0.0] * 4 for _ in range(host.uv_add)]
        rec = (list(moved[k]), list(v[1]), list(v[2]), add, v[4], bones,
               [float(x) for x in w], v[7])
        remap[i] = len(host.vertices)
        host.vertices.append(rec + (v[8],) if v[4] == 3 else rec)
    if lost:
        print(f'  {len(lost)} bones had nowhere to go and fell back to 頭: '
              f'{sorted(lost)[:6]}')
    print(f'head: {len(used)} vertices placed on {len(take)} materials')

    # -- faces and materials. Above the jaw is our head's business now, and what
    # the host leaves up there was fitted to hair we removed: 午夜沉沦's ornament
    # rides a shared metal material that also holds pieces at her ankles, so it
    # survives the material drop and then cuts straight through our shorter hair.
    keep_mats, faces, at, trimmed = [], [], 0, 0
    F = np.array(host.faces).reshape(-1, 3)
    HV = np.array([v[0] for v in host.vertices])
    for mat in host.materials:
        n = mat['fcount'] // 3
        if mat['name'] not in drop:
            tris = F[at:at + n]
            if a.trim_above is not None:
                high = (HV[tris][:, :, 1] > a.trim_above).all(1)
                trimmed += int(high.sum())
                tris = tris[~high]
                mat['fcount'] = len(tris) * 3
            keep_mats.append(mat)
            faces.append(tris)
        at += n
    if trimmed:
        print(f'  trimmed {trimmed} of their triangles above y {a.trim_above}')
    gone = len(host.materials) - len(keep_mats)
    for name in take:
        mat = dict(next(m for m in head.materials if m['name'] == name))
        tris = np.vectorize(remap.get)(gh[name])
        mat['fcount'] = len(tris) * 3
        mat['tex'] = tex.get(name, -1)
        mat['sph'] = -1
        mat['sph_mode'] = 0
        mat['toon_shared'] = keep_mats[0]['toon_shared']
        mat['toon'] = keep_mats[0]['toon']
        keep_mats.append(mat)
        faces.append(tris)
    host.materials = keep_mats
    host.faces = np.concatenate(faces).ravel().tolist()
    print(f'materials: dropped {gone} of theirs, added {len(take)} of ours')

    # -- expressions. Theirs drive a face that is gone. Ours come in two passes: the
    # vertex and material morphs first, then the grp, numbered against the list
    # that is actually kept -- numbering them against the original list put every
    # group after the first emptied morph one slot off, and まばたき winked instead.
    mat_at = {m['name']: i for i, m in enumerate(host.materials)}
    morphs, dropped, grp = [], 0, []
    for mo in head.morphs:
        m = dict(mo)
        if m['type'] == 1:
            off = [(remap[o[0]], (np.asarray(o[1]) * fit.scale).tolist())
                   for o in m['offsets'] if o[0] in remap]
        elif m['type'] in (3, 4, 5, 6, 7):
            off = [(remap[o[0]],) + tuple(o[1:]) for o in m['offsets']
                   if o[0] in remap]
        elif m['type'] == 8:
            off = []
            for o in m['offsets']:
                if o[0] < 0:
                    off.append(o)
                    continue
                nm = head.materials[o[0]]['name'] if o[0] < len(head.materials) else ''
                if nm in mat_at:
                    off.append((mat_at[nm],) + tuple(o[1:]))
        elif m['type'] in (0, 9):
            grp.append(m)
            morphs.append(m)
            continue
        else:
            off = list(m['offsets'])
        if not off and m['offsets']:
            dropped += 1
            continue
        m['offsets'] = off
        morphs.append(m)
    # grp by name, against the kept list; a group left empty goes, and the rest
    # are numbered again without it
    wants = {m['name']: [(head.morphs[o[0]]['name'],) + tuple(o[1:]) for o in m['offsets']
                         if 0 <= o[0] < len(head.morphs)] for m in grp}
    while True:
        alive = {m['name']: i for i, m in enumerate(morphs)}
        for m in grp:
            m['offsets'] = [(alive[w[0]],) + tuple(w[1:]) for w in wants[m['name']] if w[0] in alive]
        empty = [m for m in grp if not m['offsets'] and m in morphs]
        if not empty:
            break
        morphs = [m for m in morphs if m not in empty]
        grp = [m for m in grp if m not in empty]
        dropped += len(empty)
    host.morphs = morphs
    print(f'expressions: {len(morphs)} of ours kept, {dropped} emptied; '
          f'theirs discarded')

    host.frames = [f for f in host.frames if f['name'] != '表情'] + [
        dict(name='表情', name_en='Exp', special=1,
             items=[(1, i) for i, _ in enumerate(morphs)])]

    write_pmx(a.out, host)
    m = PMX(a.out)
    print(f'{a.out}: {len(m.vertices)} verts, {len(m.faces)//3} tris, '
          f'{len(m.materials)} materials, {len(m.bones)} bones, '
          f'{len(m.morphs)} morphs, {len(m.rigids)} rigid bodies, '
          f'{len(m.joints)} joints')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--host', required=True)
    p.add_argument('--head', default='public/models/reze/reze.pmx')
    p.add_argument('--out', required=True)
    p.add_argument('--take')
    p.add_argument('--drop')
    p.add_argument('--scale', type=float, default=None)
    p.add_argument('--offset', default=None,
                   help='x,y,z -- with --scale, skips the ICP')
    p.add_argument('--trim-above', type=float, default=None,
                   dest='trim_above')
    main(p.parse_args())
