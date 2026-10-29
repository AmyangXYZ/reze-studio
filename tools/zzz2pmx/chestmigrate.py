# -*- coding: utf-8 -*-
"""Carry the slim model's chest onto reze.pmx.

The two skins are one base mesh on one atlas: matched by UV and side, a vertex of
one is a vertex of the other, and off the chest the bodies agree to 0.000. So every
reze skin vertex on the front of the torso that has a partner on the slim model
takes the partner's position, offset by the constant between the two frames. What
has no partner -- reze's skin is cut away under her cups -- interpolates from its
neighbours' moves. Her cups, shirt and ribbon ride the same field, as do her 胸 bones
and rigid bodies, and her sheet becomes the slim model's bake, which was made on this
exact chest.

    python3 tools/zzz2pmx/chestmigrate.py [--slim <pmx>] [--reze <pmx>]

reze.pmx is rewritten in place; the file as it was is kept beside it.
"""
import argparse
import os
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bustfit import window, mirror, smooth_normals, Field, ANCHOR, SKIN, BREAST as SLIM_BREAST
from graftlegs import material_slice
from pmx import PMX
from pmxw import write_pmx

SLIM = 'public/models/reze-midnight/reze-midnight-slim.pmx'
REZE = 'public/models/reze/reze.pmx'
SHEET = ('public/models/reze-midnight/textures/cloth01_reze_slim.png', 'cloth01_skin_ao.png')
RIDE = ['cloth01', 'cloth01.001', 'shirt', 'ribbon']      # reze's, over the chest
BONES = ['左胸上2', '右胸上2', '左胸変形', '右胸変形']


def main(a):
    S, R = PMX(a.slim), PMX(a.reze)
    VS = np.array([v[0] for v in S.vertices]); US = np.array([v[2] for v in S.vertices])
    VR = np.array([v[0] for v in R.vertices]); UR = np.array([v[2] for v in R.vertices])
    bs = {b['name']: np.array(b['pos']) for b in S.bones}
    br = {b['name']: np.array(b['pos']) for b in R.bones}
    ss = np.unique(material_slice(S, SKIN)[0]); rs = np.unique(material_slice(R, SKIN)[0])
    KS = np.column_stack([US[ss], np.sign(VS[ss, 0]) * 0.01]); KR = np.column_stack([UR[rs], np.sign(VR[rs, 0]) * 0.01])
    d, j = cKDTree(KS).query(KR)
    pair = np.where(d < 1e-4, ss[j], -1)
    win = window(VR[rs], br[ANCHOR])
    off = (pair >= 0) & (win < 1e-3)
    shift = np.median(VS[pair[off]] - VR[rs][off], axis=0)
    agree = np.median(np.linalg.norm(VS[pair[off]] - shift - VR[rs][off], axis=1))
    print(f'skins matched {int((pair >= 0).sum())} of {len(rs)}; frames offset {np.round(shift, 3)}, '
          f'bodies agree to {agree:.3f} off the chest')

    anchors, wa, ap = rs[win > 1e-3], win[win > 1e-3], pair[win > 1e-3]
    has = ap >= 0
    D = np.zeros((len(anchors), 3))
    D[has] = (VS[ap[has]] - shift) - VR[anchors][has]
    fill = Field(VR[anchors][has], D[has], 0.25)
    D[~has] = fill._raw(VR[anchors][~has])
    step = D * wa[:, None]
    step = 0.5 * (step + step[mirror(VR[anchors])] * np.array([-1.0, 1.0, 1.0]))
    before = VR.copy()
    VR[anchors] += step
    print(f'chest: {int(has.sum())} vertices take the slim model\'s, {int((~has).sum())} interpolate; '
          f'furthest move {np.linalg.norm(step, axis=1).max():.3f}')

    field = Field(before[anchors], step, 0.45)
    F = np.array(R.faces).reshape(-1, 3)
    at, moved = 0, {}
    for mat in R.materials:
        n = mat['fcount'] // 3; tris = F[at:at + n]; at += n
        if mat['name'] in RIDE:
            idx = np.unique(tris)
            VR[idx] += field(before[idx], br[ANCHOR])
            moved[mat['name']] = int((np.linalg.norm(VR[idx] - before[idx], axis=1) > 1e-4).sum())
    print('rode the field: ' + ', '.join(f'{k} {v}' for k, v in moved.items()))

    names = [b['name'] for b in R.bones]
    for b in R.bones:
        if b['name'] in BONES:
            b['pos'] = [float(x) for x in (np.array(b['pos']) + field(b['pos'], br[ANCHOR])[0])]
    nb = 0
    for r in R.rigids:
        if 0 <= r['bone'] < len(names) and names[r['bone']] in BONES:
            r['pos'] = [float(x) for x in (np.array(r['pos']) + field(r['pos'], br[ANCHOR])[0])]
            nb += 1
    print(f'  {len(BONES)} 胸 bones and {nb} rigid bodies carried with it')

    which = np.linalg.norm(VR - before, axis=1) > 1e-5
    for i in np.where(which)[0]:
        v = list(R.vertices[i]); v[0] = [float(x) for x in VR[i]]; R.vertices[i] = tuple(v)
    at = 0
    for mat in R.materials:
        n = mat['fcount'] // 3; tris = F[at:at + n]; at += n
        idx = np.unique(tris)
        if which[idx].any():
            N = smooth_normals(VR, tris)
            for i in idx:
                v = list(R.vertices[i]); v[1] = [float(x) for x in N[i]]; R.vertices[i] = tuple(v)

    src, name = SHEET
    shutil.copyfile(src, os.path.join(os.path.dirname(a.reze), 'textures', name))
    rel = 'textures\\' + name
    if rel not in R.textures:
        R.textures.append(rel)
    next(m for m in R.materials if m['name'] == SKIN)['tex'] = R.textures.index(rel)
    keep = os.path.join(os.path.dirname(a.reze), '_reze_before_migrate.pmx')
    shutil.copyfile(a.reze, keep)
    write_pmx(a.reze, R)
    print(f'wrote {a.reze}; skin reads {name}; the file as it was is at {keep}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--slim', default=SLIM)
    p.add_argument('--reze', default=REZE)
    main(p.parse_args())
