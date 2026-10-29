# -*- coding: utf-8 -*-
"""Bring a grafted model's reze head to reze's own size, in place.

    python3 tools/zzz2pmx/headsize.py <model.pmx> [...]

The graft used to scale the head to each host's face, so the heads ran 0.995 to
1.077 of reze's; the graft now keeps them at 1.0, and a model whose host is gone
gets the same here: the head's materials, its hair bones, their bodies and joints,
and its expressions scaled about the face's centre, which is where the graft's fit
put it.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holefill
import prune
from headswap import HEAD
from pmx import PMX
from pmxw import write_pmx

REZE = 'public/models/reze/reze.pmx'
REZE_HEAD = ['头发', 'face01', '齿', '口腔', '舌', '唇', 'eyebrow', 'eyelash', 'eyelash_crease', '目白', '瞳1', '瞳2']


def face_width(m):
    V = np.array([v[0] for v in m.vertices])
    return float(np.ptp(V[np.unique(holefill.tris_of(m, ['face01']))][:, 0]))


def resize(path, reze=REZE):
    m, r = PMX(path), PMX(reze)
    s = face_width(m) / face_width(r)
    if abs(s - 1) < 1e-4:
        print(f'{path}: head already x1.0000')
        return s
    f = 1.0 / s
    V = np.array([v[0] for v in m.vertices])
    sl = holefill.slices(m)
    ids = np.unique(np.concatenate([np.unique(sl[n]) for n in REZE_HEAD if n in sl]))
    c = V[np.unique(sl['face01'])].mean(0)
    on = np.zeros(len(V), bool)
    on[ids] = True
    # her hair bones: weighted by head vertices, not part of the standard skeleton
    names = [b['name'] for b in m.bones]
    hair = set()
    for i in ids:
        v = m.vertices[int(i)]
        for b, w in zip(v[5], v[6]):
            if w > 1e-6 and names[b] not in prune.KEEP:
                hair.add(int(b))
    for i in ids:
        v = list(m.vertices[int(i)])
        v[0] = (c + f * (V[i] - c)).tolist()
        m.vertices[int(i)] = tuple(v)
    for b in hair:
        bone = m.bones[b]
        bone['pos'] = (c + f * (np.array(bone['pos']) - c)).tolist()
        if 'tail_offset' in bone:
            bone['tail_offset'] = (np.array(bone['tail_offset']) * f).tolist()
    rbs = set()
    for k, rb in enumerate(m.rigids):
        if rb['bone'] in hair:
            rb['pos'] = (c + f * (np.array(rb['pos']) - c)).tolist()
            rb['size'] = (np.array(rb['size']) * f).tolist()
            rbs.add(k)
    for j in m.joints:
        if j['a'] in rbs or j['b'] in rbs:
            j['pos'] = (c + f * (np.array(j['pos']) - c)).tolist()
    for mo in m.morphs:
        if mo['type'] == 1:
            mo['offsets'] = [(o[0], (np.array(o[1]) * f).tolist() if on[o[0]] else o[1]) for o in mo['offsets']]
    write_pmx(path, m)
    print(f'{path}: head was x{s:.4f}, now x1.0000 -- {len(ids)} vertices, {len(hair)} hair bones, '
          f'{len(rbs)} rigid bodies scaled about the face centre')
    return s


if __name__ == '__main__':
    for p in sys.argv[1:]:
        resize(p)
