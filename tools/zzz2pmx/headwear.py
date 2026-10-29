# -*- coding: utf-8 -*-
"""Move what a model wears on its head -- ears, a band, a hat -- with the bones,
bodies and joints only those materials use, by one offset, in place.

    python3 tools/zzz2pmx/headwear.py <model.pmx> <mat,mat> <dx,dy,dz>

For a host's headwear left where the host's hair crown was, after reze's smaller
head went on: the offset is reze's hair crown minus the host's.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holefill
from pmx import PMX
from pmxw import write_pmx


def crown(m, hair='头发'):
    V = np.array([v[0] for v in m.vertices])
    P = V[np.unique(holefill.tris_of(m, [hair]))]
    top = P[:, 1].max()
    c = P[P[:, 1] > top - 1.0]
    return np.array([c[:, 0].mean(), top, c[:, 2].mean()])


def move(path, mats, shift, size=1.0, about=(0.0, 0.0, 0.0)):
    """Moved by `shift`; with `size`, also scaled about `about` (a point before the move),
    for headwear sized to a host's larger head."""
    m = PMX(path)
    sh = np.array(shift, float)
    c = np.array(about, float)

    def pt(p):
        return (c + (np.array(p, float) - c) * size + sh).tolist()
    sl = holefill.slices(m)
    ids = np.unique(np.concatenate([np.unique(sl[n]) for n in mats]))
    usedby = {}
    for k in sl:
        for i in np.unique(sl[k]):
            for b, w in zip(m.vertices[int(i)][5], m.vertices[int(i)][6]):
                if w > 1e-6:
                    usedby.setdefault(int(b), set()).add(k)
    own = {b for b, ks in usedby.items() if ks <= set(mats)}
    for i in ids:
        v = list(m.vertices[int(i)])
        v[0] = pt(v[0])
        m.vertices[int(i)] = tuple(v)
    for b in own:
        m.bones[b]['pos'] = pt(m.bones[b]['pos'])
        if 'tail_offset' in m.bones[b]:
            m.bones[b]['tail_offset'] = (np.array(m.bones[b]['tail_offset']) * size).tolist()
    rbs = set()
    for k, rb in enumerate(m.rigids):
        if rb['bone'] in own:
            rb['pos'] = pt(rb['pos'])
            rb['size'] = (np.array(rb['size']) * size).tolist()
            rbs.add(k)
    for j in m.joints:
        if j['a'] in rbs or j['b'] in rbs:
            j['pos'] = pt(j['pos'])
            j['lin_min'] = (np.array(j['lin_min']) * size).tolist()
            j['lin_max'] = (np.array(j['lin_max']) * size).tolist()
    write_pmx(path, m)
    print(f'{path}: {", ".join(mats)} x{size:.3f}, moved by {np.round(sh, 3)} -- {len(ids)} vertices, '
          f'{len(own)} bones, {len(rbs)} rigid bodies')


if __name__ == '__main__':
    move(sys.argv[1], sys.argv[2].split(','), [float(x) for x in sys.argv[3].split(',')])
