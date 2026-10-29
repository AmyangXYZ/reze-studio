# -*- coding: utf-8 -*-
"""Take the base mannequin out to 塞勒涅's own silhouette.

`base_selene` is built on her 皮肤 and lands on it exactly -- thigh, knee, calf and
seat all match to 0.000 and the waist to 0.011. Two places it does not:

1103 of its 8360 body vertices sit inside her skin, by 0.011 at the median and 0.158
at the worst -- the smoothing that closed the seams took them in. Moved back out, the
bust, seat, waist, thigh, knee and calf all land on her to 0.000.

The ankle does not, and is left alone. Her boots cut hers away and what survives is a
65-vertex fragment; against it the base is not small but differently shaped, 0.036
narrower across and 0.150 deeper front to back, with 130 of its 174 vertices already
outside her. Snapping to a fragment that thin moved the silhouette further off, not
closer.

`胸口` plays no part in it. It is a garment over the breast, not the breast, so the
body it covers is the body.

Both are fixed by moving vertices, never by adding any: the body is one watertight
shell with no boundary and no non-manifold edge, and it stays that way. Each vertex
goes to the outermost surface she has there, and only outward -- pulling one in where
she happens to be thinner would undo geometry the base got right. The step is blurred
over the surface before it is applied, so the shell's rim arrives as a swell rather
than a ledge.

    python3 tools/zzz2pmx/selenebase.py

Reads `public/models/base_selene/Selene_base.pmx` and never writes it. Output is
`Selene_base_fitted.pmx` beside it.
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holefill
from pmx import PMX
from pmxw import write_pmx
from surface import Mesh

BASE = 'public/models/base_selene/Selene_base.pmx'
HER = 'public/models/塞勒涅/塞勒涅.pmx'
OUT = 'public/models/base_selene/Selene_base_fitted.pmx'

BODY = 'Body_Mannequin'
HERS = ['皮肤']            # her body. 胸口 is a garment over it, not part of it
REACH = 0.35               # further from her surface than this and it is not the same place
AGREE = 0.5                # and it has to be facing the same way
ROUNDS = 6                 # blur passes on the step, so a seam is not a ledge



def blur(D, tris, index, rounds=ROUNDS):
    """Average a per-vertex quantity over the surface it lives on."""
    nbr = defaultdict(set)
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            nbr[int(u)].add(int(v))
            nbr[int(v)].add(int(u))
    order = [np.array([index[j] for j in nbr.get(i, ()) if j in index], int)
             for i, _ in sorted(index.items(), key=lambda x: x[1])]
    for _ in range(rounds):
        out = D.copy()
        for k, ns in enumerate(order):
            if len(ns):
                out[k] = 0.5 * D[k] + 0.5 * D[ns].mean(0)
        D = out
    return D


def hug(path=BASE, her=HER, out=OUT):
    m = PMX(path)
    s = PMX(her)
    V = np.array([v[0] for v in m.vertices])
    N = np.array([v[1] for v in m.vertices])
    SV = np.array([v[0] for v in s.vertices])
    sl = holefill.slices(s)
    tris = holefill.slices(m)[BODY]
    idx = np.unique(tris)

    want = np.zeros((len(idx), 3))
    for name in HERS:
        if name not in sl:
            continue
        mesh = Mesh(SV, sl[name])
        d, t, bar = mesh.closest(V[idx])
        f = mesh.F[t]
        target = (SV[f[:, 0]] * bar[:, :1] + SV[f[:, 1]] * bar[:, 1:2]
                  + SV[f[:, 2]] * bar[:, 2:3])
        step = target - V[idx]
        out_by = np.einsum('ij,ij->i', step, N[idx])
        ok = ((d < REACH)
              & (np.einsum('ij,ij->i', N[idx], mesh.normal[t]) > AGREE)
              & (out_by > 0))
        gain = np.where(ok, out_by, 0.0)[:, None] * N[idx]
        take = np.linalg.norm(gain, axis=1) > np.linalg.norm(want, axis=1)
        want[take] = gain[take]
        print(f'  {name}: {int(ok.sum())} of {len(idx)} body vertices sit inside it, '
              f'by median {np.median(out_by[ok]) if ok.any() else 0:.4f} '
              f'max {out_by[ok].max() if ok.any() else 0:.4f}')

    index = {int(i): k for k, i in enumerate(idx)}
    want = blur(want, tris, index)
    V[idx] += want
    moved = np.linalg.norm(want, axis=1)
    print(f'  moved {int((moved > 1e-4).sum())} vertices outward, median '
          f'{np.median(moved[moved > 1e-4]):.4f} max {moved.max():.4f}')

    for i, k in index.items():
        v = list(m.vertices[i])
        v[0] = [float(x) for x in V[i]]
        m.vertices[i] = tuple(v)
    print(f'  rebuilt {holefill.renormal(m, tris, idx)} normals')
    write_pmx(out, m)
    return out


def check(path, her=HER):
    """Silhouette against hers, at the heights a dress is cut to."""
    m, s = PMX(path), PMX(her)
    V = np.array([v[0] for v in m.vertices])
    SV = np.array([v[0] for v in s.vertices])
    sl = holefill.slices(s)
    P = V[np.unique(holefill.slices(m)[BODY])]
    S = np.concatenate([SV[np.unique(sl[k])] for k in HERS if k in sl])
    tris = holefill.slices(m)[BODY]
    w = holefill.weld(V)
    e = defaultdict(int)
    for a, b, c in tris:
        for p, q in ((a, b), (b, c), (c, a)):
            if w[p] != w[q]:
                e[(min(w[p], w[q]), max(w[p], w[q]))] += 1
    print(f'  watertight: {sum(1 for v in e.values() if v == 1)} boundary edges, '
          f'{sum(1 for v in e.values() if v > 2)} non-manifold')
    rows = [('bust', 16.2, 17.4, 'body'), ('waist', 13.8, 14.4, 'body'),
            ('hip/seat', 11.6, 12.4, 'body'), ('thigh', 8.6, 9.4, 'leg'),
            ('knee', 6.0, 6.6, 'leg'), ('calf', 4.6, 5.4, 'leg'),
            ('ankle', 1.6, 2.2, 'leg')]
    for tag, lo, hi, side in rows:
        def g(Q):
            q = Q[(Q[:, 1] > lo) & (Q[:, 1] < hi)]
            q = q[(q[:, 0] > 0.15) & (q[:, 0] < 2.5)] if side == 'leg' \
                else q[np.abs(q[:, 0]) < 2.6]
            return (q[:, 0].max() - q[:, 0].min(), q[:, 2].min()) if len(q) > 8 else None
        a, b = g(P), g(S)
        if a and b:
            print(f'  {tag:9s} width {a[0]:.3f} vs hers {b[0]:.3f} ({a[0] - b[0]:+.3f}), '
                  f'reach {a[1]:+.3f} vs {b[1]:+.3f} ({a[1] - b[1]:+.3f})')


if __name__ == '__main__':
    print('--- out to her silhouette')
    p = hug()
    print(f'--- {p}')
    check(p)
