# -*- coding: utf-8 -*-
"""Rebuild the choker's rings as true tori.

Whitening a ring's covariance removes the ellipse it was squashed into and leaves
everything else: the tube centres still wander in and out by a fifth of the radius
and the tube itself is pinched by a third, which on a ring this small still reads as
"not a circle". So the ring is re-authored rather than nudged -- its own vertices
give the plane, the major radius and the tube radius, and it is laid out again from
those three numbers with segments and tube evenly spaced.

The ring carries seam duplicates: 120 vertex indices over 96 distinct positions,
24 of them doubled. The rebuild works on the distinct positions and writes each
result back to every index that shares it, so the seam stays welded -- giving them
separate positions tears the ring into halves.

    python3 tools/zzz2pmx/roundring.py <pmx> [--dry]
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pmx import PMX
from pmxw import write_pmx

MAT = 'choker'
SEGMENTS = 12
INDICES = 120


def shells(V, tris):
    used = np.unique(tris)
    rep, key, parent = {}, {}, {}
    for i in used:
        rep[i] = key.setdefault(tuple(np.round(V[i], 5)), i)

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    for t in tris:
        a, b, c = find(rep[t[0]]), find(rep[t[1]]), find(rep[t[2]])
        for u, v in ((a, b), (b, c)):
            if u != v:
                parent[u] = v
    g = {}
    for i in used:
        g.setdefault(find(rep[i]), []).append(i)
    return g


def frame(p):
    c = p.mean(0)
    q = p - c
    w, U = np.linalg.eigh(q.T @ q)
    return c, U[:, 0], U[:, 1], U[:, 2]


def segments(p, per):
    c, axis, e1, e2 = frame(p)
    q = p - c
    th = np.arctan2(q @ e2, q @ e1)
    order = np.argsort(th)
    return [order[i * per:(i + 1) * per] for i in range(SEGMENTS)]


def measure(p, per):
    c, axis, e1, e2 = frame(p)
    groups = segments(p, per)
    cen = np.array([p[gr].mean(0) for gr in groups])
    R = np.linalg.norm((cen - c) @ np.column_stack([e1, e2]), axis=1)
    tube = np.array([np.linalg.norm(p[gr] - cen[j], axis=1).mean()
                     for j, gr in enumerate(groups)])
    return R, tube


def rebuild(p, per):
    c, axis, e1, e2 = frame(p)
    groups = segments(p, per)
    cen = np.array([p[gr].mean(0) for gr in groups])
    R, tube = measure(p, per)
    Rm, rm = R.mean(), tube.mean()
    d = (cen - c) @ np.column_stack([e1, e2])
    thc = np.arctan2(d[:, 1], d[:, 0])
    base = np.angle(np.exp(1j * (thc - 2 * np.pi * np.arange(SEGMENTS) / SEGMENTS)).mean())
    out, nrm = np.zeros_like(p), np.zeros_like(p)
    for j, gr in enumerate(groups):
        tj = base + 2 * np.pi * j / SEGMENTS
        radial = np.cos(tj) * e1 + np.sin(tj) * e2
        rel = p[gr] - cen[j]
        old = (cen[j] - c) / max(np.linalg.norm(cen[j] - c), 1e-9)
        phi = np.arctan2(rel @ axis, rel @ old)
        k = np.argsort(phi)
        even = np.angle(np.exp(1j * (phi[k] - 2 * np.pi * np.arange(per) / per)).mean())
        pn = np.empty(per)
        pn[k] = even + 2 * np.pi * np.arange(per) / per
        n = np.outer(np.cos(pn), radial) + np.outer(np.sin(pn), axis)
        out[gr] = c + Rm * radial + rm * n
        nrm[gr] = n
    return out, nrm, Rm, rm


def main(path, dry=False):
    m = PMX(path)
    V = np.array([v[0] for v in m.vertices])
    N = np.array([v[1] for v in m.vertices])
    F = np.array(m.faces).reshape(-1, 3)
    at = 0
    for mat in m.materials:
        n = mat['fcount'] // 3
        if mat['name'] == MAT:
            tris = F[at:at + n]
        at += n

    for ids in shells(V, tris).values():
        if len(ids) != INDICES:
            continue
        pos = {}
        for i in ids:
            pos.setdefault(tuple(np.round(V[i], 5)), []).append(i)
        keys = list(pos)
        p = np.array(keys)
        if len(p) % SEGMENTS:
            print(f'   skipped a ring: {len(p)} positions is not a multiple of {SEGMENTS}')
            continue
        per = len(p) // SEGMENTS
        R0, t0 = measure(p, per)
        new, nn, Rm, rm = rebuild(p, per)
        R1, t1 = measure(new, per)
        print(f'ring R={Rm:.4f} tube={rm:.4f}  ({len(ids)} indices, {len(p)} positions)')
        print(f'   major radius swing {100*(R0.max()/R0.min()-1):5.1f}% -> '
              f'{100*(R1.max()/R1.min()-1):.1f}%')
        print(f'   tube radius swing  {100*(t0.max()/t0.min()-1):5.1f}% -> '
              f'{100*(t1.max()/t1.min()-1):.1f}%')
        print(f'   moved at most {np.abs(new-p).max():.4f}')
        if dry:
            continue
        for q, name in enumerate(keys):
            for i in pos[name]:
                V[i], N[i] = new[q], nn[q]
    if dry:
        return
    for i in range(len(m.vertices)):
        v = list(m.vertices[i])
        v[0] = [float(x) for x in V[i]]
        v[1] = [float(x) for x in N[i]]
        m.vertices[i] = tuple(v)
    write_pmx(path, m)
    print(f'wrote {path}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'public/models/reze/reze.pmx',
         '--dry' in sys.argv)
