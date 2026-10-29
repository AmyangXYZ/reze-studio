# -*- coding: utf-8 -*-
"""How much of an under-layer can actually be seen, by ray casting.

A rasteriser preview answers this badly: it is axis-aligned, its resolution hides
small leaks, and the angle a thing is noticed from is rarely the angle it is rendered
at. A ray answers it exactly. From the centre of each triangle of the under-layer, a
ray is cast toward the viewer; if it leaves without meeting anything, that triangle is
visible, and the area of every such triangle is what "showing" means.

Everything opaque occludes, not just the garment named. Testing against the garment
alone counts triangles the body itself hides, and then the measure moves the wrong
way: pushing the underwear INTO the skin appeared to expose more of it.

Directions fan around and below the model, because that is where a crotch is seen from.

    python3 tools/zzz2pmx/exposed.py <pmx> <under> <over>
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pmx import PMX


def prune(V, tris, box_lo, box_hi, pad=1.2):
    """Only triangles near the layer being tested can block a ray from it."""
    c = (V[tris[:, 0]] + V[tris[:, 1]] + V[tris[:, 2]]) / 3
    keep = np.all((c > box_lo - pad) & (c < box_hi + pad), axis=1)
    return tris[keep]


def slices(m):
    F = np.array(m.faces).reshape(-1, 3)
    out, at = {}, 0
    for mat in m.materials:
        n = mat['fcount'] // 3
        out[mat['name']] = F[at:at + n]
        at += n
    return out


def hits(orig, direction, V, tris, eps=1e-4):
    """Möller-Trumbore, vectorised over triangles, for one ray."""
    a, b, c = V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]
    e1, e2 = b - a, c - a
    p = np.cross(direction, e2)
    det = np.einsum('ij,j->i', e1, p) if p.ndim == 1 else np.einsum('ij,ij->i', e1, p)
    live = np.abs(det) > 1e-12
    inv = np.zeros_like(det)
    inv[live] = 1.0 / det[live]
    t0 = orig - a
    u = np.einsum('ij,ij->i', t0, p) * inv if p.ndim > 1 else np.einsum('ij,j->i', t0, p) * inv
    q = np.cross(t0, e1)
    v = np.einsum('j,ij->i', direction, q) * inv
    t = np.einsum('ij,ij->i', e2, q) * inv
    return np.any(live & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > eps))


def directions(tilts=(0, 20, 40, 60), turns=(0, 25, -25, 50, -50, 90, -90, 180)):
    out = []
    for tilt in tilts:
        for turn in turns:
            tx, ty = np.deg2rad(tilt), np.deg2rad(turn)
            d = np.array([np.sin(ty) * np.cos(tx), -np.sin(tx), np.cos(ty) * np.cos(tx)])
            out.append(d / np.linalg.norm(d))
    return out


def main(path, under, over, report=True, occluders=None):
    m = PMX(path)
    V = np.array([v[0] for v in m.vertices])
    mats = slices(m)
    if occluders is None:                    # everything opaque except the layer itself
        alpha = {mt['name']: mt['diffuse'][3] for mt in m.materials}
        occluders = [n for n in mats if n != under and alpha.get(n, 1) > 0.001]
    utris = mats[under]
    otris = np.concatenate([mats[n] for n in occluders])
    up = V[np.unique(mats[under])]
    otris = prune(V, otris, up.min(0), up.max(0))
    a, b, c = V[utris[:, 0]], V[utris[:, 1]], V[utris[:, 2]]
    cen = (a + b + c) / 3
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    nrm = np.cross(b - a, c - a)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12

    seen = np.zeros(len(utris), bool)
    for d in directions():
        facing = nrm @ d > 0.05          # only triangles turned toward that viewer
        for k in np.flatnonzero(facing & ~seen):
            if not hits(cen[k] + nrm[k] * 1e-3, d, V, otris):
                seen[k] = True
    if report:
        print(f'{under} visible past {len(occluders)} occluders: {int(seen.sum())} of {len(utris)} '
              f'triangles, {100 * area[seen].sum() / area.sum():.2f}% of its area')
        if seen.any():
            P = cen[seen]
            print(f'   they sit x {P[:,0].min():+.2f}..{P[:,0].max():+.2f}  '
                  f'y {P[:,1].min():.2f}..{P[:,1].max():.2f}  z {P[:,2].min():+.2f}..{P[:,2].max():+.2f}')
    return seen, area


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
