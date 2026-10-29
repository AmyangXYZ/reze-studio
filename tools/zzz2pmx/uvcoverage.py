# -*- coding: utf-8 -*-
"""How much of each texture a model actually samples.

A texture can be listed, indexed by a material, and still be mostly empty -- a 2K
sheet carrying one small island, or a ramp stored as a square when it varies along
one axis. This rasterises the UV triangles of every material that samples a texture
as its diffuse map and reports what fraction of the image they land on, which is
what decides whether two sheets can share one.

Sphere and toon maps are not sampled by UV -- a sphere map is indexed by the view
normal, a toon ramp by the lighting term -- so they are reported by role instead of
by coverage.

    python3 tools/zzz2pmx/uvcoverage.py <pmx>
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import PMX

GRID = 128


def roles(m):
    """texture index -> {role: [material names]}"""
    out = {}
    for mat in m.materials:
        for role, key in (('diffuse', 'tex'), ('sphere', 'sph'), ('toon', 'toon')):
            if key == 'toon' and mat['toon_shared']:
                continue
            i = mat[key]
            if 0 <= i < len(m.textures):
                out.setdefault(i, {}).setdefault(role, []).append(mat['name'])
    return out


def coverage(m, index):
    """Fraction of the unit square the diffuse UVs of `index` cover, and their bbox."""
    UV = np.array([v[2] for v in m.vertices])
    F = np.array(m.faces).reshape(-1, 3)
    at, tris = 0, []
    for mat in m.materials:
        n = mat['fcount'] // 3
        if mat['tex'] == index:
            tris.append(F[at:at + n])
        at += n
    if not tris:
        return None
    t = np.concatenate(tris)
    uv = UV[t]
    lo, hi = uv.reshape(-1, 2).min(0), uv.reshape(-1, 2).max(0)
    grid = np.zeros((GRID, GRID), bool)
    p = (uv % 1.0) * GRID
    for tri in p:
        x0, y0 = np.floor(tri.min(0)).astype(int)
        x1, y1 = np.ceil(tri.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, GRID), min(y1, GRID)
        if x1 <= x0 or y1 <= y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1) + .5, np.arange(y0, y1) + .5)
        a, b, c = tri
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            grid[y0:y1, x0:x1] = True
            continue
        w0 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / d
        w1 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / d
        hit = (w0 >= -.02) & (w1 >= -.02) & (w0 + w1 <= 1.02)
        grid[y0:y1, x0:x1] |= hit
    return grid.mean(), lo, hi, len(t)


def main(path):
    m = PMX(path)
    base = os.path.dirname(path)
    use = roles(m)
    print(f'{os.path.basename(path)}: {len(m.textures)} textures\n')
    total = 0
    for i, t in enumerate(m.textures):
        f = os.path.join(base, t.replace('\\', '/'))
        size = os.path.getsize(f) if os.path.exists(f) else 0
        total += size
        try:
            w, h = Image.open(f).size
            dims = f'{w}x{h}'
        except Exception:
            dims = '?'
        r = use.get(i, {})
        kinds = ', '.join(f'{k} ({len(v)})' for k, v in r.items()) or 'UNUSED'
        print(f'[{i:2d}] {os.path.basename(t):32s} {dims:>10s} {size/1e6:6.2f} MB  {kinds}')
        cov = coverage(m, i)
        if cov:
            frac, lo, hi, n = cov
            print(f'      {100*frac:5.1f}% of the sheet sampled by {n} triangles; '
                  f'u {lo[0]:.2f}-{hi[0]:.2f}  v {lo[1]:.2f}-{hi[1]:.2f}')
        for role, mats in r.items():
            print(f'      {role}: {", ".join(mats)}')
    print(f'\ntotal {total/1e6:.1f} MB')


if __name__ == '__main__':
    main(sys.argv[1])
