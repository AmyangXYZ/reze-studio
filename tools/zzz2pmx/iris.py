# -*- coding: utf-8 -*-
"""Paint an iris of concentric rings onto a model's iris material, in place.

    python3 tools/zzz2pmx/iris.py <model.pmx> [--material 瞳] [--sheet iris.png]

Makima's eye: rings within rings, the control devil's. The iris material's disc gets a
sheet of its own -- its UVs remapped from their corner of the face sheet onto the whole
of the new one, so the rings are drawn at 1024 pixels instead of the 120 the face sheet
spends on an iris -- and the face sheet, brows, lashes and whites stay as they were.

`RINGS` lists, outward from the centre, where each band ends (1 is the edge of the
mesh disc) and what it is: a dark dot, gold, dark rings between gold, an orange outer
gold and a dark rim.

A highlight disc sharing the iris material (英招OL's 眼睛 holds one per eye, on another
corner of the face sheet) goes into a material of its own on the face sheet, appended
last, before the iris discs are remapped.

The upper half keeps the shadow the original iris was painted with (目影, under the
upper lid): the old iris's brightness, its thin strokes -- a highlight ring, the pupil's
outline -- filtered out, divided by its brightest part and laid over the rings.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holefill
from pmx import PMX
from pmxw import write_pmx

DARK = '#120807'
RINGS = [(0.11, DARK), (0.25, 'gold'), (0.35, DARK), (0.56, 'gold'), (0.66, DARK), (0.88, 'gold'), (1.0, '#0b0504')]
GOLD = ('#f7d34c', '#e9922c')   # the gold's centre and edge


def hexc(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def paint(size=1024, ss=3, soft=0.012):
    n = size * ss
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n * 2 - 1
    r = np.hypot(x, y)
    gold = hexc(GOLD[0]) * (1 - np.clip(r, 0, 1))[..., None] + hexc(GOLD[1]) * np.clip(r, 0, 1)[..., None]
    out = np.zeros((n, n, 3))
    inner = 0.0
    for edge, what in RINGS:
        col = gold if what == 'gold' else np.broadcast_to(hexc(what), gold.shape)
        # a soft step in, a soft step out, so the rings are not aliased
        w = np.clip((r - inner) / soft + 0.5, 0, 1) * np.clip((edge - r) / soft + 0.5, 0, 1)
        out += col * w[..., None]
        inner = edge
    out += hexc(RINGS[-1][1]) * (r > 1.0)[..., None]
    # faint radial strokes over the gold, as the drawn eye has
    ang = np.arctan2(y, x)
    streak = 1 - 0.06 * (0.5 + 0.5 * np.cos(ang * 48)) * ((r > 0.36) & (r < 0.88))
    out *= streak[..., None]
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    return img.resize((size, size), Image.LANCZOS)


def shadow(sheet, c, rad, size=1024):
    """The broad light and shade of the original iris, as a multiplier on the new sheet:
    its pixels at the new sheet's own coordinates, a median filter wide enough to drop
    the thin strokes, then blurred and divided by its brightest part."""
    from scipy import ndimage
    im = Image.open(sheet).convert('RGB')
    W, H = im.size
    # new (s, t) in [0, 1] -> old uv = c + (s - 0.5) * 2 rad -> old pixels
    x0, y0 = (c[0] - rad[0]) * W, (c[1] - rad[1]) * H
    sx, sy = 2 * rad[0] * W / size, 2 * rad[1] * H / size
    # the shade is broad, so it is worked out at a quarter size and scaled up
    n = size // 4
    old = im.transform((n, n), Image.AFFINE, (sx * 4, 0, x0, 0, sy * 4, y0), Image.BICUBIC)
    L = np.asarray(old).astype(float) @ [0.299, 0.587, 0.114]
    L = ndimage.median_filter(L, size=n // 7)
    L = ndimage.gaussian_filter(L, n / 60)
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n * 2 - 1
    disc = np.hypot(x, y) < 0.85
    L = np.clip(L / np.percentile(L[disc], 95), 0.3, 1.0)
    return np.asarray(Image.fromarray(L.astype(np.float32)).resize((size, size), Image.BICUBIC))


def split_highlights(m, mat, asname):
    """Shells of `mat` far smaller in UV than its largest -- highlight discs -- into a
    material of their own, appended last so no material morph's index shifts."""
    import roundring as rr
    F = np.array(m.faces).reshape(-1, 3)
    fm = np.repeat(np.arange(len(m.materials)), [x['fcount'] // 3 for x in m.materials])
    k = next(k for k, x in enumerate(m.materials) if x['name'] == mat)
    T = F[fm == k]
    UV = np.array([v[2] for v in m.vertices])
    V = np.array([v[0] for v in m.vertices])
    sh = list(rr.shells(V, T).values())
    ext = [np.ptp(UV[list(ids)], 0).max() for ids in sh]
    small = set()
    for ids, e in zip(sh, ext):
        if e < 0.4 * max(ext):
            small |= {int(i) for i in ids}
    hl = np.array([int(t[0]) in small for t in T])
    if not hl.any():
        return
    fm[np.flatnonzero(fm == k)[hl]] = len(m.materials)
    x = dict(m.materials[k])
    x['name'] = x['name_en'] = asname
    m.materials.append(x)
    order = np.argsort(fm, kind='stable')
    for j, y in enumerate(m.materials):
        y['fcount'] = int((fm == j).sum()) * 3
    m.faces = F[order].ravel().tolist()
    print(f'  {int(hl.sum())} triangles of highlight discs out of `{mat}` into `{asname}`')


def main(path, mat='瞳', sheet='iris.png'):
    m = PMX(path)
    split_highlights(m, mat, mat + '_高光')
    ids = np.unique(holefill.slices(m)[mat])
    UV = np.array([m.vertices[i][2] for i in ids])
    # the disc's centre is its centre vertex; its radius the farthest rim vertex
    c = UV.mean(0)
    c = UV[np.argmin(np.linalg.norm(UV - c, axis=1))]
    rad = np.abs(UV - c).max(0)
    for i, uv in zip(ids, UV):
        v = list(m.vertices[int(i)])
        v[2] = list(0.5 + (uv - c) / (2 * rad))
        m.vertices[int(i)] = tuple(v)
    x = next(x for x in m.materials if x['name'] == mat)
    old = os.path.join(os.path.dirname(path), m.textures[x['tex']].replace('\\', '/'))
    shade = shadow(old, c, rad)
    rings = np.asarray(paint()).astype(float) * shade[..., None]
    Image.fromarray(np.clip(rings, 0, 255).astype(np.uint8)).save(os.path.join(os.path.dirname(path), 'textures', sheet))
    m.textures.append('textures\\' + sheet)
    x['tex'] = len(m.textures) - 1
    write_pmx(path, m)
    print(f'{path}: `{mat}` on {sheet}, rings painted; {len(ids)} vertices remapped from a '
          f'{2 * rad[0]:.3f} x {2 * rad[1]:.3f} corner of the face sheet')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--material', default='瞳')
    ap.add_argument('--sheet', default='iris.png')
    a = ap.parse_args()
    main(a.model, a.material, a.sheet)
