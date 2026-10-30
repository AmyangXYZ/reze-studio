# -*- coding: utf-8 -*-
"""Lower a face's eyebrows along the face, into a new PMX beside it.

    python3 tools/zzz2pmx/browdrop.py <model.pmx> <out.pmx> [drop] [--material 眉毛]
    python3 tools/zzz2pmx/browdrop.py <model.pmx> <out.pmx> --like <donor.pmx>

A 深空之眼 face keeps its brows, lash lines and lid creases in one material (眉毛);
the brows are its two highest shells. Each brow vertex moves down by `drop` (model
units) and keeps its depth off the face: the face's front surface is found straight
behind the vertex before and after (a ray along z through the face material), and the
vertex sits as far in front of it at the new height as it did at the old. A straight
drop would sink the brow into the brow ridge, which leans back as it rises.

Morph offsets are relative, so the brow expressions move with it. Nothing else changes.

Makima (英招2.0's face) carried her brows 0.59 of an iris height above the iris, where
塞尔凯特 and 大梵天 sit at 0.18-0.19; 0.10 down brings her to about 0.23.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holefill
import roundring as rr
from pmx import PMX
from pmxw import write_pmx


def front(P, T, xy):
    """The face's frontmost z (most negative: a PMX face looks down -z) behind each
    point, by barycentric test in the xy plane; nan where no triangle covers it."""
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    out = np.full(len(xy), np.nan)
    for k, (x, y) in enumerate(xy):
        v0, v1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
        v2 = np.array([x, y]) - a[:, :2]
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        ok = np.abs(den) > 1e-12
        s = np.where(ok, (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / np.where(ok, den, 1), -1)
        t = np.where(ok, (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / np.where(ok, den, 1), -1)
        hit = ok & (s >= -1e-6) & (t >= -1e-6) & (s + t <= 1 + 1e-6)
        if hit.any():
            z = a[hit, 2] + s[hit] * (b[hit, 2] - a[hit, 2]) + t[hit] * (c[hit, 2] - a[hit, 2])
            out[k] = z.min()
    return out


def parts(path, material='眉毛'):
    """A model's points, its face triangles, its two brows (the top two shells of the
    brow material, as vertex arrays) and its +x iris centre."""
    m = PMX(path)
    sl = holefill.slices(m)
    P = np.array([v[0] for v in m.vertices])
    face = next(x['name'] for x in m.materials if x['name'] in ('face01', '脸', '顔'))
    shells = sorted(rr.shells(P, np.array(sl[material])).values(), key=lambda v: -P[v][:, 1].mean())
    I = np.unique(np.array(sl['眼睛']))
    return m, P, np.array(sl[face]), [np.array(v) for v in shells[:2]], P[I[P[I, 0] > 0]].mean(0), len(shells)


def like(src, dst, donor, material='眉毛'):
    """The brows bent to another face's: each of ours keeps its vertices, UVs and paint
    (so its morphs still apply), and takes the donor brow's top and bottom outline at
    the same point along its length, placed as the donor's sits against its iris centre
    and as far in front of our face as the donor's stands off its own.

    A 深空之眼 brow is a strip, a top row and a bottom row along it; a row is told by
    its UV v. Makima's was straight; 塞尔凯特's hooks down at the inner end, peaks a
    third of the way out and tapers to a point."""
    m, P, F, brows, iris, _ = parts(src, material)
    d, Q, G, dbrows, diris, _ = parts(donor, material)
    UV = np.array([v[2] for v in m.vertices])
    DUV = np.array([v[2] for v in d.vertices])
    db = max(dbrows, key=lambda v: Q[v, 0].mean())                # the donor's +x brow
    dx = Q[db, 0]
    ds = (dx - dx.min()) / np.ptp(dx)
    dv = DUV[db, 1]
    dtop = dv < dv.max() - 1e-3                                   # its bottom row shares one v
    depth = Q[db, 2] - front(Q, G, Q[db, :2])
    def curve(sel, vals, s):
        o = np.argsort(ds[sel])
        return np.interp(s, ds[sel][o], vals[sel][o])
    lift = iris[1] - diris[1]
    for b in brows:
        sign = np.sign(P[b, 0].mean())
        ax = np.abs(P[b, 0])
        s = (ax - ax.min()) / np.ptp(ax)
        v = UV[b, 1]
        top = v < (v.min() + v.max()) / 2
        y = np.where(top, curve(dtop, Q[db, 1], s), curve(~dtop, Q[db, 1], s)) + lift
        off = np.where(top, curve(dtop, depth, s), curve(~dtop, depth, s))
        xy = np.stack([P[b, 0], y], -1)
        z = front(P, F, xy) + off
        assert not np.isnan(z).any(), 'a brow vertex falls off the face'
        for i, yy, zz in zip(b, y, z):
            vv = list(m.vertices[i])
            vv[0] = [vv[0][0], float(yy), float(zz)]
            m.vertices[i] = tuple(vv)
        print(f'  {"+x" if sign > 0 else "-x"} brow: {len(b)} vertices onto the donor outline, '
              f'y {y.min():.3f}..{y.max():.3f} (was {P[b, 1].min():.3f}..{P[b, 1].max():.3f})')
    write_pmx(dst, m)
    print(f'wrote {dst}')


def main(src, dst, drop=0.10, material='眉毛', face=None):
    m, P, F, brows, _, n = parts(src, material)
    brows = np.concatenate(brows)
    print(f'  brows: the top 2 of {n} shells in {material}, {len(brows)} vertices, '
          f'y {P[brows, 1].min():.3f}..{P[brows, 1].max():.3f}')
    old = front(P, F, P[brows, :2])
    new = front(P, F, P[brows, :2] - [0, drop])
    lost = np.isnan(old) | np.isnan(new)
    assert not lost.any(), f'{int(lost.sum())} brow vertices fall off the face'
    for i, zo, zn in zip(brows, old, new):
        v = list(m.vertices[i])
        p = list(v[0])
        p[1] -= drop
        p[2] += zn - zo
        v[0] = p
        m.vertices[i] = tuple(v)
    print(f'  down {drop}, depth carried: z moved {np.min(new - old):+.3f}..{np.max(new - old):+.3f}')
    write_pmx(dst, m)
    print(f'wrote {dst}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('drop', nargs='?', type=float, default=0.10)
    ap.add_argument('--material', default='眉毛')
    ap.add_argument('--like', help="another face's brows to take the shape of (see `like`)")
    a = ap.parse_args()
    if a.like:
        like(a.src, a.dst, a.like, a.material)
    else:
        main(a.src, a.dst, a.drop, a.material)
