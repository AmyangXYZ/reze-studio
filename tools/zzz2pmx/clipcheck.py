# -*- coding: utf-8 -*-
"""Report where an inner layer surfaces through the one covering it, pose by pose.

A garment is an open shell, so "which side of the nearest triangle" means nothing on
its own -- every vertex of the torso lies on some side of the shorts. Distance makes
the question answerable: a vertex is breaching only when the covering surface is
within `near` of it and it lies outside. Anything further belongs to another part of
the body.

The rest pose is not the one it is seen in, so each pair is measured across a fan of
leg poses. Exits non-zero when anything breaches deeper than `--max-depth`, so it can
gate a build.

    python3 tools/zzz2pmx/clipcheck.py <pmx> [--max-depth 0.01] [--verbose]
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pmx import PMX
from surface import Mesh
from fastpose import Skinner

# under, over, how close the two must be for a breach to be real
PAIRS = [('cloth01.001', 'shorts', 0.25),
         ('skin', 'shorts', 0.08),
         ('skin', 'socks', 0.08),
         ('skin', 'shirt', 0.08),
         ('cloth01', 'shirt', 0.20)]

POSES = [('rest', {}),
         ('open 20', {'右足': ((0, 0, 1), -20), '左足': ((0, 0, 1), 20)}),
         ('open 40', {'右足': ((0, 0, 1), -40), '左足': ((0, 0, 1), 40)}),
         ('open 60', {'右足': ((0, 0, 1), -60), '左足': ((0, 0, 1), 60)}),
         ('raise 30', {'右足': ((1, 0, 0), -30), '左足': ((1, 0, 0), -30)}),
         ('raise 60', {'右足': ((1, 0, 0), -60), '左足': ((1, 0, 0), -60)}),
         ('raise 85', {'右足': ((1, 0, 0), -85), '左足': ((1, 0, 0), -85)}),
         ('back 20', {'右足': ((1, 0, 0), 20), '左足': ((1, 0, 0), 20)}),
         ('stride 45', {'右足': ((1, 0, 0), -45), '左足': ((1, 0, 0), 15)}),
         ('stride 45 L', {'右足': ((1, 0, 0), 15), '左足': ((1, 0, 0), -45)}),
         ('twist', {'右足': ((0, 1, 0), 25), '左足': ((0, 1, 0), -25)}),
         ('knee 60', {'右足': ((1, 0, 0), -30), '右ひざ': ((1, 0, 0), 60)}),
         ('knee 90', {'右足': ((1, 0, 0), -45), '右ひざ': ((1, 0, 0), 90),
                      '左足': ((1, 0, 0), -45), '左ひざ': ((1, 0, 0), 90)}),
         ('arms down', {'右腕': ((0, 0, 1), 35), '左腕': ((0, 0, 1), -35)})]


def slices(m):
    F = np.array(m.faces).reshape(-1, 3)
    out, at = {}, 0
    for mat in m.materials:
        n = mat['fcount'] // 3
        out[mat['name']] = F[at:at + n]
        at += n
    return out


def breach(V, tris_over, idx, near):
    mesh = Mesh(V, tris_over)
    d, t, w = mesh.closest(V[idx])
    tri = tris_over[t]
    base = V[tri[:, 0]] * w[:, :1] + V[tri[:, 1]] * w[:, 1:2] + V[tri[:, 2]] * w[:, 2:3]
    side = np.einsum('ij,ij->i', V[idx] - base, mesh.normal[t])
    hit = (d < near) & (side > 0)
    return hit, side


def main(path, max_depth=0.01, verbose=False):
    m = PMX(path)
    mats = slices(m)
    skinner = Skinner(m)
    alpha = {mat['name']: mat['diffuse'][3] for mat in m.materials}
    hidden = {n for n, a in alpha.items() if a <= 0.001}
    pairs = [(u, o, n) for u, o, n in PAIRS
             if u in mats and o in mats and u not in hidden]
    if hidden:
        print(f'  not drawn, so not checked: {", ".join(sorted(hidden))}')
    print(f'{path.split("/")[-1]}: {len(pairs)} layer pairs over {len(POSES)} poses '
          f'(a breach counts past {max_depth})')
    worst = {}
    for label, rot in POSES:
        V = skinner.skin(rot)
        row = []
        for u, o, near in pairs:
            idx = np.unique(mats[u])
            hit, side = breach(V, mats[o], idx, near)
            deep = int((hit & (side > max_depth)).sum())
            top = float(side[hit].max()) if hit.any() else 0.0
            key = f'{u} in {o}'
            if top > worst.get(key, (0.0, ''))[0]:
                worst[key] = (top, label)
            if verbose or deep:
                row.append(f'{u}/{o} {int(hit.sum())}({top:.3f})')
        print(f'  {label:12s} ' + ('  '.join(row) if row else 'clean'))
    print('\n  worst over all poses:')
    fails = 0
    for key, (top, where) in sorted(worst.items(), key=lambda kv: -kv[1][0]):
        bad = top > max_depth
        fails += bad
        print(f'    {key:26s} {top:.4f} at {where}' + ('   FAIL' if bad else ''))
    return 1 if fails else 0


if __name__ == '__main__':
    a = sys.argv
    md = float(a[a.index('--max-depth') + 1]) if '--max-depth' in a else 0.01
    sys.exit(main(a[1] if len(a) > 1 and not a[1].startswith('--')
                  else 'public/models/reze/reze.pmx', md, '--verbose' in a))
