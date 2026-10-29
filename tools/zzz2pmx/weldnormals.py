# -*- coding: utf-8 -*-
"""Close the black wedges the outline opens at a hard edge.

The outline is an inverted hull: every vertex is pushed out along its own normal and
the backfaces are drawn black behind the model. Where a mesh carries a hard edge it
splits the vertex, and the copies hold normals far apart -- in these shorts, 92
positions whose copies differ by up to 97 degrees. Pushed out along those, the copies
separate, and the black hull shows between them as a wedge.

Averaging the copies' normals closes it: they extrude together, so the hull stays
sealed. The crease shades smoothly afterwards, which is the price, and only positions
whose copies actually disagree are touched -- a seam split for UVs alone already
agrees and is left as it is.

Welding is per material: a position shared by a garment and the body belongs to two
surfaces facing different ways, and averaging across them would tilt both. Copies
more than `--max-angle` apart are left alone for the same reason -- at 180 degrees
they are not a crease but two sheets back to back, a strand of hair or a lining, and
their average is a direction that points nowhere.

    python3 tools/zzz2pmx/weldnormals.py <pmx> [materials] [--spread 15] [--dry]
"""
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pmx import PMX
from pmxw import write_pmx


def main(path, names=None, spread=15.0, max_angle=120.0, dry=False):
    m = PMX(path)
    V = np.array([v[0] for v in m.vertices])
    N = np.array([v[1] for v in m.vertices])
    F = np.array(m.faces).reshape(-1, 3)
    mats, at = {}, 0
    for mat in m.materials:
        n = mat['fcount'] // 3
        mats[mat['name']] = F[at:at + n]
        at += n
    todo = names or list(mats)
    total = 0
    for name in todo:
        if name not in mats or not len(mats[name]):
            continue
        groups = defaultdict(list)
        for i in np.unique(mats[name]):
            groups[tuple(np.round(V[i], 5))].append(int(i))
        fixed, worst_before, worst_after = 0, 0.0, 0.0
        for ids in groups.values():
            if len(ids) < 2:
                continue
            ns = N[np.array(ids)]
            ns = ns / (np.linalg.norm(ns, axis=1, keepdims=True) + 1e-12)
            ang = np.degrees(np.arccos(np.clip(ns @ ns.T, -1, 1))).max()
            if ang <= spread or ang >= max_angle:
                continue
            worst_before = max(worst_before, ang)
            avg = ns.mean(0)
            n = np.linalg.norm(avg)
            if n < 1e-6:
                continue
            avg /= n
            fixed += 1
            if not dry:
                for i in ids:
                    N[i] = avg
        if fixed:
            print(f'{name:14s} welded {fixed} positions whose copies differed by '
                  f'up to {worst_before:.0f}deg')
            total += fixed
    if dry or not total:
        print(f'{total} positions would change')
        return
    for i in range(len(m.vertices)):
        v = list(m.vertices[i])
        v[1] = [float(x) for x in N[i]]
        m.vertices[i] = tuple(v)
    write_pmx(path, m)
    print(f'welded {total} positions in all; wrote {path}')


if __name__ == '__main__':
    argv, a, skip = sys.argv[1:], [], False
    sp, mx = 15.0, 120.0
    for i, x in enumerate(argv):
        if skip:
            skip = False
            continue
        if x == '--spread':
            sp = float(argv[i + 1])
            skip = True
        elif x == '--max-angle':
            mx = float(argv[i + 1])
            skip = True
        elif not x.startswith('--'):
            a.append(x)
    main(a[0], a[1:] or None, sp, mx, '--dry' in argv)
