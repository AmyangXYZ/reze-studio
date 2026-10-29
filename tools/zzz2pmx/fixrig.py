# -*- coding: utf-8 -*-
"""Move the mesh off the bones that drive the rig and onto the ones that deform it.

A semi-standard MMD leg is two chains. 足 -> ひざ -> 足首 -> つま先 is what the IK
solves against, and 足D -> ひざD -> 足首D -> 足先EX is what the mesh is supposed to
hang on; the D chain takes its rotation from the other by grant (付与回転, the 0x0100
flag) so that the two agree. Weight the mesh to the solving chain instead and it
mostly works, which is the trouble -- at rest and in most frames the two are the same
pose, and they part company exactly when the grant rate is not 1.0 or the foot is
being driven through IK.

沉醉旖旎 has this on the toe alone: 620 vertices a side on 足首D and the foot sits
right, and then 273 a side on つま先, which is the IK end effector and a child of the
NON-D 足首. So the foot turns with the granted rotation and the toe turns with the raw
one, and the toe twists away from the foot it belongs to.

    python3 tools/zzz2pmx/fixrig.py <model.pmx> [...]

Rewrites each model in place. Positions are untouched: only which bone a vertex
listens to changes, so the rest pose is identical and only animation differs.
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pmx import PMX
from pmxw import write_pmx

GRANT = 0x0100      # the bone takes its rotation from another

# What to move onto what, once both are present. A `D` twin is found by name; the toe
# is the one that is not named for its twin.
TOE = {'つま先': '足先EX'}
SIDES = ('左', '右')


def retarget(m):
    """Which bone index each vertex should be listening to instead."""
    names = [b['name'] for b in m.bones]
    index = {n: i for i, n in enumerate(names)}
    swap = {}
    for i, n in enumerate(names):
        want = None
        if n + 'D' in index:
            want = n + 'D'
        else:
            for side in SIDES:
                for a, b in TOE.items():
                    if n == side + a and side + b in index:
                        want = side + b
        if want is None:
            continue
        j = index[want]
        if not (m.bones[j].get('flag', 0) & GRANT) and want[-1] == 'D':
            continue        # a twin that is not granted is not the deform bone
        swap[i] = j
    return swap, names


def apply(m, quiet=False):
    """Do it to a model already in hand. Returns how many vertices moved."""
    swap, names = retarget(m)
    if not swap:
        return 0
    moved, verts = {}, 0
    for k, v in enumerate(m.vertices):
        hit = [bi for bi in v[5] if bi in swap]
        if not hit:
            continue
        w = list(v)
        w[5] = [swap.get(bi, bi) for bi in v[5]]
        m.vertices[k] = tuple(w)
        verts += 1
        for bi in hit:
            moved[names[bi]] = moved.get(names[bi], 0) + 1
    if not quiet:
        for n, c in sorted(moved.items(), key=lambda x: -x[1]):
            print(f'  {c:5d} vertices moved from {n} to {names[swap[names.index(n)]]}')
    return verts


def main(paths):
    for path in paths:
        m = PMX(path)
        verts = apply(m)
        if not verts:
            print(f'{path}: nothing to move')
            continue
        write_pmx(path, m)
        print(f'{path}: {verts} vertices now listen to the deform chain')


if __name__ == '__main__':
    main(sys.argv[1:])
