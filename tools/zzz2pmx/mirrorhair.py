# -*- coding: utf-8 -*-
"""Flip a model's own haircut left-for-right, keeping its rig.

Mirroring the mesh alone leaves every strand driven from the far side of the head,
because the weights still name the chain the vertex used to sit on. So the rig moves
with it -- and the move is exactly a sign flip on x, nothing else.

Three richer versions were tried and all behaved worse in the engine. Conjugating
each rotation, R -> M.R.M, is the mathematically exact reflection and matches a
perfect mirror at 0.00e+00, but it flips every body's local x axis and the strands
swing wrong. Authoring fresh frames along the mirrored bones is cleaner still on
paper and no better in practice. And refitting the skull spheres after either of
those shrinks them -- 0.79 to 0.72, 0.37 to 0.27 -- so the mirrored head collides
less than the original's, which is its own defect.

Positions flip. Rotations, limits, sizes, masses and the skull keep whatever the
source model had. Nothing outside the hair is touched.

    python3 tools/zzz2pmx/mirrorhair.py --target public/models/reze/reze.pmx
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hairphys
from pmx import PMX
from pmxw import write_pmx

HAIR_MAT = '头发'
HAIR_BONES = ('yb_', 'zb_', 'e_')


def build(args):
    m = PMX(args.target)
    faces = np.array(m.faces).reshape(-1, 3)
    off, tris, before = 0, None, 0
    for mat in m.materials:
        n = mat['fcount'] // 3
        if mat['name'] == HAIR_MAT:
            tris, before = faces[off:off + n], off
        off += n
    if tris is None:
        raise SystemExit(f'mirrorhair: {args.target} has no {HAIR_MAT} material')

    used = np.unique(tris)
    for i in used:
        v = list(m.vertices[int(i)])
        v[0] = [-v[0][0], v[0][1], v[0][2]]
        v[1] = [-v[1][0], v[1][1], v[1][2]]
        m.vertices[int(i)] = tuple(v)
    hairphys.reflect_hair_rig(m)
    if args.reauthor:
        hairphys.reauthor_hair_rig(m)

    # The reflection turns the hair inside out; turn it back. `before` is already a
    # triangle index -- dividing it by 3 again reverses some other material's faces and
    # leaves the hair backwards, which is what it looks like.
    flat = list(m.faces)
    for t in range(before, before + len(tris)):
        a, b, c = flat[3 * t], flat[3 * t + 1], flat[3 * t + 2]
        flat[3 * t], flat[3 * t + 1], flat[3 * t + 2] = a, c, b
    m.faces = flat

    print(f'mirrored {len(used)} hair vertices, weights untouched; '
          'rig differs from the source only in the sign of x')
    n = write_pmx(args.out, m)
    print(f'wrote {args.out}  {n/1e6:.2f} MB  {len(m.bones)} bones  '
          f'{len(m.rigids)} rigid bodies  {len(m.joints)} joints '
          f'(the hair\'s share of them reflected with it)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='public/models/reze/reze.pmx')
    ap.add_argument('--out', default='public/models/reze/reze-mirror-hair.pmx')
    ap.add_argument('--reauthor', action='store_true',
                    help='rebuild capsule and joint frames from the bones instead of reflecting them')
    build(ap.parse_args())
