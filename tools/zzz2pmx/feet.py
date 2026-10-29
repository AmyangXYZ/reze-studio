# -*- coding: utf-8 -*-
"""Stand a model on the floor and put its toe bones where motions expect them, in place.

    python3 tools/zzz2pmx/feet.py <model.pmx> [--back 0.3] [--up 0.2]

A rig carried over from a game can stand with its soles under y 0 -- 海拉's sank 0.35
into the stage -- and keep its つま先, the toe IK's target, wherever the game put it:
hers floated 0.8 past the shoe tip, 0.64 up. Motions are authored for a toe at the
tip of the foot on the floor, so the same toe IK key tilted her foot differently.

The whole model is lifted until the lowest vertex of its feet touches y 0; each
つま先 and its つま先ＩＫ go `back` behind the tip of the toe mesh (what 足先EX, or
足首D, carries) and `up` above the floor. Nothing is weighted to either bone, so the
mesh does not change; only how the IK aims the foot.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pmxscale
from pmx import PMX
from pmxw import write_pmx


def main(path, back=0.3, up=0.2):
    m = PMX(path)
    nm = [b['name'] for b in m.bones]
    V = np.array([v[0] for v in m.vertices])

    def carried(names, w=0.3):
        idx = {nm.index(n) for n in names if n in nm}
        return np.array([k for k, v in enumerate(m.vertices) if any(b in idx and x > w for b, x in zip(v[5], v[6]))])

    feet = carried(['左足首D', '右足首D', '左足先EX', '右足先EX', '左足首', '右足首'])
    lift = -float(V[feet][:, 1].min())
    pmxscale.scale(m, 1.0, (0.0, lift, 0.0))
    V = np.array([v[0] for v in m.vertices])
    print(f'{path}: lifted {lift:+.3f}, soles on the floor')
    for side in '左右':
        toe = carried([side + '足先EX']) if side + '足先EX' in nm else carried([side + '足首D'])
        tip = V[toe][:, 2].min()
        for n in (side + 'つま先', side + 'つま先ＩＫ'):
            b = m.bones[nm.index(n)]
            old = np.array(b['pos'])
            b['pos'] = [old[0], up, tip + back]
            print(f'  {n}: {np.round(old, 2)} -> {np.round(b["pos"], 2)}')
    write_pmx(path, m)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--back', type=float, default=0.3)
    ap.add_argument('--up', type=float, default=0.2)
    a = ap.parse_args()
    main(a.model, a.back, a.up)
