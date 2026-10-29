"""Scale a model's head about its neck joint, rig and all.

Everything the 頭 bone and its descendants own, the vertices, the bones, their
rigid bodies and joints, grows about the 頭 bone's position, so the neck stays
where it was. A vertex the neck shares with the head grows by its head share,
so the join stays smooth. Vertex morphs on the head scale with it.

    python3 tools/zzz2pmx/headscale.py --pmx public/models/reze/reze-bomb.pmx --scale 1.1
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate
from pmx import PMX
from pmxw import write_pmx


def main(a):
    P = PMX(a.pmx)
    names = [b['name'] for b in P.bones]
    head = names.index(a.bone)
    kids = {head}
    grew = True
    while grew:
        grew = False
        for i, b in enumerate(P.bones):
            if b['parent'] in kids and i not in kids:
                kids.add(i)
                grew = True
    pivot = np.array(P.bones[head]['pos'])
    s = a.scale
    V = np.array([v[0] for v in P.vertices])
    w = np.zeros(len(V))
    for k, v in enumerate(P.vertices):
        for b, x in zip(v[5], v[6][:len(v[5])]):
            if b in kids:
                w[k] += x
    w = np.clip(w, 0, 1)
    f = 1 + (s - 1) * w
    on = w > 0
    V[on] = pivot + (V[on] - pivot) * f[on, None]
    for k in np.where(on)[0]:
        v = list(P.vertices[k]); v[0] = [float(x) for x in V[k]]; P.vertices[k] = tuple(v)
    for m in P.morphs:
        if m['type'] == 1:
            m['offsets'] = [(o[0], [float(x * f[o[0]]) for x in o[1]]) if on[o[0]] else o for o in m['offsets']]
    for i in kids:
        b = P.bones[i]
        b['pos'] = [float(x) for x in pivot + (np.array(b['pos']) - pivot) * s]
        if 'tail_offset' in b:
            b['tail_offset'] = [float(x * s) for x in b['tail_offset']]
    bodies = set()
    for k, r in enumerate(P.rigids):
        if r['bone'] in kids:
            bodies.add(k)
            r['pos'] = [float(x) for x in pivot + (np.array(r['pos']) - pivot) * s]
            r['size'] = [float(x * s) for x in r['size']]
    for j in P.joints:
        if j['a'] in bodies or j['b'] in bodies:
            j['pos'] = [float(x) for x in pivot + (np.array(j['pos']) - pivot) * s]
    write_pmx(a.pmx, P)
    print(f'{os.path.basename(a.pmx)}: head scaled x{s:.3f} about {a.bone} at {np.round(pivot, 2).tolist()}: '
          f'{int(on.sum())} vertices ({int(((w > 0) & (w < 1)).sum())} shared with the neck), '
          f'{len(kids)} bones, {len(bodies)} rigid bodies')
    validate.check(a.pmx)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--pmx', required=True)
    p.add_argument('--scale', type=float, default=1.1)
    p.add_argument('--bone', default='頭', help='the bone whose subtree is the head')
    main(p.parse_args())
