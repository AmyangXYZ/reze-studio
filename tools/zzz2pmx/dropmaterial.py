# -*- coding: utf-8 -*-
"""Remove a material from a model.

Removing a material means removing its slice of the face list and renumbering
whatever indexes materials after it -- material morphs, which is the part a manual
edit forgets. Vertices are left in place: nothing draws them once their faces are
gone, and dropping them would mean renumbering every vertex morph in the file.

    python3 tools/zzz2pmx/dropmaterial.py <pmx> <material name>
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pmx import PMX
from pmxw import write_pmx


def main(path, name):
    m = PMX(path)
    at, target = 0, None
    for i, mat in enumerate(m.materials):
        n = mat['fcount'] // 3
        if mat['name'] == name:
            target, start, count = i, at, n
        at += n
    if target is None:
        raise SystemExit(f'no material named {name}; have '
                         f'{[x["name"] for x in m.materials]}')

    before_faces, before_mats = len(m.faces) // 3, len(m.materials)
    drawn_before = len(np.unique(np.array(m.faces)))

    flat = list(m.faces)
    del flat[3 * start:3 * (start + count)]
    m.faces = flat
    del m.materials[target]

    moved = []
    for k, mo in enumerate(m.morphs):
        if mo['type'] != 8:
            continue
        kept = []
        for o in mo['offsets']:
            o = list(o)
            if o[0] == target:
                continue
            if o[0] > target:
                o[0] -= 1
            kept.append(tuple(o))
        if kept != mo['offsets']:
            moved.append(k)
        mo['offsets'] = kept

    drawn_after = len(np.unique(np.array(m.faces)))
    write_pmx(path, m)
    print(f'dropped {name} (material {target}): {count} tris')
    print(f'  faces     {before_faces} -> {len(m.faces)//3}')
    print(f'  materials {before_mats} -> {len(m.materials)}')
    print(f'  vertices drawn {drawn_before} -> {drawn_after} '
          f'({drawn_before - drawn_after} now unreferenced, left in place)')
    for k in moved:
        print(f'  material morph {k} {m.morphs[k]["name"]} renumbered')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
