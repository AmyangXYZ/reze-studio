# -*- coding: utf-8 -*-
"""Put our head on the base mannequin.

`headgraft.py` expects the host to wear its face in a material of its own, and this
one does not: the base is a single watertight shell with the face welded into it. So
the face is cut back out first, against the mesh it came from -- 海滩漫步's `face`,
which our `face01` matches vertex for vertex to a median of 0.0008 after an offset of
0.883 in y, because both are the same 深空之眼 base head.

Cut at a hundredth, the host is left holding four rims, and three of them -- 26, 30
and 26 edges, the eyes and the mouth -- are exactly the rims our face arrives with.
The fourth is not: the base rebuilt the scalp, so its outer boundary runs 78 edges
against our 64, and that one is sewn.

    python3 tools/zzz2pmx/selenehead.py

The body is painted last, after the neck is sewn, because `selenepaint` gives every
face of the body its own cell in the sheet it bakes -- and run first it has no cell
for the 202 triangles that do not exist yet. Painted first, those came out sampling
across other faces' cells: 0.32% of the body, all of it around the collar.

Reads `Selene_base_fitted.pmx` and never writes it. Output is `reze-bare.pmx` in
`public/models/reze-bare/`.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holefill
import selenepaint
from pmx import PMX
from pmxw import write_pmx

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = 'public/models/base_selene/Selene_base_fitted.pmx'
FACE = 'public/models/【深空之眼】幽月·塞勒涅-海滩漫步/海滩漫步.pmx'
HEAD = 'public/models/reze/reze.pmx'
OUT = 'public/models/reze-bare/reze-bare.pmx'
TEXDIR = 'public/models/reze-bare/textures'

BODY = 'Body_Mannequin'     # `selenepaint` renames it afterwards
TOL = 0.01          # how close to her face a vertex has to be to be part of it
DROP = ['face01', 'Lips', 'Tongue', 'Mouth', 'Teeth', 'Brows', 'Lashes',
        'Eyes_White', 'Iris_L', 'Iris_R', 'Hair_F', 'Hair_B']


def cut_face(path, out):
    """Take the face back out of the shell it was welded into."""
    m = PMX(path)
    f = PMX(FACE)
    V = np.array([v[0] for v in m.vertices])
    FV = np.array([v[0] for v in f.vertices])
    tree = cKDTree(FV[np.unique(holefill.slices(f)['face'])])
    d, _ = tree.query(V)
    F = np.array(m.faces).reshape(-1, 3)
    at, keep, face = 0, [], None
    for mat in m.materials:
        k = mat['fcount'] // 3
        tris = F[at:at + k]
        at += k
        if mat['name'] == BODY:
            hers = (d[tris] < TOL).all(1)
            face = tris[hers]
            tris = tris[~hers]
            mat['fcount'] = len(tris) * 3
        keep.append(tris)
    src = next(x for x in m.materials if x['name'] == BODY)
    mat = dict(src)
    mat['name'] = mat['name_en'] = 'face01'
    mat['fcount'] = len(face) * 3
    m.materials.append(mat)
    keep.append(face)
    m.faces = np.concatenate(keep).ravel().tolist()
    write_pmx(out, m)
    print(f'  cut {len(face)} triangles of face out of the shell')


def sew_neck(path, limit=0.35):
    """Close what is left between our face and her scalp."""
    m = PMX(path)
    V = np.array([v[0] for v in m.vertices])
    names = [x['name'] for x in m.materials]
    body = holefill.slices(m)[BODY]
    ours = holefill.tris_of(m, ['face01']) if 'face01' in names else None
    if ours is None:
        return
    theirs = [l for l in holefill.boundary_loops(V, body) if len(l) >= 6]
    mine = [l for l in holefill.boundary_loops(V, ours) if len(l) >= 6]
    add, sewn = [], 0
    for a in theirs:
        ca = V[a].mean(0)
        near = [l for l in mine if np.linalg.norm(V[l].mean(0) - ca) < limit]
        if not near:
            continue
        bl = min(near, key=lambda l: np.linalg.norm(V[l].mean(0) - ca))
        mine.remove(bl)
        if abs(len(a) - len(bl)) < 1 and np.linalg.norm(V[a].mean(0) - V[bl].mean(0)) < 1e-3:
            continue
        A, B = list(a), list(bl)
        ax = V[B].mean(0) - V[A].mean(0)
        n = np.linalg.norm(ax)
        ax = ax / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        e1 = np.cross(ax, [0.0, 1.0, 0.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(ax, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax, e1)
        turn = lambda L: np.sum(np.diff(np.unwrap(np.arctan2(
            (V[L] - V[L].mean(0)) @ e2, (V[L] - V[L].mean(0)) @ e1))))
        if turn(A) * turn(B) < 0:
            B.reverse()
        k = int(np.argmin(np.linalg.norm(V[B] - V[A[0]], axis=1)))
        B = B[k:] + B[:k]
        for t3 in holefill._bridge(A, B, V):
            p, q, r = t3
            want = sum(np.array(m.vertices[int(x)][1], float) for x in t3)
            if float(np.dot(np.cross(V[q] - V[p], V[r] - V[p]), want)) < 0:
                q, r = r, q
            add.append([p, q, r])
            sewn += 1
    if not add:
        print('  nothing left to sew at the neck')
        return
    F = np.array(m.faces).reshape(-1, 3)
    at, out = 0, []
    for mat in m.materials:
        k = mat['fcount'] // 3
        tris = F[at:at + k]
        at += k
        if mat['name'] == BODY:
            tris = np.concatenate([tris, np.array(add, int)])
            mat['fcount'] = len(tris) * 3
        out.append(tris)
    m.faces = np.concatenate(out).ravel().tolist()
    write_pmx(path, m)
    print(f'  sewed {sewn} triangles between our face and her scalp')


def transform():
    """What carries our face onto hers: they are the same mesh, so it is exact."""
    f, r = PMX(FACE), PMX(HEAD)
    FV = np.array([v[0] for v in f.vertices])[np.unique(holefill.slices(f)['face'])]
    RV = np.array([v[0] for v in r.vertices])[np.unique(holefill.slices(r)['face01'])]
    off = FV.mean(0) - RV.mean(0)
    d, _ = cKDTree(FV).query(RV + off)
    print(f'  our face onto hers: offset {np.round(off, 4)}, '
          f'residual median {np.median(d):.4f} max {d.max():.4f}')
    return off


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    work = OUT + '.building'
    print('--- face out')
    cut_face(BASE, work)
    print('--- head in')
    off = transform()
    subprocess.run([sys.executable, os.path.join(HERE, 'headgraft.py'),
                    '--host', work, '--head', HEAD, '--out', work,
                    '--drop', ','.join(DROP), '--scale', '1.0',
                    '--offset=' + ','.join(f'{x:.6f}' for x in off)], check=True)
    print('--- neck')
    sew_neck(work)
    print('--- paint')
    selenepaint.main(work, work, TEXDIR)
    os.replace(work, OUT)
    m = PMX(OUT)
    print(f'{OUT}: {len(m.vertices)} verts, {len(m.faces) // 3} triangles, '
          f'{len(m.bones)} bones, {len(m.morphs)} morphs')
    print(f'  materials: {", ".join(x["name"] for x in m.materials)}')


if __name__ == '__main__':
    main()
