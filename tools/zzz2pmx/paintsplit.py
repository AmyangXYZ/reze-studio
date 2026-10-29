# -*- coding: utf-8 -*-
"""Split clothing painted onto a skin material out into a material of its own, in place.

    python3 tools/zzz2pmx/paintsplit.py <model.pmx> <material> <new-name> <bone-pattern>
        [--above y] [--below y] [--small n] [--large n] [--whole] [--refine levels]
    python3 tools/zzz2pmx/paintsplit.py <model.pmx> --merge a,b,c --into skin

A model ripped from a game can come with one body material holding skin and the
clothes painted on it -- 海拉's lace gloves are drawn over her arm and hand islands --
and the studio shades the lot as skin. Among the triangles mostly weighted to bones
matching `bone-pattern`, those whose texels are mostly not skin move to `new-name`,
appended last so no material morph's index shifts. Both keep the same texture; only
the shading group changes.

`above` and `below` bound the triangles by height; `small` keeps to shells of at most
that many vertices -- a choker's pieces, not the torso the same neck bones carry --
and `large` to shells of more: a stocking painted on the leg, not the shoes on the
same ankle bones. `whole` takes every triangle selected, for clothing the paint does
not tell from skin -- 大梵天's white stockings are a shader mask, the skin's own colour
in the diffuse, and end under her black thigh band.

`refine` makes the material edge follow the paint. A triangle straddling the painted
edge -- a long leg triangle across a stocking's lace scallops -- goes wholly one way
or the other and leaks; each level halves every straddling triangle's edges, and the
neighbours split the same edges, so nothing cracks. Where an edge lies on a UV seam,
the edge of the same positions on the other side splits with it. New vertices sit on
the old surface, their weights blended from the edge's ends.

`--merge` folds split-off materials back into one, to split again.
"""
import argparse
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import PMX
from pmxw import write_pmx

# barycentric samples across a triangle
BARY = np.array([(a, b, 1 - a - b) for a in np.linspace(0.05, 0.9, 6) for b in np.linspace(0.05, 0.9, 6)
                 if a + b <= 0.95])


def skin(c):
    return (c[..., 0] > 150) & (c[..., 0] >= c[..., 1]) & (c[..., 1] >= c[..., 2]) & (c[..., 2] > 100) \
        & (c[..., 0] - c[..., 2] < 100)


def faces_by_material(m):
    F = np.array(m.faces).reshape(-1, 3)
    fm = np.repeat(np.arange(len(m.materials)), [x['fcount'] // 3 for x in m.materials])
    return F, fm


def set_faces(m, F, fm):
    order = np.argsort(fm, kind='stable')
    F, fm = F[order], fm[order]
    for k, x in enumerate(m.materials):
        x['fcount'] = int((fm == k).sum()) * 3
    m.faces = F.ravel().tolist()


def merge(path, names, into):
    m = PMX(path)
    F, fm = faces_by_material(m)
    at = {x['name']: k for k, x in enumerate(m.materials)}
    gone = sorted(at[n] for n in names)
    # only the last materials may go: a material morph indexes the ones before
    assert gone == list(range(len(m.materials) - len(gone), len(m.materials)))
    for k in gone:
        fm[fm == k] = at[into]
    m.materials = m.materials[:gone[0]]
    set_faces(m, F, fm)
    write_pmx(path, m)
    print(f'{path}: {", ".join(names)} merged back into `{into}`')


def midpoint(m, a, b):
    va, vb = m.vertices[a], m.vertices[b]
    w = {}
    for v in (va, vb):
        for bone, x in zip(v[5], v[6]):
            if x > 0:
                w[bone] = w.get(bone, 0.0) + x / 2
    top = sorted(w.items(), key=lambda kv: -kv[1])[:4]
    s = sum(x for _, x in top)
    bones = [b_ for b_, _ in top] + [0] * (4 - len(top))
    weights = [x / s for _, x in top] + [0.0] * (4 - len(top))
    n = np.array(va[1]) + np.array(vb[1])
    n = (n / (np.linalg.norm(n) + 1e-12)).tolist()
    pos = ((np.array(va[0]) + np.array(vb[0])) / 2).tolist()
    uv = ((np.array(va[2]) + np.array(vb[2])) / 2).tolist()
    add = [((np.array(x) + np.array(y)) / 2).tolist() for x, y in zip(va[3], vb[3])]
    m.vertices.append((pos, n, uv, add, 2, bones, weights, (va[7] + vb[7]) / 2))
    return len(m.vertices) - 1


def refine(m, F, fm, flag, split):
    """Halve the faces flagged in `split`: their edges, the same edges in every face that
    shares them, and the edges of the same positions across a UV seam."""
    V = np.round(np.array([v[0] for v in m.vertices]), 5)

    def key(a, b):
        return tuple(sorted((tuple(V[a]), tuple(V[b]))))

    want = set()
    for f in F[split]:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            want.add(key(a, b))
    mids = {}

    def mid(a, b):
        e = (min(a, b), max(a, b))
        if e not in mids:
            mids[e] = midpoint(m, a, b)
        return mids[e]

    outF, outm, outf = [], [], []
    for f, k, g in zip(F, fm, flag):
        a, b, c = (int(x) for x in f)
        cut = [key(a, b) in want, key(b, c) in want, key(c, a) in want]
        n = sum(cut)
        if n == 0:
            tris = [(a, b, c)]
        elif n == 3:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            tris = [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        elif n == 1:
            # rotate the cut edge to ab, keeping the winding
            r = cut.index(True)
            a, b, c = [(a, b, c), (b, c, a), (c, a, b)][r]
            ab = mid(a, b)
            tris = [(a, ab, c), (ab, b, c)]
        else:
            # rotate the uncut edge to ca: ab and bc are cut
            r = (cut.index(False) + 1) % 3
            a, b, c = [(a, b, c), (b, c, a), (c, a, b)][r]
            ab, bc = mid(a, b), mid(b, c)
            tris = [(ab, b, bc), (a, ab, bc), (a, bc, c)]
        outF += tris
        outm += [k] * len(tris)
        outf += [g] * len(tris)
    return np.array(outF), np.array(outm), np.array(outf), len(mids)


def main(path, mat, asname, bones, above=None, small=None, large=None, below=None, whole=False, refine_levels=0):
    m = PMX(path)
    base = os.path.dirname(path)
    k_mat = next(k for k, x in enumerate(m.materials) if x['name'] == mat)
    x = m.materials[k_mat]
    im = np.asarray(Image.open(os.path.join(base, m.textures[x['tex']].replace('\\', '/'))).convert('RGB'))
    h, w, _ = im.shape
    names = [b['name'] for b in m.bones]
    F, fm = faces_by_material(m)
    T = F[fm == k_mat]

    def top(i):
        v = m.vertices[int(i)]
        return names[v[5][int(np.argmax(v[6][:len(v[5])]))]]

    near = {int(i): bool(re.search(bones, top(i))) for i in np.unique(T)}
    on = np.array([sum(near[int(i)] for i in t) >= 2 for t in T])
    V = np.array([v[0] for v in m.vertices])
    if above is not None:
        on &= (V[T][:, :, 1] > float(above)).all(1)
    if below is not None:
        on &= (V[T][:, :, 1] < float(below)).all(1)
    if small is not None or large is not None:
        import roundring as rr
        size = {}
        for ids in rr.shells(V, T).values():
            for i in ids:
                size[int(i)] = len(ids)
        n = np.array([size[int(t[0])] for t in T])
        if small is not None:
            on &= n <= int(small)
        if large is not None:
            on &= n > int(large)
    flag = np.zeros(len(F), bool)
    flag[np.flatnonzero(fm == k_mat)[on]] = True

    def painted(F):
        UV = np.array([v[2] for v in m.vertices]) % 1.0
        uv = np.einsum('sk,tkd->tsd', BARY, UV[F])
        c = im[(uv[..., 1] * (h - 1)).astype(int), (uv[..., 0] * (w - 1)).astype(int)].astype(float)
        return 1.0 - skin(c).mean(1)

    added = 0
    for _ in range(0 if whole else int(refine_levels or 0)):
        p = painted(F)
        split = flag & (p > 0.1) & (p < 0.9)
        if not split.any():
            break
        F, fm, flag, n = refine(m, F, fm, flag, split)
        added += n
    cloth = flag if whole else flag & (painted(F) >= 0.5)
    new = dict(x)
    new['name'] = new['name_en'] = asname
    m.materials.append(new)
    fm = np.where(cloth, len(m.materials) - 1, fm)
    set_faces(m, F, fm)
    write_pmx(path, m)
    print(f'{path}: {int(cloth.sum())} of {int(flag.sum())} triangles on {bones} in `{mat}` '
          f'{"taken whole" if whole else "are painted cloth"} -> `{asname}`'
          + (f'; {added} vertices added along the paint' if added else ''))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('material', nargs='?')
    ap.add_argument('name', nargs='?')
    ap.add_argument('bones', nargs='?')
    ap.add_argument('--above', type=float)
    ap.add_argument('--below', type=float)
    ap.add_argument('--small', type=int)
    ap.add_argument('--large', type=int)
    ap.add_argument('--whole', action='store_true')
    ap.add_argument('--refine', type=int, default=0)
    ap.add_argument('--merge')
    ap.add_argument('--into')
    a = ap.parse_args()
    if a.merge:
        merge(a.model, a.merge.split(','), a.into)
    else:
        main(a.model, a.material, a.name, a.bones, a.above, a.small, a.large, a.below, a.whole, a.refine)
