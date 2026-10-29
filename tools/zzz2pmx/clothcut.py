# -*- coding: utf-8 -*-
"""Cut painted clothes out of a skin mesh, along the painted edge, at texel precision.

The bodysuit and the lace bands are painted onto the body's own triangles: the edge
of the cloth runs THROUGH triangles, and 1742 of the 2752 vertices the cloth touches
are shared with skin. There is no seam to split along, so one is made.

1. The sheet is classified where the skin material owns it -- warm texels are skin,
   neutral or cool bright ones are the bodysuit, dark ones the lace -- inside a box
   the clothes are known to occupy, with small islands dropped and small holes filled.
2. Each class becomes a smooth field: its mask blurred by a texel and a half, so the
   0.5 contour is the painted edge to within a texel and has no staircase.
3. Every triangle the contour crosses is subdivided, adaptively, until its longest
   UV edge is under `FINE` texels; the pieces on one side are kept whole, the pieces
   the contour still crosses are cut by marching triangles -- the crossing point on
   each edge from the field's values at its ends. Edge vertices are shared between
   the triangles on either side, and a coarser neighbour of a subdivided triangle is
   split through the midpoint it now has on its edge, so nothing is left hanging.
4. New vertices take position, normal, UV and weights interpolated along the edge
   they sit on. The pieces go to `cloth_bodysuit` and `lace`, on the same sheet with
   the same UVs; the skin keeps the rest.

    python3 tools/zzz2pmx/clothcut.py <model.pmx> [--out <model.pmx>]
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holefill
import skinclean
from pmx import PMX
from pmxw import write_pmx

BODY = 'skin'
BOX = dict(ylo=8.0, yhi=19.2, wide=2.6)      # where the clothes are: torso and thighs
WARM = 0.02          # R - B below this is cloth for certain
MAYBE = 0.09         # and below this it may be: pink-white satin, or a highlight on skin
DARK = 0.55          # luminance below this is the lace
BRIGHT = 0.55
ISLAND = 3000        # texels; a cloth island smaller is a highlight on skin
HOLE = 8000          # texels; a hole in cloth smaller is fabric shading, filled
CLOSE = 4            # texels of closing, so sheer speckle does not fray the edge
SIGMA = 2.0          # texels of blur that make the edge smooth
FINE = 5.0           # texels; a crossed triangle is split until its UV edges are under this
LEVELS = 5
CLASSES = [('cloth_bodysuit', 1), ('lace', 2)]
# Where the paint cannot decide, the model can. The satin under the bust is painted
# in her own skin tone -- warmth 0.086 against the calf's 0.095 -- and reads as cloth
# only under lighting. Texels owned by triangles in these boxes are cloth, said so.
FORCE = [dict(ylo=15.1, yhi=16.5, wide=1.15, zmax=-0.55)]


def classify(A, own, forced=None):
    R, B = A[..., 0], A[..., 2]
    lum = A.mean(2)
    sure = own & (R - B < WARM) & (lum > BRIGHT)
    if forced is not None:
        sure = sure | (forced & own)
    maybe = own & (R - B < MAYBE) & (lum > 0.82)
    # hysteresis: the ambiguous pink-white counts as cloth only where it is joined to
    # cloth that is certain -- the satin panel under the bust is, a highlight on a
    # shin is not
    lab, n = ndimage.label(maybe | sure)
    hit = np.zeros(n + 1, bool)
    hit[np.unique(lab[sure])] = True
    hit[0] = False
    cloth = hit[lab]
    lace = own & (lum < DARK)
    def tidy(mask):
        mask = ndimage.binary_closing(mask, iterations=CLOSE)
        mask = ndimage.binary_opening(mask, iterations=2)
        lab, n = ndimage.label(mask)
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= ISLAND
        mask = keep[lab]
        holes = ~mask
        lab, n = ndimage.label(holes)
        sizes = ndimage.sum(holes, lab, range(1, n + 1))
        small = np.zeros(n + 1, bool)
        small[1:] = sizes < HOLE
        return mask | small[lab]
    cloth = tidy(cloth)
    lace = tidy(lace) & ~cloth
    label = np.zeros(own.shape, np.uint8)
    label[cloth] = 1
    label[lace] = 2
    fields = {1: ndimage.gaussian_filter(cloth.astype(np.float32), SIGMA),
              2: ndimage.gaussian_filter(lace.astype(np.float32), SIGMA)}
    return label, fields


class Cutter:
    """Seam-aware. The skin mesh is split along its UV seams -- 546 boundary edges in
    the box by index, 20 by position -- so every decision that has to agree across a
    seam is made per CANONICAL vertex (welded by position): the class field's value,
    whether an edge was split, and where the contour crosses it. Each side still gets
    its own vertex, with its own UVs, at the shared position."""

    def __init__(self, m, label, fields, w, h, own):
        self.m = m
        self.w, self.h = w, h
        self.label = label
        self.fields = fields
        self.integral = {c: np.pad(((label == c) & own).astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
                         for c in (0, 1, 2)}
        self.V = [list(v) for v in m.vertices]
        key = {}
        self.canon = []
        for v in self.V:
            k = tuple(np.round(v[0], 4))
            self.canon.append(key.setdefault(k, len(key)))
        self.dups = {}
        for i, c in enumerate(self.canon):
            self.dups.setdefault(c, []).append(i)
        self.ncanon = len(key)
        self.fval = {}       # (canon, class) -> field value, shared across a seam
        self.mid = {}        # (i, j) -> midpoint vertex, per side
        self.midc = {}       # (ci, cj) -> canonical id of that midpoint
        self.cut = {}        # (i, j, class) -> crossing vertex, per side
        self.cutc = {}       # (ci, cj, class) -> (t, canonical id), shared
        self.fresh = set()

    def px(self, uv):
        return uv[0] % 1.0 * self.w - 0.5, uv[1] % 1.0 * self.h - 0.5

    def sample(self, field, uv):
        x, y = self.px(uv)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        g = lambda yy, xx: field[yy % self.h, xx % self.w]
        return ((g(y0, x0) * (1 - fx) + g(y0, x0 + 1) * fx) * (1 - fy)
                + (g(y0 + 1, x0) * (1 - fx) + g(y0 + 1, x0 + 1) * fx) * fy)

    def f(self, i, c):
        """The class field at a vertex -- one value for every copy of it."""
        ci = self.canon[i]
        k = (ci, c)
        if k not in self.fval:
            self.fval[k] = float(np.mean([self.sample(self.fields[c], self.V[d][2])
                                          for d in self.dups[ci]]))
        return self.fval[k]

    def box_classes(self, tri):
        P = np.array([self.px(self.V[i][2]) for i in tri])
        x0, y0 = np.floor(P.min(0)).astype(int) - 1
        x1, y1 = np.ceil(P.max(0)).astype(int) + 2
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, self.w), min(y1, self.h)
        if x0 >= x1 or y0 >= y1:
            return {0}
        out = set()
        for c, I in self.integral.items():
            if I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]:
                out.add(c)
        return out

    def longest(self, tri):
        P = np.array([self.px(self.V[i][2]) for i in tri])
        return max(np.linalg.norm(P[k] - P[(k + 1) % 3]) for k in range(3))

    def between(self, i, j, t, cid):
        a, b = self.V[i], self.V[j]
        lerp = lambda p, q: [float(x * (1 - t) + y * t) for x, y in zip(p, q)]
        wd = {}
        for bi, ww in zip(a[5], a[6]):
            if ww > 0:
                wd[bi] = wd.get(bi, 0.0) + ww * (1 - t)
        for bi, ww in zip(b[5], b[6]):
            if ww > 0:
                wd[bi] = wd.get(bi, 0.0) + ww * t
        top = sorted(wd.items(), key=lambda x: -x[1])[:4] or [(0, 1.0)]
        tot = sum(x[1] for x in top) or 1.0
        n = lerp(a[1], b[1])
        nn = float(np.linalg.norm(n)) or 1.0
        v = (lerp(a[0], b[0]), [x / nn for x in n], lerp(a[2], b[2]),
             [np.zeros(4) for _ in range(self.m.uv_add)], 2,
             [x[0] for x in top] + [0] * (4 - len(top)),
             [x[1] / tot for x in top] + [0.0] * (4 - len(top)), float(a[7]))
        self.V.append(list(v))
        k = len(self.V) - 1
        self.canon.append(cid)
        self.dups.setdefault(cid, []).append(k)
        # the field at the new vertex, from the shared values at the ends
        for c in self.fields:
            self.fval[(cid, c)] = (1 - t) * self.f(i, c) + t * self.f(j, c)
        self.fresh.add(k)
        return k

    def ckey(self, i, j):
        ci, cj = self.canon[i], self.canon[j]
        return (min(ci, cj), max(ci, cj))

    def midpoint(self, i, j):
        k = (min(i, j), max(i, j))
        if k not in self.mid:
            ck = self.ckey(i, j)
            if ck not in self.midc:
                self.midc[ck] = self.ncanon
                self.ncanon += 1
            self.mid[k] = self.between(k[0], k[1], 0.5, self.midc[ck])
        return self.mid[k]

    def crossing(self, i, j, c):
        k = (min(i, j), max(i, j), c)
        if k not in self.cut:
            ck = self.ckey(i, j) + (c,)
            lo, hi = k[0], k[1]
            if ck not in self.cutc:
                fi, fj = self.f(lo, c), self.f(hi, c)
                t = (0.5 - fi) / (fj - fi) if abs(fj - fi) > 1e-9 else 0.5
                t = min(max(t, 0.02), 0.98)
                # t runs from the lower canonical id to the higher, whichever side asks
                if self.canon[lo] > self.canon[hi]:
                    t = 1 - t
                self.cutc[ck] = (t, self.ncanon)
                self.ncanon += 1
            t, cid = self.cutc[ck]
            if self.canon[lo] > self.canon[hi]:
                t = 1 - t
            self.cut[k] = self.between(lo, hi, t, cid)
        return self.cut[k]

    def refine(self, tri, level=0):
        classes = self.box_classes(tri)
        if len(classes) <= 1 or level >= LEVELS or self.longest(tri) < FINE:
            return [tri]
        a, b, c = tri
        ab, bc, ca = self.midpoint(a, b), self.midpoint(b, c), self.midpoint(c, a)
        out = []
        for t in ([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]):
            out += self.refine(t, level + 1)
        return out

    def split(self, tri, c):
        f = [self.f(i, c) for i in tri]
        inside = [x >= 0.5 for x in f]
        n = sum(inside)
        if n == 0:
            return [], [tri]
        if n == 3:
            return [tri], []
        lone = inside.index(True) if n == 1 else inside.index(False)
        a, b, cc = tri[lone], tri[(lone + 1) % 3], tri[(lone + 2) % 3]
        p = self.crossing(a, b, c)
        q = self.crossing(a, cc, c)
        one = [[a, p, q]]
        two = [[p, b, cc], [p, cc, q]]
        return (one, two) if n == 1 else (two, one)

    def balance(self, tris):
        """Split any triangle whose edge has been split anywhere -- on this side or,
        across a UV seam, on the other."""
        changed = True
        while changed:
            changed = False
            out = []
            for lab, tri in tris:
                a, b, c = tri
                edges = [(a, b), (b, c), (c, a)]
                hit = [self.ckey(i, j) in self.midc for i, j in edges]
                if not any(hit):
                    out.append((lab, tri))
                    continue
                changed = True
                k = hit.index(True)
                i, j = edges[k]
                mth = self.midpoint(i, j)
                third = tri[(k + 2) % 3]
                out.append((lab, [i, mth, third]))
                out.append((lab, [mth, j, third]))
            tris = out
        return tris

    def run(self, tris, inbox):
        fine = []
        for tri, ok in zip(tris, inbox):
            tri = [int(x) for x in tri]
            fine += [(True, t) for t in self.refine(tri)] if ok else [(False, tri)]
        # balance carries the in-box flag; a triangle outside the box is skin, whatever
        # its paint says -- the hands' highlights are cloth by colour
        fine = self.balance([(ok, t) for ok, t in fine])
        pieces = []
        for ok, sub in fine:
            # outside the box the pieces are all skin, but the triangle is still split
            # where its neighbour inside the box was, or the edge between them cracks
            rest = [sub]
            for name, c in CLASSES:
                nxt = []
                for t in rest:
                    ins, outs = self.split(t, c)
                    pieces += [(c if ok else 0, x) for x in ins]
                    nxt += outs
                rest = nxt
            pieces += [(0, x) for x in rest]
        return pieces


def main(path, out=None):
    out = out or path
    m = PMX(path)
    sl = holefill.slices(m)
    V = np.array([v[0] for v in m.vertices])
    UV = np.array([v[2] for v in m.vertices])
    here = next(i for i, x in enumerate(m.materials) if x['name'] == BODY)
    mat = m.materials[here]
    sheet = os.path.join(os.path.dirname(path), m.textures[mat['tex']].replace('\\', '/'))
    A = np.asarray(Image.open(sheet).convert('RGB'), np.float32) / 255
    h, w, _ = A.shape
    T = np.array(sl[BODY])
    own, _ = skinclean.cover(m, T, UV, w, h)
    c = V[T].mean(1)
    forced = np.zeros_like(own)
    for f in FORCE:
        sel = T[(c[:, 1] > f['ylo']) & (c[:, 1] < f['yhi']) & (np.abs(c[:, 0]) < f['wide'])
                & (c[:, 2] < f['zmax'])]
        if len(sel):
            forced |= skinclean.cover(m, sel, UV, w, h)[0]
    label, fields = classify(A, own, forced)
    inbox = (c[:, 1] > BOX['ylo']) & (c[:, 1] < BOX['yhi']) & (np.abs(c[:, 0]) < BOX['wide'])
    cutter = Cutter(m, label, fields, w, h, own)
    pieces = cutter.run(T, inbox)
    m.vertices = [tuple(v) for v in cutter.V]

    F = np.array(m.faces).reshape(-1, 3)
    at, blocks = 0, []
    for x in m.materials:
        n = x['fcount'] // 3
        blocks.append(F[at:at + n])
        at += n
    blocks[here] = np.array([t for lab, t in pieces if lab == 0], int).reshape(-1, 3)
    for name, cid in CLASSES:
        tris = np.array([t for lab, t in pieces if lab == cid], int).reshape(-1, 3)
        if not len(tris):
            continue
        new = dict(mat)
        new['name'] = new['name_en'] = name
        m.materials.append(new)
        blocks.append(tris)
    for x, b in zip(m.materials, blocks):
        x['fcount'] = len(b) * 3
    m.faces = [int(v) for v in np.concatenate(blocks).reshape(-1)]
    write_pmx(out, m)
    print(f'{out}: {BODY} {len(T)} -> {len(blocks[here])} triangles; '
          + ', '.join(f'`{n}` {len(b)}' for (n, _), b in zip(CLASSES, blocks[len(blocks) - len(CLASSES):]))
          + f'; {len(cutter.fresh)} vertices added, {len(cutter.cut)} of them on the painted edge')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--out')
    a = ap.parse_args()
    main(a.model, a.out)
