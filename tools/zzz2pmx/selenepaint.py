# -*- coding: utf-8 -*-
"""Paint the base mannequin out of the models it was built from.

It ships untextured, and its own UVs are not a basis for texturing: they are whatever
each donor vertex arrived carrying, two atlases laid over each other in the same
square, and the rebuilt parts -- the scalp, the abdomen, the feet -- carry nothing
meaningful at all. Only 56% of its 8360 body vertices can be traced back to a donor
by position and UV together.

So the UVs are not recovered, they are re-read off the donor surface underneath, which
works for a rebuilt vertex exactly as well as for a carried one. 塞勒涅 answers
wherever she has skin, since the model is hers; 海滩漫步 answers everywhere else.

Read a corner at a time it comes apart. An atlas is cut into islands, and the nearest
point to one corner of a face can land across a cut from the nearest point to the
next -- so the face stretches the whole sheet between them, and what is painted on the
sheet between those islands lands on the body as a torn scrap of lace.

So a face is read as a face: one donor triangle answers for all three of its corners,
and a vertex whose faces want different answers is split so each can carry its own.
Nothing on the sheet is repainted to make this work. An earlier pass scrubbed the
garment texels out and replaced them with the nearest skin, and that put rectangular
smears across the stomach -- the sheets are clean where the body actually reads them,
and the tearing was never the sheet's fault.

    python3 tools/zzz2pmx/selenepaint.py

Reads `Selene_base_fitted.pmx` and never writes it. Output is `Selene_base_skin.pmx`
beside it, with `textures/body_skin.png`.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holefill
from pmx import PMX
from pmxw import write_pmx
from surface import Mesh

BASE = 'public/models/base_selene/Selene_base_fitted.pmx'
OUT = 'public/models/base_selene/Selene_base_skin.pmx'
TEXDIR = 'public/models/base_selene/textures'
SHEET = 'body_skin.png'

BODY = 'Body_Mannequin'
RENAME = 'skin'      # what the studio's material classifier looks for
# In order. She is the model, so she is asked first and only what she has no skin for
# falls through -- her calves, her arms below the glove, her feet inside her boots.
DONORS = [('塞勒涅', 'public/models/塞勒涅/塞勒涅.pmx', ['皮肤'], '衣服.png'),
          ('海滩漫步', 'public/models/【深空之眼】幽月·塞勒涅-海滩漫步/海滩漫步.pmx',
           ['skin'], 'textures/cloth01.png'),
          ('沉醉旖旎', 'public/models/【深空之眼】幽月·塞勒涅-沉醉旖旎_by_Chiyuri_'
           '6dbfe0ca0200e0a1f88bbcd4d159cd48/沉醉旖旎.pmx', ['skin'],
           'textures/cloth01.png'),
          # Not 塞勒涅, but the same 深空之眼 base body, and between them they are the
          # only two with bare skin across the bust and the hip. The three above all
          # fail there at once and for opposite reasons: 塞勒涅 has the geometry and
          # her costume painted onto it, and the other two wear theirs as geometry so
          # the skin underneath was cut away. 22% of the body had no donor at all.
          ('午夜沉沦', 'public/models/【深空之眼】裁暗之锋·塞尔凯特-午夜沉沦/'
           '午夜沉沦.pmx', ['skin'], 'textures/cloth01.png'),
          ('reze', 'public/models/reze/reze.pmx', ['skin'],
           'textures/cloth01.png')]
REACH = 0.6         # further than this from a donor and it is not the same surface
SOFT = 0.12         # donors are mixed over this distance rather than chosen between

# A 深空之眼 skin sheet is not all skin. 塞勒涅's costume is painted into hers rather
# than modelled, so her chest reads 42% darker than skin even where the geometry is
# bare. 海滩漫步's swimsuit is geometry, so what is painted under it is a body.
#
# Neither can be scrubbed and reused -- replacing a garment texel with the nearest
# skin texel is a repair in atlas space, and it came back as rectangular smears across
# the stomach, because what is next to a strap on a sheet is not what is next to it on
# a body. So the garment samples are simply dropped, and the point being painted takes
# the nearest surface that a donor left bare. That is a repair in space, where
# next-to means next-to.
TONE = np.array([0.957, 0.855, 0.804])     # her skin, off the sheet
TOLER = 0.32        # how far from it a sample may be and still be skin
DARK = 0.33         # shading runs a long way down from the flat tone and is skin
BLANK = 0.97        # unpainted atlas is white, and white is not skin either
SIZE = 4096         # the sheet baked for the body
GUTTER = 2          # texels of margin round each cell, bled outward afterwards
DENSITY = 24        # sample points per donor triangle


def _ramp(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def skinness(c):
    """How much a sample reads as skin, from 0 to 1 rather than yes or no.

    A hard yes-or-no puts a hard edge in the result. Along the border of anything
    painted on a sheet the colour ramps from skin into the paint, so a texel on one
    side takes the donor's own colour and its neighbour a tenth of a millimetre away
    takes an average of samples fetched from somewhere else entirely: 22.6% of
    neighbouring triangles stepped visibly, most of it round the waist where 塞勒涅's
    costume is painted. Graded, the same border crosses over instead.
    """
    c = np.atleast_2d(np.asarray(c, float))
    if c.max() > 1.5:
        c = c / 255.0
    warm = _ramp(c[..., 0] - c[..., 2], 0.005, 0.055)
    order = _ramp(c[..., 0] - c[..., 1], -0.06, 0.0) * _ramp(c[..., 1] - c[..., 2], -0.08, 0.0)
    lit = _ramp(c[..., 0], 0.22, 0.42)
    blank = 1.0 - _ramp(c.min(-1), 0.93, 0.985)
    near = _ramp(TOLER - np.linalg.norm(c - TONE, axis=-1), 0.0, 0.08)
    return np.maximum(warm * order * lit, near) * blank


def bare(c):
    """Whether a sample reads as skin rather than as something painted onto it.

    The test is on hue, not on brightness. Skin is warm and ordered -- red over green
    over blue -- all the way from the lit side of a shoulder down into the shadow
    under a breast, and a sheet's own shading is most of what makes a body read as
    one. Cut at brightness instead and the shading goes with the costume: held to
    0.62 this dropped half of 塞勒涅's sheet and flattened her.

    What is rejected is what is not warm at any brightness -- black lace, grey straps,
    coloured trim -- and what is blank. Unpainted atlas is pure white, which is
    ordered, warm by every margin and bright, and it came back as the median colour
    of the thigh.
    """
    c = np.atleast_2d(np.asarray(c, float))
    if c.max() > 1.5:
        c = c / 255.0
    warm = ((c[..., 0] >= c[..., 1] - 0.01) & (c[..., 1] >= c[..., 2] - 0.03)
            & (c[..., 0] - c[..., 2] > 0.035))
    blank = c.min(-1) > BLANK
    return ~blank & (warm & (c[..., 0] > DARK)
                     | (np.linalg.norm(c - TONE, axis=-1) < TOLER))


def layout(n, size=SIZE, gutter=GUTTER):
    """A cell per triangle, and the UVs that put each one in its own.

    Not an unwrap. An unwrap would be nicer to edit and is not needed to bake: what
    matters is only that no two triangles share texels, and a grid guarantees that
    without anyone having to cut seams or pack islands.
    """
    across = int(np.ceil(np.sqrt(n)))
    cell = size // across
    if cell < 2 * gutter + 3:
        raise SystemExit(f'selenepaint: {n} triangles will not fit {size}px')
    r, c = np.divmod(np.arange(n), across)
    x0, y0 = c * cell + gutter, r * cell + gutter
    w = cell - 2 * gutter - 1
    px = np.stack([np.stack([x0, y0], 1),
                   np.stack([x0 + w, y0], 1),
                   np.stack([x0, y0 + w], 1)], 1).astype(np.float64)
    uv = np.empty_like(px)
    uv[..., 0] = (px[..., 0] + 0.5) / size
    uv[..., 1] = 1.0 - (px[..., 1] + 0.5) / size
    print(f'  {n} triangles in a {across}x{across} grid of {cell}px cells')
    return px, uv, cell


def donor_surface(donors):
    """Each donor as a surface that can be asked its colour anywhere on itself."""
    from PIL import Image
    out = []
    for tag, path, mats, tex in donors:
        d = PMX(path)
        V = np.array([v[0] for v in d.vertices])
        T = holefill.tris_of(d, mats)
        A = np.asarray(Image.open(os.path.join(
            os.path.dirname(path), tex.replace('\\', '/'))).convert('RGB'))
        out.append((tag, Mesh(V, T), np.array([v[2] for v in d.vertices]),
                    A.astype(np.float32) / 255))
        print(f'  {tag}: {len(T)} triangles, sheet {A.shape[1]}x{A.shape[0]}')
    return out


def look(A, uv):
    """The colour at a UV, filtered rather than snapped to a texel."""
    # PMX puts v = 0 on the sheet's top row, so v maps to the row directly. This
    # read `1 - v` for a long time, and every donor was sampled upside down.
    h, w, _ = A.shape
    x = uv[:, 0] % 1.0 * w - 0.5
    y = uv[:, 1] % 1.0 * h - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x0 %= w
    y0 %= h
    x1, y1 = (x0 + 1) % w, (y0 + 1) % h
    return ((A[y0, x0] * (1 - fx) + A[y0, x1] * fx) * (1 - fy)
            + (A[y1, x0] * (1 - fx) + A[y1, x1] * fx) * fy)


def on(mesh, uvs, A, P):
    """Where P lands on a donor, and what is painted there."""
    d, t, bar = mesh.closest(P)
    f = mesh.F[t]
    uv = (uvs[f[:, 0]] * bar[:, :1] + uvs[f[:, 1]] * bar[:, 1:2]
          + uvs[f[:, 2]] * bar[:, 2:3])
    return d, look(A, uv)


def spread(nv, tris, col, conf, rounds=96):
    """Carry colour from the vertices that found some to the vertices that did not.

    Across the surface, never through the atlas and never through space -- the nearest
    bare sample to a patch of the seat can be under an arm, and reaching for it filled
    a fifth of the body with blocks of the wrong tan.

    Held on vertices rather than faces so the fill can be interpolated across a
    triangle like anything else. Held on faces it is one colour per face, and a face
    is a whole cell of the sheet: 57% of the body came out as flat quads, which is
    the mosaic whatever colour they are.

    Every round is one pass over the edge list, so the whole surface advances at once
    rather than a vertex at a time.
    """
    E = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    E = np.concatenate([E, E[:, ::-1]]).astype(np.int64)
    E = E[np.unique(E[:, 0] * nv + E[:, 1], return_index=True)[1]]
    dst, src = E[:, 0], E[:, 1]
    deg = np.bincount(dst, minlength=nv).astype(np.float32)
    c, k = col.copy().astype(np.float32), conf.copy().astype(np.float32)
    for _ in range(rounds):
        poor = k <= 0.25
        if not poor.any():
            break
        w = k[src]
        den = np.bincount(dst, w, minlength=nv).astype(np.float32)
        take = poor & (den > 1e-6)
        if not take.any():
            break
        num = np.stack([np.bincount(dst, c[src, i] * w, minlength=nv)
                        for i in range(3)], 1).astype(np.float32)
        c[take] = num[take] / den[take, None]
        k[take] = np.minimum(0.3, den[take] / np.maximum(deg[take], 1) * 0.95)
    return c


def bake(m, donors):
    """Paint each body texel with the colour of the donor surface behind it."""
    V = np.array([v[0] for v in m.vertices])
    tris = holefill.slices(m)[BODY]
    px, uv, cell = layout(len(tris))
    surf = donor_surface(donors)

    g = np.arange(cell)
    gy, gx = np.meshgrid(g, g, indexing='ij')
    w = cell - 2 * GUTTER - 1
    u = (gx.ravel() - GUTTER) / max(w, 1)
    v = (gy.ravel() - GUTTER) / max(w, 1)
    gx, gy = gx.ravel(), gy.ravel()
    # Every texel of every cell, including the half outside the triangle. Painting
    # only the triangle leaves the rest of the cell black, and a renderer filters
    # across it: the body came back wearing the grid, a dark wedge on every face.
    bary = np.stack([1 - u - v, u, v], 1)

    n = len(tris)
    col = np.zeros((n, len(bary), 3), np.float32)
    conf = np.zeros((n, len(bary)), np.float32)
    step = max(1, 600000 // len(bary))
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        P = np.einsum('kc,fcd->fkd', bary, V[tris[lo:hi]]).reshape(-1, 3)
        acc = np.zeros((len(P), 3), np.float32)
        wsum = np.zeros(len(P), np.float32)
        for k, (tag, mesh, uvs, A) in enumerate(surf):
            d, c = on(mesh, uvs, A, P)
            ww = (np.where(d < REACH, np.exp(-(d / SOFT) ** 2), 0.0)
                  * skinness(c)).astype(np.float32)
            acc += c * ww[:, None]
            wsum += ww
        good = wsum > 1e-6
        acc[good] /= wsum[good, None]
        col[lo:hi] = acc.reshape(hi - lo, len(bary), 3)
        conf[lo:hi] = wsum.reshape(hi - lo, len(bary))

    # what each vertex knows, from the texels of the faces around it
    nv = len(V)
    vc = np.zeros((nv, 3), np.float32)
    vk = np.zeros(nv, np.float32)
    fc = (col * conf[..., None]).sum(1)
    fk = conf.sum(1)
    for c in range(3):
        np.add.at(vc, tris[:, c], fc)
        np.add.at(vk, tris[:, c], fk)
    good = vk > 1e-6
    vc[good] /= vk[good, None]
    vk = vk / max(len(bary), 1)
    print(f'  {100 * (conf.mean(1) <= 0.05).mean():.0f}% of faces found no skin '
          f'under them, {100 * (vk <= 0.25).mean():.0f}% of vertices')
    vc = spread(nv, tris, vc, vk)

    # a texel crosses over to the surface around it as its own confidence runs out,
    # interpolated across the triangle rather than flat over it
    fill = np.einsum('kc,fcd->fkd', bary, vc[tris])
    mix = np.clip(conf / 0.35, 0.0, 1.0)[..., None]
    col = col * mix + fill * (1 - mix)

    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    mask = np.zeros((SIZE, SIZE), bool)
    X = (px[:, 0, 0][:, None] - GUTTER + gx[None]).astype(int)
    Y = (px[:, 0, 1][:, None] - GUTTER + gy[None]).astype(int)
    np.clip(X, 0, SIZE - 1, out=X)
    np.clip(Y, 0, SIZE - 1, out=Y)
    img[Y.ravel(), X.ravel()] = np.clip(col.reshape(-1, 3) * 255 + 0.5, 0, 255).astype(np.uint8)
    mask[Y.ravel(), X.ravel()] = True
    return img, mask, uv


def bleed(img, mask):
    """Fill every texel nothing was painted into, from the nearest that was.

    The cells tile the sheet but do not finish it -- a grid of 124 across leaves a
    strip down two sides -- and a renderer reaching a lower mip level averages over
    it. Left black that strip is a dark edge the model wears.
    """
    from scipy import ndimage
    if mask.all():
        return img
    _, (yy, xx) = ndimage.distance_transform_edt(~mask, return_indices=True)
    img[~mask] = img[yy[~mask], xx[~mask]]
    print(f'  filled {100 * (~mask).mean():.1f}% of the sheet from its neighbours')
    return img


def main(src=BASE, dst=OUT, texdir=TEXDIR):
    global TEXDIR
    TEXDIR = texdir
    m = PMX(src)
    img, mask, uv = bake(m, DONORS)
    img = bleed(img, mask)
    os.makedirs(TEXDIR, exist_ok=True)
    from PIL import Image
    Image.fromarray(img).save(os.path.join(TEXDIR, SHEET))
    print(f'  {SHEET}: {SIZE}x{SIZE}, {100 * mask.mean():.0f}% painted')

    tris = holefill.slices(m)[BODY]
    new = []
    for fi, t in enumerate(tris):
        row = []
        for c in range(3):
            v = list(m.vertices[int(t[c])])
            v[2] = [float(uv[fi, c, 0]), float(uv[fi, c, 1])]
            m.vertices.append(tuple(v))
            row.append(len(m.vertices) - 1)
        new.append(row)
    rel = 'textures\\' + SHEET
    if rel not in m.textures:
        m.textures.append(rel)
    F = np.array(m.faces).reshape(-1, 3)
    at, out = 0, []
    for mat in m.materials:
        k = mat['fcount'] // 3
        block = F[at:at + k]
        at += k
        if mat['name'] == BODY:
            block = np.array(new, int)
            mat['fcount'] = len(block) * 3
            mat['tex'] = m.textures.index(rel)
            mat['diffuse'] = [1.0, 1.0, 1.0, 1.0]
            mat['name'] = mat['name_en'] = RENAME
        out.append(block)
    m.faces = np.concatenate(out).ravel().tolist()
    write_pmx(dst, m)
    print(f'{dst}: {len(m.vertices)} verts, {len(m.faces) // 3} triangles, '
          f'body material renamed `{RENAME}`')


if __name__ == '__main__':
    main(*sys.argv[1:])
