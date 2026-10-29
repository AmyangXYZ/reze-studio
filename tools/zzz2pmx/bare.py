# -*- coding: utf-8 -*-
"""A complete bare body, to dress in whatever comes next.

塞勒涅, made whole. Her body, her breast, her hips, her foot in her own shoes, her rig
-- 463 bones and 356 rigid bodies with a four-link breast chain -- and her own sheet.
Nothing of hers is cut. What her costume took with it is put back, each piece by the
one method that fits it, and a check runs after every step listing every open rim on
the body that is not a fingertip; the build is not done until that list is empty.

**Belly.** Her top was modelled, so the front of the torso between the underbust and
the waist is a hole, with a slit 0.2 wide between its top edge and the bottom of the
breast shell. 海滩漫步's surface inside the hole is brought onto her frame and sewn to
the hole's rim, and the slit is zipped. The fill's UVs are interpolated from her own
rims into her own atlas, which -- read the right way up -- is clean skin under the
hole. One material, one sheet.

**Hands.** Hers are gloves and went with the costume; 海滩漫步's bare ones come across
at the wrist, the way the maid build did it. The arm stays hers.

**Hair.** Her ornament chains are named `e_*`, `yb_*`, `zb_*` -- the same fifteen names
reze's hair hangs on. Left as they are, the head graft sees the names, carries nothing,
and reze's hair ends up driven by her ornaments. They are renamed first.

**Neck.** The head graft places the head and drops her face; it does not sew, and
there is nothing to sew: the face is a mask over her closed neck, and its jaw and
scalp rims are free edges. Left where they are they show as a step. They are tucked
a little under her skin, and the edge disappears into the neck.

**Paint.** Her sheet is 100% skin nearly everywhere and 76% at the hips -- her painted
underwear, kept. The sacral mark, which no donor has cleaner, is filled from her own
skin around it.

    python3 tools/zzz2pmx/bare.py

Reads the donors and never writes them. Output is `public/models/reze-bare/`.
"""
import os
import re
import shutil
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixrig
import holefill
import maidbuild
import maidselene
import progress
import publish
import skinclean
from pmx import PMX
from pmxw import write_pmx
from surface import Mesh

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = 'public/models/reze-bare'
WORK = f'{DIR}/_work'
BUILD = f'{WORK}/reze-bare.pmx'
OUT = f'{DIR}/reze-bare.pmx'

DONOR = ('海滩漫步', 'public/models/【深空之眼】幽月·塞勒涅-海滩漫步/海滩漫步.pmx', 'skin')
KEEP = {'皮肤': 'skin', '脸': 'face01', '鞋子': 'shoes'}
BODY = ('skin', 'skin_hands', 'face01', 'shoes')

BELLY = dict(ylo=14.3, yhi=16.35, zmax=0.1, wide=1.7)
HOLE = 0.15          # donor surface further than this off hers is over the hole
ZIP = 0.35           # two rims closer than this, vertex for vertex, are zipped
SEW = 0.5            # a rim pairs with the one it lies along, by median distance
CHAIN = re.compile(r'^(e|yb|zb)_\d+_\d+$')


def blocks(m):
    F = np.array(m.faces).reshape(-1, 3)
    at, out = 0, []
    for x in m.materials:
        n = x['fcount'] // 3
        out.append(F[at:at + n])
        at += n
    return out


def relay(m, bl):
    for x, b in zip(m.materials, bl):
        x['fcount'] = len(b) * 3
    m.faces = [int(v) for v in np.concatenate(
        [np.asarray(b, int).reshape(-1, 3) for b in bl]).reshape(-1)]


def weights(m, d, i):
    hb = {b['name']: k for k, b in enumerate(m.bones)}
    dn = [b['name'] for b in d.bones]
    v = d.vertices[int(i)]
    wd = {}
    for bi, ww in zip(v[5], v[6]):
        if ww > 1e-5 and 0 <= bi < len(dn) and dn[bi] in hb:
            wd[hb[dn[bi]]] = wd.get(hb[dn[bi]], 0.0) + float(ww)
    top = sorted(wd.items(), key=lambda x: -x[1])[:4] or [(0, 1.0)]
    tot = sum(x[1] for x in top) or 1.0
    return ([x[0] for x in top] + [0] * (4 - len(top)),
            [x[1] / tot for x in top] + [0.0] * (4 - len(top)))


def wind(m, tris):
    """Each triangle turned to face the way its vertices' normals do."""
    V = np.array([v[0] for v in m.vertices])
    out = []
    for p, q, r in tris:
        want = sum(np.array(m.vertices[int(x)][1], float) for x in (p, q, r))
        if float(np.dot(np.cross(V[q] - V[p], V[r] - V[p]), want)) < 0:
            q, r = r, q
        out.append([int(p), int(q), int(r)])
    return out


def pair(V, hosts, donors, limit=SEW):
    out, free = [], list(range(len(hosts)))
    for b in donors:
        best = None
        for k in free:
            d, _ = cKDTree(V[hosts[k]]).query(V[b])
            med = float(np.median(d))
            if med < limit and (best is None or med < best[0]):
                best = (med, k)
        if best is not None:
            out.append((hosts[best[1]], b, best[0]))
            free.remove(best[1])
    return out


def stitch(m, A, B):
    """Triangles sewing one whole rim to another."""
    V = np.array([v[0] for v in m.vertices])
    A, B = list(A), list(B)
    ax = V[B].mean(0) - V[A].mean(0)
    n = np.linalg.norm(ax)
    ax = ax / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ax, [0.0, 1.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(ax, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    turn = lambda L, c: np.sum(np.diff(np.unwrap(np.arctan2(
        (V[L] - c) @ e2, (V[L] - c) @ e1))))
    if turn(A, V[A].mean(0)) * turn(B, V[B].mean(0)) < 0:
        B.reverse()
    k0 = int(np.argmin(np.linalg.norm(V[B] - V[A[0]], axis=1)))
    B = B[k0:] + B[:k0]
    return wind(m, holefill._bridge(A, B, V))


def zipper(m, A, B, tol=ZIP):
    """Triangles closing a rim against another wherever the two run together.

    Partial, unlike `stitch`: only the stretch of A whose vertices each have a vertex
    of B within `tol` is closed, so a slit between two rims that also run off
    elsewhere is shut without dragging the rest of either rim along.
    """
    V = np.array([v[0] for v in m.vertices])
    A, B = list(A), list(B)
    d, j = cKDTree(V[B]).query(V[A])
    near = d < tol
    if near.sum() < 2:
        return []
    # every run of near vertices, taken cyclically
    n = len(A)
    runs, k = [], 0
    if near.all():
        runs = [list(range(n)) + [0]]
    else:
        start = next(i for i in range(n) if not near[i])
        k = start + 1
        cur = []
        while k <= start + n:
            i = k % n
            if near[i]:
                cur.append(i)
            elif cur:
                runs.append(cur)
                cur = []
            k += 1
        if cur:
            runs.append(cur)
    out = []
    nb = len(B)
    for idx in runs:
        if len(idx) < 2:
            continue
        for a0, a1 in zip(idx[:-1], idx[1:]):
            b0, b1 = int(j[a0]), int(j[a1])
            out.append([A[a0], A[a1], B[b0]])
            fwd = (b1 - b0) % nb
            back = (b0 - b1) % nb
            step = 1 if fwd <= back else -1
            b = b0
            while b != b1:
                nxt = (b + step) % nb
                out.append([A[a1], B[nxt], B[b]])
                b = nxt
    return wind(m, out)


def cap(path, mat='skin', most=16, span=2.4):
    """Close the small holes left in a material: a fan over each rim's own plane.

    On doubled vertices. A hole's rim is where an atlas island was cut, so its UVs
    are a slit that encloses no area, and a fan drawn on the rim's own vertices
    samples whatever lies along that slit -- the armpits came out black. Each rim
    vertex is copied and given the UV of the nearest vertex that is not on any rim,
    which is a real texel of the skin beside the hole.
    """
    from scipy.spatial import Delaunay
    from matplotlib.path import Path
    m = PMX(path)
    V = np.array([v[0] for v in m.vertices])
    UV = np.array([v[2] for v in m.vertices])
    here = next(i for i, x in enumerate(m.materials) if x['name'] == mat)
    bl = blocks(m)
    loops = holefill.boundary_loops(V, bl[here])
    on_rim = set(int(i) for l in loops for i in l)
    # a rim with another rim running beside it is a slit to zip, not a hole to cap
    def beside(k):
        for q, other in enumerate(loops):
            if q != k and len(other) > 2:
                d, _ = cKDTree(V[other]).query(V[loops[k]])
                if np.median(d) < 0.6:
                    return True
        return False
    inner = np.array([i for i in np.unique(bl[here]) if int(i) not in on_rim], int)
    tree = cKDTree(V[inner])
    add = []
    for k, l in enumerate(loops):
        P = V[l]
        if len(l) > most or np.ptp(P, axis=0).max() > span or len(l) < 3 or beside(k):
            continue
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c)
        flat = (P - c) @ vt[:2].T
        if len(l) == 3:
            tris = [[0, 1, 2]]
        else:
            try:
                dl = Delaunay(flat)
            except Exception:
                continue
            keep = Path(flat).contains_points(flat[dl.simplices].mean(1))
            tris = dl.simplices[keep].tolist()
        if not tris:
            continue
        twin = []
        for i in l:
            v = list(m.vertices[int(i)])
            _, j = tree.query(V[int(i)])
            v[2] = [float(x) for x in UV[inner[int(j)]]]
            m.vertices.append(tuple(v))
            twin.append(len(m.vertices) - 1)
        add += wind(m, [[twin[a], twin[b], twin[cc]] for a, b, cc in tris])
    if add:
        bl[here] = np.concatenate([bl[here], np.array(add, int)])
        relay(m, bl)
        write_pmx(path, m)
    print(f'  {len(add)} triangles capping the small holes in `{mat}`')


def slits(path, mat='skin', most=40, near=0.6):
    """Zip any two small rims that run beside each other -- a strap's channel."""
    m = PMX(path)
    V = np.array([v[0] for v in m.vertices])
    UV = np.array([v[2] for v in m.vertices])
    here = next(i for i, x in enumerate(m.materials) if x['name'] == mat)
    bl = blocks(m)
    loops = [l for l in holefill.boundary_loops(V, bl[here]) if 3 <= len(l) <= most]
    add, used = [], set()
    for a in range(len(loops)):
        for b in range(a + 1, len(loops)):
            if a in used or b in used:
                continue
            d, _ = cKDTree(V[loops[b]]).query(V[loops[a]])
            if np.median(d) > near:
                continue
            A, B = loops[a], loops[b]
            axis = int(np.argmax(np.ptp(V[A], axis=0)))
            look = interior_uv(m, mat)
            def double(ids):
                out = []
                for i in ids:
                    v = list(m.vertices[int(i)])
                    v[2] = look(V[int(i)])
                    m.vertices.append(tuple(v))
                    out.append(len(m.vertices) - 1)
                return out
            add += zip_arcs(m, double(A), double(B), axis=axis)
            used.update((a, b))
    if add:
        bl[here] = np.concatenate([bl[here], np.array(add, int)])
        relay(m, bl)
        write_pmx(path, m)
    print(f'  {len(used) // 2} pairs of rims beside each other zipped with {len(add)} triangles')


def rewind(path, mat='skin'):
    """Turn any triangle of the material that faces against its own normals."""
    m = PMX(path)
    V = np.array([v[0] for v in m.vertices])
    N = np.array([v[1] for v in m.vertices])
    here = next(i for i, x in enumerate(m.materials) if x['name'] == mat)
    bl = blocks(m)
    T = bl[here]
    g = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    bad = (g * N[T].mean(1)).sum(1) < 0
    if bad.any():
        T[bad] = T[bad][:, [0, 2, 1]]
        bl[here] = T
        relay(m, bl)
        write_pmx(path, m)
    print(f'  {int(bad.sum())} triangles of `{mat}` turned to face with their normals')


def interior_uv(m, mat='skin'):
    """A lookup from a point to the UV of the nearest vertex of `mat` not on a rim --
    a real texel of the skin, where a rim vertex's is the island's edge."""
    V = np.array([v[0] for v in m.vertices])
    UV = np.array([v[2] for v in m.vertices])
    T = holefill.tris_of(m, [mat])
    on_rim = set(int(i) for l in holefill.boundary_loops(V, T) for i in l)
    inner = np.array([i for i in np.unique(T) if int(i) not in on_rim], int)
    tree = cKDTree(V[inner])
    return lambda p: [float(x) for x in UV[inner[int(tree.query(p)[1])]]]


def zip_arcs(m, A, B, axis=0):
    """Triangles between two open arcs, walked together along `axis`.

    For a slit: both sides run left to right across the front, so sorting each by x
    and advancing whichever side has the shorter next diagonal closes it with no
    fans, no wrap and nothing left over -- the cyclic zipper skipped a stretch of the
    shell's rim and a flat cap over it was the shelf under the breast.
    """
    V = np.array([v[0] for v in m.vertices])
    A = sorted(A, key=lambda i: V[i][axis])
    B = sorted(B, key=lambda i: V[i][axis])
    i = j = 0
    out = []
    while i < len(A) - 1 or j < len(B) - 1:
        if i < len(A) - 1 and (j == len(B) - 1 or np.linalg.norm(V[A[i + 1]] - V[B[j]])
                               <= np.linalg.norm(V[A[i]] - V[B[j + 1]])):
            out.append([A[i], A[i + 1], B[j]])
            i += 1
        else:
            out.append([A[i], B[j + 1], B[j]])
            j += 1
    return wind(m, out)


def rims_of(m, mats):
    """Open rims across these materials, counted on positions rather than indices.

    Every bridge in this build sits on doubled vertices -- copies carrying the other
    side's UVs -- so a raw edge count sees two coincident rims where the surface is
    closed. Welded within a hundredth, only what is actually open remains.
    """
    V = np.array([v[0] for v in m.vertices])
    T = holefill.tris_of(m, mats)
    ids = np.unique(T)
    key = {}
    weld = {}
    for i in ids:
        k = tuple(np.round(V[int(i)], 3))
        weld[int(i)] = key.setdefault(k, int(i))
    W = np.vectorize(lambda i: weld[int(i)])(T)
    W = W[(W[:, 0] != W[:, 1]) & (W[:, 1] != W[:, 2]) & (W[:, 2] != W[:, 0])]
    return V, holefill.boundary_loops(V, W)


def check(path, label, mats=BODY):
    m = PMX(path)
    V, L = rims_of(m, [x for x in mats if any(y['name'] == x for y in m.materials)])
    odd = []
    face = set(int(i) for i in np.unique(holefill.tris_of(m, ['face01']))) \
        if any(x['name'] == 'face01' for x in m.materials) else set()
    skin = set(int(i) for i in np.unique(holefill.tris_of(m, ['skin'])))
    for l in L:
        P = V[l]
        if np.abs(P[:, 0]).min() > 6.5 and len(l) <= 10:
            continue                            # fingertips
        if P[:, 1].max() > 19.4 and len(l) <= 4:
            continue                            # the face's own pinholes
        if face and all(int(i) in face and int(i) not in skin for i in l):
            continue                            # the mask's own edge: eyes, mouth, tucked rims
        if face and P[:, 1].min() > 19.5 and np.abs(P[:, 0]).max() < 0.9:
            continue                            # her scalp opening, under the hair
        odd.append(f'{len(l)} verts at y {P[:, 1].min():.1f}-{P[:, 1].max():.1f}'
                   f' x {P[:, 0].min():.1f}..{P[:, 0].max():.1f}'
                   f' z {P[:, 2].min():.1f}..{P[:, 2].max():.1f}')
    print(f'  [{label}] {len(L)} rims, {sum(len(l) for l in L)} edges; '
          + ('none open' if not odd else f'{len(odd)} open:'))
    for o in odd:
        print(f'      {o}')
    return len(odd)


PANTY = dict(ymax=14.0, dark=0.45, lift=0.004, feather=3)


def panty(path):
    """Her painted underwear as an alpha overlay of its own, the skin bare beneath.

    Painted underwear has no edge of its own to cut along: the black stops inside
    triangles, not at them, so any triangle taken whole carries skin outside the
    black and shows the new material's shading there -- a saw-tooth fringe. So it is
    done the way models ship it. The triangles that touch the black are duplicated
    into `underwear`, a hair off the surface, sampling an RGBA copy of her original
    sheet whose alpha is the black itself, feathered; and the black on the skin sheet
    underneath is filled back to skin from around it, so the overlay can be hidden.
    """
    import selenepaint
    from PIL import Image
    from scipy import ndimage
    m = PMX(path)
    V = np.array([v[0] for v in m.vertices])
    N = np.array([v[1] for v in m.vertices])
    UV = np.array([v[2] for v in m.vertices])
    here = next(i for i, x in enumerate(m.materials) if x['name'] == 'skin')
    bl = blocks(m)
    T = bl[here]
    sheet = os.path.join(os.path.dirname(path), 'textures', '衣服.png')
    A = np.asarray(Image.open(sheet).convert('RGB'), np.float32) / 255
    h, w, _ = A.shape
    dark = A.mean(2) < PANTY['dark']
    c = V[T].mean(1)
    low = c[:, 1] < PANTY['ymax']
    # a triangle touches the black if any vertex's texel or its centre's is dark
    def darkat(uv):
        x = np.clip((uv[:, 0] % 1.0 * (w - 1)).astype(int), 0, w - 1)
        y = np.clip((uv[:, 1] % 1.0 * (h - 1)).astype(int), 0, h - 1)
        return dark[y, x]
    touch = low & (darkat(UV[T[:, 0]]) | darkat(UV[T[:, 1]]) | darkat(UV[T[:, 2]])
                   | darkat(UV[T].mean(1)))
    if not touch.any():
        print('  no underwear to lift off')
        return
    sel = T[touch]
    # only the black under those triangles, on the sheet: rasterise them to find it
    own, _ = skinclean.cover(m, sel, UV, w, h)
    black = own & dark
    # the overlay: her original texels, alpha = the black, feathered outward
    alpha = ndimage.gaussian_filter(black.astype(np.float32), PANTY['feather'])
    alpha = np.clip(alpha * 1.6, 0, 1)
    rgba = np.dstack([A, alpha])
    over = os.path.join(os.path.dirname(path), 'textures', '衣服_underwear.png')
    Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), 'RGBA').save(over)
    # the skin beneath: the black filled from the SKIN around it -- from her own
    # skin's texels, not from whatever is nearest on the sheet, which beside this
    # island is the atlas's dark background and left a third of it black
    hole = ndimage.binary_dilation(black, iterations=3)
    own_skin, _ = skinclean.cover(m, T, UV, w, h)
    src = own_skin & ~dark & ~hole
    _, (yy, xx) = ndimage.distance_transform_edt(~src, return_indices=True)
    B = A.copy()
    B[hole] = A[yy[hole], xx[hole]]
    soft = np.clip(ndimage.gaussian_filter(hole.astype(np.float32), 2) * 1.5, 0, 1)
    B = A * (1 - soft)[..., None] + B * soft[..., None]
    Image.fromarray((np.clip(B, 0, 1) * 255).astype(np.uint8)).save(sheet)
    # the duplicated triangles, a hair off the surface
    rel = 'textures\\衣服_underwear.png'
    m.textures.append(rel)
    new = dict(m.materials[here])
    new['name'] = new['name_en'] = 'underwear'
    new['tex'] = m.textures.index(rel)
    m.materials.append(new)
    twin = {}
    for i in np.unique(sel):
        v = list(m.vertices[int(i)])
        n = N[int(i)] / (np.linalg.norm(N[int(i)]) + 1e-9)
        v[0] = [float(x) for x in V[int(i)] + n * PANTY['lift']]
        m.vertices.append(tuple(v))
        twin[int(i)] = len(m.vertices) - 1
    bl.append(np.array([[twin[int(a)], twin[int(b)], twin[int(cc)]] for a, b, cc in sel], int))
    relay(m, bl)
    write_pmx(path, m)
    print(f'  {len(sel)} triangles doubled into `underwear` over {int(black.sum())} black texels; '
          f'the skin beneath filled back from around it')


def rename_chains(path):
    m = PMX(path)
    n = 0
    for b in m.bones:
        if CHAIN.match(b['name']):
            b['name'] = 'sel_' + b['name']
            b['name_en'] = 'sel_' + b['name_en']
            n += 1
    write_pmx(path, m)
    print(f'  {n} of her ornament chain bones renamed out of the way of reze\'s hair')


def belly(path, donor=DONOR, r=BELLY):
    """Fill the belly hole with his shape and her paint; zip the underbust slit."""
    tag, src, mat = donor
    m, d = PMX(path), PMX(src)
    V = np.array([v[0] for v in m.vertices])
    UV = np.array([v[2] for v in m.vertices])
    here = next(i for i, x in enumerate(m.materials) if x['name'] == 'skin')
    bl = blocks(m)
    inbox = lambda P: ((P[:, 1] > r['ylo']) & (P[:, 1] < r['yhi']) & (P[:, 2] < r['zmax'])
                       & (np.abs(P[:, 0]) < r['wide']))
    rims = [l for l in holefill.boundary_loops(V, bl[here]) if inbox(V[l]).all()]
    rims.sort(key=len, reverse=True)
    hole, slit = rims[0], rims[1:]
    src_ids = np.concatenate(rims)
    srcP, srcUV = V[src_ids], UV[src_ids]

    DV = np.array([v[0] for v in d.vertices])
    D = holefill.tris_of(d, [mat])
    centre = np.array([0.0, (r['ylo'] + r['yhi']) / 2, -0.9])
    ref, scale, gap = holefill.fit(d, m, [mat], ['skin'], near=centre, reach=3.0)
    her = Mesh(V, bl[here])
    mid = ref.V[D].mean(1)
    far, _, _ = her.closest(mid)
    sel = D[inbox(mid) & (far > HOLE)]

    used = np.unique(sel)
    tree = cKDTree(srcP)
    new = {}
    for i in used:
        p = ref.V[int(i)]
        dist, j = tree.query(p, k=min(12, len(srcP)))
        w = 1.0 / np.maximum(dist, 1e-4) ** 2
        uv = (srcUV[j] * w[:, None]).sum(0) / w.sum()
        bs, ws = weights(m, d, i)
        m.vertices.append(([float(x) for x in p], list(d.vertices[int(i)][1]),
                           [float(x) for x in uv], [np.zeros(4) for _ in range(m.uv_add)],
                           2, bs, ws, 0.0))
        new[int(i)] = len(m.vertices) - 1
    tris = np.array([[new[int(x)] for x in t] for t in sel], int)
    V = np.array([v[0] for v in m.vertices])
    drims = holefill.boundary_loops(V, tris)
    pairs = pair(V, [hole], drims)
    sewn = []
    for A, B, med in pairs:
        sewn += stitch(m, A, B)
    # The slit: the hole's top edge against the shell's front rim, both taken as the
    # arc within reach of the other, zipped along x on copies of the hole's edge
    # carrying the shell's UVs -- the hole rim's own UVs are the cut edge of an island.
    zipped = []
    UV = np.array([v[2] for v in m.vertices])
    for s_ in slit:
        dh, jh = cKDTree(V[s_]).query(V[hole])
        top = [i for i, dd in zip(hole, dh) if dd < 0.6]
        ds, _ = cKDTree(V[top]).query(V[s_]) if top else (np.full(len(s_), 9.0), None)
        front = [i for i, dd in zip(s_, ds) if dd < 0.6]
        if len(top) < 2 or len(front) < 2:
            continue
        # both sides doubled, both carrying a real skin texel: a rim vertex's own UV
        # is the edge of an island, and the strip came out black on it
        look = interior_uv(m)
        def double(ids):
            out = []
            for i in ids:
                v = list(m.vertices[int(i)])
                v[2] = look(V[int(i)])
                m.vertices.append(tuple(v))
                out.append(len(m.vertices) - 1)
            return out
        zipped += zip_arcs(m, double(top), double(front))
    add = np.concatenate([tris, np.array(sewn + zipped, int).reshape(-1, 3)])
    bl = blocks(m)
    bl[here] = np.concatenate([bl[here], add])
    relay(m, bl)
    holefill.renormal(m, bl[here], np.unique(add))
    write_pmx(path, m)
    print(f'  {len(sel)} triangles of {tag} in the hole at {gap:.3f} off her, UVs into her '
          f'atlas; hole rim {len(hole)} verts paired at '
          f'{", ".join(f"{p[2]:.2f}" for p in pairs) or "-"}, {len(sewn)} sewn; slit of '
          f'{[len(s) for s in slit]} zipped with {len(zipped)}')


ELBOW = 4.0          # the arm is his from here out: his wrist meets his own hand


def forearms(path, donor=DONOR, x=ELBOW):
    """His forearm and hand from the elbow out, one ring sewn at the elbow."""
    tag, src, mat = donor
    tex = 'textures\\cloth01.png'
    m, d = PMX(path), PMX(src)
    V = np.array([v[0] for v in m.vertices])
    here = next(i for i, x_ in enumerate(m.materials) if x_['name'] == 'skin')
    bl = blocks(m)
    gone = np.abs(V[bl[here]].mean(1)[:, 0]) > x
    bl[here] = bl[here][~gone]
    relay(m, bl)
    DV = np.array([v[0] for v in d.vertices])
    D = holefill.tris_of(d, [mat])
    sel = D[np.abs(DV[D].mean(1)[:, 0]) > x]
    took, sewn = holefill.transplant(m, d, Mesh(DV, D), {mat: sel}, ['skin'], limit=1.2)
    fp = os.path.join(os.path.dirname(src), tex.replace('\\', '/'))
    dst = f'{tag}_{os.path.basename(fp)}'
    os.makedirs(os.path.join(os.path.dirname(path), 'textures'), exist_ok=True)
    shutil.copyfile(fp, os.path.join(os.path.dirname(path), 'textures', dst))
    rel = 'textures\\' + dst
    if rel not in m.textures:
        m.textures.append(rel)
    new = dict(m.materials[here])
    new['name'] = new['name_en'] = 'skin_hands'
    new['tex'] = m.textures.index(rel)
    m.materials.append(new)
    bl = blocks(m)
    bl[-1] = np.concatenate([np.asarray(v, int).reshape(-1, 3) for v in took.values()])
    relay(m, bl)
    write_pmx(path, m)
    print(f'  {int(gone.sum())} of her forearm cut, {len(bl[-1])} of {tag} in as `skin_hands`, '
          f'{sewn} sewn at the elbow')


def same_skin(path):
    """His hands shaded like her skin: same diffuse, ambient and specular."""
    m = PMX(path)
    her = next(x for x in m.materials if x['name'] == 'skin')
    for x in m.materials:
        if x['name'] == 'skin_hands':
            for k in ('diffuse', 'specular', 'shininess', 'ambient', 'edge_color',
                      'edge_size', 'flag', 'toon_shared', 'toon', 'sph', 'sph_mode'):
                x[k] = her[k]
    write_pmx(path, m)


def wrists(path):
    """Sew whichever wrist `hands` left open -- it pairs rims by centre, and one side
    of a pair of wrists is always the other side's nearest by centre."""
    m = PMX(path)
    V = np.array([v[0] for v in m.vertices])
    UV = np.array([v[2] for v in m.vertices])
    here = next(i for i, x in enumerate(m.materials) if x['name'] == 'skin')
    hand = next(i for i, x in enumerate(m.materials) if x['name'] == 'skin_hands')
    bl = blocks(m)
    at = lambda l: 5.0 < np.abs(V[l][:, 0]).mean() < 8.0 and len(l) > 8
    sk = [l for l in holefill.boundary_loops(V, bl[here]) if at(l)]
    hd = [l for l in holefill.boundary_loops(V, bl[hand]) if at(l)]
    add = []
    for A, B, med in pair(V, sk, hd, limit=0.8):
        # her rim doubled into his atlas, so the bridge lives with the hand
        tree = cKDTree(V[B])
        twin = []
        for i in A:
            v = list(m.vertices[int(i)])
            _, j = tree.query(V[int(i)])
            v[2] = [float(x) for x in UV[B[int(j)]]]
            m.vertices.append(tuple(v))
            twin.append(len(m.vertices) - 1)
        add += stitch(m, twin, B)
    if add:
        bl = blocks(m)
        bl[hand] = np.concatenate([bl[hand], np.array(add, int)])
        relay(m, bl)
        write_pmx(path, m)
    print(f'  wrists: {len(sk)} of hers open against {len(hd)} of his, {len(add)} sewn')


def neck(path, tol=1.3):
    """Bridge her neck ring to the face's jaw, arc by arc.

    Her skin's neck ring is a real open edge -- the hole her own face was cut from --
    and the face's jaw rim runs above it. Every pair of rims in the box is taken only
    where they run within `tol` of each other and zipped along its own long axis on
    doubled vertices carrying her skin's texels. Eyes and mouth are the face's own.
    """
    m = PMX(path)
    V = np.array([v[0] for v in m.vertices])
    here = next(i for i, x in enumerate(m.materials) if x['name'] == 'skin')
    _, L = rims_of(m, ['skin', 'face01'])
    face = set(int(i) for i in np.unique(holefill.tris_of(m, ['face01'])))
    box = lambda l: (18.3 < V[l][:, 1].mean() < 21.3 and np.abs(V[l][:, 0]).max() < 1.6
                     and len(l) > 4 and not (all(int(i) in face for i in l)
                                             and np.ptp(V[l][:, 1]) < 0.4))
    loops = [l for l in L if box(l)]
    look = interior_uv(m)
    def double(ids):
        out = []
        for i in ids:
            v = list(m.vertices[int(i)])
            v[2] = look(V[int(i)])
            m.vertices.append(tuple(v))
            out.append(len(m.vertices) - 1)
        return out
    add, pairs = [], 0
    for a in range(len(loops)):
        for b in range(a + 1, len(loops)):
            A, B = loops[a], loops[b]
            da, _ = cKDTree(V[B]).query(V[A])
            db, _ = cKDTree(V[A]).query(V[B])
            arcA = [i for i, d in zip(A, da) if d < tol]
            arcB = [i for i, d in zip(B, db) if d < tol]
            if len(arcA) < 3 or len(arcB) < 3:
                continue
            axis = int(np.argmax(np.ptp(V[arcA], axis=0)))
            add += zip_arcs(m, double(arcA), double(arcB), axis=axis)
            pairs += 1
    if add:
        bl = blocks(m)
        bl[here] = np.concatenate([bl[here], np.array(add, int)])
        relay(m, bl)
        write_pmx(path, m)
    print(f'  neck: {len(loops)} rims in the box, {pairs} pairs of arcs zipped with '
          f'{len(add)} triangles')


TUCK = 0.02          # how far under her skin the face's free edge goes


def tuck_face(path, under=TUCK):
    """Tuck the face's free edges under her neck and scalp.

    The face is a mask over a closed surface; its jaw rim and scalp rim are free
    edges, not holes, and cannot be sewn to anything. Left where they are they show as
    a step. Each free rim vertex is set on the nearest point of her skin, a little
    inside it, so the edge disappears into the neck. Eyes and mouth are left.
    """
    m = PMX(path)
    V = np.array([v[0] for v in m.vertices])
    facev = set(int(i) for i in np.unique(holefill.tris_of(m, ['face01'])))
    skinv = set(int(i) for i in np.unique(holefill.tris_of(m, ['skin'])))
    skin = Mesh(V, holefill.tris_of(m, ['skin']))
    _, L = rims_of(m, ['skin', 'face01'])
    moved, worst = 0, 0.0
    for l in L:
        if len(l) < 5 or np.ptp(V[l][:, 1]) < 0.4:
            continue                                    # an eye socket or the mouth
        if not all(int(i) in facev and int(i) not in skinv for i in l):
            continue                                    # not a free edge of the mask
        d, t, bar = skin.closest(V[l])
        f = skin.F[t]
        q = (V[f[:, 0]] * bar[:, :1] + V[f[:, 1]] * bar[:, 1:2] + V[f[:, 2]] * bar[:, 2:3])
        for k, i in enumerate(l):
            if d[k] > 0.8:
                continue
            v = list(m.vertices[int(i)])
            v[0] = [float(x) for x in q[k] - skin.normal[t[k]] * under]
            m.vertices[int(i)] = tuple(v)
            moved += 1
            worst = max(worst, float(d[k]))
    write_pmx(path, m)
    print(f'  face: {moved} free-edge vertices tucked under her skin, the furthest from {worst:.2f}')


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)
    maidselene.DIR, maidselene.WORN, maidselene.RENAME = WORK, BUILD, dict(KEEP)
    with progress.step('her, undressed, in her shoes'):
        maidselene.undressed()
        rename_chains(BUILD)
        panty(BUILD)
        check(BUILD, 'as she comes')
    with progress.step('her belly'):
        belly(BUILD)
        cap(BUILD)
        check(BUILD, 'belly')
    with progress.step('his forearms'):
        forearms(BUILD)
        wrists(BUILD)
        slits(BUILD)
        cap(BUILD)
        check(BUILD, 'forearms')
    with progress.step('her costume, painted out'):
        skinclean.main(BUILD, os.path.join(WORK, 'textures', '衣服.png'))
    with progress.step('head'):
        subprocess.run([sys.executable, os.path.join(HERE, 'headgraft.py'),
                        '--host', BUILD, '--head', maidselene.HEAD, '--out', BUILD,
                        '--drop', ','.join(maidselene.DROP_HEAD), '--scale', '1'], check=True)
        neck(BUILD)
        tuck_face(BUILD)
        cap(BUILD)
        check(BUILD, 'head')
    with progress.step('the deform chain'):
        rewind(BUILD)
        maidbuild.stiffen_foot(BUILD)
        fixrig.main([BUILD])
    with progress.step('out'):
        publish.main(BUILD, DIR)
    shutil.rmtree(WORK, ignore_errors=True)
    progress.done()
    m = PMX(OUT)
    print(f'{OUT}: {len(m.vertices)} verts, {len(m.faces) // 3} triangles, '
          f'{len(m.bones)} bones, {len(m.rigids)} rigid bodies, {len(m.joints)} joints')
    print(f'  materials: {", ".join(x["name"] for x in m.materials)}')


if __name__ == '__main__':
    main()
