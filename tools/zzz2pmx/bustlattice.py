"""Slim the bust of reze-midnight.pmx with a lattice deformation run in Blender.

The donor body's breasts are a size larger than reze's. A B-spline lattice cage
round the chest draws the whole bust in and back toward the chest wall, lifts
its underside, turns its tip up and shrinks it below the tip into a drop; skin, bra cups, ruffle, knot and the 胸
bones all sit in the same cage, so they move as one field: the cup's edge stays
on the hole rim under it and nothing tears. Before the cage the skin over the
breast and the bra cups are split one triangle into four the Loop way, twice, so
both turn smooth and dense; after it Blender's Laplacian smoothing rounds the crease under
the breast. The left side is then rebuilt from the right and the
cups kept a hair off the skin.

Under each cup the donor's skin is a hole. It is filled with a breast drawn
from smooth curves: a cubic Hermite meridian from the cup's tip out to every
rim vertex, level at the tip and leaving the rim along the skin's own tangent,
full underneath and gentler above; the cup is then laid over that dome a hair
off it. The nude body is round, and another outfit can go over it too. The fill is its own UV island on a skin disc painted onto a free square of the
skin's texture, flat skin in the middle and the rim's own colours at the edge,
no nipple; the skin reads that copy of the texture. The result is written with
everything else in the file untouched.

    python3 tools/zzz2pmx/bustlattice.py [--wide 0.77] [--deep 0.85] [--lift 0.2] [--tip 0.3]

The Blender scene with the cage is saved beside the model for inspection.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dropmaterial
import validate
from PIL import Image
from pmx import PMX
from pmxw import write_pmx
from stitchhole import loops_of
from surface import Mesh

BLENDER = '/Applications/Blender.app/Contents/MacOS/Blender'
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bust_blender.py')
SKIN = 'skin'
CUP = 'cloth01'
BREAST = ['左胸', '右胸']
DROP = ['cloth01_alpha1']
CUPS = ['cloth01', 'cloth01_alpha2']
CLEAR = 0.02                # the cups sit this far off the skin
AXIS = 0.4                  # the bust draws in toward this |x|, so the cleavage holds
WALL = -1.3                 # z of the chest wall the breasts stand on
BOX_LO = np.array([-2.4, 14.0, -2.9])
BOX_HI = np.array([2.4, 18.2, 0.1])
DIMS = np.array([13, 8, 6])  # lattice points per axis: 0.4 apart across, 0.6 up and in
MIRROR = np.array([-1.0, 1.0, 1.0])
HEM = 0.03                  # edges shorter than this are a hem's fold and stay whole
HEM_ZONE = 0.05             # vertices this close to the material's boundary belong to the hem and stay
PATCH = 160                 # pixels: the square painted for the fill on a free part of the atlas


def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def smooth_normals(V, tris):
    fn = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
    acc = np.zeros_like(V)
    for c in range(3):
        np.add.at(acc, tris[:, c], fn)
    n = np.linalg.norm(acc, axis=1, keepdims=True)
    return acc / np.where(n > 1e-12, n, 1.0)


def welded_normals(V, tris, idx):
    """Area-weighted normals, vertices at one position sharing the sum, so a UV
    seam down the sternum shades as one surface."""
    fn = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
    key = {}
    group = np.zeros(len(V), int)
    for i in idx:
        group[i] = key.setdefault(tuple(np.round(V[i], 4)), len(key))
    acc = np.zeros((len(key), 3))
    for c in range(3):
        np.add.at(acc, group[tris[:, c]], fn)
    n = np.linalg.norm(acc, axis=1, keepdims=True)
    return (acc / np.where(n > 1e-12, n, 1.0))[group]


def spans(H):
    out, at = {}, 0
    for mat in H.materials:
        n = mat['fcount'] // 3
        out[mat['name']] = (at, n)
        at += n
    return out


def faces_of(H, name):
    start, n = spans(H)[name]
    return np.array(H.faces).reshape(-1, 3)[start:start + n]


def bones_of(vv):
    """A vertex's (bones, weights) padded to four."""
    b = [int(x) for x in vv[5]]
    w = [float(x) for x in vv[6]][:len(b)]
    keep = [(bn, wt) for bn, wt in zip(b, w) if wt > 0][:4] or [(b[0], 1.0)]
    tot = sum(wt for _, wt in keep)
    return ([bn for bn, _ in keep] + [0] * (4 - len(keep)),
            [wt / tot for _, wt in keep] + [0.0] * (4 - len(keep)))


def split(H, V, name, on):
    """Split every edge of material `name` with both ends in `on` at its middle.

    A triangle with one, two or three such edges becomes two, three or four, so
    nothing T-junctions with the mesh round the region; edges shorter than HEM,
    the cup's folded hem, stay whole. Positions follow Loop subdivision: a
    midpoint takes 3/8 of each end and 1/8 of the two vertices across, a vertex
    of the region moves toward its ring, and the boundary (the hole rim, the
    cup's edge) and everything within HEM_ZONE of it stay. New vertices average their parents' UV and
    weights. Returns the extended V, the map from parent edge to midpoint, and
    the material's boundary vertices.
    """
    F = np.array(H.faces).reshape(-1, 3)
    start, n = spans(H)[name]
    st = F[start:start + n]
    cnt = {}
    for t in st:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = (min(int(t[a]), int(t[b])), max(int(t[a]), int(t[b])))
            cnt[e] = cnt.get(e, 0) + 1
    rim = set(v for e, c in cnt.items() if c == 1 for v in e)
    idx = np.unique(st)
    hem = set(int(v) for v in idx[cKDTree(V[list(rim)]).query(V[idx])[0] < HEM_ZONE]) if rim else set()
    across, ring = {}, {}
    for t in st:
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            e = (min(int(t[a]), int(t[b])), max(int(t[a]), int(t[b])))
            across.setdefault(e, []).append(int(t[c]))
            ring.setdefault(int(t[a]), set()).add(int(t[b]))
            ring.setdefault(int(t[b]), set()).add(int(t[a]))
    mids, newpos = {}, []

    def loop_mid(a, b):
        key = (min(a, b), max(a, b))
        if cnt[key] == 2:
            c, d = across[key]
            if not ({a, b, c, d} & hem):
                return 0.375 * (V[a] + V[b]) + 0.125 * (V[c] + V[d])
        return 0.5 * (V[a] + V[b])

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key in mids:
            return mids[key]
        va, vb = H.vertices[a], H.vertices[b]
        acc = {}
        for vv in (va, vb):
            for bn, w in zip(vv[5], vv[6]):
                if w > 0:
                    acc[bn] = acc.get(bn, 0.0) + 0.5 * w
        items = sorted(acc.items(), key=lambda kv: -kv[1])[:4]
        tot = sum(w for _, w in items) or 1.0
        pos = loop_mid(a, b)
        rec = ([float(x) for x in pos], [0.0, 0.0, -1.0],
               [0.5 * (va[2][0] + vb[2][0]), 0.5 * (va[2][1] + vb[2][1])],
               [[0.0] * 4 for _ in range(H.uv_add)], 2,
               [bn for bn, _ in items] + [0] * (4 - len(items)),
               [w / tot for _, w in items] + [0.0] * (4 - len(items)), 0.5 * (va[7] + vb[7]))
        H.vertices.append(rec)
        newpos.append(pos)
        mids[key] = len(H.vertices) - 1
        return mids[key]

    moved = {}
    for v in np.where(on)[0]:
        v = int(v)
        if v in hem or v not in ring:
            continue
        nb = list(ring[v])
        k = len(nb)
        beta = (0.625 - (0.375 + 0.25 * np.cos(2 * np.pi / k)) ** 2) / k
        moved[v] = (1 - k * beta) * V[v] + beta * V[nb].sum(0)

    out = []
    for t in st:
        a, b, c = (int(x) for x in t)
        cut = [on[p] and on[q] and np.linalg.norm(V[p] - V[q]) > HEM for p, q in ((a, b), (b, c), (c, a))]
        if not any(cut):
            out.append((a, b, c))
            continue
        ab = mid(a, b) if cut[0] else None
        bc = mid(b, c) if cut[1] else None
        ca = mid(c, a) if cut[2] else None
        if all(cut):
            out += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        elif sum(cut) == 2:
            if not cut[0]:                       # rotate so the uncut edge is c->a
                a, b, c, ab, bc, ca = b, c, a, bc, ca, ab
            elif not cut[1]:
                a, b, c, ab, bc, ca = c, a, b, ca, ab, bc
            out += [(a, ab, c), (ab, bc, c), (ab, b, bc)]
        else:
            if cut[1]:                           # rotate so the cut edge is a->b
                a, b, c, ab = b, c, a, bc
            elif cut[2]:
                a, b, c, ab = c, a, b, ca
            out += [(a, ab, c), (ab, b, c)]
    new = np.array(out)
    H.faces = np.concatenate([F[:start], new, F[start + n:]]).ravel().tolist()
    H.materials[[m['name'] for m in H.materials].index(name)]['fcount'] = len(new) * 3
    V = np.vstack([V, np.array(newpos)]) if newpos else V
    for v, pos in moved.items():
        V[v] = pos
    print(f'{name} over the breast split: {len(mids)} edges, {len(new) - n} triangles added, '
          f'{len(moved)} vertices eased the Loop way')
    return V, mids, rim


def lattice(a, apex):
    """Rest points of the cage and the displacement of each, u fastest."""
    U, Vn, W = DIMS
    us = np.linspace(BOX_LO[0], BOX_HI[0], U)
    vs = np.linspace(BOX_LO[1], BOX_HI[1], Vn)
    ws = np.linspace(BOX_LO[2], BOX_HI[2], W)
    P = np.array([(x, y, z) for z in ws for y in vs for x in us])
    x, y, z = P.T
    gx = 1 - smoothstep(np.abs(x), 1.5, 2.4)                       # the ribs' sides and arms stay
    gy = smoothstep(y, 14.0, 15.0) * (1 - smoothstep(y, 17.4, 18.2))
    gz = smoothstep(-z, 0.5, 1.1)                                   # the back stays
    front = smoothstep(-z, 1.1, 1.7)                                # what stands off the chest wall
    D = np.zeros_like(P)
    outer = np.maximum(np.abs(x) - AXIS, 0) * np.sign(x)
    D[:, 0] = -outer * (1 - a.wide) * gx * gy * gz
    D[:, 2] = np.maximum(WALL - z, 0) * (1 - a.deep) * gx * gy
    yc = apex[1] - a.crease
    D[:, 1] = a.lift * np.exp(-((y - yc) / 0.5) ** 2) * gx * front
    # the tip: full at the apex's depth, nothing at the wall, so the breast turns up
    tip = smoothstep(-z, -WALL, -apex[2]) * np.exp(-((y - apex[1]) / 0.7) ** 2) * gx
    D[:, 1] += a.tip * tip
    # the drop: below the tip the breast shrinks toward the wall, so its underside
    # turns in under the nipple instead of hanging
    D[:, 2] += np.maximum(WALL - z, 0) * a.drop * smoothstep(apex[1] - y, 0.25, 0.75) * gx * gy
    return P, D


def ray_hits(o, d, V, tris):
    """Signed distances along d of the hits of the line through o with tris."""
    a = V[tris[:, 0]]
    e1, e2 = V[tris[:, 1]] - a, V[tris[:, 2]] - a
    p = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, p)
    ok = np.abs(det) > 1e-9
    inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
    s = o - a
    u = np.einsum('ij,ij->i', s, p) * inv
    q = np.cross(s, e1)
    v = (q @ d) * inv
    t = np.einsum('ij,ij->i', e2, q) * inv
    hit = ok & (u >= -1e-4) & (v >= -1e-4) & (u + v <= 1 + 1e-4)
    return t[hit], np.where(hit)[0]


def paint_patch(H, tex_dir, rim_ids, rim_theta):
    """Paint a skin disc for the fill on a free square of the skin's texture.

    The centre is the body's flat skin tone; toward the edge the disc takes the
    colour the texture already has at each rim vertex, so the seam matches all
    round. The skin material is pointed at the new copy of the texture. Returns
    the disc's centre and radius in UV units.
    """
    mats = {m['name']: m for m in H.materials}
    ti = mats[SKIN]['tex']
    src = os.path.join(tex_dir, H.textures[ti].replace('\\', '/'))
    im = Image.open(src).convert('RGBA')
    W, Hh = im.size
    px = np.array(im)
    F = np.array(H.faces).reshape(-1, 3)
    UV = np.array([v[2] for v in H.vertices])
    occ = Image.new('L', (W, Hh), 0)
    draw = __import__('PIL.ImageDraw', fromlist=['Draw']).Draw(occ)
    for name, (start, n) in spans(H).items():
        if mats[name]['tex'] == ti:
            for t in F[start:start + n]:
                draw.polygon([(float(UV[i, 0] * W), float(UV[i, 1] * Hh)) for i in t], fill=255)
    occ = np.array(occ) > 0
    m = PATCH + 16
    free = [(x, y) for y in range(0, Hh - m, 16) for x in range(0, W - m, 16) if not occ[y:y + m, x:x + m].any()]
    x0, y0 = free[len(free) // 2]
    cx, cy = x0 + m // 2, y0 + m // 2
    R = PATCH * 0.45

    def texel(u, v):
        i, j = int(v * Hh) % Hh, int(u * W) % W
        return px[max(0, i - 2):i + 3, max(0, j - 2):j + 3, :3].reshape(-1, 3).mean(0)

    rim_col = np.array([texel(*UV[i]) for i in rim_ids])
    centre = texel(0.6305, 0.7261)                       # the flat of the belly
    th = np.concatenate([rim_theta, [1.0]])
    col = np.vstack([rim_col, rim_col[:1]])
    yy, xx = np.mgrid[y0:y0 + m, x0:x0 + m]
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy) / R
    ang = (np.arctan2(dy, dx) / (2 * np.pi)) % 1.0
    edge = np.stack([np.interp(ang, th, col[:, k]) for k in range(3)], -1)
    f = smoothstep(r, 0.3, 1.0)[..., None]
    patch = centre * (1 - f) + edge * f
    px[y0:y0 + m, x0:x0 + m, :3] = np.clip(np.round(patch), 0, 255).astype(np.uint8)
    px[y0:y0 + m, x0:x0 + m, 3] = 255
    base, ext = os.path.splitext(H.textures[ti])
    name = base + '_slim' + ext
    Image.fromarray(px).save(os.path.join(tex_dir, name.replace('\\', '/')))
    H.textures.append(name)
    mats[SKIN]['tex'] = len(H.textures) - 1
    print(f'painted a skin disc for the fill at ({cx}, {cy}) of {name}; the skin reads it')
    return np.array([cx / W, cy / Hh]), R / W


def fill_cup(H, V, rim_ids, cup, skin, sgn, rings, crown, fullness, upper, uv_c, uv_r):
    """Fill the hole under one cup with a breast drawn from smooth curves.

    Ring 0 is a copy of the rim, so the fill is its own UV island on the painted
    disc. From the cup's tip, set a hair inside it, a cubic Hermite curve runs
    out to every rim vertex: level at the tip, and leaving the rim along the
    tangent of the skin outside it, so the dome continues the chest without a
    fold. `crown` scales the tangent at the tip (a flatter or a pointier tip),
    `fullness` the tangent at the rim on the underside and sides (how round the
    breast is there), `upper` the same on the upper slope. Rings of points along
    those curves, joined by angle, make the dome; a centre point closes the fan.
    Returns the new vertex ids, the faces and the extended V.
    """
    Pr = V[rim_ids]
    seg = np.linalg.norm(np.roll(Pr, -1, axis=0) - Pr, axis=1)
    theta = np.concatenate([[0.0], np.cumsum(seg)[:-1]]) / seg.sum()
    c = Pr.mean(0)
    n = np.linalg.svd(Pr - c)[2][2]
    n *= np.sign(n @ np.array([sgn * 0.3, 0.0, -1.0]))
    e_up = np.array([0.0, 1.0, 0.0]) - n[1] * n
    e_up /= np.linalg.norm(e_up)
    cup_idx = np.unique(cup)
    tip = V[cup_idx[np.argmax((V[cup_idx] - c) @ n)]]
    apex = tip - CLEAR * n
    # the skin's tangent into the hole at every rim vertex
    ns = smooth_normals(V, skin)
    ring_of = {}
    for t in skin:
        for a_, b_ in ((0, 1), (1, 2), (2, 0)):
            ring_of.setdefault(int(t[a_]), set()).add(int(t[b_]))
            ring_of.setdefault(int(t[b_]), set()).add(int(t[a_]))
    rim_set = set(int(i) for i in rim_ids)
    T1 = []
    for i in rim_ids:
        outer = [j for j in ring_of.get(int(i), ()) if j not in rim_set]
        d = V[outer].mean(0) - V[i] if outer else (V[i] - apex)   # on out of the hole, along the skin
        d -= (d @ ns[i]) * ns[i]
        T1.append(d / (np.linalg.norm(d) + 1e-12))
    T1 = np.array(T1)
    T1 = sum(np.roll(T1, k, axis=0) for k in range(-3, 4)) / 7.0     # eased round the rim
    T1 /= np.linalg.norm(T1, axis=1, keepdims=True) + 1e-12
    near = cKDTree(V[cup_idx])
    edge = float(np.mean([H.vertices[i][7] for i in rim_ids]))

    def at(th, A):
        th %= 1.0
        k = np.searchsorted(theta, th, side='right') - 1
        t0 = theta[k]
        t1 = theta[k + 1] if k + 1 < len(theta) else 1.0
        f = (th - t0) / max(t1 - t0, 1e-9)
        return A[k] * (1 - f) + A[(k + 1) % len(A)] * f

    def meridian(th, t):
        b = at(th, Pr)
        tb = at(th, T1)
        tb /= np.linalg.norm(tb) + 1e-12
        chord = b - apex
        L = np.linalg.norm(chord)
        flat = chord - (chord @ n) * n
        flat /= np.linalg.norm(flat) + 1e-12
        up = flat @ e_up
        k1 = fullness + (upper - fullness) * smoothstep(up, 0.0, 0.8)
        t0, t1 = crown * L * flat, k1 * L * tb
        h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        return h00 * apex + h10 * t0 + h01 * b + h11 * t1

    def disc(t, th):
        return uv_c + t * uv_r * np.array([np.cos(2 * np.pi * th), np.sin(2 * np.pi * th)])

    def add(pos, uv):
        j = near.query(pos)[1]
        bn, w = bones_of(H.vertices[cup_idx[j]])
        H.vertices.append(([float(x) for x in pos], [0.0, 0.0, -1.0], [float(uv[0]), float(uv[1])],
                           [[0.0] * 4 for _ in range(H.uv_add)], 2, bn, w, edge))
        return len(H.vertices) - 1

    new, newpos = [], []
    ring = []
    for i, th in zip(rim_ids, theta):
        rec = list(H.vertices[i])
        rec[2] = [float(x) for x in disc(1.0, th)]
        H.vertices.append(tuple(rec))
        ring.append(len(H.vertices) - 1)
        new.append(ring[-1]); newpos.append(V[i].copy())
    ring_t = theta
    faces = []
    for k in range(1, rings + 1):
        t = 1 - k / (rings + 1)
        nk = max(8, int(round(len(rim_ids) * t)))
        ths = (np.arange(nk) + 0.5 * (k % 2)) / nk
        ids = []
        for th in ths:
            pos = meridian(th, t)
            ids.append(add(pos, disc(t, th)))
            new.append(ids[-1]); newpos.append(pos)
        tA = np.concatenate([ring_t, [ring_t[0] + 1]])
        tB = np.concatenate([ths, [ths[0] + 1]])
        i = j = 0
        while i < len(ring) or j < nk:
            if j >= nk or (i < len(ring) and tA[i + 1] <= tB[j + 1]):
                faces.append((ring[i % len(ring)], ring[(i + 1) % len(ring)], ids[j % nk])); i += 1
            else:
                faces.append((ring[i % len(ring)], ids[(j + 1) % nk], ids[j % nk])); j += 1
        ring, ring_t = ids, ths
    centre = add(apex, uv_c)
    new.append(centre); newpos.append(apex)
    for j in range(len(ring)):
        faces.append((ring[j], ring[(j + 1) % len(ring)], centre))
    V = np.vstack([V, np.array(newpos)])
    out = []
    for f in faces:
        f = tuple(int(x) for x in f)
        if np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]]) @ n < 0:
            f = (f[0], f[2], f[1])
        out.append(f)
    return new, out, V


def main(a):
    H = PMX(a.host)
    V = np.array([v[0] for v in H.vertices])
    nbase = len(V)
    names = [b['name'] for b in H.bones]
    ks = [names.index(n) for n in BREAST]
    sp = spans(H)
    F = np.array(H.faces).reshape(-1, 3)
    st = faces_of(H, SKIN)
    skin_idx = np.unique(st)
    front = skin_idx[(V[skin_idx, 0] > 0.1) & (V[skin_idx, 1] > 14.5) & (V[skin_idx, 1] < 18.0)]
    for name in CUPS:
        if name in sp:
            ci = np.unique(faces_of(H, name))
            front = np.concatenate([front, ci[(V[ci, 0] > 0.1) & (V[ci, 1] > 14.5) & (V[ci, 1] < 18.0)]])
    apex = V[front[np.argmin(V[front, 2])]]
    ya = apex[1]
    print(f'apex {np.round(apex, 2)}')

    # mirror twins, material by material, found before anything moves
    twin = np.full(len(V), -1)
    for name, (start, n) in sp.items():
        idx = np.unique(F[start:start + n])
        chest = idx[(np.abs(V[idx, 1] - ya) < 2.5) & (V[idx, 2] < 0.5)]
        if len(chest):
            d, j = cKDTree(V[chest] * MIRROR).query(V[chest])
            twin[chest[d < 0.02]] = chest[j[d < 0.02]]

    # the skin over the breast and the cups split, level by level, so both have
    # vertices to curve with; at each level the rim's new midpoints go onto the
    # curve through the rim's corners, the cup's edge follows, and the cup's hem
    # row rides along with the edge, so the nude outline is round and the bra's
    # lower edge stays whole
    bones = [np.array(H.bones[k]['pos']) for k in ks]
    mids, pin = {}, {}
    for level in range(a.levels):
        before = len(V)
        ax = np.abs(V[:, 0])
        box = (ax > 0.03) & (ax < 2.0) & (V[:, 1] > ya - 1.4) & (V[:, 1] < ya + 1.2) & (V[:, 2] < WALL + 0.1)
        on = np.zeros(len(V), bool)
        on[np.unique(faces_of(H, SKIN))] = True
        V, m1, _ = split(H, V, SKIN, on & box)
        on = np.zeros(len(V), bool)
        on[np.unique(faces_of(H, CUP))] = True
        V, m2, _ = split(H, V, CUP, on & np.concatenate([box, np.ones(len(V) - len(box), bool)]))
        level_mids = {**m1, **m2}
        mids.update(level_mids)
        twin = np.concatenate([twin, np.full(len(V) - len(twin), -1)])
        for (p, q), m in level_mids.items():
            if twin[p] >= 0 and twin[q] >= 0:
                twin[m] = level_mids.get((min(twin[p], twin[q]), max(twin[p], twin[q])), -1)
        st = faces_of(H, SKIN)
        skin_idx = np.unique(st)
        # cup vertices that share a skin vertex's position are welded to it throughout
        tree = cKDTree(V[skin_idx])
        pin = {}
        for name in CUPS:
            if name in spans(H):
                idx = np.unique(faces_of(H, name))
                d, j = tree.query(V[idx])
                pin.update({int(c): int(skin_idx[k]) for c, k, dd in zip(idx, j, d) if dd < 1e-3})
        pinned_ids = np.array(sorted(pin))
        held = V.copy()
        rounded = 0
        for loop in loops_of(V, st):
            if min(np.linalg.norm(V[loop].mean(0) - b) for b in bones) > 1.4:
                continue
            orig = [v for v in loop if v < before]
            n = len(orig)
            for i in range(n):
                p, q = orig[i], orig[(i + 1) % n]
                m = level_mids.get((min(p, q), max(p, q)))
                if m is not None:
                    V[m] = (9 * (V[p] + V[q]) - V[orig[i - 1]] - V[orig[(i + 2) % n]]) / 16
                    rounded += 1
        for c, k in pin.items():
            V[c] = V[k]
        cup_idx = np.unique(faces_of(H, CUP))
        d, j = cKDTree(held[pinned_ids]).query(held[cup_idx])
        hem = cup_idx[(d < HEM_ZONE) & (d > 1e-6)]
        V[hem] += V[pinned_ids[j[(d < HEM_ZONE) & (d > 1e-6)]]] - held[pinned_ids[j[(d < HEM_ZONE) & (d > 1e-6)]]]
        print(f'level {level + 1}: rim rounded at {rounded} midpoints, {len(pin)} cup vertices on the rim, '
              f'{len(hem)} hem vertices riding with it')
    F = np.array(H.faces).reshape(-1, 3)
    sp = spans(H)
    st = faces_of(H, SKIN)
    skin_idx = np.unique(st)
    cup_idx = np.unique(faces_of(H, CUP))
    pinned = np.zeros(len(V), bool)
    pinned[list(pin)] = True
    cnt = {}
    for t in st:
        for p, q in ((0, 1), (1, 2), (2, 0)):
            e = (min(int(t[p]), int(t[q])), max(int(t[p]), int(t[q])))
            cnt[e] = cnt.get(e, 0) + 1
    held = np.zeros(len(V), bool)
    held[[v for e, c in cnt.items() if c == 1 for v in e]] = True

    # what the cage may move, and what the smoothing may
    gate = np.ones(len(V))
    for k in range(3):
        gate *= smoothstep(V[:, k] - BOX_LO[k], 0, 0.4) * (1 - smoothstep(V[:, k] - BOX_HI[k], -0.4, 0))
    ax = np.abs(V[:, 0])
    smooth = np.zeros(len(V))
    smooth[skin_idx] = 1
    smooth *= smoothstep(V[:, 1], ya - 1.5, ya - 1.1) * (1 - smoothstep(V[:, 1], ya - 0.35, ya - 0.1))
    smooth *= smoothstep(-V[:, 2], 1.1, 1.5) * smoothstep(ax, 0.03, 0.3) * (1 - smoothstep(ax, 1.6, 1.9))
    smooth[held] = 0

    P, D = lattice(a, apex)
    rig_pts, rig_ref = [], []
    for k in ks:
        rig_pts.append(H.bones[k]['pos']); rig_ref.append(('bone', k, H.bones[k]['name']))
    bodies = [i for i, r in enumerate(H.rigids) if r['bone'] in ks]
    for i in bodies:
        rig_pts.append(H.rigids[i]['pos']); rig_ref.append(('rigid', i, H.rigids[i]['name']))
    for i, jn in enumerate(H.joints):
        if jn['a'] in bodies or jn['b'] in bodies:
            rig_pts.append(jn['pos']); rig_ref.append(('joint', i, jn['name']))

    tmp = tempfile.mkdtemp(prefix='bust-')
    src, dst = os.path.join(tmp, 'in.npz'), os.path.join(tmp, 'out.npz')
    np.savez(src, V=V, mesh_names=np.array([SKIN, CUP]), faces_skin=st, faces_cloth01=faces_of(H, CUP),
             gate=gate, smooth=smooth, box_lo=BOX_LO, box_hi=BOX_HI, dims=DIMS,
             disp=D, rig=np.array(rig_pts, float), **{'lambda': a.lam, 'iterations': a.iters})
    run = subprocess.run([BLENDER, '--background', '--python', SCRIPT, '--', src, dst, os.path.abspath(a.blend)],
                         capture_output=True, text=True)
    if run.returncode:
        sys.exit(run.stdout[-3000:] + run.stderr[-3000:])
    said = [ln for ln in run.stdout.splitlines() if ln.startswith('lattice')]
    if not said:
        sys.exit('Blender gave no result:\n' + run.stdout[-2000:] + run.stderr[-2000:])
    print(said[-1])
    got = np.load(dst)
    V2, rig2 = got['V'], got['rig']
    moved = np.linalg.norm(V2 - V, axis=1)
    print(f'cage: {int((moved > 1e-4).sum())} vertices moved, furthest {moved.max():.3f}; '
          f'apex now {np.round(V2[front[np.argmin(V[front, 2])]], 2)}')
    V0, V = V, V2
    # the cup's edge welded to the rim
    for c, k in pin.items():
        V[c] = V[k]

    # the holes under the cups filled: the right one built, a round breast where
    # the cup is a cone, the left one its mirror
    bone = np.array(H.bones[ks[1]]['pos'])
    loops = [l for l in loops_of(V, st) if np.linalg.norm(V[l].mean(0) - bone) < 1.4]
    rim_r = np.array(loops[0])
    seg = np.linalg.norm(np.roll(V[rim_r], -1, axis=0) - V[rim_r], axis=1)
    theta = np.concatenate([[0.0], np.cumsum(seg)[:-1]]) / seg.sum()
    uv_c, uv_r = paint_patch(H, os.path.dirname(os.path.abspath(a.host)), rim_r, theta)
    cupf = faces_of(H, CUP)
    cen = V[cupf].mean(1)
    side = cupf[(cen[:, 0] < -0.05) & (np.abs(cen[:, 1] - ya) < 1.5) & (cen[:, 2] < WALL + 0.3)]
    new_r, faces_r, V = fill_cup(H, V, rim_r, side, st, -1.0, a.rings, a.crown, a.fullness, a.upper, uv_c, uv_r)
    lcup = np.unique(cupf[cen[:, 0] > 0.05])
    lnear = cKDTree(V[lcup])
    remap = {}
    newpos = []
    for i in new_r:
        pos = V[i] * MIRROR
        b, w = bones_of(H.vertices[lcup[lnear.query(pos)[1]]])
        rec = list(H.vertices[i]); rec[0] = [float(x) for x in pos]; rec[5], rec[6] = b, w
        H.vertices.append(tuple(rec))
        remap[i] = len(H.vertices) - 1
        newpos.append(pos)
    V = np.vstack([V, np.array(newpos)])
    V0 = np.vstack([V0, V[len(V0):]])
    twin = np.concatenate([twin, np.full(len(V) - len(twin), -1)])
    faces_l = [(remap[f[0]], remap[f[2]], remap[f[1]]) for f in faces_r]
    start, n = sp[SKIN]
    F = np.array(H.faces).reshape(-1, 3)
    H.faces = np.concatenate([F[:start + n], np.array(faces_r + faces_l), F[start + n:]]).ravel().tolist()
    H.materials[[m['name'] for m in H.materials].index(SKIN)]['fcount'] += 3 * (len(faces_r) + len(faces_l))
    sp = spans(H)
    F = np.array(H.faces).reshape(-1, 3)
    st = faces_of(H, SKIN)
    print(f'holes under the cups filled: {len(new_r)} vertices and {len(faces_r)} triangles a side')

    # the cups laid over the breast: over the fill every cup vertex goes onto the
    # dome a hair off it, the move fading to nothing at the welded edge so the
    # hem keeps its line; the ruffle is only kept out of the skin
    ns = smooth_normals(V, st)
    edge_tree = cKDTree(V[list(pin)])
    laid = 0
    for name in CUPS:
        if name not in sp:
            continue
        idx = np.unique(faces_of(H, name))
        idx = idx[(gate[idx] > 0.5) & ~pinned[idx] & (np.abs(V[idx, 0]) > 0.03)]
        for sgn in (-1.0, 1.0):
            sel = idx[np.sign(V[idx, 0]) == sgn]
            sk = st[np.sign(V[st].mean(1)[:, 0]) == sgn]
            m = Mesh(V, sk)
            d, t, w = m.closest(V[sel])
            f = sk[t]
            q = w[:, :1] * V[f[:, 0]] + w[:, 1:2] * V[f[:, 1]] + w[:, 2:] * V[f[:, 2]]
            nrm = w[:, :1] * ns[f[:, 0]] + w[:, 1:2] * ns[f[:, 1]] + w[:, 2:] * ns[f[:, 2]]
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            h = np.einsum('ij,ij->i', V[sel] - q, nrm)
            ed = edge_tree.query(V[sel])[0]
            if name == CUP:
                over = (f >= len(V0) - 2 * len(new_r)).any(1) | (d < 0.05)
                wgt = smoothstep(ed, 0.0, 0.1) * over
                V[sel] += (wgt * (CLEAR - h))[:, None] * nrm
                laid += int((wgt > 0).sum())
            else:
                low = (h < 0) & (d < 0.3) & (ed > 0.02)
                V[sel[low]] += ((0.003 - h)[low][:, None]) * nrm[low]
    print(f'  {len(pin)} cup vertices welded to the rim, {laid} laid over the breast')

    # the bra's straps, thin cords from the cup's corner over the shoulder, would
    # kink where the cage's field fades; each takes the corner's move at its foot,
    # the field at its far end, and a straight blend between, so it keeps its shape
    from collections import defaultdict
    ci = np.unique(cupf)
    cand = set(int(v) for v in ci[(V0[ci, 1] > ya + 1.1) & (V0[ci, 2] < 0.5)
                                  & (np.abs(V0[ci, 0]) > 0.2) & (np.abs(V0[ci, 0]) < 1.5)])
    adj = defaultdict(set)
    for t in cupf:
        for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            if int(p) in cand and int(q) in cand:
                adj[int(p)].add(int(q))
                adj[int(q)].add(int(p))
    rest = np.array(sorted(set(ci.tolist()) - cand))
    rest_tree = cKDTree(V0[rest])
    seen, straps = set(), 0
    for s0 in cand:
        if s0 in seen:
            continue
        comp, stack = [], [s0]
        seen.add(s0)
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        if len(comp) < 10:
            continue
        comp = np.array(comp)
        Q = V0[comp]
        c = Q.mean(0)
        axis = np.linalg.svd(Q - c)[2][0]
        axis *= np.sign(axis[1]) or 1.0
        t = (Q - c) @ axis
        t = (t - t.min()) / (t.max() - t.min())
        D = V[comp] - V0[comp]
        foot = comp[t < 0.15]
        near = rest_tree.query_ball_point(V0[foot], 0.12)
        corner = np.unique(np.concatenate([rest[k] for k in near if k])) if any(near) else None
        lo = (V[corner] - V0[corner]).mean(0) if corner is not None and len(corner) else D[t < 0.2].mean(0)
        hi = D[t > 0.8].mean(0)
        V[comp] = V0[comp] + lo + t[:, None] * (hi - lo)
        straps += 1
    print(f'  {straps} straps kept straight, their feet on the cup')

    # the left side from the right, rig included
    left = np.where((twin >= 0) & (V[:, 0] > 0.03))[0]
    V[left] = V[twin[left]] * MIRROR
    for r, (kind, i, name) in enumerate(rig_ref):
        if name.startswith('左'):
            other = [q for q, (k2, _, n2) in enumerate(rig_ref) if k2 == kind and n2 == '右' + name[1:]]
            if other:
                rig2[r] = rig2[other[0]] * MIRROR
    for i, j in remap.items():
        V[j] = V[i] * MIRROR
    # left vertices with no twin, the cup's tip among them, take the right
    # side's field at their mirrored place, so the two cups still match
    g = np.concatenate([gate, np.ones(len(V) - len(gate))]) > 0.5
    lone = np.where((twin < 0) & (V0[:, 0] > 0.03) & g)[0]
    lone = lone[lone < len(V0) - 2 * len(new_r)]
    right = np.where((V0[:, 0] < -0.03) & g)[0]
    right = right[right < len(V0) - 2 * len(new_r)]
    if len(lone) and len(right):
        d, j = cKDTree(V0[right]).query(V0[lone] * MIRROR, k=3)
        wts = 1 / (d + 1e-4)
        wts /= wts.sum(1, keepdims=True)
        D = np.einsum('ij,ijk->ik', wts, (V[right] - V0[right])[j])
        V[lone] = V0[lone] + D * MIRROR
        print(f'  {len(lone)} left vertices without a twin took the mirrored field')

    for i in range(len(V)):
        v = list(H.vertices[i]); v[0] = [float(x) for x in V[i]]; H.vertices[i] = tuple(v)
    fresh = np.zeros(len(V), bool)
    fresh[:len(moved)] = moved > 1e-4
    fresh[nbase:] = True
    for name, (start, n) in sp.items():
        tris = F[start:start + n]
        idx = np.unique(tris)
        if fresh[idx].any():
            N = welded_normals(V, tris, idx)
            for i in idx[fresh[idx]]:
                v = list(H.vertices[i]); v[1] = [float(x) for x in N[i]]; H.vertices[i] = tuple(v)
    # the rig rides the same cage, and the 胸 bodies shrink with the breast
    for (kind, i, name), p in zip(rig_ref, rig2):
        p = [float(x) for x in p]
        if kind == 'bone':
            H.bones[i]['pos'] = p
        elif kind == 'rigid':
            H.rigids[i]['pos'] = p
            H.rigids[i]['size'] = [float(s * a.deep) for s in H.rigids[i]['size']]
        else:
            H.joints[i]['pos'] = p
    write_pmx(a.out, H)
    for name in DROP:
        if name in sp:
            dropmaterial.main(a.out, name)
    print('wrote', a.out)
    validate.check(a.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--host', default='public/models/reze-midnight/reze-midnight.pmx')
    p.add_argument('--out', default='public/models/reze-midnight/reze-midnight-slim.pmx')
    p.add_argument('--blend', default='public/models/reze-midnight/reze-midnight-bust.blend')
    p.add_argument('--wide', type=float, default=0.74, help='the bust drawn in across, about |x|=0.4, by this')
    p.add_argument('--deep', type=float, default=0.87, help='and back toward the chest wall by this')
    p.add_argument('--lift', type=float, default=0.2, help='the underside lifted by this at the crease')
    p.add_argument('--tip', type=float, default=0.3, help='the tip of the breast lifted by this, its root staying')
    p.add_argument('--crease', type=float, default=0.75, help='how far below the apex the crease is')
    p.add_argument('--drop', type=float, default=0.5, help='how much of the stand-off goes below the tip, the water-drop')
    p.add_argument('--lam', type=float, default=1.0, help='Laplacian smoothing strength on the underboob')
    p.add_argument('--iters', type=int, default=4, help='and its iterations')
    p.add_argument('--levels', type=int, default=2, help='how many times the breast and cups split one into four')
    p.add_argument('--rings', type=int, default=10, help='rings of vertices filling each cup')
    p.add_argument('--crown', type=float, default=0.9, help='the dome\'s tangent at the tip, as a share of the chord: flatter above 1')
    p.add_argument('--fullness', type=float, default=1.0, help='the dome\'s tangent at the rim underneath and at the sides: rounder above 1')
    p.add_argument('--upper', type=float, default=0.6, help='the same on the upper slope')
    main(p.parse_args())
