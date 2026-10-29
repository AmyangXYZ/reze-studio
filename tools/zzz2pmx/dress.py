# -*- coding: utf-8 -*-
"""Put a garment from one body on another, the way a dresser does: by bones.

Nothing is projected onto the skin. A per-vertex conform re-applies a garment's
standoff along the target skin's normal at the nearest skin point -- right for a
strap, wrong for a collar half a unit off the neck, whose nearest skin is the jaw or
the shoulder: the collar shot out along it and crumpled. What PmxDressup does instead,
and what this does:

1. every bone of the garment resolves to a bone of the target by name, or up the
   donor's parent chain to one the target has;
2. each bone carries a rigid transform -- the rotation aligning its segment on the
   donor to its segment on the target, and the offset between the two bone positions;
3. each bone carries a GIRTH scale: the ratio of the two bodies' radial size around
   that bone, measured on the skin vertices the bone drives -- perpendicular to the
   segment only, so a sleeve narrows to a thinner arm and a shirt widens over a larger
   bust without the garment changing length;
4. a vertex goes where its weighted bones send it, blended by its own skin weights, so
   the garment keeps the shape it was made with;
5. whatever then sits inside the skin is lifted out along the garment's own normal
   (`lift_along`, a ray to the skin in front), then `push_out` for the rest -- both
   blurred moves that cannot crease it; and where the skin still comes through BETWEEN the
   garment's vertices -- a fuller bust under a coarser shirt -- the triangles over it
   are lifted by the depth it shows, blurred the same way, until nothing shows.

    python3 tools/zzz2pmx/dress.py <donor.pmx> <target.pmx> <piece>[,piece...] <out.pmx>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holefill
from graftlegs import rot_between
from pmx import PMX
from pmxw import write_pmx
from bustmerge import _hidden
from surface import Mesh, _blur, push_out

CLAMP = (0.6, 1.6)       # a girth ratio outside this is a measurement error
MIN_DRIVEN = 12          # skin vertices a bone must drive to have a girth of its own
FREE = 0.02              # the garment sits at least this far off the skin, after


def resolve(name, parent, have):
    seen = 0
    while name is not None and seen < 12:
        if name in have:
            return name
        name = parent.get(name)
        seen += 1
    return None


def segments(m):
    """Each bone's position and the direction of its segment."""
    names = [b['name'] for b in m.bones]
    pos = {b['name']: np.array(b['pos'], float) for b in m.bones}
    kids = {}
    for b in m.bones:
        if 0 <= b['parent'] < len(names):
            kids.setdefault(names[b['parent']], []).append(b['name'])
    out = {}
    for b in m.bones:
        n = b['name']
        if b['flag'] & 0x0001 and 0 <= b.get('tail', -1) < len(names):
            d = pos[names[b['tail']]] - pos[n]
        elif 'tail_offset' in b and np.linalg.norm(b['tail_offset']) > 1e-6:
            d = np.array(b['tail_offset'], float)
        elif n in kids:
            d = np.mean([pos[k] for k in kids[n]], 0) - pos[n]
        else:
            d = np.zeros(3)
        l = np.linalg.norm(d)
        out[n] = (pos[n], d / l if l > 1e-6 else np.array([0.0, 1.0, 0.0]), l)
    return out


def girths(m, body_mats, seg):
    """Median radial distance of the skin from each bone's axis, per bone."""
    names = [b['name'] for b in m.bones]
    V = np.array([v[0] for v in m.vertices])
    ids = np.unique(holefill.tris_of(m, body_mats))
    by = {}
    for i in ids:
        v = m.vertices[int(i)]
        k = int(np.argmax(v[6][:len(v[5])]))
        by.setdefault(names[v[5][k]], []).append(int(i))
    out = {}
    for n, vs in by.items():
        if len(vs) < MIN_DRIVEN or n not in seg:
            continue
        b, d, _ = seg[n]
        r = V[vs] - b
        rad = np.linalg.norm(r - np.outer(r @ d, d), axis=1)
        out[n] = float(np.median(rad))
    return out


def fit(donor, target, pieces, donor_body=('skin',), target_body=('skin',)):
    dnames = [b['name'] for b in donor.bones]
    tnames = [b['name'] for b in target.bones]
    dparent = {b['name']: (dnames[b['parent']] if 0 <= b['parent'] < len(dnames) else None)
               for b in donor.bones}
    have = set(tnames)
    ds, ts = segments(donor), segments(target)
    dg, tg = girths(donor, donor_body, ds), girths(target, target_body, ts)
    M, used = {}, {}
    for n in dnames:
        tn = resolve(n, dparent, have)
        if tn is None:
            continue
        a0, da, _ = ds[n]
        b0, db, _ = ts[tn]
        R = rot_between(da, db)
        g = 1.0
        if n in dg and tn in tg and dg[n] > 1e-6:
            g = float(np.clip(tg[tn] / dg[n], *CLAMP))
        M[n] = (a0, b0, R, db, g, tn)
    DV = np.array([v[0] for v in donor.vertices])
    DN = np.array([v[1] for v in donor.vertices])
    placed, normal = {}, {}
    for piece in pieces:
        for i in np.unique(holefill.tris_of(donor, [piece])):
            v = donor.vertices[int(i)]
            acc, nacc, tot = np.zeros(3), np.zeros(3), 0.0
            for b, w in zip(v[5], v[6]):
                if w <= 0 or dnames[b] not in M:
                    continue
                a0, b0, R, db, g, tn = M[dnames[b]]
                r = R @ (DV[i] - a0)
                par = (r @ db) * db
                acc += w * (b0 + par + g * (r - par))
                nacc += w * (R @ DN[i])
                tot += w
                used[tn] = used.get(tn, 0) + 1
            placed[int(i)] = DV[i] if tot < 1e-6 else acc / tot
            normal[int(i)] = DN[i] if tot < 1e-6 else nacc / (np.linalg.norm(nacc) + 1e-12)
    ratios = {tn: M[n][4] for n in M for tn in [M[n][5]] if tn in used}
    return placed, normal, M, ratios


def first_hit(P, D, V, T, far, chunk=64):
    """Distance along each ray to the first triangle it meets, inf where none."""
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1, e2 = b - a, c - a
    out = np.full(len(P), np.inf)
    for k in range(0, len(P), chunk):
        p, d = P[k:k + chunk], D[k:k + chunk]
        h = np.cross(d[:, None, :], e2[None, :, :])
        det = np.einsum('tj,ptj->pt', e1, h)
        ok = np.abs(det) > 1e-9
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s_ = p[:, None, :] - a[None, :, :]
        u = np.einsum('ptj,ptj->pt', s_, h) * inv
        q = np.cross(s_, e1[None, :, :])
        v = np.einsum('pj,ptj->pt', d, q) * inv
        t = np.einsum('tj,ptj->pt', e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4) & (t < far)
        t = np.where(hit, t, np.inf)
        out[k:k + chunk] = t.min(1)
    return out


def lift_along(skin, P, Nv, offset, far=0.8, rounds=20, relax=0.7, blur=0.35):
    """Lift what sits under the skin out along the garment's OWN normal: a ray from
    each vertex along it finds the skin in front, and the vertex goes to that hit plus
    the standoff. Lifting along the skin's normal at the nearest point instead folds
    a bodice between the breasts, where neighbours' nearest points sit on opposite
    breasts and are pushed apart. Blurred across the garment, as ever."""
    P = np.array(P, float)
    Nv = Nv / (np.linalg.norm(Nv, axis=1, keepdims=True) + 1e-12)
    moved = np.zeros(len(P), bool)
    for _ in range(rounds):
        t = first_hit(P, Nv, skin.V, skin.F, far)
        inside = np.isfinite(t)
        if not inside.any():
            break
        delta = np.zeros_like(P)
        delta[inside] = Nv[inside] * (t[inside] + offset)[:, None]
        delta = _blur(P, delta, blur)
        P += relax * delta
        moved |= inside
    return P, moved


def clear(t, piece, target_body, free, rounds=12):
    """No skin vertex may have the garment under it."""
    V = np.array([v[0] for v in t.vertices])
    N = np.array([v[1] for v in t.vertices])
    T = holefill.tris_of(t, [piece])
    ids = np.unique(T)
    skin = np.unique(holefill.tris_of(t, list(target_body)))
    lo, hi = V[ids, 1].min() - 0.1, V[ids, 1].max() + 0.1
    skin = skin[(V[skin, 1] > lo) & (V[skin, 1] < hi)]
    local = {int(i): k for k, i in enumerate(ids)}
    P = V[ids].copy()
    total = 0
    for _ in range(rounds):
        V[ids] = P
        covered = _hidden(V[skin], N[skin], V, T, 2.0)
        under = _hidden(V[skin], -N[skin], V, T, 0.12)
        bad = skin[~covered & under]
        if len(bad) == 0:
            break
        cm = Mesh(V, T)
        d, tri, _ = cm.closest(V[bad])
        delta = np.zeros_like(P)
        for depth, f, n in zip(d, T[tri], N[bad]):
            for i in f:
                k = local[int(i)]
                lift = n * (depth + free)
                if np.linalg.norm(lift) > np.linalg.norm(delta[k]):
                    delta[k] = lift
        P += _blur(P, delta, 0.35)
        total += len(bad)
    for k, i in enumerate(ids):
        v = list(t.vertices[int(i)])
        v[0] = [float(x) for x in P[k]]
        t.vertices[int(i)] = tuple(v)
    return len(bad)


def hair_top(m, hair):
    """Where a thing worn on the head is worn: the crown of the hair."""
    V = np.array([v[0] for v in m.vertices])
    P = V[np.unique(holefill.tris_of(m, [hair]))]
    top = P[:, 1].max()
    crown = P[P[:, 1] > top - 1.0]
    return np.array([crown[:, 0].mean(), top, crown[:, 2].mean()])


def dress(donor_path, target_path, pieces, out, sheet, donor_body=('skin',),
          target_body=('skin', 'skin_hands'), free=FREE, on_head=(), hair='头发'):
    """`on_head` names the pieces worn on the hair -- ears, a headband, a hat. A
    head bone sits at its own depth in each skull, so those go by the crown of the
    hair instead: carried whole, by the offset between the two crowns, and left where
    they land, since the scalp under the hair is skin and a lift would bury them in
    the hair or crumple them."""
    d, t = PMX(donor_path), PMX(target_path)
    placed, normal, M, ratios = fit(d, t, pieces, donor_body, target_body)
    if on_head:
        shift = hair_top(t, hair) - hair_top(d, hair)
        DV = np.array([v[0] for v in d.vertices])
        DN = np.array([v[1] for v in d.vertices])
        for piece in on_head:
            for i in np.unique(holefill.tris_of(d, [piece])):
                placed[int(i)] = DV[i] + shift
                normal[int(i)] = DN[i]
        print(f'  worn on the hair, by the crown: {", ".join(on_head)} moved by {np.round(shift, 2)}')
    tidx = {b['name']: i for i, b in enumerate(t.bones)}
    dnames = [b['name'] for b in d.bones]
    # one sheet for every piece, or (sheet=None) each piece's own, by file name --
    # the caller has copied the files beside the target
    if sheet is not None and sheet not in t.textures:
        t.textures.append(sheet)

    def texture(mat):
        if sheet is not None:
            return t.textures.index(sheet)
        if mat['tex'] < 0:
            return -1
        rel = 'textures\\' + os.path.basename(d.textures[mat['tex']].replace('\\', '/'))
        if rel not in t.textures:
            t.textures.append(rel)
        return t.textures.index(rel)
    told = ', '.join(f'{k} x{v:.2f}' for k, v in sorted(ratios.items(), key=lambda x: -abs(x[1] - 1))[:8])
    print(f'  girth by bone, the furthest from 1: {told}')
    for piece in pieces:
        tris = holefill.tris_of(d, [piece])
        keep = np.unique(tris)
        base, remap = len(t.vertices), {}
        for i in keep:
            v = d.vertices[int(i)]
            bones = [tidx.get(M[dnames[b]][5], 0) if dnames[b] in M else 0 for b in v[5]]
            w = np.array(v[6], float)
            if w[:len(v[5])].sum() > 0:
                w[:len(v[5])] /= w[:len(v[5])].sum()
            rec = ([float(x) for x in placed[int(i)]], [float(x) for x in normal[int(i)]],
                   list(v[2]), [np.zeros(4) for _ in range(t.uv_add)], v[4], bones,
                   [float(x) for x in w], v[7])
            remap[int(i)] = base + len(remap)
            t.vertices.append(rec + (v[8],) if v[4] == 3 else rec)
        faces = [[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in tris]
        t.faces = list(t.faces) + [x for f in faces for x in f]
        mat = dict(next(x for x in d.materials if x['name'] == piece))
        mat['name'] = mat['name_en'] = piece
        mat['fcount'] = len(faces) * 3
        mat['tex'] = texture(mat)
        mat['sph'] = -1
        mat['sph_mode'] = 0
        mat['toon_shared'] = t.materials[0]['toon_shared']
        mat['toon'] = t.materials[0]['toon']
        t.materials.append(mat)
        print(f'  {piece}: {len(faces)} triangles placed by bone')
    # what sank into the skin comes out, blurred across the garment
    V = np.array([v[0] for v in t.vertices])
    skin = Mesh(V, holefill.tris_of(t, list(target_body)))
    for piece in pieces:
        if piece in on_head:
            continue
        ids = np.unique(holefill.tris_of(t, [piece]))
        Nv = np.array([t.vertices[int(i)][1] for i in ids])
        moved, lifted = lift_along(skin, V[ids], Nv, offset=free)
        moved, more = push_out(skin, moved, offset=free, rounds=20)
        lifted |= more
        for k, i in enumerate(ids):
            v = list(t.vertices[int(i)])
            v[0] = [float(x) for x in moved[k]]
            t.vertices[int(i)] = tuple(v)
        print(f'  {piece}: {int(lifted.sum())} of {len(ids)} vertices lifted out of the skin')
    for piece in pieces:
        if piece in on_head:
            continue
        left = clear(t, piece, target_body, free)
        print(f'  {piece}: skin coming through between its vertices -- {left} left')
    # the lifts moved the surface; its normals are rebuilt from the faces they shade
    V = np.array([v[0] for v in t.vertices])
    for piece in pieces:
        T = holefill.tris_of(t, [piece])
        n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        acc = np.zeros_like(V)
        for k in range(3):
            np.add.at(acc, T[:, k], n)
        for i in np.unique(T):
            ln = np.linalg.norm(acc[i])
            if ln > 1e-12:
                v = list(t.vertices[int(i)])
                v[1] = (acc[i] / ln).tolist()
                t.vertices[int(i)] = tuple(v)
    write_pmx(out, t)


if __name__ == '__main__':
    a = sys.argv[1:]
    dress(a[0], a[1], a[2].split(','), a[3], a[4] if len(a) > 4 else 'textures\\cloth01.png')
