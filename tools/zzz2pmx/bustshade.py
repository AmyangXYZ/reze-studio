# -*- coding: utf-8 -*-
"""Re-bake the chest shading in the skin sheet for the chest the model now has.

`bustfit.py` gives the body reze's chest; the skin sheet still carries the shading
its author painted for the larger one -- a hard crescent under each breast that,
after the shrink, sits a third of a unit below the crease it was painted for. Seen
from below that is two creases: one in the geometry and one in the paint.

The painted shading is low-frequency, so it can be taken out without taking the
skin with it: the region's luminance blurred wide is the old shade, and dividing by
it leaves the skin's own colour and its pores. New shading is then baked off the
mesh -- ambient occlusion by hemisphere sampling against the chest itself, which
puts the shadow under and between the breasts where they actually are, plus a mild
top-light so the underside turns away as skin does. Its strength is matched to what
the sheet had, so the chest does not read flatter than the body around it. The
result is feathered into the sheet and written as its own file; only `skin` points
at it, so every other material on this atlas keeps the original.

    python3 tools/zzz2pmx/bustshade.py [--pmx <slim.pmx>] [--out-tex cloth01_slim.png]
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bustfit import smooth_normals, smoothstep, window, ANCHOR
from graftlegs import material_slice
from pmx import PMX
from pmxw import write_pmx

PMX_IN = 'public/models/reze-midnight/reze-midnight-slim.pmx'
SKIN = 'skin'
RAYS = 48           # hemisphere directions per texel
REACH = 1.1         # how far an occluder counts
TONE_SIGMA = 40     # px; the skin tone under the paint, wide enough to hold no band
DETAIL_SIGMA = 3    # px; below this is pores, kept; above it is paint, replaced
SHADE_SIGMA = 20    # px; the bake is faceted by the mesh it is cast against, and a crease
                    # that stops in a step is worse than one that fades -- 10 px still stepped
AO_STRENGTH = 0.70  # darkest a fully occluded texel goes; 0.45 was invisible in the engine, 0.85 a band
SHADOW_TINT = (1.12, 1.0, 0.88)   # per channel: red darkens most, blue least
FEATHER = 14        # px into the untouched sheet
TOPLIGHT = 0.35     # how much the underside darkens for facing down: the under-bust gradient


def blocked_any(O, D, V, T, reach, chunk=1500):
    """Whether each ray meets any triangle within `reach`: Moller-Trumbore over all
    rays and all triangles at once, in chunks of rays."""
    v0 = V[T[:, 0]]; e1 = V[T[:, 1]] - v0; e2 = V[T[:, 2]] - v0
    out = np.zeros(len(O), bool)
    for s0 in range(0, len(O), chunk):
        o, d = O[s0:s0 + chunk], D[s0:s0 + chunk]
        p = np.cross(d[:, None, :], e2[None])                       # (n,T,3)
        det = np.einsum('tj,ntj->nt', e1, p)
        ok = np.abs(det) > 1e-9
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        sv = o[:, None, :] - v0[None]
        u = np.einsum('ntj,ntj->nt', sv, p) * inv
        q = np.cross(sv, e1[None])
        v = np.einsum('nj,ntj->nt', d, q) * inv
        t = np.einsum('tj,ntj->nt', e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t < reach)
        out[s0:s0 + chunk] = hit.any(1)
    return out


def hemisphere(n, k, seed=3):
    """k cosine-weighted directions about each normal (N,3) -> (N,k,3)."""
    rng = np.random.default_rng(seed)
    u1, u2 = rng.random(k), rng.random(k)
    r, phi = np.sqrt(u1), 2 * np.pi * u2
    local = np.column_stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)])
    a = np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    t = np.cross(n, a); t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(n, t)
    return (local[None, :, 0, None] * t[:, None] + local[None, :, 1, None] * b[:, None]
            + local[None, :, 2, None] * n[:, None])


def main(a):
    m = PMX(a.pmx)
    V = np.array([v[0] for v in m.vertices])
    UV = np.array([v[2] for v in m.vertices])
    bh = {b['name']: np.array(b['pos']) for b in m.bones}
    st, mat = material_slice(m, SKIN)
    N = smooth_normals(V, st)
    tex_rel = m.textures[mat['tex']].replace('\\', '/')
    tex_path = os.path.join(os.path.dirname(a.pmx), tex_rel)
    im = np.asarray(Image.open(tex_path).convert('RGBA')).astype(np.float64)
    H, W = im.shape[:2]

    # the chest, on the sheet: triangles whose vertices the window reaches
    w = window(V, bh[ANCHOR])
    tris = st[(w[st] > 1e-3).any(1)]
    print(f'chest on the sheet: {len(tris)} skin triangles, sheet {W}x{H}')

    # rasterise them in UV space: per texel, a position and a normal
    P = np.zeros((H, W, 3)); NN = np.zeros((H, W, 3)); cov = np.zeros((H, W), bool)
    for t in tris:
        uv = UV[t] * [W, H]
        x0, y0 = np.floor(uv.min(0)).astype(int); x1, y1 = np.ceil(uv.max(0)).astype(int)
        xs, ys = np.meshgrid(np.arange(max(x0, 0), min(x1 + 1, W)), np.arange(max(y0, 0), min(y1 + 1, H)))
        pts = np.column_stack([xs.ravel() + 0.5, ys.ravel() + 0.5])
        d = (uv[1, 0] - uv[0, 0]) * (uv[2, 1] - uv[0, 1]) - (uv[2, 0] - uv[0, 0]) * (uv[1, 1] - uv[0, 1])
        if abs(d) < 1e-9:
            continue
        l1 = ((uv[1, 0] - pts[:, 0]) * (uv[2, 1] - pts[:, 1]) - (uv[2, 0] - pts[:, 0]) * (uv[1, 1] - pts[:, 1])) / d
        l2 = ((uv[2, 0] - pts[:, 0]) * (uv[0, 1] - pts[:, 1]) - (uv[0, 0] - pts[:, 0]) * (uv[2, 1] - pts[:, 1])) / d
        l3 = 1 - l1 - l2
        inside = (l1 >= -1e-3) & (l2 >= -1e-3) & (l3 >= -1e-3)
        if not inside.any():
            continue
        L = np.column_stack([l1, l2, l3])[inside]
        yy, xx = ys.ravel()[inside], xs.ravel()[inside]
        P[yy, xx] = L @ V[t]; NN[yy, xx] = L @ N[t]; cov[yy, xx] = True
    NN[cov] /= np.linalg.norm(NN[cov], axis=1, keepdims=True) + 1e-12
    ys, xs = np.where(cov)
    print(f'  {len(ys)} texels covered')

    # ambient occlusion off the chest itself
    # what casts the occlusion. reze's skin is cut away under her cups, so on her
    # the cups are the breast; without them the cleavage has nothing to shadow it.
    occ = np.concatenate([material_slice(m, k)[0] for k in a.occluders.split(',')])
    c = V[occ].mean(1) - bh[ANCHOR]
    occ = occ[(c[:, 1] > -1.9) & (c[:, 1] < 3.3) & (np.abs(c[:, 0]) < 2.4) & (c[:, 2] < 1.2)]
    origins = P[ys, xs] + NN[ys, xs] * 0.012
    dirs = hemisphere(NN[ys, xs], RAYS)
    cache = os.path.join(os.path.dirname(a.pmx), '_chest_ao.npz')
    ao = None
    if os.path.exists(cache) and not a.rebake:
        z = np.load(cache)
        if len(z['ys']) == len(ys) and (z['ys'] == ys).all() and (z['xs'] == xs).all():
            ao = z['ao']
            print('  ao from cache')
    if ao is None:
        blocked = np.zeros(len(ys))
        for k in range(RAYS):
            blocked += blocked_any(origins, dirs[:, k], V, occ, REACH)
            if k % 12 == 11:
                print(f'  ao {k + 1}/{RAYS}')
        ao = blocked / RAYS
        np.savez(cache, ys=ys, xs=xs, ao=ao)
    top = np.clip(NN[ys, xs, 1], -1, 1)                       # +1 faces up
    # a soft knee at both ends: nothing at a whisper of occlusion, and the deepest
    # of the crease not darker than the rest by a step
    ao_c = smoothstep(ao, 0.0, 0.85) ** 1.3
    shade_new = (1 - a.ao * ao_c) * (1 - a.toplight * np.clip(-top, 0, 1))

    # -- the skin under the paint. Every low-frequency tone across the island is
    # taken out, whatever colour it is: the shade painted for the old bust lives
    # in the low frequencies, and only the pore-level detail is kept. The grey
    # L-shape on the sheet is atlas padding -- no triangle maps there -- and is
    # left alone by construction. The shading is then the mesh's.
    rgb = im[:, :, :3]
    mk = cov.astype(float)
    base = np.stack([gaussian_filter(rgb[:, :, c] * mk, TONE_SIGMA) for c in range(3)], -1) \
        / np.maximum(gaussian_filter(mk, TONE_SIGMA), 1e-6)[..., None]
    low = np.stack([gaussian_filter(rgb[:, :, c] * mk, DETAIL_SIGMA) for c in range(3)], -1) \
        / np.maximum(gaussian_filter(mk, DETAIL_SIGMA), 1e-6)[..., None]
    high = rgb - low
    shade = np.ones((H, W)); shade[ys, xs] = shade_new
    shade[cov] = gaussian_filter(shade * mk, SHADE_SIGMA)[cov] / np.maximum(gaussian_filter(mk, SHADE_SIGMA)[cov], 1e-6)
    print(f'  baked shade darkest {np.percentile(shade[cov], 3):.3f}, median {np.median(shade[cov]):.3f}')

    # Darkened by a straight multiply a pink base goes orange in the crease; the
    # shadow is pulled toward mauve instead -- red darkens most, blue least.
    shade_rgb = 1 - (1 - shade)[..., None] * np.array(SHADOW_TINT)
    reshaded = np.clip(base * shade_rgb + high, 0, 255)
    feather = np.clip(distance_transform_edt(cov) / FEATHER, 0, 1)[..., None]
    out = np.where(cov[..., None], reshaded * feather + rgb * (1 - feather), rgb)
    result = im.copy(); result[:, :, :3] = out
    name = a.out_tex
    Image.fromarray(result.astype(np.uint8)).save(os.path.join(os.path.dirname(tex_path), name))

    rel = 'textures\\' + name
    if rel not in m.textures:
        m.textures.append(rel)
    mat['tex'] = m.textures.index(rel)
    write_pmx(a.pmx, m)
    print(f'wrote textures/{name}; {SKIN} now reads it, everything else keeps {os.path.basename(tex_rel)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pmx', default=PMX_IN)
    p.add_argument('--out-tex', default='cloth01_reze_slim.png', dest='out_tex')
    p.add_argument('--rebake', action='store_true')
    p.add_argument('--occluders', default=SKIN, help='materials that cast the occlusion, comma-separated')
    p.add_argument('--ao', type=float, default=AO_STRENGTH, help='darkest a fully occluded texel goes')
    p.add_argument('--toplight', type=float, default=TOPLIGHT, help='how much the underside darkens for facing down')
    main(p.parse_args())
