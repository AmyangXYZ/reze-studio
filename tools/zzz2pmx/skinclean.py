# -*- coding: utf-8 -*-
"""Repaint a region of a model's skin sheet from a donor that has it clean.

塞勒涅's sheet carries her own costume's shading baked into the lower back -- a dark V
across the sacrum that is hers, not the body's. Undressed and put in someone else's
outfit she wears it anyway, and on the maid it shows in the strip between the skirt's
hem and the underwear.

The two models do not share a UV layout, so there is no rectangle to copy: every texel
of the region is carried to its own point in space, that point is found on the donor's
skin, and the donor's sheet is read where it lands. The transferred colour is then put
to the gain that makes it match our own sheet just outside the stain, so the patch does
not read as a patch.

    python3 tools/zzz2pmx/skinclean.py

Reads the donor and never writes it. Rewrites the build's own copy of the sheet.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holefill
import selenepaint
from pmx import PMX
from surface import Mesh

BUILD = 'public/models/reze-bare/_work/reze-bare.pmx'
SHEET = 'public/models/reze-bare/_work/textures/衣服.png'
DONOR = ('public/models/【深空之眼】幽月·塞勒涅-海滩漫步/海滩漫步.pmx', 'skin')

BODY = 'skin'
REGION = None        # all of it; what is repainted is decided by the paint, not a box
FEATHER = 6          # texels the patch fades over at its edge
GRADE = 0.25         # of the region's own texels, the cleanest quarter sets the gain
# A texel this far from being skin is a garment painted into the sheet, and it is the
# garment that is replaced. `skinness` is selenepaint's: hue, saturation and value
# against the range a 深空之眼 body is painted in.
COSTUME = 0.35
# And a shadow painted onto skin is skin-coloured -- the V across her sacrum rates as
# skin by hue and is simply darker than the body around it, so a "is this skin?" test
# never saw it. This is the test that does: darker than her own body's median by this.
SHADE = 0.07
# What no donor can clean is filled from her own skin around it -- the sacral V is a
# tan mark on an otherwise even region, and 海滩漫步 is darker there than she is. Only
# moderate shadows: a texel darker than this is a garment (her underwear), and stays.
GARMENT = 0.35
LINE = 0.12          # saturation above her skin's that marks a drawn line
GROW = 3             # and its neighbours with it, so an edge is not left behind
SAFE = 0.30          # the raw donor sits ~0.10 off her; a sample within this is on her


def cover(m, tris, UV, w, h):
    """Every texel those triangles own, and the point in space each one carries."""
    P = np.array([v[0] for v in m.vertices])
    px = np.stack([UV[:, 0] % 1.0 * w - 0.5, UV[:, 1] % 1.0 * h - 0.5], 1)
    hit = np.zeros((h, w), bool)
    at = np.zeros((h, w, 3), np.float32)
    for a, b, c in tris:
        t = px[[a, b, c]]
        # a triangle straddling the atlas wrap is not one triangle on the sheet
        if np.ptp(t[:, 0]) > w * 0.5 or np.ptp(t[:, 1]) > h * 0.5:
            continue
        x0, x1 = int(np.floor(t[:, 0].min())), int(np.ceil(t[:, 0].max())) + 1
        y0, y1 = int(np.floor(t[:, 1].min())), int(np.ceil(t[:, 1].max())) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w), min(y1, h)
        if x0 >= x1 or y0 >= y1:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1) + .5, np.arange(y0, y1) + .5)
        d = ((t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1])
             - (t[2, 0] - t[0, 0]) * (t[1, 1] - t[0, 1]))
        if abs(d) < 1e-9:
            continue
        u = ((xs - t[0, 0]) * (t[2, 1] - t[0, 1])
             - (t[2, 0] - t[0, 0]) * (ys - t[0, 1])) / d
        v = ((t[1, 0] - t[0, 0]) * (ys - t[0, 1])
             - (xs - t[0, 0]) * (t[1, 1] - t[0, 1])) / d
        m_ = (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)
        if not m_.any():
            continue
        sub = hit[y0:y1, x0:x1]
        put = m_ & ~sub
        sub[put] = True
        at[y0:y1, x0:x1][put] = (P[a] * (1 - u - v)[..., None]
                                 + P[b] * u[..., None] + P[c] * v[..., None])[put]
    return hit, at


def main(build=BUILD, sheet=SHEET, donor=DONOR, region=REGION):
    m = PMX(build)
    sl = holefill.slices(m)
    V = np.array([v[0] for v in m.vertices])
    UV = np.array([v[2] for v in m.vertices])
    tris = np.array(sl[BODY])
    if region:
        tris = np.array([t for t in tris
                         if (V[t][:, 2] > region['zmin']).all()
                         and (V[t][:, 1] > region['ylo']).all()
                         and (V[t][:, 1] < region['yhi']).all()])
    if not len(tris):
        print('  nothing of the body to repaint')
        return

    img = np.asarray(Image.open(sheet).convert('RGB'), np.float32) / 255
    h, w, _ = img.shape
    own, at = cover(m, tris, UV, w, h)
    # Two texels out from every island edge as well. A rim vertex's own texel is on
    # the edge, where the rasteriser owns about half of them; the glove's cuff line
    # sat exactly there and never got repainted.
    from scipy import ndimage as _nd
    grown = _nd.binary_dilation(own, iterations=2)
    _, (yy, xx) = _nd.distance_transform_edt(~own, return_indices=True)
    at[grown & ~own] = at[yy[grown & ~own], xx[grown & ~own]]
    own = grown
    if not own.any():
        print('  those triangles own no texels')
        return

    # Where her sheet is not skin -- that is the costume, and only that is replaced.
    from scipy import ndimage
    skinny = np.zeros((h, w), np.float32)
    skinny[own] = selenepaint.skinness(img[own])
    lum = np.zeros((h, w), np.float32)
    lum[own] = img[own].mean(1)
    body = np.median(lum[own & (skinny >= COSTUME)]) if (skinny >= COSTUME).any() else 1.0
    mask = own & ((skinny < COSTUME) | (lum < body - SHADE))
    if GROW:
        mask = ndimage.binary_dilation(mask, iterations=GROW) & own
    if not mask.any():
        print(f'  {sheet}: none of it is costume')
        return

    d = PMX(donor[0])
    DV = np.array([v[0] for v in d.vertices])
    dm = Mesh(DV, holefill.slices(d)[donor[1]])
    dmat = [x for x in d.materials if x['name'] == donor[1]][0]
    sheet_d = np.asarray(Image.open(os.path.join(
        os.path.dirname(donor[0]),
        d.textures[dmat['tex']].replace('\\', '/'))).convert('RGB'), np.float32) / 255
    far, col = selenepaint.on(dm, np.array([v[2] for v in d.vertices]), sheet_d, at[mask])

    # The gain that puts the donor's skin to ours, read where both agree it is skin.
    ours = img[own & ~mask]
    _, ref = selenepaint.on(dm, np.array([v[2] for v in d.vertices]), sheet_d,
                            at[own & ~mask])
    both = selenepaint.skinness(ref) > 0.5
    gain = (np.clip(ours[both].mean(0) / np.maximum(ref[both].mean(0), 1e-6), 0.5, 2.0)
            if both.sum() > 32 else np.ones(3, np.float32))
    col = np.clip(col * gain, 0, 1)

    # A sample that landed nowhere near her keeps what was there -- and so does one
    # that is no more skin than what it would replace. 海滩漫步's sandal straps are
    # painted onto an ankle 塞勒涅 leaves bare, and without this the bare ankle was
    # repainted with the straps.
    # A donor sample is taken only if it is cleaner than what it would replace: more
    # skin by hue, and no darker. Either alone is not enough -- his bikini bottom is
    # painted across the same sacrum, and her sandal straps across a bare ankle.
    mine = img[mask]
    hold = ((far > SAFE)
            | (selenepaint.skinness(col) < selenepaint.skinness(mine) - 0.02)
            | (col.mean(1) < mine.mean(1) - 0.02))
    col[hold] = img[mask][hold]

    soft = ndimage.gaussian_filter(mask.astype(np.float32), FEATHER)
    soft = np.clip((soft - 0.5) * 2, 0, 1) * mask
    out = img.copy()
    out[mask] = img[mask] * (1 - soft[mask])[:, None] + col * soft[mask][:, None]

    # The residue: still a shadow after the transfer, but not a garment. Filled from
    # the nearest owned texel that is neither, and feathered so it cannot ring.
    l2 = np.zeros((h, w), np.float32)
    l2[own] = out[own].mean(1)
    sat = np.zeros((h, w), np.float32)
    sat[own] = out[own].max(1) - out[own].min(1)
    body_sat = np.median(sat[own & (l2 >= body - SHADE)])
    # darker than her skin, or a line drawn on it in a stronger colour -- the V's
    # border survived the darkness test because it is red, not dark
    left = own & (((l2 < body - SHADE) & (l2 > body - GARMENT)) | (sat > body_sat + LINE))
    left = ndimage.binary_dilation(left, iterations=3) & own & (l2 > body - GARMENT)
    clean = own & ~left & (l2 >= body - SHADE)
    filled = 0
    if left.any() and clean.any():
        _, (yy, xx) = ndimage.distance_transform_edt(~clean, return_indices=True)
        src = out[yy[left], xx[left]]
        f2 = ndimage.gaussian_filter(left.astype(np.float32), 3)
        f2 = np.clip(f2 * 1.5, 0, 1)[left]
        out[left] = out[left] * (1 - f2)[:, None] + src * f2[:, None]
        filled = int(left.sum())
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(sheet)

    was = selenepaint.skinness(img[own]).mean()
    now = selenepaint.skinness(out[own]).mean()
    dark_was = int((lum[own] < body - SHADE).sum())
    dark_now = int((out[own].mean(1) < body - SHADE).sum())
    print(f'{os.path.basename(sheet)}: {int(mask.sum())} of {int(own.sum())} texels were '
          f'costume, repainted from {os.path.basename(donor[0])}')
    print(f'  put to her skin by x{gain[0]:.2f}/{gain[1]:.2f}/{gain[2]:.2f}; '
          f'{int(hold.sum())} kept for want of a donor surface; '
          f'the body reads {100 * was:.0f}% skin -> {100 * now:.0f}%, and the texels '
          f'shaded darker than her own skin {dark_was} -> {dark_now}; {filled} of the '
          f'residue filled from her own skin')


if __name__ == '__main__':
    main()
