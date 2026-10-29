# -*- coding: utf-8 -*-
"""Put PmxDressup's fitted outfit onto the built model.

The fit itself is PmxDressup's (`scratchpad/dressup.py` drives it headless). This
carries the result across, which is three things and no geometry work at all:

**The height.** PmxDressup's output stands 0.282320 higher than the body it was
fitted to -- exact to a standard deviation of 0.000000, in y alone. That is not drift.
塞勒涅's model is positioned to stand in HER shoes, whose sole carries her 0.346 off
the ground; the maid's shoe is thicker and carries her 0.628. The difference is the
0.282320. So whichever shoes she ends up in decides which height is right, and since
she ends up in the maid's, the BUILD rises to meet the outfit rather than the outfit
coming down to the build.

**The names.** It prefixes everything of the costume's with `Cos:`, bones included.
Stripped, all 370 bones the dress leans on resolve against the build, and the weights
come across whole rather than collapsed onto whatever parent still existed.

**The shells.** Her `cloth01` is the whole outfit in one material, and it separates
cleanly: 32 shells below the ankle are the shoes, 45 above y 17.5 are the headdress,
and the 114 between them are the dress. Only the dress is taken.

The build already wears 沉醉旖旎's shoes, drawn for this exact foot. And it already
wears the headdress `maidbuild` placed, which is the one that sits right: the person
model PmxDressup fits against is 塞勒涅 stripped to skin and face, so it conforms a
headdress to a BALD scalp, and reze's bob then goes on underneath it. `HeadFrame`
places it off the two `face01` meshes on a head that has its hair.

    python3 tools/zzz2pmx/maiddress.py

Reads `reze-maid-dressup/reze-maid.pmx` and never writes it. Rewrites
`reze-maid-selene/reze-maid.pmx`.
"""
import os
import shutil
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dropmaterial
import fixrig
import holefill
from pmx import PMX
from pmxw import write_pmx

FITTED = 'public/models/reze-maid-dressup/reze-maid.pmx'
BUILD = 'public/models/reze-maid-selene/reze-maid.pmx'
# The same thing, kept: `bustmerge` reads it and writes BUILD, so without a copy
# to read from, re-running it means re-running this and `maidselene` first.
PLACED = 'public/models/reze-maid-selene/placed.pmx'
TEXDIR = 'public/models/reze-maid-selene/textures'

# Measured, not guessed: the difference between the two shoes' soles, read off each
# model's own bare sole against its own shoe.
LIFT = 0.282320

# Hers, and what each becomes. The names are what the studio's material classifier
# buckets on, so each piece is named for the shader it wants.
TAKE = [('Cos:cloth01', 'maid_dress', 'cloth'),
        ('Cos:metal', 'maid_metal', 'low'),
        ('Cos:cloth01_alpha', 'maid_lace', None),
        ('Cos:stocking', 'maid_stockings', None),
        ('Cos:skin_stocking', 'maid_underwear', None),
        ('Cos:skin_metal', 'maid_garter_metal', None)]

# Her shoes come too. 沉醉旖旎's were the right answer while the hosiery was the
# maid's own unfitted pair, and are the wrong one now: PmxDressup shaped this hosiery
# for the maid's shoe, so its foot reaches z -0.97 where her bare foot stops at -0.49,
# and 126 of its 140 vertices below the vamp sit outside 沉醉旖旎's shorter shoe. The
# shoe and the hosiery are drawn for each other; they travel together.
SHOES = 'maid_shoes'

# And shortened. PmxDressup scales a garment by the bones under it, and the bone it
# has for a foot is 足首->つま先, which on 塞勒涅 is 1.917 against the maid's 1.614
# because her foot is held 14 degrees flatter -- so the shoe came out 1.80 times her
# foot long where the maid's own is 1.607 times hers, overhanging her toe by 1.44.
# Scaled along its length about the ankle it keeps its height, its heel and its
# section, and only stops being a clown shoe.
TOE = 0.15           # how far past the hosiery's toe the shoe should reach

# The neckline and shoulders, lifted. Matched vertex for vertex against the build
# `maidbuild` placed -- the one that sits right -- this fit comes out 0.098 low across
# the top of the dress, tightly enough that the p90 is 0.009 of the median. Lower down
# the two disagree by 0.02 to 0.05 in both directions, which is just two fits
# differing, so the lift is faded out before it reaches the bust.
SHOULDER = 0.098
SHOULDER_BAND = (16.8, 18.2)     # nothing below the first, all of it above the second
LIFTED = ('maid_dress', 'maid_metal', 'maid_lace')

# Replaced by them, along with the hosiery the build was wearing.
SHED = ['maid_dress', 'maid_metal', 'maid_lace', 'maid_garter_metal',
        'maid_underwear', 'stockings', 'stocking_metal', 'shoes', 'shoe_metal']

ANKLE = 3.0          # a shell that ends below this is a shoe, and hers are better
# At the jaw, not at the collarbone. Her face01 starts at y 19.32, and cutting at
# 17.5 called 20 shells headdress that are nothing of the sort: the collar, the bow,
# the bell and the shoulder frills, 1106 triangles of neckline thrown away. Cut at the
# jaw it takes the 25 shells that are actually on the head -- 966 triangles, which is
# exactly what `maidbuild` places as `maid_headdress`.
CROWN = 19.3


def shells(tris):
    """Connected pieces of a material."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for t in tris:
        a = find(int(t[0]))
        for b in t[1:]:
            parent[find(int(b))] = a
    out = defaultdict(list)
    for k, t in enumerate(tris):
        out[find(int(t[0]))].append(k)
    return [tris[v] for v in out.values()]


def shoulders(path, pieces=LIFTED, rise=SHOULDER, band=SHOULDER_BAND):
    """Lift the top of the dress, and whatever is pinned to it."""
    m = PMX(path)
    sl = holefill.slices(m)
    lo, hi = band
    done = 0
    for name in pieces:
        if name not in sl:
            continue
        for i in np.unique(sl[name]):
            v = list(m.vertices[int(i)])
            w = np.clip((v[0][1] - lo) / (hi - lo), 0.0, 1.0)
            w = w * w * (3 - 2 * w)
            if w < 1e-3:
                continue
            v[0] = [v[0][0], v[0][1] + rise * float(w), v[0][2]]
            m.vertices[int(i)] = tuple(v)
            done += 1
    write_pmx(path, m)
    print(f'  lifted {done} vertices of the neckline and shoulders by up to {rise}')


def shorten(path, shoe=SHOES, sock='maid_stockings', clear=TOE):
    """Take the shoe back to the length of the foot inside it."""
    m = PMX(path)
    sl = holefill.slices(m)
    if shoe not in sl or sock not in sl:
        return
    V = np.array([v[0] for v in m.vertices])
    ankle = {b['name']: np.array(b['pos']) for b in m.bones}
    for side in ('左', '右'):
        key = side + '足首'
        if key not in ankle:
            continue
        sgn = 1 if side == '左' else -1
        ids = np.array([i for i in np.unique(sl[shoe]) if V[i][0] * sgn > 0], int)
        foot = np.array([i for i in np.unique(sl[sock])
                         if V[i][0] * sgn > 0 and V[i][1] < 2.4], int)
        if not len(ids) or not len(foot):
            continue
        z0 = float(ankle[key][2])
        toe = float(V[ids][:, 2].min())
        want = float(V[foot][:, 2].min()) - clear
        if toe >= want or abs(toe - z0) < 1e-6:
            continue
        k = (want - z0) / (toe - z0)
        for i in ids:
            v = list(m.vertices[int(i)])
            v[0] = [v[0][0], v[0][1], z0 + (v[0][2] - z0) * k]
            m.vertices[int(i)] = tuple(v)
        print(f'  {side}{shoe}: shortened by {k:.3f} about the ankle, toe '
              f'{toe:.3f} -> {want:.3f}')
    write_pmx(path, m)


def stand(path, lift=LIFT):
    """Raise the whole model, so it stands in the shoes it is about to be given."""
    m = PMX(path)
    for i, v in enumerate(m.vertices):
        w = list(v)
        w[0] = [v[0][0], v[0][1] + lift, v[0][2]]
        m.vertices[i] = tuple(w)
    for b in m.bones:
        b['pos'] = [b['pos'][0], b['pos'][1] + lift, b['pos'][2]]
    for r in m.rigids:
        r['pos'] = [r['pos'][0], r['pos'][1] + lift, r['pos'][2]]
    for j in m.joints:
        j['pos'] = [j['pos'][0], j['pos'][1] + lift, j['pos'][2]]
    write_pmx(path, m)
    print(f'  raised the build by {lift:.6f} to stand in her shoes')


def main():
    stand(BUILD)
    d, m = PMX(FITTED), PMX(BUILD)
    dsl = holefill.slices(d)
    DV = np.array([v[0] for v in d.vertices])
    dn = [b['name'] for b in d.bones]
    on = {b['name']: i for i, b in enumerate(m.bones)}
    parent = {b['name']: b.get('parent', -1) for b in d.bones}

    lut = []
    for b in d.bones:
        n = b['name'][4:] if b['name'].startswith('Cos:') else b['name']
        while n and n not in on:
            p = parent.get('Cos:' + n, parent.get(n, -1))
            n = dn[p][4:] if 0 <= p < len(dn) and dn[p].startswith('Cos:') else (
                dn[p] if 0 <= p < len(dn) else '')
        lut.append(on.get(n, 0))
    print(f'  {sum(1 for b in d.bones if (b["name"][4:] if b["name"].startswith("Cos:") else b["name"]) in on)}'
          f' of {len(d.bones)} of her bones resolve by name')

    base = os.path.dirname(FITTED)
    os.makedirs(TEXDIR, exist_ok=True)
    reg = {}

    def texture(i):
        if i < 0:
            return -1
        if i not in reg:
            fp = os.path.join(base, d.textures[i].replace('\\', '/'))
            dst = 'maid_' + os.path.basename(fp)
            if os.path.exists(fp):
                shutil.copyfile(fp, os.path.join(TEXDIR, dst))
            rel = 'textures\\' + dst
            if rel not in m.textures:
                m.textures.append(rel)
            reg[i] = m.textures.index(rel)
        return reg[i]

    # Only what is there. `dropmaterial` exits the process on a name it cannot find
    # rather than raising, so asking it for one is not a question you can catch.
    have = set(x['name'] for x in PMX(BUILD).materials)
    for name in [n for n in SHED if n in have]:
        dropmaterial.main(BUILD, name)
    m = PMX(BUILD)
    on = {b['name']: i for i, b in enumerate(m.bones)}
    reg = {}
    m_tex = m.textures

    keep = {}
    for src, want, cut in TAKE:
        if src not in dsl:
            continue
        tris = dsl[src]
        if cut == 'low':
            # Her shoe buckles, two shells of 28 at the ankle. They are hardware for
            # a shoe that is not being worn: `maidbuild` leaves them out too, which is
            # the whole of the difference between its 534 triangles of metal and this
            # material's 590.
            parts = shells(tris)
            low = [p for p in parts if DV[np.unique(p)][:, 1].max() < ANKLE]
            if low:
                tris = np.concatenate([p for p in parts
                                       if DV[np.unique(p)][:, 1].max() >= ANKLE])
                print(f'  {src}: left {len(low)} shells of shoe buckle '
                      f'({sum(len(p) for p in low)} tris) behind')
        elif cut:
            parts = shells(tris)
            span = [(DV[np.unique(p)][:, 1].min(), DV[np.unique(p)][:, 1].max(), p)
                    for p in parts]
            low = [p for lo, hi, p in span if hi < ANKLE]
            top = [p for lo, hi, p in span if lo > CROWN]
            mid = [p for lo, hi, p in span if hi >= ANKLE and lo <= CROWN]
            tris = np.concatenate(mid)
            if low:
                keep[SHOES] = (src, np.concatenate(low))
            print(f'  {src}: {len(parts)} shells -- {len(mid)} of dress, '
                  f'{len(low)} of shoe ({sum(len(p) for p in low)} tris), '
                  f'{len(top)} of headdress left behind')
        keep[want] = (src, tris)

    used = np.unique(np.concatenate([t for _, t in keep.values()]))
    fresh = {}
    for i in used:
        v = d.vertices[int(i)]
        w = list(v)
        w[0] = [float(x) for x in v[0]]
        w[5] = [lut[b] if 0 <= b < len(lut) else 0 for b in v[5]]
        m.vertices.append(tuple(w))
        fresh[int(i)] = len(m.vertices) - 1

    F = np.array(m.faces).reshape(-1, 3).tolist()
    for want, (src, tris) in keep.items():
        mat = dict(next(x for x in d.materials if x['name'] == src))
        mat['name'] = mat['name_en'] = want
        mat['tex'] = texture(mat['tex'])
        mat['sph'] = texture(mat['sph'])
        if not mat['toon_shared']:
            mat['toon'] = texture(mat['toon'])
        mat['fcount'] = len(tris) * 3
        m.materials.append(mat)
        F += [[fresh[int(x)] for x in t] for t in tris]
        print(f'  {len(tris):5d} triangles arrived as `{want}`')
    m.faces = np.array(F, int).ravel().tolist()
    write_pmx(BUILD, m)

    shoulders(BUILD)
    shorten(BUILD)
    fixrig.main([BUILD])
    shutil.copyfile(BUILD, PLACED)
    m = PMX(BUILD)
    print(f'{BUILD}: {len(m.vertices)} verts, {len(m.faces) // 3} triangles, '
          f'{len(m.bones)} bones, {len(m.morphs)} morphs')
    print(f'  materials: {", ".join(x["name"] for x in m.materials)}')


if __name__ == '__main__':
    main()
