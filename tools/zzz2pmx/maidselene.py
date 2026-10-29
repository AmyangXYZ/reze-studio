# -*- coding: utf-8 -*-
"""家政小帮手's maid outfit on a bare 塞勒涅, with our head.

`maidbuild.py` against a different host, in two steps, because the body that fits the
dress best and the body that looks best are not the same body.

**Fitted on the mannequin.** `base_selene` is one closed shell with no boundary
anywhere and a complete breast. That matters because `fit_bust` and `clear_body`
conform the outfit to whatever surface they find underneath -- and undressed,
海滩漫步 has two holes where her bikini top was, so the bodice was being fitted to a
hole. The mannequin has no texture worth keeping and does not need one: nothing of it
survives this.

**Worn by 塞勒涅 herself.** She keeps her own skin and her own sheet, untouched. Both
models stand 21.724 high in the same frame, so the fitted outfit moves across whole,
with no second fit and nothing re-solved -- only her weights are rewritten, from her
own 463-bone rig onto the 132 the outfit was built with, by name. Her skin is very
nearly a complete bare body already; what it lacks is what her costume covered, and
the dress covers the same places.

    python3 tools/zzz2pmx/maidselene.py

Her legs arrive as materials, whole. Cutting the hosiery back at the shoe's rim and
trimming her skin from under it were both worth doing eventually and neither is worth
doing yet: every cut leaves an edge to get wrong, and the shoes and the hosiery here
were drawn for one another on this exact foot, so there is nothing between them to
resolve.

Reads nothing it writes. Output is `public/models/reze-maid-selene/reze-maid.pmx`.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dropmaterial
import fixrig
import holefill
import maidbuild
from surface import Mesh, push_out
from pmx import PMX
from pmxw import write_pmx

HERE = os.path.dirname(os.path.abspath(__file__))
MANNEQUIN = 'public/models/base_selene/Selene_base_fitted.pmx'
WEARER = 'public/models/塞勒涅/塞勒涅.pmx'
# Her skin stops at the wrist -- her hands are her gloves, and the gloves are costume.
# The maid's cuffs do not reach, so the hands come from the beach version, who has
# bare ones. Nothing is fitted for it: the two agree along the forearm to 0.021 and
# share all 42 hand bones to 0.001, so the hands are simply where they already are.
HANDS = ('海滩漫步', 'public/models/【深空之眼】幽月·塞勒涅-海滩漫步/海滩漫步.pmx',
         'skin', 'textures/cloth01.png')
WRIST = 5.5              # nothing this far out on her is anything but hand
APART = 0.12             # and hers has to be missing there, not merely different

# The bodice and everything on it, given the body's own weights over the bust.
# 塞勒涅's breasts are driven by physics, and the maid was drawn on a body whose were
# not: her bow leans on 上半身 and 上半身2 for 78% and on 胸 for 14%, where the skin
# under it is 66% 胸. So the chest travels nearly five times as far as the bow does
# and comes straight through it. Transferred off the skin, the two move together.
BODICE = ('maid_dress', 'maid_metal', 'maid_lace')
# The upper bust only. Reaching down to the band under it gave that band 28% of the
# breast's motion, and a structured band that moves with the breast is a band that
# shakes: the maid's own dress holds it on 上半身2. What needs the breast is the bow
# above it.
BUST = dict(ylo=16.15, yhi=18.1, zmax=-0.85)
GATHER = 0.75            # how much skin each bodice vertex is asked about

# maidbuild's own conform settings are left alone. Widening them to chase the band
# under the bust -- reach 0.45 to 0.90, the band down from 15.00 to 14.25, the
# standoff 0.030 to 0.055 -- pulled 872 vertices by up to 0.522 and tore the bodice
# into ribbons. The band is loose because the dress was drawn for a much smaller bust
# and a conform can only move what is already nearly right.
HEAD = 'public/models/reze/reze.pmx'
DIR = 'public/models/reze-maid-selene'
OUT = f'{DIR}/reze-maid.pmx'
DRESSED = f'{DIR}/mannequin.pmx'     # the fit, before she is swapped in
WORN = f'{DIR}/host.pmx'             # her, undressed

BODY = 'Body_Mannequin'
SKIN = '皮肤'
RENAME = {SKIN: 'skin', '脸': 'face01'}
HERS = ['skin', 'face01']

# Shoes from 沉醉旖旎, not from 塞勒涅 and not the maid's. 塞勒涅 stands on a different
# foot from the rest of the series -- ankle to toe 1.917 at 41.1 degrees below
# horizontal, against 1.531 at 55.6 for both 海滩漫步 and 沉醉旖旎 -- and the rig here
# and the hosiery are both 海滩漫步's, so her own shoe arrives pitched forward off a
# foot it was never drawn for and the sock cannot reach it. 沉醉旖旎's is drawn for
# exactly this foot. It still fits HER, because what a shoe meets is the ankle, and
# the ankle is the same on all three: it passes her skin's foot-hole rim at 0.011.
# Her hosiery comes too, not just the shoes. It is drawn for this same foot and this
# same rig, so the two meet each other by construction -- which is the whole trouble
# with the maid's, conformed as it was to a mannequin. And a maid in black stockings
# is a maid.
LEGS = ('沉醉旖旎', 'public/models/【深空之眼】幽月·塞勒涅-沉醉旖旎_by_Chiyuri_'
        '6dbfe0ca0200e0a1f88bbcd4d159cd48/沉醉旖旎.pmx',
        [('cloth01', 'shoes', 'foot'), ('metal', 'shoe_metal', 'foot'),
         ('stocking', 'stockings', None), ('stocking_metal', 'stocking_metal', None)])
ANKLE = 2.3              # below this, and near the centre line, is shoe
ASIDE = 1.6

# The maid's own shoes are dropped for hers. They were fitted to the mannequin, whose
# foot is LONGER than the shoe is -- z -1.05..+1.21 against the shoe's -0.89..+1.09 --
# so the hosiery conformed to that foot was outside the shoe before anyone put it on,
# and 塞勒涅's foot then came through the lot. Hers are her own shoe on her own foot,
# and they stand the same height, y 0.00..1.85 either way.
# Dropped with them: the maid's underwear, which has nothing to do under a pair of
# her tights.
SHED = ('maid_shoes', 'maid_shoe_metal', 'maid_stockings', 'maid_underwear')

# She is rigged more finely than the mannequin -- 463 bones to its 132 -- and her skin
# leans on two the mannequin has not got. They are the second joint of each breast
# chain, and the first joint is right there.
ALIAS = {'左胸上2': '左胸', '右胸上2': '右胸'}
DROP_HEAD = ['face01', '齿', '口腔', '舌', '唇', 'eyebrow', 'eyelash', '目白',
             '瞳1', '瞳2']


def mannequin():
    """The fitting body: one closed shell, named where maidbuild looks for it."""
    shutil.copyfile(MANNEQUIN, DRESSED)
    m = PMX(DRESSED)
    for mat in m.materials:
        if mat['name'] == BODY:
            mat['name'] = mat['name_en'] = 'skin'
    write_pmx(DRESSED, m)
    print(f'  {DRESSED}: {", ".join(x["name"] for x in m.materials)}')


def undressed():
    """The body that is actually worn: her skin alone, with her own sheets."""
    shutil.copyfile(WEARER, WORN)
    src = os.path.dirname(WEARER)
    dst = os.path.join(DIR, 'textures')
    os.makedirs(dst, exist_ok=True)
    m = PMX(WORN)
    for rel in m.textures:
        fp = os.path.join(src, rel.replace('\\', '/'))
        if os.path.exists(fp):
            shutil.copyfile(fp, os.path.join(dst, os.path.basename(fp)))
    m.textures = ['textures\\' + os.path.basename(t.replace('\\', '/'))
                  for t in m.textures]
    write_pmx(WORN, m)
    for name in [x['name'] for x in PMX(WORN).materials
                 if x['name'] not in RENAME]:
        dropmaterial.main(WORN, name)
    m = PMX(WORN)
    for mat in m.materials:
        if mat['name'] in RENAME:
            mat['name'] = mat['name_en'] = RENAME[mat['name']]
    write_pmx(WORN, m)
    print(f'  {WORN}: {", ".join(x["name"] for x in m.materials)}')


def hands(path):
    """Give her hands. Hers are gloves, and the gloves went with the costume."""
    tag, src, mat, tex = HANDS
    m, d = PMX(path), PMX(src)
    V = np.array([v[0] for v in m.vertices])
    DV = np.array([v[0] for v in d.vertices])
    skin = cKDTree(V[np.unique(holefill.tris_of(m, ['skin']))])
    gap, _ = skin.query(DV)
    free = (gap > APART) & (np.abs(DV[:, 0]) > WRIST)
    T = holefill.tris_of(d, [mat])
    sel = T[free[T].all(1)]
    if not len(sel):
        return
    took, sewn = holefill.transplant(m, d, Mesh(DV, T), {mat: sel}, ['skin'],
                                     limit=1.0)
    base = os.path.dirname(src)
    fp = os.path.join(base, tex.replace('\\', '/'))
    dst = f'{tag}_{os.path.basename(fp)}'
    os.makedirs(os.path.join(DIR, 'textures'), exist_ok=True)
    if os.path.exists(fp):
        shutil.copyfile(fp, os.path.join(DIR, 'textures', dst))
    rel = 'textures\\' + dst
    if rel not in m.textures:
        m.textures.append(rel)
    F = np.array(m.faces).reshape(-1, 3).tolist()
    src_mat = next(x for x in d.materials if x['name'] == mat)
    new = dict(src_mat)
    new['name'] = new['name_en'] = 'skin_hands'
    new['tex'] = m.textures.index(rel)
    new['sph'] = -1
    new['fcount'] = sum(len(v) for v in took.values()) * 3
    m.materials.append(new)
    for v in took.values():
        F += v.tolist()
    m.faces = np.array(F, int).ravel().tolist()
    write_pmx(path, m)
    print(f'  {new["fcount"] // 3} triangles of {tag}\'s hands, {sewn} of them '
          f'sewing at the wrist')


def shoes(path):
    """Put 沉醉旖旎's legs on, where the maid's used to be."""
    tag, src, want = LEGS
    names = {a: b for a, b, _ in want}
    m, d = PMX(path), PMX(src)
    DV = np.array([v[0] for v in d.vertices])
    dsl = maidbuild.slices(d)
    take = {}
    for name, _, only in want:
        if name not in dsl:
            continue
        T = dsl[name]
        if only == 'foot':
            C = DV[T].mean(1)
            T = T[(C[:, 1] < ANKLE) & (np.abs(C[:, 0]) < ASIDE)]
        if len(T):
            take[name] = T
    if not take:
        return
    took, sewn = holefill.transplant(m, d, Mesh(DV, np.concatenate(list(take.values()))),
                                     take, ['skin'], limit=0.0)
    base = os.path.dirname(src)
    reg = {}

    def texture(i):
        if i < 0:
            return -1
        if i not in reg:
            fp = os.path.join(base, d.textures[i].replace('\\', '/'))
            dst = f'{tag}_{os.path.basename(fp)}'
            os.makedirs(os.path.join(DIR, 'textures'), exist_ok=True)
            if os.path.exists(fp):
                shutil.copyfile(fp, os.path.join(DIR, 'textures', dst))
            rel = 'textures\\' + dst
            if rel not in m.textures:
                m.textures.append(rel)
            reg[i] = m.textures.index(rel)
        return reg[i]

    F = np.array(m.faces).reshape(-1, 3).tolist()
    for name, tris in took.items():
        src_mat = next(x for x in d.materials if x['name'] == name)
        new = dict(src_mat)
        new['name'] = new['name_en'] = names[name]
        new['tex'] = texture(src_mat['tex'])
        new['sph'] = texture(src_mat['sph'])
        new['toon'] = src_mat['toon'] if src_mat['toon_shared'] else texture(src_mat['toon'])
        new['fcount'] = len(tris) * 3
        m.materials.append(new)
        F += tris.tolist()
        print(f'  {len(tris)} triangles of {tag}\'s {name} arrived as `{names[name]}`')
    m.faces = np.array(F, int).ravel().tolist()
    write_pmx(path, m)


def bodice(path, pieces=BODICE, band=BUST, radius=GATHER):
    """Let the bodice follow the breast as far as the skin around it does.

    Around it, not directly behind it. Asked for the single nearest point of skin the
    bow takes the sternum's weights, because the sternum is what is behind a bow that
    sits between the breasts -- and the sternum is the one part of a chest that does
    not move. It came back following the breast 23% against the skin's 66%, and still
    had the chest coming through it. Averaged over the skin within reach of it, it
    picks up both breasts either side, which is what is actually going to hit it.
    """
    m = PMX(path)
    sl = maidbuild.slices(m)
    V = np.array([v[0] for v in m.vertices])
    names = [b['name'] for b in m.bones]
    index = {b['name']: i for i, b in enumerate(m.bones)}
    own = set(x['name'] for x in PMX(WEARER).bones)
    # Anything hanging on a chain of the outfit's own is left alone: it is held up by
    # physics, not by the body, and giving it the body's weights would nail it down.
    chain = set()
    for i, v in enumerate(m.vertices):
        for bi, w in zip(v[5], v[6]):
            if w > 0.2 and 0 <= bi < len(names) and names[bi] not in own:
                chain.add(i)
    skin = np.array(sorted(set(int(i) for i in np.unique(sl['skin']))))
    keep = [i for i in skin if band['ylo'] < V[i][1] < band['yhi']
            and V[i][2] < band['zmax']]
    if not keep:
        return
    tree = cKDTree(V[keep])
    src = maidbuild._weights_of(m)
    for name in pieces:
        if name not in sl:
            continue
        ids = [int(i) for i in np.unique(sl[name]) if i not in chain
               and band['ylo'] < V[i][1] < band['yhi'] and V[i][2] < band['zmax']]
        if not ids:
            continue
        done, moved = 0, []
        for i in ids:
            near = tree.query_ball_point(V[i], radius)
            if not near:
                continue
            wd = {}
            tot = 0.0
            for k in near:
                j = keep[k]
                w = 1.0 / max(float(np.linalg.norm(V[j] - V[i])), 0.05) ** 2
                tot += w
                for n, x in src[j].items():
                    wd[n] = wd.get(n, 0.0) + w * x
            wd = {n: x / tot for n, x in wd.items()}
            was = dict(src[i])
            if maidbuild._set_weights(m, i, wd, index):
                done += 1
                moved.append(0.5 * sum(abs(was.get(n, 0) - wd.get(n, 0))
                                       for n in set(was) | set(wd)))
        print(f'  {name}: {done} of {len(ids)} vertices given the weights of the skin '
              f'within {radius}, moved median {np.median(moved) if moved else 0:.3f}')
    write_pmx(path, m)


def swap(dressed, worn, out, keep=HERS):
    """Take the mannequin out from under the outfit and put her in.

    A straight substitution, not a graft. The two share a skeleton bone for bone in
    the same order, so her vertices already refer to the right bones, and they stand
    in the same frame, so nothing moves. The outfit keeps the rig it was built with,
    including everything `maidbuild` did to the body's own colliders.
    """
    a, b = PMX(dressed), PMX(worn)
    an = {x['name']: i for i, x in enumerate(a.bones)}
    bn = [x['name'] for x in b.bones]
    parent = {x['name']: x.get('parent', -1) for x in b.bones}
    lut = []
    for x in b.bones:
        n = x['name']
        while n and n not in an:
            n = ALIAS.get(n) or (bn[parent.get(n, -1)]
                                 if 0 <= parent.get(n, -1) < len(bn) else '')
        lut.append(an.get(n, 0))
    unknown = sum(1 for i, x in enumerate(b.bones)
                  if x['name'] not in an and x['name'] in ALIAS)
    print(f'  her rig has {len(bn)} bones to the outfit\'s {len(an)}; '
          f'{sum(1 for x in b.bones if x["name"] in an)} match by name, '
          f'{unknown} resolve through an alias, the rest up their own parent')

    tex = {}
    for i, rel in enumerate(b.textures):
        if rel not in a.textures:
            a.textures.append(rel)
        tex[i] = a.textures.index(rel)

    base = len(a.vertices)
    for v in b.vertices:
        w = list(v)
        w[5] = [lut[i] if 0 <= i < len(lut) else 0 for i in v[5]]
        a.vertices.append(tuple(w))
    F = np.array(a.faces).reshape(-1, 3)
    G = np.array(b.faces).reshape(-1, 3) + base
    at, out_f, mats = 0, [], []
    for mat in a.materials:
        n = mat['fcount'] // 3
        if mat['name'].startswith('maid_') and mat['name'] not in SHED:
            out_f.append(F[at:at + n])
            mats.append(mat)
        at += n
    gone = len(a.materials) - len(mats)
    at = 0
    for mat in b.materials:
        n = mat['fcount'] // 3
        if mat['name'] in keep:
            out_f.append(G[at:at + n])
            mat['tex'] = tex.get(mat['tex'], -1)
            mat['sph'] = tex.get(mat['sph'], -1)
            if not mat['toon_shared']:
                mat['toon'] = tex.get(mat['toon'], -1)
            mats.append(mat)
        at += n
    a.materials = mats
    a.faces = np.concatenate(out_f).ravel().tolist()
    a.morphs = []       # the mannequin's; `headgraft` brings ours
    write_pmx(out, a)
    print(f'  dropped {gone} of the mannequin\'s materials, brought {len(keep)} of hers')
    print(f'  {", ".join(x["name"] for x in a.materials)}')


def main():
    os.makedirs(DIR, exist_ok=True)
    print('--- the fitting body')
    mannequin()
    print('--- her, undressed')
    undressed()

    maidbuild.BASE = DRESSED
    maidbuild.OUT = OUT
    maidbuild.FULL = f'{DIR}/reze-maid-fullbody.pmx'
    maidbuild.TEXDIR = f'{DIR}/textures'
    maidbuild.OUR_BODY = ['skin']
    maidbuild.CLEAR_AGAINST = ['skin']
    maidbuild.KEEP = ['skin']
    maidbuild.DROP = []
    maidbuild.REBUILD_BUST = False      # the mannequin has a breast already
    maidbuild.HAIR_SHELLS = None
    maidbuild.TRIM_LEGS = False         # her legs are trimmed after the swap

    maidbuild.main()

    print('--- she wears it')
    swap(OUT, WORN, OUT)
    # Conformed again, now that she is the one wearing it. The first pass ran against
    # the mannequin, and the mannequin is not her under the bust: it took its breast
    # from 塞勒涅 and its abdomen from 海滩漫步, so the band below the bust was fitted
    # to a chest nobody is standing in. Measured on her it stood 0.510 off, against
    # 0.356 for the maid's own dress on the maid.
    print('--- the bodice, on her')
    maidbuild.fit_bust(OUT)
    hands(OUT)
    shoes(OUT)
    bodice(OUT)
    # After the swap, not before. Both of these work on the body, and in this
    # pipeline the body the outfit was fitted against is not the body that ships --
    # run on the mannequin they are thrown away with it. That is exactly how the toe
    # came back: `stiffen_foot` is what stops the foot swinging with the knee, and it
    # had been applied to a mannequin nobody sees.
    maidbuild.stiffen_foot(OUT)     # on `skin` and the hosiery both, as it defaults

    print('--- head')
    subprocess.run([sys.executable, os.path.join(HERE, 'headgraft.py'),
                    '--host', OUT, '--head', HEAD, '--out', OUT,
                    '--drop', ','.join(DROP_HEAD)], check=True)
    # Last, because 沉醉旖旎's hosiery and shoes carry her fault with them: the toe
    # weighted to つま先, which is the IK end effector, while the foot around it is on
    # 足首D. The two then turn by different amounts and the toe twists.
    print('--- the deform chain')
    fixrig.main([OUT])

    m = PMX(OUT)
    print(f'{OUT}: {len(m.vertices)} verts, {len(m.faces) // 3} triangles, '
          f'{len(m.bones)} bones, {len(m.morphs)} morphs, {len(m.rigids)} rigid '
          f'bodies, {len(m.joints)} joints')
    print(f'  materials: {", ".join(x["name"] for x in m.materials)}')


if __name__ == '__main__':
    main()
