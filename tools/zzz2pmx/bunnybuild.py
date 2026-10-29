# -*- coding: utf-8 -*-
"""Put 芭丝特's bunny outfit on the hand-built model.

`graft.py`'s method, with a second PMX for a donor instead of a ZZZ buffer dump.
Our body is not touched: the same skin, the same rig, the same 111 morphs, the same
physics it was hand-tuned with. Only the cloth arrives.

Placement reads the two BODIES (`bodyfield.py`), never the two garments. A uniform
ICP puts hers on ours, and the low-frequency difference left over is carried across,
so each piece keeps the standoff it had from her skin and lands that far off ours.
Where ours is the fuller figure the piece still ends up inside it, and
`surface.push_out` resolves that against the surface -- blurred first, because
lifting a buried vertex while its neighbour stays put is what tears a seam, and as a
single assembly for the pieces that are hardware rather than cloth.

Two earlier versions are worth not repeating. Retargeting the outfit bone by bone
looks right at rest and tears: 左腕 sits exactly on 左肩 so its segment is zero long
and its scale falls back to 1.0 while 左腕捩1 beside it takes its own ratio, and the
upper arm came apart between them -- 552 edges over 0.5 against her 138, and the
ankle did the same under the shoes. Taking her whole body instead removed the seam
and gave her figure rather than ours.

The accessories are the part a garment graft has no answer for. Her skirt flaps,
shoulder straps, bows and ears hang on 94 bones we have none of, every one carrying a
rigid body, and resolved up the parent chain they collapse onto `下半身` as one rigid
shell. `chaingraft.py` brings the chains over first -- bones, bodies and the joints
between -- through the same field, so a chain stays inside its own mesh.

    python3 tools/zzz2pmx/bunnybuild.py

Reads `public/models/reze/reze.pmx` and never writes it. Output is
`public/models/reze-bunny/reze-bunny.pmx`, beside a `textures/` of its own.
"""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pmx import PMX
from pmxw import write_pmx
from surface import Mesh, push_out
import bodyfield
import chaingraft
import dropmaterial
import graftmaterial

BASE = 'public/models/reze/reze.pmx'
DONOR = 'public/models/深空之眼—芭丝特兔女郎/芭丝特1.0.pmx'
OUT = 'public/models/reze-bunny/reze-bunny.pmx'
TEXDIR = 'public/models/reze-bunny/textures'

TEXTURES = {'芭丝特衣服.png': 'bunny_cloth.png', '芭丝特衣服1.png': 'bunny_cloth1.png'}

# The two bodies the field is read from. Hers is in two materials because below
# y 12.31 her legs *are* the hosiery; ours is too, because `dressasskin.py` gave
# `socks` the body's UVs and everything under y 8.6 is in it.
HER_BODY = ['皮肤', '袜子']
OUR_BODY = ['skin', 'socks']

# The pieces clear both, while both are still here. `socks` is dropped afterwards --
# the hosiery replaces it outright, the way she is built, since her 皮肤 stops at
# y 12.31 because below that her legs *are* the hosiery, and it is opaque. Kept
# underneath, our leg and her hosiery are the same surface to within 0.0025 over a
# twentieth of the stocking and they shred each other. It has to stay for the clear
# though: `push_out` has no reach gate, so with only `skin` to find, a vertex at the
# ankle takes the nearest skin there is -- the thigh -- and travels 8 units to it.
CLEAR_AGAINST = ['skin', 'socks']

# The outfit. `clear` is how far the piece has to stand off our skin -- a collar sits
# on the neck, a leotard over the hip. `rigid` resolves a piece as one assembly under
# a single scale: a buckle is not cloth, and moving the buried vertices of a card cuff
# flattens it.
OUTFIT = [
    # her material, ours,             texture,            clear, rigid
    ('上衣',   'bunny_leotard',    'bunny_cloth.png',  0.045, False),
    ('胖次',   'bunny_panties',    'bunny_cloth.png',  0.015, False),
    ('袜子',   'bunny_stockings',  'bunny_cloth.png',  0.020, False),
    ('衣饰',   'bunny_trim',       'bunny_cloth1.png', 0.025, False),
    ('衣饰铁',  'bunny_hardware',   'bunny_cloth.png',  0.025, True),
    ('背带',   'bunny_straps',     'bunny_cloth.png',  0.025, False),
    ('鞋子',   'bunny_heels',      'bunny_cloth.png',  0.010, True),   # by FOOT
    ('耳朵',   'bunny_ears',       'bunny_cloth1.png', None,  True),
    ('头饰',   'bunny_headband',   'bunny_cloth.png',  None,  True),
]

CHAIN_MATERIALS = [m for m, _, _, _, _ in OUTFIT]

# The shoe rides these instead of the body field. It is hardware built around a foot
# held at an angle, and a body field smooths that angle away.
FOOT = ['足首', '足首D', 'つま先', '足先EX']
BY_FOOT = {'bunny_heels'}

# Ours, replaced by hers. Our body, legs, underwear, head and hair all stay.
DROP = ['Rubber', 'Leather', 'shirt', 'ribbon', 'shorts', 'bozi', 'socks']


def textures(path):
    """Copy her sheets in and register them, so `--tex` can name one."""
    os.makedirs(TEXDIR, exist_ok=True)
    base = os.path.dirname(DONOR)
    m = PMX(path)
    for src, dst in TEXTURES.items():
        shutil.copyfile(os.path.join(base, src), os.path.join(TEXDIR, dst))
        rel = 'textures\\' + dst
        if rel not in m.textures:
            m.textures.append(rel)
    write_pmx(path, m)
    return {dst: 'textures\\' + dst for dst in TEXTURES.values()}


def clear_body(path):
    """Lift each piece off our skin, once everything is in place."""
    m = PMX(path)
    F = np.array(m.faces).reshape(-1, 3)
    mats, at = {}, 0
    for mat in m.materials:
        n = mat['fcount'] // 3
        mats[mat['name']] = F[at:at + n]
        at += n
    V = np.array([v[0] for v in m.vertices])
    body = Mesh(V, np.concatenate([mats[k] for k in CLEAR_AGAINST]))
    # A vertex on a chain is hanging cloth, not cloth lying on the body, and nothing
    # about our figure should push it. Left in, the hip flaps were driven off the
    # back: standoff median 0.210 and max 0.574 against her own 0.148 and 0.252.
    own = set(b['name'] for b in PMX(BASE).bones)
    names = [b['name'] for b in m.bones]
    hanging = set()
    for i, v in enumerate(m.vertices):
        for bi, w in zip(v[5], v[6]):
            if w > 0.2 and 0 <= bi < len(names) and names[bi] not in own:
                hanging.add(i)
    print(f'  {len(hanging)} vertices hang on a chain and are left alone')
    for _, name, _, clear, rigid in OUTFIT:
        if clear is None or name not in mats:
            continue
        idx = np.array([i for i in np.unique(mats[name]) if int(i) not in hanging])
        if not len(idx):
            continue
        moved, lifted = push_out(body, V[idx], offset=clear, rigid=rigid, rounds=20)
        print(f'  {name:17s} lifted {int(lifted.sum()):5d} of {len(idx):5d} vertices '
              f'clear by {clear}, worst move {np.linalg.norm(moved - V[idx], axis=1).max():.3f}')
        V[idx] = moved
    for i, v in enumerate(m.vertices):
        if not np.allclose(v[0], V[i]):
            w = list(v)
            w[0] = [float(x) for x in V[i]]
            m.vertices[i] = tuple(w)
    write_pmx(path, m)


def trim_legs(path, cut=12.0, half=3.0):
    """Take our skin out from under the hosiery.

    She has none there: her 皮肤 stops at y 12.31 because below that her legs are the
    hosiery. Ours runs all the way down, and two skin-tight surfaces a hundredth
    apart do not need to interpenetrate to fight -- every vertex was outside and the
    thigh still came out shredded, because the faces between them cross. Cut to the
    legs by half-width, or it takes the hands with it: they hang at y 12.3 in an
    A-pose, which is below the cut and nowhere near the legs.
    """
    m = PMX(path)
    F = np.array(m.faces).reshape(-1, 3)
    V = np.array([v[0] for v in m.vertices])
    at, out, gone = 0, [], 0
    for mat in m.materials:
        n = mat['fcount'] // 3
        tris = F[at:at + n]
        if mat['name'] == 'skin':
            P = V[tris]
            drop = (P[:, :, 1] < cut).all(1) & (np.abs(P[:, :, 0]) < half).all(1)
            gone = int(drop.sum())
            tris = tris[~drop]
            mat['fcount'] = len(tris) * 3
        out.append(tris)
        at += n
    m.faces = np.concatenate(out).ravel().tolist()
    write_pmx(path, m)
    print(f'  took {gone} skin triangles out from under the hosiery '
          f'(below y {cut}, within |x| {half})')


def collide(path):
    """Let the chains meet our body.

    Hers are set to collide with their own group and nothing else -- her skirt panels
    pass through her own legs, which she gets away with and a fuller figure does not.
    Both sides have to agree for a pair to touch, so the chains are opened to our body
    group and our body bodies to theirs. Our hair keeps the collision setup it was
    tuned with.
    """
    m = PMX(path)
    own = set(b['name'] for b in PMX(BASE).bones)
    names = [b['name'] for b in m.bones]
    chain, body = set(), set()
    for r in m.rigids:
        n = names[r['bone']] if 0 <= r['bone'] < len(names) else ''
        if n and n not in own:
            chain.add(r['group'])
        elif r['mode'] == 0 and n in own:
            body.add(r['group'])
    for r in m.rigids:
        n = names[r['bone']] if 0 <= r['bone'] < len(names) else ''
        if n and n not in own:
            for g in body:
                r['nocollide'] &= ~(1 << g)
        elif r['mode'] == 0 and n in own:
            for g in chain:
                r['nocollide'] &= ~(1 << g)
    write_pmx(path, m)
    print(f'  chain groups {sorted(chain)} now meet body groups {sorted(body)}')


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    shutil.copyfile(BASE, OUT)
    src = os.path.join(os.path.dirname(BASE), 'textures')
    if os.path.isdir(src) and not os.path.isdir(TEXDIR):
        shutil.copytree(src, TEXDIR)

    print('--- field')
    d0, t0 = PMX(DONOR), PMX(BASE)
    field = bodyfield.Field(d0, t0, HER_BODY, OUR_BODY,
                            scale=chaingraft.body_scale(d0, t0))

    snug = bodyfield.Snug(field, d0, HER_BODY, t0, CLEAR_AGAINST)

    print('--- chains')
    chaingraft.main(DONOR, OUT, OUT, CHAIN_MATERIALS, frame='兔女郎', place=snug)
    tex = textures(OUT)

    print('--- foot')
    foot = bodyfield.BoneFrame(d0, t0, FOOT)

    print('--- outfit')
    for mat, name, sheet, _, _ in OUTFIT:
        graftmaterial.main(DONOR, mat, name, tex[sheet], target=OUT, out=OUT,
                           place=foot if name in BY_FOOT else snug)

    print('--- clear')
    clear_body(OUT)

    print('--- trim')
    trim_legs(OUT)

    print('--- collide')
    collide(OUT)

    print('--- ours out')
    for name in DROP:
        dropmaterial.main(OUT, name)

    m = PMX(OUT)
    print(f'{OUT}: {len(m.faces)//3} triangles, {len(m.bones)} bones, '
          f'{len(m.morphs)} morphs, {len(m.rigids)} rigid bodies, '
          f'{len(m.joints)} joints')
    print(f'  materials: {", ".join(x["name"] for x in m.materials)}')


if __name__ == '__main__':
    main()
