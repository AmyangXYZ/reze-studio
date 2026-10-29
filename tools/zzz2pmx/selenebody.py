# -*- coding: utf-8 -*-
"""A bare 塞勒涅, out of her own skin and the beach version's.

Her `皮肤` is very nearly a whole bare body already -- arms to the wrist, legs to the
ankle, torso and bust -- and what it is missing is what her costume covers: the belly
under the corset, the hands inside the gloves, the feet inside the boots, and a band
under the bust. 海滩漫步 has every one of those, and is missing the bust and the hip
instead, which 塞勒涅 has. The two are complements.

So nothing is baked and nothing is repainted. Each part arrives as its own material
carrying its own sheet and its own UVs, exactly as its author drew them, and the only
new geometry is the ring of triangles sewing one to the other.

An earlier version of this file measured the two bodies against each other after a
rigid fit and concluded 塞勒涅 had 24% of her arms and 56% of her calves. She has all
of both. The models are posed differently, so limbs that did not overlap read as
missing, and every decision taken from those numbers was aimed at the wrong problem.
Coverage is asked of the material's own rim now, which cannot be fooled by a pose.

    python3 tools/zzz2pmx/selenebody.py

Reads nothing it writes. Output is `public/models/reze-selene/selene-bare.pmx`.
"""
import os
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dropmaterial
import bodyfield
import holefill
from surface import Mesh
from pmx import PMX
from pmxw import write_pmx

BASE = 'public/models/塞勒涅/塞勒涅.pmx'
FILL = ('海滩漫步', 'public/models/【深空之眼】幽月·塞勒涅-海滩漫步/海滩漫步.pmx',
        'skin', 'textures/cloth01.png')
OUT = 'public/models/reze-selene/selene-bare.pmx'
TEXDIR = 'public/models/reze-selene/textures'

SKIN = '皮肤'
NAME = 'skin'            # what the studio's material classifier looks for
PATCH = 'skin_fill'      # and this still matches it

EPS = 0.10               # closer to her surface than this and she already has it

# What each hole is, because nothing local says. Every hole boundary is a ring that
# winds once about its own centre, whether a disc is missing out of a surface or a
# limb is missing past it, and the rim normals are incoherent on a curved body either
# way -- measured over these twelve, coherence runs 0.06 to 0.48 with no gap in it.
# The difference is in what the donor has on the other side, not in the rim, so it is
# stated rather than inferred. `cap` takes what lies inside the rim seen down its own
# axis; `past` takes everything beyond it, which is how a foot or a hand comes back.
HOLES = [dict(at=(0.00, 15.20), r=1.6, kind='cap', what='belly, under the corset'),
         dict(at=(0.00, 16.09), r=1.4, kind='cap', what='the band under the bust'),
         dict(at=(0.92, 0.83), r=3.5, kind='past', what='right foot, inside the boot'),
         dict(at=(-0.92, 0.83), r=3.5, kind='past', what='left foot, inside the boot'),
         dict(at=(6.43, 14.19), r=3.0, kind='past', what='right hand, in the glove'),
         dict(at=(-6.43, 14.19), r=3.0, kind='past', what='left hand, in the glove')]
MATCH = 0.45             # how near a rim has to be to one of those to be it
MIN_LOOP = 5
# The head sits in the neck, so the neck is not a hole. Nor is the crown: her scalp
# ends where her hair begins, which is how the model is built.
KEEP_OPEN = 18.6


def undress(path, keep=SKIN):
    """Everything except her skin."""
    m = PMX(path)
    for name in [x['name'] for x in m.materials if x['name'] != keep]:
        dropmaterial.main(path, name)
    m = PMX(path)
    m.materials[0]['name'] = m.materials[0]['name_en'] = NAME
    write_pmx(path, m)
    print(f'  kept {keep} alone, {m.materials[0]["fcount"] // 3} triangles, '
          f'renamed `{NAME}`')


def holes(m, mat, floor=KEEP_OPEN):
    out = []
    V = np.array([v[0] for v in m.vertices])
    for l in holefill.rims(m, [mat]):
        if len(l) < MIN_LOOP:
            continue
        if V[l][:, 1].min() > floor:
            print(f'    {len(l):4d} edges at y {V[l][:, 1].mean():6.2f}  left open: '
                  f'the head goes there')
            continue
        out.append(l)
    return out


def frame(V, loop):
    """A hole's centre, the axis it faces, and how wide it is."""
    P = V[loop]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[2], float(np.mean(np.linalg.norm(P - c, axis=1)))


def named(V, loop):
    """Which of the holes we know about this one is."""
    c = V[loop].mean(0)
    for h in HOLES:
        if abs(c[0] - h['at'][0]) < MATCH and abs(c[1] - h['at'][1]) < MATCH:
            return h
    return None


def wanted(V, loop, N, DV, h, HV):
    """Which of the donor's vertices belong in this hole.

    Pointed away from the body, by asking the body. A rim that goes around a limb has
    normals that cancel, so which way it faces cannot be read off them -- taken from
    them the foot rims pointed up the shin and fetched 19 triangles of leg. The empty
    side is the side the host has less of.
    """
    c, ax, r = frame(V, loop)
    reach = r * h['r']
    if h['kind'] == 'past':
        near = HV[np.linalg.norm(HV - c, axis=1) < reach]
        side = (near - c) @ ax
        if (side > 0.15).sum() > (side < -0.15).sum():
            ax = -ax
        q = DV - c
        return ((q @ ax) > -0.10) & (np.linalg.norm(q, axis=1) < reach)
    n = N[loop].sum(0)
    if np.dot(ax, n) < 0:
        ax = -ax
    q = DV - c
    along = q @ ax
    across = np.linalg.norm(q - along[:, None] * ax, axis=1)
    return (np.abs(along) < r * 0.9) & (across < reach)


def fill(path):
    """Bring across the parts her costume took, hole by hole."""
    tag, src, mat, tex = FILL
    m, d = PMX(path), PMX(src)
    V = np.array([v[0] for v in m.vertices])
    N = np.array([v[1] for v in m.vertices])
    rims = holes(m, NAME)
    print(f'  {len(rims)} holes in her skin, {sum(len(l) for l in rims)} rim edges')
    T = holefill.tris_of(d, [mat])
    # Warped onto her, not merely fitted to her. One rigid transform leaves 0.141
    # between these two -- they are posed differently, so a fit that lands the torso
    # cannot also land the arms -- and 0.141 is larger than the distance at which
    # "does she have this already" has to be decided. Carried through the body field
    # instead, which is a uniform fit plus the low-frequency difference between the
    # two figures read bone by bone, they agree an order of magnitude closer.
    # Scale given, not solved. Both models stand 21.72 high and share a skeleton, so
    # there is no scale between them -- left free the ICP found 0.8997 and shrank the
    # donor out of reach of the very holes it was fetched for, 30 triangles of foot.
    field = bodyfield.Field(d, m, [mat], [NAME], scale=1.0)
    ref = Mesh(field(np.array([v[0] for v in d.vertices])), T)
    check = cKDTree(V[np.unique(holefill.tris_of(m, [NAME]))]).query(
        ref.V[np.unique(T)])[0]
    print(f'  {tag} carried onto her: agreeing to {np.median(check):.3f} '
          f'(a rigid fit gives 0.141)')
    skin = cKDTree(V[np.unique(holefill.tris_of(m, [NAME]))])
    gap, _ = skin.query(ref.V)

    F = np.array(m.faces).reshape(-1, 3).tolist()
    kinds, total, sewn = [], 0, 0
    for l in rims:
        Q = V[l]
        h = named(V, l)
        if h is None:
            print(f'    {len(l):4d} edges at ({Q[:, 0].mean():+5.2f},'
                  f'{Q[:, 1].mean():5.2f}) left as it is')
            continue
        pick = wanted(V, l, N, ref.V, h, V[np.unique(holefill.tris_of(m, [NAME]))]) & (gap > EPS)
        sel = T[pick[T].all(1)]
        kind = h['kind']
        if not len(sel):
            print(f'    {h["what"]}: she has nothing there either')
            continue
        took, n = holefill.transplant(m, d, ref, {mat: sel}, [NAME], limit=0.9)
        for name, tris in took.items():
            F += tris.tolist()
            total += len(tris)
        sewn += n
        print(f'    {h["what"]:32s} {kind:4s} {len(sel):4d} triangles, sewn with {n}')
        kinds.append(kind)
    if not total:
        return
    print(f'  {total} triangles arrived, {sewn} of them sewing')

    base = os.path.dirname(src)
    reg = {}

    def texture(i):
        if i < 0:
            return -1
        if i not in reg:
            fp = os.path.join(base, d.textures[i].replace('\\', '/'))
            dst = f'{tag}_{os.path.basename(fp)}'
            os.makedirs(TEXDIR, exist_ok=True)
            if os.path.exists(fp):
                shutil.copyfile(fp, os.path.join(TEXDIR, dst))
            rel = 'textures\\' + dst
            if rel not in m.textures:
                m.textures.append(rel)
            reg[i] = m.textures.index(rel)
        return reg[i]

    src_mat = next(x for x in d.materials if x['name'] == mat)
    new = dict(src_mat)
    new['name'] = new['name_en'] = PATCH
    new['tex'] = texture(src_mat['tex'])
    new['sph'] = texture(src_mat['sph'])
    new['toon'] = src_mat['toon'] if src_mat['toon_shared'] else texture(src_mat['toon'])
    new['fcount'] = total * 3
    m.materials.append(new)
    m.faces = np.array(F, int).ravel().tolist()
    write_pmx(path, m)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    work = OUT + '.building'
    shutil.copyfile(BASE, work)
    src = os.path.dirname(BASE)
    os.makedirs(TEXDIR, exist_ok=True)
    m = PMX(BASE)
    for rel in m.textures:
        fp = os.path.join(src, rel.replace('\\', '/'))
        if os.path.exists(fp):
            shutil.copyfile(fp, os.path.join(TEXDIR, os.path.basename(fp)))
    n = PMX(work)
    n.textures = ['textures\\' + os.path.basename(t.replace('\\', '/'))
                  for t in n.textures]
    write_pmx(work, n)

    print('--- undress')
    undress(work)

    print('--- fill')
    fill(work)

    os.replace(work, OUT)
    m = PMX(OUT)
    print(f'{OUT}: {len(m.vertices)} verts, {len(m.faces) // 3} triangles, '
          f'{len(m.bones)} bones, {len(m.morphs)} morphs')
    print(f'  materials: {", ".join(x["name"] for x in m.materials)}')


if __name__ == '__main__':
    main()
