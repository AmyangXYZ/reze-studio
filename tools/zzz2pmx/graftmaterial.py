# -*- coding: utf-8 -*-
"""Re-implant a material from the donor body onto this one.

Reze's `cloth01` is a cut of 塞尔凯特2's: the cups were kept, the panties split off
into `cloth01.001`, and the straps, back band, armbands and garter were dropped --
which is why the skin behind them is painted for straps that are not there.

The bodies were re-proportioned since, so the piece is retargeted rather than
copied: each bone contributes the similarity carrying its own segment of the donor's
skeleton onto ours -- the rotation aligning the segment and the scale matching its
length -- and each vertex is placed by its own skin weights. A bone we do not have
resolves up the donor's parent chain to one we do, so the cups ride 上半身2 where the
donor had 左胸/右胸.

Shells driven by hair bones are left behind: in `cloth01` those are ornaments hanging
off the braid at z 3.2, not underwear.

    python3 tools/zzz2pmx/graftmaterial.py <donor.pmx> <material> [--as NAME] [--tex NAME]
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pmx import PMX
from pmxw import write_pmx
from graftlegs import rot_between, material_slice
from surface import Mesh
import roundring as rr

TARGET = 'public/models/reze/reze.pmx'
SKIP_BONE_PREFIX = ('hair',)
BODY = 'skin'
MIN_OFFSET = 0.025   # nothing may end up under the surface it sits on


def resolve(name, donor_parent, have):
    """The nearest ancestor of `name` (itself included) that the target owns."""
    seen = 0
    while name is not None and seen < 12:
        if name in have:
            return name
        name = donor_parent.get(name)
        seen += 1
    return None


def transforms(d, t, offsets=None, scale=None):
    """`offsets` moves a target bone onto the skin it drives, so that a garment
    welded to `頭` lands where the chains hung off `頭` land. Without it the ear cups
    ride the bone and the ear tips ride the chain, and the ear tears in half.

    `scale`, when given, replaces the per-bone segment ratio with one number for the
    whole model and squares the rotation to the world. The ratio is a good scale only
    where the segment is a limb: between 芭丝特 and us, 391 of 426 segments are
    shorter than 1.5 and their ratios run the full 0.5..2.0 clamp. 首->頭 is 0.63 on
    her and 1.02 on us, which is not that her head is 62% larger -- the two heads
    finish within 1% -- but that the bone sits at a different depth in the skull, and
    it arrived as bunny ears 1.6x oversized floating over the hair. Where the bone
    goes is the bone's business; how big the thing on it is, is the body's.
    """
    dpos = {b['name']: np.array(b['pos']) for b in d.bones}
    tpos = {b['name']: np.array(b['pos']) for b in t.bones}
    dpar = {b['name']: (d.bones[b['parent']]['name'] if b['parent'] >= 0 else None)
            for b in d.bones}
    tpar = {b['name']: (t.bones[b['parent']]['name'] if b['parent'] >= 0 else None)
            for b in t.bones}
    M = {}
    for name in dpos:
        tn = resolve(name, dpar, tpos)
        if tn is None:
            continue
        if scale is not None:
            s, R = scale, np.eye(3)
        else:
            du = dpos[name] - dpos.get(dpar.get(name) or '', dpos[name])
            tv = tpos[tn] - tpos.get(tpar.get(tn) or '', tpos[tn])
            if np.linalg.norm(du) < 1e-6 or np.linalg.norm(tv) < 1e-6:
                s, R = 1.0, np.eye(3)
            else:
                s = np.linalg.norm(tv) / np.linalg.norm(du)
                s = float(np.clip(s, 0.5, 2.0))
                R = rot_between(du, tv)
        M[name] = (dpos[name], tpos[tn] + (offsets or {}).get(tn, 0.0), s, R, tn)
    return M


def main(donor_path, matname, asname=None, texname=None, far_x=0.0, conform=False,
         min_offset=MIN_OFFSET, target=TARGET, out=None, body=BODY,
         donor_body=None, offsets=None, scale=None, cut_above=None,
         place=None):
    out = out or target
    t = PMX(target)
    d = PMX(donor_path)
    tidx = {b['name']: i for i, b in enumerate(t.bones)}
    tris, dmat = material_slice(d, matname)
    Vd = np.array([v[0] for v in d.vertices])
    Nd = np.array([v[1] for v in d.vertices])
    dnames = [b['name'] for b in d.bones]

    # drop shells that ride the hair
    keepset = set()
    for ids in rr.shells(Vd, tris).values():
        hair = 0
        for i in ids:
            v = d.vertices[int(i)]
            top = dnames[v[5][int(np.argmax(v[6][:len(v[5])]))]]
            hair += top.startswith(SKIP_BONE_PREFIX)
        p = Vd[np.array(ids)]
        if far_x and np.abs(p[:, 0]).mean() > far_x:
            print(f'   skipping {len(ids)}-vertex shell out at x '
                  f'{np.abs(p[:,0]).mean():.2f} (armband)')
            continue
        if hair / len(ids) > 0.5:
            print(f'   skipping {len(ids)}-vertex shell on hair bones '
                  f'(y {p[:,1].min():.2f}..{p[:,1].max():.2f}, z {p[:,2].mean():.2f})')
            continue
        keepset |= set(int(i) for i in ids)
    if cut_above is not None:                # the donor's own neck, in her frame
        above = set(int(i) for i in np.unique(tris) if Vd[int(i)][1] > cut_above)
        gone = sum(1 for f in tris if any(int(x) in above for x in f))
        keepset -= above
        print(f'   cutting {gone} tris above y {cut_above} -- our head goes there')
    faces = np.array([f for f in tris if all(int(x) in keepset for x in f)])
    keep = np.unique(faces)
    print(f'{matname}: keeping {len(faces)} of {len(tris)} tris, {len(keep)} vertices')

    M = transforms(d, t, offsets, scale)
    swaps = {n: M[n][4] for n in M if M[n][4] != n}
    for n, tn in sorted(swaps.items()):
        if any(n == dnames[b] for i in keep for b, w in
               zip(d.vertices[int(i)][5], d.vertices[int(i)][6]) if w > 0):
            print(f'   {n} -> {tn}')

    if place is not None:      # placed by the two bodies, not by the two skeletons
        ids = np.array(sorted(int(i) for i in keep))
        moved = place(Vd[ids])
        newpos = {int(i): moved[k] for k, i in enumerate(ids)}
        newnor = {int(i): Nd[int(i)] for i in ids}
    else:
      newpos, newnor = {}, {}
      for i in keep:
        v = d.vertices[int(i)]
        acc, nacc, tot = np.zeros(3), np.zeros(3), 0.0
        for b, w in zip(v[5], v[6]):
            if w <= 0:
                continue
            nm = dnames[b]
            if nm not in M:
                continue
            a0, b0, s, R, _ = M[nm]
            acc += w * (b0 + s * (R @ (Vd[i] - a0)))
            nacc += w * (R @ Nd[i])
            tot += w
        newpos[int(i)] = Vd[i] if tot < 1e-6 else acc / tot
        newnor[int(i)] = Nd[i] if tot < 1e-6 else nacc / (np.linalg.norm(nacc) + 1e-12)

    if conform:
        # Keep the garment's standoff from the body, but take it from OUR body: the
        # donor's chest is not ours, and a bra placed by bone alone floats off it.
        dtris, _ = material_slice(d, donor_body or body)
        ttris, _ = material_slice(t, body)
        Vt = np.array([v[0] for v in t.vertices])
        dm, tm = Mesh(Vd, dtris), Mesh(Vt, ttris)
        idxs = np.array(sorted(newpos))
        dd, dt, dw = dm.closest(Vd[idxs])
        dtri = dtris[dt]
        dbase = (Vd[dtri[:, 0]] * dw[:, :1] + Vd[dtri[:, 1]] * dw[:, 1:2]
                 + Vd[dtri[:, 2]] * dw[:, 2:3])
        stand = np.einsum('ij,ij->i', Vd[idxs] - dbase, dm.normal[dt])
        # A strap lies flat on the donor, so its standoff is near zero; carried across
        # with any error at all it lands under our skin and only its edges show, as
        # dark slivers. Hold everything clear of the surface.
        sunk = int((stand < min_offset).sum())
        stand = np.maximum(stand, min_offset)
        P = np.array([newpos[int(i)] for i in idxs])
        td, tt, tw = tm.closest(P)
        ttri = ttris[tt]
        tbase = (Vt[ttri[:, 0]] * tw[:, :1] + Vt[ttri[:, 1]] * tw[:, 1:2]
                 + Vt[ttri[:, 2]] * tw[:, 2:3])
        moved = tbase + tm.normal[tt] * stand[:, None]
        print(f'   lifted {sunk} vertices to at least {min_offset} off the body')
        print(f'   conformed to our {body}: standoff {stand.min():+.3f}..{stand.max():+.3f}, '
              f'vertices shifted by up to {np.abs(moved - P).max():.3f}')
        for k, i in enumerate(idxs):
            newpos[int(i)] = moved[k]

    tex = t.textures.index(texname) if texname else dmat['tex']
    base, remap = len(t.vertices), {}
    for i in keep:
        v = d.vertices[int(i)]
        bones = []
        for b in v[5]:
            nm = dnames[b]
            bones.append(tidx.get(M[nm][4], 0) if nm in M else 0)
        add = [np.zeros(4) for _ in range(t.uv_add)]
        wts = np.array(v[6], float)          # hers sum to 1.004 here and there
        tot = wts[:len(v[5])].sum()
        if tot > 0:
            wts[:len(v[5])] /= tot
        rec = (list(newpos[int(i)]), list(newnor[int(i)]), list(v[2]), add,
               v[4], bones, [float(x) for x in wts], v[7])
        remap[int(i)] = base + len(remap)
        t.vertices.append(rec + (v[8],) if v[4] == 3 else rec)

    flat = list(t.faces)
    for f in faces:
        flat += [remap[int(f[0])], remap[int(f[1])], remap[int(f[2])]]
    t.faces = flat

    # Normals must agree with the winding they are drawn under; conforming moves the
    # surface enough that the donor's no longer do, and borrowing the body's flips
    # one triangle in eight. Rebuilt from the grafted faces themselves.
    Vn = np.array([v[0] for v in t.vertices])
    acc = np.zeros_like(Vn)
    for f in faces:
        i0, i1, i2 = remap[int(f[0])], remap[int(f[1])], remap[int(f[2])]
        n = np.cross(Vn[i1] - Vn[i0], Vn[i2] - Vn[i0])
        for j in (i0, i1, i2):
            acc[j] += n
    for i in remap.values():
        ln = np.linalg.norm(acc[i])
        if ln > 1e-12:
            v = list(t.vertices[i])
            v[1] = list(acc[i] / ln)
            t.vertices[i] = tuple(v)

    mat = dict(dmat)
    mat['name'] = asname or matname
    mat['fcount'] = len(faces) * 3
    mat['tex'] = tex
    mat['sph'] = -1
    mat['sph_mode'] = 0
    mat['toon_shared'] = t.materials[0]['toon_shared']
    mat['toon'] = t.materials[0]['toon']
    t.materials.append(mat)
    write_pmx(out, t)
    print(f'added as "{mat["name"]}" using {t.textures[tex]}')


if __name__ == '__main__':
    a = sys.argv
    asn = a[a.index('--as') + 1] if '--as' in a else None
    tex = a[a.index('--tex') + 1] if '--tex' in a else None
    fx = float(a[a.index('--skip-far-x') + 1]) if '--skip-far-x' in a else 0.0
    mo = float(a[a.index('--min-offset') + 1]) if '--min-offset' in a else MIN_OFFSET
    ca = float(a[a.index('--cut-above') + 1]) if '--cut-above' in a else None
    tgt = a[a.index('--target') + 1] if '--target' in a else TARGET
    out = a[a.index('--out') + 1] if '--out' in a else None
    bod = a[a.index('--body') + 1] if '--body' in a else BODY
    dbod = a[a.index('--donor-body') + 1] if '--donor-body' in a else None
    main(a[1], a[2], asn, tex, fx, '--conform' in a, mo, tgt, out, bod, dbod,
         cut_above=ca)
