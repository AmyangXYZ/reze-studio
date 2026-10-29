# -*- coding: utf-8 -*-
"""Carry an accessory chain across ahead of the garment that hangs on it.

`graftmaterial.py` moves a garment's mesh, and a donor bone we do not have resolves
up the parent chain to one we do. That is right for a bone we own under another name
and wrong for a skirt: 芭丝特's 54 flap bones all resolve to `下半身`, and the flaps
become one rigid shell on the hip. Chains have to arrive as chains -- the bones, the
rigid bodies hanging off them, and the joints between -- which is what
`hairgraft.py --physics copy` does for hair, with the donor's own rig instead of a
reference build's.

Run this first. The garment then finds these bones by name and never resolves past
them.

Placement is fitted, not inherited. Taking the anchor bone's own parent segment --
what `graftmaterial` uses per vertex -- turns the whole skirt by whatever rotation
carries `センター->下半身` onto ours, and those two segments are short enough that
their angle is noise. Each chain instead gets a similarity fitted over the shared
bones standing around it, which is stable because it is overdetermined.

    python3 tools/zzz2pmx/chaingraft.py <donor.pmx> <target.pmx> <out.pmx> \
        --materials 上衣,背带,尾巴,耳朵 [--frame 兔女郎] [--dry-run]
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pmx import PMX
from pmxw import write_pmx

MIN_ANCHORS = 6      # below this a similarity is fitted to noise
NEAR = 1.5           # the falloff radius, in multiples of the chain's own radius


def umeyama(A, B, w=None):
    """Similarity s, R, t with B ~= s R A + t, R a rotation.

    Weighted, because every bone in the model is an anchor for every chain and only
    the near ones should decide it. Flat weights let 芭丝特's arms -- 18% shorter
    than ours -- set the scale of her skirt.
    """
    w = np.ones(len(A)) if w is None else w / w.sum()
    ca, cb = (w[:, None] * A).sum(0), (w[:, None] * B).sum(0)
    X, Y = A - ca, B - cb
    U, S, Vt = np.linalg.svd((w[:, None] * X).T @ Y)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = (U @ D @ Vt).T
    var = (w * (X ** 2).sum(1)).sum()
    s = float((S * np.diag(D)).sum() / var) if var > 1e-12 else 1.0
    return s, R, cb - s * R @ ca


def euler(M):
    """Back to the (rx, ry, rz) the engine composes as Ry.Rx.Rz."""
    sx = float(np.clip(M[2, 1], -1.0, 1.0))
    rx = np.arcsin(sx)
    if abs(sx) < 0.999999:
        ry = np.arctan2(M[2, 0], M[2, 2])
        rz = np.arctan2(-M[0, 1], M[1, 1])
    else:                                   # gimbal: fold rz into ry
        ry = np.arctan2(-M[0, 2], M[0, 0])
        rz = 0.0
    return [float(rx), float(ry), float(rz)]


def matrix(rot):
    rx, ry, rz = rot
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def garment_bones(d, materials):
    """Every donor bone the named materials actually lean on."""
    F = np.array(d.faces).reshape(-1, 3)
    at, used = 0, set()
    for mat in d.materials:
        n = mat['fcount'] // 3
        if mat['name'] in materials:
            for vi in np.unique(F[at:at + n]):
                v = d.vertices[int(vi)]
                for bi, w in zip(v[5], v[6]):
                    if w > 0.001 and bi >= 0:
                        used.add(int(bi))
        at += n
    return used


def chains(d, t, materials):
    """The donor bones to carry: absent here, closed over parents and over the
    joints that hold them, so a chain never arrives cut in half."""
    have = set(b['name'] for b in t.bones)
    names = [b['name'] for b in d.bones]
    keep = set(i for i in garment_bones(d, materials) if names[i] not in have)
    while True:
        grew = False
        for i in list(keep):                       # parents
            p = d.bones[i]['parent']
            if p >= 0 and names[p] not in have and p not in keep:
                keep.add(p); grew = True
        onbone = {}
        for k, r in enumerate(d.rigids):
            onbone.setdefault(r['bone'], []).append(k)
        mine = set(k for i in keep for k in onbone.get(i, []))
        for j in d.joints:                          # bodies joined to ours
            for a, b in ((j['a'], j['b']), (j['b'], j['a'])):
                if a in mine and 0 <= b < len(d.rigids):
                    ob = d.rigids[b]['bone']
                    if 0 <= ob < len(names) and names[ob] not in have and ob not in keep:
                        keep.add(ob); grew = True
        if not grew:
            return keep


def anchor_of(d, t, i):
    """The nearest ancestor of donor bone i that the target owns."""
    have = set(b['name'] for b in t.bones)
    names = [b['name'] for b in d.bones]
    while i >= 0:
        if names[i] in have:
            return names[i]
        i = d.bones[i]['parent']
    return None


def parts(d, ids):
    """The chain bones split into the accessories they are: a strap, an ear, the
    tail, one flap of a skirt. Each is placed on its own, since a fit good for the
    hip is not the one the tail wants."""
    root = {i: i for i in ids}

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]; i = root[i]
        return i
    for i in ids:
        p = d.bones[i]['parent']
        if p in root:
            root[find(i)] = find(p)
    out = {}
    for i in ids:
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def body_scale(d, t, spine=0.6):
    """One scale for the whole graft, read off the spine.

    Scale has to be global -- fitted per accessory it reads whatever the local bones
    happen to disagree about, and around the hip that is leg-root height: 芭丝特's 足
    sits high enough that 下半身->足 is 1.56x ours, nothing to do with how wide a hip
    is. And it has to come off the centre line. Fitted over the whole skeleton it
    comes out 1.16, which is the arms: hers are 17% shorter than ours and there are
    more bones in two arms than in a spine. The two models are in fact the same size
    -- their heads finish within 1% of each other.

    Taken as the median ratio of pairwise distances, which needs no correspondence of
    frames and does not care about the pose.
    """
    dpos = {b['name']: np.array(b['pos']) for b in d.bones}
    tpos = {b['name']: np.array(b['pos']) for b in t.bones}
    on_axis = [n for n in sorted(set(dpos) & set(tpos)) if abs(dpos[n][0]) < spine]
    r = [np.linalg.norm(tpos[a] - tpos[b]) / np.linalg.norm(dpos[a] - dpos[b])
         for i, a in enumerate(on_axis) for b in on_axis[i + 1:]
         if np.linalg.norm(dpos[a] - dpos[b]) > 0.5]
    s = float(np.median(r))
    print(f'body scale {s:.4f} from {len(r)} distances among {len(on_axis)} '
          f'centre-line bones (quartiles {np.percentile(r, 25):.3f}..'
          f'{np.percentile(r, 75):.3f})')
    return s


def driven(m, bone, cut=0.9):
    """The vertices that ride one bone and nothing else."""
    idx = [i for i, b in enumerate(m.bones) if b['name'] == bone]
    if not idx:
        return np.zeros((0, 3))
    idx = set(idx)
    return np.array([v[0] for v in m.vertices
                     if any(w > cut and b in idx for b, w in zip(v[5], v[6]))])


def seat(d, t, anchor, s, R, tr, rounds=12, keep=0.6, quiet=False):
    """Move the accessory onto the skin its anchor drives, rather than onto the
    anchor bone.

    芭丝特's `頭` sits 0.33 lower inside her skull than ours does inside ours, so a
    bunny ear placed by the bone floats that far off the head -- while the two skulls
    themselves finish within 1% of each other. The bones are landmarks of the rig,
    not of the head. Matched on the surface instead, trimmed to the closest 60% of
    pairs each round so that her hair -- which is nothing like ours -- is not part of
    what decides it.
    """
    A, B = driven(d, anchor), driven(t, anchor)
    if len(A) < 200 or len(B) < 200:
        return tr
    # A bone on the centre line drives skin that is symmetric about x=0 on both
    # models, so its seat cannot have a sideways component. Left to itself the fit
    # drifts 0.08 across, and since a pair of accessories is placed by one translation
    # they both slide the same way -- which is not a shift, it is a broken mirror, and
    # the viewer runs physics off it.
    axial = abs(float(np.mean([b['pos'][0] for b in d.bones
                               if b['name'] == anchor]))) < 0.1
    tree = cKDTree(B)
    tr0 = np.array(tr, float)
    tr = np.array(tr, float)
    if axial:
        tr[0] = 0.0
    P = s * (R @ A.T).T
    for _ in range(rounds):
        dist, j = tree.query(P + tr)
        k = np.argsort(dist)[:max(int(keep * len(dist)), 50)]
        step = (B[j[k]] - (P[k] + tr)).mean(0)
        if axial:
            step[0] = 0.0
        tr = tr + step
        if np.linalg.norm(step) < 1e-4:
            break
    if not quiet:
        dist, _ = tree.query(P + tr)
        print(f'    seated on {anchor}: {len(A)} of hers onto {len(B)} of ours, '
              f'moved {np.linalg.norm(tr - tr0):.3f}, trimmed residual '
              f'{np.sort(dist)[:int(keep * len(dist))].mean():.3f}')
    return tr


def seat_offsets(d, t, s=None, limit=0.5, least=200):
    """How far each bone stands from the skin it drives, donor against target.

    The correction `graftmaterial` needs for the same reason the chains do: a vertex
    welded to `頭` is placed there by where the two `頭` bones are, and those two bones
    sit at different depths in two skulls that are the same size. Returned as a
    translation per bone, to be added to where the bone would have put it.

    Only bones with real skin on both sides get one, and only up to `limit` -- past
    that the two bones are not driving the same thing and the bone is the better
    guess.
    """
    s = body_scale(d, t) if s is None else s
    R = np.eye(3)
    dpos = {b['name']: np.array(b['pos']) for b in d.bones}
    tpos = {b['name']: np.array(b['pos']) for b in t.bones}
    out = {}
    for n in sorted(set(dpos) & set(tpos)):
        A, B = driven(d, n), driven(t, n)
        if len(A) < least or len(B) < least:
            continue
        base = tpos[n] - s * R @ dpos[n]
        tr = seat(d, t, n, s, R, base, quiet=True)
        off = tr - base
        if np.linalg.norm(off) <= limit:
            out[n] = off
    for n, off in sorted(out.items(), key=lambda kv: -np.linalg.norm(kv[1]))[:6]:
        print(f'  {n} stands {np.linalg.norm(off):.3f} off its own skin '
              f'{np.round(off, 3)}')
    print(f'seated {len(out)} bones on the skin they drive')
    return out


def fit(d, t, ids, s, label, anchor=None):
    """Where one accessory goes: at the body's scale, seated on the skin its anchor
    drives, and square to the world.

    There is no rotation to find. Both models stand in the same A-pose on the same
    axes -- their upper arms agree to half a degree -- so the only honest answer is
    the identity, and fitting one anyway is how the bunny ears arrived upside down:
    a Procrustes over the six bones nearest an ear has nothing to orient itself by
    and returns whatever the noise says, which twice was a half turn.
    """
    dpos = {b['name']: np.array(b['pos']) for b in d.bones}
    tpos = {b['name']: np.array(b['pos']) for b in t.bones}
    R = np.eye(3)
    if anchor and anchor in dpos:
        tr = tpos[anchor] - s * dpos[anchor]
    else:                                   # nothing to hang it on: shared centroid
        shared = sorted(set(dpos) & set(tpos))
        tr = (np.array([tpos[n] for n in shared]).mean(0)
              - s * np.array([dpos[n] for n in shared]).mean(0))
    print(f'  {label}: {len(ids)} bones on {anchor}')
    if anchor:
        tr = seat(d, t, anchor, s, R, tr)
    return s, R, tr


def free_groups(t, donor_groups):
    """Keep a donor group number where the target leaves it free, move it where not."""
    taken = set(r['group'] for r in t.rigids)
    gmap, out = {}, set(taken)
    for g in sorted(donor_groups):
        if g not in out:
            gmap[g] = g
        else:
            g2 = next(k for k in range(16) if k not in out)
            gmap[g] = g2
            print(f'  group {g} is taken here, chains move to {g2}')
        out.add(gmap[g])
    return gmap


def remask(nocollide, gmap):
    """Rewrite a collides-with mask through the group remap."""
    out = 0xFFFF
    for g in range(16):
        if not (nocollide >> g) & 1:            # donor collides with g
            out &= ~(1 << gmap.get(g, g))
    return out & 0xFFFF


def main(donor_path, target_path, out_path, materials, frame=None, dry=False,
         scale=None, place=None):
    d, t = PMX(donor_path), PMX(target_path)
    names = [b['name'] for b in d.bones]
    ids = sorted(chains(d, t, materials))
    if not ids:
        print('no chain bones to carry'); return
    groups = sorted(parts(d, ids), key=len, reverse=True)
    print(f'{len(ids)} chain bones in {len(groups)} accessories')
    if place is not None:
        # Through the same field as the mesh, so a chain stays inside its own
        # geometry. A field is not a similarity, so the local one is read back off it:
        # where the accessory lands, and how much the field stretches it there.
        scale = place.scale
        xf = {}
        for group in groups:
            P = np.array([d.bones[i]['pos'] for i in group])
            c = P.mean(0)
            # Probed both ways along every axis. A one-sided probe is not mirror
            # symmetric, and a left strap then reads a different stretch from its
            # partner -- which reaches the rigid bodies as different sizes.
            probe = np.vstack([np.eye(3) * 0.5, -np.eye(3) * 0.5]) + c
            q = place(np.vstack([c[None], probe]))
            local = float(np.mean(np.linalg.norm(q[1:] - q[0], axis=1)) / 0.5)
            tr = q[0] - local * c
            print(f'  {names[min(group)]}: {len(group)} bones, field scale '
                  f'{local:.3f} where it sits')
            for i in group:
                xf[i] = (local, np.eye(3), tr)
    else:
        scale = body_scale(d, t) if scale is None else scale
        print(f'placing at scale {scale:.4f}')
        xf = {}
        for group in groups:
            s, R, tr = fit(d, t, group, scale, names[min(group)],
                           anchor_of(d, t, min(group)))
            for i in group:
                xf[i] = (s, R, tr)

    tidx = {b['name']: k for k, b in enumerate(t.bones)}
    clash = [names[i] for i in ids if names[i] in tidx]
    if clash:
        print(f'  refusing: {len(clash)} chain bones share a name here ({clash[:4]})')
        return

    # -- bones. Donor order keeps parents ahead of children, which PMX requires.
    bmap = {}
    for i in ids:
        b = dict(d.bones[i])
        s, R, tr = xf[i]
        b['pos'] = list(s * R @ np.array(b['pos']) + tr)
        p = d.bones[i]['parent']
        b['parent'] = bmap[p] if p in bmap else tidx.get(anchor_of(d, t, i), -1)
        if 'tail' in b:
            b['tail'] = bmap.get(b['tail'], -1)
            if b['tail'] < 0:
                del b['tail']
                b['flag'] &= ~0x0001
                b['tail_offset'] = [0.0, 0.0, 0.0]
        if 'tail_offset' in b:
            b['tail_offset'] = list(s * R @ np.array(b['tail_offset']))
        if 'grant' in b:                     # a source we did not carry drives nothing
            g, rate = b['grant']
            src = bmap.get(g, tidx.get(names[g] if 0 <= g < len(names) else '', -1))
            if src < 0:
                del b['grant']; b['flag'] &= ~0x0300
            else:
                b['grant'] = (src, rate)
        if 'axis' in b:
            b['axis'] = list(R @ np.array(b['axis']))
        if 'local_x' in b:
            b['local_x'] = list(R @ np.array(b['local_x']))
            b['local_z'] = list(R @ np.array(b['local_z']))
        if 'ik_links' in b:                  # accessory chains carry no IK worth keeping
            for k in ('ik_target', 'ik_loop', 'ik_limit', 'ik_links'):
                b.pop(k, None)
            b['flag'] &= ~0x0020
        bmap[i] = len(t.bones)
        t.bones.append(b)

    # -- rigid bodies
    onbone = {}
    for k, r in enumerate(d.rigids):
        if r['bone'] in bmap:
            onbone.setdefault(r['bone'], []).append(k)
    carried = [k for i in ids for k in onbone.get(i, [])]
    gmap = free_groups(t, set(d.rigids[k]['group'] for k in carried))
    rmap = {}
    for k in carried:
        r = dict(d.rigids[k])
        s, R, tr = xf[r['bone']]
        r['bone'] = bmap[r['bone']]
        r['group'] = gmap[r['group']]
        r['nocollide'] = remask(r['nocollide'], gmap)
        r['size'] = list(np.array(r['size']) * s)
        r['pos'] = list(s * R @ np.array(r['pos']) + tr)
        r['rot'] = euler(R @ matrix(r['rot']))
        rmap[k] = len(t.rigids)
        t.rigids.append(r)

    # -- joints. An end anchored on a donor body we did not carry re-anchors onto
    # our own body for that bone; without it the chain hangs from nothing.
    here = {}
    for k, r in enumerate(t.rigids):
        if 0 <= r['bone'] < len(t.bones):
            here.setdefault(t.bones[r['bone']]['name'], []).append((r['mode'], k))
    anchored, dropped, jn = 0, 0, 0
    for j in d.joints:
        if j['a'] not in rmap and j['b'] not in rmap:
            continue                             # nothing of ours hangs on it
        ends = []
        for e in (j['a'], j['b']):
            if e in rmap:
                ends.append(rmap[e]); continue
            bn = names[d.rigids[e]['bone']] if 0 <= e < len(d.rigids) \
                and 0 <= d.rigids[e]['bone'] < len(names) else None
            cand = sorted(here.get(bn, []))          # a static body first
            if not cand:
                ends.append(None); continue
            ends.append(cand[0][1]); anchored += 1
        if any(e is None for e in ends):
            dropped += 1; continue
        s, R, tr = xf[d.rigids[j['a']]['bone']] if j['a'] in rmap else \
            xf[d.rigids[j['b']]['bone']]
        jj = dict(j)
        jj['a'], jj['b'] = ends
        jj['pos'] = list(s * R @ np.array(j['pos']) + tr)
        jj['rot'] = euler(R @ matrix(j['rot']))
        jj['lin_min'] = list(np.array(j['lin_min']) * s)
        jj['lin_max'] = list(np.array(j['lin_max']) * s)
        t.joints.append(jj); jn += 1

    if frame:
        t.frames.append(dict(name=frame, name_en=frame, special=0,
                             items=[(0, bmap[i]) for i in ids]))

    print(f'carried {len(bmap)} bones, {len(rmap)} rigid bodies, {jn} joints '
          f'({anchored} ends re-anchored on our own bodies, {dropped} joints dropped)')
    print(f'now {len(t.bones)} bones, {len(t.rigids)} rigid bodies, {len(t.joints)} joints')
    if dry:
        print('dry run, nothing written'); return
    write_pmx(out_path, t)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    a = sys.argv
    mats = a[a.index('--materials') + 1].split(',') if '--materials' in a else []
    fr = a[a.index('--frame') + 1] if '--frame' in a else None
    sc = float(a[a.index('--scale') + 1]) if '--scale' in a else None
    main(a[1], a[2], a[3], mats, fr, '--dry-run' in a, sc)
