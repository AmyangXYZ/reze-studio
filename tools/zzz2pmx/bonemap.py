# -*- coding: utf-8 -*-
"""Match the mod's game bones to the rip's MMD bones.

The rip is a dump of the same game model, so the two skeletons are one skeleton
and the match is a bijection. Each bone is located by the weighted centroid of the
vertices it drives, which is outfit-independent enough to identify it even though
Reze's mesh is not Evelyn's.

Centroids alone put a heavy bone on 左親指２ while its mirror lands on 右親指２ with
a third of the mass. Matching mirror *pairs* instead of single bones removes that:
it doubles the evidence behind every decision and makes an asymmetric answer
unrepresentable.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _side_from_name(name):
    if name.startswith('左'):
        return 'L'
    if name.startswith('右'):
        return 'R'
    return 'C'


def _mirror_pairs(keys, cen, centre_tol):
    """Split bones into mirrored (left, right) pairs and centre bones."""
    xs = np.array([cen[k][0] for k in keys])
    centre = [k for k, x in zip(keys, xs) if abs(x) <= centre_tol]
    left = [k for k, x in zip(keys, xs) if x > centre_tol]
    right = [k for k, x in zip(keys, xs) if x < -centre_tol]
    pairs, spare = [], []
    if left and right:
        L = np.array([cen[k] for k in left])
        R = np.array([cen[k] for k in right]) * [-1, 1, 1]
        d = np.linalg.norm(L[:, None, :] - R[None, :, :], axis=2)
        ri, ci = linear_sum_assignment(d)
        used_r = set()
        for i, j in zip(ri, ci):
            pairs.append((left[i], right[j]))
            used_r.add(j)
        spare += [right[j] for j in range(len(right)) if j not in used_r]
        spare += [left[i] for i in range(len(left)) if i not in set(ri)]
    else:
        spare += left + right
    return pairs, centre, spare


def build(mod_cen, mod_mass, host_cen, host_mass, host_names, centre_tol=0.35,
          mass_weight=0.25):
    """Return {game bone: mmd bone}, plus the game bones nothing claimed."""
    mk = sorted(mod_cen)
    hk = sorted(host_cen)
    mtot = sum(mod_mass.values())
    htot = sum(host_mass.values())

    mpairs, mcentre, mspare = _mirror_pairs(mk, mod_cen, centre_tol)
    hpairs = []
    hcentre = []
    seen = set()
    byname = {}
    for k in hk:
        s = _side_from_name(host_names[k])
        if s == 'C':
            hcentre.append(k)
        else:
            byname.setdefault(host_names[k][1:], {})[s] = k
    for stem, d in byname.items():
        if 'L' in d and 'R' in d:
            hpairs.append((d['L'], d['R']))
            seen.update(d.values())
    hspare = [k for k in hk if k not in seen and k not in hcentre]

    def cost(mgroup, hgroup):
        C = np.zeros((len(mgroup), len(hgroup)))
        for i, mg in enumerate(mgroup):
            for j, hg in enumerate(hgroup):
                d = sum(np.linalg.norm(mod_cen[a] - host_cen[b]) for a, b in zip(mg, hg))
                lm = sum(mod_mass[a] for a in mg) / mtot
                lh = sum(host_mass[b] for b in hg) / htot
                C[i, j] = d / len(mg) + mass_weight * abs(np.log(lm / lh))
        return C

    mapping = {}
    report = []
    for mgroup, hgroup in ((mpairs, hpairs), ([(k,) for k in mcentre], [(k,) for k in hcentre])):
        if not mgroup or not hgroup:
            continue
        C = cost(mgroup, hgroup)
        ri, ci = linear_sum_assignment(C)
        for i, j in zip(ri, ci):
            for a, b in zip(mgroup[i], hgroup[j]):
                mapping[a] = b
                report.append((a, b, np.linalg.norm(mod_cen[a] - host_cen[b]), mod_mass[a]))

    unmatched = [k for k in mk if k not in mapping]
    return mapping, unmatched, report


def weight_matrix(bones, weights, subset, keys):
    """Dense vertex x bone weights over the bones we care about."""
    col = {b: i for i, b in enumerate(keys)}
    W = np.zeros((len(subset), len(keys)))
    for c in range(bones.shape[1]):
        b = bones[subset, c]
        w = weights[subset, c]
        for k, i in col.items():
            m = b == k
            if m.any():
                W[m, i] += w[m]
    return W


def fallback(unmatched, mapping, W, keys):
    """Send an unclaimed game bone where the bone it shares the most skin with went.

    Centroid distance is the wrong question here: a palm bone with no MMD
    counterpart sits nearest the thumb tip, and following that puts half a hand on
    a fingertip. Two bones that blend across the same vertices are neighbours in
    the skeleton, which is what the weight has to inherit.
    """
    col = {b: i for i, b in enumerate(keys)}
    resolved = dict(mapping)
    out = {}
    todo = [u for u in unmatched if u in col]
    while todo:
        best = (0.0, None, None)
        for u in todo:
            wu = W[:, col[u]]
            for anchor in resolved:
                if anchor not in col:
                    continue
                v = np.minimum(wu, W[:, col[anchor]]).sum()
                if v > best[0]:
                    best = (v, u, anchor)
        if best[1] is None:
            break        # what is left touches nothing that reaches the rig
        _, u, anchor = best
        out[u] = resolved[u] = resolved[anchor]
        todo.remove(u)
    return out


def remap(bones, weights, subset, mapping):
    """Rewrite the mod's own skin weights onto MMD bones."""
    out = []
    for i in subset:
        acc = {}
        for b, w in zip(bones[i], weights[i]):
            if w <= 0:
                continue
            tgt = mapping.get(int(b))
            if tgt is None:
                continue
            acc[tgt] = acc.get(tgt, 0.0) + float(w)
        out.append(acc)
    return out


def top4(d):
    """Reduce a bone->weight dict to the four heaviest, renormalised."""
    items = sorted(d.items(), key=lambda kv: -kv[1])[:4]
    tot = sum(w for _, w in items)
    if tot <= 0:
        return [0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]
    bones = [b for b, _ in items]
    ws = [w / tot for _, w in items]
    while len(bones) < 4:
        bones.append(0)
        ws.append(0.0)
    return bones, ws


# Bones whose position MMD reads as pure hierarchy: a motion drives their rotation,
# or physics drives them, and nothing targets them absolutely. 足D/ひざD/足首D are
# excluded on purpose — a VMD stores 足ＩＫ as a translation against the rig's own
# leg length, so moving the leg chain moves every footfall in every motion.
RETARGET = ('腕', 'ひじ', '手首', '手捩', '腕捩', '指', 'ダミー', '手先',
            '足3', '足4', '足5', 'ひざ4', '臀')


def retarget(model, groups=RETARGET, min_mass=2.0):
    """Move bones onto the joints the mod's own skin puts them at.

    The rip's bones are Evelyn's, and Evelyn wears gauntlets: her fingertip bones
    stand a full unit past the end of Reze's much shorter hand. Weighting is not the
    problem — curling a finger around a pivot outside the mesh is, and it stretches
    the finger into a spike the moment a motion touches it.
    """
    V = np.array([v[0] for v in model.vertices])
    n = len(model.bones)
    W = np.zeros((len(V), n))
    for i, v in enumerate(model.vertices):
        for b, w in zip(v[5], v[6]):
            if w > 0:
                W[i, b] += w

    names = [b['name'] for b in model.bones]
    want = [i for i, nm in enumerate(names) if any(g in nm for g in groups)]
    delta = {}
    for b in want:
        p = model.bones[b]['parent']
        if p < 0:
            continue
        w = W[:, b] * W[:, p]
        if w.sum() < min_mass:
            continue
        joint = (V * w[:, None]).sum(0) / w.sum()
        delta[b] = joint - np.array(model.bones[b]['pos'])

    # tips carry no weight of their own; they follow the segment they end
    for b in want:
        if b in delta:
            continue
        p = model.bones[b]['parent']
        if 0 <= p < n and p in delta:
            delta[b] = delta[p]

    for b, d in delta.items():
        model.bones[b]['pos'] = list(np.array(model.bones[b]['pos']) + d)
    moved = {i: delta[i] for i in delta}
    for r in model.rigids:
        if r['bone'] in moved:
            r['pos'] = list(np.array(r['pos']) + moved[r['bone']])
    body_delta = {i: moved[r['bone']] for i, r in enumerate(model.rigids) if r['bone'] in moved}
    for j in model.joints:
        if j['a'] in body_delta:
            j['pos'] = list(np.array(j['pos']) + body_delta[j['a']])
    return len(delta), float(np.median([np.linalg.norm(d) for d in delta.values()]))
