"""Fit the game->MMD placement of a ZZZ mod against a ripped PMX of the same character.

The mod and the PMX share a skeleton, so matching each bone's weighted vertex
centroid on both sides gives an outfit-independent landmark set. The head is fitted
separately: character mods routinely resize the head, and the face plate that the
PMX supplies has to land on the head the mod actually has.
"""
import numpy as np

# game (x, y, z-up), character facing -y  ->  MMD (x, y-up, z), character facing -Z.
# Swapping the last two axes alone has determinant -1: that is a reflection, and it
# hands back a model whose left and right are exchanged. The geometry is symmetric,
# so nothing looks wrong until you read the texture — Reze's choker pull-ring ends up
# on her left. Negating x as well makes it a rotation, and the winding then stands.
AXES = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float)
assert abs(np.linalg.det(AXES) - 1.0) < 1e-9, 'the axis map must be a rotation, not a reflection'
HEAD_BONES = ('頭', '首', '両目', '右目', '左目')


def to_mmd(p):
    return p @ AXES.T


def bone_centroids(pos, bones, weights, subset=None, min_mass=5.0):
    """Weighted centroid of the vertices each bone drives."""
    idx = np.arange(len(pos)) if subset is None else subset
    out = {}
    mass = {}
    for b in np.unique(bones[idx]):
        w = np.where(bones[idx] == b, weights[idx], 0).sum(1)
        t = w.sum()
        if t < min_mass:
            continue
        out[int(b)] = (pos[idx] * w[:, None]).sum(0) / t
        mass[int(b)] = float(t)
    return out, mass


def host_centroids(pmx, verts, subset, min_mass=5.0):
    acc, mass = {}, {}
    for i in subset:
        v = pmx.vertices[i]
        for b, w in zip(v[5], v[6]):
            if w <= 1e-4:
                continue
            acc[b] = acc.get(b, np.zeros(3)) + w * verts[i]
            mass[b] = mass.get(b, 0.0) + w
    return ({b: acc[b] / mass[b] for b in acc if mass[b] >= min_mass},
            {b: mass[b] for b in acc if mass[b] >= min_mass})


def match(mod_cen, mod_mass, host_cen, host_mass, s0, t0, max_dist=2.0, iters=200):
    """Mutual-nearest bone matching under a similarity, refit until stable."""
    mk = sorted(mod_cen, key=lambda b: -mod_mass[b])
    hk = sorted(host_cen, key=lambda b: -host_mass[b])
    MP = np.array([mod_cen[b] for b in mk])
    HP = np.array([host_cen[b] for b in hk])
    mm = np.array([mod_mass[b] for b in mk])
    hm = np.array([host_mass[b] for b in hk])
    s, t, pairs = s0, np.asarray(t0, float), []
    for _ in range(iters):
        d = np.linalg.norm((s * MP + t)[:, None, :] - HP[None, :, :], axis=2)
        j, i2 = d.argmin(1), d.argmin(0)
        pairs = [(i, j[i]) for i in range(len(MP)) if i2[j[i]] == i and d[i, j[i]] < max_dist]
        if len(pairs) < 8:
            break
        s, t = _fit(MP, HP, mm, hm, pairs)
    return s, t, [(mk[i], hk[j]) for i, j in pairs], (MP, HP, mm, hm, mk, hk)


def _fit(MP, HP, mm, hm, pairs):
    A = MP[[p[0] for p in pairs]]
    B = HP[[p[1] for p in pairs]]
    w = np.minimum(mm[[p[0] for p in pairs]], hm[[p[1] for p in pairs]])
    ca = (A * w[:, None]).sum(0) / w.sum()
    cb = (B * w[:, None]).sum(0) / w.sum()
    s = float((w[:, None] * (A - ca) * (B - cb)).sum() / (w[:, None] * (A - ca) ** 2).sum())
    return s, cb - s * ca


def refit(MP, HP, mm, hm, pairs):
    return _fit(MP, HP, mm, hm, pairs)


def icp_similarity(S, T, s0, t0, iters=80, trim=85, fixed_scale=False):
    """Uniform scale + translation ICP of point set S onto point set T; with
    `fixed_scale` the scale stays s0 and only the translation is solved."""
    from scipy.spatial import cKDTree
    tree = cKDTree(T)
    s, t = float(s0), np.asarray(t0, float).copy()
    for _ in range(iters):
        d, idx = tree.query(s * S + t)
        keep = d < max(np.percentile(d, trim), 1e-9)
        A, B = S[keep], T[idx[keep]]
        ca, cb = A.mean(0), B.mean(0)
        if not fixed_scale:
            s = float(((A - ca) * (B - cb)).sum() / ((A - ca) ** 2).sum())
        t = cb - s * ca
    d, _ = tree.query(s * S + t)
    return s, t, d
