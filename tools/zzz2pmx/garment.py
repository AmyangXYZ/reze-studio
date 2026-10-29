# -*- coding: utf-8 -*-
"""Cut the mod's single body draw into the garments it is made of.

XXMI dumps one vertex buffer per draw call, so the whole outfit — skin, shirt,
shorts, socks, shoes — arrives as one material. The pieces are separate surfaces
all the same, but splitting on shared vertices finds 134 fragments rather than a
dozen garments, because a UV seam duplicates every vertex along it. Welding
coincident positions first puts the seams back together and the real parts fall out.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# Each garment named by where it sits, so a part is recognised by its shape rather
# than by a component number that changes with any edit to the input. A garment can
# be several parts: the choker is a band, two rings and a cord.
BOMB_SIGNATURES = {
    'head': dict(y=(17.8, 21.6), halfwidth=(0.6, 1.1), min_verts=2000),
    'pin': dict(y=(19.4, 22.0), halfwidth=(0.8, 1.4), min_verts=200, side='back'),
    'top': dict(y=(13.6, 19.0), halfwidth=(1.0, 1.8), min_verts=1200),   # the bomb jacket
    'skirt': dict(y=(9.0, 14.8), halfwidth=(1.8, 2.5), min_verts=1000),
}

SIGNATURES = {
    'shirt': dict(y=(13.6, 19.0), halfwidth=(0.8, 2.2), min_verts=600),
    'shorts': dict(y=(9.0, 14.8), halfwidth=(1.2, 2.6), min_verts=600),
    'ribbon': dict(y=(16.6, 18.2), halfwidth=(0.0, 0.9), min_verts=40, side='front'),
    'choker': dict(y=(17.7, 18.9), halfwidth=(0.0, 0.75), min_verts=40, side='back'),
}


def parts(pos, tris, weld=1e-4):
    """Split a triangle soup into welded surface parts. Returns (labels, used, faces)."""
    used = np.unique(tris)
    idx = {int(o): i for i, o in enumerate(used)}
    f = np.vectorize(idx.get)(tris)
    P = pos[used]
    pairs = cKDTree(P).query_pairs(weld, output_type='ndarray')
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]
                       + ([pairs] if len(pairs) else []))
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(used), len(used)))
    _, lab = connected_components(A, directed=False)
    return lab, used, f


def identify(pos, tris, wanted=SIGNATURES, verbose=True):
    """Return {name: face array} for each garment the signatures describe."""
    lab, used, _ = parts(pos, tris)
    P = pos[used]
    cand = {}
    for c in range(lab.max() + 1):
        s = lab == c
        if s.sum() < 40:
            continue
        cand[c] = dict(n=int(s.sum()), y=(P[s, 1].min(), P[s, 1].max()),
                       hw=max(abs(P[s, 0].min()), abs(P[s, 0].max())),
                       zmax=P[s, 2].max(), centre=P[s].mean(0))
    idx = np.vectorize({int(o): i for i, o in enumerate(used)}.get)(tris)
    out = {}
    claimed = {}
    for name, sig in wanted.items():
        hits = []
        for c, d in cand.items():
            if d['n'] < sig['min_verts']:
                continue
            if not (sig['y'][0] <= d['y'][0] and d['y'][1] <= sig['y'][1]):
                continue
            if not (sig['halfwidth'][0] <= d['hw'] <= sig['halfwidth'][1]):
                continue
            if sig.get('side') == 'front' and d['centre'][2] > 0:
                continue
            if sig.get('side') == 'back' and d['centre'][2] < 0:
                continue
            hits.append(c)
        if not hits:
            raise SystemExit(f'garment: nothing matches the signature for {name}')
        for c in hits:
            if c in claimed:
                raise SystemExit(f'garment: part {c} matches both {claimed[c]} and {name}')
            claimed[c] = name
        keep = np.isin(lab[idx], hits).all(1)
        out[name] = tris[keep]
        if verbose:
            ys = [cand[c]['y'] for c in hits]
            print(f'  {name:8s} {keep.sum():5d} tris  from {len(hits)} part(s)  '
                  f'Y[{min(y[0] for y in ys):6.2f},{max(y[1] for y in ys):6.2f}]')
    return out


def collect(pos, tris, predicate):
    """Every welded part the predicate accepts, as one face array.

    The fuse strip is twenty-two separate rungs and the gloves are a slice off the
    arm shell; neither is a single component, so they are gathered by rule rather
    than named.
    """
    lab, used, _ = parts(pos, tris)
    P = pos[used]
    idx = np.vectorize({int(o): i for i, o in enumerate(used)}.get)(tris)
    hits = []
    for c in range(lab.max() + 1):
        s = lab == c
        if s.sum() < 20:
            continue
        d = dict(n=int(s.sum()), y=(P[s, 1].min(), P[s, 1].max()),
                 hw=float(abs(P[s, 0]).max()), z=float(P[s, 2].mean()))
        if predicate(d):
            hits.append(c)
    if not hits:
        return tris[:0]
    return tris[np.isin(lab[idx], hits).all(1)]
