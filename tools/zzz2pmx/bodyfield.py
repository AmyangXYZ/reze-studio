# -*- coding: utf-8 -*-
"""The difference between two bodies, as a field anything can be carried through.

This is `graft.py`'s placement, generalised from a ZZZ buffer dump to a second PMX
and reused for a whole outfit rather than four pieces. A uniform-scale ICP puts the
donor's body on ours, and what is left over -- where her surface stands against ours,
smoothed until only the low frequencies survive -- is the shape correction. Read it
off the two BODIES, never off the two garments: warping a garment onto the one it
replaces drags its collar to wherever the old frill was.

`graft.py` trims both bodies to `|x| < 3.2` before matching them, because in an
A-pose the nearest point on the host to a hip is the forearm hanging beside it. That
works and it throws the arms away with the problem, which is affordable for a shirt
and not for an outfit whose pieces reach the wrists, the ankles and the head. The
correspondence is gated by bone here instead: a point on her hip may only match a
point on ours that the same bone drives. The arms then carry a real correction rather
than an extrapolated one, and a hip still cannot match a wrist.

Bones are placed through the same field as the vertices they drive, so that an
accessory's chain stays inside its own mesh.
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, __file__.rsplit('/', 1)[0])
import align

RADIUS = 1.2         # graft.py's, and it is a body-shape length
TRIM = 99            # percentile of correspondences kept


def _points(m, mats):
    """A body's vertices and the bone driving each, from named materials."""
    F = np.array(m.faces).reshape(-1, 3)
    at, tris = 0, []
    for mat in m.materials:
        n = mat['fcount'] // 3
        if mat['name'] in mats:
            tris.append(F[at:at + n])
        at += n
    if not tris:
        raise SystemExit(f'bodyfield: none of {mats} in this model')
    idx = np.unique(np.concatenate(tris))
    names = [b['name'] for b in m.bones]
    V, B = [], []
    for i in idx:
        v = m.vertices[int(i)]
        V.append(v[0])
        k = int(np.argmax(v[6][:len(v[5])]))
        B.append(names[v[5][k]] if 0 <= v[5][k] < len(names) else '')
    return np.array(V), np.array(B)


class Field:
    def __init__(self, donor, host, donor_mats, host_mats, radius=RADIUS, scale=None):
        A, Ab = _points(donor, donor_mats)
        B, Bb = _points(host, host_mats)
        self.radius = radius
        if scale is None:
            s, t, d = align.icp_similarity(A, B, 1.0, B.mean(0) - A.mean(0), iters=60)
        else:
            # Scale is given, and only the offset is solved. Left free, a uniform ICP
            # cannot reconcile two bodies of the same height and different width, so
            # it splits the difference: inflates hers 12% to cover our figure and
            # drops the whole model 1.2 to keep the heads together, which put the
            # heels a unit underground and left the field carrying the error instead
            # of correcting it.
            s = float(scale)
            t = B.mean(0) - s * A.mean(0)
            for _ in range(40):
                d, idx = cKDTree(B).query(s * A + t)
                keep = d < max(np.percentile(d, 85), 1e-9)
                t = t + (B[idx[keep]] - (s * A[keep] + t)).mean(0)
            d, _ = cKDTree(B).query(s * A + t)
        self.scale, self.shift = s, t
        src = A * s + t

        # correspondence, bone by bone
        match = np.zeros_like(src)
        found = np.zeros(len(src), bool)
        whole = cKDTree(B)
        for bone in set(Ab) & set(Bb):
            here, there = np.where(Ab == bone)[0], np.where(Bb == bone)[0]
            if len(there) < 4:
                continue
            _, j = cKDTree(B[there]).query(src[here])
            match[here] = B[there[j]]
            found[here] = True
        loose = ~found
        if loose.any():
            _, j = whole.query(src[loose])
            match[loose] = B[j]
        # A rejected correspondence is dropped, not zeroed. Zeroing it -- graft.py's
        # `D[keep] = ...` over a zero array -- leaves an anchor that actively asserts
        # no correction, and the ones the trim rejects are the ones that disagree
        # most: here, the foot. She stands *in* her heels, ankle at y 1.31 against our
        # 0.99 and z 1.26 against our 0.52, so the foot is the largest honest
        # displacement in the model and the first thing a trim throws away.
        dist = np.linalg.norm(match - src, axis=1)
        keep = dist < np.percentile(dist, TRIM)
        src, D = src[keep], match[keep] - src[keep]

        self.anchors, self.D = src, D
        self.tree = cKDTree(src)
        n = np.linalg.norm(D, axis=1)
        print(f'body field: scale {s:.4f}, alignment residual median {np.median(d):.3f}; '
              f'{len(src)} anchors kept of {len(A)}, {int(found.sum())} matched within '
              f'their own bone, {int(loose.sum())} loose; correction median '
              f'{np.median(n):.3f} max {n.max():.3f}')

    def _raw(self, P, k):
        moved = P * self.scale + self.shift
        d, j = self.tree.query(moved, k=min(k, len(self.anchors)))
        w = np.exp(-(d / self.radius) ** 2)
        w /= w.sum(1, keepdims=True) + 1e-12
        return moved + (self.D[j] * w[..., None]).sum(1)

    def __call__(self, P, k=24):
        """Carry points from her frame into ours.

        Symmetrised about x=0, because both bodies are and the map between them has
        to be. Sampled raw it is not, quite: the anchors are two independently
        triangulated surfaces and the nearest 24 to a point on the left are not the
        mirror of the nearest 24 to its partner on the right. That reaches the rig --
        a left strap and a right strap took different local scales and their rigid
        bodies came out different sizes, which is not a small asymmetry in a model
        whose physics the viewer actually runs.
        """
        P = np.atleast_2d(np.asarray(P, float))
        flip = np.array([-1.0, 1.0, 1.0])
        return 0.5 * (self._raw(P, k) + self._raw(P * flip, k) * flip)


class BoneFrame:
    """Place a rigid piece by the bones it sits on, one side of the body at a time.

    A shoe is not cloth and its shape is not a shape of the body: it is built around
    a foot held at an angle, and carried through a smooth body field it flattens into
    a slipper, because a 30-degree rotation over a foot two units long is not a low
    frequency. Her foot and ours are held at nearly the same angle -- 33 degrees below
    horizontal against our 29.5 -- and differ in length, hers 1.92 against our 1.56.
    So the piece is fitted rigidly to the frame those bones make and keeps its heel.
    """

    def __init__(self, donor, host, bones):
        dp = {b['name']: np.array(b['pos']) for b in donor.bones}
        tp = {b['name']: np.array(b['pos']) for b in host.bones}
        self.sides = []
        for side in ('左', '右'):
            names = [side + b for b in bones if side + b in dp and side + b in tp]
            if len(names) < 3:
                continue
            A = np.array([dp[n] for n in names])
            B = np.array([tp[n] for n in names])
            s, R, t = _umeyama(A, B)
            sign = 1.0 if side == '左' else -1.0
            self.sides.append((sign, s, R, t))
            print(f'  {side}: fitted on {len(names)} bones, scale {s:.3f}, '
                  f'residual {np.linalg.norm((s * (R @ A.T).T + t) - B, axis=1).max():.3f}')
        self.scale = float(np.mean([s for _, s, _, _ in self.sides])) if self.sides else 1.0

    def __call__(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        out = P * self.scale
        for sign, s, R, t in self.sides:
            m = np.sign(P[:, 0]) == sign
            if m.any():
                out[m] = s * (R @ P[m].T).T + t
        return out


def _umeyama(A, B):
    ca, cb = A.mean(0), B.mean(0)
    X, Y = A - ca, B - cb
    U, S, Vt = np.linalg.svd(X.T @ Y / len(A))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = (U @ D @ Vt).T
    var = (X ** 2).sum() / len(A)
    s = float((S * np.diag(D)).sum() / var) if var > 1e-12 else 1.0
    return s, R, cb - s * R @ ca


class Snug:
    """A field that also holds hanging cloth at the standoff it had from her.

    The field carries a piece to where her body went, which is right for cloth lying
    on the body and not enough for cloth hanging off it. Her hip panels stand 0.148
    off her behind; ours is flatter there, so carried across unchanged they stand
    0.187 off ours and up to 0.556, and the gap is the first thing anyone sees. So
    the standoff is measured on her, and re-established on us.

    This is `--conform`, which failed on its own because it re-seats on the nearest
    skin and in an A-pose the nearest skin to a hip is the forearm beside it. Here
    the correspondence is already the field's, gated by bone, and only points that
    were near her body in the first place are touched at all. The correction is
    blurred before it is applied, because moving a buried vertex while its neighbour
    stays put is what tears a seam.
    """

    def __init__(self, field, her, her_body, ours, our_body, reach=0.6, blur=0.35):
        from surface import Mesh
        self.field, self.reach, self.blur = field, reach, blur
        self.scale = field.scale
        self.her = Mesh(np.array([v[0] for v in her.vertices]), _tris(her, her_body))
        self.ours = Mesh(np.array([v[0] for v in ours.vertices]), _tris(ours, our_body))

    @staticmethod
    def _stand(mesh, P):
        d, t, w = mesh.closest(P)
        f = mesh.F[t]
        base = (mesh.V[f[:, 0]] * w[:, :1] + mesh.V[f[:, 1]] * w[:, 1:2]
                + mesh.V[f[:, 2]] * w[:, 2:3])
        return d, np.einsum('ij,ij->i', P - base, mesh.normal[t]), mesh.normal[t]

    def __call__(self, P):
        from surface import _blur
        P = np.atleast_2d(np.asarray(P, float))
        moved = self.field(P)
        near, hers, _ = self._stand(self.her, P)
        _, ours, n = self._stand(self.ours, moved)
        want = np.where(near < self.reach, hers - ours, 0.0)
        if len(moved) > 3:
            want = _blur(moved, want[:, None], self.blur)[:, 0]
        return moved + n * want[:, None]


def _tris(m, mats):
    F = np.array(m.faces).reshape(-1, 3)
    at, out = 0, []
    for mat in m.materials:
        n = mat['fcount'] // 3
        if mat['name'] in mats:
            out.append(F[at:at + n])
        at += n
    return np.concatenate(out)
