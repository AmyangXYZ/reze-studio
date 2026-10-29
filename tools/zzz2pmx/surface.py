"""Closest-point queries against a triangle mesh, and weight transfer over them."""
import numpy as np
from scipy.spatial import cKDTree


def closest_on_tri(p, a, b, c):
    """Closest point on triangles (a,b,c) to points p, all (N,3). Returns point, barycentric."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = (ab * ap).sum(1)
    d2 = (ac * ap).sum(1)
    bp = p - b
    d3 = (ab * bp).sum(1)
    d4 = (ac * bp).sum(1)
    cp = p - c
    d5 = (ab * cp).sum(1)
    d6 = (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc

    u = np.zeros(len(p))
    v = np.zeros(len(p))
    w = np.zeros(len(p))

    m = (d1 <= 0) & (d2 <= 0)
    u[m] = 1
    m2 = (d3 >= 0) & (d4 <= d3) & ~m
    v[m2] = 1
    m3 = (d6 >= 0) & (d5 <= d6) & ~m & ~m2
    w[m3] = 1
    done = m | m2 | m3

    e1 = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~done
    t = np.divide(d1, d1 - d3, out=np.zeros_like(d1), where=(d1 - d3) != 0)
    u[e1] = 1 - t[e1]
    v[e1] = t[e1]
    done |= e1

    e2 = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~done
    t = np.divide(d2, d2 - d6, out=np.zeros_like(d2), where=(d2 - d6) != 0)
    u[e2] = 1 - t[e2]
    w[e2] = t[e2]
    done |= e2

    e3 = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~done
    den = (d4 - d3) + (d5 - d6)
    t = np.divide(d4 - d3, den, out=np.zeros_like(den), where=den != 0)
    v[e3] = 1 - t[e3]
    w[e3] = t[e3]
    done |= e3

    inv = np.divide(1.0, denom, out=np.zeros_like(denom), where=denom != 0)
    u[~done] = (va * inv)[~done]
    v[~done] = (vb * inv)[~done]
    w[~done] = (vc * inv)[~done]

    return u[:, None] * a + v[:, None] * b + w[:, None] * c, np.stack([u, v, w], 1)


class Mesh:
    """A triangle mesh you can ask for the closest point to."""

    def __init__(self, verts, tris):
        self.V = np.asarray(verts, float)
        self.F = np.asarray(tris, int)
        a, b, c = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        self.tree = cKDTree((a + b + c) / 3.0)
        n = np.cross(b - a, c - a)
        self.normal = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)

    def closest(self, P, k=16, chunk=4096):
        """Returns (distance, triangle index, barycentric) for each query point."""
        P = np.asarray(P, float)
        a, b, c = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        d_out = np.zeros(len(P))
        t_out = np.zeros(len(P), int)
        w_out = np.zeros((len(P), 3))
        k = min(k, len(self.F))
        for s in range(0, len(P), chunk):
            e = min(s + chunk, len(P))
            q = P[s:e]
            _, cand = self.tree.query(q, k=k)
            cand = np.atleast_2d(cand)
            best = np.full(len(q), np.inf)
            for j in range(k):
                t = cand[:, j]
                pt, bary = closest_on_tri(q, a[t], b[t], c[t])
                d = np.linalg.norm(pt - q, axis=1)
                better = d < best
                best = np.where(better, d, best)
                t_out[s:e] = np.where(better, t, t_out[s:e])
                w_out[s:e] = np.where(better[:, None], bary, w_out[s:e])
            d_out[s:e] = best
        return d_out, t_out, w_out


def transfer_weights(mesh, src_weights, P, k=16):
    """Blend the donor mesh's per-vertex weight dicts onto query points."""
    d, tri, bary = mesh.closest(P, k=k)
    out = []
    for i in range(len(P)):
        acc = {}
        for corner in range(3):
            bw = bary[i, corner]
            if bw <= 0:
                continue
            for bone, w in src_weights[mesh.F[tri[i], corner]].items():
                acc[bone] = acc.get(bone, 0.0) + bw * w
        out.append(acc)
    return out, d


def push_out(mesh, P, offset=0.06, rounds=6, relax=0.7, blur=0.35, rigid=False):
    """Lift points that have sunk inside `mesh` back out to it, plus a standoff.

    A garment fitted from another body sits slightly inside this one wherever this
    body is the fuller of the two, and the skin shows through in exactly the band
    where the two disagree. Resolving it against the surface is what makes the fit
    independent of how well the warp guessed.

    The lift is blurred across the garment before it is applied. Moving each vertex
    on its own is what tears a seam: a vertex buried deep gets yanked out to the
    surface while the neighbour it shares an edge with does not move at all.
    """
    P = np.array(P, float)
    moved = np.zeros(len(P), bool)
    for _ in range(rounds):
        d, tri, bary = mesh.closest(P)
        f = mesh.F[tri]
        q = (bary[:, 0:1] * mesh.V[f[:, 0]] + bary[:, 1:2] * mesh.V[f[:, 1]]
             + bary[:, 2:3] * mesh.V[f[:, 2]])
        n = mesh.normal[tri]
        signed = ((P - q) * n).sum(1)
        inside = signed < offset
        if not inside.any():
            break
        delta = np.zeros_like(P)
        delta[inside] = (q[inside] + n[inside] * offset) - P[inside]
        if rigid:
            # A buckle and its rings are not cloth: resolve them by growing the whole
            # assembly off the neck rather than by moving the vertices that happen to
            # be buried, which flattens the rings into ovals.
            delta = _as_similarity(P, P + delta) - P
        elif blur:
            delta = _blur(P, delta, blur)
        P += relax * delta
        moved |= inside
    return P, moved


def _as_similarity(A, B):
    """Best uniform scale + translation taking A onto B, applied to A."""
    ca, cb = A.mean(0), B.mean(0)
    num = ((A - ca) * (B - cb)).sum()
    den = ((A - ca) ** 2).sum()
    s = float(num / den) if den > 1e-12 else 1.0
    return s * (A - ca) + cb


def _blur(P, values, radius, k=12):
    tree = cKDTree(P)
    k = min(k, len(P))
    d, j = tree.query(P, k=k)
    w = np.exp(-(d / radius) ** 2)
    w /= w.sum(1, keepdims=True) + 1e-12
    return (values[j] * w[..., None]).sum(1)
