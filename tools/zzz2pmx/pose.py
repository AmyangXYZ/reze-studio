"""Linear-blend-skin a PMX into a test pose, honouring rotation grants (付与).

The rip weights its legs to the D bones, which only turn because 足 grants its
rotation to them, so a pose test that ignores grants moves nothing below the hip.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import PMX


def _quat(axis, deg):
    a = np.deg2rad(deg) / 2
    v = np.array(axis, float)
    v = v / (np.linalg.norm(v) + 1e-12)
    return np.concatenate([[np.cos(a)], np.sin(a) * v])


def _qmul(p, q):
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qscale(q, r):
    """Rotate by r times the angle of q."""
    w = np.clip(q[0], -1, 1)
    ang = 2 * np.arccos(w)
    s = np.sqrt(max(1 - w * w, 0))
    if s < 1e-9:
        return np.array([1.0, 0, 0, 0])
    return _quat(q[1:] / s, np.rad2deg(ang) * r)


def _qmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _depth(parent, i):
    d = 0
    while parent[i] >= 0 and d < 200:
        i = parent[i]
        d += 1
    return d


def skin(m, rotations):
    """rotations: {bone name: (axis, degrees)}. Returns posed vertex positions."""
    names = [b['name'] for b in m.bones]
    pos = np.array([b['pos'] for b in m.bones])
    parent = [b['parent'] for b in m.bones]
    own = [np.array([1.0, 0, 0, 0]) for _ in m.bones]
    for n, (ax, dg) in rotations.items():
        own[names.index(n)] = _quat(ax, dg)

    order = sorted(range(len(m.bones)), key=lambda i: _depth(parent, i))
    local = [None] * len(m.bones)
    for i in order:
        q = own[i]
        b = m.bones[i]
        if (b['flag'] & 0x0100) and 'grant' in b:
            src, rate = b['grant']
            if 0 <= src < len(m.bones) and local[src] is not None:
                q = _qmul(_qscale(local[src], rate), q)
        local[i] = q

    M = [None] * len(m.bones)
    for i in order:
        R = _qmat(local[i])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos[i] - R @ pos[i]
        M[i] = T if parent[i] < 0 else M[parent[i]] @ T

    V = np.array([v[0] for v in m.vertices])
    out = np.zeros_like(V)
    for k, v in enumerate(m.vertices):
        p = np.append(V[k], 1.0)
        acc = np.zeros(3)
        for b, w in zip(v[5], v[6]):
            if w > 0:
                acc += w * (M[b] @ p)[:3]
        out[k] = acc
    return out
