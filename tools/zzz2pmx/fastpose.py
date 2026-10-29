# -*- coding: utf-8 -*-
"""Linear-blend skinning that can be called in a loop.

`pose.skin` walks every vertex in Python, which is fine to check a pose and far too
slow to solve against a fan of them -- a solver asking for nine poses over twenty
rounds spends its whole budget in that loop. The weights are constant, so they are
gathered once into arrays and each pose becomes four matrix products.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose as slowpose


class Skinner:
    def __init__(self, m):
        self.m = m
        n = len(m.vertices)
        self.B = np.zeros((n, 4), int)
        self.W = np.zeros((n, 4))
        for i, v in enumerate(m.vertices):
            b, w = v[5], v[6]
            for k in range(min(4, len(b))):
                self.B[i, k] = b[k]
                self.W[i, k] = w[k] if k < len(w) else 0.0
        s = self.W.sum(1, keepdims=True)
        self.W /= np.where(s > 1e-9, s, 1.0)
        self.rest = np.array([v[0] for v in m.vertices])

    def matrices(self, rotations):
        """The same bone transforms `pose.skin` builds, honouring rotation grants."""
        names = [b['name'] for b in self.m.bones]
        pos = np.array([b['pos'] for b in self.m.bones])
        parent = [b['parent'] for b in self.m.bones]
        own = [np.array([1.0, 0, 0, 0]) for _ in self.m.bones]
        for nm, (ax, dg) in rotations.items():
            own[names.index(nm)] = slowpose._quat(ax, dg)
        order = sorted(range(len(self.m.bones)),
                       key=lambda i: slowpose._depth(parent, i))
        local = [None] * len(self.m.bones)
        for i in order:
            q = own[i]
            b = self.m.bones[i]
            if (b['flag'] & 0x0100) and 'grant' in b:
                src, rate = b['grant']
                if 0 <= src < len(self.m.bones) and local[src] is not None:
                    q = slowpose._qmul(slowpose._qscale(local[src], rate), q)
            local[i] = q
        M = np.zeros((len(self.m.bones), 4, 4))
        out = [None] * len(self.m.bones)
        for i in order:
            R = slowpose._qmat(local[i])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = pos[i] - R @ pos[i]
            out[i] = T if parent[i] < 0 else out[parent[i]] @ T
            M[i] = out[i]
        return M

    def skin(self, rotations, V=None):
        M = self.matrices(rotations)
        P = self.rest if V is None else V
        h = np.concatenate([P, np.ones((len(P), 1))], 1)
        acc = np.zeros((len(P), 3))
        for k in range(4):
            w = self.W[:, k]
            live = w > 0
            if not live.any():
                continue
            Mk = M[self.B[live, k]]
            acc[live] += w[live, None] * np.einsum('nij,nj->ni', Mk, h[live])[:, :3]
        return acc
