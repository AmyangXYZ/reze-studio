# -*- coding: utf-8 -*-
"""Close an open boundary in a mesh by stitching its two sides together.

The body has a hole at the crotch -- a 52-vertex slot running front to back -- which
the underwear was standing in for. The underwear does not follow the body there, so it
pokes through whatever covers it; the honest fix is to close the body.

The hole is a slot, not a disc: its two long sides run alongside each other about 0.32
apart while its own edges are 0.19. Cutting it in half at its far ends and walking the
two chains together works down the front and fails at the back, where the sides are not
opposite each other and the walk ends up spanning 1.28 across the seat -- a flat sheet
where the body has a crease.

So the fill is grown from the shortest connection available instead. At each step the
ear whose diagonal is shortest is clipped away, which follows whatever the body is
doing: across the gap where the gap is narrow, along the crease where it is not.

No vertices are added. Every triangle is built from vertices the body already had, so
weights, UVs and morphs come along untouched, and each one is wound to agree with the
normals around it. Where the two chains meet at the ends they share a vertex, and a
triangle built there would be degenerate -- those steps advance without emitting one.

    python3 tools/zzz2pmx/stitchhole.py <pmx> <material> [--y 10.0,13.6]
"""
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pmx import PMX
from pmxw import write_pmx


def loops_of(V, tris):
    pos, rep = {}, {}
    for i in np.unique(tris):
        rep[int(i)] = pos.setdefault(tuple(np.round(V[i], 5)), int(i))
    cnt = defaultdict(int)
    for t in tris:
        a, b, c = rep[int(t[0])], rep[int(t[1])], rep[int(t[2])]
        for u, v in ((a, b), (b, c), (c, a)):
            cnt[(min(u, v), max(u, v))] += 1
    bound = [e for e, n in cnt.items() if n == 1]
    adj = defaultdict(list)
    for a, b in bound:
        adj[a].append(b)
        adj[b].append(a)
    seen, out = set(), []
    for s in adj:
        if s in seen:
            continue
        loop, cur, prev = [s], s, None
        seen.add(s)
        while True:
            nxt = [x for x in adj[cur] if x != prev and x not in seen]
            if not nxt:
                break
            cur, prev = nxt[0], cur
            loop.append(cur)
            seen.add(cur)
        out.append(loop)
    return out


def fill_loop(V, loop):
    """Clip the ear with the shortest diagonal, until the loop is a triangle."""
    ring = list(loop)
    tris = []
    while len(ring) > 3:
        n = len(ring)
        best, bi = None, None
        for k in range(n):
            a, b, c = ring[(k - 1) % n], ring[k], ring[(k + 1) % n]
            if a == c:
                continue
            d = np.linalg.norm(V[a] - V[c])
            area = np.linalg.norm(np.cross(V[b] - V[a], V[c] - V[a])) / 2
            if area < 1e-9:
                d = -1.0                       # a degenerate ear is always worth clipping
            if best is None or d < best:
                best, bi = d, k
        a, b, c = ring[(bi - 1) % len(ring)], ring[bi], ring[(bi + 1) % len(ring)]
        if not (a == b or b == c or a == c):
            if np.linalg.norm(np.cross(V[b] - V[a], V[c] - V[a])) / 2 > 1e-9:
                tris.append((a, b, c))
        ring.pop(bi)
    a, b, c = ring
    if not (a == b or b == c or a == c):
        if np.linalg.norm(np.cross(V[b] - V[a], V[c] - V[a])) / 2 > 1e-9:
            tris.append((a, b, c))
    return tris


def main(path, name, ylo=10.0, yhi=13.6):
    m = PMX(path)
    V = np.array([v[0] for v in m.vertices])
    N = np.array([v[1] for v in m.vertices])
    F = np.array(m.faces).reshape(-1, 3)
    at, mine, mat = 0, None, None
    for mm in m.materials:
        n = mm['fcount'] // 3
        if mm['name'] == name:
            mine, mat, start = F[at:at + n], mm, at
        at += n
    if mine is None:
        raise SystemExit(f'no material {name}')
    if start + len(mine) != len(F):
        raise SystemExit(f'{name} is not the last material; stitching would land in another')

    added = []
    for loop in loops_of(V, mine):
        P = V[np.array(loop)]
        if len(loop) < 4 or not (ylo < P[:, 1].mean() < yhi):
            continue
        tris = fill_loop(V, loop)
        # wind each new triangle to agree with the normals it meets
        for a, b, c in tris:
            fn = np.cross(V[b] - V[a], V[c] - V[a])
            if fn @ (N[a] + N[b] + N[c]) < 0:
                b, c = c, b
            added.append((a, b, c))
        print(f'  closed a {len(loop)}-vertex hole at y {P[:,1].mean():.2f} '
              f'with {len(tris)} triangles')

    if not added:
        print('nothing to close')
        return
    flat = list(m.faces)
    for a, b, c in added:
        flat += [a, b, c]
    m.faces = flat
    mat['fcount'] += len(added) * 3

    # normals along the seam now have new faces to average in
    tris = np.array([list(t) for t in added])
    acc = np.zeros_like(V)
    allt = np.concatenate([mine, tris])
    fn = np.cross(V[allt[:, 1]] - V[allt[:, 0]], V[allt[:, 2]] - V[allt[:, 0]])
    for col in range(3):
        np.add.at(acc, allt[:, col], fn)
    touched = np.unique(tris)
    for i in touched:
        ln = np.linalg.norm(acc[i])
        if ln > 1e-12:
            v = list(m.vertices[int(i)])
            v[1] = [float(x) for x in acc[i] / ln]
            m.vertices[int(i)] = tuple(v)
    write_pmx(path, m)
    print(f'added {len(added)} triangles to {name}; {len(touched)} vertices re-normalled')


if __name__ == '__main__':
    a = sys.argv
    ys = a[a.index('--y') + 1].split(',') if '--y' in a else ['10.0', '13.6']
    main(a[1], a[2], float(ys[0]), float(ys[1]))
