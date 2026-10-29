"""A breast built, not shrunk.

The donor's breast cannot be made small by deforming it. It has a hole under the
cup; the crease under it is tucked as a double fold; and its weights were painted
for a rig's small moves, not a standing change of size. Every pass that shrank it
spent itself patching one of those and exposed the next.

So the breast is cut out of the skin -- the whole bulge, out to where the chest is
chest again -- and built afresh in the opening: a dome of a chosen shape over the
chest wall, stitched to the skin round it ring by ring the way a hole is closed.
Its line from the shoulder down to the bust and on to the ribs is set by four
reaches and three exponents rather than inherited, and there is nothing under it.

The dome: from an apex, the surface reaches a chosen distance up, down, out and in,
the reach turning smoothly with the direction so no line through the apex is a
crease, and its height falls as (1 - r^m)^n on the way, m and n each turning with
the direction too. m is the shape between apex and foot: near 1 a straight slope
from a pointed apex, above 2 a full round shoulder that holds its height and then
drops. n is how the foot meets the chest: near 2 it lays down without a break, near
1 it comes in round and quick, below 1 it is a cut. A drop is straight and pointed
above, full and quick below. The chest under it is a
film relaxed across the opening to least bending, with the rim and the ring of
skin outside it both held: the film passes through the rim and leaves it at the
old skin's own slope, so the seam carries neither a step nor an angle. The dome
stands off that film along the chest's outward direction. A quadric floor misses
the chest by enough to read as a collar; a thin-plate spline through the rim
overshoots between rim points; a rim let move under the relaxation shrinks and
takes the chest with it. The held film does none of these.
"""
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from stitchhole import loops_of


def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def quadric(P, z):
    X = np.column_stack([np.ones(len(P)), P[:, 0], P[:, 1], P[:, 0] ** 2, P[:, 0] * P[:, 1],
                         P[:, 1] ** 2])
    c, *_ = np.linalg.lstsq(X, z, rcond=None)

    def f(Q):
        return (c[0] + c[1] * Q[:, 0] + c[2] * Q[:, 1] + c[3] * Q[:, 0] ** 2
                + c[4] * Q[:, 0] * Q[:, 1] + c[5] * Q[:, 1] ** 2)

    def outward(Q):
        gx = c[1] + 2 * c[3] * Q[:, 0] + c[4] * Q[:, 1]
        gy = c[2] + c[4] * Q[:, 0] + 2 * c[5] * Q[:, 1]
        n = np.column_stack([gx, gy, -np.ones(len(Q))])      # the front of the chest is -z
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    return f, outward


def round_the_clock(phi, at_out, at_up, at_in, at_down):
    """A value that is `at_out` looking outward, `at_up` looking up and so on, and
    turns smoothly between them -- flat across each of the four directions, so the
    lines through the apex carry no crease."""
    q = np.mod(phi, 2 * np.pi) / (np.pi / 2)             # 0..4, one unit per quadrant
    k = np.floor(q).astype(int) % 4
    t = q - np.floor(q)
    s = t * t * (3 - 2 * t)
    ring = np.array([at_out, at_up, at_in, at_down])
    return ring[k] * (1 - s) + ring[(k + 1) % 4] * s


def dome(P, apex, sgn, reach, shape, point):
    """Height of the designed surface, 0..1, at the wall-plane points P."""
    up, down, out, inn = reach
    n_up, n_down, n_side = shape
    m_up, m_down, m_side = point
    u = (P[:, 0] - apex[0]) * sgn                  # + is outward, whichever side
    v = P[:, 1] - apex[1]
    phi = np.arctan2(v, u)
    L = round_the_clock(phi, out, up, inn, down)
    n = round_the_clock(phi, n_side, n_up, n_side, n_down)
    m = round_the_clock(phi, m_side, m_up, m_side, m_down)
    rho = np.hypot(u, v) / L
    return np.where(rho < 1, np.maximum(0.0, 1 - rho ** m) ** n, 0.0)


def thinplate(A, b, lam=0.002):
    n = len(A)
    d = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=2)
    K = np.where(d > 1e-12, d ** 2 * np.log(np.maximum(d, 1e-12)), 0.0) + lam * np.eye(n)
    Pm = np.column_stack([np.ones(n), A])
    M = np.zeros((n + 3, n + 3))
    M[:n, :n] = K; M[:n, n:] = Pm; M[n:, :n] = Pm.T
    sol = np.linalg.lstsq(M, np.concatenate([b, np.zeros(3)]), rcond=None)[0]
    w, c = sol[:n], sol[n:]

    def f(Q):
        dd = np.linalg.norm(Q[:, None, :] - A[None, :, :], axis=2)
        U = np.where(dd > 1e-12, dd ** 2 * np.log(np.maximum(dd, 1e-12)), 0.0)
        return U @ w + c[0] + Q @ c[1:]

    def outward(Q, h=1e-3):
        gx = (f(Q + [h, 0]) - f(Q - [h, 0])) / (2 * h)
        gy = (f(Q + [0, h]) - f(Q - [0, h])) / (2 * h)
        nrm = np.column_stack([gx, gy, -np.ones(len(Q))])
        return nrm / np.linalg.norm(nrm, axis=1, keepdims=True)

    return f, outward


def build(H, V, bones, skin, cut=(1.15, 1.6), centre=(0.15, 0.35), rings=12,
          reach=(1.1, 0.65, 0.85, 0.5), height=1.0, apex=(0.15, -0.3), shape=(1.2, 1.6, 1.2),
          point=(2.4, 1.8, 1.7), weight_top=0.8, front=1.0):
    """Cut the breasts out of `skin` and build them again. `bones` maps the 胸 bone
    names to their positions, `skin` names the material. Returns the new positions
    and the skin triangles, and the vertices of the two new patches."""
    F = np.array(H.faces).reshape(-1, 3)
    at, st, start, skin_mat = 0, None, None, None
    for mat in H.materials:
        n = mat['fcount'] // 3
        if mat['name'] == skin:
            st, start, skin_mat = F[at:at + n], at, mat
        at += n
    end = start + len(st)
    names = [b['name'] for b in H.bones]
    Vlist = [list(v) for v in V]
    kept = np.ones(len(st), bool)
    patches, new_tris = [], []
    for name, b in bones.items():
        sgn = 1.0 if b[0] > 0 else -1.0
        kb = names.index(name)
        cx, cy = b[0] + centre[0] * sgn, b[1] + centre[1]
        rho = lambda Q: np.sqrt(((Q[:, 0] - cx) / cut[0]) ** 2 + ((Q[:, 1] - cy) / cut[1]) ** 2)
        inside = (rho(V) < 1) & (V[:, 0] * sgn > 0.03) & (V[:, 2] < b[2] + front)
        gone = inside[st].all(1)
        kept &= ~gone
        rest = st[kept]
        # the opening's rim, and the chest under the breast: the body's own surface
        # carried across the opening, through the rim and the skin round it
        loops = [l for l in loops_of(V, rest) if len(l) >= 20]
        loop = min(loops, key=lambda l: np.linalg.norm(V[l].mean(0)[:2] - [cx, cy]))
        loop = [int(i) for i in loop]
        N = len(loop)
        rim = V[loop]
        ring = np.unique(rest)
        r = rho(V[ring])
        band = ring[(r > 1.0) & (r < 1.7) & (V[ring, 0] * sgn > 0.03) & (V[ring, 2] < b[2] + front)]
        wall, outward = quadric(V[band][:, :2], V[band][:, 2])
        rim_dz = rim[:, 2] - wall(rim[:, :2])
        cen_uv = np.mean([H.vertices[i][2] for i in loop], axis=0)
        A = np.array([b[0] + apex[0] * sgn, b[1] + apex[1]])

        def weights_at(i, g):
            v = H.vertices[i]
            others = {bn: w for bn, w in zip(v[5], v[6]) if w > 0 and bn != kb}
            tot = sum(others.values()) or 1.0
            top = weight_top * g ** 0.7
            items = [(kb, top)] + [(bn, w / tot * (1 - top)) for bn, w in others.items()]
            items = sorted(items, key=lambda kv: -kv[1])[:4]
            s = sum(w for _, w in items) or 1.0
            return ([bn for bn, _ in items] + [0] * (4 - len(items)),
                    [w / s for _, w in items] + [0.0] * (4 - len(items)))

        def add(pos, uv, i_rim, g):
            bones_, ws = weights_at(i_rim, g)
            rec = ([float(x) for x in pos], [0.0, 0.0, -1.0], [float(uv[0]), float(uv[1])],
                   [[0.0] * 4 for _ in range(H.uv_add)], 2, bones_, ws,
                   float(H.vertices[i_rim][7]))
            H.vertices.append(rec)
            Vlist.append(list(pos))
            return len(H.vertices) - 1

        # The grid: rings stepping in from the rim to one centre vertex. Their
        # places in the plane come from relaxing the grid with the rim held -- the
        # rim of an opening cut along coarse triangles is not star-shaped about its
        # centre, and rings drawn straight in from it cross their own spokes.
        # The first ring sits a hair inside the rim and is held there with it: the
        # rim zigzags along the coarse triangles it was cut through, and a first
        # ring free to relax toward the smooth rings behind it crosses the zigzag.
        grid = [loop]
        steps = np.concatenate([[0.08], np.linspace(0.08, 1.0, rings)[1:-1]])
        for f in steps:
            ring_ids = []
            for j, i in enumerate(loop):
                uv = np.array(H.vertices[i][2]) * (1 - f) + cen_uv * f
                ring_ids.append(add(rim[j] * (1 - f) + np.array([cx, cy, rim[:, 2].mean()]) * f, uv, i, 0.0))
            grid.append(ring_ids)
        c_id = add([cx, cy, rim[:, 2].mean()], cen_uv, loop[0], 0.0)
        tris = []
        for k in range(rings - 1):
            a_, b_ = grid[k], grid[k + 1]
            for j in range(N):
                tris.append((a_[j], a_[(j + 1) % N], b_[j]))
                tris.append((b_[j], a_[(j + 1) % N], b_[(j + 1) % N]))
        for j in range(N):
            tris.append((grid[-1][j], grid[-1][(j + 1) % N], c_id))
        tris = np.array(tris)
        inner = np.array([i for ring in grid[1:] for i in ring] + [c_id])
        Vn = np.array(Vlist)
        nb = defaultdict(set)
        for t in np.vstack([tris, rest]):
            for a_, b_ in ((0, 1), (1, 2), (2, 0)):
                nb[int(t[a_])].add(int(t[b_])); nb[int(t[b_])].add(int(t[a_]))
        outer = sorted(set(j for i in loop for j in nb[i]) - set(loop) - set(int(x) for x in inner))
        free = [int(i) for i in inner]
        held = set(loop) | set(outer)
        film = Vn.copy()
        for _ in range(80):                                     # a soap film first
            for i in free:
                film[i] = np.mean([film[j] for j in nb[i]], axis=0)
        involved = free + sorted(held)
        for _ in range(200):                                    # then least bending
            lap = {i: np.mean([film[j] for j in nb[i]], axis=0) - film[i] for i in involved}
            for i in free:
                film[i] += 0.5 * (lap[i] - np.mean([lap[j] for j in nb[i] if j in lap], axis=0))
        g = dome(film[inner][:, :2], A, sgn, reach, shape, point)
        pos = film[inner] + (height * g)[:, None] * outward(film[inner][:, :2])
        for k, i in enumerate(inner):
            Vlist[i] = list(pos[k])
            v = list(H.vertices[i]); v[0] = [float(x) for x in pos[k]]
            v[5], v[6] = weights_at(loop[(k % N)] if i != c_id else loop[0], float(g[k]))
            H.vertices[i] = tuple(v)
        Vn = np.array(Vlist)
        nrm = np.cross(Vn[tris[:, 1]] - Vn[tris[:, 0]], Vn[tris[:, 2]] - Vn[tris[:, 0]])
        if (np.einsum('ij,ij->i', nrm, outward(Vn[tris].mean(1)[:, :2])) < 0).mean() > 0.5:
            tris = tris[:, ::-1]
        new_tris.append(tris)
        patches.append(np.unique(tris))
        print(f'  {name}: cut {int(gone.sum())} skin triangles inside a {cut[0]:.2f} x '
              f'{cut[1]:.2f} opening; rim of {N}; built {len(tris)} triangles over it, '
              f'apex {height:.2f} off the wall')
    st = np.concatenate([st[kept]] + new_tris)
    H.faces = np.concatenate([F[:start], st, F[end:]]).ravel().tolist()
    skin_mat['fcount'] = len(st) * 3
    V = np.array(Vlist)
    for i in range(len(V)):
        v = list(H.vertices[i]); v[0] = [float(x) for x in V[i]]; H.vertices[i] = tuple(v)
    return V, st, np.concatenate(patches)
