# -*- coding: utf-8 -*-
"""Fill the holes one model's costume left in its skin, out of another model's.

A 深空之眼 body is authored around what it is wearing. 海滩漫步's skin is cut away
under her swimsuit -- both breasts, the band below them, the waist and the ankles, 452
rim edges in 18 loops -- because nothing was ever going to see it. Undressed she is
full of holes.

She does not have to be patched out of thin air, though. The whole series is one base
body: fitted bone to bone, 海滩漫步, 塞勒涅, 沉醉旖旎 and 午夜沉沦 agree on scale to
within 1%, and where two of them both have skin they agree on where it is. So a hole
in one is filled with the surface another one kept, and the filling is the same body,
not an invention.

The fill is triangulated flat, down the axis the hole faces. Taking the loop into 3D
directly does not work -- a breast's rim runs z -2.45..-1.35, nowhere near a plane, so
clipping ears off it or shrinking it toward its own centre hands back triangles that
cross in space. Seen down its own axis the surface inside a rim IS a height field, so
a Delaunay triangulation there cannot fold, and lifting each point back to the vertex
it came from puts the donor's surface in the hole at the donor's own density.
"""
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import Delaunay, cKDTree

sys.path.insert(0, __file__.rsplit('/', 1)[0])
import bodyfield
from pmx import PMX
from surface import Mesh

WELD = 1e-4

# The bones every model in the series has, and puts in the same place.
BONES = ('上半身', '上半身2', '首', '頭', '左肩', '右肩', '左腕', '右腕',
         '左ひじ', '右ひじ', '下半身', '左足', '右足', '左ひざ', '右ひざ',
         '左足首', '右足首')


def slices(m):
    """Each material's triangles, in draw order."""
    F = np.array(m.faces).reshape(-1, 3)
    at, out = 0, {}
    for mat in m.materials:
        n = mat['fcount'] // 3
        out.setdefault(mat['name'], []).append(F[at:at + n])
        at += n
    return {k: np.concatenate(v) for k, v in out.items()}


def tris_of(m, mats):
    sl = slices(m)
    return np.concatenate([sl[k] for k in mats if k in sl])


def weld(V):
    _, inv = np.unique(np.round(V / WELD).astype(np.int64), axis=0,
                       return_inverse=True)
    return inv


def boundary_loops(V, tris):
    """The rim of a triangle set, as loops of vertices walked in fill order.

    Counted against welded positions, never against vertex numbers: a UV seam is two
    vertices at one point, so by number every seam edge reads as a hole -- 48 real rim
    edges on a chest came back as 112 that way.
    """
    w = weld(V)
    count, seen = Counter(), {}
    for a, b, c in tris:
        for p, q in ((a, b), (b, c), (c, a)):
            if w[p] == w[q]:
                continue
            k = (min(w[p], w[q]), max(w[p], w[q]))
            count[k] += 1
            seen.setdefault(k, (int(p), int(q)))
    # The rim edge reads p->q on the one face that owns it, so the hole reads q->p.
    step = {}
    for k, n in count.items():
        if n == 1:
            p, q = seen[k]
            step[int(w[q])] = (int(w[p]), int(q))
    out = []
    while step:
        at = next(iter(step))
        loop = []
        while at in step:
            nxt, i = step.pop(at)
            loop.append(i)
            at = nxt
        if len(loop) > 2:
            out.append(loop)
    return out


def rims(m, mats):
    """The rim of some of a model's materials."""
    return boundary_loops(np.array([v[0] for v in m.vertices]), tris_of(m, mats))


def fit(donor, host, mats, host_mats, near=None, reach=2.4, bones=BONES):
    """The donor's surface, brought onto the host's frame.

    Bone to bone first, which is most of it in this series, and then closed on the
    host's own skin around wherever the patch is going. Fitted globally the two agree
    to about a tenth; fitted again on the neighbourhood they agree to a hundredth,
    and a hundredth is the difference between a patch that lies flush and one with a
    visible step around it.
    """
    DV = np.array([v[0] for v in donor.vertices])
    T = tris_of(donor, mats)
    db = {b['name']: np.array(b['pos']) for b in donor.bones}
    hb = {b['name']: np.array(b['pos']) for b in host.bones}
    common = [n for n in bones if n in db and n in hb]
    s, R, t = bodyfield._umeyama(np.array([db[n] for n in common]),
                                 np.array([hb[n] for n in common]))
    P = s * (R @ DV.T).T + t
    HV = np.array([v[0] for v in host.vertices])
    HP = HV[np.unique(tris_of(host, host_mats))]
    if near is not None:
        HP = HP[np.linalg.norm(HP - near, axis=1) < reach]
    idx = np.unique(T)
    tree = cKDTree(HP)
    for _ in range(40):
        Q = P[idx]
        if near is not None:
            Q = Q[np.linalg.norm(Q - near, axis=1) < reach]
        if len(Q) < 20:
            break
        d, j = tree.query(Q)
        keep = d < max(np.percentile(d, 70), 1e-9)
        s2, R2, t2 = bodyfield._umeyama(Q[keep], tree.data[j[keep]])
        P = s2 * (R2 @ P.T).T + t2
        s *= s2
    Q = P[idx]
    if near is not None:
        Q = Q[np.linalg.norm(Q - near, axis=1) < reach]
    d, _ = tree.query(Q)
    return Mesh(P, T), float(s), float(np.median(d))


def _inside(poly, P):
    """Even-odd point-in-polygon, and the distance to the nearest edge."""
    a, b = poly, np.roll(poly, -1, 0)
    x, y = P[:, :1], P[:, 1:2]
    dy = np.where(np.abs(b[:, 1] - a[:, 1]) < 1e-12, 1e-12, b[:, 1] - a[:, 1])
    cross = ((a[:, 1] > y) != (b[:, 1] > y)) & (
        x < a[:, 0] + (b[:, 0] - a[:, 0]) * (y - a[:, 1]) / dy)
    ab = b - a
    t = np.clip(((P[:, None, :] - a) * ab).sum(-1)
                / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    d = np.linalg.norm(P[:, None, :] - (a + t[..., None] * ab), axis=-1).min(1)
    return cross.sum(1) % 2 == 1, d


def mix(m, ids, ws):
    """One vertex blended from several: position, normal, UV and skin."""
    vs = [m.vertices[int(i)] for i in ids]
    ws = np.asarray(ws, float)
    ws = ws / (ws.sum() or 1.0)
    w = {}
    for v, f in zip(vs, ws):
        for bi, ww in zip(v[5], v[6]):
            if ww > 0:
                w[bi] = w.get(bi, 0.0) + float(f) * float(ww)
    top = sorted(w.items(), key=lambda x: -x[1])[:4]
    tot = sum(x[1] for x in top) or 1.0
    bones = [x[0] for x in top] + [0] * (4 - len(top))
    wts = [x[1] / tot for x in top] + [0.0] * (4 - len(top))
    blend = lambda k: [float(np.dot(ws, [v[k][i] for v in vs]))
                       for i in range(len(vs[0][k]))]
    nor = np.array(blend(1))
    n = np.linalg.norm(nor)
    return (blend(0), list(nor / n) if n > 1e-9 else list(vs[0][1]), blend(2),
            [np.zeros(4) for _ in range(m.uv_add)], 2, bones, wts,
            float(np.dot(ws, [v[7] for v in vs])))


def patch(m, loop, ref, margin=0.05, behind=0.30, power=2.0, blend=0.45):
    """Fill one hole with the donor's surface, and report the triangles.

    The donor is brought onto the rim before it is laid in. The two bodies are the
    same body and still differ by about 0.08 across the chest, and 0.08 arriving in
    one edge is a step: the cap read as a crease all the way round where it met the
    skin. So the offset is measured at the rim, where the host's own vertices say
    exactly what it is, carried into the patch by inverse distance and faded out over
    `blend`. The middle of the dome stays hers; only the approach to the rim bends.

    UVs and weights are carried in from the rim the same way. There is nothing else
    to read them off: the atlas has no island under a hole, which is why every
    attempt to solve the cup's UVs from inside it collapsed -- the boundary is a slit
    enclosing no area.
    """
    V = np.array([v[0] for v in m.vertices])
    P = V[loop]
    c = P.mean(0)
    axis = sum(np.array(m.vertices[int(i)][1], float) for i in loop)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return []
    axis /= n
    e1 = np.cross(axis, [0.0, 1.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(axis, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    flat = lambda Q: np.stack([(Q - c) @ e1, (Q - c) @ e2], 1)
    poly = flat(P)
    hers = np.unique(ref.F)
    keep, d = _inside(poly, flat(ref.V[hers]))
    keep &= d > margin
    keep &= (ref.V[hers] - c) @ axis > -behind
    inner = hers[keep]
    # what the rim says about where her surface should have been
    dd, tt, bb = ref.closest(P)
    f = ref.F[tt]
    off = P - (ref.V[f[:, 0]] * bb[:, :1] + ref.V[f[:, 1]] * bb[:, 1:2]
               + ref.V[f[:, 2]] * bb[:, 2:3])
    pts = np.concatenate([poly, flat(ref.V[inner])])
    ids = list(loop)
    for k, i in enumerate(inner):
        gap = np.maximum(np.linalg.norm(poly - pts[len(loop) + k], axis=1), 1e-6)
        w = 1.0 / gap ** power
        m.vertices.append(mix(m, loop, w))
        j = len(m.vertices) - 1
        v = list(m.vertices[j])
        fade = min(float(gap.min()) / blend, 1.0)
        shift = (off * (w / w.sum())[:, None]).sum(0) * (1.0 - fade * fade * (3 - 2 * fade))
        v[0] = [float(x) for x in ref.V[i] + shift]
        v[1] = [float(x) for x in ref.normal[np.where(ref.F == i)[0][0]]]
        m.vertices[j] = tuple(v)
        ids.append(j)
    out = []
    if len(pts) < 3:
        return out
    good, _ = _inside(poly, pts[Delaunay(pts).simplices].mean(1))
    W = np.array([v[0] for v in m.vertices])
    for t3 in Delaunay(pts).simplices[good]:
        a, b, cc = (ids[int(x)] for x in t3)
        if np.dot(np.cross(W[b] - W[a], W[cc] - W[a]), axis) < 0:
            b, cc = cc, b
        out.append([a, b, cc])
    return out


def sheets(m, loop, ref, margin=0.05, behind=0.30, cell=0.12):
    """How many layers of donor surface lie inside the loop, seen down its axis.

    This is the assumption `patch` stands on, tested directly rather than guessed at.
    Flattened down the axis it faces, the surface inside a cap's rim is a height
    field -- one donor vertex over each patch of the plane -- so a Delaunay there
    cannot fold. Inside a waist it is a tube: the front of the body and the back both
    project into the same ring, two layers deep, and triangulating that laces one to
    the other straight through the torso. The waist came back as 1387 triangles of
    lacing.

    Nothing cheaper separates the two. A breast cup's rim sits 0.485 from any surface
    because it is a deep hole, the same as a waist, and a cup's normals fan out over
    the dome as widely as a waist's do around the body.
    """
    V = np.array([v[0] for v in m.vertices])
    c, ax = loop_frame(V, loop)
    P = V[loop]
    if float(np.dot(sum(np.array(m.vertices[int(i)][1], float) for i in loop), ax)) < 0:
        ax = -ax
    e1 = np.cross(ax, [0.0, 1.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(ax, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    flat = lambda Q: np.stack([(Q - c) @ e1, (Q - c) @ e2], 1)
    poly = flat(P)
    hers = np.unique(ref.F)
    keep, d = _inside(poly, flat(ref.V[hers]))
    keep &= d > margin
    keep &= (ref.V[hers] - c) @ ax > -behind
    inner = ref.V[hers[keep]]
    if len(inner) < 4:
        return 0.0, 0
    key = np.round(flat(inner) / cell).astype(np.int64)
    return len(inner) / len(np.unique(key, axis=0)), len(inner)


def _bridge(A, B, V):
    """Triangles sewing one rim to the next, whatever their lengths.

    Advances down whichever side gives the shorter diagonal, so a 33-vertex rim meets
    a 27-vertex one with nothing long and thin between them.
    """
    na, nb, i, j, out = len(A), len(B), 0, 0, []
    d = lambda p, q: float(np.linalg.norm(V[p] - V[q]))
    while i < na or j < nb:
        if j >= nb or (i < na and d(A[(i + 1) % na], B[j % nb])
                       <= d(A[i % na], B[(j + 1) % nb])):
            out.append([A[i % na], A[(i + 1) % na], B[j % nb]])
            i += 1
        else:
            out.append([A[i % na], B[(j + 1) % nb], B[j % nb]])
            j += 1
    return out


def renormal(m, tris, ids):
    """Rebuild normals off the surface, accumulated against welded positions.

    Given only its own faces, each side of a UV seam leans a different way and the
    seam comes back as a crease down the middle of the chest.
    """
    V = np.array([v[0] for v in m.vertices])
    w = weld(V)
    acc = np.zeros((w.max() + 1, 3))
    P = V[tris]
    fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    for k in range(3):
        np.add.at(acc, w[tris[:, k]], fn)
    n = np.linalg.norm(acc, axis=1, keepdims=True)
    acc = np.where(n > 1e-12, acc / np.maximum(n, 1e-12), 0.0)
    done = 0
    for i in ids:
        new = acc[w[int(i)]]
        if not new.any():
            continue
        v = list(m.vertices[int(i)])
        v[1] = [float(x) for x in new]
        m.vertices[int(i)] = tuple(v)
        done += 1
    return done


def loop_frame(V, loop):
    """A loop's centre and the axis it lies across."""
    P = V[loop]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[2]



def transplant(m, donor, ref, take, host_mats, limit=1.2):
    """Put another model's part on this one, and sew it at the join.

    For a region too broken to repair. 海滩漫步's swimsuit bottom cuts her hip open
    behind three rims that pair with nothing and leaves a leg hole either side of the
    crotch; every attempt to close those from a donor left more boundary than it
    shut. 沉醉旖旎 wears hosiery from the waist down in one unbroken surface, so the
    whole lower body comes across instead and there is one ring to sew rather than a
    bikini's outline.

    The part keeps its own atlas, so it arrives as its own material. Weights are
    remapped by bone name, which works because the series shares a skeleton.
    """
    from scipy.spatial import cKDTree
    hb = {b['name']: i for i, b in enumerate(m.bones)}
    dn = [b['name'] for b in donor.bones]
    DUV = np.array([v[2] for v in donor.vertices])
    used = np.unique(np.concatenate(list(take.values())))
    new = {}
    for i in used:
        v = donor.vertices[int(i)]
        wd = {}
        for bi, ww in zip(v[5], v[6]):
            if ww > 1e-5 and 0 <= bi < len(dn) and dn[bi] in hb:
                wd[hb[dn[bi]]] = wd.get(hb[dn[bi]], 0.0) + float(ww)
        top = sorted(wd.items(), key=lambda x: -x[1])[:4] or [(0, 1.0)]
        tot = sum(x[1] for x in top) or 1.0
        m.vertices.append(
            ([float(x) for x in ref.V[int(i)]], list(v[1]),
             [float(x) for x in DUV[int(i)]],
             [np.zeros(4) for _ in range(m.uv_add)], 2,
             [x[0] for x in top] + [0] * (4 - len(top)),
             [x[1] / tot for x in top] + [0.0] * (4 - len(top)), v[7]))
        new[int(i)] = len(m.vertices) - 1
    out = {k: [[new[int(x)] for x in t] for t in v] for k, v in take.items()}
    owner = {}
    for k, v in out.items():
        for t in v:
            for x in t:
                owner[x] = k

    W = np.array([v[0] for v in m.vertices])
    theirs = boundary_loops(W, np.concatenate([np.array(v, int) for v in out.values()]))
    ours = boundary_loops(W, tris_of(m, host_mats))
    sewn = 0
    for loop in theirs:
        c = W[loop].mean(0)
        near = [l for l in ours if np.linalg.norm(W[l].mean(0) - c) < limit]
        if not near:
            continue
        host = min(near, key=lambda l: np.linalg.norm(W[l].mean(0) - c))
        ours.remove(host)
        # the host's rim needs UVs in the donor's atlas to cross the seam
        tree = cKDTree(W[loop])
        A = []
        for i in host:
            _, j = tree.query(W[int(i)][None])
            src = m.vertices[loop[int(j[0])]]
            v = list(m.vertices[int(i)])
            bs = list(v[5])[:4] + [0] * max(0, 4 - len(v[5]))
            ws = list(v[6])[:4] + [0.0] * max(0, 4 - len(v[6]))
            m.vertices.append((list(v[0]), list(v[1]), list(src[2]),
                               [np.zeros(4) for _ in range(m.uv_add)], 2,
                               bs, ws, v[7]))
            A.append(len(m.vertices) - 1)
        B = list(loop)
        V2 = np.array([v[0] for v in m.vertices])
        ax = W[loop].mean(0) - W[host].mean(0)
        n = np.linalg.norm(ax)
        ax = ax / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(ax, [0.0, 1.0, 0.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(ax, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax, e1)
        turn = lambda L, ctr: np.sum(np.diff(np.unwrap(np.arctan2(
            (V2[L] - ctr) @ e2, (V2[L] - ctr) @ e1))))
        if turn(A, V2[A].mean(0)) * turn(B, V2[B].mean(0)) < 0:
            B.reverse()
        k0 = int(np.argmin(np.linalg.norm(V2[B] - V2[A[0]], axis=1)))
        B = B[k0:] + B[:k0]
        for t3 in _bridge(A, B, V2):
            p, q, r = t3
            want = sum(np.array(m.vertices[int(x)][1], float) for x in t3)
            if float(np.dot(np.cross(V2[q] - V2[p], V2[r] - V2[p]), want)) < 0:
                q, r = r, q
            out[owner[B[0]]].append([p, q, r])
            sewn += 1
    return {k: np.array(v, int) for k, v in out.items()}, sewn
