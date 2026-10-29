# -*- coding: utf-8 -*-
"""Give the grafted body reze's chest, and carry the top with it.

`headgraft.py` leaves the host's figure as its author made it. 午夜沉沦's bust is a
size up on reze's -- the breast region is 3.09 wide and 1.38 deep to reze's 2.57 and
1.12 -- so this brings it down to reze's line, and nothing else.

The two bodies are the same size. Ribcage width under the bust is 2.166 against
2.165, 上半身2 to 首 is 0.998 of reze's. So reze's chest is carried across at scale 1,
上半身2 on 上半身2. A similarity fitted over the torso bones came out at 1.05 -- pulled
by where 上半身 sits, a bone placement and not a body size -- and put reze's chest back
5% larger before it was ever used as the target.

Each chest vertex is then carried radially: the ray from its breast's centre through
it is cast at reze's surface, and the vertex goes to the hit. Radial, not along the
vertex normal. A normal-wise move fails exactly at the fold under the breast -- an
underside vertex's normal points down and its nearest point is on the ribcage, so it
stays while the front collapses past it, and the old underside is left standing as a
shelf; and normals of a surface that is not a sphere shear the mesh sideways, which
is the twist. A radial map is monotone along every ray and moves nothing sideways.
reze's surface for this is her skin plus her bra cups: the skin is cut away under
them, and a ray into the hole finds only its rim. What can go wrong is done per
vertex, so it is not done per vertex:

- what is blurred is the ratio of the two radii, a scalar, and lightly -- and
  because a blurred move undershoots, it is applied in passes, re-cast each time;
- it is windowed to zero over the ribcage and does not reach the back, so there is no
  step at its edge and nothing off the breast is touched;
- it is mirrored about x=0, so the two sides are one shape;
- the spacing is restored afterwards. A shrink piles vertices up where the surface
  folds -- under the breast, edges came down to 0.29 of their length -- so the edges
  are relaxed as springs toward their original lengths, along the surface, on a graph
  welded across UV seams, and the surface is then landed on reze's again. The shape
  does not change; the texture stops pinching;
- the top, the sheer layer, the chains and the gem sample the same field, wide enough
  (0.45) that a cloth vertex takes the move of the skin beneath it and keeps its
  clearance -- so the cut stays with the body and nothing has to be pushed out
  afterwards. Pushing it out was tried and lifted moved vertices against a rim of
  unmoved ones; wrapping it to the skin triangle underneath broke in the cleavage,
  where one chain vertex rides the left breast and its neighbour the right;
- the 胸 bones and their spheres go through it too, the spheres shrinking with the
  volume, so the jiggle matches what is there.

Normals are rebuilt last, with UV-seam duplicates welded first. A seam puts two
vertices at one position, and a normal accumulated per index shades each from its own
side only -- the crease that showed on the earlier hand shrink.

`cloth01_alpha1`, the sheer overlay on the top, is removed.

    python3 tools/zzz2pmx/bustfit.py [--out <pmx>]
"""
import argparse
import copy
import os
import sys

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import breastform
import dropmaterial
from graftlegs import material_slice
from pmx import PMX
from pmxw import write_pmx
from surface import Mesh, _blur

HOST = 'public/models/reze-midnight/reze-midnight.pmx'
REF = 'public/models/reze/reze.pmx'
OUT = 'public/models/reze-midnight/reze-midnight-slim.pmx'

ANCHOR = '上半身2'                  # the bust band is measured from here
BREAST = ['左胸', '右胸']
SKIN = 'skin'
REF_SURFACE = ['skin', 'cloth01']   # reze's skin is cut away under her bra; the cups are the breast there
DROP = ['cloth01_alpha1']
WRAP = ['cloth01', 'cloth01_alpha2']   # the top and its sheer layer keep their standoff from the breast
SUBDIV = {'cloth01': 2, 'cloth01_alpha2': 1}   # Loop levels: the bra cannot be smooth at 970 triangles
SHEET = ('public/models/reze/textures/cloth01_skin.png', 'cloth01_reze.png')

FRONT_Y = (-1.7, 2.5)              # the front torso that is copied, about 上半身2: under the bust to the collarbone
FRONT_Z = 0.5                      # forward of this (about 上半身2) is the front
FRONT_X = 2.0                      # to the sides, short of the arms
CUP_STANDOFF = (0.028, 0.06)       # a cup hugs the breast: its standoff held in this band; the lower bound is the floor the blurred lift aims for
CUP_MAX = 0.22                     # a vertex further off than this is frill or strap, and keeps its own
WALL_CLEAR = 1.25                  # skin further than this from a 胸 bone is chest wall, not breast
WALL_RING = 2.1                    # and the wall is fitted from the ring out to here
CAP_R = (0.95, 1.90)               # the cap that scales, by distance from the bust point in the wall
BUMP_FADE = (0.03, 0.14)           # a vertex is on the bump once it stands this far off the wall
DOME_BAND = 0.55                   # the skin this far round the hole shapes the dome that closes it
DOME_CUP = 1.10                    # and so does the cup that lay on it, out to here
DOME_EASE = 0.30                   # how wide the dome is let down where it stands proud of the cup
DOME_SMOOTH = 0.004                # how much the dome may miss the points it is fitted through
IRON_REACH = 0.35                  # the skin this far outside the patch is ironed in with it
IRON_ROUNDS = 8                    # passes of it
SOFTEN_ROUNDS = 8                  # passes that take the edge off a fold
SOFTEN_FOLD = (40.0, 80.0)         # only a fold this sharp: a gentler gate fires over the
                                   # whole breast and sands the form off it
SPREAD_ROUNDS = 10                 # passes of sliding that even out the crowded crease
ROUND_ROUNDS = 4                   # passes of the low pass that rounds the breast
ROUND_RADIUS = 0.32                # and how wide each one reaches
AXIS_X = 0.40                      # the footprint draws in toward an axis at this |x|: the inner faces hold
OVER = {'cloth01_alpha2': 'cloth01'}   # what lies over what, and rides it
RIDE = 0.30                            # how far the layer over the bra reads the bra's own move
CUP_FLOOR = 0.03                   # the cup's standoff off the built breast
GROW = 0.05                            # how far the cup laps over the skin it was cut to
GROW_BAND = 0.14                       # and how far in from its rim that lapping fades
GROW_ON = (0.05, 0.20)                 # only where the rim runs along the cut skin
MOVED = (0.01, 0.05)                   # cloth is placed where the body under it moved this much
WRAP_NEAR = (0.16, 0.32)           # a fine vertex this far off the breast is frill or strap, not cup
WRAP_BRIDGE = (0.18, 0.45)         # and inside this |x| it is the centre piece, bridging the cleavage
CAGE_ROUNDS = 4                    # subdivide on a copy, lift the cage by the sag, again
CAGE_GAIN = 1.15                   # a little over, since the limit surface sags back
GATHER_R = (0.9, 1.5)              # gather: on the breast, by distance from its 胸 bone
GATHER_X = (0.45, 1.25)            # and most on the inner half: full inside |x| 0.45, none past 1.25
GATHER_MIN = 0.04                  # the inner faces stop this far short of the midline
BLUR = 0.18                        # of the radius ratio, over the anchors
RADIUS = 0.45                      # sampling the field off the anchors; under 0.4 the rim of the breast pinches the top
PASSES, DONE = 4, 0.01             # re-cast and go again until 90% of the radii are within this
RELAX = 15                         # tangential spring passes restoring the original spacing


def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def window(P, anchor):
    """1 over the whole front of the torso, feathered only at its rim.

    Not a window on the breast. Anything less than the whole front leaves a band
    where a vertex is part reze's and part hers -- a surface neither of them has,
    which read as a step under the breast. The bodies agree to 0.000 off the chest,
    so copying the ribcage, sternum and upper chest as well costs nothing and puts
    every vertex on one surface or the other."""
    y, z, x = P[:, 1] - anchor[1], P[:, 2] - anchor[2], np.abs(P[:, 0])
    return ((smoothstep(y, FRONT_Y[0] - 0.3, FRONT_Y[0]) * (1 - smoothstep(y, FRONT_Y[1], FRONT_Y[1] + 0.3)))
            * (1 - smoothstep(z, FRONT_Z - 0.25, FRONT_Z))
            * (1 - smoothstep(x, FRONT_X - 0.25, FRONT_X)))


def smooth_normals(V, tris):
    """Face normals accumulated onto positions rather than indices, so a UV seam
    is one surface."""
    fn = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
    acc = np.zeros_like(V)
    for c in range(3):
        np.add.at(acc, tris[:, c], fn)
    pos = {}
    for i in np.unique(tris):
        pos.setdefault(tuple(np.round(V[i], 4)), []).append(int(i))
    for group in pos.values():
        if len(group) > 1:
            n = sum(acc[i] for i in group)
            for i in group:
                acc[i] = n
    return acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)


def spring_relax(V, tris, moving, weight, before, total, iters):
    """Relax the edges of one surface as springs toward their original lengths,
    along the surface, for the vertices in `moving`.

    Springs on the original lengths, not a Laplacian: the spacing here was never
    even, and evening it stretches the short edges as far as it eases the pinched.
    The graph is welded by position first: a UV seam holds two vertices at one
    place, each seeing half the ring, and relaxed by index the seam pulls itself
    apart. Steps are mirrored about x=0 like everything else.
    """
    flip = np.array([-1.0, 1.0, 1.0])
    rep_of = {}
    rep = np.arange(len(V))
    for i in np.unique(tris):
        rep[i] = rep_of.setdefault(tuple(np.round(before[i], 4)), int(i))
    E = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    E = rep[E]
    E = np.unique(np.sort(E[E[:, 0] != E[:, 1]], axis=1), axis=0)
    L0 = np.linalg.norm(before[E[:, 0]] - before[E[:, 1]], axis=1)
    cnt = np.bincount(E.ravel(), minlength=len(V)).astype(float)
    rm = rep[moving]
    for _ in range(iters):
        n = smooth_normals(V, tris)[moving]
        d = V[E[:, 1]] - V[E[:, 0]]
        L1 = np.linalg.norm(d, axis=1)
        f = ((L1 - L0) / np.maximum(L1, 1e-9))[:, None] * d
        force = np.zeros_like(V)
        np.add.at(force, E[:, 0], f)
        np.add.at(force, E[:, 1], -f)
        force /= np.maximum(cnt, 1)[:, None]
        step = force[rm]
        step -= np.einsum('ij,ij->i', step, n)[:, None] * n
        step = 0.5 * step * weight[:, None]
        step = 0.5 * (step + step[mirror(V[moving])] * flip)
        V[moving] += step
        if total is not None:
            total += step


def origins(P, bones):
    """Where each vertex's ray starts: on the chest wall directly behind it, clamped
    to the span between the two breast centres. Continuous across the cleavage --
    the nearer bone is not, and a vertex sent toward the left centre beside a
    neighbour sent toward the right tears the sternum open."""
    b = np.array(bones)
    x = np.clip(P[:, 0], b[:, 0].min(), b[:, 0].max())
    return np.column_stack([x, np.full(len(P), b[:, 1].mean()), np.full(len(P), b[:, 2].mean())])


def subdivide(m, V, name, levels, carry=None):
    """Loop-subdivide one material in place, `levels` times.

    Positions are smoothed on a graph welded across UV seams, so the two sides of a
    seam get one surface; connectivity is split as the author had it, so each side
    keeps its own UVs and a seam stays a seam. Boundary edges (the cup's rim) take
    the boundary rule and stay a smooth curve; a rim corner of valence two is kept
    sharp. Weights interpolate and are re-cut to four bones. A second set of
    positions passed as `carry` -- the same cage before it was moved -- rides the
    same rules, so the fine mesh comes back knowing where it started. Returns the
    new V and the new faces of the material, and the carried positions with them.
    """
    from collections import defaultdict
    F = np.array(m.faces).reshape(-1, 3)
    mats, at, start = {}, 0, None
    for mat in m.materials:
        n = mat['fcount'] // 3
        if mat['name'] == name:
            start, tris = at, F[at:at + n].copy()
        at += n
    verts = list(m.vertices)
    for i in range(min(len(verts), len(V))):       # the positions handed in, not the ones on file
        v = list(verts[i]); v[0] = [float(x) for x in V[i]]; verts[i] = tuple(v)
    C = None if carry is None else [np.asarray(c, float) for c in carry]
    for _ in range(levels):
        Vn = np.array([v[0] for v in verts])
        Cn = None if C is None else np.array(C)
        rep_of, rep = {}, {}
        for i in np.unique(tris):
            rep[int(i)] = rep_of.setdefault(tuple(np.round(Vn[i], 4)), int(i))
        wt = np.vectorize(rep.get)(tris)                       # welded triangles
        # welded adjacency and boundary edges
        nb, ecount = defaultdict(set), defaultdict(int)
        for t in wt:
            for a_, b_ in ((0, 1), (1, 2), (2, 0)):
                u, v = int(t[a_]), int(t[b_])
                nb[u].add(v); nb[v].add(u); ecount[(min(u, v), max(u, v))] += 1
        boundary = set(e for e, c in ecount.items() if c == 1)
        bnb = defaultdict(set)
        for u, v in boundary:
            bnb[u].add(v); bnb[v].add(u)
        # even (existing) vertices, on welded ids
        even, even_c = {}, {}
        for r in set(rep.values()):
            if r in bnb:
                # a boundary vertex stays where the author put it. Smoothing the
                # rim pulls a strip's ends off whatever they touched -- the straps
                # round the back came apart at every join -- and the cup's rim is
                # placed by the wrap afterwards regardless.
                even[r] = Vn[r].copy()
                if Cn is not None:
                    even_c[r] = Cn[r].copy()
            else:
                k = len(nb[r])
                beta = (3.0 / 16.0) if k == 3 else (3.0 / (8.0 * k))
                even[r] = (1 - k * beta) * Vn[r] + beta * sum(Vn[j] for j in nb[r])
                if Cn is not None:
                    even_c[r] = (1 - k * beta) * Cn[r] + beta * sum(Cn[j] for j in nb[r])
        # odd (edge) positions, on welded edges
        opp = defaultdict(list)
        for t in wt:
            for a_, b_, c_ in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                u, v = int(t[a_]), int(t[b_])
                opp[(min(u, v), max(u, v))].append(int(t[c_]))
        odd, odd_c = {}, {}
        for e, cs in opp.items():
            if e in boundary or len(cs) < 2:
                odd[e] = 0.5 * (Vn[e[0]] + Vn[e[1]])
                if Cn is not None:
                    odd_c[e] = 0.5 * (Cn[e[0]] + Cn[e[1]])
            else:
                odd[e] = 0.375 * (Vn[e[0]] + Vn[e[1]]) + 0.125 * (Vn[cs[0]] + Vn[cs[1]])
                if Cn is not None:
                    odd_c[e] = 0.375 * (Cn[e[0]] + Cn[e[1]]) + 0.125 * (Cn[cs[0]] + Cn[cs[1]])
        # commit the even moves to every duplicate of a welded vertex
        for i in np.unique(tris):
            v = list(verts[i]); v[0] = [float(x) for x in even[rep[int(i)]]]; verts[i] = tuple(v)
            if C is not None:
                C[int(i)] = even_c[rep[int(i)]]
        # new vertices on unwelded edges, one per side of a seam
        made = {}
        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key in made:
                return made[key]
            va, vb = verts[a], verts[b]
            pos = odd[(min(rep[a], rep[b]), max(rep[a], rep[b]))]
            uv = [0.5 * (va[2][0] + vb[2][0]), 0.5 * (va[2][1] + vb[2][1])]
            acc = {}
            for vv in (va, vb):
                for bn, w in zip(vv[5], vv[6]):
                    if w > 0:
                        acc[bn] = acc.get(bn, 0.0) + 0.5 * w
            items = sorted(acc.items(), key=lambda kv: -kv[1])[:4]
            tot = sum(w for _, w in items) or 1.0
            bones = [b for b, _ in items] + [0] * (4 - len(items))
            ws = [w / tot for _, w in items] + [0.0] * (4 - len(items))
            rec = ([float(x) for x in pos], [0.0, 0.0, 1.0], uv, [[0.0] * 4 for _ in range(m.uv_add)],
                   2, bones, ws, 0.5 * (va[7] + vb[7]))
            made[key] = len(verts)
            verts.append(rec)
            if C is not None:
                C.append(odd_c[(min(rep[a], rep[b]), max(rep[a], rep[b]))])
            return made[key]
        out = []
        for a, b, c in tris:
            a, b, c = int(a), int(b), int(c)
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            out += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        tris = np.array(out)
    n_old = sum(mm['fcount'] // 3 for mm in m.materials if mm['name'] == name)
    faces = np.concatenate([F[:start], tris, F[start + n_old:]])
    for mat in m.materials:
        if mat['name'] == name:
            mat['fcount'] = len(tris) * 3
    m.vertices = verts
    m.faces = faces.ravel().tolist()
    Vout = np.array([v[0] for v in verts])
    if C is None:
        return Vout, tris
    return Vout, tris, np.array(C)


def fill_breast(H, V, bones, cup_tris, rings=4, relax=30, cup_gap=0.03):
    """Close the skin under the cups.

    The host's breast is cut away where its cups sit -- two rim loops of 24
    vertices, one per breast, at the front of each -- and everything downstream
    was reading the rim as the surface: the cup wrapped onto a hole's edge, the bake
    shaded a valley bounded by rims, and the cup could not cover what was not there.

    Each loop is closed with a polar grid: the rim is ring 0, `rings` rings step in
    toward one centre vertex, quads between rings, a fan at the middle. No rim edge
    is split, so nothing T-junctions with the skin around it. The grid is relaxed
    to a film with the rim held, then lifted onto a dome: a quadric fitted through
    the skin all round the hole, so the front of the breast carries on the surface
    the author modelled. Taking the cup's own inner face instead -- what this did
    first -- prints the bikini on the breast: a flat triangle with a ridge at its
    rim, which is what the shading showed. The dome is held inside the cup, since
    the cup covered it. UVs follow the same polar interpolation into the island's
    hole; weights blend from the rim. Returns the skin triangles, fill included.
    """
    from stitchhole import loops_of, fill_loop
    F = np.array(H.faces).reshape(-1, 3)
    at, skin_end, st = 0, None, None
    for mat in H.materials:
        n = mat['fcount'] // 3
        if mat['name'] == SKIN:
            st, skin_end, skin_mat = F[at:at + n], at + n, mat
        at += n
    names = [b['name'] for b in H.bones]
    cup = Mesh(V, cup_tris)
    ncup = smooth_normals(V, cup_tris)
    new_tris, added = [], 0
    Vlist = [list(v) for v in V]
    for loop in loops_of(V, st):
        if not 12 <= len(loop) <= 60:
            continue
        c = V[loop].mean(0)
        if min(np.linalg.norm(c - b) for b in bones) > 1.4:
            continue
        loop = [int(i) for i in loop]
        N = len(loop)
        cen_uv = np.mean([H.vertices[i][2] for i in loop], axis=0)
        cen_pos = c.copy()
        # bones and weights at the centre: the rim's, merged and re-cut to four
        acc = {}
        for i in loop:
            v = H.vertices[i]
            for bn, w in zip(v[5], v[6]):
                if w > 0:
                    acc[bn] = acc.get(bn, 0.0) + w / N
        def cut4(acc):
            items = sorted(acc.items(), key=lambda kv: -kv[1])[:4]
            tot = sum(w for _, w in items) or 1.0
            return [b for b, _ in items] + [0] * (4 - len(items)), [w / tot for _, w in items] + [0.0] * (4 - len(items))
        def add(pos, uv, bw):
            bones_, ws = cut4(bw)
            rec = ([float(x) for x in pos], [0.0, 0.0, -1.0], [float(uv[0]), float(uv[1])],
                   [[0.0] * 4 for _ in range(H.uv_add)], 2, bones_, ws, float(H.vertices[loop[0]][7]))
            H.vertices.append(rec); Vlist.append(list(pos))
            return len(H.vertices) - 1
        grid = [loop]
        for r in range(1, rings):
            f = r / rings
            ring = []
            for i in loop:
                v = H.vertices[i]
                bw = {bn: w * (1 - f) for bn, w in zip(v[5], v[6]) if w > 0}
                for bn, w in acc.items():
                    bw[bn] = bw.get(bn, 0.0) + w * f
                ring.append(add(V[i] * (1 - f) + cen_pos * f, np.array(v[2]) * (1 - f) + cen_uv * f, bw))
            grid.append(ring)
        centre = add(cen_pos, cen_uv, dict(acc))
        tris = []
        for r in range(rings - 1):
            a_, b_ = grid[r], grid[r + 1]
            for k in range(N):
                tris.append((a_[k], a_[(k + 1) % N], b_[k]))
                tris.append((b_[k], a_[(k + 1) % N], b_[(k + 1) % N]))
        last = grid[-1]
        for k in range(N):
            tris.append((last[k], last[(k + 1) % N], centre))
        tris = np.array(tris)
        Vn = np.array(Vlist)
        # wind the patch to face out of the breast
        nrm = np.cross(Vn[tris[:, 1]] - Vn[tris[:, 0]], Vn[tris[:, 2]] - Vn[tris[:, 0]])
        bone = min(bones, key=lambda b: np.linalg.norm(c - b))
        if (np.einsum('ij,ij->i', nrm, Vn[tris].mean(1) - bone) < 0).mean() > 0.5:
            tris = tris[:, ::-1]
        inner = np.array([i for ring in grid[1:] for i in ring] + [centre])
        nb = {i: set() for i in inner}
        for t in tris:
            for u_, v_ in ((0, 1), (1, 2), (2, 0)):
                if int(t[u_]) in nb: nb[int(t[u_])].add(int(t[v_]))
                if int(t[v_]) in nb: nb[int(t[v_])].add(int(t[u_]))
        def settle(rounds):
            for _ in range(rounds):
                for i in inner:
                    Vn[i] = np.mean([Vn[j] for j in nb[i]], axis=0)
        settle(relax)                                        # a film across the rim
        # then onto the dome the skin round the hole describes
        n = c - bone
        n /= np.linalg.norm(n)
        u = np.cross(n, [0.0, 1.0, 0.0]); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        band = np.unique(st)
        band = band[cKDTree(Vn[loop]).query(Vn[band])[0] < DOME_BAND]
        R = Vn[band] - c
        A = np.column_stack([R @ u, R @ v])
        fit = thinplate(A, R @ n)
        def dome(Q):
            r = Q - c
            a, b_ = r @ u, r @ v
            hh = fit(np.column_stack([a, b_]))
            return c + a[:, None] * u + b_[:, None] * v + hh[:, None] * n
        # the whole patch takes the dome. Blending it with the relaxed film near the
        # rim leaves the first ring flat, and the ring where flat meets dome is a
        # crease across the line from the shoulder to the bust
        # The skin round the hole says how the surface leaves the rim; the cup says
        # where the breast was in the middle, since the author cut it to this breast.
        # Off its own cage the cup is eight triangles wide and prints its facets on
        # the skin, so it is read after subdivision, where it is smooth.
        depth = np.array([r / rings for r in range(1, rings) for _ in loop] + [1.0])
        d, t, w = cup.closest(Vn[inner])
        f = cup_tris[t]
        q = w[:, :1] * Vn[f[:, 0]] + w[:, 1:2] * Vn[f[:, 1]] + w[:, 2:] * Vn[f[:, 2]]
        m = w[:, :1] * ncup[f[:, 0]] + w[:, 1:2] * ncup[f[:, 1]] + w[:, 2:] * ncup[f[:, 2]]
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
        blend = smoothstep(depth, 0.2, 0.75)
        Vn[inner] = dome(Vn[inner]) * (1 - blend[:, None]) + (q - cup_gap * m) * blend[:, None]
        # and let down until it is inside the cup, which is what covered it. Pushing
        # only the vertices that stand proud puts the cup's facets back on the
        # breast; the whole dome comes down instead, by the excess blurred over it.
        held = 0
        for _ in range(6):
            d, t, w = cup.closest(Vn[inner])
            f = cup_tris[t]
            q = w[:, :1] * Vn[f[:, 0]] + w[:, 1:2] * Vn[f[:, 1]] + w[:, 2:] * Vn[f[:, 2]]
            m = w[:, :1] * ncup[f[:, 0]] + w[:, 1:2] * ncup[f[:, 1]] + w[:, 2:] * ncup[f[:, 2]]
            m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
            past = np.maximum(np.einsum('ij,ij->i', Vn[inner] - q, m) + cup_gap, 0.0)
            held = int((past > 1e-4).sum())
            if not held:
                break
            Vn[inner] -= (_blur(Vn[inner], past[:, None], DOME_EASE)[:, 0] * blend)[:, None] * n
        settle(2)
        # the patch is the front of the breast, so it belongs to the 胸 bone as much
        # as the fullest part of the rim does. Averaged from the rim instead, it
        # holds less of the bone than the skin around it and stands proud of it when
        # the breast is made smaller.
        kb = [names.index(n) for n in BREAST if n in names]
        top = max(sum(x for b_, x in zip(H.vertices[i][5], H.vertices[i][6]) if b_ in kb)
                  for i in loop)
        for k, i in enumerate(inner):
            v = list(H.vertices[i])
            own = [(b_, x) for b_, x in zip(v[5], v[6]) if x > 0]
            mine = sum(x for b_, x in own if b_ in kb)
            rest = [(b_, x) for b_, x in own if b_ not in kb]
            keep = max(1e-6, sum(x for _, x in rest))
            want = top * smoothstep(depth[k], 0.0, 0.6) + mine * (1 - smoothstep(depth[k], 0.0, 0.6))
            near = min(kb, key=lambda b_: abs(H.bones[b_]['pos'][0] - Vn[i][0]))
            pairs = [(near, want)] + [(b_, x * (1 - want) / keep) for b_, x in rest]
            pairs = sorted(pairs, key=lambda kv: -kv[1])[:4]
            tot = sum(x for _, x in pairs) or 1.0
            v[5] = [b_ for b_, _ in pairs] + [0] * (4 - len(pairs))
            v[6] = [x / tot for _, x in pairs] + [0.0] * (4 - len(pairs))
            v[4] = 2
            H.vertices[i] = tuple(v)
        for i in inner:
            Vlist[i] = list(Vn[i])
        new_tris.append(tris)
        added += len(inner)
        settle(2)
        rise = float(np.max((Vn[inner] - c) @ n))
        print(f'  filled a {N}-vertex hole at {np.round(c, 2)} with {len(inner)} vertices, '
              f'{len(tris)} triangles; the dome stands {rise:+.3f} past the rim, '
              f'{held} of them still proud of the cup')
    if not new_tris:
        return V, st, None
    new = np.concatenate(new_tris)
    H.faces = np.concatenate([F[:skin_end], new, F[skin_end:]]).ravel().tolist()
    skin_mat['fcount'] += len(new) * 3
    V = np.array(Vlist)

    # -- iron the join. However well the patch is placed it meets the skin at its own
    # angle, and a ring of skin either side of that join carries the difference:
    # a crease all the way round the breast, which is what the shading showed.
    st = np.concatenate([st, new])
    ring = np.unique(new)
    near = np.unique(st)
    near = near[cKDTree(V[ring]).query(V[near])[0] < IRON_REACH]
    w = (1 - smoothstep(cKDTree(V[ring]).query(V[near])[0], 0.0, IRON_REACH))
    nb = defaultdict(set)
    for t in st:
        for a_, b_ in ((0, 1), (1, 2), (2, 0)):
            nb[int(t[a_])].add(int(t[b_])); nb[int(t[b_])].add(int(t[a_]))
    g = welded(V[near]); axis = np.bincount(g, np.abs(V[near, 0]) < 1e-4, g.max() + 1) > 0
    flip = np.array([-1.0, 1.0, 1.0])
    for _ in range(IRON_ROUNDS):
        mid = np.array([np.mean(V[list(nb[int(i)])], axis=0) if nb[int(i)] else V[int(i)]
                        for i in near])
        d = (mid - V[near]) * (0.5 * w)[:, None]
        d = 0.5 * (d + d[mirror(V[near])] * flip)
        V[near] += together(d, g, axis)
    for i in near:
        Vlist[i] = list(V[i])
    for i in range(len(V)):
        v = list(H.vertices[i]); v[0] = [float(x) for x in V[i]]; H.vertices[i] = tuple(v)
    return V, st, new


def blur_r(P, vals, radius):
    """Gaussian blur over everything within `radius`, not over a fixed count of
    neighbours. The filled patch is four times denser than the skin round it, and a
    twelve-nearest blur there never reaches past the patch: the seam where the two
    meet stays as a ring in the shading however many passes it gets."""
    tree = cKDTree(P)
    out = np.empty(len(P))
    for i, near in enumerate(tree.query_ball_point(P, radius)):
        d = np.linalg.norm(P[near] - P[i], axis=1)
        w = np.exp(-(d / (0.6 * radius)) ** 2)
        out[i] = float(vals[near] @ w / w.sum())
    return out


def round_bust(P, h, onb, lift=0.0, rounds=ROUND_ROUNDS, radius=ROUND_RADIUS):
    """How much each vertex's height over the chest wall must change for the breast
    to be round: the height field low-passed over the breast, so every direction
    round the bust point falls away evenly.

    A breast scaled down keeps whatever unevenness its cage had, and the scaling
    magnifies it where the mesh crowds -- the lumps read as dents in the shading
    long before they show in the silhouette. `lift` reads the field from that far
    down the body, which carries the fullest part up and brings the crease up under
    it. Only the depth moves, so the footprint the author drew, and the UVs on it,
    are untouched. Forcing the height to one
    profile of the distance from the bust point rounds it too far: that throws away
    the fall the author gave it, fuller below than above, and crowds the mesh by a
    fifth of a unit doing it.
    """
    want = h.copy()
    for _ in range(rounds):
        want = np.where(onb > 0.05, blur_r(P, want, radius), want)
    if lift:
        # read from lower down by that much where the breast is fullest and not at
        # all where it is flat. A constant shift moves the flat chest as well, and
        # the two meet in a ridge across the line from the shoulder to the bust.
        q = P.copy()
        q[:, 1] -= lift * np.clip(want / max(want.max(), 1e-6), 0.0, 1.0)
        d, j = cKDTree(P).query(q, k=min(24, len(P)))
        w = np.exp(-(d / (0.6 * radius)) ** 2)
        want = (want[j] * w).sum(1) / (w.sum(1) + 1e-12)
    return (want - h) * onb



def thinplate(A, b, lam=DOME_SMOOTH):
    """A thin-plate spline through (A, b): the surface of least bending that passes
    through the points. Fitted to the skin round the hole it carries the slope of
    that skin inward, which a quadric cannot -- a quadric matches the band on
    average and meets it at an angle, and that angle is a ring in the shading right
    where the line from the shoulder crosses it."""
    n = len(A)
    d = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=2)
    K = np.where(d > 1e-12, d ** 2 * np.log(np.maximum(d, 1e-12)), 0.0) + lam * np.eye(n)
    P_ = np.column_stack([np.ones(n), A])
    M = np.zeros((n + 3, n + 3))
    M[:n, :n] = K; M[:n, n:] = P_; M[n:, :n] = P_.T
    rhs = np.concatenate([b, np.zeros(3)])
    sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
    w, c = sol[:n], sol[n:]
    def f(Q):
        dd = np.linalg.norm(Q[:, None, :] - A[None, :, :], axis=2)
        U = np.where(dd > 1e-12, dd ** 2 * np.log(np.maximum(dd, 1e-12)), 0.0)
        return U @ w + c[0] + Q @ c[1:]
    return f


def welded(P, tol=4):
    """A group number per vertex, coincident vertices sharing one.

    The body is mirrored down the sternum and both halves carry their own copy of
    the seam. Moved apart by a hair they leave a crack the length of the cleavage,
    so any pass that slides vertices about must move a seam as one vertex.
    """
    seen, g = {}, np.empty(len(P), int)
    for k, q in enumerate(np.round(P, tol)):
        g[k] = seen.setdefault(tuple(q), len(seen))
    return g


def together(d, g, on_axis):
    """One move per welded group, and none across the midline for a seam."""
    n = g.max() + 1
    m = np.column_stack([np.bincount(g, d[:, c], n) / np.bincount(g, None, n) for c in range(3)])
    m[on_axis, 0] = 0.0
    return m[g]


def spread(V, tris, idx, weight, rounds=SPREAD_ROUNDS, step=0.5):
    """Slide the vertices of a patch over their own surface until they sit evenly.

    Scaling a breast down crowds the mesh where it was deepest -- under the bust the
    edges come to a third of their length -- and a crowded ring of quads folds
    against itself: the crease turns from a valley into a crack, 107 degrees across
    two triangles. Sliding is not reshaping: each vertex moves along the surface
    toward the middle of its neighbours and is put back on the surface it started
    on, so the silhouette is untouched and only the spacing changes.
    """
    hold = Mesh(V.copy(), tris)
    nb = defaultdict(set)
    for t in tris:
        for a_, b_ in ((0, 1), (1, 2), (2, 0)):
            nb[int(t[a_])].add(int(t[b_])); nb[int(t[b_])].add(int(t[a_]))
    w = np.asarray(weight, float)
    flip = np.array([-1.0, 1.0, 1.0])
    g = welded(V[idx])
    axis = np.bincount(g, np.abs(V[idx, 0]) < 1e-4, g.max() + 1) > 0
    for _ in range(rounds):
        n = smooth_normals(V, tris)
        mid = np.array([np.mean(V[list(nb[int(i)])], axis=0) if nb[int(i)] else V[int(i)]
                        for i in idx])
        d = mid - V[idx]
        d -= np.einsum('ij,ij->i', d, n[idx])[:, None] * n[idx]      # along the surface only
        d = 0.5 * (d + d[mirror(V[idx])] * flip)
        d = together(d * w[:, None], g, axis)
        V[idx] += step * d
        _, t, bw = hold.closest(V[idx])                              # and back onto it
        f = hold.F[t]
        V[idx] = (bw[:, :1] * hold.V[f[:, 0]] + bw[:, 1:2] * hold.V[f[:, 1]]
                  + bw[:, 2:] * hold.V[f[:, 2]])
    return V


def soften(V, tris, idx, weight, rounds=SOFTEN_ROUNDS, band=SOFTEN_FOLD, step=0.5):
    """Round off the places where the surface folds against itself.

    The crease under the bust is a real edge, and the author drew it as one -- but
    across two triangles at a hundred degrees it shades as a crack rather than a
    curve. Only the vertices on such a fold move, and only toward the middle of
    their neighbours, so the crease keeps its line and loses its edge.
    """
    nb = defaultdict(set)
    for t in tris:
        for a_, b_ in ((0, 1), (1, 2), (2, 0)):
            nb[int(t[a_])].add(int(t[b_])); nb[int(t[b_])].add(int(t[a_]))
    edges = defaultdict(list)
    for fi, t in enumerate(tris):
        for a_, b_ in ((0, 1), (1, 2), (2, 0)):
            edges[(min(int(t[a_]), int(t[b_])), max(int(t[a_]), int(t[b_])))].append(fi)
    pairs = [(e, f) for e, f in edges.items() if len(f) == 2]
    flip = np.array([-1.0, 1.0, 1.0])
    w0 = np.asarray(weight, float)
    g = welded(V[idx])
    axis = np.bincount(g, np.abs(V[idx, 0]) < 1e-4, g.max() + 1) > 0
    for _ in range(rounds):
        fn = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
        fold = np.zeros(len(V))
        for (i, j), f in pairs:
            a = np.degrees(np.arccos(np.clip(fn[f[0]] @ fn[f[1]], -1.0, 1.0)))
            fold[i] = max(fold[i], a); fold[j] = max(fold[j], a)
        w = w0 * smoothstep(fold[idx], *band)
        if w.max() < 1e-3:
            break
        mid = np.array([np.mean(V[list(nb[int(i)])], axis=0) if nb[int(i)] else V[int(i)]
                        for i in idx])
        d = (mid - V[idx]) * w[:, None]
        d = 0.5 * (d + d[mirror(V[idx])] * flip)
        V[idx] += step * together(d, g, axis)
    return V


def tangent(P, f, m):
    """A unit tangent of each triangle, taken across the normal so that it turns
    with the surface rather than with the triangle's winding."""
    t = P[f[:, 1]] - P[f[:, 0]]
    t = t - np.einsum('ij,ij->i', t, m)[:, None] * m
    return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)


def transfer_fine(V, pre, ref_tris, idx, near, own=None, hole=None, grow=0.0,
                  mid=0.08, blur=0.10, rounds=2):
    """Put each cloth vertex back on the point of the body it was cut for.

    The wrap took the nearest point on the breast as it now is. On a breast this
    much smaller that point slides up and inward and the cup follows it: the rim
    ends above the crease with the breast bare under it, and the strap it joins --
    further from the skin, left to the field -- stays behind. Here the footprint is
    read on the body as it was -- the nearest point before the scaling, kept as a
    triangle and its barycentric weights -- and the vertex carries the move that
    point of the body made, holding the standoff it had there. So
    the cloth keeps its own shape and takes the body's: a cup that covered the
    breast covers it still, whatever the breast has become, and its rim comes down
    with the strap it is sewn to. Carrying the move rather than the place is what
    keeps it whole -- the nearest point jumps from one side of a crease to the
    other, and a vertex placed outright jumps with it.

    `grow` slides its free edge that much further out where that edge runs along `hole` --
    the skin the author cut away because this cloth covered it. Cut to the cup
    exactly, the rim leaves a thread of skin showing along it once the skin is back;
    this laps it over. Returns how many vertices were placed and how much each one
    took, so that what is left can be carried by them.
    """
    flip = np.array([-1.0, 1.0, 1.0])
    out = np.zeros((len(idx), 3))
    if grow and own is not None:
        edge, nb = defaultdict(int), defaultdict(set)
        for t in own:
            for a_, b_ in ((0, 1), (1, 2), (2, 0)):
                u, v = int(t[a_]), int(t[b_])
                edge[(min(u, v), max(u, v))] += 1; nb[u].add(v); nb[v].add(u)
        rim = np.array(sorted({v for e, c in edge.items() if c == 1 for v in e}))
        away = np.array([pre[r] - np.mean([pre[j] for j in nb[r]], axis=0) for r in rim])
        away /= np.linalg.norm(away, axis=1, keepdims=True) + 1e-12
        d, j = cKDTree(pre[rim]).query(pre[idx])
        lap = away[j] * (1 - smoothstep(d, 0.0, GROW_BAND))[:, None]
        if hole is not None:                    # only where the rim runs along the cut
            skinned = Mesh(pre, hole).closest(pre[idx])[0]
            lap *= (1 - smoothstep(skinned, *GROW_ON))[:, None]
        out = out + lap * grow
    n0 = smooth_normals(pre, ref_tris)
    n1 = smooth_normals(V, ref_tris)
    cx = pre[ref_tris].mean(1)[:, 0]
    n_w = 0
    for _ in range(rounds):
        target = V[idx].copy()
        wrapw = np.zeros(len(idx))
        for sgn in (-1.0, 1.0):
            sel = (np.sign(pre[idx, 0]) == sgn) & (np.abs(pre[idx, 0]) > mid)
            if not sel.any():
                continue
            side = idx[sel]
            tris = ref_tris[np.sign(cx) == sgn]
            was = Mesh(pre, tris)
            d, t, w = was.closest(pre[side])
            f = tris[t]
            p0 = w[:, :1] * pre[f[:, 0]] + w[:, 1:2] * pre[f[:, 1]] + w[:, 2:] * pre[f[:, 2]]
            m0 = w[:, :1] * n0[f[:, 0]] + w[:, 1:2] * n0[f[:, 1]] + w[:, 2:] * n0[f[:, 2]]
            m0 /= np.linalg.norm(m0, axis=1, keepdims=True) + 1e-12
            p1 = w[:, :1] * V[f[:, 0]] + w[:, 1:2] * V[f[:, 1]] + w[:, 2:] * V[f[:, 2]]
            m1 = w[:, :1] * n1[f[:, 0]] + w[:, 1:2] * n1[f[:, 1]] + w[:, 2:] * n1[f[:, 2]]
            m1 /= np.linalg.norm(m1, axis=1, keepdims=True) + 1e-12
            # the offset from that point of the body, read in the body's own frame
            # there and written back in the frame the body has now. A hem standing
            # 0.08 off a surface whose normal turns 50 degrees keeps its 0.08:
            # carried as a plain offset it folds flat against the layer beneath.
            u0, u1 = tangent(pre, f, m0), tangent(V, f, m1)
            r = pre[side] - p0
            local = np.column_stack([np.einsum('ij,ij->i', r, u0),
                                     np.einsum('ij,ij->i', r, np.cross(m0, u0)),
                                     np.einsum('ij,ij->i', r, m0)])
            target[sel] = (p1 + local[:, :1] * u1 + local[:, 1:2] * np.cross(m1, u1)
                           + local[:, 2:] * m1 + out[sel]
                           - np.einsum('ij,ij->i', out[sel], m1)[:, None] * m1)
            # and only where the body moved under it. Elsewhere -- the halter round
            # the neck, the band round the back -- carrying the move would put the
            # cloth back where it began and tear it from the field's move next to it.
            wrapw[sel] = (1 - smoothstep(d, *near)) * smoothstep(
                np.linalg.norm(p1 - p0, axis=1), *MOVED)
        wrapw *= smoothstep(np.abs(pre[idx, 0]), *WRAP_BRIDGE)
        disp = (target - V[idx]) * wrapw[:, None]
        disp = _blur(V[idx], disp, blur)
        disp = 0.5 * (disp + disp[mirror(V[idx])] * flip)
        V[idx] += disp
        n_w = int((wrapw > 0.5).sum())
    return n_w, wrapw


def reseat(V, pre, skin_tris, idx, weight, near=WRAP_NEAR, floor=None, mid=0.08, blur=0.10,
           rounds=3):
    """Put cloth back on the body at the standoff it had, wherever it now sits.

    A band that has followed its cup up the ribs is in the right place along the
    garment and the wrong place off the body -- half of it inside. This slides it
    out to the surface again without moving it along the garment, so the seam to
    the cup holds and the band lies on the skin.
    """
    flip = np.array([-1.0, 1.0, 1.0])
    n0 = smooth_normals(pre, skin_tris)
    cx0 = pre[skin_tris].mean(1)[:, 0]
    h0 = np.zeros(len(idx)); d0 = np.full(len(idx), 9.9)
    for sgn in (-1.0, 1.0):
        sel = (np.sign(pre[idx, 0]) == sgn) & (np.abs(pre[idx, 0]) > mid)
        if not sel.any():
            continue
        side = idx[sel]
        tris = skin_tris[np.sign(cx0) == sgn]
        d, t, w = Mesh(pre, tris).closest(pre[side])
        f = tris[t]
        q = w[:, :1] * pre[f[:, 0]] + w[:, 1:2] * pre[f[:, 1]] + w[:, 2:] * pre[f[:, 2]]
        m = w[:, :1] * n0[f[:, 0]] + w[:, 1:2] * n0[f[:, 1]] + w[:, 2:] * n0[f[:, 2]]
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
        h0[sel] = np.einsum('ij,ij->i', pre[side] - q, m)
        d0[sel] = d
    if floor is not None:
        h0[:] = floor
    w0 = np.asarray(weight, float) * (1 - smoothstep(d0, *near))
    for _ in range(rounds):
        n1 = smooth_normals(V, skin_tris)
        cx = V[skin_tris].mean(1)[:, 0]
        target = V[idx].copy()
        for sgn in (-1.0, 1.0):
            sel = (np.sign(V[idx, 0]) == sgn) & (np.abs(V[idx, 0]) > mid)
            if not sel.any():
                continue
            side = idx[sel]
            tris = skin_tris[np.sign(cx) == sgn]
            d, t, w = Mesh(V, tris).closest(V[side])
            f = tris[t]
            q = w[:, :1] * V[f[:, 0]] + w[:, 1:2] * V[f[:, 1]] + w[:, 2:] * V[f[:, 2]]
            m = w[:, :1] * n1[f[:, 0]] + w[:, 1:2] * n1[f[:, 1]] + w[:, 2:] * n1[f[:, 2]]
            m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
            target[sel] = q + h0[sel][:, None] * m
        disp = (target - V[idx]) * w0[:, None]
        disp = _blur(V[idx], disp, blur)
        disp = 0.5 * (disp + disp[mirror(V[idx])] * flip)
        V[idx] += disp
    return int((w0 > 0.5).sum())


def sag(V, skin_tris, idx, floor, reach=0.3, mid=0.08):
    """How far each vertex in `idx` is short of `floor` off the skin on its own side
    (0 where clear), by the skin triangle's plane. On the midline both sides meet."""
    cx = V[skin_tris].mean(1)[:, 0]
    out = np.zeros(len(idx))
    pos = {int(i): k for k, i in enumerate(idx)}
    for sgn in (-1.0, 1.0, 0.0):
        if sgn:
            side = idx[(np.sign(V[idx, 0]) == sgn) & (np.abs(V[idx, 0]) > mid)]
            tris = skin_tris[np.sign(cx) == sgn]
        else:
            side = idx[np.abs(V[idx, 0]) <= mid]
            tris = skin_tris
        if not len(side):
            continue
        skin = Mesh(V, tris)
        d, t, w = skin.closest(V[side])
        f = tris[t]
        q = w[:, :1] * V[f[:, 0]] + w[:, 1:2] * V[f[:, 1]] + w[:, 2:] * V[f[:, 2]]
        h = np.einsum('ij,ij->i', V[side] - q, skin.normal[t])
        need = np.where(d < reach, np.maximum(0.0, floor - h), 0.0)
        for i, nd in zip(side, need):
            out[pos[int(i)]] = nd
    return out


def raycast(O, D, V, T):
    """Distance along each ray to its first hit on the triangles, nan for none."""
    v0, v1, v2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    out = np.full(len(O), np.nan)
    for i in range(len(O)):
        d = D[i]
        p = np.cross(d, e2)
        det = np.einsum('ij,ij->i', e1, p)
        ok = np.abs(det) > 1e-9
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        sv = O[i] - v0
        u = np.einsum('ij,ij->i', sv, p) * inv
        q = np.cross(sv, e1)
        v = np.einsum('j,ij->i', d, q) * inv
        t = np.einsum('ij,ij->i', e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
        if hit.any():
            out[i] = t[hit].min()
    return out


def mirror(P, tol=0.03):
    """For each point, the index of its partner across x=0 -- or its own index
    where there is none within `tol`. A strap tie has no partner, and pairing it
    with whatever is nearest hands it someone else's step."""
    d, j = cKDTree(P).query(P * np.array([-1.0, 1.0, 1.0]))
    return np.where(d < tol, j, np.arange(len(P)))


class Field:
    def __init__(self, anchors, D, radius):
        self.tree, self.A, self.D, self.radius = cKDTree(anchors), anchors, D, radius

    def _raw(self, P, k=24):
        d, j = self.tree.query(P, k=min(k, len(self.A)))
        w = np.exp(-(d / self.radius) ** 2)
        w /= w.sum(1, keepdims=True) + 1e-12
        return (self.D[j] * w[..., None]).sum(1)

    def __call__(self, P, anchor):
        P = np.atleast_2d(np.asarray(P, float))
        flip = np.array([-1.0, 1.0, 1.0])
        D = 0.5 * (self._raw(P) + self._raw(P * flip) * flip)
        return D * window(P, anchor)[:, None]


def main(a):
    H, R = PMX(a.host), PMX(a.ref)
    V = np.array([v[0] for v in H.vertices])
    bh = {b['name']: np.array(b['pos']) for b in H.bones}
    br = {b['name']: np.array(b['pos']) for b in R.bones}
    bones = [bh[n] for n in BREAST]

    # -- resolution first. A triangle bikini's cage is a handful of half-unit edges
    # and cannot be smooth however it is placed; and the skin under it is read off
    # the cup, so the cup has to be smooth before that is done.
    n_before = len(H.vertices)
    fine_tris = {}
    for name, levels in SUBDIV.items():
        if not any(m['name'] == name for m in H.materials):
            continue
        V, tris = subdivide(H, V, name, levels)
        fine_tris[name] = tris
        print(f'  {name:15s} subdivided x{levels}: {len(tris)} triangles')

    # -- the breast itself: cut out of the skin and built again over the chest wall
    V, st, patch = breastform.build(
        H, V, {n: bh[n] for n in BREAST}, SKIN,
        reach=(a.up, a.down, a.outer, a.inner), height=a.height, apex=(a.apex_out, a.apex_up),
        shape=(a.top, a.under, a.side), point=(a.point_up, a.point_under, a.point_side),
        cut=(1.15, a.cut_tall), centre=(0.15, a.cut_up))
    before = V.copy()          # the model as it stands, at the resolution it will keep

    # -- the breast, made smaller on its own bones.
    #
    # The rig already carries a breast: the 胸 bones move it and nothing else, with
    # the falloff the author painted into the weights -- full at the bust, nothing
    # at the ribs, the collarbone and the arm. Scaling the model about those bones
    # by each vertex's own weight is the deformation the rig itself would make. It
    # needs no fitted chest wall to measure from, no window and no radial gate, so
    # there is no ring where a gate fades and no seam to smooth away afterwards;
    # and the top the bikini is weighted to the same bones, so it comes along.
    names = [b['name'] for b in H.bones]
    ks = [names.index(n) for n in BREAST]
    W = np.zeros((len(V), len(ks)))
    for i, v in enumerate(H.vertices):
        for b, x in zip(v[5], v[6]):
            if x > 0 and b in ks:
                W[i, ks.index(b)] += x
    # Painted weights are smooth enough for the small moves a rig makes and not for
    # a standing change of size: neighbouring vertices differing by a few hundredths
    # come out as ripples across the breast. Blurred over the surface, the falloff
    # the author drew is kept and the noise in it is not.
    for _ in range(2):
        W = np.where(W.sum(1, keepdims=True) > 0.001, _blur(V, W, 0.12), W)
    S = np.array([a.wide, a.tall, a.deep])
    step = np.zeros_like(V)
    for j, n in enumerate(BREAST):
        b = bh[n]
        step += (b + (V - b) * S + np.array([0.0, a.lift, 0.0]) - V) * W[:, j:j + 1]
    step[np.unique(st)] = 0.0                       # the skin is built; the cloth follows
    V += step
    moved = np.linalg.norm(step, axis=1) > 1e-5
    print(f'breast on its bones: x{a.wide:.2f} wide, x{a.tall:.2f} tall, x{a.deep:.2f} deep, '
          f'raised {a.lift:+.2f}; {int(moved.sum())} vertices moved, furthest '
          f'{np.linalg.norm(step, axis=1).max():.3f}, weight up to {W.sum(1).max():.2f}')

    # -- the cup sits on the new breast. It was cut for the old one and scaled on
    # the bones toward this size; each cup vertex near the breast is put at one
    # standoff off the surface built under it.
    for name in ('cloth01',):
        if name not in fine_tris:
            continue
        idx = np.unique(fine_tris[name])
        # the cup proper, by how much of it the breast bones carry, let go smoothly
        # above the cup where the halter strap brushes the skin: a hard edge to the
        # seating stretches the cloth across it sixfold
        near = idx[W[idx].sum(1) > 0.02]
        hold = (smoothstep(W[near].sum(1), 0.05, 0.3)
                * (1 - smoothstep(V[near, 1] - max(b[1] for b in bones), 0.8, 1.3)))
        if len(near) > 8:
            n_r = reseat(V, V.copy(), st, near, hold, near=(0.25, 0.45),
                         floor=CUP_FLOOR, rounds=8)
            short = sag(V, st, near, 0.0)
            print(f'  {name:15s} {n_r} of {len(near)} cup vertices set back on the breast; '
                  f'{int((short > 2e-3).sum())} left inside it, deepest {short.max():.3f}')

    which = np.linalg.norm(V - before, axis=1) > 1e-5
    which[patch] = True                            # built this run: its normals are placeholders

    # -- the bones and their spheres go with it
    for b in H.bones:
        if b['name'] in BREAST:
            b['pos'] = [float(x) for x in (np.array(b['pos']) + np.array([0.0, a.lift, 0.0]))]
    for r in H.rigids:
        if 0 <= r['bone'] < len(names) and names[r['bone']] in BREAST:
            base = bh[names[r['bone']]]
            c = base + (np.array(r['pos']) - base) * S + np.array([0.0, a.lift, 0.0])
            r['pos'] = [float(x) for x in c]
            r['size'] = [float(x * s) for x, s in zip(r['size'], (a.wide, a.tall, a.wide))]
            print(f'  sphere on {names[r["bone"]]}: radius {r["size"][0]:.3f}')
    for j in H.joints:
        rn = [x['name'] for x in H.rigids]
        if rn[j['a']] in BREAST or rn[j['b']] in BREAST:
            base = bh[BREAST[0] if np.array(j['pos'])[0] > 0 else BREAST[1]]
            j['pos'] = [float(x) for x in
                        (base + (np.array(j['pos']) - base) * S + np.array([0.0, a.lift, 0.0]))]

    F = np.array(H.faces).reshape(-1, 3)

    # -- commit, and rebuild the normals where anything moved
    for i in np.where(which)[0]:
        v = list(H.vertices[i])
        v[0] = [float(x) for x in V[i]]
        H.vertices[i] = tuple(v)
    at = 0
    for mat in H.materials:
        nn = mat['fcount'] // 3
        tris = F[at:at + nn]
        at += nn
        idx = np.unique(tris)
        if which[idx].any():
            N = smooth_normals(V, tris)
            for i in idx:
                v = list(H.vertices[i])
                v[1] = [float(x) for x in N[i]]
                H.vertices[i] = tuple(v)

    # -- what changed, per material: no edge may have torn
    at = 0
    for mat in H.materials:
        nn = mat['fcount'] // 3
        tris = F[at:at + nn]
        at += nn
        if not which[np.unique(tris)].any():
            continue
        E = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        L0 = np.linalg.norm(before[E[:, 0]] - before[E[:, 1]], axis=1)
        L1 = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        ok = L0 > 1e-6
        r = L1[ok] / L0[ok]
        real = L0[ok] > 0.05
        rr = r[real]
        Em = E[ok][real]
        j = np.argmax(np.abs(np.log(rr)))
        print(f'  {mat["name"]:15s} edge ratio {rr.min():.3f}..{rr.max():.3f} on real edges, '
              f'{int((rr < 0.5).sum())} under 0.5, {int((rr > 1.5).sum())} over 1.5; worst at '
              f'{np.round((V[Em[j, 0]] + V[Em[j, 1]]) / 2, 2)}')

    if os.environ.get('BUSTFIT_DUMP'):
        np.savez(os.environ['BUSTFIT_DUMP'], before=before, V=V, **(stages or {}))

    # -- and the line itself, against reze's, in the breast band
    def bust(P, b):
        rel = P - b[ANCHOR]
        sel = (rel[:, 1] > 0.3) & (rel[:, 1] < 1.9) & (rel[:, 2] < 0)
        return np.ptp(rel[sel, 0]), -rel[sel, 2].min()
    rt = np.concatenate([material_slice(R, m)[0] for m in REF_SURFACE])
    sidx = np.unique(st)
    RV = np.array([v[0] for v in R.vertices]) + (bh[ANCHOR] - br[ANCHOR])
    shift = bh[ANCHOR] - br[ANCHOR]
    for label, P, b in (('was', before[sidx], bh), ('now', V[sidx], bh),
                        ('reze', RV[np.unique(rt)] - shift, br)):
        wdt, dep = bust(P, b)
        print(f'bust {label:4s} width {wdt:.3f}  depth {dep:.3f}')

    # -- the skin reads reze's sheet: one atlas, the same to 1/255 off the chest,
    # and on the chest painted for the bust the body now has
    import shutil
    src, name = SHEET
    shutil.copyfile(src, os.path.join(os.path.dirname(a.out), 'textures', name))
    rel = 'textures\\' + name
    if rel not in H.textures:
        H.textures.append(rel)
    next(m for m in H.materials if m['name'] == SKIN)['tex'] = H.textures.index(rel)
    write_pmx(a.out, H)
    for name in DROP:
        dropmaterial.main(a.out, name)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--host', default=HOST)
    p.add_argument('--ref', default=REF)
    p.add_argument('--out', default=OUT)
    p.add_argument('--up', type=float, default=1.1, help='the breast reaches this far up from its apex')
    p.add_argument('--down', type=float, default=0.65, help='this far down')
    p.add_argument('--outer', type=float, default=0.85, help='this far outward')
    p.add_argument('--inner', type=float, default=0.5, help='this far inward')
    p.add_argument('--height', type=float, default=1.0, help='and stands this far off the chest wall at the apex')
    p.add_argument('--apex-out', type=float, default=0.15, help='the apex sits this far outward of the 胸 bone')
    p.add_argument('--apex-up', type=float, default=-0.3, help='and this far above it')
    p.add_argument('--top', type=float, default=1.2, help='how the top foot lays down onto the chest: near 2 is without a break')
    p.add_argument('--under', type=float, default=1.6, help='how the under foot comes in: near 1 is round and quick, below 1 a cut')
    p.add_argument('--side', type=float, default=1.2, help='and the sides')
    p.add_argument('--point-up', type=float, default=2.4, help='the slope from the apex up: near 1 is straight, above 2 a full shoulder')
    p.add_argument('--point-under', type=float, default=1.8, help='the slope from the apex down: above 2 is full and round')
    p.add_argument('--point-side', type=float, default=1.7, help='and to the sides')
    p.add_argument('--cut-tall', type=float, default=1.6, help='the opening cut for the breast, half its height')
    p.add_argument('--cut-up', type=float, default=0.35, help='and how far above the 胸 bone its centre sits')
    p.add_argument('--wide', type=float, default=0.78, help='the top scaled about the breast bones to follow, across')
    p.add_argument('--tall', type=float, default=0.66, help='and up and down')
    p.add_argument('--deep', type=float, default=0.82, help='and forward')
    p.add_argument('--lift', type=float, default=-0.25, help='and raised by this much')
    p.add_argument('--gather', type=float, default=0.0,
                   help='pull each breast toward the midline by up to this, outer edge held')
    main(p.parse_args())
