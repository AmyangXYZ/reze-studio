# -*- coding: utf-8 -*-
"""Put the fitted bust of one dress onto the well-placed body of another.

Two builds of the same garment: `maidbuild` places it by a body field, which lands the
collar, the shoulders and the skirt where they belong and leaves the bust conformed by
`fit_bust` -- a per-vertex push along the normal, which is what goes lumpy over a bust
this much fuller than the one the dress was drawn for. PmxDressup fits it as a bone
morph, which never touches the garment's own surface and so stays smooth, and places
everything else a little off.

They are the same mesh -- 6712 vertices, 10175 triangles, identical UVs to 0.000000 --
so one can simply take the other's bust. Paired by UV and then by position, because a
UV alone is not unique: 6712 vertices share 4507 of them, the duplicates being the two
sides of every seam.

    python3 tools/zzz2pmx/bustmerge.py

Reads `previous.pmx` and `reze-maid.pmx`, writes `reze-maid.pmx`.
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holefill
from pmx import PMX
from surface import Mesh
from pmxw import write_pmx

PLACED = 'public/models/reze-maid-selene/previous.pmx'
FITTED = 'public/models/reze-maid-selene/placed.pmx'
OUT = 'public/models/reze-maid-selene/reze-maid.pmx'

DRESS = 'maid_dress'

# The shoes come back from the placed build too -- 沉醉旖旎's, drawn for this rig's
# foot. They are thinner-soled than the maid's by exactly the LIFT, so the whole model
# comes down by it: whichever shoes she stands in decides how high she stands.
# Whatever is pinned to the dress travels with it. The bell hangs on the collar and
# the hearts sit on the cuffs, and they are their own material -- left where the fit
# put them while the dress moved to the placed build, the bell came off the collar and
# the hearts off the cuffs.
FOLLOW = ['maid_metal', 'maid_lace']
REACH = 0.35         # a piece further than this from the dress is not on it

SHOES = [('shoes', 'shoes'), ('shoe_metal', 'shoe_metal')]
SHED = ['maid_shoes']

# The hosiery is the maid's and the shoes are 沉醉旖旎's: the two were never drawn
# for one another, so the sock's foot lay OVER the leather rather than in it, and the
# passes that cut back whatever stood proud left the holes at the ankle and the instep.
# Nothing is cut. The foot goes into the shoe, and the hosiery is then set on the leg
# -- every vertex of it, the way the underwear is set on the body -- so it is the leg's
# own shape at the ankle and inside the leather both.
SOCK = 'maid_stockings'
SHOE = 'shoes'
INSIDE = 0.030       # how far behind the leather the foot sits
NEAR = 0.40          # leather further off than this is not behind it
SOCK_CLING = 0.010   # the hosiery stands this far off her, and nothing more

UNDER = 'maid_underwear'
# Raised clear of her rather than set on her, at the 0.035 `reze-maid-fullbody`
# carries. Cutting the skin out from under it instead took the thigh with it: the
# garment is narrow, so the margin that keeps a cut off its edges leaves almost
# nothing to cut, and what does go is the skin between the hem and the stocking.
CLING = 0.035
PROUD = 0.030        # nothing of it left sunk in her
CLEAR = 0.012        # and nothing of her left coming through it, measured from her side        # and raised after, since a coarse shell clung to a dense body
                     # cuts her corners and she comes through them
# The skin under both comes out. Two surfaces a hundredth apart cannot both be drawn:
# whichever wins a pixel is a coin toss, and the one underneath is never the one that
# should. The hosiery takes the whole leg below its top edge, since it is a tube and
# that edge is the only one of it on show; the underwear is narrow and takes only what
# it actually covers, kept a margin in from every edge it has.
BARE = 0.060         # a garment nearer than this to her is lying on her
STRIP = [(SOCK, 0.12, True)]   # piece, margin, hem rule
HEM = 9.5            # which is the only edge of the hosiery on show

# And the dress lifted off her. With the underwear set exactly on the skin -- max
# standoff 0.0040 -- 27 of its vertices still showed through the dress, which can only
# mean the dress is inside her: nothing lying on a body can come through a garment
# that is outside it. The dress was sunk by up to 0.29.
FREE = 0.015         # how far clear of her the dress has to sit

# The skirt's top row starts inside the body's own collider, and a dynamic body that
# begins inside a static one is pushed out every frame while its joint pulls it back:
# that argument is the jitter under the bust. The maid has the same overlap on her own
# model -- 5 bodies, deepest 0.094 -- so the fix is not to remove it but to bring it
# back to what she ships with. Ours was 10 bodies at 0.187.
# The bodice floats off her rather than resting on her. On the maid's own body the
# skin behind `cloth01` sits a median 0.104 back, p5 0.015; ours came out at 0.058
# and p5 0.013, with a quarter of the covered skin inside 0.03. The breast swings
# under a bodice that is only 14% breast-driven, so at that separation it comes
# through and back every frame -- the flicker read as the underboob jitter. The
# skin behind the cloth is hidden, so it is the side that gives way.
ROOM = 0.10          # the clearance the maid's own bodice keeps
CAP = 0.09           # and no vertex gives more than this
COVER = 0.35         # cloth further off than this is not over her
SINK_FROM = (15.6, 16.1)   # nothing at the hip, all of the torso

JAM = 'skirt_0_'
SLACK = 0.094        # the deepest overlap the maid herself has
# The bodice over the bust, and nothing else. Everything outside this keeps the
# placement it already has.
BUST = dict(ylo=15.0, yhi=17.7, zmax=-0.85, fade=0.45)

# `previous.pmx` stands in 塞勒涅's own shoes and the fitted build stands in the maid's,
# whose sole is thicker by exactly this. Blending one into the other without it drops
# the collar and the sleeve by 0.28.
LIFT = 0.282320


def pair(a, ai, b, bi, AU, BU, AV, BV):
    """Which vertex of `a` is which of `b`, by UV and then by where it is."""
    bucket = defaultdict(list)
    for i in ai:
        bucket[(round(float(AU[i][0]), 6), round(float(AU[i][1]), 6))].append(int(i))
    out, lost = {}, 0
    for i in bi:
        k = (round(float(BU[i][0]), 6), round(float(BU[i][1]), 6))
        cand = bucket.get(k)
        if not cand:
            lost += 1
            continue
        out[int(i)] = min(cand, key=lambda j: float(np.linalg.norm(AV[j] - BV[i])))
    return out, lost


def _rim(V, tris):
    """The vertices on a real opening.

    Welded within a tolerance first. A garment is split along its UV seams, and after
    a fit the two sides of a seam no longer round to the same key -- taken for
    boundaries they put a slit up the whole length of the hosiery, and a guard against
    its edges then guards away the entire leg.
    """
    import collections
    from scipy.spatial import cKDTree
    ids = np.unique(tris)
    tree = cKDTree(V[ids])
    weld = {}
    for a, b in sorted(tree.query_pairs(2e-3)):
        weld[int(ids[b])] = int(ids[a])
    def root(i):
        while i in weld:
            i = weld[i]
        return i
    w = {int(i): root(int(i)) for i in ids}
    ec = collections.Counter()
    for a, c, d in tris:
        t = (w[int(a)], w[int(c)], w[int(d)])
        for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            ec[tuple(sorted(e))] += 1
    live = {x for e, n in ec.items() if n == 1 for x in e}
    return {int(i) for i in ids if w[int(i)] in live}


def intoshoe(b, piece='skin', shoe=SHOE, inside=INSIDE, near=NEAR, rounds=5):
    """Put what is proud of the shoe back inside it.

    The test is leather directly behind, and interior leather at that: the closest
    point on a pump is as often its rim as its upper, and an ankle standing clear of
    the rim is an ankle standing in the mouth, where it belongs. Excluding the
    triangles that carry the rim is what tells the two apart.
    """
    sl = holefill.slices(b)
    if piece not in sl or shoe not in sl:
        return
    V = np.array([v[0] for v in b.vertices])
    tris = sl[shoe]
    edge = _rim(V, tris)
    solid = np.array([not (int(a) in edge or int(c) in edge or int(d) in edge)
                      for a, c, d in tris])
    shell = Mesh(V, tris)
    top = float(V[np.unique(tris)][:, 1].max())
    ids = np.array([i for i in np.unique(sl[piece]) if V[i][1] < top], int)
    if not len(ids):
        return
    P = V[ids]
    d, t, bar = shell.closest(P)
    f = shell.F[t]
    q = (V[f[:, 0]] * bar[:, :1] + V[f[:, 1]] * bar[:, 1:2] + V[f[:, 2]] * bar[:, 2:3])
    proud = np.einsum('ij,ij->i', P - q, shell.normal[t])
    w = np.clip((near - d) / near, 0.0, 1.0)
    w = w * w * (3 - 2 * w) * solid[t]
    need = np.clip(proud + inside, 0.0, 0.5) * w

    nbr = {}
    for x, y, z in sl[piece]:
        for u, v in ((x, y), (y, z), (z, x)):
            nbr.setdefault(int(u), set()).add(int(v))
            nbr.setdefault(int(v), set()).add(int(u))
    at = {int(v): k for k, v in enumerate(ids)}
    order = [np.array([at[j] for j in nbr.get(int(v), ()) if j in at], int) for v in ids]
    for _ in range(rounds):
        nxt = need.copy()
        for k, ns in enumerate(order):
            if len(ns):
                nxt[k] = 0.4 * need[k] + 0.6 * need[ns].mean()
        need = nxt

    dirn = shell.normal[t]
    for k, i in enumerate(ids):
        if need[k] <= 1e-6:
            continue
        v = list(b.vertices[int(i)])
        v[0] = [float(x) for x in V[i] - dirn[k] * need[k]]
        b.vertices[int(i)] = tuple(v)
    W = np.array([v[0] for v in b.vertices])
    d2, t2, bar2 = shell.closest(W[ids])
    f2 = shell.F[t2]
    q2 = (V[f2[:, 0]] * bar2[:, :1] + V[f2[:, 1]] * bar2[:, 1:2] + V[f2[:, 2]] * bar2[:, 2:3])
    now = np.einsum('ij,ij->i', W[ids] - q2, shell.normal[t2])
    was_out = (proud > 0.003) & (solid[t2] > 0) & (d < near)
    now_out = (now > 0.003) & (solid[t2] > 0) & (d2 < near)
    print(f'  {piece}: {int(was_out.sum())} of {len(ids)} vertices stood through the '
          f'leather, worst +{proud[was_out].max() if was_out.any() else 0:.3f} -> '
          f'{int(now_out.sum())}, worst +{now[now_out].max() if now_out.any() else 0:.3f}')


def carry(b, was, paired, pieces=FOLLOW, reach=REACH):
    """Move what is pinned to the dress by however far the dress moved.

    A shell at a time, rigidly. Given each vertex the displacement of whichever dress
    vertex happened to be nearest IT, the bell came apart -- its front and its back
    took different answers and the shape went with them. A bell is not cloth: it moves
    as one thing or it stops being a bell.
    """
    from scipy.spatial import cKDTree
    from maiddress import shells
    sl = holefill.slices(b)
    now = np.array([v[0] for v in b.vertices])
    ids = np.array(sorted(paired), int)
    delta = now[ids] - was[ids]
    tree = cKDTree(was[ids])
    for name in pieces:
        if name not in sl:
            continue
        moved, left = 0, 0
        for part in shells(sl[name]):
            who = np.unique(part)
            d, j = tree.query(now[who])
            near = d < reach
            if not near.any():
                left += 1
                continue
            step = delta[j[near]].mean(0)
            for i in who:
                v = list(b.vertices[int(i)])
                v[0] = [float(x) for x in np.array(v[0]) + step]
                b.vertices[int(i)] = tuple(v)
            moved += 1
        print(f'  {name}: {moved} shells carried with the dress whole'
              + (f', {left} left where they were' if left else ''))


def unjam(b, chain=JAM, slack=SLACK):
    """Take the body's collider in until the skirt is no more jammed than hers."""
    names = [x['name'] for x in b.bones]

    def inside(p, r):
        c = np.array(r['pos'])
        size = np.array(r['size'])
        d = p - c
        if r['shape'] == 0:
            return size[0] - float(np.linalg.norm(d))
        if r['shape'] == 1:
            return float(np.min(size - np.abs(d)))
        h = max(size[1] / 2, 0.0)
        q = np.array([d[0], max(abs(d[1]) - h, 0.0) * np.sign(d[1]), d[2]])
        return size[0] - float(np.linalg.norm(q))

    top = [r for r in b.rigids
           if (names[r['bone']] if 0 <= r['bone'] < len(names) else '').startswith(chain)]
    worst = {}
    for r in b.rigids:
        if r['mode'] != 0:
            continue
        n = names[r['bone']] if 0 <= r['bone'] < len(names) else '?'
        for t in top:
            if not (r['nocollide'] >> t['group']) & 1:
                continue
            v = inside(np.array(t['pos']), r)
            if v > worst.get(n, (0.0, None))[0]:
                worst[n] = (v, r)
    for n, (v, r) in worst.items():
        if v <= slack:
            continue
        take = v - slack
        r['size'] = [max(float(x) - take, 0.01) if k != 1 or r['shape'] != 2 else float(x)
                     for k, x in enumerate(r['size'])]
        print(f'  {n}: its collider took the skirt {v:.3f} in, brought in by '
              f'{take:.3f} to the {slack:.3f} the maid herself has')


def lift(b, piece=DRESS, body='skin', offset=FREE):
    """Raise the dress out of her, wherever it has sunk in.

    The whole garment at once. `push_out` blurs its lift across whatever it is given,
    so given a band it stops dead at the band's edge and tears -- which is exactly how
    an earlier attempt at this shredded the ruffles under the bust.
    """
    from surface import push_out
    sl = holefill.slices(b)
    if piece not in sl or body not in sl:
        return
    V = np.array([v[0] for v in b.vertices])
    skin = Mesh(V, sl[body])
    ids = np.unique(sl[piece])
    d, t, bar = skin.closest(V[ids])
    f = skin.F[t]
    q = (V[f[:, 0]] * bar[:, :1] + V[f[:, 1]] * bar[:, 1:2] + V[f[:, 2]] * bar[:, 2:3])
    was = np.einsum('ij,ij->i', V[ids] - q, skin.normal[t])
    moved, lifted = push_out(skin, V[ids], offset=offset, rounds=20)
    for k, i in enumerate(ids):
        v = list(b.vertices[int(i)])
        v[0] = [float(x) for x in moved[k]]
        b.vertices[int(i)] = tuple(v)
    W = np.array([v[0] for v in b.vertices])
    d, t, bar = skin.closest(W[ids])
    f = skin.F[t]
    q = (V[f[:, 0]] * bar[:, :1] + V[f[:, 1]] * bar[:, 1:2] + V[f[:, 2]] * bar[:, 2:3])
    now = np.einsum('ij,ij->i', W[ids] - q, skin.normal[t])
    print(f'  {piece}: {int(lifted.sum())} of {len(ids)} lifted out of her; sunk in '
          f'{int((was < 0).sum())} -> {int((now < 0).sum())}, deepest '
          f'{-was.min():.3f} -> {max(0.0, -now.min()):.3f}')


def cling(b, piece=UNDER, body='skin', offset=CLING, rounds=1):
    """Set a skin-tight piece onto the skin, both ways.

    Lifting it is not enough. `push_out` only raises what has sunk and leaves what is
    proud where it is, so the piece came out at a median of +0.018 with a p90 of
    +0.105 -- and the parts standing a tenth off her were the parts coming through
    the dress. On the maid's own body this garment sits at exactly +0.0000: it is not
    near her skin, it IS her skin. So every vertex is put on the surface rather than
    only the buried ones.
    """
    sl = holefill.slices(b)
    if piece not in sl or body not in sl:
        return
    V = np.array([v[0] for v in b.vertices])
    skin = Mesh(V, sl[body])
    ids = np.unique(sl[piece])
    d, t, bar = skin.closest(V[ids])
    f = skin.F[t]
    q = (V[f[:, 0]] * bar[:, :1] + V[f[:, 1]] * bar[:, 1:2] + V[f[:, 2]] * bar[:, 2:3])
    was = np.einsum('ij,ij->i', V[ids] - q, skin.normal[t])
    delta = (q + skin.normal[t] * offset) - V[ids]
    # blurred across the piece, so a fold in it does not become a crease
    nbr = {}
    for x, y, z in sl[piece]:
        for u, v in ((x, y), (y, z), (z, x)):
            nbr.setdefault(int(u), set()).add(int(v))
            nbr.setdefault(int(v), set()).add(int(u))
    at = {int(v): k for k, v in enumerate(ids)}
    order = [np.array([at[j] for j in nbr.get(int(v), ()) if j in at], int) for v in ids]
    for _ in range(rounds):
        nxt = delta.copy()
        for k, ns in enumerate(order):
            if len(ns):
                nxt[k] = 0.6 * delta[k] + 0.4 * delta[ns].mean(0)
        delta = nxt
    for k, i in enumerate(ids):
        v = list(b.vertices[int(i)])
        v[0] = [float(x) for x in V[i] + delta[k]]
        b.vertices[int(i)] = tuple(v)
    W = np.array([v[0] for v in b.vertices])
    d, t, bar = skin.closest(W[ids])
    f = skin.F[t]
    q = (V[f[:, 0]] * bar[:, :1] + V[f[:, 1]] * bar[:, 1:2] + V[f[:, 2]] * bar[:, 2:3])
    now = np.einsum('ij,ij->i', W[ids] - q, skin.normal[t])
    print(f'  {piece}: set on her skin -- standoff median {np.median(was):+.4f} -> '
          f'{np.median(now):+.4f}, p90 {np.percentile(was, 90):+.4f} -> '
          f'{np.percentile(now, 90):+.4f}, inside {int((was < 0).sum())} -> '
          f'{int((now < 0).sum())}')


def _hidden(P, D, V, T, far, chunk=64):
    """Which of these points has cloth directly over it, along its own normal.

    A ray cast, not a proximity guess: skin just below the neckline is as near the
    collar as skin behind it is, and only one of the two is out of sight.
    """
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1, e2 = b - a, c - a
    out = np.zeros(len(P), bool)
    for k in range(0, len(P), chunk):
        p, d = P[k:k + chunk], D[k:k + chunk]
        h = np.cross(d[:, None, :], e2[None, :, :])
        det = np.einsum('tj,ptj->pt', e1, h)
        ok = np.abs(det) > 1e-9
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s_ = p[:, None, :] - a[None, :, :]
        u = np.einsum('ptj,ptj->pt', s_, h) * inv
        q = np.cross(s_, e1[None, :, :])
        v = np.einsum('pj,ptj->pt', d, q) * inv
        t = np.einsum('tj,ptj->pt', e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4) & (t < far)
        out[k:k + chunk] = hit.any(1)
    return out


def sink(b, body='skin', piece=DRESS, room=ROOM, cap=CAP, reach=COVER,
         band=SINK_FROM, rounds=4):
    """Set the skin back under the bodice, to the clearance the maid's own has.

    Only where the cloth is over her -- the test is her own normal pointing at it,
    so the neckline and the shoulders, where she is on show, do not move. The amount
    is blurred across the skin, so giving way cannot dent her.
    """
    sl = holefill.slices(b)
    if piece not in sl or body not in sl:
        return
    V = np.array([v[0] for v in b.vertices])
    N = np.array([v[1] for v in b.vertices])
    cloth = Mesh(V, sl[piece])
    ids = np.unique(sl[body])
    ids = ids[V[ids][:, 1] >= band[0]]
    d, t, bar = cloth.closest(V[ids])
    f = cloth.F[t]
    q = (V[f[:, 0]] * bar[:, :1] + V[f[:, 1]] * bar[:, 1:2] + V[f[:, 2]] * bar[:, 2:3])
    was = np.einsum('ij,ij->i', V[ids] - q, cloth.normal[t])

    nv = N[ids] / (np.linalg.norm(N[ids], axis=1, keepdims=True) + 1e-9)
    to = q - V[ids]
    face = np.einsum('ij,ij->i', nv, to / (np.linalg.norm(to, axis=1, keepdims=True) + 1e-9))
    lo, hi = band
    up = np.clip((V[ids][:, 1] - lo) / (hi - lo), 0.0, 1.0)
    up = up * up * (3 - 2 * up)
    covered = (d < reach) & (face > 0.3)
    covered[covered] = _hidden(V[ids][covered], nv[covered], V, sl[piece], reach)
    need = np.clip((room + was) / np.maximum(face, 0.4), 0.0, cap) * up
    need = np.where(covered, need, 0.0)

    nbr = {}
    for x, y, z in sl[body]:
        for u, v in ((x, y), (y, z), (z, x)):
            nbr.setdefault(int(u), set()).add(int(v))
            nbr.setdefault(int(v), set()).add(int(u))
    at = {int(v): k for k, v in enumerate(ids)}
    order = [np.array([at[j] for j in nbr.get(int(v), ()) if j in at], int) for v in ids]
    for _ in range(rounds):
        nxt = need.copy()
        for k, ns in enumerate(order):
            if len(ns):
                nxt[k] = 0.5 * need[k] + 0.5 * need[ns].mean()
        need = nxt

    for k, i in enumerate(ids):
        if need[k] <= 1e-6:
            continue
        v = list(b.vertices[int(i)])
        v[0] = [float(x) for x in V[i] - nv[k] * need[k]]
        b.vertices[int(i)] = tuple(v)
    W = np.array([v[0] for v in b.vertices])
    d2, t2, bar2 = cloth.closest(W[ids])
    f2 = cloth.F[t2]
    q2 = (V[f2[:, 0]] * bar2[:, :1] + V[f2[:, 1]] * bar2[:, 1:2] + V[f2[:, 2]] * bar2[:, 2:3])
    now = np.einsum('ij,ij->i', W[ids] - q2, cloth.normal[t2])
    m = covered & (was < 0)
    print(f'  {body}: set back under {piece} -- {int((need > 1e-6).sum())} of {len(ids)} gave way, '
          f'deepest {need.max():.3f}; clearance behind the cloth median '
          f'{np.median(-was[m]):.3f} -> {np.median(-now[m]):.3f}, p5 '
          f'{np.percentile(-was[m], 5):.3f} -> {np.percentile(-now[m], 5):.3f}, '
          f'within 0.03 {int((-was[m] < 0.03).sum())} -> {int((-now[m] < 0.03).sum())}')


def outof(b, piece, body='skin', margin=None, near=0.12, rounds=8):
    """Raise a garment until nothing of her comes through it.

    Measured from her side, not its own. A shell as coarse as this one can have every
    vertex of it outside her and still let the buttock and the inner thigh through the
    faces between them -- the standoff its own vertices report is not the standoff that
    shows. Each pass gives its vertices whatever the skin landing on them needs.
    """
    margin = CLEAR if margin is None else margin
    sl = holefill.slices(b)
    if piece not in sl or body not in sl:
        return
    ids = np.unique(sl[body])
    pids = np.unique(sl[piece])
    nbr = {}
    for x, y, z in sl[piece]:
        for u, v in ((x, y), (y, z), (z, x)):
            nbr.setdefault(int(u), set()).add(int(v))
            nbr.setdefault(int(v), set()).add(int(u))
    at = {int(v): k for k, v in enumerate(pids)}
    order = [np.array([at[j] for j in nbr.get(int(v), ()) if j in at], int) for v in pids]
    N = np.array([v[1] for v in b.vertices])
    nv = N[pids] / (np.linalg.norm(N[pids], axis=1, keepdims=True) + 1e-9)
    first = None
    for _ in range(rounds):
        V = np.array([v[0] for v in b.vertices])
        mesh = Mesh(V, sl[piece])
        d, t, bar = mesh.closest(V[ids])
        f = mesh.F[t]
        q = (V[f[:, 0]] * bar[:, :1] + V[f[:, 1]] * bar[:, 1:2] + V[f[:, 2]] * bar[:, 2:3])
        sgn = np.einsum('ij,ij->i', V[ids] - q, mesh.normal[t])
        bad = (sgn > -margin) & (d < near)
        if first is None:
            first = (int((sgn > 0).sum()), float(sgn[d < near].max()))
        if not bad.any():
            break
        need = np.zeros(len(pids))
        want = sgn[bad] + margin
        for k, tri in enumerate(np.where(bad)[0]):
            for j in f[tri]:
                i = at.get(int(j))
                if i is not None and want[k] > need[i]:
                    need[i] = want[k]
        nxt = need.copy()
        for k, ns in enumerate(order):
            if len(ns):
                nxt[k] = max(need[k], 0.7 * need[ns].max())
        need = nxt
        for k, i in enumerate(pids):
            if need[k] <= 1e-6:
                continue
            v = list(b.vertices[int(i)])
            v[0] = [float(x) for x in V[int(i)] + nv[k] * need[k]]
            b.vertices[int(i)] = tuple(v)
    V = np.array([v[0] for v in b.vertices])
    mesh = Mesh(V, sl[piece])
    d, t, bar = mesh.closest(V[ids])
    f = mesh.F[t]
    q = (V[f[:, 0]] * bar[:, :1] + V[f[:, 1]] * bar[:, 1:2] + V[f[:, 2]] * bar[:, 2:3])
    sgn = np.einsum('ij,ij->i', V[ids] - q, mesh.normal[t])
    m = d < near
    print(f'  {piece}: she came through it in {first[0]} places, worst {first[1]:+.3f}'
          f' -> {int((sgn[m] > 0).sum())}, worst {sgn[m].max():+.3f}')


def strip(b, body='skin', pieces=None, gap=None, hem=None):
    """Cut the leg out of the skin, following the hosiery.

    Guarded at the garment's top edge alone. Its other openings are inside the shoe,
    and the slit up its length is not an opening you can see through -- guarding
    against those took the cut back to nothing and left the leg showing through the
    stocking in strips.
    """
    from scipy.spatial import cKDTree
    # Resolved here, not in the signature: a default binds at definition and a module
    # constant set afterwards is then ignored.
    pieces = STRIP if pieces is None else pieces
    gap = BARE if gap is None else gap
    hem = HEM if hem is None else hem
    sl = holefill.slices(b)
    if body not in sl:
        return
    V = np.array([v[0] for v in b.vertices])
    tris = sl[body]
    ids = np.unique(tris)
    at = {int(v): k for k, v in enumerate(ids)}
    under = np.zeros(len(ids), bool)
    for piece, margin, tube in pieces:
        if piece not in sl:
            continue
        pt = sl[piece]
        mesh = Mesh(V, pt)
        d, t, bar = mesh.closest(V[ids])
        f = mesh.F[t]
        q = (V[f[:, 0]] * bar[:, :1] + V[f[:, 1]] * bar[:, 1:2] + V[f[:, 2]] * bar[:, 2:3])
        ok = d < gap
        rim = sorted([i for i in _rim(V, pt) if V[i][1] > hem] if tube else _rim(V, pt))
        if rim:
            away, _ = cKDTree(V[rim]).query(q)
            ok &= away > margin
        if tube and rim:
            # and everything below its top edge goes, however far off it the skin
            # happens to run: a shell this coarse leaves the calf standing clear of it,
            # and a distance test alone left those patches behind.
            R = V[rim]
            _, j = cKDTree(R[:, ::2]).query(V[ids][:, ::2])
            ok |= V[ids][:, 1] < R[j][:, 1] - margin
        under |= ok

    gone = np.array([under[at[int(a)]] and under[at[int(c)]] and under[at[int(d)]]
                     for a, c, d in tris])
    if not gone.any():
        print(f'  {body}: nothing of it lies under {", ".join(x[0] for x in pieces)}')
        return
    F = np.array(b.faces).reshape(-1, 3)
    out, off, cut = [], 0, 0
    for mat in b.materials:
        n = mat['fcount'] // 3
        block = F[off:off + n]
        off += n
        if mat['name'] == body:
            block = block[~gone]
            cut = int(gone.sum())
            mat['fcount'] = len(block) * 3
        out.append(block)
    b.faces = [int(x) for x in np.concatenate(out).reshape(-1)]

    # The cut takes whole triangles, so its edge dips under the hem by up to one of
    # them -- read as a comb of bare skin between the stocking's own teeth. The
    # survivors that are still under it come up to meet it, and the skin then ends on
    # the garment's top line exactly.
    raised = 0
    for piece, margin, tube in pieces:
        if piece not in sl or not tube:
            continue
        top = [i for i in _rim(V, sl[piece]) if V[i][1] > hem]
        if not top:
            continue
        R = V[top]
        live = np.unique(np.concatenate(out).reshape(-1))
        live = np.array([i for i in live if i in at], int)
        if not len(live):
            continue
        _, j = cKDTree(R[:, ::2]).query(V[live][:, ::2])
        want = R[j][:, 1]
        low = V[live][:, 1] < want
        for i, y in zip(live[low], want[low]):
            v = list(b.vertices[int(i)])
            v[0] = [v[0][0], float(y), v[0][2]]
            b.vertices[int(i)] = tuple(v)
            raised += 1
    print(f'  {body}: {cut} of {len(tris)} triangles cut out from under '
          f'{", ".join(x[0] for x in pieces)} ({int(under.sum())} of {len(ids)} vertices lie under them)'
          + (f'; {raised} left on the hem brought up to it' if raised else ''))


def reshoe(b, a):
    """Her own shoes back, and the model down to stand in them."""
    import dropmaterial
    for i, v in enumerate(b.vertices):
        w = list(v)
        w[0] = [v[0][0], v[0][1] - LIFT, v[0][2]]
        b.vertices[i] = tuple(w)
    for x in b.bones:
        x['pos'] = [x['pos'][0], x['pos'][1] - LIFT, x['pos'][2]]
    for r in b.rigids:
        r['pos'] = [r['pos'][0], r['pos'][1] - LIFT, r['pos'][2]]
    for j in b.joints:
        j['pos'] = [j['pos'][0], j['pos'][1] - LIFT, j['pos'][2]]

    asl = holefill.slices(a)
    AV = np.array([v[0] for v in a.vertices])
    an = [x['name'] for x in a.bones]
    on = {x['name']: i for i, x in enumerate(b.bones)}
    tex = {}
    for i, rel in enumerate(a.textures):
        if rel not in b.textures:
            b.textures.append(rel)
        tex[i] = b.textures.index(rel)

    F = np.array(b.faces).reshape(-1, 3)
    at, out, mats = 0, [], []
    for mat in b.materials:
        n = mat['fcount'] // 3
        if mat['name'] not in SHED:
            out.append(F[at:at + n])
            mats.append(mat)
        at += n
    dropped = len(b.materials) - len(mats)

    take = {src: asl[src] for src, _ in SHOES if src in asl}
    used = np.unique(np.concatenate(list(take.values())))
    fresh = {}
    for i in used:
        v = a.vertices[int(i)]
        w = list(v)
        w[5] = [on.get(an[x], 0) if 0 <= x < len(an) else 0 for x in v[5]]
        b.vertices.append(tuple(w))
        fresh[int(i)] = len(b.vertices) - 1
    for src, want in SHOES:
        if src not in take:
            continue
        mat = dict(next(x for x in a.materials if x['name'] == src))
        mat['name'] = mat['name_en'] = want
        mat['tex'] = tex.get(mat['tex'], -1)
        mat['sph'] = tex.get(mat['sph'], -1)
        if not mat['toon_shared']:
            mat['toon'] = tex.get(mat['toon'], -1)
        mat['fcount'] = len(take[src]) * 3
        mats.append(mat)
        out.append(np.array([[fresh[int(x)] for x in t] for t in take[src]], int))
        print(f'  {len(take[src])} triangles of her own `{want}` came back')
    b.materials = mats
    b.faces = np.concatenate(out).ravel().tolist()
    print(f'  dropped {dropped}, and lowered the model by {LIFT:.6f} to stand in them')


def main():
    a, b = PMX(PLACED), PMX(FITTED)
    AV = np.array([v[0] for v in a.vertices])
    BV = np.array([v[0] for v in b.vertices])
    AU = np.array([v[2] for v in a.vertices])
    BU = np.array([v[2] for v in b.vertices])
    ai = np.unique(holefill.slices(a)[DRESS])
    bi = np.unique(holefill.slices(b)[DRESS])
    m, lost = pair(a, ai, b, bi, AU, BU, AV, BV)
    print(f'  paired {len(m)} of {len(bi)} vertices by UV then position'
          + (f', {lost} unmatched' if lost else ''))

    # Ours is the fitted one. It keeps its bust and takes the other's placement
    # everywhere else, crossed over so there is no edge between the two.
    moved = 0
    for i, j in m.items():
        p = BV[i]
        w = (np.clip((p[1] - BUST['ylo']) / BUST['fade'], 0, 1)
             * np.clip((BUST['yhi'] - p[1]) / BUST['fade'], 0, 1)
             * np.clip((BUST['zmax'] - p[2]) / BUST['fade'], 0, 1))
        if w >= 0.999:
            continue
        v = list(b.vertices[i])
        q = AV[j] + np.array([0.0, LIFT, 0.0])
        v[0] = [float(x) for x in p * w + q * (1 - w)]
        b.vertices[i] = tuple(v)
        moved += 1
    print(f'  {moved} vertices took the placed build\'s position, '
          f'{len(m) - moved} kept the fitted bust')
    carry(b, BV, m)
    reshoe(b, a)
    unjam(b)
    intoshoe(b, "skin")
    cling(b, piece=SOCK, offset=SOCK_CLING)
    intoshoe(b, SOCK)
    lift(b)
    sink(b)
    cling(b)
    lift(b, piece=UNDER, offset=PROUD)
    outof(b, UNDER)
    strip(b)
    write_pmx(OUT, b)
    V = np.array([v[0] for v in b.vertices])
    print(f'{OUT}: {len(b.faces) // 3} triangles')


if __name__ == '__main__':
    main()
