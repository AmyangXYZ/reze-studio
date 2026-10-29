# -*- coding: utf-8 -*-
"""Put 家政小帮手's maid outfit on reze-midnight.

`bunnybuild.py`'s method against a second pair of models. The host's body, head,
hair, 110 morphs and foot pose are kept; only the cloth arrives.

The host is chosen for the heels. 家政小帮手 is authored standing in them and so is
午夜沉沦 -- their ankles sit at y 1.487 and 1.540, their toes within 0.244 -- so the
shoe lands on a foot already held at the angle it was built around. Against
`reze.pmx`, which stands flat, the same shoe has to be rotated onto the foot and the
ankle comes apart.

Placement reads the two BODIES (`bodyfield.py`), never the two garments. Hers is
`skin` plus `stocking`, because her skin stops at y 9.17 -- below that her legs *are*
the hosiery.

Three things this pair needs that the bunny did not:

**The names.** Both models number a skirt `skirt_<panel>_<link>`, and 159 of the
maid's chain bones collide with 午夜沉沦's own. A chain cannot arrive under a name
that is already here, so the host's are renamed first. They are dead by then -- every
one of them drives only cloth that is about to be dropped -- and nothing in a PMX
refers to a bone by name, so renaming costs nothing. The 33 that collide and are NOT
dead are the shared body bones (`上半身`, `左腕`, `左胸`, `頭`); those keep their
names and the dress rides them.

**The material.** `cloth01` is the whole outfit in one slice -- headdress, dress,
apron, cuffs and shoes -- and the three are not placed the same way, so it is cut by
shell into the three before anything is carried.

**The head.** The body field's anchors stop at her jaw, so a headdress read off it is
placed by extrapolating the neck. Both faces are the same 深空之眼 base head and an
ICP over the two `face01` meshes closes to 0.031, so everything above the collar
rides that fit instead.

    python3 tools/zzz2pmx/maidbuild.py

Reads `public/models/reze-midnight/reze-midnight.pmx` and never writes it. Output is
`public/models/reze-maid-midnight/reze-maid.pmx`, beside a `textures/` of its own.
"""
import os
import shutil
import sys
import tempfile

from collections import defaultdict

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pmx import PMX
from pmxw import write_pmx
from scipy.spatial import cKDTree

from surface import Mesh, push_out, transfer_weights
from graftlegs import rot_between
import align
import bodyfield
import chaingraft
import dropmaterial
import graftmaterial
import roundring as rr

BASE = 'public/models/reze-midnight/reze-midnight.pmx'
DONOR = 'public/models/女仆/家政小帮手.pmx'
OUT = 'public/models/reze-maid-midnight/reze-maid.pmx'
# The same model with its legs still whole, kept beside the trimmed one. Trimming is
# the last thing done and it is the one step that throws geometry away, so this is the
# copy to start from for a different outfit -- or to strip back to a bare body.
FULL = 'public/models/reze-maid-midnight/reze-maid-fullbody.pmx'
TEXDIR = 'public/models/reze-maid-midnight/textures'

# Hers, renamed: the host already has a `cloth01.png` and a `cloth01_alpha.png` of
# its own and they are different sheets.
TEXTURES = {'cloth01.png': 'maid_cloth.png',
            'cloth01_alpha.png': 'maid_alpha.png',
            'skin_stocking.png': 'maid_under.png',
            'skin_metal.png': 'maid_metal.png'}

HER_BODY = ['skin', 'stocking']

# The host's own `cloth01` is kept, and counted as body. Its skin has no front to the
# breasts -- two 24-edge loops at y 15.93..17.39, where its bra sat, because in this
# family of models the bra IS the breast there -- so dropping it opens the chest. The
# donor has no such hole, and her dress covers 88% of her bust against our 85%, which
# says the maid bodice was never the problem: the body under it was. Counted as body
# from the start, the field, the clearance and the bodice conform all read one
# continuous surface instead of fitting the dress to a hole.
OUR_BODY = ['skin', 'cloth01']
CLEAR_AGAINST = ['skin', 'cloth01']

# In her frame. Above the first, `cloth01` is the headdress; below the second, the
# shoe. Every shell falls cleanly on one side -- the lowest headdress shell starts at
# y 18.01 and the highest shoe shell ends at 1.94.
HEAD_Y = 17.9
FOOT_Y = 3.0

SPLIT = ['cloth01', 'metal']

# The outfit. `clear` is how far the piece has to stand off our skin. `rigid`
# resolves a piece as one assembly under a single scale: a buckle is not cloth.
# `by` is what places it -- the body field, the foot frame, or the head fit.
#
# `blur` is how wide the lift is smeared before it is applied. Draped cloth wants it
# wide, because moving a buried vertex while its neighbour stays put is what tears a
# seam. Hosiery does not: it is skin-tight over a leg 0.25 thicker than the one it
# was cut for, so the disagreement it has to resolve is per-vertex, and at 0.35 it
# cannot -- 158 of its 974 vertices stayed inside the leg through twenty rounds and
# came out as the pink streaking down both shins.
# `above` holds the clearance pass off a piece below that height, in OUR frame. The
# hosiery below the vamp is placed by the shoe's frame on purpose; pushing it out of
# the host's bare foot afterwards is exactly the inflation the frame was there to
# avoid, and the foot inside the shoe is not something anyone sees.
# The name is what the studio's auto-classifier buckets on (`lib/materials.ts`,
# CLASSIFY_KEYWORDS), so each piece is named for the shader it wants: `metal` for the
# hardware, `underwear`/`lace` for smooth cloth. Five of these used to match nothing
# and fell through to the neutral BSDF.
OUTFIT = [
    # hers,            ours,             sheet,             clear, rigid, by,     blur, above
    ('cloth01',        'maid_dress',     'maid_cloth.png',  0.020, False, 'body', 0.35, None),
    ('cloth01_head',   'maid_headdress', 'maid_cloth.png',  None,  True,  'head', 0.35, None),
    ('cloth01_foot',   'maid_shoes',     'maid_cloth.png',  0.010, True,  'foot', 0.35, None),
    ('metal',          'maid_metal',     'maid_cloth.png',  0.020, True,  'body', 0.35, None),
    ('metal_foot',     'maid_shoe_metal','maid_cloth.png',  0.010, True,  'foot', 0.35, None),
    ('stocking',       'maid_stockings', 'maid_cloth.png',  None,  False, 'sock', 0.12, 2.30),
    ('cloth01_alpha',  'maid_lace',      'maid_alpha.png',  0.020, False, 'body', 0.35, None),
    ('skin_stocking',  'maid_underwear', 'maid_under.png',  0.035, False, 'body', 0.12, None),
    ('skin_metal',     'maid_garter_metal','maid_metal.png',0.015, True,  'body', 0.35, None),
]

# Her frame: the shoe tops out at y 1.94, the calf is clear of it by 4.0.
SOCK_BAND = (1.90, 4.00)

# Her `skin_cloth` is left behind. It is her own collarbone under the neckline, and
# the host has one there already; brought across it is a second skin a hundredth off
# the first.
CHAIN_MATERIALS = [m for m, _, _, _, _, _, _, _ in OUTFIT]

# The shoe rides these instead of the body field, for the reason `BoneFrame` gives.
FOOT = ['足首', '足首D', 'つま先', '足先EX']

# The collar, widened. Measured across the bone in the band just above the shoe, this
# ankle is 1.297 wide against the donor's and 1.088 deep, against the 1.0916 the shoe
# is scaled by -- so the collar is 19% too narrow while the sole, over a foot only
# 2.5% wider, is right. `pmx_dressup` allows a thickness between 0.9 and 1.2 of the
# length scale and 1.297 sits inside that. Depth needs nothing: 1.088/1.0916 = 0.997.
SHOE_FLARE = 1.188      # extra width at the collar, on top of the uniform scale
SHOE_FLARE_BAND = (0.80, 2.24)   # our frame: none at the sole, all of it by the collar

# Ours, replaced by hers. The body, head, hair and eyes stay.
DROP = ['cloth01', 'metal01', 'cloth01_shoes', 'cloth01_alpha1', 'cloth01_alpha2',
        'metal01_2', 'gem01', 'cloth01_skin', 'cloth01_metal01', 'cloth01_metal02']

KEEP = ['skin', 'cloth01', '头发', 'face01', '齿', '口腔', '舌', '唇', 'eyebrow',
        'eyelash', 'eyelash_crease', '目白', '瞳1', '瞳2']

# Both are about 午夜沉沦 rather than about the outfit, and a host that does not need
# them turns them off. `reze-midnight` has a hole where its breasts should be and
# seven hair ornaments hanging in the air behind a bob; a 塞勒涅 in a swimsuit has
# neither, and the maid bodice covers the holes she does have.
REBUILD_BUST = True
HAIR_SHELLS = 'cloth01'


def slices(m):
    """Each material's triangles, in draw order."""
    F = np.array(m.faces).reshape(-1, 3)
    at, out = 0, {}
    for mat in m.materials:
        n = mat['fcount'] // 3
        out[mat['name']] = F[at:at + n]
        at += n
    return out


def leaning_on(m, mats):
    """The bone names the given materials actually lean on."""
    names = [b['name'] for b in m.bones]
    sl = slices(m)
    used = set()
    for tris in (sl[k] for k in mats if k in sl):
        for vi in np.unique(tris):
            v = m.vertices[int(vi)]
            for bi, w in zip(v[5], v[6]):
                if w > 0.001 and 0 <= bi < len(names):
                    used.add(names[bi])
    return used


# Taken off the waist and hip on top of the tightness match. By radius these are
# already inside the donor's scaled equivalents, so nothing measured asks for it --
# it is a look: the skirt hangs on the collider, and this one wants to sit closer to
# the body than the arithmetic puts it.
WAIST_IN = {'下半身': 0.12, '上半身2': 0.12}


def drop_hair_shells(path, material='cloth01', prefixes=('hair',), cut=0.5):
    """Take the host's hair ornaments out of the material we keep for its body.

    `cloth01` is kept whole because the breast is in it, and the rest of it rides
    along -- including seven shells hanging off `hair_B_5_1` and `hair_B_6_1` at
    z 3.01..3.46, an ornament for hair that is a waist-length braid on the model this
    body came from and a short bob on the one wearing it now. Nothing holds them up,
    so they sit in the air behind her.

    `graftmaterial` applies this rule to everything it carries from the donor; the
    kept host material never passed through it.
    """
    m = PMX(path)
    sl = slices(m)
    if material not in sl:
        return
    V = np.array([v[0] for v in m.vertices])
    names = [b['name'] for b in m.bones]

    def on_hair(i):
        v = m.vertices[int(i)]
        top = v[5][int(np.argmax(v[6][:len(v[5])]))]
        return 0 <= top < len(names) and names[top].lower().startswith(prefixes)

    gone = set()
    for ids in rr.shells(V, sl[material]).values():
        if sum(on_hair(i) for i in ids) / len(ids) > cut:
            gone |= {int(i) for i in ids}
    if not gone:
        print(f'  no hair-borne shells in {material}')
        return
    F = np.array(m.faces).reshape(-1, 3)
    at, out, cut_n = 0, [], 0
    for mat in m.materials:
        k = mat['fcount'] // 3
        tris = F[at:at + k]
        at += k
        if mat['name'] == material:
            keep = np.array([not any(int(x) in gone for x in t) for t in tris])
            cut_n = int((~keep).sum())
            tris = tris[keep]
            mat['fcount'] = len(tris) * 3
        out.append(tris)
    m.faces = np.concatenate(out).ravel().tolist()
    write_pmx(path, m)
    print(f'  dropped {cut_n} {material} triangles on {len(gone)} vertices that hang '
          f'from the old hair')


# 午夜沉沦 has no breast. `skin` stops at the dome and leaves a hole -- its rim runs
# z -2.45..-1.35, right where the apex belongs -- and `cloth01`'s bra covers the front
# of it. Welded shut, the cup keeps the bra's shape, and it facets, because the chest
# it is welded into is built at 0.24 edges.
#
# 塞勒涅 is the same 深空之眼 base body with a complete one: her `皮肤` carries the
# whole dome and breaks only at the under-bust, where her corset takes over. So the
# shape is read off her and rebuilt in our own skin -- our material, our atlas, our
# rig, her curve. None of her mesh arrives. A patch of it would bring her UVs along
# and seam straight across the upper chest, which is the one part of this that
# nothing covers.
REF = 'public/models/塞勒涅/塞勒涅.pmx'
REF_BODY = ['皮肤']
REF_BONES = ('上半身', '上半身2', '首', '頭', '左肩', '右肩', '左腕', '右腕', '下半身')

BUST_CUT = dict(rx=2.00, yc=16.55, ry=1.45, zcut=-0.55)
BUST_LEVELS = 2         # refinements inside the region: 0.24 edges -> 0.06
BUST_HOLD = 0.70        # within this much of the region the dome is hers outright
BUST_SMOOTH = 6         # blur passes on the displacement, so the rim cannot step
BUST_MARGIN = 0.06      # how clear of the rim one of her vertices has to land
BUST_REACH = 0.90       # a vertex further than this from her surface is not hers


def _bust_r(P, cut=BUST_CUT):
    """Where a point sits in the bust region: 0 at the centre, 1 at the rim."""
    P = np.atleast_2d(np.asarray(P, float))
    return np.sqrt((P[:, 0] / cut['rx']) ** 2 + ((P[:, 1] - cut['yc']) / cut['ry']) ** 2)


def _bust_in(P, cut=BUST_CUT):
    P = np.atleast_2d(np.asarray(P, float))
    return (_bust_r(P, cut) < 1.0) & (P[:, 2] < cut['zcut'])


def ref_bust(host, path=REF, mats=REF_BODY, bones=REF_BONES, cut=BUST_CUT):
    """Her bare torso, brought onto ours by the ribs around the bust.

    Fitted on the ring and never on the dome. The dome is the one thing being read
    off her, and an ICP allowed to see it closes the gap from both ends and hands
    back a blend of the two chests instead of hers.
    """
    r = PMX(path)
    RV = np.array([v[0] for v in r.vertices])
    sl = slices(r)
    T = np.concatenate([sl[k] for k in mats if k in sl])
    rb = {b['name']: np.array(b['pos']) for b in r.bones}
    hb = {b['name']: np.array(b['pos']) for b in host.bones}
    common = [n for n in bones if n in rb and n in hb]
    s, R, t = bodyfield._umeyama(np.array([rb[n] for n in common]),
                                 np.array([hb[n] for n in common]))
    P = s * (R @ RV.T).T + t

    hsl = slices(host)
    HV = np.array([v[0] for v in host.vertices])
    HT = np.concatenate([hsl[k] for k in OUR_BODY if k in hsl])

    def ring(Q):
        d = _bust_r(Q, cut)
        keep = ((d > 1.0) & (d < 1.9) & (Q[:, 2] < 0.3) & (np.abs(Q[:, 0]) < 3.0)
                & (np.abs(Q[:, 1] - cut['yc']) < 2.6))
        return Q[keep]

    tree = cKDTree(ring(HV[np.unique(HT)]))
    ri = np.unique(T)
    for _ in range(30):
        Q = ring(P[ri])
        if len(Q) < 20:
            break
        d, j = tree.query(Q)
        keep = d < max(np.percentile(d, 80), 1e-9)
        s2, R2, t2 = bodyfield._umeyama(Q[keep], tree.data[j[keep]])
        P = s2 * (R2 @ P.T).T + t2
        s *= s2
    d, _ = tree.query(ring(P[ri]))
    print(f'  {os.path.basename(path)}: fitted on {len(common)} bones then on the ribs, '
          f'scale {s:.4f}, ring residual median {np.median(d):.3f} '
          f'p90 {np.percentile(d, 90):.3f}')
    return Mesh(P, T)


def _mix(m, ids, ws):
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


def _bridge(A, B, V):
    """Triangles zipping one ring to the next, whatever their lengths.

    Advances down whichever side gives the shorter diagonal, so a 24-vertex ring
    meets a 19-vertex one with nothing long and thin between them.
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


def _inside(poly, P):
    """Even-odd point-in-polygon, and the distance to the nearest edge."""
    n = len(poly)
    a, b = poly, np.roll(poly, -1, 0)
    x, y = P[:, :1], P[:, 1:2]
    cross = ((a[:, 1] > y) != (b[:, 1] > y)) & (
        x < a[:, 0] + (b[:, 0] - a[:, 0]) * (y - a[:, 1])
        / np.where(np.abs(b[:, 1] - a[:, 1]) < 1e-12, 1e-12, b[:, 1] - a[:, 1]))
    ab = b - a
    t = np.clip(((P[:, None, :] - a) * ab).sum(-1)
                / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    d = np.linalg.norm(P[:, None, :] - (a + t[..., None] * ab), axis=-1).min(1)
    return cross.sum(1) % 2 == 1, d


def fill_bust(m, ref, name='skin', cut=BUST_CUT, margin=BUST_MARGIN):
    """Close the hole the bra was covering, with her breast.

    Rim edges are found against welded positions, not against vertex numbers. The
    chest is cut by a UV seam and the two sides of a seam are different vertices at
    the same point, so counted by number every seam edge reads as a hole -- 48 real
    rim edges came back as 112 that way.

    Her vertices fill it, triangulated flat against the plane the hole faces. Two
    ways of building the fill out of our own rim came apart first, both for the same
    reason: the loop is nowhere near flat -- one breast's runs z -2.45..-1.35 -- so
    neither clipping ears off it nor shrinking it toward its own centre stays on the
    dome, and the triangles cross in space. 1202 pairs of neighbouring normals ended
    up more than 60 degrees apart, against 4 anywhere in the source chest.

    Seen down the axis the hole faces, though, the breast IS a height field, so a
    Delaunay triangulation in that plane cannot fold, and lifting each point back to
    the vertex it came from puts her dome in the hole at her own density. UVs and
    weights are carried in from the rim: there is nothing else to read them off,
    because the atlas has no island under a hole.
    """
    from scipy.spatial import Delaunay
    sl = slices(m)
    V = np.array([v[0] for v in m.vertices])
    tris = sl[name]
    _, weld = np.unique(np.round(V / 1e-4).astype(np.int64), axis=0, return_inverse=True)
    count, seen = defaultdict(int), {}
    for a, b, c in tris:
        for p, q in ((a, b), (b, c), (c, a)):
            k = (min(weld[p], weld[q]), max(weld[p], weld[q]))
            count[k] += 1
            seen.setdefault(k, (int(p), int(q)))
    rim = [k for k, n in count.items() if n == 1
           and _bust_in(V[list(seen[k])], cut).all()]
    if not rim:
        print('  chest already closed')
        return tris, np.zeros(len(tris), bool)
    # Walked in the direction the fill runs: the rim edge reads p->q on the face that
    # owns it, so the hole reads q->p.
    step, new = {}, []
    for k in rim:
        p, q = seen[k]
        step[int(weld[q])] = (int(weld[p]), int(q))
    left = dict(step)
    hers = np.unique(ref.F)

    while left:
        at = next(iter(left))
        loop = []
        while at in left:
            nxt, a = left.pop(at)
            loop.append(a)
            at = nxt
        if len(loop) < 3:
            continue
        n0, n = len(new), len(loop)
        P = V[loop]
        c = P.mean(0)
        axis = sum(np.array(m.vertices[int(i)][1], float) for i in loop)
        axis /= np.linalg.norm(axis)
        e1 = np.cross(axis, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        flat = lambda Q: np.stack([(Q - c) @ e1, (Q - c) @ e2], 1)
        poly = flat(P)
        mine = ref.V[hers]
        keep, d = _inside(poly, flat(mine))
        keep &= d > margin
        keep &= (mine - c) @ axis > -0.25      # her surface, not her back
        inner = hers[keep]
        pts = np.concatenate([poly, flat(ref.V[inner])])
        ids = list(loop)
        for k, i in enumerate(inner):
            w = 1.0 / np.maximum(np.linalg.norm(poly - pts[n + k], axis=1), 1e-6) ** 2
            m.vertices.append(_mix(m, loop, w))
            j = len(m.vertices) - 1
            v = list(m.vertices[j])
            v[0] = [float(x) for x in ref.V[i]]
            v[1] = [float(x) for x in ref.normal[np.where(ref.F == i)[0][0]]]
            m.vertices[j] = tuple(v)
            ids.append(j)
        tri2 = Delaunay(pts).simplices
        good, _ = _inside(poly, pts[tri2].mean(1))
        out = np.array([v[0] for v in m.vertices])
        for t3 in tri2[good]:
            a, b, cc = (ids[int(x)] for x in t3)
            if np.dot(np.cross(out[b] - out[a], out[cc] - out[a]), axis) < 0:
                a, b, cc = a, cc, b
            new.append([a, b, cc])
        print(f'  filled a {n}-edge hole in the chest at y {P[:, 1].mean():.2f} with '
              f'{len(inner)} of her vertices, {len(new) - n0} triangles')
    return (np.concatenate([tris, np.array(new, int)]),
            np.concatenate([np.zeros(len(tris), bool), np.ones(len(new), bool)]))


def refine_bust(m, tris, fill, cut=BUST_CUT, levels=None):
    """Split the region into four, and keep its border watertight.

    A subset split on its own leaves T-junctions around it and those open as cracks
    under any deformation, so a face that straddles the border is split to match the
    edges its neighbour split: three marked edges make four triangles, two make
    three, one makes two.

    Marked against welded positions. The chest carries a UV seam, and across a seam
    the two faces sharing an edge do not share its vertex NUMBERS -- so matched by
    number the far side never hears that the edge was split, and the seam came out as
    10 rim edges of genuine crack.
    """
    levels = BUST_LEVELS if levels is None else levels
    for _ in range(levels):
        V = np.array([v[0] for v in m.vertices])
        _, weld = np.unique(np.round(V / 1e-4).astype(np.int64), axis=0,
                            return_inverse=True)
        key = lambda i, j: (min(int(weld[i]), int(weld[j])),
                            max(int(weld[i]), int(weld[j])))
        C = V[tris].mean(1)
        mark = set()
        for t in tris[_bust_in(C, cut)]:
            for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                mark.add(key(p, q))
        cache = {}

        def mid(i, j):
            k = (min(int(i), int(j)), max(int(i), int(j)))
            if k not in cache:
                cache[k] = len(m.vertices)
                m.vertices.append(_blend_vertex(m, k[0], k[1]))
            return cache[k]

        out, tag = [], []
        for t, f in zip(tris, fill):
            a, b, c = (int(x) for x in t)
            e = [key(a, b) in mark, key(b, c) in mark, key(c, a) in mark]
            n = sum(e)
            if n == 0:
                out.append([a, b, c])
                tag.append(f)
                continue
            if n == 3:
                ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
                out += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
                tag += [f] * 4
                continue
            for _r in range(3):     # rotate so the marked edges start at a-b
                if (n == 1 and e[0]) or (n == 2 and e[0] and e[1]):
                    break
                a, b, c = b, c, a
                e = [e[1], e[2], e[0]]
            if n == 1:
                ab = mid(a, b)
                out += [[a, ab, c], [ab, b, c]]
                tag += [f] * 2
            else:
                ab, bc = mid(a, b), mid(b, c)
                out += [[a, ab, bc], [ab, b, bc], [a, bc, c]]
                tag += [f] * 3
        tris, fill = np.array(out, int), np.array(tag, bool)
    return tris, fill


def _blur_over(D, tris, index, rounds):
    """Average a per-vertex quantity over the surface it lives on."""
    nbr = defaultdict(set)
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            nbr[int(u)].add(int(v))
            nbr[int(v)].add(int(u))
    order = [None] * len(index)
    for i, k in index.items():
        order[k] = np.array([index[j] for j in nbr.get(i, ()) if j in index], int)
    for _ in range(rounds):
        out = D.copy()
        for k, ns in enumerate(order):
            if len(ns):
                out[k] = 0.5 * D[k] + 0.5 * D[ns].mean(0)
        D = out
    return D


def shape_bust(m, tris, ref, cut=BUST_CUT, hold=BUST_HOLD, reach=BUST_REACH,
               rounds=BUST_SMOOTH):
    """Take the region onto her dome, and fade it out before the rim.

    The displacement is smoothed over the region before it is applied and weighted to
    nothing at the rim, so the chest arrives at the untouched torso with its slope
    already matching. Applied raw it steps: the two bodies disagree by up to 0.5
    across the ribs and that disagreement would land as a ledge one edge wide.
    """
    V = np.array([v[0] for v in m.vertices])
    idx = np.unique(tris)
    r = _bust_r(V[idx], cut)
    sel = (r < 1.0) & (V[idx][:, 2] < cut['zcut'])
    here = idx[sel]
    d, t, bar = ref.closest(V[here])
    f = ref.F[t]
    target = (ref.V[f[:, 0]] * bar[:, :1] + ref.V[f[:, 1]] * bar[:, 1:2]
              + ref.V[f[:, 2]] * bar[:, 2:3])
    D = target - V[here]
    D[d > reach] = 0.0
    # Nearest point is the wrong correspondence inside the cleavage: the two breasts
    # face each other across a gap narrower than the distance to either surface's own
    # far side, so a vertex on the left takes its target off the right and the cleft
    # closes into a knife edge -- 1130 pairs of neighbouring normals more than 60
    # degrees apart, against 4 in the source chest. A target has to be facing the
    # same way as the vertex that claimed it.
    ours = np.array([m.vertices[int(i)][1] for i in here], float)
    turned = np.einsum('ij,ij->i', ours, ref.normal[t]) < 0.3
    D[turned] = 0.0
    index = {int(i): k for k, i in enumerate(here)}
    D = _blur_over(D, tris, index, rounds)
    # Both sides of the UV seam are separate vertices at the same point with
    # different neighbours, so the blur hands them different steps and the seam
    # opens: 48 rim edges on the chest became 114. They move together instead.
    _, weld = np.unique(np.round(V[here] / 1e-4).astype(np.int64), axis=0,
                        return_inverse=True)
    tot = np.zeros((weld.max() + 1, 3))
    np.add.at(tot, weld, D)
    D = tot[weld] / np.bincount(weld, minlength=weld.max() + 1)[weld][:, None]
    w = np.clip((1.0 - r[sel]) / max(1.0 - hold, 1e-6), 0.0, 1.0)
    w = w * w * (3 - 2 * w)
    V[here] += w[:, None] * D
    moved = np.linalg.norm(w[:, None] * D, axis=1)
    print(f'  {len(here)} chest vertices taken onto her dome, moved median '
          f'{np.median(moved):.3f} max {moved.max():.3f}; {int((d > reach).sum())} '
          f'were out of her reach and {int(turned.sum())} faced the wrong way')
    for i, k in index.items():
        v = list(m.vertices[i])
        v[0] = [float(x) for x in V[i]]
        m.vertices[i] = tuple(v)
    return here


def _renormal(m, tris, here, cut=BUST_CUT, hold=BUST_HOLD):
    """Rebuild the region's normals off the surface it now is.

    Accumulated against welded positions, so the UV seam down the chest does not come
    back as a crease: either side of it is a different vertex at the same point, and
    given only its own faces each side leans a different way.
    """
    V = np.array([v[0] for v in m.vertices])
    _, weld = np.unique(np.round(V / 1e-4).astype(np.int64), axis=0, return_inverse=True)
    acc = np.zeros((weld.max() + 1, 3))
    P = V[tris]
    fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    for k in range(3):
        np.add.at(acc, weld[tris[:, k]], fn)
    n = np.linalg.norm(acc, axis=1, keepdims=True)
    acc = np.where(n > 1e-12, acc / np.maximum(n, 1e-12), 0.0)
    r = _bust_r(V[here], cut)
    w = np.clip((1.0 - r) / max(1.0 - hold, 1e-6), 0.0, 1.0)
    w = w * w * (3 - 2 * w)
    done = 0
    for k, i in enumerate(here):
        new = acc[weld[i]]
        if not new.any():
            continue
        old = np.array(m.vertices[int(i)][1], float)
        mix = (1 - w[k]) * old + w[k] * new
        ln = np.linalg.norm(mix)
        if ln < 1e-9:
            continue
        v = list(m.vertices[int(i)])
        v[1] = [float(x) for x in mix / ln]
        m.vertices[int(i)] = tuple(v)
        done += 1
    print(f'  rebuilt {done} normals over the region')


def strip_bra(m, material='cloth01', cut=BUST_CUT):
    """Take the bra off. The skin under it is a breast now."""
    sl = slices(m)
    if material not in sl:
        return None
    V = np.array([v[0] for v in m.vertices])
    tris = sl[material]
    C = V[tris].mean(1)
    keep = ~_bust_in(C, cut)
    print(f'  dropped {int((~keep).sum())} of {len(tris)} {material} triangles '
          f'over the bust')
    return tris[keep]


def rebuild_bust(path, cut=BUST_CUT):
    """Give her a breast, in her own skin."""
    m = PMX(path)
    ref = ref_bust(m, cut=cut)
    skin, fill = fill_bust(m, ref, cut=cut)
    n0 = len(skin)
    skin, _ = refine_bust(m, skin, fill, cut=cut)
    print(f'  chest refined: {n0} skin triangles -> {len(skin)}')
    here = shape_bust(m, skin, ref, cut=cut)
    _renormal(m, skin, here, cut=cut)
    bra = strip_bra(m, cut=cut)
    F = np.array(m.faces).reshape(-1, 3)
    at, out = 0, []
    for mat in m.materials:
        k = mat['fcount'] // 3
        tris = F[at:at + k]
        at += k
        if mat['name'] == 'skin':
            tris = skin
        elif mat['name'] == 'cloth01' and bra is not None:
            tris = bra
        mat['fcount'] = len(tris) * 3
        out.append(tris)
    m.faces = np.concatenate(out).ravel().tolist()
    write_pmx(path, m)


BRA_AS = 'maid_bra'


def _weights_of(m):
    """Per-vertex {bone name: weight}, indexed by vertex."""
    names = [b['name'] for b in m.bones]
    out = []
    for v in m.vertices:
        d = {}
        for b, w in zip(v[5], v[6]):
            if w > 1e-5 and 0 <= b < len(names):
                d[names[b]] = d.get(names[b], 0.0) + float(w)
        out.append(d)
    return out


def _set_weights(m, i, wd, index):
    """Write a weight dict back as BDEF4, biggest four, renormalised."""
    top = sorted(((w, n) for n, w in wd.items() if n in index), reverse=True)[:4]
    tot = sum(w for w, _ in top)
    if tot <= 0:
        return False
    bones = [index[n] for _, n in top] + [0] * (4 - len(top))
    wts = [w / tot for w, _ in top] + [0.0] * (4 - len(top))
    rec = list(m.vertices[i])
    rec[4], rec[5], rec[6] = 2, bones, wts
    m.vertices[i] = tuple(rec[:8])
    return True


def reweight(path, piece, source='skin', region=None, exclude=None, reach=0.8):
    """Give a piece the body's own weights where it sits on the body.

    Geometry carried from another model keeps that model's weights, and at rest that
    is invisible -- every check passes and the seam only opens once a bone moves. The
    breast cup was the clear case: it leans on 胸 for 141 of its 252 vertices where
    the skin beside it uses 胸 for 43 of 102, and 胸 is driven by physics, so the two
    halves of one breast were being deformed by different amounts every frame.
    """
    m = PMX(path)
    sl = slices(m)
    if source not in sl:
        return
    V = np.array([v[0] for v in m.vertices])
    tris = sl[source]
    if exclude is not None:
        drop = set(int(i) for i in exclude)
        tris = np.array([t for t in tris if not any(int(x) in drop for x in t)])
    if not len(tris):
        return
    body = Mesh(V, tris)
    src = _weights_of(m)
    index = {b['name']: i for i, b in enumerate(m.bones)}

    ids = np.array(sorted(piece))
    if region is not None:
        ids = np.array([i for i in ids if region(V[i])])
    if not len(ids):
        return
    before = [dict(src[int(i)]) for i in ids]
    wts, d = transfer_weights(body, src, V[ids])
    done = 0
    for k, i in enumerate(ids):
        if d[k] > reach:
            continue
        if _set_weights(m, int(i), wts[k], index):
            done += 1
    after = _weights_of(m)
    diff = [0.5 * sum(abs(before[k].get(n, 0) - after[int(i)].get(n, 0))
                      for n in set(before[k]) | set(after[int(i)]))
            for k, i in enumerate(ids)]
    write_pmx(path, m)
    print(f'  reweighted {done} of {len(ids)} vertices off "{source}"; they moved '
          f'median {np.median(diff):.3f} max {np.max(diff):.3f} away from what they had')


def tighten_colliders(path, donor, host_mats, donor_mats, pct=90):
    """Bring our body's colliders in to the tightness hers are set at.

    A skirt hangs on the collider, not on the body, so how proud of the body that
    collider stands is how far the skirt is held out. Hers are tucked inside her: her
    torso skin sits p90 0.134 OUTSIDE her `上半身` capsules. Ours is inflated past the
    body -- one capsule of radius 1.20 with the skin 0.049 INSIDE it -- so the upper
    panels rest on something wider than the torso and the whole skirt flares.

    Only inward. Growing a collider to swallow more of the body was tried and it
    swallows the skirt's own bodies with it, which costs their collision entirely.
    """
    m = PMX(path)

    def outside(model, bone, mats, caps):
        sl = slices(model)
        ids = set()
        for k in mats:
            if k in sl:
                ids.update(sl[k].ravel().tolist())
        want = {i for i, b in enumerate(model.bones) if b['name'] == bone}
        P = np.array([model.vertices[i][0] for i in sorted(ids)
                      if any(b in want and w > 0.5
                             for b, w in zip(model.vertices[i][5], model.vertices[i][6]))])
        if not len(P) or not caps:
            return None
        best = np.full(len(P), 1e9)
        for c in caps:
            best = np.minimum(best, _capsule_sd(P, c))
        return float(np.percentile(best, pct))

    dcap, hcap = {}, {}
    for r in donor.rigids:
        if r['mode'] == 0 and r['shape'] == 2:
            dcap.setdefault(donor.bones[r['bone']]['name'], []).append(r)
    for r in m.rigids:
        if r['mode'] == 0 and r['shape'] == 2:
            hcap.setdefault(m.bones[r['bone']]['name'], []).append(r)

    for bone in sorted(set(dcap) & set(hcap)):
        want = outside(donor, bone, donor_mats, dcap[bone])
        have = outside(m, bone, host_mats, hcap[bone])
        if want is None or have is None or have >= want:
            continue
        shrink = want - have
        for r in hcap[bone]:
            r['size'] = [max(0.05, r['size'][0] - shrink), r['size'][1], r['size'][2]]
        now = outside(m, bone, host_mats, hcap[bone])
        print(f'  {bone}: skin stood {have:+.3f} outside, hers {want:+.3f}; '
              f'brought in {shrink:.3f} -> {now:+.3f}')

    for bone, extra in WAIST_IN.items():
        if bone not in hcap:
            continue
        was = [r['size'][0] for r in hcap[bone]]
        for r in hcap[bone]:
            r['size'] = [max(0.05, r['size'][0] - extra), r['size'][1], r['size'][2]]
        print(f'  {bone}: taken in a further {extra} for the skirt -- '
              f'{[round(x, 3) for x in was]} -> '
              f'{[round(r["size"][0], 3) for r in hcap[bone]]}')
    write_pmx(path, m)


def carried_closure(d, mats):
    """Every donor bone that will come across, not just the ones holding cloth.

    A chain carries links that no vertex leans on -- `skirt_0_3` holds nothing and
    exists to hang `skirt_1_3` off. Asking only what the materials lean on misses
    them, their names stay occupied by the host's own outfit, and `chaingraft` then
    declines to carry the donor's version and re-anchors the children onto the host's
    stale bone instead: 11 skirt chains hung off 午夜沉沦's old skirt, as much as 4
    units from where they belong.

    Closed the way `chaingraft.chains` closes -- over parents, and over the bodies a
    joint ties to one of ours -- so the two agree on what is being carried. Bones the
    host actually uses are filtered out by the caller, so closing to `全ての親` is safe.
    """
    names = [b['name'] for b in d.bones]
    index = {n: i for i, n in enumerate(names)}
    keep = {index[n] for n in leaning_on(d, mats) if n in index}
    onbone = {}
    for k, r in enumerate(d.rigids):
        onbone.setdefault(r['bone'], []).append(k)
    while True:
        grew = False
        for i in list(keep):
            p = d.bones[i]['parent']
            if p >= 0 and p not in keep:
                keep.add(p)
                grew = True
        mine = {k for i in keep for k in onbone.get(i, [])}
        for j in d.joints:
            for a, b in ((j['a'], j['b']), (j['b'], j['a'])):
                if a in mine and 0 <= b < len(d.rigids):
                    ob = d.rigids[b]['bone']
                    if 0 <= ob < len(names) and ob not in keep:
                        keep.add(ob)
                        grew = True
        if not grew:
            return {names[i] for i in keep}


class HeadFrame:
    """Everything above the collar, placed by the two faces rather than the two
    bodies.

    Both models are 深空之眼 and their `face01` is the same base head, so a uniform
    scale and an offset carry one onto the other and ICP finds them without being
    told. The bones are not used for it: a `頭` sits at whatever depth in a skull its
    author chose.
    """

    def __init__(self, donor, host, mat='face01', body='skin', collar=18.8):
        A = np.array([donor.vertices[int(i)][0]
                      for i in np.unique(slices(donor)[mat])])
        sl = slices(host)
        if mat in sl:
            B = np.array([host.vertices[int(i)][0] for i in np.unique(sl[mat])])
        else:
            # A host whose face is welded into its body has no `face01` to fit
            # against. Everything of it above the collar is the same head, which is
            # all this needs: it is solving where a headdress sits, not who is
            # wearing it.
            B = np.array([host.vertices[int(i)][0] for i in np.unique(sl[body])])
            B = B[B[:, 1] > collar]
            print(f'  no {mat} on the host; fitting the head on {len(B)} vertices '
                  f'of {body} above y {collar}')
        s, t, d = align.icp_similarity(A, B, 1.0, B.mean(0) - A.mean(0), iters=80)
        self.scale, self.shift = s, t
        print(f'  head fit: scale {s:.4f}, offset {np.round(t, 3)}, '
              f'residual median {np.median(d):.4f} max {d.max():.4f}')

    def __call__(self, P):
        return np.atleast_2d(np.asarray(P, float)) * self.scale + self.shift


class FootFrame:
    """The shoe, placed on the foot by the only things the foot bones determine.

    `bodyfield.BoneFrame` fits a similarity over 足首, 足首D, つま先 and 足先EX. On this
    pair those are three distinct points -- 足首D sits exactly on 足首 -- and the three
    are COLLINEAR on both models: the cross product of 足首->つま先 with 足首->足先EX is
    0.000000 either side. A roll about that line changes nothing about where the three
    land, so the fit is rank deficient, and `_umeyama` returns whatever the arithmetic
    says: 166.69 degrees on the left foot and 40.42 on the right, both about the foot
    axis. It reports residual 0.000 for both, which is true and says nothing --
    collinear points map exactly under any roll. The shoes arrived rolled off the
    feet, differently on each side.

    What the bones do determine is the scale: every pairwise distance among them is
    1.0916 longer here than on her.

    The 3.50 degrees between the two foot axes is not applied either, and the floor is
    the reason. Her shoe is authored standing on two contacts -- the platform under the
    ball at z -0.40..-0.10 and the heel at z +0.80..+1.40, both at y 0.0000 -- because
    that is what a platform heel does. Pitched onto the host's foot axis it stands on
    the heel alone, with the platform 0.08 clear of the ground and the toe 0.22, and
    grounding the sole then only rests the whole shoe on that one contact. Squared to
    the world it lands on both again, exactly as she authored it. The foot inside
    finishes 3.5 degrees off the last it was cut for, which is inside an opaque shoe;
    a shoe tilted off the floor is not.
    """

    def __init__(self, donor, host, heel='足首', toe='つま先', floor=None,
                 foot=None, flare=SHOE_FLARE, flare_band=SHOE_FLARE_BAND):
        dp = {b['name']: np.array(b['pos']) for b in donor.bones}
        tp = {b['name']: np.array(b['pos']) for b in host.bones}
        self.flare, self.flare_band = flare, flare_band
        self.sides = []
        for side in ('左', '右'):
            sign = 1.0 if side == '左' else -1.0
            a0, a1 = dp[side + heel], dp[side + toe]
            b0, b1 = tp[side + heel], tp[side + toe]
            u, v = a1 - a0, b1 - b0
            s = float(np.linalg.norm(v) / np.linalg.norm(u))
            # Yaw only. Her foot toes out 2.17 degrees and this one points straight
            # ahead, and squaring the shoe to the world keeps hers -- the forefoot
            # ends up 1.74 degrees across the foot it is on. A turn about the vertical
            # axis is the one rotation that cannot lift the sole, which is why the
            # pitch stays off and this does not.
            turn = (np.arctan2(v[0], -v[2]) - np.arctan2(u[0], -u[2]))
            R = np.array([[np.cos(turn), 0.0, np.sin(turn)],
                          [0.0, 1.0, 0.0],
                          [-np.sin(turn), 0.0, np.cos(turn)]])
            t = b0 - s * R @ a0
            ang = np.degrees(np.arccos(np.clip(
                u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
            lift = 0.0
            if floor is not None:
                # The scale is right along the foot and wrong across it: the host's
                # foot is 18.7% longer front to back and only 4.9% taller, so one
                # number for both sinks the sole 0.18 under the floor. Both models
                # are authored standing on y 0 and a sole is on the ground, so that
                # one degree of freedom is taken from the floor instead of the bones.
                P = floor[np.sign(floor[:, 0]) == sign]
                lift = -float((s * (R @ P.T).T + t)[:, 1].min())
                t = t + np.array([0.0, lift, 0.0])
            # Centred on the foot, not on the ankle bone. The bone is a landmark of
            # the rig -- it sits at x 0.781 while this foot's surface is centred on
            # 0.855 -- so a shoe hung on it lands 0.094 inboard, with the leg riding
            # the outer wall of it and a gap on the inside.
            cx = float(tp[side + heel][0])
            if floor is not None and foot is not None:
                shoe = floor[np.sign(floor[:, 0]) == sign]
                here = foot[np.sign(foot[:, 0]) == sign]
                if len(shoe) and len(here):
                    q = s * (R @ shoe.T).T + t
                    off = np.abs(here[:, 0]).mean() - np.abs(q[:, 0]).mean()
                    t = t + np.array([sign * off, 0.0, 0.0])
                    cx = float(np.abs(here[:, 0]).mean()) * sign
                    print(f'  {side}: shoe moved {off:+.3f} outboard onto the foot')
            self.sides.append((sign, s, R, t, cx))
            print(f'  {side}: scale {s:.4f}, yaw {np.degrees(turn):+.2f} deg, no '
                  f'pitch ({ang:.2f} deg of it left on the foot), sole lifted '
                  f'{lift:+.3f} to the floor')
        self.scale = float(np.mean([s for _, s, _, _, _ in self.sides]))

    def __call__(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        out = P * self.scale
        lo, hi = self.flare_band
        for sign, s, R, t, cx in self.sides:
            m = np.sign(P[:, 0]) == sign
            if not m.any():
                continue
            q = s * (R @ P[m].T).T + t
            # Widened about the ankle's own centre line, and only as the collar is
            # approached: the sole and the toe box already fit the foot.
            k = 1.0 + (self.flare - 1.0) * np.clip((q[:, 1] - lo) / (hi - lo), 0, 1)
            q[:, 0] = cx + (q[:, 0] - cx) * k
            out[m] = q
        return out


class Blend:
    """One placement below, another above, mixed over a band in her frame.

    The hosiery runs from inside the shoe to the top of the thigh, and its two ends
    want different maps. Inside the shoe it has to keep the fit its author gave it,
    which is the shoe's own frame. Up the calf it has to follow a leg that is not
    hers, which is the body field.

    Carried entirely by the field -- which is what it was -- the foot inflates onto
    the host's bare foot and spills over the vamp, because the host's foot is the
    larger of the two: she deleted the part of hers the shoe hides, so the surface the
    field lands it on is a whole foot where hers was a fragment. Carried entirely by
    the shoe's frame instead, the calf comes out too thin for the leg inside it.

    Mixed linearly, so the hosiery stays one surface across the ankle.
    """

    def __init__(self, below, above, lo, hi):
        self.below, self.above, self.lo, self.hi = below, above, lo, hi
        self.scale = above.scale

    def __call__(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        w = np.clip((P[:, 1] - self.lo) / (self.hi - self.lo), 0.0, 1.0)[:, None]
        return self.below(P) * (1.0 - w) + self.above(P) * w


class Router:
    """The body field for cloth on the body, the head fit for what sits on the head.

    A hard switch, which is safe here because it is only ever handed points that are
    all on one side of it: the mesh arrives already cut into `cloth01` and
    `cloth01_head`, and the four chain bones above the collar -- the ribbons on her
    braid -- are two accessories of their own.
    """

    def __init__(self, field, head, cut):
        self.field, self.head, self.cut = field, head, cut
        self.scale = field.scale

    def __call__(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        out = self.field(P)
        up = P[:, 1] > self.cut
        if up.any():
            out[up] = self.head(P[up])
        return out


def split(src, dst):
    """Cut `cloth01` and `metal` into what sits on the head, what sits on the foot,
    and the rest, keeping every other property of the material.

    Her whole outfit is one slice of the face list. A shoe is hardware built around a
    foot held at an angle and a headdress is placed off neither body nor foot, so the
    three have to be separable before anything is carried.
    """
    m = PMX(src)
    V = np.array([v[0] for v in m.vertices])
    F = np.array(m.faces).reshape(-1, 3)
    at, faces, mats = 0, [], []
    for mat in m.materials:
        n = mat['fcount'] // 3
        tris = F[at:at + n]
        at += n
        if mat['name'] not in SPLIT:
            faces.append(tris)
            mats.append(mat)
            continue
        where = {}
        for ids in rr.shells(V, tris).values():
            P = V[np.array(ids)]
            k = ('head' if P[:, 1].min() > HEAD_Y else
                 'foot' if P[:, 1].max() < FOOT_Y else 'body')
            for i in ids:
                where[int(i)] = k
        for k, suffix in (('body', ''), ('head', '_head'), ('foot', '_foot')):
            t = tris[np.array([where[int(f[0])] == k for f in tris])]
            if not len(t):
                continue
            mm = dict(mat)
            mm['name'] = mat['name'] + suffix
            mm['name_en'] = mat['name_en'] + suffix
            mm['fcount'] = len(t) * 3
            faces.append(t)
            mats.append(mm)
            print(f'  {mm["name"]:14s} {len(t):6d} tris')
    m.faces = np.concatenate(faces).ravel().tolist()
    m.materials = mats
    write_pmx(dst, m)


def free_names(path, donor_path):
    """Rename the host's own outfit bones out of the way of the maid's.

    Both models number a skirt the same, and `chaingraft` refuses to land a chain on
    a name that is already taken -- rightly, since the two `skirt_9_3`s are different
    garments 4.18 apart. The host's are dead: they drive only cloth this build drops.
    A PMX refers to a bone by index everywhere, so a rename is free.
    """
    m = PMX(path)
    d = PMX(donor_path)
    wanted = carried_closure(d, [x[0] for x in OUTFIT])
    alive = set(leaning_on(m, KEEP))
    idx = {b['name']: i for i, b in enumerate(m.bones)}
    for n in list(alive):                       # a driven bone keeps its ancestors
        i = idx.get(n, -1)
        while i >= 0:
            alive.add(m.bones[i]['name'])
            i = m.bones[i]['parent']
    renamed = 0
    for b in m.bones:
        if b['name'] in wanted and b['name'] not in alive:
            b['name'] = '午夜_' + b['name']
            renamed += 1
    keeping = sorted(wanted & alive)
    print(f'  renamed {renamed} of the host\'s own outfit bones out of the way')
    print(f'  {len(keeping)} shared body bones keep their names: '
          f'{", ".join(keeping[:8])}, ...')
    write_pmx(path, m)
    return set(b['name'] for b in m.bones)


def textures(path):
    """Copy her sheets in and register them, so `--tex` can name one."""
    os.makedirs(TEXDIR, exist_ok=True)
    base = os.path.join(os.path.dirname(DONOR), 'textures')
    m = PMX(path)
    for src, dst in TEXTURES.items():
        shutil.copyfile(os.path.join(base, src), os.path.join(TEXDIR, dst))
        rel = 'textures\\' + dst
        if rel not in m.textures:
            m.textures.append(rel)
    write_pmx(path, m)
    return {dst: 'textures\\' + dst for dst in TEXTURES.values()}


def _even(P0, P1, tol=0.25):
    """Mirror-symmetrise a displacement.

    `push_out` blurs its lift over a KD-tree of the points, and a KD-tree is not
    mirror symmetric -- the nearest twelve neighbours of a vertex on the left are not
    the mirror of its partner's on the right. The input is symmetric to 0.0000 and the
    output was not: the underwear came out 0.112 apart between the sides. Pairs are
    taken on the input, where they are unambiguous, and both ends of a pair get the
    average of the two displacements.
    """
    flip = np.array([-1.0, 1.0, 1.0])
    d = P1 - P0
    left = np.where(P0[:, 0] > 0.02)[0]
    right = np.where(P0[:, 0] < -0.02)[0]
    if len(left) > 3 and len(right) > 3:
        dist, j = cKDTree(P0[right] * flip).query(P0[left])
        for k in np.where(dist < tol)[0]:
            a, b = left[k], right[j[k]]
            avg = (d[a] + d[b] * flip) / 2
            d[a], d[b] = avg, avg * flip
    mid = np.where(np.abs(P0[:, 0]) <= 0.02)[0]
    d[mid, 0] = 0.0
    return P0 + d


def clear_body(path, own):
    """Lift each piece off our skin, once everything is in place.

    Her figure is the slimmer one at every band -- 0.25 narrower at the waist, and
    her bust stands 0.44 less proud -- so a piece carried across at her standoff sits
    inside ours, and the skin shows through in exactly that band.
    """
    m = PMX(path)
    mats = slices(m)
    V = np.array([v[0] for v in m.vertices])
    body = Mesh(V, np.concatenate([mats[k] for k in CLEAR_AGAINST]))
    names = [b['name'] for b in m.bones]
    hanging = set()
    for i, v in enumerate(m.vertices):
        for bi, w in zip(v[5], v[6]):
            if w > 0.2 and 0 <= bi < len(names) and names[bi] not in own:
                hanging.add(i)
    print(f'  {len(hanging)} vertices hang on a chain and are left alone')
    for _, name, _, clear, rigid, _, blur, above in OUTFIT:
        if clear is None or name not in mats:
            continue
        idx = np.array([i for i in np.unique(mats[name])
                        if int(i) not in hanging
                        and (above is None or V[int(i)][1] >= above)])
        if not len(idx):
            continue
        # Forty, not `push_out`'s six or `bunnybuild`'s twenty: the lift is relaxed
        # at 0.7 a round and blurred, so a deep vertex converges geometrically and
        # the dress still had nine inside our chest at twenty.
        moved, lifted = push_out(body, V[idx], offset=clear, rigid=rigid,
                                 blur=blur, rounds=40)
        moved = _even(V[idx], moved)
        print(f'  {name:15s} lifted {int(lifted.sum()):5d} of {len(idx):5d} clear by '
              f'{clear}, worst move '
              f'{np.linalg.norm(moved - V[idx], axis=1).max():.3f}')
        V[idx] = moved
    for i, v in enumerate(m.vertices):
        if not np.allclose(v[0], V[i]):
            w = list(v)
            w[0] = [float(x) for x in V[i]]
            m.vertices[i] = tuple(w)
    write_pmx(path, m)


# The leg under the hosiery is left whole. Cutting it is the cheap way to stop it
# showing through, and it throws away the thing any later pass needs to look at --
# once the two surfaces are merged, or the leg is cut by material rather than by
# proximity, this can come back.
TRIM_LEGS = True

SOCK = 'maid_stockings'
SOCK_FLOOR = 1.50       # our frame: the fade ends at 2.20, under a collar topping at 2.235
SOCK_HEM = 0.25         # leave the lace at the top its own relief
SOCK_LACE = 4.80        # our frame: only a rim above this is the lace edge
SOCK_OFFSET = 0.020     # at the thigh
SOCK_ANKLE = 0.012      # at the ankle, where the shoe's collar is already tight
SOCK_TAPER = (2.60, 5.00)
SOCK_SMOOTH = 8         # Taubin passes; fabric spans a bone, it does not wrap it


def _blend_vertex(m, i, j):
    """A new vertex halfway along the edge i-j: position, normal, UV and skin."""
    a, b = m.vertices[i], m.vertices[j]
    w = {}
    for v in (a, b):
        for bi, ww in zip(v[5], v[6]):
            if ww > 0:
                w[bi] = w.get(bi, 0.0) + 0.5 * float(ww)
    top = sorted(w.items(), key=lambda x: -x[1])[:4]
    tot = sum(x[1] for x in top) or 1.0
    bones = [x[0] for x in top] + [0] * (4 - len(top))
    wts = [x[1] / tot for x in top] + [0.0] * (4 - len(top))
    mid = lambda k: [(p + q) / 2 for p, q in zip(a[k], b[k])]
    nor = np.array(mid(1))
    n = np.linalg.norm(nor)
    return (mid(0), list(nor / n) if n > 1e-9 else list(a[1]), mid(2),
            [np.zeros(4) for _ in range(m.uv_add)], 2, bones, wts,
            (a[7] + b[7]) / 2)


def subdivide(path, name):
    """Split every triangle of one material into four.

    The hosiery arrives as 974 vertices over two legs -- forty or so around the ankle
    -- and a tube that coarse cannot hold the shape of an ankle whichever body it is
    on. Conformed at that resolution it still reads as a cone, because the corners it
    cuts are between its own vertices. Split once it has the vertices to follow one.

    Uniformly, not just over the ankle: splitting a band leaves T-junctions on its
    border, and those open as cracks under any deformation.
    """
    m = PMX(path)
    F = np.array(m.faces).reshape(-1, 3)
    at, out = 0, []
    for mat in m.materials:
        n = mat['fcount'] // 3
        tris = F[at:at + n]
        at += n
        if mat['name'] != name:
            out.append(tris)
            continue
        edge = {}

        def mid(i, j):
            k = (min(i, j), max(i, j))
            if k not in edge:
                edge[k] = len(m.vertices)
                m.vertices.append(_blend_vertex(m, k[0], k[1]))
            return edge[k]

        new = []
        for a, b, c in tris:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        tris = np.array(new)
        mat['fcount'] = len(tris) * 3
        out.append(tris)
        print(f'  {name}: {n} triangles -> {len(tris)}, '
              f'{len(edge)} vertices added')
    m.faces = np.concatenate(out).ravel().tolist()
    write_pmx(path, m)


def _taubin(P, tris, index, w, rounds=SOCK_SMOOTH, lam=0.5, mu=-0.53):
    """Shrink-free smoothing of a point set over its own connectivity.

    Alternating a positive and a slightly larger negative step is Taubin's: the
    inward pass undoes the shrink a plain Laplacian leaves, so a tube stays the width
    it was. `w` scales it per vertex, which is how the lace and the part inside the
    shoe are held still.
    """
    nbr = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            nbr.setdefault(u, set()).add(v)
            nbr.setdefault(v, set()).add(u)
    lists = [np.array([index[j] for j in nbr.get(int(i), ()) if j in index], int)
             for i in index]
    order = [None] * len(index)
    for i, k in index.items():
        order[k] = np.array([index[j] for j in nbr.get(int(i), ()) if j in index], int)
    for r in range(rounds):
        step = lam if r % 2 == 0 else mu
        L = np.zeros_like(P)
        for k, ns in enumerate(order):
            if len(ns):
                L[k] = P[ns].mean(0) - P[k]
        P = P + step * L * w[:, None]
    return P


def _roughness(P, tris, index):
    """Mean length of the discrete Laplacian -- how bumpy the surface is."""
    nbr = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            nbr.setdefault(u, set()).add(v)
            nbr.setdefault(v, set()).add(u)
    out = []
    for i, k in index.items():
        ns = np.array([index[j] for j in nbr.get(int(i), ()) if j in index], int)
        if len(ns):
            out.append(np.linalg.norm(P[ns].mean(0) - P[k]))
    return float(np.mean(out)) if out else 0.0


def fit_socks(path, offset=SOCK_OFFSET, floor=SOCK_FLOOR, hem=SOCK_HEM, band=0.70):
    """Lay the hosiery on the leg it is actually on.

    `push_out` only lifts what has sunk, so it holds the standoff even and leaves the
    shape alone -- and the shape is the complaint: the ankle the tube was cut for is
    not this one, so the malleolus reads as a smooth flare. Hosiery is skin-tight, so
    the honest surface here is our own leg offset outward, and that is what it is set
    to.

    Two places keep the donor's shape instead. Below `floor` the piece is inside the
    shoe, where the shoe's frame decides and our bare foot is larger than the last it
    was built on. Within `hem` of the top edge is the lace, which stands off the thigh
    on purpose. Both fade over `band` so no step is left where they meet.

    The whole fade sits UNDER the shoe's collar, which tops out at 2.235. A fade is a
    band of half-conformed surface and it reads as a seam wherever it is visible: run
    from 2.10 it left a cut line right across the ankle, 0.065 clear of the collar.
    Run from 1.50 to 2.20 it is behind the shoe, and everything above the collar --
    everything anyone sees -- is conformed the same way.

    The standoff tapers to `SOCK_ANKLE` low down. The two ankles are not the same
    ankle: at 0.4 above the bone this leg has radius 0.368 against the 0.339 the
    donor's comes to under the 1.0916 the shoe is scaled by, so it is 8.5% thicker
    than the collar was cut for, and hosiery of full thickness on top of that is what
    reads as a fat ankle.
    """
    m = PMX(path)
    mats = slices(m)
    V = np.array([v[0] for v in m.vertices])
    skin = Mesh(V, mats['skin'])
    tris = mats[SOCK]
    ids = np.unique(tris)

    ec = {}
    for f in tris:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            k = (min(a, b), max(a, b))
            ec[k] = ec.get(k, 0) + 1
    # Only the top rim is the lace. The hosiery has open edges around the foot as
    # well, where its author deleted what the shoe hides, and guarding those too put
    # the whole ankle out of reach: 310 of 3622 vertices conformed instead of 3200,
    # which is the shape complaint left exactly where it was.
    rim = np.array(sorted({i for k, c in ec.items() if c == 1 for i in k
                           if V[i][1] > SOCK_LACE}))
    tree = cKDTree(V[rim])
    print(f'  {SOCK}: {len(rim)} vertices on the lace edge, above y {SOCK_LACE}')

    P = V[ids]
    d, tri, bary = skin.closest(P)
    f = skin.F[tri]
    q = (bary[:, 0:1] * skin.V[f[:, 0]] + bary[:, 1:2] * skin.V[f[:, 1]]
         + bary[:, 2:3] * skin.V[f[:, 2]])
    lo, hi = SOCK_TAPER
    thick = (SOCK_ANKLE + (offset - SOCK_ANKLE)
             * np.clip((P[:, 1] - lo) / (hi - lo), 0.0, 1.0))
    target = q + skin.normal[tri] * thick[:, None]
    hemd, _ = tree.query(P)
    w = (np.clip((P[:, 1] - floor) / band, 0, 1)
         * np.clip((hemd - hem) / band, 0, 1))
    moved = P + (target - P) * w[:, None]
    print(f'  {SOCK}: conformed {int((w > 0.9).sum())} of {len(ids)} vertices fully '
          f'and {int((w > 0.01).sum())} at all, to the leg at {SOCK_ANKLE}..{offset}; moved by up '
          f'to {np.linalg.norm(moved - P, axis=1).max():.3f}')

    # Set to the leg exactly, the hosiery reproduces the ankle bone as the hard
    # faceted lump the skin mesh draws it as. Fabric under tension spans a bony
    # prominence instead of wrapping into it, which is what the smoothing is: it
    # rounds what the offset copied, held still over the lace and inside the shoe by
    # the same weights, and shrink-free so the tube keeps its width.
    index = {int(i): k for k, i in enumerate(ids)}
    rough = _roughness(moved, tris, index)
    moved = _even(P, _taubin(moved, tris, index, w))
    print(f'  {SOCK}: smoothed the offset copy, roughness {rough:.4f} -> '
          f'{_roughness(moved, tris, index):.4f}')

    before = moved.copy()
    moved, lifted = push_out(skin, moved, offset=SOCK_ANKLE * 0.6, blur=0.10, rounds=40)
    moved = _even(before, moved)
    print(f'  {SOCK}: lifted {int(lifted.sum())} that the fade left under the leg, '
          f'by up to {np.linalg.norm(moved - before, axis=1).max():.3f}')
    V[ids] = moved

    acc = np.zeros_like(V)
    for a, b, c in tris:
        nn = np.cross(V[b] - V[a], V[c] - V[a])
        acc[a] += nn
        acc[b] += nn
        acc[c] += nn
    for i in ids:
        v = list(m.vertices[int(i)])
        v[0] = [float(x) for x in V[int(i)]]
        ln = np.linalg.norm(acc[int(i)])
        if ln > 1e-12:
            v[1] = [float(x) for x in acc[int(i)] / ln]
        m.vertices[int(i)] = tuple(v)
    write_pmx(path, m)


# reze-midnight's breasts have no front. Two 24-edge loops, one per side at
# y 15.93..17.39, are where its own bra cups sat -- the cups ARE the breast there, the
# way this family of models is built -- so dropping its outfit opens the chest. The
# donor has no such hole (her loops are 9 edges) and her dress covers 88% of her bust
# against our 85%, so the maid bodice is not the problem: the body under it is.
# The cups are kept instead of dropped, and painted the skin colour sampled off the
# chest, so they read as the breast they stand in for.
BUST = 'maid_dress'
BUST_Y = (15.00, 17.60)     # our frame: the bodice over the bust
BUST_Z = -0.90              # only what faces forward
BUST_OFFSET = 0.030
BUST_FADE = 0.40
BUST_RELAX = 6          # smoothing passes on the conform, so it cannot tear the bodice
BUST_REACH = 0.45       # skin further than this from the bodice is not under it
BUST_FOLLOWERS = ('maid_metal', 'maid_lace')
BUST_GLUE = 0.35        # a follower this far from the bodice is not attached to it


def fit_bust(path, offset=None, lo=BUST_Y[0], hi=BUST_Y[1], front=BUST_Z,
             fade=BUST_FADE):
    """Lay the bodice on the bust it is actually on.

    Over the chest the dress is a bra, and a bra is skin-tight, so the same treatment
    the hosiery got applies: set it to our own surface plus a standoff instead of
    keeping the shape it was cut for. It was cut for a much smaller bust -- 39 of 212
    bust vertices stood outside it, by as much as 0.484 -- and every one of those is a
    hole in the chest showing through, because this body has 96 boundary edges up
    there where its own bra used to be.

    Held to the front of the chest and faded at the edges of the band, so the collar,
    the shoulders and the waist keep the shape the garment was drawn with.
    """
    offset = BUST_OFFSET if offset is None else offset
    m = PMX(path)
    mats = slices(m)
    V = np.array([v[0] for v in m.vertices])
    body = [mats[k] for k in CLEAR_AGAINST if k in mats]
    skin = Mesh(V, np.concatenate(body))
    tris = mats[BUST]
    ids = np.unique(tris)
    P = V[ids]

    w = (np.clip((P[:, 1] - lo) / fade, 0, 1)
         * np.clip((hi - P[:, 1]) / fade, 0, 1)
         * np.clip((front - P[:, 2]) / fade, 0, 1))
    if not (w > 0.01).any():
        print('  no bust band found')
        return
    d, tri, bary = skin.closest(P)
    f = skin.F[tri]
    q = (bary[:, 0:1] * skin.V[f[:, 0]] + bary[:, 1:2] * skin.V[f[:, 1]]
         + bary[:, 2:3] * skin.V[f[:, 2]])
    target = q + skin.normal[tri] * offset
    # Only pull IN what the body is pushing through, and only where the body is
    # actually behind it -- a frill hanging clear of the chest is not a bra.
    w = np.where(d < BUST_REACH, w, 0.0)
    delta = (target - P) * w[:, None]
    # Smoothed over the bodice's own edges before it is applied. A weight that fades
    # moves neighbours by different amounts, and that stretches the triangles between
    # them: this pass was leaving 28 slivers in the cleavage at a thinness of up to
    # 262, against 6.6 for the worst of hers in the same band. They render as ribbons
    # strung between the breasts.
    nbr = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            nbr.setdefault(int(u), set()).add(int(v))
            nbr.setdefault(int(v), set()).add(int(u))
    at = {int(v): k for k, v in enumerate(ids)}
    order = [np.array([at[j] for j in nbr.get(int(v), ()) if j in at], int) for v in ids]
    for _ in range(BUST_RELAX):
        nxt = delta.copy()
        for k, ns in enumerate(order):
            if len(ns):
                nxt[k] = 0.5 * delta[k] + 0.5 * delta[ns].mean(0)
        delta = nxt
    moved = P + delta
    moved = _even(P, moved)
    print(f'  {BUST}: conformed {int((w > 0.9).sum())} of {len(ids)} vertices fully '
          f'and {int((w > 0.01).sum())} at all, to the bust at {offset}; moved by up '
          f'to {np.linalg.norm(moved - P, axis=1).max():.3f}')

    # Whatever was sitting ON the bodice comes with it. The two heart buttons are
    # their own material, so conforming the dress alone left them hanging 0.18 to
    # 0.22 off it, where hers sit at 0.021. Each follower takes the displacement of
    # the dress vertex it was nearest to, faded out with distance so a piece that
    # merely passes nearby is not dragged.
    delta = moved - P
    for name in BUST_FOLLOWERS:
        if name not in mats:
            continue
        fids = np.unique(mats[name])
        Q = V[fids]
        dist, j = cKDTree(P).query(Q)
        pull = np.clip((BUST_GLUE - dist) / BUST_GLUE, 0, 1)
        if not (pull > 0).any():
            continue
        V[fids] = Q + delta[j] * pull[:, None]
        for i, k in zip(fids, range(len(fids))):
            v = list(m.vertices[int(i)])
            v[0] = [float(x) for x in V[int(i)]]
            m.vertices[int(i)] = tuple(v)
        print(f'  {name}: {int((pull > 0.5).sum())} of {len(fids)} vertices carried '
              f'with the bodice, by up to {np.abs(delta[j] * pull[:, None]).max():.3f}')

    V[ids] = moved
    acc = np.zeros_like(V)
    for a, b, c in tris:
        nn = np.cross(V[b] - V[a], V[c] - V[a])
        acc[a] += nn
        acc[b] += nn
        acc[c] += nn
    for i in ids:
        v = list(m.vertices[int(i)])
        v[0] = [float(x) for x in V[int(i)]]
        ln = np.linalg.norm(acc[int(i)])
        if ln > 1e-12:
            v[1] = [float(x) for x in acc[int(i)] / ln]
        m.vertices[int(i)] = tuple(v)
    write_pmx(path, m)


HARDWARE = ('maid_metal', 'maid_lace')
HARDWARE_STANDOFF = 0.020   # her metal sits this far off her cloth, median over 11 shells
HARDWARE_REACH = 0.60       # further than this and it is not sitting on the dress


def seat_hardware(path, pieces=HARDWARE, standoff=HARDWARE_STANDOFF,
                  reach=HARDWARE_REACH):
    """Sit the buttons and trim back down on the dress.

    Each is its own material, so every pass that moved the dress left them where they
    were -- the two heart buttons ended 0.22 off the bodice against the 0.021 hers sit
    at, floating in front of the chest. Carrying the bodice's displacement closes most
    of it and not the part that was already there, since the field placed the two
    through different local scales in the first place.

    So each shell is seated: one rigid translation along the dress's normal, taking
    the shell's mean standoff to hers. Rigid because a button is hardware -- moving
    the vertices that happen to be far flattens it.
    """
    m = PMX(path)
    mats = slices(m)
    V = np.array([v[0] for v in m.vertices])
    dress = Mesh(V, mats[BUST])
    seated = 0
    for name in pieces:
        if name not in mats:
            continue
        for ids in rr.shells(V, mats[name]).values():
            idx = np.array(sorted(int(i) for i in ids))
            P = V[idx]
            d, tri, bary = dress.closest(P)
            if d.mean() > reach or d.mean() <= standoff:
                continue
            f = dress.F[tri]
            q = (bary[:, 0:1] * dress.V[f[:, 0]] + bary[:, 1:2] * dress.V[f[:, 1]]
                 + bary[:, 2:3] * dress.V[f[:, 2]])
            n = dress.normal[tri].mean(0)
            ln = np.linalg.norm(n)
            if ln < 1e-9:
                continue
            n /= ln
            step = float(d.mean() - standoff)
            V[idx] = P - n * step
            seated += 1
    for i in np.unique(np.concatenate([mats[p] for p in pieces if p in mats])):
        v = list(m.vertices[int(i)])
        v[0] = [float(x) for x in V[int(i)]]
        m.vertices[int(i)] = tuple(v)
    write_pmx(path, m)
    print(f'  seated {seated} hardware shells onto the dress at {standoff}')


def ground(path, pieces=('maid_shoes', 'maid_shoe_trim'), floor=0.0):
    """Set the shoes back down on the floor.

    `clear_body` resolves a shoe as one rigid assembly, and clearing it of a foot
    0.06 wider than the one it was cut for grows it downward as well as outward --
    the sole ended 0.080 under the floor. The width is the correction worth keeping;
    the height is not, since both models are authored standing on y 0. So the piece
    is dropped straight back to the ground afterwards, one number per side, which
    cannot undo what the clearance did across the foot.
    """
    m = PMX(path)
    mats = slices(m)
    V = np.array([v[0] for v in m.vertices])
    ids = np.unique(np.concatenate([mats[p] for p in pieces if p in mats]))
    for side in (1.0, -1.0):
        sel = np.array([i for i in ids if np.sign(V[i][0]) == side])
        if not len(sel):
            continue
        lift = floor - V[sel][:, 1].min()
        V[sel] += np.array([0.0, lift, 0.0])
        print(f'  {"left" if side > 0 else "right"} shoe lifted {lift:+.3f} '
              f'back to y {floor}')
    for i in ids:
        w = list(m.vertices[int(i)])
        w[0] = [float(x) for x in V[int(i)]]
        m.vertices[int(i)] = tuple(w)
    write_pmx(path, m)


# Pieces that sit ON the body and must deform with it. Carried geometry keeps the
# weights of the model it came from, and at rest that is invisible -- it only shows
# once a bone moves. The shoe leans on つま先 for 302 of its vertices where the foot
# under it uses つま先 for 515, so the toe bone bends the two by different amounts and
# the forefoot twists across the shoe. The dress is NOT in this list: its skirt hangs
# on carried chain bones, and taking the body's weights would take its physics with it.
BODY_FITTED = ('maid_stockings', 'maid_underwear')

# A platform heel does not flex. Hers is already close to rigid -- 足首D takes 89% of
# the middle of it -- and this foot is not: it puts 82% of its toe box on つま先, twice
# what her shoe carries there. Under a motion that curls the toes the two turn by
# different amounts and the forefoot twists across the shoe. Worse, transferring the
# body's weights onto it gave the heel block ひざD 55%, so the knee swung it.
# So the shoe is pinned to one bone per side and moves as the single rigid object it
# is. The foot inside it flexes as it likes; it is inside a closed shoe.
RIGID_ON = {'maid_shoes': '足首D', 'maid_shoe_metal': '足首D'}

# The foot inside the shoe has to be as rigid as the shoe. This body puts 61% of its
# bare foot on つま先 and the sock over it sits at 7%, so a motion that curls the toes
# swings the foot out through both, and the sock had picked up ひざD 73% at the ankle
# against the leg's 44%, so the knee swung that too. A foot in a heel does not bend:
# skin and hosiery below the vamp go on the same bone the shoe does, fading out by the
# time the ankle is above the collar and visible again.
STIFF_ON = ('skin', 'maid_stockings')
STIFF_BONE = '足首D'
STIFF_FULL = 1.70       # our frame: all of it, well inside the shoe
STIFF_FREE = 2.30       # and none of it here, just above the collar at 2.098

def rigid_weights(path, pieces=None):
    """Pin a piece to one bone per side, so nothing can bend it."""
    pieces = pieces or RIGID_ON
    m = PMX(path)
    sl = slices(m)
    V = np.array([v[0] for v in m.vertices])
    index = {b['name']: i for i, b in enumerate(m.bones)}
    for name, bone in pieces.items():
        if name not in sl:
            continue
        done = 0
        for i in np.unique(sl[name]):
            i = int(i)
            side = '左' if V[i][0] > 0 else '右'
            b = index.get(side + bone)
            if b is None:
                continue
            rec = list(m.vertices[i])
            rec[4], rec[5], rec[6] = 0, [b], [1.0]
            m.vertices[i] = tuple(rec[:8])
            done += 1
        print(f'  {name}: {done} vertices pinned to 左/右{bone}')
    write_pmx(path, m)


def stiffen_foot(path, pieces=STIFF_ON, bone=STIFF_BONE,
                 full=STIFF_FULL, free=STIFF_FREE):
    """Blend the foot onto the shoe's bone, so the two cannot turn apart."""
    m = PMX(path)
    sl = slices(m)
    V = np.array([v[0] for v in m.vertices])
    names = [b['name'] for b in m.bones]
    index = {b['name']: i for i, b in enumerate(m.bones)}
    done = 0
    for name in pieces:
        if name not in sl:
            continue
        for i in np.unique(sl[name]):
            i = int(i)
            y = V[i][1]
            if y >= free:
                continue
            w = float(np.clip((free - y) / (free - full), 0.0, 1.0))
            side = '左' if V[i][0] > 0 else '右'
            target = index.get(side + bone)
            if target is None:
                continue
            wd = {}
            for b, ww in zip(m.vertices[i][5], m.vertices[i][6]):
                if ww > 1e-5 and 0 <= b < len(names):
                    wd[names[b]] = wd.get(names[b], 0.0) + float(ww) * (1.0 - w)
            wd[side + bone] = wd.get(side + bone, 0.0) + w
            if _set_weights(m, i, wd, index):
                done += 1
    write_pmx(path, m)
    print(f'  stiffened {done} foot vertices onto 左/右{bone} below y {free}')


def trim_legs(path, reach=0.25, sock=None):
    """Take our skin out from under the hosiery.

    Two skin-tight surfaces do not have to interpenetrate to fight: the faces between
    them cross, and the thigh comes out shredded with every vertex still outside.

    What is underneath is asked of the hosiery itself rather than of a height. Hers
    tops on a diagonal -- the edge runs from y 5.08 at the inside of the knee to 9.87
    at the outside of the thigh -- so a flat cut taken from the high end opens the
    front of both thighs, and one taken from the low end leaves most of the leg
    doubled. A skin triangle goes only where all three of its corners are inside the
    hosiery and within `reach` of it, which holds the partly covered ring at the edge
    and with it the seam, and cannot reach the hands hanging beside the thigh in an
    A-pose.
    """
    m = PMX(path)
    mats = slices(m)
    V = np.array([v[0] for v in m.vertices])
    sock = Mesh(V, mats[sock or SOCK])
    ids = np.unique(mats['skin'])
    d, tri, bary = sock.closest(V[ids])
    f = sock.F[tri]
    q = (bary[:, 0:1] * sock.V[f[:, 0]] + bary[:, 1:2] * sock.V[f[:, 1]]
         + bary[:, 2:3] * sock.V[f[:, 2]])
    inside = ((V[ids] - q) * sock.normal[tri]).sum(1) < 0
    covered = np.zeros(len(V), bool)
    covered[ids] = inside & (d < reach)
    F = np.array(m.faces).reshape(-1, 3)
    at, out, gone = 0, [], 0
    for mat in m.materials:
        n = mat['fcount'] // 3
        tris = F[at:at + n]
        if mat['name'] == 'skin':
            drop = covered[tris].all(1)
            gone = int(drop.sum())
            tris = tris[~drop]
            mat['fcount'] = len(tris) * 3
        out.append(tris)
        at += n
    m.faces = np.concatenate(out).ravel().tolist()
    write_pmx(path, m)
    left = mats['skin'][~covered[mats['skin']].all(1)]
    print(f'  took {gone} skin triangles from inside the hosiery, of '
          f'{len(mats["skin"])}; the leg it leaves reaches down to y '
          f'{V[np.unique(left)][:, 1].min():.2f}')


HIP = '下半身'
# How embedded a chain body may start and still be allowed to collide. A body that
# begins deep inside a body capsule can never resolve -- it is ejected on the first
# frame -- but a grazing overlap settles in a frame or two, and demanding a clean
# start silences the panels that matter: at the butt, 0 of 68 were left free to
# collide, so the skirt simply fell through it.
CLEAR_START = -0.10


def _capsule_sd(P, r):
    """Distance from each point to a capsule's surface; negative is inside."""
    rx, ry, rz = r['rot']
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    axis = (Ry @ Rx @ Rz) @ np.array([0.0, 1.0, 0.0])
    a = np.array(r['pos']) - axis * r['size'][1] / 2
    b = np.array(r['pos']) + axis * r['size'][1] / 2
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)[:, None]
    return np.linalg.norm(P - (a + t * ab), axis=1) - r['size'][0]


def mirror_body_colliders(path, own, mid=0.30):
    """Make the body's own colliders symmetric.

    A skirt hangs on them, so any asymmetry here lands as one side of it standing off
    further than the other. Two sources. reze-midnight ships a `下半身` capsule at
    x 0.034 with a yaw of 0.259 on it, which reaches 0.068 further to one side than
    the other. And `tighten_colliders` measures each side's skin separately, so it
    brought `左足D` in to 0.924 against `右足D`'s 0.914 and `左肩` to 0.339 against
    `右肩`'s 0.349 -- small, and the panels hanging on them are not.

    A centre-line collider is put on the centre line and its yaw taken out; a 左/右
    pair is given one size and mirrored placement.
    """
    m = PMX(path)
    names = [b['name'] for b in m.bones]
    flip = np.array([-1.0, 1.0, 1.0])
    rflip = np.array([1.0, -1.0, -1.0])
    body = [r for r in m.rigids
            if r['mode'] == 0 and 0 <= r['bone'] < len(names) and names[r['bone']] in own]
    centred = paired = 0
    for r in body:
        n = names[r['bone']]
        if n.startswith(('左', '右')) or abs(r['pos'][0]) > mid:
            continue
        if abs(r['pos'][0]) > 1e-9 or abs(r['rot'][1]) > 1e-9:
            centred += 1
        r['pos'] = [0.0, float(r['pos'][1]), float(r['pos'][2])]
        r['rot'] = [float(r['rot'][0]), 0.0, float(r['rot'][2])]

    byname = {}
    for r in body:
        n = names[r['bone']]
        if n.startswith(('左', '右')):
            byname.setdefault(n[1:], {}).setdefault(n[0], []).append(r)
    for base, sides in byname.items():
        if set(sides) != {'左', '右'} or len(sides['左']) != len(sides['右']):
            continue
        L = sorted(sides['左'], key=lambda r: (r['pos'][1], r['pos'][2]))
        R = sorted(sides['右'], key=lambda r: (r['pos'][1], r['pos'][2]))
        for a, b in zip(L, R):
            size = (np.array(a['size']) + np.array(b['size'])) / 2
            pos = (np.array(a['pos']) + np.array(b['pos']) * flip) / 2
            rot = (np.array(a['rot']) + np.array(b['rot']) * rflip) / 2
            a['size'] = b['size'] = [float(x) for x in size]
            a['pos'] = [float(x) for x in pos]
            b['pos'] = [float(x) for x in pos * flip]
            a['rot'] = [float(x) for x in rot]
            b['rot'] = [float(x) for x in rot * rflip]
            paired += 1
    write_pmx(path, m)
    print(f'  put {centred} centre-line colliders back on the centre line and '
          f'matched {paired} left/right pairs')


def reseat_chains(path, donor, place, nhost):
    """Put the carried bones, bodies and joints through the same field as the mesh.

    `chaingraft` places each accessory by ONE similarity fitted to the field where the
    accessory sits, which is what its rigid-body sizes and rotations need. Over a
    fourteen-link skirt panel reaching from the waist to the hem that similarity
    drifts from the field at the ends, and the mesh does not drift with it -- the mesh
    goes through the field itself. The two then disagree in the worst possible place:
    at y 12.75..13.25 this butt reaches z 0.857, the skirt's mesh clears it by 0.5,
    and the skirt's rigid bodies sat at z 0.44..0.68 with a radius of 0.23. The cloth
    was outside the body and the physics driving it was threaded through the middle
    of it, so the panels settled into the butt however the collision was set.

    Positions only. Sizes and rotations keep the local similarity, which is the thing
    a field cannot give them.
    """
    m = PMX(path)
    dbone = {b['name']: np.array(b['pos']) for b in donor.bones}
    drig = {r['name']: np.array(r['pos']) for r in donor.rigids}
    djnt = {j['name']: np.array(j['pos']) for j in donor.joints}
    bones = rigids = joints = 0
    for i in range(nhost, len(m.bones)):
        n = m.bones[i]['name']
        if n in dbone:
            m.bones[i]['pos'] = [float(x) for x in place(dbone[n])[0]]
            bones += 1
    for r in m.rigids:
        if r['bone'] >= nhost and r['name'] in drig:
            r['pos'] = [float(x) for x in place(drig[r['name']])[0]]
            rigids += 1
    for j in m.joints:
        a, b = j['a'], j['b']
        touches = ((0 <= a < len(m.rigids) and m.rigids[a]['bone'] >= nhost)
                   or (0 <= b < len(m.rigids) and m.rigids[b]['bone'] >= nhost))
        if touches and j['name'] in djnt:
            j['pos'] = [float(x) for x in place(djnt[j['name']])[0]]
            joints += 1
    write_pmx(path, m)
    print(f'  reseated {bones} bones, {rigids} rigid bodies and {joints} joints '
          f'onto the field the mesh uses')


def unbury_chains(path, nhost, limit=0.31):
    """Lift carried bodies that start deeper inside our body than hers start in hers.

    A rigid body that begins inside a static one is pushed out on the first frame.
    Some of that is normal and her skirt is full of it -- 172 of her 364 skirt bodies
    start embedded in her own body colliders -- but hers are shallow: median 0.084,
    max 0.308. Ours went twice as deep, median 0.172 and max 0.549, and at the back
    it was 20 of 21 bodies buried by as much as 0.454. Ejected together on frame one,
    that is what threw the rear panels apart and opened the skirt.

    `limit` is her worst. Anything past it is moved straight out of the collider it is
    buried in, and its bone and joints go with it so the chain stays one piece.
    """
    m = PMX(path)
    walls = [r for r in m.rigids
             if r['mode'] == 0 and r['shape'] == 2 and 0 <= r['bone'] < nhost]
    mine = [k for k, r in enumerate(m.rigids) if r['bone'] >= nhost]
    P = np.array([m.rigids[k]['pos'] for k in mine])
    rad = np.array([m.rigids[k]['size'][0] for k in mine])
    delta = np.zeros_like(P)
    for w in walls:
        rx, ry, rz = w['rot']
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        axis = (Ry @ Rx @ Rz) @ np.array([0.0, 1.0, 0.0])
        a = np.array(w['pos']) - axis * w['size'][1] / 2
        b = np.array(w['pos']) + axis * w['size'][1] / 2
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)[:, None]
        foot = a + t * ab
        out = P - foot
        dist = np.linalg.norm(out, axis=1)
        depth = w['size'][0] + rad - dist                 # >0 means buried
        need = depth - limit
        hit = need > 0
        if not hit.any():
            continue
        dirn = out / np.maximum(dist, 1e-9)[:, None]
        want = dirn * need[:, None]
        deeper = np.linalg.norm(want, axis=1) > np.linalg.norm(delta, axis=1)
        delta[hit & deeper] = want[hit & deeper]

    moved = np.linalg.norm(delta, axis=1) > 1e-9
    bones = {}
    for i, k in enumerate(mine):
        if not moved[i]:
            continue
        r = m.rigids[k]
        r['pos'] = [float(x) for x in np.array(r['pos']) + delta[i]]
        bones.setdefault(r['bone'], []).append(delta[i])
    for b, ds in bones.items():
        d = np.mean(ds, axis=0)
        m.bones[b]['pos'] = [float(x) for x in np.array(m.bones[b]['pos']) + d]
    at = {k: delta[i] for i, k in enumerate(mine) if moved[i]}
    for j in m.joints:
        ends = [at[e] for e in (j['a'], j['b']) if e in at]
        if ends:
            j['pos'] = [float(x) for x in np.array(j['pos']) + np.mean(ends, axis=0)]
    write_pmx(path, m)
    print(f'  lifted {int(moved.sum())} of {len(mine)} carried bodies out of our '
          f'colliders, by up to {np.linalg.norm(delta, axis=1).max():.3f}')


def mirror_rig(path, nhost, tol=0.08):
    """Make the carried rig mirror-symmetric.

    Her skirt is symmetric to the digit -- bone mirror residual 0.00000, one rigid
    body out of 171 pairs with a different size, none disagreeing about collision.
    Ours was not: 72 pairs came out different sizes and 17 disagreed about collision.
    `chaingraft` fits every accessory its own similarity by probing the field where it
    sits, and a left panel and a right panel are separate accessories, so they read
    slightly different scales and their bodies come out slightly different sizes. The
    clearance rule then reads those sizes and reaches different answers on the two
    sides.

    A viewer that runs physics turns that into one side of the skirt collapsing while
    the other holds, which is not a small asymmetry in a garment made of 26 hanging
    panels. So every pair is averaged: position mirrored, size, mass, damping and the
    collision mask taken from the pair, and anything on the centre line put on it.
    """
    m = PMX(path)
    flip = np.array([-1.0, 1.0, 1.0])

    def pairs(items, pos):
        """Mirror pairs, returned as the caller's own ids -- not positions in the
        list, which is what indexed straight into `m.bones` and moved 右胸."""
        P = np.array([pos(x) for x in items])
        left = [k for k in range(len(items)) if P[k][0] > tol / 2]
        right = [k for k in range(len(items)) if P[k][0] < -tol / 2]
        mid = [items[k] for k in range(len(items)) if abs(P[k][0]) <= tol / 2]
        if not left or not right:
            return [], mid
        d, j = cKDTree(P[right] * flip).query(P[left])
        return ([(items[left[k]], items[right[j[k]]])
                 for k in range(len(left)) if d[k] < tol], mid)

    bones = list(range(nhost, len(m.bones)))
    bp, bmid = pairs(bones, lambda i: m.bones[i]['pos'])
    for a, b in bp:
        pa = np.array(m.bones[a]['pos'])
        pb = np.array(m.bones[b]['pos']) * flip
        avg = (pa + pb) / 2
        m.bones[a]['pos'] = [float(x) for x in avg]
        m.bones[b]['pos'] = [float(x) for x in avg * flip]
    for i in bmid:
        m.bones[i]['pos'] = [0.0, float(m.bones[i]['pos'][1]), float(m.bones[i]['pos'][2])]

    rig = [k for k, r in enumerate(m.rigids) if r['bone'] >= nhost]
    rp, rmid = pairs(rig, lambda k: m.rigids[k]['pos'])
    same = ('size', 'mass', 'lin_damp', 'ang_damp', 'restitution', 'friction',
            'group', 'nocollide', 'mode', 'shape')
    for a, b in rp:
        ra, rb = m.rigids[a], m.rigids[b]
        avg = (np.array(ra['pos']) + np.array(rb['pos']) * flip) / 2
        ra['pos'] = [float(x) for x in avg]
        rb['pos'] = [float(x) for x in avg * flip]
        sz = (np.array(ra['size']) + np.array(rb['size'])) / 2
        ra['size'] = rb['size'] = [float(x) for x in sz]
        for k in same:
            if k == 'size':
                continue
            rb[k] = ra[k]
        rot = (np.array(ra['rot']) + np.array(rb['rot']) * np.array([1.0, -1.0, -1.0])) / 2
        ra['rot'] = [float(x) for x in rot]
        rb['rot'] = [float(x) for x in rot * np.array([1.0, -1.0, -1.0])]
    for k in rmid:
        r = m.rigids[k]
        r['pos'] = [0.0, float(r['pos'][1]), float(r['pos'][2])]

    jj = [k for k, j in enumerate(m.joints)
          if (0 <= j['a'] < len(m.rigids) and m.rigids[j['a']]['bone'] >= nhost)
          or (0 <= j['b'] < len(m.rigids) and m.rigids[j['b']]['bone'] >= nhost)]
    jp, jmid = pairs(jj, lambda k: m.joints[k]['pos'])
    for a, b in jp:
        ja, jb = m.joints[a], m.joints[b]
        avg = (np.array(ja['pos']) + np.array(jb['pos']) * flip) / 2
        ja['pos'] = [float(x) for x in avg]
        jb['pos'] = [float(x) for x in avg * flip]
        for k in ('spring_lin', 'spring_ang'):
            v = (np.array(ja[k]) + np.array(jb[k])) / 2
            ja[k] = jb[k] = [float(x) for x in v]
    for k in jmid:
        j = m.joints[k]
        j['pos'] = [0.0, float(j['pos'][1]), float(j['pos'][2])]

    write_pmx(path, m)
    print(f'  mirrored {len(bp)} bone pairs, {len(rp)} rigid-body pairs and '
          f'{len(jp)} joint pairs ({len(bmid)}/{len(rmid)}/{len(jmid)} on the centre line)')


def prune_physics(path, own):
    """Take out the physics of the outfit and hair we dropped.

    `dropmaterial` removes a material's faces and leaves everything else alone, which
    is right for vertices -- renumbering them would mean renumbering every morph --
    and wrong for rigid bodies. 午夜沉沦's own hair and skirt keep 228 dynamic bodies
    and 355 joints between them, on 266 bones that now drive nothing at all. Nothing
    draws them and the solver still runs every one, every frame.

    A bone is dead here if no material we keep leans on it and it is not the ancestor
    of one that does. None of them is an ancestor of a carried outfit bone, so this
    cannot reach the maid's chains. Bones are left in place: removing those would mean
    renumbering the morphs.
    """
    m = PMX(path)
    sl = slices(m)
    live = set()
    for name in KEEP:
        if name not in sl:
            continue
        for vi in np.unique(sl[name]):
            v = m.vertices[int(vi)]
            for b, w in zip(v[5], v[6]):
                if w > 0.001:
                    live.add(int(b))
    for b in list(live):
        i = b
        while i >= 0:
            live.add(i)
            i = m.bones[i]['parent']
    host = {i for i, b in enumerate(m.bones) if b['name'] in own}
    dead = {i for i in host if i not in live}
    drop = {k for k, r in enumerate(m.rigids) if r['bone'] in dead}
    if not drop:
        print('  nothing to prune')
        return
    keep = [k for k in range(len(m.rigids)) if k not in drop]
    remap = {k: i for i, k in enumerate(keep)}
    joints = [j for j in m.joints if j['a'] not in drop and j['b'] not in drop]
    gone_j = len(m.joints) - len(joints)
    for j in joints:
        j['a'] = remap.get(j['a'], -1)
        j['b'] = remap.get(j['b'], -1)
    for mo in m.morphs:
        if mo['type'] != 10:
            continue
        mo['offsets'] = [(remap[o[0]],) + tuple(o[1:]) for o in mo['offsets']
                         if o[0] in remap]
    m.rigids = [m.rigids[k] for k in keep]
    m.joints = joints
    write_pmx(path, m)
    print(f'  {len(dead)} host bones drive nothing we kept; dropped {len(drop)} of '
          f'their rigid bodies and {gone_j} joints')
    print(f'  now {len(m.rigids)} rigid bodies, {len(m.joints)} joints')


def collide(path, own):
    """Check the carried collision masks, and open our body to them if it is closed.

    The mask is `非衝突グループ` and reads 1 = COLLIDES with that group. reze's hair is
    group 9 with 0xFDE0, whose bit 9 is 0 -- a chain whose links collided with each
    other would explode -- and every body collider in both models is 0xFFFF, which is
    "collides with everything". Read the other way round neither makes sense.

    `bunnybuild`'s version had it inverted, clearing bits to "enable" and setting them
    to "hold off", so every rule built on top of it did the opposite of what it said.
    Under the correct reading all 364 of her skirt bodies already collide with her
    body, and `chaingraft.remask` carries a mask across bit-for-bit, so the carried
    skirt arrives set up the way she authored it and wants nothing done to it.

    What is worth checking is our side: the chains arrive in remapped groups, and a
    body collider that does not name those groups would never meet them.
    """
    m = PMX(path)
    names = [b['name'] for b in m.bones]
    chain, body = set(), set()
    for r in m.rigids:
        n = names[r['bone']] if 0 <= r['bone'] < len(names) else ''
        if n and n not in own:
            chain.add(r['group'])
        elif r['mode'] == 0 and n in own:
            body.add(r['group'])

    opened = 0
    for r in m.rigids:
        n = names[r['bone']] if 0 <= r['bone'] < len(names) else ''
        if r['mode'] == 0 and n in own:
            before = r['nocollide']
            for g in chain:
                r['nocollide'] |= (1 << g)
            opened += r['nocollide'] != before

    meet = sum(1 for r in m.rigids
               if 0 <= r['bone'] < len(names) and names[r['bone']] not in own
               and any((r['nocollide'] >> g) & 1 for g in body))
    total = sum(1 for r in m.rigids
                if 0 <= r['bone'] < len(names) and names[r['bone']] not in own)
    write_pmx(path, m)
    print(f'  chain groups {sorted(chain)}, body groups {sorted(body)}; '
          f'opened {opened} body colliders')
    print(f'  {meet} of {total} carried bodies collide with the body, as she authored them')


def main():
    # Built under a working name and moved into place at the end. Every stage writes
    # the whole model back out, and the host's own outfit is not dropped until the
    # last one, so for the length of a build the file is reze-midnight with maid
    # pieces stacked on top of it. Anyone who loads it mid-build sees that.
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    work = OUT + '.building'
    shutil.copyfile(BASE, work)
    src = os.path.join(os.path.dirname(BASE), 'textures')
    if os.path.isdir(src) and not os.path.isdir(TEXDIR):
        shutil.copytree(src, TEXDIR)

    tmp = tempfile.mkdtemp(prefix='maidbuild-')
    donor = os.path.join(tmp, 'donor.pmx')

    print('--- split')
    split(DONOR, donor)

    print('--- names')
    own = free_names(work, donor)

    if HAIR_SHELLS and HAIR_SHELLS in slices(PMX(work)):
        print('--- stale hair')
        drop_hair_shells(work, material=HAIR_SHELLS)

    if REBUILD_BUST:
        print('--- bust')
        rebuild_bust(work)

    print('--- colliders')
    tighten_colliders(work, PMX(donor), set(OUR_BODY),
                      set(HER_BODY) | {'skin_stocking'})

    mirror_body_colliders(work, own)

    print('--- field')
    d0, t0 = PMX(donor), PMX(work)
    # Solved, not given. `chaingraft.body_scale` reads the ratio of pairwise
    # distances among centre-line bones, and here 10 of the 36 it finds are
    # `hair_B_*`: a name both models use for a chain, hanging down her braid and down
    # 午夜沉沦's, which are nowhere near each other. It returns 1.2873 against a pair
    # whose heights above `全ての親` agree at 1.085, and the field spends itself
    # undoing that -- correction median 0.732 and max 2.237 against 0.083 and 0.384
    # here. The warning its docstring carries is about two bodies of the SAME height,
    # which these are not: hers finishes at y 18.49 and ours at 19.82.
    field = bodyfield.Field(d0, t0, HER_BODY, OUR_BODY)
    snug = bodyfield.Snug(field, d0, HER_BODY, t0, CLEAR_AGAINST)

    print('--- head')
    head = HeadFrame(d0, t0)
    route = Router(snug, head, HEAD_Y)

    print('--- foot')
    sole = np.array([d0.vertices[int(i)][0] for i in
                     np.unique(np.concatenate([graftmaterial.material_slice(d0, m)[0]
                                               for m in ('cloth01_foot', 'metal_foot')]))])
    ankle = np.array([v[0] for v in t0.vertices
                      for _ in (0,)][:0]) if False else None
    hfoot = np.array([t0.vertices[int(i)][0]
                      for i in np.unique(graftmaterial.material_slice(t0, 'skin')[0])])
    hfoot = hfoot[(hfoot[:, 1] < 2.40) & (hfoot[:, 1] > 0.05)]
    foot = FootFrame(d0, t0, floor=sole, foot=hfoot)

    print('--- chains')
    chaingraft.main(donor, work, work, CHAIN_MATERIALS, frame='女仆', place=route)
    reseat_chains(work, d0, route, len(t0.bones))
    unbury_chains(work, len(t0.bones))
    mirror_rig(work, len(t0.bones))
    tex = textures(work)

    print('--- outfit')
    by = {'body': snug, 'head': head, 'foot': foot,
          'sock': Blend(foot, snug, *SOCK_BAND)}
    for mat, name, sheet, _, _, which, _, _ in OUTFIT:
        graftmaterial.main(donor, mat, name, tex[sheet], target=work, out=work,
                           place=by[which])

    print('--- socks')
    subdivide(work, 'maid_stockings')

    print('--- clear')
    clear_body(work, own)

    fit_socks(work)
    fit_bust(work)
    seat_hardware(work)

    print('--- ground')
    ground(work)

    print('--- weights')
    for name in BODY_FITTED:
        sl = slices(PMX(work))
        if name in sl:
            print(f'  {name}:')
            reweight(work, np.unique(sl[name]).tolist())

    rigid_weights(work)
    stiffen_foot(work)

    print('--- full body kept')
    shutil.copyfile(work, FULL)
    print(f'  wrote {FULL}, legs whole')

    if TRIM_LEGS:
        print('--- trim')
        trim_legs(work)

    print('--- dead physics out')
    prune_physics(work, own)

    print('--- collide')
    collide(work, own)

    print('--- ours out')
    for name in DROP:
        dropmaterial.main(work, name)

    shutil.rmtree(tmp, ignore_errors=True)
    os.replace(work, OUT)
    m = PMX(OUT)
    print(f'{OUT}: {len(m.vertices)} verts, {len(m.faces)//3} triangles, '
          f'{len(m.bones)} bones, {len(m.morphs)} morphs, {len(m.rigids)} rigid '
          f'bodies, {len(m.joints)} joints')
    print(f'  materials: {", ".join(x["name"] for x in m.materials)}')


if __name__ == '__main__':
    main()
