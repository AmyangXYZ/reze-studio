# -*- coding: utf-8 -*-
"""Build the mod's own hair rig instead of borrowing the host's.

    python3 tools/head-graft/zzzhairrig.py <in.pmx> <out.pmx> <moddir>
            [--material zzz_hair] [--stiff 10]

The mod is not only meshes. EvelynHairBlend.buf carries the author's own
skinning for the hair -- 24 bones, weights summing to exactly 1.0 -- and those
bones are laid out in chains: 8, 9, 10, 11 stepping down one side, 18, 20, 21
down the other, 3, 5, 6, 7 through the fringe, 12 to 17 across the back. The
author rigged this hair, because the character it replaces wears a completely
different one.

Binding the mesh to the host's chains instead throws all of that away and asks
proximity to guess it back, which it cannot. The host's rig is asymmetric --
右侧发 packs four links into the span 左侧发 covers with seven -- so the two
sides of a symmetric cut ended up on chains of different lengths, and the side
riding the deeper links swung further and dragged out of shape. And the host's
lowest hair bone sits at y 18.66 while this cut hangs to 17.54, so its ends had
no bone at their own height at all.

So the rig comes across with the mesh. Each of the mod's bones becomes an MMD
bone at the weighted centroid of the vertices it drives; each is parented to
the nearest bone above it in its own cluster, which recovers the chains; each
gets a capsule and a joint copied from the host's own hair bodies so the
damping and limits match the model it lives in. The vertices are then bound by
the mod's weights directly -- no proximity, no guessing.

Bone 0 is the scalp, 4034 of 4105 vertices: it is the head, and stays rigid.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import PMX
from pmxw import write_pmx
import numpy as np

SRC, OUT, MODD = sys.argv[1], sys.argv[2], sys.argv[3]
def arg(n, d): return sys.argv[sys.argv.index(n)+1] if n in sys.argv else d
MAT = arg('--material', 'zzz_hair')
SELF = '--no-self-collide' in sys.argv
# The host's own hair joints, which is what this rig has to live beside: 15 of
# its 23 use +-10 on x and z with y locked at 0. Taken rather than invented.
LIMIT = float(arg('--limit', '10'))       # joint swing, degrees

m = PMX(SRC)
def read(n): return open(os.path.join(MODD, n), 'rb').read()

bb = np.frombuffer(read('EvelynHairBlend.buf'), dtype=np.uint8).reshape(-1, 32)
BW = bb[:, 0:16].copy().view(np.float32).reshape(-1, 4)
BI = bb[:, 16:32].copy().view(np.uint32).reshape(-1, 4).astype(np.int64)
tri = np.concatenate([np.frombuffer(read(p + '.ib'), dtype=np.uint32)
                      for p in ('EvelynHairA', 'EvelynHairB')])
mod_used = sorted({int(v) for v in np.unique(tri)})

# --- the grafted vertices, in the order zzzgraft appended them ----------------
off = 0; hair_v = None
for mm in m.materials:
    if mm['name'] == MAT:
        hair_v = sorted(set(m.faces[off:off+mm['fcount']])); break
    off += mm['fcount']
if hair_v is None: raise SystemExit('no material %r' % MAT)
if len(hair_v) != len(mod_used):
    raise SystemExit('material has %d vertices, mod hair has %d -- rebuild the '
                     'graft first so the two line up' % (len(hair_v), len(mod_used)))
mod_of = {hv: mv for hv, mv in zip(hair_v, mod_used)}   # graft order is sorted both sides
P = {hv: np.array(m.vertices[hv][0]) for hv in hair_v}

# --- one MMD bone per mod bone, at the centroid of what it drives -------------
influence = {}
for hv in hair_v:
    mv = mod_of[hv]
    for bi, w in zip(BI[mv], BW[mv]):
        if w > 1e-6: influence.setdefault(int(bi), []).append((hv, float(w)))

head = next(i for i, b in enumerate(m.bones) if b['name'] == '頭')
ROOT = min(influence, key=lambda b: -len(influence[b]))     # the scalp bone
print('mod hair bones: %d; bone %d drives %d of %d vertices and is the scalp'
      % (len(influence), ROOT, len(influence[ROOT]), len(hair_v)))

cen = {}
for b, lst in influence.items():
    if b == ROOT or len(lst) < 8: continue
    W = np.array([w for _, w in lst])
    Q = np.array([P[hv] for hv, _ in lst])
    cen[b] = (Q*W[:, None]).sum(axis=0)/W.sum()

# --- recover the chains -------------------------------------------------------
# Each link hangs from the nearest bone above it whose centroid is close in the
# plane; that is what a strand is. Anything with nothing above it starts a chain
# on the head.
# Adjacency comes from the skinning, not from proximity. Two consecutive links
# of a strand share a band of vertices that blend between them, so the pair
# that co-occurs most is the pair that is joined. Parenting by nearest-bone-
# above instead let a bone at the back of the head adopt a side strand, because
# it happened to sit close in space while driving completely different hair.
order = sorted(cen, key=lambda b: -cen[b][1])
co = {}
for hv in hair_v:
    mv = mod_of[hv]
    act = [(int(bi), float(w)) for bi, w in zip(BI[mv], BW[mv]) if w > 0.05]
    for i in range(len(act)):
        for j in range(i+1, len(act)):
            a, b = act[i][0], act[j][0]
            k = (min(a, b), max(a, b))
            co[k] = co.get(k, 0.0) + min(act[i][1], act[j][1])
# The scalp is in the table too, and that is what identifies a strand's root: a
# link that blends into the scalp more than into any link above it is growing
# out of the head, not hanging off another strand. Without this the bone at the
# back of the head adopted the whole side chain -- they do share vertices where
# back meets side -- and every strand ended up hanging off one tree.
# Normalised by the smaller of the two bones' total weight. Raw co-occurrence
# is dominated by the scalp simply because it is 3714 vertices, so every link
# read as growing straight out of the head and the strands came apart into
# seventeen stubs. What matters is the fraction of a link that is shared, not
# the count.
tot = {}
for hv in hair_v:
    mv = mod_of[hv]
    for bi, w in zip(BI[mv], BW[mv]):
        if w > 1e-6: tot[int(bi)] = tot.get(int(bi), 0.0) + float(w)
def share(a, b):
    v = co.get((min(a, b), max(a, b)), 0.0)
    return v/max(min(tot.get(a, 1e-9), tot.get(b, 1e-9)), 1e-9)

parent_of = {}
for b in order:
    best, bs = None, share(ROOT, b)
    for a in order:
        if a == b or cen[a][1] <= cen[b][1] + 1e-6: continue
        w = share(a, b)
        if w > bs: best, bs = a, w
    parent_of[b] = best

proto_r = next(r for r in m.rigids if '侧发' in r['name'])
proto_j = next(j for j in m.joints if '侧发' in j['name'])

newbone = {}
for b in order:
    idx = len(m.bones)
    m.bones.append({'name': 'zh_%d' % b, 'name_en': 'zh_%d' % b,
                    'pos': [float(x) for x in cen[b]],
                    'parent': head, 'layer': 0, 'flag': 27, 'tail': -1})
    newbone[b] = idx
for b in order:
    p = parent_of[b]
    m.bones[newbone[b]]['parent'] = newbone[p] if p is not None else head
    if p is not None: m.bones[newbone[p]]['tail'] = newbone[b]
roots = [b for b in order if parent_of[b] is None]
print('  %d links in %d chains' % (len(order), len(roots)))

# --- a capsule per link, a joint per parent-child pair ------------------------
lim = math.radians(LIMIT)
rig_of = {}
# The radius is the strand's thickness, measured across the chain axis and only
# over the vertices this link actually owns. Taking the mean distance from the
# centroid over everything it touches measures how far the strand REACHES, not
# how thick it is, and produced capsules of median radius 0.419 against the
# host's 0.135 -- fat enough to collide with the skull and each other and shove
# the hair off the head. Clamped to the range the host's own hair uses.
R_MIN, R_MAX = 0.05, 0.18
for b in order:
    lst = influence[b]
    own = np.array([P[hv] for hv, w in lst if w > 0.5])
    if len(own) < 4: own = np.array([P[hv] for hv, _ in lst])
    p_ = parent_of[b]
    axis = cen[b] - (cen[p_] if p_ is not None else np.array(m.bones[head]['pos']))
    n_ = float(np.linalg.norm(axis))
    axis = axis/n_ if n_ > 1e-6 else np.array([0.0, -1.0, 0.0])
    d_ = own - cen[b]
    perp = np.linalg.norm(d_ - np.outer(d_ @ axis, axis), axis=1)
    # the 40th percentile, not the median: a strand's silhouette is set by the
    # bulk of it, and the outliers are the wisps that stray from the core
    rad = float(np.clip(np.percentile(perp, 40), R_MIN, R_MAX))
    child = [c for c in order if parent_of[c] == b]
    length = float(np.linalg.norm(cen[child[0]] - cen[b])) if child else rad*2.0
    r = dict(proto_r)
    r['name'] = 'zh_%d' % b; r['name_en'] = 'zh_%d' % b
    r['bone'] = newbone[b]
    r['pos'] = [float(x) for x in cen[b]]
    r['rot'] = [0.0, 0.0, 0.0]
    r['shape'] = 2                       # capsule, as the host's hair uses
    r['size'] = [rad*0.8, max(length, rad*2.0), 0.0]
    r['mode'] = 2                        # dynamic, follows the bone's position
    if SELF:
        # clear the hair group's own bit so the capsules pass through each
        # other; the joints still hold the chains together
        r['nocollide'] = r['nocollide'] | (1 << r['group'])
    rig_of[b] = len(m.rigids)
    m.rigids.append(r)
# the chain roots are pinned to the head by a static body it already has
head_rigid = next((i for i, r in enumerate(m.rigids)
                   if r.get('bone') == head and r.get('mode', 0) == 0), None)
if head_rigid is None:
    head_rigid = len(m.rigids)
    hr = dict(proto_r)
    hr.update({'name': 'zh_head', 'name_en': 'zh_head', 'bone': head, 'mode': 0,
               'shape': 0, 'size': [0.5, 0.0, 0.0],
               'pos': [float(x) for x in m.bones[head]['pos']],
               'rot': [0.0, 0.0, 0.0]})
    m.rigids.append(hr)

nj = 0
for b in order:
    p = parent_of[b]
    a_idx = rig_of[p] if p is not None else head_rigid
    j = dict(proto_j)
    j['name'] = 'zh_%d' % b; j['name_en'] = 'zh_%d' % b
    j['a'] = a_idx; j['b'] = rig_of[b]
    j['pos'] = [float(x) for x in cen[b]]
    j['rot'] = [0.0, 0.0, 0.0]
    j['lin_min'] = [0.0, 0.0, 0.0]; j['lin_max'] = [0.0, 0.0, 0.0]
    j['ang_min'] = [-lim, 0.0, -lim]     # y locked: hair swings, it does not twist
    j['ang_max'] = [lim, 0.0, lim]
    m.joints.append(j); nj += 1
print('  %d capsules, %d joints, swing +-%.0f degrees' % (len(order), nj, LIMIT))

# --- bind the mesh by the mod's own weights ----------------------------------
moved = 0
for hv in hair_v:
    mv = mod_of[hv]
    acc = {}
    for bi, w in zip(BI[mv], BW[mv]):
        w = float(w)
        if w <= 1e-6: continue
        bi = int(bi)
        tgt = newbone.get(bi, head)      # the scalp bone and any dropped one ride 頭
        acc[tgt] = acc.get(tgt, 0.0) + w
    it = sorted(acc.items(), key=lambda kv: -kv[1])[:4]
    tot = sum(x for _, x in it) or 1.0
    bones = [b for b, _ in it] + [0]*(4-len(it))
    ws = [x/tot for _, x in it] + [0.0]*(4-len(it))
    wt = 0 if len(it) <= 1 else (1 if len(it) == 2 else 2)
    if wt == 0: bones, ws = [bones[0]], [1.0]
    elif wt == 1: bones, ws = bones[:2], ws[:2]
    v = m.vertices[hv]
    m.vertices[hv] = v[:4] + (wt, bones, ws) + v[7:]
    if bones[0] != head: moved += 1
print('  %d of %d hair vertices now ride the mod\'s own chains' % (moved, len(hair_v)))

write_pmx(OUT, m)
print('wrote %s' % os.path.basename(OUT))
