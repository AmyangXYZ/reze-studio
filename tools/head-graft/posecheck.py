# -*- coding: utf-8 -*-
"""Pose a model's skeleton and bake the skinned result, to see what motion does.

    python3 tools/head-graft/posecheck.py <in.pmx> <out.pmx>

Every graft failure so far shipped because only the rest pose was ever
rendered, and the rest pose cannot expose a wrong weight -- geometry authored
in place looks right no matter which bones it follows. This applies a fixed
test pose by linear blend skinning (forward kinematics over the parent chain,
rotations about each bone's own origin, the way MMD applies them) and writes
the deformed positions back, so the ordinary renderer shows what an animator
would see. IK, appends and physics are not simulated; they do not need to be
-- a wrong weight tears the mesh under any rotation.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import PMX
from pmxw import write_pmx
import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]

def rot(axis, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    if axis == 'x': return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y': return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# a nod, both arms down and one elbow bent, a waist twist, one leg raised with
# a bent knee: every region the grafts have broken, in one pose
POSE = {
    '頭':   rot('x', -20) @ rot('z', 12),
    '首':   rot('x', -8),
    '上半身': rot('y', 18),
    '左腕':  rot('z', -35),
    '右腕':  rot('z', 35),
    '左ひじ': rot('y', -55),
    '右手首': rot('z', 30),
    # this model's leg vertices ride the D bones, which inherit the plain leg
    # bones through append -- not simulated here, so the D bones are posed
    # directly; for weight testing that is the same rotation
    '左足':  rot('x', -35), '左足D':  rot('x', -35),
    '左ひざ': rot('x', 40),  '左ひざD': rot('x', 40),
    '右足':  rot('x', 10),  '右足D':  rot('x', 10),
    '上半身2': rot('y', 10),
}

m = PMX(SRC)
n = len(m.bones)
name = [b['name'] for b in m.bones]
rest = np.array([b['pos'] for b in m.bones], dtype=float)
parent = [b.get('parent', -1) for b in m.bones]

Rw = [None]*n; Pw = [None]*n
order = sorted(range(n), key=lambda i: 0 if parent[i] < 0 else 1)
# parents may appear after children in index order; resolve by walking depth
def solve(i):
    if Rw[i] is not None: return
    p = parent[i]
    Rl = POSE.get(name[i], np.eye(3))
    if p < 0 or p >= n:
        Rw[i] = Rl; Pw[i] = rest[i]
    else:
        solve(p)
        Rw[i] = Rw[p] @ Rl
        Pw[i] = Pw[p] + Rw[p] @ (rest[i] - rest[p])
sys.setrecursionlimit(10000)
for i in range(n): solve(i)

moved = 0
for vi, v in enumerate(m.vertices):
    bones, ws = v[5], v[6]
    if all(name[b] not in POSE and Rw[b] is None for b in bones): pass
    p = np.array(v[0])
    out = np.zeros(3); tot = 0.0
    for b, w in zip(bones, ws):
        if w <= 1e-6: continue
        out += w*(Rw[b] @ (p - rest[b]) + Pw[b])
        tot += w
    if tot > 1e-6:
        q = out/tot
        if np.linalg.norm(q - p) > 1e-4: moved += 1
        m.vertices[vi] = ([float(x) for x in q],) + tuple(v[1:])
print('posed %d bones, %d of %d vertices moved' % (len(POSE), moved, len(m.vertices)))

# how much each material tears: the ratio of a posed edge to its rest length.
# clean skinning stretches a little at the joints; wrong weights stretch a lot
rest_v = PMX(SRC).vertices
off = 0
print('%-14s %8s %8s   (edge stretch, p99 / max)' % ('material', 'p99', 'max'))
for mm in m.materials:
    idx = m.faces[off:off+mm['fcount']]; off += mm['fcount']
    if not idx: continue
    rat = []
    step = max(3, (len(idx)//3//1200)*3)
    for k in range(0, len(idx) - 2, step):
        a, b = idx[k], idx[k+1]
        r0 = math.dist(rest_v[a][0], rest_v[b][0])
        if r0 < 1e-6: continue
        rat.append(math.dist(m.vertices[a][0], m.vertices[b][0])/r0)
    if len(rat) > 20:
        rat.sort()
        print('%-14s %8.2f %8.2f' % (mm['name'], rat[int(len(rat)*0.99)], rat[-1]))
write_pmx(OUT, m)
print('wrote %s' % os.path.basename(OUT))
