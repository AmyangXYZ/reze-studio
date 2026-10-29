# -*- coding: utf-8 -*-
"""Strip what removing a material leaves behind.

    python3 tools/head-graft/prune.py <in.pmx> <out.pmx> [--keep-edge] [--tex-max 2048]

Dropping a material takes its faces out and nothing else. reze-bomb replaces
the head, so face, eyes, mouth and hair all go -- and 5414 of their vertices
stayed in the buffer, skinned every frame and drawn never, along with the
fifteen dynamic rigid bodies that swing hair chains the model no longer shows.
None of it is visible and all of it is paid for.

Three passes, each measured on its way out:

  vertices  anything no face indexes is removed, faces renumbered, and vertex
            morphs renumbered with the offsets that pointed at removed
            vertices dropped rather than left dangling
  physics   a rigid body whose bone carries no weighted vertex has nothing to
            move, so it and every joint that names it go
  edge      the outline pass draws each flagged triangle a second time. The
            bomb gear is 71 thin shells and is the worst case for it, so new
            materials lose the flag unless --keep-edge

--tex-max also downsamples any texture wider or taller than the limit.
zzz_bomb_head.png ships at 4096x2048, which is 33.6 MB decoded for two
materials that use a corner of it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import PMX
from pmxw import write_pmx

SRC, OUT = sys.argv[1], sys.argv[2]
KEEP_EDGE = '--keep-edge' in sys.argv
NOEDGE = {'bomb_gear', 'bomb_mask'}
TEXMAX = int(sys.argv[sys.argv.index('--tex-max')+1]) if '--tex-max' in sys.argv else 0
FREEZE = (sys.argv[sys.argv.index('--freeze')+1].split(',')
          if '--freeze' in sys.argv else [])

m = PMX(SRC)
v0, r0, j0 = len(m.vertices), len(m.rigids), len(m.joints)

# --- vertices ------------------------------------------------------------------
used = sorted(set(m.faces))
newidx = {v: i for i, v in enumerate(used)}
m.vertices = [m.vertices[v] for v in used]
m.faces = [newidx[v] for v in m.faces]
kept = 0; lost = 0
for mo in m.morphs:
    if mo.get('type') != 1: continue          # vertex morph
    keep = [(newidx[o[0]],) + tuple(o[1:]) for o in mo['offsets'] if o[0] in newidx]
    lost += len(mo['offsets']) - len(keep); kept += len(keep)
    mo['offsets'] = keep
print('vertices %d -> %d (%d removed); morph offsets kept %d, dropped %d'
      % (v0, len(m.vertices), v0 - len(m.vertices), kept, lost))

# --- physics -------------------------------------------------------------------
live = set()
for v in m.vertices:
    for bi, w in zip(v[5], v[6]):
        if w > 1e-6: live.add(bi)
# A body is a candidate only if it names a real bone that no vertex follows.
# Bone index -1 is legal and means "joint network only" -- the breast bodies
# are built that way, and treating -1 as dead swept all six of them up and
# would have killed the chest physics.
def dead(i):
    r = m.rigids[i]
    if r.get('mode', 0) == 0: return False        # static bodies anchor things
    b = r.get('bone', -1)
    return 0 <= b < len(m.bones) and b not in live

# and it only goes if everything it is jointed to goes too, so a dead body
# anchoring a live chain is never pulled out from under it
link = {i: set() for i in range(len(m.rigids))}
for j in m.joints:
    a, b = j.get('a', -1), j.get('b', -1)
    if a in link and b in link: link[a].add(b); link[b].add(a)
drop = set()
for i in range(len(m.rigids)):
    if i in drop or not dead(i): continue
    comp, stack = set(), [i]
    while stack:
        k = stack.pop()
        if k in comp: continue
        comp.add(k)
        stack.extend(link[k] - comp)
    # Only the dynamic members have to be dead. Every chain is pinned to a
    # static anchor -- the hair hangs off the one on 頭 -- and counting that
    # anchor as live vetoed the whole chain. The anchor is shared, so it stays;
    # what hangs from it goes.
    dyn = {k for k in comp if m.rigids[k].get('mode', 0) != 0}
    if dyn and all(dead(k) for k in dyn): drop |= dyn
if drop:
    names = sorted({m.rigids[i]['name'] for i in drop})
    ren = {}; k = 0
    for i in range(len(m.rigids)):
        if i in drop: continue
        ren[i] = k; k += 1
    m.joints = [j for j in m.joints
                if j.get('a', -1) not in drop and j.get('b', -1) not in drop]
    for j in m.joints:
        for e in ('a', 'b'):
            if j.get(e, -1) in ren: j[e] = ren[j[e]]
    m.rigids = [r for i, r in enumerate(m.rigids) if i not in drop]
    print('rigid bodies %d -> %d, joints %d -> %d  (%s)'
          % (r0, len(m.rigids), j0, len(m.joints), ', '.join(names[:8])))
else:
    print('rigid bodies %d, joints %d -- every one drives live geometry' % (r0, j0))

# --- freeze --------------------------------------------------------------------
# Mode 0 is "follow the bone", so a chain set to it never leaves the pose it was
# authored in. That is the way to look at a graft's shape on its own: whatever
# the rest pose actually is, rather than wherever gravity and the joint limits
# drag it in the first frames. The joints between frozen bodies are dropped
# because two bone-following bodies have nothing to solve.
if FREEZE:
    froze = []
    for r in m.rigids:
        b = r.get('bone', -1)
        nm = m.bones[b]['name'] if 0 <= b < len(m.bones) else ''
        if r.get('mode', 0) != 0 and any(nm.startswith(px) for px in FREEZE):
            r['mode'] = 0; froze.append(r['name'])
    frozen_i = {i for i, r in enumerate(m.rigids) if r['name'] in set(froze)}
    before = len(m.joints)
    m.joints = [j for j in m.joints
                if not (j.get('a', -1) in frozen_i and j.get('b', -1) in frozen_i)]
    print('froze %d bodies to follow their bones, dropped %d joints: %s'
          % (len(froze), before - len(m.joints), ', '.join(sorted(froze)[:8])))

# --- outline -------------------------------------------------------------------
if not KEEP_EDGE:
    off = 0
    for mm in m.materials:
        if mm['name'] in NOEDGE and (mm.get('flag', 0) & 0x10):
            mm['flag'] = mm['flag'] & ~0x10
            print('  %s: edge outline off, %d fewer triangles drawn per frame'
                  % (mm['name'], mm['fcount']//3))
edge = sum(mm['fcount']//3 for mm in m.materials if mm.get('flag', 0) & 0x10)
print('triangles drawn twice for outline: %d of %d' % (edge, len(m.faces)//3))

# --- textures ------------------------------------------------------------------
if TEXMAX:
    from PIL import Image
    base = os.path.dirname(os.path.abspath(SRC))
    for i, t in enumerate(m.textures):
        p = os.path.join(base, *t.split('\\'))
        # only the sheets this pipeline brought in. 2.png and the rest are the
        # model's own art, shared with every other build, and resizing one in
        # place would quietly shrink it for all of them.
        if not os.path.basename(p).startswith('zzz_'): continue
        if not os.path.exists(p): continue
        im = Image.open(p)
        if max(im.size) <= TEXMAX: continue
        s = TEXMAX/max(im.size)
        sz = (max(1, int(im.size[0]*s)), max(1, int(im.size[1]*s)))
        im.resize(sz, Image.LANCZOS).save(p)
        print('  %s %sx%s -> %sx%s (%.1f -> %.1f MB decoded)'
              % (os.path.basename(p), im.size[0], im.size[1], sz[0], sz[1],
                 im.size[0]*im.size[1]*4/1e6, sz[0]*sz[1]*4/1e6))

write_pmx(OUT, m)
print('wrote %s' % OUT)
