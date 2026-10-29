# -*- coding: utf-8 -*-
"""Apply the Reze mod to the 伊芙琳 MMD model, grounded in the mod's own data.

    python3 tools/head-graft/evelyngraft.py <host.pmx> <out.pmx> <moddir>
            [--form normal|bomb]

This replaces zzzgraft.py, which grew by patches until its failures were
structural. The two decisions that make this version different:

NO PROXIMITY WEIGHTS. The mod ships its own skinning -- EvelynBodyBlend.buf,
four float32 weights then four uint32 game-bone indices per vertex, summing to
exactly 1.0 across 114 bones. The old tool ignored it and copied weights from
whatever host vertex was nearest, which put hands on thumb bones and twisted
the arms. Here every game bone is located at the weighted centroid of the
vertices it drives, matched to the host bone whose own weighted centroid is
nearest, and the mod's weights are carried over through that map verbatim.
The same correspondences then SOLVE the body's scale and offset by least
squares -- the transform and the rig come from the same measurement.

THE HEAD IS FITTED TO THE HEAD. The host is an MMD build with a proportionally
larger head than the game model, so one similarity transform cannot both seat
the neck and cover the skull -- that trade cost many rounds. The mod's BodyA
ends in a low-poly proxy of the vanilla head; ICP fits that proxy onto the
host's actual head (颜 plus 肌's neck and scalp), and the hair rides THAT
transform, landing on the skull it must wrap. The body rides its own.

肌 is trimmed, not deleted. It carries the scalp to y 20.06 and the neck; the
old tool kept only a neck tube and the head was hollow behind the hair. The
neck tube and the whole scalp stay; only chest, shoulders, arms and legs go.

Hair winding, face-texture v-flip, neck tone matching, and the mod's own hair
rig (zzzhairrig.py, run after this) are carried over from what already worked.
"""
import sys, os, math, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import PMX
from pmxw import write_pmx
import numpy as np
from PIL import Image

SRC, OUT, MODD = sys.argv[1], sys.argv[2], sys.argv[3]
def arg(n, d): return sys.argv[sys.argv.index(n)+1] if n in sys.argv else d
FORM = arg('--form', 'normal')

BODY_GONE = {'衣', '外套', '外套+', '发带', '金属', '领带', '领带+', '皮裤',
             '衬衣', '珠宝', '黑丝衣', '穗', '胸衣', '武器'}
HAIR_GONE = {'发', '侧发', '前发'}
FACE_MATS = {'颜', '痣', '眉', '眉睫影', '白目', '目', '目光', '目光2', '睫',
             '口', '舌', '齿', '目影'}
# host materials whose vertices define where a host bone "is"
HOST_SKIN = {'肌', '衣', '皮裤', '衬衣', '金属', '黑丝衣', '胸衣', '穗'}

m = PMX(SRC)
host_dir = os.path.dirname(os.path.abspath(SRC))
def read(n): return open(os.path.join(MODD, n), 'rb').read()

def mat_verts(name):
    off = 0
    for mm in m.materials:
        if mm['name'] == name:
            return sorted(set(m.faces[off:off+mm['fcount']]))
        off += mm['fcount']
    return []

bone_idx = {b['name']: i for i, b in enumerate(m.bones)}
NECK = np.array(m.bones[bone_idx['首']]['pos'])
HEAD = bone_idx['頭']

# --- mod buffers --------------------------------------------------------------
def load_part(stem):
    pos = np.frombuffer(read(stem + 'Position.buf'), dtype=np.uint8).reshape(-1, 40)
    xyz = pos[:, 0:12].copy().view(np.float32).reshape(-1, 3)
    nrm = pos[:, 12:24].copy().view(np.float32).reshape(-1, 3)
    tb = read(stem + 'Texcoord.buf')
    T = np.frombuffer(tb, dtype=np.uint8).reshape(-1, len(tb)//len(xyz))
    uv = T[:, 4:8].copy().view(np.float16).reshape(-1, 2).astype(np.float32)
    bb = np.frombuffer(read(stem + 'Blend.buf'), dtype=np.uint8).reshape(-1, 32)
    bw = bb[:, 0:16].copy().view(np.float32).reshape(-1, 4)
    bi = bb[:, 16:32].copy().view(np.uint32).reshape(-1, 4).astype(np.int64)
    # mod space (x, depth, up) -> MMD (x, up, depth); det -1, so winding is
    # reversed at emission and the componentwise-mapped normals are kept
    V = np.stack([xyz[:, 0], xyz[:, 2], xyz[:, 1]], axis=1)
    N = np.stack([nrm[:, 0], nrm[:, 2], nrm[:, 1]], axis=1)
    return V, N, uv, bw, bi

BODY_IB = 'EvelynBodyB' if FORM == 'bomb' else 'EvelynBodyA'
bV, bN, bUV, bW, bI = load_part('EvelynBody')
bTri = np.frombuffer(read(BODY_IB + '.ib'), dtype=np.uint32).reshape(-1, 3).astype(np.int64)
aTri = np.frombuffer(read('EvelynBodyA.ib'), dtype=np.uint32).reshape(-1, 3).astype(np.int64)
aU = np.unique(aTri)

# --- the throat, on both sides ------------------------------------------------
# walking up from the shoulders, width falls to the throat and rises into the
# head; measured on BodyA because BodyB's devil head would move the walk
z0_, z1_ = float(bV[aU][:, 1].min()), float(bV[aU][:, 1].max())
prof = []
for zz in np.arange(z0_ + 0.85*(z1_-z0_), z0_ + 0.985*(z1_-z0_), 0.005):
    sl = bV[aU][(bV[aU][:, 1] >= zz) & (bV[aU][:, 1] < zz + 0.005)]
    if len(sl) > 8: prof.append((zz, float(sl[:, 0].max() - sl[:, 0].min())))
# between 85% and 98% of the figure's height the only constriction is the
# throat -- shoulders are below, the head proxy above. A walk from the widest
# slab stalled on the arms and cut the collar off with the head.
MOD_NECK = min(prof, key=lambda t: t[1])[0]
print('throat: mod z %.3f (%.0f%% of height)' % (MOD_NECK, 100*(MOD_NECK-z0_)/(z1_-z0_)))

# --- game bones located, host bones located, transform SOLVED -----------------
game_tot = {}; game_acc = {}
for v in aU:
    for gb, w in zip(bI[v], bW[v]):
        if w <= 1e-6: continue
        gb = int(gb)
        game_tot[gb] = game_tot.get(gb, 0.0) + float(w)
        game_acc[gb] = game_acc.get(gb, np.zeros(3)) + float(w)*bV[v]
game_cent = {gb: game_acc[gb]/game_tot[gb] for gb in game_tot if game_tot[gb] > 2.0}

host_tot = {}; host_acc = {}
for nm in HOST_SKIN:
    for v in mat_verts(nm):
        p = np.array(m.vertices[v][0])
        for hb, w in zip(m.vertices[v][5], m.vertices[v][6]):
            if w <= 1e-6: continue
            host_tot[hb] = host_tot.get(hb, 0.0) + float(w)
            host_acc[hb] = host_acc.get(hb, np.zeros(3)) + float(w)*p
# only the deform skeleton may be matched: 金属 and the like weight to
# accessory chains ("足4"), and legs matched to those dragged the whole solve
OKTOK = ('センター', '上半身', '下半身', '首', '頭', '肩', '腕', 'ひじ', '手首',
         '親指', '人指', '中指', '薬指', '小指', '足', 'ひざ', '足首', 'つま先',
         '捩', '胸')
BADTOK = ('物理', '発', '外套', '环', '穗', '飘')
def deform_bone(hb):
    nm = m.bones[hb]['name']
    return any(t in nm for t in OKTOK) and not any(t in nm for t in BADTOK) \
        and not (nm[-1].isdigit() and '指' not in nm and '半身' not in nm)
host_cent = {hb: host_acc[hb]/host_tot[hb] for hb in host_tot
             if host_tot[hb] > 2.0 and deform_bone(hb)}
hc_idx = list(host_cent)
hc_pos = np.array([host_cent[h] for h in hc_idx])

feet_host = min(np.array(m.vertices[v][0])[1]
                for nm in HOST_SKIN for v in mat_verts(nm))
# --- one uniform transform for positions --------------------------------------
S_LOC = (NECK[1] - feet_host)/(MOD_NECK - z0_)
hostskin_v = [v for nm in HOST_SKIN for v in mat_verts(nm)]
hostskin = np.array([m.vertices[v][0] for v in hostskin_v])
mod_mid = (bV[aU].min(axis=0) + bV[aU].max(axis=0))/2.0
host_mid = (hostskin.min(axis=0) + hostskin.max(axis=0))/2.0
T_LOC = np.array([host_mid[0] - mod_mid[0]*S_LOC,
                  feet_host - z0_*S_LOC,
                  host_mid[2] - mod_mid[2]*S_LOC])
def rigid_place(v):
    return S_LOC*bV[v] + T_LOC

# --- the palette table, learned by region voting ------------------------------
# The ini binds Blend.buf straight into the vanilla draw call, so its indices
# are the game's own bone palette for this character -- and the host model was
# ripped from that same game, its weights converted from that same palette.
# The host mesh is therefore the ground truth for what each slot means. For
# every mod vertex, the host vertex under it says which host bones own that
# region; each game bone collects those votes across all of its weight, and
# the table falls out. Matching bones by centroid was a single fragile point
# per bone and put the game knee on the host thigh; a vote aggregated over
# thousands of vertices does not care that the two bodies' proportions differ.
P_all = bV[aU]*S_LOC + T_LOC
nearest = np.empty(len(aU), dtype=np.int64)
for i in range(0, len(aU), 256):
    d = ((P_all[i:i+256, None, :] - hostskin[None, :, :])**2).sum(axis=2)
    nearest[i:i+256] = d.argmin(axis=1)
votes = {}
for k, v in enumerate(aU):
    hv = m.vertices[hostskin_v[nearest[k]]]
    for gb, w in zip(bI[v], bW[v]):
        if w <= 1e-6: continue
        gb = int(gb)
        for hb, hw in zip(hv[5], hv[6]):
            if hw <= 1e-6: continue
            votes.setdefault(gb, {})[hb] = votes.get(gb, {}).get(hb, 0.0) + float(w)*float(hw)
bonedist = {}
for gb, tab in votes.items():
    cand = sorted(((hb, w) for hb, w in tab.items() if deform_bone(hb)),
                  key=lambda kv: -kv[1])
    if not cand: cand = sorted(tab.items(), key=lambda kv: -kv[1])
    top = cand[0][1]
    keep = [(hb, w) for hb, w in cand[:2] if w >= 0.35*top]
    tot = sum(w for _, w in keep)
    bonedist[gb] = [(hb, w/tot) for hb, w in keep]
print('  palette: %d slots learned by region voting; positions at uniform %.2f'
      % (len(bonedist), S_LOC))
big = sorted(votes, key=lambda g: -sum(votes[g].values()))[:14]
for g in big:
    print('    slot %3d -> %s' % (g, ' + '.join('%s %.0f%%' % (m.bones[hb]['name'], f*100)
                                                for hb, f in bonedist[g])))

# --- rest-pose retarget --------------------------------------------------------
# The two models do NOT share a bind pose. The host rests its upper arm at a
# 39.3 degree down-angle with the wrist bone at y 13.89; the mod's hand
# geometry sits at y 11.25, arms hanging near 53 degrees. So the mesh's elbow
# is nowhere near the host's elbow bone, and rotating that bone swings the arm
# about a point that is not its joint -- which is every arm and shoulder
# failure in this model, under any weighting. Weights alone cannot repair a
# pose difference; the mesh has to be carried into the host's rest pose first.
#
# Each host bone gets a rigid transform taking its limb segment from where the
# mod rests it to where the host rests it. A bone's joint is read from the mesh
# rather than assumed: it lies in the band where the bone's influence hands
# over to its parent's, so the weighted centroid of the vertices that both
# share is the hinge. The segment's direction is joint-to-child, and rotating
# the mod's direction onto the host's, about their common perpendicular, gives
# the rotation. The mesh is then blended through those transforms by its own
# weights -- the same linear blend the viewer will use, so a vertex spanning
# shoulder and arm bends smoothly instead of tearing.
prim = {}
for g, dist in bonedist.items(): prim[g] = dist[0][0]
slots_of = {}
for g, hb in prim.items(): slots_of.setdefault(hb, []).append(g)

hb_parent = {}
for hb in slots_of:
    p = m.bones[hb].get('parent', -1)
    seen = 0
    while 0 <= p < len(m.bones) and seen < 40:
        if p in slots_of: break
        p = m.bones[p].get('parent', -1); seen += 1
    hb_parent[hb] = p if (0 <= p < len(m.bones) and p in slots_of) else -1

wsum = {}
for k, v in enumerate(aU):
    for g, w in zip(bI[v], bW[v]):
        if w > 1e-6:
            wsum.setdefault(prim[int(g)], {})[int(v)] = \
                wsum.setdefault(prim[int(g)], {}).get(int(v), 0.0) + float(w)

def mod_joint(hb):
    """Where hb's influence hands over to its parent's -- the hinge."""
    par = hb_parent.get(hb, -1)
    mine = wsum.get(hb, {})
    if not mine: return None
    if par >= 0 and wsum.get(par):
        theirs = wsum[par]
        both = [(vv, min(mine[vv], theirs[vv])) for vv in mine if vv in theirs]
        both = [(vv, w) for vv, w in both if w > 0.12]
        if len(both) >= 6:
            W_ = np.array([w for _, w in both])
            P_ = np.array([bV[vv] for vv, _ in both])
            return (P_*W_[:, None]).sum(axis=0)/W_.sum()
    W_ = np.array(list(mine.values()))
    P_ = np.array([bV[vv] for vv in mine])
    c = (P_*W_[:, None]).sum(axis=0)/W_.sum()
    if par >= 0 and wsum.get(par):
        Wp = np.array(list(wsum[par].values()))
        Pp = np.array([bV[vv] for vv in wsum[par]])
        cp = (Pp*Wp[:, None]).sum(axis=0)/Wp.sum()
        return c + 0.5*(cp - c)      # halfway toward the parent
    return c

MJ = {hb: mod_joint(hb) for hb in slots_of}
MJ = {hb: j for hb, j in MJ.items() if j is not None}
hb_kids = {}
for hb, par in hb_parent.items():
    if par >= 0: hb_kids.setdefault(par, []).append(hb)

def dir_of(hb, joints, host):
    kids = [k for k in hb_kids.get(hb, []) if k in joints]
    if not kids: return None
    if host:
        base = np.array(m.bones[hb]['pos'])
        tip = np.mean([np.array(m.bones[k]['pos']) for k in kids], axis=0)
    else:
        base = joints[hb]
        tip = np.mean([joints[k] for k in kids], axis=0)
    d = tip - base
    n = np.linalg.norm(d)
    return d/n if n > 1e-6 else None

def rot_between(a, b):
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx*(1.0/(1.0 + c))

# Composed down the chain, parent first, exactly as FK composes. Solving each
# bone's rotation on its own gave neighbouring bones unrelated rotations, and
# the mesh blended between them stretched -- the forearms tapered to points.
# A child's rotation is now its parent's, plus only the local correction its
# own segment needs, so a chain that needs no correction inherits the parent's
# transform unchanged and nothing is torn between them.
XF = {}
depth = {}
for hb in slots_of:
    d, p, seen = 0, hb_parent.get(hb, -1), 0
    while p >= 0 and seen < 40:
        d += 1; p = hb_parent.get(p, -1); seen += 1
    depth[hb] = d
# The trunk is NOT retargeted. The bind-pose difference between these two
# models is in the limbs -- the mod rests its arms near 53 degrees where the
# host rests them at 39 -- while spine, neck and head agree. Retargeting the
# trunk moved the body's neck away from a face that stayed put, and the head
# came apart. The trunk keeps the plain uniform placement, which is what the
# face, the kept 肌 neck and the hair are all built against; only the limbs
# are carried into the host's rest pose.
TRUNK = ('センター', '上半身', '下半身', '首', '頭', '胸', '肩')
for hb in sorted(slots_of, key=lambda h: depth[h]):
    if hb not in MJ: continue
    nm = m.bones[hb]['name']
    if any(t in nm for t in TRUNK):
        XF[hb] = (np.eye(3), T_LOC)
        continue
    par = hb_parent.get(hb, -1)
    Rp, tp = XF[par] if par in XF else (np.eye(3), T_LOC)
    a = dir_of(hb, MJ, False); b = dir_of(hb, MJ, True)
    R = rot_between(Rp @ a, b) @ Rp if (a is not None and b is not None) else Rp
    src = MJ[hb]*S_LOC + T_LOC
    XF[hb] = (R, np.array(m.bones[hb]['pos']) - R @ src)
print('  trunk held at the uniform transform; %d limb bones retargeted'
      % sum(1 for hb in XF if not any(t in m.bones[hb]['name'] for t in TRUNK)))
moved = np.mean([np.linalg.norm(np.array(m.bones[hb]['pos']) - (MJ[hb]*S_LOC + T_LOC))
                 for hb in XF if hb in MJ])
print('  rest retarget: %d bone transforms, mean joint move %.2f' % (len(XF), moved))

def retarget(v):
    p = S_LOC*bV[v] + T_LOC
    out = np.zeros(3); tot = 0.0
    for g, w in zip(bI[v], bW[v]):
        if w <= 1e-6: continue
        hb_ = prim.get(int(g))
        xf = XF.get(hb_) if hb_ is not None else None
        if xf is None:
            out += float(w)*p; tot += float(w); continue
        out += float(w)*(xf[0] @ p + xf[1]); tot += float(w)
    return out/tot if tot > 1e-6 else p

# --- the head, fitted to the head ---------------------------------------------
stub = bV[aU][bV[aU][:, 1] > MOD_NECK - 0.05]
tgt = [np.array(m.vertices[v][0]) for v in mat_verts('颜')]
for v in mat_verts('肌'):
    p = m.vertices[v][0]
    if p[1] > 18.0: tgt.append(np.array(p))
tgt = np.array(tgt)
# Scale is the ratio of the two heads' widths -- solving it inside the ICP
# collapsed the stub to a third of its size, because with partial overlap a
# smaller cloud is always nearer to something. Only the position is iterated.
stub_head = stub[stub[:, 1] > MOD_NECK + 0.03]
host_head = tgt[tgt[:, 1] > 18.9]
S_h = (np.percentile(host_head[:, 0], 99) - np.percentile(host_head[:, 0], 1)) / \
      max(np.percentile(stub_head[:, 0], 99) - np.percentile(stub_head[:, 0], 1), 1e-9)
T_h = tgt.mean(axis=0) - S_h*stub.mean(axis=0)
for it in range(8):
    p = stub*S_h + T_h
    d = ((p[:, None, :] - tgt[None, :, :])**2).sum(axis=2)
    q = tgt[d.argmin(axis=1)]
    T_h = T_h + (q - p).mean(axis=0)
res_h = np.sqrt(((stub*S_h + T_h - q)**2).sum(axis=1))
print('head: proxy fitted by ICP, scale %.2f, residual median %.2f'
      % (S_h, np.median(res_h)))

# --- emit a mod part ----------------------------------------------------------
tex_dir = os.path.join(host_dir, 'tex_zzz')
os.makedirs(tex_dir, exist_ok=True)
src_tex = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'public', 'models', 'reze', 'textures_zzz')

new_faces = []; new_mats = []
def emit(label, V, N, UV, tri, sheet, proto_name, place, weights):
    used = sorted({int(v) for v in np.unique(tri)})
    remap = {}
    for v in used:
        n = N[v]; L = float(np.linalg.norm(n)) or 1.0
        wt, bones, ws = weights(v)
        remap[v] = len(m.vertices)
        m.vertices.append((list(place(v)), list(n/L),
                           [float(UV[v][0]), float(UV[v][1])], [],
                           wt, bones, ws, 1.0))
    faces = [remap[int(x)] for x in tri[:, ::-1].ravel()]
    p = os.path.join(src_tex, sheet)
    if os.path.exists(p): shutil.copyfile(p, os.path.join(tex_dir, sheet))
    rel = 'tex_zzz\\' + sheet
    if rel not in m.textures: m.textures.append(rel)
    proto = next(mm for mm in m.materials if mm['name'] == proto_name)
    mat = dict(proto)
    mat['name'] = label; mat['name_en'] = label
    mat['tex'] = m.textures.index(rel)
    mat['fcount'] = len(faces)
    # double-sided: their viewer culls back faces, and a strand's inner side
    # must not vanish
    mat['flag'] = mat.get('flag', 0) | 0x01
    new_mats.append(mat); new_faces.append(faces)
    # internal consistency: the wound geometric normal should agree with the
    # stored one; disagreement is the inside-out defect that cost the hair
    g = 0; n_ = 0
    step = max(3, (len(faces)//3//1500)*3)
    for k in range(0, len(faces), step):
        t3 = faces[k:k+3]
        P3 = [np.array(m.vertices[x][0]) for x in t3]
        gn = np.cross(P3[1]-P3[0], P3[2]-P3[0])
        sn = sum(np.array(m.vertices[x][1]) for x in t3)
        if np.linalg.norm(gn) < 1e-12: continue
        n_ += 1; g += (float(gn @ sn) > 0)
    print('  %-9s %6d tris, %5d verts, winding agrees with normals %.0f%%'
          % (label, len(faces)//3, len(used), 100.0*g/max(n_, 1)))
    return used

def body_weights(v):
    acc = {}
    for gb, w in zip(bI[v], bW[v]):
        if w <= 1e-6: continue
        dist = bonedist.get(int(gb))
        if dist is None:
            g2 = min(bonedist, key=lambda g: abs(g - int(gb)))
            dist = bonedist[g2]
        for hb, f in dist:
            acc[hb] = acc.get(hb, 0.0) + float(w)*f
    it = sorted(acc.items(), key=lambda kv: -kv[1])[:4]
    tot = sum(x for _, x in it) or 1.0
    bones = [b for b, _ in it]; ws = [x/tot for _, x in it]
    if len(it) == 1: return 0, bones, [1.0]
    if len(it) == 2: return 1, bones, ws
    bones += [0]*(4-len(it)); ws += [0.0]*(4-len(it))
    return 2, bones, ws

def head_only(v): return 0, [HEAD], [1.0]

# body: in the normal form the vanilla-head proxy is cut at the throat -- the
# host supplies the real head; the bomb form's devil head IS the head and stays
tri = bTri
if FORM == 'normal':
    keep = bV[tri][:, :, 1].mean(axis=1) <= MOD_NECK
    print('  head proxy dropped: %d of %d triangles' % ((~keep).sum(), len(tri)))
    tri = tri[keep]
emit('zzz_body', bV, bN, bUV, tri,
     'EvelynBodyBDiffuse.png' if FORM == 'bomb' else 'EvelynBodyADiffuse.png',
     '衣', retarget, body_weights)

if FORM == 'normal':
    hV, hN, hUV, _, _ = load_part('EvelynHair')
    hTri = np.concatenate([np.frombuffer(read(p + '.ib'), dtype=np.uint32)
                           for p in ('EvelynHairA', 'EvelynHairB')]).reshape(-1, 3).astype(np.int64)
    emit('zzz_hair', hV, hN, hUV, hTri, 'EvelynHairADiffuse.png',
         '发', lambda v: hV[v]*S_h + T_h, head_only)

# --- host materials: replace clothes and hair, trim 肌, keep the face ---------
gone = BODY_GONE | HAIR_GONE | (FACE_MATS if FORM == 'bomb' else set())
face_bot = float(min(np.array(m.vertices[v][0])[1]
                     for nm in FACE_MATS for v in mat_verts(nm) if mat_verts(nm)))
_face_samp = [(m.vertices[v][2], m.vertices[v][0][1]) for v in mat_verts('颜')]

off = 0; rebuilt = []; drop_idx = set(); neck_kept = []
for mi, mm in enumerate(m.materials):
    idx = m.faces[off:off+mm['fcount']]; off += mm['fcount']
    if mm['name'] == '肌':
        t3 = np.array(idx).reshape(-1, 3)
        VP = np.array([[m.vertices[v][0] for v in tt] for tt in t3])
        rr = np.linalg.norm(VP[:, :, ::2] - NECK[::2][None, None, :], axis=2)
        yy = VP[:, :, 1]
        # neck tube through the throat, wide only under the collar, and the
        # whole scalp -- the head was hollow behind the hair without it
        # Narrow through the neck, unrestricted above the jaw. A plane cut
        # sliced the shoulder caps and left jagged skin plates on them; a
        # radius cut all the way up severed the scalp from the neck and the
        # open edges gaped when the head turned. 颜 is the face front only --
        # the skull and scalp are 肌 -- so above the jaw everything stays, and
        # only the band below it is trimmed to a column that clears the
        # shoulders.
        ok = (yy > 18.15) | ((yy > 17.55) & (rr < 0.9))
        if FORM == 'bomb':
            ok = ((yy > 17.55) & (yy < 18.9) & (rr < 0.9))
        keep = ok.all(axis=1)
        kept = t3[keep]
        rebuilt += [int(v) for v in kept.ravel()]
        mm['fcount'] = kept.size
        neck_kept = sorted({int(v) for v in kept.ravel()})
        print('  肌 trimmed to neck + scalp: %d of %d triangles' % (keep.sum(), len(t3)))
    elif mm['name'] in gone:
        drop_idx.add(mi); mm['fcount'] = 0
    else:
        rebuilt += idx
m.faces = rebuilt
ren = {}; k = 0
for mi in range(len(m.materials)):
    if mi in drop_idx: continue
    ren[mi] = k; k += 1
m.materials = [mm for mi, mm in enumerate(m.materials) if mi not in drop_idx]
for mo in m.morphs:
    if mo.get('type') != 8: continue
    mo['offsets'] = [(ren[o[0]],) + tuple(o[1:]) for o in mo['offsets'] if o[0] in ren]
print('  %d host materials removed, %d kept' % (len(drop_idx), len(m.materials)))

for mat, faces in zip(new_mats, new_faces):
    m.materials.append(mat); m.faces = m.faces + faces

# --- textures: the mod's face art, and one skin tone --------------------------
if FORM == 'normal':
    im = Image.open(os.path.join(src_tex, 'EvelynFaceADiffuse.png'))
    im.transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(tex_dir, 'zzz_face.png'))
    rel = 'tex_zzz\\zzz_face.png'
    if rel not in m.textures: m.textures.append(rel)
    ti = m.textures.index(rel)
    n = 0
    for mm in m.materials:
        if mm['name'] in FACE_MATS and 0 <= mm['tex'] < len(m.textures) \
           and m.textures[mm['tex']].endswith('颜.tga'):
            mm['tex'] = ti; n += 1
    print('  face texture -> the mod\'s, v-flipped, on %d materials' % n)

if neck_kept:
    kim = Image.open(os.path.join(host_dir, 'tex', '衣.tga')).convert('RGBA')
    W_, H_ = kim.size
    nsam = [kim.getpixel((int(u*(W_-1)), int(vv*(H_-1))))[:3]
            for v in neck_kept for (u, vv) in [m.vertices[v][2]]
            if 0 <= u <= 1 and 0 <= vv <= 1 and m.vertices[v][0][1] < 19.0]
    fim = Image.open(os.path.join(src_tex, 'EvelynFaceADiffuse.png')).convert('RGB')
    FW_, FH_ = fim.size
    fsam = [fim.getpixel((int(u*(FW_-1)), int((1-vv)*(FH_-1))))
            for (u, vv), y_ in _face_samp if y_ < 19.4 and 0 <= u <= 1 and 0 <= vv <= 1]
    if nsam and fsam:
        gain = np.clip(np.median(fsam, axis=0)/np.maximum(np.median(nsam, axis=0), 1),
                       0.5, 2.0)
        a_ = np.asarray(kim, dtype=np.float32)
        a_[:, :, :3] = np.clip(a_[:, :, :3]*gain, 0, 255)
        Image.fromarray(a_.round().astype(np.uint8)).save(
            os.path.join(tex_dir, 'zzz_skin.png'))
        rel = 'tex_zzz\\zzz_skin.png'
        if rel not in m.textures: m.textures.append(rel)
        for mm in m.materials:
            if mm['name'] == '肌': mm['tex'] = m.textures.index(rel)
        print('  neck tone matched: gain %s' % np.round(gain, 2))

write_pmx(OUT, m)
print('wrote %s  (%d verts, %d tris, %d materials)'
      % (os.path.basename(OUT), len(m.vertices), len(m.faces)//3, len(m.materials)))
