# -*- coding: utf-8 -*-
"""Put the mod's meshes on the MMD model of the character they replace.

    python3 tools/head-graft/zzzgraft.py <host.pmx> <out.pmx> <moddir>
            [--form normal|bomb] [--no-face-tex]

The mod is a 3DMigoto replacement for one character, and 伊芙琳.pmx is an MMD
build of that same character, so the two are already in the same space -- one
uniform scale from the height ratio and they land on each other, median
nearest-vertex distance 0.369 on a 21-unit model. Nothing needs fitting.

That matters because the MMD author repacked the game's textures into custom
atlases -- 衣2.tga carries the blonde hair and the coat on one sheet where the
mod ships them separately -- so the mod's textures cannot be dropped onto this
model's UVs. Bringing the geometry, its UVs and its textures across together
sidesteps the repack completely: the mod's meshes keep the coordinates their
own sheets were painted for.

What comes from where:

  body, hair, jacket, shoulder   the mod, with the mod's UVs and sheets
  face                           the host, because the mod ships none -- it
                                 overrides the face texture and leaves the
                                 game's mesh, which is the one this model has
  face texture                   the mod's, flipped in v; the two atlases are
                                 the same layout mirrored, irises top right
                                 against bottom right
  rig, physics, morphs           the host's, untouched

Weights are transferred from the nearest host vertex, since the mod's buffers
index the game's skeleton and carry none of its bone names. The host's own hair
sits where the mod's hair sits, so the new strands pick up the hair bones and
their physics without being told which they are.
"""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import PMX
from pmxw import write_pmx
import numpy as np

SRC, OUT, MODD = sys.argv[1], sys.argv[2], sys.argv[3]
def arg(n, d): return sys.argv[sys.argv.index(n)+1] if n in sys.argv else d
FORM = arg('--form', 'normal')
FACETEX = '--no-face-tex' not in sys.argv
FLIPV   = '--flip-v' in sys.argv
NOJACKET= '--no-jacket' in sys.argv

# host materials that the mod supersedes
BODY_MATS = {'肌', '衣', '外套', '外套+', '发带', '金属', '领带', '领带+', '皮裤',
             '衬衣', '珠宝', '黑丝衣', '穗', '胸衣', '武器'}
HAIR_MATS = {'发', '侧发', '前发'}
# The transform is measured against the body alone. BODY_MATS is the list of
# what the mod supersedes, which includes the coat, the weapon and the loose
# accessories -- the coat flares well past the figure, so the box it defines is
# not the body's. Measuring against it put the median grafted vertex 1.783 from
# the nearest host vertex, with 15662 of 20038 further than 1.0 on a 21-unit
# model, and the skinning tore the mesh apart. The tight set lands at 0.369.
REF_MATS  = {'肌', '衣', '皮裤', '衬衣'}
COAT_MATS = {'外套', '外套+'}
FACE_MATS = {'颜', '痣', '眉', '眉睫影', '白目', '目', '目光', '目光2', '睫',
             '口', '舌', '齿', '目影'}

# part -> (buffer stem, index buffers, diffuse sheet, host material to copy flags from)
BODY_IB = 'EvelynBodyB' if FORM == 'bomb' else 'EvelynBodyA'
BODY_TEX = 'EvelynBodyBDiffuse.png' if FORM == 'bomb' else 'EvelynBodyADiffuse.png'
# EvelynShoulder is left out: its texcoord buffer reads non-finite as float16 at
# every offset of its 28-byte stride, so its UVs are in some other layout. It is
# a 255-triangle shoulder pad, and shipping it with guessed coordinates would
# put a smear of the wrong sheet on the model.
# the last field is the host materials a part is placed against, or None to use
# the body transform
PARTS = [
    ('body',     'EvelynBody',     [BODY_IB],                      BODY_TEX,                    '衣',   None),
    # Both halves sample A's sheet. B has its own, EvelynHairBDiffuse, but that
    # one is the warm tan of the character this mod replaces, not a second Reze
    # colour -- pointing B at it put blonde streaks through the cut.
    ('hair',     'EvelynHair',     ['EvelynHairA', 'EvelynHairB'], 'EvelynHairADiffuse.png',    '发',   'HAIR'),
    ('jacket',   'EvelynJacket',   ['EvelynJacketA'],              'EvelynJacketADiffuse.png',  '外套', 'COAT'),
]

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

HOSTP = np.array([v[0] for v in m.vertices])

# --- one transform for every part, measured on the body ------------------------
def load(stem, ibs):
    pos = np.frombuffer(read(stem + 'Position.buf'), dtype=np.uint8).reshape(-1, 40)
    xyz = pos[:, 0:12].copy().view(np.float32).reshape(-1, 3)
    nrm = pos[:, 12:24].copy().view(np.float32).reshape(-1, 3)
    # The texcoord stride is not the same for every part -- body and hair pack
    # 20 bytes, the jacket 24 -- so it is read from the buffer rather than
    # assumed. The diffuse UV is a float16 pair at offset 4 in all of them,
    # behind a four-byte vertex colour.
    tb = read(stem + 'Texcoord.buf')
    stride = len(tb)//len(xyz)
    T = np.frombuffer(tb, dtype=np.uint8).reshape(-1, stride)
    uv = T[:, 4:8].copy().view(np.float16).reshape(-1, 2).astype(np.float32)
    if not np.isfinite(uv).all():
        raise SystemExit('%s: no float16 UV pair at offset 4 of a %d-byte stride'
                         % (stem, stride))
    tri = np.concatenate([np.frombuffer(read(p + '.ib'), dtype=np.uint32) for p in ibs])
    if stem == 'EvelynBody' and FORM == 'normal':
        # BodyA ends in a head stub -- the game draws the face as a separate
        # mesh over it -- and this model has its own head in 颜. The stub is cut
        # exactly where that head begins, not at the neck: cutting at the neck
        # left the body ending at y 18.15 under a head starting at 18.64 and
        # there was simply no neck between them. Carried up to the head's own
        # underside the join is closed, and the last of the stub is inside the
        # head mesh rather than poking through it.
        t3 = tri.reshape(-1, 3)
        keep = xyz[t3][:, :, 2].mean(axis=1) <= HEAD_CUT
        print('    head stub cut at mod z %.2f: %d of %d triangles dropped'
              % (HEAD_CUT, (~keep).sum(), len(t3)))
        tri = t3[keep].ravel()
    # mod space (x, depth, up) -> mmd (x, up, depth)
    V = np.stack([xyz[:, 0], xyz[:, 2], xyz[:, 1]], axis=1)
    N = np.stack([nrm[:, 0], nrm[:, 2], nrm[:, 1]], axis=1)
    return V, N, uv, tri.reshape(-1, 3), xyz

# Always BodyA, whichever form is being built. BodyB's devil head stands taller
# than the head it replaces, so matching its height to the host's shrinks the
# body under it -- the bomb form came out at 12.05 against the normal form's
# 13.12 and the two would not have been the same character.
HEAD_CUT = 1e9              # until the transform is known
bV, _, _, bTri, _ = load('EvelynBody', ['EvelynBodyA'])
bU = np.unique(bTri)
hostbody = np.array([m.vertices[v][0] for v in
                     sorted({v for n in REF_MATS for v in mat_verts(n)})])
# Feet to NECK on both, not feet to top. This model's 肌 carries the scalp, so
# its top is the crown, while BodyA's top is the neck stub -- matching those
# stretched the body about four percent tall, pushed the shoulders up past the
# head that stayed put, and the head read as sunk into the body. The neck finds
# itself the same way it does in bombhead: walking up from the widest slab of
# the shoulders, the width falls to the throat and then rises into the head.
prof = []
z0_, z1_ = float(bV[bU][:, 1].min()), float(bV[bU][:, 1].max())
for zz in np.arange(z0_ + 0.72*(z1_-z0_), z1_, 0.02*(z1_-z0_)/1.6):
    sl = bV[bU][(bV[bU][:, 1] >= zz) & (bV[bU][:, 1] < zz + 0.02*(z1_-z0_)/1.6)]
    if len(sl) > 10: prof.append((zz, float(sl[:, 0].max() - sl[:, 0].min())))
i_ = max(range(len(prof)), key=lambda k: prof[k][1])
while i_ + 1 < len(prof) and prof[i_+1][1] <= prof[i_][1]: i_ += 1
mod_neck = prof[i_][0]
host_neck = next(b['pos'][1] for b in m.bones if b['name'] == '首')
S = (host_neck - hostbody[:, 1].min())/(mod_neck - bV[bU][:, 1].min())
print('  neck landmark: mod z %.2f -> host 首 y %.2f' % (mod_neck, host_neck))
# Two transforms, because one cannot do both jobs. The neck landmark is what
# puts the head on the shoulders, but it shortens the body about four percent
# against the host, which slides the hands inboard of the host's hands -- and
# weights are looked up by proximity, so the hand then grabbed whatever was
# nearest instead of its own counterpart, landing 44% of the arm on thumb
# bones. So the mesh is POSITIONED by the neck and its weights are SAMPLED at
# the scale that best overlays the host, where the counterpart of a knuckle is
# that knuckle.
S_W = (hostbody[:, 1].max() - hostbody[:, 1].min())/(bV[bU][:, 1].max() - bV[bU][:, 1].min())
def place_at(V, sc):
    V = V*sc
    V[:, 0] += (hostbody[:, 0].min()+hostbody[:, 0].max())/2 - (bV[bU][:, 0].min()+bV[bU][:, 0].max())/2*sc
    V[:, 1] += hostbody[:, 1].min() - bV[bU][:, 1].min()*sc
    V[:, 2] += (hostbody[:, 2].min()+hostbody[:, 2].max())/2 - (bV[bU][:, 2].min()+bV[bU][:, 2].max())/2*sc
    return V
def place(V): return place_at(V.copy(), S)
def place_w(V): return place_at(V.copy(), S_W)
Y_OFF = hostbody[:, 1].min() - bV[bU][:, 1].min()*S
hostface = np.array([m.vertices[v][0] for v in
                     sorted({v for n in FACE_MATS for v in mat_verts(n)})])
# The body stops at the throat, not at the head's underside. Carrying it up to
# the head made a second neck inside the one 肌 already provides, and the two
# cut edges tore through each other at the collar. 肌's column is the neck that
# belongs to this head; the mod body ends below it and the overlap hides under
# the collar.
HEAD_CUT = mod_neck
print('form %s: body scale %.2f, head begins at mod z %.2f' % (FORM, S, HEAD_CUT))

# The hair does not ride the body transform. The mod's BodyA stops at the neck
# -- its head is the vanilla mesh the mod does not ship -- while this model's
# 肌 carries the scalp all the way up, so a transform measured between those two
# tops is measuring different landmarks. It floated the hair 1.37 above the
# skull and left the crown bare. The hair is placed against the hair it
# replaces instead: one uniform scale from the width, crowns level.
hosthair = np.array([m.vertices[v][0] for v in
                     sorted({v for n in HAIR_MATS for v in mat_verts(n)})])

# One transform for the whole head of hair, measured once on both halves
# together. Fitting each half to the host's hair box on its own would scale the
# fringe and the crown by different amounts and pull the cut apart.
_hV, _, _, _hT, _ = load('EvelynHair', ['EvelynHairA', 'EvelynHairB'])
_hU = np.unique(_hT)
_lo, _hi = _hV[_hU].min(axis=0), _hV[_hU].max(axis=0)
_hlo, _hhi = hosthair.min(axis=0), hosthair.max(axis=0)
HAIR_S = (_hhi[0]-_hlo[0])/max(_hi[0]-_lo[0], 1e-6)
def place_hair(V, used):
    # The body's SCALE, but placed by the head. The mod authored hair and body
    # in one coordinate system, so scaling the hair to this model's hair box
    # instead made it 13.97 against the body's 12.11 -- fifteen percent
    # oversized, which is why it hung low. Taking the body's transform whole
    # does not work either: its z offset centres the body's bounding box, arms
    # and legs included, which threw the hair 1.8 forward of the face. So the
    # scale comes from the body and the position from the hair's own centroid
    # landing on the host's.
    # Crowns level, width matched to the hair it replaces. Placing by centroid
    # at the body's scale instead dropped the cap and bared the scalp: this
    # model's head is proportionally larger than the game's, so hair sized to
    # its body does not cover its head.
    # The body's scale, crowns aligned. The two are the same character: the mod
    # wears hair 11.5% of body height wide and this model wears it 12.1% wide,
    # so the mod's own scale is the right one and fitting to this model's hair
    # box made it 13.97 against the body's 12.11. At body scale the hair is
    # 2.24 wide over a 1.93 head, which covers it -- the bare scalp earlier was
    # the centroid placement dropping the cap, not the size.
    V = (V - (_lo+_hi)/2.0)*S
    V[:, 0] += (_hlo[0]+_hhi[0])/2.0
    V[:, 1] += _hhi[1] - (_hi[1]-_lo[1])*S/2.0
    V[:, 2] += (_hlo[2]+_hhi[2])/2.0
    return V
print('  hair scale %.2f on the host hair box, crown y %.2f' % (HAIR_S, _hhi[1]))

# --- build the new materials ---------------------------------------------------
tex_dir = os.path.join(host_dir, 'tex_zzz')
os.makedirs(tex_dir, exist_ok=True)
src_tex = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'public', 'models', 'reze', 'textures_zzz')

HAIR_REACH = float(sys.argv[sys.argv.index('--hair-reach')+1]) \
    if '--hair-reach' in sys.argv else 1.2
BP_ = np.array([b['pos'] for b in m.bones])
HAIRB_ = [i for i, b in enumerate(m.bones)
          if any(k in b['name'] for k in ('侧发', '刘海', '发结', '发髻', '后发'))]
HEADB_ = next(i for i, b in enumerate(m.bones) if b['name'] == '頭')

newmats = []; newfaces = []
for label, stem, ibs, sheet, protomat, ref in PARTS:
    if label == 'jacket' and NOJACKET:
        print('  jacket skipped')
        continue
    if label == 'hair' and FORM == 'bomb':
        print('  hair skipped: the devil head carries its own silhouette')
        continue
    V, N, uv, tri, RAW = load(stem, ibs)
    used = sorted({int(v) for v in np.unique(tri)})
    V = place_hair(V, used) if ref == 'HAIR' else place(V)
    # positions from V, weights sampled at the overlay scale
    W = V if ref == 'HAIR' else place_w(load(stem, ibs)[0])
    P = W[used]
    # Weights come from the matching part of the host, never from whatever
    # happens to be nearest. Searching the whole model put 2018 body vertices on
    # a little-finger tip and 1669 on the hair bun, because the weapon and the
    # loose coat sit closer to parts of the body than the body's own mesh does.
    pool = {'HAIR': HAIR_MATS, 'COAT': COAT_MATS}.get(ref, REF_MATS)
    src_v = sorted({v for n in pool for v in mat_verts(n)})
    SP_ = np.array([m.vertices[v][0] for v in src_v])
    # The four nearest, blended by inverse distance, not simply the nearest. The
    # host's 肌 is 1733 vertices for the whole skin where the mod's hands alone
    # carry a thousand, so single-nearest collapsed a whole hand onto one
    # coarse vertex -- 1041 of them landed on 小指２ and the hand skinned into a
    # smear. Blending spreads a vertex over the neighbours it sits between.
    K = 4
    knn = np.empty((len(used), K), dtype=np.int64)
    kd = np.empty((len(used), K))
    for i in range(0, len(used), 256):
        d = ((P[i:i+256, None, :] - SP_[None, :, :])**2).sum(axis=2)
        idx = np.argpartition(d, K-1, axis=1)[:, :K]
        dk = np.take_along_axis(d, idx, axis=1)
        order = np.argsort(dk, axis=1)        # argpartition does not sort
        knn[i:i+256] = np.take_along_axis(idx, order, axis=1)
        kd[i:i+256] = np.sqrt(np.take_along_axis(dk, order, axis=1))
    dist = kd[:, 0]
    print('    weights from %d %s vertices; reach median %.2f, p99 %.2f'
          % (len(src_v), ref or 'body', np.median(dist), np.percentile(dist, 99)))
    remap = {}
    for i, v in enumerate(used):
        acc = {}
        w_k = 1.0/np.maximum(kd[i], 1e-4)
        w_k = w_k/w_k.sum()
        for k in range(K):
            src = m.vertices[src_v[knn[i, k]]]
            for bi, bw in zip(src[5], src[6]):
                if bw > 1e-6: acc[bi] = acc.get(bi, 0.0) + bw*float(w_k[k])
        it = sorted(acc.items(), key=lambda kv: -kv[1])[:4]
        tot = sum(x for _, x in it) or 1.0
        bones = [bi for bi, _ in it] + [0]*(4-len(it))
        ws = [x/tot for _, x in it] + [0.0]*(4-len(it))
        wt = 0 if len(it) <= 1 else (1 if len(it) == 2 else 2)
        if wt == 0: bones, ws = [bones[0]], [1.0]
        elif wt == 1: bones, ws = bones[:2], ws[:2]
        # A bone may only move hair that is near it. The host's hair rig is
        # asymmetric -- 右侧发 packs four links into the span 左侧发 covers with
        # seven -- so on the right the bob fell onto 右侧发4, the deepest link
        # and the one that swings furthest, and physics dragged that side down
        # out of shape. 543 right-side weights sat on a bone more than 1.5
        # away. Weights reaching further than REACH are dropped and the rest
        # renormalised; a vertex left with nothing takes the nearest hair bone,
        # or rides 頭 if even that is out of range.
        if ref == 'HAIR':
            here = V[v]
            near = [(bi, bw) for bi, bw in zip(bones, ws)
                    if bw > 1e-6 and (m.bones[bi]['name'] == '頭'
                                      or np.linalg.norm(BP_[bi] - here) <= HAIR_REACH)]
            if not near:
                # No slack. Falling back to the nearest hair bone regardless of
                # distance is what put the bottom of the bob on a chain tip:
                # this cut hangs to y 17.54 and the host's lowest hair bone is
                # at 18.66, so below that line there is no bone at the right
                # height and the nearest one is always something that swings.
                # Hair with no bone of its own rides the head and holds still.
                near = [(HEADB_, 1.0)]
            tot = sum(x for _, x in near) or 1.0
            it = sorted(near, key=lambda kv: -kv[1])[:4]
            bones = [bi for bi, _ in it] + [0]*(4-len(it))
            ws = [x/tot for _, x in it] + [0.0]*(4-len(it))
            wt = 0 if len(it) <= 1 else (1 if len(it) == 2 else 2)
            if wt == 0: bones, ws = [bones[0]], [1.0]
            elif wt == 1: bones, ws = bones[:2], ws[:2]

        # The bomb form's fuse rack hangs down the front between the legs, so
        # the host vertices nearest to it are thighs -- left for one segment and
        # right for the next, and the first stride pulls the cord in half. It is
        # one piece off the hips. Anything forward of the torso folds its leg
        # weights into 下半身.
        if FORM == 'bomb' and stem == 'EvelynBody' and RAW[v][1] < -0.085:
            hips = next((k for k, b in enumerate(m.bones) if b['name'] == '下半身'), None)
            if hips is not None:
                acc2 = {}
                for bi, bw in zip(bones, ws):
                    if bw <= 1e-6: continue
                    if any(t in m.bones[bi]['name'] for t in ('足', 'ひざ', 'つま', 'かかと')):
                        bi = hips
                    acc2[bi] = acc2.get(bi, 0.0) + bw
                it = sorted(acc2.items(), key=lambda kv: -kv[1])[:4]
                tot = sum(x for _, x in it) or 1.0
                bones = [bi for bi, _ in it] + [0]*(4-len(it))
                ws = [x/tot for _, x in it] + [0.0]*(4-len(it))
                wt = 0 if len(it) <= 1 else (1 if len(it) == 2 else 2)
                if wt == 0: bones, ws = [bones[0]], [1.0]
                elif wt == 1: bones, ws = bones[:2], ws[:2]
        n = N[v]; L = float(np.linalg.norm(n)) or 1.0
        remap[v] = len(m.vertices)
        tv = 1.0 - float(uv[v][1]) if FLIPV else float(uv[v][1])
        m.vertices.append((list(V[v]), list(n/L), [float(uv[v][0]), tv],
                           [], wt, bones, ws, 1.0))
    # two axes swapped is a determinant -1 map: the winding reverses, normals do not
    faces = [remap[int(v)] for v in tri[:, ::-1].ravel()]

    p = os.path.join(src_tex, sheet)
    if os.path.exists(p): shutil.copyfile(p, os.path.join(tex_dir, sheet))
    rel = 'tex_zzz\\' + sheet
    if rel not in m.textures: m.textures.append(rel)
    proto = next((mm for mm in m.materials if mm['name'] == protomat), m.materials[0])
    mat = dict(proto)
    mat['name'] = 'zzz_' + label; mat['name_en'] = 'zzz ' + label
    mat['tex'] = m.textures.index(rel)
    mat['fcount'] = len(faces)
    newmats.append(mat); newfaces.append(faces)
    print('  %-9s %6d tris, %5d verts, sheet %s' % (label, len(faces)//3, len(used), sheet))

# --- drop what the mod supersedes ---------------------------------------------
# the bomb form's BodyB carries the devil head, so the face goes with the body
gone = BODY_MATS | HAIR_MATS | (FACE_MATS if FORM == 'bomb' else set())
# 肌 is not dropped whole. The mod's BodyA narrows to 1.13 units at the collar
# -- that is an opening, not a neck, because in the game the head mesh carries
# the neck and the mod ships no head. Deleting 肌 therefore deleted the only
# neck in the model and left the head sitting over a hole. Its neck column is
# kept and the rest of it goes.
# low enough that the trim's own cut edge sits under the collar rather than
# showing as a ring of shards at the shoulder
neck_bone = next(b['pos'] for b in m.bones if b['name'] == '首')
_face_samp = [(m.vertices[v][2], m.vertices[v][0][1]) for v in mat_verts('颜')]
neck_lo = neck_bone[1] - 1.0
neck_hi = float(np.array([m.vertices[v][0] for v in
                          sorted({v for n in FACE_MATS for v in mat_verts(n)})])[:, 1].min()) + 0.21
off = 0; drop = []; newfaces_pre = {}
for mi, mm in enumerate(m.materials):
    idx = m.faces[off:off+mm['fcount']]
    if mm['name'] == '肌' and mm['name'] in gone:
        t3 = np.array(idx).reshape(-1, 3)
        VP = np.array([[m.vertices[v][0] for v in tt] for tt in t3])
        # A neck is a cylinder, not a box. The box kept every triangle whose
        # centre fell inside |x|<1.45 above the chest line -- which is the
        # upper chest and the shoulder caps too, and 肌's triangles are large,
        # so their stray corners hung over the shirt as tan plates and jagged
        # shards. Now every vertex of a kept triangle must sit within NECK_R
        # of the throat axis, and the tube stops just above the jaw so it does
        # not double-wall the face.
        # Measured on 肌 itself: the throat runs r 0.29..0.6 between y 18.2
        # and 18.8, flares into the shoulders below and the head sides above,
        # and 颜's jaw shell spans r 0.53..1.02 -- so a tube under 0.65 slots
        # INSIDE the jaw and its top edge never shows. A single 0.95 radius
        # kept slabs of the jaw underside, and their cut edge ringed the face
        # as a white ledge. The limit now depends on height, per vertex: slim
        # through the throat, wider only below the collar line where the shirt
        # covers it.
        axz = np.array([neck_bone[0], neck_bone[2]])
        rr = np.linalg.norm(VP[:, :, ::2] - axz[None, None, :], axis=2)
        yy = VP[:, :, 1]
        # 0.55, not 0.65: the jaw shell narrows to r 0.53 at the chin, so a
        # 0.65 tube pokes through it and its cut edge ringed the face. At 0.55
        # the tube stays inside the jaw everywhere but the chin point, and the
        # top stops at 18.85 where the throat is still parallel-walled.
        lim = np.where(yy > 18.15, 0.55, 1.35)
        keep = (rr < lim).all(axis=1) & (yy.min(axis=1) > neck_lo) \
               & (yy.max(axis=1) < neck_hi)
        newfaces_pre[mi] = [int(v) for v in t3[keep].ravel()]
        _neck_kept = sorted(set(newfaces_pre[mi]))
        print('  neck kept from 肌: %d of %d triangles (y %.2f..%.2f)'
              % (keep.sum(), len(t3), neck_lo, neck_hi))
    elif mm['name'] in gone:
        drop.append((mi, off, off+mm['fcount']))
    off += mm['fcount']
# rebuild the face list so the trimmed 肌 keeps its slot
off = 0; rebuilt = []
for mi, mm in enumerate(m.materials):
    idx = m.faces[off:off+mm['fcount']]; off += mm['fcount']
    if mi in newfaces_pre:
        rebuilt += newfaces_pre[mi]; mm['fcount'] = len(newfaces_pre[mi])
    elif any(mi == d[0] for d in drop):
        mm['fcount'] = 0
    else:
        rebuilt += idx
m.faces = rebuilt
drop = [(mi, 0, 0) for mi, _, _ in drop]
dropped = {mi for mi, _, _ in drop}
ren = {}; k = 0
for mi in range(len(m.materials)):
    if mi in dropped: continue
    ren[mi] = k; k += 1
m.materials = [mm for mi, mm in enumerate(m.materials) if mi not in dropped]
for mo in m.morphs:
    if mo.get('type') != 8: continue
    mo['offsets'] = [(ren[o[0]],) + tuple(o[1:]) for o in mo['offsets'] if o[0] in ren]
print('  %d host materials replaced, %d kept' % (len(drop), len(m.materials)))

for mat, faces in zip(newmats, newfaces):
    m.materials.append(mat); m.faces = m.faces + faces

# --- the face keeps its mesh and takes the mod's paint -------------------------
if FACETEX and FORM == 'normal':
    from PIL import Image
    fp = os.path.join(src_tex, 'EvelynFaceADiffuse.png')
    if os.path.exists(fp):
        im = Image.open(fp).transpose(Image.FLIP_TOP_BOTTOM)
        out = os.path.join(tex_dir, 'zzz_face.png'); im.save(out)
        rel = 'tex_zzz\\zzz_face.png'
        if rel not in m.textures: m.textures.append(rel)
        ti = m.textures.index(rel)
        n = 0
        for mm in m.materials:
            if mm['name'] in FACE_MATS and 0 <= mm['tex'] < len(m.textures) \
               and m.textures[mm['tex']].endswith('颜.tga'):
                mm['tex'] = ti; n += 1
        print('  face texture -> the mod\'s, flipped in v, on %d materials' % n)

# --- one skin tone -------------------------------------------------------------
# The kept neck sampled Evelyn's skin on 衣.tga, median 211,160,146, against the
# mod face's 233,212,198 -- a warm ring under a pale jaw, and it read as a seam
# even where the geometry was sound. 衣.tga is shared with the clothes, so the
# gain goes onto a copy and only 肌 is repointed at it.
try:
    _neck_kept
except NameError:
    _neck_kept = []
if _neck_kept:
    from PIL import Image
    kim = Image.open(os.path.join(host_dir, 'tex', '衣.tga')).convert('RGBA')
    W_, H_ = kim.size
    nsam = []
    for v in _neck_kept:
        u, vv = m.vertices[v][2]
        if 0 <= u <= 1 and 0 <= vv <= 1:
            nsam.append(kim.getpixel((int(u*(W_-1)), int(vv*(H_-1))))[:3])
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
        rel_ = 'tex_zzz\\zzz_skin.png'
        if rel_ not in m.textures: m.textures.append(rel_)
        for mm in m.materials:
            if mm['name'] == '肌': mm['tex'] = m.textures.index(rel_)
        print('  neck tone matched to the face: %s -> %s, gain %s'
              % (np.median(nsam, axis=0).astype(int), np.median(fsam, axis=0).astype(int),
                 gain.round(2)))

write_pmx(OUT, m)
print('wrote %s  (%d verts, %d tris, %d materials, %d bones, %d rigids, %d joints)'
      % (os.path.basename(OUT), len(m.vertices), len(m.faces)//3, len(m.materials),
         len(m.bones), len(m.rigids), len(m.joints)))
