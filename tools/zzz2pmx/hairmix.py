# -*- coding: utf-8 -*-
"""A haircut put together from other models' hair, on a host that keeps its own face.

    python3 tools/zzz2pmx/hairmix.py makima [out-dir]

Every 深空之眼 head is the same base head, so a piece of one model's hair sits on
another's skull once the two faces are fitted (a uniform scale and an offset, the
graft's own fit). The host's hair comes off; each piece arrives as its mesh, the bone
chains it hangs on, their rigid bodies and joints (`chaingraft`), and its own sheet.
The host's face, body and expressions stay as they are -- unlike `headgraft`, which
brings a whole head and its morphs.

A piece is the shells of a donor's hair material picked by the bones they ride: a
braid is its own chain (`hf_*`), bangs and side locks are not.

A host can also keep part of its own hair (`trim`): the shells its recipe passes stay,
with their chains; the rest comes off with the bones only they used.

MAKIMA: 英招2.0, the original skin, for the face; 巡天·英招 听风消夏's bangs and side
locks (every shell off its braid chain); 塞尔凯特's braid with the back of the scalp it
grows from (every shell on it). Both sheets go through one red gradient so the two
pieces match whatever colour each started as. Her irises are rings within rings over
英招's shade (`iris.py`).
"""
import os
import re
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chaingraft
import headgraft
import holefill
import progress
import prune
import publish
import roundring as rr
from pmx import PMX
from pmxw import write_pmx

M = 'public/models/'


def top(m, i, names):
    v = m.vertices[int(i)]
    return names[v[5][int(np.argmax(v[6][:len(v[5])]))]]


def piece(src, mat, asname, keep, out):
    """A copy of the donor whose `mat` holds only the shells `keep` passes, as `asname`;
    its face renamed `face01` for the fit."""
    m = PMX(src)
    names = [b['name'] for b in m.bones]
    V = np.array([v[0] for v in m.vertices])
    F = np.array(m.faces).reshape(-1, 3)
    fm = np.repeat(np.arange(len(m.materials)), [x['fcount'] // 3 for x in m.materials])
    k = next(k for k, x in enumerate(m.materials) if x['name'] == mat)
    T = F[fm == k]
    kept = set()
    for ids in rr.shells(V, T).values():
        if keep([top(m, i, names) for i in ids]):
            kept |= {int(i) for i in ids}
    take = np.array([int(t[0]) in kept for t in T])
    F = np.concatenate([F[fm < k], T[take], F[fm > k]])
    m.materials[k]['fcount'] = int(take.sum()) * 3
    m.materials[k]['name'] = m.materials[k]['name_en'] = asname
    m.faces = F.ravel().tolist()
    for x in m.materials:
        if re.match(r'^(脸|顔|face)', x['name']):
            x['name'] = x['name_en'] = 'face01'
    write_pmx(out, m)
    print(f'  {os.path.basename(src)}: `{mat}` -> `{asname}`, {int(take.sum())} of {len(T)} triangles')
    return out


def trim(path, mat, keep, asname='_cut'):
    """The host's own hair cut down to the shells `keep` passes; the rest moved into
    `asname`, appended last, for `drop` to take off with the bones only it used."""
    m = PMX(path)
    names = [b['name'] for b in m.bones]
    V = np.array([v[0] for v in m.vertices])
    F = np.array(m.faces).reshape(-1, 3)
    fm = np.repeat(np.arange(len(m.materials)), [x['fcount'] // 3 for x in m.materials])
    k = next(k for k, x in enumerate(m.materials) if x['name'] == mat)
    T = F[fm == k]
    cut = set()
    for ids in rr.shells(V, T).values():
        if not keep([top(m, i, names) for i in ids]):
            cut |= {int(i) for i in ids}
    off = np.array([int(t[0]) in cut for t in T])
    fm[np.flatnonzero(fm == k)[off]] = len(m.materials)
    x = dict(m.materials[k])
    x['name'] = x['name_en'] = asname
    m.materials.append(x)
    order = np.argsort(fm, kind='stable')
    for j, y in enumerate(m.materials):
        y['fcount'] = int((fm == j).sum()) * 3
    m.faces = F[order].ravel().tolist()
    write_pmx(path, m)
    print(f'  host: `{mat}` keeps {int((~off).sum())} of {len(T)} triangles')


def drop(path, mats):
    """The host's hair off, with every bone only it used; material morphs re-pointed."""
    m = PMX(path)
    sl = holefill.slices(m)
    usedby = {}
    for k in sl:
        for i in np.unique(sl[k]):
            for b, w in zip(m.vertices[int(i)][5], m.vertices[int(i)][6]):
                if w > 1e-6:
                    usedby.setdefault(int(b), set()).add(k)
    force = [m.bones[b]['name'] for b, ks in usedby.items() if ks and ks <= set(mats)]
    F = np.array(m.faces).reshape(-1, 3)
    fm = np.repeat(np.arange(len(m.materials)), [x['fcount'] // 3 for x in m.materials])
    gone = [k for k, x in enumerate(m.materials) if x['name'] in mats]
    keep = [k for k in range(len(m.materials)) if k not in gone]
    remap = {k: i for i, k in enumerate(keep)}
    m.faces = np.concatenate([F[fm == k] for k in keep]).ravel().tolist()
    m.materials = [m.materials[k] for k in keep]
    for mo in m.morphs:
        if mo['type'] == 8:
            mo['offsets'] = [(remap[o[0]],) + tuple(o[1:]) for o in mo['offsets'] if o[0] in remap or o[0] < 0]
    write_pmx(path, m)
    print(f'  host: {", ".join(mats)} off')
    prune.main(path, force)


def _axis(rot):
    x, y, z = rot
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    R = (np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
         @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))
    return R @ [0, 1, 0]


def gap(r1, r2):
    """Clearance between two bodies, each taken as a capsule (a sphere is one of no
    length, a box the sphere round it); negative is overlap."""
    def seg(r):
        c, h = np.array(r['pos']), (r['size'][1] / 2 if r['shape'] == 2 else 0.0)
        a = _axis(r['rot'])
        rad = r['size'][0] if r['shape'] in (0, 2) else float(np.linalg.norm(r['size']))
        return c - a * h, c + a * h, rad
    p1, q1, a1 = seg(r1)
    p2, q2, a2 = seg(r2)
    d2 = q2 - p2
    best = 1e9
    for t in np.linspace(0, 1, 25):
        a = p1 + (q1 - p1) * t
        s_ = np.clip(np.dot(a - p2, d2) / max(np.dot(d2, d2), 1e-9), 0, 1)
        best = min(best, float(np.linalg.norm(a - (p2 + d2 * s_))))
    return best - a1 - a2


def confine(path, donor, chain):
    """What the carried chains collide with: only the host groups whose bodies hang on
    skeleton bones the donor has too -- the body, the chest -- never the host's own
    accessory chains. 听风消夏's strands collide with every group, which on her meant
    the body and the chest; on 英招2.0 it also meant her earrings, her neck ornament and
    the braid, and the strands snagged on them.

    A host head body the chain starts inside goes into a group of its own that the
    chain leaves out; everything else that collided with it still does. 英招2.0's head
    sphere sits 0.11 into the side locks' roots, and pushed them out from the first
    frame, by nothing anyone could see; 听风消夏's own head bodies clear them."""
    m = PMX(path)
    nm = [b['name'] for b in m.bones]
    dn = {b['name'] for b in PMX(donor).bones}
    mine = [k for k, r in enumerate(m.rigids) if 0 <= r['bone'] < len(nm) and nm[r['bone']] in chain]
    # skeleton bones are named in Japanese; accessory chains in ASCII (hf_, eh, bs, yxz,
    # d_1074ui_Bone), and two donors can share an accessory's name -- 塞尔凯特's
    # earrings are eh* too
    skeleton = lambda n: n in dn and n not in chain and any(ord(ch) > 127 for ch in n)
    shared = {r['group'] for k, r in enumerate(m.rigids) if k not in mine and 0 <= r['bone'] < len(nm)
              and skeleton(nm[r['bone']])}
    inside = [k for k, r in enumerate(m.rigids) if k not in mine and r['group'] in shared
              and nm[r['bone']] == '頭' and min(gap(m.rigids[c], r) for c in mine) < 0]
    if inside:
        free = sorted(set(range(16)) - {r['group'] for r in m.rigids})
        G = free[0]
        for k in inside:
            old = m.rigids[k]['group']
            m.rigids[k]['group'] = G
            for j, r in enumerate(m.rigids):
                if j not in mine and r['nocollide'] >> old & 1:
                    r['nocollide'] |= 1 << G
        print(f'  {", ".join(m.rigids[k]["name"] for k in inside)}: the chain starts inside, '
              f'moved to group {G} and left out of its collisions')
    mask = sum(1 << g for g in shared)
    for k in mine:
        m.rigids[k]['nocollide'] = mask
    write_pmx(path, m)
    print(f'  {len(mine)} chain bodies collide with groups {sorted(shared)} only')


def append(path, donor, mat, fit, sheet):
    """The donor material's mesh onto the host, through the fit, on the host's bones by name."""
    t, d = PMX(path), PMX(donor)
    tb = {b['name']: i for i, b in enumerate(t.bones)}
    dn = [b['name'] for b in d.bones]
    T = np.array(holefill.slices(d)[mat])
    used = np.unique(T)
    P = fit(np.array([d.vertices[i][0] for i in used]))
    remap, lost = {}, set()
    for k, i in enumerate(used):
        v = d.vertices[int(i)]
        bones = []
        for b in v[5]:
            n = dn[b] if 0 <= b < len(dn) else ''
            if n not in tb:
                lost.add(n)
            bones.append(tb.get(n, tb['頭']))
        add = [[0.0] * 4 for _ in range(t.uv_add)]
        rec = (list(P[k]), list(v[1]), list(v[2]), add, v[4], bones, list(v[6]), v[7])
        remap[int(i)] = len(t.vertices)
        t.vertices.append(rec + (v[8],) if v[4] == 3 else rec)
    os.makedirs(os.path.join(os.path.dirname(path), 'textures'), exist_ok=True)
    x = dict(next(x for x in d.materials if x['name'] == mat))
    src = os.path.join(os.path.dirname(donor), d.textures[x['tex']].replace('\\', '/'))
    shutil.copyfile(src, os.path.join(os.path.dirname(path), 'textures', sheet))
    t.textures.append('textures\\' + sheet)
    x['tex'] = len(t.textures) - 1
    # The sphere map and toon index the donor's texture list, not ours: carried as
    # they were, 塞尔凯特's braid added the host's skin sheet as its sphere map.
    for key in ('sph', 'toon'):
        if key == 'toon' and x.get('toon_shared'):
            continue
        k = x.get(key, -1)
        if 0 <= k < len(d.textures):
            base = os.path.basename(d.textures[k].replace('\\', '/'))
            have = [os.path.basename(n.replace('\\', '/')) for n in t.textures]
            if base not in have:
                shutil.copyfile(os.path.join(os.path.dirname(donor), d.textures[k].replace('\\', '/')),
                                os.path.join(os.path.dirname(path), 'textures', base))
                t.textures.append('textures\\' + base)
                have.append(base)
            x[key] = have.index(base)
    x['fcount'] = len(T) * 3
    t.materials.append(x)
    t.faces = list(t.faces) + [remap[int(i)] for i in T.ravel()]
    write_pmx(path, t)
    print(f'  {mat}: {len(used)} vertices, {len(T)} triangles' +
          (f'; {len(lost)} bones fell back to 頭: {sorted(lost)[:4]}' if lost else ''))


def coverage(m, mat, size):
    """The part of a sheet the material's triangles use."""
    from PIL import ImageDraw
    w, h = size
    img = Image.new('L', size, 0)
    d = ImageDraw.Draw(img)
    UV = np.array([v[2] for v in m.vertices])
    for t in holefill.slices(m)[mat]:
        p = UV[t] % 1.0
        if np.ptp(p[:, 0]) > 0.5 or np.ptp(p[:, 1]) > 0.5:
            continue
        d.polygon([(u * (w - 1), v * (h - 1)) for u, v in p], fill=255)
    return np.asarray(img) > 0


def recolor(path, mats, stops, base=0.62, sheen=None):
    """The materials' sheets dyed through one colour ramp, read only where their
    triangles sample.

    Brightness ratios carry over: a pixel's output colour is the point on the ramp
    whose brightness stands to the base colour's (the ramp at `base`) as the pixel's
    stands to its own sheet's median. A stroke 30% darker than the lavender around it
    comes out 30% darker than the red around it, whatever colour each sheet started as.

    Placing pixels along the ramp by stops instead was faithful only where the ramp's
    brightness climbs evenly, and it barely climbs through its red middle: 塞尔凯特's
    braid lost 2.5x of its contrast there and went flat. Stretching each sheet between
    its own percentiles amplified a narrow one without limit instead: 听风消夏's dark
    bangs, spanning 0.19-0.35, had every faint stroke and jagged line edge multiplied
    ninefold, crimson on pink.

    `sheen` brightens one material's painted highlight -- the angel ring across the
    bangs -- and nothing else: a mark standing well above the hair around it (a wide
    median of its own sheet), within what the material samples, blends to the given
    colour, with soft edges and a faint glow."""
    m = PMX(path)
    base_dir = os.path.dirname(path)
    pos = np.array([s_[0] for s_ in stops])
    col = np.array([[int(s_[1][i:i + 2], 16) for i in (1, 3, 5)] for s_ in stops], float)
    # the ramp's own brightness curve, to be read backwards
    ts = np.linspace(0, 1, 2001)
    ramp = np.stack([np.interp(ts, pos, col[:, k]) for k in range(3)], -1)
    lum = ramp @ [0.299, 0.587, 0.114] / 255
    assert np.all(np.diff(lum) > 0), 'the ramp must brighten all the way up'
    lb = float(np.interp(base, ts, lum))
    for mat in mats:
        x = next(x for x in m.materials if x['name'] == mat)
        sheet = os.path.join(base_dir, m.textures[x['tex']].replace('\\', '/'))
        im = Image.open(sheet)
        alpha = im.getchannel('A') if 'A' in im.getbands() else None
        L = (np.asarray(im.convert('RGB')).astype(float) / 255) @ [0.299, 0.587, 0.114]
        used = coverage(m, mat, im.size)
        med = np.percentile(L[used], 50)
        target = np.clip(lb * np.maximum(L, 0.01) / med, lum[0], lum[-1])
        t = np.interp(target, lum, ts)
        out = np.stack([np.interp(t, pos, col[:, k]) for k in range(3)], -1)
        if sheen and mat in sheen:
            from scipy import ndimage
            around = ndimage.median_filter(L, size=31)
            ring = np.clip((L - around - 0.08) / 0.08, 0, 1) * used
            px = im.size[0] / 1024
            ring = np.clip(0.8 * ndimage.gaussian_filter(ring, 1.6 * px)
                           + 0.35 * ndimage.gaussian_filter(ring, 7 * px), 0, 0.85) * used
            c_ = np.array([int(sheen[mat][i:i + 2], 16) for i in (1, 3, 5)], float)
            out = out * (1 - ring[..., None]) + c_ * ring[..., None]
            print(f'  {mat}: {int((ring > 0.5).sum())} pixels of painted highlight to {sheen[mat]}')
        res = Image.fromarray(out.round().astype(np.uint8))
        if alpha is not None:
            res.putalpha(alpha)
        res.save(sheet)
        r_lo, r_hi = np.percentile(L[used] / med, [1, 99])
        print(f'  {mat} ({os.path.basename(sheet)}): brightness {r_lo:.2f}x..{r_hi:.2f}x its median -> '
              f'ramp {np.interp(lb * r_lo, lum, ts):.2f}..{np.interp(min(lb * r_hi, lum[-1]), lum, ts):.2f}')


HF = re.compile(r'^hf_')

# Makima's hair, shadow to highlight: a red kept off orange -- the studio's hair shading
# warms what it is given, and #c8443a at the middle read brown-orange there -- with a
# sheen that stays pink, as a near-white top washed the crown out.
MAKIMA_RED = [(0.0, '#4a1219'), (0.35, '#a02a32'), (0.7, '#cf4a52'), (0.88, '#e86a70'), (1.0, '#f8908f')]

MIXES = {
    'makima': dict(
        host=M + '深空之眼—英招2.0/英招2.0.pmx',
        drop=['头发', '头饰'],
        pieces=[
            # 听风消夏's hair off the braid chain: bangs and side locks (its `hat` is a
            # straw hat, and stays behind)
            dict(src=M + '巡天·英招-听风消夏/英招泳装.pmx', mat='hair', name='hair_front', sheet='hair_front.png',
                 keep=lambda tops: sum(b.startswith('hair_B') for b in tops) < 0.05 * len(tops)),
            dict(src=M + '塞尔凯特2.0 2/塞尔凯特3.0.pmx', mat='头发', name='hair_braid', sheet='hair_braid.png',
                 keep=lambda tops: sum(bool(HF.match(b)) for b in tops) >= 0.05 * len(tops)),
        ],
        recolor=['hair_front', 'hair_braid'],
        ramp=MAKIMA_RED,
        sheen={'hair_front': '#ffd0c8'},
        iris=[('', '眼睛')],
    ),
}


def main(name, out_dir=None):
    import hashlib
    import iris as iris_mod
    s = MIXES[name]
    # The studio caches a texture by its file name: a sheet whose colours change keeps
    # the old ones on screen under the same name, so each name carries its colours' hash
    tag = hashlib.md5(repr((s.get('ramp'), s.get('sheen'), iris_mod.RINGS, iris_mod.GOLD, 'ratio')).encode()).hexdigest()[:6]
    for p in s['pieces']:
        stem, ext = os.path.splitext(p['sheet'])
        p['sheet'] = f'{stem}_{tag}{ext}'

    out_dir = out_dir or M + name
    work = os.path.join(out_dir, '_work')
    build = os.path.join(work, '_base.pmx')
    # the folder is this build's alone: clear what earlier recipes left in it
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(os.path.join(work, 'textures'), exist_ok=True)
    with progress.step('the host, without its hair'):
        h = PMX(s['host'])
        src = os.path.dirname(s['host'])
        for rel in h.textures:
            fp = os.path.join(src, rel.replace('\\', '/'))
            if os.path.exists(fp):
                shutil.copyfile(fp, os.path.join(work, 'textures', os.path.basename(fp)))
        h.textures = ['textures\\' + os.path.basename(t.replace('\\', '/')) for t in h.textures]
        write_pmx(build, h)
        cut = []
        if s.get('trim'):
            trim(build, s['trim']['mat'], s['trim']['keep'])
            cut = ['_cut']
        drop(build, s['drop'] + cut)
    for k, p in enumerate(s['pieces']):
        with progress.step(p['name']):
            # each donor in a folder of its own, its sheets beside it: two donors'
            # hair sheets are both 头发.png
            ddir = os.path.join(work, f'_donor{k}')
            os.makedirs(ddir, exist_ok=True)
            donor = piece(p['src'], p['mat'], p['name'], p['keep'], os.path.join(ddir, 'donor.pmx'))
            dm = PMX(donor)
            for rel in dm.textures:
                fp = os.path.join(os.path.dirname(p['src']), rel.replace('\\', '/'))
                dst = os.path.join(ddir, rel.replace('\\', '/'))
                if os.path.exists(fp):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copyfile(fp, dst)
            hm = PMX(build)
            for x in hm.materials:      # the fit reads the host's face as face01 too
                if re.match(r'^(脸|顔|face)', x['name']):
                    x['name'] = 'face01'
            fit = headgraft.Fit(dm, hm, scale=1.0)
            before = {b['name'] for b in PMX(build).bones}
            chaingraft.main(donor, build, build, [p['name']], frame='髪', place=fit)
            confine(build, donor, {b['name'] for b in PMX(build).bones} - before)
            append(build, donor, p['name'], fit, p['sheet'])
            shutil.rmtree(ddir)
    with progress.step('red'):
        recolor(build, s['recolor'], s['ramp'], sheen=s.get('sheen'))
    # a model per variant, side by side to compare
    variants = s.get('iris') or [('', None)]
    for suffix, mat in variants:
        out = os.path.join(work, name + suffix + '.pmx')
        shutil.copyfile(build, out)
        if mat:
            with progress.step('the eyes'):
                import iris
                # a sheet name per variant: the studio caches textures by name
                iris.main(out, mat, f'iris_{name}{suffix}_{tag}.png')
        with progress.step('out'):
            publish.main(out, out_dir)
    shutil.rmtree(work, ignore_errors=True)
    progress.done()
    for suffix, _ in variants:
        m = PMX(os.path.join(out_dir, name + suffix + '.pmx'))
        print(f'{out_dir}/{name}{suffix}.pmx: {len(m.vertices)} verts, {len(m.faces) // 3} triangles, '
              f'{len(m.bones)} bones, {len(m.rigids)} rigid bodies, {len(m.joints)} joints, {len(m.morphs)} morphs')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'makima', sys.argv[2] if len(sys.argv) > 2 else None)
