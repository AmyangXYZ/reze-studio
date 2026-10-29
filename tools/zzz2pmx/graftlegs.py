# -*- coding: utf-8 -*-
"""Give the model back the legs under its socks.

Reze's body was cut off at the sock line -- her skin material starts at y 8.60 and
there is nothing below it, so hiding the socks hides the legs with them. Her donor,
塞尔凯特2, still has the whole body, and two things make the transplant clean: its
skin atlas is the very sheet Reze already ships as `cloth01_skin.png`, so the UVs
land without a new texture, and every bone driving those legs exists in Reze under
the same name.

The bodies were re-proportioned though -- the knees sit 0.83 apart -- so the legs
are retargeted rather than copied. Each bone contributes the similarity that carries
its own segment of the donor's skeleton onto Reze's (the rotation that aligns the
segment, and the scale that matches its length), and the vertex is placed by its own
skin weights, which is the same blend that will later animate it.

    python3 tools/zzz2pmx/graftlegs.py [--cut 9.0] [--shrink 0.985]
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pmx import PMX
from pmxw import write_pmx

TARGET = 'public/models/reze/reze.pmx'
DONOR = 'public/models/塞尔凯特2/塞尔凯特.pmx'
SKIN = 'skin'
NAME = '足'          # the new material
CHILD = {'右足D': '右ひざD', '右ひざD': '右足首D', '右足首D': '右つま先',
         '左足D': '左ひざD', '左ひざD': '左足首D', '左足首D': '左つま先'}
FALLBACK = {'右つま先': '右足首D', '右足先EX': '右足首D',
            '左つま先': '左足首D', '左足先EX': '左足首D'}


def rot_between(u, v):
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    c = float(np.dot(u, v))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        a = np.array([1.0, 0, 0])
        if abs(u[0]) > 0.9:
            a = np.array([0, 1.0, 0])
        k = np.cross(u, a)
        k /= np.linalg.norm(k)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + 2 * K @ K
    k = np.cross(u, v)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def material_slice(m, name):
    at = 0
    F = np.array(m.faces).reshape(-1, 3)
    for mat in m.materials:
        n = mat['fcount'] // 3
        if mat['name'] == name:
            return F[at:at + n], mat
        at += n
    raise SystemExit(f'no material {name}')


def main(cut=9.0, shrink=1.0, fade=2.0):
    t = PMX(TARGET)
    d = PMX(DONOR)
    tb = {b['name']: np.array(b['pos']) for b in t.bones}
    tidx = {b['name']: i for i, b in enumerate(t.bones)}
    db = {b['name']: np.array(b['pos']) for b in d.bones}

    tris, dmat = material_slice(d, SKIN)
    Vd = np.array([v[0] for v in d.vertices])
    legs = tris[(Vd[tris][:, :, 1] < cut).all(1)]
    keep = np.unique(legs)
    print(f'donor legs: {len(legs)} tris, {len(keep)} vertices, '
          f'y {Vd[keep][:, 1].min():.2f}..{Vd[keep][:, 1].max():.2f}')

    # one similarity per bone, carrying its own segment of the donor skeleton onto ours
    M = {}
    for name in set(CHILD) | set(FALLBACK):
        src = FALLBACK.get(name, name)
        c = CHILD.get(src)
        if c is None or src not in db or c not in db or src not in tb or c not in tb:
            continue
        u, v = db[c] - db[src], tb[c] - tb[src]
        s = np.linalg.norm(v) / max(np.linalg.norm(u), 1e-9)
        R = rot_between(u, v)
        M[name] = (db[src], tb[src], s, R)
    missing = {d.bones[b]['name'] for i in keep for b, w in
               zip(d.vertices[i][5], d.vertices[i][6]) if w > 0} - set(M)
    if missing:
        print(f'   bones with no mapping (skipped): {missing}')
    for name, (a0, b0, s, R) in sorted(M.items()):
        print(f'   {name:8s} scale {s:.3f}  shift {np.linalg.norm(b0 - a0):.3f}')

    Nd = np.array([v[1] for v in d.vertices])
    newpos, newnor = {}, {}
    for i in keep:
        v = d.vertices[int(i)]
        p, n, acc, nacc, tot = Vd[i], Nd[i], np.zeros(3), np.zeros(3), 0.0
        for b, w in zip(v[5], v[6]):
            if w <= 0:
                continue
            name = d.bones[b]['name']
            if name not in M:
                continue
            a0, b0, s, R = M[name]
            acc += w * (b0 + s * (R @ (p - a0)))
            nacc += w * (R @ n)
            tot += w
        if tot < 1e-6:
            newpos[int(i)] = p
            newnor[int(i)] = n
        else:
            newpos[int(i)] = acc / tot
            newnor[int(i)] = nacc / (np.linalg.norm(nacc) + 1e-12)

    # gently shrink about each slice's centre so the legs sit inside the socks
    if shrink != 1.0:
        P = np.array([newpos[int(i)] for i in keep])
        for lo in np.arange(P[:, 1].min(), P[:, 1].max(), 0.5):
            sel = (P[:, 1] >= lo) & (P[:, 1] < lo + 0.5)
            if sel.sum() < 4:
                continue
            c = P[sel].mean(0)
            P[sel] = c + (P[sel] - c) * [shrink, 1.0, shrink]
        for k, i in enumerate(keep):
            newpos[int(i)] = P[k]

    # The donor's thigh is thinner than ours where the two meet, which would show as a
    # step at the seam. Match the radius at the join and let the correction fade out
    # down the leg, so only the top is touched.
    if fade > 0:
        Vt = np.array([v[0] for v in t.vertices])
        tri_skin, _ = material_slice(t, SKIN)
        skin_pts = Vt[np.unique(tri_skin)]
        seam = skin_pts[:, 1].min()
        P = np.array([newpos[int(i)] for i in keep])
        for sign, label in ((-1, 'right'), (1, 'left')):
            side = np.sign(P[:, 0]) == sign
            band = skin_pts[(skin_pts[:, 1] >= seam) & (skin_pts[:, 1] < seam + 0.4)]
            band = band[np.sign(band[:, 0]) == sign]
            top = P[side & (P[:, 1] > seam - 0.3)]
            if len(band) < 4 or len(top) < 4:
                continue
            cb, ct = band[:, [0, 2]].mean(0), top[:, [0, 2]].mean(0)
            rb = np.percentile(np.linalg.norm(band[:, [0, 2]] - cb, axis=1), 95)
            rt = np.percentile(np.linalg.norm(top[:, [0, 2]] - ct, axis=1), 95)
            k = rb / max(rt, 1e-6)
            w = np.clip((P[side][:, 1] - (seam - fade)) / fade, 0, 1)
            f = 1 + w * (k - 1)
            q = P[side]
            q[:, [0, 2]] = ct + (q[:, [0, 2]] - ct) * f[:, None]
            P[side] = q
            print(f'   {label} thigh: donor r {rt:.3f} -> ours {rb:.3f} '
                  f'(x{k:.3f}), faded over {fade}')
        for k2, i in enumerate(keep):
            newpos[int(i)] = P[k2]

    base = len(t.vertices)
    remap = {}
    for i in keep:
        v = d.vertices[int(i)]
        bones = []
        for b in v[5]:
            nm = d.bones[b]['name']
            bones.append(tidx.get(nm, 0))
        add = [np.zeros(4) for _ in range(t.uv_add)]
        rec = (list(newpos[int(i)]), list(newnor[int(i)]), list(v[2]), add,
               v[4], bones, list(v[6]), v[7])
        remap[int(i)] = base + len(remap)
        t.vertices.append(rec + (v[8],) if v[4] == 3 else rec)

    faces = list(t.faces)
    for tri in legs:
        faces += [remap[int(tri[0])], remap[int(tri[1])], remap[int(tri[2])]]
    t.faces = faces

    skin_mat = material_slice(t, SKIN)[1]
    mat = dict(skin_mat)
    mat['name'] = NAME
    mat['name_en'] = 'legs'
    mat['fcount'] = len(legs) * 3
    t.materials.append(mat)

    write_pmx(TARGET, t)
    print(f'added {len(remap)} vertices and {len(legs)} triangles as material "{NAME}", '
          f'sharing {t.textures[mat["tex"]]}')


if __name__ == '__main__':
    cut = float(sys.argv[sys.argv.index('--cut') + 1]) if '--cut' in sys.argv else 9.0
    sh = float(sys.argv[sys.argv.index('--shrink') + 1]) if '--shrink' in sys.argv else 1.0
    fd = float(sys.argv[sys.argv.index('--fade') + 1]) if '--fade' in sys.argv else 2.0
    main(cut, sh, fd)
