"""Build a daisy as a PMX prop.

    python3 tools/props/daisy.py                 # public/models/daisy/daisy.pmx
    python3 tools/props/daisy.py --preview out.png

White petals in two offset rings, a domed yellow disk, a green calyx with sepals,
and a gently bowed stem on a four-bone chain (茎1 → 茎2 → 茎3 → 花) so it can be bent
in a pose. Sized for a character ~20 units tall: the head is ~9 cm across, the
stem ~12 cm. Origin is the stem's cut end; attach 全ての親 to a hand bone with an
outside parent to hold it.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'zzz2pmx'))
from pmxw import write_pmx  # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
OUT_DIR = os.path.join(ROOT, 'public', 'models', 'daisy')

# MMD units: a character is ~20 tall, so 1 unit ≈ 8 cm.
STEM_LEN = 1.5
STEM_R = 0.045
DISK_R = 0.185
DISK_H = 0.075
N_PETALS = 21
HEAD_TILT = np.radians(22)     # head nods toward -Z, the side a model faces

MAT_STEM, MAT_SEPAL, MAT_PETAL, MAT_DISK = range(4)
BONE_ROOT, BONE_STEM1, BONE_STEM2, BONE_STEM3, BONE_HEAD = range(5)

rng = np.random.default_rng(7)


# ---------------------------------------------------------------- geometry helpers
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def frame_from_axis(axis):
    """Rotation taking +Y onto `axis`."""
    axis = axis / np.linalg.norm(axis)
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(y, axis)
    s = np.linalg.norm(v)
    c = float(y @ axis)
    if s < 1e-9:
        return np.eye(3) if c > 0 else rot_x(np.pi)
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


HEAD_AXIS = np.array([np.sin(np.radians(6)), np.cos(HEAD_TILT), -np.sin(HEAD_TILT)])
HEAD_AXIS /= np.linalg.norm(HEAD_AXIS)
HEAD_ROT = frame_from_axis(HEAD_AXIS)


def stem_curve(t):
    """Cubic Hermite from the cut end straight up to the head, arriving along the head axis."""
    t = np.asarray(t, float)[..., None]
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.0, STEM_LEN, 0.0]) - HEAD_AXIS * 0.0
    m0 = np.array([0.0, 1.0, 0.0]) * STEM_LEN * 1.0
    m1 = HEAD_AXIS * STEM_LEN * 0.9
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


HEAD_POS = stem_curve(1.0)


# ---------------------------------------------------------------- mesh accumulator
class Mesh:
    def __init__(self):
        self.pos, self.nor, self.uv, self.skin = [], [], [], []
        self.tris = {m: [] for m in range(4)}

    def add_patch(self, P, UV, hint, material, skin, wrap=False):
        """P: (nu, nv, 3) grid. `hint`: (nu, nv, 3) rough outward normals used to fix
        winding; the stored normal is the area-weighted average of adjacent faces,
        which is what a smooth-shaded sheet or tube wants. `skin`: callable taking
        positions (n,3) -> list of (wt, bones, weights)."""
        nu, nv, _ = P.shape
        base = len(self.pos)
        flat = P.reshape(-1, 3)
        uvf = UV.reshape(-1, 2)
        idx = np.arange(nu * nv).reshape(nu, nv)
        tris = []
        cols = nv if wrap else nv - 1
        for i in range(nu - 1):
            for j in range(cols):
                j1 = (j + 1) % nv
                a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j1], idx[i, j1]
                tris.append((a, b, c))
                tris.append((a, c, d))
        tris = np.array(tris)
        fn = np.cross(flat[tris[:, 1]] - flat[tris[:, 0]], flat[tris[:, 2]] - flat[tris[:, 0]])
        h = hint.reshape(-1, 3)
        if (fn * h[tris].mean(1)).sum() < 0:
            tris = tris[:, [0, 2, 1]]
            fn = -fn
        N = np.zeros_like(flat)
        for k in range(3):
            np.add.at(N, tris[:, k], fn)
        nrm = np.linalg.norm(N, axis=1, keepdims=True)
        N = np.where(nrm > 1e-12, N / np.maximum(nrm, 1e-12), h / np.linalg.norm(h, axis=1, keepdims=True))
        self.pos.extend(flat.tolist())
        self.nor.extend(N.tolist())
        self.uv.extend(uvf.tolist())
        self.skin.extend(skin(flat))
        self.tris[material].extend((tris + base).tolist())

    def add_fan(self, center, ring, uv_center, uv_ring, hint, material, skin):
        """A closed fan: `center` (3,), `ring` (n,3)."""
        base = len(self.pos)
        n = len(ring)
        pts = np.vstack([center[None], ring])
        tris = np.array([(0, 1 + i, 1 + (i + 1) % n) for i in range(n)])
        fn = np.cross(pts[tris[:, 1]] - pts[tris[:, 0]], pts[tris[:, 2]] - pts[tris[:, 0]])
        if (fn @ hint).sum() < 0:
            tris = tris[:, [0, 2, 1]]
            fn = -fn
        N = np.zeros_like(pts)
        for k in range(3):
            np.add.at(N, tris[:, k], fn)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        self.pos.extend(pts.tolist())
        self.nor.extend(N.tolist())
        self.uv.extend(np.vstack([uv_center[None], uv_ring]).tolist())
        self.skin.extend(skin(pts))
        self.tris[material].extend((tris + base).tolist())


def skin_head(P):
    return [(0, [BONE_HEAD], [1.0])] * len(P)


def skin_stem(P):
    """Blend between the stem bones by height along the stem."""
    out = []
    knots = [0.0, 1 / 3, 2 / 3, 1.0]
    bones = [BONE_STEM1, BONE_STEM2, BONE_STEM3, BONE_HEAD]
    for p in P:
        t = float(np.clip(p[1] / STEM_LEN, 0, 1))
        k = min(int(t * 3), 2)
        f = (t - knots[k]) / (knots[k + 1] - knots[k])
        out.append((1, [bones[k], bones[k + 1]], [1.0 - f, f]))
    return out


# ---------------------------------------------------------------- parts
def petal_sheet(length, width, tilt, twist, cup, curl, base_lift, nu=12, nv=5):
    """Petal in its own frame: +X outward along the petal, +Y up, Z across.

    Narrow at the base, widest past the middle, rounded off at the tip; the tip
    keeps a sliver of width so the last row is not a pile of zero-area triangles."""
    u = 1 - (1 - np.linspace(0, 1, nu)) ** 1.25      # denser toward the tip
    v = np.linspace(-1, 1, nv)
    body = 0.45 + 0.55 * np.sin(np.pi * u**0.85) ** 0.5
    tip = np.where(u > 0.84, np.sqrt(np.clip(1 - ((u - 0.84) / 0.16) ** 2, 0, 1)), 1.0)
    w = width * np.maximum(body * tip, 0.03)
    U, V = np.meshgrid(u, v, indexing='ij')
    W = w[:, None]
    X = U * length
    Z = V * W
    Y = base_lift + curl * length * U**2 + cup * W * np.abs(V) ** 1.5
    P = np.stack([X, Y, Z], -1)
    # twist about the petal's own axis, growing toward the tip
    for i in range(nu):
        P[i] = P[i] @ rot_x(twist * u[i]).T
    P = P @ rot_z(tilt).T
    hint = np.zeros_like(P)
    hint[..., 1] = 1.0
    UV = np.stack([0.52 + 0.46 * U, 0.5 + 0.4 * V], -1)
    return P, hint, UV


def place_head(P):
    """Petal/disk frame → world: rotate onto the head axis and move to the head."""
    return P @ HEAD_ROT.T + HEAD_POS


def build_petals(mesh):
    rings = [
        # count, azimuth offset, base r, length, width, tilt, curl, lift
        dict(n=N_PETALS, off=0.0, r0=0.12, length=0.48, width=0.072, tilt=-4, curl=0.10, lift=0.005),
        dict(n=N_PETALS, off=0.5, r0=0.13, length=0.43, width=0.062, tilt=13, curl=0.18, lift=0.02),
    ]
    for ring in rings:
        for k in range(ring['n']):
            az = 2 * np.pi * (k + ring['off']) / ring['n'] + rng.normal(0, 0.015)
            length = ring['length'] * rng.uniform(0.93, 1.07)
            width = ring['width'] * rng.uniform(0.92, 1.08)
            tilt = np.radians(ring['tilt'] + rng.normal(0, 4))
            twist = np.radians(rng.normal(0, 7))
            curl = ring['curl'] + rng.normal(0, 0.05)
            P, hint, UV = petal_sheet(length, width, tilt, twist, cup=0.35, curl=curl, base_lift=ring['lift'])
            P = P + np.array([ring['r0'], 0, 0])
            R = rot_y(az)
            P = P @ R.T
            hint = hint @ R.T
            mesh.add_patch(place_head(P), UV, hint @ HEAD_ROT.T, MAT_PETAL, skin_head)


def build_disk(mesh, nr=6, ns=28):
    base_y = 0.03
    rs = DISK_R * (np.arange(1, nr + 1) / nr) ** 0.85
    th = np.linspace(0, 2 * np.pi, ns, endpoint=False)
    rows, uvs = [], []
    for r in rs:
        q = r / DISK_R
        y = base_y + DISK_H * (1 - q**2) ** 0.75 + 0.006 * np.sin(9 * th + 5 * q) * q
        x, z = r * np.cos(th), r * np.sin(th)
        rows.append(np.stack([x, np.full(ns, 0.0) + y, z], -1))
        uvs.append(np.stack([0.25 + 0.234 * x / DISK_R, 0.5 + 0.234 * z / DISK_R], -1))
    P = np.stack(rows)
    UV = np.stack(uvs)
    hint = np.zeros_like(P)
    hint[..., 1] = 1
    center = np.array([0.0, base_y + DISK_H, 0.0])
    mesh.add_fan(place_head(center[None])[0], place_head(P[0]), np.array([0.25, 0.5]), UV[0],
                 HEAD_AXIS, MAT_DISK, skin_head)
    mesh.add_patch(place_head(P), UV, hint @ HEAD_ROT.T, MAT_DISK, skin_head, wrap=True)
    # a short lip rolling the rim down toward the petals so the dome reads as a button
    lip = np.stack([P[-1], P[-1] + np.array([0, -0.045, 0]) + P[-1] * np.array([0.12, 0, 0.12])])
    lip_uv = np.stack([UV[-1], UV[-1]])
    lip_hint = np.zeros_like(lip)
    lip_hint[..., 0] = lip[..., 0]
    lip_hint[..., 2] = lip[..., 2]
    mesh.add_patch(place_head(lip), lip_uv, lip_hint @ HEAD_ROT.T, MAT_DISK, skin_head, wrap=True)


def build_calyx(mesh, ns=20):
    """Cup under the head, from the stem's radius up to just under the petals."""
    prof_t = np.linspace(0, 1, 6)
    ys = -0.30 + 0.30 * prof_t
    rs = STEM_R + (DISK_R + 0.02 - STEM_R) * prof_t**1.8
    th = np.linspace(0, 2 * np.pi, ns, endpoint=False)
    P = np.stack([np.stack([r * np.cos(th), np.full(ns, y), r * np.sin(th)], -1) for y, r in zip(ys, rs)])
    hint = P * np.array([1, 0, 1]) + np.array([0, -0.3, 0])
    UV = np.zeros(P.shape[:2] + (2,))
    mesh.add_patch(place_head(P), UV, hint @ HEAD_ROT.T, MAT_SEPAL, skin_head, wrap=True)
    # sepals: short pointed leaves fanned out and drooping under the petals
    n = 13
    for k in range(n):
        az = 2 * np.pi * (k + 0.3) / n + rng.normal(0, 0.03)
        Pp, h, UVp = petal_sheet(0.24 * rng.uniform(0.9, 1.1), 0.055, np.radians(-28 + rng.normal(0, 5)),
                                 np.radians(rng.normal(0, 8)), cup=0.2, curl=-0.05, base_lift=0.0, nu=6, nv=3)
        Pp = Pp + np.array([0.10, -0.02, 0])
        R = rot_y(az)
        mesh.add_patch(place_head(Pp @ R.T), UVp * 0, (h @ R.T) @ HEAD_ROT.T, MAT_SEPAL, skin_head)


def build_stem(mesh, nl=22, ns=10):
    t = np.linspace(0, 1, nl)
    C = stem_curve(t)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    th = np.linspace(0, 2 * np.pi, ns, endpoint=False)
    rows, hints = [], []
    for i in range(nl):
        R = frame_from_axis(T[i])
        r = STEM_R * (1.12 - 0.22 * t[i])
        ring = np.stack([r * np.cos(th), np.zeros(ns), r * np.sin(th)], -1) @ R.T
        rows.append(C[i] + ring)
        hints.append(ring)
    P = np.stack(rows)
    hint = np.stack(hints)
    UV = np.zeros(P.shape[:2] + (2,))
    mesh.add_patch(P, UV, hint, MAT_STEM, skin_stem, wrap=True)
    # cap the cut end
    mesh.add_fan(C[0], P[0][::-1], np.zeros(2), np.zeros((ns, 2)), np.array([0, -1.0, 0]), MAT_STEM, skin_stem)


# ---------------------------------------------------------------- texture
def paint_texture(path):
    size = 512
    img = Image.new('RGB', (size, size), (250, 215, 95))
    px = np.asarray(img).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # left half: the disk, a radial gradient from an olive heart out to bright yellow
    cx, cy, R = 128.0, 256.0, 120.0
    q = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / R
    stops = [(0.0, (112, 96, 44)), (0.32, (150, 120, 40)), (0.5, (225, 165, 40)),
             (0.8, (245, 195, 60)), (1.0, (250, 220, 110))]
    disk = np.zeros((size, size, 3), np.float32)
    for (q0, c0), (q1, c1) in zip(stops, stops[1:]):
        f = np.clip((q - q0) / (q1 - q0), 0, 1)[..., None]
        seg = (q >= q0) & (q < q1)
        disk[seg] = ((1 - f) * np.array(c0) + f * np.array(c1))[seg]
    disk[q >= 1.0] = stops[-1][1]
    left = xx < size / 2
    px[left] = disk[left]
    # right half: the petal, a warm cream at the base that goes white by a third of the way up
    u = np.clip((xx - 266) / (size - 20 - 266), 0, 1)
    f = np.clip(u / 0.38, 0, 1)[..., None]
    petal = (1 - f) * np.array([236, 230, 186], np.float32) + f * np.array([254, 254, 250], np.float32)
    px[~left] = petal[~left]
    img = Image.fromarray(np.clip(px, 0, 255).astype(np.uint8))
    # florets on the disk, laid out on a sunflower spiral so they read as a texture, not a pattern
    d = ImageDraw.Draw(img)
    golden = np.pi * (3 - np.sqrt(5))
    n = 260
    for i in range(n):
        rr = R * 0.97 * np.sqrt((i + 0.5) / n)
        a = i * golden
        x, y = cx + rr * np.cos(a), cy + rr * np.sin(a)
        qq = rr / R
        rad = 3.2 + 2.2 * qq
        col = (120, 100, 40) if qq < 0.42 else (205, 145, 35)
        d.ellipse([x - rad, y - rad, x + rad, y + rad], outline=col, width=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


# ---------------------------------------------------------------- pmx assembly
class Model:
    pass


def material(name, name_en, diffuse, ambient, edge, edge_size, flag, fcount, tex):
    return dict(name=name, name_en=name_en, diffuse=list(diffuse) + [1.0], specular=[0.0, 0.0, 0.0],
                shininess=5.0, ambient=list(ambient), flag=flag, edge_color=list(edge) + [1.0],
                edge_size=edge_size, tex=tex, sph=-1, sph_mode=0, toon_shared=1, toon=0, memo='',
                fcount=fcount)


def bone(name, name_en, pos, parent, flag, tail=None, tail_offset=None):
    b = dict(name=name, name_en=name_en, pos=[float(x) for x in pos], parent=parent, layer=0, flag=flag)
    if flag & 0x0001:
        b['tail'] = tail
    else:
        b['tail_offset'] = list(tail_offset)
    return b


def assemble(mesh):
    m = Model()
    m.name, m.name_en = 'デイジー', 'daisy'
    m.comment = 'マーガレット風の白い花。茎1〜茎3で茎を曲げ、花で花首を向ける。全ての親を手首に外部親で持たせる。'
    m.comment_en = ('A white daisy. Bend the stem with 茎1-茎3, aim the head with 花; '
                    'hold it by giving 全ての親 an outside parent on a wrist bone.')
    m.uv_add = 0
    m.vertices = [(p, n, uv, [], wt, bs, ws, 1.0)
                  for p, n, uv, (wt, bs, ws) in zip(mesh.pos, mesh.nor, mesh.uv, mesh.skin)]
    order = [MAT_STEM, MAT_SEPAL, MAT_PETAL, MAT_DISK]
    m.faces = [i for mat in order for tri in mesh.tris[mat] for i in tri]
    m.textures = ['textures/daisy.png']
    green = (0.36, 0.62, 0.27)
    dark_green = (0.10, 0.26, 0.10)
    m.materials = [
        material('茎', 'stem', green, (0.32, 0.45, 0.25), dark_green, 0.6, 0x1e, len(mesh.tris[MAT_STEM]) * 3, -1),
        material('がく', 'sepal', (0.40, 0.66, 0.30), (0.34, 0.47, 0.26), dark_green, 0.6, 0x1f, len(mesh.tris[MAT_SEPAL]) * 3, -1),
        material('花びら', 'petal', (1.0, 1.0, 1.0), (0.58, 0.58, 0.56), (0.30, 0.34, 0.26), 0.7, 0x1f, len(mesh.tris[MAT_PETAL]) * 3, 0),
        material('花芯', 'disk', (1.0, 1.0, 1.0), (0.55, 0.52, 0.42), (0.42, 0.28, 0.05), 0.7, 0x1e, len(mesh.tris[MAT_DISK]) * 3, 0),
    ]
    c = [stem_curve(t) for t in (0.0, 1 / 3, 2 / 3, 1.0)]
    m.bones = [
        bone('全ての親', 'root', (0, 0, 0), -1, 0x001e, tail_offset=(0, 1.0, 0)),
        bone('茎1', 'stem1', c[0], BONE_ROOT, 0x001b, tail=BONE_STEM2),
        bone('茎2', 'stem2', c[1], BONE_STEM1, 0x001b, tail=BONE_STEM3),
        bone('茎3', 'stem3', c[2], BONE_STEM2, 0x001b, tail=BONE_HEAD),
        bone('花', 'head', c[3], BONE_STEM3, 0x001a, tail_offset=tuple(HEAD_AXIS * 0.4)),
    ]
    m.morphs = []
    m.frames = [
        dict(name='Root', name_en='Root', special=1, items=[(0, BONE_ROOT)]),
        dict(name='表情', name_en='Exp', special=1, items=[]),
        dict(name='茎', name_en='stem', special=0, items=[(0, b) for b in (BONE_STEM1, BONE_STEM2, BONE_STEM3, BONE_HEAD)]),
    ]
    m.rigids, m.joints = [], []
    return m


def build():
    mesh = Mesh()
    build_stem(mesh)
    build_calyx(mesh)
    build_petals(mesh)
    build_disk(mesh)
    return mesh, assemble(mesh)


def preview(mesh, m, out, size=520):
    sys.path.insert(0, os.path.join(HERE, '..', 'zzz2pmx'))
    import raster
    V = np.array(mesh.pos)
    N = np.array(mesh.nor)
    colors = {MAT_STEM: (0.36, 0.62, 0.27), MAT_SEPAL: (0.40, 0.66, 0.30),
              MAT_PETAL: (0.97, 0.97, 0.94), MAT_DISK: (0.93, 0.72, 0.22)}
    UV = np.array(mesh.uv)
    tex = np.asarray(Image.open(os.path.join(OUT_DIR, 'textures', 'daisy.png')).convert('RGB'), np.float32) / 255
    C = np.zeros((len(V), 3))
    F = []
    for mat, tris in mesh.tris.items():
        tris = np.array(tris)
        if not len(tris):
            continue
        vs = np.unique(tris)
        C[vs] = colors[mat]
        if m.materials[[MAT_STEM, MAT_SEPAL, MAT_PETAL, MAT_DISK].index(mat)]['tex'] >= 0:
            h, w, _ = tex.shape
            C[vs] = tex[np.clip((UV[vs, 1] * (h - 1)).astype(int), 0, h - 1),
                        np.clip((UV[vs, 0] * (w - 1)).astype(int), 0, w - 1)]
        F.append(tris)
    F = np.concatenate(F)
    shade = raster.shade(N, light=(0.3, 0.5, -0.8), base=(1, 1, 1))
    Cs = np.clip(C * shade, 0, 1)
    # whole flower from the front and side, then the head from above, the side and below
    head = (V[F][:, :, 1] > STEM_LEN - 0.9).all(1)
    views = [(V, F, Cs, (0, 1, 2, -1)), (V, F, Cs, (2, 1, 0, 1)), (V, F[head], Cs, (0, 2, 1, -1)),
             (V, F[head], Cs, (2, 1, 0, 1)), (V, F[head], Cs, (0, 2, 1, 1))]
    raster.render(views, size=size, bg=(40, 42, 48)).save(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'daisy.pmx'))
    ap.add_argument('--preview')
    a = ap.parse_args()
    mesh, m = build()
    paint_texture(os.path.join(os.path.dirname(a.out), 'textures', 'daisy.png'))
    n = write_pmx(a.out, m)
    print(a.out, n, 'bytes', len(m.vertices), 'verts', len(m.faces) // 3, 'tris')
    if a.preview:
        preview(mesh, m, a.preview)
        print(a.preview)
