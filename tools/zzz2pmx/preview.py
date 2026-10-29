"""Render a PMX from a few angles, textured, for eyeballing a build."""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raster
from pmx import PMX


def load(path, materials=None, crop=None):
    m = PMX(path)
    base = os.path.dirname(path)
    V = np.array([v[0] for v in m.vertices])
    N = np.array([v[1] for v in m.vertices])
    UV = np.array([v[2] for v in m.vertices])
    F = np.array(m.faces).reshape(-1, 3)
    C = np.full((len(V), 3), 0.8)
    cache = {}
    off = 0
    faces = []
    for mm in m.materials:
        n = mm['fcount'] // 3
        tris = F[off:off + n]
        off += n
        if materials and mm['name'] not in materials:
            continue
        faces.append(tris)
        # MMD shades as texture x (diffuse x light + ambient); without it a material
        # painted dark still renders at full texture brightness here.
        gain = np.clip(np.array(mm['diffuse'][:3]) * 0.604 + np.array(mm['ambient']), 0, 1.2)
        if mm['tex'] < 0:
            C[np.unique(tris)] = np.clip(np.array(mm['diffuse'][:3]) * gain, 0, 1)
            continue
        rel = m.textures[mm['tex']].replace('\\', '/')
        p = os.path.join(base, rel)
        if p not in cache:
            cache[p] = np.asarray(Image.open(p).convert('RGB'), np.float32) / 255 if os.path.exists(p) else None
        im = cache[p]
        if im is None:
            continue
        h, w, _ = im.shape
        v = np.unique(tris)
        u = np.clip(UV[v, 0] % 1.0, 0, 1)
        vv = np.clip(UV[v, 1] % 1.0, 0, 1)
        C[v] = np.clip(im[np.clip((vv * (h - 1)).astype(int), 0, h - 1),
                          np.clip((u * (w - 1)).astype(int), 0, w - 1)] * gain, 0, 1)
    F = np.concatenate(faces)
    if crop:
        F = F[(V[F][:, :, 1] > crop[0]).all(1) & (V[F][:, :, 1] < crop[1]).all(1)]
    return V, N, UV, F, C, m


if __name__ == '__main__':
    path = sys.argv[1]
    out = sys.argv[2]
    crop = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else None
    V, N, UV, F, C, m = load(path, crop=crop)
    views = [(V, F, C, ax) for ax in [(0, 1, 2, -1), (2, 1, 0, 1), (0, 1, 2, 1)]]
    raster.render(views, size=int(os.environ.get('SIZE', 460))).save(out)
    print(out, len(V), 'verts', len(F), 'tris')
