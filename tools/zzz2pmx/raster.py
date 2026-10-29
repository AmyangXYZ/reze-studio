"""Software rasterizer for quick mesh previews."""
import numpy as np
from PIL import Image


def render(views, size=512, bg=(24, 24, 28), bounds=None):
    """views: list of (verts Nx3, faces Mx3, colors Nx3 float 0-1 or None, label).

    Camera looks down -Y with Z up (game space). Returns PIL image row of views.
    """
    tiles = []
    for verts, faces, colors, axes in views:
        v = verts[:, axes[:2]].astype(np.float64)
        depth = verts[:, axes[2]].astype(np.float64) * (1 if len(axes) < 4 else axes[3])
        if bounds is None:
            used = np.unique(faces)
            lo, hi = v[used].min(0), v[used].max(0)
        else:
            lo, hi = bounds[0][list(axes[:2])], bounds[1][list(axes[:2])]
        c = (lo + hi) / 2
        s = (size * 0.9) / max(hi - lo)
        px = (v - c) * np.array([s, -s]) + size / 2
        img = np.full((size, size, 3), bg, np.float64)
        zb = np.full((size, size), np.inf)
        if colors is None:
            colors = np.full((len(verts), 3), 0.75)
        tri = px[faces]
        d = depth[faces]
        col = colors[faces]
        xmin = np.floor(tri[:, :, 0].min(1)).astype(int)
        xmax = np.ceil(tri[:, :, 0].max(1)).astype(int)
        ymin = np.floor(tri[:, :, 1].min(1)).astype(int)
        ymax = np.ceil(tri[:, :, 1].max(1)).astype(int)
        area = ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
        order = np.argsort(-d.mean(1))  # far to near not needed; zbuffer handles it
        for t in order:
            if abs(area[t]) < 1e-9:
                continue
            x0, x1 = max(xmin[t], 0), min(xmax[t] + 1, size)
            y0, y1 = max(ymin[t], 0), min(ymax[t] + 1, size)
            if x0 >= x1 or y0 >= y1:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1) + .5, np.arange(y0, y1) + .5)
            a, b, cc = tri[t]
            w0 = ((b[0] - a[0]) * (ys - a[1]) - (xs - a[0]) * (b[1] - a[1])) / area[t]
            w1 = ((xs - a[0]) * (cc[1] - a[1]) - (cc[0] - a[0]) * (ys - a[1])) / area[t]
            w2 = 1 - w0 - w1
            m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not m.any():
                continue
            z = w2 * d[t, 0] + w1 * d[t, 1] + w0 * d[t, 2]
            sub = zb[y0:y1, x0:x1]
            hit = m & (z < sub)
            if not hit.any():
                continue
            sub[hit] = z[hit]
            rgb = (w2[..., None] * col[t, 0] + w1[..., None] * col[t, 1] + w0[..., None] * col[t, 2])
            img[y0:y1, x0:x1][hit] = np.clip(rgb[hit], 0, 1) * 255
        tiles.append(Image.fromarray(img.astype(np.uint8)))
    out = Image.new('RGB', (size * len(tiles), size))
    for i, t in enumerate(tiles):
        out.paste(t, (i * size, 0))
    return out


def shade(normals, light=(0.4, -0.8, 0.5), base=(0.85, 0.82, 0.80)):
    l = np.array(light, float)
    l /= np.linalg.norm(l)
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
    d = np.clip(n @ l, 0, 1) * 0.7 + 0.35
    return np.clip(d[:, None] * np.array(base), 0, 1)
