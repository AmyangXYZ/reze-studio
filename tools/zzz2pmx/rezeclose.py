"""Bring reze.pmx's breasts a little closer, on a copy, with a lattice run in Blender.

reze's cups sit a little far apart and the slope from each nipple in toward
the centre falls away flat. A B-spline cage round the chest slides each breast
a little toward the sternum and fills that inner slope, skin, bra, ribbon and
the chest rig all in one field, so nothing tears. The field is a product of
smooth bells: nothing at the sternum, full over the breast, nothing at the
sides or the back. The copy is written with everything else untouched;
reze.pmx itself is never opened for writing.

    python3 tools/zzz2pmx/rezeclose.py [--closer 0.08] [--inner 0.05]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate
from bustlattice import BLENDER, SCRIPT, faces_of, smoothstep, spans, welded_normals
from pmx import PMX
from pmxw import write_pmx

SKIN = 'skin'
CUP = 'cloth01'
BOX_LO = np.array([-2.2, 14.0, -2.6])
BOX_HI = np.array([2.2, 17.6, 0.2])
DIMS = np.array([12, 7, 5])  # 0.4 apart across, 0.6 up, 0.7 in


def field(a, apex):
    U, Vn, W = DIMS
    P = np.array([(x, y, z) for z in np.linspace(BOX_LO[2], BOX_HI[2], W)
                  for y in np.linspace(BOX_LO[1], BOX_HI[1], Vn)
                  for x in np.linspace(BOX_LO[0], BOX_HI[0], U)])
    x, y, z = P.T
    ax = np.abs(x)
    gx = smoothstep(ax, 0.1, 0.4) * (1 - smoothstep(ax, 1.3, 2.0))   # the sternum and the sides stay
    gy = np.exp(-((y - apex[1]) / 0.6) ** 2)
    front = smoothstep(-z, 0.9, 1.5)                                  # what stands off the chest wall
    D = np.zeros_like(P)
    D[:, 0] = -np.sign(x) * a.closer * gx * gy * front
    # the inner slope, from the nipple in and a little down, comes forward
    inner = np.exp(-((ax - 0.4) / 0.3) ** 2) * np.exp(-((y - (apex[1] - 0.2)) / 0.4) ** 2)
    D[:, 2] = -a.inner * inner * smoothstep(-z, 1.2, 1.8) * (1 - smoothstep(ax, 1.0, 1.4))
    return P, D


def main(a):
    H = PMX(a.host)
    V = np.array([v[0] for v in H.vertices])
    sp = spans(H)
    both = np.unique(np.concatenate([faces_of(H, SKIN).ravel(), faces_of(H, CUP).ravel()]))
    r = both[(V[both, 0] > 0.1) & (V[both, 1] > 14.5) & (V[both, 1] < 17.5)]
    apex = V[r[np.argmin(V[r, 2])]]
    print(f'apex {np.round(apex, 2)}')
    gate = np.ones(len(V))
    for k in range(3):
        gate *= smoothstep(V[:, k] - BOX_LO[k], 0, 0.4) * (1 - smoothstep(V[:, k] - BOX_HI[k], -0.4, 0))
    P, D = field(a, apex)
    names = [b['name'] for b in H.bones]
    rig_pts, rig_ref = [], []
    for i, b in enumerate(H.bones):
        if '胸' in b['name'] and abs(b['pos'][0]) > 0.2:
            rig_pts.append(b['pos']); rig_ref.append(('bone', i))
    for i, rb in enumerate(H.rigids):
        if '胸' in rb['name'] and abs(rb['pos'][0]) > 0.2:
            rig_pts.append(rb['pos']); rig_ref.append(('rigid', i))
    bodies = {i for kind, i in rig_ref if kind == 'rigid'}
    for i, jn in enumerate(H.joints):
        if jn['a'] in bodies or jn['b'] in bodies:
            rig_pts.append(jn['pos']); rig_ref.append(('joint', i))
    tmp = tempfile.mkdtemp(prefix='reze-close-')
    src, dst = os.path.join(tmp, 'in.npz'), os.path.join(tmp, 'out.npz')
    np.savez(src, V=V, mesh_names=np.array([], dtype=str), gate=gate, smooth=np.zeros(len(V)),
             box_lo=BOX_LO, box_hi=BOX_HI, dims=DIMS, disp=D, rig=np.array(rig_pts, float),
             **{'lambda': 0.0, 'iterations': 0})
    run = subprocess.run([BLENDER, '--background', '--python', SCRIPT, '--', src, dst, os.path.abspath(a.blend)],
                         capture_output=True, text=True)
    said = [ln for ln in run.stdout.splitlines() if ln.startswith('lattice')]
    if run.returncode or not said:
        sys.exit(run.stdout[-3000:] + run.stderr[-3000:])
    print(said[-1])
    got = np.load(dst)
    V2, rig2 = got['V'], got['rig']
    moved = np.linalg.norm(V2 - V, axis=1)
    print(f'cage: {int((moved > 1e-4).sum())} vertices moved, furthest {moved.max():.3f}; '
          f'apex now {np.round(V2[r[np.argmin(V[r, 2])]], 2)}')
    V = V2
    F = np.array(H.faces).reshape(-1, 3)
    for i in np.where(moved > 1e-4)[0]:
        v = list(H.vertices[i]); v[0] = [float(x) for x in V[i]]; H.vertices[i] = tuple(v)
    for name, (start, n) in sp.items():
        tris = F[start:start + n]
        idx = np.unique(tris)
        if (moved[idx] > 1e-4).any():
            N = welded_normals(V, tris, idx)
            for i in idx[moved[idx] > 1e-4]:
                v = list(H.vertices[i]); v[1] = [float(x) for x in N[i]]; H.vertices[i] = tuple(v)
    for (kind, i), p in zip(rig_ref, rig2):
        p = [float(x) for x in p]
        (H.bones if kind == 'bone' else H.rigids if kind == 'rigid' else H.joints)[i]['pos'] = p
    print(f'  rig: {sum(1 for k, _ in rig_ref if k == "bone")} bones, {len(bodies)} bodies and '
          f'{sum(1 for k, _ in rig_ref if k == "joint")} joints rode the cage')
    write_pmx(a.out, H)
    print('wrote', a.out)
    validate.check(a.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--host', default='public/models/reze/reze.pmx')
    p.add_argument('--out', default='public/models/reze/reze-close.pmx')
    p.add_argument('--blend', default='public/models/reze/reze-close.blend')
    p.add_argument('--closer', type=float, default=0.08, help='how far each breast slides toward the sternum')
    p.add_argument('--inner', type=float, default=0.05, help='how far the slope from nipple to centre comes forward')
    main(p.parse_args())
