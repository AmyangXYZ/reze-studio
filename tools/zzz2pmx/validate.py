"""Structural and rig checks on a built PMX.

    python3 tools/zzz2pmx/validate.py public/models/reze-zzz/*.pmx
    python3 tools/zzz2pmx/validate.py --against <rip.pmx> <built.pmx>

`--against` compares how weight is spread over the skeleton. The mod and the rip
are the same character on the same bones, so a bone driving several times the
surface it drives in the rip is worth looking at — that is what a hand welded to a
fingertip reads as in numbers, and it is invisible to any check that only asks
whether the file parses. It is printed rather than failed: a mod is free to
re-proportion a character, and some of these are real differences.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import PMX


def bone_mass(m):
    """How much surface each bone drives, in area rather than vertices.

    Counting vertices compares tessellation, not rigging: the mod's bare hands
    carry several times the vertex density of the rip's gloved ones, and every
    finger reads as ten times overweight even when the rig is right.
    """
    V = np.array([v[0] for v in m.vertices])
    F = np.array(m.faces).reshape(-1, 3)
    tri = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1) / 2
    area = np.zeros(len(V))
    for c in range(3):
        np.add.at(area, F[:, c], tri / 3)
    out = {}
    for i, v in enumerate(m.vertices):
        for b, w in zip(v[5], v[6]):
            if w > 0:
                out[m.bones[b]['name']] = out.get(m.bones[b]['name'], 0.0) + w * area[i]
    return out


def check(path, against=None):
    m = PMX(path)
    bad = []
    nb, nv, nm, nmat, nr = (len(m.bones), len(m.vertices), len(m.morphs),
                            len(m.materials), len(m.rigids))

    for i, v in enumerate(m.vertices):
        for b in v[5]:
            if not (0 <= b < nb):
                bad.append(f'vertex {i}: bone index {b} out of range')
        s = sum(v[6][:len(v[5])]) if v[4] != 0 else 1.0
        if abs(s - 1.0) > 1e-3:
            bad.append(f'vertex {i}: weights sum to {s:.4f}')
    for f in m.faces:
        if not (0 <= f < nv):
            bad.append(f'face index {f} out of range')

    # Mesh on a bone the IK solves, where a twin exists to deform it instead. This
    # regression has come back after five separate grafts, because every new script has
    # to remember to run `fixrig` last and one of them always forgets. The check asks
    # `fixrig` itself what it would move, so the two cannot drift apart -- an earlier
    # version of this re-derived the rule and disagreed with it on the toe.
    import collections
    import fixrig
    swap, names = fixrig.retarget(m)
    if swap:
        hung = collections.Counter()
        for v in m.vertices:
            for b, w in zip(v[5], v[6][:len(v[5])]):
                if w > 0 and b in swap:
                    hung[b] += 1
        for i, n in hung.most_common():
            bad.append(f'{n} vertices hang on {names[i]}, which IK solves — '
                       f'{names[swap[i]]} is the bone that deforms them. Run fixrig.py')

    if sum(x['fcount'] for x in m.materials) != len(m.faces):
        bad.append('material face counts do not cover the face list')
    for i, b in enumerate(m.bones):
        if b['parent'] >= i and b['parent'] >= 0:
            bad.append(f'bone {i} {b["name"]}: parent {b["parent"]} comes later')
        if b['flag'] & 0x0001 and not (-1 <= b['tail'] < nb):
            bad.append(f'bone {i}: tail out of range')
    for i, mo in enumerate(m.morphs):
        lim = {0: nm, 9: nm, 1: nv, 2: nb, 8: nmat, 10: nr}.get(mo['type'], nv)
        for o in mo['offsets']:
            if not (-1 <= o[0] < lim):
                bad.append(f'morph {i} {mo["name"]}: index {o[0]} out of range')

    # a dynamic body with no joint falls away and drags its bone with it
    linked = {i for j in m.joints for i in (j['a'], j['b'])}
    for i, r in enumerate(m.rigids):
        if r['mode'] != 0 and i not in linked:
            bad.append(f'rigid body {i} {r["name"]}: dynamic but unjointed')
        if not (-1 <= r['bone'] < nb):
            bad.append(f'rigid body {i}: bone out of range')
    for i, j in enumerate(m.joints):
        if not (0 <= j['a'] < nr and 0 <= j['b'] < nr):
            bad.append(f'joint {i}: rigid index out of range')

    # A dynamic body that starts inside a collider it collides with is ejected on the
    # first frame, which reads as the hair jumping off the head before it settles.
    def axis(rot):
        rx, ry, rz = rot
        cx, sx, cy, sy, cz, sz = (np.cos(rx), np.sin(rx), np.cos(ry),
                                  np.sin(ry), np.cos(rz), np.sin(rz))
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return (Ry @ Rx @ Rz) @ np.array([0.0, 1.0, 0.0])

    def hits(a, b):
        return not ((a['nocollide'] >> b['group']) & 1 or (b['nocollide'] >> a['group']) & 1)

    for d in m.rigids:
        if d['mode'] == 0 or d['shape'] != 2:
            continue
        p0 = np.array(d['pos']) - axis(d['rot']) * d['size'][1] / 2
        p1 = np.array(d['pos']) + axis(d['rot']) * d['size'][1] / 2
        for st in m.rigids:
            if st['mode'] != 0 or st['shape'] != 0 or not hits(d, st):
                continue
            c = np.array(st['pos'])
            ab = p1 - p0
            t = 0.0 if not ab.dot(ab) else float(np.clip((c - p0).dot(ab) / ab.dot(ab), 0, 1))
            pen = (st['size'][0] + d['size'][0]) - np.linalg.norm(p0 + t * ab - c)
            if pen > 0.02:
                bad.append(f'rigid body {d["name"]}: starts {pen:.2f} inside '
                           f'the static body {st["name"]} it collides with')

    # Where a bone and its parent blend is where the joint is. A bone standing far
    # from its own joint pivots outside the mesh, and the skin around it stretches
    # into a spike the moment a motion rotates it — invisible at rest.
    V = np.array([v[0] for v in m.vertices])
    W = np.zeros((nv, nb))
    for i, v in enumerate(m.vertices):
        for b, w in zip(v[5], v[6]):
            if w > 0:
                W[i, b] += w
    off = []
    for b, bone in enumerate(m.bones):
        p = bone['parent']
        if p < 0:
            continue
        w = W[:, b] * W[:, p]
        if w.sum() < 2.0:
            continue
        joint = (V * w[:, None]).sum(0) / w.sum()
        off.append((float(np.linalg.norm(joint - np.array(bone['pos']))), bone['name']))
    for d, name in sorted(off, reverse=True):
        if d > 0.8:
            bad.append(f'bone {name}: stands {d:.2f} from the joint its own skin marks')

    # Winding against the stored normals, per material. A material whose triangles wind
    # the other way renders inside out, and reflecting geometry is the usual way to get
    # there -- as is reversing the winding of the wrong material while fixing it.
    N = np.array([v[1] for v in m.vertices])
    F = np.array(m.faces).reshape(-1, 3)
    at = 0
    for mat in m.materials:
        k = mat['fcount'] // 3
        if k:
            t = F[at:at + k]
            a, b, c = V[t[:, 0]], V[t[:, 1]], V[t[:, 2]]
            agree = float(np.mean(((np.cross(b - a, c - a)) * N[t].mean(1)).sum(1) > 0))
            if agree < 0.9:
                bad.append(f'material {mat["name"]}: {100 * (1 - agree):.0f}% of its '
                           'triangles wind against their normals — it renders inside out')
        at += k

    print(f'{os.path.basename(path)}: {nv} verts, {len(m.faces)//3} tris, {nmat} materials, '
          f'{nb} bones, {nm} morphs, {nr} rigid bodies, {len(m.joints)} joints')
    if off:
        print(f'  bones sit {np.median([d for d, _ in off]):.2f} from their own joint '
              f'(median over {len(off)}), worst {max(off)[0]:.2f}')

    if against is not None:
        mine, theirs = bone_mass(m), bone_mass(against)
        ms, ts = sum(mine.values()), sum(theirs.values())
        over = []
        for name, w in mine.items():
            if name not in theirs or w / ms < 0.003:
                continue
            ratio = (w / ms) / (theirs[name] / ts)
            if ratio > 4.0:
                over.append((ratio, name, 100 * w / ms, 100 * theirs[name] / ts))
        print(f'  weight spread over {len(mine)} bones; '
              f'{len(over)} above 4x the rip\'s share:')
        for ratio, name, a, b in sorted(over, reverse=True):
            print(f'   . {name}: {a:.1f}% of the surface against {b:.1f}% '
                  f'in the rip ({ratio:.1f}x)')

    for t in m.textures:
        if not os.path.exists(os.path.join(os.path.dirname(path), t.replace('\\', '/'))):
            bad.append(f'missing texture {t}')
    if bad:
        print(f'  {len(bad)} problems:')
        for b in bad[:40]:
            print('   -', b)
    else:
        print('  clean')
    return bad


if __name__ == '__main__':
    args = sys.argv[1:]
    ref = None
    if '--against' in args:
        i = args.index('--against')
        ref = PMX(args[i + 1])
        args = args[:i] + args[i + 2:]
    fails = sum(len(check(p, ref)) for p in args)
    sys.exit(1 if fails else 0)
