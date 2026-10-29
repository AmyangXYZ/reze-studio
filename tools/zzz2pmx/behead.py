# -*- coding: utf-8 -*-
"""Reze in two: the body without its head, and the head on its own.

    python3 tools/zzz2pmx/behead.py

The neck is cut by one horizontal plane, just above the choker. The skin crosses it,
and on the head so does the chin's underside: their triangles are split on the plane
-- position, normal, UV, weights and every expression interpolated along the cut edge
-- and each side gets a flat lid over its section. The body's throat-front, open under
the chin where the face used to cover it, is fanned shut. The hair, eyes and mouth go
whole to the head, the clothes whole to the body; the head is moved to sit on the
origin, the cut's centre at y 0. Both keep the whole skeleton, since an animation drives 頭
through 首 through 上半身 either way; the hair chains and their bodies go with the
head, the chest and skirt chains with the body.
"""
import os
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixrig
import holefill
import progress
import prune
import publish
from pmx import PMX
from pmxw import write_pmx

SRC = 'public/models/reze/reze.pmx'
CUT = 18.35            # the plane, above the choker's top at 18.3, below the chin's front
NECK = 'skin'          # the one material the plane goes through
HEAD = {'头发', 'face01', '齿', '口腔', '舌', '唇', 'eyebrow', 'eyelash', 'eyelash_crease',
        '目白', '瞳1', '瞳2'}
OUT = {'reze-headless': False, 'reze-head': True}


def lerp_vertex(a, b, t):
    """The vertex `t` of the way from `a` to `b`, as BDEF4."""
    pos = (np.array(a[0]) * (1 - t) + np.array(b[0]) * t).tolist()
    nor = np.array(a[1]) * (1 - t) + np.array(b[1]) * t
    nor = (nor / (np.linalg.norm(nor) + 1e-12)).tolist()
    uv = (np.array(a[2]) * (1 - t) + np.array(b[2]) * t).tolist()
    add = [(np.array(x) * (1 - t) + np.array(y) * t).tolist() for x, y in zip(a[3], b[3])]
    wts = {}
    for v, s in ((a, 1 - t), (b, t)):
        for bone, w in zip(v[5], v[6]):
            if w > 0:
                wts[bone] = wts.get(bone, 0.0) + s * w
    top = sorted(wts.items(), key=lambda x: -x[1])[:4]
    tot = sum(w for _, w in top)
    bones = [bn for bn, _ in top] + [0] * (4 - len(top))
    weights = [w / tot for _, w in top] + [0.0] * (4 - len(top))
    return (pos, nor, uv, add, 2, bones, weights, a[7] * (1 - t) + b[7] * t)


def split(m, name, y, above):
    """Keep of material `name` what lies on one side of the plane y=`y`, cutting the
    triangles it crosses. Returns the loop of new vertices on the plane, as directed
    edges in the kept triangles' winding."""
    F = np.array(m.faces).reshape(-1, 3)
    V = np.array([v[0] for v in m.vertices])
    side = (V[:, 1] > y) if above else (V[:, 1] < y)
    at = 0
    new_faces, cut_edges, on_edge = [], [], {}
    vmorph = {}                      # vertex -> [(morph index, offset)]
    for mi, mo in enumerate(m.morphs):
        if mo['type'] in (1, 3, 4, 5, 6, 7):
            for o in mo['offsets']:
                vmorph.setdefault(o[0], []).append((mi, np.array(o[1])))

    def crossing(i, j):
        key = (min(i, j), max(i, j))
        if key not in on_edge:
            t = (y - V[i, 1]) / (V[j, 1] - V[i, 1])
            v = lerp_vertex(m.vertices[i], m.vertices[j], t)
            v[0][1] = y
            m.vertices.append(v)
            k = len(m.vertices) - 1
            on_edge[key] = k
            offs = {}
            for src, s in ((i, 1 - t), (j, t)):
                for mi, o in vmorph.get(src, ()):
                    offs[mi] = offs.get(mi, 0) + s * o
            for mi, o in offs.items():
                m.morphs[mi]['offsets'].append((k, o.tolist()))
        return on_edge[key]

    for mat in m.materials:
        n = mat['fcount'] // 3
        tris = F[at:at + n]
        at += n
        if mat['name'] != name:
            new_faces.append(tris)
            continue
        kept = []
        for tri in tris:
            inside = [bool(side[i]) for i in tri]
            k = sum(inside)
            if k == 3:
                kept.append(list(tri))
            elif k == 0:
                continue
            elif k == 1:
                a = tri[inside.index(True)]
                b, c = [i for i in tri if i != a]
                # keep the winding: a, then the next two in the triangle's order
                order = list(tri); r = order.index(a)
                b, c = order[(r + 1) % 3], order[(r + 2) % 3]
                ab, ac = crossing(int(a), int(b)), crossing(int(a), int(c))
                kept.append([int(a), ab, ac])
                cut_edges.append((ab, ac))
            else:
                c = tri[inside.index(False)]
                order = list(tri); r = order.index(c)
                a, b = order[(r + 1) % 3], order[(r + 2) % 3]
                bc, ca = crossing(int(b), int(c)), crossing(int(c), int(a))
                kept.append([int(a), int(b), bc])
                kept.append([int(a), bc, ca])
                cut_edges.append((bc, ca))
        mat['fcount'] = len(kept) * 3
        new_faces.append(np.array(kept, int).reshape(-1, 3))
    m.faces = np.concatenate(new_faces).ravel().tolist()
    return cut_edges


def inside(C, S):
    """Which points of the plane lie inside the section whose edges are `S`
    (segments, in the plane): crossing parity along four directions, by majority.
    The section is not quite closed -- the skin's throat is open under the chin --
    and a gap fools one direction, never three."""
    votes = np.zeros(len(C), int)
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        d = np.array(d, float)
        n = np.array([-d[1], d[0]])
        a, b = S[:, 0], S[:, 1]
        sa = (a[None] - C[:, None]) @ n
        sb = (b[None] - C[:, None]) @ n
        straddle = (sa > 0) != (sb > 0)
        t = sa / (sa - sb + 1e-12)
        X = a[None] + t[..., None] * (b - a)[None]
        ahead = ((X - C[:, None]) @ d) > 0
        votes += ((straddle & ahead).sum(1) % 2 == 1)
    return votes >= 3


def lid(m, name, edges, up):
    """A flat lid over the cut. The vertices on the plane -- from every material the
    plane went through -- are triangulated in the plane (Delaunay), and the triangles
    inside the section kept: the neck's ring, and on the head the underside of the
    chin in front of it. On doubled vertices carrying a real skin texel, wound to
    face `up`."""
    from scipy.spatial import Delaunay
    if not edges:
        return 0
    rim = sorted({i for e in edges for i in e})
    V = np.array([v[0] for v in m.vertices])
    P = V[rim]
    tri = Delaunay(P[:, [0, 2]]).simplices
    C = P[tri].mean(1)[:, [0, 2]]
    S = np.array([[V[a][[0, 2]], V[b][[0, 2]]] for a, b in edges])
    tri = tri[inside(C, S)]
    nrm = [0.0, 1.0 if up else -1.0, 0.0]
    texel = skin_texel(m, name)
    double = []
    for i in rim:
        v = list(m.vertices[i])
        v[1], v[2] = list(nrm), texel(v[0])
        m.vertices.append(tuple(v))
        double.append(len(m.vertices) - 1)
    double = np.array(double)
    tris = []
    for t in tri:
        a, b, c = P[t]
        n = np.cross(b - a, c - a)
        t = t if (n[1] > 0) == up else t[[1, 0, 2]]
        tris.append(double[t].tolist())
    F = np.array(m.faces).reshape(-1, 3)
    at, out = 0, []
    for mat in m.materials:
        n = mat['fcount'] // 3
        out.append(F[at:at + n])
        at += n
        if mat['name'] == name:
            out.append(np.array(tris, int))
            mat['fcount'] += len(tris) * 3
    m.faces = np.concatenate(out).ravel().tolist()
    return len(tris)


def skin_texel(m, name):
    """UV lookup by position from the material's vertices that lie on no rim: a rim
    is where an atlas island was cut, so a rim vertex's own UV can sit on the island's
    black edge, and a fan drawn on it comes out dark."""
    V = np.array([v[0] for v in m.vertices])
    UV = np.array([v[2] for v in m.vertices])
    on_rim = {int(i) for l in holefill.rims(m, [name]) for i in l}
    inner = np.array([int(i) for i in np.unique(holefill.tris_of(m, [name])) if int(i) not in on_rim])
    tree = cKDTree(V[inner])
    return lambda p: UV[inner[tree.query(p)[1]]].tolist()


def throat(m, name, y, axis, reach=0.4):
    """The skin's throat is open at the front, under the chin, where the face used to
    cover it; with the neck cut it shows. Every rim of the skin that touches the cut
    is fanned shut, on doubled vertices carrying a nearby skin texel, facing away from
    the neck's axis."""
    V = np.array([v[0] for v in m.vertices])
    texel = skin_texel(m, name)
    loops = [l for l in holefill.rims(m, [name]) if abs(V[list(l), 1].mean() - y) < reach]
    tris = []
    for l in loops:
        l = [int(i) for i in l]
        P = V[l]
        center = P.mean(0)
        nrm = np.zeros(3)
        for k in range(len(l)):
            nrm += np.cross(P[k] - center, P[(k + 1) % len(l)] - center)
        out = center - np.array([axis[0], center[1], axis[2]])
        if nrm @ out < 0:
            l = l[::-1]
            P = P[::-1]
            nrm = -nrm
        nrm = (nrm / (np.linalg.norm(nrm) + 1e-12)).tolist()
        double = {}
        for i in l:
            v = list(m.vertices[i])
            v[1], v[2] = nrm, texel(v[0])
            m.vertices.append(tuple(v))
            double[i] = len(m.vertices) - 1
        near = l[int(np.argmin(np.linalg.norm(P - center, axis=1)))]
        c = list(m.vertices[near])
        c[0], c[1], c[2] = center.tolist(), nrm, texel(center)
        m.vertices.append(tuple(c))
        ci = len(m.vertices) - 1
        for k in range(len(l)):
            tris.append([double[l[k]], double[l[(k + 1) % len(l)]], ci])
    if tris:
        F = np.array(m.faces).reshape(-1, 3)
        at, out = 0, []
        for mat in m.materials:
            n = mat['fcount'] // 3
            out.append(F[at:at + n])
            at += n
            if mat['name'] == name:
                out.append(np.array(tris, int))
                mat['fcount'] += len(tris) * 3
        m.faces = np.concatenate(out).ravel().tolist()
    return [len(l) for l in loops]


def recenter(m, origin):
    """Move the model so `origin` sits at the world origin: vertices, bones, rigid
    bodies and joints. Offsets -- vertex and bone morphs, bone tails, SDEF -- are
    relative and stay."""
    o = np.array(origin, float)
    for k, v in enumerate(m.vertices):
        v = list(v)
        v[0] = (np.array(v[0]) - o).tolist()
        if v[4] == 3:
            c = list(v[8])
            c[0:3] = (np.array(c[0:3]) - o).tolist()
            v[8] = c
        m.vertices[k] = tuple(v)
    for b in m.bones:
        b['pos'] = (np.array(b['pos']) - o).tolist()
    for r in m.rigids:
        r['pos'] = (np.array(r['pos']) - o).tolist()
    for j in m.joints:
        j['pos'] = (np.array(j['pos']) - o).tolist()


def compact(m, want):
    """Only the materials `want` accepts, with the vertices they use, renumbered; the
    expressions renumbered with them and the empty ones dropped."""
    F = np.array(m.faces).reshape(-1, 3)
    mats, faces, at = [], [], 0
    for mat in m.materials:
        n = mat['fcount'] // 3
        if want(mat['name']) and n:
            mats.append(mat)
            faces.append(F[at:at + n])
        at += n
    F = np.concatenate(faces)
    used = np.unique(F)
    remap = {int(o): k for k, o in enumerate(used)}
    m.vertices = [m.vertices[int(i)] for i in used]
    m.faces = np.vectorize(remap.get)(F).ravel().tolist()
    old_name = {i: mat['name'] for i, mat in enumerate(m.materials)}
    new_mat = {mat['name']: i for i, mat in enumerate(mats)}
    m.materials = mats
    morphs = []
    for mo in m.morphs:
        t = mo['type']
        if t in (1, 3, 4, 5, 6, 7):
            off = [(remap[o[0]],) + tuple(o[1:]) for o in mo['offsets'] if o[0] in remap]
        elif t == 8:
            off = [o for o in mo['offsets'] if o[0] < 0] + [
                (new_mat[old_name[o[0]]],) + tuple(o[1:]) for o in mo['offsets']
                if o[0] >= 0 and old_name.get(o[0]) in new_mat]
        else:
            off = list(mo['offsets'])
        if not off and mo['offsets']:
            continue
        mo['offsets'] = off
        morphs.append(mo)
    alive = {mo['name']: i for i, mo in enumerate(morphs)}
    for mo in morphs:
        if mo['type'] in (0, 9):
            names = [m.morphs[o[0]]['name'] for o in mo['offsets']]
            mo['offsets'] = [(alive[nm],) + tuple(o[1:])
                             for nm, o in zip(names, mo['offsets']) if nm in alive]
    m.morphs = morphs
    m.frames = [f for f in m.frames if f['name'] != '表情'] + [
        dict(name='表情', name_en='Exp', special=1, items=[(1, i) for i in range(len(morphs))])]
    return len(used), len(F)


def bones_only_of(m, dropped):
    """Bones nothing but the dropped materials weighted."""
    sl = holefill.slices(m)
    usedby = {}
    for k in sl:
        for i in np.unique(sl[k]):
            for b, w in zip(m.vertices[int(i)][5], m.vertices[int(i)][6]):
                if w > 1e-6:
                    usedby.setdefault(int(b), set()).add(k)
    return [m.bones[b]['name'] for b, ks in usedby.items() if ks and ks <= dropped]


def main(src=SRC, cut=CUT):
    for name, head in OUT.items():
        out_dir = os.path.join('public/models', name)
        work = os.path.join(out_dir, '_work')
        build = os.path.join(work, name + '.pmx')
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(os.path.join(work, 'textures'), exist_ok=True)
        with progress.step(name):
            m = PMX(src)
            base = os.path.dirname(src)
            for rel in m.textures:
                fp = os.path.join(base, rel.replace('\\', '/'))
                if os.path.exists(fp):
                    shutil.copyfile(fp, os.path.join(work, 'textures', os.path.basename(fp)))
            m.textures = ['textures\\' + os.path.basename(t.replace('\\', '/')) for t in m.textures]
            m.name = m.name_en = name
            want = (lambda n: n in HEAD or n == NECK) if head else (lambda n: n not in HEAD)
            dropped = {x['name'] for x in m.materials if not want(x['name'])}
            force = bones_only_of(m, dropped)
            edges = split(m, NECK, cut, above=head)
            through = [NECK]
            if head:
                # the chin's underside dips below the plane, in front of the neck
                for x in m.materials:
                    if x['name'] in HEAD and x['name'] != '头发':
                        more = split(m, x['name'], cut, above=True)
                        if more:
                            edges += more
                            through.append(x['name'])
            V = np.array([v[0] for v in m.vertices])
            axis = V[sorted({i for e in edges for i in e})].mean(0)
            n_lid = lid(m, NECK, edges, up=not head)
            holes = [] if head else throat(m, NECK, cut, axis)
            nv, nf = compact(m, want)
            if head:
                # the head sits on the origin: the cut's centre at y 0
                V = np.array([v[0] for v in m.vertices])
                on = np.abs(V[:, 1] - cut) < 1e-4
                recenter(m, [V[on, 0].mean(), cut, V[on, 2].mean()])
            print(f'  cut at y {cut} through {", ".join(through)}: {len(edges)} edges on the plane, '
                  f'lid of {n_lid} triangles; rims beside the cut shut: {holes}')
            print(f'  {nv} vertices, {nf} triangles, {len(m.morphs)} expressions kept; '
                  f'{len(force)} bones go with the dropped materials')
            write_pmx(build, m)
            prune.main(build, force)
            fixrig.main([build])
            publish.main(build, out_dir)
        shutil.rmtree(work, ignore_errors=True)
        m = PMX(os.path.join(out_dir, name + '.pmx'))
        print(f'  {out_dir}/{name}.pmx: {len(m.vertices)} verts, {len(m.faces) // 3} triangles, '
              f'{len(m.bones)} bones, {len(m.rigids)} rigid bodies, {len(m.joints)} joints')
        print(f'  materials: {", ".join(x["name"] for x in m.materials)}')
    progress.done()


if __name__ == '__main__':
    main(*sys.argv[1:])
