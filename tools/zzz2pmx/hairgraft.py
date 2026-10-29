# -*- coding: utf-8 -*-
"""Put the ZZZ haircut on the hand-built Reze model.

Two ways, and the difference is what drives the strands.

`--physics head` (the default) gives the model the same hair `reze-zzz/reze-casual.pmx`
has, rig included: the haircut from the corrected pipeline, the right way round, every
vertex on `頭` at weight 1, and no sway chain at all. The 15 sway rigid bodies and their
16 joints are dropped with it, so nothing simulates that drives nothing. Reze's cut is
short and carries nothing long enough to need sway, and a strand welded to the skull
cannot come apart -- which is more than the alternatives managed.

`--physics copy` moves a working haircut across whole instead: mesh, weights, the
positions of the sway bones, the rigid bodies hanging off them and the joints between
those, taken verbatim from a build that already has them fitted. Binding new strands
to chains shaped for the old ones is what breaks the strand on the character's left,
so the rig travels with the mesh or not at all. It keeps whatever handedness that
build had.

Either way the rest of the target is untouched — including the four leg and hip
rigid bodies `reze-new-nomorph.pmx` has and `reze-zzz.pmx` lost.

    python3 tools/zzz2pmx/hairgraft.py
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import align
import presentation
import hairphys
import zzzmod
from pmx import PMX
from pmxw import write_pmx

MOD = 'zzz-mod/EvelynMod'
HAIR_MAT = '头发'
HAIR_TEX = 'zzz_EvelynHairADiffuse.png'
HAIR_BONES = ('yb_', 'zb_', 'e_')

# Where the haircut sits on this head. Fitted once against the earlier graft in
# reze-zzz.pmx -- which had placed it by hand -- to a residual of 0.0008, then kept as
# a constant so the build needs only the mod and the base model. Pass --reference to
# refit against another build's placement.
PLACEMENT = (0.93974, np.array([-0.02096, 0.79107, -0.78353]))


class _MatHolder:
    def __init__(self, materials):
        self.materials = materials


def material_groups(pmx):
    faces = np.array(pmx.faces).reshape(-1, 3)
    out, off = {}, 0
    for m in pmx.materials:
        n = m['fcount'] // 3
        out[m['name']] = faces[off:off + n]
        off += n
    return out


def mod_hair():
    """The haircut in MMD space, correctly handed."""
    hair = zzzmod.load(MOD)['Hair']
    V = align.to_mmd(hair.pos) * 12.5055 + np.array([0.0, 0.3052, -0.0006])
    N = align.to_mmd(hair.normal)
    tris = np.concatenate([zzzmod.ib(MOD, 'HairA'), zzzmod.ib(MOD, 'HairB')])
    return V, N, hair.uv, tris


def placement(V, ref, ref_group):
    """Reuse the placement an existing graft already fitted, unmirrored."""
    if ref is None:
        return PLACEMENT[0], PLACEMENT[1], 0.0
    RV = np.array([v[0] for v in ref.vertices])
    target = RV[np.unique(ref_group)] * [-1, 1, 1]
    s, t, d = align.icp_similarity(V, target, 1.0, target.mean(0) - V.mean(0), iters=150)
    return s, t, np.median(d)


def _axis(rot):
    rx, ry, rz = rot
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (Ry @ Rx @ Rz) @ np.array([0.0, 1.0, 0.0])


def is_hair_body(model, r):
    return 0 <= r['bone'] < len(model.bones) and \
        model.bones[r['bone']]['name'].startswith(HAIR_BONES)


def build(args):
    host = PMX(args.target)
    ref = PMX(args.reference) if args.reference else None
    hg = material_groups(host)
    rg = material_groups(ref) if ref else {}
    hname = [b['name'] for b in host.bones]
    rname = [b['name'] for b in ref.bones] if ref else []
    bone_of = {n: i for i, n in enumerate(hname)}

    old = np.unique(hg[HAIR_MAT])
    for mo in host.morphs:
        if mo['type'] in (1, 3, 4, 5, 6, 7):
            lost = [o for o in mo['offsets'] if o[0] in set(old.tolist())]
            if lost:
                raise SystemExit(f'hairgraft: morph {mo["name"]} drives {len(lost)} '
                                 'of the hair vertices being replaced')
    drop = set(old.tolist())
    keep_old = [i for i in range(len(host.vertices)) if i not in drop]
    old2new = {o: i for i, o in enumerate(keep_old)}
    verts = [host.vertices[i] for i in keep_old]
    textures = list(host.textures)

    if args.physics == 'follow':
        V, N, UV, tris = mod_hair()
        s, t, _ = placement(V, ref, rg.get(HAIR_MAT))
        V = V * s + t
        print(f'placement: scale {s:.4f} translate {np.round(t, 3)}')
        head = bone_of['頭']
        chains = {}
        for i, n in enumerate(hname):
            if n.startswith(HAIR_BONES):
                chains.setdefault(n.split('_')[0], []).append(
                    (i, np.array(host.bones[i]['pos'])))
        for c in chains.values():
            c.sort(key=lambda x: -x[1][1])
        used = np.unique(tris)
        wts = hairphys.bind_to_chains(V[used], list(chains.values()))
        bound = 0
        for w in wts:
            tot = sum(w.values())
            if tot > 0.01:
                bound += 1
            w[head] = w.get(head, 0.0) + max(0.0, 1.0 - tot)
        print(f'rig: {len(chains)} chains kept as they are '
              f'({", ".join(f"{k} x{len(v)}" for k, v in sorted(chains.items()))}); '
              f'{bound} of {len(used)} hair vertices bound to one, the rest on 頭')
        remap = {int(o): len(verts) + i for i, o in enumerate(used)}
        for k, o in enumerate(used):
            bs, ws = top4(wts[k])
            n = N[o] / (np.linalg.norm(N[o]) + 1e-9)
            verts.append((list(V[o]), list(n), list(UV[o]),
                          [[0.0] * 4] * host.uv_add, 2, bs, ws, 1.0))
        hair_faces = np.vectorize(remap.get)(tris)
        rel = 'textures\\' + HAIR_TEX
        tex_dir = os.path.join(os.path.dirname(args.target), 'textures')
        if not os.path.exists(os.path.join(tex_dir, HAIR_TEX)):
            Image.open(os.path.join(MOD, 'EvelynHairADiffuse.dds')).convert('RGB').save(
                os.path.join(tex_dir, HAIR_TEX))
        rigids, joints = host.rigids, host.joints
    elif args.physics == 'build':
        V, N, UV, tris = mod_hair()
        s, t, resid = placement(V, ref, rg.get(HAIR_MAT))
        V = V * s + t
        print(f'placement: scale {s:.4f} translate {np.round(t, 3)}'
              + (f' residual {resid:.4f}' if ref else ' (fitted constant)'))
        hair = zzzmod.load(MOD)['Hair']
        keys, W, cap, mass, chains = hairphys.recover(V, hair.bones, hair.weights)
        head = bone_of['頭']

        # one bone per link, at the point where it and its parent share the strand
        link_bone, link_pos, depth, parent_link = {}, {}, {}, {}
        order = []
        for ci, (start, parent) in enumerate(chains):
            kids = {}
            for k, v in parent.items():
                kids.setdefault(v, []).append(k)
            stack = [(start, 0)]
            while stack:
                i, d = stack.pop(0)
                p = parent[i]
                link_pos[i] = hairphys.joint_point(V, W, i, p if p is not None else cap)
                depth[i], parent_link[i] = d, p
                order.append((ci, i))
                stack += [(k, d + 1) for k in sorted(kids.get(i, []), key=lambda x: -mass[x])]
        for ci, i in order:
            link_bone[i] = len(host.bones)
            host.bones.append(dict(
                name=f'髪{ci}_{depth[i]}_{int(keys[i])}', name_en=f'hair{ci}_{depth[i]}',
                pos=list(link_pos[i]), parent=head if parent_link[i] is None else link_bone[parent_link[i]],
                layer=0, flag=hairphys.BONE_FLAG, tail_offset=[0.0, -0.2, 0.0]))
        for i in link_pos:
            kid = [k for k in link_pos if parent_link[k] == i]
            if kid:
                host.bones[link_bone[i]]['tail_offset'] = list(link_pos[kid[0]] - link_pos[i])

        used = np.unique(tris)
        col = {int(b): k for k, b in enumerate(keys)}
        remap = {int(o): len(verts) + i for i, o in enumerate(used)}
        for o in used:
            acc = {}
            for b, w in zip(hair.bones[o], hair.weights[o]):
                if w <= 0:
                    continue
                k = col.get(int(b))
                tgt = link_bone.get(k, head) if k is not None else head
                acc[tgt] = acc.get(tgt, 0.0) + float(w)
            items = sorted(acc.items(), key=lambda kv: -kv[1])[:4]
            tot = sum(x[1] for x in items)
            bs = [b for b, _ in items] + [0] * (4 - len(items))
            ws = [w / tot for _, w in items] + [0.0] * (4 - len(items))
            n = N[o] / (np.linalg.norm(N[o]) + 1e-9)
            verts.append((list(V[o]), list(n), list(UV[o]), [[0.0] * 4] * host.uv_add,
                          2, bs, ws, 1.0))
        hair_faces = np.vectorize(remap.get)(tris)

        # capsules along each segment, and a cone that opens toward the tip
        rigids = [r for r in host.rigids if not is_hair_body(host, r)]
        head_body = next(i for i, r in enumerate(rigids)
                         if host.bones[r['bone']]['name'] == '頭')
        index = {r['name']: i for i, r in enumerate(rigids)}
        joints = []
        for j in host.joints:
            a, b = host.rigids[j['a']], host.rigids[j['b']]
            if is_hair_body(host, a) or is_hair_body(host, b):
                continue
            jj = dict(j)
            jj['a'], jj['b'] = index[a['name']], index[b['name']]
            joints.append(jj)

        maxdepth = {ci: max(depth[i] for c2, i in order if c2 == ci) for ci, _ in order}
        body_of = {}
        for ci, i in order:
            kid = [k for k in link_pos if parent_link[k] == i]
            if kid:
                tip = link_pos[kid[0]]
            else:
                back = link_pos[i] - link_pos[parent_link[i]] if parent_link[i] is not None \
                    else np.array([0.0, -0.25, 0.0])
                tip = link_pos[i] + back
            seg = tip - link_pos[i]
            length = float(np.linalg.norm(seg)) or 0.2
            w = W[:, i]
            near = V[w > 0.2]
            radius = float(np.clip(np.median(np.linalg.norm(near - link_pos[i], axis=1)) * 0.5,
                                   *hairphys.RADIUS)) if len(near) > 4 else 0.09
            frac = depth[i] / max(maxdepth[ci], 1)
            body_of[i] = len(rigids)
            rigids.append(dict(
                name=host.bones[link_bone[i]]['name'], name_en=host.bones[link_bone[i]]['name_en'],
                bone=link_bone[i], group=hairphys.GROUP, nocollide=hairphys.NOCOLLIDE,
                shape=2, size=[radius, max(length - radius, 0.05), 0.104],
                pos=list(link_pos[i] + seg * 0.5), rot=hairphys.capsule_rot(seg),
                mass=float(args.root_mass + (hairphys.TIP_MASS - args.root_mass) * frac),
                lin_damp=hairphys.DAMPING, ang_damp=hairphys.DAMPING,
                restitution=0.0, friction=0.0, mode=1))
            cone = np.deg2rad(args.cone) * frac
            joints.append(dict(
                name=host.bones[link_bone[i]]['name'], name_en=host.bones[link_bone[i]]['name_en'],
                type=0, a=head_body if parent_link[i] is None else body_of[parent_link[i]],
                b=body_of[i], pos=list(link_pos[i]), rot=hairphys.capsule_rot(seg),
                lin_min=[0.0] * 3, lin_max=[0.0] * 3,
                ang_min=[-cone, 0.0, -cone], ang_max=[cone, 0.0, cone],
                spring_lin=[0.0] * 3, spring_ang=[0.0] * 3))
        # give the skull something the hair can actually land on, sized so no strand
        # begins inside it
        caps = []
        for r in rigids[len(rigids) - len(order):]:
            R = hairphys.capsule_rot(np.array([0.0, 1.0, 0.0]))
            caps.append((np.array(r['pos']), _axis(r['rot']), r['size'][1] / 2, r['size'][0]))
        lifted, shrunk = 0, []
        for r in rigids:
            if 0 <= r['bone'] < len(host.bones) and host.bones[r['bone']]['name'] == '頭' \
                    and r['mode'] == 0 and r['nocollide'] == 0xFFFF:
                r['nocollide'] = hairphys.HEAD_COLLIDER_MASK
                new = hairphys.fit_collider(np.array(r['pos']), r['size'][0], caps)
                if new < r['size'][0] - 1e-6:
                    shrunk.append((r['size'][0], new))
                    r['size'] = [new] + list(r['size'][1:])
                lifted += 1
        note = '; ' + ', '.join(f'{a:.2f}->{b:.2f}' for a, b in shrunk) if shrunk else ''
        print(f'  {lifted} head spheres now collide with the hair{note}')

        rel = 'textures\\' + HAIR_TEX
        tex_dir = os.path.join(os.path.dirname(args.target), 'textures')
        if not os.path.exists(os.path.join(tex_dir, HAIR_TEX)):
            Image.open(os.path.join(MOD, 'EvelynHairADiffuse.dds')).convert('RGB').save(
                os.path.join(tex_dir, HAIR_TEX))
        print(f'rig: {len(chains)} strands, {len(order)} links; '
              f'{len(order)} capsules and joints authored, {len(rigids) - len(order)} bodies kept')
    elif args.physics == 'head':
        V, N, UV, tris = mod_hair()
        s, t, resid = placement(V, ref, rg.get(HAIR_MAT))
        V = V * s + t
        print(f'placement: scale {s:.4f} translate {np.round(t, 3)}')
        head = bone_of['頭']
        used = np.unique(tris)
        remap = {int(o): len(verts) + i for i, o in enumerate(used)}
        for o in used:
            n = N[o] / (np.linalg.norm(N[o]) + 1e-9)
            verts.append((list(V[o]), list(n), list(UV[o]), [[0.0] * 4] * host.uv_add,
                          0, [head], [1.0], 1.0))
        hair_faces = np.vectorize(remap.get)(tris)
        tex_dir = os.path.join(os.path.dirname(args.target), 'textures')
        if not os.path.exists(os.path.join(tex_dir, HAIR_TEX)):
            Image.open(os.path.join(MOD, 'EvelynHairADiffuse.dds')).convert('RGB').save(
                os.path.join(tex_dir, HAIR_TEX))
        rel = 'textures\\' + HAIR_TEX
        rigids = [r for r in host.rigids if not is_hair_body(host, r)]
        index = {r['name']: i for i, r in enumerate(rigids)}
        joints = []
        for j in host.joints:
            a, b = host.rigids[j['a']], host.rigids[j['b']]
            if is_hair_body(host, a) or is_hair_body(host, b):
                continue
            jj = dict(j)
            jj['a'], jj['b'] = index[a['name']], index[b['name']]
            joints.append(jj)
        print(f'rig: {len(used)} hair vertices on 頭 at weight 1; dropped '
              f'{len(host.rigids) - len(rigids)} sway rigid bodies and '
              f'{len(host.joints) - len(joints)} joints, kept {len(rigids)} and {len(joints)}')
    else:
        if ref is None:
            raise SystemExit('hairgraft: --physics copy needs --reference')
        moved = 0
        for i, n in enumerate(rname):
            if not n.startswith(HAIR_BONES):
                continue
            j = bone_of[n]
            moved += not np.allclose(host.bones[j]['pos'], ref.bones[i]['pos'])
            host.bones[j]['pos'] = list(ref.bones[i]['pos'])
        rigids = [r for r in host.rigids if not is_hair_body(host, r)]
        for r in ref.rigids:
            if is_hair_body(ref, r):
                rr = dict(r)
                rr['bone'] = bone_of[ref.bones[r['bone']]['name']]
                rigids.append(rr)
        index = {r['name']: i for i, r in enumerate(rigids)}
        joints = []
        for j in host.joints:
            a, b = host.rigids[j['a']], host.rigids[j['b']]
            if is_hair_body(host, a) or is_hair_body(host, b):
                continue
            jj = dict(j)
            jj['a'], jj['b'] = index[a['name']], index[b['name']]
            joints.append(jj)
        carried = 0
        for j in ref.joints:
            a, b = ref.rigids[j['a']], ref.rigids[j['b']]
            if not (is_hair_body(ref, a) or is_hair_body(ref, b)):
                continue
            jj = dict(j)
            jj['a'], jj['b'] = index[a['name']], index[b['name']]
            joints.append(jj)
            carried += 1
        used = np.unique(rg[HAIR_MAT])
        remap = {int(o): len(verts) + i for i, o in enumerate(used)}
        for o in used:
            v = list(ref.vertices[int(o)])
            v[5] = [bone_of[rname[b]] for b in v[5]]
            verts.append(tuple(v))
        hair_faces = np.vectorize(remap.get)(rg[HAIR_MAT])
        rel = ref.textures[ref.materials[[m['name'] for m in ref.materials].index(HAIR_MAT)]['tex']]
        print(f'rig: {moved} sway bones repositioned, {len(rigids) - len([r for r in host.rigids if not is_hair_body(host, r)])} '
              f'rigid bodies and {carried} joints taken from the reference')
    if rel not in textures:
        textures.append(rel)

    materials, faces = [], []
    for m in host.materials:
        mm = dict(m)
        if m['name'] == HAIR_MAT:
            f = hair_faces
            mm['tex'] = textures.index(rel)
        else:
            f = np.vectorize(old2new.get)(hg[m['name']])
        mm['fcount'] = len(f) * 3
        faces.append(f)
        materials.append(mm)

    presentation.apply(_MatHolder(materials))
    out = Grafted(host, verts, np.concatenate(faces).ravel().tolist(), materials,
                  textures, old2new, rigids, joints)
    n = write_pmx(args.out, out)
    print(f'wrote {args.out}  {n/1e6:.2f} MB  {len(verts)} verts  {len(out.faces)//3} tris  '
          f'{len(out.bones)} bones  {len(out.morphs)} morphs  {len(rigids)} rigid bodies  '
          f'{len(joints)} joints')


def top4(d):
    items = sorted(d.items(), key=lambda kv: -kv[1])[:4]
    tot = sum(w for _, w in items)
    if tot <= 0:
        return [0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]
    bones = [b for b, _ in items] + [0] * (4 - len(items))
    ws = [w / tot for _, w in items] + [0.0] * (4 - len(items))
    return bones, ws


class Grafted:
    def __init__(self, host, verts, faces, materials, textures, old2new, rigids, joints):
        self.version = 2.0
        self.uv_add = host.uv_add
        self.name = host.name
        self.name_en = host.name_en
        self.comment = host.comment
        self.comment_en = host.comment_en
        self.vertices = verts
        self.faces = faces
        self.materials = materials
        self.textures = textures
        self.bones = host.bones
        self.morphs = [self._morph(m, old2new) for m in host.morphs]
        self.frames = host.frames
        self.rigids = rigids
        self.joints = joints

    @staticmethod
    def _morph(mo, old2new):
        m = dict(mo)
        if m['type'] in (1, 3, 4, 5, 6, 7):
            m['offsets'] = [(old2new[o[0]],) + tuple(o[1:]) for o in m['offsets']
                            if o[0] in old2new]
        return m


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='public/models/reze/reze-base.pmx')
    ap.add_argument('--reference', default=None,
                    help='refit the placement against this build; required by --physics copy')
    ap.add_argument('--physics', choices=('follow', 'build', 'head', 'copy'),
                    default='follow')
    ap.add_argument('--cone', type=float, default=12.0,
                    help='degrees of bend at the tip of a strand; 0 at the scalp')
    ap.add_argument('--root-mass', type=float, default=8.0,
                    help='mass at the scalp, tapering to 1 at the tip')
    ap.add_argument('--out', default='public/models/reze/_hair.pmx',
                    help='an intermediate; the leading underscore keeps it out of git')
    build(ap.parse_args())
