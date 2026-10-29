# -*- coding: utf-8 -*-
"""Recover the mod's hair rig and author MMD sway physics for it.

The mod skins its haircut to 24 bones but ships no skeleton, so the chains have to
be read back out of the weights. Two bones that blend across the same vertices are
neighbours on a strand, which makes shared weight the edge measure — but the cap
bone at the scalp shares weight with every strand, so including it collapses the
whole thing into a star. Dropping it first leaves the strands as separate
components, and a maximum spanning tree inside each one, rooted at its highest
link, recovers the order along the strand and any branch off it.

Bodies and joints are authored to the same recipe as the model's own hair chains:
capsules in group 9, mass tapering to 1 at the tip, damping 0.99, and a cone that
opens from nothing at the scalp to a dozen degrees at the end.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Copied from the model's own yb_/zb_/e_ chains rather than invented, with two
# corrections those chains got away with because they were short and far apart.
GROUP = 9
# 0xFDE0 there, which leaves bit 9 clear: the strands collide with each other. Twenty
# capsules on one head, several of them branches a few centimetres apart, then spend
# the frame shoving their own neighbours around and fighting the joints holding them.
# Setting bit 9 keeps the body collisions (groups 0-4) and drops the self-collision.
NOCOLLIDE = 0xFDE0 | 0x0200
# The three spheres on 頭 carry nocollide 0xFFFF -- they collide with nothing at all,
# so there is no skull for the hair to rest on and it sinks straight through. Clearing
# bit 9 makes them collide with the hair group and nothing else, which is all a hair
# collider is for and leaves every other body's behaviour alone.
HEAD_COLLIDER_MASK = 0xFFFF & ~0x0200
BONE_FLAG = 0x001E
ROOT_MASS = 8.0
TIP_MASS = 1.0
TIP_CONE = np.deg2rad(12.0)
DAMPING = 0.99
# The originals run 0.083 to 0.156. Fatter capsules than the strand they stand for get
# pushed further off the skull than the hair actually is.
RADIUS = (0.05, 0.13)


def _weight_matrix(bones, weights, n_vert):
    keys = np.unique(bones)
    col = {int(b): k for k, b in enumerate(keys)}
    W = np.zeros((n_vert, len(keys)))
    for c in range(bones.shape[1]):
        for b in keys:
            m = bones[:, c] == b
            W[m, col[int(b)]] += weights[m, c]
    return keys, W


def recover(pos, bones, weights, min_mass=3.0, percentile=35):
    """Return [(chain root, {link: parent}), ...] plus the cap bone, from the weights."""
    keys, W = _weight_matrix(bones, weights, len(pos))
    mass = W.sum(0)
    cap = int(np.argmax(mass))
    n = len(keys)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            C[i, j] = C[j, i] = np.minimum(W[:, i], W[:, j]).sum()

    live = [i for i in range(n) if i != cap and mass[i] > min_mass]
    sub = C[np.ix_(live, live)]
    thr = np.percentile(sub[sub > 0], percentile)
    ncomp, lab = connected_components(coo_matrix((sub > thr).astype(float)), directed=False)

    def centre(i):
        w = W[:, i]
        return (pos * w[:, None]).sum(0) / w.sum()

    chains = []
    for c in range(ncomp):
        idx = [live[k] for k in range(len(live)) if lab[k] == c]
        start = max(idx, key=lambda i: centre(i)[1])
        inn, parent = [start], {start: None}
        while len(inn) < len(idx):
            best = (-1.0, None, None)
            for a in inn:
                for b in idx:
                    if b in parent:
                        continue
                    if C[a, b] > best[0]:
                        best = (C[a, b], a, b)
            if best[2] is None:
                break
            parent[best[2]] = best[1]
            inn.append(best[2])
        chains.append((start, parent))
    return keys, W, cap, mass, chains


def joint_point(pos, W, child, parent):
    """Where two links blend is where the joint between them is."""
    w = W[:, child] * W[:, parent]
    if w.sum() < 1e-6:
        w = W[:, child]
    return (pos * w[:, None]).sum(0) / w.sum()


def capsule_rot(direction):
    """Euler for a capsule whose local +Y runs along `direction`.

    The engine composes rigid-body rotation as Ry.Rx.Rz, so fixing rz at zero leaves
    a closed form: rx tilts +Y off vertical, ry swings it around.
    """
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    rx = float(np.arccos(np.clip(d[1], -1.0, 1.0)))
    ry = float(np.arctan2(d[0], d[2])) if abs(d[1]) < 0.999999 else 0.0
    return [rx, ry, 0.0]


def seg_point_distance(a, b, p):
    """Distance from p to the segment ab."""
    ab = b - a
    t = 0.0 if not np.dot(ab, ab) else np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return float(np.linalg.norm(a + t * ab - p))


def fit_collider(sphere_pos, radius, capsules, margin=0.01, floor=0.12):
    """Shrink a skull sphere until no strand starts inside it.

    A capsule that begins overlapping the collider is ejected on the first frame,
    which reads as the hair jumping off the head before it settles. The sphere only
    ever meets the hair, so trimming it to clear the rest pose costs nothing else.
    """
    worst = 0.0
    for pos, axis, half, r in capsules:
        d = seg_point_distance(pos - axis * half, pos + axis * half, sphere_pos)
        worst = max(worst, (radius + r) - d)
    if worst <= 0:
        return radius
    want = radius - worst - margin
    if want < floor:
        # The floor used to be 0.35, which is larger than the smallest skull sphere
        # here -- so it clamped, the sphere stayed over a strand, and that strand was
        # ejected on the first frame. Better a small collider than one a capsule is
        # born inside.
        want = floor
    return want


def bind_to_chains(P, chains, reach=0.62, soft=0.55):
    """Weight points onto existing sway chains, the way this model's own hair is bound.

    Each chain is a polyline through its links. A point takes the two links bracketing
    the closest point on that polyline, blended by where it falls between them, scaled
    by how near it is; the remainder stays on the head. Binding by distance to the
    CHAIN rather than by proximity to the hair being replaced is what matters here --
    the two haircuts are different shapes, and transferring vertex to vertex put the
    strand on the character's left onto the wrong chain.

    Returns a list of {bone: weight} with the head's share left for the caller to add.
    """
    out = [dict() for _ in P]
    best = np.full(len(P), np.inf)
    for links in chains:
        pos = np.array([p for _, p in links])
        for k in range(len(pos) - 1):
            a, b = pos[k], pos[k + 1]
            ab = b - a
            denom = float(ab @ ab) or 1.0
            t = np.clip((P - a) @ ab / denom, 0, 1)
            q = a + t[:, None] * ab
            d = np.linalg.norm(P - q, axis=1)
            take = d < np.minimum(best, reach)
            if not take.any():
                continue
            best = np.where(take, d, best)
            w = np.clip((reach - d) / max(reach * (1 - soft), 1e-6), 0, 1)
            w = w * w * (3 - 2 * w)                       # smoothstep, so no hard edge
            for i in np.nonzero(take)[0]:
                out[i] = {links[k][0]: float(w[i] * (1 - t[i])),
                          links[k + 1][0]: float(w[i] * t[i])}
    return out


def _capsule_axis(rot):
    rx, ry, rz = rot
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (Ry @ Rx @ Rz) @ np.array([0.0, 1.0, 0.0])


def fix_hair_collision(model, groups=('yb_', 'zb_', 'e_', '髪'), verbose=True):
    """Give the hair a skull to rest on, and stop it fighting itself.

    Two faults this model ships with. The three spheres on the head bone carry
    nocollide 0xFFFF, which is not "collides with the body" but "collides with
    nothing" -- there is no skull under the hair at all, so it sinks through the
    moment anything moves. And the strands collide with each other, which short
    chains far apart get away with and longer ones do not.

    Clearing bit 9 on the skull makes it meet the hair group and nothing else, which
    is all a hair collider is for. Setting bit 9 on the hair keeps the body collisions
    and drops the self-collision. Any sphere a strand already starts inside is then
    trimmed, because a body born inside a collider is ejected on the first frame.
    """
    names = [b['name'] for b in model.bones]

    def bone_of(r):
        return names[r['bone']] if 0 <= r['bone'] < len(names) else ''

    hair = [r for r in model.rigids if bone_of(r).startswith(groups)]
    if not hair:
        return 0
    group = hair[0]['group']
    changed = 0
    for r in hair:
        if not (r['nocollide'] >> group) & 1:
            r['nocollide'] |= 1 << group
            changed += 1
    if verbose and changed:
        print('  %d hair bodies no longer collide with each other' % changed)

    caps = [(np.array(r['pos']), _capsule_axis(r['rot']), r['size'][1] / 2, r['size'][0])
            for r in hair if r['shape'] == 2]
    # Refit unconditionally. Gating this on the mask still reading 0xFFFF meant a
    # sphere was sized once, against whatever capsules existed then, and never again --
    # so mirroring or re-authoring the chains could push a capsule inside a sphere that
    # had already been "fixed". That is what threw the fringe tip: e_3_1 sat 0.077 deep
    # in the skull and was ejected on the first frame.
    opened, trimmed = 0, []
    for r in model.rigids:
        if bone_of(r) != '頭' or r['mode'] != 0:
            continue
        if r['nocollide'] == 0xFFFF:
            r['nocollide'] = 0xFFFF & ~(1 << group)
            opened += 1
        if caps:
            new = fit_collider(np.array(r['pos']), r['size'][0], caps, margin=0.03)
            if new < r['size'][0] - 1e-6:
                trimmed.append((r['size'][0], new))
                r['size'] = [new] + list(r['size'][1:])
    if verbose and (opened or trimmed):
        note = ('trimmed ' + ', '.join('%.2f->%.2f' % t for t in trimmed)) if trimmed else ''
        head = ('%d skull spheres now collide with the hair; ' % opened) if opened else ''
        print('  %s%s' % (head, note))
    return changed + opened + len(trimmed)


def reflect_hair_rig(model, groups=('yb_', 'zb_', 'e_', '髪'), verbose=True):
    """Reflect the hair's chains, bodies and joints in x, to match a mirrored mesh.

    Rebinding a mirrored haircut onto the chains as they stand does not work, and the
    reason is that no chain here is its own mirror or another's: `e` leans from x +0.02
    at the scalp to -0.28 at the tip, so the flipped fringe ends up half a unit from the
    chain meant to carry it, and `yb` and `zb` are 0.69 apart on average with six links
    against five. Reflect the rig with the mesh instead and every strand keeps the chain
    it already had -- the weights need no remapping at all, because each bone moves to
    where its own vertices went.

    Positions flip outright. Of the three Euler angles only rz does: it is the tilt in
    the xy plane, the component that actually leans a capsule left or right, and a
    capsule that leans outward on one side of the head leans into it on the other --
    which is what buried the zb strand.

    ry is left alone, and that is a deliberate departure from the exact reflection. The
    exact conjugate is (rx, -ry, -rz), it matches a perfect mirror at 0.00e+00, and the
    solver does not survive it: negating ry rolls each body about its own axis, which
    relabels the local axes a joint reads its limits in, and the yb and e strands swing
    wrong. Keeping the roll and flipping only the tilt puts zb exactly right and leaves
    the other two nearer the mirror than doing nothing did -- 32 and 28 degrees off
    their mirrored axis against 58 and 68.

    A joint's limits here are symmetric on every axis, so nothing has to swap with them.
    """
    names = [b['name'] for b in model.bones]
    bones = {i for i, n in enumerate(names) if n.startswith(groups)}
    if not bones:
        return 0

    for i in bones:
        b = model.bones[i]
        b['pos'] = [-b['pos'][0], b['pos'][1], b['pos'][2]]
        for key in ('tail_offset', 'axis', 'local_x', 'local_z'):
            if key in b:
                b[key] = [-b[key][0], b[key][1], b[key][2]]

    bodies = set()
    for k, r in enumerate(model.rigids):
        if r['bone'] in bones:
            bodies.add(k)
            r['pos'] = [-r['pos'][0], r['pos'][1], r['pos'][2]]
            r['rot'] = [r['rot'][0], r['rot'][1], -r['rot'][2]]

    joints = 0
    for j in model.joints:
        if j['a'] not in bodies and j['b'] not in bodies:
            continue
        joints += 1
        j['pos'] = [-j['pos'][0], j['pos'][1], j['pos'][2]]
        j['rot'] = [j['rot'][0], j['rot'][1], -j['rot'][2]]
        lo, hi = j['lin_min'], j['lin_max']
        j['lin_min'] = [-hi[0], lo[1], lo[2]]
        j['lin_max'] = [-lo[0], hi[1], hi[2]]
        lo, hi = j['ang_min'], j['ang_max']
        j['ang_min'] = [lo[0], -hi[1], -hi[2]]
        j['ang_max'] = [hi[0], -lo[1], -lo[2]]
    if verbose:
        print('  reflected %d sway bones, %d rigid bodies, %d joints'
              % (len(bones), len(bodies), joints))
    return len(bones)


if __name__ == '__main__':
    import sys as _sys
    from pmx import PMX as _PMX
    from pmxw import write_pmx as _write
    for _p in _sys.argv[1:]:
        _m = _PMX(_p)
        print(_p.split('/')[-1] + ':')
        if fix_hair_collision(_m):
            _write(_p, _m)
        else:
            print('  nothing to fix')


def author_hair_frames(model, groups=('yb_', 'zb_', 'e_', '髪'), verbose=True):
    """Lay every hair capsule along the segment its bone spans, and give each joint
    the frame of the body it drives.

    This exists because reflecting the frames does not survive the solver. Conjugating
    a rotation, R -> M.R.M, is exact -- positions, frames and limits all match a
    perfect mirror at 0.00e+00 -- but it flips each body's local x axis, and the
    strands then swing wrong. Leaving the rotations alone behaves better and is
    plainly incorrect: twelve of fifteen capsules end up pointing off their own
    strand, one of them by 68 degrees.

    Authoring instead gives frames built the same way for both sides -- rz pinned at
    zero, +Y laid along the segment -- with no conjugation and no flipped axis
    anywhere.
    """
    names = [b['name'] for b in model.bones]
    hair = {i for i, n in enumerate(names) if n.startswith(groups)}
    if not hair:
        return 0
    child = {}
    for i, b in enumerate(model.bones):
        if i in hair and b['parent'] in hair:
            child[b['parent']] = i

    frame = {}
    for i in hair:
        b = model.bones[i]
        if i in child:
            d = np.array(model.bones[child[i]]['pos']) - np.array(b['pos'])
        elif 'tail_offset' in b:
            d = np.array(b['tail_offset'], float)
        else:
            d = np.array([0.0, -0.2, 0.0])
        if np.linalg.norm(d) < 1e-6:
            d = np.array([0.0, -0.2, 0.0])
        frame[i] = capsule_rot(d)

    body_frame, n_bodies = {}, 0
    for k, r in enumerate(model.rigids):
        if r['bone'] in hair:
            r['rot'] = list(frame[r['bone']])
            body_frame[k] = frame[r['bone']]
            n_bodies += 1
    n_joints = 0
    for j in model.joints:
        f = body_frame.get(j['b'], body_frame.get(j['a']))
        if f is not None:
            j['rot'] = list(f)
            n_joints += 1
    if verbose:
        print('  authored %d capsule frames and %d joint frames along the bones'
              % (n_bodies, n_joints))
    return n_bodies
