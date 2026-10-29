# -*- coding: utf-8 -*-
"""Take out of a model the bones, rigid bodies and joints nothing uses any more.

After a head swap the host's hair is gone but its hair rig is not: chains of bones
with nothing weighted to them, each carrying a dynamic body, each body joined to the
next and the first to the skull. They cost physics every frame and clutter every bone
list. A bone is kept if any vertex is weighted to it, or if a kept bone needs it -- as
parent, tail, grant source, IK target or IK link -- or if it is an IK bone whose target
is kept. Rigid bodies on a removed bone go, and the joints on those bodies with them;
bodies bound to no bone (the 深空之眼 breast anchors) stay. Everything that indexes a
bone or a body is renumbered.

    python3 tools/zzz2pmx/prune.py <model.pmx>
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pmx import PMX
from pmxw import write_pmx


# The standard MMD skeleton, kept by name whatever is weighted to it: a motion drives
# these, and a model that has lost one cannot be animated. 陵光 weights its toes to
# 足先EX, so つま先 had no weights, and the toe IK went with it; a hair vertex carried a
# sliver of センター, and the root of all motion went with the hair.
KEEP = {'全ての親', 'センター', 'グルーブ', '腰', '操作中心', '左ダミー', '右ダミー', '両目',
        '上半身', '上半身2', '下半身', '首', '頭', '左目', '右目',
        '左肩P', '左肩', '左肩C', '右肩P', '右肩', '右肩C',
        '左腕', '右腕', '左腕捩', '右腕捩', '左ひじ', '右ひじ', '左手捩', '右手捩', '左手首', '右手首',
        '左足', '右足', '左ひざ', '右ひざ', '左足首', '右足首', '左つま先', '右つま先',
        '左足D', '右足D', '左ひざD', '右ひざD', '左足首D', '右足首D', '左足先EX', '右足先EX',
        '左足ＩＫ', '右足ＩＫ', '左つま先ＩＫ', '右つま先ＩＫ', '左足IK親', '右足IK親',
        '腰キャンセル左', '腰キャンセル右', '左胸', '右胸', '左胸先', '右胸先'}
for _s in ('左', '右'):
    for _f in ('親指', '人指', '中指', '薬指', '小指'):
        for _k in ('０', '１', '２', '３'):
            KEEP.add(_s + _f + _k)


def prune(m, force=()):
    """`force`: names of bones to remove whatever depends on them -- the bones a
    dropped material alone used. Their children go to their parent, a kept bone whose
    display tail pointed at one keeps the direction as an offset instead, and their
    bodies and joints go with them."""
    nb = len(m.bones)
    force = {i for i, b in enumerate(m.bones) if b['name'] in force and b['name'] not in KEEP}
    if force:
        # never a bone something live hangs from: the weapon's chain had no live
        # descendants, センター has the whole skeleton
        weighted = set()
        for v in m.vertices:
            for b, w in zip(v[5], v[6]):
                if w > 1e-6:
                    weighted.add(int(b))
        def ancestors(i):
            out = set()
            while 0 <= i < nb:
                i = m.bones[i]['parent']
                out.add(i)
            return out
        hangs = set()
        for b in weighted:
            if b not in force:
                hangs |= ancestors(b)
        force -= hangs
    if force:
        pos = {i: np.array(b['pos']) for i, b in enumerate(m.bones)}
        def up(i):
            while i in force:
                i = m.bones[i]['parent']
            return i
        for i, b in enumerate(m.bones):
            if i in force:
                continue
            if b['parent'] in force:
                b['parent'] = up(b['parent'])
            if (b['flag'] & 0x0001) and b.get('tail', -1) in force:
                b['flag'] &= ~0x0001
                b['tail_offset'] = [float(x) for x in pos[b['tail']] - pos[i]]
                del b['tail']
            if 'grant' in b and b['grant'][0] in force:
                b['grant'] = (up(b['grant'][0]), b['grant'][1])
            if 'ik_target' in b:
                if b['ik_target'] in force:
                    b['ik_target'] = up(b['ik_target'])
                b['ik_links'] = [(j, lim) for j, lim in b['ik_links'] if j not in force]
        for n, v in enumerate(m.vertices):
            if not any(b in force for b in v[5]):
                continue
            v = list(v)
            bs = list(v[5])
            ws = list(v[6])
            # what a removed bone held goes to the nearest kept ancestor
            for k, b in enumerate(bs):
                if b in force:
                    bs[k] = up(b)
            merged = {}
            for b, w in zip(bs, ws):
                if w > 0:
                    merged[b] = merged.get(b, 0.0) + float(w)
            top = sorted(merged.items(), key=lambda x: -x[1])[:4] or [(0, 1.0)]
            tot = sum(w for _, w in top) or 1.0
            v[4] = 2
            v[5] = [b for b, _ in top] + [0] * (4 - len(top))
            v[6] = [w / tot for _, w in top] + [0.0] * (4 - len(top))
            m.vertices[n] = tuple(v[:8])
    live = np.array([b['name'] in KEEP and i not in force for i, b in enumerate(m.bones)])
    for v in m.vertices:
        for b, w in zip(v[5], v[6]):
            if w > 1e-6 and 0 <= b < nb and b not in force:
                live[b] = True
    # grow: what a live bone depends on, and IK bones over a live target
    changed = True
    while changed:
        changed = False
        for i, b in enumerate(m.bones):
            need = []
            if live[i]:
                need.append(b['parent'])
                if b['flag'] & 0x0001:
                    need.append(b.get('tail', -1))
                if 'grant' in b:
                    need.append(b['grant'][0])
                if 'ik_target' in b:
                    need.append(b['ik_target'])
                    need += [j for j, _ in b.get('ik_links', ())]
            elif 'ik_target' in b and 0 <= b['ik_target'] < nb and live[b['ik_target']] and i not in force:
                live[i] = True
                changed = True
            for j in need:
                if 0 <= j < nb and not live[j] and j not in force:
                    live[j] = True
                    changed = True
    # Anchors. A breast chain hangs from a STATIC body on a bone nothing is weighted
    # to (左胸変形); pruned as unused, the chain fell to the floor. A bone carrying a
    # static body joined to a body that stays is a bone that stays -- to a fixed point,
    # since an anchor can hang off an anchor. Dynamic bodies on dead bones still go.
    rb_live = lambda r: r['bone'] < 0 or (0 <= r['bone'] < nb and live[r['bone']])
    changed = True
    while changed:
        changed = False
        for j in m.joints:
            for me, other in ((j['a'], j['b']), (j['b'], j['a'])):
                if not (0 <= me < len(m.rigids) and 0 <= other < len(m.rigids)):
                    continue
                r = m.rigids[me]
                if r['mode'] == 0 and r['bone'] >= 0 and not live[r['bone']] \
                        and r['bone'] not in force and rb_live(m.rigids[other]):
                    live[r['bone']] = True
                    changed = True
        if changed:
            for i, b in enumerate(m.bones):        # and what those bones depend on
                if live[i]:
                    for jj in ([b['parent']] + ([b['grant'][0]] if 'grant' in b else [])):
                        if 0 <= jj < nb and not live[jj] and jj not in force:
                            live[jj] = True
    keep = np.where(live)[0]
    bmap = {int(o): n for n, o in enumerate(keep)}
    rb = lambda j: bmap.get(int(j), -1) if j is not None else -1

    bones = []
    for i in keep:
        b = dict(m.bones[i])
        b['parent'] = rb(b['parent'])
        if b['flag'] & 0x0001:
            b['tail'] = rb(b.get('tail', -1))
        if 'grant' in b:
            b['grant'] = (rb(b['grant'][0]), b['grant'][1])
        if 'ik_target' in b:
            b['ik_target'] = rb(b['ik_target'])
            b['ik_links'] = [(rb(j), lim) for j, lim in b['ik_links'] if rb(j) >= 0]
        bones.append(b)
    m.bones = bones
    for k, v in enumerate(m.vertices):
        v = list(v)
        v[5] = [rb(b) if rb(b) >= 0 else 0 for b in v[5]]
        m.vertices[k] = tuple(v)
    for mo in m.morphs:
        if mo['type'] == 2:
            mo['offsets'] = [(rb(o[0]),) + tuple(o[1:]) for o in mo['offsets'] if rb(o[0]) >= 0]
    for f in m.frames:
        f['items'] = [(k, rb(i) if k == 0 else i) for k, i in f['items']
                      if k != 0 or rb(i) >= 0]
    rkeep = [i for i, r in enumerate(m.rigids) if r['bone'] < 0 or rb(r['bone']) >= 0]
    rmap = {o: n for n, o in enumerate(rkeep)}
    rigids = []
    for i in rkeep:
        r = dict(m.rigids[i])
        r['bone'] = rb(r['bone']) if r['bone'] >= 0 else -1
        rigids.append(r)
    joints = []
    for j in m.joints:
        a = rmap.get(j['a'], -1) if j['a'] >= 0 else -1
        b = rmap.get(j['b'], -1) if j['b'] >= 0 else -1
        if (j['a'] >= 0 and a < 0) or (j['b'] >= 0 and b < 0):
            continue
        j = dict(j)
        j['a'], j['b'] = a, b
        joints.append(j)
    gone = (nb - len(bones), len(m.rigids) - len(rigids), len(m.joints) - len(joints))
    m.rigids, m.joints = rigids, joints
    return gone


def main(path, force=()):
    m = PMX(path)
    b, r, j = prune(m, force)
    write_pmx(path, m)
    print(f'  pruned {b} bones, {r} rigid bodies, {j} joints nothing used')


if __name__ == '__main__':
    main(sys.argv[1])
