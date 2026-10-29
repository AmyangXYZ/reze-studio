# -*- coding: utf-8 -*-
"""reze's head on another 深空之眼 body, in one command.

The series shares one body frame -- 頭 within 0.05, 首 0.12, the wrist 0.02 between
塞勒涅 and 塞赫麦特 -- so an outfit is worn by swapping the head, not by dressing.
The host is left exactly as shipped except for its head:

  1. the frame is checked against 塞勒涅 and a drift over 0.3 at any landmark is
     reported before anything else runs;
  2. the host's face, mouth, eyes, brows, lashes and hair come off, found by name;
  3. any host chain bones sharing reze's hair-chain names (`e_*`, `yb_*`, `zb_*`) are
     renamed first, or the graft finds the names, carries nothing, and her hair hangs
     on the host's ornaments;
  4. `headgraft` puts reze's head on;
  5. whatever the host still wears above the head -- a cap, a hat -- is headwear, and
     every strand of hair outside it is pushed back under it (parity against the
     shell, so hair hanging under the brim is left);
  6. `prune`, `fixrig`, `publish`.

    python3 tools/zzz2pmx/headswap.py <host.pmx> <out-dir> [--drop a,b] [--rename a=b,c=d]

Reads the host and never writes it.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixrig
import holefill
import maidselene
import progress
import prune
import publish
from pmx import PMX
from pmxw import write_pmx
from surface import Mesh

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME = 'public/models/塞勒涅/塞勒涅.pmx'
LANDMARKS = ['頭', '首', '上半身', '左肩', '左手首', '右手首', '左足首', '右足首']
# 塞勒涅's landmarks, for when her folder is not on disk
FRAME_BONES = {'上半身': [0.0, 14.4326, -0.8837], '首': [0.0, 18.5235, -0.0884], '頭': [0.0, 19.526, -0.3128],
               '右手首': [-6.2836, 14.2371, -0.2748], '左肩': [0.2349, 18.1605, -0.1633],
               '左手首': [6.2836, 14.2371, -0.2748], '右足首': [-0.9028, 1.3765, 0.6393],
               '左足首': [0.9028, 1.3765, 0.6393]}
DRIFT = 0.3
HEAD = re.compile(r'face|脸|顔|唇|齿|歯|牙|舌|口腔|eyebrow|眉|eyelash|睫|目白|瞳|眼白|眼睛|表情|eye'
                  r'|hair|髪|头发|后发', re.I)
CHAIN = re.compile(r'^(e|yb|zb|hf)_\d+_\d+$')
# Where reze's face sits over the top of the neck, read off reze-selene (her own
# family's body): face centre minus the centre of the neck opening
SEAT = np.array([0.0, 0.2613, -0.6338])
UNDER = 0.03
BLUR = 6


def neck_opening(m):
    """The open top of the host's neck: the highest wide loop on the centre line
    between 首 and a unit above 頭, on any material the graft keeps."""
    hb = {b['name']: b['pos'] for b in m.bones}
    V = np.array([v[0] for v in m.vertices])
    best = None
    for name, T in holefill.slices(m).items():
        if HEAD.search(name) or not len(T):
            continue
        for L in holefill.boundary_loops(V, np.array(T)):
            c = V[L].mean(0)
            if (abs(c[0]) < 0.05 and hb['首'][1] < c[1] < hb['頭'][1] + 1.0 and np.ptp(V[L][:, 0]) > 1.0
                    and (best is None or c[1] > best[1])):
                best = c
    return best


def seat(m):
    """reze's head seated on the host's neck as it sits on 塞勒涅's, for a host whose
    face is another size and would carry the head off the neck in a face fit."""
    head = PMX(maidselene.HEAD)
    face = np.array([v[0] for v in head.vertices])[np.unique(holefill.tris_of(head, ['face01']))].mean(0)
    ring = neck_opening(m)
    off = ring + SEAT - face
    print(f'  neck opening at {np.round(ring, 3)}: head offset {np.round(off, 3)}')
    return off


def frame_check(host):
    h = PMX(host)
    hb = {b['name']: np.array(b['pos']) for b in h.bones}
    if os.path.exists(FRAME):
        fb = {b['name']: np.array(b['pos']) for b in PMX(FRAME).bones}
    else:
        fb = {k: np.array(v) for k, v in FRAME_BONES.items()}
    worst = []
    for n in LANDMARKS:
        if n in hb and n in fb:
            d = float(np.linalg.norm(hb[n] - fb[n]))
            worst.append((d, n))
    worst.sort(reverse=True)
    print('  frame against 塞勒涅: ' + ', '.join(f'{n} {d:.2f}' for d, n in worst[:4]))
    if worst and worst[0][0] > DRIFT:
        print(f'  !! {worst[0][1]} sits {worst[0][0]:.2f} off -- this may not be the same body')


def crossings(P, C, V, T, chunk=64):
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1, e2 = b - a, c - a
    out = np.zeros(len(P), int)
    for k in range(0, len(P), chunk):
        p = P[k:k + chunk]
        d = C[None, :] - p
        L = np.linalg.norm(d, axis=1, keepdims=True)
        d = d / np.maximum(L, 1e-9)
        h = np.cross(d[:, None, :], e2[None, :, :])
        det = np.einsum('tj,ptj->pt', e1, h)
        ok = np.abs(det) > 1e-9
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s_ = p[:, None, :] - a[None, :, :]
        u = np.einsum('ptj,ptj->pt', s_, h) * inv
        q = np.cross(s_, e1[None, :, :])
        v = np.einsum('pj,ptj->pt', d, q) * inv
        t = np.einsum('tj,ptj->pt', e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4) & (t < L)
        out[k:k + chunk] = hit.sum(1)
    return out


def underwear(path, hair, shell, under=UNDER, rounds=BLUR):
    """Every strand outside the shell goes back under it; the rest is left."""
    m = PMX(path)
    sl = holefill.slices(m)
    V = np.array([v[0] for v in m.vertices])
    T = np.array(sl[shell])
    cap = Mesh(V, T)
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    axis = n.sum(0)
    axis /= np.linalg.norm(axis) + 1e-9
    if axis[1] < 0:
        axis = -axis
    loops = holefill.boundary_loops(V, T)
    inside = (V[np.concatenate(loops)].mean(0) if loops else V[np.unique(T)].mean(0)) + axis * 0.25
    ids = np.unique(sl[hair])
    out = crossings(V[ids], inside, V, T) % 2 == 1
    if not out.any():
        print(f'  hair: nothing outside `{shell}`')
        return
    if out.mean() > 0.4:
        print(f'  hair: `{shell}` is no shell over the head ({100 * out.mean():.0f}% of the hair '
              f'reads outside it) -- left alone')
        return
    d, t, bar = cap.closest(V[ids])
    f = cap.F[t]
    q = V[f[:, 0]] * bar[:, :1] + V[f[:, 1]] * bar[:, 1:2] + V[f[:, 2]] * bar[:, 2:3]
    proud = np.einsum('ij,ij->i', V[ids] - q, cap.normal[t])
    need = np.where(out, np.abs(proud) + under, 0.0)
    nbr = {}
    for x, y, z in sl[hair]:
        for u, v in ((x, y), (y, z), (z, x)):
            nbr.setdefault(int(u), set()).add(int(v))
            nbr.setdefault(int(v), set()).add(int(u))
    at = {int(v): k for k, v in enumerate(ids)}
    order = [np.array([at[j] for j in nbr.get(int(v), ()) if j in at], int) for v in ids]
    full = need.copy()
    for _ in range(rounds):
        nxt = need.copy()
        for k, ns in enumerate(order):
            if len(ns):
                nxt[k] = 0.5 * need[k] + 0.5 * need[ns].mean()
        need = nxt
    need = np.maximum(need, full)
    moved = 0
    for k, i in enumerate(ids):
        if need[k] > 1e-3:
            v = list(m.vertices[int(i)])
            v[0] = [float(x) for x in V[int(i)] - cap.normal[t[k]] * need[k]]
            m.vertices[int(i)] = tuple(v)
            moved += 1
    write_pmx(path, m)
    W = np.array([v[0] for v in m.vertices])
    still = crossings(W[ids], inside, V, T) % 2 == 1
    print(f'  hair: {int(out.sum())} of {len(ids)} vertices outside `{shell}` '
          f'(worst {np.abs(proud[out]).max():+.3f}) -> {int(still.sum())}; {moved} moved')


def hatsplit(m, mat, asname, on=r'^頭$', hair=r'^hair'):
    """A veil or hood that shares the body's cloth material: its shells anchored on
    頭 become their own material `asname`, and shells riding the host's hair bones --
    ornaments on hair we remove -- are dropped from the material altogether. A hat
    with bones of its own is found by those instead: `on` and `hair` match the bone
    names a shell is mostly weighted to."""
    import roundring as rr
    names = [b['name'] for b in m.bones]
    V = np.array([v[0] for v in m.vertices])
    F = np.array(m.faces).reshape(-1, 3)
    at, blocks = 0, {}
    for x in m.materials:
        n = x['fcount'] // 3
        blocks[x['name']] = (at, n)
        at += n
    a, n = blocks[mat]
    T = F[a:a + n]
    on_hat, on_hair = set(), set()
    for ids in rr.shells(V, T).values():
        head = hairs = 0
        for i in ids:
            v = m.vertices[int(i)]
            top = names[v[5][int(np.argmax(v[6][:len(v[5])]))]]
            head += bool(re.search(on, top))
            hairs += bool(re.search(hair, top))
        if head / len(ids) >= 0.3:
            on_hat |= {int(i) for i in ids}
        elif hairs / len(ids) >= 0.5:
            on_hair |= {int(i) for i in ids}
    hat = np.array([all(int(i) in on_hat for i in t) for t in T])
    gone = np.array([any(int(i) in on_hair for i in t) for t in T])
    keep = T[~hat & ~gone]
    hatT = T[hat]
    x = next(x for x in m.materials if x['name'] == mat)
    x['fcount'] = len(keep) * 3
    new = dict(x)
    new['name'] = new['name_en'] = asname
    new['fcount'] = len(hatT) * 3
    m.materials.insert(m.materials.index(x) + 1, new)
    m.faces = np.concatenate([F[:a], keep, hatT, F[a + n:]]).ravel().tolist()
    print(f'  {mat}: {len(hatT)} triangles of headwear split off as `{asname}`, '
          f'{int(gone.sum())} on her hair bones dropped')


def main(host, out_dir, drop=None, rename=None, hat=None, wear=None, scale=None, trim=None, seated=False,
         head=None, take=None):
    name = os.path.basename(out_dir.rstrip('/'))
    work = os.path.join(out_dir, '_work')
    build = os.path.join(work, name + '.pmx')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.join(work, 'textures'), exist_ok=True)

    with progress.step('the host, as shipped'):
        m = PMX(host)
        if scale:
            # a smaller figure brought up to 塞勒涅's size before the check
            import pmxscale
            if callable(scale):
                scale(m)
            else:
                k, shift = scale
                pmxscale.scale(m, k, shift)
                print(f'  host scaled x{k} and moved by {shift}')
            write_pmx(build, m)
            frame_check(build)
        else:
            frame_check(host)
        src = os.path.dirname(host)
        for rel in m.textures:
            fp = os.path.join(src, rel.replace('\\', '/'))
            if os.path.exists(fp):
                shutil.copyfile(fp, os.path.join(work, 'textures', os.path.basename(fp)))
        m.textures = ['textures\\' + os.path.basename(t.replace('\\', '/')) for t in m.textures]
        n = 0
        for b in m.bones:
            if CHAIN.match(b['name']):
                b['name'], b['name_en'] = 'host_' + b['name'], 'host_' + b['name_en']
                n += 1
        if n:
            print(f'  {n} host chain bones renamed out of the way of reze\'s hair')
        for x in m.materials:
            if rename and x['name'] in rename:
                x['name'] = x['name_en'] = rename[x['name']]
        for mat, asname in (hat or {}).items():
            # a name, or (name, hat-bone pattern, hair-bone pattern)
            hatsplit(m, mat, *((asname,) if isinstance(asname, str) else asname))
        # The graft fits by the host's face and looks for it as `face01`; a host that
        # calls it 脸 gets it renamed, since it comes off anyway.
        faces = [x for x in m.materials if re.match(r'^(face|脸|顔)', x['name'], re.I)]
        if faces and not any(x['name'] == 'face01' for x in m.materials):
            sl0 = holefill.slices(m)
            face = max(faces, key=lambda x: len(sl0.get(x['name'], ())))
            old = face['name']
            face['name'] = face['name_en'] = 'face01'
            if drop:
                drop = ['face01' if d == old else d for d in drop]
            print(f'  the host\'s face `{old}` renamed `face01` for the graft')
        write_pmx(build, m)
        if drop is None:
            drop = [x['name'] for x in m.materials if HEAD.search(x['name'])]
        print(f'  head materials to drop: {", ".join(drop)}')
        # The bones only the dropped materials used -- a weapon's, a hair ornament's --
        # go with them, whatever hangs off them in the rig.
        sl = holefill.slices(m)
        usedby = {}
        for k in sl:
            for i in np.unique(sl[k]):
                for b, w in zip(m.vertices[int(i)][5], m.vertices[int(i)][6]):
                    if w > 1e-6:
                        usedby.setdefault(int(b), set()).add(k)
        force = [m.bones[b]['name'] for b, ks in usedby.items() if ks and ks <= set(drop)]
        if force:
            print(f'  bones only those used, to go with them: {len(force)}')

    with progress.step('head'):
        # Another model's head than reze's: a copy beside it, for its sheets, with its
        # face called face01 as the graft reads it
        source = maidselene.HEAD
        if head:
            hm = PMX(head)
            for x in hm.materials:
                if re.match(r'^(脸|顔|face)', x['name']):
                    x['name'] = x['name_en'] = 'face01'
            take = ['face01' if re.match(r'^(脸|顔|face)', t) else t for t in take]
            source = os.path.join(os.path.dirname(head), '_head.pmx')
            write_pmx(source, hm)
        before = {b['name'] for b in PMX(build).bones}
        subprocess.run([sys.executable, os.path.join(HERE, 'headgraft.py'),
                        '--host', build, '--head', source, '--out', build,
                        '--drop', ','.join(drop), '--scale', '1']
                       + (['--take', ','.join(take)] if take else [])
                       + (['--offset=' + ','.join(f'{x:.5f}' for x in seat(PMX(build)))] if seated else [])
                       + (['--trim-above', str(trim)] if trim is not None else []), check=True)
        if head:
            # the head's own chains, kept off the host's accessories and out of any
            # head body they start inside, as hairmix does
            import hairmix
            hairmix.confine(build, source, {b['name'] for b in PMX(build).bones} - before)
            os.remove(source)

    with progress.step('headwear'):
        m = PMX(build)
        sl = holefill.slices(m)
        V = np.array([v[0] for v in m.vertices])
        top = V[np.unique(sl['头发'])][:, 1].max() if '头发' in sl else 20.0
        # Headwear is a shell that spans the head, not a badge on it: its footprint
        # has to cover most of the head's width and depth. The cap's badge, 246
        # triangles wide as a coin, once passed for headwear and parity against it
        # moved every hair vertex by up to three units.
        if wear is None:
            wear = []
            for x in m.materials:
                nm = x['name']
                if nm == '头发' or nm not in sl or not len(sl[nm]) or HEAD.search(nm) or 'metal' in nm.lower():
                    continue
                P = V[np.unique(sl[nm])]
                if (P[:, 1].min() > top - 2.5 and P[:, 1].max() > top - 0.5
                        and np.ptp(P[:, 0]) > 1.4 and np.ptp(P[:, 2]) > 1.4):
                    wear.append(nm)
        if wear:
            for w in wear:
                underwear(build, '头发', w)
        else:
            print('  nothing worn on the head')

    with progress.step('the rig, cleaned'):
        prune.main(build, force)
    with progress.step('the deform chain'):
        fixrig.main([build])
    with progress.step('out'):
        publish.main(build, out_dir)
    shutil.rmtree(work, ignore_errors=True)
    progress.done()
    m = PMX(os.path.join(out_dir, name + '.pmx'))
    print(f'{out_dir}/{name}.pmx: {len(m.vertices)} verts, {len(m.faces) // 3} triangles, '
          f'{len(m.bones)} bones, {len(m.rigids)} rigid bodies, {len(m.joints)} joints')
    print(f'  materials: {", ".join(x["name"] for x in m.materials)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('host')
    ap.add_argument('out')
    ap.add_argument('--drop', help='host materials to drop, comma-separated (default: by name)')
    ap.add_argument('--rename', help='a=b,c=d material renames for the studio\'s styling')
    a = ap.parse_args()
    main(a.host, a.out,
         drop=a.drop.split(',') if a.drop else None,
         rename=dict(x.split('=') for x in a.rename.split(',')) if a.rename else None)
