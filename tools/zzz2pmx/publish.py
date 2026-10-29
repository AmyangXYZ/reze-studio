# -*- coding: utf-8 -*-
"""Copy a built model out to a folder of its own, carrying only what it samples.

A build directory accumulates: the intermediates it was made from, the sheets of every
donor it borrowed a material from, the sheets those donors' other materials wanted. The
model itself names a fraction of them. This takes the PMX, drops the texture-table
entries no material indexes into, folds together the paths and then the contents that
duplicate -- a sheet copied in from two donors is one file here -- and copies across
exactly what survives, flat, so the table needs no directory tree under it.

    python3 tools/zzz2pmx/publish.py public/models/reze-maid-selene/reze-maid.pmx \\
                                     public/models/reze-maid

Reads the build and never writes it.
"""
import hashlib
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixrig
import prune
import textures
from pmx import PMX
from pmxw import write_pmx

SRC = 'public/models/reze-maid-selene/reze-maid.pmx'
DST = 'public/models/reze-maid'


def main(src=SRC, dst=DST):
    m = PMX(src)
    base = os.path.dirname(os.path.abspath(src))
    # Last, always. Every graft can put mesh back on a bone the IK solves, and every
    # trim can leave a chain's tail bones and bodies swinging with nothing on them;
    # no amount of remembering has stopped either coming back, so here they cannot leave.
    gone = prune.prune(m)
    if any(gone):
        print(f'  pruned {gone[0]} bones, {gone[1]} rigid bodies, {gone[2]} joints nothing used')
    moved = fixrig.apply(m, quiet=True)
    if moved:
        print(f'  {moved} vertices moved onto the deform chain on the way out')
    # a host's own weights can sum to 1.004 here and there; the engine trusts the sum
    fixed = 0
    for k, v in enumerate(m.vertices):
        if v[4] in (2, 4):
            tot = sum(v[6])
            if tot > 0 and abs(tot - 1.0) > 1e-4:
                v = list(v)
                v[6] = [w / tot for w in v[6]]
                m.vertices[k] = tuple(v)
                fixed += 1
    if fixed:
        print(f'  {fixed} vertices had their weights normalised')
    before = len(m.textures)
    dropped = textures.prune(m)

    os.makedirs(os.path.join(dst, 'textures'), exist_ok=True)
    seen, byhash, missing, bytes_, folded = {}, {}, [], 0, 0
    for i, rel in enumerate(m.textures):
        fp = os.path.normpath(os.path.join(base, rel.replace('\\', '/')))
        if not os.path.exists(fp):
            missing.append(rel)
            m.textures[i] = 'textures\\' + os.path.basename(fp)
            continue
        with open(fp, 'rb') as fh:
            digest = hashlib.md5(fh.read()).hexdigest()
        if digest in byhash:
            m.textures[i] = byhash[digest]
            folded += 1
            continue
        name = os.path.basename(fp)
        # Two donors can ship different sheets under the same basename; keep both.
        while name in seen and seen[name] != fp:
            stem, ext = os.path.splitext(name)
            name = f'{stem}_{i}{ext}'
        seen[name] = fp
        out = os.path.join(dst, 'textures', name)
        shutil.copyfile(fp, out)
        bytes_ += os.path.getsize(out)
        m.textures[i] = byhash[digest] = 'textures\\' + name

    out = os.path.join(dst, os.path.basename(src))
    write_pmx(out, m)
    print(f'{out}: {len(m.vertices)} verts, {len(m.faces) // 3} triangles, '
          f'{len(m.materials)} materials, {len(m.bones)} bones, {len(m.morphs)} morphs, '
          f'{len(m.rigids)} rigid bodies, {len(m.joints)} joints')
    textures.prune(m)
    print(f'  textures {before} -> {len(m.textures)} ({dropped} unused, {folded} the same '
          f'sheet under another name), {bytes_ / 1e6:.1f} MB copied')
    for rel in missing:
        print(f'  MISSING {rel}')

    check = PMX(out)
    bad = [t for t in check.textures
           if not os.path.exists(os.path.join(dst, t.replace('\\', '/')))]
    print('  every texture the model names is present' if not bad
          else f'  {len(bad)} still unresolved: {bad}')
    return out


if __name__ == '__main__':
    main(*(sys.argv[1:3] or []))
