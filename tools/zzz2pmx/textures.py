# -*- coding: utf-8 -*-
"""Cut a model's texture folder down to the files it actually draws with.

Grafting leaves files beside a model that nothing samples any more, and a second
folder of raw mod dumps beside that. This finds every asset file the materials do
not reach and, with --delete, removes it.

The PMX is not written. The texture table keeps its entries, its order and its
indices, so a run cannot disturb anything else in the model -- entries the
materials never index are left where they are, dangling, which the loader records
and never fetches. Pass --prune-table to drop them, which does rewrite the file.

Orphans are judged against every model named in one run, so cleaning one model of
a pair keeps what the other draws with.

    python3 tools/zzz2pmx/textures.py <pmx>...            # report
    python3 tools/zzz2pmx/textures.py <pmx>... --delete   # remove the orphans
    python3 tools/zzz2pmx/textures.py <pmx> --rename=old.png=new.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import PMX
from pmxw import write_pmx

ASSET_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tga', '.dds', '.spa', '.sph',
                  '.obj', '.ini', '.psd')


def slots(material):
    """The texture-table slots a material indexes into."""
    yield 'tex'
    yield 'sph'
    if not material['toon_shared']:
        yield 'toon'


def prune(model):
    """Drop unused entries, fold duplicate paths together, remap the materials."""
    keep, at, remap = [], {}, {}
    for m in model.materials:
        for key in slots(m):
            i = m[key]
            if not (0 <= i < len(model.textures)):
                continue
            path = model.textures[i]
            if path not in at:
                at[path] = len(keep)
                keep.append(path)
            remap[i] = at[path]

    dropped = len(model.textures) - len(keep)
    model.textures = keep
    for m in model.materials:
        for key in slots(m):
            m[key] = remap.get(m[key], -1)
    return dropped


def used_files(model, path, table=False):
    """Files the model needs: what its materials sample, or the whole table."""
    base = os.path.dirname(os.path.abspath(path))
    if table:
        want = model.textures
    else:
        want = []
        for m in model.materials:
            for key in slots(m):
                if 0 <= m[key] < len(model.textures):
                    want.append(model.textures[m[key]])
    return {os.path.normpath(os.path.join(base, t.replace('\\', '/'))) for t in want}


def orphans(base, keep):
    """Asset files under `base` that no model's texture table names."""
    found = []
    for root, _, files in os.walk(base):
        for f in files:
            full = os.path.normpath(os.path.join(root, f))
            if full.lower().endswith(ASSET_SUFFIXES) and full not in keep:
                found.append(full)
    return sorted(found)


def rename(model, base, old, new):
    """Rename a texture file and repoint the table entries that name it.

    Only the path string changes -- entry count, order and indices stay as they are,
    so the materials go on sampling exactly what they sampled before.
    """
    hits = []
    for i, t in enumerate(model.textures):
        sep = '\\' if '\\' in t else '/'
        folder, _, name = t.rpartition(sep)
        if name != old:
            continue
        model.textures[i] = f'{folder}{sep}{new}' if folder else new
        hits.append(i)
    if not hits:
        raise SystemExit(f'textures: no table entry names {old}')
    src = os.path.join(base, model.textures[hits[0]].replace('\\', '/').rpartition('/')[0], old)
    dst = os.path.join(base, model.textures[hits[0]].replace('\\', '/'))
    if os.path.exists(src):
        os.rename(src, dst)
    elif not os.path.exists(dst):
        raise SystemExit(f'textures: {src} is not there to rename')
    return hits


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith('--')]
    do_delete = '--delete' in sys.argv
    do_prune = '--prune-table' in sys.argv
    keep, bases = set(), []
    for path in paths:
        model = PMX(path)
        base = os.path.dirname(os.path.abspath(path))
        for spec in [a[9:] for a in sys.argv[1:] if a.startswith('--rename=')]:
            was, _, now = spec.partition('=')
            hits = rename(model, base, was, now)
            write_pmx(path, model)
            model = PMX(path)
            print(f'{os.path.basename(path)}: {was} -> {now} '
                  f'({len(hits)} table {"entry" if len(hits) == 1 else "entries"} repointed)')
        if do_prune:
            before = len(model.textures)
            if prune(model):
                write_pmx(path, model)
            print(f'{os.path.basename(path)}: texture table {before} -> {len(model.textures)}')
        keep |= used_files(model, path)
        bases.append(base)
        drawn = sorted({os.path.basename(f) for f in used_files(model, path)})
        print(f'{os.path.basename(path)}: draws with {len(drawn)} textures'
              + ('' if do_prune else ', table untouched'))
        for t in drawn:
            print(f'   keeps {t}')

    for base in sorted(set(bases)):
        dead = orphans(base, keep)
        if not dead:
            print(f'{os.path.basename(base)}/: no orphaned files')
            continue
        size = sum(os.path.getsize(f) for f in dead)
        print(f'{os.path.basename(base)}/: {len(dead)} orphaned files, {size/1e6:.1f} MB'
              + (' — deleting' if do_delete else ' — pass --delete to remove'))
        for f in dead:
            if do_delete:
                os.remove(f)
            print(f'   {"removed" if do_delete else "orphan"} {os.path.relpath(f, base)}')
        if do_delete:
            for root, _, _ in os.walk(base, topdown=False):
                if root != base and not os.listdir(root):
                    os.rmdir(root)
                    print(f'   removed empty {os.path.relpath(root, base)}/')


if __name__ == '__main__':
    main()
