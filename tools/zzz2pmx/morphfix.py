# -*- coding: utf-8 -*-
"""Re-point a grafted model's group expressions at the morphs they name in reze.pmx.

    python3 tools/zzz2pmx/morphfix.py <model.pmx> [...]

For a model whose source chain can no longer run (reze-maid's mannequin is gone).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import PMX
from pmxw import write_pmx

REZE = 'public/models/reze/reze.pmx'


def check(m, r):
    """Group morphs whose members differ from reze's, by name."""
    rn = [x['name'] for x in r.morphs]
    mn = [x['name'] for x in m.morphs]
    bad = []
    for g in r.morphs:
        if g['type'] not in (0, 9) or g['name'] not in mn:
            continue
        want = [(rn[o[0]], o[1]) for o in g['offsets'] if rn[o[0]] in mn]
        have = [(mn[o[0]], o[1]) for o in m.morphs[mn.index(g['name'])]['offsets']]
        if want != have:
            bad.append((g['name'], have, want))
    return bad


def fix(path, reze=REZE):
    m, r = PMX(path), PMX(reze)
    bad = check(m, r)
    if not bad:
        print(f'{path}: group expressions already right')
        return 0
    rn = [x['name'] for x in r.morphs]
    at = {x['name']: i for i, x in enumerate(m.morphs)}
    for g in r.morphs:
        if g['type'] in (0, 9) and g['name'] in at:
            m.morphs[at[g['name']]]['offsets'] = [(at[rn[o[0]]], o[1]) for o in g['offsets'] if rn[o[0]] in at]
    write_pmx(path, m)
    print(f'{path}: {len(bad)} group expressions re-pointed: ' + ', '.join(b[0] for b in bad))
    return len(bad)


if __name__ == '__main__':
    for p in sys.argv[1:]:
        fix(p)
