# -*- coding: utf-8 -*-
"""Material presentation fixes that outlive a rebuild.

    python3 tools/zzz2pmx/presentation.py <pmx> [<pmx> ...]     # patch in place
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import PMX
from pmxw import write_pmx

# The model outlines skin in a dull red (0.502, 0.251, 0.251) and everything else in
# black. `socks` is a garment that was given the skin colour, so it is drawn with a red
# line down the thigh where the shorts beside it have a black one.
OUTLINE_MATCH = {'socks': 'shorts'}


def apply(model, verbose=True):
    by_name = {m['name']: m for m in model.materials}
    fixed = []
    for name, like in OUTLINE_MATCH.items():
        a, b = by_name.get(name), by_name.get(like)
        if a is None or b is None or a['edge_color'] == b['edge_color']:
            continue
        fixed.append((name, list(a['edge_color']), like))
        a['edge_color'] = list(b['edge_color'])
        a['edge_size'] = b['edge_size']
    if verbose:
        for name, was, like in fixed:
            print(f'  {name}: outline {[round(v, 3) for v in was]} -> {like}\'s')
    return len(fixed)


if __name__ == '__main__':
    for path in sys.argv[1:]:
        m = PMX(path)
        n = apply(m)
        if n:
            write_pmx(path, m)
            print(f'{os.path.basename(path)}: {n} material(s) fixed')
        else:
            print(f'{os.path.basename(path)}: nothing to fix')
