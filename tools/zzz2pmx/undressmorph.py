# -*- coding: utf-8 -*-
"""Hide the underwear, and give it a morph that trades places with the shorts.

The panties sit entirely inside the shorts, and two surfaces that close together are
decided by whichever happens to bulge -- so wherever the shorts pinch, the panties
surface. Seating them deeper closes the poses you test and opens the ones you did
not: the correction is baked into the rest pose while the collision belongs to the
pose being played.

Geometry that is not drawn cannot poke through. So the panties start at zero alpha,
which settles it for every pose at once, and a material morph raises them back while
dropping the shorts -- one slider that takes the shorts off. Both offsets are additive,
whose neutral value is zero, so the morph at rest changes nothing.

    python3 tools/zzz2pmx/undressmorph.py <pmx> [--under cloth01.001] [--over shorts]
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pmx import PMX
from pmxw import write_pmx

NAME, NAME_EN = 'ショーツ', 'shorts off'
FRAME = '衣装'
ADD = 1
PANEL = 4          # the "other" tab, where costume switches belong


def offset(index, alpha):
    """A material-morph offset that only moves diffuse alpha."""
    return (index, ADD,
            [0.0, 0.0, 0.0, alpha],      # diffuse
            [0.0, 0.0, 0.0],             # specular
            0.0,                         # shininess
            [0.0, 0.0, 0.0],             # ambient
            [0.0, 0.0, 0.0, 0.0],        # edge colour
            0.0,                         # edge size
            [0.0, 0.0, 0.0, 0.0],        # texture
            [0.0, 0.0, 0.0, 0.0],        # sphere
            [0.0, 0.0, 0.0, 0.0])        # toon


def main(path, under='cloth01.001', over='shorts'):
    m = PMX(path)
    idx = {mat['name']: i for i, mat in enumerate(m.materials)}
    for n in (under, over):
        if n not in idx:
            raise SystemExit(f'no material {n}')

    was = m.materials[idx[under]]['diffuse'][3]
    m.materials[idx[under]]['diffuse'][3] = 0.0
    print(f'{under}: alpha {was:.2f} -> 0.00, so nothing of it is drawn')

    for mo in m.morphs:
        if mo['name'] == NAME:
            raise SystemExit(f'{path} already has a {NAME} morph')
    m.morphs.append({'name': NAME, 'name_en': NAME_EN, 'panel': PANEL, 'type': 8,
                     'offsets': [offset(idx[under], 1.0), offset(idx[over], -1.0)]})
    mi = len(m.morphs) - 1
    print(f'morph {mi} "{NAME}": {under} alpha +1, {over} alpha -1')

    for fr in m.frames:
        if fr['name'] == FRAME:
            fr['items'].append((1, mi))
            break
    else:
        m.frames.append({'name': FRAME, 'name_en': 'costume', 'special': 0,
                         'items': [(1, mi)]})
        print(f'added display frame "{FRAME}"')
    write_pmx(path, m)
    print(f'wrote {path}')


if __name__ == '__main__':
    a = sys.argv
    g = lambda k, d: a[a.index(k) + 1] if k in a else d
    main(a[1], g('--under', 'cloth01.001'), g('--over', 'shorts'))
