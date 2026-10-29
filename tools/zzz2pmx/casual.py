# -*- coding: utf-8 -*-
"""reze's own shirt and shorts on the 塞勒涅 body.

`reze-bare` is the dressing target -- complete skin, the thinner arms and legs, the
full rig -- and reze.pmx's clothes are carried onto it with `dress`: by bone, the way
a dresser does it -- rotation and offset from the two skeletons, a girth scale per
bone from the two bodies, a vertex blended by its own weights -- and nothing projected
onto the skin. The skin is then set back under the cloth where the bust comes through
between its vertices.

    python3 tools/zzz2pmx/casual.py

Reads reze.pmx and reze-bare and never writes them. Output is
`public/models/reze-casual-selene/`.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bustmerge
import dress
import fixrig
import progress
import publish
from pmx import PMX
from pmxw import write_pmx

BASE = 'public/models/reze-bare/reze-bare.pmx'
DONOR = 'public/models/reze/reze.pmx'
SHEET = 'zzz_EvelynBodyADiffuse.png'
PIECES = ['shirt', 'shorts', 'ribbon']
DIR = 'public/models/reze-casual-selene'
WORK = f'{DIR}/_work'
BUILD = f'{WORK}/reze-casual-selene.pmx'


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(os.path.join(WORK, 'textures'), exist_ok=True)
    with progress.step('the bare body'):
        shutil.copyfile(BASE, BUILD)
        for f in os.listdir(os.path.join(os.path.dirname(BASE), 'textures')):
            shutil.copyfile(os.path.join(os.path.dirname(BASE), 'textures', f),
                            os.path.join(WORK, 'textures', f))
        shutil.copyfile(os.path.join(os.path.dirname(DONOR), 'textures', SHEET),
                        os.path.join(WORK, 'textures', SHEET))
        m = PMX(BUILD)
        rel = 'textures\\' + SHEET
        if rel not in m.textures:
            m.textures.append(rel)
        write_pmx(BUILD, m)
    with progress.step('her clothes, by bone'):
        dress.dress(DONOR, BUILD, PIECES, BUILD, rel, donor_body=('skin',),
                    target_body=('skin', 'skin_hands'))
    with progress.step('the skin, under the cloth'):
        m = PMX(BUILD)
        for piece, band in (('shirt', (13.4, 13.9)), ('shorts', (9.3, 9.8))):
            bustmerge.sink(m, body='skin', piece=piece, room=0.03, cap=0.05, reach=0.3,
                           band=band)
        write_pmx(BUILD, m)
    with progress.step('the deform chain'):
        fixrig.main([BUILD])
    with progress.step('out'):
        publish.main(BUILD, DIR)
    shutil.rmtree(WORK, ignore_errors=True)
    progress.done()
    m = PMX(f'{DIR}/reze-casual-selene.pmx')
    print(f'  materials: {", ".join(x["name"] for x in m.materials)}')


if __name__ == '__main__':
    main()
