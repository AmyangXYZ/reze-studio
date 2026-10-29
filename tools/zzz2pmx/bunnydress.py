# -*- coding: utf-8 -*-
"""芭丝特's bunny outfit on the 塞勒涅 body.

`reze-bare` is the dressing target and 芭丝特1.0.pmx the donor -- a different frame,
0.9 forward and shorter -- so every piece is carried by `dress`: by bone, with a girth
scale per bone from the two bodies. Her legs are stockings with no skin under them, so
the stockings themselves stand in for her skin when the leg girths are measured. The
stockings' feet are cut off above 塞勒涅's shoes, which stay; her heels and nails do
not come. The ears and headband are worn on
the hair and go by its crown. The tail rides 下半身 rigidly -- its chain is not carried.

    python3 tools/zzz2pmx/bunnydress.py

Output is `public/models/reze-bunny-selene/`.
"""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bustmerge
import dress
import fixrig
import holefill
import progress
import publish
from pmx import PMX
from pmxw import write_pmx

BASE = 'public/models/reze-bare/reze-bare.pmx'
DONOR = 'public/models/深空之眼—芭丝特兔女郎/芭丝特1.0.pmx'
SHEETS = ['芭丝特衣服.png', '芭丝特衣服1.png']
PIECES = ['上衣', '衣饰铁', '尾巴', '背带', '胖次', '袜子', '耳朵', '头饰', '衣饰']
RENAME = {'上衣': '上衣_cloth', '衣饰铁': '衣饰铁_metal', '尾巴': '尾巴_cloth', '背带': '背带_cloth',
          '胖次': '胖次_cloth', '袜子': '袜子_stockings', '耳朵': '耳朵_cloth', '头饰': '头饰_metal',
          '衣饰': '衣饰_cloth'}
SINK = {'上衣': (11.0, 11.5), '胖次': (11.3, 11.6)}
SHOE_TOP = 1.85           # 塞勒涅's shoes end here; the stockings' feet go
DIR = 'public/models/reze-bunny-selene'
WORK = f'{DIR}/_work'
BUILD = f'{WORK}/reze-bunny-selene.pmx'


def cut_below(m, name, y):
    F = np.array(m.faces).reshape(-1, 3)
    V = np.array([v[0] for v in m.vertices])
    at, out, gone = 0, [], 0
    for mat in m.materials:
        n = mat['fcount'] // 3
        T = F[at:at + n]
        at += n
        if mat['name'] == name:
            keep = ~(V[T][:, :, 1] < y).all(1)
            gone = int((~keep).sum())
            T = T[keep]
            mat['fcount'] = len(T) * 3
        out.append(T)
    m.faces = np.concatenate(out).ravel().tolist()
    return gone


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(os.path.join(WORK, 'textures'), exist_ok=True)
    with progress.step('the bare body'):
        shutil.copyfile(BASE, BUILD)
        for f in os.listdir(os.path.join(os.path.dirname(BASE), 'textures')):
            shutil.copyfile(os.path.join(os.path.dirname(BASE), 'textures', f),
                            os.path.join(WORK, 'textures', f))
        for f in SHEETS:
            shutil.copyfile(os.path.join(os.path.dirname(DONOR), f), os.path.join(WORK, 'textures', f))
    with progress.step('her outfit, by bone'):
        dress.dress(DONOR, BUILD, PIECES, BUILD, None, donor_body=('皮肤', '袜子'),
                    target_body=('skin', 'skin_hands'), on_head=('耳朵', '头饰'))
    with progress.step('the stockings end at the shoes; the skin gives way under the cloth'):
        m = PMX(BUILD)
        gone = cut_below(m, '袜子', SHOE_TOP)
        print(f'  袜子: {gone} triangles of the feet cut below y {SHOE_TOP}')
        for piece, band in SINK.items():
            bustmerge.sink(m, body='skin', piece=piece, room=0.03, cap=0.05, reach=0.3, band=band)
        for x in m.materials:
            if x['name'] in RENAME:
                x['name'] = x['name_en'] = RENAME[x['name']]
        write_pmx(BUILD, m)
    with progress.step('the deform chain'):
        fixrig.main([BUILD])
    with progress.step('out'):
        publish.main(BUILD, DIR)
    shutil.rmtree(WORK, ignore_errors=True)
    progress.done()
    m = PMX(f'{DIR}/reze-bunny-selene.pmx')
    print(f'  materials: {", ".join(x["name"] for x in m.materials)}')


if __name__ == '__main__':
    main()
