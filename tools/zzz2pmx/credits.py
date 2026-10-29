# -*- coding: utf-8 -*-
"""Write the model's name and credit block.

The header is the only place the provenance travels with the file, and this model
is assembled from four sources with terms of their own. Kept as a script rather
than typed once into an editor so the text stays in the repo and a rebuild can put
it back.

Only the four header strings are written; everything else round-trips untouched.

    python3 tools/zzz2pmx/credits.py public/models/reze/reze.pmx
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import PMX
from pmxw import write_pmx

NAME = '蕾塞'
NAME_EN = 'Reze'

COMMENT = '''蕾塞 / Reze — 电锯人

素体：塞尔凯特（深空之眼）   模型编辑：神帝宇
头部：塞勒涅（深空之眼）     模型编辑：神帝宇
服装：绝区零 伊芙琳 Reze mod（3DMigoto / XXMI）  作者：Stelziey（把心静下来zzz）
改造：黯灭小羊 AmyangXYZ

原模型条款：请勿二次配布 / 请勿商用 / 请勿用于18禁作品
版权所属 深空之眼 · miHoYo
'''

COMMENT_EN = '''蕾塞 / Reze (Chainsaw Man).

Body from Selket, head and face from Selene — both Aethergazer, model edits by
神帝宇. Outfit from the Reze mod for Evelyn (Zenless Zone Zero, miHoYo) by
Stelziey (把心静下来zzz), ported from its 3DMigoto buffers. Rework by 黯灭小羊 (AmyangXYZ).

Source models are shared on their own terms: no redistribution, no commercial
use, nothing R18. Copyright remains with Aethergazer and miHoYo.
'''


def main(path):
    m = PMX(path)
    was = (m.name, m.name_en)
    m.name, m.name_en = NAME, NAME_EN
    m.comment, m.comment_en = COMMENT, COMMENT_EN
    write_pmx(path, m)
    print(f'{os.path.basename(path)}: name {was[0]!r}/{was[1]!r} -> {NAME!r}/{NAME_EN!r}')
    print(f'  comment {len(COMMENT)} chars, comment_en {len(COMMENT_EN)} chars')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'public/models/reze/reze.pmx')
