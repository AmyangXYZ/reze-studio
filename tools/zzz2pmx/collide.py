# -*- coding: utf-8 -*-
"""Move rigid bodies into a collision group of their own and say what they collide with, in place.

    python3 tools/zzz2pmx/collide.py <model.pmx> <bone-pattern> --group 3 --with 2 [--admit skirt]

The stored mask is what a body collides with (the engine pairs two bodies only when each
one's mask holds the other's group bit). The bodies on bones matching `bone-pattern` go
into `group` and collide with the groups in `--with`; bodies on bones matching `--admit`
take the new group into their own mask, or the pair never forms.

For a decoration resting against the body: 海拉's waist bow hangs a hair's breadth off
her torso capsule and jittered against it every frame, while passing through the skirt
beside it. Off the body and onto the skirt, it rests on the dress, which the legs push.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import PMX
from pmxw import write_pmx


def main(path, bones, group, with_, admit=None):
    m = PMX(path)
    names = [b['name'] for b in m.bones]
    mask = 0
    for g in with_:
        mask |= 1 << g
    moved, admitted = [], []
    for r in m.rigids:
        bn = names[r['bone']] if r['bone'] >= 0 else ''
        if re.search(bones, bn):
            r['group'], r['nocollide'] = group, mask
            moved.append(r['name'])
        elif admit and re.search(admit, bn):
            r['nocollide'] |= 1 << group
            admitted.append(r['name'])
    write_pmx(path, m)
    print(f'{path}: {len(moved)} bodies on {bones} into group {group}, colliding with {with_}; '
          f'{len(admitted)} on {admit} admit them')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('bones')
    ap.add_argument('--group', type=int, required=True)
    ap.add_argument('--with', dest='with_', default='', help='groups to collide with, comma-separated')
    ap.add_argument('--admit')
    a = ap.parse_args()
    main(a.model, a.bones, a.group, [int(g) for g in a.with_.split(',') if g], a.admit)
