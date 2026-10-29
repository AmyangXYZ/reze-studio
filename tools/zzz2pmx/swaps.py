# -*- coding: utf-8 -*-
"""Every 深空之眼 host reze's head goes on, in one place, so they all rebuild
together when the graft changes.

    python3 tools/zzz2pmx/swaps.py [name ...]

Reze's head goes on at its own size (headgraft --scale 1); only where it sits is read
off the host's face. The host folders for all but 芭丝特 were removed on 2026-09-15,
so those entries stand as the record of how each model was made; `headsize.py` is
what brings an already published model's head to x1.0.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import headswap

M = 'public/models/'


def neck_scale(m, neck=18.524, ramp=(1.4, 4.0), centre_z=0.0):
    """A smaller figure brought to 塞勒涅's neck height with its feet kept."""
    import pmxscale
    y0 = next(b['pos'][1] for b in m.bones if b['name'] == '首')
    ya, yb = ramp
    body = (neck - ya - (yb - ya) / 2) / ((yb - ya) / 2 + (y0 - yb))
    pmxscale.profile(m, 1.0, body, ramp, centre_z)
    print(f'  host: feet kept, body above the shin x{body:.4f}, neck {y0:.2f} -> {neck:.2f}, centred from z {centre_z}')


def ratio_scale(m, to=M + 'reze-maid/reze-maid.pmx', centre_z=0.0):
    """A figure of other proportions brought to another's: widths by the shoulder-to-
    neck scale, heights remapped so the ankle keeps that scale and the hip and neck
    land on the target's."""
    from pmx import PMX
    import pmxscale
    hb = {b['name']: b['pos'] for b in m.bones}
    tb = {b['name']: b['pos'] for b in PMX(to).bones}
    k = tb['首'][1] / hb['首'][1]
    ankle = hb['左足首'][1]
    pts = [(0.0, 0.0), (ankle, ankle * k), (hb['左足'][1], tb['左足'][1]), (hb['首'][1], tb['首'][1])]
    Y = pmxscale.knots(m, k, pts, centre_z)
    print(f'  host: widths x{k:.4f}; ankle, hip, neck {", ".join(f"{a:.2f}->{Y(a):.2f}" for a, _ in pts[1:])}')


def makima_head():
    """Makima's head, built fresh by hairmix into a folder of its own for the swap to
    read; her own folder cleared, since the swap does not."""
    import shutil
    import hairmix
    shutil.rmtree(M + 'makima', ignore_errors=True)
    hairmix.main('makima', M + '_makima_head')


def gone(path):
    def post(out):
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    return post


def headwear_on_reze(swap, mats):
    """For a host whose head is larger than reze's: what it wears on the head goes on
    her head at her size, scaled by the face-width ratio about the host's hair crown
    and set on hers."""
    def post(out):
        from pmx import PMX
        import headsize
        import headwear
        h = PMX(SWAPS[swap]['host'])
        for x in h.materials:
            if x['name'] == '脸':
                x['name'] = 'face01'
        SWAPS[swap]['scale'](h)
        path = os.path.join(out, os.path.basename(out) + '.pmx')
        m = PMX(path)
        s = headsize.face_width(m) / headsize.face_width(h)
        hc, rc = headwear.crown(h), headwear.crown(m)
        headwear.move(path, mats, rc - hc, s, hc)
    return post


def brahma_drop():
    from pmx import PMX
    m = PMX(M + '【深空之眼】三相·梵天-轻纱梦华/大梵天皮肤/大梵天皮肤.pmx')
    keep = {'hair_acce', 'eye_mask'}
    return [x['name'] for x in m.materials if headswap.HEAD.search(x['name']) and x['name'] not in keep] + ['weapon']


THOTH = {'皮肤': 'skin', '手': '手_skin', '指甲': '指甲_skin', '裤子': '裤子_cloth', '衣饰': '衣饰_metal',
         '裙带': '裙带_cloth', '裙带1': '裙带1_cloth', '胸口布': '胸口布_cloth', '胸口布1': '胸口布1_cloth',
         '头巾白': '头巾白_cloth', '头巾小白': '头巾小白_cloth', '衣花白': '衣花白_cloth', '衣花白1': '衣花白1_cloth',
         '手套': '手套_cloth', '袖子': '袖子_cloth'}

SWAPS = {
    'reze-instructor': dict(host=M + '【深空之眼】狂狮·塞赫麦特-危险教官/危险教官.pmx',
                            rename={'handcuff': 'handcuff_metal'}),
    'reze-brahma': dict(host=M + '【深空之眼】三相·梵天-轻纱梦华/大梵天皮肤/大梵天皮肤.pmx',
                        drop=brahma_drop),
    'reze-director': dict(host=M + '深空之眼-太一·庚辰-代班董事/庚辰制服2.0.pmx',
                          drop=lambda: None, extra_drop=['发饰', '发饰带', '头饰']),
    'reze-seawhisper': dict(host=M + '深空之眼—太一·庚辰-海上的私语/太一·庚辰-海上的私语/庚辰6.0.pmx'),
    'reze-lingguang': dict(host=M + '陵光-珠绣联动/陵光-珠绣联动/陵光-珠绣联动3.0.pmx',
                           rename={'旗袍': 'qipao_dress'}),
    'reze-hodur': dict(host=M + '【深空之眼】谧光之刻·霍德尔-幽情祷告/幽情祷告.pmx',
                       hat={'cloth01': 'veil_cloth'}, wear=['veil_cloth']),
    # 芭丝特 is a smaller figure standing 0.86 forward: her feet are already 塞勒涅's
    # size (foot bone 1.917 on both), her neck sits at 17.19 against 18.52, so the body
    # above the shin is scaled to put the neck there and the feet stay
    'reze-bunny': dict(host=M + '深空之眼—芭丝特兔女郎/芭丝特1.0.pmx', scale=lambda m: neck_scale(m, centre_z=0.859),
                       rename={'皮肤': 'skin', '袜子': '袜子_stockings', '衣饰铁': '衣饰铁_metal', '头饰': '头饰_metal',
                               '上衣': '上衣_cloth', '尾巴': '尾巴_cloth', '背带': '背带_cloth', '胖次': '胖次_cloth',
                               '鞋子': '鞋子_cloth', '耳朵': '耳朵_cloth', '衣饰': '衣饰_cloth', '指甲': '指甲_skin'}),
    # the 塞勒涅 family proper; 午夜沉沦's shared metal has a hair ornament above y 19.6
    'reze-selene': dict(host=M + '塞勒涅/塞勒涅.pmx',
                        rename={'皮肤': 'skin', '衣饰': '衣饰_metal', '鞋子': '鞋子_cloth', '裙饰': '裙饰_metal',
                                '裙子': '裙子_cloth', '衣服': '衣服_cloth', '肩饰': '肩饰_metal', '头饰': '头饰_metal',
                                '手套': '手套_cloth', '胸口': '胸口_cloth'}),
    'reze-beach': dict(host=M + '【深空之眼】幽月·塞勒涅-海滩漫步/海滩漫步.pmx',
                       rename={'gem': 'gem_metal', 'glasses': 'glasses_metal'}),
    'reze-barmaid': dict(host=M + '【深空之眼】幽月·塞勒涅-沉醉旖旎/沉醉旖旎.pmx',
                       rename={'gem': 'gem_metal', 'stocking': 'stockings'}),
    'reze-midnight': dict(host=M + '【深空之眼】裁暗之锋·塞尔凯特-午夜沉沦/午夜沉沦.pmx',
                          rename={'earring': 'earring_metal', 'gem01': 'gem01_metal'}, trim=19.6),
    'reze-serket': dict(host=M + '塞尔凯特2.0/塞尔凯特3.0.pmx', extra_drop=['发饰'],
                        rename={'皮肤': 'skin', '袜子': '袜子_stockings', '裙带': '裙带_cloth', '裙布': '裙布_cloth',
                                '鞋子': '鞋子_cloth', '鞋子饰': '鞋子饰_metal', '腿环': '腿环_metal', '裤子': '裤子_cloth',
                                '外套': '外套_cloth', '外套饰': '外套饰_metal', '裙子': '裙子_cloth', '衣服': '衣服_cloth'}),
    'reze-tropic': dict(host=M + '【深空之眼】硝芒·埃克什瓦-热带蓝调/热带蓝调.pmx',
                        rename={'hat': 'hat_cloth', 'gem01': 'gem01_metal', 'gem012': 'gem012_metal',
                                'glasses': 'glasses_metal', 'bikini': 'bikini_cloth', 'panty': 'panty_cloth'}),
    # 托特 is a smaller figure too (首 at 17.07, foot bone already 1.917): once as shipped,
    # once brought to 塞勒涅's neck height with the feet kept
    'reze-thoth': dict(host=M + '深空之眼—苍鹭·托特「扉页之吻」/苍鹭·托特「扉页之吻」白衣1.0.pmx', rename=THOTH),
    'reze-thoth-tall': dict(host=M + '深空之眼—苍鹭·托特「扉页之吻」/苍鹭·托特「扉页之吻」白衣1.0.pmx', rename=THOTH,
                            scale=lambda m: neck_scale(m)),
    # Makima: her head (hairmix.py makima, built first) on 塞尔凯特's body; 塞尔凯特's
    # braid tie rides the braid that comes off
    'makima': dict(host=M + '塞尔凯特2.0 2/塞尔凯特3.0.pmx', head=M + '_makima_head/makima.pmx',
                   pre=[makima_head], post=[gone(M + '_makima_head')],
                   take=['脸', '口腔', '舌头', '牙齿', '眉毛', '眼白', '眼睛', '眼睛_高光', '表情',
                         'hair_front', 'hair_braid'],
                   extra_drop=['发饰'],
                   rename={'皮肤': 'skin', '袜子': '袜子_stockings', '裙带': '裙带_cloth', '裙布': '裙布_cloth',
                           '鞋子': '鞋子_cloth', '鞋子饰': '鞋子饰_metal', '腿环': '腿环_metal', '裤子': '裤子_cloth',
                           '外套': '外套_cloth', '外套饰': '外套饰_metal', '裙子': '裙子_cloth', '衣服': '衣服_cloth'}),
    'reze-ekshva': dict(host=M + '埃克什瓦/埃克什瓦1.0.pmx',
                        rename={'皮肤': 'skin', '衣服': '衣服_cloth', '衣服1': '衣服1_cloth', '裤子': '裤子_cloth',
                                '帽子': '帽子_cloth', '胸口': '胸口_cloth', '腰带': '腰带_cloth',
                                '耳环': '耳环_metal', '帽子饰': '帽子饰_metal', '衣服饰': '衣服饰_metal',
                                '胸饰': '胸饰_metal', '鞋子饰': '鞋子饰_metal'}),
    # 南瓜小鬼头 is a small figure of a child's build (首 at 16.66, legs short for it):
    # brought to reze-maid's ankle, hip and neck heights
    # and her face is 1.3 times reze's, so the head is seated on her neck, not her face
    'reze-pumpkin': dict(host=M + '前鬼坊天狗·南瓜小鬼头/南瓜小鬼头.pmx', scale=ratio_scale, seated=True,
                         extra_drop=['二重', '高光右', '高光左', '嘴'],
                         rename={'身体': 'skin', '右腿': '右腿_stockings', '左腿': '左腿_stockings', '鞋子': '鞋子_cloth',
                                 '裙子': '裙子_cloth', '背后翅膀': '背后翅膀_cloth', '头饰': '头饰_cloth',
                                 '手环': '手环_metal'}, post=[headwear_on_reze('reze-pumpkin', ['头饰_cloth'])]),
    # 海拉 魔女见习夜, another small figure (首 at 14.24) with a head 1.3 times reze's; her
    # witch hat rides its own Bone_hat* bones inside the skin and dress materials. The
    # host is gone; what followed the build was applied in place:
    #   paintsplit.py <m> skin glove_cloth '手首|手捩|ひじ|腕捩|指|elbow' --refine 3
    #   paintsplit.py <m> skin choker_cloth '首|上半身2' --above 17.5 --small 600 --whole
    #   paintsplit.py <m> skin stockings '^左(ひざ|足首)|Bone_L_Knee' --large 1000 --refine 3
    #   paintsplit.py <m> skin panty_cloth '^下半身$|TwistThigh|butt' --above 11.5 --below 14.4 --large 1000 --refine 3
    #   collide.py <m> bow --group 5 --with 2 --admit skirt   (her waist bow jittered on the torso capsule)
    #   feet.py <m>   (soles sank 0.35 into the floor; the toe IK target floated past the shoe)
    'reze-hela': dict(host=M + '119401-海拉-魔女见习夜/119401-海拉-魔女见习夜.pmx', scale=ratio_scale, seated=True,
                      extra_drop=['刘海'],
                      rename={'身体': 'skin', '衣裙': '衣裙_cloth', '饰品': '饰品_metal', '披风衣': '披风衣_cloth'},
                      hat={'skin': ('hat_cloth', r'hat', r'hair'), '衣裙_cloth': ('hat2_cloth', r'hat', r'hair')},
                      wear=['hat_cloth'],
                      post=[headwear_on_reze('reze-hela', ['hat_cloth', 'hat2_cloth'])]),
}


def main(names=None):
    for name, s in SWAPS.items():
        if names and name not in names:
            continue
        print(f'=== {name}')
        for pre in s.get('pre', []):
            pre()
        drop = s.get('drop')
        drop = drop() if callable(drop) else drop
        if s.get('extra_drop'):
            from pmx import PMX
            m = PMX(s['host'])
            drop = [x['name'] for x in m.materials if headswap.HEAD.search(x['name'])] + s['extra_drop']
        headswap.main(s['host'], M + name, drop=drop, rename=s.get('rename'), head=s.get('head'), take=s.get('take'),
                      hat=s.get('hat'), wear=s.get('wear'), scale=s.get('scale'), trim=s.get('trim'),
                      seated=s.get('seated', False))
        for post in s.get('post', []):
            post(M + name)


if __name__ == '__main__':
    main(sys.argv[1:] or None)
