# -*- coding: utf-8 -*-
import struct

import numpy as np

class W:
    def __init__(self, enc='utf-16-le'):
        self.b = bytearray(); self.enc = enc
    def u1(self,v): self.b += struct.pack('<B', v)
    def i1(self,v): self.b += struct.pack('<b', v)
    def u2(self,v): self.b += struct.pack('<H', v)
    def i2(self,v): self.b += struct.pack('<h', v)
    def i4(self,v): self.b += struct.pack('<i', v)
    def u4(self,v): self.b += struct.pack('<I', v)
    def f(self,v):  self.b += struct.pack('<f', v)
    def vec(self,a): self.b += struct.pack('<%df'%len(a), *a)
    def raw(self,d): self.b += d
    def s(self,t):
        d = t.encode(self.enc); self.u4(len(d)); self.raw(d)

def isize(n, signed=True):
    if signed:
        return 1 if n < 128 else (2 if n < 32768 else 4)
    return 1 if n < 256 else (2 if n < 65536 else 4)

def write_pmx(path, m):
    """m: object with the same attributes the PMX reader produces."""
    sz = dict(vtx=isize(len(m.vertices), signed=False),
              tex=isize(len(m.textures)),
              mat=isize(len(m.materials)),
              bone=isize(len(m.bones)),
              morph=isize(len(m.morphs)),
              rb=isize(len(m.rigids)))
    w = W('utf-16-le')
    w.raw(b'PMX '); w.f(2.0)
    w.u1(8)
    w.raw(bytes([0, m.uv_add, sz['vtx'], sz['tex'], sz['mat'], sz['bone'], sz['morph'], sz['rb']]))
    w.s(m.name); w.s(m.name_en); w.s(m.comment); w.s(m.comment_en)

    def widx(kind, v):
        n = sz[kind]
        if kind == 'vtx':
            {1:w.u1, 2:w.u2, 4:w.i4}[n](v)
        else:
            {1:w.i1, 2:w.i2, 4:w.i4}[n](v)

    # vertices -- one pack per vertex off a cached layout, rather than a dozen
    w.i4(len(m.vertices))
    bc = {1: 'b', 2: 'h', 4: 'i'}[sz['bone']]
    base = '<%df' % (8 + 4 * m.uv_add)
    lay = {0: struct.Struct(base + 'B' + bc + 'f'),
           1: struct.Struct(base + 'B' + bc * 2 + 'ff'),
           2: struct.Struct(base + 'B' + bc * 4 + '4ff'),
           3: struct.Struct(base + 'B' + bc * 2 + 'f' + '9f' + 'f'),
           4: struct.Struct(base + 'B' + bc * 4 + '4ff')}
    blank = [0.0, 0.0, 0.0, 0.0]
    out = []
    for v in m.vertices:
        pos, nor, uv, add, wt, bs, ws, edge = v[:8]
        a = list(pos) + list(nor) + list(uv)
        for i in range(m.uv_add):
            a = a + (list(add[i]) if i < len(add) else blank)
        a.append(wt)
        if wt == 0:
            a = a + [bs[0]]
        elif wt == 1:
            a = a + [bs[0], bs[1], ws[0]]
        elif wt == 3:
            a = a + [bs[0], bs[1], ws[0]] + list(v[8])
        else:
            a = a + [bs[0], bs[1], bs[2], bs[3]] + list(ws[:4])
        a.append(edge)
        out.append(lay[wt].pack(*a))
    w.b += b''.join(out)
    # faces
    w.i4(len(m.faces))
    w.b += np.asarray(m.faces,
                      {1: np.uint8, 2: np.uint16, 4: np.int32}[sz['vtx']]).tobytes()
    # textures
    w.i4(len(m.textures))
    for t in m.textures: w.s(t)
    # materials
    w.i4(len(m.materials))
    for mm in m.materials:
        w.s(mm['name']); w.s(mm['name_en'])
        w.vec(mm['diffuse']); w.vec(mm['specular']); w.f(mm['shininess'])
        w.vec(mm['ambient']); w.u1(mm['flag']); w.vec(mm['edge_color']); w.f(mm['edge_size'])
        widx('tex', mm['tex']); widx('tex', mm['sph']); w.u1(mm['sph_mode'])
        w.u1(mm['toon_shared'])
        if mm['toon_shared']: w.u1(mm['toon'])
        else: widx('tex', mm['toon'])
        w.s(mm['memo']); w.i4(mm['fcount'])
    # bones
    w.i4(len(m.bones))
    for b in m.bones:
        w.s(b['name']); w.s(b['name_en']); w.vec(b['pos'])
        widx('bone', b['parent']); w.i4(b['layer'])
        fl = b['flag']; w.u2(fl)
        if fl & 0x0001: widx('bone', b['tail'])
        else: w.vec(b['tail_offset'])
        if fl & 0x0300: widx('bone', b['grant'][0]); w.f(b['grant'][1])
        if fl & 0x0400: w.vec(b['axis'])
        if fl & 0x0800: w.vec(b['local_x']); w.vec(b['local_z'])
        if fl & 0x2000: w.i4(b['ext_key'])
        if fl & 0x0020:
            widx('bone', b['ik_target']); w.i4(b['ik_loop']); w.f(b['ik_limit'])
            w.i4(len(b['ik_links']))
            for li, lo in b['ik_links']:
                widx('bone', li)
                if lo is None: w.u1(0)
                else: w.u1(1); w.vec(lo[0]); w.vec(lo[1])
    # morphs
    w.i4(len(m.morphs))
    for mo in m.morphs:
        w.s(mo['name']); w.s(mo['name_en']); w.u1(mo['panel']); w.u1(mo['type'])
        offs = mo['offsets']; t = mo['type']
        w.i4(len(offs))
        for o in offs:
            if t == 0: widx('morph', o[0]); w.f(o[1])
            elif t == 1: widx('vtx', o[0]); w.vec(o[1])
            elif t == 2: widx('bone', o[0]); w.vec(o[1]); w.vec(o[2])
            elif t in (3,4,5,6,7): widx('vtx', o[0]); w.vec(o[1])
            elif t == 8:
                widx('mat', o[0]); w.u1(o[1]); w.vec(o[2]); w.vec(o[3]); w.f(o[4])
                w.vec(o[5]); w.vec(o[6]); w.f(o[7]); w.vec(o[8]); w.vec(o[9]); w.vec(o[10])
            elif t == 9: widx('morph', o[0]); w.f(o[1])
            elif t == 10: widx('rb', o[0]); w.u1(o[1]); w.vec(o[2]); w.vec(o[3])
    # frames
    w.i4(len(m.frames))
    for fr in m.frames:
        w.s(fr['name']); w.s(fr['name_en']); w.u1(fr['special'])
        w.i4(len(fr['items']))
        for k, i in fr['items']:
            w.u1(k); widx('bone' if k == 0 else 'morph', i)
    # rigid bodies
    w.i4(len(m.rigids))
    for r in m.rigids:
        w.s(r['name']); w.s(r['name_en']); widx('bone', r['bone'])
        w.u1(r['group']); w.u2(r['nocollide']); w.u1(r['shape']); w.vec(r['size'])
        w.vec(r['pos']); w.vec(r['rot']); w.f(r['mass']); w.f(r['lin_damp'])
        w.f(r['ang_damp']); w.f(r['restitution']); w.f(r['friction']); w.u1(r['mode'])
    # joints
    w.i4(len(m.joints))
    for j in m.joints:
        w.s(j['name']); w.s(j['name_en']); w.u1(j['type'])
        widx('rb', j['a']); widx('rb', j['b']); w.vec(j['pos']); w.vec(j['rot'])
        w.vec(j['lin_min']); w.vec(j['lin_max']); w.vec(j['ang_min']); w.vec(j['ang_max'])
        w.vec(j['spring_lin']); w.vec(j['spring_ang'])
    with open(path, 'wb') as fh:
        fh.write(bytes(w.b))
    return len(w.b)
