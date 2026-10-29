import struct, sys, io

import numpy as np

class R:
    def __init__(self, data):
        self.d = data; self.i = 0
    def u1(self):
        v = self.d[self.i]; self.i += 1; return v
    def i1(self):
        v = struct.unpack_from('<b', self.d, self.i)[0]; self.i += 1; return v
    def u2(self):
        v = struct.unpack_from('<H', self.d, self.i)[0]; self.i += 2; return v
    def i2(self):
        v = struct.unpack_from('<h', self.d, self.i)[0]; self.i += 2; return v
    def i4(self):
        v = struct.unpack_from('<i', self.d, self.i)[0]; self.i += 4; return v
    def u4(self):
        v = struct.unpack_from('<I', self.d, self.i)[0]; self.i += 4; return v
    def f(self):
        v = struct.unpack_from('<f', self.d, self.i)[0]; self.i += 4; return v
    def vec(self, n):
        v = struct.unpack_from('<%df' % n, self.d, self.i); self.i += 4*n; return list(v)
    def raw(self, n):
        v = self.d[self.i:self.i+n]; self.i += n; return v

class PMX:
    def __init__(self, path):
        with open(path,'rb') as fh: data = fh.read()
        r = R(data); self.r = r
        assert r.raw(4) == b'PMX ', 'not pmx'
        self.version = r.f()
        gcount = r.u1()
        g = r.raw(gcount)
        self.encoding = g[0]          # 0=utf16le 1=utf8
        self.uv_add = g[1]
        self.sz = dict(vtx=g[2], tex=g[3], mat=g[4], bone=g[5], morph=g[6], rb=g[7])
        self.enc = 'utf-16-le' if self.encoding==0 else 'utf-8'
        self.name = self.s(); self.name_en = self.s()
        self.comment = self.s(); self.comment_en = self.s()
        self.read_vertices(); self.read_faces(); self.read_textures()
        self.read_materials(); self.read_bones(); self.read_morphs()
        self.read_frames(); self.read_rigid(); self.read_joints()

    def s(self):
        n = self.r.u4()
        return self.r.raw(n).decode(self.enc, errors='replace')

    def idx(self, kind):
        n = self.sz[kind]; r = self.r
        if kind == 'vtx':   # unsigned
            return {1:r.u1, 2:r.u2, 4:r.i4}[n]()
        return {1:r.i1, 2:r.i2, 4:r.i4}[n]()

    def read_vertices(self):
        """Vectorised. The records are variable-length -- the weight type decides the
        size -- so one Python pass walks the buffer for the offsets, reading a single
        byte each, and numpy lifts every field out of them at once."""
        r = self.r; n = r.i4(); self.vcount = n
        d = r.d
        bs = self.sz['bone']
        pre = 32 + 16 * self.uv_add
        rec = {0: pre + 1 + bs + 4, 1: pre + 1 + 2 * bs + 4 + 4,
               2: pre + 1 + 4 * bs + 16 + 4, 3: pre + 1 + 2 * bs + 4 + 36 + 4,
               4: pre + 1 + 4 * bs + 16 + 4}
        buf0 = np.frombuffer(d, np.uint8)
        t0 = d[r.i + pre]
        step = rec[t0]
        flat = (r.i + n * step <= len(d)
                and (buf0[r.i + pre: r.i + pre + n * step: step] == t0).all())
        if flat:
            # every record the same size, so there is nothing to walk
            off = r.i + np.arange(n, dtype=np.int64) * step
            wt = np.full(n, t0, np.uint8)
            r.i += n * step
        else:
            off = np.empty(n, np.int64); wt = np.empty(n, np.uint8)
            o = r.i
            for k in range(n):
                off[k] = o
                t = d[o + pre]
                wt[k] = t
                o += rec[t]
            r.i = o
        if n == 0:
            self.vertices = []
            return
        buf = np.frombuffer(d, np.uint8)

        def grab(at, size, dt):
            return buf[at[:, None] + np.arange(size)].copy().view(dt)

        head = grab(off, pre, np.float32)
        pos = head[:, 0:3].tolist(); nor = head[:, 3:6].tolist(); uv = head[:, 6:8].tolist()
        add = (head[:, 8:].reshape(n, self.uv_add, 4).tolist() if self.uv_add
               else [[] for _ in range(n)])
        span = (np.full(n, rec[t0], np.int64) if flat
                else np.array([rec[int(t)] for t in wt], np.int64))
        edge = grab(off + span - 4, 4, np.float32)[:, 0].tolist()
        bdt = {1: np.int8, 2: np.int16, 4: np.int32}[bs]

        B = [None] * n; W = [None] * n; S = {}
        for t in np.unique(wt):
            sel = np.where(wt == t)[0]
            at = off[sel] + pre + 1
            if t == 0:
                b = grab(at, bs, bdt).tolist()
                wv = [[1.0]] * len(sel)
            elif t in (1, 3):
                b = grab(at, 2 * bs, bdt).tolist()
                w0 = grab(at + 2 * bs, 4, np.float32)[:, 0]
                wv = np.stack([w0, 1 - w0], 1).tolist()
                if t == 3:
                    for k, v in zip(sel, grab(at + 2 * bs + 4, 36, np.float32).tolist()):
                        S[int(k)] = v
            else:
                b = grab(at, 4 * bs, bdt).tolist()
                wv = grab(at + 4 * bs, 16, np.float32).tolist()
            for k, bb, ww in zip(sel, b, wv):
                B[int(k)] = bb; W[int(k)] = ww
        self.vertices = [
            (pos[k], nor[k], uv[k], add[k], int(wt[k]), B[k], W[k], edge[k])
            if int(wt[k]) != 3 else
            (pos[k], nor[k], uv[k], add[k], 3, B[k], W[k], edge[k], S[k])
            for k in range(n)]

    def read_faces(self):
        r = self.r; n = r.i4()
        dt = {1: np.uint8, 2: np.uint16, 4: np.int32}[self.sz['vtx']]
        self.faces = np.frombuffer(r.d, dt, n, r.i).tolist()
        r.i += n * self.sz['vtx']

    def read_textures(self):
        n = self.r.i4(); self.textures = [self.s() for _ in range(n)]

    def read_materials(self):
        r = self.r; n = r.i4(); M = []
        for _ in range(n):
            m = {}
            m['name'] = self.s(); m['name_en'] = self.s()
            m['diffuse'] = r.vec(4); m['specular'] = r.vec(3); m['shininess'] = r.f()
            m['ambient'] = r.vec(3); m['flag'] = r.u1(); m['edge_color'] = r.vec(4)
            m['edge_size'] = r.f(); m['tex'] = self.idx('tex'); m['sph'] = self.idx('tex')
            m['sph_mode'] = r.u1(); m['toon_shared'] = r.u1()
            m['toon'] = r.u1() if m['toon_shared'] else self.idx('tex')
            m['memo'] = self.s(); m['fcount'] = r.i4()
            M.append(m)
        self.materials = M

    def read_bones(self):
        r = self.r; n = r.i4(); B = []
        for _ in range(n):
            b = {}
            b['name'] = self.s(); b['name_en'] = self.s()
            b['pos'] = r.vec(3); b['parent'] = self.idx('bone'); b['layer'] = r.i4()
            fl = r.u2(); b['flag'] = fl
            if fl & 0x0001: b['tail'] = self.idx('bone')
            else: b['tail_offset'] = r.vec(3)
            if fl & 0x0300: b['grant'] = (self.idx('bone'), r.f())
            if fl & 0x0400: b['axis'] = r.vec(3)
            if fl & 0x0800: b['local_x'] = r.vec(3); b['local_z'] = r.vec(3)
            if fl & 0x2000: b['ext_key'] = r.i4()
            if fl & 0x0020:
                b['ik_target'] = self.idx('bone'); b['ik_loop'] = r.i4(); b['ik_limit'] = r.f()
                lc = r.i4(); links=[]
                for _ in range(lc):
                    li = self.idx('bone'); has = r.u1()
                    lo = (r.vec(3), r.vec(3)) if has else None
                    links.append((li, lo))
                b['ik_links'] = links
            B.append(b)
        self.bones = B

    def read_morphs(self):
        r = self.r; n = r.i4(); Mo = []
        for _ in range(n):
            m = {}
            m['name'] = self.s(); m['name_en'] = self.s()
            m['panel'] = r.u1(); m['type'] = r.u1(); c = r.i4()
            t = m['type']; off=[]
            for _ in range(c):
                if t==0: off.append((self.idx('morph'), r.f()))
                elif t==1: off.append((self.idx('vtx'), r.vec(3)))
                elif t==2: off.append((self.idx('bone'), r.vec(3), r.vec(4)))
                elif t in (3,4,5,6,7): off.append((self.idx('vtx'), r.vec(4)))
                elif t==8:
                    off.append((self.idx('mat'), r.u1(), r.vec(4), r.vec(3), r.f(), r.vec(3),
                                r.vec(4), r.f(), r.vec(4), r.vec(4), r.vec(4)))
                elif t==9: off.append((self.idx('morph'), r.f()))
                elif t==10: off.append((self.idx('rb'), r.u1(), r.vec(3), r.vec(3)))
                else: raise ValueError('bad morph type %d' % t)
            m['offsets'] = off
            Mo.append(m)
        self.morphs = Mo

    def read_frames(self):
        r = self.r; n = r.i4(); F=[]
        for _ in range(n):
            f={}; f['name']=self.s(); f['name_en']=self.s(); f['special']=r.u1()
            c=r.i4(); items=[]
            for _ in range(c):
                k=r.u1()
                items.append((k, self.idx('bone') if k==0 else self.idx('morph')))
            f['items']=items; F.append(f)
        self.frames=F

    def read_rigid(self):
        r=self.r; n=r.i4(); R_=[]
        for _ in range(n):
            b={}; b['name']=self.s(); b['name_en']=self.s(); b['bone']=self.idx('bone')
            b['group']=r.u1(); b['nocollide']=r.u2(); b['shape']=r.u1(); b['size']=r.vec(3)
            b['pos']=r.vec(3); b['rot']=r.vec(3); b['mass']=r.f(); b['lin_damp']=r.f()
            b['ang_damp']=r.f(); b['restitution']=r.f(); b['friction']=r.f(); b['mode']=r.u1()
            R_.append(b)
        self.rigids=R_

    def read_joints(self):
        r=self.r; n=r.i4(); J=[]
        for _ in range(n):
            j={}; j['name']=self.s(); j['name_en']=self.s(); j['type']=r.u1()
            j['a']=self.idx('rb'); j['b']=self.idx('rb'); j['pos']=r.vec(3); j['rot']=r.vec(3)
            j['lin_min']=r.vec(3); j['lin_max']=r.vec(3); j['ang_min']=r.vec(3); j['ang_max']=r.vec(3)
            j['spring_lin']=r.vec(3); j['spring_ang']=r.vec(3)
            J.append(j)
        self.joints=J
