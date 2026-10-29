# -*- coding: utf-8 -*-
"""Scale a model uniformly about the origin and shift it: everything with a length.

Vertices, SDEF points, bones and their tail offsets, rigid bodies and their sizes,
joints and their linear limits, vertex-morph offsets and bone-morph translations.
Angles, masses, damping and springs are left as they are.
"""
import numpy as np


def scale(m, k, shift=(0.0, 0.0, 0.0)):
    sh = np.array(shift, float)

    def pt(p):
        return (np.array(p, float) * k + sh).tolist()

    def ln(p):
        return (np.array(p, float) * k).tolist()

    for i, v in enumerate(m.vertices):
        v = list(v)
        v[0] = pt(v[0])
        if v[4] == 3:
            c = list(v[8])
            c[0:3], c[3:6], c[6:9] = pt(c[0:3]), pt(c[3:6]), pt(c[6:9])
            v[8] = c
        m.vertices[i] = tuple(v)
    for b in m.bones:
        b['pos'] = pt(b['pos'])
        if 'tail_offset' in b:
            b['tail_offset'] = ln(b['tail_offset'])
    for r in m.rigids:
        r['pos'] = pt(r['pos'])
        r['size'] = ln(r['size'])
    for j in m.joints:
        j['pos'] = pt(j['pos'])
        j['lin_min'] = ln(j['lin_min'])
        j['lin_max'] = ln(j['lin_max'])
    for mo in m.morphs:
        if mo['type'] == 1:
            mo['offsets'] = [(o[0], ln(o[1])) for o in mo['offsets']]
        elif mo['type'] == 2:
            mo['offsets'] = [(o[0], ln(o[1]), o[2]) for o in mo['offsets']]


def profile(m, foot, body, ramp, centre_z=0.0):
    """Scale by height: the feet stay their size (`foot`, up to y ramp[0]), the body
    above is `body` times larger (from y ramp[1] up), and the shin between blends.
    Widths scale about the centre line (x 0, z `centre_z`), which then moves to z 0;
    heights accumulate, so a vertex keeps its place in the figure."""
    ya, yb = ramp

    def s(y):
        if y <= ya:
            return foot
        if y >= yb:
            return body
        return foot + (body - foot) * (y - ya) / (yb - ya)

    def Y(y):
        if y <= ya:
            return y * foot
        if y <= yb:
            return ya * foot + (y - ya) * (foot + s(y)) / 2
        return ya * foot + (yb - ya) * (foot + body) / 2 + (y - yb) * body

    def pt(p):
        x, y, z = p
        k = s(y)
        return [x * k, Y(y), (z - centre_z) * k]

    def ln(p, y):
        return (np.array(p, float) * s(y)).tolist()

    V = [v[0] for v in m.vertices]
    for i, v in enumerate(m.vertices):
        v = list(v)
        y = v[0][1]
        v[0] = pt(v[0])
        if v[4] == 3:
            c = list(v[8])
            c[0:3], c[3:6], c[6:9] = pt(c[0:3]), pt(c[3:6]), pt(c[6:9])
            v[8] = c
        m.vertices[i] = tuple(v)
    for b in m.bones:
        y = b['pos'][1]
        b['pos'] = pt(b['pos'])
        if 'tail_offset' in b:
            b['tail_offset'] = ln(b['tail_offset'], y)
    for r in m.rigids:
        y = r['pos'][1]
        r['pos'] = pt(r['pos'])
        r['size'] = ln(r['size'], y)
    for j in m.joints:
        y = j['pos'][1]
        j['pos'] = pt(j['pos'])
        j['lin_min'] = ln(j['lin_min'], y)
        j['lin_max'] = ln(j['lin_max'], y)
    for mo in m.morphs:
        if mo['type'] == 1:
            mo['offsets'] = [(o[0], ln(o[1], V[o[0]][1])) for o in mo['offsets']]
        elif mo['type'] == 2:
            mo['offsets'] = [(o[0], ln(o[1], m.bones[o[0]]['pos'][1]), o[2]) for o in mo['offsets']]
    return Y


def knots(m, width, pts, centre_z=0.0, blend=0.6):
    """Scale widths by `width` and remap heights through `pts`, (host y, target y)
    pairs starting at (0, 0): each span between two knots stretches by its own rate,
    corners rounded over `blend`, and above the last knot the rate is `width`.
    A figure of other proportions gets another's hip, neck and ankle heights."""
    ys = np.array([p[0] for p in pts], float)
    Ys = np.array([p[1] for p in pts], float)
    rates = np.diff(Ys) / np.diff(ys)
    g = np.linspace(-5.0, 40.0, 45001)
    dy = g[1] - g[0]
    r = np.where(g < ys[0], rates[0], width)
    for a, b, k in zip(ys[:-1], ys[1:], rates):
        r = np.where((g >= a) & (g < b), k, r)
    w = max(1, int(round(blend / dy)))
    r = np.convolve(np.pad(r, w, mode='edge'), np.ones(2 * w + 1) / (2 * w + 1), mode='same')[w:-w]
    G = np.cumsum(r) * dy
    G -= np.interp(0.0, g, G)

    def Y(y):
        return float(np.interp(y, g, G))

    def rate(y):
        return float(np.interp(y, g, r))

    def pt(p):
        x, y, z = p
        return [x * width, Y(y), (z - centre_z) * width]

    def ln(p, y):
        return [p[0] * width, p[1] * rate(y), p[2] * width]

    V = [v[0] for v in m.vertices]
    By = [b['pos'][1] for b in m.bones]
    for i, v in enumerate(m.vertices):
        v = list(v)
        v[0] = pt(v[0])
        if v[4] == 3:
            c = list(v[8])
            c[0:3], c[3:6], c[6:9] = pt(c[0:3]), pt(c[3:6]), pt(c[6:9])
            v[8] = c
        m.vertices[i] = tuple(v)
    for b in m.bones:
        y = b['pos'][1]
        b['pos'] = pt(b['pos'])
        if 'tail_offset' in b:
            b['tail_offset'] = ln(b['tail_offset'], y)
    for rb in m.rigids:
        rb['pos'] = pt(rb['pos'])
        rb['size'] = (np.array(rb['size'], float) * width).tolist()
    for j in m.joints:
        y = j['pos'][1]
        j['pos'] = pt(j['pos'])
        j['lin_min'] = ln(j['lin_min'], y)
        j['lin_max'] = ln(j['lin_max'], y)
    for mo in m.morphs:
        if mo['type'] == 1:
            mo['offsets'] = [(o[0], ln(o[1], V[o[0]][1])) for o in mo['offsets']]
        elif mo['type'] == 2:
            mo['offsets'] = [(o[0], ln(o[1], By[o[0]]), o[2]) for o in mo['offsets']]
    return Y
