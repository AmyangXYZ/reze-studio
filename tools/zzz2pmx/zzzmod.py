"""Read a 3DMigoto/XXMI ZZZ mod folder into meshes."""
import os
import re
import numpy as np

POS_STRIDE = 40      # POSITION float3, NORMAL float3, TANGENT float4
BLEND_STRIDE = 32    # BLENDWEIGHTS float4, BLENDINDICES uint4


class Component:
    """One vertex-buffer group (Body, Hair, ...) plus its index-buffer draws."""

    def __init__(self, name, folder):
        self.name = name
        p = np.fromfile(os.path.join(folder, f'Evelyn{name}Position.buf'), np.uint8).reshape(-1, POS_STRIDE)
        self.pos = p[:, 0:12].copy().view(np.float32).reshape(-1, 3)
        self.normal = p[:, 12:24].copy().view(np.float32).reshape(-1, 3)
        self.tangent = p[:, 24:40].copy().view(np.float32).reshape(-1, 4)

        b = np.fromfile(os.path.join(folder, f'Evelyn{name}Blend.buf'), np.uint8).reshape(-1, BLEND_STRIDE)
        self.weights = b[:, 0:16].copy().view(np.float32).reshape(-1, 4)
        self.bones = b[:, 16:32].copy().view(np.uint32).reshape(-1, 4)

        t = np.fromfile(os.path.join(folder, f'Evelyn{name}Texcoord.buf'), np.uint8)
        stride = len(t) // len(self.pos)
        t = t.reshape(-1, stride)
        self.color = t[:, 0:4] / 255.0
        self.uv = np.ascontiguousarray(t[:, 4:8]).view(np.float16).reshape(-1, 2).astype(np.float32)
        self.texcoord_stride = stride

    def __len__(self):
        return len(self.pos)


def load(folder):
    comps = {}
    for name in ('Body', 'Hair', 'Jacket', 'Shoulder'):
        if os.path.exists(os.path.join(folder, f'Evelyn{name}Position.buf')):
            comps[name] = Component(name, folder)
    return comps


def ib(folder, name):
    return np.fromfile(os.path.join(folder, f'Evelyn{name}.ib'), np.uint32).reshape(-1, 3)
