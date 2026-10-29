"""Runs inside Blender: render the named objects of a .blend from a few angles.

    Blender -b <file.blend> --python tools/maid/render.py -- <out.png> [obj,obj,...]
"""
import sys, os, math
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index('--')+1:]
OUT = os.path.abspath(argv[0])
WANT = argv[1].split(',') if len(argv) > 1 and argv[1] else None
VIEWS = [('front', 0.0), ('side', math.pi/2), ('back', math.pi)]
SIZE = int(os.environ.get('SIZE', 560))

for ob in list(bpy.data.objects):
    if ob.type == 'MESH':
        ob.hide_render = bool(WANT) and ob.name not in WANT
    else:
        ob.hide_render = True

shown = [o for o in bpy.data.objects if o.type == 'MESH' and not o.hide_render]
if not shown:
    raise SystemExit(f'nothing to render; have {[o.name for o in bpy.data.objects]}')
pts = [o.matrix_world @ Vector(c) for o in shown for c in o.bound_box]
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
z0 = float(os.environ.get('ZMIN', lo.z))
z1 = float(os.environ.get('ZMAX', hi.z))
if 'ZMIN' in os.environ or 'ZMAX' in os.environ:
    lo.z, hi.z = z0, z1
    span = max(hi.x - lo.x, hi.z - lo.z)
    mid = (lo + hi) / 2
else:
    mid, span = (lo + hi) / 2, max(hi - lo)

scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
scene.display.shading.light = 'STUDIO'
scene.display.shading.color_type = 'TEXTURE'
scene.display.shading.show_cavity = True
scene.render.film_transparent = False
scene.world = scene.world or bpy.data.worlds.new('w')
scene.world.color = (0.09, 0.09, 0.11)
scene.render.resolution_x = scene.render.resolution_y = SIZE
scene.render.image_settings.file_format = 'PNG'

cam_data = bpy.data.cameras.new('cam')
cam_data.type = 'ORTHO'
cam_data.ortho_scale = span * 1.06
cam = bpy.data.objects.new('cam', cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

tiles = []
for name, ang in VIEWS:
    r = span * 3
    cam.location = mid + Vector((math.sin(ang) * r, -math.cos(ang) * r, 0))
    cam.rotation_euler = (math.pi/2, 0, ang)
    scene.render.filepath = OUT.replace('.png', f'_{name}.png')
    bpy.ops.render.render(write_still=True)
    tiles.append(scene.render.filepath)

# stitch
import gpu  # noqa: F401  (ensures image module is ready)
imgs = [bpy.data.images.load(t) for t in tiles]
w, h = imgs[0].size
out = bpy.data.images.new('sheet', width=w * len(imgs), height=h)
buf = [0.0] * (w * len(imgs) * h * 4)
for k, im in enumerate(imgs):
    px = list(im.pixels)
    for y in range(h):
        row = px[y*w*4:(y+1)*w*4]
        at = (y * w * len(imgs) + k * w) * 4
        buf[at:at+w*4] = row
out.pixels = buf
out.filepath_raw = OUT
out.file_format = 'PNG'
out.save()
for t in tiles:
    os.remove(t)
print(f'### wrote {OUT}')
