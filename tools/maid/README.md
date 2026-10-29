# 家政小帮手's outfit on reze-midnight

Blender 5.2 + mmd_tools. Each step reads the previous step's `.blend`, prints what it
measured, and writes the next one, so any stage can be opened and looked at.

```sh
B=/Applications/Blender.app/Contents/MacOS/Blender
$B --background --python tools/maid/step1_extract.py -- /tmp/step1.blend
$B -b /tmp/step1.blend  --python tools/maid/step2_fit.py     -- /tmp/step2.blend
$B -b /tmp/step2.blend  --python tools/maid/step3_deform.py  -- /tmp/step3.blend
$B -b /tmp/step3.blend  --python tools/maid/step4_clear.py   -- /tmp/step4.blend 0.015
$B -b /tmp/step4.blend  --python tools/maid/step4b_patch.py  -- /tmp/step4b.blend
$B -b /tmp/step4b.blend --python tools/maid/step5_undress.py -- /tmp/step5.blend
$B -b /tmp/step5.blend  --python tools/maid/step6_rig.py     -- \
    public/models/reze-maid-blender/reze-maid.pmx /tmp/step6.blend
```

`render.py` draws any of them: `$B -b /tmp/step5.blend --python tools/maid/render.py --
out.png maid_cloth,target_mesh`, with `ZMIN`/`ZMAX`/`SIZE` in the environment to frame a
region.

## Why it fits

After one uniform scale the two bodies agree to **0.023 median, 0.434 max**, with zero
correspondences longer than 1.2. They are the same 深空之眼 base body at different
sizes, so the whole job is a 10% scale and a small shape delta, and a Surface Deform
bound to the body the outfit was authored on carries it across with its own drape
intact.

The same measurement separates the bones. Carried through the same deform, a bone that
is really the same bone lands on the target's own -- `頭` 0.058, `上半身` 0.135, out to
`胸` at 0.436. A bone that only shares a name -- both models number a skirt
`skirt_<panel>_<link>` -- lands 1.755 away at the nearest. Nothing in between.

## Things that cost time

- Corrective Smooth after an *applied* Shrinkwrap has no deformation left to read, so
  it smooths the mesh itself: 11028 vertices moved and the drape went with them.
- `BVHTree` sign tests need `sub.normal_update()` first, or every point reads outside.
- mmd_tools parents each piece to a root empty; setting a scale on the piece *and* the
  root applies it twice.
- Working copies must have `shape_key_clear()`. A `bmesh` round-trip leaves imported
  morphs stale, the evaluated mesh then disagrees with the vertices, and Surface Deform
  binds to the disagreement.
- Both models ship a `cloth01.png`, and they are different sheets. The donor's are
  renamed on the way out or the target's skin is drawn with the maid's dress.
- Opening a `.blend` restores its preferences and turns the extension back off.

## Open

The maid's skirt has no physics: the rigid bodies and joints on her 339 outfit bones
are still on the donor side. What the model carries is reze-midnight's own -- 272
bodies, 422 joints, hair included.
