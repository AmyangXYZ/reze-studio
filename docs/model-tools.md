# Model tools: handoff notes

The Python under `tools/` builds and repairs the PMX models the studio loads. The models
themselves stay local (`public/models/` is ignored); these scripts are the record of how
each was made, and most rebuild one with a single command.

Everything runs from the repo root with `python3` and numpy, scipy and Pillow. Each
script's docstring says what it does and why; this page is the map.

## Reading and writing PMX

- `pmx.py` reads a PMX into plain lists and dicts; `pmxw.py` writes one back. The pair
  round-trips `reze.pmx` byte-identical, so a script that writes only what it changed
  leaves the rest exactly as it was.
- A rigid body's `nocollide` field holds the collides-with mask as stored: the engine
  pairs two bodies only when each one's mask holds the other's group bit
  (`node_modules/reze-engine/dist/physics/body.js`).
- `holefill.slices(m)` gives each material's triangles; `roundring.shells(V, T)` splits
  triangles into connected shells, welded by position.
- `preview.py <pmx> out.png [ymin ymax]` renders front, side and back. It colours per
  vertex, with no transparency and no shading groups. It shows shape and placement, and
  it cannot judge colour, eyes or skin layers.

## Head swaps: reze, or any head, on another body

`headswap.py` puts a head on a 深空之眼 body; `swaps.py` holds every swap's recipe, so
they all rebuild together:

    python3 tools/zzz2pmx/swaps.py reze-tropic makima

The series shares one body frame, so the host stays as shipped apart from its head.
`headgraft.py` places the head by fitting the two faces at the head's own size,
`chaingraft.py` carries its hair chains (bones, rigid bodies, joints), and `prune`,
`fixrig` and `publish` run last. Recipe options worth knowing:

| option | for |
| --- | --- |
| `rename` | material names the studio's classifier understands (`皮肤 -> skin`, `袜子 -> 袜子_stockings`) |
| `scale=neck_scale` / `ratio_scale` | a smaller host brought to 塞勒涅's neck, or to reze-maid's ankle, hip and neck (`pmxscale.knots`) |
| `seated=True` | a host whose face is another size: seat the head on the neck opening instead of the face |
| `hat` | a hat inside a body material, split out by the bones it rides |
| `head`, `take` | another model's head instead of reze's (Makima) |
| `pre`, `post` | steps before and after: build a head, fit headwear to the new head (`headwear_on_reze`) |

## Repairs on a built model, in place

| tool | does |
| --- | --- |
| `paintsplit.py` | clothing painted onto a skin material into its own material: by bone, height and shell size, triangles mostly not skin-coloured; `--refine` splits triangles along the painted edge, `--whole` takes a selection the paint cannot tell from skin, `--merge` undoes it |
| `collide.py` | moves bodies into a collision group and says what they collide with |
| `feet.py` | stands a model on the floor and puts its toe IK targets at the toe tip |
| `headwear.py` | moves (and scales) headwear with its bones and bodies |
| `headsize.py`, `morphfix.py` | bring a head to reze's size; check or repair expression morphs against `reze.pmx` |
| `fixrig.py` | moves weight off bones that have a granted `D` twin; always last |

## A new character from parts: `hairmix.py`

    python3 tools/zzz2pmx/hairmix.py makima [out-dir]

A host that keeps its own face and morphs, its hair off (or trimmed), hair pieces from
other models grafted on with their chains, every hair sheet through one colour ramp, and
an iris from `iris.py`. Makima's head is built this way, then `swaps.py makima` puts it
on 塞尔凯特's body.

## Lessons that cost time

**Looks are judged in the studio.** The preview renderer and every metric here miss
toon shading, alpha and the shader groups. Make one change, hand it over, wait for a
screenshot. Reze-design's Aether Gazer style shades the Face and Body graphs differently,
so texture colours that match on paper do not match on screen.

**The studio caches textures by file name.** A recoloured sheet under its old name shows
the old colours. `hairmix` puts a hash of the colours in every sheet name.

**Copied materials carry indices into the donor's texture list.** A material's `sph` and
`toon` must be re-pointed by file, or they land on whatever the host has at that index
(Makima's braid once wore the host's skin as a sphere map).

**Grafted chains need their collisions confined.** A donor's "collide with everything"
means its own body there, and on a new host it also catches the host's earrings, neck
ornaments and clothing ribbons. A host head body a strand starts inside pushes it away
from the first frame. `hairmix.confine` handles both.

**Anything that touches the body runs after the swap**, and `fixrig` runs last of all:
every graft can put weight back on the IK effector.

**Hand-tuned models are edited only where asked.** `reze.pmx` carries tuning that exists
nowhere else. Record the md5 before and after, keep an untouched copy somewhere that
outlives the session, and prove the rest unchanged.

**Measure before you cut.** Read the mesh, the bones and the texture before deciding;
render the result and look at it before trusting a number. A per-pixel key can flicker
across a painted texture's grain and mottle it; a material edge can only follow
triangle edges.

**Physics can be tested headless.** reze-engine's `PmxLoader` and `RezePhysics` run in
Node with the studio's resolve hook (`node --import ./lib/__tests__/register.mjs`).
`model.update()` resets local rotations, so drive bones with `setBoneRotationOffset`, and
switch IK off or the foot IK holds the legs in place.
