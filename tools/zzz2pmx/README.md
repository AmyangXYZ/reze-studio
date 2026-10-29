# zzz2pmx

Rebuilds the Reze mod for ZZZ's 伊芙琳 as an MMD model.

```
python3 tools/zzz2pmx/build.py --variant casual   # public/models/reze-zzz/reze-casual.pmx
python3 tools/zzz2pmx/build.py --variant bomb     # public/models/reze-zzz/reze-bomb.pmx
python3 tools/zzz2pmx/validate.py --against public/models/伊芙琳_by_绝区零/伊芙琳.pmx \
    public/models/reze-zzz/reze-casual.pmx
python3 tools/zzz2pmx/preview.py <pmx> out.png [ymin ymax]
python3 tools/zzz2pmx/graft.py                    # the outfit onto the base model
python3 tools/zzz2pmx/bombgraft.py                # then the bomb form onto that

# the mod's haircut instead of the model's own, kept as a variant to revisit
python3 tools/zzz2pmx/hairgraft.py --out public/models/reze/_zzzhair.pmx
python3 tools/zzz2pmx/graft.py --target public/models/reze/_zzzhair.pmx \
    --out public/models/reze/reze-nomorph-zzzhair.pmx
```

| model | what it is |
| --- | --- |
| `reze-zzz.pmx` | the model's own haircut, wearing the mod's outfit |
| `reze-zzzhair.pmx` | the same with the mod's haircut and recovered sway |
| `reze-bomb.pmx` | the bomb form, built on the first |
| `reze-mirror-hair.pmx` | `reze.pmx` with its own haircut flipped left-for-right |

`hairphys.py` doubles as a CLI that repairs hair collision in a built model:

    python3 tools/zzz2pmx/hairphys.py <pmx> ...

This model ships with the three spheres on `頭` set to `nocollide 0xFFFF` — not
"collides with the body" but *collides with nothing*, so there is no skull under the
hair and it sinks through as soon as anything moves. Its strands also collide with each
other, which short chains far apart get away with and longer ones do not. The repair
clears bit 9 on the skull so it meets the hair group and nothing else, sets bit 9 on the
hair to drop the self-collision, and trims any sphere a strand already starts inside,
because a body born inside a collider is ejected on the first frame.

`mirrorhair.py` makes the last of those, and applies the same repair. Mirroring the mesh alone leaves every strand
driven from the far side of the head, and the chains cannot simply be swapped — `yb`
has six links and `zb` five, so there is no partner map. The flipped hair is rebound by
distance to the chains instead, which puts each strand on whichever one now runs down
its side. Only the hair is written; bones, rigid bodies and joints come through as they
are.

`graft.py` reads a base model that is no longer in the tree — the build without a morph
baked into the rest pose. Restore it with

    git checkout 47124a3 -- public/models/reze/reze-new-nomorph.pmx

and point `--target` at it. `bombgraft.py` needs only `reze-zzz.pmx` and runs as it
stands.

Inputs are `zzz-mod/EvelynMod` and `public/models/伊芙琳_by_绝区零/伊芙琳.pmx`; neither is
written to, so runs are idempotent.

## Reze's head on a 深空之眼 body

```
python3 tools/zzz2pmx/swaps.py [name ...]   # ~5s each -> public/models/reze-<name>/
```

The 深空之眼 adult figures share one body, so an outfit that exists on one is reze's
by swapping the head: `headswap.main(host, out, drop, rename, hat, wear)` checks the
host's frame against 塞勒涅's, drops its head materials by name, renames its chains out
of the way of reze's, grafts reze's head by the host's face, pushes reze's hair under
whatever the host wears on the head, prunes the rig and publishes. Every host is listed
in `swaps.py` with its own drop list and renames for the studio's styling; a hat that
lives inside a body-wide cloth material (霍德尔's veil in `cloth01`) is split off with
`hat={'cloth01': 'veil_cloth'}` -- the shells anchored on 頭 become the hat, shells
riding the host's hair bones go with the hair -- and named as headwear with `wear`.

A host of another size is scaled first: `pmxscale.scale` uniformly, or
`pmxscale.profile` by height -- 芭丝特's feet are already 塞勒涅's size, so her body
above the shin is scaled x1.092 to put the neck at 塞勒涅's and the feet stay; the
figure is centred from where she stood. Bones, rigid bodies and their sizes, joints
and their limits, and every morph offset scale with the mesh.

Reze's head goes on at its own size: the graft runs with `--scale 1` and reads only
where the head sits off the host's face, so every reze model has the same head.
`headsize.py` brings an already published model there in place -- its head materials,
hair bones, bodies, joints and expressions scaled about the face centre -- for a model
whose host is gone.

What the host wears on the head stays where the host's hair crown was; `headwear.py`
moves those materials, with the bones, bodies and joints only they use, by reze's
crown minus the host's.

Expressions come across in two passes: vertex and material morphs first, then the
groups (まばたき is both winks), numbered against the list actually kept. `morphfix.py`
checks any shipped model's groups against reze.pmx by name, and repairs one whose
build chain can no longer run.

## Reze's clothes on the bare model

```
python3 tools/zzz2pmx/casual.py    # ~7s -> public/models/reze-casual-selene/
```

reze.pmx's `shirt`, `shorts` and `ribbon` on `reze-bare`, put there by `dress.py` the
way a dresser does it: by bone. Every garment bone resolves to a bone of the target;
each carries the rotation and offset between the two skeletons and a *girth* scale --
the ratio of the two skins' radial size around that bone, perpendicular to the segment
only, so a sleeve narrows to the thinner arm without changing length. A vertex goes
where its weighted bones send it. Then two clearance passes: sunk vertices are lifted
out of the skin (blurred, so no crease), and where the skin still shows *between* the
garment's vertices -- her fuller bust under the coarser shirt -- the triangles over it
are lifted by the depth it shows until nothing shows. Nothing is projected onto the
skin: a per-vertex conform takes a collar's standoff along the jaw's normal and
crumples it, which is how the first build's neck went wrong.

## Reze in two

```
python3 tools/zzz2pmx/behead.py    # ~2s -> public/models/reze-headless/, public/models/reze-head/
```

reze.pmx cut at the neck by one horizontal plane (y 18.35, just above the choker). The
skin crosses it, and on the head so does the chin's underside: those triangles are
split on the plane, every attribute and every expression interpolated along the cut
edge, and each section gets a flat lid -- the plane's vertices triangulated in the
plane, the triangles inside the section kept by a crossing-parity vote in four
directions, since the skin's throat is open under the chin and a gap fools one
direction, never three. The body's throat-front is fanned shut. Hair, eyes and mouth go
whole to the head, the clothes whole to the body; both keep the whole skeleton, and
each keeps only its own chains and bodies.

## The bare model

```
python3 tools/zzz2pmx/bare.py      # ~16s -> public/models/reze-bare/
```

塞勒涅 made whole, to dress in whatever comes next: her body, breast, hips, foot in her
own shoes, rig and sheet. **Nothing of hers is cut.** What her costume took with it is
put back, each piece by the one method that fits it, and `check()` runs after every
step listing every open rim on the body; the build is not done until that list is only
the face's own eye and mouth openings.

| what was missing | how it is put back |
| --- | --- |
| belly — a hole, y 14.5–16.0, with a 0.2 slit to the breast shell above | 海滩漫步's surface inside the hole, fitted, sewn to the hole's rim by `stitch`; the slit zipped by `zip_arcs` — both sides as open arcs walked together along x; UVs interpolated from her own rims into her own atlas, which is clean skin under the hole. One material, one sheet |
| forearms and hands — hers are gloves | 海滩漫步's from the elbow out as `skin_hands`, so his wrist meets his own hand; one ring sewn at the elbow; her material colours |
| her painted underwear | `underwear`, an alpha overlay the way models ship it: the triangles touching the black are duplicated a hair off the surface, sampling an RGBA copy of her original sheet whose alpha is the black itself (feathered); the black on the skin sheet beneath is filled back to skin, so the overlay can be hidden. Whole triangles cut out instead carry skin outside the black and show a saw-tooth fringe |
| small holes and strap channels | `slits` zips any two small rims beside each other; `cap` fans the rest, on doubled vertices carrying a neighbouring skin texel's UV |
| neck — the head graft places, it does not sew | her skin's neck ring is the open edge her own face was cut from; `neck` zips it to the face's jaw arc where the two run within reach, then `tuck_face` sets the face's remaining free rims (the scalp, under the hair) a little under her skin |
| hair — her ornament chains share reze's fifteen chain-bone names | `rename_chains` first, so the graft carries reze's |
| costume paint — the sacral mark, the cuff line | `skinclean`: costume by hue, or a shadow against her own median, read off 海滩漫步 at its point in space and taken only if cleaner; what no donor has cleaner is filled from her own skin |

Rims are paired by the median distance between their vertices, never by their centres;
`zipper` closes every run where two rims lie within 0.35, `stitch` sews one whole rim to
another. Anything drawn across a rim gets **doubled vertices with one side's UVs** — a
triangle spanning two islands samples across the atlas and comes out black.

**Textures are read with v as-is.** PMX puts v = 0 on the top row. A reader that flips
v samples every sheet upside down and returns numbers that look plausible — that is
what the old mannequin atlas was built from. And **`preview.py` colours per vertex and skips edge-on triangles** —
it shows dark bands and a slot under the breast where the data has none. The arbiter is
`check()` (open rims, counted on welded positions) and a ray cast through the region;
both say closed before a render is allowed to say otherwise.

## Cutting painted clothes out of a skin mesh

`clothcut.py` splits a `skin` material along the painted edge of its clothes — refine
the crossed triangles, cut them on a smoothed class field, per vertex welded by
position so UV seams stay closed. On 大梵天皮肤 it produced a watertight split into
`cloth_bodysuit` and `lace`, and the result was rejected on sight; `reze-brahma` ships
uncut, with the bodysuit painted on `skin`. The tool is kept for a next attempt.

## Dressing a model in a donor's outfit

Four stages, each reading the one before and never writing it. Run them in order; the
last two are re-runnable on their own.

```
python3 tools/zzz2pmx/maidselene.py   # ~13s  fit the outfit on a closed mannequin,
                                      #       then swap the real body in under it
python3 tools/zzz2pmx/maiddress.py    # ~8s   carry the fitted outfit onto the build
                                      #       -> also snapshots it as placed.pmx
python3 tools/zzz2pmx/bustmerge.py    # ~4s   seat every layer against every other
python3 tools/zzz2pmx/publish.py      # ~2s   -> public/models/reze-maid/
```

`bustmerge` reads `placed.pmx` and writes `reze-maid.pmx`, so it can be run alone as
many times as it takes to settle a layer. Without that snapshot it wrote over its own
input and every tweak cost the whole chain.

### The passes that seat a layer

All of them live in `bustmerge.py`, and between them they replaced fourteen one-off
scripts that each fixed one garment in one place.

| pass | what it does |
| --- | --- |
| `lift` | raises a garment out of the body wherever it has sunk in |
| `cling` | sets a skin-tight piece **on** the body, both ways — proud and sunk alike |
| `outof` | raises a garment until nothing of the body comes through it, measured from the body's side |
| `sink` | sets the hidden body back under a garment, to a clearance that survives physics |
| `intoshoe` | puts the foot inside the leather rather than trimming what stands proud |
| `strip` | cuts the skin out from under an opaque garment, along that garment's own edge |
| `carry` | moves hardware — bells, hearts, buckles — with the garment it hangs on, per shell and rigid |
| `unjam` | brings a collider back to the overlap the donor itself ships with |

### Rules these were paid for

- **Anchor every measurement to a landmark, never to a coordinate.** Two models' chests
  sit 1.65 apart in y. Comparing the same y band on both produced a weight discrepancy
  that did not exist and sent a day down the wrong path.
- **Fit is measured from the body's side.** A garment can have every vertex outside the
  skin and still let the body through the faces between them — 1627 places on a piece
  reporting `inside 0/618` by its own vertices.
- **Weld before trusting a boundary.** A raw edge count calls UV seams openings. The
  hosiery's seam runs the length of the leg; guarding a cut against it guarded away the
  whole leg.
- **Move geometry, do not cut it.** Every snip left a hole. The only cut that works is a
  wholesale one along a garment's own edge, with the survivors brought up onto that edge
  so the cut cannot read as a comb.
- **Two surfaces a hundredth apart cannot both be drawn.** It shreds into strips under a
  camera and flickers under animation — which reads as a physics fault and is not one.
- **Resolve module constants inside the function, not in its signature.** A default binds
  at definition; a constant set afterwards is silently ignored. This has now cost two
  wrong conclusions.


## What the mod actually contains

`EvelynMod` is a 3DMigoto/XXMI dump: four vertex-buffer groups (Body, Hair, Jacket,
Shoulder) plus index buffers that carve materials out of them. `Evelyn.ini` names the
strides, and `EvelynMod/Evelyn.ini` — not the copy beside it, which is a stale build
whose counts no longer match the buffers — is the one that describes these files.

| buffer | stride | layout |
| --- | --- | --- |
| Position | 40 | POSITION float3, NORMAL float3, TANGENT float4 |
| Blend | 32 | BLENDWEIGHTS float4, BLENDINDICES uint4 |
| Texcoord | 20 | COLOR unorm4, **TEXCOORD half2**, then 8 bytes that are not a UV |

The UV is the half2 at byte 4. The float pair at byte 8 spans exactly [-1,1] and looks
like a UV until you check it against the index buffers: it is constant (0,1) across every
vertex BodyB draws, and edge lengths in it do not track edge lengths on the surface.

Two draws share the Body buffer and never share a vertex: `BodyA` (20038 verts) is the
jacket outfit, `BodyB` (29618) the bomb form. Jacket and Shoulder are dumped but never
drawn — the ini nulls their index buffers and only reads the jacket's draw call as the
signal for which body variant the game is in.

The mod has no face. ZZZ draws the head as its own call and the mod only swaps the
texture on it (`EvelynFaceADiffuse`), so `BodyA` carries the neck, jaw, ears and skull
with the face left open, and `BodyB` replaces the whole head with the bomb. The bomb form
also skips the face draw, which is why it gets no face plate and no facial morphs here.

## Placement

The rip and the mod share one surface, untouched by either: the head shell. Fitting the
mod's shell onto the rip's own head skin converges to **scale 12.5055, translate
(0, 0.3052, -0.0006)** with a median residual of 0.0015 — vertex-exact, and it is the
game→MMD transform for the whole model. Two independent checks fall out of it: the mod's
sole lands at Y 0.029 against the rip's 0.043, and the body spans 0.03–20.05 against the
rip's 0.04–20.06.

Fitting the *body* instead gives 13.64, and that number is wrong. Half of `BodyA`'s
vertices are arm and hand, ICP over a tube is set by the tube's radius rather than its
length, and Reze's arms are thicker than Evelyn's. Taken at 13.64 the head comes out 12%
oversized and the wrist bone sits 0.06 from the skin instead of inside the wrist.

Game axes are Z-up and the character faces -y; MMD is Y-up facing -Z, so positions and
normals map `(x,y,z) -> (-x,z,y)`.

Swapping the last two axes alone is the obvious map and it is wrong: determinant -1, a
reflection, so the model comes back with its left and right exchanged. The mod's body
geometry is symmetric to within 0.0002, so nothing about the shape gives it away, and the
reflected winding presents as inward-facing triangles — reverse those and the model looks
entirely correct. It reads only in the texture: Reze's choker pull-ring sits on her left
instead of her right. Negating x makes the map a rotation, the winding stands as dumped
(agreement 0.9999), and the ring is where it belongs. `align.py` asserts the determinant.

`align.py` also matches game bones to MMD bones by weighted vertex centroid, which seeds
the head fit. The correspondence it finds is clean and symmetric (game 86/8 → 左/右足首D,
64/74 → 右/左手首, fingers to fingers) — that is what establishes that the two skeletons
are one skeleton, and it is what the rig is then built on.

## Rig

The mod's own skin weights are carried over, with each game bone rewritten to the MMD
bone it is (`bonemap.py`). Bones are identified by the weighted centroid of the vertices
they drive, matched as a bijection with the Hungarian algorithm.

Two things make that match trustworthy. It runs on **mirror pairs** rather than single
bones, which doubles the evidence behind each decision and makes an asymmetric answer
unrepresentable — matching bones singly put a 467-mass bone on 左親指２ while its mirror
took 右親指２ at a third the mass. And it runs **per variant**, against host bones with
the accessory chains removed; over both variants at once the bomb form's dress had
nothing to match and landed on Evelyn's sword-sheath bone at a distance of 3.6.

86 of 112 game bones match directly, with the worst at 1.61. The remaining 26 are the
game's extra segments — it carries four thumb bones where MMD weights three — and they
merge into whichever matched bone they **share the most skin with**, resolved in
co-occurrence order so a chain can follow its parent. Nearest centroid is the wrong
question for this: the game's distal thumb bone sits nearest the thumb tip, and following
that put half a hand on a fingertip.

### Why not read the weights off the rip's surface

The first version of this did, by closest triangle with a normal gate. It is the obvious
approach and it is wrong here, because Evelyn wears gauntlets. Her gloved hand reaches
X 7.30 where Reze's bare hand stops at 6.34, so the nearest donor triangle to a point on
Reze's forearm is Evelyn's thumb. The result was static-perfect and broke the moment
anything rotated:

| region | closest-triangle transfer | carried over from the mod |
| --- | --- | --- |
| palm and knuckles | 小指２ 26%, 親指２ 21%, 小指１ 18% | 小指３ 18%, 手首 13%, 小指１/人指１/中指１/薬指１ |
| fingers | 小指３ 46%, 親指２ 18% | 人指３ 32%, 中指３ 27%, 薬指３ 26% |

Nothing about this shows up in a rest pose, which is why `validate.py --against` exists.

Coat, ribbon, tie, weapon and Evelyn's own hair are held out of the bone candidates. They
are geometry the mod replaces, and their bones would otherwise claim Reze's skin.

### The bones then have to move

Correct weights are not enough, and for the same reason: the rip's bones are Evelyn's,
and Evelyn's gloved hand reaches a full unit past the end of Reze's. 左人指３ stands at
X 6.96 when Reze's whole hand stops at 6.35, so curling that finger swings it around a
pivot outside the mesh and draws it out into a spike. At rest the model is perfect; one
frame of any motion with finger channels shreds both hands.

Where a bone and its parent blend across a joint is where that joint is, so
`Σ (w_b · w_parent) · v / Σ (w_b · w_parent)` locates it from the skin alone. On the rip,
whose bones and mesh do agree, that estimator lands a median 0.16 from the real bone —
accurate enough to trust. On the build before retargeting:

| | fingers | arm | leg | torso and head |
| --- | --- | --- | --- | --- |
| bone to its own joint | 1.17 | 0.87 | 0.42 | 0.27 |

So the bones are moved onto the joints the mod's own skin marks, and their rigid bodies
and joint anchors with them. Median offset afterwards is 0.00, worst 0.44 against the
rip's own worst of 0.45.

Only bones nothing addresses absolutely are moved: the arm and finger chains, which
motions drive by rotation, and the thigh and hip jiggle bones, which only physics drives.
`足D`/`ひざD`/`足首D` stay: a VMD stores `足ＩＫ` as a translation against the rig's leg
length, so moving the leg chain moves every footfall in every motion.

The face plate comes over whole, with the vertices first in the list so its 49 morphs
keep their indices. `EvelynFaceADiffuse` is flipped vertically to sit on the rip's face
UVs.

Everything the mod replaced is then dropped: physics for the removed chains (180 rigid
bodies → 61, 181 joints → 40), the bones left behind (293 → 157, 145 for the bomb form),
and any morph left
empty. Bones are kept if a vertex, a rigid body or an IK chain reaches them, closed over
parents, tails, grant sources and IK links; morphs follow the bones rather than pinning
them, so the bomb form does not ship expressions that drive nothing.

`validate.py` checks index ranges, weight sums, material coverage, parent ordering and
texture paths, and flags any dynamic rigid body left without a joint — that one falls
away under physics and drags its bone with it. It also reports how far each bone stands
from the joint its own skin marks, which is the check that catches a rig correct at rest
and broken in motion. `--against <rip.pmx>` additionally prints
how much surface each bone drives next to the share it drives in the rip. Measure that in
**area**, not vertices: the mod's bare hands are several times denser than the rip's
gloved ones, and counting vertices reports all 26 finger bones as tenfold overweight on a
rig that is right.

## Known gaps

The hair is rigid on `頭`. The mod ships a 24-bone hair rig, but recovering its chains
from weight co-occurrence produces a star out of the cap bone rather than strands, and a
wrong chain sways worse than no chain. The ponytail and the bomb form's fuse strip are
the two places this shows.

The bomb form's dress and fuse strip have no counterpart in the rip, so they merge onto
`下半身` and move with the hip rather than draping. `validate.py --against` reports it:
`下半身` drives 30% of that model's surface against 1.7% in the rip.

`体` is one material over skin, cloth and shoes, so all three take one toon ramp.

## Putting the outfit on the hand-built model

`graft.py` moves the mod's shirt, shorts, ribbon and choker onto
`public/models/reze/reze-zzz.pmx`, whose body is 塞尔凯特2's and whose face is 塞勒涅's.
Its shirt was 20,224 triangles of subdivided mesh over `Swatch_3_Diffuse_1_toon.png`, a
photographic grey swatch — 84 tones at mean chroma 2.5, which is why it takes the
environment's hue. The mod's is 1,940 triangles with the collar, placket and seams
painted where the geometry has them.

The mod ships the whole outfit as one draw, so `garment.py` splits it: weld the vertices
UV seams duplicated, take connected components, and 134 fragments become 26 real parts.
Each garment is then recognised by where it sits rather than by a component number, and
one garment can be several parts — the choker is a band, two rings and a cord.

Placement reads the two **bodies**, not the two garments. Warping the mod's shirt onto
the old one would drag its collar down to where the frill used to be; instead each
garment keeps its standoff from the skin, and the low-frequency difference between the
two bodies is carried across (they agree to a median 0.36, and the mod's is 5% larger).

Two things the warp cannot do on its own:

**Interpenetration.** A garment fitted from another body sits inside this one wherever
this one is fuller, and the skin shows through in exactly that band. `surface.push_out`
resolves it against the surface, which is what makes the result independent of how well
the warp guessed — 381 shirt and 397 shorts vertices needed lifting. The lift is blurred
across the garment first: moving each vertex alone tore edges to 18x their length.

**Cloth versus hardware.** A buckle is not cloth. The choker and its rings are resolved
as one assembly under a single scale, which grows it off the neck without flattening the
rings into ovals — that alone took its worst edge from 12.5x to 3.1x.

The model's inner shorts sit entirely inside the incoming pair, and the mod's are the
tighter cut, so a little showed at the crotch. They are tucked back under rather than
deleted, which would leave a hole if anything ever moved the outer pair.

Weights come from the model's own clothed surface by closest point. That is the same
method that failed on the hands above, and it is right here for the reason it was wrong
there: the new garment lands within 0.06 of the old one and covers the same region, so
there is no wrong digit to land on.

## The bomb form

`bombgraft.py` takes the finished outfit build and turns it into the mod's other body
draw. The head goes outright -- face, eyes, mouth and hair off, mask on -- which is
what the model's own earlier bomb build did and what takes the 111 facial morphs with
it. Below the neck it keeps the outfit and adds the jacket, the gloves and the strip.

Two things it has to do that the casual outfit did not:

**Fit and field want different samples.** The similarity is fitted on the torso, because
the arms are what the two models agree least about and letting them into the fit dragged
the mask 1.9 units up and off centre. The correction field is then built over the whole
body, arms included, or the gloves get no shape correction and land on the wrong part of
the hand. Conflating those two cost two rounds.

**A glove is not a slice, and not a body either.** Cutting the arm shell at `|X| > 2.6`
threw the upper arm away; the glove is separated by what it is instead, since it is
painted dark and the arm under it is not (luminance 0.18 against 0.67 at the shoulder).
It is then placed and lengthened rather than fitted. The body correction field is built
from nearest points, and near the shoulder the nearest point on her is the torso — it
drags the top of the sleeve inward and splits the glove in two, leaving a bare upper arm
between two dark cuffs. A sleeve wants its own axis: each glove is stretched along the
arm it belongs to, anchored at the cuff, since anchoring at the shoulder pulls the cuff
outward and opens the same gap.

What that costs is in `validate.py`: nine hand and finger bones stand up to 1.36 from
the skin they drive, because the geometry is the arm the glove was cut for and not hers.
Fitting the glove instead fixes those numbers and looks wrong, which is the wrong trade —
the fingers deform approximately, and the sleeve is whole.

The mask is grown about the head bone until it clears the skull it replaces, solved
directly rather than iterated -- iterating ran away and swallowed the shoulders -- and
about the bone rather than its own centroid, which is off axis and drifts it sideways.

## The haircut

`hairgraft.py` is optional: `reze-nomorph-zzz.pmx` keeps the model's own haircut and the
sway rig that was built for it. Run it to fit the mod's haircut instead. The default,
`--physics build`, recovers the mod's own hair rig and authors MMD sway for it.

The mod skins the haircut to 24 bones and ships no skeleton, so the chains come back
out of the weights: two bones that blend across the same vertices are neighbours on a
strand. The cap bone at the scalp shares weight with every strand, so leaving it in
collapses the graph into a star — dropping it first leaves the strands as separate
components, and a maximum spanning tree inside each, rooted at its highest link,
recovers the order along it and any branch. Four strands, twenty links: the fringe,
both side locks and the ponytail, with the cap staying rigid on `頭`.

Bodies and joints follow the recipe of the model's own chains rather than invented
numbers — capsules in group 9, mask 0xFDE0, mass tapering from 10 to 1, damping 0.99,
and a cone opening from nothing at the scalp to 12° at the tip. Capsule orientation is
exact: the engine composes rigid-body rotation as `Ry.Rx.Rz`, so fixing `rz` at zero
leaves `rx = acos(dy)`, `ry = atan2(dx, dz)` to put the capsule's local +Y along the
segment.

`--physics head` instead puts every vertex on `頭` at weight 1 and drops the sway
bodies, which is what `reze-zzz/reze-casual.pmx` does. `--physics copy` moves a rig
that already works across whole — mesh, weights, sway bone positions, bodies and
joints — from a build that has one; binding new strands to chains shaped for the old
ones is what breaks a strand, so the rig travels with the mesh or not at all.

All three leave the rest of the target alone, which is why the base is
`reze-new-nomorph.pmx`: it still has the four leg and hip rigid bodies (`右腿物理`,
`左腿物理`, `右臀物理`, `左臀物理`) that the earlier ZZZ graft had lost.

Where the haircut sits on that head is a constant, fitted once against the earlier
graft — which had placed it by hand — to a residual of 0.0008. Baking it means the
build needs only the mod and the base; `--reference` refits against another build.

Two faults in the collision setup had to be corrected on the way, both inherited and
both made worse by twenty capsules where the original had six. The three spheres on
`頭` carry `nocollide` `0xFFFF`, which is not "collides with the body" but "collides
with nothing" — there was no skull under the hair at all. And the strands collided
with each other, which the original chains got away with because `yb`, `zb` and `e`
sit far apart. One skull sphere is trimmed 0.86 to 0.70 so no strand begins inside
it: a body that starts inside a collider is ejected on the first frame, which reads
as the hair jumping off the head before it settles.

## Known gaps

The hair is rigid on `頭`. The mod ships a 24-bone hair rig, but recovering its chains
from weight co-occurrence produces a star out of the cap bone rather than strands, and a
wrong chain sways worse than no chain. The ponytail and the bomb form's fuse strip are
the two places this shows.

The bomb form's dress and fuse strip have no counterpart in the rip, so they merge onto
`下半身` and move with the hip rather than draping. `validate.py --against` reports it:
`下半身` drives 30% of that model's surface against 1.7% in the rip.

`体` is one material over skin, cloth and shoes, so all three take one toon ramp.

## Putting the outfit on the hand-built model

`graft.py` moves the mod's shirt, shorts, ribbon and choker onto
`public/models/reze/reze-zzz.pmx`, whose body is 塞尔凯特2's and whose face is 塞勒涅's.
Its shirt was 20,224 triangles of subdivided mesh over `Swatch_3_Diffuse_1_toon.png`, a
photographic grey swatch — 84 tones at mean chroma 2.5, which is why it takes the
environment's hue. The mod's is 1,940 triangles with the collar, placket and seams
painted where the geometry has them.

The mod ships the whole outfit as one draw, so `garment.py` splits it: weld the vertices
UV seams duplicated, take connected components, and 134 fragments become 26 real parts.
Each garment is then recognised by where it sits rather than by a component number, and
one garment can be several parts — the choker is a band, two rings and a cord.

Placement reads the two **bodies**, not the two garments. Warping the mod's shirt onto
the old one would drag its collar down to where the frill used to be; instead each
garment keeps its standoff from the skin, and the low-frequency difference between the
two bodies is carried across (they agree to a median 0.36, and the mod's is 5% larger).

Two things the warp cannot do on its own:

**Interpenetration.** A garment fitted from another body sits inside this one wherever
this one is fuller, and the skin shows through in exactly that band. `surface.push_out`
resolves it against the surface, which is what makes the result independent of how well
the warp guessed — 381 shirt and 397 shorts vertices needed lifting. The lift is blurred
across the garment first: moving each vertex alone tore edges to 18x their length.

**Cloth versus hardware.** A buckle is not cloth. The choker and its rings are resolved
as one assembly under a single scale, which grows it off the neck without flattening the
rings into ovals — that alone took its worst edge from 12.5x to 3.1x.

The model's inner shorts sit entirely inside the incoming pair, and the mod's are the
tighter cut, so a little showed at the crotch. They are tucked back under rather than
deleted, which would leave a hole if anything ever moved the outer pair.

Weights come from the model's own clothed surface by closest point. That is the same
method that failed on the hands above, and it is right here for the reason it was wrong
there: the new garment lands within 0.06 of the old one and covers the same region, so
there is no wrong digit to land on.

## The haircut

`hairgraft.py` runs first, and `graft.py` takes its output. It offers two rigs.

`--physics head` (the default) is what `reze-zzz/reze-casual.pmx` does: the haircut
comes from the corrected pipeline, the right way round, and every vertex rides `頭` at
weight 1. The sway rigid bodies and their joints are dropped with it, so nothing
simulates that drives nothing. Reze's cut is short and carries nothing long enough to
need sway, and a strand welded to the skull cannot come apart.

`--physics copy` moves a working haircut across whole instead — mesh, weights, sway bone
positions, rigid bodies and the joints between them, verbatim from a build that already
has them fitted. Binding new strands to chains shaped for the old ones is what breaks
the strand on the character's left, so the rig has to travel with the mesh or not at all.

Both leave the rest of the target alone, which is why the base is `reze-new-nomorph.pmx`:
it still has the four leg and hip rigid bodies (`右腿物理`, `左腿物理`, `右臀物理`,
`左臀物理`) and their joints that `reze-zzz.pmx` lost.

Reflecting the earlier graft in place was tried and abandoned. It is sound on paper —
reflecting x conjugates a rotation to `(rx, -ry, -rz)` whatever order the Euler angles
compose in — but it means rewriting every sway body and joint, and the physics is the
part of this model least worth gambling on a derivation.
