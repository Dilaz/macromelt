# macromelt

Decoder for Macromedia Director **Shockwave 3D** cast members, with a glTF 2.0 bake.

## What it is

A Director 8.x movie stores each 3D cast member as an `XMED` chunk of its RIFX container.
The payload is a `3DEM` stream built on Intel's IFX technology — the ancestor of U3D —
and it carries the whole authored scene:

- **CLOD progressive meshes.** Geometry is not a vertex array but a base mesh plus a
  sequence of resolution updates, the whole thing packed behind an arithmetic-coded
  bitstream with per-symbol contexts and escapes.
- **Skins** with authored per-vertex bone weights, and the bone hierarchy that drives them.
- **Motions**: bones motions (one TRS track per joint) and keyframe motions (a single
  track driving a model's own transform, Director's `keyframePlayer`).
- **Textures**: JPEG colour planes, with 4-channel textures carrying their alpha as a
  separate zlib-deflated 8-bit plane.
- **Materials**, **cameras** and the **transform node** graph that places everything.

`macromelt` decodes all of that and bakes it to a single glTF 2.0 binary (`.glb`).
It is a library first; the CLI is a thin shell over it.

Coordinates stay in Director's **Z-up** convention. Times are seconds, quaternions are
stored **wxyz**, and inverse-quantisation factors multiply decoded integers back to floats.

## Install

Requires Rust 1.85 or newer.

```sh
cargo install --git https://github.com/Dilaz/macromelt --tag v0.1.1 --locked macromelt
```

## CLI

```
macromelt <input.xmed> [options]
```

With no action flag the file's chunk tables are summarised (`--info`).

| Flag | Effect |
| --- | --- |
| `--info` | Summarise every chunk table: bones, mesh nodes, geometry, declarations, shaders, textures, cameras, materials, motions, skins. |
| `--decode-motion` | Decode every motion, report track health, print the keyframes as JSON on stdout. |
| `--decode-weights` | Print each 0x4B skin's bone hierarchy as JSON on stdout. |
| `--probe-geometry` | Dump the raw layout of every 0x49 geometry chunk. |
| `--try-decode-positions` | Attempt a position-only decode of every geometry chunk and report it. |
| `--probe-clod` | Report how each chunk's CLOD update records are laid out. |
| `--probe-basemesh` | Report each chunk's base-mesh (resolution 0) block. |
| `--probe-hand` | Run the hand-written reference decoder over every geometry chunk. |
| `--dump-schedule` | Print the 0x47 CLOD resolution schedule of every mesh. |
| `--decode-geom` | Decode every mesh group and print its element counts. |
| `--dump-geom <out.json>` | Write every decoded mesh group's arrays to a JSON file. |
| `--dump-textures <dir>` | Write every 0x21 texture payload into a directory. |
| `--native-check <obj_dir>` | Compare the decode against reference OBJs; exit 1 on mismatch. Short-circuits every other action. |
| `--export-glb <out.glb>` | Bake the file to a glTF binary. |
| `--flat-skins` | Bake the flat-skin layout (see the GLB contract below). |
| `--anim <anim.xmed>` | Take motions from this additional file as well. Repeatable. |
| `--take <motion>=<clip>` | Take only this motion from the **preceding** `--anim`, under this clip name. Repeatable. |
| `--clip-name <from>=<to>` | Rename a clip after decoding. Repeatable. |
| `--dump-keyframes <substring>` | Print keyframe diagnostics for tracks whose name contains the substring (`""` matches all). |
| `-q`, `--quiet` | Log warnings and errors only. |
| `-v`, `--verbose` | Raise the log level: once for debug, twice for trace. |

`--take` attaches to the `--anim` it follows on the command line, so the order matters:
`--anim a.xmed --take x=y --anim b.xmed` takes `x` from `a.xmed`, and everything from
`b.xmed`. A `--take` before any `--anim` is a usage error.

### Examples

Inspect a file:

```sh
macromelt horse.xmed --info
```

Bake a rig with motions taken from other cast members — the second file contributes only
one of its motions, under the clip name the movie's Lingo clones it as — in the
flat-skin layout:

```sh
macromelt horse.xmed --export-glb horse.glb --flat-skins \
  --anim horse_walk.xmed \
  --anim extras.xmed --take stand_long=horse_idle \
  --clip-name "logo-Key=logo-key"
```

Extract the embedded textures (JPEG colour planes and zlib alpha planes, as stored):

```sh
macromelt scene.xmed --dump-textures /tmp/tex
```

Look at what a motion actually contains, per keyframe:

```sh
macromelt horse_walk.xmed --dump-keyframes "" | head -40
```

## Library

```toml
[dependencies]
macromelt = { git = "https://github.com/Dilaz/macromelt", tag = "v0.1.1", default-features = false }
```

`default-features = false` drops the CLI's `clap` and `env_logger` dependencies.

```rust
use macromelt::{ExportOptions, decode_motion_full, export_glb, parse_xmed};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: bake <file.xmed>");
    let bytes = std::fs::read(&path)?;
    let xmed = parse_xmed(&bytes)?;

    // Motions living in this same file are baked automatically; this shows how
    // to decode them yourself, e.g. to inspect or to re-bake from another file.
    for motion in &xmed.motions {
        let decoded = decode_motion_full(motion);
        println!("clip {:?}: {} tracks", decoded.name, decoded.tracks.len());
    }

    let glb = export_glb(&xmed, &ExportOptions::default())?;
    std::fs::write("out.glb", glb)?;
    Ok(())
}
```

The pipeline is `parse_xmed` → `decode_mesh_group` (once per `XmedFile::geometry_groups()`
entry) / `decode_motion_full` (once per `XmedFile::motions` entry) → `export_glb`.
`export_glb` runs the mesh, material and texture stages itself; you only need
`decode_motion_full` to feed *another* file's motions into a bake through
`ExportOptions::extra_motions`, which is what Director's `cloneMotionFromCastmember` does.

`cargo doc --open` documents every chunk type field by field.

## GLB contract

This is what a baked `.glb` looks like. Consumers depend on it, so it is specified rather
than incidental.

### Node layout

```
SceneRoot
├── Mesh:<skin>              one per skinned mesh, NO TRS of its own
├── <static>                 one per unskinned Director model, TRS = its 0x72 matrix
├── <static>:start           parent inserted only for keyframe-animated statics
│   └── <static>
├── <group>                  authored parent transform node, when the XMED has one
│   └── SkinRoot:<skin>
└── SkinRoot:<skin>          otherwise a direct child of SceneRoot
    └── <root bone>          first child, at the skin's rest displacement
```

- **Skinned mesh nodes** are named `Mesh:<skin>` and carry no TRS. Per the glTF spec a
  skinned mesh ignores its node transform, so placement lives on the `SkinRoot` and in the
  joint hierarchy.
- **`SkinRoot:<name>`** carries the skin's placement. Its first child is the skin's root
  bone at its rest displacement. (Note that `GLTFLoader` in three.js reads the node name
  back as `SkinRoot<name>`.)
- **Static nodes** are named exactly as the Director mesh node, are direct children of
  `SceneRoot`, carry the authored 0x72 matrix as TRS (decomposed, mirror-safe: a mesh
  authored with a negative axis gets `scale (-1, 1, 1)`), and their vertices are
  node-local. A multi-material mesh stays ONE node with one primitive per 0x45 shader
  slot — a material is not a Director model.
- **Authored parent groups.** When the XMED parents a skin's mesh node under a transform
  node, that node is emitted as a group node under `SceneRoot` (name = the Director model
  name, TRS = its world matrix) and the `SkinRoot`s hang under it.
- **`flat_skins`** overrides that: every `SkinRoot` becomes a direct sibling under
  `SceneRoot` with identity rotation and scale, and translation = the skin's authored 0x72
  node offset **plus** one shared `ground_shift` that puts the lowest hull vertex of the
  skin with toe bones at Director Z = 0. The authored group translation is deliberately
  **not** baked in: a consumer that seats one rig on another re-creates that offset at run
  time. `ground_shift` applies to flat-skin bakes only; every other bake keeps the authored
  node offsets.

### Animation clips

- One clip per skin. Channels are translation, rotation and scale (scale only when it is
  not identity), quaternions converted from the file's **wxyz** to glTF's **xyzw**,
  coordinates Z-up like the rest of the bake.
- Clips are **retargeted at bake time** so keyframe 0 equals the destination skin's bind
  pose: the root joint of every clip is re-anchored onto that skin's bind TRS, while limb
  rotations stay the authored values. This lets one authored motion drive a differently
  proportioned skin without a first-frame pop.
- Every clip is **rebased on its earliest key**, so `t = 0` is always the first pose.
  Director keyframe times are absolute in the motion's own timeline and `bonesPlayer.play`
  starts a motion at its first keyframe, but a glTF `AnimationClip` spans `[0, max t]` —
  without the rebase a loop whose first key sits at 0.2333 s holds its first pose for a
  quarter second on every cycle.
- **Keyframe motions** (`keyframePlayer`: a single track that is not a bone) become clips
  targeting the static node named after the track, or the bake's only static node when the
  Lingo clones the motion onto a differently named model. Director plays these *relative to
  the model's transform at play start* and a glTF mixer writes absolute TRS, so the
  authored 0x72 TRS moves onto a parent `<name>:start` node and the animated `<name>` node
  rests at identity. A scene that sets the model's transform itself must set or neutralise
  `<name>:start`, exactly as the Lingo replaces `transform`.

### Materials and textures

- An **untextured** shader gets `baseColorFactor = diffuse.rgb + opacity` from its 0x10
  material block. A **textured** shader keeps the white base: the texture replaces the
  diffuse term, which is what the original authoring tool also assumed.
- `opacity < 1` adds `alphaMode: "BLEND"` in both cases.
- Emissive is decoded but not exported: the stored values are 3ds Max self-illumination
  percentages, which blow out under a physically-based renderer.
- A 4-channel texture (0x20 `channels = 4`) is merged into an RGBA PNG from its JPEG colour
  plane and its zlib alpha plane. On a **static** primitive it gets `alphaMode: "BLEND"`;
  on a **skinned** primitive it keeps the cut-out `MASK` with `alphaCutoff: 0.5`.
- Texcoords are mirrored on export (`[u, 1 - v]`): XMED stores them bottom-up
  (Director/OpenGL, `v = 0` at the image's bottom row) while glTF's UV origin is top-left.

### Skin weights

Every skinned XMED carries its own per-vertex weights in the 0x49 geometry stream: per
POSITION a count (arithmetic context `0x12`), then that many bone ids (`0x13`), and a
quantised weight (`0x14`, `w = raw / 32768`) for each entry after the first — the first
takes `1 - Σrest`, the U3D "last weight is not written" rule. Those authored weights are
decoded and bound directly. The IFXSkin
`regenerate → joint-cross-section → smooth → remove-rogue` envelope pipeline is only the
**fallback** for skins whose weights could not be decoded; deriving weights by proximity
binds held props to whatever bone happens to be nearest, which is visibly wrong.

## Format notes

| Chunk | Content |
| --- | --- |
| `0x10` | Material: name, attributes, ambient/diffuse/specular/emissive RGBA, reflectivity, opacity. |
| `0x20` | Texture declaration: name, width, height, channel count. |
| `0x21` | Texture image: JPEG payload, or a zlib-deflated 8-bit alpha plane. |
| `0x36` | Shader: flags, texture layers, material and texture references. |
| `0x45` | Mesh declaration: element counts, per-shader-slot counts, bounding sphere, inverse-quantisation factors, sibling (instance) meshes. |
| `0x47` | CLOD resolution schedule, per submesh. |
| `0x49` | Geometry: the arithmetic-coded CLOD stream, including authored skin weights. |
| `0x4B` | Skin: the bone definitions (rest length, displacement, orientation, parent). |
| `0x67` | Motion resource: tracks of TRS keyframes. |
| `0x70` | Transform node / bone. |
| `0x72` | Mesh node: name, parent, geometry reference, shader, 4×4 matrix. |
| `0x74` | Camera: parent, position, fov, hither/yon, viewport rect, projection. |

Several 0x49 chunks may refine one mesh; `XmedFile::geometry_groups()` keys them on the
mesh name, in first-appearance order, with file order inside a group. See `cargo doc` for
the per-chunk field layout.

### Mesh variants

The 0x49 record layout is not the same for every file. Bit 1 of the 0x45 declaration's
`attributes` word says that each new face is followed by an **attribute-face record**: a
parallel 28-byte-per-face structure (three per-corner attribute indices, three next-face
links, three next-corner bytes and a type byte) that the original fills to track which faces
share an attribute index. Nothing in it feeds position or face reconstruction, so macromelt
reads and discards it — but it has to be read, because the next face's corner type follows
its bits.

Both values occur in the wild, and the flag is per mesh, not per title: across the two
Tallitytöt discs 659 of 1281 mesh declarations are `attributes = 4` and 622 are
`attributes = 6`. The first disc is uniformly 4; the second mixes them 235/622, though never
inside a single file — 69 of its 112 3D files are all-4 and the remaining 43 all-6.

## Debugging

Nothing is printed by the library: every diagnostic goes through the [`log`] facade.

```sh
RUST_LOG=macromelt=debug  macromelt file.xmed --export-glb out.glb
RUST_LOG=macromelt=trace  macromelt file.xmed --decode-geom
```

The bit-level dumps sit behind their own targets so they can be enabled one at a time:

| Target | Dump |
| --- | --- |
| `macromelt::bittrace` | Every arithmetic-coder read: context, symbol, escape. |
| `macromelt::posdump` | Decoded positions, texcoords and faces per record. |
| `macromelt::facedump` | Face-array construction and deferred updates. |
| `macromelt::fulltrace` | Raw bit reads of the whole stream. |
| `macromelt::mdraw` | The 0x45 mesh declaration as parsed. |

```sh
RUST_LOG=macromelt::bittrace=trace macromelt file.xmed --decode-geom
```

The CLI's `-v`/`-vv` set `macromelt=debug`/`macromelt=trace`; `RUST_LOG` overrides them.

## Testing

The unit tests run anywhere. The integration tests read real Director assets, which are
copyrighted and are not part of this repository, so they resolve their fixtures through an
environment variable and **skip** when it is unset:

```sh
cargo test                                    # fixture-backed tests print "skip:"
MACROMELT_FIXTURES=/path/to/fixtures cargo test
MACROMELT_FIXTURES=/path/to/disc1:/path/to/disc2 cargo test
```

`MACROMELT_FIXTURES` is a `:`-separated list of directories, each holding
`extracted/assets/…` (the raw `.xmed` files), `assets/models/obj/…` (reference OBJs, for the
decode-parity tests) and `assets/models/gltf/…` (previously baked GLBs, for the regression
tests). A fixture is looked up in each root in turn, so one run can cover several discs —
which matters because they do not all use the same mesh variant (see
[Format notes](#format-notes)).

## License

Apache-2.0. See [LICENSE](LICENSE).
