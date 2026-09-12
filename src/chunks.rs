//! XMED (3DEM) chunk parser using nom combinators.
//!
//! The 3DEM format uses Intel IFX v2 chunk structure:
//!   Header: 3DEM(4) + total_size(4 BE) + version(4) + inner_size(4) + IFX sub-header
//!   Chunks: type(1) + 0xFFFFFF(3) + size(4 LE) + data(size)

use crate::error::ParseError;
use nom::{
    IResult, Parser,
    bytes::complete::{tag, take},
    error::{Error as NomError, ErrorKind},
    number::complete::{le_f32, le_i32, le_u16, le_u32, u8 as nom_u8},
};
use serde::Serialize;

// ── Primitive parsers ──────────────────────────────────────

/// Parse a length-prefixed string (u16 LE length + ASCII bytes).
fn parse_string(input: &[u8]) -> IResult<&[u8], String> {
    let (input, len) = le_u16(input)?;
    let (input, bytes) = take(len as usize)(input)?;
    Ok((input, String::from_utf8_lossy(bytes).to_string()))
}

/// Parse a 4x4 float matrix (16 x f32 LE = 64 bytes).
fn parse_matrix(input: &[u8]) -> IResult<&[u8], [f32; 16]> {
    let mut m = [0f32; 16];
    let mut rest = input;
    for v in &mut m {
        let (r, val) = le_f32(rest)?;
        *v = val;
        rest = r;
    }
    Ok((rest, m))
}

/// Parse 3 floats as [f32; 3].
fn parse_vec3(input: &[u8]) -> IResult<&[u8], [f32; 3]> {
    let (input, (x, y, z)) = (le_f32, le_f32, le_f32).parse(input)?;
    Ok((input, [x, y, z]))
}

/// Parse 4 floats as [f32; 4].
fn parse_vec4(input: &[u8]) -> IResult<&[u8], [f32; 4]> {
    let (input, (a, b, c, d)) = (le_f32, le_f32, le_f32, le_f32).parse(input)?;
    Ok((input, [a, b, c, d]))
}

// ── Chunk data types ───────────────────────────────────────

/// Transform node (0x70 chunk): one named node of the scene's transform
/// hierarchy — the authored group/dummy nodes a mesh or skin hangs from.
#[derive(Debug, Clone, Serialize)]
pub struct TransformNode {
    /// Node name as authored in Director.
    pub name: String,
    /// Name of the parent 0x70 node, or `"World"` when the node hangs directly
    /// off the scene root.
    pub parent: String,
    /// Placement relative to `parent`, row-vector convention: rows 0..2 are the
    /// scaled rotation basis, row 3 (elements 12..14) is the translation. World
    /// placement is the product up the `parent` chain. Director space is Z-up;
    /// consumers convert.
    pub matrix: [f32; 16],
    /// Translation row of [`TransformNode::matrix`] (elements 12..14), in
    /// Director Z-up units.
    pub position: [f32; 3],
}

/// Mesh node (0x72 chunk): places one mesh resource in the scene graph.
///
/// Its matrix is the authored TRS baked onto the static glTF node of the same
/// name; `mesh_ref` links it to the 0x45 declaration and the 0x49 geometry
/// chunks that supply the shape.
#[derive(Debug, Clone, Serialize)]
pub struct MeshNode {
    /// Node name as authored in Director — the exported static node carries
    /// exactly this name.
    pub name: String,
    /// Name of the parent 0x70 transform node, or `"World"` for the scene root.
    pub parent: String,
    /// Translation row of [`MeshNode::matrix`] (elements 12..14), in Director
    /// Z-up units.
    pub position: [f32; 3],
    /// Placement relative to `parent`, row-vector convention: rows 0..2 the
    /// scaled rotation basis, row 3 the translation. Director Z-up.
    pub matrix: [f32; 16],
    /// Name of the mesh this node draws: matches a 0x45 mesh declaration and
    /// the 0x49 geometry chunks of the same name.
    pub mesh_ref: String,
    /// Name of the backing mesh resource, written directly after `mesh_ref`.
    pub resource_ref: String,
    /// Shader name (a 0x36 chunk) bound to this instance; empty when the chunk
    /// ends before the string.
    pub shader: String,
}

/// Compressed CLOD geometry (0x49 chunk): one arithmetic-coded record stream
/// for a mesh.
///
/// Only the first u32 after the name is a real count. The binary's 0x49 handler
/// (`FUN_7a183c50`) raw-reads the name plus that single u32 while the
/// arithmetic coder is still pristine (`FUN_7a11dbf0` reads raw bytes while
/// `high = 0xffff, code = 0`); the next four u32 already belong to the coded
/// body, so `geometry.rs` re-prepends their little-endian bytes to
/// `compressed_data` to rebuild the stream the decoder actually sees.
///
/// One mesh may be split across 2-3 chunks of the same name — see
/// [`XmedFile::geometry_groups`].
#[derive(Debug, Clone, Serialize)]
pub struct GeometryChunk {
    /// Mesh name; matches the 0x45 declaration and a 0x72 node's `mesh_ref`.
    pub name: String,
    /// Number of resolution-update records this chunk carries (the handler's
    /// `maxRes`) — a record count, not a vertex count. Summed over a mesh's
    /// chunks it equals the declaration's [`MeshDescription::max_resolution`].
    pub num_positions: u32,
    /// Second raw u32 of the chunk. Despite the name it is not a face count:
    /// it is already the first word of the arithmetic-coded body and is
    /// re-prepended to `compressed_data` before decoding.
    pub num_faces: u32,
    /// Third raw u32 — likewise part of the arithmetic-coded body, not a
    /// normal count.
    pub num_normals: u32,
    /// Fourth raw u32 — likewise part of the arithmetic-coded body, not a
    /// texcoord count.
    pub num_texcoords: u32,
    /// Fifth raw u32 — likewise part of the arithmetic-coded body; it has no
    /// standalone meaning, hence the placeholder name.
    pub field5: u32,
    /// Chunk bytes after those five u32: the remainder of the arithmetic-coded
    /// record stream.
    pub compressed_data: Vec<u8>,
}

/// CLOD resolution-update schedule (0x47 chunk).
///
/// Handler `FUN_7a183a20` (dispatch table byte `0x00` at `0x7a18a92f`, jump
/// target `0x7a18a75b`; disassembly `0x7a183bc4-0x7a183c35`). Layout:
///
/// ```text
///   name : string
///   for submesh s in 0 .. decoder->0x78 (= MeshDescription::shaders.len()):
///       for j in 0 .. decoder->0xb4[s] (= ShaderSlot::num_texcoords):
///           acc += ReadCompressedU32(ctx 1)      // acc resets to 0 per submesh
///           schedule[s][j] = acc
/// ```
///
/// `schedule[s][j]` is the *resolution step* at which submesh `s`'s `j`-th
/// progressive record becomes decodable: the 0x49 decoder (`FUN_7a18a990`,
/// `dec_50_55_7a18a990.c:164-181`) walks a global resolution counter and, at
/// each step, decodes at most one record per submesh, gated on
/// `schedule[s][cursor[s]] <= resolution`. Without it a multi-shader mesh
/// cannot be split back into its interleaved per-submesh record streams.
#[derive(Debug, Clone, Serialize)]
pub struct ClodScheduleChunk {
    /// Mesh name this schedule belongs to (the 0x45 / 0x49 name).
    pub name: String,
    /// Chunk bytes after the name: the arithmetic-coded `schedule[s][j]` table,
    /// decoded by `geometry::decode_clod_schedule`.
    #[serde(skip)]
    pub compressed_data: Vec<u8>,
}

/// One shading group ("shader slot") of an author-mesh declaration (0x45 chunk).
///
/// A 0x45 chunk declares `num_shaders` shading groups; group `i` owns the faces
/// whose shading id is `i` and draws them with `material` / `shader`. The counts
/// are that group's share of the mesh.
#[derive(Debug, Clone, Serialize)]
pub struct ShaderSlot {
    /// Per-vertex attribute mask — `uStack_f0` in the 0x49 decoder
    /// (`tools/re/decoders/dec_50_55_7a18a990.c:241-242`, read from the author
    /// mesh object through vtable +0x18): `&1` positions, `&2` normals,
    /// `(>>4)&0xf` texcoord layers. Only 0x3 and 0x13 occur in this corpus
    /// (113 × 0x3, 866 × 0x13 across 424 declarations).
    pub attrs: u32,
    /// This group's share of the mesh's positions.
    pub num_positions: u32,
    /// This group's share of the mesh's faces (the faces whose shading id is
    /// this group's index).
    pub num_faces: u32,
    /// This group's share of the mesh's normals.
    pub num_normals: u32,
    /// Positionally the group's texcoord count, but the engine also uses it as
    /// the group's **progressive-record count**: the 0x45 handler
    /// (`FUN_7a1858d0`, `dec_9_18_7a1858d0.c:481-518`) sizes both the CLOD
    /// record table (`FUN_7a1847c0(obj, 0xb4[s], 0xb0[s])`) and the 0x47
    /// schedule array from this field. `Hey/arrow.xmed` declares `0x3` (no UV
    /// layers at all) yet 11 here — and decodes to exactly 11 records.
    pub num_texcoords: u32,
    /// This group's share of the mesh's vertex colours (0 throughout this corpus).
    pub num_colors: u32,
    /// Material name for this shading id — matches a 0x36 shader chunk name.
    pub material: String,
    /// Shader name for this shading id (always `DefaultShader` in this corpus).
    pub shader: String,
}

/// Material list of a further mesh that shares one 0x45 declaration.
///
/// Rare: 3 of the 424 declarations in the corpus (`Rydde/stall.xmed` Box01..Box10,
/// `Rydde/clickplane_1.xmed`, `Hey/hay_scene.xmed`) declare several identically
/// shaped meshes in one chunk. The mesh the chunk is named after lives in
/// [`MeshDescription::shaders`]; the rest land here.
#[derive(Debug, Clone, Serialize)]
pub struct SiblingMesh {
    /// Name of the sibling mesh (the head of its name array).
    pub name: String,
    /// Its material names in shading-id order, one per shading group of the
    /// shared declaration; each names a 0x36 shader chunk.
    pub materials: Vec<String>,
}

/// Mesh description / Author Mesh Declaration (0x45 chunk).
///
/// Layout, validated against all 1281 declarations of the two Tallitytöt discs:
///
/// ```text
///   name : string
///   u32    attributes                     (bit 1 = per-face attribute record)
///   u32    num_shaders = N                (shading-group count)
///   N × u32[6] { positions, faces, normals, texcoords, colors, attribute_mask }
///   u32    extra_count = M                (number of name arrays)
///   M × (N+1) strings                     (array 0: "default" + N shader names;
///                                          arrays 1..: mesh name + N materials)
///   f32[3] bounding-sphere centre
///   32-byte tail: 7 × f32 + u32
///     (bounding_sphere_radius, scale, pos_iq, norm_iq, tc_iq, diff_iq, spec_iq,
///      max_resolution)
/// ```
///
/// The inverse-quantization factors (pos_iq etc.) are the critical outputs —
/// they let the 0x49 compressed geometry body recover real float positions from
/// quantized integer deltas.
#[derive(Debug, Clone, Serialize)]
pub struct MeshDescription {
    /// Mesh name; the 0x49 geometry chunks and the 0x72 node's `mesh_ref` use it.
    pub name: String,
    /// Mesh-level attribute word (first u32 after the name).
    ///
    /// Bit 1 (`& 2`) says every new face in the 0x49 stream carries the
    /// attribute-face record of `call_7a18a990.c:1209-1324` — the parallel
    /// per-corner attribute-index array. It is set per mesh, not per file or
    /// per title: of the 1281 declarations on the two Tallitytöt discs, 659 are
    /// `4` and 622 are `6` (disc 1 all 4; disc 2 mixes 235 and 622).
    pub attributes: u32,
    /// Shading-group count (== `shaders.len()`).
    pub num_shaders: u32,
    /// Total position count: the sum over the shading groups.
    pub num_positions: u32,
    /// Total face count: the sum over the shading groups.
    pub num_faces: u32,
    /// Total normal count: the sum over the shading groups.
    pub num_normals: u32,
    /// Total texcoord / progressive-record count: the sum over the shading
    /// groups (see [`ShaderSlot::num_texcoords`]).
    pub num_texcoords: u32,
    /// Total vertex-colour count: the sum over the shading groups.
    pub num_colors: u32,
    /// Number of name arrays: the shader array plus one material array per mesh.
    pub extra_count: u32,
    /// Shading groups in shading-id order.
    pub shaders: Vec<ShaderSlot>,
    /// Further meshes sharing this declaration (empty for 421 of 424 chunks).
    pub sibling_meshes: Vec<SiblingMesh>,
    /// Bounding-sphere centre in the mesh's own Director Z-up object space.
    pub bounding_center: [f32; 3],
    /// Bounding-sphere radius, in Director units.
    pub bounding_sphere: f32,
    /// Authored mesh scale written in the 32-byte tail.
    pub scale: f32,
    /// Inverse-quantization factor for positions: multiplies the 0x49 stream's
    /// decoded integer position deltas back into Director-unit floats.
    pub position_inv_quant: f32,
    /// Inverse-quantization factor for normals: multiplies decoded integer
    /// normal deltas back into unit-vector components.
    pub normal_inv_quant: f32,
    /// Inverse-quantization factor for texture coordinates: multiplies decoded
    /// integer UV deltas back into floats.
    pub texcoord_inv_quant: f32,
    /// Inverse-quantization factor for diffuse vertex colours (1/512 here).
    /// Not the skin-weight factor — 0x49 weights use a fixed `1/32768`.
    pub diffuse_inv_quant: f32,
    /// Inverse-quantization factor for specular vertex colours.
    pub specular_inv_quant: f32,
    /// Highest resolution step of this mesh: the total number of
    /// resolution-update records, equal to the sum of
    /// [`GeometryChunk::num_positions`] over the mesh's 0x49 chunks.
    pub max_resolution: u32,
}

impl MeshDescription {
    /// Material names in shading-id order (each names a 0x36 shader chunk).
    pub fn material_names(&self) -> Vec<&str> {
        self.shaders.iter().map(|s| s.material.as_str()).collect()
    }

    /// Per-vertex attribute mask for the whole mesh: the bitwise OR over the
    /// shading groups. A declaration without groups (never seen in this corpus)
    /// falls back to the declared counts.
    pub fn attribute_mask(&self) -> u32 {
        if !self.shaders.is_empty() {
            return self.shaders.iter().fold(0, |mask, slot| mask | slot.attrs);
        }
        let normals = if self.num_normals > 0 { 2 } else { 0 };
        let uv = if self.num_texcoords > 0 { 0x10 } else { 0 };
        1 | normals | uv
    }
}

/// Shader (0x36 chunk): binds a material and, optionally, a texture.
#[derive(Debug, Clone, Serialize)]
pub struct ShaderChunk {
    /// Shader name; a 0x45 shading group and a 0x72 mesh node reference it.
    pub name: String,
    /// `&1` material name present, `&2` texture name present (1 or 3 here).
    pub flags: u16,
    /// Texture layer count: 1 when `texture` is set, else 0.
    pub texture_layers: u16,
    /// Material name this shader binds — a 0x10 material chunk.
    pub material: String,
    /// Texture name (a 0x20/0x21 chunk name); empty for an untextured shader.
    pub texture: String,
    /// Trailing float/flag block: 140 bytes when textured, empty otherwise.
    #[serde(skip)]
    pub raw_flags: Vec<u8>,
}

/// Texture declaration (0x20 chunk): the dimensions of the matching 0x21 image.
#[derive(Debug, Clone, Serialize)]
pub struct TextureDecl {
    /// Texture name; the matching 0x21 image chunk(s) carry the same name.
    pub name: String,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// 3 = opaque JPEG, 4 = JPEG plus a zlib alpha plane.
    pub channels: u8,
}

/// Payload of a 0x21 texture chunk.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum TextureImage {
    /// Type byte 01: a complete JFIF stream.
    Jpeg {
        /// Raw JFIF bytes, from the `FF D8 FF` start-of-image marker on.
        #[serde(skip)]
        data: Vec<u8>,
    },
    /// Type byte 03: an 8-bit alpha plane, zlib-deflated to `width * height` bytes.
    Plane {
        /// Plane width in pixels (matches the 0x20 declaration).
        width: u32,
        /// Plane height in pixels (matches the 0x20 declaration).
        height: u32,
        /// zlib stream inflating to `width * height` 8-bit alpha samples.
        #[serde(skip)]
        zlib: Vec<u8>,
    },
}

/// Texture image (0x21 chunk). A 4-channel texture emits two chunks under the
/// same name: the RGB JPEG and the alpha `Plane`.
#[derive(Debug, Clone, Serialize)]
pub struct TextureChunk {
    /// Texture name; matches the 0x20 declaration and, for a 4-channel
    /// texture, the sibling chunk carrying the other half.
    pub name: String,
    /// This chunk's payload: the RGB JPEG or the alpha plane.
    pub image: TextureImage,
}

/// Camera (0x74 chunk).
#[derive(Debug, Clone, Serialize)]
pub struct CameraChunk {
    /// Camera name (`defaultview` for the runtime camera Director itself writes).
    pub name: String,
    /// Name of the parent 0x70 transform node, or `"World"`.
    pub parent: String,
    /// Row-major 4×4: rows 0..2 carry the rotation basis, row 3 the translation.
    pub matrix: [f32; 16],
    /// Translation row of [`CameraChunk::matrix`] (elements 12..14), in
    /// Director Z-up units.
    pub position: [f32; 3],
    /// Projection kind: 8 for 505 of the corpus' 508 cameras, 9 for the other 3.
    pub projection: u32,
    /// Near clipping-plane distance, in Director units.
    pub hither: f32,
    /// Far clipping-plane distance, in Director units.
    pub yon: f32,
    /// Field of view in degrees.
    pub fov: f32,
    /// Viewport rect (x, y, width, height): (0,0,640,480) for `defaultview`,
    /// (0,0,1,1) for cameras authored in MAX.
    pub rect: [f32; 4],
}

/// Material (0x10 chunk): the colour block a 0x36 shader binds by name.
///
/// Payload after the name is `u32 attributes, ambient[4], diffuse[4],
/// specular[4], emissive[4], f32 reflectivity, f32 opacity` — 76 bytes in
/// every chunk of the corpus (215 distinct materials), `attributes` always
/// 0x3F (all six fields written). Cross-checked against the SW3D converter's
/// own MTL for `christian_bike`: `yellow` writes `Ka 0.780392 0.494118
/// 0.164706` = ambient, `Kd 1.0 0.811765 0.356863` = diffuse, `Ks` = specular,
/// `Ns 24.32` = `reflectivity * 128`, `d 1.0` = opacity.
#[derive(Debug, Clone, Serialize)]
pub struct Material {
    /// Material name; a 0x36 shader chunk binds it by this name.
    pub name: String,
    /// Field mask; 0x3F throughout this corpus.
    pub attributes: u32,
    /// Ambient colour RGBA in 0..1 (the MTL `Ka`).
    pub ambient: [f32; 4],
    /// Diffuse colour RGBA in 0..1 (the MTL `Kd`); the exporter's
    /// `baseColorFactor` for untextured shaders.
    pub diffuse: [f32; 4],
    /// Specular colour RGBA in 0..1 (the MTL `Ks`).
    pub specular: [f32; 4],
    /// Emissive colour RGBA in 0..1.
    pub emissive: [f32; 4],
    /// Specular exponent as a 0..1 fraction; the MTL `Ns` is `reflectivity * 128`.
    pub reflectivity: f32,
    /// Opacity in 0..1 (the MTL `d`); below 1 the exported material blends.
    pub opacity: f32,
}

/// A single bone definition from a 0x4B chunk.
#[derive(Debug, Clone, Serialize)]
pub struct BoneDef {
    /// Bone name; motion tracks address bones by this name.
    pub name: String,
    /// Index of the parent bone in [`BoneWeightChunk::bones`], or negative for
    /// a root bone.
    pub parent_index: i32,
    /// Bone length along its own local +X axis, in Director units. A child's
    /// local translation is `displacement` offset by the parent's
    /// `rest_length` along X (`bone_weights::compute_bone_world_positions`).
    pub rest_length: f32,
    /// Rest-pose offset from the parent bone's end, in the parent's local
    /// Director Z-up space.
    pub displacement: [f32; 3],
    /// Rest-pose orientation quaternion relative to the parent, stored **wxyz**
    /// (real part first, per ECMA-363) — every consumer reads it through
    /// `Quat::from_bone_wxyz`. (An earlier comment on this field labelled the
    /// order `xyzw`; the skinning and export math disagree with that label.)
    pub orientation: [f32; 4],
}

/// Skeleton definition + bone weight data (0x4B chunk).
#[derive(Debug, Clone, Serialize)]
pub struct BoneWeightChunk {
    /// Name of the mesh this skeleton skins (a 0x45 / 0x49 mesh name, or the
    /// 0x72 node name).
    pub mesh_name: String,
    /// Number of bone definitions that follow.
    pub bone_count: u32,
    /// Bone definitions in index order; `parent_index` refers into this vector.
    pub bones: Vec<BoneDef>,
    /// Raw trailing data after bone definitions (contains vertex weight assignments).
    pub weight_data: Vec<u8>,
}

/// A single keyframe in a motion track.
#[derive(Debug, Clone, Serialize)]
pub struct KeyFrame {
    /// Keyframe time in seconds, absolute in the motion's own timeline.
    pub time: f32,
    /// Bone translation at this time, in its parent's local Director Z-up space.
    pub displacement: [f32; 3],
    /// Bone rotation at this time, **wxyz** (real part first, per ECMA-363).
    pub rotation: [f32; 4],
    /// Per-axis bone scale at this time.
    pub scale: [f32; 3],
}

/// A motion track for one bone (first track only — rest are in compressed data).
#[derive(Debug, Clone, Serialize)]
pub struct MotionTrack {
    /// Name of the bone this track animates (a [`BoneDef::name`]).
    pub name: String,
    /// Number of keyframes in the track.
    pub time_count: u32,
    /// Inverse-quantization factor for this track's displacement deltas:
    /// multiplies decoded integers back into Director units.
    pub displacement_inv_quant: f32,
    /// Inverse-quantization factor for this track's scale deltas.
    pub scale_inv_quant: f32,
    /// First keyframe (raw F32, unquantized).
    pub first_keyframe: KeyFrame,
    /// Remaining compressed keyframe data (arithmetic coded).
    #[serde(skip)]
    pub compressed_data: Vec<u8>,
}

/// Motion resource from a 0x67 chunk.
#[derive(Debug, Clone, Serialize)]
pub struct MotionResource {
    /// Motion name; becomes the exported animation clip name.
    pub name: String,
    /// Number of bone tracks in the motion.
    pub track_count: u32,
    /// Inverse-quantization factor for keyframe times: multiplies decoded
    /// integer time deltas back into seconds.
    pub time_inv_quant: f32,
    /// Inverse-quantization factor for keyframe rotations: multiplies decoded
    /// integer quaternion deltas back into floats.
    pub rotation_inv_quant: f32,
    /// First track parsed from header; remaining tracks are in compressed data.
    pub first_track: Option<MotionTrack>,
    /// Raw data after the motion header (all track data).
    #[serde(skip)]
    pub raw_data: Vec<u8>,
    /// Complete chunk data for full bitstream decoding.
    #[serde(skip)]
    pub chunk_data: Vec<u8>,
}

/// Every chunk table of one parsed XMED (`3DEM`) stream, in file order.
#[derive(Debug, Clone, Serialize)]
pub struct XmedFile {
    /// 0x70 transform nodes: the scene's transform hierarchy.
    pub bones: Vec<TransformNode>,
    /// 0x72 mesh nodes: authored placements of the meshes.
    pub mesh_nodes: Vec<MeshNode>,
    /// 0x49 geometry chunks: the arithmetic-coded CLOD record streams.
    pub geometry: Vec<GeometryChunk>,
    /// 0x47 CLOD resolution-update schedules, keyed by mesh name.
    pub clod_schedules: Vec<ClodScheduleChunk>,
    /// 0x45 author-mesh declarations: counts and inverse-quantization factors.
    pub mesh_descriptions: Vec<MeshDescription>,
    /// 0x36 shader chunks: material → texture bindings.
    pub shaders: Vec<ShaderChunk>,
    /// 0x20 texture declarations: dimensions and channel count.
    pub texture_decls: Vec<TextureDecl>,
    /// 0x21 texture images: JPEG streams and zlib alpha planes.
    pub textures: Vec<TextureChunk>,
    /// 0x10 material chunks: the colour blocks shaders bind by name.
    pub materials: Vec<Material>,
    /// 0x67 motion resources: keyframe animation, one entry per motion.
    pub motions: Vec<MotionResource>,
    /// 0x4B skin chunks: skeleton definitions plus their weight payload.
    pub bone_weights: Vec<BoneWeightChunk>,
    /// 0x74 camera chunks.
    pub cameras: Vec<CameraChunk>,
}

impl XmedFile {
    /// The 0x49 chunks that carry each mesh, keyed by mesh name, in file order.
    ///
    /// 30 of the 91 geometry names in the corpus split one mesh across 2-3
    /// chunks: `Stelle/lynet.xmed` writes "lynet" three times with header record
    /// counts 255 + 255 + 103 = 613, which is exactly the `max_resolution` its
    /// single 0x45 declaration carries. The arithmetic coder restarts per chunk
    /// (each has its own bitstream) but the mesh state does not, so a group has
    /// to be decoded into one `LiveMesh`.
    ///
    /// Grouping is by *name*, not adjacency: 22 files (every `together*`,
    /// `c_together`, `TitleMeny/bakgrunn`, `Tillit/3d*`) interleave the
    /// continuation chunks of two meshes (`lynet, monica, lynet, monica, monica`),
    /// and splitting those runs restarts the resolution counter mid-mesh. Keying
    /// on the name is safe: over all 467 geometry-bearing files no file declares
    /// one 0x45 name twice, and the per-name sum of chunk record counts equals
    /// that declaration's `max_resolution` in every file.
    pub fn geometry_groups(&self) -> Vec<Vec<&GeometryChunk>> {
        let mut groups: Vec<Vec<&GeometryChunk>> = Vec::new();
        for g in &self.geometry {
            match groups.iter_mut().find(|grp| grp[0].name == g.name) {
                Some(grp) => grp.push(g),
                None => groups.push(vec![g]),
            }
        }
        groups
    }

    /// The 0x47 schedule chunk that belongs to a mesh name, if the file has one.
    pub fn clod_schedule(&self, name: &str) -> Option<&ClodScheduleChunk> {
        self.clod_schedules.iter().find(|c| c.name == name)
    }
}

// ── Chunk parsers ──────────────────────────────────────────

fn parse_transform(input: &[u8]) -> IResult<&[u8], TransformNode> {
    let (input, name) = parse_string(input)?;
    let (input, parent) = parse_string(input)?;
    let (input, _pad) = take(2usize)(input)?;
    let (input, matrix) = parse_matrix(input)?;
    Ok((
        input,
        TransformNode {
            name,
            parent,
            position: [matrix[12], matrix[13], matrix[14]],
            matrix,
        },
    ))
}

fn parse_mesh_node(input: &[u8]) -> IResult<&[u8], MeshNode> {
    let (input, name) = parse_string(input)?;
    let (input, parent) = parse_string(input)?;
    let (input, _pad) = take(2usize)(input)?;
    let (input, matrix) = parse_matrix(input)?;
    let (input, mesh_ref) = parse_string(input)?;
    let (input, resource_ref) = parse_string(input)?;
    let (input, _count) = le_u32(input)?;
    let (input, shader) = if input.len() >= 2 {
        parse_string(input)?
    } else {
        (input, String::new())
    };
    Ok((
        input,
        MeshNode {
            name,
            parent,
            position: [matrix[12], matrix[13], matrix[14]],
            matrix,
            mesh_ref,
            resource_ref,
            shader,
        },
    ))
}

fn parse_geometry(data: &[u8]) -> IResult<&[u8], GeometryChunk> {
    let (input, name) = parse_string(data)?;
    let (input, (a, b, c, d, e)) = (le_u32, le_u32, le_u32, le_u32, le_u32).parse(input)?;
    Ok((
        &[],
        GeometryChunk {
            name,
            num_positions: a,
            num_faces: b,
            num_normals: c,
            num_texcoords: d,
            field5: e,
            compressed_data: input.to_vec(),
        },
    ))
}

fn parse_clod_schedule(data: &[u8]) -> IResult<&[u8], ClodScheduleChunk> {
    let (input, name) = parse_string(data)?;
    Ok((
        &[],
        ClodScheduleChunk {
            name,
            compressed_data: input.to_vec(),
        },
    ))
}

fn parse_mesh_description(data: &[u8]) -> IResult<&[u8], MeshDescription> {
    let (input, name) = parse_string(data)?;
    let (input, (attributes, num_shaders)) = (le_u32, le_u32).parse(input)?;

    // Per-shading-group counts, 6 u32 each. `take` bounds the (untrusted) group
    // count against the chunk length before anything is allocated.
    let (input, count_bytes) = take((num_shaders as usize).saturating_mul(24))(input)?;
    let mut counts: Vec<[u32; 6]> = Vec::with_capacity(num_shaders as usize);
    let mut rest = count_bytes;
    while !rest.is_empty() {
        let (r, (positions, faces, normals, texcoords, colors, attrs)) =
            (le_u32, le_u32, le_u32, le_u32, le_u32, le_u32).parse(rest)?;
        counts.push([positions, faces, normals, texcoords, colors, attrs]);
        rest = r;
    }

    let (mut input, extra_count) = le_u32(input)?;

    // `extra_count` name arrays of `num_shaders + 1` strings each: array 0 is
    // ["default", shader × N], every later array [mesh name, material × N].
    let per_array = num_shaders as usize + 1;
    let mut arrays: Vec<Vec<String>> = Vec::new();
    for _ in 0..extra_count {
        let mut array: Vec<String> = Vec::with_capacity(per_array);
        for _ in 0..per_array {
            let (next, s) = parse_string(input)?;
            array.push(s);
            input = next;
        }
        arrays.push(array);
    }

    let (input, bounding_center) = parse_vec3(input)?;
    let (input, (bsphere, scale, pos_iq, norm_iq, tc_iq, diff_iq, spec_iq, max_res)) = (
        le_f32, le_f32, le_f32, le_f32, le_f32, le_f32, le_f32, le_u32,
    )
        .parse(input)?;

    // Array 0 names the shaders; the first array whose head matches the chunk name
    // is this mesh's material list (falls back to array 1 when the chunk is named
    // after something else). Any remaining arrays are sibling meshes.
    let shader_names = arrays.first();
    let primary = arrays
        .iter()
        .skip(1)
        .position(|a| a.first().is_some_and(|n| *n == name))
        .unwrap_or(0)
        + 1;
    let materials = arrays.get(primary);
    let shaders: Vec<ShaderSlot> = counts
        .iter()
        .enumerate()
        .map(|(i, c)| ShaderSlot {
            num_positions: c[0],
            num_faces: c[1],
            num_normals: c[2],
            num_texcoords: c[3],
            num_colors: c[4],
            attrs: c[5],
            material: materials
                .and_then(|a| a.get(i + 1))
                .cloned()
                .unwrap_or_default(),
            shader: shader_names
                .and_then(|a| a.get(i + 1))
                .cloned()
                .unwrap_or_default(),
        })
        .collect();
    let sibling_meshes: Vec<SiblingMesh> = arrays
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(i, _)| *i != primary)
        .map(|(_, a)| SiblingMesh {
            name: a[0].clone(),
            materials: a[1..].to_vec(),
        })
        .collect();

    if log::log_enabled!(target: "macromelt::mdraw", log::Level::Trace) {
        let shader_names: Vec<&str> = shaders.iter().map(|s| s.shader.as_str()).collect();
        let material_names: Vec<&str> = shaders.iter().map(|s| s.material.as_str()).collect();
        let sibling_names: Vec<&str> = sibling_meshes.iter().map(|s| s.name.as_str()).collect();
        log::trace!(
            target: "macromelt::mdraw",
            "  [MDRAW {:?} attrs=0x{:x} N={} M={} mask=0x{:x} counts={:?} shaders={:?} materials={:?} siblings={:?}]",
            name,
            attributes,
            num_shaders,
            extra_count,
            shaders.iter().fold(0, |m, s| m | s.attrs),
            counts,
            shader_names,
            material_names,
            sibling_names,
        );
    }

    Ok((
        input,
        MeshDescription {
            name,
            attributes,
            num_shaders,
            num_positions: counts.iter().fold(0u32, |t, c| t.saturating_add(c[0])),
            num_faces: counts.iter().fold(0u32, |t, c| t.saturating_add(c[1])),
            num_normals: counts.iter().fold(0u32, |t, c| t.saturating_add(c[2])),
            num_texcoords: counts.iter().fold(0u32, |t, c| t.saturating_add(c[3])),
            num_colors: counts.iter().fold(0u32, |t, c| t.saturating_add(c[4])),
            extra_count,
            shaders,
            sibling_meshes,
            bounding_center,
            bounding_sphere: bsphere,
            scale,
            position_inv_quant: pos_iq,
            normal_inv_quant: norm_iq,
            texcoord_inv_quant: tc_iq,
            diffuse_inv_quant: diff_iq,
            specular_inv_quant: spec_iq,
            max_resolution: max_res,
        },
    ))
}

/// 0x36 shader: `name, u16 flags, u16 texture_layers, u32 reserved, material,
/// [texture], raw float block`. `flags` is 3 (textured, 503 chunks) or 1
/// (untextured, 94 chunks).
fn parse_shader(data: &[u8]) -> IResult<&[u8], ShaderChunk> {
    let (input, name) = parse_string(data)?;
    let (input, (flags, texture_layers, _reserved)) = (le_u16, le_u16, le_u32).parse(input)?;
    let (input, material) = parse_string(input)?;
    let (input, texture) = if flags & 2 != 0 {
        parse_string(input)?
    } else {
        (input, String::new())
    };
    Ok((
        &[],
        ShaderChunk {
            name,
            flags,
            texture_layers,
            material,
            texture,
            raw_flags: input.to_vec(),
        },
    ))
}

/// 0x20 texture declaration: `name, 07, width, height, channels`.
fn parse_texture_decl(data: &[u8]) -> IResult<&[u8], TextureDecl> {
    let (input, name) = parse_string(data)?;
    let (input, _marker) = tag([0x07u8].as_slice())(input)?;
    let (input, (width, height, channels)) = (le_u32, le_u32, nom_u8).parse(input)?;
    Ok((
        input,
        TextureDecl {
            name,
            width,
            height,
            channels,
        },
    ))
}

/// 0x21 texture image: `name, kind`, then either a JFIF stream (kind 01) or a
/// zlib-deflated 8-bit alpha plane `width, height, size, data` (kind 03).
fn parse_texture(data: &[u8]) -> IResult<&[u8], TextureChunk> {
    let (input, name) = parse_string(data)?;
    let (input, kind) = nom_u8(input)?;
    match kind {
        1 => {
            let start = input
                .windows(3)
                .position(|w| w[0] == 0xFF && w[1] == 0xD8 && w[2] == 0xFF)
                .unwrap_or(0);
            Ok((
                &[],
                TextureChunk {
                    name,
                    image: TextureImage::Jpeg {
                        data: input[start..].to_vec(),
                    },
                },
            ))
        }
        3 => {
            let (input, (width, height, size)) = (le_u32, le_u32, le_u32).parse(input)?;
            let (input, zlib) = take(size as usize)(input)?;
            Ok((
                input,
                TextureChunk {
                    name,
                    image: TextureImage::Plane {
                        width,
                        height,
                        zlib: zlib.to_vec(),
                    },
                },
            ))
        }
        _ => Err(nom::Err::Error(NomError::new(input, ErrorKind::Tag))),
    }
}

/// 0x74 camera: `name, parent, u16 flag, [1 byte when flag != 0], 4x4 matrix,
/// u32 projection, hither, yon, fov, rect[4], 44 bytes of background/clear state,
/// the parent name again, 32-byte tail`.
///
/// The extra byte after `flag` is present for every `defaultview` camera (466 of
/// 508) and absent for MAX-authored cameras. Evidence: `Hey/npc_00.xmed` and
/// `Hey/npc_00_1.xmed` differ inside their 0x74 chunk only in the byte runs
/// 0x17-0x20, 0x27-0x31, 0x37-0x41, 0x47-0x51 — the three basis rows and the
/// translation row of a matrix starting at chunk offset 0x17, leaving the
/// constant (0,0,0,1) fourth column at 0x23/0x33/0x43/0x53 untouched.
fn parse_camera(data: &[u8]) -> IResult<&[u8], CameraChunk> {
    let (input, name) = parse_string(data)?;
    let (input, parent) = parse_string(input)?;
    let (input, flag) = le_u16(input)?;
    let (input, _extra) = take(if flag != 0 { 1usize } else { 0 })(input)?;
    let (input, matrix) = parse_matrix(input)?;
    let (input, (projection, hither, yon, fov)) = (le_u32, le_f32, le_f32, le_f32).parse(input)?;
    let (input, rect) = parse_vec4(input)?;
    Ok((
        input,
        CameraChunk {
            name,
            parent,
            position: [matrix[12], matrix[13], matrix[14]],
            matrix,
            projection,
            hither,
            yon,
            fov,
            rect,
        },
    ))
}

fn parse_material(input: &[u8]) -> IResult<&[u8], Material> {
    let (input, name) = parse_string(input)?;
    let (input, attributes) = le_u32(input)?;
    let (input, ambient) = parse_vec4(input)?;
    let (input, diffuse) = parse_vec4(input)?;
    let (input, specular) = parse_vec4(input)?;
    let (input, emissive) = parse_vec4(input)?;
    let (input, (reflectivity, opacity)) = (le_f32, le_f32).parse(input)?;
    Ok((
        input,
        Material {
            name,
            attributes,
            ambient,
            diffuse,
            specular,
            emissive,
            reflectivity,
            opacity,
        },
    ))
}

fn parse_bone_def(input: &[u8]) -> IResult<&[u8], BoneDef> {
    let (input, name) = parse_string(input)?;
    let (input, parent_index) = le_i32(input)?;
    let (input, rest_length) = le_f32(input)?;
    let (input, displacement) = parse_vec3(input)?;
    let (input, orientation) = parse_vec4(input)?;
    Ok((
        input,
        BoneDef {
            name,
            parent_index,
            rest_length,
            displacement,
            orientation,
        },
    ))
}

fn parse_bone_weight_chunk(data: &[u8]) -> IResult<&[u8], BoneWeightChunk> {
    let (input, mesh_name) = parse_string(data)?;
    let (input, bone_count) = le_u32(input)?;

    let mut bones = Vec::with_capacity(bone_count as usize);
    let mut remaining = input;

    for _i in 0..bone_count as usize {
        let (rest, bone) = parse_bone_def(remaining)?;

        // Bone Attributes (U32) — flags controlling optional sections.
        // Validated formula (100% across 885 bones in 28 files):
        //   extra = 8 * ((attrs & 0x12) == 0x12) + 8 * ((attrs & 0x24) == 0x24)
        if rest.len() < 4 {
            bones.push(bone);
            remaining = rest;
            continue;
        }
        let attrs = u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]);
        let extra_a = if (attrs & 0x12) == 0x12 { 8usize } else { 0 };
        let extra_b = if (attrs & 0x24) == 0x24 { 8usize } else { 0 };
        let skip = 4 + extra_a + extra_b;

        bones.push(bone);
        remaining = &rest[skip.min(rest.len())..];
    }

    Ok((
        &[],
        BoneWeightChunk {
            mesh_name,
            bone_count,
            bones,
            weight_data: remaining.to_vec(),
        },
    ))
}

/// Parse the first motion track: header + first unquantized keyframe.
fn parse_first_motion_track(input: &[u8]) -> IResult<&[u8], MotionTrack> {
    let (input, track_name) = parse_string(input)?;
    let (input, time_count) = le_u32(input)?;
    let (input, disp_inv_quant) = le_f32(input)?;
    let (input, scale_inv_quant) = le_f32(input)?;
    // First keyframe: time(f32) + displacement(3xf32) + rotation(4xf32) + scale(3xf32)
    let (input, time) = le_f32(input)?;
    let (input, displacement) = parse_vec3(input)?;
    let (input, rotation) = parse_vec4(input)?;
    let (input, scale) = parse_vec3(input)?;
    Ok((
        input,
        MotionTrack {
            name: track_name,
            time_count,
            displacement_inv_quant: disp_inv_quant,
            scale_inv_quant,
            first_keyframe: KeyFrame {
                time,
                displacement,
                rotation,
                scale,
            },
            compressed_data: input.to_vec(),
        },
    ))
}

fn parse_motion_resource(data: &[u8]) -> IResult<&[u8], MotionResource> {
    let (input, name) = parse_string(data)?;
    let (input, track_count) = le_u32(input)?;
    let (input, time_inv_quant) = le_f32(input)?;
    let (input, rotation_inv_quant) = le_f32(input)?;

    // Try to parse first track header + first keyframe
    let first_track = parse_first_motion_track(input).ok().map(|(_, t)| t);

    Ok((
        &[],
        MotionResource {
            name,
            track_count,
            time_inv_quant,
            rotation_inv_quant,
            first_track,
            raw_data: input.to_vec(),
            chunk_data: data.to_vec(),
        },
    ))
}

// ── File parser ────────────────────────────────────────────

const MARKER: &[u8] = &[0xFF, 0xFF, 0xFF];

/// Parse a single chunk: type(u8) + 0xFFFFFF(3) + size(u32 LE) + data(size bytes).
fn parse_chunk(input: &[u8]) -> IResult<&[u8], (u8, &[u8])> {
    let (input, chunk_type) = nom_u8(input)?;
    let (input, _marker) = tag(MARKER)(input)?;
    let (input, size) = le_u32(input)?;
    let (input, data) = take(size as usize)(input)?;
    Ok((input, (chunk_type, data)))
}

/// Parse a full XMED byte stream into its chunk tables.
///
/// `data` is the raw `XMED` cast-member payload of a Director movie: a `3DEM`
/// container holding Intel IFX v2 chunks. Chunk types the decoder does not
/// model are skipped; the bytes between chunks are scanned for the next
/// `<type> FF FF FF` marker, which is how Director's own reader resynchronises.
pub fn parse_xmed(data: &[u8]) -> Result<XmedFile, ParseError> {
    if data.len() < 0x1C {
        return Err(ParseError::Header(format!(
            "stream is {} bytes, shorter than the 0x1C-byte 3DEM header",
            data.len()
        )));
    }
    if &data[0..4] != b"3DEM" {
        return Err(ParseError::Header(format!(
            "expected magic \"3DEM\", found {:?}",
            String::from_utf8_lossy(&data[0..4])
        )));
    }
    let mut file = XmedFile {
        bones: vec![],
        mesh_nodes: vec![],
        geometry: vec![],
        clod_schedules: vec![],
        mesh_descriptions: vec![],
        shaders: vec![],
        texture_decls: vec![],
        textures: vec![],
        materials: vec![],
        motions: vec![],
        bone_weights: vec![],
        cameras: vec![],
    };

    // `expected` is the byte right after the last chunk that parsed. While the
    // walk is still contiguous a malformed length means the stream is
    // truncated and the file is unusable; once we have fallen off the chunk
    // stream the `FF FF FF` marker is only a 3-byte heuristic, so a failure
    // there is a false positive and the scan simply resynchronises.
    let mut expected = 0x1C;
    let mut pos = 0x1C;
    while pos + 8 <= data.len() {
        if pos + 4 > data.len() {
            break;
        }
        if data[pos + 1] == 0xFF && data[pos + 2] == 0xFF && data[pos + 3] == 0xFF {
            if let Ok((_, (ct, chunk_data))) = parse_chunk(&data[pos..]) {
                let chunk_end = pos + 8 + chunk_data.len();
                match ct {
                    0x70 => {
                        if let Ok((_, n)) = parse_transform(chunk_data) {
                            file.bones.push(n);
                        }
                    }
                    0x72 => {
                        if let Ok((_, n)) = parse_mesh_node(chunk_data) {
                            file.mesh_nodes.push(n);
                        }
                    }
                    0x45 => {
                        if let Ok((_, md)) = parse_mesh_description(chunk_data) {
                            file.mesh_descriptions.push(md);
                        }
                    }
                    0x47 => {
                        if let Ok((_, s)) = parse_clod_schedule(chunk_data) {
                            file.clod_schedules.push(s);
                        }
                    }
                    0x49 => {
                        if let Ok((_, g)) = parse_geometry(chunk_data) {
                            file.geometry.push(g);
                        }
                    }
                    0x36 => {
                        if let Ok((_, s)) = parse_shader(chunk_data) {
                            file.shaders.push(s);
                        }
                    }
                    0x20 => {
                        if let Ok((_, t)) = parse_texture_decl(chunk_data) {
                            file.texture_decls.push(t);
                        }
                    }
                    0x21 => {
                        if let Ok((_, t)) = parse_texture(chunk_data) {
                            file.textures.push(t);
                        }
                    }
                    0x10 => {
                        if let Ok((_, m)) = parse_material(chunk_data) {
                            file.materials.push(m);
                        }
                    }
                    0x67 => {
                        if let Ok((_, m)) = parse_motion_resource(chunk_data) {
                            file.motions.push(m);
                        }
                    }
                    0x4B => {
                        if let Ok((_, bw)) = parse_bone_weight_chunk(chunk_data) {
                            file.bone_weights.push(bw);
                        }
                    }
                    0x74 => {
                        if let Ok((_, c)) = parse_camera(chunk_data) {
                            file.cameras.push(c);
                        }
                    }
                    _ => {}
                }
                pos = chunk_end;
                expected = chunk_end;
            } else if pos == expected {
                return Err(ParseError::Chunk {
                    offset: pos,
                    reason: format!(
                        "chunk type 0x{:02X} declares a length that overruns the remaining {} bytes",
                        data[pos],
                        data.len() - pos
                    ),
                });
            } else {
                pos += 1;
            }
        } else {
            pos += 1;
        }
    }
    Ok(file)
}
