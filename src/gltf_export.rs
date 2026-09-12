//! glTF 2.0 (.glb) exporter for decoded XMED models.
//!
//! Combines:
//! - mesh geometry decoded from the XMED's own 0x49 CLOD chunks (positions,
//!   normals, UVs, faces), split into one primitive per material run
//! - textures embedded in the XMED itself (0x36 shader -> 0x21 image), with the
//!   4-channel ones' zlib alpha planes merged in
//! - skeleton hierarchy from 0x4B chunks ([`BoneDef`])
//! - authored per-vertex skin weights, falling back to the IFXSkin envelope
//!   pipeline when a skin carries none
//! - decoded animation clips from 0x67 chunks (this file's own, plus motions
//!   supplied by the caller through [`ExportOptions::extra_motions`])
//!
//! Produces the bytes of a single `.glb` binary.

// bone_weights module used for compute_bone_world_positions + point_to_segment_dist_pub
use crate::chunks::{BoneDef, Material, TextureImage, XmedFile};
use crate::clod_state::LiveMesh;
use crate::error::ExportError;
use crate::geometry::{DecodedGeometry, decode_mesh_group};
use crate::motion::{DecodedMotion, decode_motion_full};
use flate2::read::ZlibDecoder;
use log::{debug, info, warn};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{Cursor, Read};
use zune_jpeg::JpegDecoder;

// ── Source mesh ───────────────────────────────────────────

/// Mesh data for the bake, split into one group per material run.
///
/// Skinned groups carry positions in **node space** (the 0x72 translation
/// applied): the exporter subtracts the node translation again (`mesh_offset`)
/// so the vertices land in bone space, and puts it back on the `SkinRoot`.
/// Unskinned meshes stay in their own local space and are listed in `statics`;
/// each becomes a glTF node named after the Director model, carrying the 0x72
/// matrix as TRS, so a keyframe motion can drive that node exactly as
/// Director's `keyframePlayer` drives the model.
#[derive(Debug)]
struct BakeMesh {
    /// All groups in order: one per (mesh node, 0x45 shader slot).
    groups: Vec<BakeGroup>,
    /// Unskinned meshes, in group order (`BakeGroup::name` keys into it).
    statics: Vec<StaticNode>,
}

/// One unskinned Director model: a 0x72 mesh node with no 0x4B skin.
#[derive(Debug, Clone)]
struct StaticNode {
    name: String,
    /// The 0x72 node matrix, row-vector convention (translation in row 3).
    /// Authored relative to `parent` (Director `transform` is parent-relative).
    matrix: [f32; 16],
    /// The 0x72 parent name; `World` or another mesh node (`Skogen/3d_1.xmed`:
    /// `trees2 -> segment_2`), in which case the glTF node hangs under that
    /// static so the scene can move the parent and carry its children.
    parent: String,
}

/// One primitive's worth of de-indexed geometry: a (mesh node, shader slot) run.
#[derive(Debug)]
struct BakeGroup {
    /// The Director model this run came from: the geometry name for skinned
    /// meshes (the skin routing key), the node name for statics.
    name: String,
    /// Material name of the 0x45 shader slot this run belongs to.
    material: String,
    /// De-indexed vertices: each face vertex becomes a unique vertex.
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    texcoords: Vec<[f32; 2]>,
    /// Authored per-vertex skin weights from the XMED's 0x49 stream, parallel
    /// to `positions`; empty when the geometry carries none.
    weights: Vec<Vec<(u32, f32)>>,
    /// Triangle indices (already triangulated).
    indices: Vec<u32>,
}

/// Build the bake mesh from the XMED's own 0x49 geometry.
///
/// One Director model = one 0x72 mesh NODE. Several nodes may instance the same
/// geometry (`Rydde/stall.xmed`: `Box02..Box10` reference `Box01`, each with its
/// own transform and its own material list in the declaration's
/// `sibling_meshes`), so the walk is over mesh nodes, decoding each geometry
/// once. Per (node, shader slot) one group; faces from
/// `DecodedGeometry::submesh_of_face`, vertices compacted per group so
/// unreferenced positions drop out. A geometry with a 0x4B skin keeps its local
/// coordinates plus the node translation (the exporter's `mesh_offset`
/// contract) under the geometry's name; any other node stays local and is
/// listed in `statics` under the node's name.
fn native_mesh(xmed: &XmedFile) -> Result<BakeMesh, ExportError> {
    let mut groups = Vec::new();
    let mut statics: Vec<StaticNode> = Vec::new();
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let geometry_groups = xmed.geometry_groups();
    let mut decoded: HashMap<String, DecodedGeometry> = HashMap::new();
    // Every mesh node, then any geometry no node references (identity placement).
    let mut targets: Vec<(String, String, [f32; 16], String)> = xmed
        .mesh_nodes
        .iter()
        .map(|n| {
            (
                n.name.clone(),
                n.mesh_ref.clone(),
                n.matrix,
                n.parent.clone(),
            )
        })
        .collect();
    for g in &geometry_groups {
        let name = &g[0].name;
        if !targets.iter().any(|(_, geo, _, _)| geo == name) {
            warn!(
                "  geometry \"{}\" has no 0x72 mesh node; placed at identity",
                name
            );
            targets.push((name.clone(), name.clone(), identity, "World".to_string()));
        }
    }

    for (node_name, geo_name, m, parent) in targets {
        let Some(chunks) = geometry_groups.iter().find(|g| g[0].name == geo_name) else {
            warn!(
                "  mesh node \"{}\" references geometry \"{}\" that has no 0x49 chunks; skipped",
                node_name, geo_name
            );
            continue;
        };
        let md = xmed
            .mesh_descriptions
            .iter()
            .find(|m| m.name == geo_name)
            .ok_or_else(|| {
                ExportError::Other(format!("mesh \"{}\" has no 0x45 declaration", geo_name))
            })?;
        if !decoded.contains_key(&geo_name) {
            let mut live = LiveMesh::new();
            decode_mesh_group(xmed, chunks, false, &mut live);
            let geo = DecodedGeometry::from(&live);
            if geo.positions.len() != md.num_positions as usize {
                return Err(ExportError::IncompleteDecode {
                    mesh: geo_name.clone(),
                    what: "positions",
                    got: geo.positions.len(),
                    declared: md.num_positions as usize,
                });
            }
            if geo.faces.len() != md.num_faces as usize {
                return Err(ExportError::IncompleteDecode {
                    mesh: geo_name.clone(),
                    what: "faces",
                    got: geo.faces.len(),
                    declared: md.num_faces as usize,
                });
            }
            decoded.insert(geo_name.clone(), geo);
        }
        let geo = &decoded[&geo_name];
        let skinned = xmed.bone_weights.iter().any(|bw| {
            (bw.mesh_name == geo_name || bw.mesh_name == node_name) && !bw.bones.is_empty()
        });
        let linear_is_identity =
            (0..3).all(|r| (0..3).all(|c| (m[r * 4 + c] - identity[r * 4 + c]).abs() < 1e-5));
        if skinned && !linear_is_identity {
            warn!(
                "  skinned mesh \"{}\" has a rotated/scaled node matrix; only its translation is applied",
                node_name
            );
        }
        // Skinned groups are named after the geometry (skin routing key) and
        // carry the node translation; static groups are named after the node.
        let group_name = if skinned {
            geo_name.clone()
        } else {
            node_name.clone()
        };
        let apply = |p: [f32; 3]| -> [f32; 3] {
            if skinned {
                [p[0] + m[12], p[1] + m[13], p[2] + m[14]]
            } else {
                p
            }
        };
        if !skinned {
            if statics.iter().any(|s| s.name == node_name) {
                warn!(
                    "  duplicate mesh node \"{}\"; second occurrence skipped",
                    node_name
                );
                continue;
            }
            statics.push(StaticNode {
                name: node_name.clone(),
                matrix: m,
                parent,
            });
        }
        // An instance node takes its own material list from the declaration.
        let sibling = md.sibling_meshes.iter().find(|s| s.name == node_name);

        for (s, slot) in md.shaders.iter().enumerate() {
            let material = sibling
                .and_then(|sib| sib.materials.get(s).cloned())
                .unwrap_or_else(|| slot.material.clone());
            let mut remap: HashMap<u32, u32> = HashMap::new();
            let mut group = BakeGroup {
                name: group_name.clone(),
                material,
                positions: Vec::new(),
                normals: Vec::new(),
                texcoords: Vec::new(),
                weights: Vec::new(),
                indices: Vec::new(),
            };
            for (fi, face) in geo.faces.iter().enumerate() {
                if geo.submesh_of_face.get(fi).copied().unwrap_or(0) as usize != s {
                    continue;
                }
                for &corner in face {
                    let idx = *remap.entry(corner).or_insert_with(|| {
                        let i = corner as usize;
                        group.positions.push(apply(geo.positions[i]));
                        group
                            .weights
                            .push(geo.bone_weights.get(i).cloned().unwrap_or_default());
                        if let Some(n) = geo.normals.get(i) {
                            group.normals.push(*n);
                        }
                        if let Some(tc) = geo.texcoords.get(i) {
                            // XMED texcoords are bottom-up (Director/OpenGL: v = 0
                            // at the image's bottom row); glTF's origin is the
                            // top-left, so every native V is mirrored. Verified on
                            // Tillit/bane `Material #17` (tree band) and #18 (alpha
                            // trees) and the Krets fence banners, which all put
                            // v = 1 at their highest vertices and rendered inverted
                            // before this; the OBJ route never hit it because the
                            // converter wrote OBJ (v-up) UVs that Three's default
                            // `flipY` handled.
                            group.texcoords.push([tc[0], 1.0 - tc[1]]);
                        }
                        (group.positions.len() - 1) as u32
                    });
                    group.indices.push(idx);
                }
            }
            if group.indices.is_empty() {
                debug!(
                    "  Native mesh \"{}\" slot[{}] \"{}\": no faces, skipped",
                    node_name, s, group.material
                );
                continue;
            }
            debug!(
                "  Native mesh \"{}\" (geometry \"{}\") slot[{}] \"{}\": {} verts, {} faces, {}",
                node_name,
                geo_name,
                s,
                group.material,
                group.positions.len(),
                group.indices.len() / 3,
                if skinned { "skinned" } else { "static" }
            );
            groups.push(group);
        }
    }
    if groups.is_empty() {
        return Err(ExportError::Other(String::from(
            "no 0x49 geometry in this XMED",
        )));
    }
    Ok(BakeMesh { groups, statics })
}

/// A single combined mesh with all groups merged (used for skinning weight computation).
#[derive(Debug)]
struct FlatMesh {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    texcoords: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

// ── Embedded XMED textures ────────────────────────────────

/// A texture taken from the XMED itself, ready to embed in the GLB.
struct EmbeddedTexture {
    /// 0x21 chunk name (the 0x36 shader's `texture`).
    name: String,
    mime: &'static str,
    bytes: Vec<u8>,
    /// True when a 0x21 alpha `Plane` was merged in, so the material needs
    /// `alphaMode: "MASK"`.
    masked: bool,
}

/// Key that matches an XMED material name against an OBJ `usemtl` name: the
/// SW3D converter drops underscores when it writes the MTL (`hest_set` ->
/// `hestset`, `dirt_gummiskrape1` -> `dirtgummiskrape1`; verified against
/// `assets/models/obj/lynet/lynet.MTL` and
/// `assets/models/obj/together_trav2/together_trav2.MTL`).
fn material_key(name: &str) -> String {
    name.chars().filter(|c| *c != '_').collect()
}

/// Inflate a zlib alpha plane to `width * height` 8-bit samples.
fn inflate_alpha_plane(zlib: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let mut alpha = Vec::with_capacity((width * height) as usize);
    ZlibDecoder::new(zlib)
        .read_to_end(&mut alpha)
        .map_err(|e| format!("alpha plane inflate failed: {}", e))?;
    let expected = (width as usize) * (height as usize);
    if alpha.len() != expected {
        return Err(format!(
            "alpha plane inflated to {} bytes, expected {}x{} = {}",
            alpha.len(),
            width,
            height,
            expected
        ));
    }
    Ok(alpha)
}

/// Merge a 0x21 JPEG and its 0x21 alpha `Plane` into one RGBA PNG.
///
/// The plane is stored bottom-up relative to the JPEG: the SW3D converter's
/// `assets/models/obj/lynet/hale.png` has RGB byte-equal to the embedded JPEG
/// and alpha byte-equal to the inflated plane *flipped vertically* (64x128,
/// 8192 samples). So the plane rows are read in reverse here.
fn merge_alpha_png(jpeg: &[u8], plane: (u32, u32, &[u8])) -> Result<Vec<u8>, String> {
    let (plane_w, plane_h, zlib) = plane;
    let alpha = inflate_alpha_plane(zlib, plane_w, plane_h)?;

    let mut decoder = JpegDecoder::new(Cursor::new(jpeg));
    let pixels = decoder
        .decode()
        .map_err(|e| format!("JPEG decode failed: {}", e))?;
    let info = decoder.info().ok_or("JPEG has no image info")?;
    let (w, h) = (u32::from(info.width), u32::from(info.height));
    if (w, h) != (plane_w, plane_h) {
        return Err(format!(
            "alpha plane {}x{} does not match JPEG {}x{}",
            plane_w, plane_h, w, h
        ));
    }
    let px_count = (w as usize) * (h as usize);
    if px_count == 0 || pixels.len() % px_count != 0 {
        return Err(format!(
            "unexpected JPEG buffer: {} bytes for {}x{}",
            pixels.len(),
            w,
            h
        ));
    }
    let channels = pixels.len() / px_count;
    if channels != 1 && channels != 3 {
        return Err(format!("unsupported JPEG channel count {}", channels));
    }

    let mut rgba = Vec::with_capacity(px_count * 4);
    for y in 0..h {
        // Plane row (h - 1 - y) matches JPEG row y.
        let alpha_row = ((h - 1 - y) as usize) * (w as usize);
        for x in 0..w {
            let src = ((y as usize) * (w as usize) + (x as usize)) * channels;
            if channels == 1 {
                let g = pixels[src];
                rgba.extend_from_slice(&[g, g, g]);
            } else {
                rgba.extend_from_slice(&pixels[src..src + 3]);
            }
            rgba.push(alpha[alpha_row + (x as usize)]);
        }
    }

    let mut png_bytes: Vec<u8> = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut png_bytes, w, h);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|e| format!("PNG header write failed: {}", e))?;
        writer
            .write_image_data(&rgba)
            .map_err(|e| format!("PNG write failed: {}", e))?;
        writer
            .finish()
            .map_err(|e| format!("PNG finish failed: {}", e))?;
    }
    Ok(png_bytes)
}

/// Resolve the XMED's own texture bindings, keyed by [`material_key`].
///
/// Binding chain: 0x36 shader `material` -> that shader's `texture` -> the 0x21
/// chunk(s) of that name. A 4-channel texture (0x20 `channels == 4`) emits two
/// 0x21 chunks under one name — the RGB JPEG and a zlib alpha plane — which are
/// merged into a masked RGBA PNG. Opaque textures are embedded as the original
/// JPEG, untouched.
fn build_embedded_textures(xmed: &XmedFile) -> HashMap<String, EmbeddedTexture> {
    let mut out: HashMap<String, EmbeddedTexture> = HashMap::new();
    for shader in &xmed.shaders {
        if shader.material.is_empty() || shader.texture.is_empty() {
            continue;
        }
        let key = material_key(&shader.material);
        if out.contains_key(&key) {
            continue;
        }
        let jpeg = xmed.textures.iter().find_map(|t| match &t.image {
            TextureImage::Jpeg { data } if t.name == shader.texture => Some(data),
            _ => None,
        });
        let plane = xmed.textures.iter().find_map(|t| match &t.image {
            TextureImage::Plane {
                width,
                height,
                zlib,
            } if t.name == shader.texture => Some((*width, *height, zlib.as_slice())),
            _ => None,
        });
        let Some(jpeg) = jpeg else {
            warn!(
                "  shader \"{}\" names texture \"{}\" but the XMED has no JPEG chunk for it",
                shader.name, shader.texture
            );
            continue;
        };
        let (mime, bytes, masked) = match plane {
            Some(p) => match merge_alpha_png(jpeg, p) {
                Ok(png) => ("image/png", png, true),
                Err(e) => {
                    warn!(
                        "  texture \"{}\" alpha merge failed ({}) — embedding opaque JPEG",
                        shader.texture, e
                    );
                    ("image/jpeg", jpeg.clone(), false)
                }
            },
            None => ("image/jpeg", jpeg.clone(), false),
        };
        out.insert(
            key,
            EmbeddedTexture {
                name: shader.texture.clone(),
                mime,
                bytes,
                masked,
            },
        );
    }
    out
}

/// Flatten all groups into a single mesh (used for skinning weight computation).
fn flatten_groups(obj: &BakeMesh) -> FlatMesh {
    let mut flat = FlatMesh {
        positions: Vec::new(),
        normals: Vec::new(),
        texcoords: Vec::new(),
        indices: Vec::new(),
    };
    for group in &obj.groups {
        let base = flat.positions.len() as u32;
        flat.positions.extend_from_slice(&group.positions);
        flat.normals.extend_from_slice(&group.normals);
        flat.texcoords.extend_from_slice(&group.texcoords);
        for &idx in &group.indices {
            flat.indices.push(base + idx);
        }
    }
    flat
}

// ── Buffer helpers ────────────────────────────────────────

/// A growing binary buffer that tracks byte offsets for each accessor.
struct GltfBuffer {
    data: Vec<u8>,
}

impl GltfBuffer {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    /// Pad to alignment boundary.
    fn align(&mut self, alignment: usize) {
        while self.data.len() % alignment != 0 {
            self.data.push(0);
        }
    }

    fn write_f32(&mut self, v: f32) {
        self.data.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u16(&mut self, v: u16) {
        self.data.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u32(&mut self, v: u32) {
        self.data.extend_from_slice(&v.to_le_bytes());
    }

    /// Write a 4x4 matrix as 16 floats (column-major for glTF).
    fn write_mat4(&mut self, m: &[f32; 16]) {
        for &v in m {
            self.write_f32(v);
        }
    }

    /// Write raw bytes directly.
    fn write_bytes(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
    }
}

// ── Skeleton math ─────────────────────────────────────────

/// Simple quaternion type for transform math.
#[derive(Debug, Clone, Copy)]
struct Quat {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl Quat {
    /// Create from BoneDef orientation (stored as wxyz per ECMA-363 9.6.1.1.6.7).
    fn from_bone_wxyz(q: [f32; 4]) -> Self {
        Self {
            w: q[0],
            x: q[1],
            y: q[2],
            z: q[3],
        }
    }

    #[allow(dead_code)]
    fn to_xyzw(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }
}

/// Quaternion multiply for wxyz-ordered inputs. Returns wxyz.
/// Matches bone_weights::quat_mul_wxyz / main::quat_mul_wxyz.
fn quat_mul_wxyz(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let (aw, ax, ay, az) = (a[0], a[1], a[2], a[3]);
    let (bw, bx, by, bz) = (b[0], b[1], b[2], b[3]);
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

/// Quaternion conjugate for wxyz-ordered input. For unit quats this equals
/// the inverse.
fn quat_conjugate_wxyz(q: [f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

/// Renormalize a wxyz quaternion (safety after repeated multiplies).
fn quat_normalize_wxyz(q: [f32; 4]) -> [f32; 4] {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if n2 > 1e-12 {
        let inv = 1.0 / n2.sqrt();
        [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

/// Multiply two column-major 4x4 matrices: result = a * b.
fn mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut r = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            r[col * 4 + row] = a[row] * b[col * 4]
                + a[4 + row] * b[col * 4 + 1]
                + a[8 + row] * b[col * 4 + 2]
                + a[12 + row] * b[col * 4 + 3];
        }
    }
    r
}

/// Build a column-major 4x4 TRS matrix with per-axis scale.
fn trs_to_mat4(t: [f32; 3], q: &Quat, s: [f32; 3]) -> [f32; 16] {
    let Quat { x, y, z, w } = *q;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    [
        s[0] * (1.0 - 2.0 * (yy + zz)),
        s[0] * (2.0 * (xy + wz)),
        s[0] * (2.0 * (xz - wy)),
        0.0,
        s[1] * (2.0 * (xy - wz)),
        s[1] * (1.0 - 2.0 * (xx + zz)),
        s[1] * (2.0 * (yz + wx)),
        0.0,
        s[2] * (2.0 * (xz + wy)),
        s[2] * (2.0 * (yz - wx)),
        s[2] * (1.0 - 2.0 * (xx + yy)),
        0.0,
        t[0],
        t[1],
        t[2],
        1.0,
    ]
}

/// Decompose a column-major 4x4 TRS matrix into (translation, rotation_xyzw, scale).
/// Assumes the matrix has no shear (only scale, rotation, translation).
fn decompose_trs(m: &[f32; 16]) -> ([f32; 3], [f32; 4], [f32; 3]) {
    let t = [m[12], m[13], m[14]];

    // Column lengths give per-axis scale magnitudes.
    let sx = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
    let sy = (m[4] * m[4] + m[5] * m[5] + m[6] * m[6]).sqrt();
    let sz = (m[8] * m[8] + m[9] * m[9] + m[10] * m[10]).sqrt();

    // Sign-correct the scale if the rotation block has negative determinant
    // (one column is mirrored). Without this, decompose -> recompose changes
    // chirality.
    let det = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[4] * (m[1] * m[10] - m[2] * m[9])
        + m[8] * (m[1] * m[6] - m[2] * m[5]);
    let (sx, sy, sz) = if det < 0.0 {
        (-sx, sy, sz)
    } else {
        (sx, sy, sz)
    };

    let inv_sx = if sx.abs() > 1e-8 { 1.0 / sx } else { 0.0 };
    let inv_sy = if sy.abs() > 1e-8 { 1.0 / sy } else { 0.0 };
    let inv_sz = if sz.abs() > 1e-8 { 1.0 / sz } else { 0.0 };

    // Normalised rotation matrix (column-major 3x3).
    let r00 = m[0] * inv_sx;
    let r01 = m[1] * inv_sx;
    let r02 = m[2] * inv_sx;
    let r10 = m[4] * inv_sy;
    let r11 = m[5] * inv_sy;
    let r12 = m[6] * inv_sy;
    let r20 = m[8] * inv_sz;
    let r21 = m[9] * inv_sz;
    let r22 = m[10] * inv_sz;

    // Rotation-matrix-to-quaternion (Shoemake's stable form). Note: stored in
    // column-major; r_ij accessed above means r[col i][row j].
    // For clarity, build a 3x3 view in row-major: m3[row][col].
    let m3 = [[r00, r10, r20], [r01, r11, r21], [r02, r12, r22]];
    let tr = m3[0][0] + m3[1][1] + m3[2][2];
    let (qw, qx, qy, qz) = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        (
            0.25 * s,
            (m3[2][1] - m3[1][2]) / s,
            (m3[0][2] - m3[2][0]) / s,
            (m3[1][0] - m3[0][1]) / s,
        )
    } else if m3[0][0] > m3[1][1] && m3[0][0] > m3[2][2] {
        let s = (1.0 + m3[0][0] - m3[1][1] - m3[2][2]).sqrt() * 2.0;
        (
            (m3[2][1] - m3[1][2]) / s,
            0.25 * s,
            (m3[0][1] + m3[1][0]) / s,
            (m3[0][2] + m3[2][0]) / s,
        )
    } else if m3[1][1] > m3[2][2] {
        let s = (1.0 + m3[1][1] - m3[0][0] - m3[2][2]).sqrt() * 2.0;
        (
            (m3[0][2] - m3[2][0]) / s,
            (m3[0][1] + m3[1][0]) / s,
            0.25 * s,
            (m3[1][2] + m3[2][1]) / s,
        )
    } else {
        let s = (1.0 + m3[2][2] - m3[0][0] - m3[1][1]).sqrt() * 2.0;
        (
            (m3[1][0] - m3[0][1]) / s,
            (m3[0][2] + m3[2][0]) / s,
            (m3[1][2] + m3[2][1]) / s,
            0.25 * s,
        )
    };

    // Round near-unit components to clean defaults so glTF stays compact.
    let q = [qx, qy, qz, qw];
    let q = if (q[0].abs() < 1e-6)
        && (q[1].abs() < 1e-6)
        && (q[2].abs() < 1e-6)
        && (q[3] - 1.0).abs() < 1e-6
    {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        q
    };

    ([t[0], t[1], t[2]], q, [sx, sy, sz])
}

/// Invert a column-major 4x4 matrix (general case, supports scale).
fn invert_mat4(m: &[f32; 16]) -> [f32; 16] {
    let [
        m0,
        m1,
        m2,
        m3,
        m4,
        m5,
        m6,
        m7,
        m8,
        m9,
        m10,
        m11,
        m12,
        m13,
        m14,
        m15,
    ] = *m;
    let a00 = m0;
    let a01 = m1;
    let a02 = m2;
    let a03 = m3;
    let a10 = m4;
    let a11 = m5;
    let a12 = m6;
    let a13 = m7;
    let a20 = m8;
    let a21 = m9;
    let a22 = m10;
    let a23 = m11;
    let a30 = m12;
    let a31 = m13;
    let a32 = m14;
    let a33 = m15;

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if det.abs() < 1e-10 {
        return [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
    }
    let inv_det = 1.0 / det;
    [
        (a11 * b11 - a12 * b10 + a13 * b09) * inv_det,
        (-a01 * b11 + a02 * b10 - a03 * b09) * inv_det,
        (a31 * b05 - a32 * b04 + a33 * b03) * inv_det,
        (-a21 * b05 + a22 * b04 - a23 * b03) * inv_det,
        (-a10 * b11 + a12 * b08 - a13 * b07) * inv_det,
        (a00 * b11 - a02 * b08 + a03 * b07) * inv_det,
        (-a30 * b05 + a32 * b02 - a33 * b01) * inv_det,
        (a20 * b05 - a22 * b02 + a23 * b01) * inv_det,
        (a10 * b10 - a11 * b08 + a13 * b06) * inv_det,
        (-a00 * b10 + a01 * b08 - a03 * b06) * inv_det,
        (a30 * b04 - a31 * b02 + a33 * b00) * inv_det,
        (-a20 * b04 + a21 * b02 - a23 * b00) * inv_det,
        (-a10 * b09 + a11 * b07 - a12 * b06) * inv_det,
        (a00 * b09 - a01 * b07 + a02 * b06) * inv_det,
        (-a30 * b03 + a31 * b01 - a32 * b00) * inv_det,
        (a20 * b03 - a21 * b01 + a22 * b00) * inv_det,
    ]
}

// ── Main export ───────────────────────────────────────────

/// Knobs for [`export_glb`].
#[derive(Debug, Default, Clone)]
pub struct ExportOptions {
    /// Motions decoded from other XMED files, appended as clips. This is the
    /// Lingo's `cloneMotionFromCastmember`: a Director movie routinely keeps a
    /// rig's motions in separate cast members and clones them onto the model.
    pub extra_motions: Vec<DecodedMotion>,
    /// Clip renames applied last, `from -> to`. Director plays a motion under
    /// the name the Lingo clones it as, which need not be the name in the XMED
    /// (`n_logo-Key` in the file, `n_logo-key` in `TitleMeny/INITIALIZE.ls`).
    pub clip_names: HashMap<String, String>,
    /// Keep every `SkinRoot` a direct child of `SceneRoot` even when the XMED
    /// parents the skins under an authored transform node, and ground the rig
    /// so its lowest hull vertex sits at Director Z = 0. Used for mount rigs
    /// whose rider is seated at run time rather than at bake time.
    pub flat_skins: bool,
}

/// Bake a parsed XMED into the bytes of a glTF 2.0 binary (`.glb`).
///
/// Geometry, skins, materials and textures all come from the XMED's own
/// chunks; `options` only adds motions, renames clips and selects the
/// flat-skin layout. Coordinates stay Director Z-up — consumers convert.
pub fn export_glb(xmed: &XmedFile, options: &ExportOptions) -> Result<Vec<u8>, ExportError> {
    let extra_anim_motions: &[DecodedMotion] = &options.extra_motions;
    let clip_names = &options.clip_names;
    let flat_skins = options.flat_skins;
    let obj = native_mesh(xmed)?;

    // Textures the XMED carries itself (0x36 shader -> 0x21 image). These win
    // over the converter's MTL PNGs: they are the original texture the Director
    // shader binds, and 4-channel ones carry their alpha plane.
    let embedded_textures = build_embedded_textures(xmed);
    for tex in embedded_textures.values() {
        info!(
            "Embedded texture: {} ({} bytes, {}{})",
            tex.name,
            tex.bytes.len(),
            tex.mime,
            if tex.masked {
                ", alpha plane merged"
            } else {
                ""
            }
        );
    }

    let flat_mesh = flatten_groups(&obj);

    info!(
        "Mesh: {} groups, {} total positions, {} total normals, {} total texcoords, {} total indices ({} triangles)",
        obj.groups.len(),
        flat_mesh.positions.len(),
        flat_mesh.normals.len(),
        flat_mesh.texcoords.len(),
        flat_mesh.indices.len(),
        flat_mesh.indices.len() / 3,
    );

    // ── Per-skin setup ────────────────────────────────────────
    // The XMED can contain multiple skeletons (e.g. c_together has chhest + christian).
    // Each bone_weights entry defines one skin. OBJ groups are routed to skins by name.
    // Motions are routed to skins by bone-name overlap.

    struct SkinData<'a> {
        mesh_name: String, // e.g. "chhest" or "christian"
        bones: &'a [BoneDef],
        mesh_offset: [f32; 3], // from matching mesh_node (0x72)
        correction: [f32; 16],
        bone_world_pos: Vec<crate::bone_weights::BoneWorldPos>,
        bone_name_to_idx: HashMap<String, usize>,
        joint_node_offset: usize, // filled in later: glTF node index of joint 0
    }

    let identity_mat4 = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let mut skins_data: Vec<SkinData> = Vec::new();
    for bw in &xmed.bone_weights {
        if bw.bones.is_empty() {
            continue;
        }
        // Find the mesh_node whose mesh_ref matches this skin's mesh_name
        let mesh_offset: [f32; 3] = xmed
            .mesh_nodes
            .iter()
            .find(|mn| mn.mesh_ref == bw.mesh_name || mn.name == bw.mesh_name)
            .map(|mn| mn.position)
            .unwrap_or([0.0; 3]);
        if mesh_offset != [0.0; 3] {
            info!(
                "Skin \"{}\" mesh node offset: ({:.2}, {:.2}, {:.2})",
                bw.mesh_name, mesh_offset[0], mesh_offset[1], mesh_offset[2]
            );
        }
        let bone_world_pos = crate::bone_weights::compute_bone_world_positions(&bw.bones);
        let bone_name_to_idx: HashMap<String, usize> = bw
            .bones
            .iter()
            .enumerate()
            .map(|(i, b)| (b.name.clone(), i))
            .collect();
        skins_data.push(SkinData {
            mesh_name: bw.mesh_name.clone(),
            bones: &bw.bones,
            mesh_offset,
            correction: identity_mat4, // filled in below after routing
            bone_world_pos,
            bone_name_to_idx,
            joint_node_offset: 0, // assigned below
        });
    }

    let has_skeleton = !skins_data.is_empty();
    let num_skins = skins_data.len();

    // Groups are named after their XMED mesh, so routing is exact: a mesh with
    // a same-named 0x4B skin is skinned, anything else (`sal`, `pled`, scene
    // props) stays a static primitive on its own mesh node.
    let group_to_skin: Vec<Option<usize>> = obj
        .groups
        .iter()
        .map(|g| skins_data.iter().position(|sk| sk.mesh_name == g.name))
        .collect();

    // 0x49 positions are already in bone space (the 0x4B displacements sit
    // inside the decoded hull), so every skin keeps the identity correction it
    // was built with.
    for skin in &skins_data {
        info!("  Skin \"{}\": correction = identity", skin.mesh_name);
    }

    // Compute ground_shift: a uniform Director-Z translation applied to every
    // SkinRoot so the "ground skin" (the one with toe bones) renders its lowest
    // hoof vertex at Director Z = 0 (three.js Y = 0 after the SceneRoot rotation).
    //
    // The lowest decoded vertex of the toe-bone skin (node-local, i.e. what its
    // SkinRoot renders at `mesh_offset`) is lifted to z = 0.
    //
    // Applied uniformly to ALL skins so relative positions are preserved
    // (rider stays in his XMED-defined position relative to horse).
    //
    // The horse's origin ends up at its hooves, which is the contract
    // `RiderSeat` relies on. The authored `together` / `c_together` group
    // translation (z 21.71) is NOT baked in: the scene supplies it as
    // `RigNodes::group` when it seats the rider.
    // Only the flat-skins mount rigs are grounded; every other bake keeps the
    // authored node offsets (its scene places the `SkinRoot` itself, e.g.
    // Questions), and its statics (lynet's `sal`/`pled`) stay where the 0x72
    // chunks put them relative to the skin.
    let ground_shift_z: f32 = if has_skeleton && flat_skins {
        let ground_skin = skins_data
            .iter()
            .enumerate()
            .find(|(_, s)| s.bones.iter().any(|b| b.name.contains("toe")));
        match ground_skin {
            Some((si, skin)) => {
                let min_z = obj
                    .groups
                    .iter()
                    .enumerate()
                    .filter(|(gi, _)| group_to_skin[*gi] == Some(si))
                    .flat_map(|(_, g)| g.positions.iter().map(|p| p[2]))
                    .fold(f32::INFINITY, f32::min);
                if !min_z.is_finite() {
                    0.0
                } else {
                    let shift = -(min_z + skin.mesh_offset[2]);
                    info!(
                        "  ground_shift: ground skin[{}] \"{}\" lowest local Z={:.3} mesh_offset Z={:.3} shift Z={:.3}",
                        si, skin.mesh_name, min_z, skin.mesh_offset[2], shift
                    );
                    shift
                }
            }
            None => {
                info!("  ground_shift: no skin has toe bones; ground_shift=0");
                0.0
            }
        }
    } else {
        0.0
    };

    // Decode animations from base XMED, then the --anim files. `clip_names`
    // renames clips to the name the Lingo clones them under (`n_logo-Key` in the
    // XMED is played as `n_logo-key`, TitleMeny/INITIALIZE.ls:41-43).
    let mut decoded_motions: Vec<DecodedMotion> =
        xmed.motions.iter().map(|m| decode_motion_full(m)).collect();
    decoded_motions.extend_from_slice(extra_anim_motions);
    for m in &mut decoded_motions {
        if let Some(to) = clip_names.get(&m.name) {
            info!("  Clip \"{}\" renamed to \"{}\"", m.name, to);
            m.name = to.clone();
        }
    }
    info!(
        "Total animations: {} (base: {}, extra: {})",
        decoded_motions.len(),
        xmed.motions.len(),
        extra_anim_motions.len(),
    );

    // ── Keyframe (`keyframePlayer`) motions ──────────────────────────────
    // A Director keyframe motion is a single track that drives a MODEL's
    // transform (`Hey/trolleyguy.ls:15,25` clones `tralle_pickuphay` as
    // `tralle_pickuphay-Key`; `TitleMeny/INITIALIZE.ls:41-43` plays `n_logo-key`
    // on the language logo). It targets the native static node of the same
    // name, or - the Lingo clones motions onto differently named models - the
    // bake's only static node. Bones motions never have exactly one track.
    let statics: &[StaticNode] = &obj.statics;
    let is_bone_track = |name: &str| {
        skins_data
            .iter()
            .any(|sk| sk.bone_name_to_idx.contains_key(name))
    };
    let keyframe_targets: Vec<(usize, usize)> = decoded_motions
        .iter()
        .enumerate()
        .filter(|(_, m)| m.tracks.len() == 1 && !is_bone_track(&m.tracks[0].name))
        .filter_map(|(mi, m)| {
            let track = m.tracks[0].name.as_str();
            let target = statics
                .iter()
                .position(|s| s.name == track)
                .or_else(|| (statics.len() == 1).then_some(0));
            match target {
                Some(si) => {
                    info!(
                        "  Keyframe motion \"{}\" (track \"{}\") → static node \"{}\"",
                        m.name, track, statics[si].name
                    );
                    Some((mi, si))
                }
                None => {
                    warn!(
                        "  dropping keyframe motion \"{}\" — no static node named \"{}\" ({} statics)",
                        m.name,
                        track,
                        statics.len()
                    );
                    None
                }
            }
        })
        .collect();
    let static_animated: Vec<bool> = (0..statics.len())
        .map(|si| keyframe_targets.iter().any(|&(_, t)| t == si))
        .collect();

    // Node layout:
    //   0: SceneRoot
    //   1..: skin mesh nodes (one per skin), or the single `Mesh` node of an
    //       unskinned OBJ bake
    //   then per native static mesh, in order: [`<name>:start` (the authored
    //       0x72 TRS, only when a keyframe motion drives it)] `<name>` (the
    //       Director model node: mesh, keyframe channels, identity rest when
    //       animated - Director's keyframePlayer plays relative to the transform
    //       at play start, and Three's mixer writes absolute TRS)
    //   then SkinRoot nodes (parents of each skin's joints, carrying
    //       T(mesh_offset) - mesh-node TRS is ignored on SkinnedMeshes per glTF)
    //   then joint nodes (skin 0's joints, then skin 1's, ...)
    let num_skin_mesh_nodes = if has_skeleton {
        num_skins
    } else {
        usize::from(statics.is_empty())
    };
    // glTF node index of each static's model node, and of its `:start` parent.
    let mut static_node_idx: Vec<usize> = Vec::with_capacity(statics.len());
    let mut static_start_idx: Vec<Option<usize>> = Vec::with_capacity(statics.len());
    {
        let mut next = 1 + num_skin_mesh_nodes;
        for si in 0..statics.len() {
            if static_animated[si] {
                static_start_idx.push(Some(next));
                next += 1;
            } else {
                static_start_idx.push(None);
            }
            static_node_idx.push(next);
            next += 1;
        }
        let first_skin_root = next;
        let mut next_joint_node = first_skin_root + num_skins;
        for skin in skins_data.iter_mut() {
            skin.joint_node_offset = next_joint_node;
            next_joint_node += skin.bones.len();
        }
    }
    let first_skin_root_node =
        1 + num_skin_mesh_nodes + static_node_idx.len() + static_start_idx.iter().flatten().count();
    // glTF mesh index of each static: after the skin (or OBJ) meshes.
    let static_mesh_of = |si: usize| num_skin_mesh_nodes + si;
    let num_gltf_meshes = num_skin_mesh_nodes + statics.len();

    // For backward compatibility / printing: alias the "primary" skin (first one).
    let primary_skin_idx: usize = 0;
    let primary_bones: &[BoneDef] = if has_skeleton {
        skins_data[primary_skin_idx].bones
    } else {
        &[]
    };
    let primary_mesh_offset: [f32; 3] = if has_skeleton {
        skins_data[primary_skin_idx].mesh_offset
    } else {
        [0.0; 3]
    };

    // Debug: compare BoneDef rest pose vs animation first keyframe
    if has_skeleton && !decoded_motions.is_empty() {
        let bone_name_to_idx: HashMap<String, usize> = primary_bones
            .iter()
            .enumerate()
            .map(|(i, b)| (b.name.clone(), i))
            .collect();
        debug!("=== BoneDef vs Animation first KF comparison ===");
        for track in &decoded_motions[0].tracks {
            if let Some(&bi) = bone_name_to_idx.get(&track.name) {
                let bone = &primary_bones[bi];
                let kf0 = &track.keyframes[0];
                let d_diff = [
                    kf0.displacement[0] - bone.displacement[0],
                    kf0.displacement[1] - bone.displacement[1],
                    kf0.displacement[2] - bone.displacement[2],
                ];
                let r_diff = [
                    kf0.rotation[0] - bone.orientation[0],
                    kf0.rotation[1] - bone.orientation[1],
                    kf0.rotation[2] - bone.orientation[2],
                    kf0.rotation[3] - bone.orientation[3],
                ];
                let d_mag =
                    (d_diff[0] * d_diff[0] + d_diff[1] * d_diff[1] + d_diff[2] * d_diff[2]).sqrt();
                let r_mag = (r_diff[0] * r_diff[0]
                    + r_diff[1] * r_diff[1]
                    + r_diff[2] * r_diff[2]
                    + r_diff[3] * r_diff[3])
                    .sqrt();
                if d_mag > 0.01 || r_mag > 0.01 {
                    debug!(
                        "  MISMATCH bone[{}] \"{}\": disp_diff=({:.4},{:.4},{:.4}) mag={:.4}  rot_diff=({:.4},{:.4},{:.4},{:.4}) mag={:.4}",
                        bi,
                        track.name,
                        d_diff[0],
                        d_diff[1],
                        d_diff[2],
                        d_mag,
                        r_diff[0],
                        r_diff[1],
                        r_diff[2],
                        r_diff[3],
                        r_mag,
                    );
                    debug!(
                        "    BoneDef: d=({:.4},{:.4},{:.4}) q=({:.4},{:.4},{:.4},{:.4})",
                        bone.displacement[0],
                        bone.displacement[1],
                        bone.displacement[2],
                        bone.orientation[0],
                        bone.orientation[1],
                        bone.orientation[2],
                        bone.orientation[3],
                    );
                    debug!(
                        "    AnimKF0: d=({:.4},{:.4},{:.4}) q=({:.4},{:.4},{:.4},{:.4})",
                        kf0.displacement[0],
                        kf0.displacement[1],
                        kf0.displacement[2],
                        kf0.rotation[0],
                        kf0.rotation[1],
                        kf0.rotation[2],
                        kf0.rotation[3],
                    );
                }
            }
        }
        debug!("=== End comparison ===");
    }

    // Build the glTF
    let mut buf = GltfBuffer::new();
    let mut accessors: Vec<Value> = Vec::new();
    let mut buffer_views: Vec<Value> = Vec::new();

    // Helper: add a buffer view + accessor
    let add_accessor = |buf: &mut GltfBuffer,
                        bv_list: &mut Vec<Value>,
                        acc_list: &mut Vec<Value>,
                        component_type: u32,
                        count: usize,
                        acc_type: &str,
                        min_val: Option<Value>,
                        max_val: Option<Value>,
                        target: Option<u32>|
     -> (usize, usize) {
        let bv_idx = bv_list.len();
        let acc_idx = acc_list.len();
        let byte_offset = buf.len();

        // We'll fill the data after this call, so mark the start
        let mut bv = json!({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": 0, // placeholder
        });
        if let Some(t) = target {
            bv["target"] = json!(t);
        }
        bv_list.push(bv);

        let mut acc = json!({
            "bufferView": bv_idx,
            "componentType": component_type,
            "count": count,
            "type": acc_type,
        });
        if let Some(mn) = min_val {
            acc["min"] = mn;
        }
        if let Some(mx) = max_val {
            acc["max"] = mx;
        }
        acc_list.push(acc);

        // Return (buffer_view_index, accessor_index)
        // IMPORTANT: use bv_idx for buffer_views updates, acc_idx for accessor references
        (bv_idx, acc_idx)
    };

    // ── Per-group mesh data ──────────────────────────────────
    // Each OBJ group (usemtl block) becomes a separate glTF primitive.
    // With multi-skeleton support, each primitive is routed to one of the skins
    // (via OBJ group name → bone_weights mesh_name matching).

    struct PrimitiveInfo {
        pos_acc: usize,
        normal_acc: Option<usize>,
        texcoord_acc: Option<usize>,
        index_acc: usize,
        joints_acc: Option<usize>,
        weights_acc: Option<usize>,
        material_name: String,
        skin_idx: Option<usize>, // which skin this primitive belongs to (None for unrigged)
        /// Native route: index into `obj.statics` for an unskinned primitive.
        static_idx: Option<usize>,
    }

    // The XMED skeleton (0x4B) and mesh (0x49) are in the same local coordinate space.
    // Bones represent the centerline of each limb — the mesh extends around them with
    // finite thickness. No scaling is needed; non-uniform root_scale causes animation
    // deformation artifacts because it permanently scales all joint transforms.
    let root_scale: [f32; 3] = [1.0; 3];

    // Both bones and mesh should be in the same local coordinate space.
    // Instead of adding mesh_offset to bones (bone_offset = mesh_offset), we
    // SUBTRACT mesh_offset from vertices (transform_vertex = v - mesh_offset).
    // This puts everything in the XMED's local space where bones are defined.
    // bone_offset = [0,0,0] means the root joint sits at its raw displacement.
    let bone_offset: [f32; 3] = [0.0; 3];

    // Debug-print bone info for the primary skin (preserves existing logging behavior).
    if has_skeleton {
        debug!(
            "  bone_offset: ({:.2},{:.2},{:.2}) (bones in local space)",
            bone_offset[0], bone_offset[1], bone_offset[2]
        );
        debug!(
            "  primary mesh_offset subtracted from vertices: ({:.2},{:.2},{:.2})",
            primary_mesh_offset[0], primary_mesh_offset[1], primary_mesh_offset[2]
        );
        // Debug: print key bone positions
        let primary_bwp = &skins_data[primary_skin_idx].bone_world_pos;
        for (i, bp) in primary_bwp.iter().enumerate() {
            let name = &primary_bones[i].name;
            if name.contains("head")
                || name.contains("neck")
                || name.contains("tail")
                || name.contains("tale")
                || name == "center"
                || name.contains("body")
                || name.contains("leg")
                || name.contains("toe")
                || name.contains("front")
            {
                debug!(
                    "  bone[{:2}] {:25} start=({:7.2},{:7.2},{:7.2}) end=({:7.2},{:7.2},{:7.2}) len={:.2}",
                    i,
                    name,
                    bp.start[0],
                    bp.start[1],
                    bp.start[2],
                    bp.end[0],
                    bp.end[1],
                    bp.end[2],
                    primary_bones[i].rest_length
                );
            }
        }
    }

    // ── Skin weights, per skin ────────────────────────────────────────────
    // The XMED's own 0x49 stream carries authored per-vertex weights (count /
    // bone ids / quantized weights, `LiveMesh::bone_weights`); those are what
    // Shockwave binds with, so they win whenever the geometry route decoded
    // them. Only a rig without them (the OBJ fallback route) falls back to
    // regenerating weights with the IFXSkin pipeline, which needs the WHOLE
    // skin mesh at once (joint cross-sections + neighbour smoothing), so it
    // runs over the combined bone-space vertices and is split back per group.
    let mut group_weights: Vec<Vec<Vec<(usize, f32)>>> = obj
        .groups
        .iter()
        .map(|g| vec![Vec::new(); g.positions.len()])
        .collect();
    if has_skeleton {
        for si in 0..num_skins {
            let correction = skins_data[si].correction;
            let mesh_offset = skins_data[si].mesh_offset;
            let raw = |p: &[f32; 3]| -> [f32; 3] {
                [
                    correction[0] * (p[0] - mesh_offset[0]) + correction[12],
                    correction[5] * (p[1] - mesh_offset[1]) + correction[13],
                    correction[10] * (p[2] - mesh_offset[2]) + correction[14],
                ]
            };
            let real_bones: Vec<usize> = (0..skins_data[si].bones.len())
                .filter(|&bi| !skins_data[si].bones[bi].name.starts_with("noanimate"))
                .collect();

            let mut verts: Vec<[f32; 3]> = Vec::new();
            let mut faces: Vec<[usize; 3]> = Vec::new();
            let mut spans: Vec<(usize, usize, usize)> = Vec::new(); // (group_idx, base, count)
            for (gi, g) in obj.groups.iter().enumerate() {
                if group_to_skin[gi] != Some(si) {
                    continue;
                }
                let base = verts.len();
                for p in &g.positions {
                    verts.push(raw(p));
                }
                for tri in g.indices.chunks(3) {
                    if tri.len() == 3 {
                        faces.push([
                            base + tri[0] as usize,
                            base + tri[1] as usize,
                            base + tri[2] as usize,
                        ]);
                    }
                }
                spans.push((gi, base, g.positions.len()));
            }
            if verts.is_empty() {
                continue;
            }
            // Authored weights: bone ids index this skin's bone table directly.
            let authored = spans
                .iter()
                .all(|&(gi, _, _)| obj.groups[gi].weights.iter().all(|w| !w.is_empty()));
            if authored {
                let nbones = skins_data[si].bones.len();
                for &(gi, _, _) in &spans {
                    group_weights[gi] = obj.groups[gi]
                        .weights
                        .iter()
                        .map(|list| {
                            list.iter()
                                .filter(|&&(b, w)| (b as usize) < nbones && w > 0.0)
                                .map(|&(b, w)| (b as usize, w))
                                .collect()
                        })
                        .collect();
                }
                continue;
            }
            let dense = crate::skin_weights::generate_weights(
                &verts,
                &faces,
                skins_data[si].bones,
                &real_bones,
            );
            for (gi, base, count) in spans {
                group_weights[gi] = dense[base..base + count].to_vec();
            }
        }
    }

    let mut primitives_info: Vec<PrimitiveInfo> = Vec::new();

    for (group_idx, group) in obj.groups.iter().enumerate() {
        let group_vertex_count = group.positions.len();

        // Routing was computed up-front (above), to ensure per-skin correction
        // sees only this skin's own OBJ groups.
        let group_skin_idx = if has_skeleton {
            group_to_skin[group_idx]
        } else {
            None
        };
        let (mesh_offset, correction): ([f32; 3], [f32; 16]) = match group_skin_idx {
            Some(si) => (skins_data[si].mesh_offset, skins_data[si].correction),
            None => (
                [0.0; 3],
                [
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
            ),
        };

        // Transform a vertex into this group's skin BONE space by applying the
        // OBJ→bone correction (axis-aligned scale + offset) directly to the
        // POSITION. This puts the mesh vertices in the SAME space as the bones
        // (bones anatomically inside the mesh) so animated bone rotations pivot
        // about points that are actually near their vertices — without this the
        // correction lived on the SkinRoot and bones sat ~30u behind the mesh,
        // tearing it apart on rotation. The SkinRoot below drops the correction
        // (scale=1, translation = mesh_offset + ground_shift) so the rendered
        // rest position is unchanged.
        let transform_vertex = |p: &[f32; 3]| -> [f32; 3] {
            if group_skin_idx.is_some() {
                [
                    correction[0] * (p[0] - mesh_offset[0]) + correction[12],
                    correction[5] * (p[1] - mesh_offset[1]) + correction[13],
                    correction[10] * (p[2] - mesh_offset[2]) + correction[14],
                ]
            } else {
                *p
            }
        };

        // Compute AABB for transformed positions
        let mut pos_min = [f32::MAX; 3];
        let mut pos_max = [f32::MIN; 3];
        for p in &group.positions {
            let tp = transform_vertex(p);
            for i in 0..3 {
                pos_min[i] = pos_min[i].min(tp[i]);
                pos_max[i] = pos_max[i].max(tp[i]);
            }
        }

        // Positions
        buf.align(4);
        let (pos_acc_bv, pos_acc) = add_accessor(
            &mut buf,
            &mut buffer_views,
            &mut accessors,
            5126, // FLOAT
            group_vertex_count,
            "VEC3",
            Some(json!([pos_min[0], pos_min[1], pos_min[2]])),
            Some(json!([pos_max[0], pos_max[1], pos_max[2]])),
            Some(34962), // ARRAY_BUFFER
        );
        let start = buf.len();
        for p in &group.positions {
            let tp = transform_vertex(p);
            buf.write_f32(tp[0]);
            buf.write_f32(tp[1]);
            buf.write_f32(tp[2]);
        }
        buffer_views[pos_acc_bv]["byteLength"] = json!(buf.len() - start);

        // Normals
        let normal_acc = if !group.normals.is_empty() {
            buf.align(4);
            let (acc_bv, acc) = add_accessor(
                &mut buf,
                &mut buffer_views,
                &mut accessors,
                5126,
                group.normals.len(),
                "VEC3",
                None,
                None,
                Some(34962),
            );
            // Normals transform by the inverse-transpose of the linear part of
            // the correction (a diagonal scale): n' = normalize(n / scale).
            let nsx = if group_skin_idx.is_some() {
                correction[0]
            } else {
                1.0
            };
            let nsy = if group_skin_idx.is_some() {
                correction[5]
            } else {
                1.0
            };
            let nsz = if group_skin_idx.is_some() {
                correction[10]
            } else {
                1.0
            };
            let start = buf.len();
            for n in &group.normals {
                let mut x = n[0] / nsx;
                let mut y = n[1] / nsy;
                let mut z = n[2] / nsz;
                let len = (x * x + y * y + z * z).sqrt();
                if len > 1e-8 {
                    x /= len;
                    y /= len;
                    z /= len;
                }
                buf.write_f32(x);
                buf.write_f32(y);
                buf.write_f32(z);
            }
            buffer_views[acc_bv]["byteLength"] = json!(buf.len() - start);
            Some(acc)
        } else {
            None
        };

        // TexCoords
        let texcoord_acc = if !group.texcoords.is_empty() {
            buf.align(4);
            let (acc_bv, acc) = add_accessor(
                &mut buf,
                &mut buffer_views,
                &mut accessors,
                5126,
                group.texcoords.len(),
                "VEC2",
                None,
                None,
                Some(34962),
            );
            let start = buf.len();
            for tc in &group.texcoords {
                buf.write_f32(tc[0]);
                buf.write_f32(tc[1]);
            }
            buffer_views[acc_bv]["byteLength"] = json!(buf.len() - start);
            Some(acc)
        } else {
            None
        };

        // Indices
        buf.align(4);
        let max_index = group.indices.iter().copied().max().unwrap_or(0);
        let index_acc = if max_index <= 65535 {
            let (acc_bv, acc) = add_accessor(
                &mut buf,
                &mut buffer_views,
                &mut accessors,
                5123, // UNSIGNED_SHORT
                group.indices.len(),
                "SCALAR",
                None,
                None,
                Some(34963), // ELEMENT_ARRAY_BUFFER
            );
            let start = buf.len();
            for &idx in &group.indices {
                buf.write_u16(idx as u16);
            }
            buffer_views[acc_bv]["byteLength"] = json!(buf.len() - start);
            acc
        } else {
            let (acc_bv, acc) = add_accessor(
                &mut buf,
                &mut buffer_views,
                &mut accessors,
                5125, // UNSIGNED_INT
                group.indices.len(),
                "SCALAR",
                None,
                None,
                Some(34963),
            );
            let start = buf.len();
            for &idx in &group.indices {
                buf.write_u32(idx);
            }
            buffer_views[acc_bv]["byteLength"] = json!(buf.len() - start);
            acc
        };

        // Skinning data: compute per-vertex weights against this group's skin.
        let (joints_acc, weights_acc) = if let Some(skin_idx) = group_skin_idx {
            let skin = &skins_data[skin_idx];
            let bones = skin.bones;
            let bw_pos = &skin.bone_world_pos;
            let correction = skin.correction;

            // Build list of real (non-noanimate) bone indices for this skin
            let real_bones: Vec<usize> = (0..bones.len())
                .filter(|&bi| !bones[bi].name.starts_with("noanimate"))
                .collect();

            // Diagnostic: compare OBJ vertex bbox with skeleton bbox
            if group.positions.len() > 20 {
                let mut obj_min = [f32::MAX; 3];
                let mut obj_max = [f32::MIN; 3];
                for p in &group.positions {
                    for i in 0..3 {
                        obj_min[i] = obj_min[i].min(p[i]);
                        obj_max[i] = obj_max[i].max(p[i]);
                    }
                }
                let mut skel_min = [f32::MAX; 3];
                let mut skel_max = [f32::MIN; 3];
                for bi in &real_bones {
                    let bp = &bw_pos[*bi];
                    for i in 0..3 {
                        skel_min[i] = skel_min[i].min(bp.start[i]).min(bp.end[i]);
                        skel_max[i] = skel_max[i].max(bp.start[i]).max(bp.end[i]);
                    }
                }
                debug!(
                    "  [skin \"{}\"] OBJ vertex bbox: ({:.2},{:.2},{:.2}) to ({:.2},{:.2},{:.2})",
                    skin.mesh_name,
                    obj_min[0],
                    obj_min[1],
                    obj_min[2],
                    obj_max[0],
                    obj_max[1],
                    obj_max[2]
                );
                debug!(
                    "  OBJ-mesh_offset: ({:.2},{:.2},{:.2}) to ({:.2},{:.2},{:.2})",
                    obj_min[0] - mesh_offset[0],
                    obj_min[1] - mesh_offset[1],
                    obj_min[2] - mesh_offset[2],
                    obj_max[0] - mesh_offset[0],
                    obj_max[1] - mesh_offset[1],
                    obj_max[2] - mesh_offset[2]
                );
                debug!(
                    "  Skeleton bbox:   ({:.2},{:.2},{:.2}) to ({:.2},{:.2},{:.2})",
                    skel_min[0], skel_min[1], skel_min[2], skel_max[0], skel_max[1], skel_max[2]
                );
            }

            // ── Skin weights (resolved per skin, see above) ───────────────
            // Reduce each vertex's normalized (bone, weight) list to the glTF
            // top-4 and renormalize. The list is the XMED's own authored skin
            // weights when the geometry route decoded them, otherwise the
            // IFXSkin regenerate → joint-cross-section → smooth → remove-rogue
            // pipeline (`skin_weights::generate_weights`).
            let mut joint_data: Vec<u8> = Vec::with_capacity(group_vertex_count * 4);
            let mut weight_data: Vec<f32> = Vec::with_capacity(group_vertex_count * 4);
            let mut bone_counts = vec![0usize; bones.len()];

            let gw = &group_weights[group_idx];
            for vi in 0..group_vertex_count {
                let mut iw: Vec<(usize, f32)> = gw.get(vi).cloned().unwrap_or_default();
                iw.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                iw.truncate(4);
                let s: f32 = iw.iter().map(|(_, w)| *w).sum();
                let mut js = [0u8; 4];
                let mut ws = [0f32; 4];
                if s > 0.0 {
                    for (k, &(bi, w)) in iw.iter().enumerate() {
                        js[k] = bi as u8;
                        ws[k] = w / s;
                    }
                    bone_counts[iw[0].0] += 1;
                } else {
                    // Fallback: nearest bone segment (rare — orphaned vertex).
                    let p = &group.positions[vi];
                    let raw = [
                        correction[0] * (p[0] - mesh_offset[0]) + correction[12],
                        correction[5] * (p[1] - mesh_offset[1]) + correction[13],
                        correction[10] * (p[2] - mesh_offset[2]) + correction[14],
                    ];
                    let mut best = (real_bones[0], f32::MAX);
                    for &bi in &real_bones {
                        let d = crate::bone_weights::point_to_segment_dist_pub(
                            &raw,
                            &bw_pos[bi].start,
                            &bw_pos[bi].end,
                        );
                        if d < best.1 {
                            best = (bi, d);
                        }
                    }
                    js[0] = best.0 as u8;
                    ws[0] = 1.0;
                    bone_counts[best.0] += 1;
                }
                joint_data.extend_from_slice(&js);
                weight_data.extend_from_slice(&ws);
            }

            if group.positions.len() > 20 {
                debug!(
                    "  Bone assignments (IFXSkin) for group \"{}\" → skin \"{}\" ({} verts):",
                    group.name,
                    skin.mesh_name,
                    group.positions.len()
                );
                for (bi, &count) in bone_counts.iter().enumerate() {
                    if count > 0 {
                        debug!("    bone[{:2}] {:25} : {} verts", bi, bones[bi].name, count);
                    }
                }
            }

            // JOINTS_0
            buf.align(4);
            let (joints_acc_bv, joints_acc) = add_accessor(
                &mut buf,
                &mut buffer_views,
                &mut accessors,
                5121,
                group_vertex_count,
                "VEC4",
                None,
                None,
                Some(34962),
            );
            let start = buf.len();
            buf.write_bytes(&joint_data);
            buffer_views[joints_acc_bv]["byteLength"] = json!(buf.len() - start);

            // WEIGHTS_0
            buf.align(4);
            let (weights_acc_bv, weights_acc) = add_accessor(
                &mut buf,
                &mut buffer_views,
                &mut accessors,
                5126,
                group_vertex_count,
                "VEC4",
                None,
                None,
                Some(34962),
            );
            let start = buf.len();
            for &w in &weight_data {
                buf.write_f32(w);
            }
            buffer_views[weights_acc_bv]["byteLength"] = json!(buf.len() - start);

            (Some(joints_acc), Some(weights_acc))
        } else {
            (None, None)
        };

        primitives_info.push(PrimitiveInfo {
            pos_acc,
            normal_acc,
            texcoord_acc,
            index_acc,
            joints_acc,
            weights_acc,
            material_name: group.material.clone(),
            skin_idx: group_skin_idx,
            static_idx: if group_skin_idx.is_none() {
                statics.iter().position(|s| s.name == group.name)
            } else {
                None
            },
        });
    }

    // ── Inverse bind matrices (one accessor per skin) ──────────────

    let mut ibm_accs: Vec<usize> = Vec::with_capacity(skins_data.len());
    for skin in &skins_data {
        buf.align(4);
        let (acc_bv, acc) = add_accessor(
            &mut buf,
            &mut buffer_views,
            &mut accessors,
            5126,
            skin.bones.len(),
            "MAT4",
            None,
            None,
            None,
        );
        let start = buf.len();

        // IBM = inverse(boneWorld).
        // The correction (and per-skin mesh_offset) is applied via the SkinRoot
        // parent node above each skin's joint roots. Baking it into the IBM as
        // well would double-apply it.
        let world_mats = compute_bone_world_matrices(skin.bones, bone_offset, root_scale);
        for m in &world_mats {
            let ibm = invert_mat4(m);
            buf.write_mat4(&ibm);
        }
        buffer_views[acc_bv]["byteLength"] = json!(buf.len() - start);
        ibm_accs.push(acc);
    }

    // ── Materials and texture embedding ───────────────────
    // One material per distinct primitive material name, in primitive order.
    // Each takes its baseColorTexture from the XMED's own 0x36 shader binding
    // when there is one, and from the converter's MTL PNG otherwise. Every
    // material is double-sided: Shockwave 3D draws both faces, and the models
    // depend on it (the tail, mane and blanket are single quads).

    let mut gltf_samplers: Vec<Value> = Vec::new();
    let mut gltf_images: Vec<Value> = Vec::new();
    let mut gltf_textures: Vec<Value> = Vec::new();
    let mut gltf_materials: Vec<Value> = Vec::new();

    // Map: material_name -> glTF material index
    let mut material_index_map: HashMap<String, usize> = HashMap::new();
    // Image source key ("xmed:<texture>" or "mtl:<file>") -> glTF texture index,
    // so a texture shared by several materials is embedded once.
    let mut texture_cache: HashMap<String, usize> = HashMap::new();
    // The XMED's own 0x10 colour blocks, keyed the way `material_key` keys the
    // shader slots so an OBJ `usemtl` (`grey2`) still finds `grey_2`.
    let xmed_colours: HashMap<String, &Material> = xmed
        .materials
        .iter()
        .map(|m| (material_key(&m.name), m))
        .collect();
    for prim in &primitives_info {
        let mat_name = &prim.material_name;
        if mat_name.is_empty() || material_index_map.contains_key(mat_name) {
            continue;
        }

        let resolved: Option<(String, &str, &[u8], &str, bool)> =
            embedded_textures.get(&material_key(mat_name)).map(|tex| {
                (
                    format!("xmed:{}", tex.name),
                    tex.mime,
                    tex.bytes.as_slice(),
                    tex.name.as_str(),
                    tex.masked,
                )
            });

        let mut material = json!({
            "name": mat_name,
            "doubleSided": true,
            "pbrMetallicRoughness": {
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
        });

        // The 0x10 colour block. An untextured shader is drawn in its diffuse
        // colour — without it the moped, the Rydde barn and the Hey yard come
        // out default white. A textured shader keeps a white base: the SW3D
        // converter writes no `Kd` for those either (`christian_bike.MTL`'s
        // `Material #8dfg` has `map_Kd` and no diffuse), i.e. the texture
        // replaces the diffuse term rather than modulating it. `opacity` is a
        // shader property in both cases and always applies.
        if let Some(colour) = xmed_colours.get(&material_key(mat_name)) {
            let alpha = colour.opacity.clamp(0.0, 1.0);
            if resolved.is_none() {
                material["pbrMetallicRoughness"]["baseColorFactor"] = json!([
                    colour.diffuse[0],
                    colour.diffuse[1],
                    colour.diffuse[2],
                    alpha,
                ]);
            }
            if alpha < 1.0 {
                material["alphaMode"] = json!("BLEND");
            }
        }

        if let Some((cache_key, mime, bytes, tex_name, masked)) = resolved {
            let tex_idx = match texture_cache.get(&cache_key) {
                Some(&idx) => idx,
                None => {
                    // One shared sampler (linear filtering, repeat wrap)
                    if gltf_samplers.is_empty() {
                        gltf_samplers.push(json!({
                            "magFilter": 9729, // LINEAR
                            "minFilter": 9987, // LINEAR_MIPMAP_LINEAR
                            "wrapS": 10497,    // REPEAT
                            "wrapT": 10497,    // REPEAT
                        }));
                    }

                    // Image bytes go into the binary buffer as a bufferView
                    buf.align(4);
                    let bv_idx = buffer_views.len();
                    buffer_views.push(json!({
                        "buffer": 0,
                        "byteOffset": buf.len(),
                        "byteLength": bytes.len(),
                    }));
                    buf.write_bytes(bytes);

                    let img_idx = gltf_images.len();
                    gltf_images.push(json!({
                        "name": tex_name,
                        "mimeType": mime,
                        "bufferView": bv_idx,
                    }));

                    let idx = gltf_textures.len();
                    gltf_textures.push(json!({
                        "name": tex_name,
                        "source": img_idx,
                        "sampler": 0,
                    }));
                    texture_cache.insert(cache_key, idx);
                    idx
                }
            };
            material["pbrMetallicRoughness"]["baseColorTexture"] = json!({ "index": tex_idx });
            if masked {
                // 4-channel Director texture (0x20 channels=4: JPEG colour + zlib
                // alpha plane). Shockwave alpha-blends such textures. Static
                // props (`fontene`'s water) get BLEND; skinned rigs keep the
                // cut-out MASK (lynet's `hale`/dirt overlays).
                if prim.skin_idx.is_none() {
                    material["alphaMode"] = json!("BLEND");
                } else {
                    material["alphaMode"] = json!("MASK");
                    material["alphaCutoff"] = json!(0.5);
                }
            }
        }

        material_index_map.insert(mat_name.clone(), gltf_materials.len());
        gltf_materials.push(material);
    }

    // ── Animation data ────────────────────────────────────

    let mut gltf_animations: Vec<Value> = Vec::new();

    if has_skeleton && !decoded_motions.is_empty() {
        // Route each motion to the skin with the highest bone-name overlap.
        // Motions whose tracks reference no known bones in ANY skin are dropped.
        for (mi, motion) in decoded_motions.iter().enumerate() {
            if keyframe_targets.iter().any(|&(k, _)| k == mi) {
                continue;
            }
            // Count bone-name overlaps per skin
            let mut best_skin: Option<usize> = None;
            let mut best_overlap: usize = 0;
            for (si, skin) in skins_data.iter().enumerate() {
                let overlap = motion
                    .tracks
                    .iter()
                    .filter(|t| skin.bone_name_to_idx.contains_key(&t.name))
                    .count();
                if overlap > best_overlap {
                    best_overlap = overlap;
                    best_skin = Some(si);
                }
            }
            let Some(skin_idx) = best_skin else {
                warn!(
                    "  dropping animation \"{}\" — no track bones match any skin",
                    motion.name
                );
                continue;
            };
            let skin = &skins_data[skin_idx];
            let bones = skin.bones;
            let joint_node_offset = skin.joint_node_offset;
            info!(
                "  Animation \"{}\" → skin[{}] \"{}\" ({} of {} tracks match)",
                motion.name,
                skin_idx,
                skin.mesh_name,
                best_overlap,
                motion.tracks.len(),
            );

            // Director keyframe times are absolute in the motion's own
            // timeline and `bonesPlayer.play` starts a motion at its first
            // keyframe. A Three.js AnimationClip instead spans [0, max t], so
            // a motion whose first key sits at t > 0 holds that first pose for
            // the lead-in on every repeat. Every locomotion loop on the disc
            // starts at 0.2333 s (7 frames at 30 fps), which showed up as
            // Monica freezing for a quarter second after each two steps.
            // Shift the whole clip so its earliest key is t = 0.
            let time_origin = motion
                .tracks
                .iter()
                .filter_map(|t| t.keyframes.first().map(|k| k.time))
                .fold(f32::INFINITY, f32::min);
            let time_origin = if time_origin.is_finite() {
                time_origin
            } else {
                0.0
            };

            let mut samplers: Vec<Value> = Vec::new();
            let mut channels: Vec<Value> = Vec::new();
            let mut dropped_tracks = 0usize;

            for track in motion.tracks.iter() {
                if track.keyframes.is_empty() {
                    continue;
                }

                // Match track to joint by name within this skin.
                // Tracks for bones that don't exist in this skin (e.g. monica's
                // ponytail tracks against the christian skeleton) are dropped.
                let Some(&joint_idx) = skin.bone_name_to_idx.get(&track.name) else {
                    dropped_tracks += 1;
                    continue;
                };

                let joint_node_idx = joint_node_offset + joint_idx;

                // ── Retarget to this skin's bind pose ──────────────────────
                // The animation may have been authored against a slightly
                // different rest skeleton (e.g. monica_climbon's `center`
                // track was authored in horse-skeleton coordinates, ~22× the
                // rider's actual bind displacement). Compute the per-bone
                // offset that takes kf0 to the target skin's bind-pose
                // values, then apply that offset to every keyframe so kf0
                // matches the joint node's rest TRS exactly.
                let target_bone = &bones[joint_idx];
                let kf0 = &track.keyframes[0];
                // Retarget the ROOT bone only. The original concern this solves
                // is that a clip's root (`center`) track may be authored in a
                // different rest skeleton's coordinates (e.g. monica_climbon's
                // center was authored ~22x off in horse-skeleton space). Aligning
                // kf0 to the destination bind positions the actor correctly.
                //
                // Applying it to LIMB bones is harmful: it pre-multiplies every
                // limb rotation so kf0 equals the bind orientation. When the bake
                // rest pose differs from the clip's authoring rest (e.g.
                // together_trav2's monica binds with arms out, but the clips pose
                // her holding the reins), this overwrites the authored limb pose
                // with the bind pose — a near-static `stand` then leaves the arms
                // stuck at the arms-out bind (the "T-pose" the rider showed).
                // Limb bones carry absolute authored rotations already, so leaving
                // them untouched reproduces the intended pose.
                let is_root = bones[joint_idx].parent_index < 0;
                let t_offset = if is_root {
                    [
                        target_bone.displacement[0] - kf0.displacement[0],
                        target_bone.displacement[1] - kf0.displacement[1],
                        target_bone.displacement[2] - kf0.displacement[2],
                    ]
                } else {
                    [0.0; 3]
                };
                // q_offset such that q_offset * kf0.rotation == target_bone.orientation.
                // Both kf0.rotation and target_bone.orientation are wxyz unit quats;
                // the inverse of a unit quat is its conjugate. Identity for limbs.
                let q_offset = if is_root {
                    let kf0_inv = quat_conjugate_wxyz(kf0.rotation);
                    quat_mul_wxyz(target_bone.orientation, kf0_inv)
                } else {
                    [1.0, 0.0, 0.0, 0.0]
                };

                if t_offset.iter().any(|v| v.abs() > 0.01) {
                    debug!(
                        "  Retarget motion \"{}\" track \"{}\" bone \"{}\": t_offset=({:.3},{:.3},{:.3})",
                        motion.name,
                        track.name,
                        target_bone.name,
                        t_offset[0],
                        t_offset[1],
                        t_offset[2]
                    );
                }

                // Input: time values, rebased on the motion's first keyframe.
                let times: Vec<f32> = track
                    .keyframes
                    .iter()
                    .map(|k| k.time - time_origin)
                    .collect();
                let time_min = times.first().copied().unwrap_or(0.0);
                let time_max = times.last().copied().unwrap_or(0.0);

                buf.align(4);
                let (time_acc_bv, time_acc) = add_accessor(
                    &mut buf,
                    &mut buffer_views,
                    &mut accessors,
                    5126,
                    track.keyframes.len(),
                    "SCALAR",
                    Some(json!([time_min])),
                    Some(json!([time_max])),
                    None,
                );
                let start = buf.len();
                for &t in &times {
                    buf.write_f32(t);
                }
                buffer_views[time_acc_bv]["byteLength"] = json!(buf.len() - start);

                // Translation output. Apply the retarget t_offset to the raw
                // bind-frame displacement BEFORE adding parent_length + extra
                // (the joint-node TRS shift), so kf0 + t_offset == target_bone.displacement
                // — matching the joint node's rest translation exactly.
                buf.align(4);
                let (trans_acc_bv, trans_acc) = add_accessor(
                    &mut buf,
                    &mut buffer_views,
                    &mut accessors,
                    5126,
                    track.keyframes.len(),
                    "VEC3",
                    None,
                    None,
                    None,
                );
                let parent_length = if bones[joint_idx].parent_index >= 0 {
                    bones[bones[joint_idx].parent_index as usize].rest_length
                } else {
                    0.0
                };
                let extra = if bones[joint_idx].parent_index < 0 {
                    bone_offset
                } else {
                    [0.0; 3]
                };
                let start = buf.len();
                for kf in &track.keyframes {
                    let dx = kf.displacement[0] + t_offset[0];
                    let dy = kf.displacement[1] + t_offset[1];
                    let dz = kf.displacement[2] + t_offset[2];
                    buf.write_f32(dx + parent_length + extra[0]);
                    buf.write_f32(dy + extra[1]);
                    buf.write_f32(dz + extra[2]);
                }
                buffer_views[trans_acc_bv]["byteLength"] = json!(buf.len() - start);

                // Rotation output (wxyz -> xyzw, NO axis remapping — Z-up space).
                // Pre-multiply by q_offset in wxyz, then the wxyz->xyzw remap below
                // produces a quat that equals target_bone.orientation at kf0.
                buf.align(4);
                let (rot_acc_bv, rot_acc) = add_accessor(
                    &mut buf,
                    &mut buffer_views,
                    &mut accessors,
                    5126,
                    track.keyframes.len(),
                    "VEC4",
                    None,
                    None,
                    None,
                );
                let start = buf.len();
                // Decoded rotations are absolute (wxyz). After retarget pre-mul,
                // write as glTF xyzw. Hemisphere canonicalization is done at
                // decode time in motion.rs.
                for kf in &track.keyframes {
                    let retargeted = quat_mul_wxyz(q_offset, kf.rotation);
                    let normalized = quat_normalize_wxyz(retargeted);
                    let [w, x, y, z] = normalized;
                    buf.write_f32(x);
                    buf.write_f32(y);
                    buf.write_f32(z);
                    buf.write_f32(w);
                }
                buffer_views[rot_acc_bv]["byteLength"] = json!(buf.len() - start);

                // Translation channel
                let trans_sampler_idx = samplers.len();
                samplers.push(json!({
                    "input": time_acc,
                    "output": trans_acc,
                    "interpolation": "LINEAR",
                }));
                channels.push(json!({
                    "sampler": trans_sampler_idx,
                    "target": {
                        "node": joint_node_idx,
                        "path": "translation",
                    },
                }));

                // Rotation channel
                let rot_sampler_idx = samplers.len();
                samplers.push(json!({
                    "input": time_acc,
                    "output": rot_acc,
                    "interpolation": "LINEAR",
                }));
                channels.push(json!({
                    "sampler": rot_sampler_idx,
                    "target": {
                        "node": joint_node_idx,
                        "path": "rotation",
                    },
                }));
            }

            if dropped_tracks > 0 {
                debug!(
                    "    dropped {} tracks (bones not in target skin)",
                    dropped_tracks
                );
            }
            if !channels.is_empty() {
                gltf_animations.push(json!({
                    "name": motion.name,
                    "samplers": samplers,
                    "channels": channels,
                }));
            }
        }
    }

    // ── Keyframe motions → channels on the static model node ──────────────
    // Raw track values, wxyz → xyzw, Z-up like the bone channels (SceneRoot
    // rotates the whole scene to Y-up). kf0 of every keyframe motion in the
    // corpus is the identity: Director plays them relative to the model's
    // transform at play start, which is why the model node rests at identity
    // under its `:start` parent.
    for &(mi, si) in &keyframe_targets {
        let motion = &decoded_motions[mi];
        let track = &motion.tracks[0];
        if track.keyframes.is_empty() {
            continue;
        }
        let node_idx = static_node_idx[si];
        // Rebased on the first keyframe, like the bone clips above.
        let time_origin = track.keyframes[0].time;
        let times: Vec<f32> = track
            .keyframes
            .iter()
            .map(|k| k.time - time_origin)
            .collect();
        buf.align(4);
        let (time_bv, time_acc) = add_accessor(
            &mut buf,
            &mut buffer_views,
            &mut accessors,
            5126,
            times.len(),
            "SCALAR",
            Some(json!([times.first().copied().unwrap_or(0.0)])),
            Some(json!([times.last().copied().unwrap_or(0.0)])),
            None,
        );
        let start = buf.len();
        for &t in &times {
            buf.write_f32(t);
        }
        buffer_views[time_bv]["byteLength"] = json!(buf.len() - start);

        let mut samplers: Vec<Value> = Vec::new();
        let mut channels: Vec<Value> = Vec::new();
        let mut emit = |buf: &mut GltfBuffer,
                        bv_list: &mut Vec<Value>,
                        acc_list: &mut Vec<Value>,
                        path: &str,
                        n: usize,
                        values: &[f32]| {
            buf.align(4);
            let (bv, acc) = add_accessor(
                buf,
                bv_list,
                acc_list,
                5126,
                times.len(),
                if n == 4 { "VEC4" } else { "VEC3" },
                None,
                None,
                None,
            );
            let start = buf.len();
            for v in values {
                buf.write_f32(*v);
            }
            bv_list[bv]["byteLength"] = json!(buf.len() - start);
            samplers.push(json!({ "input": time_acc, "output": acc, "interpolation": "LINEAR" }));
            channels.push(json!({
                "sampler": samplers.len() - 1,
                "target": { "node": node_idx, "path": path },
            }));
        };
        let translation: Vec<f32> = track
            .keyframes
            .iter()
            .flat_map(|k| k.displacement)
            .collect();
        let rotation: Vec<f32> = track
            .keyframes
            .iter()
            .flat_map(|k| {
                let [w, x, y, z] = quat_normalize_wxyz(k.rotation);
                [x, y, z, w]
            })
            .collect();
        let scale: Vec<f32> = track.keyframes.iter().flat_map(|k| k.scale).collect();
        emit(
            &mut buf,
            &mut buffer_views,
            &mut accessors,
            "translation",
            3,
            &translation,
        );
        emit(
            &mut buf,
            &mut buffer_views,
            &mut accessors,
            "rotation",
            4,
            &rotation,
        );
        if scale.iter().any(|s| (s - 1.0).abs() > 1e-5) {
            emit(
                &mut buf,
                &mut buffer_views,
                &mut accessors,
                "scale",
                3,
                &scale,
            );
        }
        info!(
            "  Keyframe clip \"{}\" → node #{} \"{}\": {} keyframes, {:.3}s",
            motion.name,
            node_idx,
            statics[si].name,
            times.len(),
            times.last().copied().unwrap_or(0.0)
        );
        gltf_animations.push(json!({
            "name": motion.name,
            "samplers": samplers,
            "channels": channels,
        }));
    }
    // ── Build glTF JSON ───────────────────────────────────

    // Node layout:
    //   0: SceneRoot (contains mesh nodes + skeleton roots as children)
    //   1..1+num_skins: Mesh nodes (one per skin), or single mesh node when no skeleton
    //   1+num_skins..: Joint nodes (skin 0's joints, then skin 1's, ...)

    let mut nodes: Vec<Value> = Vec::new();

    // Native route: a skin's mesh node may hang under an authored transform node
    // (`bakgrunn.xmed`: `lynet`/`monica` -> `together` -> World). Director
    // addresses that node as `scene.model("together")`, so it is emitted as a
    // group node under SceneRoot carrying its authored world TRS, and the
    // skins' SkinRoots become its children (static-node contract). Groups are
    // appended after every other node; SceneRoot's children are patched then.
    struct ParentGroup {
        name: String,
        matrix: [f32; 16],
        skins: Vec<usize>,
    }
    let mut parent_groups: Vec<ParentGroup> = Vec::new();
    let mut skin_parent_group: Vec<Option<usize>> = vec![None; num_skins];
    // `flat_skins` (config `"skins": "flat"`) opts a mount rig out: its
    // authored `together`/`c_together` group is what `horse.ls` re-parents
    // away at run time, and the scene re-creates it from `RigNodes::group`
    // (`src/systems/RiderSeat.ts`), so the SkinRoots stay SceneRoot siblings.
    if has_skeleton && !flat_skins {
        for (si, skin) in skins_data.iter().enumerate() {
            let parent_name = xmed
                .mesh_nodes
                .iter()
                .find(|mn| mn.mesh_ref == skin.mesh_name || mn.name == skin.mesh_name)
                .map(|mn| mn.parent.as_str())
                .unwrap_or("World");
            if parent_name == "World" {
                continue;
            }
            // World matrix of the transform node: walk the parent chain.
            let mut matrix = identity_mat4;
            let mut cursor = parent_name;
            let mut found = false;
            while let Some(node) = xmed.bones.iter().find(|b| b.name == cursor) {
                found = true;
                matrix = mat4_mul(&node.matrix, &matrix);
                if node.parent == "World" {
                    break;
                }
                cursor = node.parent.as_str();
            }
            if !found {
                warn!(
                    "  skin \"{}\" parent \"{}\" is not a transform node; SkinRoot stays under SceneRoot",
                    skin.mesh_name, parent_name
                );
                continue;
            }
            let gi = match parent_groups.iter().position(|g| g.name == parent_name) {
                Some(gi) => gi,
                None => {
                    parent_groups.push(ParentGroup {
                        name: parent_name.to_string(),
                        matrix,
                        skins: Vec::new(),
                    });
                    parent_groups.len() - 1
                }
            };
            parent_groups[gi].skins.push(si);
            skin_parent_group[si] = Some(gi);
        }
    }

    // Node 0: Scene root with Z-up → Y-up rotation.
    // SceneRoot children: the skin mesh nodes (or the OBJ `Mesh`), every native
    // static model node (its `:start` parent when animated), the SkinRoots not
    // hanging under an authored parent group, and (patched in below) the groups.
    let mut root_children: Vec<Value> = Vec::new();
    for i in 0..num_skin_mesh_nodes {
        root_children.push(json!(1 + i));
    }
    let static_parent_of = |si: usize| -> Option<usize> {
        statics
            .iter()
            .position(|p| p.name == statics[si].parent && p.name != statics[si].name)
    };
    for si in 0..statics.len() {
        if static_parent_of(si).is_none() {
            root_children.push(json!(static_start_idx[si].unwrap_or(static_node_idx[si])));
        }
    }
    if has_skeleton {
        // Every skin's SkinRoot is a child of SceneRoot (siblings). This matches
        // how the original game loads the riding scene: both `monica` and `lynet`
        // are sibling models from one w3d scene (horse.ls), positioned in a common
        // space, NOT parented to each other. The rider is placed in that common
        // space by composing the horse's correction onto her own (see below), so
        // she follows the horse purely via her authored `monica_*` animations.
        for i in 0..num_skins {
            if skin_parent_group[i].is_none() {
                root_children.push(json!(first_skin_root_node + i));
            }
        }
    }
    nodes.push(json!({
        "name": "SceneRoot",
        "children": root_children,
        "rotation": [-0.7071068, 0.0, 0.0, 0.7071068],
    }));

    // Group primitives per glTF mesh: one per skin (or the single OBJ mesh), then
    // one per native static model. An unrigged primitive with no static (an OBJ
    // group that failed routing) falls back to mesh 0.
    let mut prims_per_mesh: Vec<Vec<&PrimitiveInfo>> = vec![Vec::new(); num_gltf_meshes];
    for prim in &primitives_info {
        let mesh_idx = match (prim.skin_idx, prim.static_idx) {
            (Some(si), _) => si,
            (None, Some(st)) => static_mesh_of(st),
            (None, None) => 0,
        };
        if mesh_idx < num_gltf_meshes {
            prims_per_mesh[mesh_idx].push(prim);
        }
    }

    // glTF mesh names: per skin, per static model, or plain `mesh` for an OBJ bake.
    let mesh_label = |mesh_idx: usize| -> String {
        if mesh_idx >= num_skin_mesh_nodes {
            format!(":{}", statics[mesh_idx - num_skin_mesh_nodes].name)
        } else if has_skeleton {
            format!(":{}", skins_data[mesh_idx].mesh_name)
        } else {
            String::new()
        }
    };

    // Skin mesh nodes (or the OBJ `Mesh`). No TRS here: per the glTF spec, a
    // node's TRS is ignored for the SkinnedMesh it carries. All per-skin
    // placement lives on the SkinRoot below.
    for mesh_idx in 0..num_skin_mesh_nodes {
        let mut mesh_node = json!({
            "name": format!("Mesh{}", mesh_label(mesh_idx)),
            "mesh": mesh_idx,
        });
        if has_skeleton {
            mesh_node["skin"] = json!(mesh_idx);
        }
        nodes.push(mesh_node);
    }

    // Native static model nodes: `<name>` carries the mesh and the authored 0x72
    // TRS; when a keyframe motion drives it, that TRS moves to a `<name>:start`
    // parent and `<name>` rests at identity (see the keyframe block above).
    for (si, st) in statics.iter().enumerate() {
        let (t, q, s) = decompose_trs(&st.matrix);
        let mut trs = json!({});
        if t != [0.0; 3] {
            trs["translation"] = json!(t);
        }
        if q != [0.0, 0.0, 0.0, 1.0] {
            trs["rotation"] = json!(q);
        }
        if s.iter().any(|v| (v - 1.0).abs() > 1e-6) {
            trs["scale"] = json!(s);
        }
        let mut model_node = json!({
            "name": st.name,
            "mesh": static_mesh_of(si),
        });
        // Authored children (mesh nodes whose 0x72 parent is this one).
        let child_nodes: Vec<Value> = statics
            .iter()
            .enumerate()
            .filter(|(ci, c)| *ci != si && c.parent == st.name)
            .map(|(ci, _)| json!(static_start_idx[ci].unwrap_or(static_node_idx[ci])))
            .collect();
        if !child_nodes.is_empty() {
            model_node["children"] = json!(child_nodes);
        }
        if let Some(start_idx) = static_start_idx[si] {
            debug_assert_eq!(nodes.len(), start_idx);
            let mut start = json!({
                "name": format!("{}:start", st.name),
                "children": [static_node_idx[si]],
            });
            for (k, v) in trs.as_object().unwrap() {
                start[k] = v.clone();
            }
            nodes.push(start);
        } else {
            for (k, v) in trs.as_object().unwrap() {
                model_node[k] = v.clone();
            }
        }
        debug_assert_eq!(nodes.len(), static_node_idx[si]);
        debug!(
            "  Static node \"{}\": t=({:.3},{:.3},{:.3}) q=({:.4},{:.4},{:.4},{:.4}) s=({:.3},{:.3},{:.3}){}",
            st.name,
            t[0],
            t[1],
            t[2],
            q[0],
            q[1],
            q[2],
            q[3],
            s[0],
            s[1],
            s[2],
            if static_start_idx[si].is_some() {
                " on :start (keyframe-animated)"
            } else {
                ""
            }
        );
        nodes.push(model_node);
    }

    // Nodes 1+num_skins..1+2*num_skins: SkinRoot nodes — one per skin. Each carries
    // T(mesh_offset + ground_shift) * correction so Three.js positions each skin in
    // scene space through the joint hierarchy (which IS applied, unlike mesh-node TRS).
    // ground_shift is a single per-bake Z translation that lifts the assembly onto
    // the ground plane (Director Z=0); it's the same for every skin so relative
    // positions are preserved.
    //
    if has_skeleton {
        for skin in &skins_data {
            let correction = skin.correction;
            // The OBJ→bone correction is now baked into the vertex POSITIONs
            // (see transform_vertex), so the SkinRoot is pure translation:
            // mesh_offset + ground_shift. This keeps the rendered rest position
            // identical while the bones now sit inside the mesh.
            let _ = correction;
            let c_scale = [1.0, 1.0, 1.0];
            let c_trans = [
                skin.mesh_offset[0],
                skin.mesh_offset[1],
                skin.mesh_offset[2] + ground_shift_z,
            ];
            let joint_root_children: Vec<Value> = skin
                .bones
                .iter()
                .enumerate()
                .filter(|(_, b)| b.parent_index < 0)
                .map(|(i, _)| json!(skin.joint_node_offset + i))
                .collect();
            let mut skin_root = json!({
                "name": format!("SkinRoot:{}", skin.mesh_name),
                "children": joint_root_children,
            });

            if c_scale != [1.0, 1.0, 1.0] {
                skin_root["scale"] = json!([c_scale[0], c_scale[1], c_scale[2]]);
            }
            if c_trans != [0.0, 0.0, 0.0] {
                skin_root["translation"] = json!([c_trans[0], c_trans[1], c_trans[2]]);
            }
            info!(
                "  SkinRoot \"{}\": scale=({:.4},{:.4},{:.4}) trans=({:.4},{:.4},{:.4})  (mesh_offset=({:.2},{:.2},{:.2}), ground_shift_z={:.3})",
                skin.mesh_name,
                c_scale[0],
                c_scale[1],
                c_scale[2],
                c_trans[0],
                c_trans[1],
                c_trans[2],
                skin.mesh_offset[0],
                skin.mesh_offset[1],
                skin.mesh_offset[2],
                ground_shift_z,
            );
            nodes.push(skin_root);
        }
    }

    // Joint nodes: skin 0's joints contiguous, then skin 1's, etc.
    if has_skeleton {
        for skin in skins_data.iter() {
            let bones = skin.bones;
            let offset = skin.joint_node_offset;
            for (i, bone) in bones.iter().enumerate() {
                let is_root = bone.parent_index < 0;
                let parent_length = if !is_root {
                    bones[bone.parent_index as usize].rest_length
                } else {
                    0.0
                };
                let extra_offset = if is_root { bone_offset } else { [0.0_f32; 3] };
                let t = [
                    bone.displacement[0] + parent_length + extra_offset[0],
                    bone.displacement[1] + extra_offset[1],
                    bone.displacement[2] + extra_offset[2],
                ];
                let q = Quat::from_bone_wxyz(bone.orientation);

                let mut joint_node = json!({
                    "name": bone.name,
                    "translation": [t[0], t[1], t[2]],
                    "rotation": [q.x, q.y, q.z, q.w],
                });
                if is_root {
                    joint_node["scale"] = json!([root_scale[0], root_scale[1], root_scale[2]]);
                }

                let children: Vec<Value> = bones
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| b.parent_index == i as i32)
                    .map(|(j, _)| json!(offset + j))
                    .collect();

                if !children.is_empty() {
                    joint_node["children"] = json!(children);
                }

                nodes.push(joint_node);
            }
        }
    }

    // Build meshes (one per skin, or one for unskinned).
    let mut meshes_json: Vec<Value> = Vec::new();
    for mesh_idx in 0..num_gltf_meshes {
        let mut mesh_primitives: Vec<Value> = Vec::new();
        for prim in &prims_per_mesh[mesh_idx] {
            let mut attributes = json!({
                "POSITION": prim.pos_acc,
            });
            if let Some(na) = prim.normal_acc {
                attributes["NORMAL"] = json!(na);
            }
            if let Some(ta) = prim.texcoord_acc {
                attributes["TEXCOORD_0"] = json!(ta);
            }
            if let Some(ja) = prim.joints_acc {
                attributes["JOINTS_0"] = json!(ja);
            }
            if let Some(wa) = prim.weights_acc {
                attributes["WEIGHTS_0"] = json!(wa);
            }

            let mut primitive = json!({
                "attributes": attributes,
                "indices": prim.index_acc,
            });

            if let Some(&mat_idx) = material_index_map.get(&prim.material_name) {
                primitive["material"] = json!(mat_idx);
            }

            mesh_primitives.push(primitive);
        }

        let mesh_name = format!("mesh{}", mesh_label(mesh_idx));
        meshes_json.push(json!({
            "name": mesh_name,
            "primitives": mesh_primitives,
        }));
    }

    // Skins: one per skeleton.
    let skins: Vec<Value> = if has_skeleton {
        skins_data
            .iter()
            .enumerate()
            .map(|(si, skin)| {
                let offset = skin.joint_node_offset;
                let joint_indices: Vec<Value> =
                    (0..skin.bones.len()).map(|i| json!(offset + i)).collect();
                let skeleton_root = skin
                    .bones
                    .iter()
                    .position(|b| b.parent_index < 0)
                    .map(|i| offset + i)
                    .unwrap_or(offset);
                json!({
                    "name": format!("Skeleton:{}", skin.mesh_name),
                    "inverseBindMatrices": ibm_accs[si],
                    "joints": joint_indices,
                    "skeleton": skeleton_root,
                })
            })
            .collect()
    } else {
        vec![]
    };

    // Buffer (placeholder byteLength, we'll fill it in)
    let total_buf_len = buf.len();

    // Authored parent groups (native route): appended last, then hooked into
    // SceneRoot. Their TRS is the transform node's authored world matrix.
    for group in &parent_groups {
        let group_idx = nodes.len();
        let (t, q, s) = decompose_trs(&group.matrix);
        let mut node = json!({
            "name": group.name,
            "children": group
                .skins
                .iter()
                .map(|&si| json!(first_skin_root_node + si))
                .collect::<Vec<Value>>(),
        });
        if t != [0.0; 3] {
            node["translation"] = json!(t);
        }
        if q != [0.0, 0.0, 0.0, 1.0] {
            node["rotation"] = json!(q);
        }
        if s.iter().any(|v| (v - 1.0).abs() > 1e-6) {
            node["scale"] = json!(s);
        }
        nodes.push(node);
        if let Some(children) = nodes[0]["children"].as_array_mut() {
            children.push(json!(group_idx));
        }
        debug!(
            "  Parent group \"{}\": t=({:.3},{:.3},{:.3}) skins={:?}",
            group.name,
            t[0],
            t[1],
            t[2],
            group
                .skins
                .iter()
                .map(|&si| skins_data[si].mesh_name.as_str())
                .collect::<Vec<_>>()
        );
    }

    let mut gltf = json!({
        "asset": {
            "version": "2.0",
            "generator": "xmed-decoder",
        },
        "scene": 0,
        "scenes": [{
            "name": "Scene",
            "nodes": json!([0]),
        }],
        "nodes": nodes,
        "meshes": meshes_json,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{
            "byteLength": total_buf_len,
        }],
    });

    if !skins.is_empty() {
        gltf["skins"] = json!(skins);
    }
    if !gltf_animations.is_empty() {
        gltf["animations"] = json!(gltf_animations);
    }
    if !gltf_samplers.is_empty() {
        gltf["samplers"] = json!(gltf_samplers);
    }
    if !gltf_images.is_empty() {
        gltf["images"] = json!(gltf_images);
    }
    if !gltf_textures.is_empty() {
        gltf["textures"] = json!(gltf_textures);
    }
    if !gltf_materials.is_empty() {
        gltf["materials"] = json!(gltf_materials);
    }

    // ── Write GLB ─────────────────────────────────────────

    let json_str = serde_json::to_string(&gltf).unwrap();
    let json_bytes = json_str.as_bytes();

    // Pad JSON to 4-byte boundary with spaces
    let json_padded_len = (json_bytes.len() + 3) & !3;
    let mut json_padded = json_bytes.to_vec();
    while json_padded.len() < json_padded_len {
        json_padded.push(b' ');
    }

    // Pad binary buffer to 4-byte boundary with zeros
    let bin_padded_len = (buf.len() + 3) & !3;
    while buf.data.len() < bin_padded_len {
        buf.data.push(0);
    }

    // GLB structure:
    // Header: 12 bytes (magic + version + length)
    // JSON chunk: 8 bytes header + padded JSON
    // BIN chunk: 8 bytes header + padded binary
    let total_length = 12 + 8 + json_padded.len() + 8 + buf.data.len();

    let mut glb = Vec::with_capacity(total_length);

    // Header
    glb.extend_from_slice(&0x46546C67u32.to_le_bytes()); // magic "glTF"
    glb.extend_from_slice(&2u32.to_le_bytes()); // version
    glb.extend_from_slice(&(total_length as u32).to_le_bytes());

    // JSON chunk
    glb.extend_from_slice(&(json_padded.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // "JSON"
    glb.extend_from_slice(&json_padded);

    // BIN chunk
    glb.extend_from_slice(&(buf.data.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E4942u32.to_le_bytes()); // "BIN\0"
    glb.extend_from_slice(&buf.data);

    info!("Built {} bytes of GLB", glb.len());
    let total_prims: usize = primitives_info.len();
    info!(
        "  {} nodes, {} accessors, {} buffer views, {} primitives, {} meshes",
        nodes.len(),
        accessors.len(),
        buffer_views.len(),
        total_prims,
        meshes_json.len(),
    );
    info!(
        "  {} textures, {} materials, {} images",
        gltf_textures.len(),
        gltf_materials.len(),
        gltf_images.len(),
    );
    if has_skeleton {
        let total_bones: usize = skins_data.iter().map(|s| s.bones.len()).sum();
        info!(
            "  {} skins, {} total bones, {} animations",
            skins_data.len(),
            total_bones,
            gltf_animations.len()
        );
    }

    Ok(glb)
}

/// Compute world 4x4 matrices for each bone, with per-axis root scale.
/// Uses full matrix multiplication (no TRS decomposition) to correctly handle
/// non-uniform scale + rotation combinations.
fn compute_bone_world_matrices(
    bones: &[BoneDef],
    bone_offset: [f32; 3],
    root_scale: [f32; 3],
) -> Vec<[f32; 16]> {
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let mut world_mats = vec![identity; bones.len()];

    for (i, bone) in bones.iter().enumerate() {
        let is_root = bone.parent_index < 0;
        let parent_length = if !is_root {
            bones[bone.parent_index as usize].rest_length
        } else {
            0.0
        };
        let extra = if is_root { bone_offset } else { [0.0; 3] };
        let t = [
            bone.displacement[0] + parent_length + extra[0],
            bone.displacement[1] + extra[1],
            bone.displacement[2] + extra[2],
        ];
        let q = Quat::from_bone_wxyz(bone.orientation);
        let s = if is_root { root_scale } else { [1.0; 3] };
        let local = trs_to_mat4(t, &q, s);

        if is_root {
            world_mats[i] = local;
        } else {
            let p = bone.parent_index as usize;
            world_mats[i] = mat4_mul(&world_mats[p], &local);
        }
    }
    world_mats
}
