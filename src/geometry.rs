//! Geometry chunk (0x49) decoder: CLOD progressive meshes.
//!
//! XMED 0x49 stores compressed mesh data. Unlike U3D which splits mesh
//! into Declaration / BaseMesh / Progressive blocks (types 0x31 / 0x3B / 0x3C),
//! XMED uses a single type 0x49 chunk per sub-mesh.
//!
//! A mesh is therefore a **group**: the run of consecutive 0x49 chunks sharing
//! one name (`XmedFile::geometry_groups`). Each chunk restarts the arithmetic
//! coder and carries further resolution updates that refine the same mesh, so
//! the group has to be decoded in order into one [`crate::clod_state::LiveMesh`];
//! the mesh's 0x45 declaration supplies the shading-group slots and the 0x47
//! chunk the resolution schedule that says which group each record belongs to.
//!
//! Positions come out in Director units and **Z-up**, texture coordinates with
//! V running bottom-up (the glTF exporter mirrors V).

use crate::bitstream::BitStream;
use crate::chunks::{ClodScheduleChunk, GeometryChunk, MeshDescription, XmedFile};
use crate::clod_state::LiveMesh;
use log::{Level, debug, info, log_enabled, trace, warn};

/// Decode a 0x47 chunk into one resolution-step array per shading group.
///
/// `record_counts[s]` is `MeshDescription::shaders[s].num_texcoords` — the
/// engine's `decoder->0xb4[s]`. Port of `FUN_7a183a20` @`0x7a183bc4-0x7a183c35`:
/// a single ctx-1 arithmetic stream, read straight through, with the running
/// sum reset to zero at each submesh boundary.
pub fn decode_clod_schedule(chunk: &ClodScheduleChunk, record_counts: &[u32]) -> Vec<Vec<u32>> {
    let mut bs = BitStream::new(&chunk.compressed_data);
    record_counts
        .iter()
        .map(|&n| {
            let mut acc = 0u32;
            (0..n)
                .map(|_| {
                    acc = acc.wrapping_add(bs.read_compressed_u32(1));
                    acc
                })
                .collect()
        })
        .collect()
}

// U3D/IFX AC context IDs for position coding (IFXACContext.h)
const CTX_POSITION_DIFF_SIGNS: u32 = 20;
const CTX_POSITION_DIFF_MAG_X: u32 = 21;
const CTX_POSITION_DIFF_MAG_Y: u32 = 22;
const CTX_POSITION_DIFF_MAG_Z: u32 = 23;

// XMED-specific AC contexts recovered from the Shockwave 3D Asset.x32 geometry
// decoder (FUN_7a18a990 = U3D CIFXAuthorCLODDecoder_P). See
// docs/methodology/reference/xmed-decoder-re-progress.md. uACStaticFull = 0x400.
const X_CTX_NEW_POS_COUNT: u32 = 1; // # new positions in this resolution update
const X_CTX_C3: u32 = 3; // second count (faces?)
const X_CTX_C2: u32 = 2; // third count
const X_CTX_PRED_COUNT: u32 = 4; // # predecessor positions for the split
const X_CTX_POS_PRED_TYPE: u32 = 6; // u8 position prediction type (4 = use split)
const X_CTX_SPLIT_IDX: u32 = 5; // split position index
const X_CTX_POS_SIGN: u32 = 7; // u8 position diff sign bits (x=1,y=2,z=4)
const X_CTX_POS_MAG: u32 = 8; // position diff magnitude (X, Y, Z all use ctx 8)
const X_AC_STATIC_FULL: u32 = 0x400;

/// Inverse quantization of a ctx 0x14 skin weight: the codes are 15-bit fixed
/// point (`raw / 32768`). Measured over the corpus — the largest secondary
/// weight any XMED stores is 10922 (`= 32768/3`, a three-way split) and the
/// common ones are 8192 (1/4) and 16384 (1/2); no vertex's decoded weights sum
/// above 1. The 0x45 chunk's `diffuse_inv_quant` (1/512 here) is a different
/// field and gives impossible weights above 20.
const WEIGHT_INV_QUANT: f32 = 1.0 / 32768.0;

/// Experimental decode following the CONFIRMED progressive-CLOD read order from
/// the original decoder. Decodes the first resolution update's header counts and
/// per-position sign+magnitude stream, printing results for comparison against
/// the converter OBJ ground truth (e.g. arrow.xmed → assets/models/obj/arrow).
///
/// This is a diagnostic step toward the full port; it does not yet reconstruct
/// faces/normals/texcoords or the prediction graph.
pub fn probe_clod_progressive(g: &GeometryChunk, md: Option<&MeshDescription>) {
    let pos_iq = md.map(|m| m.position_inv_quant).unwrap_or(1.0);
    info!(
        "\n═══ probe_clod_progressive {:?} np={} nf={} pos_iq={:.10} body={}B ═══",
        g.name,
        g.num_positions,
        g.num_faces,
        pos_iq,
        g.compressed_data.len(),
    );

    let mut bs = BitStream::new(&g.compressed_data);

    // One resolution-update record header.
    let new_pos = bs.read_compressed_u32(X_CTX_NEW_POS_COUNT);
    let c3 = bs.read_compressed_u32(X_CTX_C3);
    let c2 = bs.read_compressed_u32(X_CTX_C2);
    let pred_count = bs.read_compressed_u32(X_CTX_PRED_COUNT);
    debug!(
        "  header: new_pos(ctx1)={} c3(ctx3)={} c2(ctx2)={} pred_count(ctx4)={} @bit{}",
        new_pos,
        c3,
        c2,
        pred_count,
        bs.bit_count(),
    );

    // Predecessor position indices (uACStaticFull + base) — only when pred_count>0.
    let mut preds: Vec<u32> = Vec::new();
    if pred_count > 0 && pred_count < 4096 {
        let first = bs.read_compressed_u32(X_AC_STATIC_FULL); // base = 0 for empty mesh
        preds.push(first);
        for _ in 1..pred_count {
            let d = bs.read_compressed_u32(X_AC_STATIC_FULL);
            preds.push(preds.last().unwrap().wrapping_add(d));
        }
        debug!(
            "  preds={:?} @bit{}",
            &preds[..preds.len().min(16)],
            bs.bit_count()
        );
    }

    // Per-new-position decode.
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let limit = new_pos.min(64);
    for i in 0..limit {
        let pred_type = bs.read_compressed_u8(X_CTX_POS_PRED_TYPE);
        let mut base = [0.0f32; 3];
        if pred_type != 4 {
            let split = bs.read_compressed_u32(X_CTX_SPLIT_IDX);
            if let Some(p) = positions.get(split as usize) {
                base = *p;
            }
        }
        let sign = bs.read_compressed_u8(X_CTX_POS_SIGN);
        let mx = bs.read_compressed_u32(X_CTX_POS_MAG);
        let my = bs.read_compressed_u32(X_CTX_POS_MAG);
        let mz = bs.read_compressed_u32(X_CTX_POS_MAG);
        let sx = if sign & 1 != 0 { -1.0 } else { 1.0 };
        let sy = if sign & 2 != 0 { -1.0 } else { 1.0 };
        let sz = if sign & 4 != 0 { -1.0 } else { 1.0 };
        let p = [
            base[0] + sx * (mx as f32) * pos_iq,
            base[1] + sy * (my as f32) * pos_iq,
            base[2] + sz * (mz as f32) * pos_iq,
        ];
        debug!(
            "  pos[{:2}] predT={} sign=0x{:02x} mag=({:>8},{:>8},{:>8}) → ({:9.4},{:9.4},{:9.4}) @bit{}",
            i,
            pred_type,
            sign,
            mx,
            my,
            mz,
            p[0],
            p[1],
            p[2],
            bs.bit_count(),
        );
        positions.push(p);
    }
    info!(
        "  consumed {}/{} bits",
        bs.bit_count(),
        g.compressed_data.len() * 8
    );
}

/// CORRECTED probe: the binary 0x49 handler (FUN_7a183c50) raw-consumes only
/// name + ONE u32 (the first count = maxRes) while the AC coder is still pristine
/// (FUN_7a11dbf0 reads raw bytes while high=0xffff,code=0). chunks.rs strips name
/// + FIVE u32s, so it over-strips by 4 u32 (16 bytes). Those 16 bytes are actually
/// the FIRST arithmetic-coded reads of FUN_7a18a990. This probe reconstructs the
/// real AC stream (the 4 over-stripped counts re-prepended) and runs the
/// progressive ctx-1.. read order from the true start.
pub fn probe_basemesh(g: &GeometryChunk, md: Option<&MeshDescription>) {
    let pos_iq = md.map(|m| m.position_inv_quant).unwrap_or(1.0);
    // Reconstruct the AC stream the binary actually sees: chunks.rs already stripped
    // name + 5 u32. The binary kept count[0] (=num_positions) as maxRes (raw). The
    // remaining 4 counts belong to the AC stream. Re-prepend them.
    let mut ac: Vec<u8> = Vec::with_capacity(16 + g.compressed_data.len());
    for v in [g.num_faces, g.num_normals, g.num_texcoords, g.field5] {
        ac.extend_from_slice(&v.to_le_bytes());
    }
    ac.extend_from_slice(&g.compressed_data);

    info!(
        "\n═══ probe_basemesh {:?} maxRes(np)={} nf={} nn={} ntc={} f5={} pos_iq={:.10} ac_len={}B ═══",
        g.name,
        g.num_positions,
        g.num_faces,
        g.num_normals,
        g.num_texcoords,
        g.field5,
        pos_iq,
        ac.len(),
    );

    let mut bs = BitStream::new(&ac);

    // First record header reads (FUN_7a18a990 lines 184/187/191/195).
    let c1 = bs.read_compressed_u32(1);
    let c3 = bs.read_compressed_u32(3);
    let c2 = bs.read_compressed_u32(2);
    let c4 = bs.read_compressed_u32(4);
    debug!(
        "  rec0 header: ctx1={} ctx3={} ctx2={} ctx4={} @bit{}",
        c1,
        c3,
        c2,
        c4,
        bs.bit_count(),
    );

    // Predecessor position index array (ctx = StaticFull + index, lines 212/228).
    // First is read with ctx (base + 0x400). Subsequent are delta-coded.
    if c4 > 0 && c4 < 4096 {
        let mut preds = Vec::new();
        let first = bs.read_compressed_u32(X_AC_STATIC_FULL); // base index 0
        preds.push(first);
        for i in 1..c4 {
            // ctx = (range - preds[i-1]) + 0x400; range bound unknown here, use 0x400 as a probe
            let d = bs.read_compressed_u32(X_AC_STATIC_FULL);
            preds.push(preds[(i - 1) as usize].wrapping_add(d));
        }
        debug!("  preds(ctx~0x400)={:?} @bit{}", preds, bs.bit_count());
    }

    // Per-split position decode (ctx 6 predType / 5 splitIdx / 7 signs / 8,8,8 mags).
    bs.trace_reads = true;
    bs.trace_label = "bm".into();
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let n = c1.min(64);
    for i in 0..n {
        if i >= 2 {
            bs.trace_reads = false;
        }
        let pred_type = bs.read_compressed_u8(6);
        let mut base = [0.0f32; 3];
        if pred_type != 4 {
            let split = bs.read_compressed_u32(5);
            if let Some(p) = positions.get(split as usize) {
                base = *p;
            }
        }
        let sign = bs.read_compressed_u8(7);
        let mx = bs.read_compressed_u32(8) as i32;
        let my = bs.read_compressed_u32(8) as i32;
        let mz = bs.read_compressed_u32(8) as i32;
        // Binary sign convention (FUN_7a18a990 lines 363-389):
        //   if (sign & bit)==0: comp = base + mag*iq ; else: comp = base - mag*iq
        let recon = |b: f32, m: i32, bit: u8| -> f32 {
            let mf = m as f32; // negative-int fix omitted (mags are positive magnitudes)
            if sign & bit == 0 {
                b + mf * pos_iq
            } else {
                b - mf * pos_iq
            }
        };
        let p = [
            recon(base[0], mx, 1),
            recon(base[1], my, 2),
            recon(base[2], mz, 4),
        ];
        debug!(
            "  pos[{:2}] predT={} sign=0x{:02x} mag=({},{},{}) → ({:.6},{:.6},{:.6}) @bit{}",
            i,
            pred_type,
            sign,
            mx,
            my,
            mz,
            p[0],
            p[1],
            p[2],
            bs.bit_count(),
        );
        positions.push(p);
    }
    info!("  consumed {}/{} bits", bs.bit_count(), ac.len() * 8);
}

/// Hand-decode probe: builds the binary's true AC stream (skip only maxRes=1 u32,
/// keep the other 4 counts which are the first AC reads), then decodes the
/// resolution-update header (ctx 1,3,2,4) and per-position records exactly per
/// FUN_7a18a990, dumping AC state after each read so we can spot desync vs the
/// expected quantized integers.
pub fn probe_handdecode(g: &GeometryChunk, md: Option<&MeshDescription>) {
    let pos_iq = md.map(|m| m.position_inv_quant).unwrap_or(1.0);
    // Binary: 0x49 handler reads name + maxRes(1 u32) raw. chunks.rs stripped name + 5 u32.
    // So re-prepend 4 over-stripped u32 (num_faces, num_normals, num_texcoords, field5):
    // those are the genuine first bytes of the AC stream.
    let mut ac: Vec<u8> = Vec::new();
    for v in [g.num_faces, g.num_normals, g.num_texcoords, g.field5] {
        ac.extend_from_slice(&v.to_le_bytes());
    }
    ac.extend_from_slice(&g.compressed_data);

    info!(
        "\n═══ probe_handdecode {:?} maxRes={} pos_iq={:.10e} ac_len={}B ═══",
        g.name,
        g.num_positions,
        pos_iq,
        ac.len()
    );
    // Expected quantized ints for arrow v[0..3] (from arrow.obj / pos_iq):
    info!("  expected v[0] quant ≈ (-100469, -106791, 2727164)");

    let mut bs = BitStream::new(&ac);

    // Resolution-update header: ctx 1,3,2,4 (CompU32).
    let n_pos = bs.read_compressed_u32(1);
    bs.dump_ac_state("after ctx1");
    let c3 = bs.read_compressed_u32(3);
    let c2 = bs.read_compressed_u32(2);
    let pred_count = bs.read_compressed_u32(4);
    debug!(
        "  header: n_pos(ctx1)={} c3={} c2={} pred_count(ctx4)={} @bit{}",
        n_pos,
        c3,
        c2,
        pred_count,
        bs.bit_count()
    );
    bs.dump_ac_state("after header");

    // Predecessor index array (ctx = 0x400 + ...). pred_count==0 for arrow → skip.
    let mut preds: Vec<u32> = Vec::new();
    if pred_count > 0 && pred_count < 256 {
        let first = bs.read_compressed_u32(X_AC_STATIC_FULL);
        preds.push(first);
        for i in 1..pred_count {
            let d = bs.read_compressed_u32(X_AC_STATIC_FULL);
            preds.push(preds[(i - 1) as usize].wrapping_add(d));
        }
        debug!("  preds={:?}", preds);
    }

    // Per-position records: ctx6 predType (CompU8), [ctx5 split if predType!=4],
    // ctx7 sign (CompU8), ctx8 magX, ctx8 magY, ctx8 magZ (CompU32).
    let bias = 4294967296.0f64; // _DAT_7a281438 = 2^32
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let count = n_pos.min(32);
    for i in 0..count {
        let pred_type = bs.read_compressed_u8(6);
        let mut base = [0.0f32; 3];
        let mut split = u32::MAX;
        if pred_type != 4 {
            split = bs.read_compressed_u32(5);
            if let Some(p) = positions.get(split as usize) {
                base = *p;
            }
        }
        let sign = bs.read_compressed_u8(7);
        let mx = bs.read_compressed_u32(8) as i32;
        let my = bs.read_compressed_u32(8) as i32;
        let mz = bs.read_compressed_u32(8) as i32;
        let recon = |b: f32, m: i32, bit: u8| -> f32 {
            // mag<0 fix: add 2^32 to treat as unsigned magnitude
            let mf = if m < 0 { (m as f64) + bias } else { m as f64 } as f32;
            if sign & bit == 0 {
                b + mf * pos_iq
            } else {
                b - mf * pos_iq
            }
        };
        let p = [
            recon(base[0], mx, 1),
            recon(base[1], my, 2),
            recon(base[2], mz, 4),
        ];
        debug!(
            "  pos[{:2}] predT={} split={} sign=0x{:02x} mag=({},{},{}) → ({:.5},{:.5},{:.5}) @bit{}",
            i,
            pred_type,
            split as i32,
            sign,
            mx,
            my,
            mz,
            p[0],
            p[1],
            p[2],
            bs.bit_count()
        );
        positions.push(p);
    }
    info!("  consumed {}/{} bits", bs.bit_count(), ac.len() * 8);
}

/// One decoded mesh: the flattened result of a mesh's group of 0x49 chunks.
///
/// Vertex attributes are parallel arrays indexed by the face corner indices in
/// `faces`; the exporter de-indexes them per shading group.
#[derive(Debug, Clone, Default)]
pub struct DecodedGeometry {
    /// Vertex positions in Director units, **Z-up**; consumers convert.
    pub positions: Vec<[f32; 3]>,
    /// Vertex normals, Z-up and unit length. Empty when the mesh's attribute
    /// mask declares no normals.
    pub normals: Vec<[f32; 3]>,
    /// Texture coordinates as stored in the file: V runs **bottom-up**, so the
    /// exporter mirrors V (`1 - v`) for glTF's top-down convention.
    pub texcoords: Vec<[f32; 2]>,
    /// Triangles as index triples into `positions` / `normals` / `texcoords`.
    pub faces: Vec<[u32; 3]>,
    /// Authored per-position skin weights (`LiveMesh::bone_weights`).
    ///
    /// One `(bone_index, weight)` list per position, decoded from the 0x49
    /// stream (count in arithmetic context 0x12, bone ids in 0x13, quantised
    /// weights in 0x14 with `w = raw / 32768`; the first entry takes
    /// `1 - sum(rest)` per the U3D "last weight is not written" rule). Empty
    /// for unskinned meshes.
    pub bone_weights: Vec<Vec<(u32, f32)>>,
    /// Shading-group id per face — indexes `MeshDescription::shaders`, i.e. the
    /// 0x45 declaration's shader slot the face is rendered with, which the
    /// exporter turns into one glTF primitive per slot.
    pub submesh_of_face: Vec<u32>,
}

// ── Full progressive-CLOD record (FUN_7a18a990) ──────────────────────────────
//
// Resolved float constants from the binary (read_memory @0x7a100000 base):
//   _DAT_7a281438 = f32 0x4f800000 = 4294967296.0   (2^32 neg-int fix)
//   _DAT_7a281304 = f32 0x3f800000 = 1.0
//   _DAT_7a281308 = f32 0x00000000 = 0.0
//   _DAT_7a285fb4 = f32 0xbf800000 = -1.0
//   _DAT_7a2866e0 = f64 1.0
//   _DAT_7a2893c8 = f64 -1.0
const NEG_FIX: f64 = 4294967296.0; // 2^32

/// Apply the binary's signed-magnitude reconstruction:
///   f = (float)mag;  if (mag < 0) f += 2^32;
///   comp = sign_bit ? base - f*iq : base + f*iq
fn recon_sm(base: f32, mag: u32, sign_bit: bool, iq: f32) -> f32 {
    let m = mag as i32;
    let mut f = m as f64;
    if m < 0 {
        f += NEG_FIX;
    }
    let f = f as f32;
    if sign_bit {
        base - f * iq
    } else {
        base + f * iq
    }
}

/// Reconstruct one normal from the two spherical angles, exactly per
/// FUN_7a18a990 lines 482-555. `mag1` = ctx0xc value (iStack_9c), `mag2` =
/// the acos-context value (iStack_98), `nsign` = ctx0xb bits (bStack_189),
/// `pred` = predecessor normal (fStack_180/17c/178), normal_iq = decoder+0xdc,
/// normal_iq2 = decoder+0xe0. Returns a normalized normal.
#[allow(clippy::too_many_arguments)]
fn recon_normal(mag1: u32, mag2: u32, nsign: u8, pred: [f32; 3], niq: f32, niq2: f32) -> [f32; 3] {
    // fStack_1e4 = mag1 (neg-fixed) * niq ; clamp to <=1.0
    let mut z = {
        let m = mag1 as i32;
        let mut f = m as f64;
        if m < 0 {
            f += NEG_FIX;
        }
        (f as f32) * niq
    };
    if z > 1.0 {
        z = 1.0;
    }

    // Tangent-plane angle: fVar16 = sin*r, fStack_f4 = cos*r where
    //   r = sqrt((1+z)(1-z)), phi = (mag2_negfixed * niq2)/r
    let (mut sin_part, mut cos_part): (f64, f64);
    if z.abs() < 1.0 {
        let r = (((z as f64) + 1.0) * (1.0 - (z as f64))).sqrt();
        let a2 = {
            let m = mag2 as i32;
            let mut f = m as f64;
            if m < 0 {
                f += NEG_FIX;
            }
            f
        };
        let phi = (a2 * (niq2 as f64)) / r;
        cos_part = phi.cos() * r;
        sin_part = phi.sin() * r;
    } else {
        cos_part = 0.0; // fStack_f4
        sin_part = 0.0; // fVar16 = _DAT_7a281308 = 0.0
    }

    // Octant sign flips (bStack_189): x->cos_part, y->sin_part, z->z
    let mut x = cos_part;
    if nsign & 1 != 0 {
        x = cos_part * -1.0;
    }
    if nsign & 2 != 0 {
        sin_part *= -1.0;
    }
    let mut zf = z as f64;
    if nsign & 4 != 0 {
        zf *= -1.0;
    }
    cos_part = x;

    // Rotate into predecessor frame if a predecessor normal exists
    // (|pred.z| < 1.0); else use raw (cos_part, sin_part, z).
    let (px, py, pz) = (pred[0] as f64, pred[1] as f64, pred[2] as f64);
    let mut out;
    if pz.abs() >= 1.0 {
        out = [cos_part, sin_part, zf];
    } else {
        let denom = (1.0 - pz * pz).sqrt();
        let inv = 1.0 / denom; // _DAT_7a281304 / sqrt(...)
        let m34 = pz * px * inv;
        let m30 = py * (-1.0) * inv; // _DAT_7a285fb4 = -1.0
        let m24 = pz * py * inv;
        let m18 = denom * (-1.0);
        out = [
            zf * px + cos_part * m34 + m30 * sin_part,
            cos_part * m24 + py * zf + inv * px * sin_part,
            sin_part * 0.0 /*_DAT_7a281308*/ + cos_part * m18 + pz * zf,
        ];
    }
    // Normalize
    let len = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2]).sqrt();
    if len > 0.0 {
        let s = 1.0 / len;
        out = [out[0] * s, out[1] * s, out[2] * s];
    }
    [out[0] as f32, out[1] as f32, out[2] as f32]
}

/// One deferred position-face corner rewrite, decoded from a record's group_a
/// entries. This is the XMED spelling of U3D's `IFXAuthorFaceUpdate`
/// (`CIFXAuthorCLODDecoder_P.cpp:702-726`) minus the `Attribute` field:
/// `dec_50_55_7a18a990.c:885-894` stores exactly four u32 per entry
/// (stride 0x10) — face index, corner, IncrValue, DecrValue.
#[derive(Debug, Clone, Copy)]
struct FaceUpdate {
    face: u32,
    corner: usize,
    /// IncrValue: the position index the corner moves TO.
    incr: u32,
    /// DecrValue: the position index it moves FROM (read back off the live face
    /// array through ctx 0x18, or a global escape when thirdType == 3).
    decr: u32,
}

/// When a record's group_a corner rewrites reach the live face array.
///
/// `FUN_7a18a990` never applies its own updates: the record end
/// (`dec_50_55_7a18a990.c:1330-1341`) writes the three counts into the group's
/// update array at the group's record cursor, and it is the CLOD manager's
/// `SetResolution`, called at the TOP of every resolution step with the
/// PRE-increment counter (`:158-160`), that walks them in. That walk is U3D's
/// `CIFXCLODManager::IncreaseTo(r)` (`IFXCLODManager.cpp:151`):
///
/// ```text
///   while (i < maxLocalRes && r > synchTable[i]) i++;
/// ```
///
/// i.e. a group's record `j` is applied once the pre-increment counter is
/// STRICTLY greater than its 0x47 schedule entry. Per-group schedules are
/// strictly increasing, so record `j` decodes exactly at post-counter
/// `S_j = schedule[j]`, and its rewrites are visible to every record decoded at
/// post-counter `>= S_j + REVEAL_DELAY`. On a contiguous schedule (arrow:
/// `[2..12]`) that is the "two records later" the winedbg traces measured
/// (`arrow_live_facearray_trace.log`: 13/13 snapshots, 22/22 face writes; one
/// record later gives 4/13 and 18/22). Counting group-local records instead of
/// resolution steps was wrong for every multi-group mesh with schedule gaps -
/// `bakgrunn/klippe` rec 64 (S=270) had to be visible to rec 65 (S=276), and
/// the record rule waited for rec 66 (S=277), which mispredicted vertex #132 by
/// exactly `pos[25] - pos[105]`.
const REVEAL_DELAY: u32 = 2;

/// `preds[]` holds FACE indices; `pred_type` 0/1/2 selects a corner of that
/// face, and the corner's value is a POSITION index. Every attribute predicts
/// from the attribute stored AT THAT POSITION INDEX — positions
/// (`dec_50_55_7a18a990.c:306-342`), normals (`:471-478` + `:427-456`) and
/// texcoords (`:651-664` + `:577-642`) all go through the same
/// `positionFaces[preds[split]].corner[predType]` indirection. That is the
/// per-corner attribute question settled: attributes are per-VERTEX and share
/// the position index space, so the U3D per-face attribute dup/split reads
/// (`CIFXAuthorCLODDecoder_P.cpp:1174-1230`) have no XMED counterpart.
fn pred_vertex(live: &LiveMesh, preds: &[u32], split: usize, pred_type: u8) -> Option<u32> {
    let face = *preds.get(split)?;
    live.face_corner(face, (pred_type as usize).min(2))
}

/// Emit one live-face-array snapshot in the format of
/// `tools/re/traces/arrow_live_facearray_trace.log` so a decode run can be
/// diffed against the winedbg capture by `tools/re/traces/diff_facearray.py`.
fn facedump(seq: &mut u32, site: &str, rec: u32, extra: &str, live: &LiveMesh, capacity: usize) {
    let cap = capacity.max(live.faces.len());
    let mut arr = String::from("[");
    for i in 0..cap {
        if i > 0 {
            arr.push_str(", ");
        }
        let f = live.faces.get(i).copied().unwrap_or([0, 0, 0]);
        arr.push_str(&format!("[{}, {}, {}]", f[0], f[1], f[2]));
    }
    arr.push(']');
    trace!(
        target: "macromelt::facedump",
        "FACEDUMP #{} site={} rec={} {} count={} arr={}",
        *seq, site, rec, extra, cap, arr
    );
    *seq += 1;
}

/// Decode a single 0x49 chunk on its own, returning the mesh it builds.
///
/// Convenience for the one-chunk case (a probe or a test looking at a single
/// chunk); a mesh the file split across several chunks must go through
/// [`decode_mesh_group`] instead.
///
/// Full progressive-record decode (port of `FUN_7a18a990`), driving a live
/// author-CLOD mesh.
///
/// The mesh is built by a sequence of resolution-update records (outer loop).
/// Each record:
///   header: ctx1=new_vert_count, ctx3=face_group_a, ctx2=face_group_b,
///           ctx4=predecessor_count
///   predecessor FACE index array (first: ctx = faceCount+0x400; then
///   delta-coded with ctx = faceCount-prev+0x400)
///   per new vertex (gated by the attribute mask):
///     position  (attr&1): ctx6 predType, [ctx5 split], ctx7 sign, ctx8 x/y/z
///     normal    (attr&2): ctx10 predType, [ctx9 split], ctx0xb sign,
///                         ctx0xc mag, acos→static ctx (round(acos)+0x401),
///                         spherical reconstruction
///     texcoord  (#layers = (attr>>4)&0xf):
///                         ctx0xe predType, [ctx0xf split], ctx0x10 sign,
///                         ctx0x11 u, ctx0x11 v
///     per-position shading list: ctx0x12 count, then ctx0x13 id
///                         (+ ctx0x14 blend weight for entries after the first)
///   group_a loop (ctx3 times): ctx0x15 split, 0x16 corner, 0x17 IncrValue
///                         (0 → global escape), 0x18 corner-of-DecrValue
///                         (3 → global escape). Deferred face corner rewrites.
///   group_b loop (ctx2 times): three corners, each ctx0x19 predType then
///                         0 → global escape, 1 → ctx0x1a delta off the record's
///                         first new position, 2/3/4 → ctx0x1b split + ctx0x1c
///                         delta off `liveFaces[preds[split]].corner[ptype-2]`.
///                         Appends one face per iteration.
///
/// `attr` is the author-mesh attribute mask (uStack_f0) — from MeshDescription,
/// not the bitstream. arrow declares 0x3 (positions + normals), kost and lynet
/// 0x13 (one UV layer).
pub fn decode_geometry(
    g: &GeometryChunk,
    md: Option<&MeshDescription>,
    trace: bool,
) -> DecodedGeometry {
    decode_geometry_group(&[g], md, trace)
}

/// Decode one mesh that the file wrote as a run of consecutive 0x49 chunks
/// (`XmedFile::geometry_groups`) into a fresh mesh. A single-chunk mesh is the
/// one-element case. No 0x47 schedule, so see [`decode_geometry_group_into`].
fn decode_geometry_group(
    chunks: &[&GeometryChunk],
    md: Option<&MeshDescription>,
    trace: bool,
) -> DecodedGeometry {
    let mut live = LiveMesh::new();
    decode_geometry_group_into(chunks, md, trace, &mut live);
    DecodedGeometry::from(&live)
}

/// Decode a group of 0x49 chunks against caller-owned mesh state, without a 0x47
/// schedule.
///
/// Refines `live` in place, so consecutive calls keep refining the same mesh.
/// Correct only for single-shading-group meshes: without the schedule there is
/// no way to tell which shading group a record belongs to. Use
/// [`decode_mesh_group`], which looks the schedule up, for everything else.
pub fn decode_geometry_group_into(
    chunks: &[&GeometryChunk],
    md: Option<&MeshDescription>,
    trace: bool,
    live: &mut LiveMesh,
) {
    decode_geometry_scheduled_into(chunks, md, None, trace, live);
}

/// Decode one of `xmed.geometry_groups()` into `live`, with the mesh's 0x45
/// declaration and 0x47 resolution schedule looked up by mesh name.
///
/// This is the route every consumer of native geometry (`--decode-geom`, the
/// GLB bake) goes through: it is the only one that keeps the records of an
/// interleaved multi-shading-group mesh apart. Read the result out of `live`
/// with `DecodedGeometry::from(&live)`.
pub fn decode_mesh_group(
    xmed: &XmedFile,
    group: &[&GeometryChunk],
    trace: bool,
    live: &mut LiveMesh,
) {
    let name = group[0].name.as_str();
    let md = xmed.mesh_descriptions.iter().find(|m| m.name == name);
    let sched = md.zip(xmed.clod_schedule(name)).map(|(m, c)| {
        let counts: Vec<u32> = m.shaders.iter().map(|s| s.num_texcoords).collect();
        decode_clod_schedule(c, &counts)
    });
    decode_geometry_scheduled_into(group, md, sched.as_deref(), trace, live);
}

/// Decode a mesh's run of 0x49 chunks into `live`.
///
/// 30 of the 91 XMED geometry names split one mesh across 2-3 chunks. Each
/// chunk restarts the arithmetic coder — the 0x49 handler `FUN_7a183c50`
/// releases `decoder->0x118` and creates a fresh one per chunk
/// (`0x7a183c8e-0x7a183cb1`) — while the mesh state, the per-shading-group
/// record cursors and the global resolution counter all carry over.
///
/// **The record stream is interleaved across shading groups.** `FUN_7a18a990`
/// (`dec_50_55_7a18a990.c:156-181`) is a resolution loop, not a record loop:
///
/// ```text
///   repeat chunk->numResolutionUpdates times:      // the chunk's first raw u32
///       resolution += 1                            // decoder->0x94, persists
///       for s in 0 .. numShadingGroups:            // decoder->0x78
///           if cursor[s] < numRecords[s]                       // 0x98[s], 0xb4[s]
///              && schedule[s][cursor[s]] <= resolution:        // 0x47 chunk
///               decode one record into shading group s
///               cursor[s] += 1
/// ```
///
/// Each shading group is its own `IFXAuthorMesh` (`decoder->0x7c[s]`) with its
/// own attribute mask, position array, face array and adjacency; the groups are
/// concatenated into the render mesh afterwards. Decoding the records as one
/// flat stream against one mesh — what this decoder did before — reads the
/// right bits but resolves every predecessor index, `pos_base` and face-corner
/// reference against the wrong array the moment the stream switches group.
/// That, not the ctx 0x12/0x13/0x14 shading list, was the multi-shader desync.
fn decode_geometry_scheduled_into(
    chunks: &[&GeometryChunk],
    md: Option<&MeshDescription>,
    schedule: Option<&[Vec<u32>]>,
    trace: bool,
    live: &mut LiveMesh,
) {
    let slots = md.map(|m| m.shaders.as_slice()).unwrap_or(&[]);
    let n_sub = slots.len().max(1);
    let fallback_attr = md.map(MeshDescription::attribute_mask).unwrap_or(0x3);
    let mut subs: Vec<SubMesh> = (0..n_sub)
        .map(|s| {
            let slot = slots.get(s);
            SubMesh {
                live: LiveMesh {
                    record_appends: live.record_appends,
                    ..LiveMesh::new()
                },
                schedule: schedule
                    .and_then(|sc| sc.get(s))
                    .cloned()
                    .unwrap_or_default(),
                num_records: slot.map(|x| x.num_texcoords as usize),
                cursor: 0,
                deferred: Vec::new(),
                attr: slot.map(|x| x.attrs).unwrap_or(fallback_attr),
                target_positions: slot
                    .map(|x| x.num_positions)
                    .or_else(|| md.map(|m| m.num_positions))
                    .filter(|&n| n > 0),
                target_faces: slot
                    .map(|x| x.num_faces)
                    .or_else(|| md.map(|m| m.num_faces))
                    .filter(|&n| n > 0),
            }
        })
        .collect();

    let mut carry = ClodCarry::default();
    for g in chunks {
        decode_chunk_into(g, md, trace, &mut subs, &mut carry);
    }
    // The mesh's final resolution reveals every remaining vertex update, which
    // is why the last records' rewrites are part of the render mesh (validated
    // against arrow.obj).
    for sub in &mut subs {
        flush_deferred(sub, &mut carry.dump_seq, trace, md);
    }
    merge_submeshes(live, &subs);

    if trace {
        info!(
            "  MESH {:?} positions={} normals={} texcoords={} faces={} ({} chunk{}, {} shading group{})",
            chunks.first().map(|c| c.name.as_str()).unwrap_or(""),
            live.positions.len(),
            live.normals.len(),
            live.texcoords.len(),
            live.faces.len(),
            chunks.len(),
            if chunks.len() == 1 { "" } else { "s" },
            n_sub,
            if n_sub == 1 { "" } else { "s" }
        );
        for (s, sub) in subs.iter().enumerate() {
            info!(
                "    group[{}] records {}/{} positions {}/{} faces {}/{} attr=0x{:x}",
                s,
                sub.cursor,
                sub.num_records.unwrap_or(0),
                sub.live.positions.len(),
                sub.target_positions.unwrap_or(0),
                sub.live.faces.len(),
                sub.target_faces.unwrap_or(0),
                sub.attr
            );
        }
        info!(
            "  TOPOLOGY orphan_positions={:?} degenerate_faces={:?}",
            live.orphan_positions(),
            live.degenerate_faces()
        );
    }
    if log_enabled!(target: "macromelt::posdump", Level::Trace) {
        for (i, p) in live.positions.iter().enumerate() {
            trace!(target: "macromelt::posdump", "POS {} {:.5} {:.5} {:.5}", i, p[0], p[1], p[2]);
        }
        for (i, t) in live.texcoords.iter().enumerate() {
            trace!(target: "macromelt::posdump", "TC {} {:.6} {:.6}", i, t[0], t[1]);
        }
        for (i, f) in live.faces.iter().enumerate() {
            trace!(target: "macromelt::posdump", "FACE {} {} {} {}", i, f[0], f[1], f[2]);
        }
    }
}

/// One shading group's decode state — the engine's `decoder->0x7c[s]` author
/// mesh plus the per-group cursors (`0x98[s]` record, `0x9c[s]` position,
/// `0xa0[s]` face).
struct SubMesh {
    live: LiveMesh,
    /// `schedule[j]` = resolution step at which record `j` unlocks (0x47 chunk).
    /// Empty means ungated: records decode one per resolution step, in order,
    /// which is what a single-group mesh does anyway.
    schedule: Vec<u32>,
    /// `decoder->0xb4[s]`; `None` when the file has no 0x45 declaration.
    num_records: Option<usize>,
    /// `decoder->0x98[s]`: records consumed by this group so far.
    cursor: usize,
    /// Group_a corner rewrites not yet applied, each with the post-counter at
    /// which `SetResolution` reveals it (see `REVEAL_DELAY`). Decode order.
    deferred: Vec<(u32, FaceUpdate)>,
    /// This group's per-vertex attribute mask (`uStack_f0`).
    attr: u32,
    target_positions: Option<u32>,
    target_faces: Option<u32>,
}

impl SubMesh {
    /// The engine's decode gate: `dec_50_55_7a18a990.c:178-181`.
    fn ready(&self, resolution: u32) -> bool {
        if self.num_records.is_some_and(|n| self.cursor >= n) {
            return false;
        }
        match self.schedule.get(self.cursor) {
            Some(&step) => step <= resolution,
            // Ungated (no 0x47 chunk): one record per resolution step.
            None if self.schedule.is_empty() => true,
            None => false,
        }
    }

    /// Every declared record consumed. A group with no 0x45 declaration is
    /// bounded only by the schedule / the chunk's resolution-step count.
    fn exhausted(&self) -> bool {
        match self.num_records {
            Some(n) => self.cursor >= n,
            None => false,
        }
    }
}

/// Concatenate the shading groups into the render mesh, in shading-id order —
/// the merge the 0x45 handler's tail performs (`dec_9_18_7a1858d0.c:660-673`,
/// `IFXMeshGroup::Allocate(decoder->0x7c, decoder->0x80)`). Group `s` owns the
/// contiguous position range starting at the sum of the earlier groups' counts,
/// which is exactly how the declaration's totals are stated.
fn merge_submeshes(live: &mut LiveMesh, subs: &[SubMesh]) {
    if subs.len() == 1 {
        let only = &subs[0];
        live.clone_from(&only.live);
        live.submesh_of_position = vec![0; live.positions.len()];
        live.submesh_of_face = vec![0; live.faces.len()];
        return;
    }
    let record_appends = live.record_appends;
    live.record_appends = false;
    for (s, sub) in subs.iter().enumerate() {
        let off = live.positions.len() as u32;
        for p in &sub.live.positions {
            live.push_position(*p);
            live.submesh_of_position.push(s as u32);
        }
        live.normals.extend_from_slice(&sub.live.normals);
        live.texcoords.extend_from_slice(&sub.live.texcoords);
        live.bone_weights.extend_from_slice(&sub.live.bone_weights);
        for f in &sub.live.faces {
            live.push_face([f[0] + off, f[1] + off, f[2] + off]);
            live.submesh_of_face.push(s as u32);
        }
        if record_appends {
            live.appends.extend(
                sub.live
                    .appends
                    .iter()
                    .map(|f| [f[0] + off, f[1] + off, f[2] + off]),
            );
        }
    }
    live.record_appends = record_appends;
}

/// Decoder state that outlives a single 0x49 chunk.
#[derive(Default)]
struct ClodCarry {
    /// `decoder->0x94`: the global resolution counter. Persists across chunks.
    resolution: u32,
    /// `macromelt::facedump` trace snapshot counter.
    dump_seq: u32,
}

/// Immutable per-chunk decode parameters shared by every record.
struct RecordEnv {
    pos_iq: f32,
    tc_iq: f32,
    nrm_iq: f32,
    nrm_iq2: f32,
    /// Ceiling on a record's header counts; a bigger value means the AC desynced.
    count_cap: u32,
    /// Ceiling on a decoded face corner (a position index).
    pos_cap: u32,
    /// Whether each new face carries the attribute-face record of
    /// `call_7a18a990.c:1209-1324` (mesh `attributes` bit 1).
    attribute_faces: bool,
    trace: bool,
    dump_faces: bool,
    face_capacity: usize,
}

/// Decode one 0x49 chunk, driving every shading group's records off the shared
/// arithmetic stream in resolution order.
fn decode_chunk_into(
    g: &GeometryChunk,
    md: Option<&MeshDescription>,
    trace: bool,
    subs: &mut [SubMesh],
    carry: &mut ClodCarry,
) {
    let m = md;
    let nrm_iq = m.map(|x| x.normal_inv_quant).unwrap_or(1.0);
    // This chunk's `numResolutionUpdates` — the ONE raw u32 the 0x49 handler
    // reads before handing the rest to the arithmetic coder
    // (`FUN_7a183c50` @0x7a183dd4: `decoder->ReadU32X(&this->0x11c)`).
    let updates = g.num_positions;
    if trace && let Some(m) = md {
        let slots: Vec<(&str, u32, u32)> = m
            .shaders
            .iter()
            .map(|s| (s.material.as_str(), s.attrs, s.num_texcoords))
            .collect();
        info!(
            "  [MDPROBE {:?} attrs=0x{:x} npos={} nfaces={} nnorm={} ntc={} nshaders={} slots(material,mask,records)={:?}]",
            m.name,
            m.attributes,
            m.num_positions,
            m.num_faces,
            m.num_normals,
            m.num_texcoords,
            m.num_shaders,
            slots
        );
    }

    // AC stream framing. chunks.rs strips name + 5 u32; only the first is raw.
    let mut ac: Vec<u8> = Vec::with_capacity(16 + g.compressed_data.len());
    for v in [g.num_faces, g.num_normals, g.num_texcoords, g.field5] {
        ac.extend_from_slice(&v.to_le_bytes());
    }
    ac.extend_from_slice(&g.compressed_data);

    let env = RecordEnv {
        pos_iq: m.map(|x| x.position_inv_quant).unwrap_or(1.0),
        tc_iq: m.map(|x| x.texcoord_inv_quant).unwrap_or(1.0),
        nrm_iq,
        // The binary uses two normal factors (+0xdc, +0xe0). The 0x45 chunk
        // parses one normal_inv_quant; +0xe0 is empirically the same factor.
        nrm_iq2: nrm_iq,
        count_cap: md
            .map(|x| x.num_positions)
            .unwrap_or(0)
            .max(updates)
            .max(64),
        pos_cap: md
            .map(|x| x.num_positions)
            .unwrap_or(u32::MAX / 4)
            .max(updates),
        attribute_faces: md.is_some_and(|m| m.attributes & 2 != 0),
        trace,
        dump_faces: log_enabled!(target: "macromelt::facedump", Level::Trace),
        // Declared face capacity — the allocated slot count the winedbg
        // snapshots dump (`count=22` for arrow). Only pads facedump trace output.
        face_capacity: md.map(|m| m.num_faces as usize).unwrap_or(0),
    };

    if trace {
        info!(
            "\n═══ decode_geometry {:?} updates={} groups={} pos_iq={:.6e} tc_iq={:.6e} nrm_iq={:.6e} ac={}B ═══",
            g.name,
            updates,
            subs.len(),
            env.pos_iq,
            env.tc_iq,
            nrm_iq,
            ac.len()
        );
    }

    let mut bs = BitStream::new(&ac);
    bs.full_trace = log_enabled!(target: "macromelt::fulltrace", Level::Trace);
    let total_bits = bs.total_bits();
    let mut chunk_rec = 0usize;

    // `dec_50_55_7a18a990.c:156-181`: repeat `numResolutionUpdates` times,
    // bumping the global resolution counter, and give each shading group at
    // most one record per step. The counter persists across chunks.
    'updates: for _ in 0..updates {
        carry.resolution += 1;
        // `SetResolution(pre-increment counter)` on every group's author mesh:
        // reveal the corner rewrites whose schedule step is now two behind.
        for sub in subs.iter_mut() {
            reveal_updates(
                sub,
                carry.resolution,
                &mut carry.dump_seq,
                trace,
                env.dump_faces,
                env.face_capacity,
            );
        }
        if bs.bit_count() >= total_bits {
            if trace {
                debug!("  stream exhausted @bit{}", bs.bit_count());
            }
            break;
        }
        // Every group has consumed its declared record count: nothing left to
        // read even if the chunk claims more resolution steps.
        if subs.iter().all(SubMesh::exhausted) {
            if trace {
                debug!("  all records consumed @bit{}", bs.bit_count());
            }
            break;
        }
        for s in 0..subs.len() {
            // NB: no "mesh already full" gate. A group's last records may add
            // no positions at all, and skipping them would leave their bits in
            // the stream and desync every later group.
            if !subs[s].ready(carry.resolution) {
                continue;
            }
            if trace {
                debug!(
                    "  ── res {} group {} record {} @bit{}",
                    carry.resolution,
                    s,
                    subs[s].cursor,
                    bs.bit_count()
                );
            }
            if !decode_record(
                &mut bs,
                &mut subs[s],
                &env,
                carry.resolution,
                &mut carry.dump_seq,
            ) {
                break 'updates;
            }
            subs[s].cursor += 1;
            chunk_rec += 1;
        }
    }

    if trace {
        info!(
            "  CHUNK done: {} records, resolution={} consumed {}/{} bits",
            chunk_rec,
            carry.resolution,
            bs.bit_count(),
            ac.len() * 8
        );
    }
}

/// `SetResolution` for one group: apply every pending corner rewrite whose
/// reveal step is `<= resolution` (the post-increment counter of the step that
/// is about to decode). See `REVEAL_DELAY`.
fn reveal_updates(
    sub: &mut SubMesh,
    resolution: u32,
    dump_seq: &mut u32,
    trace: bool,
    dump_faces: bool,
    face_capacity: usize,
) {
    let rec = sub.cursor as u32;
    let SubMesh { live, deferred, .. } = sub;
    // Pending rewrites are in decode order, so the due ones form a prefix.
    let due = deferred
        .iter()
        .take_while(|(at, _)| *at <= resolution)
        .count();
    for (_, u) in deferred.drain(..due) {
        apply_face_update(live, &u, trace);
        if dump_faces {
            facedump(
                dump_seq,
                "update",
                rec,
                &format!(
                    "face={} corner={} incr={} decr={}",
                    u.face, u.corner, u.incr, u.decr
                ),
                live,
                face_capacity,
            );
        }
    }
}

/// Decode one progressive record into one shading group at global resolution
/// step `resolution` (post-increment counter). Returns false when the stream
/// desynced and the chunk must be abandoned.
fn decode_record(
    bs: &mut BitStream,
    sub: &mut SubMesh,
    env: &RecordEnv,
    resolution: u32,
    dump_seq: &mut u32,
) -> bool {
    let trace = env.trace;
    let dump_faces = env.dump_faces;
    let face_capacity = env.face_capacity;
    let pos_iq = env.pos_iq;
    let tc_iq = env.tc_iq;
    let nrm_iq = env.nrm_iq;
    let nrm_iq2 = env.nrm_iq2;
    let count_cap = env.count_cap;
    let pos_cap = env.pos_cap;
    // Per-vertex attribute mask (uStack_f0): &1 positions, &2 normals,
    // (>>4)&0xf texcoord layers — this GROUP's mask, read off its own author
    // mesh (`dec_50_55_7a18a990.c:241-242`, vtable +0x18), not the mesh-wide OR.
    let has_normals = sub.attr & 2 != 0;
    let n_layers = ((sub.attr >> 4) & 0xf) as usize;
    let rec = sub.cursor;
    let SubMesh { live, deferred, .. } = sub;

    {
        // Record header
        bs.trace_reads =
            trace && log_enabled!(target: "macromelt::bittrace", Level::Trace) && rec == 1;
        bs.trace_label = format!("r{}", rec);
        let new_verts = bs.read_compressed_u32(1);
        let group_a = bs.read_compressed_u32(3); // uStack_100 (deferred face updates)
        let group_b = bs.read_compressed_u32(2); // uStack_104 (new faces)
        let pred_count = bs.read_compressed_u32(4);
        // Desync guard: per-record counts never legitimately exceed the mesh size.
        // A garbage count means the AC stream desynced — bail rather than spin.
        if new_verts > count_cap || group_a > count_cap || group_b > count_cap {
            if trace {
                debug!(
                    "  REC counts out of range (verts={} a={} b={} cap={}) — desync/end, bail @bit{}",
                    new_verts,
                    group_a,
                    group_b,
                    count_cap,
                    bs.bit_count()
                );
            }
            return false;
        }
        if trace {
            debug!(
                "  REC verts(ctx1)={} a(ctx3)={} b(ctx2)={} preds(ctx4)={} @bit{}",
                new_verts,
                group_a,
                group_b,
                pred_count,
                bs.bit_count()
            );
        }

        // Predecessor FACE index array. Binary (FUN_7a18a990 @0x7a18ab88-0x7a18ac14,
        // decompile lines 211-239): the static-context base `T` is
        // `*(int*)(decoder+0xa0)[submesh]`, the running position-FACE count — the
        // same cell the group_b loop writes each finished face to and increments
        // (decompile lines 1188-1207, 1325-1326).
        //   first read : ctx = T + 0x400
        //   read i (≥1): ctx = (T - pred[i-1]) + 0x400  ; pred[i] = pred[i-1] + value
        // The array is positional (`puStack_1c0[split]`), so it stays a Vec even
        // though its values are ascending like a `SetX`.
        let pred_base = live.faces.len() as u32;
        let mut preds: Vec<u32> = Vec::new();
        if pred_count > 0 && pred_count < 65536 {
            let first = bs.read_compressed_u32(pred_base.wrapping_add(X_AC_STATIC_FULL));
            preds.push(first);
            for i in 1..pred_count {
                let prev = preds[(i - 1) as usize];
                let ctx = pred_base.wrapping_sub(prev).wrapping_add(X_AC_STATIC_FULL);
                let d = bs.read_compressed_u32(ctx);
                preds.push(prev.wrapping_add(d));
            }
        }

        // First position index this record adds — the binary's
        // `positionCount - newVertCount` (decompile lines 834-835, 939).
        let pos_base = live.resolution;

        // Per-new-vertex loop
        for vi in 0..new_verts {
            // ── position (attr&1) ──
            // predType (ctx6): 4 = no predecessor (absolute, base 0). Values 0/1/2
            // select a CORNER of a predecessor FACE as the reconstruction base.
            let mut base = [0.0f32; 3];
            let pred_type = bs.read_compressed_u8(6);
            if pred_type != 4 {
                let split = bs.read_compressed_u32(5) as usize;
                if let Some(v) = pred_vertex(live, &preds, split, pred_type)
                    && let Some(p) = live.positions.get(v as usize)
                {
                    base = *p;
                } else if trace {
                    debug!(
                        "         ! pos pred unresolved (split={} predT={} preds={:?})",
                        split, pred_type, preds
                    );
                }
            }
            let sign = bs.read_compressed_u8(7);
            let mx = bs.read_compressed_u32(8);
            let my = bs.read_compressed_u32(8);
            let mz = bs.read_compressed_u32(8);
            let p = [
                recon_sm(base[0], mx, sign & 1 != 0, pos_iq),
                recon_sm(base[1], my, sign & 2 != 0, pos_iq),
                recon_sm(base[2], mz, sign & 4 != 0, pos_iq),
            ];
            live.push_position(p);
            if trace {
                debug!(
                    "    v{:<3} pos predT={} sign=0x{:02x} mag=({},{},{}) → ({:.5},{:.5},{:.5}) @bit{}",
                    vi,
                    pred_type,
                    sign,
                    mx as i32,
                    my as i32,
                    mz as i32,
                    p[0],
                    p[1],
                    p[2],
                    bs.bit_count()
                );
            }

            // ── normal (attr&2) ──
            if has_normals {
                let npred = bs.read_compressed_u8(10);
                let nsign: u8;
                let mag1: u32; // iStack_9c (ctx0xc)
                let mut pred_n = [0.0f32, 0.0, 1.0]; // fStack_180/17c/178 default (0,0,1)
                if npred == 4 {
                    nsign = bs.read_compressed_u8(0xb);
                    mag1 = bs.read_compressed_u32(0xc);
                } else {
                    let split = bs.read_compressed_u32(9) as usize;
                    if let Some(v) = pred_vertex(live, &preds, split, npred)
                        && let Some(n) = live.normals.get(v as usize)
                    {
                        pred_n = *n;
                    }
                    // predicted-normal path also re-reads ctx0xb/0xc in case4 of inner switch
                    nsign = bs.read_compressed_u8(0xb);
                    mag1 = bs.read_compressed_u32(0xc);
                }
                // acos→static-context read for the 2nd angle (iStack_98).
                // EXACT binary math (FUN_7a18a990 disasm @0x7a18b3a4-b428):
                //   z   = (float)(mag1 negfix) * niq(+0xdc);  if z>1.0 → z=1.0
                //   r   = sqrt((1.0 - z) * (1.0 + z))
                //   acc = _CIacos(0.0) * r            ; _CIacos(0.0) == PI/2 (const!)
                //   acc = acc * (1.0 / niq)           ; FLD niq; FDIVR 1.0; FMULP
                //   ctx = (int)(acc + 0.5) + 0x401    ; FADD 0.5; truncate-to-int
                // i.e. the 2nd-angle context is a static context whose index is the
                // *tangent-plane arc length* (PI/2)·sin(theta) quantized by 1/niq.
                let z_clamped: f32 = {
                    let mm = mag1 as i32;
                    let mut f = mm as f64;
                    if mm < 0 {
                        f += NEG_FIX;
                    }
                    let mut v = (f as f32) * nrm_iq;
                    if v > 1.0 {
                        v = 1.0;
                    }
                    v
                };
                let zf = z_clamped as f64;
                let r = ((1.0 - zf) * (1.0 + zf)).max(0.0).sqrt();
                let acc = (std::f64::consts::FRAC_PI_2 * r) * (1.0 / (nrm_iq as f64));
                // FADD 0.5 then x87 FISTP = round-to-nearest-even (default control word).
                let ctx2 = ((acc + 0.5).floor() as i64 as u32).wrapping_add(0x401);
                let mag2 = bs.read_compressed_u32(ctx2);
                let n = recon_normal(mag1, mag2, nsign, pred_n, nrm_iq, nrm_iq2);
                live.normals.push(n);
                if trace {
                    debug!(
                        "         nrm predT={} sign=0x{:02x} m1={} acosCtx={} m2={} → ({:.4},{:.4},{:.4}) @bit{}",
                        npred,
                        nsign,
                        mag1 as i32,
                        ctx2,
                        mag2 as i32,
                        n[0],
                        n[1],
                        n[2],
                        bs.bit_count()
                    );
                }
            }

            // ── texcoords (n_layers) ──
            for _layer in 0..n_layers {
                let tpred = bs.read_compressed_u8(0xe);
                let mut tbase = [0.5f32, 0.5];
                if tpred != 4 {
                    let split = bs.read_compressed_u32(0xf) as usize;
                    if let Some(v) = pred_vertex(live, &preds, split, tpred)
                        && let Some(t) = live.texcoords.get(v as usize)
                    {
                        tbase = *t;
                    }
                }
                let tsign = bs.read_compressed_u8(0x10);
                let tu = bs.read_compressed_u32(0x11);
                let tv = bs.read_compressed_u32(0x11);
                let t = [
                    recon_sm(tbase[0], tu, tsign & 1 != 0, tc_iq),
                    recon_sm(tbase[1], tv, tsign & 2 != 0, tc_iq),
                ];
                live.texcoords.push(t);
                if trace {
                    debug!(
                        "         tc l{} predT={} sign=0x{:02x} uv=({},{}) → ({:.6},{:.6}) @bit{}",
                        _layer,
                        tpred,
                        tsign,
                        tu as i32,
                        tv as i32,
                        t[0],
                        t[1],
                        bs.bit_count()
                    );
                }
            }

            // ── per-position authored skin weights (ctx0x12 count, ctx0x13
            // bone ids, ctx0x14 quantized weights for the entries after the
            // first; w[0] = 1 - Σrest). Stored per POSITION by the binary
            // (decompile lines 709-793), not per face.
            let weight_count = bs.read_compressed_u32(0x12);
            let mut vweights: Vec<(u32, f32)> = Vec::new();
            if weight_count > 0 && weight_count < 65536 {
                let mut rest = 0.0f32;
                for fi in 0..weight_count {
                    let bone = bs.read_compressed_u32(0x13);
                    if fi == 0 {
                        vweights.push((bone, 0.0));
                    } else {
                        let raw = bs.read_compressed_u32(0x14);
                        let w = raw as f32 * WEIGHT_INV_QUANT;
                        rest += w;
                        vweights.push((bone, w));
                    }
                }
                vweights[0].1 = (1.0 - rest).max(0.0);
            }
            if trace {
                debug!(
                    "         weights(ctx0x12)={} {:?} @bit{}",
                    weight_count,
                    vweights,
                    bs.bit_count()
                );
            }
            live.bone_weights.push(vweights);
        }

        // ── group_a: deferred position-face corner rewrites (ctx0x15..0x18) ──
        // decompile lines 802-900:
        //   ctx0x15 split      → face index preds[split]
        //   ctx0x16 corner     → which corner moves (0/1/2)
        //   ctx0x17 IncrValue  → 0: ReadU32X global escape;
        //                        else pos_base + value - 1 (this record's k-th new
        //                        position, 1-based)
        //   ctx0x18 cornerOfDecr → 0/1/2: DecrValue = liveFaces[preds[split]][that];
        //                          3: ReadU32X global escape
        // The four values are stored as one 16-byte face-update record; they are
        // NOT applied here (see FACE_UPDATE_LAG).
        for ga in 0..group_a {
            let split = bs.read_compressed_u32(0x15) as usize;
            let corner = bs.read_compressed_u8(0x16) as usize;
            let third = bs.read_compressed_u32(0x17);
            let incr = if third == 0 {
                bs.read_u32_via_context8()
            } else {
                pos_base.wrapping_add(third).wrapping_sub(1)
            };
            let third_type = bs.read_compressed_u8(0x18);
            let decr = if third_type == 3 {
                bs.read_u32_via_context8()
            } else {
                pred_vertex(live, &preds, split, third_type).unwrap_or(u32::MAX)
            };
            let Some(&face) = preds.get(split) else {
                if trace {
                    debug!(
                        "         ! group_a split {} out of preds {:?}",
                        split, preds
                    );
                }
                continue;
            };
            let update = FaceUpdate {
                face,
                corner,
                incr,
                decr,
            };
            if trace {
                let set_b = live.adj.face_set(decr);
                let mut set_c = live.adj.position_set(set_b, &live.faces);
                set_c.remove(decr);
                debug!(
                    "         GA[{}] face={} corner={} incr={} decr={} setB({})={:?} setC={:?} @bit{}",
                    ga,
                    face,
                    corner,
                    incr,
                    decr,
                    decr,
                    set_b.as_slice(),
                    set_c.as_slice(),
                    bs.bit_count()
                );
            }
            deferred.push((resolution + REVEAL_DELAY, update));
        }

        // ── group_b: new faces (corner graph 0x19..0x1c) ──
        // Each face has 3 corners; each corner reads ctx0x19 predType then, per
        // case: 0 → ReadU32X escape, 1 → ctx0x1a delta off pos_base, 2/3/4 →
        // ctx0x1b split + ctx0x1c delta off the LIVE face array
        // (decompile lines 909-1013: `liveFaces[preds[split]].corner[ptype-2] + delta`).
        // The object-method reads (+0x1c on the author-mesh object) are array
        // descriptor fetches, not bitstream reads.
        for fg in 0..group_b {
            let mut corners = [0u32; 3];
            for corner in 0..3usize {
                let ptype = bs.read_compressed_u8(0x19);
                let idx: u32 = match ptype {
                    0 => bs.read_u32_via_context8(),
                    1 => {
                        let d = bs.read_compressed_u32(0x1a);
                        pos_base.wrapping_add(d)
                    }
                    2 | 3 | 4 => {
                        let split = bs.read_compressed_u32(0x1b) as usize;
                        let d = bs.read_compressed_u32(0x1c);
                        if dump_faces {
                            facedump(
                                dump_seq,
                                "corner",
                                rec as u32,
                                &format!(
                                    "fg={} corner={} ptype={} split={} idx(preds[split])={}",
                                    fg,
                                    corner,
                                    ptype,
                                    split,
                                    preds.get(split).copied().unwrap_or(u32::MAX)
                                ),
                                live,
                                face_capacity,
                            );
                        }
                        pred_vertex(live, &preds, split, ptype - 2)
                            .unwrap_or(0)
                            .wrapping_add(d)
                    }
                    _ => 0,
                };
                corners[corner] = idx;
                if trace {
                    debug!(
                        "         faceB[{}] corner{} ptype={} idx={} @bit{}",
                        fg,
                        corner,
                        ptype,
                        idx,
                        bs.bit_count()
                    );
                }
            }
            if corners.iter().any(|&c| c >= pos_cap) {
                if trace {
                    debug!(
                        "  face corner out of range ({:?} cap={}) — desync, bail @bit{}",
                        corners,
                        pos_cap,
                        bs.bit_count()
                    );
                }
                return false;
            }
            // ── per-face attribute-face record (ctx 0x1d..0x24) ──
            // `call_7a18a990.c:1209-1324`, inside this same new-face loop and
            // gated on `decoder->0x64 != 0 && decoder->0x84 != 0`. It fills a
            // second, 28-byte-per-face array parallel to the position faces:
            // three per-corner attribute indices at +0x00, three next-face links
            // at +0x0c, three next-corner bytes at +0x18 and the ctx0x1d byte at
            // +0x1b — a per-corner linked list of faces sharing one attribute
            // index. Nothing in it feeds position or face reconstruction, so the
            // values are read and dropped; what matters is that the bits are
            // consumed, because the next face's ctx0x19 corner type follows them.
            //
            // Discriminator: the 0x45 declaration's mesh `attributes` word. It is
            // 4 on every one of disc 1's 424 declarations (block absent) and 6 on
            // every one of disc 2's (block present); the whole disc-2 corpus
            // desyncs on the second face of the first record without it.
            if env.attribute_faces {
                // `faceRec + 0x1b`; `decoder->0xa0[group]` still holds this
                // face's index — it is incremented at `:1325`, after the block.
                let face_index = live.faces.len() as u32;
                bs.read_compressed_u8(0x1d);
                for _ in 0..3 {
                    // `auStack_80[j]`, stored at `faceRec + 0x18 + j`.
                    bs.read_compressed_u8(0x1e);
                }
                for _ in 0..3 {
                    match bs.read_compressed_u8(0x1f) {
                        // Walk the existing linked list from a predecessor face
                        // (`FUN_7a15b810` + the loop at `:1266-1279`): split index
                        // then step count, both read, the walk itself consuming no
                        // bits.
                        0..=2 => {
                            bs.read_compressed_u32(0x22);
                            bs.read_compressed_u32(0x23);
                        }
                        // `:1292-1296` — absolute index in a static context keyed
                        // by this group's running face count.
                        4 => {
                            bs.read_compressed_u32(0x20);
                            bs.read_compressed_u32(face_index.wrapping_add(X_AC_STATIC_FULL));
                        }
                        // `:1301-1305` — delta off this group's face count.
                        5 => {
                            bs.read_compressed_u32(0x20);
                            bs.read_compressed_u32(0x21);
                        }
                        // `:1284-1289` — delta off `preds[type - 6]`.
                        _ => {
                            bs.read_compressed_u32(0x20);
                            bs.read_compressed_u32(0x24);
                        }
                    }
                }
            }
            live.push_face(corners);
            if dump_faces {
                facedump(
                    dump_seq,
                    "append",
                    rec as u32,
                    &format!("face={}", live.faces.len() - 1),
                    live,
                    face_capacity,
                );
            }
        }

        if trace {
            debug!(
                "  REC done: positions={} faces={} @bit{}",
                live.positions.len(),
                live.faces.len(),
                bs.bit_count()
            );
        }
        // An all-zero record is legal: the engine still counts it as one
        // resolution update for this group (`dec_50_55_7a18a990.c:1337-1339`
        // writes the three zero counts and advances the cursor). The record
        // count from the 0x45 declaration bounds the loop, so there is no
        // no-progress heuristic to fall back on.
    }
    true
}

/// Apply the face updates still pending when the mesh reaches its final
/// resolution (`SetResolution(max)`). Runs once per mesh, after its last chunk.
fn flush_deferred(
    sub: &mut SubMesh,
    dump_seq: &mut u32,
    trace: bool,
    md: Option<&MeshDescription>,
) {
    let dump_faces = log_enabled!(target: "macromelt::facedump", Level::Trace);
    let face_capacity = md.map(|m| m.num_faces as usize).unwrap_or(0);
    reveal_updates(sub, u32::MAX, dump_seq, trace, dump_faces, face_capacity);
}

/// Apply one deferred corner rewrite to the live mesh. `decr` is the binary's
/// read-back of the corner's current value; a mismatch means the update targets
/// a corner the decode has already diverged on, so it is reported and the
/// rewrite still applied (the binary stores Incr/Decr and applies Incr).
fn apply_face_update(live: &mut LiveMesh, u: &FaceUpdate, trace: bool) {
    if trace && u.decr != u32::MAX && live.face_corner(u.face, u.corner) != Some(u.decr) {
        debug!(
            "  ! face-update decr mismatch: face {} corner {} is {:?}, stream says {}",
            u.face,
            u.corner,
            live.face_corner(u.face, u.corner),
            u.decr
        );
    }
    if !live.set_corner(u.face, u.corner, u.incr) && trace {
        debug!(
            "  ! face-update out of range: face {} corner {} (faces={})",
            u.face,
            u.corner,
            live.faces.len()
        );
    }
}

/// Experimental decode attempt: treat body as sign+magnitude position deltas
/// using the U3D position-diff context IDs (20/21/22/23) and the inverse
/// quantization factor from the matching 0x45 MeshDescription.
///
/// Prints the first decoded positions for diagnosis.
pub fn try_decode_positions(g: &GeometryChunk, md: Option<&MeshDescription>) {
    let pos_iq = md.map(|m| m.position_inv_quant).unwrap_or(1.0);
    let count = g.num_positions.min(32);

    info!(
        "\n═══ try_decode_positions {:?} (np={}) pos_iq={:.10} body={}B ═══",
        g.name,
        g.num_positions,
        pos_iq,
        g.compressed_data.len(),
    );

    let mut bs = BitStream::new(&g.compressed_data);
    let mut prev = [0.0f32; 3];
    for i in 0..count {
        let sign = bs.read_compressed_u8(CTX_POSITION_DIFF_SIGNS);
        let dx = bs.read_compressed_u32(CTX_POSITION_DIFF_MAG_X);
        let dy = bs.read_compressed_u32(CTX_POSITION_DIFF_MAG_Y);
        let dz = bs.read_compressed_u32(CTX_POSITION_DIFF_MAG_Z);
        let sx = if sign & 1 != 0 { -1.0 } else { 1.0 };
        let sy = if sign & 2 != 0 { -1.0 } else { 1.0 };
        let sz = if sign & 4 != 0 { -1.0 } else { 1.0 };
        let p = [
            prev[0] + sx * (dx as f32) * pos_iq,
            prev[1] + sy * (dy as f32) * pos_iq,
            prev[2] + sz * (dz as f32) * pos_iq,
        ];
        debug!(
            "  pos[{:2}] sign=0x{:02x} d=({:>7},{:>7},{:>7}) → ({:8.4}, {:8.4}, {:8.4}) bit={}",
            i,
            sign,
            dx,
            dy,
            dz,
            p[0],
            p[1],
            p[2],
            bs.bit_count(),
        );
        prev = p;
    }
    info!(
        "  consumed {} bits of {} available",
        bs.bit_count(),
        g.compressed_data.len() * 8
    );
}

/// Dump the geometry chunk's bitstream as several different read-interpretations
/// in parallel. We read a few bytes, rewind, try another interpretation.
pub fn probe(g: &GeometryChunk) {
    info!(
        "\n═══ Geometry chunk: {:?}  header=(np={}, nf={}, nn={}, ntc={}, f5={}) comp={}B",
        g.name,
        g.num_positions,
        g.num_faces,
        g.num_normals,
        g.num_texcoords,
        g.field5,
        g.compressed_data.len(),
    );

    // We start from the compressed_data portion (chunks.rs already consumed
    // the name + 5 u32 header). So bit 0 of our BitStream = first byte after header.
    probe_as_u3d_static(g);
    probe_as_raw_positions(g);
    probe_ac_symbols(g);
    probe_as_declaration(g);
    try_decode_declaration(g);
    probe_ac_decode(g);
    probe_static_full(g);
}

/// Hypothesis D: XMED 0x49 starts with an IFX v2 CLOD Declaration sub-block:
///   NumDiffuseColors  (u32 raw via ReadU32X at pristine AC)
///   NumSpecularColors (u32)
///   NumShaders        (u32)
///   For each shader:
///     MaterialAttributes (u32)   - bit0=diffuse, bit1=specular
///     NumTextureLayers   (u32)
///     TexCoordDimensions (u32 × NumTextureLayers)
///     OriginalShadingID  (u32)
///   MinResolution      (u32)
///   FinalMaxResolution (u32)
///   3× u32 quality factors
///   5× f32 inverse quantization
///   3× f32 resource parameters
///
/// If this layout matches, first bytes of arrow (13v, 2 materials) should read:
///   NumDiff=0, NumSpec=0, NumShaders=1 or 2, then MaterialAttributes=0 or 3, etc.
fn probe_as_declaration(g: &GeometryChunk) {
    info!("  -- Hypothesis D: IFX v2 Declaration sub-block --");
    let mut bs = BitStream::new(&g.compressed_data);
    // Read 20 raw u32s and print — let's see if any pattern emerges
    info!("    first 20 raw u32 LE (starting offset 0):");
    for i in 0..20 {
        let v = bs.read_u32();
        let v_signed = v as i32;
        let f = f32::from_bits(v);
        debug!("      u32[{i:2}] = {v:10} (0x{v:08X}) signed={v_signed} f32={f:e}");
    }

    // Try u16-granular reads
    let mut bs2 = BitStream::new(&g.compressed_data);
    info!("    first 20 raw u16 LE:");
    for i in 0..20 {
        let v = bs2.read_u16();
        debug!("      u16[{i:2}] = {v:6} (0x{v:04X})");
    }

    // Try u8-granular reads
    let mut bs3 = BitStream::new(&g.compressed_data);
    info!("    first 40 raw u8:");
    let bytes: Vec<u8> = (0..40).map(|_| bs3.read_u8()).collect();
    info!(
        "      hex: {}",
        bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join(" ")
    );
    info!(
        "      dec: {}",
        bytes
            .iter()
            .map(|b| format!("{:3}", b))
            .collect::<Vec<_>>()
            .join(" ")
    );
}

/// Try reading as if this were a U3D Static block body (counts + positions + ...).
fn probe_as_u3d_static(g: &GeometryChunk) {
    info!("  -- Hypothesis A: U3D-style Static block body --");
    let mut bs = BitStream::new(&g.compressed_data);
    // In U3D Static block, after Name+ChainIndex (which we've notionally consumed
    // via the raw chunk header), comes:
    //   6 × u32 counts (NumFaces, NumPositions, NumNormals, NumDiffuse, NumSpecular, NumTexCoords)
    // then raw F32 position triples.
    // Try first: read 6 context8-u32 counts, then read first 3 position triples as context8-f32.
    // (Motion decoder used context8 for "pristine-or-first-after-AC" raw reads.)
    let c = [
        bs.read_u32_via_context8(),
        bs.read_u32_via_context8(),
        bs.read_u32_via_context8(),
        bs.read_u32_via_context8(),
        bs.read_u32_via_context8(),
        bs.read_u32_via_context8(),
    ];
    info!(
        "    c8 u32 ×6 = [{}, {}, {}, {}, {}, {}] (NumFaces?, NumPositions?, NumNormals?, NumDiff?, NumSpec?, NumTex?)",
        c[0], c[1], c[2], c[3], c[4], c[5]
    );
    info!("    c8 f32 ×9 (first 3 positions x/y/z):");
    for i in 0..3 {
        let x = bs.read_f32_via_context8();
        let y = bs.read_f32_via_context8();
        let z = bs.read_f32_via_context8();
        debug!("      pos[{i}] = ({:.4}, {:.4}, {:.4})", x, y, z);
    }
}

/// Try reading compressed_data as raw F32 triples (no AC, no bitstream).
fn probe_as_raw_positions(g: &GeometryChunk) {
    info!("  -- Hypothesis B: raw F32 positions right after header --");
    let d = &g.compressed_data;
    for i in 0..5usize.min(d.len() / 12) {
        let o = i * 12;
        let x = f32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]);
        let y = f32::from_le_bytes([d[o + 4], d[o + 5], d[o + 6], d[o + 7]]);
        let z = f32::from_le_bytes([d[o + 8], d[o + 9], d[o + 10], d[o + 11]]);
        debug!("    raw_f32[{i}] = ({:.6}, {:.6}, {:.6})", x, y, z);
    }
}

/// Attempt to decode the IFX v2 Declaration sub-block structure in order.
/// Prints each field with its offset so we can see what reads align to sensible values.
fn try_decode_declaration(g: &GeometryChunk) {
    info!(
        "\n-- DECODE ATTEMPT: IFX v2 Declaration for {:?} --",
        g.name
    );
    let data = &g.compressed_data;
    let read_u32_le = |o: usize| -> u32 {
        if o + 4 > data.len() {
            return 0;
        }
        u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]])
    };
    let read_f32_le = |o: usize| -> f32 {
        if o + 4 > data.len() {
            return 0.0;
        }
        f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]])
    };
    let mut o = 0usize;

    macro_rules! r_u32 {
        ($label:expr) => {{
            let v = read_u32_le(o);
            debug!("    @{:4} {:32} u32 = {:10} (0x{:08X})", o, $label, v, v);
            o += 4;
            v
        }};
    }
    macro_rules! r_f32 {
        ($label:expr) => {{
            let v = read_f32_le(o);
            debug!("    @{:4} {:32} f32 = {:e} ({:.6})", o, $label, v, v);
            o += 4;
        }};
    }

    let _exclude_normals = r_u32!("ExcludeNormals flag");
    let num_shaders = r_u32!("NumShaders");
    if num_shaders > 16 {
        warn!(
            "    (NumShaders={} looks wrong — aborting decl parse)",
            num_shaders
        );
        return;
    }
    for s in 0..num_shaders {
        r_u32!(&format!("Shader[{}] MaterialAttrs", s));
        let n_layers = r_u32!(&format!("Shader[{}] NumTextureLayers", s));
        if n_layers > 8 {
            warn!("    (NumTextureLayers={} too big — aborting)", n_layers);
            return;
        }
        for l in 0..n_layers {
            r_u32!(&format!("Shader[{}] TexCoordDim[{}]", s, l));
        }
        r_u32!(&format!("Shader[{}] OriginalShadingID", s));
    }
    r_u32!("MinResolution");
    r_u32!("FinalMaxResolution");
    r_u32!("PositionQuality");
    r_u32!("NormalQuality");
    r_u32!("TexCoordQuality");
    r_f32!("InverseQuantPosition");
    r_f32!("InverseQuantNormal");
    r_f32!("InverseQuantTexCoord");
    r_f32!("InverseQuantDiffuseColor");
    r_f32!("InverseQuantSpecularColor");
    r_f32!("NormalCreaseParameter");
    r_f32!("NormalUpdateParameter");
    r_f32!("NormalTolerance");
    info!(
        "    Declaration ended at offset {} (of {} bytes)",
        o,
        data.len()
    );
}

/// Hypothesis E: try AC-decoding the body using read_compressed_u16 with various contexts.
/// The motion decoder uses contexts 1..=8 for time/disp/rot/scale sign-and-diff pairs.
/// Maybe geometry uses similar contexts. Print first 30 decoded values.
fn probe_ac_decode(g: &GeometryChunk) {
    for ctx in [1u32, 2, 3, 4, 5, 6, 7, 8] {
        info!("\n  -- Hypothesis E: read_compressed_u16 with context={ctx} --");
        let mut bs = BitStream::new(&g.compressed_data);
        let mut vals = Vec::new();
        for _ in 0..40 {
            let v = bs.read_compressed_u16(ctx);
            vals.push(v);
            if bs.bit_count() >= (g.compressed_data.len() as u32) * 8 {
                break;
            }
        }
        debug!("    vals: {:?}", vals);
    }
}

/// Hypothesis F: maybe body is ReadCompressedU32(ctx=StaticFull+N) calls, which are
/// U3D's pattern for index-into-N-array reads (fixed range, uniform).
/// Try small N values (13 for arrow positions).
fn probe_static_full(g: &GeometryChunk) {
    const AC_STATIC_FULL: u32 = 0x400;
    let test_sizes = [g.num_positions, g.num_faces, g.num_texcoords, 13u32, 2, 6];
    for n in test_sizes {
        if n == 0 {
            continue;
        }
        info!("\n  -- Hypothesis F: read_compressed_u32(StaticFull + {n}) --");
        let mut bs = BitStream::new(&g.compressed_data);
        let mut vals = Vec::new();
        for _ in 0..30 {
            let v = bs.read_compressed_u32(AC_STATIC_FULL + n);
            vals.push(v);
        }
        debug!("    vals: {:?}", vals);
    }
}

/// Try reading the first several AC symbols in various static contexts.
fn probe_ac_symbols(g: &GeometryChunk) {
    info!("  -- Hypothesis C: direct AC stream (context8 bytes then ctx1-8) --");
    let mut bs = BitStream::new(&g.compressed_data);
    let mut c8_bytes = Vec::new();
    for _ in 0..16 {
        c8_bytes.push(bs.read_symbol_context8());
    }
    debug!(
        "    first 16 c8 bytes (hex): {}",
        c8_bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join(" ")
    );
    debug!(
        "    first 16 c8 bytes (ascii): {:?}",
        c8_bytes
            .iter()
            .map(|&b| if (0x20..0x7F).contains(&b) {
                b as char
            } else {
                '.'
            })
            .collect::<String>()
    );
}
