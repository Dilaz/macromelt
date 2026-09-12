//! Motion resource decoder for XMED 0x67 chunks.
//!
//! A motion is a named set of per-bone tracks, each a list of TRS keyframes:
//! time in seconds (absolute in the motion's own timeline), displacement in
//! Director units, rotation as a **wxyz** quaternion and a scale triple, all in
//! Director's **Z-up** space. The glTF exporter rebases each clip on its first
//! key so `t = 0` is the first pose.
//!
//! Implements the IFX v2 motion decoder. Key differences from U3D:
//!   - Every track carries its own header (name, keyframe count, displacement
//!     and scale quantization), not just track 0
//!   - Escape reads use AC static with SwapBits8 (ReadSymbolContext8)

use log::{debug, info, warn};

use crate::bitstream::BitStream;
use crate::chunks::{KeyFrame, MotionResource};

// Context IDs from IFXACContext.h (Intel U3D SDK)
const CTX_TIME_SIGN: u32 = 1;
const CTX_TIME_DIFF: u32 = 2;
const CTX_DISP_SIGN: u32 = 3;
const CTX_DISP_DIFF: u32 = 4;
const CTX_ROT_SIGN: u32 = 5;
const CTX_ROT_DIFF: u32 = 6;
const CTX_SCALE_SIGN: u32 = 7;
const CTX_SCALE_DIFF: u32 = 8;

fn inverse_quant(predicted: f32, sign_bit: bool, diff: u32, inv_quant: f32) -> f32 {
    let delta = inv_quant * diff as f32;
    if sign_bit {
        predicted - delta
    } else {
        predicted + delta
    }
}

/// A fully decoded motion: one [`DecodedTrack`] of TRS keyframes per bone.
///
/// Keyframe times are **absolute** in the motion's own timeline — Director's
/// `bonesPlayer.play` starts a motion at its first keyframe, so a motion whose
/// first key sits at 0.2333 s is not held for a quarter second. The glTF
/// exporter rebases every clip on its earliest key so that `t = 0` is the first
/// pose.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DecodedMotion {
    /// Motion name as authored in Director, e.g. `monica_walk`. This is the
    /// name the exporter turns into a glTF animation clip.
    pub name: String,
    /// Number of tracks the 0x67 chunk header declares. `tracks.len()` is what
    /// was actually decoded and may be shorter if the stream desynced.
    pub track_count: u32,
    /// Inverse quantization step for keyframe times, in seconds per code.
    pub time_inv_quant: f32,
    /// Inverse quantization step for the rotation quaternion components
    /// (dimensionless; components are normalized after reconstruction).
    pub rotation_inv_quant: f32,
    /// One track per animated bone, in the file's track order.
    pub tracks: Vec<DecodedTrack>,
}

/// One bone's animation: the keyframes of a single track of a [`DecodedMotion`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct DecodedTrack {
    /// Bone name the track drives; matches a `BoneDef::name` of the skeleton.
    pub name: String,
    /// Keyframes in file order (ascending time). Times are seconds and
    /// absolute in the motion's timeline, displacements are Director units in
    /// the bone's parent space, rotations are **wxyz** quaternions and the
    /// whole thing is Director **Z-up**.
    pub keyframes: Vec<KeyFrame>,
}

/// Read a u16-length-prefixed string from the bitstream (via ReadSymbolContext8).
fn read_string(bs: &mut BitStream) -> String {
    // U3D SDK ReadIFXStringX uses ReadU16X → 2 × ReadSymbolContext8
    let lo = bs.read_symbol_context8() as u16;
    let hi = bs.read_symbol_context8() as u16;
    let len = (lo | (hi << 8)) as usize;
    if len > 4096 {
        panic!(
            "read_string: absurd length {} — bitstream likely desynced",
            len
        );
    }
    let mut bytes = Vec::with_capacity(len);
    for _ in 0..len {
        bytes.push(bs.read_symbol_context8());
    }
    String::from_utf8_lossy(&bytes).to_string()
}

fn read_compressed_kf(
    bs: &mut BitStream,
    predicted: &KeyFrame,
    time_inv_quant: f32,
    rot_inv_quant: f32,
    disp_inv_quant: f32,
    scale_inv_quant: f32,
) -> KeyFrame {
    let sign = bs.read_compressed_u8(CTX_TIME_SIGN);
    let diff = bs.read_compressed_u32(CTX_TIME_DIFF);
    let time = inverse_quant(predicted.time, (sign & 1) != 0, diff, time_inv_quant);

    let sign = bs.read_compressed_u8(CTX_DISP_SIGN);
    let dx = bs.read_compressed_u32(CTX_DISP_DIFF);
    let dy = bs.read_compressed_u32(CTX_DISP_DIFF);
    let dz = bs.read_compressed_u32(CTX_DISP_DIFF);
    let displacement = [
        inverse_quant(
            predicted.displacement[0],
            (sign & 1) != 0,
            dx,
            disp_inv_quant,
        ),
        inverse_quant(
            predicted.displacement[1],
            (sign & 2) != 0,
            dy,
            disp_inv_quant,
        ),
        inverse_quant(
            predicted.displacement[2],
            (sign & 4) != 0,
            dz,
            disp_inv_quant,
        ),
    ];

    let sign = bs.read_compressed_u8(CTX_ROT_SIGN);
    let ub = bs.read_compressed_u32(CTX_ROT_DIFF);
    let uc = bs.read_compressed_u32(CTX_ROT_DIFF);
    let ud = bs.read_compressed_u32(CTX_ROT_DIFF);

    let mut fb = rot_inv_quant * ub as f32;
    let mut fc = rot_inv_quant * uc as f32;
    let mut fd = rot_inv_quant * ud as f32;
    if fb > 1.0 {
        fb = 1.0;
    }
    if fc > 1.0 {
        fc = 1.0;
    }
    if fd > 1.0 {
        fd = 1.0;
    }

    let fa_sq = 1.0 - (fb as f64 * fb as f64 + fc as f64 * fc as f64 + fd as f64 * fd as f64);
    let mut fa = if fa_sq > 0.0 {
        fa_sq.sqrt() as f32
    } else {
        0.0
    };

    if (sign & 1) != 0 {
        fa = -fa;
    }
    if (sign & 2) != 0 {
        fb = -fb;
    }
    if (sign & 4) != 0 {
        fc = -fc;
    }
    if (sign & 8) != 0 {
        fd = -fd;
    }

    let [pw, pi, pj, pk] = predicted.rotation;
    let rw = pw * fa - pi * fb - pj * fc - pk * fd;
    let ri = pw * fb + pi * fa + pj * fd - pk * fc;
    let rj = pw * fc - pi * fd + pj * fa + pk * fb;
    let rk = pw * fd + pi * fc - pj * fb + pk * fa;

    let mag = (rw * rw + ri * ri + rj * rj + rk * rk).sqrt();
    let rotation = if mag > 1e-10 {
        [rw / mag, ri / mag, rj / mag, rk / mag]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    };

    let sign = bs.read_compressed_u8(CTX_SCALE_SIGN);
    let sx = bs.read_compressed_u32(CTX_SCALE_DIFF);
    let sy = bs.read_compressed_u32(CTX_SCALE_DIFF);
    let sz = bs.read_compressed_u32(CTX_SCALE_DIFF);
    let scale = [
        inverse_quant(predicted.scale[0], (sign & 1) != 0, sx, scale_inv_quant),
        inverse_quant(predicted.scale[1], (sign & 2) != 0, sy, scale_inv_quant),
        inverse_quant(predicted.scale[2], (sign & 4) != 0, sz, scale_inv_quant),
    ];

    KeyFrame {
        time,
        displacement,
        rotation,
        scale,
    }
}

/// Decode a 0x67 motion resource into its per-bone tracks of TRS keyframes.
///
/// The chunk is one arithmetic-coded stream: a raw header (motion name, track
/// count, time and rotation quantization steps), then per track a header (bone
/// name, keyframe count, displacement and scale steps) followed by
/// delta-coded keyframes — each predicted from the previous one, sign bits and
/// magnitudes read from the IFX contexts named at the top of this module.
///
/// Times come out in seconds and stay **absolute** in this motion's timeline
/// (see [`DecodedMotion`]); displacements are Director units, rotations are
/// **wxyz** quaternions, and everything is Director **Z-up**.
pub fn decode_motion_full(motion: &MotionResource) -> DecodedMotion {
    let mut bs = BitStream::new(&motion.chunk_data);

    // Motion header (at pristine AC state = raw reads)
    let _motion_name = read_string(&mut bs);
    let _track_count = bs.read_u32();
    let time_inv_quant = bs.read_f32();
    let rot_inv_quant = bs.read_f32();

    let track_count = motion.track_count as usize;
    let mut tracks: Vec<DecodedTrack> = Vec::with_capacity(track_count);

    for track_idx in 0..track_count {
        // Per-track header via context8 (U3D SDK: each track has its own header)
        // At Track 0, AC is pristine so context8 = raw reads.
        // At Track 1+, AC is non-pristine so context8 = AC static + SwapBits8.
        let track_name = read_string(&mut bs);
        let time_count = bs.read_u32_via_context8();
        if time_count > 100_000 {
            panic!(
                "time_count {} is absurd at track {} — bitstream desynced",
                time_count, track_idx
            );
        }
        let disp_inv_quant = bs.read_f32_via_context8();
        let scale_inv_quant = bs.read_f32_via_context8();

        if track_idx < 5 || track_idx == track_count - 1 {
            debug!(
                "  Track {} \"{}\" kf={} disp_iq={:.6e} scale_iq={:.6e} @bit{}",
                track_idx,
                track_name,
                time_count,
                disp_inv_quant,
                scale_inv_quant,
                bs.bit_count(),
            );
        }

        let mut predicted = KeyFrame {
            time: 0.0,
            displacement: [0.0; 3],
            rotation: [1.0, 0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        };

        let mut keyframes = Vec::with_capacity(time_count as usize);

        for j in 0..time_count {
            let is_first = j == 0;
            let is_last = j == time_count - 1 && time_count > 1;

            if bs.bit_count() >= bs.total_bits() {
                warn!(
                    "    bitstream exhausted at track {} kf {} — stopping",
                    track_idx, j
                );
                break;
            }

            let mut kf = if is_first {
                // First KF: all raw via context8 (U3D SDK ReadF32X)
                let time = bs.read_f32_via_context8();
                let displacement = [
                    bs.read_f32_via_context8(),
                    bs.read_f32_via_context8(),
                    bs.read_f32_via_context8(),
                ];
                let rotation = [
                    bs.read_f32_via_context8(),
                    bs.read_f32_via_context8(),
                    bs.read_f32_via_context8(),
                    bs.read_f32_via_context8(),
                ];
                let sx = bs.read_f32_via_context8();
                let sy = bs.read_f32_via_context8();
                let sz = bs.read_f32_via_context8();
                KeyFrame {
                    time: predicted.time + time,
                    displacement,
                    rotation,
                    scale: [
                        predicted.scale[0] + sx,
                        predicted.scale[1] + sy,
                        predicted.scale[2] + sz,
                    ],
                }
            } else if is_last {
                // Last KF: raw time via context8, compressed disp/rot/scale
                let diff_time = bs.read_f32_via_context8();
                let time = predicted.time + diff_time;

                let sign = bs.read_compressed_u8(CTX_DISP_SIGN);
                let dx = bs.read_compressed_u32(CTX_DISP_DIFF);
                let dy = bs.read_compressed_u32(CTX_DISP_DIFF);
                let dz = bs.read_compressed_u32(CTX_DISP_DIFF);
                let displacement = [
                    inverse_quant(
                        predicted.displacement[0],
                        (sign & 1) != 0,
                        dx,
                        disp_inv_quant,
                    ),
                    inverse_quant(
                        predicted.displacement[1],
                        (sign & 2) != 0,
                        dy,
                        disp_inv_quant,
                    ),
                    inverse_quant(
                        predicted.displacement[2],
                        (sign & 4) != 0,
                        dz,
                        disp_inv_quant,
                    ),
                ];

                let sign = bs.read_compressed_u8(CTX_ROT_SIGN);
                let ub = bs.read_compressed_u32(CTX_ROT_DIFF);
                let uc = bs.read_compressed_u32(CTX_ROT_DIFF);
                let ud = bs.read_compressed_u32(CTX_ROT_DIFF);
                let mut fb = rot_inv_quant * ub as f32;
                let mut fc = rot_inv_quant * uc as f32;
                let mut fd = rot_inv_quant * ud as f32;
                if fb > 1.0 {
                    fb = 1.0;
                }
                if fc > 1.0 {
                    fc = 1.0;
                }
                if fd > 1.0 {
                    fd = 1.0;
                }
                let fa_sq =
                    1.0 - (fb as f64 * fb as f64 + fc as f64 * fc as f64 + fd as f64 * fd as f64);
                let mut fa = if fa_sq > 0.0 {
                    fa_sq.sqrt() as f32
                } else {
                    0.0
                };
                if (sign & 1) != 0 {
                    fa = -fa;
                }
                if (sign & 2) != 0 {
                    fb = -fb;
                }
                if (sign & 4) != 0 {
                    fc = -fc;
                }
                if (sign & 8) != 0 {
                    fd = -fd;
                }
                let [pw, pi, pj, pk] = predicted.rotation;
                let rw = pw * fa - pi * fb - pj * fc - pk * fd;
                let ri = pw * fb + pi * fa + pj * fd - pk * fc;
                let rj = pw * fc - pi * fd + pj * fa + pk * fb;
                let rk = pw * fd + pi * fc - pj * fb + pk * fa;
                let mag = (rw * rw + ri * ri + rj * rj + rk * rk).sqrt();
                let rotation = if mag > 1e-10 {
                    [rw / mag, ri / mag, rj / mag, rk / mag]
                } else {
                    [1.0, 0.0, 0.0, 0.0]
                };

                let sign = bs.read_compressed_u8(CTX_SCALE_SIGN);
                let sx = bs.read_compressed_u32(CTX_SCALE_DIFF);
                let sy = bs.read_compressed_u32(CTX_SCALE_DIFF);
                let sz = bs.read_compressed_u32(CTX_SCALE_DIFF);
                let scale = [
                    inverse_quant(predicted.scale[0], (sign & 1) != 0, sx, scale_inv_quant),
                    inverse_quant(predicted.scale[1], (sign & 2) != 0, sy, scale_inv_quant),
                    inverse_quant(predicted.scale[2], (sign & 4) != 0, sz, scale_inv_quant),
                ];
                KeyFrame {
                    time,
                    displacement,
                    rotation,
                    scale,
                }
            } else {
                // Differential keyframes: fully compressed
                read_compressed_kf(
                    &mut bs,
                    &predicted,
                    time_inv_quant,
                    rot_inv_quant,
                    disp_inv_quant,
                    scale_inv_quant,
                )
            };

            if (track_idx < 5 || track_idx == track_count - 1) && (j < 3 || j == time_count - 1) {
                let qmag = (kf.rotation[0].powi(2)
                    + kf.rotation[1].powi(2)
                    + kf.rotation[2].powi(2)
                    + kf.rotation[3].powi(2))
                .sqrt();
                debug!(
                    "    kf[{}] @bit{}: t={:.4} d=({:.4},{:.4},{:.4}) r=({:.4},{:.4},{:.4},{:.4}) |q|={:.4} s=({:.4},{:.4},{:.4})",
                    j,
                    bs.bit_count(),
                    kf.time,
                    kf.displacement[0],
                    kf.displacement[1],
                    kf.displacement[2],
                    kf.rotation[0],
                    kf.rotation[1],
                    kf.rotation[2],
                    kf.rotation[3],
                    qmag,
                    kf.scale[0],
                    kf.scale[1],
                    kf.scale[2],
                );
            }

            // Quaternion sign neighborhood: q and -q represent the same rotation, but
            // slerp(q_prev, q_i) takes the long arc when dot < 0. Flip when needed so
            // runtime interpolation picks the short arc.
            let dot = predicted.rotation[0] * kf.rotation[0]
                + predicted.rotation[1] * kf.rotation[1]
                + predicted.rotation[2] * kf.rotation[2]
                + predicted.rotation[3] * kf.rotation[3];
            if dot < 0.0 {
                kf.rotation[0] = -kf.rotation[0];
                kf.rotation[1] = -kf.rotation[1];
                kf.rotation[2] = -kf.rotation[2];
                kf.rotation[3] = -kf.rotation[3];
            }

            predicted = kf.clone();
            keyframes.push(kf);
        }

        if track_idx < 5 || track_idx == track_count - 1 {
            debug!(
                "  Track {} end @bit{} underflow={}",
                track_idx,
                bs.bit_count(),
                bs.ac_underflow(),
            );
        }

        tracks.push(DecodedTrack {
            name: track_name,
            keyframes,
        });
    }

    info!(
        "  Bits consumed: {} / {} total ({:.1}%)",
        bs.bit_count(),
        bs.total_bits(),
        bs.bit_count() as f64 / bs.total_bits() as f64 * 100.0
    );

    DecodedMotion {
        name: motion.name.clone(),
        track_count: motion.track_count,
        time_inv_quant: motion.time_inv_quant,
        rotation_inv_quant: motion.rotation_inv_quant,
        tracks,
    }
}
