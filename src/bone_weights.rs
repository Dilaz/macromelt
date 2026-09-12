//! Skeleton walk: world-space bone segments for XMED models.
//!
//! Computes world-space bone positions by walking the skeleton hierarchy,
//! using bone length + orientation to find each bone's tail (end) position.
//! Per ECMA-363 9.6.1.1.6.6: "Bone Displacement is the displacement of the
//! start of this bone from the END of its parent bone."
//!
//! Skin weights themselves are *not* computed here: every skinned XMED carries
//! authored per-vertex weights in its 0x49 geometry stream (per position a
//! count in arithmetic context 0x12, that many bone ids in 0x13, and a
//! quantised weight in 0x14 with `w = raw / 32768` for every entry after the
//! first, the first taking `1 - sum(rest)` per the U3D "last weight is not
//! written" rule) and `geometry.rs` decodes them. The segments produced here
//! serve the proximity queries of the [`crate::skin_weights`] IFXSkin fallback
//! and of the exporter's rogue-weight checks.

use crate::chunks::BoneDef;

/// Bone start and end positions in world space, in Director units and Z-up.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BoneWorldPos {
    /// Head of the bone: the world position of its node.
    pub(crate) start: [f32; 3],
    /// Tail of the bone: `start` displaced by `rest_length` along the bone's
    /// own X axis.
    pub(crate) end: [f32; 3],
}

/// Compute world-space bone start/end positions by walking the parent chain.
///
/// This MUST match the same math as gltf_export.rs's compute_bone_world_transforms:
///   local_translation = (displacement[0] + parent_rest_length, displacement[1], displacement[2])
///   local_rotation = bone.orientation (wxyz)
///   world = parent_world * local
///
/// The bone START is the world position of the node.
/// The bone END = START + rotate(rest_length along local X, by world_orientation).
pub(crate) fn compute_bone_world_positions(bones: &[BoneDef]) -> Vec<BoneWorldPos> {
    let mut result = vec![
        BoneWorldPos {
            start: [0.0; 3],
            end: [0.0; 3]
        };
        bones.len()
    ];
    let mut world_positions: Vec<[f32; 3]> = vec![[0.0; 3]; bones.len()];
    let mut world_orientations: Vec<[f32; 4]> = vec![[1.0, 0.0, 0.0, 0.0]; bones.len()];

    for (i, bone) in bones.iter().enumerate() {
        // Local translation: displacement from parent END = displacement + parent_length along X
        let parent_length = if bone.parent_index >= 0 {
            bones[bone.parent_index as usize].rest_length
        } else {
            0.0
        };
        let local_t = [
            bone.displacement[0] + parent_length,
            bone.displacement[1],
            bone.displacement[2],
        ];

        if bone.parent_index < 0 {
            // Root bone: world = local
            world_positions[i] = local_t;
            world_orientations[i] = bone.orientation;
        } else {
            let p = bone.parent_index as usize;
            // Rotate local translation by parent's world orientation
            let rotated_t = quat_rotate_vec_wxyz(world_orientations[p], local_t);
            world_positions[i] = [
                world_positions[p][0] + rotated_t[0],
                world_positions[p][1] + rotated_t[1],
                world_positions[p][2] + rotated_t[2],
            ];
            // World orientation = parent * local
            world_orientations[i] = quat_mul_wxyz(world_orientations[p], bone.orientation);
        }

        let start = world_positions[i];
        let dir = quat_rotate_vec_wxyz(world_orientations[i], [bone.rest_length, 0.0, 0.0]);
        let end = [start[0] + dir[0], start[1] + dir[1], start[2] + dir[2]];
        result[i] = BoneWorldPos { start, end };
    }

    result
}

/// Quaternion multiply (both in wxyz format).
fn quat_mul_wxyz(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

/// Rotate a vector by a quaternion (wxyz format).
fn quat_rotate_vec_wxyz(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let qv = [0.0, v[0], v[1], v[2]]; // pure quaternion
    let qc = [q[0], -q[1], -q[2], -q[3]]; // conjugate
    let result = quat_mul_wxyz(quat_mul_wxyz(q, qv), qc);
    [result[1], result[2], result[3]]
}

/// Distance from a point to the bone segment `a`–`b` (crate-visible wrapper
/// used from `gltf_export`).
pub(crate) fn point_to_segment_dist_pub(p: &[f32; 3], a: &[f32; 3], b: &[f32; 3]) -> f32 {
    point_to_segment_dist(p, a, b)
}

/// Distance from point to line segment.
fn point_to_segment_dist(p: &[f32; 3], a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let ab_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];

    if ab_sq < 0.001 {
        return (ap[0] * ap[0] + ap[1] * ap[1] + ap[2] * ap[2]).sqrt();
    }

    let t = ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / ab_sq).clamp(0.0, 1.0);
    let closest = [a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]];
    let dx = p[0] - closest[0];
    let dy = p[1] - closest[1];
    let dz = p[2] - closest[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}
