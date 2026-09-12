//! Faithful port of Intel U3D `IFXSkin` automatic bone-weight generation —
//! the **fallback** weight source.
//!
//! Every skinned XMED actually carries authored per-vertex weights in its 0x49
//! geometry stream: per position a count (arithmetic context 0x12), that many
//! bone ids (0x13) and a quantised weight (0x14, `w = raw / 32768`) for each
//! entry after the first, the first taking `1 - sum(rest)` per the U3D "last
//! weight is not written" rule. `geometry.rs` decodes those and the exporter
//! binds with them. This module's `regenerate -> joint-cross-section -> smooth
//! -> remove-rogue` pipeline runs only for skins whose authored weights were
//! not decoded.
//!
//! XMED 0x4B skeleton chunks carry NO per-vertex weights — the Shockwave-3D
//! runtime regenerates them from mesh topology + skeleton via the IFXSkin
//! algorithm (`tools/u3d-sdk/RTL/Component/Bones/IFXSkin.cpp`). This module
//! reproduces that pipeline so the baked GLB has anatomically-correct weights
//! instead of the previous hand-tuned inverse-square + body-zone heuristic.
//!
//! Call sequence (matches `IFXMeshGroup_Character::CleanupWeights`, the full
//! regenerate + remove-rogue + smooth path):
//!   1. RegenerateWeights(use_joints=false)  — proximity pick, one bone/vertex
//!   2. CalculateJointCrossSections(base)     — crude elliptical joint bounds
//!   3. SmoothWeights(3)                       — diffuse across neighbours
//!   4. CalculateJointCrossSections(base+tip)  — approximate joint bounds
//!   5. RegenerateWeights(use_joints=true)     — re-pick with joint scaling
//!   6. RemoveRogueWeights()                   — flood-fill disjoint islands
//!   7. CalculateJointCrossSections(base)      — final joint bounds
//!   8. SmoothWeights(10)                       — final diffuse
//!
//! All geometry is in the model's "bone space"; bone start/orientation come
//! from the same skeleton walk as `bone_weights::compute_bone_world_positions`.

use crate::chunks::BoneDef;

// ── Tunable constants (IFXSkin defaults / empirically validated) ───────────
const SMOOTH_ITERATIONS_REGEN: i32 = 3;
const SMOOTH_ITERATIONS_FINAL: i32 = 10;
/// `maxratio` in SmoothWeights: max weight delta per (distance/jointsize).
const SMOOTH_THRESHOLD: f32 = 0.5;
/// `weldmax` as a fraction of model size: below this, neighbours smooth fully.
const SMOOTH_WELDMAX: f32 = 0.0;

// IFXSkin joint clamps.
const JOINT_MAXRAD: f32 = 5.0;
const JOINT_MAXASPECT: f32 = 5.0;
const JOINT_MAXDISPLACE: f32 = 0.3;
const JOINT_MAXCHILDASPECT: f32 = 1.0;
const JOINT_MAXTIPCHANGE: f32 = 0.1;
const JOINT_MINTIPCHANGE: f32 = -0.5;
const REGEN_MAXASPECT: f32 = 2.0;

// ── Bone rest transform ────────────────────────────────────────────────────

/// A bone's rest-pose transform in bone space: base position + world
/// orientation (wxyz) + bone length. Mirrors `IFXBoneNode::StoredTransform`.
#[derive(Debug, Clone, Copy)]
struct BoneXform {
    /// Head of the bone in bone space (Director units, Z-up).
    start: [f32; 3],
    /// World orientation of the bone, wxyz.
    q: [f32; 4],
    /// Bone length along its own X axis.
    length: f32,
}

impl BoneXform {
    /// `ReverseTransformVector`: world point → bone-local offset.
    /// result[0] is the coordinate along the bone's X axis.
    #[inline]
    fn reverse(&self, v: &[f32; 3]) -> [f32; 3] {
        let d = [
            v[0] - self.start[0],
            v[1] - self.start[1],
            v[2] - self.start[2],
        ];
        let qc = [self.q[0], -self.q[1], -self.q[2], -self.q[3]];
        quat_rotate_vec_wxyz(qc, d)
    }
}

/// Walk the skeleton hierarchy building per-bone rest transforms.
/// Identical math to `bone_weights::compute_bone_world_positions`, additionally
/// exposing the accumulated world orientation.
fn bone_xforms(bones: &[BoneDef]) -> Vec<BoneXform> {
    let n = bones.len();
    let mut wpos = vec![[0.0f32; 3]; n];
    let mut wq = vec![[1.0f32, 0.0, 0.0, 0.0]; n];
    let mut out = vec![
        BoneXform {
            start: [0.0; 3],
            q: [1.0, 0.0, 0.0, 0.0],
            length: 0.0,
        };
        n
    ];

    for (i, bone) in bones.iter().enumerate() {
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
            wpos[i] = local_t;
            wq[i] = bone.orientation;
        } else {
            let p = bone.parent_index as usize;
            let rt = quat_rotate_vec_wxyz(wq[p], local_t);
            wpos[i] = [wpos[p][0] + rt[0], wpos[p][1] + rt[1], wpos[p][2] + rt[2]];
            wq[i] = quat_mul_wxyz(wq[p], bone.orientation);
        }
        out[i] = BoneXform {
            start: wpos[i],
            q: wq[i],
            length: bone.rest_length,
        };
    }
    out
}

// ── Joint cross-sections ───────────────────────────────────────────────────

/// Elliptical joint bounds for one bone: base (index 0) + tip (index 1).
/// `scale[k][1]`, `scale[k][2]` are the y/z ellipse radii; `center[k]` the
/// offset of the ellipse centre from the bone axis.
#[derive(Debug, Clone, Copy)]
struct JointBounds {
    center: [[f32; 3]; 2],
    scale: [[f32; 3]; 2],
}

impl JointBounds {
    fn empty() -> Self {
        JointBounds {
            center: [[0.0; 3]; 2],
            scale: [[0.0; 3]; 2],
        }
    }
    #[inline]
    fn base_size(&self) -> f32 {
        0.5 * (self.scale[0][1] + self.scale[0][2])
    }
    #[inline]
    fn tip_size(&self) -> f32 {
        0.5 * (self.scale[1][1] + self.scale[1][2])
    }
}

/// One vertex-weight entry (a vertex may have several, one per influencing bone).
#[derive(Debug, Clone, Copy)]
struct Vw {
    vert: usize,
    bone: usize,
    weight: f32,
    score: f32,
}

struct SkinCtx<'a> {
    positions: &'a [[f32; 3]],
    xforms: &'a [BoneXform],
    bones: &'a [BoneDef],
    real_bones: &'a [usize],
    /// bone index → child bone indices
    children: Vec<Vec<usize>>,
    /// per-vertex topological neighbours (face-mate + edge-opposite + replicant)
    neighbors: Vec<Vec<usize>>,
    influential: Vec<bool>,
    model_size: f32,
    // live weight state
    weights: Vec<Vw>,
    vmap: Vec<Vec<usize>>, // vert → indices into weights
    // per-bone best (lowest-score) vertex, for rogue seeds
    best_score: Vec<f32>,
    best_vert: Vec<i32>,
}

/// Generate per-vertex bone weights via the IFXSkin pipeline.
///
/// Fallback only: used for skins whose authored 0x49 weights were not decoded.
/// `positions` and the bone transforms must share one coordinate space.
/// Returns, per input vertex, a normalized list of `(bone_index, weight)`.
pub(crate) fn generate_weights(
    positions: &[[f32; 3]],
    faces: &[[usize; 3]],
    bones: &[BoneDef],
    real_bones: &[usize],
) -> Vec<Vec<(usize, f32)>> {
    let nverts = positions.len();
    if nverts == 0 || real_bones.is_empty() {
        return vec![vec![]; nverts];
    }

    let xforms = bone_xforms(bones);

    // bone → children (real bones only)
    let mut children = vec![Vec::new(); bones.len()];
    for &b in real_bones {
        let p = bones[b].parent_index;
        if p >= 0 {
            children[p as usize].push(b);
        }
    }

    let model_size = model_diagonal(positions).max(1e-3);
    let neighbors = build_neighbors(nverts, faces, positions, SMOOTH_WELDMAX * model_size);

    let mut ctx = SkinCtx {
        positions,
        xforms: &xforms,
        bones,
        real_bones,
        children,
        neighbors,
        influential: vec![true; bones.len()],
        model_size,
        weights: Vec::with_capacity(nverts),
        vmap: vec![Vec::new(); nverts],
        best_score: vec![-1.0; bones.len()],
        best_vert: vec![-1; bones.len()],
    };

    // 1. crude proximity pick (no joints)
    let empty_joints = vec![JointBounds::empty(); bones.len()];
    ctx.regenerate_weights(false, &empty_joints);
    // influence set is fixed from the first proximity pass
    ctx.influential = (0..bones.len()).map(|b| ctx.best_score[b] >= 0.0).collect();

    // 2. crude joints (base only) + 3. smooth
    let joints = ctx.calculate_joints(false);
    ctx.smooth_weights(SMOOTH_ITERATIONS_REGEN, &joints);

    // 4. approximate joints (base+tip) + 5. re-pick with joints
    let joints = ctx.calculate_joints(true);
    ctx.regenerate_weights(true, &joints);

    // 6. remove rogue islands
    ctx.remove_rogue_weights();

    // 7. final joints (base) + 8. final smooth
    let joints = ctx.calculate_joints(false);
    ctx.smooth_weights(SMOOTH_ITERATIONS_FINAL, &joints);

    // Collect normalized per-vertex weights.
    let mut out = vec![Vec::new(); nverts];
    for vi in 0..nverts {
        let mut sum = 0.0f32;
        for &wi in &ctx.vmap[vi] {
            sum += ctx.weights[wi].weight.max(0.0);
        }
        if sum <= 0.0 {
            continue;
        }
        for &wi in &ctx.vmap[vi] {
            let w = ctx.weights[wi].weight.max(0.0);
            if w > 0.0 {
                out[vi].push((ctx.weights[wi].bone, w / sum));
            }
        }
    }
    out
}

impl SkinCtx<'_> {
    /// The joint-scaled local distance from `vert` to `bone`.
    /// Returns `None` when the bone contributes no valid joint (jointsize<=0).
    fn scaled_dist(
        &self,
        vi: usize,
        bone: usize,
        use_joints: bool,
        jb: &JointBounds,
    ) -> Option<f32> {
        let x = &self.xforms[bone];
        let bonelength = x.length;
        let mut result = x.reverse(&self.positions[vi]);

        let (basejoint, tipjoint) = if use_joints {
            (jb.base_size(), jb.tip_size())
        } else {
            (1.0, 1.0)
        };

        let mut jointsize;
        if result[0] > 0.0 {
            if result[0] > bonelength && bonelength > 0.0 {
                result[0] -= bonelength;
                let mut js = 1.0 - result[0] / bonelength;
                if js < 0.1 {
                    js = 0.1;
                }
                jointsize = js * tipjoint;
            } else {
                jointsize = if bonelength > 0.0 {
                    basejoint + (tipjoint - basejoint) * result[0] / bonelength
                } else {
                    basejoint
                };
                result[0] = 0.0;
            }
        } else {
            let mut js = if bonelength > 0.0 {
                1.0 + result[0] / bonelength
            } else {
                1.0
            };
            if js < 0.1 {
                js = 0.1;
            }
            jointsize = js * basejoint;
        }

        if jointsize > bonelength * REGEN_MAXASPECT {
            jointsize = bonelength * REGEN_MAXASPECT;
        }
        if !use_joints {
            jointsize = 1.0;
        }
        if jointsize <= 0.0 {
            return None;
        }
        let inv = 1.0 / jointsize;
        let r = [result[0] * inv, result[1] * inv, result[2] * inv];
        Some((r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt())
    }

    /// RegenerateWeights: clear and assign each vertex to its best bone (w=1).
    fn regenerate_weights(&mut self, use_joints: bool, joints: &[JointBounds]) {
        self.weights.clear();
        for v in self.vmap.iter_mut() {
            v.clear();
        }
        for s in self.best_score.iter_mut() {
            *s = -1.0;
        }
        for b in self.best_vert.iter_mut() {
            *b = -1;
        }

        let nverts = self.positions.len();
        for vi in 0..nverts {
            let mut picked: i32 = -1;
            let mut pickdist = f32::MAX;
            // fallback (unscaled) pick, guarantees every vertex gets a bone
            let mut fb_bone: i32 = -1;
            let mut fb_dist = f32::MAX;

            for &b in self.real_bones {
                if use_joints && !self.influential[b] {
                    continue;
                }
                if let Some(d) = self.scaled_dist(vi, b, false, &joints[b]) {
                    if d < fb_dist {
                        fb_dist = d;
                        fb_bone = b as i32;
                    }
                }
                if let Some(d) = self.scaled_dist(vi, b, use_joints, &joints[b]) {
                    if picked < 0 || d < pickdist {
                        picked = b as i32;
                        pickdist = d;
                    }
                }
            }

            if picked < 0 {
                picked = fb_bone;
                pickdist = fb_dist;
            }
            if picked < 0 {
                continue;
            }
            let pb = picked as usize;
            if self.best_score[pb] < 0.0 || pickdist < self.best_score[pb] {
                self.best_score[pb] = pickdist;
                self.best_vert[pb] = vi as i32;
            }
            let wi = self.weights.len();
            self.weights.push(Vw {
                vert: vi,
                bone: pb,
                weight: 1.0,
                score: pickdist,
            });
            self.vmap[vi].push(wi);
        }
    }

    /// CalculateJointCrossSections: fit elliptical bounds per bone from the
    /// vertices currently weighted to the bone's family (self+parent+children).
    fn calculate_joints(&self, computetips: bool) -> Vec<JointBounds> {
        let mut out = vec![JointBounds::empty(); self.bones.len()];
        // parent-first (bone indices are topologically sorted).
        for &b in self.real_bones {
            let bonelength = self.xforms[b].length;

            // family set
            let mut family = vec![b];
            let p = self.bones[b].parent_index;
            if p >= 0 {
                family.push(p as usize);
            }
            for &c in &self.children[b] {
                family.push(c);
            }
            let has_children = !self.children[b].is_empty();

            // gather base offsets (local to THIS bone) for family-weighted verts
            let mut offsets: Vec<[f32; 3]> = Vec::new();
            let mut has_weights = false;
            for w in &self.weights {
                if family.contains(&w.bone) {
                    let off = self.xforms[b].reverse(&self.positions[w.vert]);
                    if w.bone == b {
                        has_weights = true;
                    }
                    offsets.push(off);
                }
            }
            if !has_weights {
                continue;
            }

            // base ellipse
            let base = fit_ellipse(
                &offsets,
                bonelength,
                false,
                false,
                &[0.0; 3],
                has_children,
                &[0.0; 3],
            );
            out[b].center[0] = base.0;
            out[b].scale[0] = base.1;

            if computetips {
                // tip offsets = base offsets shifted by -bonelength along axis
                let mut tip_offsets = offsets.clone();
                for o in tip_offsets.iter_mut() {
                    o[0] -= bonelength;
                }
                // check parent tip aspect
                let mut check_parent = false;
                let mut parentscale = [0.0f32; 3];
                if p >= 0 {
                    parentscale = out[p as usize].scale[1];
                    let mag = (parentscale[0] * parentscale[0]
                        + parentscale[1] * parentscale[1]
                        + parentscale[2] * parentscale[2])
                        .sqrt();
                    if mag > bonelength * 0.01 {
                        check_parent = true;
                    }
                }
                let tip = fit_ellipse(
                    &tip_offsets,
                    bonelength,
                    true,
                    check_parent,
                    &parentscale,
                    has_children,
                    &out[b].scale[0],
                );
                out[b].center[1] = tip.0;
                out[b].scale[1] = tip.1;
                out[b].center[1][0] = bonelength;
            }
        }
        out
    }

    /// SmoothWeights: iterative diffusion of weights across neighbours, with the
    /// permissible delta scaled by neighbour distance / joint size.
    fn smooth_weights(&mut self, iterations: i32, joints: &[JointBounds]) {
        if iterations <= 0 {
            return;
        }
        let maxratio = SMOOTH_THRESHOLD;
        let weldmax = SMOOTH_WELDMAX * self.model_size;
        let nverts = self.positions.len();

        for pass in 0..iterations {
            let lastpass = pass == iterations - 1;
            for vi in 0..nverts {
                // iterate a snapshot of this vertex's current weights
                let wlist = self.vmap[vi].clone();
                for &wi in &wlist {
                    let boneid = self.weights[wi].bone;
                    let jointsize = joints[boneid].base_size().max(1e-4);

                    let nbrs = self.neighbors[vi].clone();
                    for nb in nbrs {
                        let weight = self.weights[wi].weight;
                        // find neighbour's weight for the same bone
                        let mut found: i32 = -1;
                        for &nwi in &self.vmap[nb] {
                            if self.weights[nwi].bone == boneid {
                                found = nwi as i32;
                                break;
                            }
                        }
                        let neighweight = if found >= 0 {
                            self.weights[found as usize].weight
                        } else {
                            0.0
                        };
                        let delta = weight - neighweight;

                        let dv = [
                            self.positions[vi][0] - self.positions[nb][0],
                            self.positions[vi][1] - self.positions[nb][1],
                            self.positions[vi][2] - self.positions[nb][2],
                        ];
                        let mut distance = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
                        if distance < weldmax {
                            distance = 0.0;
                        } else if lastpass {
                            continue;
                        }

                        let maxdiff = distance * maxratio / jointsize;
                        let mut adjust = delta.abs() - maxdiff;
                        if adjust <= 0.0 {
                            continue;
                        }

                        let found = if found < 0 {
                            // create a zero-weight influence on the neighbour
                            let nwi = self.weights.len();
                            self.weights.push(Vw {
                                vert: nb,
                                bone: boneid,
                                weight: 0.0,
                                score: self.weights[wi].score,
                            });
                            self.vmap[nb].push(nwi);
                            nwi
                        } else {
                            found as usize
                        };

                        adjust *= 0.5;
                        if weight < neighweight {
                            adjust = -adjust;
                        }

                        if lastpass {
                            // equalize coincident (welded) neighbours deterministically
                            if vi < nb {
                                self.weights[found].weight = self.weights[wi].weight;
                            } else {
                                self.weights[wi].weight = self.weights[found].weight;
                            }
                        } else {
                            self.weights[wi].weight -= adjust;
                            self.weights[found].weight += adjust;
                        }
                    }
                }
            }

            // normalize per vertex
            for vi in 0..nverts {
                let mut sum = 0.0f32;
                for &wi in &self.vmap[vi] {
                    sum += self.weights[wi].weight;
                }
                if sum > 0.0 {
                    for &wi in &self.vmap[vi] {
                        self.weights[wi].weight /= sum;
                    }
                }
            }
        }
    }

    /// RemoveRogueWeights: flood-fill valid patches from each bone's best vertex,
    /// then reassign disjoint islands to a valid neighbour's bone.
    /// Operates on the single-bone assignment produced by RegenerateWeights.
    fn remove_rogue_weights(&mut self) {
        let nverts = self.positions.len();
        // primary bone per vertex (single-bone state)
        let mut assign: Vec<i32> = vec![-1; nverts];
        for vi in 0..nverts {
            if let Some(&wi) = self.vmap[vi].first() {
                assign[vi] = self.weights[wi].bone as i32;
            }
        }
        let mut valid = vec![false; nverts];

        // Pass 1: flood valid patches from each bone's best vertex.
        for &b in self.real_bones {
            let seed = self.best_vert[b];
            if seed < 0 {
                continue;
            }
            let seed = seed as usize;
            if assign[seed] != b as i32 || valid[seed] {
                continue;
            }
            self.flood(seed, b as i32, &assign, &mut valid);
        }

        // Pass 2: reassign invalid islands to a valid neighbour's bone.
        // Iterate to convergence (islands adjacent to newly-valid regions).
        let mut changed = true;
        while changed {
            changed = false;
            for vi in 0..nverts {
                if valid[vi] || assign[vi] < 0 {
                    continue;
                }
                let mut newbone = -1;
                for &nb in &self.neighbors[vi] {
                    if valid[nb] {
                        newbone = assign[nb];
                        break;
                    }
                }
                if newbone >= 0 {
                    // flood this vertex's old-bone island to `newbone`
                    let oldbone = assign[vi];
                    self.flood_reassign(vi, oldbone, newbone, &mut assign, &mut valid);
                    changed = true;
                }
            }
        }

        // Write reassignments back into the single-bone weight list.
        for vi in 0..nverts {
            if let Some(&wi) = self.vmap[vi].first() {
                if assign[vi] >= 0 {
                    self.weights[wi].bone = assign[vi] as usize;
                }
            }
        }
    }

    fn flood(&self, seed: usize, bone: i32, assign: &[i32], valid: &mut [bool]) {
        let mut stack = vec![seed];
        valid[seed] = true;
        while let Some(v) = stack.pop() {
            for &nb in &self.neighbors[v] {
                if !valid[nb] && assign[nb] == bone {
                    valid[nb] = true;
                    stack.push(nb);
                }
            }
        }
    }

    fn flood_reassign(
        &self,
        seed: usize,
        oldbone: i32,
        newbone: i32,
        assign: &mut [i32],
        valid: &mut [bool],
    ) {
        let mut stack = vec![seed];
        while let Some(v) = stack.pop() {
            if valid[v] || assign[v] != oldbone {
                continue;
            }
            assign[v] = newbone;
            valid[v] = true;
            for &nb in &self.neighbors[v] {
                if !valid[nb] && assign[nb] == oldbone {
                    stack.push(nb);
                }
            }
        }
    }
}

/// Fit an elliptical joint cross-section (port of `CalculateJointForBone`).
/// Returns `(center, scale)` where scale[1],scale[2] are the y/z radii.
fn fit_ellipse(
    offsets: &[[f32; 3]],
    bonelength: f32,
    tip: bool,
    check_parent: bool,
    parentscale: &[f32; 3],
    has_children: bool,
    basescale: &[f32; 3],
) -> ([f32; 3], [f32; 3]) {
    const PASSES: i32 = 16;
    const EXPANDER: f32 = 1.4;
    const UNITDELTAMAX: f32 = 0.25;
    const DELTAINFLUENCE: f32 = 0.5;
    const XFADE: f32 = 0.5;

    let mut center = [0.0f32; 3];
    let mut scale = [0.0f32, 0.1, 0.1];

    let n = offsets.len();
    if n == 0 {
        return (center, [0.0; 3]);
    }

    // crude scale = min radial distance (tightest vertex)
    let mut crudescale = 1e6f32;
    for o in offsets {
        let x = o[0].abs();
        let radial = (o[1] * o[1] + o[2] * o[2]).sqrt();
        let radius = radial + x;
        if crudescale > radius {
            crudescale = radius;
        }
    }
    let deltamax = UNITDELTAMAX * crudescale;

    if n < 3 {
        return (center, scale);
    }

    scale[1] = crudescale;
    scale[2] = crudescale;

    for pass in 0..PASSES {
        let crude2 = DELTAINFLUENCE * crudescale * ((PASSES - pass) as f32 / PASSES as f32);
        scale[1] *= EXPANDER;
        scale[2] *= EXPANDER;

        let mut delta = [0.0f32; 2];

        for cycle in 0..2 {
            for o in offsets {
                let mut radial = [o[0] - center[0], o[1] - center[1], o[2] - center[2]];
                let x = radial[0].abs() * XFADE;
                radial[0] = 0.0;

                let angle = radial[2].atan2(radial[1]);
                let f = [angle.cos().abs() + 0.001, angle.sin().abs() + 0.001];

                for k in 0..2 {
                    let axis = (radial[k + 1].abs() + x) / f[k];
                    if scale[k + 1] > axis {
                        if cycle == 1 {
                            scale[k + 1] = axis;
                        } else {
                            let change = crude2 * (1.0 - axis / scale[k + 1]);
                            if change > 0.0 {
                                delta[k] += if o[k + 1] > 0.0 { -change } else { change };
                            }
                        }
                    }
                }
            }

            if cycle == 0 {
                for k in 0..2 {
                    if delta[k].abs() > deltamax {
                        delta[k] = if delta[k] > 0.0 { deltamax } else { -deltamax };
                    }
                    center[k + 1] += delta[k];
                }
            }

            for j in 1..3 {
                if scale[j] > bonelength * JOINT_MAXRAD {
                    scale[j] = bonelength * JOINT_MAXRAD;
                }
                if scale[j] > scale[3 - j] * JOINT_MAXASPECT {
                    scale[j] = scale[3 - j] * JOINT_MAXASPECT;
                }
                if center[j].abs() > scale[j] * JOINT_MAXDISPLACE {
                    center[j] *= scale[j] * JOINT_MAXDISPLACE / center[j].abs();
                }

                if tip {
                    let maxdiff = JOINT_MAXTIPCHANGE * bonelength;
                    let mindiff = JOINT_MINTIPCHANGE * bonelength;
                    let mut diff = scale[j] - basescale[j];
                    if diff < mindiff {
                        diff = mindiff;
                    } else if !has_children && diff > 0.0 {
                        diff = 0.0;
                    } else if diff > maxdiff {
                        diff = maxdiff;
                    }
                    scale[j] = basescale[j] + diff;
                } else if check_parent && scale[j] > parentscale[j] * JOINT_MAXCHILDASPECT {
                    scale[j] = parentscale[j] * JOINT_MAXCHILDASPECT;
                }
            }
        }
    }

    (center, scale)
}

// ── Mesh topology ──────────────────────────────────────────────────────────

/// Build per-vertex neighbours: face-mates (shared triangle), edge-opposite
/// (vertices facing the same edge), and replicants (coincident positions).
/// Mirrors `IFXSkin::FindNeighbors`.
fn build_neighbors(
    nverts: usize,
    faces: &[[usize; 3]],
    positions: &[[f32; 3]],
    weld: f32,
) -> Vec<Vec<usize>> {
    let mut neigh: Vec<Vec<usize>> = vec![Vec::new(); nverts];
    let add = |a: usize, b: usize, neigh: &mut Vec<Vec<usize>>| {
        if a != b && a < nverts && b < nverts && !neigh[a].contains(&b) {
            neigh[a].push(b);
        }
    };

    // edge → third vertex (for edge-opposite mirroring)
    use std::collections::HashMap;
    let mut edge_third: HashMap<(usize, usize), usize> = HashMap::new();

    for tri in faces {
        for m in 0..3 {
            let v = tri[m];
            let v2 = tri[(m + 1) % 3];
            let v3 = tri[(m + 2) % 3];
            // face-mate mirror
            add(v, v2, &mut neigh);
            add(v2, v, &mut neigh);

            let (lo, hi) = if v < v2 { (v, v2) } else { (v2, v) };
            if let Some(&v4) = edge_third.get(&(lo, hi)) {
                // both triangles on this edge: mirror the opposing verts
                add(v3, v4, &mut neigh);
                add(v4, v3, &mut neigh);
            } else {
                edge_third.insert((lo, hi), v3);
            }
        }
    }

    // replicants: vertices at (near-)coincident positions
    if nverts > 0 {
        let weld2 = (weld.max(1e-6)) * (weld.max(1e-6));
        // bucket by quantized position to avoid O(n^2)
        let q = 1.0 / (weld.max(1e-4));
        let mut buckets: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (i, p) in positions.iter().enumerate() {
            let key = (
                (p[0] * q).round() as i64,
                (p[1] * q).round() as i64,
                (p[2] * q).round() as i64,
            );
            buckets.entry(key).or_default().push(i);
        }
        for group in buckets.values() {
            for a_i in 0..group.len() {
                for b_i in (a_i + 1)..group.len() {
                    let (a, b) = (group[a_i], group[b_i]);
                    let d = [
                        positions[a][0] - positions[b][0],
                        positions[a][1] - positions[b][1],
                        positions[a][2] - positions[b][2],
                    ];
                    if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= weld2 {
                        add(a, b, &mut neigh);
                        add(b, a, &mut neigh);
                    }
                }
            }
        }
    }

    neigh
}

fn model_diagonal(positions: &[[f32; 3]]) -> f32 {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for p in positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let d = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

// ── Quaternion helpers (wxyz; identical convention to bone_weights) ─────────

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

fn quat_rotate_vec_wxyz(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let qv = [0.0, v[0], v[1], v[2]];
    let qc = [q[0], -q[1], -q[2], -q[3]];
    let r = quat_mul_wxyz(quat_mul_wxyz(q, qv), qc);
    [r[1], r[2], r[3]]
}
