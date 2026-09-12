//! Golden-file test for the native XMED 0x49 progressive-CLOD geometry decoder
//! (`geometry::decode_geometry`, a port of the original Shockwave 3D Asset.x32
//! `FUN_7a18a990` = U3D CIFXAuthorCLODDecoder_P).
//!
//! Validates the decoded POSITIONS of arrow.xmed against its Wine-converter OBJ
//! ground truth (`assets/models/obj/arrow/arrow.obj`). Face connectivity has its
//! own golden file, `arrow_faces_byte_exact.rs`.
//!
//! State (2026-09-02, plan step 3.1): all 18 positions decode byte-exact and in
//! OBJ vertex order, and all 22 faces match the OBJ index-for-index — the
//! decoder now maintains the live author-CLOD face array
//! (`macromelt::clod_state::LiveMesh`) and applies the vertex-split corner
//! rewrites, so there is no residual connectivity gap. Every geometry read
//! agrees with the binary's own trace in type and context
//! (`tools/re/traces/arrow_full_trace.log`, checked by
//! `tools/re/traces/diff_facearray.py`).

use macromelt::chunks::parse_xmed;
use macromelt::geometry::decode_geometry;

mod common;

const EPS: f32 = 1e-3;

fn approx(a: [f32; 3], b: [f32; 3]) -> bool {
    (a[0] - b[0]).abs() < EPS && (a[1] - b[1]).abs() < EPS && (a[2] - b[2]).abs() < EPS
}

/// The confirmed byte-exact prefix: the first resolution-update record decodes
/// 6 vertex slots (3 distinct model-space positions, each appearing twice) and 2
/// triangles referencing slots 0..5.
#[test]
fn arrow_first_record_is_byte_exact() {
    let Some(path) = common::fixture("extracted/assets/Hey/arrow.xmed") else {
        return;
    };
    let data = std::fs::read(&path).expect("read arrow.xmed");
    let xmed = parse_xmed(&data).expect("parse arrow.xmed");
    let g = xmed.geometry.first().expect("arrow has a geometry chunk");
    let md = xmed.mesh_descriptions.iter().find(|m| m.name == g.name);

    // max-res positions for arrow == 13 (from the 0x45 chunk).
    assert_eq!(g.num_positions, 13, "arrow max-res position count");

    let decoded = decode_geometry(g, md, false);

    // The first resolution-update record decodes 6 vertex slots; the whole mesh
    // (18 slots) is checked by `arrow_full_position_decode_byte_exact`.
    assert!(
        decoded.positions.len() >= 6,
        "at least the first record's 6 vertex slots decoded (got {})",
        decoded.positions.len()
    );

    // First record: 3 distinct model-space positions (verified bit-exact).
    let a = [-0.43100, -0.45811, 0.08471];
    let b = [0.46336, 0.46089, 0.07683];
    let c = [-1.11260, 1.10268, 2.48078];
    assert!(
        approx(decoded.positions[0], a),
        "pos0 {:?}",
        decoded.positions[0]
    );
    assert!(
        approx(decoded.positions[1], b),
        "pos1 {:?}",
        decoded.positions[1]
    );
    assert!(
        approx(decoded.positions[2], c),
        "pos2 {:?}",
        decoded.positions[2]
    );
    assert!(
        approx(decoded.positions[3], a),
        "pos3 {:?}",
        decoded.positions[3]
    );
    assert!(
        approx(decoded.positions[4], c),
        "pos4 {:?}",
        decoded.positions[4]
    );
    assert!(
        approx(decoded.positions[5], b),
        "pos5 {:?}",
        decoded.positions[5]
    );

    // Face connectivity: the live face array (`clod_state::LiveMesh`) has the
    // group_a corner rewrites applied, so faces[0]/[1] hold the FINAL render
    // corners, not the pre-mutation base triangles [0,1,2]/[3,4,5] that group_b
    // appended. Full connectivity is locked in `arrow_faces_byte_exact.rs`.
    assert!(decoded.faces.len() >= 2, "at least 2 faces decoded");
    assert_eq!(decoded.faces[0], [6, 5, 10], "face0 render mesh");
    assert_eq!(decoded.faces[1], [3, 4, 16], "face1 render mesh");
}

/// Full-mesh decode validated against the Wine-converter OBJ ground truth
/// (`assets/models/obj/arrow/arrow.obj`, world→model space: ×2.610, z+30.313).
///
/// The decoder reproduces the binary's resolution-update records byte-exact (ctx=8
/// magnitudes + ctx=7 signs match `tools/re/traces/arrow_full_trace.log`). Two
/// findings pinned the full mesh this session:
///   1. `preds` holds FACE indices; a split vertex's base is
///      `positions[faces[preds[split]].corner[predType]]` (predType 0/1/2 picks the
///      corner), and group_b corner case 2/3/4 = `faces[preds[split]].corner[ptype-2]
///      + delta`.
///   2. The mesh has **18 vertices**, NOT the 0x49 header's `num_positions`
///      (=13, the distinct/base count). The 0x45 MeshDescription declares the
///      real 18/22 counts and the update loop runs until it reaches them —
///      records 8..12 add OBJ verts 10/11/12 (the y-mirror side).
/// With both, all 18 decoded positions match the 18 OBJ vertices IN ORDER, and 22
/// position faces are emitted (matching the OBJ face count).
///
/// Face *connectivity* is byte-exact as of plan step 3.1 and is locked
/// index-for-index in `arrow_faces_byte_exact.rs`; the canonical-set check below
/// stays as a cheap 22/22 regression guard.
#[test]
fn arrow_full_position_decode_byte_exact() {
    let Some(path) = common::fixture("extracted/assets/Hey/arrow.xmed") else {
        return;
    };
    let data = std::fs::read(&path).expect("read arrow.xmed");
    let xmed = parse_xmed(&data).expect("parse arrow.xmed");
    let g = xmed.geometry.first().expect("geometry");
    let md = xmed.mesh_descriptions.iter().find(|m| m.name == g.name);
    let decoded = decode_geometry(g, md, false);

    // Load OBJ ground-truth vertices, transform world→model space.
    let Some(obj_path) = common::fixture("assets/models/obj/arrow/arrow.obj") else {
        return;
    };
    const S: f32 = 2.610;
    const ZT: f32 = 30.313;
    let obj_verts: Vec<[f32; 3]> = std::fs::read_to_string(&obj_path)
        .expect("read arrow.obj")
        .lines()
        .filter(|l| l.starts_with("v "))
        .map(|l| {
            let n: Vec<f32> = l
                .split_whitespace()
                .skip(1)
                .take(3)
                .map(|x| x.parse().unwrap())
                .collect();
            [n[0] / S, n[1] / S, (n[2] - ZT) / S]
        })
        .collect();

    // The full arrow mesh is 18 vertices (num_positions=13 is the distinct count).
    assert_eq!(obj_verts.len(), 18, "arrow.obj has 18 vertices");
    assert_eq!(
        decoded.positions.len(),
        18,
        "all 18 arrow vertices decode (got {})",
        decoded.positions.len()
    );

    // Decoded vertices match the OBJ vertices IN ORDER, byte-exact (quant tolerance).
    for (i, (got, want)) in decoded.positions.iter().zip(obj_verts.iter()).enumerate() {
        assert!(
            approx(*got, *want),
            "vertex {} = {:?}, expected OBJ {:?}",
            i,
            got,
            want
        );
    }

    // 13 distinct model-space positions (the OBJ's distinct count).
    let mut uniq: Vec<[f32; 3]> = Vec::new();
    for p in &decoded.positions {
        if !uniq.iter().any(|u| approx(*u, *p)) {
            uniq.push(*p);
        }
    }
    assert_eq!(uniq.len(), 13, "13 distinct model-space positions");

    // 22 position faces emitted (matching the OBJ face count).
    assert_eq!(decoded.faces.len(), 22, "22 position faces decoded");

    // Face connectivity vs OBJ ground truth. The live face array (group_a corner
    // rewrites applied on top of the group_b appends) reconstructs the render
    // mesh; compare as canonical position-face sets (decoder vertex i ≡ OBJ
    // vertex i — the in-order position match above — collapsed by coincident
    // position, since the OBJ splits verts on UV/normal seams). 22/22.
    let mut cid = vec![0usize; decoded.positions.len()];
    let mut canon: Vec<[f32; 3]> = Vec::new();
    for (i, p) in decoded.positions.iter().enumerate() {
        match canon.iter().position(|u| approx(*u, *p)) {
            Some(k) => cid[i] = k,
            None => {
                cid[i] = canon.len();
                canon.push(*p);
            }
        }
    }
    let norm = |f: &[u32; 3]| {
        let mut v = [cid[f[0] as usize], cid[f[1] as usize], cid[f[2] as usize]];
        v.sort_unstable();
        v
    };
    let obj_faces: std::collections::HashSet<[usize; 3]> = std::fs::read_to_string(&obj_path)
        .expect("read arrow.obj")
        .lines()
        .filter(|l| l.starts_with("f "))
        .map(|l| {
            let idx: Vec<u32> = l
                .split_whitespace()
                .skip(1)
                .take(3)
                .map(|t| t.split('/').next().unwrap().parse::<u32>().unwrap() - 1)
                .collect();
            norm(&[idx[0], idx[1], idx[2]])
        })
        .collect();
    let dec_faces: std::collections::HashSet<[usize; 3]> =
        decoded.faces.iter().map(|f| norm(f)).collect();
    let overlap = dec_faces.intersection(&obj_faces).count();
    assert_eq!(
        overlap, 22,
        "native render faces match arrow.obj in canonical position space"
    );
}
