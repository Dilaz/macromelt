//! Face-corner rewrites are revealed by the CLOD manager on the GLOBAL
//! resolution clock, two steps after the record's 0x47 schedule entry
//! (`CIFXCLODManager::IncreaseTo`: `r > synchTable[i]`, called with the
//! pre-increment counter) - not two group-local records later.
//!
//! `TitleMeny/bakgrunn.xmed`'s `klippe` (3 shading groups, schedule gaps) is the
//! smallest ground truth: with the record rule, group 0's record 65 (step 276)
//! read face 50 corner 2 before record 64's (step 270) rewrite `25 -> 105` had
//! landed, built face 121 as `(51,106,25)` instead of `(51,106,105)`, and record
//! 83 then predicted position #132 off the wrong corner - exactly
//! `pos[25] - pos[105]` away from the converter's vertex. The converter OBJ
//! writes vertices in decode order with the 0x72 node matrix applied, so the
//! comparison is index-for-index.

use std::path::Path;

use macromelt::chunks::parse_xmed;
use macromelt::clod_state::LiveMesh;
use macromelt::geometry::decode_mesh_group;

mod common;

/// `v` lines of one `g <name>` block, in file order.
fn obj_group_positions(path: &Path, group: &str) -> Vec<[f32; 3]> {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    let mut inside = false;
    let mut out = Vec::new();
    for line in text.lines() {
        if let Some(name) = line.strip_prefix("g ") {
            inside = name.trim() == group;
        } else if inside && let Some(rest) = line.strip_prefix("v ") {
            let v: Vec<f32> = rest
                .split_whitespace()
                .map(|t| t.parse().unwrap())
                .collect();
            out.push([v[0], v[1], v[2]]);
        }
    }
    out
}

#[test]
fn klippe_positions_match_converter_index_for_index() {
    let Some(xmed_path) = common::fixture("extracted/assets/TitleMeny/bakgrunn.xmed") else {
        return;
    };
    let Some(obj_path) = common::fixture("assets/models/obj/bakgrunn/bakgrunn.obj") else {
        return;
    };
    let data = std::fs::read(xmed_path).expect("read bakgrunn.xmed");
    let xmed = parse_xmed(&data).expect("parse bakgrunn.xmed");
    let groups = xmed.geometry_groups();
    let klippe = groups
        .iter()
        .find(|g| g[0].name == "klippe")
        .expect("klippe group");
    let mut live = LiveMesh::new();
    decode_mesh_group(&xmed, klippe, false, &mut live);

    let node = xmed
        .mesh_nodes
        .iter()
        .find(|n| n.mesh_ref == "klippe")
        .expect("klippe node");
    let m = node.matrix; // row-vector convention: p' = p * L + t
    let world = |p: [f32; 3]| -> [f32; 3] {
        [
            p[0] * m[0] + p[1] * m[4] + p[2] * m[8] + m[12],
            p[0] * m[1] + p[1] * m[5] + p[2] * m[9] + m[13],
            p[0] * m[2] + p[1] * m[6] + p[2] * m[10] + m[14],
        ]
    };

    let obj = obj_group_positions(&obj_path, "klippe");
    assert_eq!(live.positions.len(), 375);
    assert_eq!(obj.len(), 375);
    let bad: Vec<usize> = live
        .positions
        .iter()
        .zip(&obj)
        .enumerate()
        .filter(|(_, (p, o))| {
            let w = world(**p);
            (0..3).any(|k| (w[k] - o[k]).abs() > 2e-3)
        })
        .map(|(i, _)| i)
        .collect();
    assert!(
        bad.is_empty(),
        "klippe vertices off vs converter OBJ: {:?}",
        bad
    );
}
