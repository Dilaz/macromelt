//! `lynet_native_matches_obj` — the parity plan's named check of the native
//! decode of `Stelle/lynet.xmed` against the Wine-converter
//! `assets/models/obj/lynet/lynet.obj`.
//!
//! The converter OBJ is NOT a position ground truth for the horse: `lynet` is
//! a skinned mesh and the converter exports skinned meshes bone-posed (the
//! rest pose run through the bind skeleton), while the 0x49 stream holds the
//! authored bind-space positions. The OBJ is also a lower CLOD resolution
//! (706 unique vertices for 808 declared positions), so even the face index
//! sets cannot be compared vertex-for-vertex. What CAN be asserted, and is:
//!
//! * every mesh group of `lynet.xmed` decodes to its exact 0x45 declaration
//!   (`lynet` 808/1068, `sal` 288/354, `pled` 62/68), with no degenerate face
//!   and no corner past the position count;
//! * the two unskinned groom props `sal` and `pled` do have a position ground
//!   truth in the OBJ: every converter vertex has an exact decoded twin
//!   (tolerance `1e-3 * bbox`), the converter's remaining vertices being the
//!   coarser-LOD subset - the same rule `--native-check` applies.
//!
//! `cargo run --release -- extracted/assets/Stelle/lynet.xmed --native-check
//! assets/models/obj` prints the same verdict per group.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use macromelt::chunks::parse_xmed;
use macromelt::clod_state::LiveMesh;
use macromelt::geometry::decode_mesh_group;

mod common;

/// Global `v` list plus, per `g` block, the set of vertex indices its faces use.
fn load_obj(path: &Path) -> (Vec<[f64; 3]>, HashMap<String, Vec<usize>>) {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    let mut verts = Vec::new();
    let mut blocks: HashMap<String, HashSet<usize>> = HashMap::new();
    let mut current = String::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let xyz: Vec<f64> = it.take(3).map(|t| t.parse().unwrap()).collect();
                verts.push([xyz[0], xyz[1], xyz[2]]);
            }
            Some("g") => current = it.next().unwrap_or("").to_string(),
            Some("f") => {
                let block = blocks.entry(current.clone()).or_default();
                for t in it {
                    block.insert(t.split('/').next().unwrap().parse::<usize>().unwrap() - 1);
                }
            }
            _ => {}
        }
    }
    let blocks = blocks
        .into_iter()
        .map(|(k, v)| {
            let mut v: Vec<usize> = v.into_iter().collect();
            v.sort_unstable();
            (k, v)
        })
        .collect();
    (verts, blocks)
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[test]
fn lynet_native_matches_obj() {
    let Some(xmed_path) = common::fixture("extracted/assets/Stelle/lynet.xmed") else {
        return;
    };
    let Some(obj_path) = common::fixture("assets/models/obj/lynet/lynet.obj") else {
        return;
    };
    let data =
        std::fs::read(&xmed_path).unwrap_or_else(|e| panic!("read {}: {}", xmed_path.display(), e));
    let xmed = parse_xmed(&data).expect("parse lynet.xmed");
    let (verts, blocks) = load_obj(&obj_path);

    let mut seen = Vec::new();
    for group in xmed.geometry_groups() {
        let name = group[0].name.clone();
        let md = xmed
            .mesh_descriptions
            .iter()
            .find(|m| m.name == name)
            .expect("0x45 declaration");
        let mut live = LiveMesh::new();
        decode_mesh_group(&xmed, &group, false, &mut live);
        assert_eq!(
            (live.positions.len(), live.faces.len()),
            (md.num_positions as usize, md.num_faces as usize),
            "{} decoded short of its declaration",
            name
        );
        assert!(
            live.degenerate_faces().is_empty(),
            "{} has degenerate faces",
            name
        );
        let max_corner = live.faces.iter().flatten().copied().max().unwrap() as usize;
        assert!(
            max_corner < live.positions.len(),
            "{} corner past its positions",
            name
        );

        let skinned = xmed
            .bone_weights
            .iter()
            .any(|bw| bw.mesh_name == name && !bw.bones.is_empty());
        if skinned {
            // Bone-posed in the OBJ: positions are not ground truth. Count-only.
            assert_eq!(name, "lynet", "only the horse is skinned in lynet.xmed");
            seen.push(name);
            continue;
        }

        // Static prop: node matrix (row-vector convention) then reverse-NN.
        let node = xmed
            .mesh_nodes
            .iter()
            .find(|n| n.mesh_ref == name || n.name == name)
            .expect("mesh node");
        let m: Vec<f64> = node.matrix.iter().map(|v| *v as f64).collect();
        let dec: Vec<[f64; 3]> = live
            .positions
            .iter()
            .map(|p| {
                let p = [p[0] as f64, p[1] as f64, p[2] as f64];
                [
                    p[0] * m[0] + p[1] * m[4] + p[2] * m[8] + m[12],
                    p[0] * m[1] + p[1] * m[5] + p[2] * m[9] + m[13],
                    p[0] * m[2] + p[1] * m[6] + p[2] * m[10] + m[14],
                ]
            })
            .collect();
        let block = blocks
            .get(&name)
            .unwrap_or_else(|| panic!("no `g {}` in lynet.obj", name));
        let objv: Vec<[f64; 3]> = block.iter().map(|&i| verts[i]).collect();
        assert!(!objv.is_empty());
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for v in &objv {
            for a in 0..3 {
                lo[a] = lo[a].min(v[a]);
                hi[a] = hi[a].max(v[a]);
            }
        }
        let tol = 1e-3 * dist(lo, hi).max(1.0);
        let worst = objv
            .iter()
            .map(|o| {
                dec.iter()
                    .map(|d| dist(*o, *d))
                    .fold(f64::INFINITY, f64::min)
            })
            .fold(0.0, f64::max);
        assert!(
            worst <= tol,
            "{}: a converter vertex has no decoded twin (worst {:.4} > tol {:.4})",
            name,
            worst,
            tol
        );
        seen.push(name);
    }
    seen.sort();
    assert_eq!(seen, ["lynet", "pled", "sal"]);
}
