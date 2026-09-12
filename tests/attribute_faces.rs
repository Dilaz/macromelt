//! Every new face of a mesh whose 0x45 declaration sets `attributes` bit 1
//! carries an attribute-face record — `call_7a18a990.c:1209-1324`, gated in the
//! original on `decoder->0x64 != 0 && decoder->0x84 != 0`. It fills a second,
//! 28-byte-per-face array parallel to the position faces (three per-corner
//! attribute indices, three next-face links, three next-corner bytes and the
//! ctx0x1d byte), and none of it feeds position or face reconstruction.
//!
//! It still has to be decoded, because the next face's ctx0x19 corner type
//! follows its bits. The flag is per mesh, and both values occur in the wild:
//! across the two Tallitytöt discs, 659 of 1281 mesh declarations are
//! `attributes = 4` and 622 are `attributes = 6` (disc 1 is uniformly 4; disc 2
//! mixes 235 and 622, never within one file). Before this, every
//! `attributes = 6` mesh desynced on the *second* face of its first record —
//! `camerabox` stopped at 6 of 24 positions and 1 of 12 faces,
//! `background_house` at 3 of 4 and 1 of 2.
//!
//! The contract asserted here is the one the bake depends on
//! (`ExportError::IncompleteDecode`): a mesh decodes to exactly the position
//! and face counts its own declaration promises. That is a whole-bitstream
//! check — 14 000 positions across 20 meshes only land on their declared totals
//! if every record boundary in the file is bit-exact.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use macromelt::chunks::{XmedFile, parse_xmed};
use macromelt::clod_state::LiveMesh;
use macromelt::geometry::decode_mesh_group;

mod common;

/// One mesh's `(name, attributes, decoded, declared)` position/face counts.
type MeshCounts = (String, u32, (usize, usize), (usize, usize));

/// Decode every mesh of `xmed`.
fn decode_xmed(xmed: &XmedFile) -> Vec<MeshCounts> {
    let mut out = Vec::new();
    for group in xmed.geometry_groups() {
        let name = group[0].name.clone();
        let Some(md) = xmed.mesh_descriptions.iter().find(|m| m.name == name) else {
            continue;
        };
        let mut live = LiveMesh::new();
        decode_mesh_group(xmed, &group, false, &mut live);
        out.push((
            name,
            md.attributes,
            (live.positions.len(), live.faces.len()),
            (md.num_positions as usize, md.num_faces as usize),
        ));
    }
    out
}

/// Decode every mesh of the fixture at `rel`.
fn decode_all(rel: &str) -> Option<Vec<MeshCounts>> {
    let path = common::fixture(rel)?;
    let data = std::fs::read(&path).expect("read fixture");
    let xmed = parse_xmed(&data).expect("parse fixture");
    Some(decode_xmed(&xmed))
}

/// Every `.xmed` under `dir`, recursively, in sorted order.
fn xmed_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "xmed") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Disc 2's title scene: 20 meshes, every one `attributes = 6`. Without the
/// attribute-face record `camerabox` yields 6 positions / 1 face instead of
/// 24 / 12.
#[test]
fn disc2_meshes_reach_their_declared_counts() {
    let Some(meshes) = decode_all("extracted/assets/TitleMeny/scene.xmed") else {
        return;
    };
    assert_eq!(meshes.len(), 20, "TitleMeny/scene.xmed mesh count");

    for (name, attributes, decoded, declared) in &meshes {
        assert_eq!(
            *attributes, 6,
            "{name}: disc 2 declares the attribute-face record"
        );
        assert_eq!(
            decoded, declared,
            "{name}: decoded vs declared (positions, faces)"
        );
    }

    let camerabox = meshes
        .iter()
        .find(|(n, ..)| n == "camerabox")
        .expect("camerabox");
    assert_eq!(camerabox.2, (24, 12), "camerabox is a 24-position box");
}

/// The whole-corpus guard, over every `.xmed` under every configured fixture
/// root: no file, of either variant, may decode short. This is what keeps disc
/// 1 byte-identical — it takes the `attributes = 4` path on every one of its
/// meshes and must stay untouched by the record added for `attributes = 6`.
///
/// It also prints the `attributes` histogram, which is how the one-variant-per-
/// title claim in this file's header is checked rather than assumed.
#[test]
fn every_fixture_mesh_reaches_its_declared_counts() {
    let roots = common::fixture_roots();
    if roots.is_empty() {
        eprintln!("skip: MACROMELT_FIXTURES unset");
        return;
    }

    let (mut files, mut meshes) = (0usize, 0usize);
    let mut by_attributes: BTreeMap<u32, usize> = BTreeMap::new();
    for root in &roots {
        for path in xmed_files(&root.join("extracted/assets")) {
            let Ok(data) = std::fs::read(&path) else {
                continue;
            };
            // Director text members share the extension; they are rejected here.
            let Ok(xmed) = parse_xmed(&data) else {
                continue;
            };
            if xmed.mesh_descriptions.is_empty() {
                continue;
            }
            files += 1;
            for (name, attributes, decoded, declared) in decode_xmed(&xmed) {
                meshes += 1;
                *by_attributes.entry(attributes).or_default() += 1;
                assert_eq!(
                    decoded,
                    declared,
                    "{}: {name} (attributes = {attributes}) decoded vs declared",
                    path.display(),
                );
            }
        }
    }

    assert!(files > 0, "no 3D fixtures found under {roots:?}");
    eprintln!("{meshes} meshes in {files} files; attributes histogram {by_attributes:?}");
}
