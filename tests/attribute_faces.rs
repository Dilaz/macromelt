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
///
/// A directory that cannot be listed is a broken fixture root, not something
/// to walk past: a silent `continue` here would turn a mistyped
/// `MACROMELT_FIXTURES` into a test that passes over nothing.
fn xmed_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let entries =
            std::fs::read_dir(&d).unwrap_or_else(|e| panic!("read_dir {}: {e}", d.display()));
        for e in entries {
            let p = e
                .unwrap_or_else(|e| panic!("dirent under {}: {e}", d.display()))
                .path();
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

/// One fixture root's full `.xmed` census.
#[derive(Debug, Default, PartialEq, Eq)]
struct Corpus {
    /// Files with at least one mesh declaration.
    geometry: usize,
    /// Meshes across those files.
    meshes: usize,
    /// Director *text* members, which share the `.xmed` extension.
    text: usize,
    /// Files that parse as XMED but declare no mesh.
    empty: usize,
    attributes: BTreeMap<u32, usize>,
}

/// What each Tallitytöt disc must yield, measured 2026-09-12.
///
/// Pinning the totals is the point: without them a half-finished extraction or
/// a `parse_xmed` regression would quietly reclassify geometry as text members
/// and this test would sweep a shrunken corpus while still reporting success.
fn expected_corpus(root: &Path) -> Option<Corpus> {
    let (geometry, meshes, text, empty, attributes): (_, _, _, _, &[(u32, usize)]) =
        match root.file_name()?.to_str()? {
            "tallitytot" => (198, 424, 640, 269, &[(4, 424)]),
            "tallitytot2" => (112, 857, 206, 238, &[(4, 235), (6, 622)]),
            _ => return None,
        };
    Some(Corpus {
        geometry,
        meshes,
        text,
        empty,
        attributes: attributes.iter().copied().collect(),
    })
}

/// The whole-corpus guard, over every `.xmed` under every configured fixture
/// root: no file, of either variant, may decode short. This is what keeps disc
/// 1 byte-identical — it takes the `attributes = 4` path on every one of its
/// meshes and must stay untouched by the record added for `attributes = 6`.
///
/// Nothing is skipped quietly and nothing is merely printed. Every `.xmed` is
/// read and classified, a root of a known disc is held to its exact census
/// (`expected_corpus`), and the two structural claims this file's header makes
/// are asserted for every root, known or not:
///  - only `4` and `6` occur, i.e. no third variant slips in undecoded;
///  - no single file mixes them, which is what makes the gate per mesh and not
///    per title.
#[test]
fn every_fixture_mesh_reaches_its_declared_counts() {
    let roots = common::fixture_roots();
    if roots.is_empty() {
        eprintln!("skip: MACROMELT_FIXTURES unset");
        return;
    }

    let mut total = Corpus::default();
    for root in &roots {
        let dir = root.join("extracted/assets");
        let paths = xmed_files(&dir);
        assert!(!paths.is_empty(), "no .xmed under {}", dir.display());

        let mut corpus = Corpus::default();
        for path in paths {
            let data =
                std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            // Director text members share the extension; they are not geometry.
            let Ok(xmed) = parse_xmed(&data) else {
                corpus.text += 1;
                continue;
            };
            if xmed.mesh_descriptions.is_empty() {
                corpus.empty += 1;
                continue;
            }
            corpus.geometry += 1;

            let mut seen: BTreeMap<u32, usize> = BTreeMap::new();
            for (name, attributes, got, declared) in decode_xmed(&xmed) {
                corpus.meshes += 1;
                *corpus.attributes.entry(attributes).or_default() += 1;
                *seen.entry(attributes).or_default() += 1;
                assert_eq!(
                    got,
                    declared,
                    "{}: {name} (attributes = {attributes}) decoded vs declared",
                    path.display(),
                );
            }
            assert_eq!(
                seen.len(),
                1,
                "{}: one file mixes attribute variants {seen:?}",
                path.display(),
            );
        }

        assert!(
            corpus.attributes.keys().all(|a| *a == 4 || *a == 6),
            "{}: unknown attributes variant in {:?}; only bit 1 is decoded",
            root.display(),
            corpus.attributes,
        );
        if let Some(want) = expected_corpus(root) {
            assert_eq!(corpus, want, "{} census", root.display());
        } else {
            eprintln!("{}: unpinned fixture root, {corpus:?}", root.display());
        }

        total.geometry += corpus.geometry;
        total.meshes += corpus.meshes;
        total.text += corpus.text;
        total.empty += corpus.empty;
        for (k, v) in corpus.attributes {
            *total.attributes.entry(k).or_default() += v;
        }
    }

    assert!(total.geometry > 0, "no 3D fixtures found under {roots:?}");
    eprintln!(
        "{} meshes in {} files (+{} text members, {} without geometry); histogram {:?}",
        total.meshes, total.geometry, total.text, total.empty, total.attributes,
    );
}
