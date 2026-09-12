//! Golden tests for the 0x45 Author Mesh Declaration parser (`chunks.rs`).
//!
//! The chunk declares, per mesh: an attribute word, a shading-group count `N`,
//! `N` count records (positions/faces/normals/texcoords/colors + the per-vertex
//! attribute mask the 0x49 decoder needs), a name-array count `M`, and `M`
//! arrays of `N + 1` strings — array 0 being `["default", shader × N]` and every
//! later array `[mesh name, material × N]`.
//!
//! Ground truth for the counts is `tools/re/ground-truth.md:19-21` (arrow);
//! ground truth for the material order is the Wine-converter OBJ, whose
//! `usemtl` runs appear in shading-id order.

use macromelt::chunks::{XmedFile, parse_xmed};

mod common;

fn parse(rel: &str) -> Option<XmedFile> {
    let path = common::fixture(rel)?;
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    Some(parse_xmed(&data).unwrap_or_else(|e| panic!("parse {}: {}", path.display(), e)))
}

/// `together_trav2.xmed` declares two meshes. Their shader-slot material lists
/// must come out in shading-id order, matching the `usemtl` runs of
/// `assets/models/obj/together_trav2/together_trav2.obj`:
///   `g lynet`  → hestset, hale, stropp        (lines 3, 2369, 2651)
///   `g monica` → helmet, helmet2, monica00, monica01 (2693, 2853, 2874, 3683)
/// The OBJ names are the XMED names with underscores stripped by the converter.
#[test]
fn together_trav2_material_lists() {
    let Some(xmed) = parse("extracted/assets/Christian/together_trav2.xmed") else {
        return;
    };
    assert_eq!(xmed.mesh_descriptions.len(), 2, "two 0x45 chunks");

    let lynet = &xmed.mesh_descriptions[0];
    assert_eq!(lynet.name, "lynet");
    assert_eq!(lynet.num_shaders, 3);
    assert_eq!(lynet.material_names(), vec!["hest_set", "hale", "stropp"]);

    let monica = &xmed.mesh_descriptions[1];
    assert_eq!(monica.name, "monica");
    assert_eq!(monica.num_shaders, 4);
    assert_eq!(
        monica.material_names(),
        vec!["helmet", "helmet2", "monica_00", "monica_01"]
    );

    // Every slot names the shader it draws with, and both meshes carry one UV
    // layer plus normals.
    for md in [lynet, monica] {
        assert_eq!(md.attribute_mask(), 0x13, "{} attribute mask", md.name);
        for slot in &md.shaders {
            assert_eq!(slot.shader, "DefaultShader");
            assert_eq!(slot.attrs, 0x13);
        }
        assert!(md.sibling_meshes.is_empty());
    }

    // Slot counts sum to the mesh totals (lynet: 547 + 71 + 12 positions).
    assert_eq!(
        lynet.shaders.iter().map(|s| s.num_positions).sum::<u32>(),
        lynet.num_positions
    );
    assert_eq!(lynet.shaders[0].num_positions, 547);
    assert_eq!(lynet.shaders[1].num_positions, 71);
    assert_eq!(lynet.shaders[2].num_positions, 12);
}

/// `tools/re/ground-truth.md:19-21`: arrow's counts after the name are
/// `04 01 12 16 1E 0B 00 03 02` — attributes 4, one shading group, then that
/// group's 18 positions / 22 faces / 30 normals / 11 texcoords / 0 colors and
/// attribute mask 0x3, then two name arrays.
#[test]
fn arrow_counts_match_ground_truth() {
    let Some(xmed) = parse("extracted/assets/Hey/arrow.xmed") else {
        return;
    };
    let md = &xmed.mesh_descriptions[0];

    assert_eq!(md.name, "arrow");
    assert_eq!(md.attributes, 4);
    assert_eq!(md.num_shaders, 1);
    assert_eq!(md.extra_count, 2);
    assert_eq!(md.num_positions, 18);
    assert_eq!(md.num_faces, 22);
    assert_eq!(md.num_normals, 30);
    assert_eq!(md.num_texcoords, 11);
    assert_eq!(md.num_colors, 0);
    assert_eq!(md.max_resolution, 13);

    // arrow.obj faces are `a//a` — no UV layer, so the mask is positions+normals.
    assert_eq!(md.attribute_mask(), 0x3);
    assert_eq!(md.material_names(), vec!["pil"]);
}

/// `kost.xmed` is the single-group counterexample: one UV layer, mask 0x13, and
/// 56 positions — the vertex count of `assets/models/obj/kost/kost.obj`.
#[test]
fn kost_declares_one_uv_layer() {
    let Some(xmed) = parse("extracted/assets/Rydde/kost.xmed") else {
        return;
    };
    let md = &xmed.mesh_descriptions[0];

    assert_eq!(md.name, "kost");
    assert_eq!(md.num_positions, 56);
    assert_eq!(md.attribute_mask(), 0x13);
    assert_eq!(md.material_names(), vec!["kost"]);
}

/// `stall.xmed` packs nine identically shaped boxes into one declaration:
/// `M = 10` name arrays (the shader array plus one material array per box).
#[test]
fn stall_declares_sibling_meshes() {
    let Some(xmed) = parse("extracted/assets/Rydde/stall.xmed") else {
        return;
    };
    let md = xmed
        .mesh_descriptions
        .iter()
        .find(|m| m.name == "Box01")
        .expect("Box01 declaration");

    assert_eq!(md.num_shaders, 1);
    assert_eq!(md.extra_count, 10);
    assert_eq!(md.material_names(), vec!["Default Material116"]);
    assert_eq!(md.sibling_meshes.len(), 8);
    assert_eq!(md.sibling_meshes[0].name, "Box02");
    assert_eq!(md.sibling_meshes[0].materials, vec!["Default Material117"]);
    assert_eq!(md.sibling_meshes[7].name, "Box10");
}

/// The 0x74 camera chunk. Layout was pinned by diffing `Hey/npc_00.xmed`
/// against `Hey/npc_00_1.xmed`: inside their (equally sized, 202-byte) camera
/// chunks the only differing byte runs are 0x17-0x20, 0x27-0x31, 0x37-0x41 and
/// 0x47-0x51 — the three basis rows and the translation row of a row-major 4x4
/// starting at chunk offset 0x17, with the constant (0,0,0,1) fourth column at
/// 0x23/0x33/0x43/0x53 untouched and everything from 0x57 on identical.
#[test]
fn camera_chunk_layout() {
    let Some(arrow) = parse("extracted/assets/Hey/arrow.xmed") else {
        return;
    };
    assert_eq!(arrow.cameras.len(), 1);
    let cam = &arrow.cameras[0];
    assert_eq!(cam.name, "defaultview");
    assert_eq!(cam.parent, "World");
    assert_eq!(cam.projection, 8);
    assert_eq!(cam.hither, 1.0);
    assert_eq!(cam.yon, f32::MAX);
    assert!((cam.fov - 33.75).abs() < 0.01, "fov {}", cam.fov);
    assert_eq!(cam.rect, [0.0, 0.0, 640.0, 480.0]);
    assert_eq!(cam.matrix[15], 1.0);
    assert_eq!(
        cam.position,
        [cam.matrix[12], cam.matrix[13], cam.matrix[14]]
    );
    assert!(
        (cam.position[0] - 51.476).abs() < 1e-3,
        "{:?}",
        cam.position
    );
    assert!(
        (cam.position[1] + 92.7148).abs() < 1e-3,
        "{:?}",
        cam.position
    );
    assert!(
        (cam.position[2] - 23.5641).abs() < 1e-3,
        "{:?}",
        cam.position
    );

    // Same chunk in both npc_00 variants: identical projection block, different
    // transform.
    let Some(a) = parse("extracted/assets/Hey/npc_00.xmed") else {
        return;
    };
    let Some(b) = parse("extracted/assets/Hey/npc_00_1.xmed") else {
        return;
    };
    let (ca, cb) = (&a.cameras[0], &b.cameras[0]);
    assert_eq!(ca.name, cb.name);
    assert_eq!(
        (ca.projection, ca.fov, ca.hither, ca.yon, ca.rect),
        (cb.projection, cb.fov, cb.hither, cb.yon, cb.rect)
    );
    assert_ne!(ca.position, cb.position);
    for i in [3, 7, 11, 15] {
        assert_eq!(ca.matrix[i], cb.matrix[i], "matrix column 3 row {}", i / 4);
    }

    // MAX-authored cameras take the shorter header (no 0x20 byte after the flag)
    // and a normalized viewport rect.
    let Some(scene) = parse("extracted/assets/Map1_1/1_1_scene.xmed") else {
        return;
    };
    let max_cam = scene
        .cameras
        .iter()
        .find(|c| c.name == "Camera01")
        .expect("Camera01");
    assert_eq!(max_cam.rect, [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(max_cam.matrix[15], 1.0);
    assert!(max_cam.fov > 0.0 && max_cam.fov < 180.0, "{}", max_cam.fov);
}
