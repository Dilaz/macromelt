//! Interleaved 0x49 continuation chunks must be regrouped per mesh name.
//!
//! `Christian/together_trav2.xmed` writes its five geometry chunks as
//! `lynet, monica, lynet, monica, monica`. Grouping by file adjacency (what the
//! decoder did before) yields four groups and restarts the arithmetic coder and
//! the CLOD resolution counter in the middle of each mesh, decoding lynet to
//! 375/630 positions and monica to 652/1252. Grouping by name yields the two
//! meshes the 0x45 declarations describe, at their exact declared counts.

use macromelt::chunks::parse_xmed;
use macromelt::clod_state::LiveMesh;
use macromelt::geometry::decode_mesh_group;

mod common;

#[test]
fn together_trav2_interleaved_chunks_decode_to_declared_counts() {
    let Some(path) = common::fixture("extracted/assets/Christian/together_trav2.xmed") else {
        return;
    };
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    let xmed = parse_xmed(&data).expect("parse together_trav2.xmed");

    let order: Vec<&str> = xmed.geometry.iter().map(|g| g.name.as_str()).collect();
    assert_eq!(
        order,
        ["lynet", "monica", "lynet", "monica", "monica"],
        "fixture no longer interleaves; pick another file"
    );

    let groups = xmed.geometry_groups();
    let shape: Vec<(&str, usize)> = groups
        .iter()
        .map(|g| (g[0].name.as_str(), g.len()))
        .collect();
    assert_eq!(shape, [("lynet", 2), ("monica", 3)]);

    for group in &groups {
        let md = xmed
            .mesh_descriptions
            .iter()
            .find(|m| m.name == group[0].name)
            .expect("0x45 declaration");
        let mut live = LiveMesh::new();
        decode_mesh_group(&xmed, group, false, &mut live);
        assert_eq!(
            (live.positions.len(), live.faces.len()),
            (md.num_positions as usize, md.num_faces as usize),
            "{} decoded short of its declaration",
            md.name
        );
        assert!(
            live.degenerate_faces().is_empty(),
            "{} has degenerate faces",
            md.name
        );
        let max_corner = live.faces.iter().flatten().copied().max().unwrap();
        assert!((max_corner as usize) < live.positions.len());
    }
}
