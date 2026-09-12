//! Golden test for the native XMED 0x49 CONNECTIVITY decode (plan step 3.1).
//!
//! `geometry_decode.rs` locks the decoded POSITIONS against the Wine-converter
//! OBJ ground truth. This file locks the face graph, against three independent
//! captures of the original Shockwave 3D decoder (`FUN_7a18a990`):
//!
//! 1. `tools/re/traces/arrow_faces_trace.log` — the 22 position-face WRITES at
//!    the append site `0x7a18c814`, i.e. every face exactly as group_b first
//!    emits it, before any vertex-split corner rewrite. Locked here as
//!    `EXPECTED_APPENDS`.
//! 2. `tools/re/traces/arrow_live_facearray_trace.log` — 13 winedbg snapshots of
//!    the LIVE face array taken mid-decode at the case-2/3/4 corner-lookup
//!    sites, which prove the array is mutated in place. Two of those snapshots
//!    (the first and the last) are locked here as `SNAPSHOT_0` / `SNAPSHOT_12`;
//!    all 13 are checked by `tools/re/traces/diff_facearray.py`.
//! 3. `assets/models/obj/arrow/arrow.obj` — the converter's exported render
//!    mesh, which the final live array must equal index-for-index.
//!
//! The decoder reproduces all three: 22/22 appends, 22/22 final faces in OBJ
//! order with orientation, and OBJ vertex 2 unreferenced in both.

use macromelt::chunks::parse_xmed;
use macromelt::clod_state::LiveMesh;
use macromelt::geometry::decode_geometry_group_into;

mod common;

/// `tools/re/traces/arrow_faces_trace.log`, writes #0..#21.
const EXPECTED_APPENDS: [[u32; 3]; 22] = [
    [0, 1, 2],
    [3, 4, 5],
    [0, 6, 4],
    [6, 3, 5],
    [7, 5, 0],
    [7, 3, 5],
    [0, 1, 8],
    [7, 3, 8],
    [9, 7, 5],
    [0, 9, 1],
    [5, 4, 10],
    [6, 10, 4],
    [3, 6, 11],
    [11, 6, 10],
    [12, 9, 0],
    [12, 3, 1],
    [13, 8, 1],
    [0, 13, 5],
    [3, 14, 6],
    [15, 0, 12],
    [16, 10, 5],
    [1, 8, 17],
];

/// `arrow_live_facearray_trace.log` snapshot #0 — the base mesh, first 2 slots.
const SNAPSHOT_0: [[u32; 3]; 2] = [[0, 1, 2], [3, 4, 5]];

/// `arrow_live_facearray_trace.log` snapshot #12 — the last captured state, all
/// 16 dumped slots. Reached after the record-9 face updates land.
const SNAPSHOT_12: [[u32; 3]; 16] = [
    [6, 5, 10],
    [3, 4, 1],
    [3, 11, 4],
    [6, 3, 5],
    [7, 9, 12],
    [7, 8, 13],
    [12, 1, 8],
    [7, 12, 8],
    [9, 7, 13],
    [0, 9, 13],
    [1, 4, 10],
    [11, 10, 4],
    [3, 6, 11],
    [11, 6, 10],
    [12, 9, 0],
    [12, 3, 1],
];

fn decode_arrow() -> Option<LiveMesh> {
    let path = common::fixture("extracted/assets/Hey/arrow.xmed")?;
    let data = std::fs::read(&path).expect("read arrow.xmed");
    let xmed = parse_xmed(&data).expect("parse arrow.xmed");
    let g = xmed.geometry.first().expect("arrow has a geometry chunk");
    let md = xmed.mesh_descriptions.iter().find(|m| m.name == g.name);
    let mut live = LiveMesh::new();
    live.record_appends = true;
    decode_geometry_group_into(&[g], md, false, &mut live);
    Some(live)
}

/// The final live face array equals `arrow.obj` face-for-face, corner-for-corner
/// (the OBJ's `f a//a` corners are position indices into the same 18-vertex
/// array `geometry_decode.rs` validates).
#[test]
fn arrow_faces_match_obj_in_order() {
    let Some(live) = decode_arrow() else { return };

    let Some(obj_path) = common::fixture("assets/models/obj/arrow/arrow.obj") else {
        return;
    };
    let text = std::fs::read_to_string(&obj_path).expect("read arrow.obj");
    let obj_faces: Vec<[u32; 3]> = text
        .lines()
        .filter(|l| l.starts_with("f "))
        .map(|l| {
            let mut it = l.split_whitespace().skip(1).map(|t| {
                t.split('/')
                    .next()
                    .unwrap()
                    .parse::<u32>()
                    .expect("obj face index")
                    - 1
            });
            [
                it.next().expect("corner 0"),
                it.next().expect("corner 1"),
                it.next().expect("corner 2"),
            ]
        })
        .collect();

    assert_eq!(obj_faces.len(), 22, "arrow.obj face count");
    assert_eq!(
        live.faces.len(),
        22,
        "decoded face count (0x45 declares 22 faces)"
    );
    for (i, (got, want)) in live.faces.iter().zip(obj_faces.iter()).enumerate() {
        assert_eq!(got, want, "face {i} corners");
    }

    // The author mesh leaves one seam duplicate unreferenced; the OBJ agrees.
    assert_eq!(
        live.orphan_positions(),
        vec![2],
        "only position 2 is unreferenced"
    );
    assert!(
        live.degenerate_faces().is_empty(),
        "no repeated-corner faces: {:?}",
        live.degenerate_faces()
    );
}

/// Every face, as group_b first appends it, equals the binary's face-write trace
/// byte for byte. This pins all 66 group_b corner reads — including the 36
/// case-2/3/4 reads that resolve through the live face array — independently of
/// the split rewrites that mutate them afterwards.
#[test]
fn arrow_group_b_appends_match_binary_write_trace() {
    let Some(live) = decode_arrow() else { return };
    assert_eq!(
        live.appends.len(),
        EXPECTED_APPENDS.len(),
        "22 group_b appends"
    );
    for (i, (got, want)) in live.appends.iter().zip(EXPECTED_APPENDS.iter()).enumerate() {
        assert_eq!(got, want, "append #{i} (arrow_faces_trace.log write #{i})");
    }
}

/// The two winedbg snapshots that bracket the mutation sequence. Snapshot #0 is
/// the base mesh, so it must equal the first two appends. Snapshot #12 is the
/// last capture; every slot it dumps must either already hold its final value or
/// differ only where a later record's face update rewrites that exact corner —
/// so replaying the snapshot against the final array pins which corners records
/// 10-11 still touch. `tools/re/traces/diff_facearray.py` checks all 13
/// snapshots against a `XMED_FACEDUMP=1` run.
#[test]
fn arrow_live_array_reaches_captured_snapshots() {
    let Some(live) = decode_arrow() else { return };

    for (i, want) in SNAPSHOT_0.iter().enumerate() {
        assert_eq!(
            &live.appends[i], want,
            "snapshot #0 slot {i} is the appended base face"
        );
    }

    // Corners still rewritten after snapshot #12, read off the decode: slot →
    // corner → (captured, final). Records 10 and 11 move 8 corners across 7
    // slots; every other dumped slot is already final at snapshot #12.
    const POST_SNAPSHOT_REWRITES: [(usize, usize, u32, u32); 8] = [
        (1, 2, 1, 16),
        (3, 1, 3, 14),
        (6, 1, 1, 17),
        (9, 0, 0, 15),
        (10, 0, 1, 16),
        (14, 2, 0, 15),
        (15, 1, 3, 0),
        (15, 2, 1, 17),
    ];
    let mut expected = SNAPSHOT_12;
    for (slot, corner, captured, final_value) in POST_SNAPSHOT_REWRITES {
        assert_eq!(
            expected[slot][corner], captured,
            "snapshot #12 slot {slot} corner {corner} captured value"
        );
        expected[slot][corner] = final_value;
    }
    for (i, want) in expected.iter().enumerate() {
        assert_eq!(
            &live.faces[i], want,
            "slot {i}: snapshot #12 plus the record 10-11 rewrites"
        );
    }
}
