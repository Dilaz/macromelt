//! A Director `keyframePlayer` motion (one track driving a MODEL's transform)
//! must come out of the native bake as a glTF animation on the model's node.
//!
//! `Hey/ParentScript 15 - trolleyguy.ls:15,21-29` adds a keyframePlayer to the
//! cart and clones `tralle_pickuphay` as `tralle_pickuphay-Key`; the motion's
//! single track is named after its own file's mesh (`tralle_pickuphay`), and the
//! Lingo plays it on the `tralle` model. Director plays keyframe motions
//! relative to the transform at play start (every such motion in the corpus has
//! kf0 = identity), so the bake puts the authored 0x72 TRS on a `tralle:start`
//! parent and lets the channels drive the identity-rest `tralle` node.

use macromelt::chunks::parse_xmed;
use macromelt::clod_state::LiveMesh;
use macromelt::geometry::decode_mesh_group;
use macromelt::gltf_export::{ExportOptions, export_glb};
use macromelt::motion::decode_motion_full;
use serde_json::Value;

mod common;

fn read_glb_json(data: &[u8]) -> (Value, Vec<u8>) {
    assert_eq!(&data[0..4], b"glTF");
    let mut off = 12usize;
    let mut json = None;
    let mut bin = None;
    while off + 8 <= data.len() {
        let len = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        let body = &data[off + 8..off + 8 + len];
        match &data[off + 4..off + 8] {
            b"JSON" => json = Some(serde_json::from_slice(body).unwrap()),
            b"BIN\0" => bin = Some(body.to_vec()),
            _ => {}
        }
        off += 8 + len;
    }
    (json.unwrap(), bin.unwrap())
}

fn read_f32s(json: &Value, bin: &[u8], accessor: usize) -> Vec<f32> {
    let acc = &json["accessors"][accessor];
    let bv = &json["bufferViews"][acc["bufferView"].as_u64().unwrap() as usize];
    let n = match acc["type"].as_str().unwrap() {
        "SCALAR" => 1,
        "VEC3" => 3,
        "VEC4" => 4,
        t => panic!("unexpected accessor type {t}"),
    };
    let start = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
    let count = acc["count"].as_u64().unwrap() as usize * n;
    (0..count)
        .map(|i| f32::from_le_bytes(bin[start + 4 * i..start + 4 * i + 4].try_into().unwrap()))
        .collect()
}

#[test]
fn tralle_pickuphay_key_drives_the_tralle_node() {
    let Some(rest_path) = common::fixture("extracted/assets/Hey/tralle.xmed") else {
        return;
    };
    let Some(anim_path) = common::fixture("extracted/assets/Hey/tralle_pickuphay.xmed") else {
        return;
    };
    let rest = parse_xmed(&std::fs::read(rest_path).unwrap()).unwrap();
    let anim = parse_xmed(&std::fs::read(anim_path).unwrap()).unwrap();
    let motions: Vec<_> = anim.motions.iter().map(decode_motion_full).collect();
    assert_eq!(motions.len(), 1);
    assert_eq!(motions[0].name, "tralle_pickuphay-Key");
    assert_eq!(motions[0].tracks.len(), 1);

    let glb = export_glb(
        &rest,
        &ExportOptions {
            extra_motions: motions,
            ..Default::default()
        },
    )
    .expect("export_glb");
    let (json, bin) = read_glb_json(&glb);

    // Nodes: SceneRoot > tralle:start (authored 0x72 TRS) > tralle (mesh, identity rest).
    let nodes = json["nodes"].as_array().unwrap();
    let model = nodes
        .iter()
        .position(|n| n["name"] == "tralle")
        .expect("tralle node");
    let start = nodes
        .iter()
        .position(|n| n["name"] == "tralle:start")
        .expect("tralle:start node");
    assert!(
        nodes[model].get("mesh").is_some(),
        "tralle carries the mesh"
    );
    assert!(nodes[model].get("translation").is_none() && nodes[model].get("rotation").is_none());
    assert_eq!(nodes[start]["children"], serde_json::json!([model]));
    let node72 = rest
        .mesh_nodes
        .iter()
        .find(|n| n.mesh_ref == "tralle")
        .unwrap();
    let t = nodes[start]["translation"].as_array().unwrap();
    for k in 0..3 {
        assert!((t[k].as_f64().unwrap() as f32 - node72.matrix[12 + k]).abs() < 1e-5);
    }

    // Positions are LOCAL: every GLB vertex is one of the natively decoded
    // positions verbatim (the 0x72 matrix lives on `tralle:start`, not in the data).
    let group = rest
        .geometry_groups()
        .into_iter()
        .find(|g| g[0].name == "tralle")
        .unwrap();
    let mut live = LiveMesh::new();
    decode_mesh_group(&rest, &group, false, &mut live);
    let prim = &json["meshes"][nodes[model]["mesh"].as_u64().unwrap() as usize]["primitives"][2];
    let pos = read_f32s(
        &json,
        &bin,
        prim["attributes"]["POSITION"].as_u64().unwrap() as usize,
    );
    assert_eq!(pos.len(), 52 * 3);
    for p in pos.chunks(3) {
        assert!(
            live.positions
                .iter()
                .any(|q| (0..3).all(|k| (q[k] - p[k]).abs() < 1e-6)),
            "GLB vertex {p:?} is not a decoded local position"
        );
    }

    // One clip, named after the XMED motion, with translation + rotation
    // channels on the tralle node, kf0 = identity, 24 keyframes over 1.433 s.
    let anims = json["animations"].as_array().unwrap();
    assert_eq!(anims.len(), 1);
    assert_eq!(anims[0]["name"], "tralle_pickuphay-Key");
    let channels = anims[0]["channels"].as_array().unwrap();
    let paths: Vec<&str> = channels
        .iter()
        .map(|c| c["target"]["path"].as_str().unwrap())
        .collect();
    assert_eq!(paths, ["translation", "rotation"]);
    for c in channels {
        assert_eq!(c["target"]["node"].as_u64().unwrap() as usize, model);
    }
    let sampler = &anims[0]["samplers"][0];
    let times = read_f32s(&json, &bin, sampler["input"].as_u64().unwrap() as usize);
    assert_eq!(times.len(), 24);
    assert!((times[23] - 1.4333).abs() < 1e-3);
    let trans = read_f32s(&json, &bin, sampler["output"].as_u64().unwrap() as usize);
    assert!(
        trans[..3].iter().all(|v| v.abs() < 1e-4),
        "kf0 translation is identity"
    );
    assert!(
        (trans[3 * 23 + 2] - 5.9699).abs() < 1e-3,
        "last keyframe lifts the cart by 5.97 (z)"
    );
    let rot = read_f32s(
        &json,
        &bin,
        anims[0]["samplers"][1]["output"].as_u64().unwrap() as usize,
    );
    assert!(
        (rot[3] - 1.0).abs() < 1e-5,
        "kf0 rotation is identity (xyzw)"
    );
}
