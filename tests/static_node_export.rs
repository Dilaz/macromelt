//! An unskinned native bake keeps the Director model as a NODE: named after the
//! XMED mesh node, a direct child of `SceneRoot`, carrying the authored 0x72
//! matrix as TRS, with vertices in node-local space. Lingo assigns
//! `scene.model(name).transform.position` to that node, so folding the matrix
//! into the vertices (what the OBJ converter does) would double the placement.
//!
//! Oracle from the MapScene owner: `Map1_1/karifontene_sit.xmed` has node
//! translation z = 19.3168 and local vertex z in [-14.5, 19.4] (world would be
//! [4.8, 38.7]); `mettefontene_sit` is authored with a 0.2447 uniform scale and
//! an x-mirror.

use macromelt::chunks::parse_xmed;
use macromelt::gltf_export::{ExportOptions, export_glb};
use serde_json::Value;

mod common;

fn read_glb_json(data: &[u8]) -> (Value, Vec<u8>) {
    assert_eq!(&data[0..4], b"glTF");
    let mut off = 12usize;
    let (mut json, mut bin) = (None, None);
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

/// (min, max) over every POSITION accessor of the glTF mesh on `node`.
fn position_bounds(json: &Value, bin: &[u8], node: &Value) -> ([f32; 3], [f32; 3]) {
    let mesh = &json["meshes"][node["mesh"].as_u64().unwrap() as usize];
    let (mut min, mut max) = ([f32::MAX; 3], [f32::MIN; 3]);
    for prim in mesh["primitives"].as_array().unwrap() {
        let acc = &json["accessors"][prim["attributes"]["POSITION"].as_u64().unwrap() as usize];
        let bv = &json["bufferViews"][acc["bufferView"].as_u64().unwrap() as usize];
        let start = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
        for i in 0..acc["count"].as_u64().unwrap() as usize {
            for k in 0..3 {
                let o = start + (i * 3 + k) * 4;
                let v = f32::from_le_bytes(bin[o..o + 4].try_into().unwrap());
                min[k] = min[k].min(v);
                max[k] = max[k].max(v);
            }
        }
    }
    (min, max)
}

fn bake(rel: &str) -> Option<(Value, Vec<u8>)> {
    let path = common::fixture(rel)?;
    let xmed = parse_xmed(&std::fs::read(path).unwrap()).unwrap();
    let glb = export_glb(&xmed, &ExportOptions::default()).expect("export_glb");
    Some(read_glb_json(&glb))
}

#[test]
fn karifontene_sit_keeps_node_translation_and_local_vertices() {
    let Some((json, bin)) = bake("extracted/assets/Map1_1/karifontene_sit.xmed") else {
        return;
    };
    let nodes = json["nodes"].as_array().unwrap();
    let idx = nodes
        .iter()
        .position(|n| n["name"] == "karifontene_sit")
        .expect("model node");
    assert!(
        nodes[0]["children"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!(idx)),
        "model node is a direct child of SceneRoot"
    );
    let t = nodes[idx]["translation"].as_array().unwrap();
    assert!(
        (t[2].as_f64().unwrap() - 19.3168).abs() < 1e-3,
        "node z translation, got {t:?}"
    );
    assert!(nodes[idx].get("rotation").is_none() && nodes[idx].get("scale").is_none());
    let (min, max) = position_bounds(&json, &bin, &nodes[idx]);
    assert!(
        min[2] > -14.6 && min[2] < -13.0,
        "local z min {} (world would be ~4.8)",
        min[2]
    );
    assert!(
        max[2] > 18.0 && max[2] < 19.5,
        "local z max {} (world would be ~38.7)",
        max[2]
    );
    assert_eq!(json["meshes"].as_array().unwrap().len(), 1);
    assert!(json.get("skins").is_none());
}

#[test]
fn mettefontene_sit_keeps_scale_and_mirror_on_the_node() {
    let Some((json, _bin)) = bake("extracted/assets/Map1_1/mettefontene_sit.xmed") else {
        return;
    };
    let node = json["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["name"] == "mettefontene_sit")
        .expect("model node");
    let s: Vec<f64> = node["scale"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert!(
        (s[0] + 0.2447).abs() < 1e-3
            && (s[1] - 0.2447).abs() < 1e-3
            && (s[2] - 0.2447).abs() < 1e-3,
        "{s:?}"
    );
    let t = node["translation"].as_array().unwrap();
    assert!((t[2].as_f64().unwrap() - 19.2663).abs() < 1e-3);
}

#[test]
fn fontene_alpha_plane_texture_blends() {
    let Some((json, _bin)) = bake("extracted/assets/Map1_1/fontene.xmed") else {
        return;
    };
    let mat = &json["materials"][0];
    assert_eq!(mat["name"], "fontene");
    assert_eq!(mat["alphaMode"], "BLEND");
    assert_eq!(mat["doubleSided"], true);
    let tex = mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        .as_u64()
        .unwrap() as usize;
    let img = &json["images"][json["textures"][tex]["source"].as_u64().unwrap() as usize];
    assert_eq!(
        img["mimeType"], "image/png",
        "JPEG colour + zlib alpha plane merged to RGBA"
    );
}
