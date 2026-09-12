//! The XMED's 0x49 stream carries the model's AUTHORED skin weights - per
//! position a count (ctx 0x12), that many bone ids (ctx 0x13) and a quantized
//! weight (ctx 0x14) for every entry after the first, the first taking the
//! remainder. The bake binds with those, never with regenerated proximity
//! weights.
//!
//! The oracle is a held prop. `Rydde/monica_wkost.xmed` is Monica carrying a
//! broom whose bristles reach the floor beside her feet: proximity weighting
//! (the IFXSkin fallback) splits that material group between `right fingers`
//! and the foot/leg bones and the broom visibly stretches as she walks, while
//! the authored weights put all 52 of its vertices on the hand alone. The
//! same holds for the fork (`monica_wfork`) and the hay bale
//! (`monica_whay`).

use std::collections::HashMap;

use macromelt::chunks::parse_xmed;
use macromelt::gltf_export::{ExportOptions, export_glb};
use serde_json::Value;

mod common;

fn bake(rel: &str) -> Option<(Value, Vec<u8>)> {
    let path = common::fixture(rel)?;
    let xmed = parse_xmed(&std::fs::read(path).unwrap()).unwrap();
    let data = export_glb(&xmed, &ExportOptions::default()).expect("export_glb");

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
    Some((json.unwrap(), bin.unwrap()))
}

fn accessor_offset(json: &Value, idx: usize) -> (usize, usize) {
    let acc = &json["accessors"][idx];
    let bv = &json["bufferViews"][acc["bufferView"].as_u64().unwrap() as usize];
    let base = bv["byteOffset"].as_u64().unwrap_or(0) as usize
        + acc["byteOffset"].as_u64().unwrap_or(0) as usize;
    (base, acc["count"].as_u64().unwrap() as usize)
}

/// Joint names carrying weight in the primitive drawn with `material`, with the
/// vertex count each one influences.
fn influences(json: &Value, bin: &[u8], material: &str) -> HashMap<String, usize> {
    let joints: Vec<String> = json["skins"][0]["joints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| {
            json["nodes"][j.as_u64().unwrap() as usize]["name"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    let mut out: HashMap<String, usize> = HashMap::new();
    for mesh in json["meshes"].as_array().unwrap() {
        for prim in mesh["primitives"].as_array().unwrap() {
            let mat = &json["materials"][prim["material"].as_u64().unwrap() as usize];
            if mat["name"].as_str() != Some(material) {
                continue;
            }
            let (jo, count) = accessor_offset(
                json,
                prim["attributes"]["JOINTS_0"].as_u64().unwrap() as usize,
            );
            let (wo, _) = accessor_offset(
                json,
                prim["attributes"]["WEIGHTS_0"].as_u64().unwrap() as usize,
            );
            for v in 0..count {
                let mut sum = 0.0f32;
                for k in 0..4 {
                    let joint = bin[jo + v * 4 + k] as usize;
                    let o = wo + (v * 4 + k) * 4;
                    let w = f32::from_le_bytes(bin[o..o + 4].try_into().unwrap());
                    sum += w;
                    if w > 0.0 {
                        *out.entry(joints[joint].clone()).or_default() += 1;
                    }
                }
                assert!(
                    (sum - 1.0).abs() < 1e-3,
                    "{material} vertex {v} weights sum to {sum}"
                );
            }
        }
    }
    assert!(!out.is_empty(), "no primitive drawn with \"{material}\"");
    out
}

#[test]
fn held_props_bind_to_the_hand_alone() {
    for (rel, material, verts) in [
        ("extracted/assets/Rydde/monica_wkost.xmed", "kost", 52),
        ("extracted/assets/Rydde/monica_wfork.xmed", "fork", 87),
        ("extracted/assets/Rydde/monica_whay.xmed", "hay", 8),
    ] {
        let Some((json, bin)) = bake(rel) else {
            continue;
        };
        let inf = influences(&json, &bin, material);
        assert_eq!(
            inf,
            HashMap::from([("right fingers".to_string(), verts)]),
            "{material}: a held prop is rigid on the hand bone, got {inf:?}"
        );
    }
}

#[test]
fn the_body_keeps_its_authored_multi_bone_weighting() {
    let Some((json, bin)) = bake("extracted/assets/Rydde/monica_wkost.xmed") else {
        return;
    };
    let inf = influences(&json, &bin, "monica_01");
    assert!(
        inf.len() > 15,
        "the torso/limb group spreads over the skeleton, got {} bones",
        inf.len()
    );
    for bone in ["upper right leg", "upper body", "upper left arm"] {
        assert!(inf.contains_key(bone), "{bone} carries no weight: {inf:?}");
    }
}
