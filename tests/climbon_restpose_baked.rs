//! Verify the BAKED GLB's animation kf[0] against the bake-time retarget
//! contract in `gltf_export.rs`:
//!
//! * the ROOT joint of every clip (`center`, `parent_index < 0`) is re-anchored
//!   onto the destination skin's bind TRS - kf0 translation/rotation equal the
//!   joint node's rest TRS (this is what puts `monica_climbon`, authored in
//!   horse-skeleton coordinates, on the rider);
//! * every other joint keeps the AUTHORED keyframe verbatim: kf0 rotation equals
//!   the source XMED track's kf0 (wxyz -> xyzw), kf0 translation equals the
//!   track's displacement plus the parent bone's rest length. Director's
//!   `bonesPlayer` applies motion rotations absolutely, so re-anchoring limbs to
//!   the bind pose would replace the authored pose with a T-pose; `c_together`
//!   plays the `together_*` clips authored on `together_trav2`'s rig, whose
//!   limb bind differs from `chhest`/`christian`, which is exactly the case the
//!   old "every kf0 == bind" expectation got wrong (253 limb rotations).
//!
//! Sources come from `scripts/bake-models.config.json` (rest XMED + animations +
//! clip renames), decoded with the same `decode_motion_full` the bake uses.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use macromelt::chunks::parse_xmed;
use macromelt::motion::{DecodedMotion, decode_motion_full};
use serde_json::Value;

mod common;

const TRANS_EPS: f32 = 0.05; // joint translation tolerance (units)
const ROT_EPS: f32 = 0.01; // 1 - |dot| tolerance on unit quaternions

/// One bone of one skin, as the bake sees it.
struct Bone {
    name: String,
    root: bool,
    parent_length: f32,
}

/// The clips a config entry bakes, by final clip name, plus each skin's bones.
struct Sources {
    motions: HashMap<String, DecodedMotion>,
    skins: Vec<Vec<Bone>>,
}

/// Mirror of the bake's motion naming: a motion named after the rest mesh (or
/// unnamed) takes its file stem; `clip_names` renames apply last. `root` is the
/// game directory the config's `extracted/assets/…` paths resolve against.
fn load_sources(root: &Path, config: &Path, entry_name: &str) -> Sources {
    let cfg: Value = serde_json::from_slice(&std::fs::read(config).unwrap()).unwrap();
    let entry = cfg["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"] == entry_name)
        .unwrap_or_else(|| panic!("no config entry {entry_name}"));
    let rest_path = root.join(entry["rest_xmed"].as_str().unwrap());
    let rest_stem = rest_path.file_stem().unwrap().to_str().unwrap().to_string();
    let rest = parse_xmed(&std::fs::read(&rest_path).unwrap()).unwrap();
    let renames: HashMap<String, String> = entry["clip_names"]
        .as_object()
        .map(|o| {
            o.iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                .collect()
        })
        .unwrap_or_default();

    let mut motions = HashMap::new();
    let mut add = |mut m: DecodedMotion, stem: &str| {
        if m.name.is_empty() || m.name == rest_stem {
            m.name = stem.to_string();
        }
        if let Some(to) = renames.get(&m.name) {
            m.name = to.clone();
        }
        motions.insert(m.name.clone(), m);
    };
    for m in &rest.motions {
        add(decode_motion_full(m), &rest_stem);
    }
    for anim in entry["animations"].as_array().unwrap() {
        let path = root.join(anim.as_str().unwrap());
        let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
        let x = parse_xmed(&std::fs::read(&path).unwrap()).unwrap();
        for m in &x.motions {
            add(decode_motion_full(m), &stem);
        }
    }
    let skins = rest
        .bone_weights
        .iter()
        .filter(|bw| !bw.bones.is_empty())
        .map(|bw| {
            bw.bones
                .iter()
                .map(|b| Bone {
                    name: b.name.clone(),
                    root: b.parent_index < 0,
                    parent_length: if b.parent_index >= 0 {
                        bw.bones[b.parent_index as usize].rest_length
                    } else {
                        0.0
                    },
                })
                .collect()
        })
        .collect();
    Sources { motions, skins }
}

/// Minimal GLB loader: returns (json_value, binary_chunk_bytes).
fn load_glb(path: &PathBuf) -> (Value, Vec<u8>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    assert!(bytes.len() >= 12, "GLB too small");
    assert_eq!(&bytes[0..4], b"glTF", "not a GLB file");
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    assert_eq!(version, 2, "GLB version != 2");

    // Chunk 0: JSON
    let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let json_type = &bytes[16..20];
    assert_eq!(json_type, b"JSON", "first chunk is not JSON");
    let json_start = 20;
    let json_end = json_start + json_len;
    let json_bytes = &bytes[json_start..json_end];
    let json: Value = serde_json::from_slice(json_bytes).expect("parse glTF JSON chunk");

    // Chunk 1: BIN
    let bin_chunk_hdr = json_end;
    let bin_len =
        u32::from_le_bytes(bytes[bin_chunk_hdr..bin_chunk_hdr + 4].try_into().unwrap()) as usize;
    let bin_type = &bytes[bin_chunk_hdr + 4..bin_chunk_hdr + 8];
    assert_eq!(bin_type, b"BIN\0", "second chunk is not BIN");
    let bin_data_start = bin_chunk_hdr + 8;
    let bin = bytes[bin_data_start..bin_data_start + bin_len].to_vec();

    (json, bin)
}

/// Read the first VEC3 entry from an accessor.
fn read_vec3_first(json: &Value, bin: &[u8], accessor_idx: usize) -> [f32; 3] {
    let acc = &json["accessors"][accessor_idx];
    assert_eq!(
        acc["componentType"].as_u64().unwrap(),
        5126,
        "VEC3 needs f32"
    );
    assert_eq!(acc["type"].as_str().unwrap(), "VEC3");
    let bv_idx = acc["bufferView"].as_u64().unwrap() as usize;
    let bv = &json["bufferViews"][bv_idx];
    let bv_off = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
    let acc_off = acc["byteOffset"].as_u64().unwrap_or(0) as usize;
    let off = bv_off + acc_off;
    let x = f32::from_le_bytes(bin[off..off + 4].try_into().unwrap());
    let y = f32::from_le_bytes(bin[off + 4..off + 8].try_into().unwrap());
    let z = f32::from_le_bytes(bin[off + 8..off + 12].try_into().unwrap());
    [x, y, z]
}

/// Read the first VEC4 entry from an accessor (glTF xyzw order).
fn read_vec4_first(json: &Value, bin: &[u8], accessor_idx: usize) -> [f32; 4] {
    let acc = &json["accessors"][accessor_idx];
    assert_eq!(
        acc["componentType"].as_u64().unwrap(),
        5126,
        "VEC4 needs f32"
    );
    assert_eq!(acc["type"].as_str().unwrap(), "VEC4");
    let bv_idx = acc["bufferView"].as_u64().unwrap() as usize;
    let bv = &json["bufferViews"][bv_idx];
    let bv_off = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
    let acc_off = acc["byteOffset"].as_u64().unwrap_or(0) as usize;
    let off = bv_off + acc_off;
    let x = f32::from_le_bytes(bin[off..off + 4].try_into().unwrap());
    let y = f32::from_le_bytes(bin[off + 4..off + 8].try_into().unwrap());
    let z = f32::from_le_bytes(bin[off + 8..off + 12].try_into().unwrap());
    let w = f32::from_le_bytes(bin[off + 12..off + 16].try_into().unwrap());
    [x, y, z, w]
}

fn array_to_vec3(v: &Value, default: [f32; 3]) -> [f32; 3] {
    if let Some(arr) = v.as_array() {
        if arr.len() == 3 {
            return [
                arr[0].as_f64().unwrap() as f32,
                arr[1].as_f64().unwrap() as f32,
                arr[2].as_f64().unwrap() as f32,
            ];
        }
    }
    default
}

fn array_to_vec4(v: &Value, default: [f32; 4]) -> [f32; 4] {
    if let Some(arr) = v.as_array() {
        if arr.len() == 4 {
            return [
                arr[0].as_f64().unwrap() as f32,
                arr[1].as_f64().unwrap() as f32,
                arr[2].as_f64().unwrap() as f32,
                arr[3].as_f64().unwrap() as f32,
            ];
        }
    }
    default
}

fn check_glb(name: &str, rel_glb: &str) {
    let Some(glb_path) = common::fixture(rel_glb) else {
        return;
    };
    let Some(config) = common::fixture("bake-models.config.json") else {
        return;
    };
    let root = config
        .parent()
        .expect("config lives in the game dir")
        .to_path_buf();
    let src = load_sources(&root, &config, name);
    let (json, bin) = load_glb(&glb_path);
    let nodes = json["nodes"].as_array().expect("nodes array");

    // glTF joint node -> (skin index, bone) through skins[].joints, which the
    // bake writes in bone_weights order.
    let mut joint_of: HashMap<usize, (usize, &Bone)> = HashMap::new();
    for (si, skin) in json["skins"].as_array().expect("skins").iter().enumerate() {
        for (bi, j) in skin["joints"].as_array().unwrap().iter().enumerate() {
            joint_of.insert(j.as_u64().unwrap() as usize, (si, &src.skins[si][bi]));
        }
    }

    let mut failures: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for anim in json["animations"].as_array().expect("animations") {
        let clip = anim["name"].as_str().unwrap();
        let motion = src
            .motions
            .get(clip)
            .unwrap_or_else(|| panic!("{name}: clip {clip} has no source motion"));
        let samplers = anim["samplers"].as_array().unwrap();
        for chan in anim["channels"].as_array().unwrap() {
            let output = samplers[chan["sampler"].as_u64().unwrap() as usize]["output"]
                .as_u64()
                .unwrap() as usize;
            let node_idx = chan["target"]["node"].as_u64().unwrap() as usize;
            let path = chan["target"]["path"].as_str().unwrap();
            let Some(&(_, bone)) = joint_of.get(&node_idx) else {
                continue; // keyframe (node) channels are covered by keyframe_export.rs
            };
            let joint = &nodes[node_idx];
            let track = motion.tracks.iter().find(|t| t.name == bone.name);
            match path {
                "translation" => {
                    let kf0 = read_vec3_first(&json, &bin, output);
                    let want = if bone.root {
                        array_to_vec3(&joint["translation"], [0.0; 3])
                    } else {
                        let d = track.expect("authored track").keyframes[0].displacement;
                        [d[0] + bone.parent_length, d[1], d[2]]
                    };
                    if (0..3).any(|k| (kf0[k] - want[k]).abs() > TRANS_EPS) {
                        failures.push(format!(
                            "[{}] {} {:>22} kf0.trans {:?} want {:?} ({})",
                            name,
                            clip,
                            bone.name,
                            kf0,
                            want,
                            if bone.root { "bind" } else { "authored" }
                        ));
                    }
                    checked += 1;
                }
                "rotation" => {
                    let kf0 = read_vec4_first(&json, &bin, output);
                    let want = if bone.root {
                        array_to_vec4(&joint["rotation"], [0.0, 0.0, 0.0, 1.0])
                    } else {
                        let [w, x, y, z] = track.expect("authored track").keyframes[0].rotation;
                        [x, y, z, w]
                    };
                    let dot: f32 = (0..4).map(|k| kf0[k] * want[k]).sum();
                    if (1.0 - dot.abs()) > ROT_EPS {
                        failures.push(format!(
                            "[{}] {} {:>22} kf0.rot {:?} want {:?} |dot|={:.5} ({})",
                            name,
                            clip,
                            bone.name,
                            kf0,
                            want,
                            dot.abs(),
                            if bone.root { "bind" } else { "authored" }
                        ));
                    }
                    checked += 1;
                }
                _ => {}
            }
        }
    }
    assert!(checked > 0, "{}: no joint animation channels checked", name);
    assert!(
        failures.is_empty(),
        "{} kf0 mismatches ({} checked):\n  {}",
        failures.len(),
        checked,
        failures.join("\n  ")
    );
    eprintln!(
        "{}: OK - {} channels: roots on bind, limbs authored",
        name, checked
    );
}

#[test]
fn c_together_animations_kf0_match_bind_pose() {
    check_glb("c_together", "assets/models/gltf/c_together.glb");
}

#[test]
fn lynet_animations_kf0_match_bind_pose() {
    check_glb("lynet", "assets/models/gltf/lynet.glb");
}

#[test]
fn monica_animations_kf0_match_bind_pose() {
    check_glb("monica", "assets/models/gltf/monica.glb");
}
