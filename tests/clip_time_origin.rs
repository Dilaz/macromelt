//! Every exported clip must start at t = 0.
//!
//! Director keyframe times are absolute in the motion's own timeline and
//! `bonesPlayer.play(name)` starts a motion at its first keyframe. A Three.js
//! `AnimationClip`, by contrast, spans `[0, max t]` and `LoopRepeat` replays
//! that whole span, so a clip whose earliest key sits at t > 0 holds its first
//! pose for the lead-in on every cycle.
//!
//! `Map1_1/monica_walk.xmed` is the case that showed: its keys run
//! 0.2333..1.1333 s (7 frames of lead-in at 30 fps), which made Monica freeze
//! for a quarter second after every two steps on the hub maps. The bake rebases
//! the clip on its own first key.

use macromelt::chunks::parse_xmed;
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
        let kind = &data[off + 4..off + 8];
        let body = &data[off + 8..off + 8 + len];
        if kind == b"JSON" {
            json = Some(serde_json::from_slice(body).unwrap());
        } else {
            bin = Some(body.to_vec());
        }
        off += 8 + len + (4 - len % 4) % 4;
    }
    (json.unwrap(), bin.unwrap())
}

fn read_f32s(json: &Value, bin: &[u8], accessor: usize) -> Vec<f32> {
    let acc = &json["accessors"][accessor];
    let bv = &json["bufferViews"][acc["bufferView"].as_u64().unwrap() as usize];
    let start = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
    let count = acc["count"].as_u64().unwrap() as usize;
    (0..count)
        .map(|i| f32::from_le_bytes(bin[start + 4 * i..start + 4 * i + 4].try_into().unwrap()))
        .collect()
}

#[test]
fn monica_walk_is_rebased_on_its_first_keyframe() {
    let Some(rest_path) = common::fixture("extracted/assets/Hey/monica.xmed") else {
        return;
    };
    let Some(anim_path) = common::fixture("extracted/assets/Map1_1/monica_walk.xmed") else {
        return;
    };
    let rest = parse_xmed(&std::fs::read(rest_path).unwrap()).unwrap();
    let anim = parse_xmed(&std::fs::read(anim_path).unwrap()).unwrap();
    let motions: Vec<_> = anim.motions.iter().map(decode_motion_full).collect();
    let walk = motions
        .iter()
        .find(|m| m.name == "monica_walk")
        .expect("monica_walk motion");

    // The source really does start late — this is what the bake has to absorb.
    let src_first = walk
        .tracks
        .iter()
        .filter_map(|t| t.keyframes.first().map(|k| k.time))
        .fold(f32::INFINITY, f32::min);
    let src_last = walk
        .tracks
        .iter()
        .filter_map(|t| t.keyframes.last().map(|k| k.time))
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (src_first - 0.2333).abs() < 1e-3,
        "source lead-in is 7 frames, got {src_first}"
    );
    assert!((src_last - 1.1333).abs() < 1e-3, "source ends at 1.1333 s");

    let glb = export_glb(
        &rest,
        &ExportOptions {
            extra_motions: motions,
            ..Default::default()
        },
    )
    .expect("export_glb");
    let (json, bin) = read_glb_json(&glb);

    let clip = json["animations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["name"] == "monica_walk")
        .expect("monica_walk clip");

    let mut first = f32::INFINITY;
    let mut last = f32::NEG_INFINITY;
    for sampler in clip["samplers"].as_array().unwrap() {
        let acc = sampler["input"].as_u64().unwrap() as usize;
        let times = read_f32s(&json, &bin, acc);
        assert!(!times.is_empty());
        // The accessor bounds a loader may trust must agree with the payload.
        let min = json["accessors"][acc]["min"][0].as_f64().unwrap() as f32;
        assert!((min - times[0]).abs() < 1e-6, "accessor min matches data");
        first = first.min(times[0]);
        last = last.max(*times.last().unwrap());
    }

    assert!(
        first.abs() < 1e-6,
        "clip must start at t = 0, got {first} — LoopRepeat would freeze for that long every cycle"
    );
    // The span is preserved: only the origin moved.
    assert!(
        (last - (src_last - src_first)).abs() < 1e-4,
        "clip span {last} should equal the source span {}",
        src_last - src_first
    );
}
