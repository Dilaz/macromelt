// Lives in tests/ and consumes the crate as a library via a new src/lib.rs
// (least invasive: main.rs left untouched, lib.rs re-exports the modules).
//
// Verify each climbon motion's kf[0] matches the destination skin's bind
// pose. If they differ by more than ~1 unit per axis, the animation was
// authored against a different rest skeleton than the one in c_together.xmed.
//
// Per docs/plans/2026-05-31-tillit-animation-quaternion-fix.md, Task 5B.
// Intentionally fails: monica_climbon's `center` track kf0 is ~3.5 units off
// christian's bind pose because the source XMED was authored against a
// different rest skeleton. Task 6B's bake-time retarget fixes the OUTPUT GLB
// (verified by tests/climbon_restpose_baked.rs). This test stays as
// documentation of the source-data quirk — marked #[ignore] so it doesn't
// fail CI. Run explicitly via:
//   cargo test --release climbon_restpose_kf0_matches_bind_pose -- --ignored

use macromelt::chunks::parse_xmed;
use macromelt::motion::decode_motion_full;

mod common;

fn load_xmed(rel: &str) -> Option<macromelt::chunks::XmedFile> {
    let path = common::fixture(rel)?;
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    Some(
        parse_xmed(&data)
            .unwrap_or_else(|e| panic!("parse_xmed failed for {}: {}", path.display(), e)),
    )
}

#[test]
#[ignore = "documents source-XMED quirk; bake retarget fixes it in the GLB — see climbon_restpose_baked.rs"]
fn climbon_restpose_kf0_matches_bind_pose() {
    let Some(rest) = load_xmed("extracted/assets/Christian/c_together.xmed") else {
        return;
    };
    let chhest = rest
        .bone_weights
        .iter()
        .find(|b| b.mesh_name == "chhest")
        .expect("chhest skin present in c_together.xmed");
    let christian = rest
        .bone_weights
        .iter()
        .find(|b| b.mesh_name == "christian")
        .expect("christian skin present in c_together.xmed");

    let Some(anim) = load_xmed("extracted/assets/Tillit/together_climbon.xmed") else {
        return;
    };
    let decoded: Vec<_> = anim.motions.iter().map(decode_motion_full).collect();

    let mut failures: Vec<String> = Vec::new();

    let skins: [(&str, &macromelt::chunks::BoneWeightChunk); 2] =
        [("chhest", chhest), ("christian", christian)];

    for motion_name in ["lynet_climbon", "monica_climbon"] {
        let motion = decoded
            .iter()
            .find(|m| m.name == motion_name)
            .unwrap_or_else(|| panic!("motion {} not present", motion_name));

        // Replicate the bake's per-motion routing: pick the skin with the
        // highest bone-name overlap against this motion's tracks.
        let mut best_skin: Option<(&str, &macromelt::chunks::BoneWeightChunk)> = None;
        let mut best_overlap: usize = 0;
        for (label, skin) in &skins {
            let overlap = motion
                .tracks
                .iter()
                .filter(|t| skin.bones.iter().any(|b| b.name == t.name))
                .count();
            if overlap > best_overlap {
                best_overlap = overlap;
                best_skin = Some((label, skin));
            }
        }
        let Some((skin_label, skin)) = best_skin else {
            continue;
        };

        for track in &motion.tracks {
            let Some(kf0) = track.keyframes.first() else {
                continue;
            };

            // Only compare against the routed destination skin.
            let Some(bone) = skin.bones.iter().find(|b| b.name == track.name) else {
                continue;
            };
            for axis in 0..3 {
                let kf_v = kf0.displacement[axis];
                let bind_v = bone.displacement[axis];
                let diff = (kf_v - bind_v).abs();
                if !(diff < 1.0) {
                    failures.push(format!(
                        "motion {:>14} track {:>20} kf0.disp axis {} = {:+.3} \
                         but {} bind = {:+.3} (Δ {:.3})",
                        motion_name, track.name, axis, kf_v, skin_label, bind_v, diff
                    ));
                }
            }
        }
    }

    if !failures.is_empty() {
        let count = failures.len();
        let joined = failures.join("\n  ");
        panic!(
            "{} climbon kf0 mismatches against bind pose (>=1.0 unit):\n  {}",
            count, joined
        );
    }
}
