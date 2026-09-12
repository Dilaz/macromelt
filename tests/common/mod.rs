//! Fixture lookup shared by the integration tests.
//!
//! The tests decode real Director cast members and compare them against the
//! Shockwave 3D converter's exports. That material is copyrighted disc content
//! and never enters this repository, so it is addressed through the
//! `MACROMELT_FIXTURES` environment variable. Without it — and without the
//! individual file — the test prints a `skip:` line and returns, so
//! `cargo test` is green on a bare checkout.

#![allow(dead_code)]

use std::path::PathBuf;

/// Game directories (`games/tallitytot`, `games/tallitytot2`) holding
/// `extracted/assets` and `assets/models/gltf`, in `MACROMELT_FIXTURES` order.
///
/// The variable is a `:`-separated list so one run can cover discs that differ
/// in format — disc 2's meshes carry the attribute-face record that disc 1's do
/// not (`geometry::RecordEnv::attribute_faces`).
pub fn fixture_roots() -> Vec<PathBuf> {
    match std::env::var_os("MACROMELT_FIXTURES") {
        Some(v) => std::env::split_paths(&v).collect(),
        None => Vec::new(),
    }
}

/// Resolve `rel` under the first [`fixture_roots`] entry that has it, or `None`
/// — after printing why — when `MACROMELT_FIXTURES` is unset or no root has it.
pub fn fixture(rel: &str) -> Option<PathBuf> {
    let roots = fixture_roots();
    if roots.is_empty() {
        eprintln!("skip: MACROMELT_FIXTURES unset, cannot resolve {rel}");
        return None;
    }
    for root in &roots {
        let path = root.join(rel);
        if path.exists() {
            return Some(path);
        }
    }
    eprintln!("skip: {rel} missing under every MACROMELT_FIXTURES root");
    None
}
