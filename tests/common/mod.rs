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

/// Root of a tallitytot game directory (`games/tallitytot`) holding
/// `extracted/assets`, `assets/models/obj` and `assets/models/gltf`.
pub fn fixtures_root() -> Option<PathBuf> {
    std::env::var_os("MACROMELT_FIXTURES").map(PathBuf::from)
}

/// Resolve `rel` under [`fixtures_root`], or `None` — after printing why — when
/// `MACROMELT_FIXTURES` is unset or the file is not there.
pub fn fixture(rel: &str) -> Option<PathBuf> {
    let Some(root) = fixtures_root() else {
        eprintln!("skip: MACROMELT_FIXTURES unset, cannot resolve {rel}");
        return None;
    };
    let path = root.join(rel);
    if !path.exists() {
        eprintln!("skip: {rel} missing under {}", root.display());
        return None;
    }
    Some(path)
}
