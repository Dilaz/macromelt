//! Decoder for Macromedia Director **Shockwave 3D** cast members.
//!
//! A Director 8.x movie stores each 3D cast member as an `XMED` chunk of its
//! RIFX container. The payload is a `3DEM` stream built on Intel's IFX format —
//! the same technology that became U3D — carrying CLOD progressive meshes
//! behind an arithmetic-coded bitstream, skins with authored per-vertex weights,
//! bones and keyframe motions, JPEG textures with separate zlib alpha planes,
//! materials, cameras and transform nodes. This crate decodes that stream and
//! bakes it to a glTF 2.0 binary (`.glb`).
//!
//! Coordinates stay in Director's **Z-up** convention throughout; consumers
//! convert. Angles are degrees only where Director stores them so, times are
//! seconds, and quaternions are stored **wxyz**.
//!
//! # Pipeline
//!
//! ```text
//! parse_xmed(bytes)
//!   -> decode_mesh_group  (once per XmedFile::geometry_groups() entry)
//!   -> decode_motion_full (once per XmedFile::motions entry)
//!   -> export_glb
//! ```
//!
//! [`export_glb`] runs the mesh and material stages itself; you only need
//! [`decode_motion_full`] when you want to feed a *different* file's motions
//! into the bake through [`ExportOptions::extra_motions`], which is what
//! Director's `cloneMotionFromCastmember` does.
//!
//! # Example
//!
//! ```no_run
//! use macromelt::{ExportOptions, decode_motion_full, export_glb, parse_xmed};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let path = std::env::args().nth(1).expect("usage: bake <file.xmed>");
//! let bytes = std::fs::read(&path)?;
//! let xmed = parse_xmed(&bytes)?;
//!
//! // Motions living in this same file are baked automatically; this shows how
//! // to decode them yourself, e.g. to inspect or to re-bake from another file.
//! for motion in &xmed.motions {
//!     let decoded = decode_motion_full(motion);
//!     println!("clip {:?}: {} tracks", decoded.name, decoded.tracks.len());
//! }
//!
//! let glb = export_glb(&xmed, &ExportOptions::default())?;
//! std::fs::write("out.glb", glb)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Logging
//!
//! Nothing is printed: every diagnostic goes through the [`log`] facade. Install
//! any logger and raise the level to watch the decode
//! (`RUST_LOG=macromelt=debug`). The bit-level traces are gated behind their own
//! targets — `macromelt::bittrace`, `macromelt::posdump`,
//! `macromelt::facedump`, `macromelt::fulltrace`, `macromelt::mdraw` — so they
//! can be switched on one at a time.

#![warn(missing_docs)]

pub(crate) mod bitstream;
pub(crate) mod bone_weights;
pub mod chunks;
pub mod clod_state;
pub mod error;
pub mod geometry;
pub mod gltf_export;
pub mod motion;
pub(crate) mod skin_weights;

pub use chunks::{BoneDef, Material, MotionResource, TextureImage, XmedFile, parse_xmed};
pub use clod_state::LiveMesh;
pub use error::{ExportError, ParseError};
pub use geometry::{DecodedGeometry, decode_mesh_group};
pub use gltf_export::{ExportOptions, export_glb};
pub use motion::{DecodedMotion, DecodedTrack, decode_motion_full};
