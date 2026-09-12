//! Live author-CLOD mesh state for the native XMED 0x49 progressive decoder.
//!
//! The original Shockwave 3D decoder (`FUN_7a18a990`, decompile
//! `tools/re/decoders/dec_50_55_7a18a990.c`) does not append a static author
//! face list: it MUTATES a live position-face array in place as vertices split,
//! and the array's final state IS the render mesh. Proven by
//! `tools/re/traces/arrow_live_facearray_trace.log` (13 winedbg snapshots of the
//! array mid-decode) and by the final state matching
//! `assets/models/obj/arrow/arrow.obj` face-for-face.
//!
//! This module holds that live state plus the two U3D containers the engine
//! keeps alongside it (`tools/u3d-sdk/RTL/Component/CLODAuthor/CIFXSetX.cpp`,
//! `CIFXSetAdjacencyX.cpp`).

use crate::geometry::DecodedGeometry;

/// Sorted, duplicate-free index set — port of U3D `CIFXSetX`
/// (`tools/u3d-sdk/RTL/Component/CLODAuthor/CIFXSetX.cpp`). Only the operations
/// the decoder actually performs are ported: add, remove, emptiness, ordered
/// member access.
///
/// `CIFXSetX::AddX` keeps the backing array sorted ascending, which is what
/// makes "local index into the set" a stable wire encoding.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SetX {
    members: Vec<u32>,
}

impl SetX {
    /// An empty set.
    pub(crate) fn new() -> Self {
        Self {
            members: Vec::new(),
        }
    }

    /// `CIFXSetX::AddX` — insert keeping ascending order; no-op if present.
    /// Returns true when the value was newly inserted.
    pub(crate) fn add(&mut self, value: u32) -> bool {
        match self.members.binary_search(&value) {
            Ok(_) => false,
            Err(at) => {
                self.members.insert(at, value);
                true
            }
        }
    }

    /// `CIFXSetX::RemoveX` — drop the value if present.
    pub(crate) fn remove(&mut self, value: u32) -> bool {
        match self.members.binary_search(&value) {
            Ok(at) => {
                self.members.remove(at);
                true
            }
            Err(_) => false,
        }
    }

    /// True when the set has no members.
    pub(crate) fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Iterate the members in ascending order.
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, u32> {
        self.members.iter()
    }

    /// The members as an ascending slice.
    pub(crate) fn as_slice(&self) -> &[u32] {
        &self.members
    }
}

/// Per-position incident-face sets — port of U3D `CIFXSetAdjacencyX`
/// (`tools/u3d-sdk/RTL/Component/CLODAuthor/CIFXSetAdjacencyX.cpp`), restricted
/// to `AddX` / `RemoveX` / `GetFaceSetX` / `GetPositionSetX`.
#[derive(Debug, Clone, Default)]
pub(crate) struct SetAdjacency {
    /// Indexed by position; each entry is the set of faces using that position.
    faces_by_position: Vec<SetX>,
    empty: SetX,
}

impl SetAdjacency {
    /// `CIFXSetAdjacencyX::AddX(position, face)`.
    pub(crate) fn add(&mut self, position: u32, face: u32) {
        let idx = position as usize;
        if idx >= self.faces_by_position.len() {
            self.faces_by_position.resize_with(idx + 1, SetX::new);
        }
        self.faces_by_position[idx].add(face);
    }

    /// `CIFXSetAdjacencyX::RemoveX(position, face)`.
    pub(crate) fn remove(&mut self, position: u32, face: u32) {
        if let Some(set) = self.faces_by_position.get_mut(position as usize) {
            set.remove(face);
        }
    }

    /// `CIFXSetAdjacencyX::GetFaceSetX(position)` — faces incident to `position`
    /// (U3D "set B" when `position` is the split position).
    pub(crate) fn face_set(&self, position: u32) -> &SetX {
        self.faces_by_position
            .get(position as usize)
            .unwrap_or(&self.empty)
    }

    /// `CIFXSetAdjacencyX::GetPositionSetX(faceSet)` — every position used by the
    /// faces in `face_set` (U3D "set C" before the split position is removed).
    /// U3D's adjacency object owns the mesh; here the face table is passed in.
    pub(crate) fn position_set(&self, face_set: &SetX, faces: &[[u32; 3]]) -> SetX {
        let mut out = SetX::new();
        for &f in face_set.iter() {
            if let Some(face) = faces.get(f as usize) {
                out.add(face[0]);
                out.add(face[1]);
                out.add(face[2]);
            }
        }
        out
    }
}

/// The live author-CLOD mesh the decoder mutates as it consumes resolution
/// updates. `faces` is the array the winedbg trace snapshots; `adj` is kept in
/// lockstep with it by `push_face` / `set_corner`.
///
/// A CLOD stream is progressive: it starts from a base mesh and then replays
/// *resolution update* records, each of which splits one position into two and
/// re-points some corners at the new position. So every array here grows as
/// decoding proceeds — `push_position` appends one entry to `positions` (and to
/// the per-position arrays the stream also carries) per update, `push_face`
/// appends the new faces the split creates, and `set_corner` rewrites corners
/// of faces that already exist. At any point the state is a valid mesh: the
/// state after the last record is the full-resolution render mesh.
#[derive(Debug, Clone, Default)]
pub struct LiveMesh {
    /// Vertex positions in Director's **Z-up** model space, in Director units.
    /// Index order is decode order, which is also the CLOD resolution order.
    pub positions: Vec<[f32; 3]>,
    /// Per-position normals, unit length, same Z-up space as `positions`.
    /// Empty when the stream carried none.
    pub normals: Vec<[f32; 3]>,
    /// Per-position texture coordinates `(u, v)` exactly as Director stores
    /// them. Empty when the stream carried none.
    pub texcoords: Vec<[f32; 2]>,
    /// Triangles as three indices into `positions`, in the stream's winding.
    /// Both the corners and the array itself change as updates are applied.
    pub faces: Vec<[u32; 3]>,
    /// Per-POSITION authored skin weights, parallel to `positions`:
    /// `(bone index, weight)` pairs summing to 1.
    ///
    /// `dec_50_55_7a18a990.c:709-793` reads, per new position, a count at ctx
    /// 0x12, that many ids at ctx 0x13 and a quantized weight at ctx 0x14 for
    /// every entry after the first (`w[0] = 1 - Σrest`). The ids index the
    /// model's **bone table**, not the shader slots: `monica_wkost` declares 3
    /// shading groups yet its ids run 1..31 over a 32-bone skeleton, and
    /// `lynet` reaches 32 with 19 slots. Face materials come from the 0x45
    /// shader slots (`submesh_of_face`); nothing here is shading.
    pub bone_weights: Vec<Vec<(u32, f32)>>,
    /// Shading-group id per position / per face, filled when the mesh's
    /// shading groups are concatenated into the render mesh. A group's faces
    /// draw with `MeshDescription::shaders[id].material`.
    pub(crate) submesh_of_position: Vec<u32>,
    /// Shading-group id per face, parallel to `faces`; see
    /// `submesh_of_position`.
    pub(crate) submesh_of_face: Vec<u32>,
    /// Incident-face set per position, maintained by `push_face` /
    /// `set_corner` so the split records can look up a position's faces.
    pub(crate) adj: SetAdjacency,
    /// Position count == the decoder's `m_uCurrentResolution` (the index the
    /// next new position takes).
    pub(crate) resolution: u32,
    /// Opt-in: keep every face's value AS APPENDED, before any split rewrite.
    /// Off by default (an empty `Vec` never allocates); turned on by the golden
    /// test that compares the append order against the binary's face-write trace
    /// `tools/re/traces/arrow_faces_trace.log`.
    pub record_appends: bool,
    /// Append-time face values; only filled while `record_appends` is set.
    pub appends: Vec<[u32; 3]>,
}

impl LiveMesh {
    /// An empty mesh: no positions, no faces, resolution 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a position; keeps `resolution` == `positions.len()`.
    pub(crate) fn push_position(&mut self, p: [f32; 3]) -> u32 {
        let idx = self.positions.len() as u32;
        self.positions.push(p);
        self.resolution = self.positions.len() as u32;
        idx
    }

    /// Append a face and register it in the adjacency
    /// (`CIFXAuthorCLODDecoder_P.cpp:1262-1275`, "update set adjacency for new
    /// faces"). Returns the new face index.
    pub(crate) fn push_face(&mut self, face: [u32; 3]) -> u32 {
        let idx = self.faces.len() as u32;
        self.faces.push(face);
        if self.record_appends {
            self.appends.push(face);
        }
        for c in face {
            self.adj.add(c, idx);
        }
        idx
    }

    /// Rewrite one corner of an existing face and move the face between the two
    /// positions' adjacency sets (`CIFXAuthorCLODDecoder_P.cpp:1249-1260`,
    /// "update set adjacency for move faces"). Returns false when the face index
    /// is out of range (a decode error, reported by the caller).
    pub(crate) fn set_corner(&mut self, face: u32, corner: usize, value: u32) -> bool {
        let Some(f) = self.faces.get_mut(face as usize) else {
            return false;
        };
        if corner > 2 {
            return false;
        }
        let old = f[corner];
        if old == value {
            return true;
        }
        f[corner] = value;
        let still_used = f.contains(&old);
        if !still_used {
            self.adj.remove(old, face);
        }
        self.adj.add(value, face);
        true
    }

    /// Corner `corner` (0..=2) of face `face`, or `None` when either index is
    /// out of range.
    pub(crate) fn face_corner(&self, face: u32, corner: usize) -> Option<u32> {
        self.faces.get(face as usize)?.get(corner).copied()
    }

    /// Positions no face references — an **orphan position** is a vertex that
    /// survived every split but ended up in no triangle. A non-empty result on
    /// a fully decoded mesh means the connectivity decode dropped or
    /// mis-targeted a corner, so the decoder reports it as the generalization
    /// gate when sweeping the other XMED names.
    pub fn orphan_positions(&self) -> Vec<u32> {
        (0..self.positions.len() as u32)
            .filter(|&p| self.adj.face_set(p).is_empty())
            .collect()
    }

    /// Faces with a repeated corner — a **degenerate face** covers zero area
    /// and renders as nothing, so like an orphan position it is a decode
    /// symptom rather than authored content. Same generalization gate.
    pub fn degenerate_faces(&self) -> Vec<u32> {
        self.faces
            .iter()
            .enumerate()
            .filter(|(_, f)| f[0] == f[1] || f[1] == f[2] || f[0] == f[2])
            .map(|(i, _)| i as u32)
            .collect()
    }
}

impl From<&LiveMesh> for DecodedGeometry {
    fn from(m: &LiveMesh) -> Self {
        DecodedGeometry {
            positions: m.positions.clone(),
            normals: m.normals.clone(),
            texcoords: m.texcoords.clone(),
            faces: m.faces.clone(),
            bone_weights: m.bone_weights.clone(),
            submesh_of_face: m.submesh_of_face.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setx_stays_sorted_and_deduped() {
        let mut s = SetX::new();
        assert!(s.add(7));
        assert!(s.add(2));
        assert!(!s.add(7));
        assert!(s.add(5));
        assert_eq!(s.as_slice(), &[2, 5, 7]);
        assert!(s.remove(5));
        assert!(!s.remove(5));
        assert_eq!(s.as_slice(), &[2, 7]);
    }

    #[test]
    fn adjacency_tracks_corner_rewrites() {
        let mut m = LiveMesh::new();
        for _ in 0..4 {
            m.push_position([0.0; 3]);
        }
        m.push_face([0, 1, 2]);
        assert_eq!(m.adj.face_set(0).as_slice(), &[0]);
        // move corner 0 from position 0 to position 3
        assert!(m.set_corner(0, 0, 3));
        assert!(m.adj.face_set(0).is_empty());
        assert_eq!(m.adj.face_set(3).as_slice(), &[0]);
        assert_eq!(m.faces[0], [3, 1, 2]);
        // set C = positions of the faces at 3, minus 3
        let mut set_c = m.adj.position_set(m.adj.face_set(3), &m.faces);
        set_c.remove(3);
        assert_eq!(set_c.as_slice(), &[1, 2]);
        assert_eq!(m.orphan_positions(), vec![0]);
        assert!(m.degenerate_faces().is_empty());
    }
}
