//! `--native-check <obj_dir>`: compare the native 0x49 decode of every mesh
//! group in an XMED against the Wine-converter OBJ `<obj_dir>/<name>/<name>.obj`
//! (`<name>` = the XMED file stem), when one exists.
//!
//! Same rule set as `tools/re/traces/fit_obj.py` (the corpus sweep is
//! `tools/re/traces/fit_corpus.py`; this flag is the single-file command the
//! parity plan names):
//!
//! * The OBJ `g` block is matched by mesh name, underscore-tolerant (the
//!   converter strips `_`: `monica_wfork` -> `monicawfork`).
//! * **Static** mesh (no bone weights): decoded positions are taken through
//!   the mesh node's 0x72 matrix (row-vector convention) and compared
//!   IN ORDER against the block's vertices, tolerance `1e-3 * bbox`, after
//!   removing a common median shift. Different counts mean the converter
//!   dumped a lower CLOD resolution and/or welded duplicates: every converter
//!   vertex must then still have an exact decoded twin (reverse nearest
//!   neighbour); decoded vertices without a twin are the finer-LOD detail.
//! * **Skinned** mesh: the converter exports it bone-posed, so positions are
//!   not comparable; it is reported `skinned - count-only`, plus - when the
//!   counts agree, which pins the vertex order - the raw face index sets are
//!   compared.
//! * An OBJ with no matching block, or a block without vertices, is reported
//!   as truncated/empty and not scored.
//!
//! Exit status: 0 when every scored mesh agrees, 1 otherwise.

use macromelt::{DecodedGeometry, LiveMesh, XmedFile, decode_mesh_group};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// One `g` block of the converter OBJ: global 0-based `v` indices and faces.
struct ObjBlock {
    verts: Vec<usize>,
    faces: Vec<[usize; 3]>,
}

fn load_obj(path: &Path) -> Result<(Vec<[f64; 3]>, HashMap<String, ObjBlock>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let mut verts: Vec<[f64; 3]> = Vec::new();
    let mut blocks: HashMap<String, ObjBlock> = HashMap::new();
    let mut current: Option<String> = None;
    let mut seen: Vec<HashSet<usize>> = Vec::new();
    let mut order: Vec<String> = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let xyz: Vec<f64> = it.take(3).filter_map(|t| t.parse().ok()).collect();
                if xyz.len() == 3 {
                    verts.push([xyz[0], xyz[1], xyz[2]]);
                }
            }
            Some("g") => {
                let name = it.next().unwrap_or("").to_string();
                if !blocks.contains_key(&name) {
                    blocks.insert(
                        name.clone(),
                        ObjBlock {
                            verts: Vec::new(),
                            faces: Vec::new(),
                        },
                    );
                    order.push(name.clone());
                    seen.push(HashSet::new());
                }
                current = Some(name);
            }
            Some("f") => {
                let Some(name) = current.as_ref() else {
                    continue;
                };
                let idx: Vec<usize> = it
                    .filter_map(|t| t.split('/').next()?.parse::<isize>().ok())
                    .map(|i| {
                        if i < 0 {
                            (verts.len() as isize + i) as usize
                        } else {
                            (i - 1) as usize
                        }
                    })
                    .collect();
                let bi = order.iter().position(|n| n == name).unwrap();
                let block = blocks.get_mut(name).unwrap();
                for &i in &idx {
                    if seen[bi].insert(i) {
                        block.verts.push(i);
                    }
                }
                // Fan-triangulate polygons, as the converter's are already triangles.
                for k in 1..idx.len().saturating_sub(1) {
                    block.faces.push([idx[0], idx[k], idx[k + 1]]);
                }
            }
            _ => {}
        }
    }
    // Vertex order inside a block is the global `v` order, as fit_obj.py's `sorted(set(...))`.
    for block in blocks.values_mut() {
        block.verts.sort_unstable();
    }
    Ok((verts, blocks))
}

fn find_block<'a>(blocks: &'a HashMap<String, ObjBlock>, name: &str) -> Option<&'a ObjBlock> {
    if let Some(b) = blocks.get(name) {
        return Some(b);
    }
    let key: String = name.chars().filter(|c| *c != '_').collect();
    blocks
        .iter()
        .find(|(n, _)| n.chars().filter(|c| *c != '_').collect::<String>() == key)
        .map(|(_, b)| b)
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn nearest(p: [f64; 3], set: &[[f64; 3]]) -> f64 {
    set.iter()
        .map(|q| dist(p, *q))
        .fold(f64::INFINITY, f64::min)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Run the check; prints one line per mesh group. Returns `true` when every
/// scored mesh agrees.
pub fn native_check(xmed: &XmedFile, xmed_path: &Path, obj_dir: &Path) -> bool {
    let stem = xmed_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let obj_path = obj_dir.join(stem).join(format!("{}.obj", stem));
    let obj = if obj_path.exists() {
        match load_obj(&obj_path) {
            Ok(o) => Some(o),
            Err(e) => {
                println!("OBJ unreadable: {}", e);
                None
            }
        }
    } else {
        println!("no converter OBJ at {} - counts only", obj_path.display());
        None
    };

    let mut all_ok = true;
    for group in xmed.geometry_groups() {
        let name = group[0].name.as_str();
        let mut live = LiveMesh::new();
        decode_mesh_group(xmed, &group, false, &mut live);
        let geo = DecodedGeometry::from(&live);
        let declared = xmed
            .mesh_descriptions
            .iter()
            .find(|m| m.name == name)
            .map(|m| (m.num_positions as usize, m.num_faces as usize));
        let counts_ok = declared
            .map(|(p, f)| p == geo.positions.len() && f == geo.faces.len())
            .unwrap_or(true);
        let count_note = match declared {
            Some((p, f)) => format!(
                "decl {}/{} dec {}/{}{}",
                p,
                f,
                geo.positions.len(),
                geo.faces.len(),
                if counts_ok { "" } else { " COUNT MISMATCH" }
            ),
            None => format!(
                "dec {}/{} (no 0x45 declaration)",
                geo.positions.len(),
                geo.faces.len()
            ),
        };
        if !counts_ok {
            all_ok = false;
        }
        let skinned = xmed
            .bone_weights
            .iter()
            .any(|bw| bw.mesh_name == name && !bw.bones.is_empty());

        let Some((verts, blocks)) = obj.as_ref() else {
            println!(
                "{} {}: {}",
                if counts_ok { "OK  " } else { "FAIL" },
                name,
                count_note
            );
            continue;
        };
        let Some(block) = find_block(blocks, name) else {
            println!(
                "SKIP {}: {} | no `g {}` block in the OBJ (truncated/empty export)",
                name, count_note, name
            );
            continue;
        };
        if block.verts.is_empty() {
            println!(
                "SKIP {}: {} | OBJ block has no vertices (truncated/empty export)",
                name, count_note
            );
            continue;
        }

        if skinned {
            let dec_n = geo.positions.len();
            let obj_n = block.verts.len();
            if dec_n != obj_n {
                println!(
                    "OK   {}: {} | skinned - count-only (obj {} vertices, bone-posed, not comparable)",
                    name, count_note, obj_n
                );
                continue;
            }
            let base = block.verts[0];
            let obj_faces: HashSet<[usize; 3]> = block
                .faces
                .iter()
                .map(|f| {
                    let mut t = [f[0] - base, f[1] - base, f[2] - base];
                    t.sort_unstable();
                    t
                })
                .collect();
            let dec_faces: HashSet<[usize; 3]> = geo
                .faces
                .iter()
                .map(|f| {
                    let mut t = [f[0] as usize, f[1] as usize, f[2] as usize];
                    t.sort_unstable();
                    t
                })
                .collect();
            let only_dec = dec_faces.difference(&obj_faces).count();
            let only_obj = obj_faces.difference(&dec_faces).count();
            let ok = only_dec == 0 && only_obj == 0;
            all_ok &= ok;
            println!(
                "{} {}: {} | skinned - count-only n={} face sets dec {} obj {} dec-only {} obj-only {}",
                if ok { "OK  " } else { "FAIL" },
                name,
                count_note,
                dec_n,
                dec_faces.len(),
                obj_faces.len(),
                only_dec,
                only_obj
            );
            continue;
        }

        // Static: node matrix (row-vector convention, as `--info` prints it).
        let node = xmed
            .mesh_nodes
            .iter()
            .find(|n| n.mesh_ref == name || n.name == name);
        let dec: Vec<[f64; 3]> = geo
            .positions
            .iter()
            .map(|p| {
                let p = [p[0] as f64, p[1] as f64, p[2] as f64];
                match node {
                    Some(n) => {
                        let m: Vec<f64> = n.matrix.iter().map(|v| *v as f64).collect();
                        [
                            p[0] * m[0] + p[1] * m[4] + p[2] * m[8] + m[12],
                            p[0] * m[1] + p[1] * m[5] + p[2] * m[9] + m[13],
                            p[0] * m[2] + p[1] * m[6] + p[2] * m[10] + m[14],
                        ]
                    }
                    None => p,
                }
            })
            .collect();
        let objv: Vec<[f64; 3]> = block.verts.iter().map(|&i| verts[i]).collect();
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for v in &objv {
            for a in 0..3 {
                lo[a] = lo[a].min(v[a]);
                hi[a] = hi[a].max(v[a]);
            }
        }
        let bbox = dist(lo, hi);
        let tol = 1e-3 * bbox.max(1.0);

        if dec.len() == objv.len() {
            let shift = [
                median(objv.iter().zip(&dec).map(|(o, d)| o[0] - d[0]).collect()),
                median(objv.iter().zip(&dec).map(|(o, d)| o[1] - d[1]).collect()),
                median(objv.iter().zip(&dec).map(|(o, d)| o[2] - d[2]).collect()),
            ];
            let mut off = 0;
            let mut offs = 0;
            let mut max_d = 0.0f64;
            let mut max_ds = 0.0f64;
            for (o, d) in objv.iter().zip(&dec) {
                let dd = dist(*o, *d);
                let ds = dist(*o, [d[0] + shift[0], d[1] + shift[1], d[2] + shift[2]]);
                if dd > tol {
                    off += 1;
                }
                if ds > tol {
                    offs += 1;
                }
                max_d = max_d.max(dd);
                max_ds = max_ds.max(ds);
            }
            let ok = offs == 0;
            all_ok &= ok;
            println!(
                "{} {}: {} | static in-order n={} off={} (max {:.4}) | shift ({:.3}, {:.3}, {:.3}) off={} (max {:.4}) bbox {:.1}",
                if ok { "OK  " } else { "FAIL" },
                name,
                count_note,
                dec.len(),
                off,
                max_d,
                shift[0],
                shift[1],
                shift[2],
                offs,
                max_ds,
                bbox
            );
        } else {
            let rev: Vec<f64> = objv.iter().map(|o| nearest(*o, &dec)).collect();
            let fwd: Vec<f64> = dec.iter().map(|d| nearest(*d, &objv)).collect();
            let off = rev.iter().filter(|d| **d > tol).count();
            let fwd_off = fwd.iter().filter(|d| **d > tol).count();
            let ok = off == 0;
            all_ok &= ok;
            println!(
                "{} {}: {} | static COUNT dec {} obj {} obj->dec max {:.4} off={} (dec->obj off={}: LOD detail) bbox {:.1}",
                if ok { "OK  " } else { "FAIL" },
                name,
                count_note,
                dec.len(),
                objv.len(),
                rev.iter().cloned().fold(0.0, f64::max),
                off,
                fwd_off,
                bbox
            );
        }
    }
    all_ok
}
