//! `macromelt` command line: inspect and bake Director Shockwave 3D cast members.

mod cli;

use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser};
use env_logger::Env;
use macromelt::chunks::MeshDescription;
use macromelt::clod_state::LiveMesh;
use macromelt::geometry;
use macromelt::gltf_export::ExportOptions;
use macromelt::{
    DecodedGeometry, DecodedMotion, ExportError, TextureImage, XmedFile, decode_mesh_group,
    decode_motion_full, export_glb, parse_xmed,
};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Decode a Macromedia Director Shockwave 3D cast member (XMED / W3D).
///
/// With no action flag the file's chunk tables are summarised (`--info`).
#[derive(Debug, Parser)]
#[command(name = "macromelt", version, about, long_about = None)]
struct Cli {
    /// The `.xmed` file to read.
    input: PathBuf,

    /// Summarise the file's chunk tables. Implied when no other action is given.
    #[arg(long)]
    info: bool,

    /// Decode every motion and print the keyframes as JSON on stdout.
    #[arg(long)]
    decode_motion: bool,

    /// Print each 0x4B skin's bone hierarchy as JSON on stdout.
    #[arg(long)]
    decode_weights: bool,

    /// Dump the raw layout of every 0x49 geometry chunk.
    #[arg(long)]
    probe_geometry: bool,

    /// Attempt a position-only decode of every geometry chunk and report it.
    #[arg(long)]
    try_decode_positions: bool,

    /// Report how each geometry chunk's CLOD update records are laid out.
    #[arg(long)]
    probe_clod: bool,

    /// Report each geometry chunk's base-mesh (resolution 0) block.
    #[arg(long)]
    probe_basemesh: bool,

    /// Run the hand-written reference decoder over every geometry chunk.
    #[arg(long)]
    probe_hand: bool,

    /// Print the 0x47 CLOD resolution schedule of every mesh.
    #[arg(long)]
    dump_schedule: bool,

    /// Decode every mesh group and print its element counts.
    #[arg(long)]
    decode_geom: bool,

    /// Write every decoded mesh group's arrays to this JSON file.
    #[arg(long, value_name = "out.json")]
    dump_geom: Option<PathBuf>,

    /// Write every 0x21 texture payload into this directory.
    #[arg(long, value_name = "dir")]
    dump_textures: Option<PathBuf>,

    /// Compare the native decode against converter OBJs in this directory; exit 1 on mismatch.
    #[arg(long, value_name = "obj_dir")]
    native_check: Option<PathBuf>,

    /// Bake the file to this glTF binary.
    #[arg(long, value_name = "out.glb")]
    export_glb: Option<PathBuf>,

    /// Keep every SkinRoot a direct child of SceneRoot and ground the rig at Z = 0.
    #[arg(long)]
    flat_skins: bool,

    /// Take motions from this additional XMED file as well; repeatable.
    #[arg(long, value_name = "anim.xmed")]
    anim: Vec<PathBuf>,

    /// Take only this motion from the preceding --anim, under this clip name; repeatable.
    #[arg(long, value_name = "motion=clip")]
    take: Vec<String>,

    /// Rename a clip after decoding, `from=to`; repeatable.
    #[arg(long, value_name = "from=to")]
    clip_name: Vec<String>,

    /// Print keyframe diagnostics for tracks whose name contains this substring ("" matches all).
    #[arg(long, value_name = "substring")]
    dump_keyframes: Option<String>,

    /// Only report warnings and errors.
    #[arg(short, long, conflicts_with = "verbose")]
    quiet: bool,

    /// Raise the log level: once for debug, twice for trace.
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

/// An `--anim` file together with the `--take` selections that followed it.
struct AnimFile {
    path: PathBuf,
    /// `(motion name in the file, clip name in the bake)`; empty takes every motion.
    takes: Vec<(String, String)>,
}

/// Pair each `--take` with the `--anim` it follows on the command line.
///
/// clap collects repeated options into flat vectors and loses their interleaving,
/// so the original argv positions are read back from `ArgMatches` to restore it.
/// A `--take` before any `--anim` has nothing to attach to and is a usage error.
fn group_anims(cli: &Cli, matches: &ArgMatches) -> Result<Vec<AnimFile>, String> {
    let anim_at: Vec<usize> = matches
        .indices_of("anim")
        .map(|i| i.collect())
        .unwrap_or_default();
    let take_at: Vec<usize> = matches
        .indices_of("take")
        .map(|i| i.collect())
        .unwrap_or_default();

    let mut anims: Vec<AnimFile> = cli
        .anim
        .iter()
        .map(|p| AnimFile {
            path: p.clone(),
            takes: Vec::new(),
        })
        .collect();

    for (t, raw) in take_at.iter().zip(&cli.take) {
        let Some((motion, clip)) = raw.split_once('=') else {
            return Err(format!("--take takes <motion>=<clip>, got {:?}", raw));
        };
        let owner = anim_at
            .iter()
            .rposition(|a| a < t)
            .ok_or_else(|| format!("--take {:?} must follow an --anim it applies to", raw))?;
        anims[owner]
            .takes
            .push((motion.to_string(), clip.to_string()));
    }
    Ok(anims)
}

/// Read and parse an XMED file, turning any failure into a CLI message.
fn read_xmed(path: &Path) -> Result<XmedFile, String> {
    let data = fs::read(path).map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    parse_xmed(&data).map_err(|e| format!("{}: {}", path.display(), e))
}

/// Decode the motions the `--anim` files contribute, applying their `--take` filters.
fn collect_extra_motions(
    anims: &[AnimFile],
    mesh_name: &str,
) -> Result<Vec<DecodedMotion>, String> {
    let mut out = Vec::new();
    for anim in anims {
        let xmed = read_xmed(&anim.path)?;
        let anim_name = anim
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        for motion in &xmed.motions {
            let mut decoded = decode_motion_full(motion);
            // A motion with no name of its own, or one merely echoing the mesh,
            // is identified by the file it came from.
            if decoded.name.is_empty() || decoded.name == mesh_name {
                decoded.name = anim_name.to_string();
            }
            if !anim.takes.is_empty() {
                match anim.takes.iter().find(|(m, _)| *m == decoded.name) {
                    Some((_, clip)) => decoded.name = clip.clone(),
                    None => {
                        log::debug!(
                            "skipping motion {:?} from {} (not in --take)",
                            decoded.name,
                            anim.path.display()
                        );
                        continue;
                    }
                }
            }
            log::info!(
                "decoding animation {:?} from {}",
                decoded.name,
                anim.path.display()
            );
            out.push(decoded);
        }
    }
    Ok(out)
}

fn main() -> ExitCode {
    let matches = Cli::command().get_matches();
    let cli = match Cli::from_arg_matches(&matches) {
        Ok(cli) => cli,
        Err(e) => e.exit(),
    };

    let level = if cli.quiet {
        "warn"
    } else {
        match cli.verbose {
            0 => "info",
            1 => "debug",
            _ => "trace",
        }
    };
    env_logger::Builder::from_env(Env::default().default_filter_or(level)).init();

    match run(&cli, &matches) {
        Ok(code) => code,
        Err(message) => {
            eprintln!("error: {}", message);
            ExitCode::FAILURE
        }
    }
}

fn run(cli: &Cli, matches: &ArgMatches) -> Result<ExitCode, String> {
    let anims = group_anims(cli, matches)?;
    let xmed = read_xmed(&cli.input)?;
    let mesh_name = cli
        .input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    if let Some(dir) = &cli.dump_textures {
        dump_textures(&xmed, dir)?;
    }

    // `--native-check` is a pass/fail report, so it short-circuits everything else.
    if let Some(obj_dir) = &cli.native_check {
        let ok = cli::native_check::native_check(&xmed, &cli.input, obj_dir);
        return Ok(if ok {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        });
    }

    let wants_motions = cli.export_glb.is_some() || cli.dump_keyframes.is_some();
    let extra_motions = if wants_motions {
        collect_extra_motions(&anims, mesh_name)?
    } else {
        Vec::new()
    };

    // Runs before the bake so one invocation can both dump and export.
    if let Some(substring) = &cli.dump_keyframes {
        let own: Vec<DecodedMotion> = xmed.motions.iter().map(decode_motion_full).collect();
        let all: Vec<&DecodedMotion> = own.iter().chain(extra_motions.iter()).collect();
        cli::keyframes::dump_keyframes(&all, substring);
    }

    if let Some(out) = &cli.export_glb {
        let mut clip_names: HashMap<String, String> = HashMap::new();
        for raw in &cli.clip_name {
            let Some((from, to)) = raw.split_once('=') else {
                return Err(format!("--clip-name takes <from>=<to>, got {:?}", raw));
            };
            clip_names.insert(from.to_string(), to.to_string());
        }
        log::info!("exporting {} -> {}", cli.input.display(), out.display());
        if !anims.is_empty() {
            log::info!("  + {} extra animation file(s)", anims.len());
        }
        let options = ExportOptions {
            extra_motions,
            clip_names,
            flat_skins: cli.flat_skins,
        };
        let glb = export_glb(&xmed, &options).map_err(|e: ExportError| e.to_string())?;
        fs::write(out, &glb).map_err(|e| format!("cannot write {}: {}", out.display(), e))?;
        log::info!("wrote {} bytes to {}", glb.len(), out.display());
        return Ok(ExitCode::SUCCESS);
    }

    // `--dump-keyframes` on its own is a complete action; do not fall through
    // to the `--info` default.
    let explicit_action = cli.info
        || cli.decode_motion
        || cli.decode_weights
        || cli.probe_geometry
        || cli.try_decode_positions
        || cli.probe_clod
        || cli.probe_basemesh
        || cli.probe_hand
        || cli.dump_schedule
        || cli.decode_geom
        || cli.dump_geom.is_some()
        || cli.dump_textures.is_some();
    if cli.dump_keyframes.is_some() && !explicit_action {
        return Ok(ExitCode::SUCCESS);
    }

    if cli.info || !explicit_action {
        print_info(&xmed);
    }

    if cli.decode_motion {
        decode_motion_report(&xmed)?;
    }

    if cli.probe_geometry {
        for g in &xmed.geometry {
            geometry::probe(g);
        }
    }

    if cli.try_decode_positions {
        for (g, md) in with_declarations(&xmed) {
            geometry::try_decode_positions(g, md);
        }
    }

    if cli.probe_clod {
        for (g, md) in with_declarations(&xmed) {
            geometry::probe_clod_progressive(g, md);
        }
    }

    if cli.probe_basemesh {
        for (g, md) in with_declarations(&xmed) {
            geometry::probe_basemesh(g, md);
        }
    }

    if cli.probe_hand {
        for (g, md) in with_declarations(&xmed) {
            geometry::probe_handdecode(g, md);
        }
    }

    if cli.dump_schedule {
        dump_schedule(&xmed);
    }

    if cli.decode_geom {
        for group in xmed.geometry_groups() {
            let name = group[0].name.as_str();
            let mut live = LiveMesh::new();
            decode_mesh_group(&xmed, &group, true, &mut live);
            println!(
                "{}: {} positions, {} faces, {} normals, {} texcoords, {} chunk(s), orphans {:?}, degenerate {:?}",
                name,
                live.positions.len(),
                live.faces.len(),
                live.normals.len(),
                live.texcoords.len(),
                group.len(),
                live.orphan_positions(),
                live.degenerate_faces()
            );
        }
    }

    if let Some(out) = &cli.dump_geom {
        dump_geom(&xmed, out)?;
    }

    if cli.decode_weights {
        for bw in &xmed.bone_weights {
            let skeleton = serde_json::json!({
                "mesh_name": bw.mesh_name,
                "bones": bw.bones.iter().enumerate().map(|(i, b)| {
                    serde_json::json!({
                        "index": i,
                        "name": b.name,
                        "parent_index": b.parent_index,
                        "rest_length": b.rest_length,
                        "displacement": b.displacement,
                        "orientation": b.orientation,
                    })
                }).collect::<Vec<_>>(),
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&skeleton).map_err(|e| e.to_string())?
            );
        }
    }

    Ok(ExitCode::SUCCESS)
}

/// Pair every geometry chunk with its 0x45 declaration, when it has one.
fn with_declarations(
    xmed: &XmedFile,
) -> Vec<(&macromelt::chunks::GeometryChunk, Option<&MeshDescription>)> {
    xmed.geometry
        .iter()
        .map(|g| (g, xmed.mesh_descriptions.iter().find(|m| m.name == g.name)))
        .collect()
}

/// `--dump-textures`: write every 0x21 payload for inspection.
fn dump_textures(xmed: &XmedFile, dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {}", dir.display(), e))?;
    for (i, tex) in xmed.textures.iter().enumerate() {
        let (suffix, bytes) = match &tex.image {
            TextureImage::Jpeg { data } => (String::from("jpg"), data),
            TextureImage::Plane {
                width,
                height,
                zlib,
            } => (format!("{}x{}.alpha.zlib", width, height), zlib),
        };
        let path = dir.join(format!("{:02}_{}.{}", i, tex.name, suffix));
        fs::write(&path, bytes).map_err(|e| format!("cannot write {}: {}", path.display(), e))?;
        log::info!("wrote {} ({}B)", path.display(), bytes.len());
    }
    Ok(())
}

/// `--dump-schedule`: the 0x47 CLOD resolution schedule of every mesh.
fn dump_schedule(xmed: &XmedFile) {
    for md in &xmed.mesh_descriptions {
        let counts: Vec<u32> = md.shaders.iter().map(|s| s.num_texcoords).collect();
        let Some(chunk) = xmed.clod_schedule(&md.name) else {
            println!("{}: no 0x47 schedule chunk", md.name);
            continue;
        };
        let sched = geometry::decode_clod_schedule(chunk, &counts);
        println!(
            "{}: {} submesh(es), counts={:?}, {}B",
            md.name,
            sched.len(),
            counts,
            chunk.compressed_data.len()
        );
        for (s, arr) in sched.iter().enumerate() {
            println!("  [{}] {:?}", s, arr);
        }
    }
}

/// `--dump-geom`: every mesh group's decoded arrays, for offline comparison.
fn dump_geom(xmed: &XmedFile, out: &Path) -> Result<(), String> {
    let meshes: Vec<serde_json::Value> = xmed
		.geometry_groups()
		.iter()
		.map(|group| {
			let mut live = LiveMesh::new();
			decode_mesh_group(xmed, group, false, &mut live);
			let geo = DecodedGeometry::from(&live);
			let node = xmed.mesh_nodes.iter().find(|n| n.mesh_ref == group[0].name);
			serde_json::json!({
				"name": group[0].name,
				"node_matrix": node.map(|n| n.matrix),
				"skinned": xmed.bone_weights.iter().any(|bw| bw.mesh_name == group[0].name && !bw.bones.is_empty()),
				"positions": geo.positions,
				"normals": geo.normals,
				"texcoords": geo.texcoords,
				"faces": geo.faces,
				"submesh_of_face": geo.submesh_of_face,
			})
		})
		.collect();
    let bytes = serde_json::to_vec(&meshes).map_err(|e| e.to_string())?;
    fs::write(out, bytes).map_err(|e| format!("cannot write {}: {}", out.display(), e))?;
    log::info!("wrote {} mesh(es) to {}", meshes.len(), out.display());
    Ok(())
}

/// `--decode-motion`: decode every motion, report track health, print JSON.
fn decode_motion_report(xmed: &XmedFile) -> Result<(), String> {
    let mut all_motions = Vec::new();
    // The first 0x4B chunk carries the skeleton the tracks are indexed against.
    let skeleton_bones: Vec<String> = xmed
        .bone_weights
        .first()
        .map(|bw| bw.bones.iter().map(|b| b.name.clone()).collect())
        .unwrap_or_default();

    for m in &xmed.motions {
        log::info!("decoding motion {:?}", m.name);
        let decoded = decode_motion_full(m);
        log::info!("  decoded {} tracks", decoded.tracks.len());

        log::debug!("  bone/track index alignment:");
        let max_len = decoded.tracks.len().max(skeleton_bones.len());
        for i in 0..max_len {
            let track_name = decoded
                .tracks
                .get(i)
                .map(|t| t.name.as_str())
                .unwrap_or("<none>");
            let bone_name = skeleton_bones
                .get(i)
                .map(|s| s.as_str())
                .unwrap_or("<none>");
            let mismatch = if track_name != bone_name {
                " MISMATCH"
            } else {
                ""
            };
            log::debug!(
                "    [{:>3}] track={:<24} bone={:<24}{}",
                i,
                format!("\"{}\"", track_name),
                format!("\"{}\"", bone_name),
                mismatch
            );
        }

        for track in &decoded.tracks {
            let mut valid = true;
            for (i, kf) in track.keyframes.iter().enumerate() {
                let qmag = (kf.rotation[0].powi(2)
                    + kf.rotation[1].powi(2)
                    + kf.rotation[2].powi(2)
                    + kf.rotation[3].powi(2))
                .sqrt();
                if (qmag - 1.0).abs() > 0.1 {
                    log::warn!("track {:?} kf[{}] |q|={:.4}", track.name, i, qmag);
                    valid = false;
                }
                if !kf.time.is_finite()
                    || !kf.displacement.iter().all(|v| v.is_finite())
                    || !kf.rotation.iter().all(|v| v.is_finite())
                {
                    log::warn!("track {:?} kf[{}] NaN/Inf", track.name, i);
                    valid = false;
                }
            }
            if valid {
                log::debug!(
                    "  track {:?} OK: {} kf, t=[{:.3}..{:.3}]",
                    track.name,
                    track.keyframes.len(),
                    track.keyframes.first().map(|k| k.time).unwrap_or(0.0),
                    track.keyframes.last().map(|k| k.time).unwrap_or(0.0),
                );
            }
        }
        all_motions.push(decoded);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&all_motions).map_err(|e| e.to_string())?
    );
    Ok(())
}

/// `--info`: summarise every chunk table the file carries.
fn print_info(xmed: &XmedFile) {
    println!("Bones: {}", xmed.bones.len());
    for b in &xmed.bones {
        println!(
            "  {} → {} ({:.2}, {:.2}, {:.2})",
            b.name, b.parent, b.position[0], b.position[1], b.position[2]
        );
    }
    println!("Meshes: {}", xmed.mesh_nodes.len());
    for m in &xmed.mesh_nodes {
        println!(
            "  {} → {} ref={} shader={}",
            m.name, m.parent, m.mesh_ref, m.shader
        );
        let mat = &m.matrix;
        for row in 0..4 {
            println!(
                "    [{:8.4} {:8.4} {:8.4} {:8.4}]",
                mat[row * 4],
                mat[row * 4 + 1],
                mat[row * 4 + 2],
                mat[row * 4 + 3]
            );
        }
    }
    println!("Geometry: {}", xmed.geometry.len());
    for g in &xmed.geometry {
        println!(
            "  {} {}v {}f {}n {}tc ({}B compressed)",
            g.name,
            g.num_positions,
            g.num_faces,
            g.num_normals,
            g.num_texcoords,
            g.compressed_data.len()
        );
    }
    println!("Mesh Descriptions: {}", xmed.mesh_descriptions.len());
    for md in &xmed.mesh_descriptions {
        println!(
            "  \"{}\" attrs=0x{:x} mask=0x{:x} pos={} faces={} norm={} tc={} col={} shaders={} arrays={}",
            md.name,
            md.attributes,
            md.attribute_mask(),
            md.num_positions,
            md.num_faces,
            md.num_normals,
            md.num_texcoords,
            md.num_colors,
            md.num_shaders,
            md.extra_count,
        );
        for (i, slot) in md.shaders.iter().enumerate() {
            println!(
                "    [{}] material={:?} shader={:?} mask=0x{:x} pos={} faces={} norm={} tc={} col={}",
                i,
                slot.material,
                slot.shader,
                slot.attrs,
                slot.num_positions,
                slot.num_faces,
                slot.num_normals,
                slot.num_texcoords,
                slot.num_colors,
            );
        }
        for sib in &md.sibling_meshes {
            println!("    + mesh \"{}\" materials={:?}", sib.name, sib.materials);
        }
        println!(
            "    bsphere={:.4} centre=({:.4}, {:.4}, {:.4}) scale={:.4} pos_iq={:.8} norm_iq={:.8} tc_iq={:.8} diff_iq={:.8} spec_iq={:.8} max_res={}",
            md.bounding_sphere,
            md.bounding_center[0],
            md.bounding_center[1],
            md.bounding_center[2],
            md.scale,
            md.position_inv_quant,
            md.normal_inv_quant,
            md.texcoord_inv_quant,
            md.diffuse_inv_quant,
            md.specular_inv_quant,
            md.max_resolution,
        );
    }
    println!("Shaders: {}", xmed.shaders.len());
    for s in &xmed.shaders {
        println!(
            "  \"{}\" flags=0x{:x} layers={} material={:?} texture={:?} ({}B raw)",
            s.name,
            s.flags,
            s.texture_layers,
            s.material,
            s.texture,
            s.raw_flags.len()
        );
    }
    println!("Texture Declarations: {}", xmed.texture_decls.len());
    for t in &xmed.texture_decls {
        println!(
            "  \"{}\" {}x{} channels={}",
            t.name, t.width, t.height, t.channels
        );
    }
    println!("Texture Images: {}", xmed.textures.len());
    for t in &xmed.textures {
        match &t.image {
            TextureImage::Jpeg { data } => {
                println!("  \"{}\" jpeg {}B", t.name, data.len())
            }
            TextureImage::Plane {
                width,
                height,
                zlib,
            } => println!(
                "  \"{}\" alpha-plane {}x{} ({}B zlib)",
                t.name,
                width,
                height,
                zlib.len()
            ),
        }
    }
    println!("Cameras: {}", xmed.cameras.len());
    for c in &xmed.cameras {
        println!(
            "  \"{}\" → {} pos=({:.4}, {:.4}, {:.4}) fov={:.4} hither={:.4} yon={:.4e} rect=({}, {}, {}, {}) proj={}",
            c.name,
            c.parent,
            c.position[0],
            c.position[1],
            c.position[2],
            c.fov,
            c.hither,
            c.yon,
            c.rect[0],
            c.rect[1],
            c.rect[2],
            c.rect[3],
            c.projection,
        );
    }
    println!("Materials: {}", xmed.materials.len());
    println!("Motions: {}", xmed.motions.len());
    for m in &xmed.motions {
        println!(
            "  \"{}\" tracks={} time_iq={:.6} rot_iq={:.6e}",
            m.name, m.track_count, m.time_inv_quant, m.rotation_inv_quant
        );
        if let Some(track) = &m.first_track {
            println!(
                "    first_track: \"{}\" keyframes={} disp_iq={:.6e} scale_iq={:.6e}",
                track.name, track.time_count, track.displacement_inv_quant, track.scale_inv_quant
            );
            let kf = &track.first_keyframe;
            println!(
                "    first_kf: t={:.4} pos=({:.4}, {:.4}, {:.4}) rot=({:.4}, {:.4}, {:.4}, {:.4}) scale=({:.4}, {:.4}, {:.4})",
                kf.time,
                kf.displacement[0],
                kf.displacement[1],
                kf.displacement[2],
                kf.rotation[0],
                kf.rotation[1],
                kf.rotation[2],
                kf.rotation[3],
                kf.scale[0],
                kf.scale[1],
                kf.scale[2],
            );
            println!("    compressed_data: {} bytes", track.compressed_data.len());
        }
    }
    println!("Bone weights: {}", xmed.bone_weights.len());
    for bw in &xmed.bone_weights {
        println!(
            "  mesh=\"{}\" bones={} weight_data={}B",
            bw.mesh_name,
            bw.bone_count,
            bw.weight_data.len()
        );
        for (i, b) in bw.bones.iter().enumerate() {
            let qmag = (b.orientation[0].powi(2)
                + b.orientation[1].powi(2)
                + b.orientation[2].powi(2)
                + b.orientation[3].powi(2))
            .sqrt();
            println!(
                "    [{i}] \"{}\" parent={} restLen={:.4} disp=({:.4},{:.4},{:.4}) |q|={:.6}",
                b.name,
                b.parent_index,
                b.rest_length,
                b.displacement[0],
                b.displacement[1],
                b.displacement[2],
                qmag
            );
        }
    }
}
