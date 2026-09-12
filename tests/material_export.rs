//! Golden tests for glTF material emission (`gltf_export.rs`, plan step 4.2).
//!
//! The material's texture comes from the XMED itself: material name → 0x36
//! shader with that `material` → its `texture` → the 0x21 image chunk(s) of
//! that name. A 4-channel texture ships as two 0x21 chunks — the RGB JPEG plus
//! a zlib alpha plane — which are merged into an RGBA PNG and drawn with
//! `alphaMode: "MASK"`. Every material is double-sided, matching Shockwave 3D
//! (the tail, straps and helmet visor are single quads).

use macromelt::chunks::parse_xmed;
use macromelt::gltf_export::{ExportOptions, export_glb};
use serde_json::Value;

mod common;

/// The JSON chunk of a .glb file; the BIN chunk is checked to be there.
struct Glb {
    json: Value,
}

fn read_glb(data: &[u8]) -> Glb {
    assert_eq!(&data[0..4], b"glTF", "glB magic");
    let total = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    assert_eq!(total, data.len(), "glB length header");

    let mut json: Option<Value> = None;
    let mut has_bin = false;
    let mut off = 12usize;
    while off + 8 <= total {
        let len = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        let kind = &data[off + 4..off + 8];
        let body = &data[off + 8..off + 8 + len];
        match kind {
            b"JSON" => json = Some(serde_json::from_slice(body).expect("glB JSON chunk")),
            b"BIN\0" => has_bin = true,
            other => panic!("unexpected glB chunk {:?}", other),
        }
        off += 8 + len;
    }
    assert!(has_bin, "glB BIN chunk");
    Glb {
        json: json.expect("JSON chunk"),
    }
}

fn material<'a>(glb: &'a Glb, name: &str) -> &'a Value {
    glb.json["materials"]
        .as_array()
        .expect("materials array")
        .iter()
        .find(|m| m["name"] == name)
        .unwrap_or_else(|| panic!("material \"{}\"", name))
}

/// `helmet` proves the chain follows the shader's `texture` field: its material
/// is `helmet` but the bound texture is `helmet1`.
#[test]
fn helmet_material_binds_its_shader_texture_name() {
    let Some(glb) = bake_native("extracted/assets/Christian/together_trav2.xmed") else {
        return;
    };

    let helmet = material(&glb, "helmet");
    let tex_idx = helmet["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        .as_u64()
        .expect("helmet baseColorTexture") as usize;
    let texture = &glb.json["textures"].as_array().unwrap()[tex_idx];
    assert_eq!(texture["name"], "helmet1");

    // `helmet2` is 4-channel, so it is masked while `helmet` is not.
    assert!(helmet.get("alphaMode").is_none(), "{}", helmet);
    assert_eq!(material(&glb, "helmet2")["alphaMode"], "MASK");
}

/// Bake an XMED from its own 0x49 geometry.
fn bake_native(rel_xmed: &str) -> Option<Glb> {
    let path = common::fixture(rel_xmed)?;
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    let xmed = parse_xmed(&data).unwrap_or_else(|e| panic!("parse {rel_xmed}: {e}"));
    let glb = export_glb(&xmed, &ExportOptions::default()).expect("export_glb");
    Some(read_glb(&glb))
}

/// Christian's moped: four untextured shaders whose colour lives in the 0x10
/// material chunk. Ground truth is the SW3D converter's own
/// `assets/models/obj/christian_bike/christian_bike.MTL`, which writes
/// `Kd 1.000000 0.811765 0.356863` for `yellow` and
/// `Kd 0.321569 0.321569 0.321569` for `grey1`.
#[test]
fn untextured_shaders_carry_their_director_diffuse() {
    let Some(glb) = bake_native("extracted/assets/Map1_6/cycle.xmed") else {
        return;
    };

    let yellow = material(&glb, "yellow")["pbrMetallicRoughness"]["baseColorFactor"]
        .as_array()
        .expect("yellow baseColorFactor")
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect::<Vec<_>>();
    assert!((yellow[0] - 1.0).abs() < 1e-4, "{:?}", yellow);
    assert!((yellow[1] - 0.811765).abs() < 1e-4, "{:?}", yellow);
    assert!((yellow[2] - 0.356863).abs() < 1e-4, "{:?}", yellow);
    assert!((yellow[3] - 1.0).abs() < 1e-6, "{:?}", yellow);
    assert!(material(&glb, "yellow").get("alphaMode").is_none());

    let grey1 = material(&glb, "grey1")["pbrMetallicRoughness"]["baseColorFactor"][0]
        .as_f64()
        .expect("grey1 baseColorFactor");
    assert!((grey1 - 0.321569).abs() < 1e-4, "{}", grey1);
}

/// A shader that binds a texture keeps the default white base — the converter
/// writes no `Kd` for those either (`Material #8dfg` in the same MTL).
#[test]
fn textured_shaders_keep_a_white_base_colour() {
    let Some(glb) = bake_native("extracted/assets/Map1_2/christian_bike.xmed") else {
        return;
    };

    let christian = material(&glb, "Material #8dfg");
    assert!(
        christian["pbrMetallicRoughness"]["baseColorTexture"].is_object(),
        "{}",
        christian
    );
    assert!(
        christian["pbrMetallicRoughness"]
            .get("baseColorFactor")
            .is_none(),
        "{}",
        christian
    );
}

/// `opacity` is the shader's own alpha: the hint markers Director draws
/// see-through (`sporre`'s question mark at 0.6) must blend, not read as an
/// opaque block.
#[test]
fn shader_opacity_becomes_a_blended_alpha() {
    let Some(glb) = bake_native("extracted/assets/Map1_2/sporre.xmed") else {
        return;
    };

    let mat = material(&glb, "2 - Default");
    assert_eq!(mat["alphaMode"], "BLEND", "{}", mat);
    let alpha = mat["pbrMetallicRoughness"]["baseColorFactor"][3]
        .as_f64()
        .expect("baseColorFactor alpha");
    assert!((alpha - 0.6).abs() < 1e-4, "{}", alpha);
}
