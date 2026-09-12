//! Golden tests for the 0x36 shader, 0x20 texture-declaration and 0x21
//! texture-image parsers (`chunks.rs`).
//!
//! A 4-channel texture ships as two 0x21 chunks under one name: the RGB JPEG and
//! a zlib-deflated 8-bit alpha plane. Phase 4.2 merges the two into a masked
//! glTF material, so the binding chain material → 0x36 shader → texture name →
//! 0x20 dimensions → 0x21 payload has to resolve from the parse alone.

use macromelt::chunks::{TextureImage, XmedFile, parse_xmed};

mod common;

fn parse(rel: &str) -> Option<XmedFile> {
    let path = common::fixture(rel)?;
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    Some(parse_xmed(&data).unwrap_or_else(|e| panic!("parse {}: {}", path.display(), e)))
}

#[test]
fn together_trav2_texture_chunks() {
    let Some(xmed) = parse("extracted/assets/Christian/together_trav2.xmed") else {
        return;
    };

    // 0x20: `hale` is a 16x16 4-channel texture — RGB plus an alpha plane.
    let decl = xmed
        .texture_decls
        .iter()
        .find(|t| t.name == "hale")
        .expect("hale texture declaration");
    assert_eq!((decl.width, decl.height, decl.channels), (16, 16, 4));

    // 0x21: `hale` therefore appears twice, once per payload kind.
    let plane = xmed
        .textures
        .iter()
        .find(|t| t.name == "hale" && matches!(t.image, TextureImage::Plane { .. }))
        .expect("hale alpha plane");
    match &plane.image {
        TextureImage::Plane {
            width,
            height,
            zlib,
        } => {
            assert_eq!((*width, *height), (16, 16));
            assert_eq!(&zlib[..2], &[0x78, 0xDA], "zlib header");
        }
        other => panic!("expected a plane, got {:?}", other),
    }
    let hale_jpeg = xmed
        .textures
        .iter()
        .find(|t| t.name == "hale" && matches!(t.image, TextureImage::Jpeg { .. }))
        .expect("hale jpeg");
    match &hale_jpeg.image {
        TextureImage::Jpeg { data } => assert_eq!(&data[..3], &[0xFF, 0xD8, 0xFF]),
        other => panic!("expected a jpeg, got {:?}", other),
    }

    // The mesh named `lynet` resolves to a JPEG through its shading groups.
    // There is no texture literally named "lynet" — that is the mesh name; its
    // first shading group binds material `hest_set`, whose 0x36 shader names
    // texture `hest_set`, which carries the JPEG.
    let lynet = xmed
        .mesh_descriptions
        .iter()
        .find(|m| m.name == "lynet")
        .expect("lynet mesh description");
    let material = lynet.material_names()[0];
    assert_eq!(material, "hest_set");
    let shader = xmed
        .shaders
        .iter()
        .find(|s| s.name == material)
        .expect("hest_set shader");
    assert_eq!(shader.material, "hest_set");
    assert_eq!(shader.texture, "hest_set");
    assert_eq!(shader.texture_layers, 1);
    let lynet_jpeg = xmed
        .textures
        .iter()
        .find(|t| t.name == shader.texture)
        .expect("hest_set texture image");
    match &lynet_jpeg.image {
        TextureImage::Jpeg { data } => assert_eq!(&data[..3], &[0xFF, 0xD8, 0xFF]),
        other => panic!("expected a jpeg for the lynet mesh, got {:?}", other),
    }

    // Every 4-channel declaration has a matching alpha plane, every 3-channel one
    // does not.
    for decl in &xmed.texture_decls {
        let has_plane = xmed
            .textures
            .iter()
            .any(|t| t.name == decl.name && matches!(t.image, TextureImage::Plane { .. }));
        assert_eq!(has_plane, decl.channels == 4, "{}", decl.name);
    }
}

/// An untextured shader (0x36 flags 1) carries a material name and nothing else.
#[test]
fn arrow_shader_has_no_texture() {
    let Some(xmed) = parse("extracted/assets/Hey/arrow.xmed") else {
        return;
    };
    assert_eq!(xmed.shaders.len(), 1);
    let shader = &xmed.shaders[0];
    assert_eq!(shader.name, "pil");
    assert_eq!(shader.material, "pil");
    assert_eq!(shader.flags, 1);
    assert_eq!(shader.texture_layers, 0);
    assert!(shader.texture.is_empty());
    assert!(shader.raw_flags.is_empty());
    assert!(xmed.texture_decls.is_empty());
    assert!(xmed.textures.is_empty());
}

/// A shader whose texture name differs from its material name — the binding must
/// follow the 0x36 `texture` field, not the material.
#[test]
fn helmet_shader_binds_a_differently_named_texture() {
    let Some(xmed) = parse("extracted/assets/Christian/together_trav2.xmed") else {
        return;
    };
    let shader = xmed
        .shaders
        .iter()
        .find(|s| s.name == "helmet")
        .expect("helmet shader");
    assert_eq!(shader.material, "helmet");
    assert_eq!(shader.texture, "helmet1");
    assert!(xmed.texture_decls.iter().any(|t| t.name == "helmet1"));
    assert!(!xmed.texture_decls.iter().any(|t| t.name == "helmet"));
}
