//! `--dump-keyframes`: per-track keyframe diagnostics for a decoded motion.

use macromelt::DecodedMotion;

/// Print per-track keyframe diagnostics for any track whose name contains `substring`.
/// An empty substring matches every track. Output goes to stdout (greppable).
///
/// For each matched track we print one header line, then one line per keyframe:
///   kf[i]  t=<time>  d=(dx,dy,dz)  r=(rw,rx,ry,rz)  s=(sx,sy,sz)
///          Δd_from_kf0=(...)  Δrot_axis_angle_from_prev=(axis,angle_deg)
///
/// Δrot is the axis-angle form of `q_i * conjugate(q_{i-1})` (wxyz quaternion
/// storage). The kf[0] entry shows `(—, 0.0)` since there is no previous frame.
pub fn dump_keyframes(motions: &[&DecodedMotion], substring: &str) {
    for motion in motions {
        for track in &motion.tracks {
            if !track.name.contains(substring) {
                continue;
            }
            println!(
                "=== motion \"{}\" track \"{}\" ({} keyframes) ===",
                motion.name,
                track.name,
                track.keyframes.len()
            );
            let kf0 = match track.keyframes.first() {
                Some(k) => k.clone(),
                None => continue,
            };
            let mut prev_rot = kf0.rotation;
            for (i, kf) in track.keyframes.iter().enumerate() {
                let dd = [
                    kf.displacement[0] - kf0.displacement[0],
                    kf.displacement[1] - kf0.displacement[1],
                    kf.displacement[2] - kf0.displacement[2],
                ];
                let delta_rot_str = if i == 0 {
                    "(—, 0.0)".to_string()
                } else {
                    // q_delta = q_i * conjugate(q_{i-1}); quaternions stored wxyz.
                    let q_conj = [prev_rot[0], -prev_rot[1], -prev_rot[2], -prev_rot[3]];
                    let qd = quat_mul_wxyz(kf.rotation, q_conj);
                    let w = qd[0].clamp(-1.0, 1.0);
                    let angle_rad = 2.0 * w.acos();
                    let angle_deg = angle_rad.to_degrees();
                    let axis_mag = (qd[1] * qd[1] + qd[2] * qd[2] + qd[3] * qd[3]).sqrt();
                    let axis = if axis_mag > 1e-8 {
                        [qd[1] / axis_mag, qd[2] / axis_mag, qd[3] / axis_mag]
                    } else {
                        [0.0, 0.0, 0.0]
                    };
                    format!(
                        "(({:+.4},{:+.4},{:+.4}), {:.3})",
                        axis[0], axis[1], axis[2], angle_deg
                    )
                };
                println!(
                    "  kf[{:>3}]  t={:>8.4}  d=({:+8.3},{:+8.3},{:+8.3})  \
                     r=({:+.4},{:+.4},{:+.4},{:+.4})  s=({:+.3},{:+.3},{:+.3})  \
                     Δd_from_kf0=({:+8.3},{:+8.3},{:+8.3})  Δrot_axis_angle_from_prev={}",
                    i,
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
                    dd[0],
                    dd[1],
                    dd[2],
                    delta_rot_str,
                );
                prev_rot = kf.rotation;
            }
        }
    }
}

/// Hamilton product of two wxyz quaternions.
fn quat_mul_wxyz(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}
