//! Morphing one genome into another as flam3 does for Electric Sheep's
//! edges (interpolation.c): pad both to the same xforms (flam3_align, with
//! its special padding for spherical-like and fan / rings variations),
//! interpolate each affine column in log-polar form (angles unwrapped, and
//! kept turning the same way when one side is a symmetry xform), palettes in
//! HSV with the hue taking the short way round, everything else linearly.

use crate::genome::{Affine, Genome, IDENTITY, VarUse, Xform, hsv2rgb, rgb2hsv};
use crate::var;
use core::f32::consts::PI;

const EPS: f32 = 1e-10;

/// A padding xform (initialize_xforms) shaped after its non-padding partner.
fn pad_for(other: &Xform, index: usize) -> Xform {
    let mut x = Xform::new();
    x.weight = 0.0;
    x.color = (index & 1) as f32;
    x.padding = true;
    x.vars.clear();
    let has = |id: i32| other.vars.iter().any(|v| v.id == id && v.w > 0.0);
    // spherical-like: a 180-degree turned identity with linear weight -1
    if [2, 38, 32, 33, 5, 79, 78].iter().any(|&id| has(id)) {
        x.vars.push(VarUse { id: 0, w: -1.0, p: [0.0; crate::genome::NP] });
        x.c = [[-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]];
        return x;
    }
    let mut found = 0;
    for (id, p) in [
        (40, [0.0; 8]),
        (26, [0.0; 8]),
        (25, [0.0; 8]),
        (23, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (30, [0.0, other.var(30).map_or(0.0, |v| v.p[1]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (39, [0.0; 8]),
        // super_shape: n1 = n2 = n3 = 2, rnd = holes = 0, m kept
        (50, [0.0, other.var(50).map_or(0.0, |v| v.p[1]), 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]),
    ] {
        if has(id) {
            x.vars.push(VarUse { id, w: 1.0, p });
            found += 1;
        }
    }
    if found == 0 {
        for id in [22, 21] {
            if has(id) {
                x.vars.push(VarUse { id, w: 1.0, p: [0.0; crate::genome::NP] });
                found += 1;
            }
        }
        if found > 0 {
            x.c = [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
        }
    }
    if found == 0 {
        x.vars.push(VarUse { id: 0, w: 1.0, p: [0.0; crate::genome::NP] });
    } else {
        let n = x.vars.len() as f32;
        for v in x.vars.iter_mut() {
            v.w /= n;
        }
    }
    x
}

fn pad_final() -> Xform {
    let mut x = Xform::new();
    x.weight = 0.0;
    x.animate = false;
    x.speed = 0.0;
    x.padding = true;
    x
}

/// Both genomes with the same xforms (and both with or without a final).
pub fn align(a: &Genome, b: &Genome) -> (Genome, Genome) {
    let (mut a, mut b) = (a.clone(), b.clone());
    let n = a.xforms.len().max(b.xforms.len());
    for i in a.xforms.len()..n {
        let x = pad_for(&b.xforms[i], i);
        a.xforms.push(x);
    }
    for i in b.xforms.len()..n {
        let x = pad_for(&a.xforms[i], i);
        b.xforms.push(x);
    }
    if a.fin.is_some() || b.fin.is_some() {
        a.fin.get_or_insert_with(pad_final);
        b.fin.get_or_insert_with(pad_final);
    }
    (a, b)
}

/// The angle and length of each linear column, with flam3's zero-length rule.
fn polar(c: &Affine) -> ([f32; 2], [f32; 2]) {
    let mut ang = [0.0; 2];
    let mut mag = [0.0; 2];
    for col in 0..2 {
        ang[col] = c[col][1].atan2(c[col][0]);
        mag[col] = (c[col][0] * c[col][0] + c[col][1] * c[col][1]).sqrt();
    }
    if mag[0] == 0.0 && mag[1] != 0.0 {
        ang[0] = ang[1];
    } else if mag[1] == 0.0 && mag[0] != 0.0 {
        ang[1] = ang[0];
    }
    (ang, mag)
}

/// flam3's log interpolation of one affine between `a` (weight 1 - t) and `b`.
/// `wind`: the asymmetric reference angle per column (establish_asymmetric_refangles).
fn interp_affine(a: &Affine, b: &Affine, t: f32, wind: [Option<f32>; 2]) -> Affine {
    let (mut aa, am) = polar(a);
    let (mut ba, bm) = polar(b);
    for col in 0..2 {
        if let Some(w) = wind[col] {
            let refang = w - 2.0 * PI;
            for v in [&mut aa[col], &mut ba[col]] {
                while *v < refang {
                    *v += 2.0 * PI;
                }
                while *v > refang + 2.0 * PI {
                    *v -= 2.0 * PI;
                }
            }
        } else {
            let d = ba[col] - aa[col];
            if d > PI + EPS {
                ba[col] -= 2.0 * PI;
            } else if d < -(PI - EPS) {
                ba[col] += 2.0 * PI;
            }
        }
    }
    let mut out = [[0.0; 2]; 3];
    for col in 0..2 {
        let ang = aa[col] * (1.0 - t) + ba[col] * t;
        let lin = am[col].ln() < -10.0 || bm[col].ln() < -10.0;
        let mag = if lin { am[col] * (1.0 - t) + bm[col] * t } else { (am[col].ln() * (1.0 - t) + bm[col].ln() * t).exp() };
        out[col] = [mag * ang.cos(), mag * ang.sin()];
        out[2][col] = a[2][col] * (1.0 - t) + b[2][col] * t;
    }
    out
}

/// The wind angles of an aligned pair: when exactly one side doesn't turn
/// (a symmetry xform), both are kept within the turning side's angle + 2 pi.
fn winds(a: &Xform, b: &Xform) -> [Option<f32>; 2] {
    let (aa, _) = polar(&a.c);
    let (mut ba, _) = polar(&b.c);
    let mut w = [None; 2];
    for col in 0..2 {
        let d = ba[col] - aa[col];
        if d > PI + EPS {
            ba[col] -= 2.0 * PI;
        } else if d < -(PI - EPS) {
            ba[col] += 2.0 * PI;
        }
        let (s0, s1) = (!a.animate, !b.animate);
        if s1 && !s0 {
            w[col] = Some(aa[col] + 2.0 * PI);
        } else if s0 && !s1 {
            w[col] = Some(ba[col] + 2.0 * PI);
        }
    }
    w
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// The xform between aligned `a` and `b`; the affines are given already turned.
fn interp_xform(a: &Xform, b: &Xform, ac: &Affine, bc: &Affine, t: f32, wind: [Option<f32>; 2]) -> Xform {
    let mut x = Xform::new();
    x.weight = lerp(a.weight, b.weight, t).max(0.0);
    x.color = lerp(a.color, b.color, t).clamp(0.0, 1.0);
    x.speed = lerp(a.speed, b.speed, t).clamp(0.0, 1.0);
    x.opacity = lerp(a.opacity, b.opacity, t);
    x.animate = if t < 0.5 { a.animate } else { b.animate };
    x.c = interp_affine(ac, bc, t, wind);
    x.post = if a.post == IDENTITY && b.post == IDENTITY {
        IDENTITY
    } else {
        interp_affine(&a.post, &b.post, t, [None; 2])
    };
    // the union of the variations; a side without one has weight 0 and the
    // other side's parameters
    let mut ids: Vec<i32> = a.vars.iter().chain(b.vars.iter()).map(|v| v.id).collect();
    ids.sort();
    ids.dedup();
    x.vars = ids
        .into_iter()
        .filter_map(|id| {
            let va = a.var(id);
            let vb = b.var(id);
            let (wa, pa) = va.map_or((0.0, None), |v| (v.w, Some(v.p)));
            let (wb, pb) = vb.map_or((0.0, None), |v| (v.w, Some(v.p)));
            let pa = pa.or(pb).unwrap_or(var::defaults(id));
            let pb = pb.unwrap_or(pa);
            let w = lerp(wa, wb, t);
            (w != 0.0).then(|| VarUse { id, w, p: std::array::from_fn(|i| lerp(pa[i], pb[i], t)) })
        })
        .collect();
    let n = a.xaos.len().max(b.xaos.len());
    if n > 0 {
        x.xaos = (0..n)
            .map(|j| lerp(a.xaos.get(j).copied().unwrap_or(1.0), b.xaos.get(j).copied().unwrap_or(1.0), t).max(0.0))
            .collect();
    }
    x
}

/// The palette between `a` and `b`: HSV, hue the short way round.
pub fn interp_palette(a: &[[f32; 3]], b: &[[f32; 3]], t: f32) -> Vec<[f32; 3]> {
    a.iter()
        .zip(b.iter())
        .map(|(ca, cb)| {
            let mut ha = rgb2hsv(*ca);
            let hb = rgb2hsv(*cb);
            if hb[0] - ha[0] > 3.0 {
                ha[0] += 6.0;
            } else if ha[0] - hb[0] > 3.0 {
                ha[0] -= 6.0;
            }
            let h = [lerp(ha[0], hb[0], t), lerp(ha[1], hb[1], t), lerp(ha[2], hb[2], t)];
            let c = hsv2rgb(h);
            [c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0), c[2].clamp(0.0, 1.0)]
        })
        .collect()
}

/// sheep_edge: `a` and `b` (aligned) both turned by `angle`, blended at `t`
/// (already eased). The result is flattened with angle 0.
pub fn edge(a: &Genome, b: &Genome, angle: f32, t: f32) -> Genome {
    let (s, c) = angle.sin_cos();
    let turn = |x: &Xform| if x.animate { crate::genome::rotate_linear(&x.c, s, c) } else { x.c };
    let mut g = a.clone();
    g.xforms = a
        .xforms
        .iter()
        .zip(b.xforms.iter())
        .map(|(xa, xb)| interp_xform(xa, xb, &turn(xa), &turn(xb), t, winds(xa, xb)))
        .collect();
    for x in g.xforms.iter_mut() {
        // already turned
        x.animate = false;
    }
    g.fin = match (&a.fin, &b.fin) {
        (Some(fa), Some(fb)) => Some(interp_xform(fa, fb, &fa.c, &fb.c, t, [None; 2])),
        _ => None,
    };
    g.palette = interp_palette(&a.palette, &b.palette, t);
    g.center = [lerp(a.center[0], b.center[0], t), lerp(a.center[1], b.center[1], t)];
    g.scale = lerp(a.scale, b.scale, t);
    g.rotate = lerp(a.rotate, b.rotate, t);
    g.brightness = lerp(a.brightness, b.brightness, t);
    g.gamma = lerp(a.gamma, b.gamma, t);
    g.gamma_threshold = lerp(a.gamma_threshold, b.gamma_threshold, t);
    g.vibrancy = lerp(a.vibrancy, b.vibrancy, t);
    g.highlight = lerp(a.highlight, b.highlight, t);
    for i in 0..3 {
        g.background[i] = lerp(a.background[i], b.background[i], t);
        g.de[i] = lerp(a.de[i], b.de[i], t);
    }
    g.name = if t < 0.5 { a.name.clone() } else { b.name.clone() };
    g
}

/// flam3's smoother: 3t^2 - 2t^3.
pub fn smoother(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
