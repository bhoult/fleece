//! The variations for the CPU, eight points at a time: the same formulas as
//! shaders/variations.glsl (flam3's variations.c, then Fractorium's 3D,
//! plugin and direct-colour ones), branch-free.

use crate::fmath::Rng8;
use crate::simd::{F8, I8, M8};
use core::f32::consts::PI;

const EPS: f32 = 1e-10;

/// The per-lane inputs of one variation.
pub struct VarIn {
    pub tx: F8,
    pub ty: F8,
    pub sumsq: F8,
    pub sq: F8,
    pub w: F8,
    pub p: [F8; 8],
    /// the xform's affine (waves, popcorn, rings, fan, dc_carpet)
    pub c00: F8,
    pub c01: F8,
    pub c10: F8,
    pub c11: F8,
    pub c20: F8,
    pub c21: F8,
}

#[inline(always)]
fn sel(m: M8, a: F8, b: F8) -> F8 {
    F8::select(m, a, b)
}
#[inline(always)]
fn s(v: f32) -> F8 {
    F8::splat(v)
}
#[inline(always)]
fn sinhf(x: F8) -> F8 {
    let e = x.clamp(-80.0, 80.0).fexp();
    (e - F8::one() / e) * 0.5
}
#[inline(always)]
fn coshf(x: F8) -> F8 {
    let e = x.clamp(-80.0, 80.0).fexp();
    (e + F8::one() / e) * 0.5
}
#[inline(always)]
fn rand_bit(rng: &mut Rng8) -> M8 {
    I8::zero().gt(rng.next())
}
/// a sum of four uniforms minus 2 (flam3's cheap gaussian)
#[inline(always)]
fn gauss(rng: &mut Rng8) -> F8 {
    rng.f01() + rng.f01() + rng.f01() + rng.f01() - 2.0
}
#[inline(always)]
fn atan_xy(v: &VarIn) -> F8 {
    v.tx.atan2(v.ty)
}
#[inline(always)]
fn atan_yx(v: &VarIn) -> F8 {
    v.ty.atan2(v.tx)
}
#[inline(always)]
fn polar(r: F8, a: F8) -> (F8, F8) {
    let (sn, cs) = a.sin_cos();
    (r * cs, r * sn)
}
/// Fractorium's Zeps: a divisor that is never 0
#[inline(always)]
fn zeps(x: F8) -> F8 {
    sel(x.eq(F8::zero()), s(f32::EPSILON), x)
}
/// bMod's and bSwirl's bipolar coordinates (tau, sigma)
#[inline(always)]
fn bipolar(tx: F8, ty: F8) -> (F8, F8) {
    let xp1 = tx + 1.0;
    let xm1 = tx - 1.0;
    let y2 = ty * ty;
    let tau = ((xp1 * xp1 + y2).ln() - (xm1 * xm1 + y2).ln()) * 0.5;
    (tau, s(PI) - ty.atan2(xp1) - ty.atan2(F8::one() - tx))
}

/// pre_blur: the point moved by a gaussian of `w` in a random direction,
/// before the precalcs (flam3 applies it right after the affine).
#[inline(always)]
pub fn pre_blur(tx: F8, ty: F8, w: F8, rng: &mut Rng8) -> (F8, F8) {
    let g = w * gauss(rng);
    let (sn, cs) = (rng.f01() * (2.0 * PI)).sin_cos();
    (g.mul_add(cs, tx), g.mul_add(sn, ty))
}

/// Variation `id`'s contribution (flam3's numbering).
#[inline(always)]
pub fn eval(id: i32, v: &VarIn, rng: &mut Rng8) -> (F8, F8) {
    let (tx, ty, w, sq, sumsq, p) = (v.tx, v.ty, v.w, v.sq, v.sumsq, &v.p);
    match id {
        0 => (w * tx, w * ty),
        1 => (w * tx.fsin(), w * ty.fsin()),
        2 => {
            let r2 = w / (sumsq + EPS);
            (r2 * tx, r2 * ty)
        }
        3 => {
            let (sn, c) = sumsq.sin_cos();
            (w * (sn * tx - c * ty), w * (c * tx + sn * ty))
        }
        4 => {
            let r = w / (sq + EPS);
            ((tx - ty) * (tx + ty) * r, 2.0 * tx * ty * r)
        }
        5 => (w * atan_xy(v) * (1.0 / PI), w * (sq - 1.0)),
        6 => {
            let a = atan_xy(v);
            (w * sq * (a + sq).fsin(), w * sq * (a - sq).fcos())
        }
        7 => {
            let a = sq * atan_xy(v);
            let r = w * sq;
            let (sn, c) = a.sin_cos();
            (r * sn, -r * c)
        }
        8 => {
            let a = atan_xy(v) * (1.0 / PI);
            let (sn, c) = (sq * PI).sin_cos();
            (w * sn * a, w * c * a)
        }
        9 => {
            let r = sq + EPS;
            let r1 = w / r;
            let (sn, c) = r.sin_cos();
            (r1 * (ty / sq + sn), r1 * (tx / sq - c))
        }
        10 => {
            let r = sq + EPS;
            (w * tx / sq / r, w * ty / sq * r)
        }
        11 => {
            let (sn, c) = sq.sin_cos();
            (w * tx / sq * c, w * ty / sq * sn)
        }
        12 => {
            let a = atan_xy(v);
            let n0 = (a + sq).fsin();
            let n1 = (a - sq).fcos();
            let m0 = n0 * n0 * n0 * sq;
            let m1 = n1 * n1 * n1 * sq;
            (w * (m0 + m1), w * (m0 - m1))
        }
        13 => {
            let a = atan_xy(v) * 0.5;
            let a = sel(rand_bit(rng), a + PI, a);
            polar(w * sq.sqrt(), a)
        }
        14 => {
            let neg_x = tx.lt(F8::zero());
            let neg_y = ty.lt(F8::zero());
            (w * sel(neg_x, tx * 2.0, tx), w * sel(neg_y, ty * 0.5, ty))
        }
        15 => {
            let dx2 = F8::one() / (v.c20 * v.c20 + EPS);
            let dy2 = F8::one() / (v.c21 * v.c21 + EPS);
            (w * (tx + v.c10 * (ty * dx2).fsin()), w * (ty + v.c11 * (tx * dy2).fsin()))
        }
        16 => {
            let r = 2.0 * w / (sq + 1.0);
            (r * ty, r * tx)
        }
        17 => (w * (tx + v.c20 * (ty * 3.0).tan().fsin()), w * (ty + v.c21 * (tx * 3.0).tan().fsin())),
        18 => {
            let dx = w * (tx - 1.0).fexp();
            let (sn, c) = (ty * PI).sin_cos();
            (dx * c, dx * sn)
        }
        19 => {
            let sa = tx / sq;
            let r = w * sq.pow(sa);
            (r * (ty / sq), r * sa)
        }
        20 => {
            let (sn, c) = (tx * PI).sin_cos();
            (w * c * coshf(ty), -w * sn * sinhf(ty))
        }
        21 => {
            let dx = v.c20 * v.c20 + EPS;
            let r = w * ((sq + dx).fmod(dx * 2.0) - dx + sq * (1.0 - dx));
            (r * ty / sq, r * tx / sq)
        }
        22 => {
            let dx = (v.c20 * v.c20 + EPS) * PI;
            let dy = v.c21;
            let dx2 = dx * 0.5;
            let a = atan_xy(v);
            let a = a + sel((a + dy).fmod(dx).gt(dx2), -dx2, dx2);
            polar(w * sq, a)
        }
        23 => {
            let a = atan_xy(v);
            let r = sq * (p[0] + (p[1] - p[0]) * ((p[2] * a).fsin().mul_add(s(0.5), s(0.5))));
            (w * r * tx / sq, w * r * ty / sq)
        }
        24 => (w * ((p[0] * ty).fsin() - (p[1] * tx).fcos()), w * ((p[2] * tx).fsin() - (p[3] * ty).fcos())),
        25 => {
            let dy = p[1];
            let dx = (p[0] * p[0] + EPS) * PI;
            let dx2 = dx * 0.5;
            let a = atan_xy(v);
            let t = a + dy - dx * ((a + dy) / dx).trunc();
            let a = sel(t.gt(dx2), a - dx2, a + dx2);
            let (sn, c) = a.sin_cos();
            (w * sq * sn, w * sq * c)
        }
        26 => {
            let dx = p[0] * p[0] + EPS;
            let r = sq - dx * 2.0 * ((sq + dx) / (dx * 2.0)).trunc() + sq * (1.0 - dx);
            (w * r * tx / sq, w * r * ty / sq)
        }
        27 => {
            let r = 2.0 * w / (sq + 1.0);
            (r * tx, r * ty)
        }
        28 => {
            let r = w / (sumsq * 0.25 + 1.0);
            (r * tx, r * ty)
        }
        29 => (w * tx.fsin(), w * ty),
        30 => {
            let t = F8::one() / (p[0] - ty * p[1]);
            (w * p[0] * tx * t, w * p[2] * ty * t)
        }
        31 => {
            let a = rng.f01() * (2.0 * PI);
            let r = w * rng.f01();
            let (sn, c) = a.sin_cos();
            (tx * r * c, ty * r * sn)
        }
        32 => {
            let t = (p[1] * rng.f01()).trunc();
            let a = (atan_yx(v) + t * (2.0 * PI)) / p[0];
            polar(w * sumsq.pow(p[2]), a)
        }
        33 => {
            let t = (p[1] * rng.f01()).trunc();
            let yx = atan_yx(v);
            let even = (t.round_i8() & 1).eq(I8::zero());
            let a = (t * (2.0 * PI) + sel(even, yx, -yx)) / p[0];
            polar(w * sumsq.pow(p[2]), a)
        }
        34 => {
            let a = rng.f01() * (2.0 * PI);
            let r = w * rng.f01();
            polar(r, a)
        }
        35 => {
            let a = rng.f01() * (2.0 * PI);
            let r = w * gauss(rng);
            polar(r, a)
        }
        36 => {
            let g = w * gauss(rng);
            let a = atan_yx(v) + p[0] * g;
            let rz = p[1] * g - 1.0;
            let (sn, c) = a.sin_cos();
            (sq * c + rz * tx, sq * sn + rz * ty)
        }
        37 => {
            let sl = (rng.f01() * p[0] + 0.5).trunc();
            let a = p[1] + (sl + rng.f01() * p[2]) * (2.0 * PI) / p[0];
            polar(w * rng.f01(), a)
        }
        38 => {
            let rf = sumsq.pow(p[1] * 0.5);
            let th = atan_yx(v);
            let b = (2.0 * PI) / p[0];
            let ph = th - b * (th / b).floor();
            let ph = sel(ph.gt(b * 0.5), ph - b, ph);
            let amp = (p[3] * (F8::one() / (ph.fcos() + EPS) - 1.0) + p[2]) / (rf + EPS);
            (w * tx * amp, w * ty * amp)
        }
        39 => {
            let re = F8::one() + p[0] * tx + p[1] * (tx * tx - ty * ty);
            let im = p[0] * ty + 2.0 * p[1] * tx * ty;
            let r = w / (re * re + im * im);
            ((tx * re + ty * im) * r, (ty * re - tx * im) * r)
        }
        40 => {
            let x = ((tx / p[0]).floor() * 2.0 + 1.0) * p[0] - tx;
            let y = ((ty / p[1]).floor() * 2.0 + 1.0) * p[1] - ty;
            (w * sel(p[0].eq(F8::zero()), tx, x), w * sel(p[1].eq(F8::zero()), ty, y))
        }
        41 => {
            let (sn, c) = (rng.f01() * w * PI).sin_cos();
            (w * sn, w * sn * sn / c)
        }
        42 => (w * tx.fsin() / ty.fcos(), w * ty.tan()),
        43 => (w * (rng.f01() - 0.5), w * (rng.f01() - 0.5)),
        44 => {
            let a = w * rng.f01() * PI;
            let r = w / (sumsq + EPS);
            let t = w * a.tan() * r;
            (t * tx.fcos(), t * ty.fsin())
        }
        45 => {
            let (sn, c) = (rng.f01() * w * sq).sin_cos();
            (w * tx * (c + sn), w * tx * (c - sn))
        }
        46 => {
            let c = (w * sq).fcos();
            let ic = F8::one() / c;
            (w * tx, w * sel(c.lt(F8::zero()), ic + 1.0, ic - 1.0))
        }
        47 => {
            let (sn, c) = (rng.f01() * w * sq).sin_cos();
            let d = (sn * sn).log10() + c;
            // log10(0) is -38 here, not -inf: flam3 substitutes -30 for bad values
            let d = sel(d.bad() | (sn * sn).le(F8::splat(1e-37)), s(-30.0), d);
            (w * tx * d, w * tx * (d - sn * PI))
        }
        48 => {
            let q = tx * tx - ty * ty;
            let r = w * (F8::one() / (q * q + EPS)).sqrt();
            (tx * r, ty * r)
        }
        49 => {
            let t = p[0] * (tx + ty);
            let r = w * atan_xy(v) * (1.0 / PI);
            let (sn, c) = t.sin_cos();
            ((sn + p[2]) * r, (c + p[1]) * r)
        }
        50 => {
            let th = p[0] * atan_yx(v) + core::f32::consts::FRAC_PI_4;
            let (st, ct) = th.sin_cos();
            let t1 = ct.abs().pow(p[1]);
            let t2 = st.abs().pow(p[2]);
            let r = w * ((p[4] * rng.f01() + (F8::one() - p[4]) * sq) - p[5]) * (t1 + t2).max(s(1e-30)).pow(p[3]) / sq;
            (r * tx, r * ty)
        }
        51 => {
            let r = w * (rng.f01() - p[1]) * (p[0] * atan_yx(v)).fcos() / sq;
            (r * tx, r * ty)
        }
        52 => {
            let ct = tx / sq;
            let r = w * (rng.f01() - p[1]) * p[0] / (p[0] * ct + 1.0) / sq;
            (r * tx, r * ty)
        }
        53 => {
            let (sn, c) = sq.sin_cos();
            (p[0] * w * sn * sn * rng.f01(), p[1] * w * c * rng.f01())
        }
        54 => {
            let neg_x = tx.lt(F8::zero());
            let neg_y = ty.lt(F8::zero());
            (w * sel(neg_x, tx * p[0], tx), w * sel(neg_y, ty * p[1], ty))
        }
        55 => {
            let t = sumsq + 1.0;
            let x2 = tx * 2.0;
            let y = (ty * 2.0).atan2(sumsq - 1.0) * 0.5 - p[0] * (0.5 * PI);
            let hi = y.gt(s(0.5 * PI));
            let lo = y.lt(s(-0.5 * PI));
            let y = sel(hi, (y + 0.5 * PI).fmod(s(PI)) - 0.5 * PI, y);
            let y = sel(lo, s(0.5 * PI) - (s(0.5 * PI) - y).fmod(s(PI)), y);
            (w * (0.25 * 2.0 / PI) * ((t + x2) / (t - x2)).ln(), w * (2.0 / PI) * y)
        }
        56 => {
            let rx = tx.round();
            let ry = ty.round();
            let ox = tx - rx;
            let oy = ty - ry;
            let corner = rng.f01().ge(s(0.75));
            let xside = ox.abs().ge(oy.abs());
            let half = (ox * 0.5 + rx, oy * 0.5 + ry);
            // the x side: +-0.25 on x, +-0.25 oy / ox on y; the y side likewise
            let sx = sel(ox.ge(F8::zero()), s(0.25), s(-0.25));
            let sy = sel(oy.ge(F8::zero()), s(0.25), s(-0.25));
            let (xa, ya) = (half.0 + sx, half.1 + sx * oy / ox);
            let (xb, yb) = (half.0 + sy * ox / oy, half.1 + sy);
            let x = sel(corner, half.0, sel(xside, xa, xb));
            let y = sel(corner, half.1, sel(xside, ya, yb));
            (w * x, w * y)
        }
        57 => {
            let y2 = ty * 2.0;
            let r = w * 1.302_940_1 * ((ty * tx).abs() / (tx * tx + y2 * y2 + EPS)).sqrt();
            (r * tx, r * y2)
        }
        58 => {
            let inv = F8::one() / p[0];
            let x = (tx * inv).floor();
            let y = (ty * inv).floor();
            let dx = tx - x * p[0];
            let dy = ty - y * p[0];
            let x = sel(x.ge(F8::zero()), x * 2.0, -(x * 2.0 + 1.0));
            let y = sel(y.ge(F8::zero()), y * 2.0, -(y * 2.0 + 1.0));
            (w * (dx + x * p[0]), -(w * (dy + y * p[0])))
        }
        59 => {
            let a = atan_yx(v);
            let lnr = sumsq.ln() * 0.5;
            let va = (2.0 * PI) / p[2];
            let vc = p[0] / p[2];
            let vd = p[1] / p[2];
            let ang = vc * a + vd * lnr + va * (p[2] * rng.f01()).floor();
            polar(w * (vc * lnr - vd * a).fexp(), ang)
        }
        60 => {
            let xl = (p[2] * p[2]).max(s(1e-20));
            let yl = (p[3] * p[3]).max(s(1e-20));
            (w * (tx + p[0] * (-ty * ty / xl).fexp()), w * (ty + p[1] * (-tx * tx / yl).fexp()))
        }
        61 => {
            let tmp = sumsq + 1.0;
            let tmp2 = tx * 2.0;
            let xmax = ((tmp + tmp2).sqrt() + (tmp - tmp2).sqrt()) * 0.5;
            let a1 = (xmax + (xmax - 1.0).max(F8::zero()).sqrt()).ln();
            let a2 = -acos((tx / xmax).clamp(-1.0, 1.0));
            let ww = w * (1.0 / 11.570_346);
            let (snv, csv) = a1.sin_cos();
            let snv = sel(ty.gt(F8::zero()), -snv, snv);
            (ww * coshf(a2) * csv, ww * sinhf(a2) * snv)
        }
        62 => {
            let tmp = sumsq + 1.0;
            let x2 = tx * 2.0;
            let xmax = ((tmp + x2).sqrt() + (tmp - x2).sqrt()) * 0.5;
            let a = tx / xmax;
            let b = (F8::one() - a * a).max(F8::zero()).sqrt();
            let ssx = (xmax - 1.0).max(F8::zero()).sqrt();
            let ww = w * (2.0 / PI);
            let l = (xmax + ssx).ln();
            (ww * a.atan2(b), ww * sel(ty.gt(F8::zero()), l, -l))
        }
        63 => {
            let a = atan_yx(v);
            let lnr = sumsq.ln() * 0.5;
            let m = w * (p[0] * lnr - p[1] * a).fexp();
            polar(m, p[0] * a + p[1] * lnr)
        }
        64 => {
            let ex = tx.fexp() * 0.5;
            let enx = F8::splat(0.25) / ex;
            let (sn, c) = ty.sin_cos();
            let t = w / (ex + enx - c);
            (t * (ex - enx), t * sn)
        }
        65 => {
            let x = tx - p[0];
            let y = ty + p[1];
            let r = (x * x + y * y).sqrt();
            let inside = r.lt(w);
            let a = y.atan2(x) + p[2] + p[4] * (w - r);
            let (sn, cs) = a.sin_cos();
            let ri = w * r;
            let ro = w * (F8::one() + p[3] / r);
            (sel(inside, ri * cs, ro * x) + p[0], sel(inside, ri * sn, ro * y) - p[1])
        }
        66 => {
            let w2 = w * w;
            let inside = sumsq.lt(w2);
            let r = sel(inside, w * (w2 / sumsq - 1.0).max(F8::zero()).sqrt(), w);
            (r * tx, r * ty)
        }
        68 => (w * modulus(tx, p[0]), w * modulus(ty, p[1])),
        69 => {
            let tpf = p[1] * (2.0 * PI);
            let damp = sel(p[3].eq(F8::zero()), F8::one(), (-tx.abs() * p[3]).fexp());
            let t = p[2] * (tpf * tx).fcos() * damp + p[0];
            (w * tx, w * sel(ty.abs().le(t), -ty, ty))
        }
        70 => {
            let pv = w * (1.0 / PI);
            (pv * atan_xy(v), pv * 0.5 * sumsq.ln())
        }
        71 => (w * (tx + p[0] * (ty * p[2]).tan().fsin()), w * (ty + p[1] * (tx * p[2]).tan().fsin())),
        72 => {
            let r = F8::one() / (sq * (sumsq + F8::one() / (w + EPS)));
            (r * tx, r * ty)
        }
        73 => {
            let ex = (tx * tx + p[0] * p[0]).sqrt();
            let ey = (ty * ty + p[2] * p[2]).sqrt();
            let x = sel(tx.gt(F8::zero()), ex - tx * p[1], -(ex + tx * p[1]));
            let y = sel(ty.gt(F8::zero()), ey - ty * p[3], -(ey + ty * p[3]));
            (w * x, w * y)
        }
        74 => {
            let fx = (ty * p[1] * PI).fcos().ge(F8::zero());
            let fy = (tx * p[0] * PI).fcos().ge(F8::zero());
            (w * sel(fx, tx, -tx), w * sel(fy, ty, -ty))
        }
        75 => (w * sel(tx.ge(F8::zero()), tx + p[0], tx - p[0]), w * sel(ty.ge(F8::zero()), ty + p[1], ty - p[1])),
        76 => {
            let rx = (tx + 0.5).floor();
            let ox = tx - rx;
            (w * (ox * (1.0 - p[0]) + rx), w * (ty + ox * ox * p[1]))
        }
        77 => {
            let a = atan_yx(v) + p[3] * sq;
            let c = ((p[2] * a + PI) * (0.5 / PI)).floor();
            let cf = F8::one() - p[0] * p[2] * (0.5 / PI);
            polar(w * (sq + p[1]), a * cf + c * p[0])
        }
        78 => {
            let r = w * sumsq.pow(p[5]);
            let t = (p[4] * rng.f01()).trunc();
            let a = (atan_yx(v) + t * (2.0 * PI)) / p[2];
            let c = ((p[1] * a + PI) * (0.5 / PI)).floor();
            polar(r, a * p[3] + c * p[0])
        }
        79 => {
            let r = F8::one() / (sq + EPS);
            let a = atan_yx(v) + p[3] * r;
            let c = ((p[1] * a + PI) * (0.5 / PI)).floor();
            let cf = F8::one() - p[0] * p[1] * (0.5 / PI);
            polar(w * (r + p[2]), a * cf + c * p[0])
        }
        80 => {
            let a = atan_yx(v) + sel(sq.lt(w), p[0], p[1]) / (w - sq);
            polar(w * sq, a)
        }
        81 => (w * (tx + p[0] * (ty * p[2]).fsin()), w * (ty + p[1] * (tx * p[3]).fsin())),
        82 => polar(w * tx.fexp(), ty),
        83 => (w * 0.5 * sumsq.ln(), w * atan_yx(v)),
        84 => {
            let (sn, c) = tx.sin_cos();
            (w * sn * coshf(ty), w * c * sinhf(ty))
        }
        85 => {
            let (sn, c) = tx.sin_cos();
            (w * c * coshf(ty), -(w * sn * sinhf(ty)))
        }
        86 => {
            let (sn, c) = (tx * 2.0).sin_cos();
            let d = F8::one() / (c + coshf(ty * 2.0));
            (w * d * sn, w * d * sinhf(ty * 2.0))
        }
        87 => {
            let (sn, c) = tx.sin_cos();
            let d = F8::splat(2.0) / ((tx * 2.0).fcos() + coshf(ty * 2.0));
            (w * d * c * coshf(ty), w * d * sn * sinhf(ty))
        }
        88 => {
            let (sn, c) = tx.sin_cos();
            let d = F8::splat(2.0) / (coshf(ty * 2.0) - (tx * 2.0).fcos());
            (w * d * sn * coshf(ty), -(w * d * c * sinhf(ty)))
        }
        89 => {
            let (sn, c) = (tx * 2.0).sin_cos();
            let d = F8::one() / (coshf(ty * 2.0) - c);
            (w * d * sn, -(w * d * sinhf(ty * 2.0)))
        }
        90 => {
            let (sn, c) = ty.sin_cos();
            (w * sinhf(tx) * c, w * coshf(tx) * sn)
        }
        91 => {
            let (sn, c) = ty.sin_cos();
            (w * coshf(tx) * c, w * sinhf(tx) * sn)
        }
        92 => {
            let (sn, c) = (ty * 2.0).sin_cos();
            let d = F8::one() / (c + coshf(tx * 2.0));
            (w * d * sinhf(tx * 2.0), w * d * sn)
        }
        93 => {
            let (sn, c) = ty.sin_cos();
            let d = F8::splat(2.0) / ((ty * 2.0).fcos() + coshf(tx * 2.0));
            (w * d * c * coshf(tx), -(w * d * sn * sinhf(tx)))
        }
        94 => {
            let (sn, c) = ty.sin_cos();
            let d = F8::splat(2.0) / (coshf(tx * 2.0) - (ty * 2.0).fcos());
            (w * d * sinhf(tx) * c, -(w * d * coshf(tx) * sn))
        }
        95 => {
            let (sn, c) = (ty * 2.0).sin_cos();
            let d = F8::one() / (coshf(tx * 2.0) - c);
            (w * d * sinhf(tx * 2.0), w * d * sn)
        }
        96 => {
            let sn = (p[0] * tx).fsin();
            let t = (p[0] * ty).fsin();
            let dy = ty + p[1] * (p[2] * sn * 0.5 + ty.abs() * sn);
            let dx = tx + p[1] * (p[2] * t * 0.5 + tx.abs() * t);
            (w * (tx + p[3] * (dx - tx)), w * dy)
        }
        97 => {
            let xpw = tx + w;
            let xmw = tx - w;
            let r = w * (p[0] + 2.0) * ((ty * ty + xpw * xpw).sqrt() / (ty * ty + xmw * xmw).sqrt()).sqrt();
            let a = (ty.atan2(xmw) - ty.atan2(xpw)) * 0.5;
            polar(r, a)
        }
        98 => {
            let reu = p[0] * tx - p[1] * ty + p[2];
            let imu = p[0] * ty + p[1] * tx + p[3];
            let rev = p[4] * tx - p[5] * ty + p[6];
            let imv = p[4] * ty + p[5] * tx + p[7];
            let rad = w / (rev * rev + imv * imv);
            (rad * (reu * rev + imu * imv), rad * (imu * rev - reu * imv))
        }
        // the plugins, after Fractorium's EclipseVariation, EpispiralVariation,
        // CirclizeVariation, GlynniaVariation and SphericalNVariation (both in
        // their flam3-compatible form, Fractorium's default), XHeartVariation,
        // PetalVariation, SquirrelVariation, HoleVariation, FlipCircleVariation,
        // FlipYVariation, BModVariation, BSwirlVariation, CollideoscopeVariation
        112 => {
            let c2 = (w * w - ty * ty).max(F8::zero()).sqrt();
            let inside = ty.abs().le(w) & tx.abs().le(c2);
            let xs = p[0].mul_add(w, tx);
            (w * sel(inside, sel(xs.abs().ge(c2), -tx, xs), tx), w * ty)
        }
        113 => {
            let th = atan_yx(v);
            let tw = p[1] * w;
            let t = sel(tw.eq(F8::zero()), w, tw * rng.f01()) / (p[0] * th).fcos() - p[2] * w;
            polar(t, th)
        }
        114 => {
            let (ax, ay) = (tx.abs(), ty.abs());
            let xside = ax.ge(ay);
            let side = sel(xside, ax, ay);
            let per = sel(xside, sel(tx.ge(ay), ax + ty, ax * 5.0 - ty), sel(ty.ge(ax), ay * 3.0 - tx, ay * 7.0 + tx));
            let r = w * (4.0 / PI) * side + p[0];
            polar(r, s(PI / 4.0) * per / side - PI / 4.0)
        }
        115 => {
            let v2 = w * 0.707_106_77;
            let outside = sq.gt(F8::one());
            let sg = sel(outside, F8::one(), s(-1.0));
            let first = rng.f01().gt(s(0.5));
            let d1 = (sq + tx).sqrt();
            let d1 = sel(outside, d1, zeps(d1));
            let d2 = sq + tx;
            let s2 = (sq * (ty * ty + d2 * d2)).sqrt();
            let r = w / sel(outside, s2, zeps(s2));
            (sel(first, sg * v2 * d1, sg * r * d2), sel(first, -(v2 / d1 * ty), r * ty))
        }
        116 => {
            let r = zeps(sq.pow(p[1]));
            let n = (p[0] * rng.f01()).floor();
            polar(w / r, atan_yx(v) + n * (2.0 * PI) / zeps(p[0].floor()))
        }
        117 => {
            let r24 = sumsq + 4.0;
            let bx = s(4.0) / r24;
            let by = p[2] / r24;
            let x = p[0] * (bx * tx) - p[1] * (by * ty);
            let y = p[1] * (bx * tx) + p[0] * (by * ty);
            (w * x, w * sel(x.gt(F8::zero()), y, -y))
        }
        118 => {
            let (sx, cx) = tx.sin_cos();
            let cy = ty.fcos();
            let (bx, by) = (cx * cy, sx * cy);
            (w * cx * (bx * bx * bx), w * cx * (by * by * by))
        }
        119 => {
            let u = (p[0] * tx * tx + p[1] * ty * ty).max(F8::zero()).sqrt();
            let (sn, cs) = u.sin_cos();
            (w * cs * tx.clamp(-8_388_607.0, 8_388_607.0).tan(), w * sn * ty.clamp(-8_388_607.0, 8_388_607.0).tan())
        }
        120 => {
            let d = (atan_yx(v) * (1.0 / PI) + 1.0).pow(p[0]);
            let r = sel(p[1].ne(F8::zero()), w * d / (sq + d), w.mul_add(sq, d)) / zeps(sq);
            (r * tx, r * ty)
        }
        121 => (w * tx, w * sel(sumsq.gt(w * w), ty, -ty)),
        122 => (w * tx, w * sel(tx.gt(F8::zero()), -ty, ty)),
        123 => {
            let (tau, sigma) = bipolar(tx, ty);
            let inside = tau.lt(p[0]) & (-tau).lt(p[0]);
            let tau = sel(inside, (tau + p[0] + p[1] * p[0]).fmod(p[0] * 2.0) - p[0], tau);
            let t = zeps(coshf(tau) - sigma.fcos());
            (w * sinhf(tau) / t, w * sigma.fsin() / t)
        }
        124 => {
            let (tau, sigma) = bipolar(tx, ty);
            let sigma = sigma + tau * p[1] + p[0] / zeps(tau);
            let t = zeps(coshf(tau) - sigma.fcos());
            (w * sinhf(tau) / t, w * sigma.fsin() / t)
        }
        125 => {
            let a = atan_yx(v);
            let alt = (a.abs() * p[0]).trunc();
            // alt pi/n plus the angle past it, shifted by +-pi a/n by parity
            let k = sel((alt.round_i8() & 1).eq(I8::zero()), p[2], -p[2]);
            let pos = alt * p[1] + (k + a).fmod(p[1]);
            let neg = -(alt * p[1] + (k - a).fmod(p[1]));
            polar(w * sq, sel(a.ge(F8::zero()), pos, neg))
        }
        _ => (F8::zero(), F8::zero()),
    }
}

/// What the 3D chaos game adds to a variation's inputs: its input z, the
/// sum of the variations before it (the dc_ ones read it) and the colour
/// index (the xform's blend, or what a dc_ variation before it set), which
/// the dc_ ones set.
pub struct V3 {
    pub tz: F8,
    pub sx: F8,
    pub sy: F8,
    pub col: F8,
}

/// Variation `id` in the 3D chaos game: x, y and z (Fractorium's z rules;
/// flam3's and the plugins through `eval`), the dc_ ones setting `e.col`.
#[inline(always)]
pub fn eval3(id: i32, v: &VarIn, e: &mut V3, rng: &mut Rng8) -> (F8, F8, F8) {
    let (tx, ty, tz, w, sq, sumsq, p) = (v.tx, v.ty, e.tz, v.w, v.sq, v.sumsq, &v.p);
    let zero = F8::zero();
    match id {
        // after Fractorium's Linear3DVariation, Spherical3DVariation,
        // Julia3DVariation, Julia3DzVariation, HemisphereVariation,
        // ZConeVariation, ZblurVariation, ZTranslateVariation,
        // ZScaleVariation, Blur3DVariation, Curl3DVariation, Pie3DVariation
        // and Sinusoidal3DVariation
        99 => (w * tx, w * ty, w * tz),
        100 => {
            let r2 = w / zeps(sumsq + tz * tz);
            (r2 * tx, r2 * ty, r2 * tz)
        }
        101 => {
            let z = tz / p[1];
            let r = w * (sumsq + z * z).pow(p[2]);
            let t = (rng.f01() * p[1]).trunc();
            let (x, y) = polar(r * sq, (atan_yx(v) + t * (2.0 * PI)) / p[0]);
            (x, y, r * z)
        }
        102 => {
            let r = w * sumsq.pow(p[2]);
            let t = (rng.f01() * p[1]).trunc();
            let (x, y) = polar(r, (atan_yx(v) + t * (2.0 * PI)) / p[0]);
            (x, y, r * tz / zeps(sq * p[1]))
        }
        103 => {
            let t = w / (sumsq + 1.0).sqrt();
            (t * tx, t * ty, t)
        }
        104 => (zero, zero, w * sq),
        105 => (zero, zero, w * gauss(rng)),
        106 => (zero, zero, w),
        107 => (zero, zero, w * tz),
        108 => {
            let a = rng.f01() * (2.0 * PI);
            let r = w * gauss(rng);
            let (sb, cb) = (rng.f01() * PI).sin_cos();
            let (x, y) = polar(r * sb, a);
            (x, y, r * cb)
        }
        109 => {
            let r2 = sumsq + tz * tz;
            let r = w / zeps(r2 * p[3] + p[0] * 2.0 * tx - p[1] * 2.0 * ty + p[2] * 2.0 * tz + 1.0);
            (r * (tx + p[0] * r2), r * (ty - p[1] * r2), r * (tz + p[2] * r2))
        }
        110 => {
            let sl = (rng.f01() * p[0] + 0.5).trunc();
            let a = p[1] + (sl + rng.f01() * p[2]) * (2.0 * PI) / p[0];
            let r = w * rng.f01();
            let (x, y) = polar(r, a);
            (x, y, w * r.fsin())
        }
        111 => (w * tx.fsin(), w * ty.fsin(), w * (tx * tx).atan2(ty * ty) * tz.fcos()),
        // direct colour, after Fractorium's DCLinearVariation,
        // DCBubbleVariation, DCCarpetVariation, DCCylinderVariation and
        // DCZTranslVariation
        126 => {
            let (x, y) = (w * tx, w * ty);
            let (sx, sy) = (x + e.sx, y + e.sy);
            e.col = ((p[3] * (p[2] * sx + p[1] * sy + p[0]) + 1.0) * 0.5).abs().fract();
            (x, y, w * tz)
        }
        127 => {
            let r = w / zeps(sumsq * 0.25 + 1.0);
            let (x, y) = (r * tx, r * ty);
            let (sx, sy) = (x + e.sx + p[0], y + e.sy + p[1]);
            e.col = (p[2] * (sx * sx + sy * sy)).abs().fract();
            (x, y, w * (s(2.0) / r - 1.0))
        }
        128 => {
            let x0 = sel(rand_bit(rng), s(-1.0), F8::one());
            let y0 = sel(rand_bit(rng), s(-1.0), F8::one());
            let (x, y) = (tx + x0, ty + y0);
            // Fractorium's (int) x0 ^ (int) y0: 0 for the same sign, -2 else
            let xr = sel(x0.eq(y0), zero, s(-2.0));
            let h = (F8::one() - xr) * p[0] - p[0];
            e.col = (e.col * 0.5 * (h + 1.0) + xr * (F8::one() - h) * 0.5).abs().fract();
            (w * (v.c00 * x + v.c10 * y + v.c11), w * (v.c20 * x + v.c01 * y + v.c21), zero)
        }
        129 => {
            let (sn, cs) = (rng.f01() * (2.0 * PI)).sin_cos();
            let r = p[6] * gauss(rng);
            let x = w * r.mul_add(sn, tx).fsin() * p[4];
            let y = r + ty * p[5];
            let (sx, sy) = (x + e.sx, y + e.sy);
            e.col = ((p[3] * (p[2] * sx + p[1] * sy + p[0]) + 1.0) * 0.5).abs().fract();
            (x, y, w * r.mul_add(cs, tx).fcos())
        }
        130 => {
            let zf = p[2] * (e.col - p[0]) / p[1];
            let zf = sel(p[4].ne(zero), zf.clamp(0.0, 1.0), zf);
            (w * tx, w * ty, sel(p[3].eq(zero), w * tz * zf, w * zf))
        }
        _ => {
            let (x, y) = eval(id, v, rng);
            (x, y, zdef(id, v, tz))
        }
    }
}

/// The z a flam3 variation passes on: Fractorium's (w z for those in
/// var::ZPASS; bubble's, cylinder's and flux's own; 0, DefaultZ, for the
/// rest, the plugins among them).
#[inline(always)]
fn zdef(id: i32, v: &VarIn, tz: F8) -> F8 {
    match id {
        28 => v.w * (s(2.0) / zeps(v.sumsq.mul_add(s(0.25), F8::one()) - 1.0)),
        29 => v.w * v.tx.fcos(),
        97 => tz,
        0..128 if crate::var::ZPASS_MASK[id as usize >> 5] & (1 << (id & 31)) != 0 => v.w * tz,
        _ => F8::zero(),
    }
}

#[inline(always)]
fn modulus(t: F8, m: F8) -> F8 {
    let hi = t.gt(m);
    let lo = t.lt(-m);
    let a = (t + m).fmod(m * 2.0) - m;
    let b = m - (m - t).fmod(m * 2.0);
    sel(hi, a, sel(lo, b, t))
}

/// acos for |x| <= 1, through atan2(sqrt(1 - x^2), x).
#[inline(always)]
fn acos(x: F8) -> F8 {
    (F8::one() - x * x).max(F8::zero()).sqrt().atan2(x)
}
