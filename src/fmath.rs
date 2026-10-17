//! Eight-lane versions of the libm functions the variations call, accurate
//! to a few ulp over the ranges flames use (Cephes polynomials), and a
//! per-lane xorshift generator. The GPU uses its own built-ins; both only
//! have to land on the same attractor, not on the same orbit.
#![allow(dead_code)]

use crate::simd::{F8, I8, M8};
use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

impl F8 {
    /// (sin x, cos x): reduced by multiples of pi/2, degree 9 / 8 polynomials.
    #[inline(always)]
    pub fn sin_cos(self) -> (F8, F8) {
        // past 1e5 the argument keeps no useful phase; wrap it the way the
        // GPU's sine does (the fraction of x / 2 pi), so both stay bounded
        // and agree
        let t = self * (0.5 / PI);
        let wrapped = (t - t.round()) * (2.0 * PI);
        let x = F8::select(self.abs().gt(F8::splat(1e5)), wrapped, self);
        let q = (x * (2.0 / PI)).round();
        // x - q pi/2 in three parts (Cody-Waite)
        let r = q.neg_mul_add(F8::splat(1.570_312_5), x);
        let r = q.neg_mul_add(F8::splat(4.837_513e-4), r);
        let r = q.neg_mul_add(F8::splat(7.549_79e-8), r);
        // |r| <= pi/4: Taylor to degree 9 (sin) and 8 (cos), errors under 3e-8
        let z = r * r;
        let s = F8::splat(2.755_731_9e-6).mul_add(z, F8::splat(-1.984_127e-4));
        let s = s.mul_add(z, F8::splat(8.333_333e-3)).mul_add(z, F8::splat(-1.666_666_7e-1));
        let s = (s * z).mul_add(r, r);
        let c = F8::splat(2.480_158_7e-5).mul_add(z, F8::splat(-1.388_888_9e-3)).mul_add(z, F8::splat(4.166_666_8e-2));
        let c = (c * z).mul_add(z, F8::splat(-0.5).mul_add(z, F8::one()));
        // quadrant: 0 (s, c), 1 (c, -s), 2 (-s, -c), 3 (-c, s)
        let qi = q.round_i8();
        let swap = (qi & 1).eq(I8::splat(1));
        let (sn, cs) = (F8::select(swap, c, s), F8::select(swap, s, c));
        let neg_s = (qi & 2).eq(I8::splat(2));
        let neg_c = ((qi + 1) & 2).eq(I8::splat(2));
        (F8::select(neg_s, -sn, sn), F8::select(neg_c, -cs, cs))
    }
    #[inline(always)]
    pub fn fsin(self) -> F8 {
        self.sin_cos().0
    }
    #[inline(always)]
    pub fn fcos(self) -> F8 {
        self.sin_cos().1
    }
    #[inline(always)]
    pub fn tan(self) -> F8 {
        let (s, c) = self.sin_cos();
        s / c
    }

    /// atan(x) for x in [0, 1]: reduced at tan(pi/8), Cephes atanf.
    #[inline(always)]
    fn atan01(a: F8) -> F8 {
        let mid = a.gt(F8::splat(0.414_213_57));
        let x = F8::select(mid, (a - 1.0) / (a + 1.0), a);
        let y0 = F8::splat(FRAC_PI_4).and(mid);
        let z = x * x;
        let p = F8::splat(8.053_744_5e-2).mul_add(z, F8::splat(-1.387_768_6e-1));
        let p = p.mul_add(z, F8::splat(1.997_771_1e-1)).mul_add(z, F8::splat(-3.333_295e-1));
        y0 + (p * z).mul_add(x, x)
    }

    /// atan2(y = self, x), in (-pi, pi]; atan2(0, 0) = 0.
    #[inline(always)]
    pub fn atan2(self, x: F8) -> F8 {
        let (ay, ax) = (self.abs(), x.abs());
        let hi = ay.max(ax);
        let lo = ay.min(ax);
        let r = F8::atan01(lo / hi.max(F8::splat(1e-37)));
        let r = F8::select(ay.gt(ax), F8::splat(FRAC_PI_2) - r, r);
        let r = F8::select(x.lt(F8::zero()), F8::splat(PI) - r, r);
        r.copysign(self)
    }
    #[inline(always)]
    pub fn atan(self) -> F8 {
        self.atan2(F8::one())
    }

    /// ln(x) for x > 0 (Cephes logf); 0 gives about -88, not -inf.
    #[inline(always)]
    pub fn ln(self) -> F8 {
        {
            let bits = self.max(F8::splat(1e-38)).to_bits();
            let e = (bits.shr::<23>() & 0xff) - 127;
            let m = (bits & 0x007f_ffff | 0x3f80_0000).cast_f8();
            // m in [1, 2): fold to [sqrt(1/2), sqrt(2))
            let big = m.gt(F8::splat(core::f32::consts::SQRT_2));
            let m = F8::select(big, m * 0.5, m);
            let e = e.to_f8() + F8::one().and(big);
            let x = m - 1.0;
            let z = x * x;
            let y = F8::splat(7.037_683_6e-2).mul_add(x, F8::splat(-1.151_461e-1));
            let y = y.mul_add(x, F8::splat(1.167_699_9e-1)).mul_add(x, F8::splat(-1.242_014_1e-1));
            let y = y.mul_add(x, F8::splat(1.424_932_3e-1)).mul_add(x, F8::splat(-1.666_805_8e-1));
            let y = y.mul_add(x, F8::splat(2.000_071_4e-1)).mul_add(x, F8::splat(-2.499_999_4e-1));
            let y = y.mul_add(x, F8::splat(3.333_333e-1)) * x * z;
            let y = e.mul_add(F8::splat(-2.121_944_4e-4), y);
            let y = F8::splat(-0.5).mul_add(z, y);
            e.mul_add(F8::splat(0.693_359_4), x + y)
        }
    }
    #[inline(always)]
    pub fn log10(self) -> F8 {
        self.ln() * core::f32::consts::LOG10_E
    }

    /// e^x (Cephes expf), x clamped to [-87, 88].
    #[inline(always)]
    pub fn fexp(self) -> F8 {
        let x = self.clamp(-87.0, 88.0);
        let n = (x * core::f32::consts::LOG2_E).round();
        let g = n.neg_mul_add(F8::splat(0.693_359_4), x);
        let g = n.neg_mul_add(F8::splat(-2.121_944_4e-4), g);
        let z = g * g;
        let y = F8::splat(1.987_569_1e-4).mul_add(g, F8::splat(1.398_199_9e-3));
        let y = y.mul_add(g, F8::splat(8.333_452e-3)).mul_add(g, F8::splat(4.166_579_6e-2));
        let y = y.mul_add(g, F8::splat(1.666_666_5e-1)).mul_add(g, F8::splat(0.5));
        let y = y.mul_add(z, g + 1.0);
        let scale = ((n.round_i8() + 127).shl::<23>()).cast_f8();
        y * scale
    }

    /// x^p for x >= 0 (0^p = 0 for p > 0).
    #[inline(always)]
    pub fn pow(self, p: F8) -> F8 {
        let r = (p * self.ln()).fexp();
        F8::select(self.le(F8::zero()), F8::zero(), r)
    }
    #[inline(always)]
    pub fn sinh_cosh(self) -> (F8, F8) {
        let e = self.fexp();
        let ie = F8::one() / e;
        ((e - ie) * 0.5, (e + ie) * 0.5)
    }
    /// C fmod: self - b * trunc(self / b)
    #[inline(always)]
    pub fn fmod(self, b: F8) -> F8 {
        (self / b).trunc().neg_mul_add(b, self)
    }
    /// any lane NaN, infinite or beyond +-1e10 (flam3's badvalue)
    #[inline(always)]
    pub fn bad(self) -> M8 {
        // NaN fails the compare, so it counts as bad
        !self.abs().le(F8::splat(1e10))
    }
}

/// Eight xorshift32 streams.
#[derive(Clone, Copy)]
pub struct Rng8(pub I8);

impl Rng8 {
    pub fn new(seed: u32) -> Rng8 {
        let mut a = [0i32; 8];
        let mut s = seed ^ 0x9E37_79B9;
        for v in a.iter_mut() {
            s = s.wrapping_mul(0x85EB_CA6B).wrapping_add(0xC2B2_AE35);
            let mut h = s ^ (s >> 15);
            h = h.wrapping_mul(0x2C1B_3C6D);
            h ^= h >> 12;
            *v = (h | 1) as i32;
        }
        Rng8(I8::from_array(a))
    }
    #[inline(always)]
    pub fn next(&mut self) -> I8 {
        let mut x = self.0;
        x = x ^ x.shl::<13>();
        x = x ^ x.shr::<17>();
        x = x ^ x.shl::<5>();
        self.0 = x;
        x
    }
    /// [0, 1)
    #[inline(always)]
    pub fn f01(&mut self) -> F8 {
        (self.next().shr::<9>() | 0x3F80_0000).cast_f8() - 1.0
    }
}
