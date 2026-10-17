//! The variations this program has (flam3's numbering and XML names, then
//! Fractorium's 3D, plugin and direct-colour ones from 99 up), their
//! parameters, flam3's random ranges for them (flam3.c, flam3_random) and the
//! precalcs that turn them into what the shader reads.

use crate::genome::NP;
use crate::rng::Rng;
use core::f32::consts::PI;

pub struct VarInfo {
    pub id: i32,
    pub name: &'static str,
    /// XML attribute names of the parameters, in storage order
    pub params: &'static [&'static str],
}

macro_rules! vars {
    ($($id:literal $name:literal [$($p:literal),*];)*) => {
        pub const VARS: &[VarInfo] = &[$(VarInfo { id: $id, name: $name, params: &[$($p),*] }),*];
    };
}

vars! {
    0 "linear" []; 1 "sinusoidal" []; 2 "spherical" []; 3 "swirl" []; 4 "horseshoe" []; 5 "polar" [];
    6 "handkerchief" []; 7 "heart" []; 8 "disc" []; 9 "spiral" []; 10 "hyperbolic" []; 11 "diamond" [];
    12 "ex" []; 13 "julia" []; 14 "bent" []; 15 "waves" []; 16 "fisheye" []; 17 "popcorn" [];
    18 "exponential" []; 19 "power" []; 20 "cosine" []; 21 "rings" []; 22 "fan" [];
    23 "blob" ["blob_low", "blob_high", "blob_waves"];
    24 "pdj" ["pdj_a", "pdj_b", "pdj_c", "pdj_d"];
    25 "fan2" ["fan2_x", "fan2_y"];
    26 "rings2" ["rings2_val"];
    27 "eyefish" []; 28 "bubble" []; 29 "cylinder" [];
    30 "perspective" ["perspective_angle", "perspective_dist"];
    31 "noise" [];
    32 "julian" ["julian_power", "julian_dist"];
    33 "juliascope" ["juliascope_power", "juliascope_dist"];
    34 "blur" []; 35 "gaussian_blur" [];
    36 "radial_blur" ["radial_blur_angle"];
    37 "pie" ["pie_slices", "pie_rotation", "pie_thickness"];
    38 "ngon" ["ngon_sides", "ngon_power", "ngon_circle", "ngon_corners"];
    39 "curl" ["curl_c1", "curl_c2"];
    40 "rectangles" ["rectangles_x", "rectangles_y"];
    41 "arch" []; 42 "tangent" []; 43 "square" []; 44 "rays" []; 45 "blade" []; 46 "secant2" [];
    47 "twintrian" []; 48 "cross" [];
    49 "disc2" ["disc2_rot", "disc2_twist"];
    50 "super_shape" ["super_shape_rnd", "super_shape_m", "super_shape_n1", "super_shape_n2", "super_shape_n3", "super_shape_holes"];
    51 "flower" ["flower_petals", "flower_holes"];
    52 "conic" ["conic_eccentricity", "conic_holes"];
    53 "parabola" ["parabola_height", "parabola_width"];
    54 "bent2" ["bent2_x", "bent2_y"];
    55 "bipolar" ["bipolar_shift"];
    56 "boarders" [];
    57 "butterfly" [];
    58 "cell" ["cell_size"];
    59 "cpow" ["cpow_r", "cpow_i", "cpow_power"];
    60 "curve" ["curve_xamp", "curve_yamp", "curve_xlength", "curve_ylength"];
    61 "edisc" []; 62 "elliptic" [];
    63 "escher" ["escher_beta"];
    64 "foci" [];
    65 "lazysusan" ["lazysusan_x", "lazysusan_y", "lazysusan_spin", "lazysusan_space", "lazysusan_twist"];
    66 "loonie" [];
    67 "pre_blur" [];
    68 "modulus" ["modulus_x", "modulus_y"];
    69 "oscilloscope" ["oscilloscope_separation", "oscilloscope_frequency", "oscilloscope_amplitude", "oscilloscope_damping"];
    70 "polar2" [];
    71 "popcorn2" ["popcorn2_x", "popcorn2_y", "popcorn2_c"];
    72 "scry" [];
    73 "separation" ["separation_x", "separation_xinside", "separation_y", "separation_yinside"];
    74 "split" ["split_xsize", "split_ysize"];
    75 "splits" ["splits_x", "splits_y"];
    76 "stripes" ["stripes_space", "stripes_warp"];
    77 "wedge" ["wedge_angle", "wedge_hole", "wedge_count", "wedge_swirl"];
    78 "wedge_julia" ["wedge_julia_angle", "wedge_julia_count", "wedge_julia_power", "wedge_julia_dist"];
    79 "wedge_sph" ["wedge_sph_angle", "wedge_sph_count", "wedge_sph_hole", "wedge_sph_swirl"];
    80 "whorl" ["whorl_inside", "whorl_outside"];
    81 "waves2" ["waves2_scalex", "waves2_scaley", "waves2_freqx", "waves2_freqy"];
    82 "exp" []; 83 "log" []; 84 "sin" []; 85 "cos" []; 86 "tan" []; 87 "sec" []; 88 "csc" []; 89 "cot" [];
    90 "sinh" []; 91 "cosh" []; 92 "tanh" []; 93 "sech" []; 94 "csch" []; 95 "coth" [];
    96 "auger" ["auger_freq", "auger_weight", "auger_scale", "auger_sym"];
    97 "flux" ["flux_spread"];
    98 "mobius" ["mobius_re_a", "mobius_im_a", "mobius_re_b", "mobius_im_b", "mobius_re_c", "mobius_im_c", "mobius_re_d", "mobius_im_d"];
    // Fractorium's 3D variations (they write z)
    99 "linear3D" []; 100 "Spherical3D" [];
    101 "julia3D" ["julia3D_power"];
    102 "julia3Dz" ["julia3Dz_power"];
    103 "hemisphere" []; 104 "zcone" []; 105 "zblur" []; 106 "ztranslate" []; 107 "zscale" []; 108 "blur3D" [];
    109 "curl3D" ["curl3D_cx", "curl3D_cy", "curl3D_cz"];
    110 "pie3D" ["pie3D_slices", "pie3D_rotation", "pie3D_thickness"];
    111 "sinusoidal3D" [];
    // Apophysis plugins as Fractorium has them (2D)
    112 "eclipse" ["eclipse_shift"];
    113 "epispiral" ["epispiral_n", "epispiral_thickness", "epispiral_holes"];
    114 "circlize" ["circlize_hole"];
    115 "glynnia" [];
    116 "SphericalN" ["SphericalN_Power", "SphericalN_Dist"];
    117 "xheart" ["xheart_angle", "xheart_ratio"];
    118 "petal" [];
    119 "squirrel" ["squirrel_a", "squirrel_b"];
    120 "hole" ["hole_a", "hole_inside"];
    121 "flipcircle" []; 122 "flipy" [];
    123 "bMod" ["bMod_radius", "bMod_distance"];
    124 "bSwirl" ["bSwirl_in", "bSwirl_out"];
    125 "collideoscope" ["collideoscope_a", "collideoscope_num"];
    // direct colour: they set the colour index (dc_ztransl reads it into z)
    126 "dc_linear" ["dc_linear_offset", "dc_linear_angle", "dc_linear_scale"];
    127 "dc_bubble" ["dc_bubble_centerx", "dc_bubble_centery", "dc_bubble_scale"];
    128 "dc_carpet" ["dc_carpet_origin"];
    129 "dc_cylinder" ["dc_cylinder_offset", "dc_cylinder_angle", "dc_cylinder_scale", "dc_cylinder_x", "dc_cylinder_y", "dc_cylinder_blur"];
    130 "dc_ztransl" ["dc_ztransl_x0", "dc_ztransl_x1", "dc_ztransl_factor", "dc_ztransl_overwrite", "dc_ztransl_clamp"];
}

/// flam3's variations: the first 99 entries of VARS.
const FLAM3_VARS: usize = 99;
/// The 3D variations (and dc_ztransl, which writes z too), the 2D plugin
/// ones, the direct-colour ones.
const VARS_3D: core::ops::RangeInclusive<i32> = 99..=111;
const VARS_2D: core::ops::RangeInclusive<i32> = 112..=125;
const VARS_DC: core::ops::RangeInclusive<i32> = 126..=129;

/// Whether a variation needs the 3D chaos game (walker z, direct colour):
/// the 3D ones and the dc_ ones.
pub fn needs_3d(id: i32) -> bool {
    VARS_3D.contains(&id) || VARS_DC.contains(&id) || id == 130
}

/// The flam3 variations that pass z on as w z in Fractorium (the rest give
/// z = 0, Fractorium's DefaultZ for a regular variation; bubble, cylinder and
/// flux have their own z, and waves2 is w z with its scalez of 0).
const ZPASS: [i32; 36] =
    [0, 1, 2, 3, 4, 5, 8, 9, 10, 24, 25, 26, 27, 31, 32, 33, 34, 35, 36, 38, 40, 43, 48, 55, 63, 64, 65, 66, 70, 73, 75, 77, 81, 83, 96, 98];

/// ZPASS as a 128-bit mask.
pub const ZPASS_MASK: [u32; 4] = {
    let mut m = [0u32; 4];
    let mut i = 0;
    while i < ZPASS.len() {
        m[ZPASS[i] as usize >> 5] |= 1 << (ZPASS[i] & 31);
        i += 1;
    }
    m
};

/// What iterate.comp needs for its 3D variant (walker z, the 3D camera,
/// direct colour).
pub fn three_d_glsl() -> String {
    let m = ZPASS_MASK;
    format!("#define THREE_D 1\n#define ZPASS uvec4({}u, {}u, {}u, {}u)\n", m[0], m[1], m[2], m[3])
}

pub fn info(id: i32) -> &'static VarInfo {
    VARS.iter().find(|v| v.id == id).unwrap_or(&VARS[0])
}

fn pad(p: [f32; 4]) -> [f32; NP] {
    [p[0], p[1], p[2], p[3], 0.0, 0.0, 0.0, 0.0]
}

/// The defaults of initialize_xforms (variations.c) for parameters a genome
/// lacks (used when interpolating towards a flame that has the variation).
pub fn defaults(id: i32) -> [f32; NP] {
    match id {
        // super_shape: rnd m n1 n2 n3 holes
        50 => [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        // wedge_julia: angle count power dist
        78 => [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        // mobius: a = 1, b = c = 0, d = 1 (the identity)
        98 => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        // dc_cylinder: offset angle scale x y blur (Fractorium's defaults)
        129 => [0.0, 0.0, 0.5, 0.125, 0.125, 1.0, 0.0, 0.0],
        // dc_ztransl: x0 x1 factor overwrite clamp
        130 => [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        _ => pad(defaults4(id)),
    }
}

fn defaults4(id: i32) -> [f32; 4] {
    match id {
        23 => [0.0, 1.0, 1.0, 0.0],
        30 => [0.0, 0.0, 0.0, 0.0],
        32 | 33 => [1.0, 1.0, 0.0, 0.0],
        37 => [6.0, 0.0, 0.5, 0.0],
        38 => [5.0, 3.0, 1.0, 2.0],
        53 => [1.0, 1.0, 0.0, 0.0],
        54 => [1.0, 1.0, 0.0, 0.0],
        58 => [1.0, 0.0, 0.0, 0.0],
        59 => [1.0, 0.0, 1.0, 0.0],
        60 => [0.0, 0.0, 1.0, 1.0],
        69 => [1.0, PI, 1.0, 0.0],
        77 | 79 => [0.0, 0.0, 1.0, 0.0],
        81 => [0.0, 0.0, 0.0, 0.0],
        96 => [1.0, 0.5, 1.0, 0.0],
        // Fractorium's defaults for the new ones
        101 | 102 => [2.0, 0.0, 0.0, 0.0],
        110 => [6.0, 0.5, 0.5, 0.0],
        113 => [6.0, 0.0, 1.0, 0.0],
        116 | 119 => [1.0, 1.0, 0.0, 0.0],
        120 | 123 => [1.0, 0.0, 0.0, 0.0],
        125 => [0.0, 1.0, 0.0, 0.0],
        126 | 127 => [0.0, 0.0, 1.0, 0.0],
        _ => [0.0; 4],
    }
}

/// flam3_random's parameters for a newly chosen variation.
pub fn random_params(id: i32, r: &mut Rng) -> [f32; NP] {
    match id {
        // flam3 casts before multiplying: (int) random01() * 6 is 0
        50 => {
            let n2 = r.f01() * 20.0;
            [r.f01(), 0.0, r.f01() * 40.0, n2, n2, 0.0, 0.0, 0.0]
        }
        65 => [2.0 * r.signed(), 2.0 * r.signed(), PI * r.signed(), 2.0 * r.signed(), 2.0 * r.signed(), 0.0, 0.0, 0.0],
        78 => {
            let power = (5.0 * r.f01() + 2.0).trunc();
            let count = (3.0 * r.f01() + 1.0).trunc();
            [PI * r.f01(), count, power, 1.0, 0.0, 0.0, 0.0, 0.0]
        }
        98 => std::array::from_fn(|_| r.signed()),
        129 => {
            let (off, ang, scale) = (r.signed(), PI * r.signed(), 0.25 + r.f01());
            [off, ang, scale, 0.125 + 0.5 * r.f01(), 0.125 + 0.5 * r.f01(), r.f01(), 0.0, 0.0]
        }
        130 => [0.0, 1.0, 0.5 + r.f01(), 1.0, 0.0, 0.0, 0.0, 0.0],
        _ => pad(random_params4(id, r)),
    }
}

fn random_params4(id: i32, r: &mut Rng) -> [f32; 4] {
    let u = |r: &mut Rng| r.f01();
    let s = |r: &mut Rng| r.signed();
    match id {
        23 => [0.2 + 0.5 * u(r), 0.8 + 0.4 * u(r), (2.0 + 5.0 * u(r)).trunc(), 0.0],
        24 => [3.0 * s(r), 3.0 * s(r), 3.0 * s(r), 3.0 * s(r)],
        25 => [s(r), s(r), 0.0, 0.0],
        26 => [2.0 * u(r), 0.0, 0.0, 0.0],
        30 => [u(r), 2.0 * u(r) + 1.0, 0.0, 0.0],
        32 | 33 => [(5.0 * u(r) + 2.0).trunc(), 1.0, 0.0, 0.0],
        36 => [2.0 * u(r) - 1.0, 0.0, 0.0, 0.0],
        // flam3 casts before multiplying: (int) 10.0 * random01()
        37 => [10.0 * u(r), 2.0 * PI * s(r), u(r), 0.0],
        38 => {
            // (int) flam3_random01() * 10 + 3 is always 3 in flam3
            let circle = 3.0 * u(r);
            let power = 3.0 * u(r) + 1.0;
            [3.0, power, circle, 2.0 * u(r) * circle]
        }
        39 => [u(r), u(r), 0.0, 0.0],
        40 => [u(r), u(r), 0.0, 0.0],
        49 => [0.5 * u(r), 0.5 * u(r), 0.0, 0.0],
        51 => [4.0 * u(r), u(r), 0.0, 0.0],
        52 => [u(r), u(r), 0.0, 0.0],
        53 => [0.5 + u(r), 0.5 + u(r), 0.0, 0.0],
        54 => [3.0 * (u(r) - 0.5), 3.0 * (u(r) - 0.5), 0.0, 0.0],
        55 => [2.0 * u(r) - 1.0, 0.0, 0.0, 0.0],
        58 => [2.0 * u(r) + 0.5, 0.0, 0.0, 0.0],
        // a cpow power of 0 divides by zero; flam3 lets it (bad values)
        59 => [3.0 * u(r), u(r) - 0.5, (5.0 * u(r)).trunc().max(1.0), 0.0],
        60 => [5.0 * (u(r) - 0.5), 4.0 * (u(r) - 0.5), 2.0 * (u(r) + 0.5), 2.0 * (u(r) + 0.5)],
        63 => [PI * s(r), 0.0, 0.0, 0.0],
        68 => [s(r), s(r), 0.0, 0.0],
        69 => [1.0 + s(r), PI * s(r), 1.0 + 2.0 * u(r), u(r)],
        71 => [0.2 * u(r), 0.2 * u(r), 5.0 * u(r), 0.0],
        73 => [1.0 + s(r), s(r), 1.0 + s(r), s(r)],
        74 => [s(r), s(r), 0.0, 0.0],
        75 => [s(r), s(r), 0.0, 0.0],
        76 => [u(r), 5.0 * u(r), 0.0, 0.0],
        77 => [PI * u(r), 0.5 * s(r), (5.0 * u(r)).floor() + 1.0, u(r)],
        79 => [PI * u(r), (5.0 * u(r)).floor() + 1.0, 0.5 * s(r), u(r)],
        80 => [u(r), u(r), 0.0, 0.0],
        81 => [0.5 + u(r), 0.5 + u(r), 4.0 * u(r), 4.0 * u(r)],
        96 => [(5.0 * u(r)).floor() + 1.0, 0.5 + u(r) / 2.0, u(r), 0.0],
        97 => [0.5 + u(r) / 2.0, 0.0, 0.0, 0.0],
        // julia3D(z): Fractorium's Random, rand(5) + 2 and negative half the time
        101 | 102 => {
            let n = (r.below(5) + 2) as f32;
            [if r.bit() { -n } else { n }, 0.0, 0.0, 0.0]
        }
        // Fractorium's generic Random: each parameter in [-1, 1)
        109 => [s(r), s(r), s(r), 0.0],
        // pie3D: Fractorium's Random (slices rounded, at least 1)
        110 => [(10.0 * u(r) + 0.5).floor().max(1.0), 2.0 * PI * s(r), u(r), 0.0],
        // the plugins have no Random of their own in Fractorium (it would put
        // every parameter in [-1, 1)); these ranges keep them in shape
        112 => [2.0 * s(r), 0.0, 0.0, 0.0],
        113 => [(1.0 + 6.0 * u(r)).floor(), 0.3 * u(r), s(r), 0.0],
        114 => [0.5 * s(r), 0.0, 0.0, 0.0],
        116 => [(1.0 + 6.0 * u(r)).floor(), 0.5 + 1.5 * u(r), 0.0, 0.0],
        117 => [2.0 * s(r), s(r), 0.0, 0.0],
        119 => [0.5 + 1.5 * u(r), 0.5 + 1.5 * u(r), 0.0, 0.0],
        120 => [0.25 + 1.5 * u(r), if r.bit() { 1.0 } else { 0.0 }, 0.0, 0.0],
        123 => [0.5 + 1.5 * u(r), 2.0 * u(r), 0.0, 0.0],
        124 => [s(r), s(r), 0.0, 0.0],
        125 => [s(r), (1.0 + 6.0 * u(r)).floor(), 0.0, 0.0],
        126 => [s(r), PI * s(r), 0.25 + u(r), 0.0],
        127 => [0.5 * s(r), 0.5 * s(r), 0.25 + u(r), 0.0],
        128 => [s(r), 0.0, 0.0, 0.0],
        _ => defaults4(id),
    }
}

/// The parameters as the shader reads them (flam3's xform_precalc).
pub fn gpu_params(id: i32, p: [f32; NP]) -> [f32; NP] {
    match id {
        // super_shape: (m / 4, n2, n3, -1 / n1), (rnd, holes)
        50 => [p[1] / 4.0, p[3], p[4], -1.0 / p[2], p[0], p[5], 0.0, 0.0],
        // wedge_julia: (angle, count, power, cf), (rN, cn)
        78 => {
            let cf = 1.0 - p[0] * p[1] / PI * 0.5;
            [p[0], p[1], p[2], cf, p[2].abs(), p[3] / p[2] / 2.0, 0.0, 0.0]
        }
        // dc_cylinder: (offset, sin angle, cos angle, 1 / scale), (x, y, blur)
        129 => {
            let (sa, ca) = p[1].sin_cos();
            [p[0], sa, ca, 1.0 / if p[2] == 0.0 { 1e-5 } else { p[2] }, p[3], p[4], p[5], 0.0]
        }
        // dc_ztransl: (min x, max x - min x, factor, overwrite), (clamp)
        130 => [p[0].min(p[1]), zeps(p[0].max(p[1]) - p[0].min(p[1])), p[2], p[3], p[4], 0.0, 0.0, 0.0],
        _ => {
            let g = gpu_params4(id, [p[0], p[1], p[2], p[3]]);
            [g[0], g[1], g[2], g[3], p[4], p[5], p[6], p[7]]
        }
    }
}

fn gpu_params4(id: i32, p: [f32; 4]) -> [f32; 4] {
    match id {
        30 => {
            let (s, c) = (p[0] * PI / 2.0).sin_cos();
            [p[1], s, p[1] * c, 0.0]
        }
        32 | 33 => [p[0], p[0].abs(), p[1] / p[0] / 2.0, 0.0],
        36 => {
            let (s, c) = (p[0] * PI / 2.0).sin_cos();
            [s, c, 0.0, 0.0]
        }
        49 => {
            let add = p[1];
            let (mut sa, mut ca) = add.sin_cos();
            ca -= 1.0;
            let k = if add > 2.0 * PI {
                1.0 + add - 2.0 * PI
            } else if add < -2.0 * PI {
                1.0 + add + 2.0 * PI
            } else {
                1.0
            };
            sa *= k;
            ca *= k;
            [p[0] * PI, sa, ca, 0.0]
        }
        63 => {
            let (s, c) = p[0].sin_cos();
            [0.5 * (1.0 + c), 0.5 * s, 0.0, 0.0]
        }
        // julia3D: (n, |n|, (1 / n - 1) / 2); julia3Dz: (n, |n|, 1 / n / 2)
        101 => [p[0], p[0].abs(), (1.0 / p[0] - 1.0) / 2.0, 0.0],
        102 => [p[0], p[0].abs(), 1.0 / p[0] / 2.0, 0.0],
        // curl3D: (cx, cy, cz, cx^2 + cy^2 + cz^2)
        109 => [p[0], p[1], p[2], p[0] * p[0] + p[1] * p[1] + p[2] * p[2]],
        // xheart: (cos, sin) of pi/4 + pi/8 angle, 6 + 2 ratio
        117 => {
            let (s, c) = (PI / 4.0 + 0.5 * PI / 4.0 * p[0]).sin_cos();
            [c, s, 6.0 + 2.0 * p[1], 0.0]
        }
        119 => [zeps(p[0]), zeps(p[1]), 0.0, 0.0],
        // collideoscope: (num / pi, pi / num, pi a / num) with num never 0
        125 => {
            let n = zeps(p[1]);
            [n / PI, PI / n, PI * p[0] / n, 0.0]
        }
        // dc_linear: (offset, sin angle, cos angle, 1 / scale)
        126 => {
            let (s, c) = p[1].sin_cos();
            [p[0], s, c, 1.0 / if p[2] == 0.0 { 1e-5 } else { p[2] }]
        }
        // dc_bubble: (centre x, centre y, 1 / scale)
        127 => [p[0], p[1], 1.0 / if p[2] == 0.0 { 1e-5 } else { p[2] }, 0.0],
        // dc_carpet: h = origin / 10
        128 => [0.1 * p[0], 0.0, 0.0, 0.0],
        _ => p,
    }
}

/// Fractorium's Zeps: 0 becomes the float epsilon (a divisor that is never 0).
pub fn zeps(x: f32) -> f32 {
    if x == 0.0 { f32::EPSILON } else { x }
}

/// How many parameters the shader / lanes read for a variation.
pub fn gpu_param_count(id: i32) -> usize {
    match id {
        50 | 78 => 6,
        65 | 130 => 5,
        129 => 7,
        98 => 8,
        // precalcs can turn 1-2 XML parameters into 2-3 shader ones
        _ if info(id).params.is_empty() => 0,
        _ => 4,
    }
}

/// The variations flam3-genome leaves out of random genomes (noise, blur,
/// gaussian_blur, radial_blur, ngon, square, rays, cross, pre_blur,
/// separation, split, splits).
const NOT_RANDOM: [i32; 12] = [31, 34, 35, 36, 38, 43, 44, 48, 67, 73, 74, 75];

/// A variation for a random genome: uniform over the rest of flam3's.
pub fn random_id(r: &mut Rng) -> i32 {
    loop {
        let id = VARS[r.below(FLAM3_VARS as u32) as usize].id;
        if !NOT_RANDOM.contains(&id) {
            return id;
        }
    }
}

/// A variation for a random genome with the new ones: one draw in sixteen a
/// 3D one (dc_ztransl among them), else uniform over flam3's random set, the
/// 2D plugins and the direct-colour ones.
#[allow(dead_code)] // for the v2 breeder
pub fn random_id_v2(r: &mut Rng) -> i32 {
    if r.below(16) == 0 {
        let n = VARS_3D.end() - VARS_3D.start() + 2;
        let k = r.below(n as u32) as i32;
        return if k == n - 1 { 130 } else { VARS_3D.start() + k };
    }
    let flat = FLAM3_VARS as u32 + (VARS_2D.end() - VARS_2D.start() + 1) as u32 + (VARS_DC.end() - VARS_DC.start() + 1) as u32;
    loop {
        let k = r.below(flat) as usize;
        // the plugins and the dc ones are numbered one after the other
        let id = if k < FLAM3_VARS { VARS[k].id } else { VARS_2D.start() + (k - FLAM3_VARS) as i32 };
        if !NOT_RANDOM.contains(&id) {
            return id;
        }
    }
}
