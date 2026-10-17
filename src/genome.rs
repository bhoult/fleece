//! A flame genome as flam3 has it: xforms (an affine, weighted variations,
//! a post transform, a colour), an optional final xform, a palette, the
//! camera and the tone settings. `random` is flam3_random; `sample` flattens
//! a genome (at some rotation) for the chaos game.

use crate::rng::Rng;
use crate::sample::{MAXX, NV, PICKN, Sample, XfGpu};
use crate::var;

/// Parameters per variation (mobius has eight).
pub const NP: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VarUse {
    pub id: i32,
    pub w: f32,
    /// flam3's parameters for this variation (var::VARS gives their names)
    pub p: [f32; NP],
}

pub type Affine = [[f32; 2]; 3];
pub const IDENTITY: Affine = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];

#[derive(Clone, Debug)]
pub struct Xform {
    /// flam3's "density": the chance of being picked, relative to the others
    pub weight: f32,
    pub color: f32,
    pub speed: f32,
    pub opacity: f32,
    /// turns with the loop (symmetry xforms don't)
    pub animate: bool,
    pub c: Affine,
    pub post: Affine,
    pub vars: Vec<VarUse>,
    /// xaos: weight multipliers for the xform after this one (empty: all 1)
    pub xaos: Vec<f32>,
    /// added to match another genome's xform count (flam3's padding)
    pub padding: bool,
    /// flam3's <motion> elements: values that swing over each loop
    pub motion: Vec<Motion>,
}

/// One <motion> element: deltas added to its xform, scaled by a function of
/// the loop's phase (flam3's apply_motion_parameters).
#[derive(Clone, Debug, Default)]
pub struct Motion {
    pub freq: f32,
    /// 0 sin, 1 triangle, 2 hill
    pub func: u8,
    pub weight: f32,
    pub color: f32,
    pub speed: f32,
    pub opacity: f32,
    /// variation weight deltas (w) and parameter deltas (p)
    pub vars: Vec<VarUse>,
    pub c: Affine,
    pub post: Affine,
    /// added by fleece (motion for every sheep), played only while that is on
    pub extra: bool,
}

impl Motion {
    /// flam3's motion_funcs: cyclic, zero at whole loops, at most 1.
    pub fn amount(&self, blend: f32) -> f32 {
        let t = self.freq * blend;
        match self.func {
            1 => {
                let fr = t.rem_euclid(1.0);
                if fr <= 0.25 {
                    4.0 * fr
                } else if fr <= 0.75 {
                    -4.0 * fr + 2.0
                } else {
                    4.0 * fr - 4.0
                }
            }
            2 => (1.0 - (2.0 * std::f32::consts::PI * t).cos()) * 0.5,
            _ => (2.0 * std::f32::consts::PI * t).sin(),
        }
    }
}

impl Xform {
    /// The xform with its motion applied at loop phase `blend` (0..1); the
    /// extra (fleece's own) motion only with `extra`.
    pub fn moved(&self, blend: f32, extra: bool) -> Xform {
        let mut x = self.clone();
        for m in self.motion.iter().filter(|m| extra || !m.extra) {
            let k = m.amount(blend);
            x.weight += m.weight * k;
            x.color += m.color * k;
            x.speed += m.speed * k;
            x.opacity += m.opacity * k;
            for i in 0..3 {
                for j in 0..2 {
                    x.c[i][j] += m.c[i][j] * k;
                    x.post[i][j] += m.post[i][j] * k;
                }
            }
            for d in &m.vars {
                match x.vars.iter_mut().find(|v| v.id == d.id) {
                    Some(v) => {
                        v.w += d.w * k;
                        for (vp, dp) in v.p.iter_mut().zip(d.p.iter()) {
                            *vp += dp * k;
                        }
                    }
                    None => {
                        let mut p = crate::var::defaults(d.id);
                        for (vp, dp) in p.iter_mut().zip(d.p.iter()) {
                            *vp += dp * k;
                        }
                        x.vars.push(VarUse { id: d.id, w: d.w * k, p });
                    }
                }
            }
        }
        x.weight = x.weight.max(0.0);
        x.color = x.color.clamp(0.0, 1.0);
        x.speed = x.speed.clamp(0.0, 1.0);
        x.motion.clear();
        x
    }
}

impl Genome {
    /// Whether any xform has motion elements.
    pub fn has_motion(&self) -> bool {
        self.xforms.iter().chain(self.fin.iter()).any(|x| !x.motion.is_empty())
    }

    /// The genome with its motion elements applied at loop phase `blend`
    /// (fleece's extra ones only with `extra`).
    pub fn moved(&self, blend: f32, extra: bool) -> Genome {
        let mut g = self.clone();
        for x in g.xforms.iter_mut() {
            *x = x.moved(blend, extra);
        }
        if let Some(f) = g.fin.as_mut() {
            *f = f.moved(blend, extra);
        }
        g
    }
}

impl Xform {
    pub fn new() -> Xform {
        Xform {
            weight: 0.0,
            color: 0.0,
            speed: 0.5,
            opacity: 1.0,
            animate: true,
            c: IDENTITY,
            post: IDENTITY,
            vars: vec![VarUse { id: 0, w: 1.0, p: [0.0; NP] }],
            xaos: Vec::new(),
            padding: false,
            motion: Vec::new(),
        }
    }
    pub fn has_post(&self) -> bool {
        self.post != IDENTITY
    }
    pub fn var(&self, id: i32) -> Option<&VarUse> {
        self.vars.iter().find(|v| v.id == id)
    }
}

#[derive(Clone, Debug)]
pub struct Genome {
    pub xforms: Vec<Xform>,
    pub fin: Option<Xform>,
    pub palette: Vec<[f32; 3]>,
    /// which of flam3's palettes, and its hue rotation (0..1)
    pub palette_index: i32,
    pub hue: f32,
    pub center: [f32; 2],
    /// pixels per unit as a fraction of the image width (flam3 stores pixels
    /// per unit for a given size; scale-free here)
    pub scale: f32,
    /// degrees
    pub rotate: f32,
    pub brightness: f32,
    pub gamma: f32,
    pub gamma_threshold: f32,
    pub vibrancy: f32,
    pub highlight: f32,
    pub background: [f32; 3],
    /// density estimation: radius, minimum, curve (flam3 defaults 9, 0, 0.4)
    pub de: [f32; 3],
    /// palette_mode="linear": colour indices blend neighbouring palette
    /// entries instead of picking one
    pub palette_linear: bool,
    pub symmetry: i32,
    /// where it came from, for the HUD
    pub name: String,
    /// a real sheep from the genebank, as it was: keep its framing and colours
    pub curated: bool,
    /// the designer of a hand-made sheep (flam3's nick attribute)
    pub nick: String,
    /// is, or descends from, an evolved Electric Sheep (CC BY-NC: its
    /// descendants stay non-commercial however they are crossed)
    pub nc: bool,
    /// flam3's spatial supersampling (buckets per pixel on each axis; the
    /// Electric Sheep server rendered at 4) and its gaussian filter's radius
    /// in pixels (flam3's default 0.5; the server's sheep say 1)
    pub supersample: u32,
    pub filter: f32,
    /// a 3D camera (Apophysis 7X's and Fractorium's cam_* attributes)
    pub cam: Cam3,
}

/// The 3D camera: the flame is tilted by `pitch` and turned by `yaw`
/// (radians) about the view's centre, seen with `persp` perspective; `zpos`
/// moves the points along z and `dof` blurs them with their distance from
/// the focal plane. All zero: the flat flam3 camera.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cam3 {
    pub pitch: f32,
    pub yaw: f32,
    pub persp: f32,
    pub zpos: f32,
    pub dof: f32,
}

impl Cam3 {
    pub fn is_flat(&self) -> bool {
        *self == Cam3::default()
    }
}

static PALETTES: &[u8] = include_bytes!("../assets/flam3-palettes.bin");
pub const NPALETTES: i32 = (PALETTES.len() / 768) as i32;

/// flam3's palette `n` with its hue turned by `hue` (0..1 of a full turn).
pub fn palette(n: i32, hue: f32) -> Vec<[f32; 3]> {
    let n = n.rem_euclid(NPALETTES) as usize;
    (0..256)
        .map(|i| {
            let b = &PALETTES[n * 768 + i * 3..];
            let rgb = [b[0] as f32 / 255.0, b[1] as f32 / 255.0, b[2] as f32 / 255.0];
            if hue != 0.0 {
                let mut hsv = rgb2hsv(rgb);
                hsv[0] = (hsv[0] + hue * 6.0).rem_euclid(6.0);
                hsv2rgb(hsv)
            } else {
                rgb
            }
        })
        .collect()
}

/// flam3's rgb2hsv: h in [0, 6).
pub fn rgb2hsv(c: [f32; 3]) -> [f32; 3] {
    let (r, g, b) = (c[0], c[1], c[2]);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let del = max - min;
    let v = max;
    let s = if max != 0.0 { del / max } else { 0.0 };
    let mut h = 0.0;
    if s != 0.0 {
        let rc = (max - r) / del;
        let gc = (max - g) / del;
        let bc = (max - b) / del;
        h = if r == max {
            bc - gc
        } else if g == max {
            2.0 + rc - bc
        } else {
            4.0 + gc - rc
        };
        if h < 0.0 {
            h += 6.0;
        }
    }
    [h, s, v]
}

pub fn hsv2rgb(c: [f32; 3]) -> [f32; 3] {
    let (mut h, s, v) = (c[0], c[1], c[2]);
    while h >= 6.0 {
        h -= 6.0;
    }
    while h < 0.0 {
        h += 6.0;
    }
    let j = h.floor();
    let f = h - j;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match j as i32 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// flam3's default xform counts for random genomes.
const XFORM_DISTRIB: [usize; 14] = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6];
const SYM_DISTRIB: [i32; 15] = [-4, -3, -2, -2, -2, -1, -1, -1, 2, 2, 2, 3, 3, 4, 4];

impl Genome {
    pub fn empty() -> Genome {
        Genome {
            xforms: Vec::new(),
            fin: None,
            palette: palette(0, 0.0),
            palette_index: 0,
            hue: 0.0,
            center: [0.0, 0.0],
            scale: 0.25,
            rotate: 0.0,
            brightness: 4.0,
            gamma: 4.0,
            gamma_threshold: 0.01,
            vibrancy: 1.0,
            highlight: -1.0,
            background: [0.0; 3],
            de: [9.0, 0.0, 0.4],
            palette_linear: false,
            symmetry: 0,
            name: String::new(),
            curated: false,
            nick: String::new(),
            nc: false,
            supersample: 1,
            filter: 0.5,
            cam: Cam3::default(),
        }
    }

    /// flam3_random as flam3-genome calls it (flam3.c): 2-6 xforms of equal
    /// weight with random affines; each xform half the time copies the
    /// previous one's variations, else gets 2+ random ones (fewer more
    /// likely); a post transform 40 % of the time; a final xform 15 %;
    /// symmetry a quarter of the time without one; a random palette,
    /// hue-turned one time in eight.
    pub fn random(seed: u32) -> Genome {
        let mut r = Rng::new(seed);
        let mut g = Genome::empty();
        g.hue = if r.below(8) != 0 { 0.0 } else { r.f01() };
        g.palette_index = r.below(NPALETTES as u32) as i32;
        g.palette = palette(g.palette_index, g.hue);
        let n = r.pick(&XFORM_DISTRIB);
        let add_final = r.f01() < 0.15;
        let samed = r.bit();
        let postid = r.f01() < 0.6;
        let samepost = r.bit();
        let total = n + add_final as usize;
        for i in 0..total {
            let mut x = Xform::new();
            x.weight = 1.0 / total as f32;
            x.color = (i & 1) as f32;
            x.speed = 0.5;
            for j in 0..3 {
                for k in 0..2 {
                    x.c[j][k] = r.signed();
                }
            }
            let is_final = add_final && i == total - 1;
            if !is_final {
                if !postid {
                    if samepost || i == 0 {
                        for j in 0..3 {
                            for k in 0..2 {
                                x.post[j][k] = r.signed();
                            }
                        }
                    } else {
                        x.post = g.xforms[0].post;
                    }
                }
                x.vars = if samed && i > 0 {
                    g.xforms[i - 1].vars.clone()
                } else {
                    let mut k = 2;
                    while r.bit() {
                        k += 1;
                    }
                    random_vars(&mut r, k)
                };
                g.xforms.push(x);
            } else {
                let k = if r.bit() { 2 } else { 1 };
                x.vars = random_vars(&mut r, k);
                x.animate = false;
                g.fin = Some(x);
            }
        }
        if !add_final && r.below(4) == 0 {
            g.add_symmetry(&mut r, 0);
        }
        g.name = format!("random {seed:08x}");
        g
    }

    /// flam3_add_symmetry: -k adds a mirror and k-fold rotation, k a k-fold
    /// rotation, each as a weight-1 linear xform that keeps its colour still.
    pub fn add_symmetry(&mut self, r: &mut Rng, sym: i32) {
        let mut sym = sym;
        if sym == 0 {
            sym = if r.bit() {
                r.pick(&SYM_DISTRIB)
            } else if r.below(32) != 0 {
                r.below(13) as i32 - 6
            } else {
                r.below(51) as i32 - 25
            };
        }
        // room for the symmetry xforms
        let room = MAXX as i32 - self.xforms.len() as i32;
        let need = |s: i32| if s < 0 { -s } else { s - 1 };
        while sym.abs() > 1 && need(sym) > room {
            sym -= sym.signum();
        }
        if sym == 0 || sym == 1 {
            return;
        }
        self.symmetry = sym;
        let sym_xf = |c: Affine, color: f32| Xform {
            weight: 1.0,
            color,
            speed: 0.0,
            opacity: 1.0,
            animate: false,
            c,
            post: IDENTITY,
            vars: vec![VarUse { id: 0, w: 1.0, p: [0.0; NP] }],
            xaos: Vec::new(),
            padding: false,
            motion: Vec::new(),
        };
        if sym < 0 {
            self.xforms.push(sym_xf([[-1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], 1.0));
            sym = -sym;
        }
        let a = 2.0 * std::f64::consts::PI / sym as f64;
        let round6 = |x: f64| ((x * 1e6).round() / 1e6) as f32;
        for k in 1..sym {
            let (s, c) = (k as f64 * a).sin_cos();
            let color = if sym < 3 { 0.0 } else { (k as f32 - 1.0) / (sym as f32 - 2.0) };
            let (c00, c01) = (round6(c), round6(s));
            self.xforms.push(sym_xf([[c00, c01], [-c01, c00], [0.0, 0.0]], color));
        }
    }

    /// Flattens the genome for the chaos game: the xforms turned by `angle`
    /// (radians; flam3_rotate turns each animated xform's linear part) and
    /// the camera for a `w` x `h` histogram.
    pub fn sample(&self, angle: f32, w: u32, h: u32, out: &mut Sample) {
        let n = self.xforms.len().min(MAXX);
        out.info = [n as i32, self.fin.is_some() as i32, 0, self.palette_linear as i32];
        let (s, c) = angle.sin_cos();
        for (k, x) in self.xforms.iter().take(n).enumerate() {
            let mut a = x.c;
            if x.animate && angle != 0.0 {
                a = rotate_linear(&x.c, s, c);
            }
            out.xf[k] = flatten(x, &a);
        }
        for k in n..MAXX {
            out.xf[k] = XfGpu::default();
            out.xf[k].vid = [-1; NV];
        }
        match &self.fin {
            Some(f) => out.xf[MAXX] = flatten(f, &f.c),
            None => {
                out.xf[MAXX] = XfGpu::default();
                out.xf[MAXX].vid = [-1; NV];
            }
        }
        // the choice tables: one row, or one per previous xform with xaos
        let xaos = self.xforms.iter().take(n).any(|x| x.xaos.iter().take(n).any(|&v| v != 1.0));
        out.info[2] = xaos as i32;
        let rows = if xaos { n } else { 1 };
        for row in 0..rows {
            let wts: Vec<f32> = (0..n)
                .map(|j| {
                    let m = if xaos { self.xforms[row].xaos.get(j).copied().unwrap_or(1.0) } else { 1.0 };
                    (self.xforms[j].weight * m).max(0.0)
                })
                .collect();
            fill_pick(&wts, &mut out.pick[row * PICKN..(row + 1) * PICKN]);
        }
        for (i, p) in self.palette.iter().enumerate().take(256) {
            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
            out.pal[i] = q(p[0]) | q(p[1]) << 8 | q(p[2]) << 16 | 255 << 24;
        }
        // the camera (rect.c): turn about the centre, then pixels per unit
        let ppu = self.scale * w as f32;
        let (rs, rc) = (self.rotate.to_radians()).sin_cos();
        let (cx, cy) = (self.center[0], self.center[1]);
        // px = ppu (rc (x - cx) - rs (y - cy)) + w/2; py = ppu (rs (x - cx) + rc (y - cy)) + h/2
        out.cam = [
            ppu * rc,
            -ppu * rs,
            ppu * (-rc * cx + rs * cy) + w as f32 * 0.5,
            0.0,
            ppu * rs,
            ppu * rc,
            ppu * (-rs * cx - rc * cy) + h as f32 * 0.5,
            0.0,
        ];
    }
}

/// flam3 picks 2+ variations at random (repeats merge), with random weights
/// summing to 1; at most five are kept (truncate_variations).
fn random_vars(r: &mut Rng, n: usize) -> Vec<VarUse> {
    let mut v: Vec<VarUse> = Vec::new();
    for _ in 0..n {
        let id = var::random_id(r);
        let w = r.f01();
        match v.iter_mut().find(|u| u.id == id) {
            Some(u) => u.w = w,
            None => v.push(VarUse { id, w, p: var::random_params(id, r) }),
        }
    }
    v.sort_by(|a, b| b.w.total_cmp(&a.w));
    v.truncate(5);
    let sum: f32 = v.iter().map(|u| u.w).sum();
    if sum <= 0.0 {
        return vec![VarUse { id: 0, w: 1.0, p: [0.0; NP] }];
    }
    for u in v.iter_mut() {
        u.w /= sum;
    }
    v
}

/// The 2x2 part turned by the angle with sine `s`, cosine `c` (flam3_rotate:
/// the rotation matrix times the xform's coefficients).
pub fn rotate_linear(a: &Affine, s: f32, c: f32) -> Affine {
    // T = [[c, s], [-s, c]] in flam3's c[row][col] layout; result = T * a for the
    // two linear rows; the translation row stays
    [
        [c * a[0][0] - s * a[0][1], s * a[0][0] + c * a[0][1]],
        [c * a[1][0] - s * a[1][1], s * a[1][0] + c * a[1][1]],
        a[2],
    ]
}

fn flatten(x: &Xform, a: &Affine) -> XfGpu {
    let mut o = XfGpu { c: [a[0][0], a[0][1], a[1][0], a[1][1], a[2][0], a[2][1], x.color, x.speed], ..Default::default() };
    let p = &x.post;
    o.post = [p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1], x.opacity, if x.has_post() { 1.0 } else { 0.0 }];
    o.vid = [-1; NV];
    for (j, v) in x.vars.iter().filter(|v| v.w != 0.0).take(NV).enumerate() {
        o.vid[j] = v.id;
        o.vw[j] = v.w;
        o.vp[j] = var::gpu_params(v.id, v.p);
    }
    o
}

/// PICKN xform indices with each xform's share proportional to its weight.
fn fill_pick(w: &[f32], out: &mut [u8]) {
    let total: f32 = w.iter().sum();
    if total <= 0.0 {
        out.fill(0);
        return;
    }
    let mut j = 0;
    let mut acc = w[0];
    let n = out.len() as f32;
    for (e, o) in out.iter_mut().enumerate() {
        let t = (e as f32 + 0.5) / n * total;
        while acc < t && j + 1 < w.len() {
            j += 1;
            acc += w[j];
        }
        // never an xform of weight zero
        while w[j] <= 0.0 && j + 1 < w.len() {
            j += 1;
            acc += w[j];
        }
        *o = j as u8;
    }
}

impl Genome {
    /// flam3's XML (one <flame>), for flam3-render, Apophysis, Fractorium.
    pub fn to_flam3(&self, w: u32, h: u32) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let f = |v: f32| {
            let t = format!("{v:.6}");
            let t = t.trim_end_matches('0').trim_end_matches('.').to_string();
            if t == "-0" { "0".to_string() } else { t }
        };
        let aff = |a: &Affine| format!("{} {} {} {} {} {}", f(a[0][0]), f(a[0][1]), f(a[1][0]), f(a[1][1]), f(a[2][0]), f(a[2][1]));
        let _ = writeln!(
            s,
            "<flame name=\"{}\"{} version=\"fleece\" size=\"{w} {h}\" center=\"{} {}\" scale=\"{}\" rotate=\"{}\" brightness=\"{}\" gamma=\"{}\" gamma_threshold=\"{}\" vibrancy=\"{}\" highlight_power=\"{}\" background=\"{} {} {}\" estimator_radius=\"{}\" estimator_minimum=\"{}\" estimator_curve=\"{}\" palette_mode=\"{}\" supersample=\"{}\" filter=\"{}\" filter_shape=\"gaussian\"{} interpolation=\"linear\" palette_interpolation=\"hsv_circular\" symmetry=\"{}\">",
            self.name.replace('"', "'"),
            if self.nick.is_empty() { String::new() } else { format!(" nick=\"{}\"", self.nick.replace('"', "'")) }
                + if self.nc { " fleece_nc=\"1\"" } else { "" },
            f(self.center[0]),
            f(self.center[1]),
            f(self.scale * w as f32),
            f(self.rotate),
            f(self.brightness),
            f(self.gamma),
            f(self.gamma_threshold),
            f(self.vibrancy),
            f(self.highlight),
            f(self.background[0]),
            f(self.background[1]),
            f(self.background[2]),
            f(self.de[0]),
            f(self.de[1]),
            f(self.de[2]),
            if self.palette_linear { "linear" } else { "step" },
            self.supersample,
            f(self.filter),
            if self.cam.is_flat() {
                String::new()
            } else {
                format!(
                    " cam_pitch=\"{}\" cam_yaw=\"{}\" cam_perspective=\"{}\" cam_zpos=\"{}\" cam_dof=\"{}\"",
                    f(self.cam.pitch),
                    f(self.cam.yaw),
                    f(self.cam.persp),
                    f(self.cam.zpos),
                    f(self.cam.dof)
                )
            },
            self.symmetry
        );
        let xf = |s: &mut String, tag: &str, x: &Xform, fin: bool| {
            let _ = write!(s, "   <{tag} ");
            if !fin {
                let _ = write!(s, "weight=\"{}\" ", f(x.weight));
            }
            let _ = write!(
                s,
                "color=\"{}\" color_speed=\"{}\" animate=\"{}\" opacity=\"{}\" ",
                f(x.color),
                f(x.speed),
                x.animate as i32,
                f(x.opacity)
            );
            for v in &x.vars {
                let info = crate::var::info(v.id);
                let _ = write!(s, "{}=\"{}\" ", info.name, f(v.w));
                for (i, p) in info.params.iter().enumerate() {
                    let _ = write!(s, "{p}=\"{}\" ", f(v.p[i]));
                }
            }
            let _ = write!(s, "coefs=\"{}\"", aff(&x.c));
            if x.has_post() {
                let _ = write!(s, " post=\"{}\"", aff(&x.post));
            }
            if !x.xaos.is_empty() {
                let c: Vec<String> = x.xaos.iter().map(|&v| f(v)).collect();
                let _ = write!(s, " chaos=\"{}\"", c.join(" "));
            }
            if x.motion.is_empty() {
                let _ = writeln!(s, "/>");
            } else {
                let _ = writeln!(s, ">");
                for m in &x.motion {
                    let func = ["sin", "triangle", "hill"][m.func.min(2) as usize];
                    let _ = write!(s, "      <motion motion_frequency=\"{}\" motion_function=\"{func}\"", f(m.freq));
                    if m.extra {
                        let _ = write!(s, " fleece_extra=\"1\"");
                    }
                    for (k, v) in [("weight", m.weight), ("color", m.color), ("color_speed", m.speed), ("opacity", m.opacity)] {
                        if v != 0.0 {
                            let _ = write!(s, " {k}=\"{}\"", f(v));
                        }
                    }
                    for d in &m.vars {
                        let info = crate::var::info(d.id);
                        if d.w != 0.0 {
                            let _ = write!(s, " {}=\"{}\"", info.name, f(d.w));
                        }
                        for (i, p) in info.params.iter().enumerate() {
                            if d.p[i] != 0.0 {
                                let _ = write!(s, " {p}=\"{}\"", f(d.p[i]));
                            }
                        }
                    }
                    if m.c != [[0.0; 2]; 3] {
                        let _ = write!(s, " coefs=\"{}\"", aff(&m.c));
                    }
                    if m.post != [[0.0; 2]; 3] {
                        let _ = write!(s, " post=\"{}\"", aff(&m.post));
                    }
                    let _ = writeln!(s, "/>");
                }
                let _ = writeln!(s, "   </{tag}>");
            }
        };
        for x in &self.xforms {
            xf(&mut s, "xform", x, false);
        }
        if let Some(x) = &self.fin {
            xf(&mut s, "finalxform", x, true);
        }
        for (i, c) in self.palette.iter().enumerate() {
            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as i32;
            let _ = writeln!(s, "   <color index=\"{i}\" rgb=\"{} {} {}\"/>", q(c[0]), q(c[1]), q(c[2]));
        }
        s.push_str("</flame>\n");
        s
    }
}
