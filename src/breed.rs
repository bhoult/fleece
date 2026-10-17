//! Where new sheep come from, as on the Electric Sheep server: flam3-genome's
//! mutation and crossover of the flock's sheep (picked by rating), random
//! genomes, the bounding-box framing, and the test render that rejects
//! flames that are too dark, too full or blown out.

use crate::cpu;
use crate::genome::{self, Genome};
use crate::interp;
use crate::rng::Rng;
use crate::sample::{MAXX, Sample};

/// flam3_mutate: one of seven mutations (flam3.c's probabilities).
pub fn mutate(g: &Genome, r: &mut Rng) -> Genome {
    let mut g = g.clone();
    g.curated = false;
    let x = r.f01();
    let what;
    if x < 0.1 {
        // all variations from a random genome with as many xforms
        let m = random_with(r.next_u32(), g.xforms.len());
        for (a, b) in g.xforms.iter_mut().zip(m.xforms.iter()) {
            a.vars = b.vars.clone();
        }
        what = "variations";
    } else if x < 0.3 {
        let m = random_with(r.next_u32(), 2);
        let k = r.below(g.xforms.len() as u32) as usize;
        if g.xforms.len() <= 2 {
            g.xforms[k].c[2] = m.xforms[0].c[2];
        } else {
            g.xforms[k].c = m.xforms[0].c;
        }
        what = "one xform";
    } else if x < 0.5 {
        g.add_symmetry(r, 0);
        what = "symmetry";
    } else if x < 0.6 {
        mutate_posts(&mut g, r);
        what = "post";
    } else if x < 0.7 {
        // flam3: 40 % new colour indices, 40 % new indices and palette (the
        // best of the tries), 20 % just a new palette
        let s = r.f01();
        if s < 0.4 {
            improve_colors(&mut g, 100, false, r);
        } else if s < 0.8 {
            improve_colors(&mut g, 25, true, r);
        } else {
            g.palette_index = r.below(genome::NPALETTES as u32) as i32;
            g.hue = if r.below(8) != 0 { 0.0 } else { r.f01() };
            g.palette = genome::palette(g.palette_index, g.hue);
        }
        what = "colour";
    } else if x < 0.8 {
        if g.xforms.len() > 1 {
            let k = r.below(g.xforms.len() as u32) as usize;
            g.xforms.remove(k);
            for x in g.xforms.iter_mut() {
                if !x.xaos.is_empty() && k < x.xaos.len() {
                    x.xaos.remove(k);
                }
            }
        }
        what = "delete";
    } else {
        // every coefficient nudged by a tenth of a random genome's
        let m = random_with(r.next_u32(), g.xforms.len());
        for (a, b) in g.xforms.iter_mut().zip(m.xforms.iter()) {
            for i in 0..3 {
                for j in 0..2 {
                    a.c[i][j] += 0.1 * b.c[i][j];
                }
            }
        }
        what = "coefs";
    }
    g.name = format!("{} ({what})", g.name.split(" (").next().unwrap_or(""));
    g
}

/// A flam3_random genome with exactly `n` xforms (mutation material).
fn random_with(seed: u32, n: usize) -> Genome {
    let mut s = seed;
    loop {
        let g = Genome::random(s);
        if g.xforms.len() >= n {
            let mut g = g;
            g.xforms.truncate(n.max(1));
            return g;
        }
        s = s.wrapping_add(0x9E37_79B9);
    }
}

fn mutate_posts(g: &mut Genome, r: &mut Rng) {
    let b = 1 + r.below(6);
    let same = r.below(4) != 0;
    let first_post = g.xforms[0].post;
    for i in 0..g.xforms.len() {
        if i > 0 && same {
            g.xforms[i].post = first_post;
            continue;
        }
        let x = &mut g.xforms[i];
        if b & 1 != 0 {
            let f = std::f32::consts::PI * r.signed();
            x.c = turn(&x.c, f);
            x.post = turn(&x.post, -f);
        }
        if b & 2 != 0 {
            let mut f = 0.2 + r.f01();
            let mut gg = 0.2 + r.f01();
            if r.bit() {
                f = 1.0 / f;
            }
            if r.bit() {
                gg = f;
            } else if r.bit() {
                gg = 1.0 / gg;
            }
            x.c[0][0] /= f;
            x.c[0][1] /= f;
            x.c[1][1] /= gg;
            x.c[1][0] /= gg;
            x.post[0][0] *= f;
            x.post[1][0] *= f;
            x.post[0][1] *= gg;
            x.post[1][1] *= gg;
        }
        if b & 4 != 0 {
            let f = r.signed();
            let gg = r.signed();
            x.c[2][0] -= f;
            x.c[2][1] -= gg;
            x.post[2][0] += f;
            x.post[2][1] += gg;
        }
    }
}

/// The 2x2 part turned by `f` the way flam3_mutate's post mutation does.
fn turn(a: &genome::Affine, f: f32) -> genome::Affine {
    let (s, c) = f.sin_cos();
    [[a[0][0] * c - a[0][1] * s, a[0][0] * s + a[0][1] * c], [a[1][0] * c - a[1][1] * s, a[1][0] * s + a[1][1] * c], a[2]]
}

/// flam3_cross: union 10 %, interpolation 10 %, alternating xforms 80 %; 40 %
/// of the time the palettes are spliced too.
pub fn cross(a: &Genome, b: &Genome, r: &mut Rng) -> Genome {
    let x = r.f01();
    let mut out;
    let what;
    if x < 0.1 {
        out = a.clone();
        for xf in &b.xforms {
            if out.xforms.len() < MAXX {
                out.xforms.push(xf.clone());
            }
        }
        for xf in out.xforms.iter_mut() {
            xf.xaos.clear();
        }
        what = "union";
    } else if x < 0.2 {
        let (pa, pb) = interp::align(a, b);
        out = interp::edge(&pa, &pb, 0.0, r.f01());
        for (xf, src) in out.xforms.iter_mut().zip(pa.xforms.iter()) {
            xf.animate = src.animate;
        }
        out.xforms.retain(|x| x.weight > 0.0);
        what = "blend";
    } else {
        let mut tries = 0;
        loop {
            let from_b = r.bit();
            let (base, other) = if from_b { (b, a) } else { (a, b) };
            out = base.clone();
            let (mut got_base, mut got_other) = (base.fin.is_some(), false);
            for i in 0..out.xforms.len() {
                if r.bit() && i < other.xforms.len() && other.xforms[i].weight > 0.0 {
                    out.xforms[i] = other.xforms[i].clone();
                    got_other = true;
                } else {
                    got_base = true;
                }
            }
            tries += 1;
            if (got_base && got_other) || out.xforms.len() <= 1 || tries > 20 {
                break;
            }
        }
        what = "cross";
    }
    for (i, xf) in out.xforms.iter_mut().enumerate() {
        if xf.animate {
            xf.color = (i & 1) as f32;
        }
    }
    if r.f01() < 0.4 {
        let mut from_b = r.bit();
        out.palette = (0..256)
            .map(|i| {
                if r.f01() < 0.01 {
                    from_b = !from_b;
                }
                if from_b { b.palette[i] } else { a.palette[i] }
            })
            .collect();
    }
    out.name = format!("{} x {} ({what})", short(&a.name), short(&b.name));
    out.curated = false;
    out.nc = a.nc || b.nc;
    out.nick = match (a.nick.is_empty(), b.nick.is_empty()) {
        (false, false) if a.nick != b.nick => format!("{} and {}", a.nick, b.nick),
        (false, _) => a.nick.clone(),
        (_, false) => b.nick.clone(),
        _ => String::new(),
    };
    out
}

fn short(n: &str) -> String {
    n.split(" (").next().unwrap_or("").chars().take(24).collect()
}

/// flam3-genome's framing: the box holding all but 1 % of the points on each
/// side; 30 % of the time centred on it, else at a golden section of it.
pub fn frame(g: &mut Genome, r: &mut Rng) {
    let mut s = Box::new(Sample::default());
    let mut probe = g.clone();
    probe.center = [0.0, 0.0];
    probe.scale = 0.25;
    probe.rotate = 0.0;
    probe.sample(0.0, 128, 128, &mut s);
    let mut pts = Vec::with_capacity(100_000);
    cpu::sample_points(&s, 100_000, r.next_u32(), &mut pts);
    if pts.len() < 1000 {
        return;
    }
    let trim = pts.len() / 100;
    let mut xs: Vec<f32> = pts.iter().map(|p| p[0]).collect();
    let mut ys: Vec<f32> = pts.iter().map(|p| p[1]).collect();
    xs.sort_by(f32::total_cmp);
    ys.sort_by(f32::total_cmp);
    let (x0, x1) = (xs[trim], xs[xs.len() - 1 - trim]);
    let (y0, y1) = (ys[trim], ys[ys.len() - 1 - trim]);
    if !(x1 - x0 > 1e-6) {
        return;
    }
    let golden = |r: &mut Rng| if r.bit() { 0.38196 } else { 0.61804 };
    let (m0, m1) = if r.f01() < 0.3 {
        (0.5, 0.5)
    } else if r.bit() {
        (golden(r) + r.signed() / 5.0, golden(r))
    } else if r.bit() {
        (golden(r), golden(r) + r.signed() / 5.0)
    } else {
        (golden(r) + r.signed() / 5.0, golden(r) + r.signed() / 5.0)
    };
    g.center = [m0 * x0 + (1.0 - m0) * x1, m1 * y0 + (1.0 - m1) * y1];
    g.scale = 1.0 / (x1 - x0);
}

/// flam3-genome's test: a 128 x 128 render of [-1, 1]^2 at quality 1 with
/// highlight power 1; it passes with a mean channel of 20 or more, at least
/// 1 % black pixels and at most 5 % white ones.
pub fn fitness(g: &Genome, seed: u32) -> (bool, f32, f32, f32) {
    let mut probe = g.clone();
    probe.center = [0.0, 0.0];
    probe.scale = 0.5;
    probe.rotate = 0.0;
    let (avg, fb, fw) = test_render(&probe, 128, 128, 0.0, 1.0, 1.0, seed);
    (avg >= 20.0 && fb >= 0.01 && fw <= 0.05, avg, fb, fw)
}

/// The darkest a new sheep's framed view may get (mean channel, 0-255): below
/// this a frame looks black.
pub const MIN_VIEW: f32 = 8.0;

/// The whole test a new sheep goes through: flam3-genome's test render,
/// flam3-genome's framing, then the framed view at four turns of its loop
/// (flam3's test looks at [-1, 1]^2 at one angle only, so about one sheep in
/// ten that passes it, and one mutation in five, goes dark on screen).
/// Returns the framed genome, whether it passed, and a score for choosing
/// the best of failed tries.
pub fn trial(mut g: Genome, seed: u32) -> (Genome, bool, f32) {
    if g.curated {
        // a real sheep: its own framing; only make sure it shows
        let view = view_brightness(&g, seed);
        return (g, view >= MIN_VIEW, view);
    }
    let (ok, avg, ..) = fitness(&g, seed);
    let mut r = Rng::new(seed ^ 0x0005_1ee9);
    frame(&mut g, &mut r);
    let view = view_brightness(&g, seed);
    (g, ok && view >= MIN_VIEW, view.min(avg))
}

/// Up to ten tries of `make` through `trial` (flam3-genome tries ten too);
/// the first that passes, else the best one. Counts into `tested` /
/// `rejected`.
pub fn first_good(mut make: impl FnMut() -> (Genome, u32), tested: &mut u32, rejected: &mut u32) -> Genome {
    let mut best: Option<(Genome, f32)> = None;
    for _ in 0..10 {
        let (g, seed) = make();
        let (mut g, ok, score) = trial(g, seed);
        *tested += 1;
        if ok && !g.curated {
            // flam3-genome's colour pass (working, and on every sheep: random
            // and crossed genomes otherwise use only palette ends 0 and 1)
            improve_colors(&mut g, 100, false, &mut Rng::new(seed ^ 0xc010));
            return g;
        }
        if ok {
            return g;
        }
        *rejected += 1;
        if best.as_ref().is_none_or(|b| score > b.1) {
            best = Some((g, score));
        }
    }
    best.unwrap().0
}

/// The mean channel (0-255) of the flame as it will be framed on screen, at
/// the darkest of four turns of its loop: a small, quick render with the
/// genome's own tone settings.
pub fn view_brightness(g: &Genome, seed: u32) -> f32 {
    (0..4u32)
        .map(|q| test_render(g, 160, 90, q as f32 * std::f32::consts::FRAC_PI_2, 3.0, g.highlight, seed.wrapping_add(q)).0)
        .fold(f32::MAX, f32::min)
}

/// A `w` x `h` render on the CPU at `quality` samples per pixel through the
/// genome's camera, turned by `angle`, tone-mapped as flam3 does (no density
/// estimation): (mean channel, black share, white share).
fn test_render(g: &Genome, w: usize, h: usize, angle: f32, quality: f32, highpow: f32, seed: u32) -> (f32, f32, f32) {
    let img = render_small(g, w, h, angle, quality, highpow, seed);
    let (mut tot, mut black, mut white) = (0f64, 0, 0);
    for rgb in &img {
        tot += rgb.iter().map(|&v| v as f64).sum::<f64>();
        if *rgb == [0, 0, 0] {
            black += 1;
        }
        if *rgb == [255, 255, 255] {
            white += 1;
        }
    }
    let n = (w * h) as f32;
    ((tot / (3.0 * n as f64)) as f32, black as f32 / n, white as f32 / n)
}

/// The pixels of that render.
fn render_small(g: &Genome, w: usize, h: usize, angle: f32, quality: f32, highpow: f32, seed: u32) -> Vec<[u8; 3]> {
    let mut s = Box::new(Sample::default());
    g.sample(angle, w as u32, h as u32, &mut s);
    let samples = ((w * h) as f32 * quality) as usize;
    let mut pts = Vec::with_capacity(samples);
    cpu::sample_points(&s, samples, seed, &mut pts);
    let mut acc = vec![[0f32; 4]; w * h];
    for p in &pts {
        let px = s.cam[0] * p[0] + s.cam[1] * p[1] + s.cam[2];
        let py = s.cam[4] * p[0] + s.cam[5] * p[1] + s.cam[6];
        if !(px >= 0.0 && py >= 0.0 && px < w as f32 && py < h as f32) {
            continue;
        }
        let i = py as usize * w + px as usize;
        let c = g.palette[((p[2] * 256.0) as i32).clamp(0, 255) as usize];
        acc[i][0] += c[0];
        acc[i][1] += c[1];
        acc[i][2] += c[2];
        acc[i][3] += 1.0;
    }
    // t = brightness * 268/256 * ln(1 + hits * ppu^2 / samples)
    let ppu = g.scale * w as f32;
    let k = ppu * ppu / samples.max(1) as f32;
    let gam = 1.0 / g.gamma;
    acc.iter()
        .map(|a| {
            let mut rgb = [0u8; 3];
            if a[3] > 0.0 {
                let t = g.brightness * 268.0 / 256.0 * (1.0 + a[3] * k).ln();
                let alpha = calc_alpha(t, gam, g.gamma_threshold);
                let ls = g.vibrancy * 256.0 * alpha / t;
                // cbuf / PREFILTER_WHITE = average palette colour * t
                let cb = [a[0] / a[3] * t, a[1] / a[3] * t, a[2] / a[3] * t];
                let c = newrgb(cb, ls, highpow);
                for i in 0..3 {
                    let v = c[i] + (1.0 - g.vibrancy) * 256.0 * cb[i].max(0.0).powf(gam);
                    rgb[i] = v.clamp(0.0, 255.0) as u8;
                }
            }
            rgb
        })
        .collect()
}

/// flam3's try_colors: the share of the res^3 colour cells a quick render
/// (about 10,000 pixels, quality 1) lights. flam3 3.x divides in integers
/// here and always gets 0, which turned flam3_improve_colors into a no-op;
/// this is the version that does what it meant.
fn try_colors(g: &Genome, res: usize, seed: u32) -> f32 {
    let img = render_small(g, 133, 75, 0.0, 1.0, 1.0, seed);
    let mut cells = vec![false; res * res * res];
    for p in &img {
        let i = p[0] as usize * res / 256 + (p[1] as usize * res / 256) * res + (p[2] as usize * res / 256) * res * res;
        cells[i] = true;
    }
    cells.iter().filter(|&&c| c).count() as f32 / cells.len() as f32
}

/// flam3's change_colors: every xform a random colour index, one of them 0
/// and another 1 half the time each; with `palette`, a random palette too.
fn change_colors(g: &mut Genome, palette: bool, r: &mut Rng) {
    if palette {
        g.hue = 0.0;
        g.palette_index = r.below(genome::NPALETTES as u32) as i32;
        g.palette = genome::palette(g.palette_index, 0.0);
    }
    for x in g.xforms.iter_mut() {
        x.color = r.f01();
    }
    let n = g.xforms.len() as u32;
    if n > 0 {
        let x0 = r.below(n) as usize;
        let x1 = (x0 + 1 + r.below(n.max(2) - 1) as usize) % n as usize;
        if r.bit() {
            g.xforms[x0].color = 0.0;
        }
        if r.bit() && x1 != x0 {
            g.xforms[x1].color = 1.0;
        }
    }
}

/// flam3_improve_colors: the best of `tries` colourings by how many distinct
/// colours the flame shows.
pub fn improve_colors(g: &mut Genome, tries: u32, palette: bool, r: &mut Rng) {
    let seed = r.next_u32();
    let mut best = try_colors(g, 10, seed);
    for _ in 0..tries {
        let mut c = g.clone();
        change_colors(&mut c, palette, r);
        let b = try_colors(&c, 10, seed);
        if b > best {
            best = b;
            *g = c;
        }
    }
}

/// flam3_calc_alpha
pub fn calc_alpha(t: f32, g: f32, lin: f32) -> f32 {
    if t <= 0.0 {
        0.0
    } else if t < lin {
        let f = t / lin;
        (1.0 - f) * t * (lin.powf(g) / lin) + f * t.powf(g)
    } else {
        t.powf(g)
    }
}

/// flam3_calc_newrgb: `cb` is the colour over PREFILTER_WHITE (0..1 per hit
/// scale), `ls` the scale; channels come out 0..255+.
pub fn newrgb(cb: [f32; 3], ls: f32, highpow: f32) -> [f32; 3] {
    if ls == 0.0 || cb == [0.0; 3] {
        return [0.0; 3];
    }
    let (mut maxa, mut maxc) = (-1.0f32, 0.0f32);
    for &c in &cb {
        let a = ls * c;
        if a > maxa {
            maxa = a;
            maxc = c;
        }
    }
    let newls = 255.0 / maxc;
    if maxa > 255.0 && highpow >= 0.0 {
        let ratio = (newls / ls).powf(highpow);
        let mut hsv = genome::rgb2hsv([newls * cb[0] / 255.0, newls * cb[1] / 255.0, newls * cb[2] / 255.0]);
        hsv[1] *= ratio;
        let c = genome::hsv2rgb(hsv);
        [c[0] * 255.0, c[1] * 255.0, c[2] * 255.0]
    } else {
        let adj = if maxa <= 255.0 { 1.0 } else { (-highpow).min(1.0) };
        let s = (1.0 - adj) * newls + adj * ls;
        [s * cb[0], s * cb[1], s * cb[2]]
    }
}

/// A sheep in the flock.
#[derive(Clone)]
pub struct Sheep {
    pub genome: Genome,
    pub rating: i32,
    pub id: u32,
}

/// The local flock: Electric Sheep's server, in miniature. New sheep are
/// 70 % descendants of the flock (a third mutations, the rest crosses), the
/// rest random; parents are picked by rating; 20 % get a symmetry
/// singularity, as the server did. Every candidate is framed and tested.
pub struct Flock {
    pub sheep: Vec<Sheep>,
    /// real Electric Sheep genomes (the server's genebank): cloned and crossed
    pub genebank: Vec<Genome>,
    /// each one's weight: its Electric Sheep rating (likes) plus one
    pub bank_weight: Vec<f32>,
    r: Rng,
    next_id: u32,
    pub tested: u32,
    pub rejected: u32,
}

pub const FLOCK_MAX: usize = 32;

impl Flock {
    pub fn new(seed: u32) -> Flock {
        Flock {
            sheep: Vec::new(),
            genebank: Vec::new(),
            bank_weight: Vec::new(),
            r: Rng::new(seed),
            next_id: 1,
            tested: 0,
            rejected: 0,
        }
    }

    /// A liked sheep, picked in proportion to its rating (the server's rule).
    fn parent(&mut self) -> Genome {
        let rated: Vec<usize> = (0..self.sheep.len()).filter(|&i| self.sheep[i].rating > 0).collect();
        if rated.is_empty() {
            return self.banked().unwrap_or_else(|| Genome::random(self.r.next_u32()));
        }
        let total: i32 = rated.iter().map(|&i| self.sheep[i].rating).sum();
        let mut x = self.r.below(total as u32) as i32;
        for &i in &rated {
            x -= self.sheep[i].rating;
            if x < 0 {
                return self.sheep[i].genome.clone();
            }
        }
        self.sheep[rated[0]].genome.clone()
    }

    pub fn next_id(&mut self) -> u32 {
        self.next_id += 1;
        self.next_id - 1
    }

    pub fn seed(&mut self) -> u32 {
        self.r.next_u32()
    }

    /// A genebank genome, if there are any: the more Electric Sheep viewers
    /// liked it, the likelier (square root of the rating, so the favourites
    /// lead without crowding out the rest).
    fn banked(&mut self) -> Option<Genome> {
        if self.genebank.is_empty() {
            return None;
        }
        if self.bank_weight.len() != self.genebank.len() {
            return Some(self.genebank[self.r.below(self.genebank.len() as u32) as usize].clone());
        }
        let total: f32 = self.bank_weight.iter().sum();
        let mut x = self.r.f01() * total;
        for (g, w) in self.genebank.iter().zip(&self.bank_weight) {
            x -= w;
            if x <= 0.0 {
                return Some(g.clone());
            }
        }
        self.genebank.last().cloned()
    }

    /// One candidate genome (untested). The Electric Sheep server bred 70 %
    /// of new sheep from a flock thousands strong, rated by thousands of
    /// people; a one-person flock of 32 bred that way soon shows the same
    /// few ancestors. So only liked sheep breed (half the new sheep once
    /// there are any), the genebank of real sheep gives clones and crosses
    /// (as the server's did), and the rest are random.
    pub fn conceive(&mut self) -> Genome {
        let liked = self.sheep.iter().any(|s| s.rating > 0);
        let bank = !self.genebank.is_empty();
        let x = self.r.f01();
        let mut g = if liked && x < 0.5 {
            if self.r.f01() < 0.33 {
                let p = self.parent();
                mutate(&p, &mut self.r)
            } else {
                // a liked sheep with another liked one, or with a real sheep
                let a = self.parent();
                let b = if bank && self.r.bit() { self.banked().unwrap() } else { self.parent() };
                cross(&a, &b, &mut self.r)
            }
        } else if bank && x < if liked { 0.8 } else { 0.6 } {
            let a = self.banked().unwrap();
            if self.r.bit() {
                a
            } else {
                let b = self.banked().unwrap();
                cross(&a, &b, &mut self.r)
            }
        } else {
            Genome::random(self.r.next_u32())
        };
        if self.r.f01() < 0.2 {
            // a symmetry singularity (as many as fit)
            let s = 11 + self.r.below(45) as i32;
            let s = if self.r.bit() { s } else { -s };
            let mut r2 = self.r;
            g.add_symmetry(&mut r2, s);
            self.r.next_u32();
        }
        g
    }

    /// The first sheep of a run: a random real sheep from the genebank
    /// (weighted by its rating), or a new random genome without one, so
    /// every run opens differently (breeding from the flock would favour
    /// the same liked family each time).
    pub fn opener(&mut self) -> Sheep {
        let (mut tested, mut rejected) = (self.tested, self.rejected);
        let g = first_good(
            || {
                let g = self.banked().unwrap_or_else(|| Genome::random(self.r.next_u32()));
                (g, self.r.next_u32())
            },
            &mut tested,
            &mut rejected,
        );
        (self.tested, self.rejected) = (tested, rejected);
        let id = self.next_id();
        Sheep { genome: g, rating: 0, id }
    }

    /// A new sheep that passed the tests (or the best of ten tries).
    pub fn breed(&mut self) -> Sheep {
        let (mut tested, mut rejected) = (self.tested, self.rejected);
        let g = first_good(
            || {
                let g = self.conceive();
                (g, self.r.next_u32())
            },
            &mut tested,
            &mut rejected,
        );
        (self.tested, self.rejected) = (tested, rejected);
        let id = self.next_id();
        Sheep { genome: g, rating: 0, id }
    }

    pub fn add(&mut self, s: Sheep) {
        self.sheep.push(s);
        if self.sheep.len() > FLOCK_MAX {
            // the lowest rated (oldest first among equals) dies
            let worst = (0..self.sheep.len()).min_by_key(|&i| (self.sheep[i].rating, i)).unwrap();
            self.sheep.remove(worst);
        }
    }

    /// The flock as (genome, rating) for the flock file.
    pub fn snapshot(&self) -> Vec<(Genome, i32)> {
        self.sheep.iter().map(|s| (s.genome.clone(), s.rating)).collect()
    }

    pub fn vote(&mut self, id: u32, v: i32) {
        if let Some(s) = self.sheep.iter_mut().find(|s| s.id == id) {
            s.rating += v;
        }
        if v < 0 {
            self.sheep.retain(|s| s.id != id || s.rating >= 0);
        }
    }
}

