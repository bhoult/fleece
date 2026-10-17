//! A taste in flames: a few measures of a small render (how colourful,
//! how much of the frame it fills, how much detail, how symmetric) and a
//! linear score over them. The prior was fitted to what Electric Sheep's
//! viewers liked (the genebank's ratings, `fleece --tastefit`); the votes
//! here move it towards what this viewer likes. Breeding keeps the best
//! scoring of a few candidates.

use crate::genome::Genome;

/// Measures per flame.
pub const NF: usize = 9;
pub const NAMES: [&str; NF] = ["colourful", "coverage", "coverage^2", "entropy", "edges", "symmetry", "colours", "luma", "streaks"];
/// The streaks measure below which 99 % of real sheep are (--tastefit).
const STREAKS_REAL_P99: f32 = 0.385;
/// The render the measures come from (Electric Sheep's frame shape).
const W: usize = 160;
const H: usize = 118;

/// The measures of `g` as it will be framed (its loop's first frame).
pub fn features(g: &Genome, seed: u32) -> [f32; NF] {
    let img = crate::breed::render_small(g, W, H, 0.0, 6.0, g.highlight, seed);
    let px: Vec<[f32; 3]> = img.iter().map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]).collect();
    let luma: Vec<f32> = px.iter().map(|c| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]).collect();
    let n = px.len() as f32;
    // Hasler and Suesstrunk's colourfulness: the spread and strength of the opponent colours
    let (mut rg, mut yb) = (Vec::with_capacity(px.len()), Vec::with_capacity(px.len()));
    for c in &px {
        rg.push(c[0] - c[1]);
        yb.push(0.5 * (c[0] + c[1]) - c[2]);
    }
    let ms = |v: &[f32]| {
        let m = v.iter().sum::<f32>() / n;
        (m, (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n).sqrt())
    };
    let ((mrg, srg), (myb, syb)) = (ms(&rg), ms(&yb));
    let colourful = (srg * srg + syb * syb).sqrt() + 0.3 * (mrg * mrg + myb * myb).sqrt();
    let coverage = luma.iter().filter(|&&l| l > 0.06).count() as f32 / n;
    // the luma's entropy (16 levels), in bits over 4
    let mut hist = [0f32; 16];
    for &l in &luma {
        hist[((l * 16.0) as usize).min(15)] += 1.0;
    }
    let entropy = -hist.iter().filter(|&&c| c > 0.0).map(|&c| (c / n) * (c / n).log2()).sum::<f32>() / 4.0;
    // detail: the mean gradient of the luma; streaks: how much the
    // gradients of the whole picture point one way (1 for stripes, near 0
    // for most flames), from the structure tensor
    let mut edges = 0.0;
    for y in 0..H - 1 {
        for x in 0..W - 1 {
            let i = y * W + x;
            edges += (luma[i + 1] - luma[i]).abs() + (luma[i + W] - luma[i]).abs();
        }
    }
    // the colour structure tensor (stripes may differ in hue, not
    // brightness) of the picture averaged over 2 x 2 blocks (the render's
    // noise points every way), from central differences
    let (w2, h2) = (W / 2, H / 2);
    let half: Vec<[f32; 3]> = (0..w2 * h2)
        .map(|j| {
            let (x, y) = (2 * (j % w2), 2 * (j / w2));
            let mut c = [0.0; 3];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let p = px[(y + dy) * W + x + dx];
                for k in 0..3 {
                    c[k] += 0.25 * p[k];
                }
            }
            c
        })
        .collect();
    let (mut jxx, mut jyy, mut jxy) = (0.0f32, 0.0f32, 0.0f32);
    for y in 1..h2 - 1 {
        for x in 1..w2 - 1 {
            let i = y * w2 + x;
            for c in 0..3 {
                let (gx, gy) = (half[i + 1][c] - half[i - 1][c], half[i + w2][c] - half[i - w2][c]);
                jxx += gx * gx;
                jyy += gy * gy;
                jxy += gx * gy;
            }
        }
    }
    edges /= n;
    let streaks = ((jxx - jyy).powi(2) + 4.0 * jxy * jxy).sqrt() / (jxx + jyy).max(1e-9);
    // symmetry: the best correlation with its mirror images and half turn
    let corr = |f: &dyn Fn(usize, usize) -> usize| {
        let m = luma.iter().sum::<f32>() / n;
        let (mut num, mut den) = (0.0, 0.0);
        for y in 0..H {
            for x in 0..W {
                let a = luma[y * W + x] - m;
                let b = luma[f(x, y)] - m;
                num += a * b;
                den += a * a;
            }
        }
        if den > 0.0 { num / den } else { 0.0 }
    };
    let symmetry = corr(&|x, y| y * W + (W - 1 - x))
        .max(corr(&|x, y| (H - 1 - y) * W + x))
        .max(corr(&|x, y| (H - 1 - y) * W + (W - 1 - x)));
    // how many of 8^3 colour cells the lit pixels use
    let mut cells = [0u16; 512];
    for (c, &l) in px.iter().zip(&luma) {
        if l > 0.06 {
            let q = |v: f32| ((v * 8.0) as usize).min(7);
            cells[q(c[0]) * 64 + q(c[1]) * 8 + q(c[2])] += 1;
        }
    }
    let colours = cells.iter().filter(|&&c| c >= 3).count() as f32 / 64.0;
    let luma_mean = luma.iter().sum::<f32>() / n;
    [colourful, coverage, (coverage - 0.5) * (coverage - 0.5), entropy, edges, symmetry, colours, luma_mean, streaks]
}

/// A score over standardised measures: w . (f - mean) / std + bias.
#[derive(Clone, Debug)]
pub struct Taste {
    pub mean: [f32; NF],
    pub std: [f32; NF],
    pub w: [f32; NF],
    pub bias: f32,
}

impl Taste {
    /// `fleece --tastefit` on 3,000 genebank sheep (1) and 3,000 new random
    /// sheep that passed the tests (0); cross-validated correlation 0.58.
    /// Real sheep are more colourful, more detailed and use more distinct
    /// colours than random ones; a score near 1 looks like one of them.
    pub fn prior() -> Taste {
        Taste {
            mean: [0.2133, 0.4777, 0.0297, 0.6061, 0.2335, 0.3272, 1.7614, 0.1932, 0.0811],
            std: [0.1105, 0.1709, 0.0357, 0.1487, 0.1067, 0.2082, 1.1975, 0.0926, 0.0954],
            w: [0.1282, -0.1038, 0.0484, -0.0169, 0.1651, 0.0636, 0.1065, -0.0993, -0.0218],
            bias: 0.4989,
        }
    }

    fn z(&self, f: &[f32; NF]) -> [f32; NF] {
        std::array::from_fn(|i| ((f[i] - self.mean[i]) / self.std[i].max(1e-4)).clamp(-4.0, 4.0))
    }

    pub fn score(&self, f: &[f32; NF]) -> f32 {
        let z = self.z(f);
        // streaks past what 99 % of real sheep show (0.385; random ones reach
        // 0.8): a direct-colour flame painting stripes across the frame is
        // colourful, which the score likes, and too rare to have been learnt
        let streaky = 2.0 * (f[8] - STREAKS_REAL_P99).max(0.0);
        self.bias + z.iter().zip(&self.w).map(|(a, b)| a * b).sum::<f32>() - streaky
    }

    /// The linear score alone (what votes teach).
    fn linear(&self, f: &[f32; NF]) -> f32 {
        let z = self.z(f);
        self.bias + z.iter().zip(&self.w).map(|(a, b)| a * b).sum::<f32>()
    }

    /// One vote: a small least-squares step taking a liked flame's score
    /// towards 1 (a real sheep), a disliked one's towards 0.
    pub fn learn(&mut self, f: &[f32; NF], v: i32) {
        let target = if v > 0 { 1.0 } else { 0.0 };
        let err = target - self.linear(f);
        let z = self.z(f);
        for i in 0..NF {
            self.w[i] += 0.02 * err * z[i];
        }
    }

    /// The taste saved by `save`, else the prior.
    pub fn load(path: &str) -> Taste {
        let mut t = Taste::prior();
        let Ok(s) = std::fs::read_to_string(path) else { return t };
        for line in s.lines() {
            let v: Vec<&str> = line.split_whitespace().collect();
            if v.len() == 4 {
                if let (Some(i), Ok(w)) = (NAMES.iter().position(|n| *n == v[0]), v[1].parse::<f32>()) {
                    t.w[i] = w;
                }
            }
        }
        t
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut s = String::from("# fleece's taste: measure, weight, mean, std (weights move with your votes)\n");
        for i in 0..NF {
            s.push_str(&format!("{} {} {} {}\n", NAMES[i], self.w[i], self.mean[i], self.std[i]));
        }
        if let Some(dir) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, s)
    }
}
