//! Music made up as it plays: a composer that never repeats itself and a
//! small synthesiser to play it, for the show to dance to (and with).
//!
//! The composer is seeded: a genre and a seed always give the same music. It
//! writes pieces of a few minutes, each with a key, a mode, a tempo, a chord
//! progression, drum patterns, a bassline, an arpeggio, a melody and sounds of
//! its own, laid out in sections (intro, build, main, breakdown, outro) with
//! layers entering and leaving. Within a piece patterns mutate at phrase ends,
//! the melody comes back varied (call and response, sequences, new endings),
//! one optional layer drops out now and then, and filters drift. Then the next
//! piece, usually in a neighbouring key. A mix (polka+techno, say) is one
//! genre's music drummed by another, with instruments from both.
//!
//! The synthesiser: polyBLEP oscillators, trapezoidal state-variable filters,
//! two-operator FM (bells, electric piano), drums made of sines and noise, a
//! Freeverb, a ping-pong delay, a gated reverb (synthwave's snare) or a rumble
//! (techno's kick), sidechain ducking, and a master with a DC blocker, a slow
//! leveller, a soft knee and a lookahead limiter. 48 kHz stereo, played through
//! pacat (`Player`) or written to a WAV (`render_wav`).

use std::collections::VecDeque;
use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::samples::{self, Bank, SAMPLER, Sampler};

/// Genre names, shown on screen in capitals (the app's font has only A-Z, 0-9 and " -.:/!+()[],%").
pub const GENRES: &[&str] = &["AMBIENT", "TECHNO", "SYNTHWAVE", "LO-FI", "POLKA", "TAVERN", "SKA", "MARIACHI"];
const AMBIENT: usize = 0;
const TECHNO: usize = 1;
const SYNTHWAVE: usize = 2;
const LOFI: usize = 3;
const POLKA: usize = 4;
const TAVERN: usize = 5;
const SKA: usize = 6;
const MARIACHI: usize = 7;
/// How many genres.
const NG: usize = GENRES.len();
/// The music's ids: a genre alone (below `NG`), or a mix of a primary genre and a
/// secondary one (`mix_id`): the primary writes the music (its form, meter,
/// harmony and tunes), the secondary drums it (its beat, its effects, its
/// tempo, near enough), and each part has an instrument of each.
pub const IDS: usize = NG * (NG + 1);
/// Each genre's tempo, about (a mix's lies 60 % of the way to its drummer's).
const MID_BPM: [f64; NG] = [94.0, 132.0, 112.0, 90.0, 120.0, 115.0, 135.0, 118.0];

/// The id of `primary` alone, or mixed with `secondary` (another genre).
pub fn mix_id(primary: usize, secondary: Option<usize>) -> usize {
    match secondary.filter(|&g| g != primary && g < NG) {
        Some(g) => primary % NG + NG * (g + 1),
        None => primary % NG,
    }
}

/// An id's primary genre and secondary one, if a mix.
pub fn parts(id: usize) -> (usize, Option<usize>) {
    let (p, s) = (id % NG, (id % IDS) / NG);
    (p, s.checked_sub(1).filter(|&g| g != p))
}

/// An id's name: the genre's, or the mix's two ("POLKA+TECHNO").
pub fn name(id: usize) -> String {
    match parts(id) {
        (p, Some(g)) => format!("{}+{}", GENRES[p], GENRES[g]),
        (p, None) => GENRES[p].to_string(),
    }
}

/// `id` in its one spelling (a mix of a genre with itself is the genre alone).
fn canon(id: usize) -> usize {
    let (p, s) = parts(id);
    mix_id(p, s)
}

/// The music called `name` (any case, spaces and dashes ignored; a mix's
/// primary genre first), or its id.
pub fn find(name_: &str) -> Option<usize> {
    let squash = |s: &str| s.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_uppercase();
    let want = squash(name_);
    (0..IDS)
        .filter(|&id| canon(id) == id)
        .find(|&id| squash(&name(id)) == want)
        .or_else(|| name_.trim().parse().ok().filter(|&n| n < IDS))
}

const RATE: u32 = 48000;
const SR: f32 = RATE as f32;
/// Samples per control block: filters, envelopes' coefficients and the
/// mixer's smoothing update once a block (and events split blocks).
const BLOCK: usize = 32;
/// The player's writes to pacat, in frames (21 ms).
const CHUNK: usize = 1024;
/// A genre switch's crossfade, in samples.
const FADE: u32 = 58000;

// ---------------------------------------------------------------- randomness

/// splitmix64: the composer's dice.
#[derive(Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x6A09_E667_F3BC_C909)
    }
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// [0, 1)
    fn f(&mut self) -> f32 {
        (self.u64() >> 40) as f32 * (1.0 / 16_777_216.0)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
    fn signed(&mut self) -> f32 {
        self.f() * 2.0 - 1.0
    }
    /// [0, n)
    fn below(&mut self, n: usize) -> usize {
        (((self.u64() >> 32) * n as u64) >> 32) as usize
    }
    fn chance(&mut self, p: f32) -> bool {
        self.f() < p
    }
    fn bit_chance(&mut self) -> bool {
        self.u64() >> 63 == 1
    }
    fn pick<T: Copy>(&mut self, t: &[T]) -> T {
        t[self.below(t.len())]
    }
    /// An index, by weight.
    fn weighted(&mut self, w: &[f32]) -> usize {
        let mut x = self.f() * w.iter().sum::<f32>();
        for (i, &wi) in w.iter().enumerate() {
            if x < wi {
                return i;
            }
            x -= wi;
        }
        w.len() - 1
    }
}

/// A slow random walk in [-1, 1], pulled gently back to 0.
fn walk(v: f32, r: &mut Rng, step: f32) -> f32 {
    (v + r.signed() * step - v * 0.04).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------- DSP

/// White noise in [-1, 1) (xorshift32: cheap enough per sample).
#[derive(Clone, Copy)]
struct Noise(u32);

impl Default for Noise {
    fn default() -> Noise {
        Noise(0x2545_F491)
    }
}

impl Noise {
    #[inline]
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x as i32 as f32 * (1.0 / 2_147_483_648.0)
    }
    fn seed(&mut self) -> u32 {
        self.next();
        self.0 | 1
    }
}

/// sin(2π t), to about 1e-3 (a parabola, refined).
#[inline]
fn sin01(t: f32) -> f32 {
    let x = t - (t + 0.5).floor();
    let y = 8.0 * x - 16.0 * x * x.abs();
    0.225 * (y * y.abs() - y) + y
}

/// tanh, near enough (a Padé approximant, clamped where it meets ±1).
#[inline]
fn soft(x: f32) -> f32 {
    let x = x.clamp(-3.0, 3.0);
    x * (27.0 + x * x) / (27.0 + 9.0 * x * x)
}

fn mtof(n: f32) -> f32 {
    440.0 * ((n - 69.0) / 12.0).exp2()
}

/// The per-sample factor that decays to -60 dB in `t` seconds.
fn t60(t: f32) -> f32 {
    (-6.9 / (t.max(1e-4) * SR)).exp()
}

/// The polyBLEP residual at phase `t` for phase step `dt`.
#[inline]
fn blep(t: f32, dt: f32) -> f32 {
    if t < dt {
        let x = t / dt;
        x + x - x * x - 1.0
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt;
        x * x + x + x + 1.0
    } else {
        0.0
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Wave {
    Saw,
    Square,
    Tri,
    Sine,
}

/// One sample of a band-limited waveform at phase `t` in [0, 1).
#[inline]
fn osc(w: Wave, t: f32, dt: f32, pw: f32) -> f32 {
    match w {
        Wave::Saw => 2.0 * t - 1.0 - blep(t, dt),
        Wave::Square => {
            let mut t2 = t - pw;
            if t2 < 0.0 {
                t2 += 1.0;
            }
            (if t < pw { 1.0 } else { -1.0 }) + blep(t, dt) - blep(t2, dt)
        }
        // a triangle's harmonics fall 12 dB an octave: little left to alias
        Wave::Tri => 1.0 - 4.0 * (t - 0.5).abs(),
        Wave::Sine => sin01(t),
    }
}

#[inline]
fn wrap(t: f32) -> f32 {
    if t >= 1.0 { t - 1.0 } else { t }
}

/// The trapezoidal (zero-delay feedback) state-variable filter of Zavalishin
/// and Simper: stable under fast modulation, resonance up to self-oscillation.
#[derive(Clone, Copy, Default)]
struct Svf {
    ic1: f32,
    ic2: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    k: f32,
}

impl Svf {
    /// Cutoff in Hz; resonance 0 (none) .. 1 (ringing).
    fn set(&mut self, fc: f32, res: f32) {
        let g = (std::f32::consts::PI * fc.clamp(20.0, 0.42 * SR) / SR).tan();
        let k = (2.0 - 2.0 * res.clamp(0.0, 1.0)).max(0.05);
        self.k = k;
        self.a1 = 1.0 / (1.0 + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
    }
    /// (low, band) outputs.
    #[inline]
    fn run(&mut self, v0: f32) -> (f32, f32) {
        let v3 = v0 - self.ic2;
        let v1 = self.a1 * self.ic1 + self.a2 * v3;
        let v2 = self.ic2 + self.a2 * self.ic1 + self.a3 * v3;
        self.ic1 = 2.0 * v1 - self.ic1;
        self.ic2 = 2.0 * v2 - self.ic2;
        (v2, v1)
    }
    #[inline]
    fn lp(&mut self, x: f32) -> f32 {
        self.run(x).0
    }
    /// Band-pass with unity gain at the centre.
    #[inline]
    fn bp(&mut self, x: f32) -> f32 {
        self.k * self.run(x).1
    }
    #[inline]
    fn hp(&mut self, x: f32) -> f32 {
        let (l, b) = self.run(x);
        x - self.k * b - l
    }
}

const IDLE: u8 = 0;
const ATTACK: u8 = 1;
const DECAY: u8 = 2;
const RELEASE: u8 = 3;

/// ADSR: a linear attack (2 ms at least) from wherever the level is, an
/// exponential decay to the sustain level and an exponential release (3 ms at
/// least); times are to about -60 dB.
#[derive(Clone, Copy, Default)]
struct Adsr {
    stage: u8,
    level: f32,
    att: f32,
    dec: f32,
    sus: f32,
    rel: f32,
}

impl Adsr {
    fn set(&mut self, a: f32, d: f32, s: f32, r: f32) {
        self.att = 1.0 / (a.max(0.002) * SR);
        self.dec = t60(d.max(0.003));
        self.sus = s;
        self.rel = t60(r.max(0.003));
    }
    fn on(&mut self) {
        self.stage = ATTACK;
    }
    fn off(&mut self) {
        if self.stage != IDLE {
            self.stage = RELEASE;
        }
    }
    #[inline]
    fn next(&mut self) -> f32 {
        match self.stage {
            ATTACK => {
                self.level += self.att;
                if self.level >= 1.0 {
                    self.level = 1.0;
                    self.stage = DECAY;
                }
            }
            DECAY => self.level = self.sus + (self.level - self.sus) * self.dec,
            RELEASE => {
                self.level *= self.rel;
                // -60 dB: inaudible under the music, and a voice the less
                if self.level < 1e-3 {
                    self.level = 0.0;
                    self.stage = IDLE;
                }
            }
            _ => {}
        }
        self.level
    }
}

// ---------------------------------------------------------------- instruments

/// A tonal sound: subtractive (up to three detuned oscillators, a second
/// waveform, a sub, noise, a filter with its own envelope) or two-operator FM.
#[derive(Clone, Copy)]
struct Patch {
    wave: Wave,
    pw: f32,
    /// oscillators, spread over `detune` cents and `width` of the stereo field
    unison: u8,
    detune: f32,
    width: f32,
    wave2: Wave,
    mix2: f32,
    /// the second waveform's pitch, semitones from the note
    semi2: f32,
    sub: f32,
    noise: f32,
    /// FM (when > 0): a sine carrier, phase modulated by sines at `ratio` and
    /// `ratio2` times its frequency, with indices (radians) that decay over
    /// `fm_dec` and `fm2_dec` seconds
    fm: f32,
    ratio: f32,
    fm_dec: f32,
    fm2: f32,
    ratio2: f32,
    fm2_dec: f32,
    filter: bool,
    cutoff: f32,
    res: f32,
    /// the filter envelope's reach, in octaves, and its decay
    env_amt: f32,
    fdec: f32,
    /// how far the cutoff follows the pitch (1: all the way)
    keytrack: f32,
    /// two filters in series (24 dB an octave), the acid bass's
    poles4: bool,
    a: f32,
    d: f32,
    s: f32,
    r: f32,
    /// vibrato (semitones, fading in) and tremolo depth, at `vib_rate` Hz
    vib: f32,
    vib_rate: f32,
    trem: f32,
    /// portamento time for slid notes
    glide: f32,
    drive: f32,
    gain: f32,
    /// Karplus-Strong (when > 0): a plucked string ringing for `ks` seconds,
    /// `ks_tone` 0 (dull) .. 1 (bright)
    ks: f32,
    ks_tone: f32,
    /// a vowel's formants (1 a, 2 e, 3 i, 4 o, 5 u; 0 none) on the oscillators
    vowel: u8,
    /// semitones the note starts away from itself, gliding in (`glide`)
    bend: f32,
    /// a recorded instrument (an id from `samples::multi`, `gm` or `drum`;
    /// 0: the synthesiser). It plays in place of the oscillators (the
    /// waves, unison, FM, the string, the noise unused); the vowel, the
    /// filter and the drive after it only as the patch turns them on
    /// (`SAMPLED` has them off). Its envelope: `a` the attack, then level 1
    /// while the gate is open (`d` and `s` unused: the recording decays as it
    /// does), `r` the release (0: the instrument's own). Pitch, glide, bend,
    /// vibrato, tremolo, velocity and gain as any voice's.
    samp: u16,
}

/// The vowels' first three formants, Hz.
const VOWELS: [[f32; 3]; 5] = [
    [730.0, 1090.0, 2440.0],
    [530.0, 1840.0, 2480.0],
    [300.0, 2200.0, 2950.0],
    [570.0, 840.0, 2410.0],
    [320.0, 800.0, 2240.0],
];

/// A string's delay line, a power of two (the lowest note: 48000 / 2048 Hz, 23 Hz).
const KS_LEN: usize = 2048;

const PATCH: Patch = Patch {
    wave: Wave::Saw,
    pw: 0.5,
    unison: 1,
    detune: 0.0,
    width: 0.0,
    wave2: Wave::Sine,
    mix2: 0.0,
    semi2: 0.0,
    sub: 0.0,
    noise: 0.0,
    fm: 0.0,
    ratio: 1.0,
    fm_dec: 1.0,
    fm2: 0.0,
    ratio2: 1.0,
    fm2_dec: 0.1,
    filter: true,
    cutoff: 2000.0,
    res: 0.1,
    env_amt: 0.0,
    fdec: 0.3,
    keytrack: 0.5,
    poles4: false,
    a: 0.005,
    d: 0.3,
    s: 0.7,
    r: 0.2,
    vib: 0.0,
    vib_rate: 5.0,
    trem: 0.0,
    glide: 0.0,
    drive: 0.0,
    gain: 1.0,
    ks: 0.0,
    ks_tone: 0.5,
    vowel: 0,
    bend: 0.0,
    samp: 0,
};

/// Where a sampled patch starts (`samp` set): no filter, the instrument's
/// own release.
#[allow(dead_code)]
const SAMPLED: Patch = Patch { filter: false, a: 0.002, r: 0.0, ..PATCH };

#[derive(Clone, Copy)]
struct Voice {
    on: bool,
    p: Patch,
    note: f32,
    target: f32,
    vel: f32,
    accent: f32,
    /// samples left before the gate closes
    hold: i64,
    age: u32,
    /// phases: unison oscillators, the second waveform, the sub
    ph: [f32; 5],
    mph: f32,
    mph2: f32,
    fme: f32,
    fme2: f32,
    fmc: f32,
    fmc2: f32,
    amp: Adsr,
    fenv: f32,
    f: [Svf; 2],
    lfo: f32,
    noise: Noise,
    /// the gain (velocity, tremolo, accent) as applied: it glides (4 ms), so
    /// a sounding voice retriggered softer or louder does not step
    g: f32,
    /// the string: its delay line, where it writes, its length, its loop's
    /// low-pass (coefficient, state) and its loss each time round
    kb: [f32; KS_LEN],
    ki: usize,
    klen: f32,
    kc: f32,
    klp: f32,
    kfb: f32,
    /// a sounding voice restarted on another source (a string re-plucked, a
    /// stolen voice of another kind): its output carried over the jump, the
    /// last sample, an offset decaying from it, whether one is due
    last: [f32; 2],
    off: [f32; 2],
    fix: bool,
    /// the formants: three band-passes a side
    ff: [Svf; 6],
    /// a sampled patch's zones playing (none: the oscillators)
    sm: Sampler,
}

/// The per-sample step of a voice's gain glide.
const GLIDE_G: f32 = 0.0052;

const VOICE: Voice = Voice {
    on: false,
    p: PATCH,
    note: 60.0,
    target: 60.0,
    vel: 0.0,
    accent: 0.0,
    hold: 0,
    age: 0,
    ph: [0.0; 5],
    mph: 0.0,
    mph2: 0.0,
    fme: 0.0,
    fme2: 0.0,
    fmc: 1.0,
    fmc2: 1.0,
    amp: Adsr { stage: IDLE, level: 0.0, att: 0.0, dec: 0.0, sus: 0.0, rel: 0.0 },
    fenv: 0.0,
    f: [Svf { ic1: 0.0, ic2: 0.0, a1: 0.0, a2: 0.0, a3: 0.0, k: 2.0 }; 2],
    lfo: 0.0,
    noise: Noise(0x2545_F491),
    g: 0.0,
    kb: [0.0; KS_LEN],
    ki: 0,
    klen: 100.0,
    kc: 0.5,
    klp: 0.0,
    kfb: 0.0,
    last: [0.0; 2],
    off: [0.0; 2],
    fix: false,
    ff: [Svf { ic1: 0.0, ic2: 0.0, a1: 0.0, a2: 0.0, a3: 0.0, k: 2.0 }; 6],
    sm: SAMPLER,
};

impl Voice {
    /// A note: `len` samples of gate. A slide into a voice whose gate is still
    /// open glides there without a new attack (the 303's slide). A sampled
    /// patch's zones come from `bank` (round robin `rr`); none there, the
    /// synthesiser plays it.
    #[allow(clippy::too_many_arguments)]
    fn start(
        &mut self,
        p: &Patch,
        note: f32,
        vel: f32,
        len: i64,
        slide: bool,
        seed: u32,
        bank: Option<&Bank>,
        rr: u32,
    ) {
        let held = self.on && self.amp.stage != RELEASE;
        let accent = ((vel - 0.85) / 0.15).clamp(0.0, 1.0);
        if slide && held {
            self.target = note;
            self.hold = len;
            self.vel = vel;
            self.accent = accent;
            return;
        }
        let fresh = !self.on;
        let sampled = p.samp != 0 && bank.is_some_and(|b| self.sm.start(b, p.samp, note, vel, len, rr));
        if !sampled {
            self.sm.n = 0;
        }
        if fresh {
            // a fresh voice: oscillators at scattered phases (the attack
            // hides the start), filters empty
            self.noise = Noise(seed);
            for ph in self.ph.iter_mut() {
                *ph = self.noise.next() * 0.5 + 0.5;
            }
            self.ph[4] = 0.0;
            self.mph = 0.0;
            self.mph2 = 0.0;
            self.f = [Svf::default(); 2];
            self.ff = [Svf::default(); 6];
            self.lfo = self.noise.next() * 0.5 + 0.5;
            self.amp.level = 0.0;
            if p.ks > 0.0 {
                self.kb.fill(0.0);
                self.klp = 0.0;
            }
            self.last = [0.0; 2];
            self.off = [0.0; 2];
        } else {
            let kind = |q: &Patch| {
                if q.samp != 0 {
                    3
                } else if q.ks > 0.0 {
                    2
                } else if q.fm > 0.0 {
                    1
                } else {
                    0
                }
            };
            if p.ks > 0.0 {
                // a string still ringing, plucked again: damped and restarted
                self.kb.fill(0.0);
                self.klp = 0.0;
            }
            // the output carried over the jump to the new source (no click;
            // a sample restarts from its beginning)
            self.fix = p.ks > 0.0 || sampled || kind(p) != kind(&self.p);
        }
        self.p = *p;
        self.on = true;
        self.note = note + p.bend;
        self.target = note;
        if p.ks > 0.0 {
            // the string: a delay a period long (less its loop filter's lag),
            // losing -60 dB over `ks` seconds, plucked with a burst of noise
            // as bright as its tone (added to what still rings: no click)
            let f = mtof(note);
            self.kc = 0.25 + 0.7 * p.ks_tone;
            self.klen = (SR / f - (1.0 - self.kc) / self.kc).clamp(2.0, (KS_LEN - 2) as f32);
            self.kfb = (-6.9 / (f * p.ks.max(0.01))).exp();
            let n = self.klen.ceil() as usize;
            let ce = 0.15 + 0.8 * p.ks_tone;
            let amp = 1.4 / ce.sqrt();
            let mut lp = 0.0;
            for k in 0..n {
                lp += (self.noise.next() - lp) * ce;
                self.kb[(self.ki + KS_LEN - n + k) & (KS_LEN - 1)] += lp * amp;
            }
        }
        if p.vowel > 0 {
            let fs = VOWELS[(p.vowel as usize - 1) % VOWELS.len()];
            for (k, f) in self.ff.iter_mut().enumerate() {
                f.set(fs[k % 3], [0.9, 0.93, 0.93][k % 3]);
            }
        }
        self.vel = vel;
        self.accent = accent;
        self.hold = len;
        self.age = 0;
        if sampled {
            // the attack, level 1 while held, the release (or the instrument's own)
            self.amp.set(p.a, p.d, 1.0, if p.r > 0.0 { p.r } else { self.sm.rel });
        } else {
            self.amp.set(p.a, p.d, p.s, p.r);
        }
        self.amp.on();
        self.fenv = 1.0;
        // harder notes, brighter FM
        self.fme = 0.45 + 0.75 * vel;
        self.fme2 = self.fme;
        self.fmc = t60(p.fm_dec);
        self.fmc2 = t60(p.fm2_dec);
    }

    /// One sample out, carried over a restart's jump when one is due.
    #[inline]
    fn out(&mut self, o: &mut [f32; 2], l: f32, r: f32) {
        if self.fix {
            self.off = [self.last[0] - l, self.last[1] - r];
            self.fix = false;
        }
        let (l, r) = (l + self.off[0], r + self.off[1]);
        self.off[0] *= 0.995;
        self.off[1] *= 0.995;
        self.last = [l, r];
        o[0] += l;
        o[1] += r;
    }

    /// Adds the voice into `buf`; `cut` scales the cutoff (the channel's
    /// drift); a sampled voice's zones are in `bank`.
    fn render(&mut self, cut: f32, buf: &mut [[f32; 2]], bank: Option<&Bank>) {
        let n = buf.len();
        let nf = n as f32;
        let p = self.p;
        if self.hold > 0 {
            self.hold -= n as i64;
            if self.hold <= 0 {
                self.amp.off();
            }
        }
        if self.note != self.target {
            let k = if p.glide > 0.0 { 1.0 - (-nf / (p.glide * SR)).exp() } else { 1.0 };
            self.note += (self.target - self.note) * k;
            if (self.target - self.note).abs() < 1e-3 {
                self.note = self.target;
            }
        }
        self.lfo = wrap(self.lfo + p.vib_rate * nf / SR);
        let lfo = sin01(self.lfo);
        let pitch = self.note + p.vib * lfo * (self.age as f32 / (0.4 * SR)).min(1.0);
        let f = mtof(pitch);
        let dt0 = (f / SR).min(0.45);
        let mut gain = self.vel * p.gain * (1.0 - p.trem * (0.5 + 0.5 * lfo)) * (1.0 + 0.3 * self.accent);
        let sampled = self.sm.n > 0;
        if sampled {
            gain *= samples::CAL;
        }
        if self.amp.level == 0.0 {
            self.g = gain;
        }
        let stereo = sampled || (p.unison > 1 && p.width > 0.0);
        if p.filter {
            let oct =
                (p.env_amt * (0.55 + 0.45 * self.vel) + self.accent) * self.fenv + (pitch - 60.0) * p.keytrack / 12.0;
            // envelopes peak at 9 kHz: brighter only zaps
            let fc = (p.cutoff * cut * oct.exp2()).min(9000.0);
            let res = (p.res + 0.08 * self.accent).min(0.97);
            if p.poles4 {
                self.f[0].set(fc, 0.3);
                self.f[1].set(fc, res);
            } else {
                self.f[0].set(fc, res);
                if stereo {
                    self.f[1].set(fc, res);
                }
            }
            self.fenv *= (-6.9 * nf / (p.fdec * SR)).exp();
        }
        if sampled {
            let gate = self.amp.stage != RELEASE;
            let mut live = true;
            let mut tmp = [[0.0f32; 2]; BLOCK];
            for chunk in buf.chunks_mut(BLOCK) {
                let t = &mut tmp[..chunk.len()];
                live &= bank.is_some_and(|b| self.sm.fill(b, pitch, gate, t));
                for (o, s) in chunk.iter_mut().zip(t.iter()) {
                    let (mut l, mut r) = (s[0], s[1]);
                    if p.vowel > 0 {
                        let ff = &mut self.ff;
                        let form = |f: &mut [Svf], x: f32| 1.6 * (f[0].bp(x) + 0.6 * f[1].bp(x) + 0.35 * f[2].bp(x));
                        l = form(&mut ff[..3], l);
                        r = form(&mut ff[3..], r);
                    }
                    if p.filter {
                        if p.poles4 {
                            let m = self.f[0].lp((l + r) * 0.5);
                            let m = self.f[1].lp(m);
                            l = m;
                            r = m;
                        } else {
                            l = self.f[0].lp(l);
                            r = self.f[1].lp(r);
                        }
                    }
                    if p.drive > 0.0 {
                        l = soft(l * (1.0 + p.drive));
                        r = soft(r * (1.0 + p.drive));
                    }
                    self.g += (gain - self.g) * GLIDE_G;
                    let a = self.amp.next() * self.g;
                    self.out(o, l * a, r * a);
                }
            }
            // the recording over (or died away): the voice with it
            if !live {
                self.amp.stage = IDLE;
                self.amp.level = 0.0;
            }
        } else if p.ks > 0.0 {
            for o in buf.iter_mut() {
                let rp = self.ki as f32 + KS_LEN as f32 - self.klen;
                let i0 = rp as usize;
                let fr = rp - i0 as f32;
                let a0 = self.kb[i0 & (KS_LEN - 1)];
                let y = a0 + (self.kb[(i0 + 1) & (KS_LEN - 1)] - a0) * fr;
                self.klp += (y - self.klp) * self.kc;
                self.kb[self.ki] = self.klp * self.kfb;
                self.ki = (self.ki + 1) & (KS_LEN - 1);
                let mut x = y;
                if p.filter {
                    x = self.f[0].lp(x);
                }
                if p.drive > 0.0 {
                    x = soft(x * (1.0 + p.drive));
                }
                self.g += (gain - self.g) * GLIDE_G;
                let a = self.amp.next() * self.g;
                self.out(o, x * a, x * a);
            }
        } else if p.fm > 0.0 {
            let (r1, r2) = (dt0 * p.ratio, dt0 * p.ratio2);
            // the second modulator (a tine's ping, high above) eases off up the keyboard
            let fm2 = p.fm2 * (400.0 / f).min(1.0);
            for o in buf.iter_mut() {
                self.mph += r1;
                while self.mph >= 1.0 {
                    self.mph -= 1.0;
                }
                self.mph2 += r2;
                while self.mph2 >= 1.0 {
                    self.mph2 -= 1.0;
                }
                self.fme *= self.fmc;
                self.fme2 *= self.fmc2;
                let pm = (p.fm * self.fme * sin01(self.mph) + fm2 * self.fme2 * sin01(self.mph2))
                    * (1.0 / std::f32::consts::TAU);
                self.ph[0] = wrap(self.ph[0] + dt0);
                let mut x = sin01(self.ph[0] + pm);
                if p.filter {
                    x = self.f[0].lp(x);
                }
                self.g += (gain - self.g) * GLIDE_G;
                let a = self.amp.next() * self.g;
                self.out(o, x * a, x * a);
            }
        } else {
            let u = (p.unison as usize).clamp(1, 3);
            let norm = 1.0 / (u as f32).sqrt();
            let (mut dts, mut gl, mut gr) = ([0.0f32; 3], [0.0f32; 3], [0.0f32; 3]);
            for k in 0..u {
                let x = if u == 1 { 0.0 } else { -1.0 + 2.0 * k as f32 / (u - 1) as f32 };
                dts[k] = (dt0 * (x * p.detune / 1200.0).exp2()).min(0.45);
                let pan = x * p.width;
                gl[k] = (1.0 - pan).min(1.0) * norm;
                gr[k] = (1.0 + pan).min(1.0) * norm;
            }
            let dt2 = (dt0 * (p.semi2 / 12.0).exp2()).min(0.45);
            let dsub = dt0 * 0.5;
            for o in buf.iter_mut() {
                let (mut l, mut r) = (0.0, 0.0);
                for k in 0..u {
                    let s = osc(p.wave, self.ph[k], dts[k], p.pw);
                    self.ph[k] = wrap(self.ph[k] + dts[k]);
                    l += s * gl[k];
                    r += s * gr[k];
                }
                let mut c = 0.0;
                if p.mix2 > 0.0 {
                    c += p.mix2 * osc(p.wave2, self.ph[3], dt2, 0.5);
                    self.ph[3] = wrap(self.ph[3] + dt2);
                }
                if p.sub > 0.0 {
                    c += p.sub * sin01(self.ph[4]);
                    self.ph[4] = wrap(self.ph[4] + dsub);
                }
                if p.noise > 0.0 {
                    c += p.noise * self.noise.next();
                }
                l += c;
                r += c;
                if p.vowel > 0 {
                    let ff = &mut self.ff;
                    let form = |f: &mut [Svf], x: f32| 1.6 * (f[0].bp(x) + 0.6 * f[1].bp(x) + 0.35 * f[2].bp(x));
                    if stereo {
                        l = form(&mut ff[..3], l);
                        r = form(&mut ff[3..], r);
                    } else {
                        let m = form(&mut ff[..3], (l + r) * 0.5);
                        l = m;
                        r = m;
                    }
                }
                if p.filter {
                    if stereo {
                        l = self.f[0].lp(l);
                        r = self.f[1].lp(r);
                    } else {
                        let mut m = self.f[0].lp((l + r) * 0.5);
                        if p.poles4 {
                            m = self.f[1].lp(m);
                        }
                        l = m;
                        r = m;
                    }
                }
                if p.drive > 0.0 {
                    l = soft(l * (1.0 + p.drive));
                    r = soft(r * (1.0 + p.drive));
                }
                self.g += (gain - self.g) * GLIDE_G;
                let a = self.amp.next() * self.g;
                self.out(o, l * a, r * a);
            }
        }
        self.age = self.age.saturating_add(n as u32);
        if self.amp.stage == IDLE {
            self.on = false;
        }
    }
}

/// A drum: a sine with pitch and amplitude envelopes, plus noise (or the
/// 808's six square waves, for metal) through a filter with its own envelope,
/// in a few bursts for a clap.
#[derive(Clone, Copy)]
struct DrumPatch {
    tone: f32,
    f0: f32,
    f1: f32,
    /// the pitch envelope's time constant, the tone's -60 dB time
    pdec: f32,
    tdec: f32,
    tone2: f32,
    ratio2: f32,
    noise: f32,
    /// 0 low-pass, 1 band-pass, 2 high-pass (two, steeper)
    nmode: u8,
    ncut: f32,
    nres: f32,
    ndec: f32,
    natt: f32,
    metal: f32,
    mf: f32,
    claps: u8,
    click: f32,
    drive: f32,
    gain: f32,
    /// a recorded hit (an id from `samples::drum` or `multi`; 0: the
    /// synthesiser's): played once through in place of the tone and the
    /// noise (the gate and the loops ignored), then `drive` and `gain`
    samp: u16,
}

const DRUM: DrumPatch = DrumPatch {
    tone: 0.0,
    f0: 100.0,
    f1: 100.0,
    pdec: 0.05,
    tdec: 0.2,
    tone2: 0.0,
    ratio2: 1.5,
    noise: 0.0,
    nmode: 2,
    ncut: 5000.0,
    nres: 0.2,
    ndec: 0.1,
    natt: 0.0005,
    metal: 0.0,
    mf: 1.0,
    claps: 0,
    click: 0.0,
    drive: 0.0,
    gain: 1.0,
    samp: 0,
};

/// The 808's cymbal oscillators, Hz.
const METAL: [f32; 6] = [205.3, 304.4, 369.6, 522.7, 540.0, 800.0];
/// A clap's bursts, samples apart (9 ms).
const CLAP_GAP: u32 = 432;

#[derive(Clone, Copy)]
struct Drum {
    on: bool,
    kind: u8,
    p: DrumPatch,
    vel: f32,
    t: u32,
    ph: f32,
    ph2: f32,
    pe: f32,
    pc: f32,
    ta: f32,
    tc: f32,
    na: f32,
    nc: f32,
    burst: f32,
    bc: f32,
    natt: u32,
    mph: [f32; 6],
    f: Svf,
    f2: Svf,
    noise: Noise,
    /// a recorded hit's zones, and its level (a choke fades it) and the
    /// level's step a sample
    sm: Sampler,
    sg: f32,
    sgc: f32,
}

const DRUMV: Drum = Drum {
    on: false,
    kind: 0,
    p: DRUM,
    vel: 0.0,
    t: 0,
    ph: 0.0,
    ph2: 0.0,
    pe: 0.0,
    pc: 0.0,
    ta: 0.0,
    tc: 0.0,
    na: 0.0,
    nc: 0.0,
    burst: 0.0,
    bc: 0.0,
    natt: 1,
    mph: [0.0; 6],
    f: Svf { ic1: 0.0, ic2: 0.0, a1: 0.0, a2: 0.0, a3: 0.0, k: 2.0 },
    f2: Svf { ic1: 0.0, ic2: 0.0, a1: 0.0, a2: 0.0, a3: 0.0, k: 2.0 },
    noise: Noise(0x2545_F491),
    sm: SAMPLER,
    sg: 1.0,
    sgc: 1.0,
};

impl Drum {
    /// A hit; a recorded one's zones from `bank` (round robin `rr`), and
    /// none there, the synthesiser's.
    fn start(&mut self, p: &DrumPatch, kind: u8, vel: f32, seed: u32, bank: Option<&Bank>, rr: u32) {
        let mut sm = SAMPLER;
        let sampled = p.samp != 0 && bank.is_some_and(|b| sm.start(b, p.samp, 60.0, vel, 0, rr));
        sm.loops = false;
        let mut n = Noise(seed);
        let mut mph = [0.0; 6];
        for m in mph.iter_mut() {
            *m = n.next() * 0.5 + 0.5;
        }
        *self = Drum {
            on: true,
            kind,
            p: *p,
            vel,
            pe: 1.0,
            pc: (-1.0 / (p.pdec.max(1e-4) * SR)).exp(),
            ta: if p.tone > 0.0 && !sampled { 1.0 } else { 0.0 },
            tc: t60(p.tdec),
            na: if p.noise > 0.0 && !sampled { 1.0 } else { 0.0 },
            nc: t60(p.ndec),
            burst: 1.0,
            bc: t60(0.02),
            natt: ((p.natt * SR) as u32).max(1),
            mph,
            noise: n,
            sm,
            ..DRUMV
        };
        self.f.set(p.ncut, p.nres);
        self.f2.set(p.ncut, p.nres);
    }

    /// Fast fade (an open hat closed by a closed one).
    fn choke(&mut self) {
        self.tc = t60(0.012);
        self.nc = t60(0.012);
        self.sgc = t60(0.012);
    }

    fn render(&mut self, buf: &mut [[f32; 2]], bank: Option<&Bank>) {
        let p = self.p;
        if self.sm.n > 0 {
            let g = self.vel * p.gain * samples::CAL_DRUM;
            let mut live = true;
            let mut tmp = [[0.0f32; 2]; BLOCK];
            for chunk in buf.chunks_mut(BLOCK) {
                let t = &mut tmp[..chunk.len()];
                live &= bank.is_some_and(|b| self.sm.fill(b, self.sm.key, false, t));
                for (o, s) in chunk.iter_mut().zip(t.iter()) {
                    let (mut l, mut r) = (s[0], s[1]);
                    if p.drive > 0.0 {
                        l = soft(l * (1.0 + p.drive));
                        r = soft(r * (1.0 + p.drive));
                    }
                    self.sg *= self.sgc;
                    o[0] += l * g * self.sg;
                    o[1] += r * g * self.sg;
                }
                self.t += chunk.len() as u32;
            }
            if !live || self.sg < 1e-4 {
                self.on = false;
            }
            return;
        }
        let g = self.vel * p.gain;
        let clap_end = p.claps as u32 * CLAP_GAP;
        let minc: [f32; 6] = std::array::from_fn(|k| METAL[k] * p.mf / SR);
        for o in buf.iter_mut() {
            let mut x = 0.0;
            if self.ta > 1e-5 {
                self.pe *= self.pc;
                let f = p.f1 + (p.f0 - p.f1) * self.pe;
                self.ph = wrap(self.ph + f / SR);
                let mut s = sin01(self.ph);
                if p.tone2 > 0.0 {
                    self.ph2 = wrap(self.ph2 + (f * p.ratio2 / SR).min(0.45));
                    s += p.tone2 * sin01(self.ph2);
                }
                x += s * self.ta * p.tone;
                self.ta *= self.tc;
            }
            if self.na > 1e-5 {
                let mut w = self.noise.next();
                if p.metal > 0.0 {
                    let mut m = 0.0;
                    for k in 0..6 {
                        self.mph[k] = wrap(self.mph[k] + minc[k]);
                        m += if self.mph[k] < 0.5 { 1.0 } else { -1.0 };
                    }
                    w = w * (1.0 - p.metal) + m * p.metal * 0.3;
                }
                w = match p.nmode {
                    0 => self.f.lp(w),
                    1 => self.f.bp(w),
                    _ => self.f2.hp(self.f.hp(w)),
                };
                let env = if self.t < clap_end {
                    let b = self.burst;
                    self.burst = if (self.t + 1) % CLAP_GAP == 0 { 1.0 } else { b * self.bc };
                    b
                } else {
                    self.na *= self.nc;
                    self.na
                };
                let att = if self.t < self.natt { self.t as f32 / self.natt as f32 } else { 1.0 };
                x += w * env * att * p.noise;
            }
            if p.click > 0.0 && self.t < 96 {
                x += p.click * self.noise.next() * (1.0 - self.t as f32 / 96.0);
            }
            if p.drive > 0.0 {
                x = soft(x * (1.0 + p.drive));
            }
            o[0] += x * g;
            o[1] += x * g;
            self.t += 1;
        }
        if self.ta <= 1e-5 && self.na <= 1e-5 && self.t >= clap_end {
            self.on = false;
        }
    }
}

// ---------------------------------------------------------------- effects

struct Comb {
    b: Vec<f32>,
    i: usize,
    z: f32,
}

struct Allpass {
    b: Vec<f32>,
    i: usize,
}

/// Freeverb (Jezar's): eight damped combs and four allpasses a side, its
/// lengths scaled from 44.1 to 48 kHz and by `size`.
struct Reverb {
    cl: Vec<Comb>,
    cr: Vec<Comb>,
    al: Vec<Allpass>,
    ar: Vec<Allpass>,
    /// comb feedback (0.7 .. 0.98: the length of the tail) and damping (0 .. 1: its darkness)
    fb: f32,
    damp: f32,
}

impl Reverb {
    fn new(size: f32) -> Reverb {
        const COMBS: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
        const APS: [usize; 4] = [556, 441, 341, 225];
        let s = size * SR / 44100.0;
        let len = |n: usize| ((n as f32 * s) as usize).max(16);
        let comb = |n: usize| Comb { b: vec![0.0; len(n)], i: 0, z: 0.0 };
        let ap = |n: usize| Allpass { b: vec![0.0; len(n)], i: 0 };
        Reverb {
            cl: COMBS.iter().map(|&n| comb(n)).collect(),
            cr: COMBS.iter().map(|&n| comb(n + 23)).collect(),
            al: APS.iter().map(|&n| ap(n)).collect(),
            ar: APS.iter().map(|&n| ap(n + 23)).collect(),
            fb: 0.84,
            damp: 0.4,
        }
    }

    #[inline]
    fn tick(&mut self, x: f32) -> (f32, f32) {
        // a whisper of DC keeps the loops out of denormals
        let x = x * 0.015 + 1e-18;
        let (fb, damp) = (self.fb, self.damp);
        let comb = |c: &mut Comb| {
            let y = c.b[c.i];
            c.z = y * (1.0 - damp) + c.z * damp;
            c.b[c.i] = x + c.z * fb;
            c.i += 1;
            if c.i == c.b.len() {
                c.i = 0;
            }
            y
        };
        let ap = |a: &mut Allpass, x: f32| {
            let b = a.b[a.i];
            a.b[a.i] = x + b * 0.5;
            a.i += 1;
            if a.i == a.b.len() {
                a.i = 0;
            }
            b - x
        };
        let mut l: f32 = self.cl.iter_mut().map(comb).sum();
        let mut r: f32 = self.cr.iter_mut().map(comb).sum();
        for a in self.al.iter_mut() {
            l = ap(a, l);
        }
        for a in self.ar.iter_mut() {
            r = ap(a, r);
        }
        (l * 3.0, r * 3.0)
    }
}

const DLEN: usize = 1 << 17;

/// A ping-pong delay (in on the left, crossing sides each repeat) with a
/// band-pass in the loop so repeats darken and thin; its time glides to a new
/// tempo rather than jumping.
struct Delay {
    l: Vec<f32>,
    r: Vec<f32>,
    w: usize,
    time: f32,
    target: f32,
    fb: f32,
    lpk: f32,
    lz: [f32; 2],
    hz: [f32; 2],
}

impl Delay {
    fn new() -> Delay {
        Delay {
            l: vec![0.0; DLEN],
            r: vec![0.0; DLEN],
            w: 0,
            time: 12000.0,
            target: 12000.0,
            fb: 0.4,
            lpk: 0.4,
            lz: [0.0; 2],
            hz: [0.0; 2],
        }
    }

    #[inline]
    fn tick(&mut self, x: f32) -> (f32, f32) {
        // kept inside the line (a time past it read garbage and went NaN)
        self.time = (self.time + (self.target - self.time) * 0.0001).clamp(1.0, DLEN as f32 - 2.0);
        let rp = self.w as f32 + DLEN as f32 - self.time;
        let i0 = rp as usize;
        let fr = rp - i0 as f32;
        let (a, b) = (i0 & (DLEN - 1), (i0 + 1) & (DLEN - 1));
        let dl = self.l[a] + (self.l[b] - self.l[a]) * fr;
        let dr = self.r[a] + (self.r[b] - self.r[a]) * fr;
        let mut tone = |i: usize, v: f32| {
            self.lz[i] += (v - self.lz[i]) * self.lpk;
            self.hz[i] += (self.lz[i] - self.hz[i]) * 0.02;
            self.lz[i] - self.hz[i]
        };
        let (fl, fr2) = (tone(0, dl), tone(1, dr));
        self.l[self.w] = x + fr2 * self.fb + 1e-18;
        self.r[self.w] = fl * self.fb + 1e-18;
        self.w = (self.w + 1) & (DLEN - 1);
        (dl, dr)
    }
}

const LOOK: usize = 48;

/// A lookahead peak limiter: the gain each sample needs, its minimum over the
/// next millisecond, a release, then a millisecond's moving average, applied
/// to the signal delayed a millisecond (so it is down before any peak arrives).
struct Limiter {
    ceil: f32,
    dl: [[f32; 2]; LOOK],
    req: VecDeque<(u64, f32)>,
    avg: [f32; LOOK],
    sum: f64,
    g: f32,
    rel: f32,
    n: u64,
    i: usize,
}

impl Limiter {
    fn new(ceil: f32) -> Limiter {
        Limiter {
            ceil,
            dl: [[0.0; 2]; LOOK],
            req: VecDeque::with_capacity(LOOK + 2),
            avg: [1.0; LOOK],
            sum: LOOK as f64,
            g: 1.0,
            rel: 1.0 - (-1.0 / (0.08 * SR)).exp(),
            n: 0,
            i: 0,
        }
    }

    #[inline]
    fn tick(&mut self, l: f32, r: f32) -> (f32, f32) {
        let pk = l.abs().max(r.abs());
        let need = if pk > self.ceil { self.ceil / pk } else { 1.0 };
        while self.req.back().is_some_and(|&(_, v)| v >= need) {
            self.req.pop_back();
        }
        self.req.push_back((self.n, need));
        while self.req.front().is_some_and(|&(t, _)| t + (LOOK as u64) < self.n) {
            self.req.pop_front();
        }
        let m = self.req.front().map_or(1.0, |&(_, v)| v);
        self.g = if m < self.g { m } else { self.g + (m - self.g) * self.rel };
        self.sum += (self.g - self.avg[self.i]) as f64;
        self.avg[self.i] = self.g;
        let gs = (self.sum / LOOK as f64) as f32;
        let [ol, or] = self.dl[self.i];
        self.dl[self.i] = [l, r];
        self.i = (self.i + 1) % LOOK;
        self.n += 1;
        (ol * gs, or * gs)
    }
}

/// The master: an optional low-pass and tape wow (lo-fi), gain, a slow
/// leveller (half the way to the target loudness, within ±6 dB), a DC
/// blocker, a soft knee and the limiter.
struct Master {
    gain: f32,
    target: f32,
    lp: f32,
    f: [Svf; 2],
    /// wow depth in samples, gliding to `wow_to` (a new piece's depth must
    /// not jump the read position: a splice)
    wow: f32,
    wow_to: f32,
    wrate: f32,
    wbuf: Vec<[f32; 2]>,
    wi: usize,
    wph: f32,
    fph: f32,
    dc: [f32; 4],
    ms: f32,
    agc: f32,
    /// how much of the level's error the leveller takes out (as a power),
    /// and the most it adds
    lk: f32,
    lmax: f32,
    lim: Limiter,
}

impl Master {
    fn new() -> Master {
        Master {
            gain: 1.0,
            target: 0.2,
            lp: 20000.0,
            f: [Svf::default(); 2],
            wow: 0.0,
            wow_to: 0.0,
            wrate: 0.5,
            wbuf: vec![[0.0; 2]; 2048],
            wi: 0,
            wph: 0.0,
            fph: 0.0,
            dc: [0.0; 4],
            ms: 0.04,
            agc: 1.0,
            lk: 0.5,
            lmax: 2.0,
            lim: Limiter::new(0.89),
        }
    }

    fn set_lp(&mut self, hz: f32) {
        self.lp = hz;
        for f in self.f.iter_mut() {
            f.set(hz, 0.05);
        }
    }

    #[inline]
    fn tick(&mut self, mut l: f32, mut r: f32) -> (f32, f32) {
        if self.lp < 19000.0 {
            l = self.f[0].lp(l);
            r = self.f[1].lp(r);
        }
        if self.wow_to > 0.0 || self.wow > 0.0 {
            self.wow += (self.wow_to - self.wow) * 2e-5;
            let len = self.wbuf.len();
            self.wbuf[self.wi] = [l, r];
            self.wph = wrap(self.wph + self.wrate / SR);
            self.fph = wrap(self.fph + 6.3 / SR);
            let d = 4.0 + self.wow * (1.0 + sin01(self.wph)) + 0.06 * self.wow * (1.0 + sin01(self.fph));
            let rp = self.wi as f32 + len as f32 - d;
            let i0 = rp as usize;
            let fr = rp - i0 as f32;
            let (a, b) = (self.wbuf[i0 % len], self.wbuf[(i0 + 1) % len]);
            l = a[0] + (b[0] - a[0]) * fr;
            r = a[1] + (b[1] - a[1]) * fr;
            self.wi = (self.wi + 1) % len;
        }
        l *= self.gain;
        r *= self.gain;
        self.ms += ((l * l + r * r) * 0.5 - self.ms) * (1.0 / (3.0 * SR));
        l *= self.agc;
        r *= self.agc;
        // DC blocker (-3 dB at 4 Hz)
        let yl = l - self.dc[0] + 0.9995 * self.dc[1];
        let yr = r - self.dc[2] + 0.9995 * self.dc[3];
        self.dc = [l, yl, r, yr];
        let knee = |x: f32| {
            let a = x.abs();
            if a <= 0.7 { x } else { (0.7 + 0.3 * soft((a - 0.7) / 0.3)).copysign(x) }
        };
        let (l, r) = self.lim.tick(knee(yl), knee(yr));
        (l.clamp(-0.944, 0.944), r.clamp(-0.944, 0.944))
    }

    /// The leveller, once a block.
    fn block(&mut self, n: usize) {
        let want = (self.target / (self.ms.sqrt() + 1e-6)).powf(self.lk).clamp(0.5, self.lmax);
        self.agc += (want - self.agc) * (n as f32 / (2.5 * SR));
    }
}

/// Noise textures: a riser (noise swept up a band-pass), wind (pinkish noise
/// through a wandering band-pass), vinyl crackle and hiss.
struct Fx {
    riser_left: u32,
    riser_len: u32,
    riser_vel: f32,
    rf: [Svf; 2],
    wind: f32,
    wind_to: f32,
    wf: [Svf; 2],
    wz: [f32; 2],
    wwalk: f32,
    crackle: f32,
    crackle_rate: f32,
    /// rain: a soft noise bed and drops (a ping through a resonant band-pass)
    rain: f32,
    rain_to: f32,
    rz: [f32; 2],
    drop: Svf,
    dpan: f32,
    pop: f32,
    pan: f32,
    hiss: f32,
    hz: [f32; 2],
    n: Noise,
    n2: Noise,
}

impl Fx {
    fn new() -> Fx {
        Fx {
            riser_left: 0,
            riser_len: 1,
            riser_vel: 0.0,
            rf: [Svf::default(); 2],
            wind: 0.0,
            wind_to: 0.0,
            wf: [Svf::default(); 2],
            wz: [0.0; 2],
            wwalk: 0.0,
            crackle: 0.0,
            crackle_rate: 0.0,
            rain: 0.0,
            rain_to: 0.0,
            rz: [0.0; 2],
            drop: Svf::default(),
            dpan: 0.0,
            pop: 0.0,
            pan: 0.0,
            hiss: 0.0,
            hz: [0.0; 2],
            n: Noise(0x1234_5679),
            n2: Noise(0x0BAD_F00D),
        }
    }

    fn riser(&mut self, secs: f32, vel: f32) {
        self.riser_len = ((secs * SR) as u32).max(1);
        self.riser_left = self.riser_len;
        self.riser_vel = vel;
    }

    /// Adds into `buf`; whether anything sounded.
    fn render(&mut self, buf: &mut [[f32; 2]]) -> bool {
        let n = buf.len();
        let mut any = false;
        if self.riser_left > 0 {
            any = true;
            let x = 1.0 - self.riser_left as f32 / self.riser_len as f32;
            for f in self.rf.iter_mut() {
                f.set(220.0 * (x * 5.4).exp2(), 0.55);
            }
            let a = self.riser_vel * x * x;
            for o in buf.iter_mut() {
                if self.riser_left == 0 {
                    break;
                }
                let fade = (self.riser_left as f32 / 480.0).min(1.0);
                o[0] += self.rf[0].bp(self.n.next()) * a * fade;
                o[1] += self.rf[1].bp(self.n2.next()) * a * fade;
                self.riser_left -= 1;
            }
        }
        self.wind += (self.wind_to - self.wind) * (n as f32 / (3.0 * SR));
        if self.wind > 1e-4 {
            any = true;
            self.wwalk = (self.wwalk + self.n.next() * 0.01).clamp(-1.0, 1.0);
            let fc = 500.0 * (self.wwalk * 1.6).exp2();
            self.wf[0].set(fc, 0.6);
            self.wf[1].set(fc * 1.12, 0.6);
            for o in buf.iter_mut() {
                self.wz[0] += (self.n.next() - self.wz[0]) * 0.15;
                self.wz[1] += (self.n2.next() - self.wz[1]) * 0.15;
                o[0] += self.wf[0].bp(self.wz[0]) * self.wind * 2.0;
                o[1] += self.wf[1].bp(self.wz[1]) * self.wind * 2.0;
            }
        }
        self.rain += (self.rain_to - self.rain) * (n as f32 / (3.0 * SR));
        if self.rain > 1e-4 {
            any = true;
            for o in buf.iter_mut() {
                self.rz[0] += (self.n.next() - self.rz[0]) * 0.3;
                self.rz[1] += (self.n2.next() - self.rz[1]) * 0.3;
                let mut imp = 0.0;
                if self.n.next() * 0.5 + 0.5 < 30.0 / SR {
                    self.drop.set(1500.0 * (self.n2.next() * 1.5 + 1.6).exp2(), 0.96);
                    self.dpan = self.n2.next() * 0.8;
                    imp = self.n.next().abs() * 6.0;
                }
                let d = self.drop.bp(imp);
                o[0] += (self.rz[0] * 0.5 + d * (1.0 - self.dpan)) * self.rain;
                o[1] += (self.rz[1] * 0.5 + d * (1.0 + self.dpan)) * self.rain;
            }
        }
        if self.crackle > 0.0 {
            any = true;
            let p = self.crackle_rate / SR;
            for o in buf.iter_mut() {
                if self.n.next() * 0.5 + 0.5 < p {
                    let a = self.n2.next().abs();
                    self.pop = self.crackle * a * a * a * 4.0;
                    self.pan = self.n2.next() * 0.6;
                }
                let c = self.pop;
                self.pop *= -0.55;
                self.hz[0] += (self.n.next() - self.hz[0]) * 0.3;
                self.hz[1] += (self.n2.next() - self.hz[1]) * 0.3;
                o[0] += c * (1.0 - self.pan) + self.hz[0] * self.hiss;
                o[1] += c * (1.0 + self.pan) + self.hz[1] * self.hiss;
            }
        }
        any
    }
}

// ---------------------------------------------------------------- the synthesiser

const CH_KICK: usize = 0;
const CH_SNARE: usize = 1;
const CH_HAT: usize = 2;
const CH_PERC: usize = 3;
const CH_CYM: usize = 4;
const CH_BASS: usize = 5;
const CH_PAD: usize = 6;
const CH_ARP: usize = 7;
const CH_LEAD: usize = 8;
const CH_BELL: usize = 9;
const CH_KEYS: usize = 10;
const CH_DRONE: usize = 11;
const CH_FX: usize = 12;
const NCH: usize = 13;

/// Drum kinds: on the hat channel, closed and open; on the snare channel the
/// main hit (which opens the gated reverb) and a second; on the percussion
/// channel a rim, three toms and a shaker.
const HAT_C: u8 = 0;
const HAT_O: u8 = 1;
const RIM: u8 = 0;
const TOM_LO: u8 = 1;
const TOM_MID: u8 = 2;
const TOM_HI: u8 = 3;
const SHAKER: u8 = 4;

/// What the composer asks of the synthesiser.
#[derive(Clone, Copy)]
enum Ev {
    Hit {
        ch: u8,
        kind: u8,
        vel: f32,
    },
    /// `len`: seconds of gate
    Note {
        ch: u8,
        note: f32,
        vel: f32,
        len: f32,
        slide: bool,
    },
    Riser {
        secs: f32,
        vel: f32,
    },
}

struct Channel {
    patch: Patch,
    voices: Vec<Voice>,
    mono: bool,
    drums: Vec<DrumPatch>,
    dv: Vec<Drum>,
    gain: f32,
    gain_s: f32,
    pan: f32,
    /// sends: reverb, delay, the second reverb (gate or rumble)
    rev: f32,
    dly: f32,
    rev2: f32,
    /// how deep the kick ducks it
    duck: f32,
    /// the cutoff's drift (a factor), smoothed
    cut: f32,
    cut_s: f32,
    /// a low-pass over the whole channel (Hz; off from 19 kHz), its ceiling, smoothed
    lp: f32,
    lp_base: f32,
    lp_s: f32,
    filt: [Svf; 2],
    filtering: bool,
    /// a slow swell of the level (an accordion's bellows): depth, rate (Hz), phase
    swell: f32,
    swell_s: f32,
    swell_rate: f32,
    swell_ph: f32,
    buf: [[f32; 2]; BLOCK],
    /// the round robin the next sampled note (0) or hit (by its kind)
    /// takes: each cycles in order (a shaker's strokes down and up)
    rr: [u32; 8],
}

impl Channel {
    fn new(voices: usize, drums: usize) -> Channel {
        Channel {
            patch: PATCH,
            voices: vec![VOICE; voices],
            mono: false,
            drums: Vec::new(),
            dv: vec![DRUMV; drums],
            gain: 0.0,
            gain_s: 0.0,
            pan: 0.0,
            rev: 0.0,
            dly: 0.0,
            rev2: 0.0,
            duck: 0.0,
            cut: 1.0,
            cut_s: 1.0,
            lp: 20000.0,
            lp_base: 20000.0,
            lp_s: 20000.0,
            filt: [Svf::default(); 2],
            filtering: false,
            swell: 0.0,
            swell_s: 0.0,
            swell_rate: 0.5,
            swell_ph: 0.0,
            buf: [[0.0; 2]; BLOCK],
            rr: [0; 8],
        }
    }

    /// Renders the block into `buf`; whether anything sounded. (The swell
    /// keeps time whether or not the channel sounds.)
    fn render(&mut self, n: usize, bank: Option<&Bank>) -> bool {
        self.swell_s += (self.swell - self.swell_s) * 0.02;
        self.swell_ph = wrap(self.swell_ph + self.swell_rate * n as f32 / SR);
        let Channel { voices, dv, buf, .. } = self;
        let buf = &mut buf[..n];
        buf.fill([0.0; 2]);
        self.cut_s += (self.cut - self.cut_s) * 0.02;
        let mut any = false;
        for v in voices.iter_mut().filter(|v| v.on) {
            v.render(self.cut_s, buf, bank);
            any = true;
        }
        for d in dv.iter_mut().filter(|d| d.on) {
            d.render(buf, bank);
            any = true;
        }
        any
    }

    /// A hit of drum `kind`; on the hat channel (`hats`) a closed hat
    /// chokes the open one.
    fn hit(&mut self, hats: bool, kind: u8, vel: f32, seed: u32, bank: Option<&Bank>) {
        if self.drums.is_empty() || self.dv.is_empty() {
            return;
        }
        if hats && kind == HAT_C {
            for d in self.dv.iter_mut().filter(|d| d.on && d.kind == HAT_O) {
                d.choke();
            }
        }
        let p = self.drums[(kind as usize).min(self.drums.len() - 1)];
        let i = self
            .dv
            .iter()
            .position(|d| !d.on)
            .unwrap_or_else(|| (0..self.dv.len()).max_by_key(|&i| self.dv[i].t).unwrap_or(0));
        let k = (kind as usize).min(7);
        self.dv[i].start(&p, kind, vel, seed, bank, self.rr[k]);
        self.rr[k] = self.rr[k].wrapping_add(1);
        // the SoundFont's exclusive classes (an open hat, its pedal and closed hats)
        let ex = self.dv[i].sm.excl;
        if ex != 0 && self.dv[i].sm.n > 0 {
            for (j, d) in self.dv.iter_mut().enumerate() {
                if j != i && d.on && d.sm.n > 0 && d.sm.excl == ex {
                    d.choke();
                }
            }
        }
    }

    /// The channel filter, then into the buses.
    fn mix(&mut self, n: usize, sc: &[f32], dry: &mut [[f32; 2]], rs: &mut [f32], ds: &mut [f32], r2s: &mut [f32]) {
        self.lp_s += (self.lp - self.lp_s) * 0.03;
        let on = self.lp_s < 19000.0;
        if on && !self.filtering {
            // coming in near the top of the range, the filter all but follows
            // its input: start it from there, not from stale state
            for (k, f) in self.filt.iter_mut().enumerate() {
                f.ic1 = 0.0;
                f.ic2 = self.buf[0][k];
            }
        }
        self.filtering = on;
        if on {
            for f in self.filt.iter_mut() {
                f.set(self.lp_s, 0.1);
            }
            for o in self.buf[..n].iter_mut() {
                o[0] = self.filt[0].lp(o[0]);
                o[1] = self.filt[1].lp(o[1]);
            }
        }
        self.gain_s += (self.gain - self.gain_s) * 0.05;
        let mut g = self.gain_s;
        if self.swell_s > 1e-4 {
            g *= 1.0 - self.swell_s * (0.5 - 0.5 * sin01(self.swell_ph));
        }
        let gl = g * (1.0 - self.pan).min(1.0);
        let gr = g * (1.0 + self.pan).min(1.0);
        for i in 0..n {
            let d = 1.0 - self.duck * sc[i];
            let l = self.buf[i][0] * gl * d;
            let r = self.buf[i][1] * gr * d;
            dry[i][0] += l;
            dry[i][1] += r;
            let m = (l + r) * 0.5;
            rs[i] += m * self.rev;
            ds[i] += m * self.dly;
            r2s[i] += m * self.rev2;
        }
    }
}

/// What the second reverb does: nothing, gate (synthwave's snare), rumble (techno's kick).
const R2_OFF: u8 = 0;
const R2_GATE: u8 = 1;
const R2_RUMBLE: u8 = 2;

struct Synth {
    ch: Vec<Channel>,
    rev: Reverb,
    rev_wet: f32,
    rev_s: f32,
    dly: Delay,
    dly_out: f32,
    dly_rev: f32,
    rev2: Reverb,
    r2mode: u8,
    r2gain: f32,
    gate: f32,
    gate_left: u32,
    gate_hold: u32,
    gate_rel: f32,
    r2lp: [Svf; 2],
    sc: f32,
    sc_att: u32,
    sc_rel: f32,
    fx: Fx,
    m: Master,
    noise: Noise,
    dry: [[f32; 2]; BLOCK],
    rs: [f32; BLOCK],
    ds: [f32; BLOCK],
    r2s: [f32; BLOCK],
    scb: [f32; BLOCK],
}

impl Synth {
    fn new(genre: usize) -> Synth {
        // voices per channel (tonal, drums)
        const POOL: [(usize, usize); NCH] =
            [(0, 2), (0, 4), (0, 4), (0, 4), (0, 2), (2, 0), (16, 0), (8, 0), (4, 0), (8, 0), (16, 0), (6, 0), (0, 0)];
        // the room: a mix's between its two genres'
        let room = |g: usize| match g {
            AMBIENT => 1.35,
            TECHNO => 1.0,
            SYNTHWAVE => 1.1,
            LOFI => 0.8,
            TAVERN => 0.65,
            SKA => 0.85,
            MARIACHI => 0.85,
            _ => 0.9,
        };
        let (writer, drummer) = parts(genre);
        let size = drummer.map_or(room(writer), |d| 0.5 * (room(writer) + room(d)));
        let techno = writer == TECHNO || drummer == Some(TECHNO);
        let mut r2lp = [Svf::default(); 2];
        for f in r2lp.iter_mut() {
            f.set(110.0, 0.1);
        }
        Synth {
            ch: POOL.iter().map(|&(v, d)| Channel::new(v, d)).collect(),
            rev: Reverb::new(size),
            rev_wet: 0.3,
            rev_s: 0.3,
            dly: Delay::new(),
            dly_out: 1.0,
            dly_rev: 0.2,
            rev2: Reverb::new(if techno { 1.3 } else { 0.7 }),
            r2mode: R2_OFF,
            r2gain: 1.0,
            gate: 0.0,
            gate_left: 0,
            gate_hold: 12000,
            gate_rel: t60(0.04),
            r2lp,
            sc: 0.0,
            sc_att: 0,
            sc_rel: (-1.0 / (0.12 * SR)).exp(),
            fx: Fx::new(),
            m: Master::new(),
            noise: Noise(0x9E37_79B9),
            dry: [[0.0; 2]; BLOCK],
            rs: [0.0; BLOCK],
            ds: [0.0; BLOCK],
            r2s: [0.0; BLOCK],
            scb: [0.0; BLOCK],
        }
    }

    fn fire(&mut self, e: Ev) {
        let seed = self.noise.seed();
        let bank = samples::bank();
        match e {
            Ev::Hit { ch, kind, vel } => {
                let ch = ch as usize;
                // the sidechain: the kick proper (not a frame drum's light
                // strokes, not a ghost)
                if ch == CH_KICK && kind == 0 && vel >= 0.5 {
                    self.sc_att = (0.003 * SR) as u32;
                }
                if ch == CH_SNARE && kind == 0 {
                    self.gate_left = self.gate_hold;
                }
                self.ch[ch].hit(ch == CH_HAT, kind, vel, seed, bank);
            }
            Ev::Note { ch, note, vel, len, slide } => {
                let c = &mut self.ch[ch as usize];
                if c.voices.is_empty() {
                    return;
                }
                let i = if c.mono {
                    0
                } else {
                    // a free voice, else the quietest releasing one, else the oldest
                    c.voices.iter().position(|v| !v.on).unwrap_or_else(|| {
                        let rel = (0..c.voices.len())
                            .filter(|&i| c.voices[i].amp.stage == RELEASE)
                            .min_by(|&a, &b| c.voices[a].amp.level.total_cmp(&c.voices[b].amp.level));
                        rel.unwrap_or_else(|| (0..c.voices.len()).max_by_key(|&i| c.voices[i].age).unwrap_or(0))
                    })
                };
                let p = c.patch;
                c.voices[i].start(&p, note, vel, (len * SR) as i64, slide, seed, bank, c.rr[0]);
                c.rr[0] = c.rr[0].wrapping_add(1);
            }
            Ev::Riser { secs, vel } => self.fx.riser(secs, vel),
        }
    }

    /// Renders up to BLOCK frames, interleaved stereo.
    fn render(&mut self, out: &mut [f32]) {
        let n = out.len() / 2;
        let sc_k = 1.0 - (-1.0 / (0.0015 * SR)).exp();
        for i in 0..n {
            if self.sc_att > 0 {
                self.sc += (1.0 - self.sc) * sc_k;
                self.sc_att -= 1;
            } else {
                self.sc *= self.sc_rel;
            }
            self.scb[i] = self.sc;
        }
        self.dry[..n].fill([0.0; 2]);
        self.rs[..n].fill(0.0);
        self.ds[..n].fill(0.0);
        self.r2s[..n].fill(0.0);
        let Synth { ch, fx, dry, rs, ds, r2s, scb, .. } = self;
        let bank = samples::bank();
        for (ci, c) in ch.iter_mut().enumerate() {
            let mut any = c.render(n, bank);
            if ci == CH_FX {
                any |= fx.render(&mut c.buf[..n]);
            }
            if any {
                c.mix(n, &scb[..n], &mut dry[..n], &mut rs[..n], &mut ds[..n], &mut r2s[..n]);
            }
        }
        self.rev_s += (self.rev_wet - self.rev_s) * 0.01;
        for i in 0..n {
            let (dl, dr) = self.dly.tick(self.ds[i]);
            let (rl, rr) = self.rev.tick(self.rs[i] + (dl + dr) * self.dly_rev);
            let mut l = self.dry[i][0] + dl * self.dly_out + rl * self.rev_s;
            let mut r = self.dry[i][1] + dr * self.dly_out + rr * self.rev_s;
            if self.r2mode != R2_OFF {
                let (a, b) = self.rev2.tick(self.r2s[i]);
                if self.r2mode == R2_GATE {
                    if self.gate_left > 0 {
                        self.gate_left -= 1;
                        self.gate += (1.0 - self.gate) * 0.02;
                    } else {
                        self.gate *= self.gate_rel;
                    }
                    l += a * self.gate * self.r2gain;
                    r += b * self.gate * self.r2gain;
                } else {
                    let m = self.r2lp[0].lp((a + b) * 0.5);
                    let m = self.r2lp[1].lp(m);
                    let m = soft(m * 2.0) * self.r2gain * (1.0 - 0.9 * self.scb[i]);
                    l += m;
                    r += m;
                }
            }
            let (l, r) = self.m.tick(l, r);
            out[2 * i] = l;
            out[2 * i + 1] = r;
        }
        self.m.block(n);
    }
}

// ---------------------------------------------------------------- music theory

const MODES: [[i32; 7]; 7] = [
    [0, 2, 4, 5, 7, 9, 11], // ionian
    [0, 2, 3, 5, 7, 9, 10], // dorian
    [0, 1, 3, 5, 7, 8, 10], // phrygian
    [0, 2, 4, 6, 7, 9, 11], // lydian
    [0, 2, 4, 5, 7, 9, 10], // mixolydian
    [0, 2, 3, 5, 7, 8, 10], // aeolian
    [0, 2, 3, 5, 7, 8, 11], // harmonic minor
];
const IONIAN: usize = 0;
const LYDIAN: usize = 3;
const MIXOLYDIAN: usize = 4;
const AEOLIAN: usize = 5;

/// Semitones above the tonic of scale degree `deg` (0: the tonic; 7: an octave up).
fn deg_semi(mode: usize, deg: i32) -> i32 {
    deg.div_euclid(7) * 12 + MODES[mode][deg.rem_euclid(7) as usize]
}

/// Chord progressions, eight bars each (a numeral a bar; A+B: two chords in
/// the bar), by genre and by the mode's colour (major: ionian, lydian,
/// mixolydian; minor: the rest). Borrowed chords, secondary dominants,
/// suspensions, sevenths and ninths; inversions (^1, ^2) for bass lines that move.
const AMB_MAJ: &[&str] = &[
    "Imaj7 Imaj7^1 IVmaj7 IVmaj7 vi9 V^1 IVmaj9 Vsus4",
    "Iadd9 V^1 vi7 iii7 IVmaj7 Iadd9^1 ii9 Vsus4",
    "I69 II vi9 IVmaj7 I69 II IVmaj9 Vsus4",
    "IVmaj7 V^1 vi9 Iadd9^1 IVmaj9 iii7 ii9 Vsus4",
    "Isus2 bVIImaj7 IVadd9 Isus2 vi9 IVmaj7 bVIImaj7 Vsus4",
    "Imaj7 vi9 IVmaj9 iv6 Iadd9 iii7 vi7 V7sus4",
    "vi9 IVmaj7 Imaj7 V^1 vi9 IVmaj7 bVIImaj7 Vsus4",
    "Imaj9 iii7 IVmaj7 V7/vi vi9 V^1 IVmaj7 iv6",
];
const AMB_MIN: &[&str] = &[
    "i9 IV9 i9 IV9 bVImaj7 bVIImaj7 i9 V7sus4",
    "i7 bVImaj7 bIIImaj7 bVII i7 iv9 bVImaj7 V7sus4",
    "i9 bVIImaj7 bVImaj7 bVIImaj7 i9 iv9 bIIImaj7 bVII",
    "iv9 bVII bIIImaj7 bVImaj7 iim7b5 V7sus4 i9 i9",
    "i9 i9^2 bVImaj7 bVImaj7 iv9 bVII bIIImaj7 V7sus4",
    "bVImaj7 bVII i9 v7 bVImaj7 bVII bIIImaj7 bVIIsus4",
];
const TECH_MIN: &[&str] = &[
    "i7 i7 i7 i7 bVI bVI bVII bVII",
    "i9 i9 iv9 iv9 i9 i9 bVIImaj7 bVIImaj7",
    "i7 i7 bIIImaj7 bIIImaj7 i7 i7 v7 v7",
    "i7 i7 i7 i7 i7 i7 i7 i7",
    "i7 bII i7 bII i7 bII bVII bVII",
    "i9 i9 i9 i9 bVImaj7 bVImaj7 iv7 iv7",
    "i11 i11 bVII bVII i11 i11 bVI bVII",
    "i7 i7 iv7 iv7 bVImaj7 bVImaj7 V7sus4 V7sus4",
];
const SW_MAJ: &[&str] = &[
    "I V vi IV I V^1 vi IV",
    "vi IV I V vi IV bVII V",
    "I iii IV V I iii bVII IV",
    "IV V iii vi ii V I I",
    "I V^1 vi I^2 IV I^1 ii V7",
    "vi V IV V vi V bVI bVII",
];
const SW_MIN: &[&str] = &[
    "i bVI bIII bVII i bVI iv V7",
    "bVI bVII i i bVI bVII bIII V7sus4+V7",
    "i v bVI bIII iv i^2 bVI+bVII i",
    "i bVII bVI bVII i bVII bVI+bVII V7",
    "iv bVII bIIImaj7 bVImaj7 iim7b5 V7 i i",
    "i bIII bVII iv bVI bIII iv+V7 i",
    "i bVII^1 bVI bVII i bVII bVI V7sus4",
    "bVI bIII bVII i bVI bIII V7/V V7",
];
const LOFI_MAJ: &[&str] = &[
    "ii9 V9 Imaj9 Imaj9 ii9 V9 Imaj9 V7/ii",
    "IVmaj7 iii7 vi9 V7/V ii9 V7 Imaj9 vi9",
    "Imaj7 V7/vi vi9 ii9+V7 IVmaj7 iv6 iii7 ii7+bII7",
    "Imaj9 I7 IVmaj7 iv6 iii7 VI7 ii9 V9",
    "IVmaj9 V9 iii7 vi9 ii7 V7 Imaj7 Imaj7",
    "ii9+V9 Imaj9+vi9 ii9+V9 Imaj9 IVmaj7+iv6 iii7+VI7 ii9+V9 Imaj9",
    "Imaj7 IVmaj7 iii7 vi7 ii7 V7 Imaj7 bVIImaj7",
];
const LOFI_MIN: &[&str] = &[
    "i9 iv9 bVIImaj7 bIIImaj7 bVImaj7 iim7b5 V7 i9",
    "i9 IV9 i9 IV9 bVImaj7 bVII9 i9 i9",
    "i9 bVImaj7 iim7b5+V7 i9 iv9 bVII9 bIIImaj7 V7",
    "i7 iv7 bVIImaj7 bIIImaj7 iv7 V7 i7 bVI7",
    "iv9 v7 i9 i9 bVImaj7 bVII7 i9 V7sus4+V7",
];

/// The genre's pool for the mode.
fn pool(genre: usize, mode: usize) -> &'static [&'static str] {
    let major = matches!(mode, IONIAN | LYDIAN | MIXOLYDIAN);
    match (genre, major) {
        (AMBIENT, true) => AMB_MAJ,
        (AMBIENT, false) => AMB_MIN,
        (TECHNO, _) => TECH_MIN,
        (SYNTHWAVE, true) => SW_MAJ,
        (SYNTHWAVE, false) => SW_MIN,
        (_, true) => LOFI_MAJ,
        (_, false) => LOFI_MIN,
    }
}

/// A progression from the pools, a chord per half bar.
fn halves(plan: &str) -> Vec<Harm> {
    plan.split_whitespace()
        .flat_map(|t| {
            let v = harms(&t.replace('+', " "));
            match v.len() {
                1 => [v[0], v[0]],
                _ => [v[0], v[1]],
            }
        })
        .collect()
}

/// A voicing of the pitch classes `pcs` (absolute) within [lo, hi], moving as
/// little as it can from `prev` (close position, or drop-2 when `open`).
fn voicing_pcs(pcs: &[i32], lo: i32, hi: i32, prev: &[i32], open: bool) -> Vec<i32> {
    let k = pcs.len();
    let mean = |v: &[i32]| v.iter().sum::<i32>() as f32 / v.len().max(1) as f32;
    let centre = if prev.is_empty() { (lo + hi) as f32 / 2.0 } else { mean(prev) };
    let mut best: Option<(f32, Vec<i32>)> = None;
    for inv in 0..k {
        let mut v = Vec::with_capacity(k);
        let mut last = lo - 1;
        for j in 0..k {
            let pc = pcs[(inv + j) % k];
            let n = last + 1 + (pc - (last + 1)).rem_euclid(12);
            v.push(n);
            last = n;
        }
        if open && k >= 4 {
            v[k - 2] -= 12;
            v.sort();
            if v[0] < lo - 5 {
                v.iter_mut().for_each(|x| *x += 12);
            }
        }
        for shift in [0, 12, -12] {
            let w: Vec<i32> = v.iter().map(|x| x + shift).collect();
            if w[0] < lo - 5 || w[k - 1] > hi {
                continue;
            }
            let cost = if prev.len() == k {
                w.iter().zip(prev).map(|(a, b)| (a - b).abs() as f32).sum::<f32>()
            } else {
                (mean(&w) - centre).abs() * k as f32
            };
            if best.as_ref().is_none_or(|(c0, _)| cost < *c0) {
                best = Some((cost, w));
            }
        }
    }
    best.map(|b| b.1).unwrap_or_else(|| pcs.iter().map(|&pc| lo + (pc - lo).rem_euclid(12)).collect())
}

// -- chords anywhere in the chromatic scale

/// A chord: its root (semitones above the tonic), its quality, and the chord
/// tone its bass plays (0 the root, 1 the third, 2 the fifth, 3 the seventh).
#[derive(Clone, Copy, PartialEq, Debug)]
struct Harm {
    root: i32,
    q: u8,
    inv: u8,
}

const Q_MAJ: u8 = 0;
const Q_MIN: u8 = 1;
const Q_DIM: u8 = 2;
const Q_AUG: u8 = 3;
const Q_MAJ7: u8 = 4;
const Q_MIN7: u8 = 5;
const Q_DOM7: u8 = 6;
const Q_HDIM7: u8 = 7;
const Q_DIM7: u8 = 8;
const Q_SUS2: u8 = 9;
const Q_SUS4: u8 = 10;
const Q_ADD9: u8 = 11;
const Q_MADD9: u8 = 12;
const Q_MAJ9: u8 = 13;
const Q_MIN9: u8 = 14;
const Q_DOM9: u8 = 15;
const Q_7SUS4: u8 = 16;
const Q_SIX: u8 = 17;
const Q_MIN6: u8 = 18;
const Q_SIX9: u8 = 19;
const Q_MIN11: u8 = 20;

/// A quality's intervals above the root.
fn q_iv(q: u8) -> &'static [i32] {
    match q {
        Q_MIN => &[0, 3, 7],
        Q_DIM => &[0, 3, 6],
        Q_AUG => &[0, 4, 8],
        Q_MAJ7 => &[0, 4, 7, 11],
        Q_MIN7 => &[0, 3, 7, 10],
        Q_DOM7 => &[0, 4, 7, 10],
        Q_HDIM7 => &[0, 3, 6, 10],
        Q_DIM7 => &[0, 3, 6, 9],
        Q_SUS2 => &[0, 2, 7],
        Q_SUS4 => &[0, 5, 7],
        Q_ADD9 => &[0, 4, 7, 14],
        Q_MADD9 => &[0, 3, 7, 14],
        Q_MAJ9 => &[0, 4, 7, 11, 14],
        Q_MIN9 => &[0, 3, 7, 10, 14],
        Q_DOM9 => &[0, 4, 7, 10, 14],
        Q_7SUS4 => &[0, 5, 7, 10],
        Q_SIX => &[0, 4, 7, 9],
        Q_MIN6 => &[0, 3, 7, 9],
        Q_SIX9 => &[0, 4, 7, 9, 14],
        Q_MIN11 => &[0, 3, 7, 10, 14, 17],
        _ => &[0, 4, 7],
    }
}

/// A Roman numeral, relative to the tonic's major scale: semitones, whether
/// it is lower case (minor), and what follows it.
fn numeral(s: &str) -> Option<(i32, bool, &str)> {
    let (acc, s) = match s.as_bytes().first() {
        Some(b'b') => (-1, &s[1..]),
        Some(b'#') => (1, &s[1..]),
        _ => (0, s),
    };
    let n = s.chars().take_while(|c| matches!(c, 'I' | 'V' | 'i' | 'v')).count();
    let (num, rest) = s.split_at(n);
    let semi = match num.to_ascii_uppercase().as_str() {
        "I" => 0,
        "II" => 2,
        "III" => 4,
        "IV" => 5,
        "V" => 7,
        "VI" => 9,
        "VII" => 11,
        _ => return None,
    };
    Some((semi + acc, num.starts_with(['i', 'v']), rest))
}

/// A chord from its Roman numeral: upper case major, lower case minor; b or #
/// before; after it o, +, 7, maj7, m7b5, o7, 9, maj9, add9, sus2, sus4, 7sus4,
/// 6, 69, 11; /X for a secondary chord (V7/V); ^1 ^2 ^3 for the bass on the
/// third, fifth or seventh. None if it does not parse.
fn parse_harm(tok: &str) -> Option<Harm> {
    let (tok, inv) = match tok.split_once('^') {
        Some((a, b)) => (a, b.parse().ok()?),
        None => (tok, 0),
    };
    let (main, of) = match tok.split_once('/') {
        Some((a, b)) => (a, Some(numeral(b)?.0)),
        None => (tok, None),
    };
    let (root, minor, rest) = numeral(main)?;
    let q = match (minor, rest) {
        (false, "") => Q_MAJ,
        (true, "") => Q_MIN,
        (_, "o") => Q_DIM,
        (_, "+") => Q_AUG,
        (false, "7") => Q_DOM7,
        (true, "7") => Q_MIN7,
        (_, "maj7") => Q_MAJ7,
        (_, "m7b5") => Q_HDIM7,
        (_, "o7") => Q_DIM7,
        (false, "9") => Q_DOM9,
        (true, "9") => Q_MIN9,
        (_, "maj9") => Q_MAJ9,
        (false, "add9") => Q_ADD9,
        (true, "add9") => Q_MADD9,
        (_, "sus2") => Q_SUS2,
        (_, "sus4") => Q_SUS4,
        (_, "7sus4") => Q_7SUS4,
        (false, "6") => Q_SIX,
        (true, "6") => Q_MIN6,
        (_, "69") => Q_SIX9,
        (_, "11") => Q_MIN11,
        _ => return None,
    };
    if inv as usize >= q_iv(q).len().min(4) {
        return None;
    }
    Some(Harm { root: (root + of.unwrap_or(0)).rem_euclid(12), q, inv })
}

/// A progression from its numerals (the pools are tested to parse).
fn harms(s: &str) -> Vec<Harm> {
    s.split_whitespace().map(|t| parse_harm(t).unwrap_or(Harm { root: 0, q: Q_MAJ, inv: 0 })).collect()
}

impl Harm {
    /// Its pitch classes, above the tonic.
    fn pcs(&self) -> Vec<i32> {
        q_iv(self.q).iter().map(|i| (self.root + i).rem_euclid(12)).collect()
    }
    /// The pitch class its bass plays.
    fn bass(&self) -> i32 {
        let iv = q_iv(self.q);
        (self.root + iv[(self.inv as usize).min(iv.len() - 1)]).rem_euclid(12)
    }
    fn mask(&self, tonic: i32) -> u16 {
        self.pcs().iter().fold(0, |m, &p| m | 1 << (p + tonic).rem_euclid(12))
    }
}

/// The scale a line runs through over chord `h`: the mode's, each chord tone
/// it lacks put in place of the nearest scale tone that is not a chord tone.
fn chord_scale(mode: usize, h: Harm) -> Vec<i32> {
    let ct = h.pcs();
    let mut sc = MODES[mode].to_vec();
    let dist = |a: i32, b: i32| (a - b).rem_euclid(12).min((b - a).rem_euclid(12));
    for &t in &ct {
        if !sc.contains(&t)
            && let Some(i) = (0..sc.len()).filter(|&i| !ct.contains(&sc[i])).min_by_key(|&i| dist(sc[i], t))
        {
            sc[i] = t;
        }
    }
    sc.sort();
    sc.dedup();
    sc
}

/// The pitches in [lo, hi] (semitones above a tonic) of the pitch classes `pcs`.
fn in_range(pcs: &[i32], lo: i32, hi: i32) -> Vec<i32> {
    (lo..=hi).filter(|p| pcs.contains(&p.rem_euclid(12))).collect()
}

/// The index of the entry of `v` nearest `p`.
fn nearest(v: &[i32], p: i32) -> usize {
    (0..v.len()).min_by_key(|&i| (v[i] - p).abs()).unwrap_or(0)
}

/// Euclidean rhythm: `k` hits spread over `n` steps, rotated.
fn euclid(k: usize, n: usize, rot: usize) -> Vec<bool> {
    (0..n).map(|i| ((i + rot) * k) % n < k).collect()
}

/// A note of a melody: a 16th-step, a length in steps, a scale degree.
#[derive(Clone, Copy)]
struct MNote {
    step: u16,
    len: u16,
    deg: i32,
    vel: f32,
}

/// Where a 16th sits in the bar, as a weight (downbeat 1).
const METRIC: [f32; 16] = [1.0, 0.12, 0.4, 0.15, 0.7, 0.12, 0.45, 0.2, 0.85, 0.12, 0.4, 0.15, 0.7, 0.15, 0.45, 0.2];

/// A motif cell of `steps` 16ths: a rhythm drawn by metric weight and a
/// contour that mostly steps, sometimes skips, and turns back after a leap.
/// A cadence ends long, on the tonic.
fn make_cell(r: &mut Rng, steps: u16, density: f32, lo: i32, hi: i32, cadence: bool) -> Vec<MNote> {
    let mut on: Vec<u16> =
        (0..steps).filter(|&i| r.chance((METRIC[i as usize % 16] * density * 1.5).min(0.95))).collect();
    if on.first().is_none_or(|&s| s > 4) {
        on.insert(0, r.pick(&[0, 0, 2]));
    }
    if cadence {
        let end = steps.saturating_sub(8);
        on.retain(|&s| s < end);
        on.push(end.max(on.last().map_or(0, |&s| s + 1)));
    }
    let mut deg = r.pick(&[0, 2, 4, 7]).clamp(lo, hi);
    let mut last = 0;
    let mut notes = Vec::with_capacity(on.len());
    for (k, &st) in on.iter().enumerate() {
        if k > 0 {
            let mut iv = match r.weighted(&[0.12, 0.5, 0.26, 0.12]) {
                0 => 0,
                1 => 1,
                2 => 2,
                _ => r.pick(&[3, 4, 5, 7]),
            };
            let centre = (lo + hi) as f32 / 2.0;
            let up = if last >= 3 {
                false
            } else if last <= -3 {
                true
            } else {
                r.chance((0.5 + 0.07 * (centre - deg as f32)).clamp(0.1, 0.9))
            };
            if i32::abs(last) >= 3 {
                iv = iv.min(2);
            }
            if !up {
                iv = -iv;
            }
            deg = (deg + iv).clamp(lo, hi);
            last = iv;
        }
        let next = on.get(k + 1).copied().unwrap_or(steps);
        let mut len = (next - st).max(1);
        if len > 2 && r.chance(0.3) {
            len /= 2;
        }
        let vel = (0.72 + 0.25 * METRIC[st as usize % 16] + r.range(-0.06, 0.06)).min(1.0);
        notes.push(MNote { step: st, len, deg, vel });
    }
    if cadence && let Some(n) = notes.last_mut() {
        let mut t = (n.deg as f32 / 7.0).round() as i32 * 7;
        if t > hi {
            t -= 7;
        }
        if t < lo {
            t += 7;
        }
        n.deg = t;
        n.len = steps - n.step;
    }
    notes
}

/// A cell moved by `by` scale degrees.
fn moved(v: &[MNote], by: i32, lo: i32, hi: i32) -> Vec<MNote> {
    v.iter().map(|n| MNote { deg: (n.deg + by).clamp(lo, hi), ..*n }).collect()
}

/// A sixteen-bar melody, developed from a two-bar motif: the motif (a), its
/// sequence a step or a third away with a new ending (a'), a contrasting
/// cell (b), a half cadence (c); then the motif inverted (a''), the
/// contrast moved (b'), a new cell (d) and the cadence home (e).
fn make_melody(r: &mut Rng, density: f32, lo: i32, hi: i32) -> Vec<MNote> {
    let a = make_cell(r, 32, density, lo, hi, false);
    let mut a2 = moved(&a, r.pick(&[1, -1, 2, -2, 3]), lo, hi);
    let k = a2.len();
    for n in a2[k.saturating_sub(2)..].iter_mut() {
        n.deg = (n.deg + r.pick(&[-2, -1, 1, 2])).clamp(lo, hi);
    }
    let db = (density * r.range(0.7, 1.4)).min(0.95);
    let b = make_cell(r, 32, db, lo, hi, false);
    let mut c = make_cell(r, 32, density * 0.8, lo, hi, true);
    if let Some(n) = c.last_mut() {
        // half way: rest on the fifth or the second, not home
        n.deg = (n.deg + r.pick(&[4, 1, -3])).clamp(lo, hi);
    }
    let first = a.first().map_or(0, |n| n.deg);
    let a3: Vec<MNote> = a.iter().map(|n| MNote { deg: (2 * first - n.deg).clamp(lo, hi), ..*n }).collect();
    let b2 = moved(&b, r.pick(&[2, -2, 1]), lo, hi);
    let d = make_cell(r, 32, density, lo, hi, false);
    let e = make_cell(r, 32, density * 0.7, lo, hi, true);
    let mut v = Vec::new();
    for (k, cell) in [&a, &a2, &b, &c, &a3, &b2, &d, &e].iter().enumerate() {
        v.extend(cell.iter().map(|n| MNote { step: n.step + 32 * k as u16, ..*n }));
    }
    v
}

/// An arpeggio's order: indices into the chord's tones over two octaves.
fn make_arp(r: &mut Rng) -> Vec<u8> {
    match r.below(6) {
        0 => vec![0, 1, 2, 3, 4, 5, 6, 7],
        1 => vec![7, 6, 5, 4, 3, 2, 1, 0],
        2 => vec![0, 1, 2, 3, 4, 3, 2, 1],
        3 => vec![0, 2, 1, 3, 2, 4, 3, 5],
        4 => vec![0, 3, 1, 4, 2, 5, 3, 6],
        _ => (0..8).map(|_| r.below(7) as u8).collect(),
    }
}

// ---------------------------------------------------------------- the composer

const L_KICK: u32 = 1;
const L_SNARE: u32 = 1 << 1;
const L_HAT: u32 = 1 << 2;
const L_OHAT: u32 = 1 << 3;
const L_PERC: u32 = 1 << 4;
const L_BASS: u32 = 1 << 5;
const L_PAD: u32 = 1 << 6;
const L_ARP: u32 = 1 << 7;
const L_LEAD: u32 = 1 << 8;
const L_BELL: u32 = 1 << 9;
const L_KEYS: u32 = 1 << 10;
const L_DRONE: u32 = 1 << 11;
const L_TEX: u32 = 1 << 12;
const L_RUMBLE: u32 = 1 << 13;
/// a second voice: answering the melody, or walking the chords' guide tones
const L_CTR: u32 = 1 << 14;

const INTRO: u8 = 0;
const BUILD: u8 = 1;
const MAIN: u8 = 2;
const BREAK: u8 = 3;
const OUTRO: u8 = 4;

#[derive(Clone, Copy)]
struct Sec {
    kind: u8,
    bars: u32,
    mask: u32,
    energy: f32,
}

#[derive(Clone, Copy, Default)]
struct Step {
    on: bool,
    /// semitones above the root
    note: i32,
    accent: bool,
    slide: bool,
}

/// One bar being written: event times from 16th steps (swing delays the odd ones).
struct Bar<'a> {
    t0: f64,
    step: f64,
    swing: f32,
    /// in 6/8 and 9/8, how late a beat's second and third eighths come (16ths)
    lilt: f32,
    /// semitones added to every note
    lift: i32,
    ev: &'a mut Vec<(u64, Ev)>,
}

impl Bar<'_> {
    fn at(&self, step: f32, late: f32) -> u64 {
        let whole = step.fract() == 0.0;
        let odd = whole && (step as i32) % 2 == 1;
        // a compound beat's second eighth comes late, its third half as
        // late, and the 16ths between them in proportion (a beat's grid
        // bends; it does not stumble)
        let lilt = if self.lilt == 0.0 {
            0.0
        } else {
            let x = step.rem_euclid(6.0);
            self.lilt * if x < 2.0 { 0.5 * x } else { 1.0 - 0.25 * (x - 2.0) }
        };
        let st = step + if odd { self.swing } else { 0.0 } + lilt;
        (self.t0 + st as f64 * self.step + late as f64 * SR as f64).round().max(0.0) as u64
    }
    fn hit(&mut self, step: f32, late: f32, ch: usize, kind: u8, vel: f32) {
        let t = self.at(step, late);
        self.ev.push((t, Ev::Hit { ch: ch as u8, kind, vel: vel.clamp(0.0, 1.0) }));
    }
    fn note(&mut self, step: f32, late: f32, ch: usize, note: i32, vel: f32, steps: f32, slide: bool) {
        let t = self.at(step, late);
        let len = self.secs(steps);
        let note = (note + self.lift) as f32;
        self.ev.push((t, Ev::Note { ch: ch as u8, note, vel: vel.clamp(0.0, 1.0), len, slide }));
    }
    fn secs(&self, steps: f32) -> f32 {
        (steps as f64 * self.step) as f32 / SR
    }
}

struct Song {
    genre: usize,
    rng: Rng,
    piece: u32,
    bpm: f64,
    /// how late the odd 16ths come, in 16ths
    swing: f32,
    /// the most a hit comes late, seconds
    human: f32,
    /// the tonic's pitch class, and the mode
    root: i32,
    mode: usize,
    /// a chord per half bar, looping: the section's, drawn from the piece's two
    prog: Vec<Harm>,
    progs: Vec<Vec<Harm>>,
    /// each section's key (semitones from the piece's tonic), and a pivot
    /// chord (a half bar's index, the new key's dominant) leading into a change
    lifts: Vec<i32>,
    pivot: Option<(u32, Harm)>,
    /// the melody's bar (of sixteen) this bar, and the counter-melody's last note
    mel_bar: u32,
    mel_pb: u32,
    guide: i32,
    /// semitones the whole piece is lifted (synthwave's last chorus)
    lift: i32,
    /// the melody's range, scale degrees about the tonic
    prange: (i32, i32),
    plan: Vec<Sec>,
    sec: usize,
    sec_bar: u32,
    kick: [f32; 16],
    snare: [f32; 16],
    hat: [f32; 16],
    ohat: [f32; 16],
    perc: [f32; 16],
    perc_kind: u8,
    stab: [f32; 16],
    bass: [Step; 16],
    bass_style: u8,
    arp: Vec<u8>,
    arp_rate: usize,
    phrase: Vec<MNote>,
    /// the phrase's transposition (scale degrees) this time round
    ptrans: i32,
    lead_center: i32,
    pad_v: Vec<i32>,
    keys_v: Vec<i32>,
    walk: [f32; 4],
    phase: f32,
    period: f32,
    /// the patterns' B variants, which 4-bar groups of the section play
    /// them (bit k: group k), and the ambient pad's pulse (a beat's divisions)
    kick2: [f32; 16],
    snare2: [f32; 16],
    hat2: [f32; 16],
    perc2: [f32; 16],
    bass2: [Step; 16],
    arp2: Vec<u8>,
    scheme: u8,
    pulse: f32,
    bass_style2: u8,
    stab_shift: usize,
    /// techno's hook: a bar of melody
    hook: Vec<MNote>,
    /// the piece's instruments, and whether ambient's texture is rain (or wind)
    band: Band,
    rain: bool,
    /// an optional layer sitting out this phrase
    mute: u32,
    anticipated: bool,
    bell_motif: bool,
    snare_kind: u8,
    /// each piece's tempo, tonic and mode, for --musicwav's summary
    log: Vec<(f64, i32, usize, &'static str)>,
    pk: Polka,
    tv: Tavern,
    sk: Ska,
    mc: Mariachi,
    /// a mix's drummer (the genre `genre` writes for), a synthesiser its
    /// drums are made on, and its patterns
    guest: Option<usize>,
    scratch: Option<Box<Synth>>,
    dr: Drummer,
}

impl Song {
    fn new(genre: usize, seed: u64, s: &mut Synth) -> Song {
        let (writer, guest) = parts(genre);
        let mut song = Song {
            genre: writer,
            rng: Rng::new(seed ^ (genre as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03)),
            piece: 0,
            bpm: 120.0,
            swing: 0.0,
            human: 0.0,
            root: 0,
            mode: AEOLIAN,
            prog: vec![Harm { root: 0, q: Q_MAJ, inv: 0 }],
            progs: vec![vec![Harm { root: 0, q: Q_MAJ, inv: 0 }]],
            lifts: Vec::new(),
            pivot: None,
            mel_bar: 0,
            mel_pb: 0,
            guide: 0,
            lift: 0,
            prange: (-2, 9),
            plan: Vec::new(),
            sec: 0,
            sec_bar: 0,
            kick: [0.0; 16],
            snare: [0.0; 16],
            hat: [0.0; 16],
            ohat: [0.0; 16],
            perc: [0.0; 16],
            perc_kind: RIM,
            stab: [0.0; 16],
            bass: [Step::default(); 16],
            bass_style: 0,
            arp: vec![0],
            arp_rate: 2,
            phrase: Vec::new(),
            ptrans: 0,
            lead_center: 72,
            pad_v: Vec::new(),
            keys_v: Vec::new(),
            walk: [0.0; 4],
            phase: 0.0,
            period: 32.0,
            kick2: [0.0; 16],
            snare2: [0.0; 16],
            hat2: [0.0; 16],
            perc2: [0.0; 16],
            bass2: [Step::default(); 16],
            arp2: vec![0],
            scheme: 0,
            pulse: 1.0,
            bass_style2: 0,
            stab_shift: 0,
            hook: Vec::new(),
            band: Band::default(),
            rain: false,
            mute: 0,
            anticipated: false,
            bell_motif: false,
            snare_kind: 0,
            log: Vec::new(),
            pk: Polka::default(),
            tv: Tavern::default(),
            sk: Ska::default(),
            mc: Mariachi::default(),
            guest,
            scratch: guest.map(|g| Box::new(Synth::new(g))),
            dr: Drummer::default(),
        };
        song.new_piece(s);
        song.section_start(s);
        song
    }

    /// 16ths a bar: 2/4 for polka, 4/4 for the rest.
    fn bar_steps(&self) -> u32 {
        match self.genre {
            POLKA => 8,
            TAVERN => self.tv.bar_steps.max(1),
            SKA => self.sk.bar_steps.max(1),
            MARIACHI => self.mc.bar_steps.max(1),
            _ => 16,
        }
    }

    /// 16ths a beat: 6 in 6/8 and 9/8 (the beat a dotted quarter), else 4.
    fn beat_steps(&self) -> u32 {
        match self.genre {
            TAVERN => self.tv.beat_steps.max(1),
            SKA => self.sk.beat_steps.max(1),
            MARIACHI => self.mc.beat_steps.max(1),
            _ => 4,
        }
    }

    /// A 16th, in samples.
    fn step_len(&self) -> f64 {
        60.0 / self.bpm * SR as f64 / self.beat_steps() as f64
    }

    /// The tonic nearest `center` (MIDI), in [center - 6, center + 6).
    fn tonic_near(&self, center: i32) -> i32 {
        center - 6 + (self.root - (center - 6)).rem_euclid(12)
    }

    fn semi(&self, deg: i32) -> i32 {
        deg_semi(self.mode, deg)
    }

    fn chord(&self, half: u32) -> Harm {
        match self.pivot {
            Some((h, c)) if h == half => c,
            _ => self.prog[half as usize % self.prog.len()],
        }
    }

    /// The chords of the bar just written, as pitch-class sets (bit k: pitch class k).
    fn bar_chords(&self) -> Vec<u16> {
        if self.genre == POLKA {
            return vec![self.pk.mask];
        }
        if self.genre == TAVERN {
            return vec![self.tv.mask];
        }
        if self.genre == MARIACHI {
            return self.mc.mask.to_vec();
        }
        let bar = self.sec_bar.saturating_sub(1);
        (0..2).map(|h| self.chord(bar * 2 + h).mask(self.root + self.lift)).collect()
    }

    /// The chord's bass (its root, or the chord tone an inversion puts there) in [lo, lo + 12).
    fn bass_note(&self, c: Harm, lo: i32) -> i32 {
        lo + (self.root + c.bass() - lo).rem_euclid(12)
    }

    /// The chord's pitch classes (absolute).
    fn pcs(&self, c: Harm) -> Vec<i32> {
        let mut v: Vec<i32> = c.pcs().iter().map(|p| (self.root + p).rem_euclid(12)).collect();
        v.dedup();
        v
    }

    /// A melody note: scale degree `deg` (of the mode), on a strong beat or
    /// long moved to the nearest tone of chord `c`, otherwise to the nearest
    /// tone of its chord-scale. Semitones above the tonic.
    fn mel_pitch(&self, deg: i32, c: Harm, strong: bool) -> i32 {
        let p = self.semi(deg);
        let set = if strong { c.pcs() } else { chord_scale(self.mode, c) };
        (0..7).flat_map(|k| [p - k, p + k]).find(|q| set.contains(&q.rem_euclid(12))).unwrap_or(p)
    }

    /// A voicing of `c` within [lo, hi], moving as little as it can from
    /// `prev` (close position, or drop-2 when `open`); an eleventh chord
    /// leaves its fifth out.
    fn voicing(&self, c: Harm, lo: i32, hi: i32, prev: &[i32], open: bool) -> Vec<i32> {
        let mut pcs = self.pcs(c);
        if pcs.len() > 5 {
            pcs.remove(2);
        }
        voicing_pcs(&pcs, lo, hi, prev, open)
    }

    /// Chord tones from `lo` upward over two octaves.
    fn arp_tones(&self, c: Harm, lo: i32) -> Vec<i32> {
        let mut v: Vec<i32> = self
            .pcs(c)
            .iter()
            .flat_map(|&pc| {
                let n = lo + (pc - lo).rem_euclid(12);
                [n, n + 12]
            })
            .collect();
        v.sort();
        v.dedup();
        v
    }

    /// A progression from the genre's pool: eight bars, or sixteen with each
    /// chord held two (ambient, sometimes).
    fn draw_prog(&mut self) -> Vec<Harm> {
        let r = &mut self.rng;
        let v = halves(r.pick(pool(self.genre, self.mode)));
        if self.genre == AMBIENT && r.chance(0.35) {
            v.chunks(2).flat_map(|c| [c[0], c[1], c[0], c[1]]).collect()
        } else {
            v
        }
    }

    /// The piece's progressions: A (verses), B (choruses), C (bridges and
    /// breaks), each another than the last when the pool allows.
    fn draw_progs(&mut self) {
        let mut v: Vec<Vec<Harm>> = Vec::new();
        for _ in 0..3 {
            let mut p = self.draw_prog();
            for _ in 0..6 {
                if !v.contains(&p) {
                    break;
                }
                p = self.draw_prog();
            }
            v.push(p);
        }
        self.prog = v[0].clone();
        self.progs = v;
        self.plan_keys();
    }

    /// Each section's key: home, but two pieces in five move a step, a third
    /// or a fourth away from a main section on (home again after two sections
    /// half the time); synthwave's last chorus may lift too.
    fn plan_keys(&mut self) {
        let n = self.plan.len();
        self.lifts = vec![0; n];
        let r = &mut self.rng;
        let mains: Vec<usize> = (2..n.saturating_sub(1)).filter(|&k| self.plan[k].kind == MAIN).collect();
        if !mains.is_empty() && r.chance(0.4) {
            let k = mains[r.below(mains.len())];
            let m = r.pick(&[2, -2, 5, -5, 3, -3]);
            let end = if r.chance(0.5) { (k + 2).min(n - 1) } else { n };
            for l in self.lifts[k..end].iter_mut() {
                *l = m;
            }
        }
        if self.genre == SYNTHWAVE && r.chance(0.5) {
            if let Some(k) = self.plan.iter().rposition(|x| x.kind == MAIN) {
                let up = r.pick(&[1, 2, 2]);
                for l in self.lifts[k..].iter_mut() {
                    *l += up;
                }
            }
        }
    }

    /// The next bar: its events into `ev`, starting at sample `t0`; its length in samples.
    fn bar(&mut self, s: &mut Synth, t0: f64, ev: &mut Vec<(u64, Ev)>) -> f64 {
        if self.sec_bar >= self.plan[self.sec].bars {
            self.sec_bar = 0;
            self.sec += 1;
            if self.sec >= self.plan.len() {
                self.new_piece(s);
            }
            self.section_start(s);
        }
        let sec = self.plan[self.sec];
        // the last bar before a key change ends on the new key's dominant
        if self.genre != POLKA && self.sec_bar + 1 == sec.bars {
            let next = self.lifts.get(self.sec + 1).copied();
            if let Some(l) = next.filter(|&l| l != self.lift) {
                let root = (l - self.lift + 7).rem_euclid(12);
                self.pivot = Some((self.sec_bar * 2 + 1, Harm { root, q: Q_DOM7, inv: 0 }));
            }
        }
        let len = self.bar_steps() as f64 * self.step_len();
        // (a jig's; mariachi's 6/8 is even)
        let lilt = if self.beat_steps() == 6 && self.genre != MARIACHI { 0.18 } else { 0.0 };
        let mut b = Bar { t0, step: self.step_len(), swing: self.swing, lilt, lift: self.lift, ev };
        if self.sec_bar.is_multiple_of(8) && self.sec_bar > 0 && sec.kind == MAIN {
            // now and then an optional layer sits out a phrase
            let opt: &[u32] = match self.genre {
                AMBIENT => &[L_BELL, L_DRONE, L_TEX, L_HAT, L_KICK],
                TECHNO => &[L_OHAT, L_PERC, L_KEYS, L_SNARE],
                SYNTHWAVE => &[L_ARP, L_OHAT, L_PAD, L_PERC],
                LOFI => &[L_LEAD, L_PERC, L_HAT],
                SKA => &[L_DRONE, L_PAD, L_PERC, L_OHAT],
                MARIACHI => &[L_BELL, L_DRONE, L_PAD, L_HAT],
                _ => &[L_HAT],
            };
            self.mute = if self.rng.chance(0.5) { self.rng.pick(opt) } else { 0 };
        }
        let m = sec.mask & !self.mute;
        for w in self.walk.iter_mut() {
            *w = walk(*w, &mut self.rng, 0.12);
        }
        let n0 = b.ev.len();
        match self.genre {
            AMBIENT => self.ambient(s, &mut b, sec, m),
            TECHNO => self.techno(s, &mut b, sec, m),
            SYNTHWAVE => self.synthwave(s, &mut b, sec, m),
            LOFI => self.lofi(s, &mut b, sec, m),
            TAVERN => self.tavern(s, &mut b, sec, m),
            SKA => self.ska(s, &mut b, sec, m),
            MARIACHI => self.mariachi(s, &mut b, sec, m),
            _ => self.polka(&mut b, sec),
        }
        // a mix: the drummer's beat for the writer's (but for polka's,
        // tavern's, ska's and mariachi's last bar, their ending)
        let ending = matches!(self.genre, POLKA | TAVERN | SKA | MARIACHI) && sec.kind == OUTRO && self.sec_bar + 1 == sec.bars;
        if self.guest.is_some() && !ending {
            let mut k = n0;
            for j in n0..b.ev.len() {
                if !matches!(b.ev[j].1, Ev::Hit { .. }) {
                    b.ev.swap(k, j);
                    k += 1;
                }
            }
            b.ev.truncate(k);
            self.drum(s, &mut b, sec, m);
        }
        // the arc: an intro opens the melodic parts' filters, an outro closes them
        let x = (self.sec_bar as f32 + 1.0) / sec.bars as f32;
        let sweep = match sec.kind {
            _ if self.genre >= POLKA => 20000.0,
            INTRO => 600.0 * (x * 5.05).exp2(),
            OUTRO => 600.0 * ((1.0 - x) * 3.5 + 1.5).exp2(),
            _ => 20000.0,
        };
        for c in [CH_PAD, CH_ARP, CH_KEYS, CH_LEAD, CH_BELL, CH_BASS] {
            s.ch[c].lp = s.ch[c].lp_base.min(sweep);
        }
        self.sec_bar += 1;
        len
    }

    fn new_piece(&mut self, s: &mut Synth) {
        self.piece += 1;
        let r = &mut self.rng;
        // the key: at first anywhere, then usually a neighbour on the circle of fifths
        self.root = if self.piece == 1 {
            r.below(12) as i32
        } else {
            (self.root + r.pick(&[0, 5, 7, 7, 5, 2, 10, 3, 9])).rem_euclid(12)
        };
        let w: [f32; 7] = match self.genre {
            AMBIENT => [2.0, 2.0, 0.0, 3.0, 1.0, 2.0, 0.0],
            TECHNO => [0.0, 2.0, 2.0, 0.0, 0.0, 3.0, 0.5],
            SYNTHWAVE => [1.5, 2.0, 0.0, 0.0, 0.0, 5.0, 1.5],
            LOFI => [3.0, 3.0, 0.0, 0.5, 1.0, 1.0, 0.0],
            TAVERN => [2.0, 3.0, 0.0, 0.0, 3.0, 2.0, 0.0],
            SKA => [5.0, 1.0, 0.0, 0.0, 1.5, 1.5, 0.5],
            // mariachi: major mostly, minor ones in harmonic minor
            MARIACHI => [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4],
            // polka: major mostly, now and then a minor one
            _ => [4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        };
        self.mode = r.weighted(&w);
        self.sec = 0;
        self.sec_bar = 0;
        self.ptrans = 0;
        self.lift = 0;
        self.pad_v.clear();
        self.keys_v.clear();
        if let (Some(g), Some(x)) = (self.guest, self.scratch.as_deref_mut()) {
            // a mix's drums: tuned and levelled as the drummer's own pieces have them
            sounds(g, &mut self.rng, x);
        }
        match self.genre {
            AMBIENT => self.piece_ambient(s),
            TECHNO => self.piece_techno(s),
            SYNTHWAVE => self.piece_synthwave(s),
            LOFI => self.piece_lofi(s),
            TAVERN => self.piece_tavern(s),
            SKA => self.piece_ska(s),
            MARIACHI => self.piece_mariachi(s),
            _ => self.piece_polka(s),
        }
        self.fuse(s);
        let meter = match (self.genre, self.tv.kind) {
            (POLKA, _) => "2/4",
            (TAVERN, T_JIG) => "6/8 jig",
            (TAVERN, T_SLIP) => "9/8 slip jig",
            (TAVERN, T_REEL) => "4/4 reel",
            (TAVERN, T_WALTZ) => "3/4",
            (TAVERN, _) if self.tv.bar_steps == 12 => "3/4 ballad",
            (TAVERN, _) => "4/4 ballad",
            (SKA, _) => self.sk.meter,
            (MARIACHI, _) => self.mc.meter,
            _ => "4/4",
        };
        self.log.push((self.bpm, self.root, self.mode, meter));
    }

    fn section_start(&mut self, s: &mut Synth) {
        let sec = self.plan[self.sec];
        self.mute = 0;
        // which 4-bar groups play the B patterns: they alternate, never all A
        self.scheme = if sec.kind == MAIN {
            self.rng.pick(&[0b0110, 0b1010, 0b0100, 0b1110, 0b0101, 0b1001, 0b0010])
        } else {
            self.rng.pick(&[0, 0b10])
        };
        // which progression: A for verses, B for choruses and later mains,
        // C for bridges and breaks
        let mains_before = self.plan[..self.sec].iter().filter(|x| x.kind == MAIN).count();
        let r = &mut self.rng;
        let which = match (self.genre, sec.kind) {
            (POLKA | TAVERN, _) => 0,
            (_, BREAK) => 2,
            (AMBIENT, MAIN) => self.sec % 2,
            (TECHNO, MAIN) => (mains_before >= 1) as usize,
            (SYNTHWAVE, MAIN) => (sec.mask & L_LEAD != 0) as usize,
            (LOFI, MAIN) => (sec.mask & L_LEAD != 0 || r.chance(0.3)) as usize,
            (SKA, _) => self.sk.map.get(self.sec).map_or(0, |x| x.dress as usize),
            _ => 0,
        };
        self.prog = self.progs[which.min(self.progs.len() - 1)].clone();
        self.pivot = None;
        let dress = if self.genre == TAVERN {
            self.tv.map.get(self.sec).map_or(0, |m| (m.1 / 2 % 2) as usize)
        } else if self.genre == POLKA {
            match self.pk.map.get(self.sec).map_or(0, |m| m.0) {
                0 => 0,
                1 => 1,
                _ => r.below(3),
            }
        } else if self.genre == MARIACHI {
            self.mc.map.get(self.sec).map_or(0, |m| m.dress)
        } else {
            which
        };
        let lift = self.lifts.get(self.sec).copied().unwrap_or(0);
        if lift != self.lift {
            self.lift = lift;
            self.pad_v.clear();
            self.keys_v.clear();
        }
        match self.genre {
            AMBIENT => {
                let tex = sec.mask & L_TEX != 0;
                s.fx.wind_to = if tex && !self.rain { r.range(0.06, 0.12) } else { 0.0 };
                s.fx.rain_to = if tex && self.rain { r.range(0.05, 0.1) } else { 0.0 };
                if sec.kind == MAIN && r.chance(0.4) {
                    self.phrase = make_melody(r, 0.3, 0, 11);
                }
                if r.chance(0.5) {
                    self.arp = make_arp(r);
                } else {
                    self.arp2 = make_arp(r);
                }
            }
            TECHNO => {
                s.ch[CH_KICK].rev2 = if sec.mask & L_RUMBLE != 0 { 0.7 } else { 0.0 };
                s.fx.wind_to = if sec.kind == BREAK && r.chance(0.5) { 0.05 } else { 0.0 };
                if sec.kind == MAIN && r.chance(0.5) {
                    self.bass = new_bass(r, self.mode, self.bass_style);
                }
                self.bass2 = mutate_bass(r, &self.bass, self.mode, 3);
                if r.chance(0.4) {
                    let k = r.pick(&[3, 5, 7]);
                    let e = euclid(k, 16, r.below(16));
                    for i in 0..16 {
                        self.perc[i] = if e[i] { r.range(0.45, 0.7) } else { 0.0 };
                    }
                    self.perc_kind = r.pick(&[RIM, RIM, TOM_MID, SHAKER]);
                }
            }
            SYNTHWAVE => {
                self.arp = make_arp(r);
                self.arp2 = make_arp(r);
                if sec.kind == MAIN && r.chance(0.3) {
                    self.phrase = make_melody(r, 0.55, -2, 9);
                }
            }
            LOFI => {
                if sec.kind == MAIN && r.chance(0.35) {
                    self.phrase = make_melody(r, 0.35, -2, 8);
                }
            }
            TAVERN => self.tavern_section(),
            SKA => self.ska_section(),
            MARIACHI => self.mariachi_section(),
            _ => self.polka_section(),
        }
        self.dress(s, dress);
    }

    // -- the parts all genres share

    /// The section's instruments: the B ones for choruses and bridges (`which`
    /// 1 and 2), the A ones for verses, a fifth of the time the other way
    /// round; the bass and drone keep theirs more often; the drum kits swap
    /// together a third of the time.
    fn dress(&mut self, s: &mut Synth, which: usize) {
        let kit_b = self.rng.chance(0.35);
        for k in 0..self.band.tones.len() {
            let (ch, pair, _) = self.band.tones[k];
            let melodic = matches!(ch, CH_PAD | CH_ARP | CH_LEAD | CH_BELL | CH_KEYS);
            let b = if melodic { (which >= 1) != self.rng.chance(0.2) } else { which == 2 || self.rng.chance(0.25) };
            s.ch[ch].patch = pair[b as usize];
        }
        for (ch, kits) in &self.band.kits {
            s.ch[*ch].drums = kits[kit_b as usize].clone();
        }
    }

    /// Whether this 4-bar group plays the B patterns.
    fn var_b(&self) -> bool {
        (self.scheme >> (self.sec_bar / 4 % 8)) & 1 == 1
    }

    /// The fill this bar ends with: 0 none, 1 a small one (closing a 4-bar
    /// group), 2 a big one (closing an 8-bar group or the section).
    fn fill(&self, sec: Sec) -> u8 {
        let i = self.sec_bar;
        if i + 1 == sec.bars || i % 8 == 7 {
            2
        } else if i % 4 == 3 {
            1
        } else {
            0
        }
    }

    /// The pad: a chord at each change, voice-led, held until the next.
    fn pads(&mut self, b: &mut Bar, ch: usize, lo: i32, hi: i32, vel: f32, open: bool, tail: f32) {
        for h in 0..2 {
            let half = self.sec_bar * 2 + h;
            let c = self.chord(half);
            if !(half == 0 || self.chord(half - 1) != c) {
                continue;
            }
            let mut n = 1;
            while n < 16 && self.chord(half + n) == c {
                n += 1;
            }
            let v = self.voicing(c, lo, hi, &self.pad_v, open);
            for (i, &note) in v.iter().enumerate() {
                let vv = vel * (1.0 - 0.05 * i as f32);
                b.note(h as f32 * 8.0, 0.0, ch, note, vv, n as f32 * 8.0 + tail, false);
            }
            self.pad_v = v;
        }
    }

    /// The melody's bar: the sixteen-bar melody (varied each time round)
    /// against the chords.
    fn melody(&mut self, b: &mut Bar, ch: usize, vel: f32, late: f32, legato: f32) {
        let pb = self.mel_bar % 16;
        if pb == 0 && self.mel_bar > 0 {
            self.vary_phrase();
        }
        self.mel_pb = pb;
        self.mel_bar += 1;
        let tonic = self.tonic_near(self.lead_center);
        for k in 0..self.phrase.len() {
            let n = self.phrase[k];
            if n.step as u32 / 16 != pb {
                continue;
            }
            let st = n.step % 16;
            let c = self.chord(self.sec_bar * 2 + (st >= 8) as u32);
            let p = self.mel_pitch(n.deg + self.ptrans, c, st.is_multiple_of(8) || n.len >= 4);
            let j = self.rng.range(0.0, late);
            b.note(st as f32, j, ch, tonic + p, n.vel * vel, n.len as f32 * legato, false);
        }
    }

    /// A second voice answering the melody: in its rests of a quarter note
    /// or more, a short figure of chord tones in its own register leading to
    /// the melody's next note; where no melody plays, a slow line through
    /// the chords' thirds and sevenths, each a step or so from the last.
    fn counter(&mut self, b: &mut Bar, ch: usize, center: i32, vel: f32, answering: bool) {
        let i = self.sec_bar;
        let base = self.tonic_near(center);
        if answering {
            let pb = self.mel_pb;
            let mut busy = [false; 16];
            let mut next_pitch = Vec::new();
            for n in self.phrase.iter().filter(|n| n.step as u32 / 16 == pb) {
                let st = (n.step % 16) as usize;
                for x in busy[st..(st + n.len as usize).min(16)].iter_mut() {
                    *x = true;
                }
                next_pitch.push((st, n.deg));
            }
            let mut st = 0;
            let mut figures = 0;
            while st < 16 && figures < 2 {
                let run = (st..16).take_while(|&k| !busy[k]).count();
                if run >= 4 && st % 2 == 0 {
                    let c = self.chord(i * 2 + (st >= 8) as u32);
                    let tones: Vec<i32> = self.arp_tones(c, base - 7).into_iter().filter(|&p| p <= base + 7).collect();
                    let k0 = nearest(&tones, self.guide.clamp(base - 7, base + 7)) as isize;
                    let up = self.rng.bit_chance();
                    let sixteenths = run >= 6 && self.rng.chance(0.3);
                    let (count, gap) = if sixteenths { (4, 1) } else { ((run / 2).min(3), 2) };
                    for j in 0..count {
                        let k = (k0 + if up { j as isize } else { -(j as isize) }).clamp(0, tones.len() as isize - 1);
                        let p = tones[k as usize];
                        let at = st + j * gap;
                        let v = vel * if j == 0 { 1.0 } else { 0.85 };
                        b.note(at as f32, self.rng.range(0.0, self.human), ch, p, v, gap as f32 * 0.9, false);
                        self.guide = p;
                    }
                    figures += 1;
                    st += run;
                } else {
                    st += run.max(1);
                }
            }
        } else {
            for h in 0..2u32 {
                let half = i * 2 + h;
                let c = self.chord(half);
                if !(half == 0 || self.chord(half - 1) != c || (h == 0 && i % 2 == 0)) {
                    continue;
                }
                let iv = q_iv(c.q);
                let guides: Vec<i32> = [iv[1], *iv.get(3).unwrap_or(&iv[2])]
                    .iter()
                    .map(|x| (self.root + c.root + x).rem_euclid(12))
                    .collect();
                let cands = in_range(&guides, base - 5, base + 9);
                if cands.is_empty() {
                    continue;
                }
                let p = cands[nearest(&cands, if self.guide == 0 { base } else { self.guide })];
                let mut n = 1;
                while n < 4 && self.chord(half + n) == c {
                    n += 1;
                }
                b.note(h as f32 * 8.0, 0.0, ch, p, vel * 0.8, n as f32 * 8.0, false);
                self.guide = p;
            }
        }
    }

    /// A new turn of the melody: as it was, moved, rephrased or re-ended.
    fn vary_phrase(&mut self) {
        let r = &mut self.rng;
        match r.weighted(&[0.25, 0.25, 0.2, 0.15, 0.15]) {
            0 => self.ptrans = 0,
            1 => self.ptrans = r.pick(&[-2, 2, 1, -1, 3]),
            2 => {
                // a few notes nudged a 16th
                for _ in 0..3 {
                    let k = r.below(self.phrase.len().max(1));
                    if let Some(n) = self.phrase.get_mut(k)
                        && n.step % 16 != 0
                        && n.step % 16 < 15
                    {
                        n.step += 1;
                        n.len = n.len.saturating_sub(1).max(1);
                    }
                }
            }
            3 => {
                // a few notes moved
                for _ in 0..4 {
                    let k = r.below(self.phrase.len().max(1));
                    if let Some(n) = self.phrase.get_mut(k) {
                        n.deg += r.pick(&[-2, -1, 1, 2]);
                    }
                }
            }
            _ => {
                // new last two bars, home again
                let cad = make_cell(r, 32, 0.45, self.prange.0, self.prange.1, true);
                self.phrase.retain(|n| n.step < 224);
                self.phrase.extend(cad.iter().map(|n| MNote { step: n.step + 224, ..*n }));
            }
        }
    }

    /// Keeps a phrase's notes and their pitches near the mode's pentatonic (bells).
    fn penta(&self, d: i32) -> i32 {
        let major = MODES[self.mode][2] == 4;
        let x = d.rem_euclid(7);
        let avoid = if major { x == 3 || x == 6 } else { x == 1 || x == 5 };
        if avoid { d - 1 } else { d }
    }

    // -- ambient

    fn piece_ambient(&mut self, s: &mut Synth) {
        let r = &mut self.rng;
        self.bpm = r.range(88.0, 100.0) as f64;
        self.swing = 0.0;
        self.human = 0.006;
        // a pulse throughout: the arpeggiated sequence from the first bar,
        // soft percussion, and in fuller sections a bass pulse and a pad
        // that swells with the beat
        let core = L_PAD | L_ARP;
        self.plan = vec![
            Sec { kind: INTRO, bars: 4, mask: core | L_DRONE, energy: 0.35 },
            Sec { kind: MAIN, bars: 8, mask: core | L_DRONE | L_BELL | L_PERC, energy: 0.5 },
            Sec { kind: MAIN, bars: 8, mask: core | L_BELL | L_PERC | L_BASS | L_HAT | L_CTR, energy: 0.65 },
        ];
        if r.chance(0.5) {
            self.plan.push(Sec { kind: BREAK, bars: 4, mask: L_PAD | L_BELL | L_TEX | L_DRONE, energy: 0.4 });
        } else {
            self.plan.push(Sec { kind: MAIN, bars: 8, mask: core | L_DRONE | L_BELL | L_TEX, energy: 0.55 });
        }
        self.plan.push(Sec {
            kind: MAIN,
            bars: 8,
            mask: core | L_BELL | L_PERC | L_BASS | L_HAT | L_KICK | L_TEX,
            energy: 0.75,
        });
        self.plan.push(Sec {
            kind: MAIN,
            bars: 8,
            mask: core | L_DRONE | L_BELL | L_PERC | L_TEX | L_CTR,
            energy: 0.6,
        });
        if r.chance(0.5) {
            self.plan.push(Sec {
                kind: MAIN,
                bars: 8,
                mask: core | L_BELL | L_PERC | L_BASS | L_HAT | L_KICK,
                energy: 0.7,
            });
        }
        self.plan.push(Sec { kind: OUTRO, bars: 4, mask: core | L_DRONE | L_TEX, energy: 0.35 });
        self.phrase = make_melody(r, 0.3, 0, 11);
        self.arp = make_arp(r);
        self.arp2 = make_arp(r);
        self.arp_rate = r.pick(&[2, 2, 1]);
        self.pulse = r.pick(&[1.0, 2.0]);
        self.lead_center = 78;
        // a shaker in 8ths (A) or 16ths (B); a soft kick on the one (and three);
        // a rim on two and four (A) or the off-beats (B)
        self.perc = std::array::from_fn(|i| match i % 4 {
            0 => 0.4,
            2 => 0.28,
            _ => 0.0,
        });
        self.perc2 = std::array::from_fn(|i| [0.42, 0.18, 0.3, 0.2][i % 4]);
        self.kick = [0.0; 16];
        self.kick[0] = 0.55;
        self.kick2 = self.kick;
        self.kick2[8] = 0.4;
        if r.chance(0.5) {
            self.kick2[10] = 0.3;
        }
        self.hat = [0.0; 16];
        self.hat[4] = 0.35;
        self.hat[12] = 0.35;
        self.hat2 = std::array::from_fn(|i| if i % 4 == 2 { 0.25 } else { 0.0 });
        self.hat2[12] = 0.35;
        self.prange = (0, 11);
        sounds_ambient(r, s);
        self.band = band_ambient(r, s);
        self.rain = r.chance(0.4);
        self.draw_progs();
    }

    fn ambient(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let e = sec.energy;
        let i = self.sec_bar;
        let vb = self.var_b();
        let fill = self.fill(sec);
        s.ch[CH_PAD].cut = (0.9 * self.walk[0] + 1.2 * (e - 0.5)).exp2();
        s.ch[CH_BELL].cut = 1.0;
        s.rev_wet = 0.45 + 0.1 * self.walk[1];
        // the pad pulses with the beat where the bass does, its peaks on the beats
        let pad = &mut s.ch[CH_PAD];
        pad.swell = if m & L_BASS != 0 { 0.35 } else { 0.0 };
        pad.swell_rate = (self.bpm / 60.0) as f32 * self.pulse;
        pad.swell_ph = 0.25;
        if m & L_PAD != 0 {
            self.pads(b, CH_PAD, 48, 76, 0.55 + 0.25 * e, true, 6.0);
        }
        if m & L_DRONE != 0 && i.is_multiple_of(4) {
            let t = self.tonic_near(38);
            let len = 4.0 * 16.0 + 4.0;
            b.note(0.0, 0.0, CH_DRONE, t, 0.7, len, false);
            if self.rng.chance(0.5) {
                b.note(0.0, 0.0, CH_DRONE, t + 7, 0.4, len, false);
            }
        }
        if m & L_BASS != 0 {
            // a soft pulse on the chord's root: quarters (A) or 8ths (B)
            let every = if vb { 2 } else { 4 };
            for st in (0..16).step_by(every) {
                let c = self.chord(i * 2 + (st >= 8) as u32);
                let n = self.bass_note(c, 36);
                let v = if st % 8 == 0 { 0.8 } else { 0.55 };
                b.note(st as f32, 0.0, CH_BASS, n, v, every as f32 * 0.8, false);
            }
        }
        if m & L_BELL != 0 {
            // the motif in the B groups, loose notes in the A ones
            self.bell_motif = vb;
            if self.bell_motif {
                let pb = self.mel_bar % 16;
                if pb == 0 && self.mel_bar > 0 {
                    self.vary_phrase();
                }
                self.mel_pb = pb;
                self.mel_bar += 1;
                let tonic = self.tonic_near(self.lead_center);
                for k in 0..self.phrase.len() {
                    let n = self.phrase[k];
                    if n.step as u32 / 16 != pb || self.rng.chance(0.1) {
                        continue;
                    }
                    let st = n.step % 16;
                    let c = self.chord(i * 2 + (st >= 8) as u32);
                    let p = self.mel_pitch(self.penta(n.deg + self.ptrans), c, st.is_multiple_of(8));
                    let j = self.rng.range(0.0, self.human);
                    b.note(st as f32, j, CH_BELL, tonic + p, 0.35 + 0.35 * n.vel, 4.0, false);
                }
            } else {
                for beat in 0..4 {
                    if !self.rng.chance(0.25 + 0.25 * e) || (fill > 0 && beat == 3) {
                        continue;
                    }
                    let st = beat * 4 + self.rng.pick(&[0, 0, 2]);
                    let c = self.chord(i * 2 + (st >= 8) as u32);
                    let pc = self.rng.pick(&self.pcs(c));
                    let base = self.tonic_near(72);
                    let note = base + (pc - base).rem_euclid(12) + 12 * self.rng.below(2) as i32;
                    let j = self.rng.range(0.0, self.human);
                    b.note(st as f32, j, CH_BELL, note, self.rng.range(0.3, 0.65), 4.0, false);
                }
            }
            if m & L_CTR != 0 {
                // the ocarina answers the bells' motif, or walks the chords' guide tones
                let answering = self.bell_motif;
                self.counter(b, CH_LEAD, 66, 0.55, answering);
            }
        }
        if m & L_ARP != 0 {
            // the sequence: thin at first in the intro, an order per variant,
            // and a rising flourish to close an 8-bar group
            let rate = self.arp_rate;
            let thin = sec.kind == INTRO && i < 2;
            for st in (0..16).step_by(rate) {
                if fill == 2 && st >= 12 {
                    continue;
                }
                if !self.rng.chance(if thin { 0.5 } else { 0.9 }) {
                    continue;
                }
                let c = self.chord(i * 2 + (st >= 8) as u32);
                let tones = self.arp_tones(c, self.tonic_near(72) - 5);
                let order = if vb { &self.arp2 } else { &self.arp };
                let k = order[(st / rate) % order.len()] as usize;
                let vel = if st % 4 == 0 { 0.6 } else { 0.42 };
                b.note(st as f32, 0.0, CH_ARP, tones[k % tones.len()], vel, rate as f32 * 2.0, false);
            }
            if fill == 2 {
                let c = self.chord(i * 2 + 1);
                let tones = self.arp_tones(c, self.tonic_near(72) - 5);
                for (k, st) in (12..16).enumerate() {
                    let n = tones[(k + 2).min(tones.len() - 1)];
                    b.note(st as f32, 0.0, CH_ARP, n, 0.4 + 0.08 * k as f32, 2.0, false);
                }
            }
        }
        if m & L_PERC != 0 {
            let p = if vb { self.perc2 } else { self.perc };
            for (st, &v) in p.iter().enumerate() {
                if v > 0.0 {
                    let j = self.rng.range(0.0, self.human);
                    b.hit(st as f32, j, CH_HAT, HAT_C, v * self.rng.range(0.7, 1.0));
                }
            }
        }
        if m & L_HAT != 0 {
            let p = if vb { self.hat2 } else { self.hat };
            for (st, &v) in p.iter().enumerate() {
                if v > 0.0 {
                    b.hit(st as f32, self.rng.range(0.0, self.human), CH_PERC, RIM, v);
                }
            }
        }
        if m & L_KICK != 0 {
            let p = if vb { self.kick2 } else { self.kick };
            for (st, &v) in p.iter().enumerate() {
                if v > 0.0 && !(fill == 2 && st >= 8) {
                    b.hit(st as f32, 0.0, CH_KICK, 0, v);
                }
            }
        }
    }

    // -- techno

    fn piece_techno(&mut self, s: &mut Synth) {
        let r = &mut self.rng;
        self.bpm = if self.piece == 1 {
            r.range(128.0, 136.0) as f64
        } else {
            (self.bpm + r.range(-3.0, 3.0) as f64).clamp(128.0, 138.0)
        };
        self.swing = r.pick(&[0.0, 0.0, 0.0, 0.08, 0.15]);
        self.human = 0.0;
        let rumble = if r.chance(0.6) { L_RUMBLE } else { 0 };
        let stabs = if r.chance(0.6) { L_KEYS } else { 0 };
        let drums = L_KICK | L_HAT | L_OHAT | L_SNARE | L_PERC;
        let brk =
            L_PAD | L_TEX | L_KEYS | if r.chance(0.6) { L_BASS } else { 0 } | if r.chance(0.5) { L_HAT } else { 0 };
        self.plan = vec![
            Sec { kind: INTRO, bars: 8, mask: L_KICK | L_HAT | L_PERC, energy: 0.4 },
            Sec { kind: BUILD, bars: 8, mask: L_KICK | L_HAT | L_OHAT | L_SNARE | L_BASS, energy: 0.6 },
            Sec { kind: MAIN, bars: 16, mask: drums | L_BASS | rumble | stabs, energy: 0.8 },
            Sec { kind: BREAK, bars: 8, mask: brk, energy: 0.35 },
            Sec { kind: MAIN, bars: 16, mask: drums | L_BASS | L_RUMBLE | L_KEYS | L_LEAD, energy: 0.95 },
        ];
        if r.chance(0.5) {
            self.plan.push(Sec { kind: BREAK, bars: 4, mask: L_PAD | L_KEYS | L_HAT | L_BASS, energy: 0.45 });
            self.plan.push(Sec { kind: MAIN, bars: 8, mask: drums | L_BASS | rumble | L_KEYS | L_LEAD, energy: 0.9 });
        }
        self.plan.push(Sec { kind: OUTRO, bars: 8, mask: L_KICK | L_HAT | L_PERC | L_BASS, energy: 0.5 });
        // A: four on the floor, claps on two and four, 8ths of hats with 16ths
        // among them; B: a ghost kick before the bar, an extra clap and a
        // ghost, 16ths of hats throughout
        self.kick = std::array::from_fn(|i| if i % 4 == 0 { 1.0 } else { 0.0 });
        self.kick2 = self.kick;
        self.kick2[r.pick(&[15, 14, 11])] = 0.5;
        self.snare = [0.0; 16];
        self.snare[4] = 0.9;
        self.snare[12] = 0.9;
        self.snare2 = self.snare;
        self.snare2[r.pick(&[7, 15, 14])] = 0.45;
        self.snare2[r.pick(&[10, 3, 9])] = 0.22;
        let acc = r.range(0.7, 0.9);
        self.hat = std::array::from_fn(|i| [0.3, 0.2, acc, 0.22][i % 4]);
        self.hat2 = std::array::from_fn(|i| [0.32, 0.26, acc, 0.28][i % 4]);
        self.ohat = std::array::from_fn(|i| if i % 4 == 2 { 0.55 } else { 0.0 });
        let e = euclid(r.pick(&[3, 5, 7]), 16, r.below(16));
        self.perc = std::array::from_fn(|i| if e[i] { 0.6 } else { 0.0 });
        let e2 = euclid(r.pick(&[4, 5, 6, 9]), 16, r.below(16));
        self.perc2 = std::array::from_fn(|i| if e2[i] { 0.55 } else { 0.0 });
        self.perc_kind = r.pick(&[RIM, RIM, TOM_MID, SHAKER]);
        let st = euclid(r.pick(&[3, 3, 5]), r.pick(&[8, 16]), r.below(8));
        self.stab = std::array::from_fn(|i| if st[i % st.len()] { 0.8 } else { 0.0 });
        self.bass_style = if r.chance(0.65) { 0 } else { 1 };
        self.bass = new_bass(r, self.mode, self.bass_style);
        self.bass2 = mutate_bass(r, &self.bass, self.mode, 3);
        self.period = r.pick(&[8.0, 16.0, 24.0, 32.0]);
        self.hook = make_cell(r, 16, 0.45, 0, 9, false);
        self.lead_center = r.pick(&[67, 69, 72]);
        sounds_techno(r, s, self.bass_style);
        self.band = band_techno(r, s, self.bass_style == 0);
        s.dly.target = (self.step_len() * 3.0) as f32;
        self.draw_progs();
    }

    fn techno(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let e = sec.energy;
        let i = self.sec_bar;
        let left = sec.bars - i;
        let vb = self.var_b();
        let fill = self.fill(sec);
        let next_main = self.plan.get(self.sec + 1).is_some_and(|x| x.kind == MAIN);
        // the acid filter: a slow sweep over `period` bars, a walk, the energy
        self.phase = (self.phase + 1.0 / self.period).fract();
        let sweep = sin01(self.phase);
        let open = if self.bass_style == 0 { 1.5 * sweep + 0.6 * self.walk[0] } else { 0.4 * self.walk[0] };
        s.ch[CH_BASS].cut = (open + 2.0 * (e - 0.7)).exp2();
        s.ch[CH_PAD].cut = (0.7 * self.walk[1] + (left as f32 / sec.bars as f32 - 0.5) * -1.5).exp2();
        s.ch[CH_KEYS].cut = (0.8 * self.walk[2] + 0.5 * sweep).exp2();
        let pre_drop = next_main && left == 1 && sec.kind != OUTRO;
        let rolling = next_main && ((sec.kind == BREAK && left <= 4) || (sec.kind == BUILD && left <= 2));
        // in a main section, an 8-bar group ends with the kick out for a beat
        // now and then, and a snare run; a 4-bar group with a short flourish
        let big = fill == 2 && sec.kind == MAIN && !pre_drop;
        let drop = big && self.rng.chance(0.4);
        if m & L_KICK != 0 {
            let p = if vb { self.kick2 } else { self.kick };
            for (st, &v) in p.iter().enumerate() {
                if v > 0.0 && !(pre_drop && st >= 8) && !(drop && st >= 12) {
                    b.hit(st as f32, 0.0, CH_KICK, 0, v);
                }
            }
        }
        if m & L_SNARE != 0 && !rolling {
            let p = if vb { self.snare2 } else { self.snare };
            for (st, &v) in p.iter().enumerate() {
                if v > 0.0 && !(big && st >= 12) {
                    b.hit(st as f32, 0.0, CH_SNARE, 0, v);
                }
            }
        }
        if big {
            for st in 12..16 {
                b.hit(st as f32, 0.0, CH_SNARE, 1, 0.3 + 0.12 * (st - 12) as f32);
            }
        } else if fill == 1 && sec.kind == MAIN {
            if self.rng.chance(0.5) {
                for (k, st) in [13.0, 14.0, 15.0].iter().enumerate() {
                    b.hit(*st, 0.0, CH_SNARE, 1, 0.25 + 0.1 * k as f32);
                }
            } else {
                b.hit(15.0, 0.0, CH_HAT, HAT_O, 0.5);
            }
        }
        if rolling {
            let div = match left {
                4 => 4,
                3 => 2,
                _ => 1,
            };
            for st in (0..16).step_by(div) {
                let x = ((4 - left.min(4)) as f32 * 16.0 + st as f32) / 64.0;
                b.hit(st as f32, 0.0, CH_SNARE, 1, 0.25 + 0.65 * x);
            }
        }
        if next_main && left == sec.bars.min(8) && (sec.kind == BREAK || sec.kind == BUILD) {
            b.ev.push((b.at(0.0, 0.0), Ev::Riser { secs: b.secs(16.0 * left as f32), vel: 0.5 }));
        }
        if m & L_HAT != 0 {
            let p = if vb { self.hat2 } else { self.hat };
            for (st, &v) in p.iter().enumerate() {
                if v > 0.0 && (st % 4 == 2 || vb || self.rng.chance(0.2 + 0.75 * e)) {
                    b.hit(st as f32, 0.0, CH_HAT, HAT_C, v * self.rng.range(0.8, 1.0));
                }
            }
        }
        if m & L_OHAT != 0 {
            for st in 0..16 {
                if self.ohat[st] > 0.0 && !(fill == 1 && st == 14) {
                    b.hit(st as f32, 0.0, CH_HAT, HAT_O, self.ohat[st]);
                }
            }
        }
        if m & L_PERC != 0 {
            let p = if vb { self.perc2 } else { self.perc };
            for (st, &v) in p.iter().enumerate() {
                if v > 0.0 {
                    b.hit(st as f32, 0.0, CH_PERC, self.perc_kind, v * self.rng.range(0.8, 1.0));
                }
            }
        }
        if sec.kind == MAIN && i.is_multiple_of(8) {
            b.hit(0.0, 0.0, CH_CYM, 0, if i == 0 { 0.8 } else { 0.55 });
        }
        if m & L_BASS != 0 {
            let c = self.chord(i * 2);
            let root = self.bass_note(c, self.tonic_near(36) - 3);
            let line = if vb { self.bass2 } else { self.bass };
            let prev_slide = line[15].on && line[15].slide;
            for st in 0..16 {
                let x = line[st];
                if !x.on || (drop && st >= 12) {
                    continue;
                }
                let slide_in = if st == 0 { prev_slide } else { line[st - 1].on && line[st - 1].slide };
                let next_on = line[(st + 1) % 16].on;
                let len = if x.slide && next_on { 1.15 } else { 0.55 };
                let vel = if x.accent { 1.0 } else { 0.72 };
                b.note(st as f32, 0.0, CH_BASS, root + x.note, vel, len, slide_in && self.bass_style == 0);
            }
        }
        if m & L_KEYS != 0 {
            let c = self.chord(i * 2);
            let v = self.voicing(c, 55, 74, &self.keys_v, false);
            // the stab pattern, shifted a 16th in the B groups
            let shift = if vb { self.stab_shift } else { 0 };
            for st in 0..16 {
                if self.stab[(st + 16 - shift) % 16] > 0.0 {
                    for &n in &v {
                        b.note(
                            st as f32,
                            0.0,
                            CH_KEYS,
                            n,
                            self.stab[(st + 16 - shift) % 16] * self.rng.range(0.85, 1.0),
                            1.0,
                            false,
                        );
                    }
                }
            }
            self.keys_v = v;
        }
        if m & L_PAD != 0 {
            self.pads(b, CH_PAD, 50, 72, 0.6, false, 4.0);
        }
        if m & L_LEAD != 0 {
            // the hook: a bar of plucks, moved up in the B groups, the stabs answering
            let tonic = self.tonic_near(self.lead_center);
            let by = if vb { 2 } else { 0 };
            for k in 0..self.hook.len() {
                let n = self.hook[k];
                let c = self.chord(i * 2 + (n.step >= 8) as u32);
                let p = self.mel_pitch(n.deg + by, c, n.step % 8 == 0);
                b.note(n.step as f32, 0.0, CH_LEAD, tonic + p, n.vel * 0.85, (n.len as f32).min(3.0) * 0.8, false);
            }
        }
        if i % 8 == 7 {
            // the 8-bar group's end: the line mutates, and its variant with it
            let r = &mut self.rng;
            let n = r.pick(&[1, 2, 2, 3]);
            self.bass = mutate_bass(r, &self.bass, self.mode, n);
            self.bass2 = mutate_bass(r, &self.bass, self.mode, 3);
            if r.chance(0.4) {
                let k = r.pick(&[1, 3, 5, 7, 9, 11, 13, 15]);
                self.hat[k] = if self.hat[k] > 0.0 { 0.0 } else { 0.22 };
            }
            self.stab_shift = r.pick(&[1, 2, 3, 6]);
            if r.chance(0.4) {
                // the hook changes a note or two
                for _ in 0..2 {
                    let k = r.below(self.hook.len().max(1));
                    if let Some(n) = self.hook.get_mut(k) {
                        n.deg += r.pick(&[-2, -1, 1, 2]);
                    }
                }
            }
        }
    }

    // -- synthwave

    fn piece_synthwave(&mut self, s: &mut Synth) {
        let r = &mut self.rng;
        self.bpm = r.range(105.0, 120.0) as f64;
        self.swing = 0.0;
        self.human = 0.002;
        let verse = L_KICK | L_SNARE | L_HAT | L_BASS | L_PAD | L_ARP;
        let chorus = verse | L_LEAD | L_OHAT | L_CTR;
        let bridge = L_PAD | L_ARP | L_BASS | L_CTR | if r.chance(0.5) { L_LEAD } else { L_PERC } | L_HAT;
        self.plan = vec![
            Sec { kind: INTRO, bars: 4, mask: L_PAD | L_ARP | if r.chance(0.5) { L_BASS } else { 0 }, energy: 0.35 },
            Sec { kind: MAIN, bars: 8, mask: verse, energy: 0.6 },
            Sec { kind: MAIN, bars: 8, mask: chorus, energy: 0.9 },
            Sec { kind: MAIN, bars: 8, mask: verse | L_PERC, energy: 0.65 },
            Sec { kind: MAIN, bars: 8, mask: chorus, energy: 0.95 },
            Sec { kind: BREAK, bars: 8, mask: bridge, energy: 0.5 },
            Sec { kind: MAIN, bars: 8, mask: chorus | L_PERC, energy: 1.0 },
        ];
        if r.chance(0.4) {
            self.plan.push(Sec { kind: MAIN, bars: 8, mask: chorus, energy: 1.0 });
        }
        self.plan.push(Sec { kind: OUTRO, bars: 4, mask: L_PAD | L_ARP | L_KICK, energy: 0.35 });
        // A: the kick on one and three (or four on the floor), B: an extra
        // kick, ghost snares; hats in 8ths (A) or 16ths (B)
        self.kick = [0.0; 16];
        let four = r.chance(0.4);
        for i in (0..16).step_by(if four { 4 } else { 8 }) {
            self.kick[i] = 1.0;
        }
        self.kick2 = self.kick;
        self.kick2[r.pick(&[10, 7, 11, 14])] = 0.75;
        self.snare = [0.0; 16];
        self.snare[4] = 0.95;
        self.snare[12] = 0.95;
        self.snare2 = self.snare;
        self.snare2[r.pick(&[15, 7])] = 0.25;
        self.snare2[r.pick(&[10, 14, 3])] = 0.2;
        self.hat = std::array::from_fn(|i| match i % 4 {
            0 => 0.5,
            2 => 0.42,
            _ => 0.0,
        });
        self.hat2 = std::array::from_fn(|i| [0.45, 0.25, 0.4, 0.25][i % 4]);
        self.ohat = [0.0; 16];
        self.ohat[14] = 0.45;
        if r.chance(0.5) {
            self.ohat[6] = 0.4;
        }
        self.perc = [0.0; 16];
        self.bass_style = r.pick(&[0, 0, 1, 2]) as u8;
        self.bass_style2 = r.pick(&[0, 1, 2, 2]) as u8;
        self.arp = make_arp(r);
        self.arp2 = make_arp(r);
        self.arp_rate = r.pick(&[1, 1, 2]);
        self.lead_center = r.pick(&[72, 74, 76]);
        self.phrase = make_melody(r, 0.55, -2, 9);
        self.prange = (-2, 9);
        sounds_synthwave(r, s);
        self.band = band_synthwave(r, s);
        s.dly.target = (self.step_len() * 3.0) as f32;
        self.draw_progs();
    }

    fn synthwave(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let e = sec.energy;
        let i = self.sec_bar;
        let vb = self.var_b();
        let fill = self.fill(sec);
        let big = fill == 2 && m & L_SNARE != 0;
        let small = fill == 1 && m & L_SNARE != 0;
        s.ch[CH_PAD].cut = (0.6 * self.walk[0] + (e - 0.7)).exp2();
        s.ch[CH_ARP].cut = (0.8 * self.walk[1] + 1.2 * (e - 0.6)).exp2();
        if m & L_KICK != 0 {
            let p = if vb { self.kick2 } else { self.kick };
            for (st, &v) in p.iter().enumerate() {
                if v > 0.0 && !(big && st >= 12) {
                    b.hit(st as f32, 0.0, CH_KICK, 0, v);
                }
            }
        }
        if m & L_SNARE != 0 {
            let p = if vb { self.snare2 } else { self.snare };
            for (st, &v) in p.iter().enumerate() {
                if v > 0.0 && !(big && st >= 12) {
                    b.hit(st as f32, 0.0, CH_SNARE, if v < 0.5 { 1 } else { 0 }, v);
                }
            }
        }
        if big {
            if self.rng.chance(0.6) {
                let toms: &[(f32, u8)] = if self.rng.chance(0.5) {
                    &[(12.0, TOM_HI), (13.0, TOM_HI), (14.0, TOM_MID), (15.0, TOM_LO)]
                } else {
                    &[(12.0, TOM_HI), (14.0, TOM_MID), (15.0, TOM_LO)]
                };
                for &(st, k) in toms {
                    b.hit(st, 0.0, CH_PERC, k, 0.85);
                }
            } else {
                for st in 12..16 {
                    b.hit(st as f32, 0.0, CH_SNARE, 1, 0.5 + 0.12 * (st - 12) as f32);
                }
            }
        } else if small {
            // a flam into the next group: a ghost and the hat opened
            b.hit(15.0, 0.0, CH_SNARE, 1, 0.35);
            b.hit(14.0, 0.0, CH_HAT, HAT_O, 0.4);
        }
        if i.is_multiple_of(8) && sec.kind == MAIN && m & L_KICK != 0 {
            b.hit(0.0, 0.0, CH_CYM, 0, if i == 0 { 0.75 } else { 0.5 });
        }
        if m & L_HAT != 0 {
            let p = if vb { self.hat2 } else { self.hat };
            for (st, &v) in p.iter().enumerate() {
                if v > 0.0 {
                    b.hit(st as f32, 0.0, CH_HAT, HAT_C, v * self.rng.range(0.85, 1.0));
                }
            }
        }
        if m & L_OHAT != 0 {
            for st in 0..16 {
                if self.ohat[st] > 0.0 && !(small && st == 14) {
                    b.hit(st as f32, 0.0, CH_HAT, HAT_O, self.ohat[st]);
                }
            }
        }
        if m & L_PERC != 0 {
            // a rim on the off-beat of four, a clap layering two and four in B
            b.hit(14.0, 0.0, CH_PERC, RIM, 0.5);
            if vb {
                b.hit(7.0, 0.0, CH_PERC, RIM, 0.35);
            }
        }
        if m & L_BASS != 0 {
            let lo = self.tonic_near(36) - 4;
            // the verse's bass, or in a chorus the B groups' other one
            let style = if e >= 0.9 && vb { self.bass_style2 } else { self.bass_style };
            for st in (0..16).step_by(if style == 2 { 1 } else { 2 }) {
                let c = self.chord(i * 2 + (st >= 8) as u32);
                let root = self.bass_note(c, lo);
                let next = self.chord(i * 2 + (st >= 8) as u32 + 1);
                let (note, len, vel) = match style {
                    0 => (root + if st % 4 == 2 { 12 } else { 0 }, 1.6, 0.85),
                    1 => {
                        let n = if st % 8 == 6 && next != c {
                            let fifth = (self.root + c.root + q_iv(c.q)[2]).rem_euclid(12);
                            root + (fifth - root).rem_euclid(12)
                        } else {
                            root
                        };
                        (n, 1.6, if st % 4 == 0 { 0.9 } else { 0.75 })
                    }
                    _ => {
                        if st % 4 == 1 {
                            continue;
                        }
                        (root, 0.9, if st % 4 == 0 { 0.9 } else { 0.7 })
                    }
                };
                b.note(st as f32, 0.0, CH_BASS, note, vel, len, false);
            }
        }
        if m & L_PAD != 0 {
            self.pads(b, CH_PAD, 52, 76, 0.6, false, 2.0);
        }
        if m & L_ARP != 0 {
            // the arpeggio's order per variant; the B groups an octave up in a chorus
            let rate = self.arp_rate;
            let up = if vb && e >= 0.9 { 12 } else { 0 };
            for st in (0..16).step_by(rate) {
                let c = self.chord(i * 2 + (st >= 8) as u32);
                let tones = self.arp_tones(c, self.tonic_near(64) - 4);
                let order = if vb { &self.arp2 } else { &self.arp };
                let k = order[(st / rate) % order.len()] as usize;
                let vel = if st % 4 == 0 { 0.85 } else { 0.65 };
                b.note(st as f32, 0.0, CH_ARP, tones[k % tones.len()] + up, vel, rate as f32 * 0.8, false);
            }
        }
        if m & L_LEAD != 0 {
            self.melody(b, CH_LEAD, 0.9, 0.0, 0.95);
        }
        if m & L_CTR != 0 {
            self.counter(b, CH_BELL, self.lead_center - 9, 0.7, m & L_LEAD != 0);
        }
    }

    // -- lo-fi

    fn piece_lofi(&mut self, s: &mut Synth) {
        let r = &mut self.rng;
        self.bpm = r.range(85.0, 95.0) as f64;
        self.swing = r.range(0.12, 0.28);
        self.human = 0.01;
        let drums = L_KICK | L_SNARE | L_HAT | L_BASS | L_KEYS | L_TEX;
        self.plan = vec![
            Sec { kind: INTRO, bars: 4, mask: L_KEYS | L_TEX | if r.chance(0.6) { L_BASS } else { 0 }, energy: 0.35 },
            Sec { kind: MAIN, bars: 8, mask: drums, energy: 0.6 },
            Sec { kind: MAIN, bars: 8, mask: drums | L_LEAD | L_CTR, energy: 0.75 },
            Sec { kind: MAIN, bars: 8, mask: drums | L_PERC | L_CTR, energy: 0.7 },
            Sec {
                kind: BREAK,
                bars: 4,
                mask: L_KEYS | L_TEX | if r.chance(0.5) { L_LEAD } else { L_BASS | L_HAT },
                energy: 0.4,
            },
            Sec { kind: MAIN, bars: 8, mask: drums | L_LEAD | L_PERC | L_CTR, energy: 0.8 },
        ];
        if r.chance(0.5) {
            self.plan.push(Sec { kind: MAIN, bars: 8, mask: drums | L_LEAD, energy: 0.75 });
        }
        self.plan.push(Sec { kind: OUTRO, bars: 4, mask: L_KEYS | L_TEX | L_BASS, energy: 0.35 });
        // two boom-bap kicks (A and B), snares on two and four with more
        // ghosts in B; hats in 8ths with a few 16ths (A) or swung 16ths (B)
        let kicks: [&[usize]; 6] =
            [&[0, 7, 10], &[0, 10, 11], &[0, 6, 9], &[0, 3, 10], &[0, 7, 9, 10], &[0, 8, 11, 14]];
        let ka = r.below(kicks.len());
        let kb = (ka + 1 + r.below(kicks.len() - 1)) % kicks.len();
        self.kick = [0.0; 16];
        self.kick2 = [0.0; 16];
        for (p, k) in [(&mut self.kick, ka), (&mut self.kick2, kb)] {
            for &st in kicks[k] {
                p[st] = if st == 0 { 1.0 } else { 0.8 };
            }
        }
        self.snare = [0.0; 16];
        self.snare[4] = 0.85;
        self.snare[12] = 0.9;
        self.snare2 = self.snare;
        self.snare2[7] = 0.18;
        self.snare2[15] = 0.2;
        self.snare_kind = if r.chance(0.3) { 1 } else { 0 };
        self.hat = std::array::from_fn(|i| match i % 4 {
            0 => 0.55,
            2 => 0.4,
            _ => 0.0,
        });
        self.hat2 = std::array::from_fn(|i| [0.5, 0.22, 0.38, 0.26][i % 4]);
        self.perc = std::array::from_fn(|i| if i % 2 == 1 { 0.35 } else { 0.0 });
        self.perc2 = std::array::from_fn(|i| [0.0, 0.3, 0.2, 0.35][i % 4]);
        self.lead_center = r.pick(&[72, 74, 76]);
        self.phrase = make_melody(r, 0.4, -2, 8);
        self.anticipated = false;
        sounds_lofi(r, s);
        self.band = band_lofi(r, s);
        self.prange = (-2, 8);
        let echo = r.pick(&[4.0, 3.0, 6.0]);
        s.dly.target = (self.step_len() * echo) as f32;
        self.draw_progs();
    }

    fn lofi(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let e = sec.energy;
        let i = self.sec_bar;
        let h = self.human;
        let vb = self.var_b();
        let fill = self.fill(sec);
        // an 8-bar group closes with the drums out on the last beat (or a
        // snare pickup), a 4-bar group with the kick dropped and ghosts
        let gap = fill == 2 && self.rng.chance(0.5);
        s.ch[CH_KEYS].cut = (0.5 * self.walk[0]).exp2();
        if m & L_KICK != 0 {
            let p = if vb { self.kick2 } else { self.kick };
            for (st, &v) in p.iter().enumerate() {
                if v > 0.0 && !(fill > 0 && st >= 12) {
                    let j = self.rng.range(0.0, h * 0.3);
                    b.hit(st as f32, j, CH_KICK, 0, v * self.rng.range(0.9, 1.0));
                }
            }
        }
        if m & L_SNARE != 0 {
            let p = if vb { self.snare2 } else { self.snare };
            for (st, &v) in p.iter().enumerate() {
                if v > 0.0 && !(gap && st >= 12) {
                    let j = self.rng.range(0.004, h + 0.004);
                    let kind = if v < 0.5 { 0 } else { self.snare_kind };
                    b.hit(st as f32, j, CH_SNARE, kind, v * self.rng.range(0.85, 1.0));
                }
            }
            for st in [3usize, 9] {
                if self.rng.chance(0.1 + 0.15 * e) {
                    b.hit(st as f32, self.rng.range(0.0, h), CH_SNARE, 0, self.rng.range(0.12, 0.22));
                }
            }
            if fill == 1 || (fill == 2 && !gap) {
                for (k, st) in [13.0, 14.0, 15.0].iter().enumerate() {
                    b.hit(*st, self.rng.range(0.0, h), CH_SNARE, 0, 0.14 + 0.06 * k as f32);
                }
            }
        }
        if m & L_HAT != 0 {
            let p = if vb { self.hat2 } else { self.hat };
            for (st, &v) in p.iter().enumerate() {
                if v > 0.0 && !(gap && st >= 12) {
                    let j = self.rng.range(0.0, h);
                    b.hit(st as f32, j, CH_HAT, HAT_C, v * self.rng.range(0.7, 1.0));
                }
            }
            if vb || gap {
                b.hit(14.0, self.rng.range(0.0, h), CH_HAT, HAT_O, 0.4);
            }
        }
        if m & L_PERC != 0 {
            let p = if vb { self.perc2 } else { self.perc };
            for (st, &v) in p.iter().enumerate() {
                if v > 0.0 && self.rng.chance(0.8) {
                    b.hit(st as f32, self.rng.range(0.0, h), CH_PERC, SHAKER, v * self.rng.range(0.6, 1.0));
                }
            }
        }
        if m & L_KEYS != 0 {
            let c0 = self.chord(i * 2);
            let c1 = self.chord(i * 2 + 1);
            let next = self.chord(i * 2 + 2);
            if !self.anticipated {
                let len = if c1 == c0 { 15.0 } else { 7.5 };
                self.strum(b, 0.0, c0, len, 0.6);
            }
            self.anticipated = false;
            if c1 != c0 {
                self.strum(b, 8.0, c1, 7.5, 0.55);
            } else if self.rng.chance(if vb { 0.75 } else { 0.45 }) {
                let st = self.rng.pick(&[6.0, 10.0, 11.0]);
                self.strum(b, st, c0, 16.0 - st, 0.42);
            }
            if next != c1 && self.rng.chance(if vb { 0.5 } else { 0.25 }) && i + 1 < sec.bars {
                self.strum(b, 14.0, next, 16.0, 0.55);
                self.anticipated = true;
            }
            let k = s.ch[CH_KEYS].patch;
            if k.ks > 0.0 || k.s < 0.1 {
                // a guitar or a piano dies away: strum again on the off-beats
                for st in [6.0, 11.0] {
                    let c = if st < 8.0 { c0 } else { c1 };
                    self.strum(b, st, c, 4.0, 0.4);
                }
            }
            if m & L_KICK == 0 {
                // with no drums the keys keep moving: a broken chord over the second half
                let tones = self.arp_tones(c1, 60);
                let up = self.rng.bit_chance();
                for (k, st) in [8.0, 10.0, 12.0].iter().enumerate() {
                    let n = tones[if up { k + 1 } else { 3 - k }.min(tones.len() - 1)];
                    b.note(*st, self.rng.range(0.0, h), CH_KEYS, n, 0.35, 3.5, false);
                }
            }
        }
        if m & L_BASS != 0 {
            let lo = self.tonic_near(42) - 5;
            let next_root = self.bass_note(self.chord(i * 2 + 2), lo);
            // the B groups push into the next chord a 16th early, or walk to it
            let push = vb && next_root != self.bass_note(self.chord(i * 2 + 1), lo) && self.rng.chance(0.6);
            let approach = !push && self.rng.chance(0.35) && next_root != self.bass_note(self.chord(i * 2 + 1), lo);
            for hh in 0..2u32 {
                let c = self.chord(i * 2 + hh);
                let root = self.bass_note(c, lo);
                let st0 = hh as f32 * 8.0;
                if hh == 1 && c == self.chord(i * 2) && self.rng.chance(0.4) {
                    // the second half answers with the fifth, or the octave pops
                    let n = root + self.rng.pick(&[7, 12]);
                    b.note(st0 + 2.0, h * 0.5, CH_BASS, n, 0.65, 2.0, false);
                    continue;
                }
                b.note(st0, h * 0.3, CH_BASS, root, 0.85, if hh == 1 && push { 5.0 } else { 5.8 }, false);
                if !(hh == 1 && (approach || push)) && self.rng.chance(0.6) {
                    let n = root + self.rng.pick(&[7, 12, 7, 10]);
                    b.note(st0 + 6.0, h * 0.5, CH_BASS, n, 0.6, 1.8, false);
                }
            }
            if approach {
                let n = next_root + self.rng.pick(&[-1, 1, -2]);
                b.note(14.0, h * 0.5, CH_BASS, n, 0.6, 2.0, false);
            }
            if push {
                b.note(15.0, h * 0.3, CH_BASS, next_root, 0.8, 1.6, false);
            }
        }
        if m & L_LEAD != 0 {
            self.melody(b, CH_LEAD, 0.8, h, 0.9);
        }
        if m & L_CTR != 0 {
            self.counter(b, CH_BELL, self.lead_center - 8, 0.6, m & L_LEAD != 0);
        }
    }

    // -- polka

    fn piece_polka(&mut self, s: &mut Synth) {
        let tonic = self.tonic_near(74);
        let mode = self.mode;
        let minor = mode != IONIAN;
        let r = &mut self.rng;
        self.bpm = r.range(110.0, 130.0) as f64;
        self.swing = 0.0;
        self.human = 0.003;
        let (a_plans, b_plans) = if minor { (POLKA_MINOR, POLKA_MINOR) } else { (POLKA_A, POLKA_B) };
        // strain lengths, and whether a dogfight and a last trio follow: the
        // piece lasts a minute and a half at least (bars of two beats)
        let mut lens: [usize; 3] = std::array::from_fn(|_| if r.chance(0.6) { 16 } else { 8 });
        let mut dog = r.chance(0.5);
        let bar_s = 120.0 / self.bpm;
        let total = |l: &[usize; 3], dog: bool| 10 + 2 * (l[0] + l[1] + l[2]) + if dog { 8 + l[2] } else { 0 };
        while total(&lens, dog) as f64 * bar_s < 95.0 {
            match lens.iter().position(|&x| x == 8) {
                _ if !dog => dog = true,
                Some(k) => lens[k] = 16,
                None => break,
            }
        }
        let [la, lb, lc] = lens;
        let a = polka_strain(r, a_plans, la, 0, mode);
        let mut b = polka_strain(r, b_plans, lb, 0, mode);
        for _ in 0..4 {
            if b.chords != a.chords {
                break;
            }
            b = polka_strain(r, b_plans, lb, 0, mode);
        }
        // the trio: in the subdominant (a minor polka's, in its relative
        // major), up or down so the tune stays in its range
        let (up, down) = if minor { (3, -9) } else { (5, -7) };
        let tk = if tonic + up < 80 { up } else { down };
        let c = polka_strain(r, POLKA_A, lc, tk, IONIAN);
        let na = a.chords.len() as u32;
        let (nb, nc) = (b.chords.len() as u32, c.chords.len() as u32);
        let all = L_KICK | L_SNARE | L_HAT | L_BASS | L_KEYS | L_LEAD | L_ARP;
        let sec = |kind: u8, bars: u32, energy: f32| Sec { kind, bars, mask: all, energy };
        let mut plan = vec![
            (sec(INTRO, 4, 0.8), (0, 0, na - 4)),
            (sec(MAIN, na, 0.8), (0, 0, 0)),
            (sec(MAIN, na, 0.85), (0, 1, 0)),
            (sec(MAIN, nb, 0.85), (1, 0, 0)),
            (sec(MAIN, nb, 0.9), (1, 1, 0)),
        ];
        if r.chance(0.7) {
            // two bars on the trio's dominant lead into it
            plan.push((sec(BUILD, 2, 0.8), (2, 0, 0)));
        }
        plan.push((sec(MAIN, nc, 0.7), (2, 0, 0)));
        plan.push((sec(MAIN, nc, 0.95), (2, 1, 0)));
        // a dogfight (the instruments trading phrases) and the trio once more
        let d = polka_strain(r, POLKA_B, 8, tk, IONIAN);
        if dog {
            plan.push((sec(MAIN, 8, 0.9), (3, 0, 0)));
            plan.push((sec(MAIN, nc, 1.0), (2, 2, 0)));
        }
        plan.push((sec(OUTRO, 2, 1.0), (2, 0, 0)));
        self.plan = plan.iter().map(|x| x.0).collect();
        self.pk = Polka {
            strains: vec![a, b, c, d],
            map: plan.iter().map(|x| x.1).collect(),
            tonic,
            hats: r.chance(0.5),
            b_other: r.chance(0.5),
            lead: CH_LEAD,
            ..Polka::default()
        };
        let trumpet = r.chance(0.5);
        sounds_polka(r, s, trumpet);
        self.band = band_polka(r, s);
    }

    /// A polka section begins: its tune as it goes this time through (a
    /// repeat re-figures a bar, adds graces and turns the answers round), who
    /// leads, and whether the other instrument doubles a third below.
    fn polka_section(&mut self) {
        let sec = self.plan[self.sec];
        let (si, rep, _) = self.pk.map[self.sec];
        let (chords, mode) = (self.pk.strains[si].chords.clone(), self.pk.strains[si].mode);
        let mut mel = self.pk.strains[si].mel.clone();
        let mut ans = self.pk.strains[si].ans.clone();
        let r = &mut self.rng;
        if rep > 0 && sec.kind == MAIN {
            let bars = chords.len();
            let b = r.below(bars - 1);
            if b % 4 != 3 && r.chance(0.7) {
                let prev = mel.iter().filter(|n| (n.step as usize) < 8 * b).last().map_or(7, |n| n.pitch);
                let kind = r.pick(&MID_FIGS);
                let fig = polka_figure(r, kind, chords[b], mode, prev, None, PK_LO, PK_HI);
                mel.retain(|n| n.step as usize / 8 != b);
                mel.extend(fig.iter().map(|n| PNote { step: n.step + 8 * b as u16, ..*n }));
                mel.sort_by_key(|n| n.step);
            }
            polka_graces(r, &mut mel, 0.12);
            // each answer's notes in reverse order: an upward figure turned down
            let mut k = 0;
            while k < ans.len() {
                let bar = ans[k].step / 8;
                let end = (k..ans.len()).find(|&j| ans[j].step / 8 != bar).unwrap_or(ans.len());
                let pitches: Vec<i32> = ans[k..end].iter().map(|n| n.pitch).rev().collect();
                for (n, p) in ans[k..end].iter_mut().zip(pitches) {
                    n.pitch = p;
                }
                k = end;
            }
        }
        let other_leads = (si == 1 && self.pk.b_other) || (si == 2 && rep == 2);
        self.pk.lead = if other_leads { CH_ARP } else { CH_LEAD };
        self.pk.trade = si == 3;
        self.pk.thirds = sec.kind == MAIN && rep > 0 && (si == 2 || r.chance(0.5)) && !self.pk.trade;
        self.pk.mel = mel;
        self.pk.ans = ans;
    }

    /// A bar of polka: oom on the beats (the bass, root then fifth, walking
    /// into a new strain now and then; the kick), pah off them (the chord,
    /// short, with the snare), the tune, answers and thirds; rolls into a new
    /// strain, a crash at its top; the trio's dominant leading in; a button
    /// to end on.
    fn polka(&mut self, b: &mut Bar, sec: Sec) {
        let (si, rep, first) = self.pk.map[self.sec];
        let (key, mode, bars) = {
            let st = &self.pk.strains[si];
            (st.key, st.mode, st.chords.len() as u32)
        };
        let e = sec.energy;
        let i = self.sec_bar;
        let bi = (i + first) % bars;
        let last = i + 1 == sec.bars;
        let dom = Harm { root: 7, q: Q_DOM7, inv: 0 };
        let h = match sec.kind {
            BUILD => dom,
            OUTRO if i == 0 => dom,
            OUTRO => Harm { root: 0, q: Q_MAJ, inv: 0 },
            _ => self.pk.strains[si].chords[bi as usize],
        };
        let tonic_pc = self.root + key;
        self.pk.mask = h.mask(tonic_pc);
        let tune = self.pk.tonic + key;
        let mut lead = self.pk.lead;
        let mut other = if lead == CH_LEAD { CH_ARP } else { CH_LEAD };
        if self.pk.trade && (bi / 2) % 2 == 1 {
            (lead, other) = (other, lead);
        }
        let blo = 34;
        let root = blo + (tonic_pc + h.bass() - blo).rem_euclid(12);
        if sec.kind == OUTRO && i == 1 {
            // the button: one stab, everyone
            b.hit(0.0, 0.0, CH_KICK, 0, 1.0);
            b.hit(0.0, 0.0, CH_SNARE, 0, 0.8);
            b.hit(0.0, 0.0, CH_CYM, 0, 0.8);
            b.note(0.0, 0.0, CH_BASS, root, 1.0, 3.0, false);
            let pcs: Vec<i32> = h.pcs().iter().map(|p| (tonic_pc + p).rem_euclid(12)).collect();
            for &n in &voicing_pcs(&pcs, 55, 70, &self.keys_v, false) {
                b.note(0.0, 0.0, CH_KEYS, n, 0.9, 3.0, false);
            }
            let t = tune + if tune - self.pk.tonic > 6 { 0 } else { 12 };
            b.note(0.0, 0.0, CH_LEAD, t, 0.95, 4.0, false);
            b.note(0.0, 0.0, CH_ARP, t - 8, 0.85, 4.0, false);
            return;
        }
        // the beats and the off-beats
        b.hit(0.0, 0.0, CH_KICK, 0, 0.8 + 0.15 * e);
        b.hit(4.0, 0.0, CH_KICK, 0, 0.7 + 0.15 * e);
        let roll = last && self.rng.chance(0.75);
        for st in [2.0, 6.0] {
            if !(roll && st == 6.0) {
                b.hit(st, 0.0, CH_SNARE, 0, 0.35 + 0.35 * e);
            }
            if self.pk.hats && e > 0.75 {
                b.hit(st, 0.0, CH_HAT, HAT_C, 0.4);
            }
        }
        if roll {
            for k in 0..8 {
                b.hit(4.0 + 0.5 * k as f32, 0.0, CH_SNARE, 1, 0.3 + 0.07 * k as f32);
            }
        }
        if i == 0 && sec.kind == MAIN && (si > 0 && (rep == 0 || si >= 2)) {
            b.hit(0.0, 0.0, CH_CYM, 0, 0.6 + 0.2 * e);
        }
        // the bass: root and fifth, or a walk into the next strain's first chord
        let next = self.pk.map.get(self.sec + 1).filter(|_| last && sec.kind == MAIN).map(|&(s2, _, f2)| {
            let st = &self.pk.strains[s2];
            st.key + st.chords[f2 as usize].bass()
        });
        let sc: Vec<i32> = in_range(&chord_scale(mode, h), blo - 6 - tonic_pc, blo + 18 - tonic_pc)
            .into_iter()
            .map(|p| p + tonic_pc)
            .collect();
        match next.filter(|&n| (n + self.root - root).rem_euclid(12) != 0 && self.rng.chance(0.6)) {
            Some(n) => {
                let target = root - 6 + (n + self.root - (root - 6)).rem_euclid(12);
                let d: isize = if target >= root { 1 } else { -1 };
                let k = nearest(&sc, root) as isize;
                for j in 0..4 {
                    let p = sc[(k + d * j).clamp(0, sc.len() as isize - 1) as usize];
                    b.note(2.0 * j as f32, 0.0, CH_BASS, if j == 0 { root } else { p }, 0.85, 1.7, false);
                }
            }
            None => {
                let fifth = (tonic_pc + h.root + 7).rem_euclid(12);
                let f = root - 7 + (fifth - (root - 7)).rem_euclid(12);
                let f = if f < 30 { f + 12 } else { f };
                b.note(0.0, 0.0, CH_BASS, root, 0.9, 1.8, false);
                b.note(4.0, 0.0, CH_BASS, f, 0.8, 1.8, false);
            }
        }
        // the pah: the chord, short and bright (a seventh as its shell)
        let iv: &[i32] = if h.q == Q_DOM7 { &[0, 4, 10] } else { &q_iv(h.q)[..3] };
        let pcs: Vec<i32> = iv.iter().map(|x| (tonic_pc + h.root + x).rem_euclid(12)).collect();
        let v = voicing_pcs(&pcs, 55, 70, &self.keys_v, false);
        for (st, acc) in [(2.0, 1.0), (6.0, 0.9)] {
            for &n in &v {
                b.note(st, 0.0, CH_KEYS, n, (0.5 + 0.25 * e) * acc, 1.0, false);
            }
        }
        self.keys_v = v;
        // the tune
        let vel = 0.7 + 0.3 * e;
        let mel: Vec<PNote> = match sec.kind {
            BUILD | OUTRO => {
                let kind = if i == 0 { F_ARP_UP } else { F_RUN_HALF };
                let prev = self.pk.mel.last().map_or(4, |n| n.pitch);
                polka_figure(&mut self.rng, kind, h, mode, prev, None, PK_LO, PK_HI)
            }
            _ => self
                .pk
                .mel
                .iter()
                .filter(|n| n.step as u32 / 8 == bi)
                .map(|n| PNote { step: n.step % 8, ..*n })
                .collect(),
        };
        let csc = in_range(&chord_scale(mode, h), PK_LO - 12, PK_HI);
        let grace_steps = 0.06 / b.secs(1.0);
        for n in &mel {
            let st = n.step as f32;
            let len = n.len as f32 * if lead == CH_ARP { 0.95 } else { 0.85 };
            let mut late = 0.0;
            if n.grace != 0 {
                b.note(st, 0.0, lead, tune + n.pitch + n.grace as i32, n.vel * vel * 0.7, grace_steps, false);
                late = 0.045;
            }
            b.note(st, late, lead, tune + n.pitch, n.vel * vel, len, n.grace != 0 && lead == CH_ARP);
            if self.pk.thirds {
                let k = nearest(&csc, n.pitch);
                let third = csc[k.saturating_sub(2)];
                b.note(st, late, other, tune + third, n.vel * vel * 0.8, len, false);
            }
        }
        if matches!(sec.kind, MAIN | INTRO) && !self.pk.trade {
            for n in self.pk.ans.iter().filter(|n| n.step as u32 / 8 == bi) {
                b.note((n.step % 8) as f32, 0.0, other, tune + n.pitch, n.vel * vel, n.len as f32 * 0.9, false);
            }
        }
    }

    // -- tavern

    fn piece_tavern(&mut self, s: &mut Synth) {
        let r = &mut self.rng;
        // the dance: its meter and tempo
        let kind = r.weighted(&[3.0, 1.5, 3.0, 1.5, 1.0]) as u8;
        let (bar_steps, beat_steps) = match kind {
            T_JIG => (12, 6),
            T_SLIP => (18, 6),
            T_REEL => (16, 4),
            T_WALTZ => (12, 4),
            _ => {
                if r.bit_chance() {
                    (12, 4)
                } else {
                    (16, 4)
                }
            }
        };
        self.bpm = if kind == T_BALLAD { r.range(64.0, 80.0) } else { r.range(100.0, 130.0) } as f64;
        self.swing = if kind == T_REEL { r.range(0.05, 0.15) } else { 0.0 };
        self.human = 0.012;
        // a fiddler's and a piper's keys
        self.root = r.pick(&[2, 7, 9, 4, 0, 2, 7, 5]);
        let mode = self.mode;
        let pools = match mode {
            IONIAN => TAV_ION,
            MIXOLYDIAN => TAV_MIX,
            AEOLIAN => TAV_AEO,
            _ => TAV_DOR,
        };
        self.tv = Tavern { kind, bar_steps, beat_steps, drone: r.chance(0.75), ..Tavern::default() };
        let mut strains = Vec::new();
        for (k, pool) in pools.iter().enumerate() {
            let chords = halves(r.pick(pool));
            // the B strain sits higher
            let (lo, hi) = if k == 0 { (-3, 12) } else { (0, 15) };
            let (mel, ans) = tavern_tune(r, &self.tv, &chords, mode, lo, hi);
            strains.push(TStrain { chords, mel, ans });
        }
        self.tv.strains = strains;
        // AABB as many times through as a minute and a half takes; each time
        // round more join (the answering instrument, the tambourine, the
        // spoons, a harmony above, at last the claps)
        let bar_s = bar_steps as f64 / beat_steps as f64 * 60.0 / self.bpm;
        let rounds = ((95.0 / (32.0 * bar_s)).ceil() as u32).clamp(1, 4);
        let ballad = kind == T_BALLAD;
        let drone = if self.tv.drone { L_DRONE } else { 0 };
        let bass = if r.chance(0.4) { L_BASS } else { 0 };
        let mut plan = vec![(
            Sec { kind: INTRO, bars: if ballad { 4 } else { 2 }, mask: L_KEYS | drone | L_KICK, energy: 0.5 },
            (0, 0),
        )];
        for round in 0..rounds {
            for (si, half) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let q = 2 * round + half;
                let mut mask = L_LEAD | L_KEYS | drone;
                if !ballad || q >= 1 {
                    mask |= L_KICK;
                }
                if q >= 1 {
                    mask |= L_BELL;
                }
                if q >= 1 && !ballad {
                    mask |= L_HAT;
                }
                if q >= 2 {
                    mask |= bass | L_CTR | if ballad { 0 } else { L_PERC };
                }
                if round + 1 == rounds && si == 1 && half == 1 && !ballad {
                    mask |= L_SNARE;
                }
                let energy = (0.55 + 0.12 * q as f32 + 0.05 * si as f32).min(1.0);
                plan.push((Sec { kind: MAIN, bars: 8, mask, energy }, (si, q)));
            }
        }
        plan.push((Sec { kind: OUTRO, bars: 2, mask: L_LEAD | L_KEYS | drone | L_KICK | L_BELL, energy: 0.8 }, (0, 0)));
        self.plan = plan.iter().map(|x| x.0).collect();
        self.tv.map = plan.iter().map(|x| x.1).collect();
        sounds_tavern(r, s);
        self.band = band_tavern(r, s);
    }

    /// A tavern section begins: the strain's tune this time through (a bar
    /// re-figured on repeats, more ornaments each time), a harmony above
    /// (a third or a sixth) where the second instrument doubles, the B
    /// strain up an octave late on.
    fn tavern_section(&mut self) {
        let sec = self.plan[self.sec];
        let (si, rep) = self.tv.map[self.sec];
        let st = self.tv.strains[si].clone();
        let (bs, beat) = (self.tv.bar_steps as u16, self.tv.beat_steps as u16);
        let beats = bs / beat;
        let mode = self.mode;
        let r = &mut self.rng;
        let mut mel = st.mel;
        if sec.kind == MAIN && rep > 0 && r.chance(0.7) {
            let bar = r.pick(&[0u16, 1, 2, 4, 5, 6]);
            let mut p = mel.iter().filter(|n| n.step / bs < bar).last().map_or(4, |n| n.pitch);
            let mut dir = if r.bit_chance() { 1 } else { -1 };
            let (lo, hi) = if si == 0 { (-3, 12) } else { (0, 15) };
            let mut v = Vec::new();
            for k in 0..beats {
                let h = st.chords[bar as usize * 2 + ((k * 2) / beats) as usize];
                for (s0, len, pitch) in tavern_beat(r, self.tv.kind, beat as u32, h, mode, p, dir, lo, hi) {
                    v.push(PNote {
                        step: bar * bs + k * beat + s0,
                        len,
                        pitch,
                        vel: if s0 == 0 { 0.9 } else { 0.75 },
                        grace: 0,
                    });
                    p = pitch;
                }
                if !(lo + 2..=hi - 2).contains(&p) {
                    dir = -dir;
                }
            }
            mel.retain(|n| n.step / bs != bar);
            mel.extend(v);
            mel.sort_by_key(|n| n.step);
        }
        tavern_ornaments(r, &mut mel, 0.06 + 0.07 * rep as f32);
        self.tv.harmony = if sec.mask & L_CTR != 0 { r.pick(&[2, 5, 2]) } else { 0 };
        self.tv.up = if si == 1 && rep >= 3 && r.chance(0.5) { 12 } else { 0 };
        self.tv.mel = mel;
        self.tv.ans = st.ans;
    }

    /// A bar of tavern music: the bodhran (and tambourine, spoons, claps)
    /// in the dance's pattern, the lute's bass on the beat and chord after it
    /// (or a broken chord), the drone, the bass viol, the tune with its cuts
    /// and rolls, and the second instrument answering or in harmony.
    fn tavern(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let (si, rep) = self.tv.map[self.sec];
        let (bs, beat) = (self.tv.bar_steps, self.tv.beat_steps);
        let beats = bs / beat;
        let kind = self.tv.kind;
        let i = self.sec_bar;
        let e = sec.energy;
        let h = self.human;
        let bi = (i % 8) as usize;
        let minor = MODES[self.mode][2] == 3;
        let home = Harm { root: 0, q: if minor { Q_MIN } else { Q_MAJ }, inv: 0 };
        let chords: [Harm; 2] = match sec.kind {
            MAIN => [self.tv.strains[si].chords[bi * 2], self.tv.strains[si].chords[bi * 2 + 1]],
            OUTRO if i == 0 => {
                let last = self.tv.strains[1].chords[15];
                [last, last]
            }
            _ => [home, home],
        };
        let chord_at = |k: u32| chords[((k * 2) / beats) as usize];
        self.tv.mask = chords[0].mask(self.root);
        // the drone buzzes with the beat (a hurdy-gurdy's trompette)
        let d = &mut s.ch[CH_DRONE];
        d.swell = 0.25;
        d.swell_rate = (self.bpm / 60.0) as f32;
        d.swell_ph = 0.25;
        let vb = rep % 2 == 1;
        if m & L_KICK != 0 {
            let soft = if kind == T_BALLAD { 0.6 } else { 1.0 };
            for k in 0..beats {
                let at = (k * beat) as f32;
                let v = if k == 0 { 0.9 } else { 0.72 };
                match (kind, beat) {
                    (T_BALLAD, _) => {
                        if k == 0 {
                            b.hit(0.0, 0.0, CH_KICK, 0, 0.55);
                        }
                    }
                    (_, 6) => {
                        b.hit(at, self.rng.range(0.0, h), CH_KICK, 0, v * soft);
                        b.hit(at + 2.0, self.rng.range(0.0, h), CH_KICK, 1, 0.42);
                        b.hit(at + 4.0, self.rng.range(0.0, h), CH_KICK, 1, 0.5);
                        if vb && self.rng.chance(0.35) {
                            b.hit(at + 3.0, self.rng.range(0.0, h), CH_KICK, 1, 0.3);
                        }
                    }
                    (T_WALTZ, _) => {
                        b.hit(at, self.rng.range(0.0, h), CH_KICK, if k == 0 { 0 } else { 1 }, v);
                    }
                    _ => {
                        b.hit(
                            at,
                            self.rng.range(0.0, h),
                            CH_KICK,
                            if k % 2 == 0 { 0 } else { 1 },
                            if k % 2 == 0 { v } else { 0.6 },
                        );
                        b.hit(at + 2.0, self.rng.range(0.0, h), CH_KICK, 1, 0.4);
                        if vb && self.rng.chance(0.3) {
                            b.hit(at + 3.0, self.rng.range(0.0, h), CH_KICK, 1, 0.28);
                        }
                    }
                }
            }
        }
        if m & L_HAT != 0 {
            for k in 0..beats {
                let at = (k * beat) as f32;
                if beat == 6 {
                    b.hit(at + 2.0, self.rng.range(0.0, h), CH_HAT, 0, 0.35);
                    b.hit(at + 4.0, self.rng.range(0.0, h), CH_HAT, 0, 0.3);
                    if k % 2 == 1 {
                        b.hit(at, self.rng.range(0.0, h), CH_HAT, 1, 0.45);
                    }
                } else {
                    b.hit(at + 2.0, self.rng.range(0.0, h), CH_HAT, 0, 0.35);
                    if k % 2 == 1 {
                        b.hit(at, self.rng.range(0.0, h), CH_HAT, 1, 0.45);
                    }
                }
            }
        }
        if m & L_PERC != 0 {
            for st in (0..bs).step_by(2) {
                let on_beat = st % beat == 0;
                if on_beat || self.rng.chance(0.7) {
                    b.hit(st as f32, self.rng.range(0.0, h), CH_PERC, 0, if on_beat { 0.55 } else { 0.32 });
                }
            }
        }
        if m & L_SNARE != 0 {
            for k in 1..beats {
                if beat == 6 || k % 2 == 1 {
                    b.hit((k * beat) as f32, self.rng.range(0.0, h), CH_SNARE, 0, 0.6);
                }
            }
        }
        if i == 0 && sec.kind == MAIN && rep == 0 {
            b.hit(0.0, 0.0, CH_PERC, 3, 0.45);
        }
        if m & L_DRONE != 0 && i % 4 == 0 {
            let t = self.tonic_near(45);
            let len = 4.0 * bs as f32 + 4.0;
            b.note(0.0, 0.0, CH_DRONE, t, 0.6, len, false);
            b.note(0.0, 0.0, CH_DRONE, t + 7, 0.45, len, false);
        }
        if m & L_KEYS != 0 {
            let arpeggio = kind == T_BALLAD || s.ch[CH_KEYS].patch.ks > 2.0;
            for k in 0..beats {
                let c = chord_at(k);
                let lo = self.tonic_near(48) - 5;
                let bass = lo + (self.root + c.bass() - lo).rem_euclid(12);
                let v = self.voicing(c, 55, 69, &self.keys_v, false);
                let at = (k * beat) as f32;
                let j = self.rng.range(0.0, h);
                if arpeggio {
                    let tones = self.arp_tones(c, 55);
                    b.note(at, j, CH_KEYS, bass, 0.6, beat as f32, false);
                    for (n, x) in (2..beat).step_by(2).enumerate() {
                        b.note(at + x as f32, j, CH_KEYS, tones[(n + k as usize) % tones.len()], 0.45, 3.0, false);
                    }
                } else if beat == 6 {
                    // strings ring until the next stroke
                    b.note(at, j, CH_KEYS, bass, 0.7, 5.5, false);
                    for x in [2.0, 4.0] {
                        for &n in &v {
                            b.note(at + x, j, CH_KEYS, n, 0.38, 1.9, false);
                        }
                    }
                } else if kind == T_WALTZ {
                    if k == 0 {
                        b.note(at, j, CH_KEYS, bass, 0.7, bs as f32 - 0.5, false);
                    } else {
                        for &n in &v {
                            b.note(at, j, CH_KEYS, n, 0.4, beat as f32 - 0.2, false);
                        }
                    }
                } else {
                    b.note(at, j, CH_KEYS, bass, 0.7, 3.8, false);
                    for &n in &v {
                        b.note(at + 2.0, j, CH_KEYS, n, 0.4, 1.9, false);
                    }
                }
                self.keys_v = v;
            }
        }
        if m & L_BASS != 0 {
            let lo = self.tonic_near(38) - 5;
            for k in [0, beats / 2].iter().take(if beats >= 4 { 2 } else { 1 }) {
                let c = chord_at(*k);
                let n = lo + (self.root + c.bass() - lo).rem_euclid(12);
                b.note((k * beat) as f32, 0.0, CH_BASS, n, 0.7, (beats / 2).max(1) as f32 * beat as f32 * 0.9, false);
            }
        }
        // the tune (silent in the intro; in the outro, the cadence and a last long tonic)
        let tune = self.tonic_near(74) + self.tv.up;
        let lead = CH_LEAD;
        let mel: Vec<PNote> = match sec.kind {
            MAIN => self
                .tv
                .mel
                .iter()
                .filter(|n| n.step as usize / bs as usize == bi)
                .map(|n| PNote { step: n.step % bs as u16, ..*n })
                .collect(),
            OUTRO if i == 0 => self.tv.strains[1]
                .mel
                .iter()
                .filter(|n| n.step as usize / bs as usize == 7)
                .map(|n| PNote { step: n.step % bs as u16, ..*n })
                .collect(),
            OUTRO => vec![PNote { step: 0, len: bs as u16, pitch: 0, vel: 0.85, grace: 0 }],
            _ => Vec::new(),
        };
        if m & L_LEAD != 0 {
            let vel = 0.7 + 0.25 * e;
            let mono = s.ch[lead].mono;
            let st_s = b.secs(1.0);
            for n in &mel {
                let st = n.step as f32;
                let len = n.len as f32 * 0.92;
                let p = tune + n.pitch;
                let j = self.rng.range(0.0, h);
                match n.grace {
                    0 => b.note(st, j, lead, p, n.vel * vel, len, false),
                    9 => {
                        // a roll: cut, the note, a tap below, the note
                        let g = 0.045 / st_s;
                        b.note(st, j, lead, p + 2, n.vel * vel * 0.7, g, false);
                        b.note(st, j + 0.045, lead, p, n.vel * vel * 0.8, g, mono);
                        b.note(st, j + 0.09, lead, p - 1, n.vel * vel * 0.7, g, mono);
                        b.note(st, j + 0.135, lead, p, n.vel * vel, len - 0.135 / st_s, mono);
                    }
                    g => {
                        // a cut: a grace above, slid off into the note
                        b.note(st, j, lead, p + g as i32, n.vel * vel * 0.7, 0.05 / st_s, false);
                        b.note(st, j + 0.04, lead, p, n.vel * vel, len - 0.04 / st_s, mono);
                    }
                }
            }
            // the second instrument: a harmony above, or answers under the long notes
            if m & L_CTR != 0 && self.tv.harmony > 0 {
                for n in &mel {
                    let c = chord_at(n.step as u32 / beat);
                    let sc = in_range(&chord_scale(self.mode, c), PK_LO - 2, PK_HI + 12);
                    let k = nearest(&sc, n.pitch);
                    let p = sc[(k + self.tv.harmony as usize).min(sc.len() - 1)];
                    b.note(
                        n.step as f32,
                        self.rng.range(0.0, h),
                        CH_BELL,
                        tune + p - 12 * (self.tv.harmony == 5) as i32,
                        n.vel * vel * 0.75,
                        n.len as f32 * 0.9,
                        false,
                    );
                }
            }
        }
        if m & L_BELL != 0 && sec.kind == MAIN && !(m & L_CTR != 0 && self.tv.harmony > 0) {
            for n in self.tv.ans.iter().filter(|n| n.step as usize / bs as usize == bi) {
                b.note(
                    (n.step % bs as u16) as f32,
                    self.rng.range(0.0, h),
                    CH_BELL,
                    tune + n.pitch,
                    n.vel * 0.8,
                    n.len as f32 * 0.9,
                    false,
                );
            }
        }
    }

    // -- ska

    /// A ska piece: first-wave (Jamaica, the early sixties: a shuffle at 110
    /// to 130 bpm, a jazz band's horns, the piano and the guitar on the
    /// off-beats, a Hammond), 2-tone (Britain, the late seventies: nearly
    /// straight, 132 to 158, a Farfisa or a Vox bubbling, a picked bass, the
    /// kick now and then on every beat) or the third wave (ska-punk, the
    /// nineties on: straight, 135 to 148, the skank in the verses and the
    /// punk drive in the choruses, a four-piece horn section in close
    /// harmony carrying the tune). An intro (a vamp, a horn riff over it, or
    /// the drums leading in), the head (AB or AABA, eight bars a strain), a
    /// solo or two over its changes, now and then a breakdown, the head out
    /// (from the bridge, when long) and a tag ending on a final hit: a minute
    /// and a half to two and a half. (The third wave's form is its own.)
    fn piece_ska(&mut self, s: &mut Synth) {
        let mode = self.mode;
        let minor = MODES[mode][2] == 3;
        let shuffled = self.guest.is_none();
        let r = &mut self.rng;
        // the style: the third wave and 2-tone lean minor, the first wave major
        let style = if r.chance(if minor { 0.6 } else { 0.21 }) {
            SKS_THIRD
        } else if r.chance(if minor { 0.7 } else { 0.4 }) {
            SKS_TWO
        } else {
            SKS_FIRST
        };
        let (two_tone, third) = (style == SKS_TWO, style == SKS_THIRD);
        self.bpm = match style {
            SKS_THIRD => r.range(135.0, 148.0),
            SKS_TWO => r.range(132.0, 158.0),
            _ => r.range(110.0, 130.0),
        } as f64;
        self.swing = 0.0;
        self.human = match style {
            SKS_FIRST => 0.007,
            SKS_TWO => 0.004,
            _ => 0.003,
        };
        // the drummer's shuffle (the off-beat eighths late); none under a
        // mix's drummer, nor in the third wave
        let shuffle = match (shuffled, style) {
            (false, _) | (_, SKS_THIRD) => 0.0,
            (_, SKS_TWO) => r.range(0.0, 0.06),
            _ => r.range(0.12, 0.32),
        };
        // the horns' keys (Bb, Eb, F, C, G, Ab, D), and the minor keys a guitar likes
        self.root = if minor { r.pick(&[9, 2, 7, 4, 0, 5]) } else { r.pick(&[10, 3, 5, 0, 7, 8, 2]) };
        let (pa, pb, pv) = match mode {
            _ if third && minor => (SKA_TW_MIN_A, SKA_TW_MIN_B, SKA_TW_VAMP_MIN),
            _ if third => (SKA_TW_MAJ_A, SKA_TW_MAJ_B, SKA_TW_VAMP_MAJ),
            _ if minor => (SKA_MIN_A, SKA_MIN_B, SKA_VAMP_MIN),
            MIXOLYDIAN => (SKA_MIX_A, SKA_MIX_B, SKA_VAMP_MIX),
            _ => (SKA_MAJ_A, SKA_MAJ_B, SKA_VAMP_MAJ),
        };
        let a = halves(r.pick(pa));
        let b = halves(r.pick(pb));
        let vamp = halves(r.pick(pv));
        // the A strain comes home, the bridge (higher) ends on the dominant
        let (am, aa) = ska_tune(r, &a, mode, SK_LO, SK_HI, true);
        let (bm, ba) = ska_tune(r, &b, mode, SK_LO + 2, SK_HI + 2, false);
        // the form (at the tempo a mix's drummer will pull it to)
        let bpm = self.guest.map_or(self.bpm, |g| self.bpm + 0.6 * (MID_BPM[g] - self.bpm));
        let (plan, form, solos, intro) =
            if third { (ska_third_form(r), vec![0, 1], 0, SKI_RIFF) } else { ska_form(r, 240.0 / bpm, two_tone) };
        self.plan = plan.iter().map(|x| x.0).collect();
        // the riffs: the intro's (on the lead's register; the third wave's
        // a run) and the one behind the solos
        let home = Harm { root: 0, q: if minor { Q_MIN } else { Q_MAJ }, inv: 0 };
        let riff = ska_riff(r, home, mode, SK_LO + 2, SK_HI, if third { SKA_TW_RUNS } else { SKA_RIFFS });
        let back = ska_riff(r, home, mode, SK_LO - 5, SK_HI - 7, SKA_RIFFS);
        let mut strains = vec![SkaStrain { chords: a, mel: am, ans: aa }, SkaStrain { chords: b, mel: bm, ans: ba }];
        let mut run = Vec::new();
        if third {
            // the bridge's chords, and the horn break's runs over the chorus's
            let pc = if minor { SKA_TW_MIN_C } else { SKA_TW_MAJ_C };
            strains.push(SkaStrain { chords: halves(r.pick(pc)), mel: Vec::new(), ans: Vec::new() });
            run = ska_runs(r, &strains[1].chords, mode, SK_LO, SK_HI + 2);
        }
        self.sk = Ska {
            bar_steps: 16,
            beat_steps: 4,
            meter: match style {
                SKS_THIRD => "4/4 third wave",
                SKS_TWO => "4/4 2-tone",
                _ => "4/4 first wave",
            },
            two_tone,
            third,
            shuffle,
            four: third || (two_tone && self.bpm > 140.0 && r.chance(0.6)),
            open: r.chance(0.4),
            bubble: two_tone || r.chance(0.4),
            intro,
            end: r.below(3) as u8,
            strains,
            vamp,
            form,
            map: plan.iter().map(|x| x.1).collect(),
            riff,
            back,
            run,
            ..Ska::default()
        };
        sounds_ska(r, s, style);
        self.band = band_ska(r, s, style);
        // the soloists, each in its register, as loud as the lead's horn
        let orig = s.ch[CH_LEAD].patch;
        let mut who: Vec<usize> = (0..SKA_SOLOISTS.len()).collect();
        for k in 0..solos as usize {
            let j = k + r.below(who.len() - k);
            who.swap(k, j);
            let (_, make, center) = SKA_SOLOISTS[who[k]];
            self.sk.solos.push((levelled(&orig, center as f32, make(r)), center));
        }
        self.progs = vec![self.sk.strains[0].chords.clone(), self.sk.strains[1].chords.clone()];
        self.prog = self.progs[0].clone();
        // the third wave's last chorus now and then a half or a whole step up
        self.lifts.clear();
        if third && r.chance(0.35) {
            let up = r.pick(&[1, 2, 2]);
            let k = plan.len() - 2;
            self.lifts = (0..plan.len()).map(|j| if j >= k { up } else { 0 }).collect();
        }
        s.dly.target = (self.step_len() * 3.0) as f32;
    }

    /// A ska section begins: its chords (the vamp, a strain, the whole
    /// chorus under a solo, the tag: the head's last two bars and home),
    /// the head as it goes this time through (a bar re-figured and the
    /// ending made anew on repeats, more scoops each time, falls off the
    /// long notes in the head out), how the horns voice it, and whether the
    /// organ holds chords under a solo.
    fn ska_section(&mut self) {
        let x = self.sk.map[self.sec];
        let sec = self.plan[self.sec];
        let home = self.ska_home();
        let st = &self.sk.strains;
        self.prog = match x.part {
            SK_INTRO => self.sk.vamp.clone(),
            SK_HEAD | SK_BREAK | SK_RUN => st[x.strain as usize].chords.clone(),
            SK_SOLO => self.sk.form.iter().flat_map(|&k| st[k as usize].chords.iter().copied()).collect(),
            _ => {
                let tail = &st[0].chords[12..16];
                (0..(sec.bars - 1) / 2).flat_map(|_| tail.iter().copied()).chain([home, home]).collect()
            }
        };
        let mode = self.mode;
        let r = &mut self.rng;
        if matches!(x.part, SK_HEAD | SK_TAG) {
            let si = x.strain as usize;
            let (lo, hi) = (SK_LO + 2 * si as i32, SK_HI + 2 * si as i32);
            let strain = &self.sk.strains[si];
            let mut mel = strain.mel.clone();
            let chords = &strain.chords;
            let refigure = |r: &mut Rng, mel: &mut Vec<PNote>, bar: u16| {
                let b2 = bar as usize * 2;
                let c = [chords[b2], chords[b2 + 1], chords[(b2 + 2) % chords.len()]];
                let prev = mel.iter().filter(|n| n.step / 16 < bar).last().map_or(4, |n| n.pitch);
                let mut dir = if r.bit_chance() { 1 } else { -1 };
                let cell = if bar == 7 { r.pick(SKA_CADENCE) } else { r.pick(SKA_CELLS) };
                let mut fig = ska_figure(r, cell, c, mode, prev, &mut dir, lo, hi);
                if bar == 7 {
                    ska_land(&mut fig, c, if si == 0 { &[0, 12, 4] } else { &[7, 11, 2] }, lo, hi);
                }
                mel.retain(|n| n.step / 16 != bar);
                mel.extend(fig.iter().map(|n| PNote { step: n.step + 16 * bar, ..*n }));
                mel.sort_by_key(|n| n.step);
            };
            if x.rep > 0 && x.part == SK_HEAD {
                if r.chance(0.7) {
                    let bar = r.pick(&[1u16, 2, 5, 6]);
                    refigure(r, &mut mel, bar);
                }
                // a new ending, now and then
                if r.chance(0.4) {
                    refigure(r, &mut mel, 6);
                    refigure(r, &mut mel, 7);
                }
            }
            ska_scoops(r, &mut mel, 0.05 + 0.05 * x.rep.min(3) as f32);
            if x.out || x.part == SK_TAG {
                // falls off the phrases' last long notes
                for bar in [3u16, 7] {
                    if let Some(n) = mel.iter_mut().filter(|n| n.step / 16 == bar && n.len >= 4).last()
                        && r.chance(0.55)
                    {
                        n.grace = SKA_FALL;
                    }
                }
            }
            self.sk.mel = mel;
            self.sk.ans = strain.ans.clone();
        }
        self.sk.harmony = match x.part {
            SK_INTRO => r.pick(&[SKH_OCT, SKH_OCT, SKH_THIRD]),
            SK_HEAD if x.out => r.pick(&[SKH_TRIAD, SKH_THIRD, SKH_OCT]),
            SK_HEAD if x.strain == 1 => r.pick(&[SKH_THIRD, SKH_SIXTH, SKH_TRIAD]),
            SK_HEAD if x.rep == 0 => r.pick(&[SKH_OCT, SKH_OCT, SKH_THIRD]),
            SK_HEAD => r.pick(&[SKH_THIRD, SKH_SIXTH, SKH_TRIAD]),
            SK_TAG => SKH_TRIAD,
            _ => r.pick(&[SKH_SIXTH, SKH_OCT]),
        };
        if x.part == SK_RUN {
            self.sk.mel = self.sk.run.clone();
            self.sk.ans.clear();
        }
        if self.sk.third {
            // the third wave's horns: the lines in unison and octaves or in
            // thirds, the choruses in close harmony; the punk beat (a rock
            // beat more often than the skank beat) and the power chords
            self.sk.harmony = match x.part {
                SK_HEAD if x.strain == 1 => r.pick(&[SKH_TRIAD, SKH_TRIAD, SKH_THIRD]),
                SK_HEAD | SK_INTRO | SK_RUN => r.pick(&[SKH_OCT, SKH_THIRD]),
                _ => SKH_TRIAD,
            };
            self.sk.drive = r.weighted(&[0.15, 0.85]) as u8;
            self.sk.ring = r.chance(0.35);
        }
        self.sk.hold = x.part == SK_SOLO && r.chance(0.4);
        if x.part == SK_SOLO {
            self.sk.solo_prev = r.pick(&[4, 7, 0, 9]);
            self.sk.solo_dir = 1;
        }
    }

    /// The home chord.
    fn ska_home(&self) -> Harm {
        Harm { root: 0, q: if MODES[self.mode][2] == 3 { Q_MIN } else { Q_MAJ }, inv: 0 }
    }

    /// The chord the next bar opens on (across a section's end, the next
    /// section's first; home after the piece).
    fn ska_next(&self, sec: Sec) -> Harm {
        let i = self.sec_bar;
        if i + 1 < sec.bars {
            return self.chord(i * 2 + 2);
        }
        let st = &self.sk.strains;
        match self.sk.map.get(self.sec + 1) {
            Some(x) if matches!(x.part, SK_HEAD | SK_BREAK | SK_RUN) => st[x.strain as usize].chords[0],
            Some(x) if x.part == SK_SOLO => st[0].chords[0],
            Some(x) if x.part == SK_TAG && !self.sk.third => st[0].chords[12],
            _ => self.ska_home(),
        }
    }

    /// How late a 16th comes in the shuffle, in seconds: the off-beat eighth
    /// `shuffle` 16ths late, the 16ths either side of it half as late.
    fn ska_late(&self, b: &Bar, st: f32) -> f32 {
        let x = st.rem_euclid(4.0);
        let f = if x <= 2.0 { 0.5 * x } else { 0.5 * (4.0 - x) };
        self.sk.shuffle * f * b.secs(1.0)
    }

    /// A drum hit on the shuffle's grid, a little loose.
    fn ska_hit(&mut self, b: &mut Bar, st: f32, ch: usize, kind: u8, vel: f32) {
        let late = self.ska_late(b, st) + self.rng.range(0.0, self.human);
        b.hit(st, late, ch, kind, vel);
    }

    /// A bar of ska: the drums, the walking bass, the skank (the guitar's
    /// chop on every off-beat, the organ's bubble, the piano), and the
    /// horns: the head in unison or harmony with stabs answering it, the
    /// intro's riff, a solo with a riff behind it, the tag and its last hit.
    fn ska(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let x = self.sk.map[self.sec];
        let i = self.sec_bar;
        let c = [self.chord(i * 2), self.chord(i * 2 + 1), self.ska_next(sec)];
        // the third wave's distorted guitar and fourth horn never sit a phrase out
        let m = if self.sk.third { m | (sec.mask & (L_DRONE | L_PAD)) } else { m };
        if sec.kind == OUTRO && i + 1 == sec.bars {
            self.ska_button(b, m);
            return;
        }
        if x.punk {
            self.ska_punk(b, sec, m);
        } else {
            self.ska_drums(b, sec, m);
        }
        if x.part == SK_INTRO && self.sk.intro == SKI_DRUMS && i == 0 {
            // the drums lead in alone
            return;
        }
        if m & L_BASS != 0 {
            if x.punk {
                self.ska_drive(b, c, sec.energy, x.half);
            } else {
                self.ska_walk(b, c, sec.energy, x.part == SK_BREAK && i < 4);
            }
        }
        self.ska_skank(s, b, c, sec, m);
        if x.punk && m & L_DRONE != 0 {
            self.ska_power(b, c, sec, x.half);
        }
        if x.part == SK_SOLO
            && i == 0
            && let Some(&(p, _)) = self.sk.solos.get(x.solo as usize)
        {
            // the soloist steps up: the lead's sound for this chorus
            s.ch[CH_LEAD].patch = p;
        }
        match x.part {
            SK_HEAD | SK_RUN => self.ska_horns(b, sec, m, c, i % 8),
            SK_TAG => self.ska_horns(b, sec, m, c, 6 + i % 2),
            SK_SOLO => self.ska_solo(b, sec, m, c),
            SK_BREAK if self.sk.third => self.ska_swell(b, sec, m, c, x.half),
            SK_INTRO if self.sk.intro == SKI_RIFF => self.ska_riff_bar(b, m, c, true, 0.7 + 0.25 * sec.energy),
            SK_INTRO if i + 1 == sec.bars => self.ska_pickup(b, m, c),
            _ => {}
        }
    }

    /// Ska's drums, its own: the kick on one and three (on every beat in a
    /// fast 2-tone piece), the backbeat on two and four on the kit's snare
    /// or cross-stick (on the rim in a breakdown), ghost notes in the B
    /// groups, hats in eighths with the open one on the off-beats in places,
    /// a tambourine's jingle on them when everyone plays; a roll or the toms
    /// into each new eight bars, a "ba-dap" closing four, a crash at a
    /// section's top. A drum intro opens with a bar of them alone, the
    /// cross-stick on two and a roll over the second half.
    fn ska_drums(&mut self, b: &mut Bar, sec: Sec, m: u32) {
        let x = self.sk.map[self.sec];
        let (i, e) = (self.sec_bar, sec.energy);
        let vb = self.var_b();
        let fill = self.fill(sec);
        let lead_in = x.part == SK_INTRO && self.sk.intro == SKI_DRUMS && i == 0;
        let quiet = if sec.kind == BREAK { 0.65 } else { 1.0 };
        let roll = fill == 2 || lead_in;
        let toms = roll && !lead_in && self.rng.chance(0.35);
        // where the fill takes over
        let from = if lead_in {
            8.0
        } else if roll {
            12.0
        } else {
            16.0
        };
        if m & L_KICK != 0 {
            for k in 0..4 {
                if self.sk.four || k % 2 == 0 {
                    let v = [0.9, 0.62, 0.8, 0.62][k] * (0.85 + 0.15 * e) * quiet;
                    self.ska_hit(b, (4 * k) as f32, CH_KICK, 0, v);
                }
            }
            if vb && !roll && !self.sk.third && self.rng.chance(0.3) {
                self.ska_hit(b, 14.0, CH_KICK, 0, 0.5 * quiet);
            }
        }
        if m & L_SNARE != 0 {
            for st in [4.0, 12.0] {
                if st < from {
                    if sec.kind == BREAK || lead_in {
                        self.ska_hit(b, st, CH_PERC, RIM, 0.6);
                    } else {
                        let v = (0.78 + 0.12 * e) * self.rng.range(0.92, 1.0);
                        self.ska_hit(b, st, CH_SNARE, 0, v);
                    }
                }
            }
            if vb {
                for st in [7.0, 10.0, 15.0] {
                    if st < from && self.rng.chance(0.3) {
                        let v = self.rng.range(0.13, 0.22);
                        self.ska_hit(b, st, CH_SNARE, 1, v);
                    }
                }
            }
            if fill == 1 && sec.kind != BREAK {
                // ba-dap
                self.ska_hit(b, 14.0, CH_SNARE, 1, 0.4);
                self.ska_hit(b, 15.0, CH_SNARE, 0, 0.55);
            }
        }
        if roll && (m & L_SNARE != 0 || lead_in) {
            if toms {
                for (st, kind) in [(12.0, TOM_HI), (13.0, TOM_HI), (14.0, TOM_MID), (15.0, TOM_LO)] {
                    self.ska_hit(b, st, CH_PERC, kind, 0.8);
                }
            } else {
                // 16ths, then 32nds over the last beat, louder as they go
                let mut st = from;
                while st < 16.0 {
                    let v = 0.25 + 0.55 * (st - from) / (16.0 - from);
                    self.ska_hit(b, st, CH_SNARE, 1, v * quiet);
                    st += if st >= 14.0 && e >= 0.7 { 0.5 } else { 1.0 };
                }
            }
        }
        if m & (L_HAT | L_OHAT) != 0 {
            for k in 0..4 {
                let on = (4 * k) as f32;
                if lead_in && on >= from {
                    break;
                }
                self.ska_hit(b, on, CH_HAT, HAT_C, if k == 0 { 0.5 } else { 0.42 });
                // (the third wave's verses keep it closed, the beat ahead of the skank)
                let open = m & L_OHAT != 0
                    && sec.kind != BREAK
                    && !self.sk.third
                    && (self.sk.open || (vb && k % 2 == 1) || e >= 0.9);
                let (kind, v) = if open { (HAT_O, 0.42) } else { (HAT_C, 0.42) };
                let v = v * self.rng.range(0.9, 1.0);
                self.ska_hit(b, on + 2.0, CH_HAT, kind, v);
            }
        }
        if m & L_PERC != 0 && e >= 0.75 && sec.kind != BREAK {
            for k in 0..4 {
                if !(roll && k == 3) {
                    let v = 0.3 * self.rng.range(0.8, 1.0);
                    self.ska_hit(b, (4 * k + 2) as f32, CH_PERC, SHAKER, v);
                }
            }
        }
        let top = (i == 0 && matches!(sec.kind, MAIN | OUTRO))
            || (x.part == SK_INTRO && self.sk.intro == SKI_DRUMS && i == 1);
        if top {
            self.ska_hit(b, 0.0, CH_CYM, 0, 0.6 + 0.2 * e);
        }
    }

    /// A bar of walking bass: quarter notes, the chord's root on the one
    /// (and on the three when the chord changes), chord and scale tones
    /// between, the fourth beat leading into the next bar's root (a
    /// semitone from below or above, a scale step, or its fifth). A 2-tone
    /// bass pushes two eighths into the next bar now and then; a breakdown
    /// opens in half notes.
    fn ska_walk(&mut self, b: &mut Bar, c: [Harm; 3], e: f32, halves: bool) {
        let lo = self.tonic_near(38) - 5;
        let hi = lo + 19;
        let tonic = self.root;
        let fit = |p: i32| {
            let mut p = p;
            while p < lo {
                p += 12;
            }
            while p > lo + 14 {
                p -= 12;
            }
            p
        };
        let near = |pc: i32, to: i32| to - 6 + (pc - (to - 6)).rem_euclid(12);
        let abs = |h: Harm, scale: bool| -> Vec<i32> {
            let pcs = if scale { chord_scale(self.mode, h) } else { h.pcs() };
            in_range(&pcs, lo - 3 - tonic, hi - tonic).iter().map(|p| p + tonic).collect()
        };
        let at = |v: &[i32], k: isize| v[k.clamp(0, v.len() as isize - 1) as usize];
        let prev = if self.sk.bass_prev > 0 { self.sk.bass_prev } else { lo + 5 };
        let r0 = fit(near(tonic + c[0].bass(), prev));
        let tgt = fit(near(tonic + c[2].bass(), r0));
        let (ct0, sc0, sc2) = (abs(c[0], false), abs(c[0], true), abs(c[2], true));
        let r = &mut self.rng;
        let mut n = [r0; 4];
        if c[1] != c[0] {
            n[2] = fit(near(tonic + c[1].bass(), r0));
            let between: Vec<i32> = sc0.iter().copied().filter(|&p| p > n[0].min(n[2]) && p < n[0].max(n[2])).collect();
            n[1] = if !between.is_empty() && r.chance(0.75) {
                between[r.below(between.len())]
            } else {
                at(&ct0, nearest(&ct0, r0) as isize + if n[2] >= r0 { 1 } else { -1 })
            };
        } else {
            let up = if r0 > lo + 10 {
                false
            } else if tgt > r0 + 2 || r0 < lo + 3 {
                true
            } else if tgt < r0 - 2 {
                false
            } else {
                r.bit_chance()
            };
            let (k, j) = (nearest(&ct0, r0) as isize, nearest(&sc0, r0) as isize);
            // turning where the range ends
            let d: isize = match (up, k + 2 < ct0.len() as isize, k >= 2) {
                (true, true, _) | (false, _, false) => 1,
                _ => -1,
            };
            match r.weighted(&[0.45, 0.3, 0.25]) {
                // up or down the chord, or the scale, or root-fifth-third
                0 => (n[1], n[2]) = (at(&ct0, k + d), at(&ct0, k + 2 * d)),
                1 => (n[1], n[2]) = (at(&sc0, j + d), at(&sc0, j + 2 * d)),
                _ => (n[1], n[2]) = (at(&ct0, k + 2 * d), at(&ct0, k + d)),
            }
        }
        let from = n[2];
        let j = nearest(&sc2, tgt) as isize;
        n[3] = match r.weighted(&[0.55, 0.25, 0.2]) {
            0 => tgt + if from > tgt { 1 } else { -1 },
            1 => at(&sc2, j + if from > tgt { 1 } else { -1 }),
            _ => {
                if tgt + 7 <= hi {
                    tgt + 7
                } else {
                    tgt - 5
                }
            }
        };
        let push = (self.sk.two_tone || self.sk.third) && r.chance(0.2);
        self.sk.bass_prev = n[3];
        let lift = 0.85 + 0.15 * e;
        let h = self.human;
        if halves {
            for (st, p) in [(0.0, n[0]), (8.0, n[2])] {
                let late = self.ska_late(b, st) + self.rng.range(0.0, h);
                b.note(st, late, CH_BASS, p, 0.85 * lift, 7.0, false);
            }
            return;
        }
        for k in 0..4 {
            let st = 4.0 * k as f32;
            let late = self.ska_late(b, st) + self.rng.range(0.0, h);
            let v = [0.9, 0.68, 0.8, 0.7][k] * lift * self.rng.range(0.93, 1.0);
            if k == 3 && push {
                // two eighths: a chord tone, then the approach
                b.note(st, late, CH_BASS, n[2] + 12 * (n[2] + 12 <= hi) as i32, v, 1.7, false);
                let late = self.ska_late(b, st + 2.0) + self.rng.range(0.0, h);
                b.note(st + 2.0, late, CH_BASS, n[3], v, 1.7, false);
            } else {
                b.note(st, late, CH_BASS, n[k], v, 3.5, false);
            }
        }
    }

    /// Off-beat chords on channel `ch` (voiced in `range`, led on from
    /// `prev`), `len` 16ths long; on the off-beats in `twice` (bit k: beat
    /// k) a second stab on the 16th after, softer (the organ's bubble, the
    /// guitar's double chop); spread `spread` seconds a string, top first
    /// (an upstroke). Held chords instead when `hold`, or when the sound
    /// swells in too slowly to stab. In the first wave on the straight
    /// off-beat, a hair behind it (the shuffle is the drums' and the
    /// horns'), as the records have it. The last voicing back.
    #[allow(clippy::too_many_arguments)]
    fn ska_offbeats(
        &mut self,
        s: &Synth,
        b: &mut Bar,
        ch: usize,
        c: [Harm; 3],
        range: (i32, i32),
        prev: &[i32],
        vel: f32,
        len: f32,
        twice: u8,
        spread: f32,
        hold: bool,
    ) -> Vec<i32> {
        let v0 = self.voicing(c[0], range.0, range.1, prev, false);
        let v1 = self.voicing(c[1], range.0, range.1, &v0, false);
        let h = self.human;
        if hold || s.ch[ch].patch.a > 0.08 {
            let split = c[1] != c[0];
            for (st, v) in [(0.0, &v0), (8.0, &v1)] {
                if st == 8.0 && !split {
                    break;
                }
                let late = self.rng.range(0.0, h);
                for &n in v {
                    b.note(st, late, ch, n, vel * 0.75, if split { 8.0 } else { 16.0 }, false);
                }
            }
            return v1;
        }
        let swung = self.sk.two_tone || self.sk.third;
        // the third wave's upstroke looser
        let h = if self.sk.third && ch == CH_KEYS { 0.012 } else { h };
        for k in 0..4 {
            let st = (4 * k + 2) as f32;
            let v = if k < 2 { &v0 } else { &v1 };
            let late = if swung { self.ska_late(b, st) } else { 0.004 } + self.rng.range(0.0, h);
            for (j, &n) in v.iter().rev().enumerate() {
                let vv = vel * self.rng.range(0.88, 1.0);
                b.note(st, late + spread * j as f32, ch, n, vv, len, false);
            }
            if twice >> k & 1 == 1 {
                let late = if swung { self.ska_late(b, st + 1.0) } else { 0.004 } + self.rng.range(0.0, h);
                for &n in v.iter() {
                    b.note(st + 1.0, late, ch, n, vel * 0.68, len * 0.8, false);
                }
            }
        }
        v1
    }

    /// The skank: the guitar's chop on every off-beat (an upstroke; in a
    /// 2-tone piece's B groups doubled on the 16th after beats two and
    /// four), the organ's bubble (two stabs an off-beat) or its stab, held
    /// chords under a solo now and then, and the piano with the guitar.
    fn ska_skank(&mut self, s: &Synth, b: &mut Bar, c: [Harm; 3], sec: Sec, m: u32) {
        let e = sec.energy;
        let vb = self.var_b();
        let straight = self.sk.two_tone || self.sk.third;
        if m & L_KEYS != 0 {
            let twice = if straight && vb { 0b1010 } else { 0 };
            let prev = std::mem::take(&mut self.keys_v);
            // the first wave's chop tighter
            // the first wave's chop tighter, the third wave's upstroke a wider sweep
            let wide = match (straight, self.sk.third) {
                (_, true) => 2.5,
                (true, _) => 1.0,
                _ => 0.4,
            };
            let spread = self.rng.range(0.004, 0.008) * wide;
            self.keys_v =
                self.ska_offbeats(s, b, CH_KEYS, c, (57, 72), &prev, 0.6 + 0.2 * e, 1.2, twice, spread, false);
        }
        // (the third wave's organ and piano channels are a horn and a guitar)
        if m & L_PAD != 0 && !self.sk.third {
            let (twice, len) = if self.sk.bubble { (0b1111, 0.7) } else { (0, 1.7) };
            let prev = std::mem::take(&mut self.pad_v);
            let hold = self.sk.hold;
            self.pad_v = self.ska_offbeats(s, b, CH_PAD, c, (55, 70), &prev, 0.42 + 0.15 * e, len, twice, 0.0, hold);
        }
        if m & L_DRONE != 0 && !self.sk.third {
            let twice = if vb && !self.sk.two_tone { 0b1000 } else { 0 };
            let prev = std::mem::take(&mut self.sk.piano_v);
            self.sk.piano_v =
                self.ska_offbeats(s, b, CH_DRONE, c, (52, 67), &prev, 0.42 + 0.15 * e, 1.0, twice, 0.0, false);
        }
    }

    /// The second and third horns' notes under the lead's `p` (over chord
    /// `h`), above the tonic: an octave and two below (the lowest an octave
    /// up when it would go too low), a third or a sixth below in the chord's
    /// scale with the third horn an octave down, or close harmony (a third
    /// below, and the chord tone under that).
    fn ska_voices(&self, p: i32, h: Harm, how: u8, tune: i32) -> (i32, i32) {
        let sc = in_range(&chord_scale(self.mode, h), p - 14, p);
        let k = nearest(&sc, p);
        let third = sc[k.saturating_sub(2)];
        let low = |q: i32| if tune + q - 24 >= 40 { q - 24 } else { q - 12 };
        match how {
            SKH_OCT => (p - 12, low(p)),
            SKH_THIRD => (third, low(p)),
            SKH_SIXTH => (sc[k.saturating_sub(5)], low(p)),
            _ => {
                let under = in_range(&h.pcs(), third - 9, third - 2);
                (third, under.last().copied().unwrap_or(p - 12))
            }
        }
    }

    /// The head's bar `bi` (of eight) on the horns: the lead with its scoops
    /// and falls, the others under it as this time through has them; in the
    /// lead's rests the two low horns answer with stabs.
    fn ska_horns(&mut self, b: &mut Bar, sec: Sec, m: u32, c: [Harm; 3], bi: u32) {
        let tune = self.tonic_near(SK_TUNE);
        let vel = 0.72 + 0.25 * sec.energy;
        let h = self.human;
        for k in 0..self.sk.mel.len() {
            let n = self.sk.mel[k];
            if (n.step / 16) as u32 != bi {
                continue;
            }
            let st = (n.step % 16) as f32;
            let late = self.ska_late(b, st) + self.rng.range(0.0, h);
            let len = ska_len(n.len);
            let v = n.vel * vel;
            let p = tune + n.pitch;
            if m & L_LEAD != 0 {
                match n.grace {
                    0 => b.note(st, late, CH_LEAD, p, v, len, false),
                    SKA_FALL => {
                        // held, then slid off downward
                        let held = (len - 1.0).max(0.8);
                        b.note(st, late, CH_LEAD, p, v, held + 0.3, false);
                        let at = (st + held).min(15.0);
                        for (j, d) in [1, 3, 6].iter().enumerate() {
                            b.note(at + 0.25 * j as f32, late, CH_LEAD, p - d, v * (0.7 - 0.15 * j as f32), 0.35, true);
                        }
                    }
                    g => {
                        // a scoop: up from below, slid into the note
                        b.note(st, late, CH_LEAD, p + g as i32, v * 0.6, 0.6, false);
                        b.note(st, late + 0.035, CH_LEAD, p, v, len, true);
                    }
                }
            }
            let hc = c[ska_half(n.step % 16, n.len)];
            if self.sk.third {
                // the third wave's four: the sax, the trombone, the baritone under the trumpet
                let [p2, p3, p4] = self.ska_voices4(n.pitch, hc, self.sk.harmony, tune);
                for (ch, layer, q, f) in
                    [(CH_PAD, L_PAD, p2, 0.85), (CH_BELL, L_BELL, p3, 0.85), (CH_ARP, L_ARP, p4, 0.8)]
                {
                    if m & layer != 0 {
                        b.note(st, late, ch, tune + q, v * f, len, false);
                    }
                }
                continue;
            }
            let (p2, p3) = self.ska_voices(n.pitch, hc, self.sk.harmony, tune);
            if m & L_BELL != 0 {
                b.note(st, late, CH_BELL, tune + p2, v * 0.85, len, false);
            }
            if m & L_ARP != 0 {
                b.note(st, late, CH_ARP, tune + p3, v * 0.8, len, false);
            }
        }
        if m & (L_BELL | L_ARP) != 0 {
            for k in 0..self.sk.ans.len() {
                let n = self.sk.ans[k];
                if (n.step / 16) as u32 != bi {
                    continue;
                }
                let st = (n.step % 16) as f32;
                let late = self.ska_late(b, st) + self.rng.range(0.0, h);
                let hc = c[ska_half(n.step % 16, n.len)];
                let (_, under) = self.ska_voices(n.pitch, hc, SKH_TRIAD, tune);
                let v = n.vel * vel;
                b.note(st, late, CH_BELL, tune + n.pitch, v * 0.85, 0.8, false);
                b.note(st, late, CH_ARP, tune + under, v * 0.8, 0.8, false);
            }
        }
    }

    /// The third wave's four horns under the trumpet's `p` (over chord `h`):
    /// the sax, the trombone and the baritone. In octaves the sax with the
    /// trumpet, the trombone an octave down, the baritone two; in thirds or
    /// sixths the sax under the trumpet, the others in octaves; in close
    /// harmony the sax a third under, the trombone the chord tone below, the
    /// baritone on the chord's root.
    fn ska_voices4(&self, p: i32, h: Harm, how: u8, tune: i32) -> [i32; 3] {
        let (p2, p3) = self.ska_voices(p, h, how, tune);
        match how {
            SKH_OCT => [p, p2, p3],
            SKH_TRIAD => {
                let root = in_range(&[h.root.rem_euclid(12)], p3 - 12, p3 - 1);
                [p2, p3, root.first().copied().unwrap_or(p3 - 12)]
            }
            _ => [p2, p - 12, p3],
        }
    }

    /// A power chord on `h`'s root for the distorted guitar: the root (E2
    /// to Eb3), the fifth, the octave.
    fn ska_fifths(&self, h: Harm) -> [i32; 3] {
        let root = 40 + (self.root + h.root - 40).rem_euclid(12);
        [root, root + 7, root + 12]
    }

    /// The third wave's drums in the punk drive: the skank beat now and then
    /// (the kick on every beat, the snare on every off-beat, the crash
    /// riding the beats) or a rock beat (the kick on one and three, on every
    /// beat in the B groups, the snare on two and four, the crash riding the
    /// beats when it is strong), else the hats in eighths; in half time the
    /// kick on one, the snare on three, the hats in quarters; a crash on each
    /// bar's one (every other bar when it is less strong), the toms or a
    /// roll into each new eight bars, a "ba-dap" closing four.
    fn ska_punk(&mut self, b: &mut Bar, sec: Sec, m: u32) {
        let x = self.sk.map[self.sec];
        let (i, e) = (self.sec_bar, sec.energy);
        let vb = self.var_b();
        let fill = self.fill(sec);
        let roll = fill == 2;
        let toms = roll && self.rng.chance(0.6);
        let from = if roll { 12.0 } else { 16.0 };
        let lv = 0.85 + 0.15 * e;
        let skank = self.sk.drive == 0 && !x.half;
        let ride = !x.half && (skank || e >= 0.85);
        let mut hits: Vec<(f32, usize, u8, f32)> = Vec::with_capacity(32);
        if m & L_KICK != 0 {
            let ks: &[f32] = match (x.half, skank || vb) {
                (true, _) => &[0.0],
                (_, true) => &[0.0, 4.0, 8.0, 12.0],
                _ => &[0.0, 8.0],
            };
            for &st in ks.iter().filter(|&&st| st < from || st == 0.0) {
                hits.push((st, CH_KICK, 0, if st % 8.0 == 0.0 { 0.95 } else { 0.78 } * lv));
            }
        }
        if m & L_SNARE != 0 {
            let ss: &[f32] = match (x.half, skank) {
                (true, _) => &[8.0],
                (_, true) => &[2.0, 6.0, 10.0, 14.0],
                _ => &[4.0, 12.0],
            };
            for &st in ss.iter().filter(|&&st| st < from) {
                hits.push((st, CH_SNARE, 0, if skank { 0.75 } else { 0.9 } * lv));
            }
            if fill == 1 && !x.half {
                // ba-dap
                hits.push((14.0, CH_SNARE, 1, 0.45));
                hits.push((15.0, CH_SNARE, 0, 0.6));
            }
        }
        if m & (L_HAT | L_OHAT) != 0 {
            for k in 0..8 {
                let st = 2.0 * k as f32;
                if st >= from {
                    break;
                }
                if ride {
                    // the crash on the beats, the hat's foot between
                    if k % 2 == 0 && k > 0 {
                        hits.push((st, CH_CYM, 0, 0.4));
                    }
                } else if !x.half || k % 2 == 0 {
                    let open = k % 2 == 1 && m & L_OHAT != 0 && (vb || self.sk.open);
                    let v = if k % 2 == 0 { 0.5 } else { 0.36 };
                    hits.push((st, CH_HAT, if open { HAT_O } else { HAT_C }, v));
                }
            }
        }
        if i % if e >= 0.85 { 1 } else { 2 } == 0 {
            hits.push((0.0, CH_CYM, 0, 0.55 + 0.3 * e));
        }
        if roll && m & L_SNARE != 0 {
            if toms {
                for (st, kind) in [(12.0, TOM_HI), (13.0, TOM_HI), (14.0, TOM_MID), (15.0, TOM_LO)] {
                    hits.push((st, CH_PERC, kind, 0.85));
                }
            } else {
                // 16ths, louder as they go
                for k in 0..4 {
                    hits.push((12.0 + k as f32, CH_SNARE, 1, 0.35 + 0.15 * k as f32));
                }
            }
        }
        let h = self.human;
        for (st, ch, kind, v) in hits {
            let late = self.rng.range(0.0, h);
            b.hit(st, late, ch, kind, v * self.rng.range(0.93, 1.0));
        }
    }

    /// The third wave's bass in the punk drive: eighths on each half bar's
    /// root, the beats dug in a little, the last now and then a semitone from
    /// the next bar's root; in half time the root on the one and the three.
    fn ska_drive(&mut self, b: &mut Bar, c: [Harm; 3], e: f32, half: bool) {
        let lo = self.tonic_near(38) - 5;
        let fit = |p: i32| lo + (p - lo).rem_euclid(12);
        let roots = [fit(self.root + c[0].bass()), fit(self.root + c[1].bass())];
        let next = fit(self.root + c[2].bass());
        let lift = 0.85 + 0.15 * e;
        let h = self.human;
        if half {
            for (st, k, len) in [(0.0, 0, 7.0), (8.0, 1, 7.0)] {
                let late = self.rng.range(0.0, h);
                b.note(st, late, CH_BASS, roots[k], 0.9 * lift, len, false);
            }
            return;
        }
        let walk = next != roots[1] && self.rng.chance(0.3);
        for k in 0..8 {
            let mut p = roots[k / 4];
            if k == 7 && walk {
                p = next + if next > p { -1 } else { 1 };
            }
            let v = if k % 2 == 0 { 0.92 } else { 0.88 } * lift * self.rng.range(0.94, 1.0);
            let late = self.rng.range(0.0, h);
            b.note(2.0 * k as f32, late, CH_BASS, p, v, 1.6, false);
        }
        self.sk.bass_prev = roots[1];
    }

    /// The third wave's distorted guitar in the punk drive: power chords on
    /// each half bar's root, palm-muted eighths with the beats dug in, or
    /// rung out on the one and the three; in half time a bar rung out (or
    /// two halves).
    fn ska_power(&mut self, b: &mut Bar, c: [Harm; 3], sec: Sec, half: bool) {
        let vel = 0.75 + 0.2 * sec.energy;
        let split = c[1] != c[0];
        let hits: Vec<(f32, f32, f32, usize)> = if half || self.sk.ring {
            match (half, split) {
                (true, false) => vec![(0.0, 15.5, 1.0, 0)],
                _ => vec![(0.0, 7.5, 1.0, 0), (8.0, 7.5, 0.95, 1)],
            }
        } else {
            (0..8).map(|k| (2.0 * k as f32, 0.9, if k % 2 == 0 { 1.0 } else { 0.7 }, k / 4)).collect()
        };
        let h = self.human;
        for (st, len, f, k) in hits {
            let late = self.rng.range(0.0, h);
            for n in self.ska_fifths(c[k]) {
                b.note(st, late, CH_DRONE, n, vel * f, len, false);
            }
        }
    }

    /// The third wave's bridge on the horns: the four holding each chord in
    /// close harmony, swelling as it goes (a quiet bridge's from its second
    /// half), the trumpet moving as little as it can.
    fn ska_swell(&mut self, b: &mut Bar, sec: Sec, m: u32, c: [Harm; 3], half: bool) {
        let i = self.sec_bar;
        if !half && i < sec.bars / 2 {
            return;
        }
        let tune = self.tonic_near(SK_TUNE);
        let vel = 0.45 + 0.45 * (i + 1) as f32 / sec.bars as f32;
        let split = c[1] != c[0];
        for (k, st) in [(0usize, 0.0f32), (1, 8.0)] {
            if k == 1 && !split {
                break;
            }
            let h = c[k];
            let ct = in_range(&h.pcs(), SK_LO + 2, SK_HI);
            let top = ct[nearest(&ct, self.sk.solo_prev)];
            self.sk.solo_prev = top;
            let len = if split { 7.5 } else { 15.5 };
            let [p2, p3, p4] = self.ska_voices4(top, h, SKH_TRIAD, tune);
            for (ch, layer, n, f) in [
                (CH_LEAD, L_LEAD, top, 1.0),
                (CH_PAD, L_PAD, p2, 0.9),
                (CH_BELL, L_BELL, p3, 0.9),
                (CH_ARP, L_ARP, p4, 0.85),
            ] {
                if m & layer != 0 {
                    b.note(st, 0.0, ch, tune + n, vel * f, len, false);
                }
            }
        }
    }

    /// A bar of a horn riff fitted to this bar's chords (the riffs are two
    /// bars over the home chord): the intro's on all three horns as this
    /// section voices them, or the one behind a solo on the second horn
    /// with the third under it.
    fn ska_riff_bar(&mut self, b: &mut Bar, m: u32, c: [Harm; 3], intro: bool, vel: f32) {
        let home = self.ska_home();
        let bar = (self.sec_bar % 2) as u16;
        let src = if intro { &self.sk.riff } else { &self.sk.back };
        let notes: Vec<PNote> =
            src.iter().filter(|n| n.step / 16 == bar).map(|n| PNote { step: n.step % 16, ..*n }).collect();
        let (lo, hi) = if intro { (SK_LO, SK_HI + 2) } else { (SK_LO - 7, SK_HI - 5) };
        let fit = ska_refit(&notes, [home; 3], c, self.mode, lo, hi);
        let tune = self.tonic_near(SK_TUNE);
        let h = self.human;
        for n in &fit {
            let st = n.step as f32;
            let late = self.ska_late(b, st) + self.rng.range(0.0, h);
            let len = ska_len(n.len) * 0.85;
            let v = n.vel * vel;
            let how = if intro { self.sk.harmony } else { SKH_THIRD };
            let (p2, p3) = self.ska_voices(n.pitch, c[ska_half(n.step, n.len)], how, tune);
            if intro && self.sk.third {
                let [q2, q3, q4] = self.ska_voices4(n.pitch, c[ska_half(n.step, n.len)], how, tune);
                for (ch, layer, q, f) in [
                    (CH_LEAD, L_LEAD, n.pitch, 1.0),
                    (CH_PAD, L_PAD, q2, 0.85),
                    (CH_BELL, L_BELL, q3, 0.85),
                    (CH_ARP, L_ARP, q4, 0.8),
                ] {
                    if m & layer != 0 {
                        b.note(st, late, ch, tune + q, v * f, len, false);
                    }
                }
            } else if intro {
                if m & L_LEAD != 0 {
                    b.note(st, late, CH_LEAD, tune + n.pitch, v, len, false);
                }
                if m & L_BELL != 0 {
                    b.note(st, late, CH_BELL, tune + p2, v * 0.85, len, false);
                }
                if m & L_ARP != 0 {
                    b.note(st, late, CH_ARP, tune + p3, v * 0.8, len, false);
                }
            } else {
                b.note(st, late, CH_BELL, tune + n.pitch, v * 0.8, len, false);
                b.note(st, late, CH_ARP, tune + p2 - 12 * (tune + p2 >= 60) as i32, v * 0.75, len, false);
            }
        }
    }

    /// The intro's last bar: the horns' pickup into the head, two or three
    /// eighths climbing the scale to its first note.
    fn ska_pickup(&mut self, b: &mut Bar, m: u32, c: [Harm; 3]) {
        let Some(first) = self.sk.strains[0].mel.first().copied() else { return };
        if first.step > 2 || !self.rng.chance(0.7) {
            return;
        }
        let tune = self.tonic_near(SK_TUNE);
        let sc = in_range(&chord_scale(self.mode, c[1]), SK_LO - 5, SK_HI + 2);
        let j = nearest(&sc, first.pitch);
        let count = if self.rng.bit_chance() { 2 } else { 3 };
        for k in 0..count {
            let st = (16 - 2 * (count - k)) as f32;
            let p = sc[j.saturating_sub(count - k)];
            let late = self.ska_late(b, st) + self.rng.range(0.0, self.human);
            let (p2, p3) = self.ska_voices(p, c[1], SKH_OCT, tune);
            for (ch, layer, note, v) in
                [(CH_LEAD, L_LEAD, p, 0.8), (CH_BELL, L_BELL, p2, 0.7), (CH_ARP, L_ARP, p3, 0.65)]
            {
                if m & layer != 0 {
                    b.note(st, late, ch, tune + note, v, 1.5, false);
                }
            }
        }
    }

    /// A bar of a solo over the chorus's changes: phrases of two to four
    /// bars with a breath between (a pickup into the next), mostly eighths
    /// (a run of 16ths now and then, three against four), chord tones on
    /// the beats, scale tones and chromatic approaches between, the blue
    /// third slid into; the chorus's last bar comes to rest. Behind it, once
    /// it is under way (the second solo all through), the other horns' riff.
    fn ska_solo(&mut self, b: &mut Bar, sec: Sec, m: u32, c: [Harm; 3]) {
        let x = self.sk.map[self.sec];
        let i = self.sec_bar;
        let last = i + 1 == sec.bars;
        let center = self.sk.solos.get(x.solo as usize).map_or(SK_TUNE, |s| s.1);
        let base = self.tonic_near(center);
        if m & L_LEAD != 0 {
            let mode = self.mode;
            let major = MODES[mode][2] == 4;
            let r = &mut self.rng;
            let breathe = !last && ((i % 4 == 3 && r.chance(0.45)) || r.chance(0.06));
            let cell = if last {
                r.pick(SKA_CADENCE)
            } else if breathe {
                r.pick(SKA_PICKUPS)
            } else {
                r.pick(SKA_SOLO_CELLS)
            };
            let mut dir = self.sk.solo_dir;
            let mut v = ska_figure(r, cell, c, mode, self.sk.solo_prev, &mut dir, -5, 14);
            for k in 0..v.len().saturating_sub(1) {
                let nx = v[k + 1];
                if nx.step.is_multiple_of(4) && nx.step - v[k].step <= 2 && r.chance(0.3) {
                    // a chromatic approach from below
                    v[k].pitch = nx.pitch - 1;
                }
            }
            for n in v.iter_mut() {
                if major && n.pitch.rem_euclid(12) == 4 && n.len >= 2 && r.chance(0.35) {
                    // the blue third
                    n.grace = -1;
                }
                if n.step % 4 == 2 {
                    n.vel = (n.vel + 0.06).min(1.0);
                }
            }
            if let Some(n) = v.last() {
                self.sk.solo_prev = n.pitch;
            }
            self.sk.solo_dir = dir;
            let vel = 0.7 + 0.25 * sec.energy;
            let h = self.human;
            for n in &v {
                let st = n.step as f32;
                let late = self.ska_late(b, st) + self.rng.range(0.0, h);
                let len = ska_len(n.len);
                if n.grace != 0 {
                    b.note(st, late, CH_LEAD, base + n.pitch + n.grace as i32, n.vel * vel * 0.6, 0.6, false);
                    b.note(st, late + 0.035, CH_LEAD, base + n.pitch, n.vel * vel, len, true);
                } else {
                    b.note(st, late, CH_LEAD, base + n.pitch, n.vel * vel, len, false);
                }
            }
        }
        if m & L_CTR != 0 && (x.solo % 2 == 1 || i >= sec.bars / 2) && !last {
            self.ska_riff_bar(b, m, c, false, 0.55 + 0.25 * sec.energy);
        }
    }

    /// The last bar: the band's final hit (on the one; or the Charleston's
    /// one and the and-of-two; or the chord held over a roll and cut off on
    /// the four), the tonic chord (a sixth chord when held) everywhere, a
    /// crash; the lead now and then falls off it.
    fn ska_button(&mut self, b: &mut Bar, m: u32) {
        let minor = MODES[self.mode][2] == 3;
        let six = self.sk.end == SKE_HOLD || self.rng.chance(0.3);
        let q = match (minor, six) {
            (false, true) => Q_SIX,
            (false, false) => Q_MAJ,
            (true, true) => Q_MIN6,
            (true, false) => Q_MIN,
        };
        let h = Harm { root: 0, q, inv: 0 };
        let tune = self.tonic_near(SK_TUNE);
        let bass = self.tonic_near(36);
        let pcs = self.pcs(h);
        let hit: f32 = match self.sk.end {
            SKE_CHARLESTON => 6.0,
            SKE_HOLD => 12.0,
            _ => 0.0,
        };
        // the horns' top: the tonic, the third or the sixth, near the tune's tonic an octave up
        let top = in_range(&h.pcs(), 7, 16);
        let lead = top[self.rng.below(top.len())];
        if self.sk.third {
            // the third wave stops dead: one hit, an eighth long, everyone
            let [q2, q3, q4] = self.ska_voices4(lead, h, SKH_TRIAD, tune);
            for (ch, layer, n) in
                [(CH_LEAD, L_LEAD, lead), (CH_PAD, L_PAD, q2), (CH_BELL, L_BELL, q3), (CH_ARP, L_ARP, q4)]
            {
                if m & layer != 0 {
                    b.note(0.0, 0.0, ch, tune + n, 0.95, 1.8, false);
                }
            }
            if m & L_DRONE != 0 {
                for n in self.ska_fifths(h) {
                    b.note(0.0, 0.0, CH_DRONE, n, 0.95, 1.8, false);
                }
            }
            if m & L_BASS != 0 {
                b.note(0.0, 0.0, CH_BASS, bass, 0.95, 1.8, false);
            }
            b.hit(0.0, 0.0, CH_KICK, 0, 1.0);
            b.hit(0.0, 0.0, CH_SNARE, 0, 0.9);
            b.hit(0.0, 0.0, CH_CYM, 0, 0.9);
            return;
        }
        let (p2, p3) = self.ska_voices(lead, h, SKH_TRIAD, tune);
        // the chord: the rhythm section's `len` long, the horns' and the organ's `ring`
        let chord = |b: &mut Bar, st: f32, len: f32, ring: f32, vel: f32, v: &[Vec<i32>; 3]| {
            for (ch, layer, notes, l) in
                [(CH_KEYS, L_KEYS, &v[0], len), (CH_PAD, L_PAD, &v[1], ring), (CH_DRONE, L_DRONE, &v[2], len)]
            {
                if m & layer != 0 {
                    for &n in notes {
                        b.note(st, 0.0, ch, n, vel, l, false);
                    }
                }
            }
            for (ch, layer, n) in [(CH_LEAD, L_LEAD, lead), (CH_BELL, L_BELL, p2), (CH_ARP, L_ARP, p3)] {
                if m & layer != 0 {
                    b.note(st, 0.0, ch, tune + n, vel, ring, false);
                }
            }
            if m & L_BASS != 0 {
                b.note(st, 0.0, CH_BASS, bass, vel, len, false);
            }
        };
        let v = [
            voicing_pcs(&pcs, 57, 72, &self.keys_v, false),
            voicing_pcs(&pcs, 55, 70, &self.pad_v, false),
            voicing_pcs(&pcs, 52, 67, &self.sk.piano_v, false),
        ];
        match self.sk.end {
            SKE_HOLD => {
                // held over a roll, then cut off
                chord(b, 0.0, 11.5, 11.5, 0.8, &v);
                b.hit(0.0, 0.0, CH_KICK, 0, 0.8);
                b.hit(0.0, 0.0, CH_CYM, 0, 0.45);
                let mut st = 0.0;
                while st < 12.0 {
                    b.hit(st, 0.0, CH_SNARE, 1, 0.18 + 0.5 * st / 12.0);
                    st += 0.5;
                }
            }
            SKE_CHARLESTON => chord(b, 0.0, 1.5, 1.5, 0.85, &v),
            _ => {}
        }
        let ring: f32 = (16.0 - hit - 4.0).max(3.5);
        chord(b, hit, 4.0, ring, 0.95, &v);
        b.hit(hit, 0.0, CH_KICK, 0, 1.0);
        b.hit(hit, 0.0, CH_SNARE, 0, 0.85);
        b.hit(hit, 0.0, CH_CYM, 0, 0.85);
        if m & L_LEAD != 0 && self.rng.chance(0.4) {
            let at = hit + ring - 1.5;
            for (j, d) in [1, 3, 6, 10].iter().enumerate() {
                b.note(at + 0.3 * j as f32, 0.0, CH_LEAD, tune + lead - d, 0.6 - 0.12 * j as f32, 0.4, true);
            }
        }
    }

    /// Ska's beat in any meter, for a mix: the kick on the one (and the
    /// middle beat of four; on every beat in some pieces), the backbeat on
    /// the snare (two and four; in three, the second and third; in two, the
    /// second), the hats on the beats and on the off-beats (the skank's
    /// place: open there in some pieces, in the B groups, when it is
    /// strong), ghosts before the backbeat in the B groups, a tambourine on
    /// the off-beats when strong; a roll or the toms into each new eight
    /// bars, a "ba-dap" closing four, a crash at a full section's top. Each
    /// layer plays where ska's sense of the section says, or where the
    /// writer's arrangement has its own drums: in an intro the hats and the
    /// cross-stick (and the roll into what follows), in a break the kick and
    /// the cross-stick, quietly.
    fn ska_beat(&mut self, b: &mut Bar, sec: Sec, m: u32) {
        let (bs, beat) = (self.bar_steps(), self.beat_steps());
        let beats = bs / beat;
        let (e, i, h) = (sec.energy, self.sec_bar, self.human);
        let vb = self.var_b();
        let fill = self.fill(sec);
        let full = matches!(sec.kind, MAIN | BUILD | OUTRO);
        // this piece's drummer: four on the floor, the hat open on every
        // off-beat; a third-wave one's punk drive (the kick on every beat,
        // the snare on every off-beat) where the band is in full
        let (four, open) = (self.dr.snare_kind == 1, self.dr.tambourine);
        let punk = self.sk.third && full;
        let r = &mut self.rng;
        let last = (beats - 1) * beat;
        let off = beat / 2;
        let roll = fill == 2 && (full || sec.kind == INTRO);
        let toms = roll && full && r.chance(0.35);
        let cut = |st: u32| roll && st >= last;
        let kick = full || m & L_KICK != 0;
        if kick || sec.kind == BREAK {
            let soft = if kick { 1.0 } else { 0.6 };
            for k in 0..beats {
                let on = k == 0 || four || punk || (beats == 4 && k == 2);
                if on && !(cut(k * beat) && k > 0) {
                    let v = if k == 0 { 0.9 } else { 0.75 } * (0.85 + 0.15 * e) * soft;
                    b.hit((k * beat) as f32, r.range(0.0, h), CH_KICK, 0, v);
                }
            }
        }
        let backs: Vec<u32> = if punk {
            (0..beats).map(|k| k * beat + off).collect()
        } else {
            (1..beats).filter(|&k| beats == 3 || k % 2 == 1).map(|k| k * beat).collect()
        };
        let snare = full || m & L_SNARE != 0;
        for st in backs {
            if cut(st) {
                continue;
            }
            if snare {
                b.hit(st as f32, r.range(0.0, h), CH_SNARE, 0, (0.78 + 0.12 * e) * r.range(0.92, 1.0));
                if vb && r.chance(0.35) {
                    b.hit((st - 1) as f32, r.range(0.0, h), CH_SNARE, 1, r.range(0.13, 0.22));
                }
            } else if matches!(sec.kind, INTRO | BREAK) {
                b.hit(st as f32, r.range(0.0, h), CH_PERC, RIM, 0.5);
            }
        }
        if sec.kind != BREAK || m & (L_HAT | L_OHAT) != 0 {
            for k in 0..beats {
                let at = k * beat;
                if cut(at) && toms {
                    break;
                }
                b.hit(at as f32, r.range(0.0, h), CH_HAT, HAT_C, if k == 0 { 0.5 } else { 0.42 });
                let opened = full && (open || (vb && k % 2 == 1) || e >= 0.9);
                let (kind, v) = if opened { (HAT_O, 0.42) } else { (HAT_C, 0.4) };
                b.hit((at + off) as f32, r.range(0.0, h), CH_HAT, kind, v * r.range(0.9, 1.0));
            }
        }
        if (full && e >= 0.8) || m & L_PERC != 0 {
            for k in 0..beats {
                if !cut(k * beat) {
                    b.hit((k * beat + off) as f32, r.range(0.0, h), CH_PERC, SHAKER, 0.3 * r.range(0.8, 1.0));
                }
            }
        }
        if toms {
            let subs: [u32; 4] = if beat == 6 { [0, 2, 3, 5] } else { [0, 1, 2, 3] };
            for (st, kind) in subs.iter().zip([TOM_HI, TOM_HI, TOM_MID, TOM_LO]) {
                b.hit((last + st) as f32, r.range(0.0, h), CH_PERC, kind, 0.8);
            }
        } else if roll {
            // 16ths, then 32nds over the beat's second half, louder as they go
            let mut st = last as f32;
            while st < bs as f32 {
                let x = (st - last as f32) / beat as f32;
                b.hit(st, r.range(0.0, h), CH_SNARE, 1, 0.25 + 0.55 * x);
                st += if x >= 0.5 && e >= 0.7 { 0.5 } else { 1.0 };
            }
        } else if fill == 1 && full {
            // ba-dap
            b.hit((bs - 2) as f32, r.range(0.0, h), CH_SNARE, 1, 0.4);
            b.hit((bs - 1) as f32, r.range(0.0, h), CH_SNARE, 0, 0.55);
        }
        if full && i == 0 {
            b.hit(0.0, 0.0, CH_CYM, 0, 0.6 + 0.2 * e);
        }
    }

    /// The electric piano's chord, rolled upward.
    fn strum(&mut self, b: &mut Bar, step: f32, c: Harm, steps: f32, vel: f32) {
        let v = self.voicing(c, 50, 74, &self.keys_v, true);
        let gap = self.rng.range(0.006, 0.02);
        let late = self.rng.range(0.0, self.human);
        for (k, &n) in v.iter().enumerate() {
            let vv = vel * self.rng.range(0.85, 1.05);
            b.note(step, late + gap * k as f32, CH_KEYS, n, vv, steps, false);
        }
        self.keys_v = v;
    }

    // -- the mixes

    /// A mix's piece, once its writer has planned it: 60 % of the way to
    /// the drummer's tempo, the drummer's drums and the effects its beat
    /// drives, its patterns in the writer's meter, and in each part's pair
    /// of instruments one of the drummer's (now and then both).
    fn fuse(&mut self, s: &mut Synth) {
        let (Some(g), Some(x)) = (self.guest, self.scratch.as_deref()) else { return };
        self.bpm += 0.6 * (MID_BPM[g] - self.bpm);
        s.dly.target = (self.step_len() * 3.0) as f32;
        let (bs, beats) = (self.bar_steps() as usize, (self.bar_steps() / self.beat_steps()) as usize);
        let compound = self.beat_steps() == 6;
        let r = &mut self.rng;
        // the drums, the second reverb as the drummer has it (techno's
        // rumble, synthwave's gate) and its sidechain's release
        for c in [CH_KICK, CH_SNARE, CH_HAT, CH_PERC, CH_CYM] {
            let (to, from) = (&mut s.ch[c], &x.ch[c]);
            to.drums.clone_from(&from.drums);
            (to.gain, to.pan, to.rev, to.dly, to.rev2) = (from.gain, from.pan, from.rev, from.dly, from.rev2);
            (to.lp_base, to.lp) = (from.lp_base, from.lp);
        }
        (s.r2mode, s.r2gain, s.rev2.fb, s.rev2.damp, s.gate_hold) = (x.r2mode, x.r2gain, x.rev2.fb, x.rev2.damp, x.gate_hold);
        s.sc_rel = x.sc_rel;
        // the leveller more thorough: one genre's levels were set for its own
        // drums (techno's music under ambient's would be 6 dB down)
        (s.m.lk, s.m.lmax) = (0.75, 2.8);
        // the channel the riser, wind, rain, crackle and hiss play on: the
        // drummer's if it has one, else the writer's, else opened
        let fx = &x.ch[CH_FX];
        if fx.gain > 0.0 {
            mix(&mut s.ch[CH_FX], fx.gain, fx.pan, fx.rev, fx.dly, 0.0);
        } else if s.ch[CH_FX].gain == 0.0 {
            mix(&mut s.ch[CH_FX], 0.5, 0.0, 0.4, 0.0, 0.0);
        }
        // the pump under the kick is the drummer's (a techno writer's music
        // ducking under a frame drum's strokes fights its beat), and a drum
        // machine plays the band tight
        for c in [CH_BASS, CH_PAD, CH_ARP, CH_LEAD, CH_BELL, CH_KEYS, CH_DRONE] {
            s.ch[c].duck = 0.0;
        }
        if matches!(g, TECHNO | SYNTHWAVE) {
            self.human = self.human.min(0.003);
        }
        match g {
            AMBIENT => s.rev_wet = (s.rev_wet + 0.15).min(0.5),
            TECHNO => {
                // everything ducking under the kick
                for (c, d) in
                    [(CH_BASS, 0.35), (CH_KEYS, 0.35), (CH_PAD, 0.5), (CH_ARP, 0.25), (CH_LEAD, 0.2), (CH_BELL, 0.2), (CH_DRONE, 0.3)]
                {
                    s.ch[c].duck = d;
                }
            }
            SYNTHWAVE => {
                for (c, d) in [(CH_BASS, 0.25), (CH_PAD, 0.45), (CH_ARP, 0.2), (CH_BELL, 0.2)] {
                    s.ch[c].duck = d;
                }
            }
            LOFI => {
                // the record's crackle and hiss, the tape's wow, a dull top, a swing
                (s.fx.crackle, s.fx.crackle_rate, s.fx.hiss) = (x.fx.crackle, x.fx.crackle_rate, x.fx.hiss);
                (s.m.wow_to, s.m.wrate) = (x.m.wow_to, x.m.wrate);
                s.m.set_lp(x.m.lp + 1000.0);
                for (c, d) in [(CH_BASS, 0.15), (CH_KEYS, 0.2), (CH_LEAD, 0.1), (CH_BELL, 0.1)] {
                    s.ch[c].duck = d;
                }
                if !compound {
                    self.swing = self.swing.max(r.range(0.12, 0.22));
                }
            }
            SKA => {
                // a third-wave drummer a third of the time: the punk drive on a rock kit
                self.sk.third = r.chance(0.33);
                if self.sk.third {
                    s.ch[CH_KICK].drums = ska_kit_punk_kick(r);
                    s.ch[CH_SNARE].drums = ska_kit_punk_snare(r);
                    (s.ch[CH_KICK].gain, s.ch[CH_SNARE].gain) = (0.45, 1.4);
                }
            }
            _ => {}
        }
        // the patterns: a euclidean percussion line (over two bars of a
        // two-beat meter), the boom-bap kicks (16ths of four beats, or of
        // three), lo-fi's snare
        let n = bs * if beats == 2 { 2 } else { 1 };
        let e = euclid(r.pick(&[3, 5, 7]), n, r.below(n));
        self.dr.perc = e.iter().map(|&on| if on { 0.6 } else { 0.0 }).collect();
        self.dr.perc_kind = r.pick(&[RIM, RIM, TOM_MID, SHAKER]);
        let kicks: &[&[u16]] = if beats == 3 {
            &[&[0, 7], &[0, 6, 10], &[0, 3, 8], &[0, 9, 11]]
        } else {
            &[&[0, 7, 10], &[0, 10, 11], &[0, 6, 9], &[0, 3, 10], &[0, 7, 9, 10], &[0, 8, 11, 14]]
        };
        let ka = r.below(kicks.len());
        let kb = (ka + 1 + r.below(kicks.len() - 1)) % kicks.len();
        self.dr.kicks = [kicks[ka].to_vec(), kicks[kb].to_vec()];
        self.dr.snare_kind = if r.chance(0.3) { 1 } else { 0 };
        self.dr.tambourine = r.bit_chance();
        self.period = r.pick(&[8.0, 16.0, 32.0]);
        // the instruments
        for k in 0..self.band.tones.len() {
            let (ch, mut pair, note) = self.band.tones[k];
            let Some(pal) = palette(g, ch) else { continue };
            let orig = s.ch[ch].patch;
            let pick = |r: &mut Rng| levelled(&orig, note, (pal[r.below(pal.len())].1)(r));
            match r.weighted(&[2.0, 2.0, 1.0]) {
                0 => pair[0] = pick(r),
                1 => pair[1] = pick(r),
                _ => pair = [pick(r), pick(r)],
            }
            self.band.tones[k] = (ch, pair, note);
        }
        // the kits: the drummer's own and another, the kick's the drummer's,
        // the others' as often the writer's
        self.band.kits.clear();
        for c in [CH_KICK, CH_SNARE, CH_HAT, CH_PERC, CH_CYM] {
            if s.ch[c].drums.is_empty() {
                continue;
            }
            let pal = match (kit_palette(g, c), kit_palette(self.genre, c)) {
                (Some(a), Some(b)) => {
                    if c != CH_KICK && r.bit_chance() {
                        b
                    } else {
                        a
                    }
                }
                (Some(a), None) | (None, Some(a)) => a,
                (None, None) => continue,
            };
            self.band.kit(r, s, c, pal);
        }
    }

    /// A mix's bar of drums, in the writer's meter. Each beat plays a layer
    /// where its own sense of the section says, and wherever the writer's
    /// arrangement (`m`) has its own drums play one (a techno intro is
    /// drums alone).
    fn drum(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        // the drums' sends as the drummer has them (a techno writer opens
        // its kick's rumble each section)
        if let Some(x) = self.scratch.as_deref() {
            for c in [CH_KICK, CH_SNARE, CH_HAT, CH_PERC, CH_CYM] {
                s.ch[c].rev2 = x.ch[c].rev2;
            }
        }
        match self.guest {
            Some(AMBIENT) => self.ambient_beat(b, sec, m),
            Some(TECHNO) => self.techno_beat(s, b, sec, m),
            Some(SYNTHWAVE) => self.synthwave_beat(b, sec, m),
            Some(LOFI) => self.lofi_beat(s, b, sec, m),
            Some(POLKA) => self.polka_beat(b, sec, m),
            Some(SKA) => self.ska_beat(b, sec, m),
            Some(MARIACHI) => self.mariachi_beat(s, b, sec, m),
            Some(_) => self.tavern_drums(b, sec, m),
            None => {}
        }
    }

    /// Ambient's beat in any meter, soft and sparse: a frame drum on the one
    /// (every other bar but in a full section; halfway too when it is
    /// strong), a shaker in eighths (16ths in the B groups) and the
    /// percussion line, quietly, once it is under way; a roll on the hand
    /// drums closing a group. Nothing in an intro or a break.
    fn ambient_beat(&mut self, b: &mut Bar, sec: Sec, m: u32) {
        let (bs, beat) = (self.bar_steps(), self.beat_steps());
        let beats = bs / beat;
        let (e, i, h) = (sec.energy, self.sec_bar, self.human);
        let vb = self.var_b();
        let fill = self.fill(sec);
        let quiet = matches!(sec.kind, INTRO | BREAK);
        let full = sec.kind == MAIN;
        let r = &mut self.rng;
        let last = (beats - 1) * beat;
        if (!quiet || m & L_KICK != 0) && (full || i % 2 == 0) {
            b.hit(0.0, r.range(0.0, h), CH_KICK, 0, 0.6);
        }
        if full && e >= 0.6 && beats >= 4 {
            b.hit((2 * beat) as f32, r.range(0.0, h), CH_KICK, 0, 0.42);
        }
        if (full && e >= 0.5) || m & (L_HAT | L_OHAT) != 0 {
            for st in (0..bs).step_by(if vb { 1 } else { 2 }) {
                let v = if st % beat == 0 { 0.4 } else { 0.22 };
                b.hit(st as f32, r.range(0.0, h), CH_HAT, HAT_C, v * r.range(0.7, 1.0));
            }
        }
        if (full && (e >= 0.7 || vb)) || m & L_PERC != 0 {
            let span = (self.dr.perc.len() as u32 / bs).max(1);
            let off = (i % span) * bs;
            for st in 0..bs {
                let v = self.dr.perc.get((off + st) as usize).copied().unwrap_or(0.0);
                if v > 0.0 && !(fill == 2 && st >= last) {
                    b.hit(st as f32, r.range(0.0, h), CH_PERC, self.dr.perc_kind, 0.5 * v * r.range(0.8, 1.0));
                }
            }
        }
        if fill == 2 && full {
            for (k, st) in (last..bs).enumerate() {
                let kind = if k % 2 == 0 { TOM_MID } else { TOM_LO };
                b.hit(st as f32, r.range(0.0, h), CH_PERC, kind, 0.22 + 0.25 * k as f32 / beat as f32);
            }
        }
    }

    /// Synthwave's beat in any meter: the kick on the odd beats (one and
    /// three; in three, the one), the gated snare on the even ones (in
    /// three, the last), hats in eighths (16ths when strong or in the B
    /// groups) with an open one at the bar's end, an extra kick and a ghost
    /// snare in the B groups, a rim on the last off-beat; toms or a snare run
    /// closing a group, a crash at a full section's top. The kick and hats
    /// alone in an intro, the hats in a break.
    fn synthwave_beat(&mut self, b: &mut Bar, sec: Sec, m: u32) {
        let (bs, beat) = (self.bar_steps(), self.beat_steps());
        let beats = bs / beat;
        let (e, i) = (sec.energy, self.sec_bar);
        let vb = self.var_b();
        let fill = self.fill(sec);
        let full = sec.kind == MAIN;
        let (big, small) = (fill == 2 && full, fill == 1 && full);
        let r = &mut self.rng;
        let last = (beats - 1) * beat;
        let kick = sec.kind != BREAK || m & L_KICK != 0;
        let snare = matches!(sec.kind, MAIN | BUILD | OUTRO) || m & L_SNARE != 0;
        let sixteens = full && (vb || e >= 0.85);
        let hats: &[(u32, f32)] = match (beat, sixteens) {
            (6, false) => &[(0, 0.45), (2, 0.3), (4, 0.35)],
            (6, true) => &[(0, 0.45), (1, 0.2), (2, 0.3), (3, 0.22), (4, 0.35), (5, 0.2)],
            (_, false) => &[(0, 0.5), (2, 0.38)],
            _ => &[(0, 0.45), (1, 0.25), (2, 0.4), (3, 0.25)],
        };
        for k in 0..beats {
            let at = k * beat;
            let back = if beats == 3 { k == 2 } else { (i * beats + k) % 2 == 1 };
            let on = if beats == 3 { k == 0 } else { !back };
            if kick && on && !(big && at >= last) {
                b.hit(at as f32, 0.0, CH_KICK, 0, if k == 0 { 1.0 } else { 0.9 });
            }
            if snare && back && !(big && at >= last) {
                b.hit(at as f32, 0.0, CH_SNARE, 0, 0.95);
            }
            for &(sd, v) in hats {
                b.hit((at + sd) as f32, 0.0, CH_HAT, HAT_C, v * r.range(0.85, 1.0));
            }
        }
        if full && vb {
            b.hit((last - beat / 2) as f32, 0.0, CH_KICK, 0, 0.75);
            if !big {
                b.hit((bs - 1) as f32, 0.0, CH_SNARE, 1, 0.22);
            }
        }
        if full && e >= 0.8 {
            b.hit((last + beat / 2) as f32, 0.0, CH_PERC, RIM, 0.5);
            if !small {
                b.hit((bs - 2) as f32, 0.0, CH_HAT, HAT_O, 0.45);
            }
        }
        if big {
            if r.chance(0.6) {
                let subs: [u32; 4] = if beat == 6 { [0, 2, 3, 5] } else { [0, 1, 2, 3] };
                for (st, kind) in subs.iter().zip([TOM_HI, TOM_HI, TOM_MID, TOM_LO]) {
                    b.hit((last + st) as f32, 0.0, CH_PERC, kind, 0.85);
                }
            } else {
                for st in last..bs {
                    b.hit(st as f32, 0.0, CH_SNARE, 1, 0.5 + 0.4 * (st - last) as f32 / beat as f32);
                }
            }
        } else if small {
            b.hit((bs - 1) as f32, 0.0, CH_SNARE, 1, 0.35);
            b.hit((bs - 2) as f32, 0.0, CH_HAT, HAT_O, 0.4);
        }
        if full && i % 8 == 0 {
            b.hit(0.0, 0.0, CH_CYM, 0, if i == 0 { 0.75 } else { 0.5 });
        }
    }

    /// Polka's beat in any meter: oom on the beat (the bass drum), pah off it
    /// (the snare on the off-eighth; in a compound meter the second and third
    /// eighths; in three, oom-pah-pah), the tambourine with the pahs when it
    /// is strong (or the writer has hats); a roll into each new section, a
    /// crash at a full one's top. The ooms alone in an intro, the pahs alone
    /// in a break.
    fn polka_beat(&mut self, b: &mut Bar, sec: Sec, m: u32) {
        let (bs, beat) = (self.bar_steps(), self.beat_steps());
        let beats = bs / beat;
        let (e, i, h) = (sec.energy, self.sec_bar, self.human);
        let fill = self.fill(sec);
        let full = sec.kind == MAIN;
        let waltz = beats == 3 && beat == 4;
        let r = &mut self.rng;
        let last = (beats - 1) * beat;
        let roll = fill == 2 && (full || sec.kind == BUILD) && r.chance(0.75);
        if sec.kind != BREAK || m & L_KICK != 0 {
            for k in 0..beats {
                if !(waltz && k > 0) {
                    let v = if k == 0 { 0.8 } else { 0.7 } + 0.15 * e;
                    b.hit((k * beat) as f32, r.range(0.0, h), CH_KICK, 0, v);
                }
            }
        }
        let pah = sec.kind != INTRO || m & L_SNARE != 0;
        let tambourine = (self.dr.tambourine && e > 0.75 && full) || m & (L_HAT | L_OHAT) != 0;
        let quiet = if sec.kind == BREAK { 0.6 } else { 1.0 };
        for k in 0..beats {
            let at = k * beat;
            let pahs: &[u32] = match (beat, waltz) {
                (6, _) => &[2, 4],
                (_, true) if k == 0 => &[],
                (_, true) => &[0],
                _ => &[2],
            };
            for &p in pahs {
                let st = at + p;
                if roll && st >= last {
                    continue;
                }
                if pah {
                    b.hit(st as f32, r.range(0.0, h), CH_SNARE, 0, (0.35 + 0.35 * e) * quiet);
                }
                if tambourine {
                    b.hit(st as f32, r.range(0.0, h), CH_HAT, HAT_C, 0.4);
                }
            }
        }
        if roll {
            for k in 0..2 * beat {
                b.hit(last as f32 + 0.5 * k as f32, 0.0, CH_SNARE, 1, 0.3 + 0.55 * k as f32 / (2 * beat) as f32);
            }
        }
        if full && i == 0 {
            b.hit(0.0, 0.0, CH_CYM, 0, 0.6 + 0.2 * e);
        }
    }

    /// Tavern's beat in any meter, a bodhran's: the frame drum's low stroke on
    /// the beat and lighter ones between (three to a beat in a compound
    /// meter; one a beat in three), a ghost in the B groups, a shaker off the
    /// beat once it is under way, bones in eighths when it is strong, claps
    /// at its height; loosely timed. The drum alone on the one in an intro,
    /// softly in a break.
    fn tavern_drums(&mut self, b: &mut Bar, sec: Sec, m: u32) {
        let (bs, beat) = (self.bar_steps(), self.beat_steps());
        let beats = bs / beat;
        let (e, h) = (sec.energy, self.human);
        let vb = self.var_b();
        let waltz = beats == 3 && beat == 4;
        let r = &mut self.rng;
        if sec.kind == INTRO && m & L_KICK == 0 {
            b.hit(0.0, r.range(0.0, h), CH_KICK, 0, 0.7);
            return;
        }
        let quiet = if sec.kind == BREAK && m & L_KICK == 0 { 0.6 } else { 1.0 };
        for k in 0..beats {
            let at = (k * beat) as f32;
            let v = quiet * if k == 0 { 0.9 } else { 0.72 };
            if beat == 6 {
                b.hit(at, r.range(0.0, h), CH_KICK, 0, v);
                b.hit(at + 2.0, r.range(0.0, h), CH_KICK, 1, 0.42 * quiet);
                b.hit(at + 4.0, r.range(0.0, h), CH_KICK, 1, 0.5 * quiet);
                if vb && r.chance(0.35) {
                    b.hit(at + 3.0, r.range(0.0, h), CH_KICK, 1, 0.3 * quiet);
                }
            } else if waltz {
                b.hit(at, r.range(0.0, h), CH_KICK, if k == 0 { 0 } else { 1 }, v);
            } else {
                let strong = k % 2 == 0;
                b.hit(at, r.range(0.0, h), CH_KICK, if strong { 0 } else { 1 }, if strong { v } else { 0.6 * quiet });
                b.hit(at + 2.0, r.range(0.0, h), CH_KICK, 1, 0.4 * quiet);
                if vb && r.chance(0.3) {
                    b.hit(at + 3.0, r.range(0.0, h), CH_KICK, 1, 0.28 * quiet);
                }
            }
        }
        if (sec.kind != BREAK && e >= 0.6) || m & (L_HAT | L_OHAT) != 0 {
            for k in 0..beats {
                let at = (k * beat) as f32;
                b.hit(at + 2.0, r.range(0.0, h), CH_HAT, HAT_C, 0.35);
                if beat == 6 {
                    b.hit(at + 4.0, r.range(0.0, h), CH_HAT, HAT_C, 0.3);
                }
                if k % 2 == 1 {
                    b.hit(at, r.range(0.0, h), CH_HAT, HAT_O, 0.45);
                }
            }
        }
        if (sec.kind == MAIN && e >= 0.75) || m & L_PERC != 0 {
            for st in (0..bs).step_by(2) {
                let on_beat = st % beat == 0;
                if on_beat || r.chance(0.7) {
                    b.hit(st as f32, r.range(0.0, h), CH_PERC, 0, if on_beat { 0.55 } else { 0.32 });
                }
            }
        }
        if sec.kind == MAIN && e >= 0.9 {
            for k in 1..beats {
                if beat == 6 || k % 2 == 1 {
                    b.hit((k * beat) as f32, r.range(0.0, h), CH_SNARE, 0, 0.6);
                }
            }
        }
    }

    /// Techno's beat in any meter: the kick on every beat, claps on every
    /// other, hats in 16ths with an open one between the beats, the
    /// percussion line, all busier in the B groups; a clap run closing a
    /// group, a roll into a full section, a crash at its top, a riser over
    /// the bars before it, the kick's rumble when everything plays, and the
    /// bass's and chords' filters sweeping as acid's do.
    fn techno_beat(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let (bs, beat) = (self.bar_steps(), self.beat_steps());
        let beats = bs / beat;
        let (e, i) = (sec.energy, self.sec_bar);
        let left = sec.bars - i;
        let vb = self.var_b();
        let fill = self.fill(sec);
        let next_main = self.plan.get(self.sec + 1).is_some_and(|x| x.kind == MAIN);
        let full = sec.kind == MAIN;
        self.phase = (self.phase + 1.0 / self.period).fract();
        let sweep = sin01(self.phase);
        s.ch[CH_BASS].cut = (1.2 * sweep - 0.3 + 0.5 * self.walk[0]).exp2();
        s.ch[CH_KEYS].cut = (0.6 * self.walk[2] + 0.4 * sweep).exp2();
        s.ch[CH_KICK].rev2 = if full && e >= 0.85 { 0.6 } else { 0.0 };
        let rolling = next_main && !full && left <= 2;
        let big = fill == 2 && full;
        let r = &mut self.rng;
        let drop = big && r.chance(0.35);
        // the last beat's first step
        let last = (beats - 1) * beat;
        let hats: &[f32] = match (beat, vb) {
            (6, _) => &[0.3, 0.18, 0.75, 0.2, 0.55, 0.2],
            (_, false) => &[0.3, 0.2, 0.8, 0.22],
            _ => &[0.32, 0.26, 0.8, 0.28],
        };
        for k in 0..beats {
            let at = k * beat;
            if (sec.kind != BREAK || m & L_KICK != 0) && !(drop && at >= last) {
                b.hit(at as f32, 0.0, CH_KICK, 0, if k == 0 { 1.0 } else { 0.9 });
            }
            let claps = full || sec.kind == BUILD || m & L_SNARE != 0;
            if claps && (i * beats + k) % 2 == 1 && !rolling && !(big && at >= last) {
                b.hit(at as f32, 0.0, CH_SNARE, 0, 0.9);
            }
            if sec.kind != BREAK || vb || m & (L_HAT | L_OHAT) != 0 {
                for (sub, &v) in hats.iter().enumerate() {
                    let main = sub == 2 || (beat == 6 && sub == 4);
                    if main || vb || r.chance(0.15 + 0.75 * e) {
                        b.hit((at + sub as u32) as f32, 0.0, CH_HAT, HAT_C, v * r.range(0.8, 1.0));
                    }
                }
            }
            if full && e >= 0.8 && !(fill == 1 && k + 1 == beats) {
                b.hit((at + if beat == 6 { 3 } else { 2 }) as f32, 0.0, CH_HAT, HAT_O, 0.5);
            }
        }
        if (full && (e >= 0.85 || vb)) || m & L_PERC != 0 {
            let span = (self.dr.perc.len() as u32 / bs).max(1);
            let off = (i % span) * bs;
            for st in 0..bs {
                let v = self.dr.perc.get((off + st) as usize).copied().unwrap_or(0.0);
                if v > 0.0 {
                    b.hit(st as f32, 0.0, CH_PERC, self.dr.perc_kind, v * r.range(0.8, 1.0));
                }
            }
        }
        if big {
            for st in last..bs {
                b.hit(st as f32, 0.0, CH_SNARE, 1, 0.3 + 0.5 * (st - last) as f32 / beat as f32);
            }
        } else if fill == 1 && full {
            if r.chance(0.5) {
                for (k, st) in (bs - 3..bs).enumerate() {
                    b.hit(st as f32, 0.0, CH_SNARE, 1, 0.25 + 0.1 * k as f32);
                }
            } else {
                b.hit((bs - 1) as f32, 0.0, CH_HAT, HAT_O, 0.5);
            }
        }
        if rolling {
            // into the drop: eighths, then 16ths, louder as they go
            let div = if left == 2 { 2 } else { 1 };
            for st in (0..bs).step_by(div) {
                let x = ((2 - left) * bs + st) as f32 / (2 * bs) as f32;
                b.hit(st as f32, 0.0, CH_SNARE, 1, 0.25 + 0.65 * x);
            }
        }
        if full && i % 8 == 0 {
            b.hit(0.0, 0.0, CH_CYM, 0, if i == 0 { 0.8 } else { 0.55 });
        }
        if next_main && !full && left == sec.bars.min(4) {
            b.ev.push((b.at(0.0, 0.0), Ev::Riser { secs: b.secs((bs * left) as f32), vel: 0.45 }));
        }
    }

    /// Lo-fi's beat in any meter: the boom-bap kick (its A and B patterns,
    /// over two bars of a two-beat meter), the snare on every other beat (in
    /// three, the last) with ghosts before it in the B groups, eighths of
    /// hats with 16ths among them in the B groups, a shaker when everything
    /// plays; the drums out for the last beat or a pickup closing a group.
    /// No drums in an intro, only the hats in a break.
    fn lofi_beat(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let (bs, beat) = (self.bar_steps(), self.beat_steps());
        let beats = bs / beat;
        let (e, i, h) = (sec.energy, self.sec_bar, self.human);
        let vb = self.var_b();
        let fill = self.fill(sec);
        s.ch[CH_KEYS].cut = (0.5 * self.walk[0]).exp2();
        if sec.kind == INTRO && m & (L_KICK | L_SNARE | L_HAT | L_OHAT | L_PERC) == 0 {
            return;
        }
        let r = &mut self.rng;
        let gap = fill == 2 && r.chance(0.5);
        let last = (beats - 1) * beat;
        let drums = matches!(sec.kind, MAIN | OUTRO | BUILD) || m & L_KICK != 0;
        // a 16th of a quarter to a step of this beat
        let sub = |q: u32| if beat == 6 { [0, 2, 4, 5][q as usize] } else { q };
        if drums {
            let span = if beats == 2 { 2 } else { 1 };
            for &p in &self.dr.kicks[vb as usize] {
                let (q, s16) = (p as u32 / 4, p as u32 % 4);
                let st = (q % beats) * beat + sub(s16);
                if q / beats == i % span && !(fill > 0 && st >= last && p != 0) {
                    let v = if p == 0 { 1.0 } else { 0.8 };
                    b.hit(st as f32, r.range(0.0, h * 0.3), CH_KICK, 0, v * r.range(0.9, 1.0));
                }
            }
        }
        if (drums && (e >= 0.6 || sec.kind != MAIN)) || m & L_SNARE != 0 {
            let backs: &[u32] = match beats {
                3 => &[2],
                4 => &[1, 3],
                _ => &[1],
            };
            for &k in backs.iter().filter(|&&k| k < beats) {
                let at = k * beat;
                if !(gap && at >= last) {
                    let v = 0.88 * r.range(0.85, 1.0);
                    b.hit(at as f32, r.range(0.004, h + 0.004), CH_SNARE, self.dr.snare_kind, v);
                }
                if vb {
                    b.hit((at - 1) as f32, r.range(0.0, h), CH_SNARE, 0, r.range(0.14, 0.22));
                }
            }
            if fill == 1 || (fill == 2 && !gap) {
                for (k, st) in (bs - 3..bs).enumerate() {
                    b.hit(st as f32, r.range(0.0, h), CH_SNARE, 0, 0.14 + 0.06 * k as f32);
                }
            }
        }
        let hats: &[(u32, f32)] = match (beat, vb) {
            (6, false) => &[(0, 0.5), (2, 0.3), (4, 0.4)],
            (6, true) => &[(0, 0.5), (2, 0.3), (3, 0.2), (4, 0.4), (5, 0.22)],
            (_, false) => &[(0, 0.55), (2, 0.4)],
            _ => &[(0, 0.5), (1, 0.22), (2, 0.38), (3, 0.26)],
        };
        for k in 0..beats {
            for &(sd, v) in hats {
                let st = k * beat + sd;
                if !(gap && st >= last) {
                    b.hit(st as f32, r.range(0.0, h), CH_HAT, HAT_C, v * r.range(0.7, 1.0));
                }
            }
        }
        if vb || gap {
            b.hit((last + beat / 2) as f32, r.range(0.0, h), CH_HAT, HAT_O, 0.4);
        }
        if (sec.kind == MAIN && e >= 0.75) || m & L_PERC != 0 {
            for st in (1..bs).step_by(2) {
                if r.chance(0.8) {
                    b.hit(st as f32, r.range(0.0, h), CH_PERC, SHAKER, 0.3 * r.range(0.6, 1.0));
                }
            }
        }
    }

    // -- mariachi

    /// A mariachi piece: its style (and so its meter and tempo), its strains
    /// (verse, chorus, the contrasting part, the fanfare) with their tunes,
    /// and its form: the fanfare; verses and choruses, AABB or ABAB, the
    /// fanfare's end again between them; the contrasting part; the last
    /// chorus with everyone; the ending. A minute and a half to two and a half.
    fn piece_mariachi(&mut self, s: &mut Synth) {
        let center = self.rng.pick(&[67, 69, 70, 72]);
        let tonic = self.tonic_near(center);
        let minor = self.mode != IONIAN;
        let mode = if minor { MARIACHI_MINOR } else { IONIAN };
        self.mode = mode;
        let r = &mut self.rng;
        let style = r.weighted(&[3.0, 2.0, 2.0, 2.0, 2.0]) as u8;
        let (bs, beat, bpm, meter) = match style {
            MARIACHI_SON => (12u32, 6u32, r.range(108.0, 132.0), "6/8 son jalisciense"),
            MARIACHI_HUAPANGO => (12, 6, r.range(88.0, 108.0), "6/8 huapango"),
            MARIACHI_VALS => (12, 4, r.range(115.0, 145.0), "3/4 ranchera valseada"),
            MARIACHI_POLKA => (8, 4, r.range(125.0, 145.0), "2/4 ranchera polkeada"),
            _ => (16, 4, r.range(70.0, 88.0), "4/4 bolero ranchero"),
        };
        self.bpm = bpm as f64;
        self.swing = 0.0;
        self.human = 0.008;
        // the son's bars in 3/4 against its 6/8: alternating (the
        // sesquialtera), two and two, or the cadence's alone; a huapango's
        // cadences now and then
        let hemi: u8 = match style {
            MARIACHI_SON => r.pick(&[0b1010, 0b1010, 0b1000, 0b1100]),
            MARIACHI_HUAPANGO if r.chance(0.5) => 0b1000,
            _ => 0,
        };
        let bar_s = bs as f64 / beat as f64 * 60.0 / self.bpm;
        // strains of sixteen bars, eight for the bolero's long ones
        let lb: usize = if bar_s < 1.8 { 16 } else { 8 };
        let lc = if lb == 16 && r.chance(0.4) { 8 } else { lb };
        let lf = lb / 2;
        let (pa, pb, pc, pf) = if minor {
            (MARIACHI_MIN_A, MARIACHI_MIN_B, MARIACHI_MIN_C, MARIACHI_FANFARE_MIN)
        } else {
            (MARIACHI_MAJ_A, MARIACHI_MAJ_B, MARIACHI_MAJ_C, MARIACHI_FANFARE)
        };
        let dom = Harm { root: 7, q: Q_DOM7, inv: 0 };
        let polkeada = style == MARIACHI_POLKA;
        // a strain's chords: a plan, or two (the first closing on the
        // dominant); a polkeada's a chord a bar
        let chords_of = |r: &mut Rng, pool: &[&str], bars: usize| {
            let mut v = halves(r.pick(pool));
            if bars == 16 {
                let n = v.len();
                v[n - 2] = dom;
                v[n - 1] = dom;
                v.extend(halves(r.pick(pool)));
            }
            if polkeada {
                for k in (0..v.len()).step_by(2) {
                    v[k + 1] = v[k];
                }
            }
            v
        };
        let a = chords_of(r, pa, lb);
        let mut bc = chords_of(r, pb, lb);
        for _ in 0..4 {
            if bc != a {
                break;
            }
            bc = chords_of(r, pb, lb);
        }
        let c = chords_of(r, pc, lc);
        let mut f = halves(r.pick(pf));
        if lf == 8 {
            f.extend(halves(r.pick(pf)));
        }
        let mut strains = Vec::with_capacity(4);
        for (k, (chords, fanfare)) in [(a, false), (bc, false), (c, false), (f, true)].into_iter().enumerate() {
            let (mel, ans) = if polkeada && !fanfare {
                // the polkeada's tunes are a polka's
                let per_bar: Vec<Harm> = chords.iter().step_by(2).copied().collect();
                let (mut mel, ans) = polka_tune(r, &per_bar, mode);
                polka_graces(r, &mut mel, 0.06);
                (mel, ans)
            } else {
                mariachi_tune(r, style, hemi, &chords, bs as u16, beat as u16, mode, fanfare, k == 1)
            };
            strains.push(MariachiStrain { chords, mel, ans });
        }
        // the form: (section, its part, how soon it goes when the piece runs long: 0 never)
        let rs = L_KEYS | L_PAD | L_BASS;
        let tunes = L_LEAD | L_ARP;
        let harp = if r.chance(if style == MARIACHI_BOLERO { 1.0 } else { 0.6 }) { L_BELL } else { 0 };
        let zap = if matches!(style, MARIACHI_SON | MARIACHI_HUAPANGO) { L_KICK } else { 0 };
        let claps = match style {
            MARIACHI_SON | MARIACHI_POLKA => L_SNARE,
            MARIACHI_HUAPANGO if r.chance(0.4) => L_SNARE,
            _ => 0,
        };
        let shake = match style {
            MARIACHI_BOLERO => r.pick(&[L_HAT, L_HAT | L_PERC, 0]),
            MARIACHI_VALS if r.chance(0.3) => L_PERC,
            _ => 0,
        };
        let (lb, lc, lf) = (lb as u32, lc as u32, lf as u32);
        let li = lb / 4;
        let mk = |kind: u8, bars: u32, mask: u32, energy: f32| Sec { kind, bars, mask, energy };
        let part = |strain: usize, rep: u32, first: u32, role: u8, dress: usize| MariachiSection {
            strain,
            rep,
            first,
            role,
            dress,
        };
        let intro_role = if r.chance(0.65) { MARIACHI_TRUMPETS } else { MARIACHI_VIOLINS };
        let verse2 = if r.chance(0.35) { MARIACHI_TRADE } else { MARIACHI_VIOLINS };
        let chorus = if r.chance(0.6) { MARIACHI_TRUMPETS } else { MARIACHI_TRADE };
        let bridge = if r.chance(0.7) { MARIACHI_VIOLINS } else { MARIACHI_TRUMPETS };
        let intro = (mk(INTRO, lf, rs | tunes | L_CTR | harp, 0.65), part(3, 0, 0, intro_role, 1), 0u8);
        let a0 = (mk(MAIN, lb, rs | tunes | L_DRONE | harp, 0.6), part(0, 0, 0, MARIACHI_VIOLINS, 0), 0);
        let a1 = (mk(MAIN, lb, rs | tunes | L_CTR | harp, 0.7), part(0, 1, 0, verse2, 0), 4);
        let inter = |rep: u32, drop: u8| {
            (mk(BUILD, li, rs | tunes | L_CTR | harp, 0.8), part(3, rep, lf - li, MARIACHI_TRUMPETS, 1), drop)
        };
        let full = rs | tunes | L_CTR | L_DRONE | harp;
        let b0 = (mk(MAIN, lb, full | zap | (shake & L_HAT), 0.85), part(1, 0, 0, chorus, 1), 0);
        let b1 = (mk(MAIN, lb, full | zap | shake, 0.9), part(1, 1, 0, MARIACHI_TUTTI, 1), 2);
        let cc = (mk(MAIN, lc, full, 0.6), part(2, 0, 0, bridge, 2), 0);
        let b2 = (mk(MAIN, lb, full | zap | claps | shake, 1.0), part(1, 2, 0, MARIACHI_TUTTI, 1), 0);
        let outro = (mk(OUTRO, 2, full | zap | claps, 1.0), part(1, 3, 0, MARIACHI_TUTTI, 1), 0);
        let mut plan = if r.chance(0.55) {
            vec![intro, a0, a1, inter(1, 0), b0, b1, cc, inter(2, 3), b2, outro]
        } else {
            vec![intro, a0, b0, inter(1, 0), a1, b1, cc, b2, outro]
        };
        let secs = |p: &[(Sec, MariachiSection, u8)]| p.iter().map(|x| x.0.bars).sum::<u32>() as f64 * bar_s;
        while secs(&plan) > 150.0 {
            let Some(k) = (0..plan.len()).filter(|&k| plan[k].2 > 0).max_by_key(|&k| plan[k].2) else { break };
            plan.remove(k);
        }
        if secs(&plan) < 90.0 {
            let verse3 = (mk(MAIN, lb, rs | tunes | L_CTR | harp, 0.75), part(0, 2, 0, MARIACHI_TRADE, 0), 0);
            let k = plan.len() - 2;
            plan.insert(k, verse3);
        }
        self.plan = plan.iter().map(|x| x.0).collect();
        // the trumpets an octave down if the tunes climb past their comfort
        let top = strains.iter().flat_map(|s| s.mel.iter().chain(&s.ans)).map(|n| n.pitch).max().unwrap_or(12);
        self.mc = Mariachi {
            bar_steps: bs,
            beat_steps: beat,
            meter,
            style,
            hemi,
            strains,
            map: plan.iter().map(|x| x.1).collect(),
            tonic,
            tr_oct: if tonic + top > 84 { -12 } else { 0 },
            button: r.chance(match style {
                MARIACHI_BOLERO => 0.0,
                MARIACHI_VALS => 0.4,
                _ => 0.6,
            }),
            call: if r.chance(0.4) { lf / 4 } else { 0 },
            ..Mariachi::default()
        };
        sounds_mariachi(r, s);
        // the slow styles' guitarrón (a note or three a bar) louder, its low
        // end as full as the others'
        s.ch[CH_BASS].gain *= match style {
            MARIACHI_VALS => 1.3,
            MARIACHI_BOLERO => 1.25,
            _ => 1.0,
        };
        self.band = band_mariachi(r, s);
        s.dly.target = (self.step_len() * 3.0) as f32;
    }

    /// A mariachi section begins: its tune this time through (on a repeat a
    /// bar figured anew, the answers turned round; ornaments, more each
    /// time), who leads, a third or a sixth under the tune, which mánico.
    fn mariachi_section(&mut self) {
        let sec = self.plan[self.sec];
        let ms = self.mc.map[self.sec];
        let (bs, beat) = (self.mc.bar_steps as u16, self.mc.beat_steps as u16);
        let (style, hemi, mode) = (self.mc.style, self.mc.hemi, self.mode);
        let st = self.mc.strains[ms.strain].clone();
        let r = &mut self.rng;
        let (mut mel, mut ans) = (st.mel, st.ans);
        let bars = (st.chords.len() / 2) as u16;
        if ms.rep > 0 && sec.kind == MAIN && r.chance(0.75) {
            let bar = r.pick(&[1u16, 2, 5, 6]) + if bars > 8 && r.bit_chance() { 8 } else { 0 };
            let prev = mel.iter().filter(|n| n.step / bs < bar).last().map_or(4, |n| n.pitch);
            let chord_at = |s0: u16| st.chords[bar as usize * 2 + (s0 as usize * 2 / bs as usize).min(1)];
            let fig = if style == MARIACHI_POLKA {
                let kind = r.pick(&MID_FIGS);
                polka_figure(r, kind, chord_at(0), mode, prev, None, PK_LO, PK_HI)
            } else {
                let hb = (hemi >> (bar % 4)) & 1 == 1;
                let three = hb || (bs == 12 && beat == 4);
                let tpl = r.pick(mariachi_rhythms(style, bs, three, false, false));
                let mut dir = if r.bit_chance() { 1 } else { -1 };
                let leaps = style == MARIACHI_HUAPANGO;
                mariachi_figure(r, tpl, &chord_at, mode, prev, &mut dir, if three { 4 } else { beat }, false, None, leaps)
            };
            mel.retain(|n| n.step / bs != bar);
            mel.extend(fig.iter().map(|n| PNote { step: n.step + bar * bs, ..*n }));
            mel.sort_by_key(|n| n.step);
        }
        if ms.rep % 2 == 1 {
            // each bar's answer turned round (a run up now runs down)
            let mut k = 0;
            while k < ans.len() {
                let bar = ans[k].step / bs;
                let end = (k..ans.len()).find(|&j| ans[j].step / bs != bar).unwrap_or(ans.len());
                let pitches: Vec<i32> = ans[k..end].iter().map(|n| n.pitch).rev().collect();
                for (n, p) in ans[k..end].iter_mut().zip(pitches) {
                    n.pitch = p;
                }
                k = end;
            }
        }
        mariachi_ornaments(r, &mut mel, style, 0.06 + 0.05 * ms.rep.min(3) as f32);
        self.mc.role = ms.role;
        self.mc.harm = if sec.mask & L_CTR != 0 { r.pick(&[2, 2, 5]) } else { 0 };
        self.mc.alt = r.bit_chance();
        self.mc.mel = mel;
        self.mc.ans = ans;
        self.mc.sus_v.clear();
    }

    /// A bar of mariachi: the vihuela's and guitar's mánico, the
    /// guitarrón's roots and fifths (walking into a new chord at a phrase's
    /// end), the harp, the tune (violins or trumpets leading, a third or a
    /// sixth under it, the other section answering), the second violins
    /// holding the harmony, the zapateado, palmas, maracas or guiro where
    /// asked; at the end, the closing figure and the last chord.
    fn mariachi(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let ms = self.mc.map[self.sec];
        let i = self.sec_bar;
        let last = i + 1 == sec.bars;
        let minor = MODES[self.mode][2] == 3;
        let home = Harm { root: 0, q: if minor { Q_MIN } else { Q_MAJ }, inv: 0 };
        let dom = Harm { root: 7, q: Q_DOM7, inv: 0 };
        let (chords, next, bi, hold) = {
            let ch = &self.mc.strains[ms.strain].chords;
            let nb = (ch.len() / 2) as u32;
            let bi = (i + ms.first) % nb;
            let at = |k: u32| ch[(k % (2 * nb)) as usize];
            let chords = match sec.kind {
                OUTRO if i == 0 => [dom, dom],
                OUTRO => [home, home],
                _ => [at(2 * bi), at(2 * bi + 1)],
            };
            let next = if sec.kind == OUTRO {
                home
            } else if !last {
                at(2 * bi + 2)
            } else {
                match self.mc.map.get(self.sec + 1) {
                    Some(n) if self.plan[self.sec + 1].kind != OUTRO => {
                        self.mc.strains[n.strain].chords[2 * n.first as usize]
                    }
                    Some(_) => dom,
                    None => home,
                }
            };
            // how many half bars each of the bar's chords is held from here
            // (0: held over from before)
            let hold = [0u32, 1].map(|hh| {
                let k = 2 * bi + hh;
                if !((i == 0 && hh == 0) || at(k + 2 * nb - 1) != at(k)) {
                    return 0;
                }
                let left = 2 * (sec.bars - i) - hh;
                let mut n = 1;
                while n < 8 && n < left && at(k + n) == at(k) {
                    n += 1;
                }
                n
            });
            (chords, next, bi, hold)
        };
        self.mc.mask = [chords[0].mask(self.root), chords[1].mask(self.root)];
        let hemi = (self.mc.hemi >> (bi % 4)) & 1 == 1;
        if sec.kind == OUTRO && last {
            self.mariachi_ending(s, b, m, home);
            return;
        }
        let vb = self.var_b();
        let alt = self.mc.alt != vb;
        let strength = 0.85 + 0.15 * sec.energy;
        // the fanfare's call, alone, before the band comes in
        let m = if sec.kind == INTRO && i < self.mc.call { m & !(L_KEYS | L_PAD | L_BASS | L_BELL) } else { m };
        if m & L_KEYS != 0 {
            self.mariachi_strums(s, b, CH_KEYS, chords, hemi, alt, last && sec.kind != OUTRO, strength);
        }
        if m & L_PAD != 0 {
            self.mariachi_strums(s, b, CH_PAD, chords, hemi, alt, false, strength);
        }
        if m & L_BASS != 0 {
            let walk = last || bi % 4 == 3 || sec.kind == OUTRO;
            self.mariachi_bass(b, chords, next, hemi, vb, walk, hold[0] == 0);
        }
        if m & L_BELL != 0 {
            // a bolero's requinto runs from an eighth after the tune's last
            // note in the bar, where it holds and nothing answers it
            let bs = self.mc.bar_steps;
            let run = if self.mc.style == MARIACHI_BOLERO && sec.kind != OUTRO {
                let notes = self.mc.mel.iter().chain(&self.mc.ans);
                let last = notes.filter(|n| n.step as u32 / bs == bi).map(|n| n.step as u32 % bs).max();
                last.map_or(bs / 2, |l| (l / 2 + 1) * 2)
            } else {
                bs
            };
            self.mariachi_harp(b, chords, hemi, vb, run);
        }
        if sec.kind == OUTRO {
            self.mariachi_closing(s, b, m, dom);
        } else {
            self.mariachi_lines(s, b, sec, m, chords, bi, hold);
        }
        let fill = if matches!(sec.kind, MAIN | BUILD) { self.fill(sec) } else { 0 };
        self.mariachi_hits(b, m & (L_KICK | L_SNARE | L_HAT | L_PERC), hemi, fill, sec.energy, 1.0);
    }

    /// The vihuela's chord (its five strings within A3-E5, any inversion) or
    /// the guitar's (from the root on its low strings up, six at most).
    fn mariachi_voicing(&self, c: Harm, guitar: bool) -> Vec<i32> {
        let pcs = self.pcs(c);
        if guitar {
            let lo = 40 + (self.root + c.root - 40).rem_euclid(12);
            let mut v = in_range(&pcs, lo, lo + 19);
            v.truncate(6);
            v
        } else {
            let mut v = in_range(&pcs, 57, 76);
            let n = v.len();
            if n > 5 {
                v.drain(..n - 5);
            }
            v
        }
    }

    /// A stroke on `ch` across voicing `v` (low to high), ringing `len` steps.
    fn mariachi_stroke(&mut self, b: &mut Bar, ch: usize, st: f32, v: &[i32], len: f32, vel: f32, kind: u8) {
        let r = &mut self.rng;
        let late = r.range(0.0, self.human);
        let gap = if ch == CH_KEYS { 0.005 } else { 0.009 } * r.range(0.8, 1.25);
        match kind {
            MARIACHI_UP => {
                for (k, &p) in v.iter().rev().take(3).enumerate() {
                    b.note(st, late + gap * k as f32, ch, p, vel * r.range(0.85, 1.0), len, false);
                }
            }
            MARIACHI_CHOP => {
                for (k, &p) in v.iter().enumerate() {
                    b.note(st, late + 0.3 * gap * k as f32, ch, p, vel * 0.8, 0.3, false);
                }
            }
            MARIACHI_RASGUEO => {
                // down, down, up in a 16th, the last left ringing
                let third = b.secs(1.0) / 3.0;
                for j in 0..3 {
                    let (t, l) = (late + third * j as f32, if j == 2 { len } else { 0.35 });
                    for k in 0..v.len() {
                        let p = if j == 2 { v[v.len() - 1 - k] } else { v[k] };
                        b.note(st, t + 0.6 * gap * k as f32, ch, p, vel * if j == 0 { 0.8 } else { 1.0 }, l, false);
                    }
                }
            }
            _ => {
                // the guitar's down strokes weighted to its bass strings (the
                // thumb on the lowest)
                let w: &[f32] = if ch == CH_PAD { &[1.6, 1.25, 1.0, 0.8, 0.8, 0.8] } else { &[1.0] };
                for (k, &p) in v.iter().enumerate() {
                    let t = w[k.min(w.len() - 1)];
                    b.note(st, late + gap * k as f32, ch, p, vel * t * r.range(0.88, 1.0), len, false);
                }
            }
        }
    }

    /// A bar of a strummed part (the vihuela on CH_KEYS, the guitar on
    /// CH_PAD) in the style's mánico, its last stroke a rasgueo when
    /// `rasgueo`; held chords where the instrument at hand cannot be
    /// strummed (a mix's pad, its attack slow).
    fn mariachi_strums(
        &mut self,
        s: &Synth,
        b: &mut Bar,
        ch: usize,
        chords: [Harm; 2],
        hemi: bool,
        alt: bool,
        rasgueo: bool,
        vel: f32,
    ) {
        let bs = self.mc.bar_steps as f32;
        let guitar = ch == CH_PAD;
        let v = [self.mariachi_voicing(chords[0], guitar), self.mariachi_voicing(chords[1], guitar)];
        if s.ch[ch].patch.a > 0.08 {
            let two = chords[1] != chords[0];
            for (hh, vv) in v.iter().enumerate().take(if two { 2 } else { 1 }) {
                for &n in vv {
                    b.note(hh as f32 * bs / 2.0, 0.0, ch, n, 0.45 * vel, if two { bs / 2.0 } else { bs }, false);
                }
            }
            return;
        }
        let pat = mariachi_manico(self.mc.style, hemi, guitar, alt);
        for (k, &(st, kind, v0)) in pat.iter().enumerate() {
            let st = st as f32;
            let end = pat.get(k + 1).map_or(bs, |x| x.0 as f32);
            let kind = if rasgueo && k + 1 == pat.len() { MARIACHI_RASGUEO } else { kind };
            let half = (st * 2.0 >= bs) as usize;
            self.mariachi_stroke(b, ch, st, &v[half], end - st, v0 * vel, kind);
        }
    }

    /// The guitarrón: roots and fifths where the style puts them (the root
    /// often doubled an octave up, plucked hard), the second chord's root
    /// where it changes; walking by step into the next chord when `walk` and
    /// it moves. A waltz's bar on a chord held over plays its fifth.
    fn mariachi_bass(&mut self, b: &mut Bar, chords: [Harm; 2], next: Harm, hemi: bool, vb: bool, walk: bool, held: bool) {
        let (bs, style) = (self.mc.bar_steps, self.mc.style);
        // roots in the octave up from C2 (fundamentals of 65-125 Hz), the
        // octave above them doubling
        let lo = MARIACHI_BASS_LO;
        let root = |c: Harm| lo + (self.root + c.bass() - lo).rem_euclid(12);
        let fifth = |c: Harm| {
            let r0 = root(c);
            let f = r0 - 7 + (self.root + c.root + 7 - (r0 - 7)).rem_euclid(12);
            if f < lo { f + 12 } else { f }
        };
        let (rn, roots, fifths) = (root(next), [root(chords[0]), root(chords[1])], [fifth(chords[0]), fifth(chords[1])]);
        let scale = |p: i32| MODES[self.mode].contains(&(p - self.root).rem_euclid(12));
        // (16th, the fifth rather than the root)
        let pat: &[(u32, bool)] = match style {
            MARIACHI_SON | MARIACHI_HUAPANGO if hemi => &[(0, false), (4, true), (8, false)],
            MARIACHI_SON | MARIACHI_HUAPANGO => &[(0, false), (6, true)],
            MARIACHI_VALS if held => &[(0, true)],
            MARIACHI_VALS => &[(0, false)],
            MARIACHI_POLKA => &[(0, false), (4, true)],
            _ if vb => &[(0, false), (6, true), (8, true)],
            _ => &[(0, false), (8, true), (12, true)],
        };
        let mut notes: Vec<(u32, i32, f32)> = Vec::with_capacity(6);
        let mut second = false;
        for &(st, fifth_) in pat {
            let hh = (st * 2 >= bs) as usize;
            let changed = hh == 1 && chords[1] != chords[0] && !second;
            second |= hh == 1;
            let p = if changed || !fifth_ { roots[hh] } else { fifths[hh] };
            notes.push((st, p, if st == 0 { 0.95 } else { 0.8 }));
        }
        let cur = notes.last().map_or(roots[1], |n| n.1);
        if walk && rn != cur {
            // two steps of the scale into the next root, from below or above;
            // a waltz's on its second and third beats
            let wa = if style == MARIACHI_VALS { 4 } else { bs - 4 };
            let q = if style == MARIACHI_VALS { 4 } else { 2 };
            notes.retain(|n| n.0 < wa || n.0 == 0);
            // (into a root near the bottom from above: the guitarrón's
            // lowest string is A1)
            let up = rn >= cur && rn - 4 >= 33;
            let path: Vec<i32> = if up {
                (rn - 4..rn).rev().filter(|&p| scale(p)).take(2).collect()
            } else {
                (rn + 1..rn + 5).filter(|&p| scale(p)).take(2).collect()
            };
            for (k, &p) in path.iter().rev().enumerate() {
                notes.push((wa + q * k as u32, p, 0.75));
            }
        }
        let octave = self.rng.chance(match style {
            MARIACHI_BOLERO => 0.25,
            MARIACHI_VALS => 0.4,
            _ => 0.55,
        });
        for (k, &(st, p, v)) in notes.iter().enumerate() {
            let end = notes.get(k + 1).map_or(bs, |n| n.0);
            let len = (end - st) as f32 * 0.92;
            let j = self.rng.range(0.0, self.human * 0.5);
            b.note(st as f32, j, CH_BASS, p, v, len, false);
            if st == 0 && octave {
                b.note(0.0, j + 0.004, CH_BASS, p + 12, 0.7 * v, len, false);
            }
        }
    }

    /// The harp: its bass (the root, low) on the bar's accents, the chord's
    /// tones climbing in eighths between (a bolero's in 16ths), falling in
    /// the B groups. From 16th `run` on, a run instead (the requinto's, where
    /// the tune holds): down the chord-scale in 16th triplets into the next
    /// bar, up in the B groups.
    fn mariachi_harp(&mut self, b: &mut Bar, chords: [Harm; 2], hemi: bool, vb: bool, run: u32) {
        let (bs, beat, style) = (self.mc.bar_steps, self.mc.beat_steps, self.mc.style);
        let acc = if hemi { 4 } else { beat };
        let q = if style == MARIACHI_BOLERO { 1 } else { 2 };
        let lo = self.tonic_near(48) - 5;
        let tones = [self.arp_tones(chords[0], 55), self.arp_tones(chords[1], 55)];
        let roots = chords.map(|c| lo + (self.root + c.bass() - lo).rem_euclid(12));
        let down = vb;
        if run < bs {
            let sc = chord_scale(self.mode, chords[1]);
            let mut v: Vec<i32> = in_range(&sc, 62 - self.root, 86 - self.root).iter().map(|p| p + self.root).collect();
            let n = ((bs - run) * 3 / 2) as usize;
            v.drain(..v.len().saturating_sub(n));
            if !vb {
                v.reverse();
            }
            for (k, &p) in v.iter().enumerate() {
                let st = run as f32 + k as f32 * 2.0 / 3.0;
                let j = self.rng.range(0.0, self.human * 0.5);
                b.note(st, j, CH_BELL, p, if k % 3 == 0 { 0.55 } else { 0.47 }, 1.5, false);
            }
        }
        let mut k = 0;
        for st in (0..run.min(bs)).step_by(q as usize) {
            let hh = (st * 2 >= bs) as usize;
            let j = self.rng.range(0.0, self.human);
            if st == 0 || (st * 2 == bs && st % acc == 0) {
                b.note(st as f32, j, CH_BELL, roots[hh], 0.6, (2 * acc) as f32, false);
            } else {
                let t = &tones[hh];
                let n = if down { t[t.len() - 1 - k % t.len()] } else { t[k % t.len()] };
                let v = if st % acc == 0 { 0.55 } else { 0.45 };
                b.note(st as f32, j, CH_BELL, n, v, (3 * q) as f32, false);
                k += 1;
            }
        }
    }

    /// The bar's tune and what goes with it: the leading section (violins or
    /// trumpets; trading two-bar phrases; or everyone, the violins an octave
    /// above the trumpets), a third or a sixth under it (the second trumpet,
    /// or the second violins), the other section's answers (the trumpets'
    /// in thirds), the second violins holding the chords' thirds and
    /// sevenths where they do not play under the tune.
    fn mariachi_lines(&mut self, s: &Synth, b: &mut Bar, sec: Sec, m: u32, chords: [Harm; 2], bi: u32, hold: [u32; 2]) {
        let bs = self.mc.bar_steps;
        let (tune, tro, role, harm, mode) = (self.mc.tonic, self.mc.tr_oct, self.mc.role, self.mc.harm, self.mode);
        let pick = |v: &[PNote]| -> Vec<PNote> {
            v.iter()
                .filter(|n| n.step as u32 / bs == bi)
                .map(|n| PNote { step: (n.step as u32 % bs) as u16, ..*n })
                .collect()
        };
        let mel = pick(&self.mc.mel);
        let ans = pick(&self.mc.ans);
        let trumpets = match role {
            MARIACHI_VIOLINS => false,
            MARIACHI_TRADE => (bi / 2) % 2 == 1,
            _ => true,
        };
        let tutti = role == MARIACHI_TUTTI;
        let (lead, other) = if trumpets { (CH_ARP, CH_LEAD) } else { (CH_LEAD, CH_ARP) };
        let plays = |ch: usize| m & if ch == CH_ARP { L_ARP } else { L_LEAD } != 0;
        let oct = |ch: usize| if ch == CH_ARP { tro } else { 0 };
        let legato = |ch: usize| if ch == CH_ARP { 0.85 } else { 0.95 };
        let chord_at = |st: u16| chords[(st as u32 * 2 / bs).min(1) as usize];
        let vel = 0.72 + 0.25 * sec.energy;
        // a drone at hand (a mix's, its attack slow) cannot follow a tune: it holds
        let second = if trumpets { CH_ARP } else { CH_DRONE };
        let under = harm > 0 && m & L_CTR != 0 && (trumpets || s.ch[CH_DRONE].patch.a <= 0.08);
        if plays(lead) {
            for n in &mel {
                let h = chord_at(n.step);
                self.mariachi_note(s, b, lead, n, tune + oct(lead) + n.pitch, n.vel * vel, legato(lead), h, false);
                if under {
                    // a sixth below too low for the second trumpet: a third above
                    let mut p = tune + oct(lead) + mariachi_below(mode, h, n.pitch, harm);
                    if second == CH_ARP && p < 55 {
                        p += 12;
                    }
                    let j = self.rng.range(0.0, self.human);
                    let len = n.len as f32 * legato(lead);
                    b.note(n.step as f32, j, second, p, n.vel * vel * 0.75, len, false);
                }
                // everyone: the violins an octave above the trumpets
                if tutti && plays(CH_LEAD) && tune + tro + n.pitch + 12 <= 98 {
                    self.mariachi_note(s, b, CH_LEAD, n, tune + tro + n.pitch + 12, n.vel * vel * 0.85, 0.95, h, false);
                }
            }
        }
        if !tutti && plays(other) {
            // the trumpets' answer an octave up if it would go below their range
            let low = ans.iter().map(|n| n.pitch).min().unwrap_or(0) - 4;
            let up = if other == CH_ARP && tune + tro + low < 55 { 12 } else { 0 };
            for n in &ans {
                let j = self.rng.range(0.0, self.human);
                let len = n.len as f32 * legato(other);
                b.note(n.step as f32, j, other, tune + oct(other) + up + n.pitch, n.vel * vel * 0.9, len, false);
                if other == CH_ARP && m & L_CTR != 0 {
                    let p = mariachi_below(mode, chord_at(n.step), n.pitch, 2);
                    b.note(n.step as f32, j, CH_ARP, tune + tro + up + p, n.vel * vel * 0.7, len, false);
                }
            }
        }
        if m & L_DRONE != 0 && !(under && second == CH_DRONE) {
            for (hh, &c) in chords.iter().enumerate() {
                if hold[hh] == 0 {
                    continue;
                }
                let iv = q_iv(c.q);
                let pcs: Vec<i32> =
                    [iv[1], *iv.get(3).unwrap_or(&iv[2])].iter().map(|x| (self.root + c.root + x).rem_euclid(12)).collect();
                let v = voicing_pcs(&pcs, tune - 5, tune + 9, &self.mc.sus_v, false);
                let len = hold[hh] as f32 * bs as f32 / 2.0 + 0.5;
                for &p in &v {
                    b.note(hh as f32 * bs as f32 / 2.0, 0.0, CH_DRONE, p, 0.35 + 0.1 * sec.energy, len, false);
                }
                self.mc.sus_v = v;
            }
        }
    }

    /// A note of a tune on `ch` with its ornament (unless `plain`): a grace
    /// note, a trill or a turn (on the chord-scale's neighbours), a falsete;
    /// slid into on a mono channel (the first violins).
    fn mariachi_note(&mut self, s: &Synth, b: &mut Bar, ch: usize, n: &PNote, p: i32, vel: f32, legato: f32, h: Harm, plain: bool) {
        let mono = s.ch[ch].mono;
        let st = n.step as f32;
        let len = n.len as f32 * legato;
        let j = self.rng.range(0.0, self.human);
        let step_s = b.secs(1.0);
        let g = match n.grace {
            _ if plain => 0,
            MARIACHI_TRILL if !mono => 0,
            g => g,
        };
        match g {
            0 => b.note(st, j, ch, p, vel, len, false),
            g @ (MARIACHI_TRILL | MARIACHI_TURN) => {
                let sc = chord_scale(self.mode, h);
                let inscale = |q: i32| sc.contains(&(q - self.root).rem_euclid(12));
                let up = (1..=3).map(|d| p + d).find(|&q| inscale(q)).unwrap_or(p + 2);
                let down = (1..=3).map(|d| p - d).find(|&q| inscale(q)).unwrap_or(p - 2);
                let d = 0.06;
                let k = if g == MARIACHI_TRILL { ((len * step_s * 0.5) / d).clamp(2.0, 10.0) as usize } else { 3 };
                for x in 0..k {
                    let q = if g == MARIACHI_TURN { [up, p, down][x] } else if x % 2 == 0 { p } else { up };
                    b.note(st, j + d * x as f32, ch, q, vel * if x == 0 { 1.0 } else { 0.85 }, 1.5 * d / step_s, mono && x > 0);
                }
                let t = d * k as f32;
                b.note(st, j + t, ch, p, vel, (len - t / step_s).max(0.5), mono);
            }
            MARIACHI_FALSETE => {
                b.note(st, j, ch, p - 5, vel * 0.7, 0.08 / step_s, false);
                b.note(st, j + 0.06, ch, p, vel, (len - 0.06 / step_s).max(0.5), mono);
            }
            g => {
                b.note(st, j, ch, p + g as i32, vel * 0.7, 0.06 / step_s, false);
                b.note(st, j + 0.045, ch, p, vel, (len - 0.045 / step_s).max(0.5), mono);
            }
        }
    }

    /// The bar before the last, on the dominant: the trumpets (in thirds)
    /// and the violins (an octave above) climb its arpeggio to the leading
    /// tone; the second violins hold its seventh.
    fn mariachi_closing(&mut self, s: &Synth, b: &mut Bar, m: u32, dom: Harm) {
        let (bs, style) = (self.mc.bar_steps, self.mc.style);
        let (tune, tro) = (self.mc.tonic, self.mc.tr_oct);
        let q = if style == MARIACHI_BOLERO { 4 } else { 2 };
        let n = (bs / q) as usize;
        let ct = in_range(&dom.pcs(), -20, 11);
        let fig = &ct[ct.len().saturating_sub(n)..];
        for (k, &p) in fig.iter().enumerate() {
            let st = (k as u32 * q) as f32;
            let lastn = k + 1 == fig.len();
            let len = if lastn { (bs as f32 - st) + 0.5 } else { q as f32 * 0.85 };
            let v = 0.75 + 0.25 * k as f32 / fig.len() as f32;
            let j = self.rng.range(0.0, self.human);
            if m & L_ARP != 0 {
                b.note(st, j, CH_ARP, tune + tro + p, v, len, false);
                if k > 0 {
                    b.note(st, j, CH_ARP, tune + tro + fig[k - 1], v * 0.75, len, false);
                }
            }
            if m & L_LEAD != 0 {
                let note = PNote { step: st as u16, len: if lastn { bs as u16 - st as u16 } else { q as u16 }, pitch: p, vel: v, grace: 0 };
                self.mariachi_note(s, b, CH_LEAD, &note, tune + 12 + p, v, 0.9, dom, true);
            }
        }
        if m & L_DRONE != 0 {
            let v = voicing_pcs(&[(self.root + 11).rem_euclid(12), (self.root + 5).rem_euclid(12)], tune - 5, tune + 9, &self.mc.sus_v, false);
            for &p in &v {
                b.note(0.0, 0.0, CH_DRONE, p, 0.5, bs as f32 + 0.5, false);
            }
            self.mc.sus_v = v;
        }
    }

    /// The last bar: everyone on the tonic. Held (the vihuela's rasgueo and
    /// the guitar left ringing, the guitarrón's octave, the harp swept up,
    /// the trumpets high, the violins trilling), or two stabs, the second the
    /// louder; a stomp and palmas with them.
    fn mariachi_ending(&mut self, s: &Synth, b: &mut Bar, m: u32, home: Harm) {
        let bs = self.mc.bar_steps as f32;
        let (tune, tro, button) = (self.mc.tonic, self.mc.tr_oct, self.mc.button);
        let third = if home.q == Q_MIN { 3 } else { 4 };
        let beat = self.mc.beat_steps as f32;
        let second = if bs / beat == 3.0 { 2.0 * beat } else { bs / 2.0 };
        let stabs: &[(f32, f32, f32)] =
            if button { &[(0.0, 1.5, 0.85), (second, bs - second + 2.0, 1.0)] } else { &[(0.0, bs + 4.0, 1.0)] };
        let mixed = self.guest.is_some();
        let vk = [self.mariachi_voicing(home, false), self.mariachi_voicing(home, true)];
        let root = MARIACHI_BASS_LO + (self.root - MARIACHI_BASS_LO).rem_euclid(12);
        for &(st, len, v) in stabs {
            if m & L_KICK != 0 || mixed {
                b.hit(st, 0.0, CH_KICK, 0, v);
            }
            if m & L_SNARE != 0 || mixed {
                b.hit(st, 0.0, CH_SNARE, 0, 0.8 * v);
            }
            if m & L_KEYS != 0 {
                let kind = if button { MARIACHI_DOWN } else { MARIACHI_RASGUEO };
                self.mariachi_stroke(b, CH_KEYS, st, &vk[0], len, v, kind);
            }
            if m & L_PAD != 0 {
                self.mariachi_stroke(b, CH_PAD, st, &vk[1], len, v * 0.9, MARIACHI_DOWN);
            }
            if m & L_BASS != 0 {
                b.note(st, 0.0, CH_BASS, root, v, len, false);
                b.note(st, 0.004, CH_BASS, root + 12, 0.75 * v, len, false);
            }
            if m & L_ARP != 0 {
                for p in [tune + tro + third, tune + tro + 7] {
                    b.note(st, 0.0, CH_ARP, p, 0.95 * v, len, false);
                }
            }
            if m & L_LEAD != 0 {
                let n = PNote { step: st as u16, len: len as u16, pitch: 12, vel: v, grace: if button { 0 } else { MARIACHI_TRILL } };
                self.mariachi_note(s, b, CH_LEAD, &n, tune + 12, v, 1.0, home, false);
            }
            if m & L_DRONE != 0 {
                b.note(st, 0.0, CH_DRONE, tune + third, 0.6 * v, len, false);
                b.note(st, 0.0, CH_DRONE, tune + 7, 0.55 * v, len, false);
            }
        }
        if !button && m & L_BELL != 0 {
            // the harp swept up the chord, into the downbeat's ring
            for (k, &p) in in_range(&self.pcs(home), 48, 84).iter().enumerate() {
                b.note(0.0, 0.025 * k as f32, CH_BELL, p, 0.55, bs, false);
            }
        }
    }

    /// Mariachi's percussion, the `layers` asked for, in any meter: L_KICK
    /// the zapateado (heel stomps on the accents, toe taps on the eighths
    /// between, a shuffle in the B groups, a run of taps closing a group:
    /// `fill`), L_SNARE palmas (on a compound meter's accents, else off the
    /// beat), L_HAT maracas in eighths, L_PERC the guiro (a long scrape on
    /// the accents, short ones between); in 6/8 turned to 3/4 (`hemi`) the
    /// accents are the three quarters.
    fn mariachi_hits(&mut self, b: &mut Bar, layers: u32, hemi: bool, fill: u8, e: f32, quiet: f32) {
        let (bs, beat) = (self.bar_steps(), self.beat_steps());
        let beats = bs / beat;
        let vb = self.var_b();
        let h = self.human;
        let r = &mut self.rng;
        let acc = if hemi { 4 } else { beat };
        let last = bs - acc;
        let lift = quiet * (0.8 + 0.2 * e);
        if layers & L_KICK != 0 {
            for st in (0..bs).step_by(2) {
                let on = st % acc == 0;
                if fill == 2 && st > last {
                    continue;
                }
                if on {
                    b.hit(st as f32, r.range(0.0, h), CH_KICK, 0, lift * if st == 0 { 0.95 } else { 0.82 });
                } else if r.chance(0.85) {
                    b.hit(st as f32, r.range(0.0, h), CH_KICK, 1, lift * r.range(0.4, 0.55));
                }
                if vb && !on && st + 1 < bs && r.chance(0.3) {
                    b.hit((st + 1) as f32, r.range(0.0, h), CH_KICK, 1, lift * 0.35);
                }
            }
            if fill == 2 {
                // a redoble: taps in 16ths, louder to the bar line
                for st in last + 1..bs {
                    b.hit(st as f32, r.range(0.0, h), CH_KICK, 1, lift * (0.35 + 0.45 * (st - last) as f32 / acc as f32));
                }
            }
        }
        if layers & L_SNARE != 0 {
            let at: &[u32] = match (beat, beats) {
                _ if hemi => &[0, 4, 8],
                (4, 2) => &[2, 6],
                (4, 3) => &[4, 8],
                (4, _) => &[4, 12],
                (_, 3) => &[0, 6, 12],
                _ => &[0, 6],
            };
            for &st in at.iter().filter(|&&st| st < bs) {
                b.hit(st as f32, r.range(0.0, h), CH_SNARE, 0, lift * (0.55 + 0.25 * e));
            }
            if vb {
                b.hit((bs - 2) as f32, r.range(0.0, h), CH_SNARE, 1, lift * 0.4);
            }
        }
        if layers & L_HAT != 0 {
            for st in (0..bs).step_by(2) {
                let on = st % acc == 0;
                let kind = if fill > 0 && st + 2 >= bs { HAT_O } else { HAT_C };
                b.hit(st as f32, r.range(0.0, h), CH_HAT, kind, lift * if on { 0.5 } else { 0.32 } * r.range(0.8, 1.0));
                if vb && !on && r.chance(0.5) {
                    b.hit((st + 1) as f32, r.range(0.0, h), CH_HAT, HAT_C, lift * 0.22);
                }
            }
        }
        if layers & L_PERC != 0 {
            for st in (0..bs).step_by(2) {
                let on = st % acc == 0;
                if on || r.chance(0.6) {
                    b.hit(st as f32, r.range(0.0, h), CH_PERC, SHAKER, lift * if on { 0.5 } else { 0.3 });
                }
            }
        }
    }

    /// Mariachi's beat in any meter, for a mix: the zapateado where there is
    /// a beat at all (in 6/8, every other bar turned to 3/4 in half the
    /// sections: the son's hemiola), palmas when it is strong, maracas once
    /// under way (softly in a break), the guiro in a strong section's B
    /// groups; each where the section asks or the writer's arrangement (`m`)
    /// has those drums (nothing in an intro, else).
    fn mariachi_beat(&mut self, _s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let (bs, beat) = (self.bar_steps(), self.beat_steps());
        if self.sec_bar == 0 {
            self.mc.dhemi = bs == 12 && beat == 6 && self.rng.chance(0.5);
        }
        let hemi = self.mc.dhemi && self.sec_bar % 2 == 1;
        let (e, full) = (sec.energy, sec.kind == MAIN);
        let vb = self.var_b();
        let fill = if full || sec.kind == BUILD { self.fill(sec) } else { 0 };
        let mut layers = 0;
        if full || matches!(sec.kind, BUILD | OUTRO) || m & L_KICK != 0 {
            layers |= L_KICK;
        }
        if (full && e >= 0.8) || (sec.kind == BUILD && e >= 0.7) || m & L_SNARE != 0 {
            layers |= L_SNARE;
        }
        if (full && (e >= 0.6 || vb)) || sec.kind == BREAK || m & (L_HAT | L_OHAT) != 0 {
            layers |= L_HAT;
        }
        if (full && e >= 0.75 && vb) || m & L_PERC != 0 {
            layers |= L_PERC;
        }
        let quiet = if sec.kind == BREAK && m & (L_KICK | L_HAT | L_OHAT) == 0 { 0.6 } else { 1.0 };
        self.mariachi_hits(b, layers, hemi, fill, e, quiet);
    }
}

/// A mix's drummer, this piece: a percussion line (a bar, or two of a
/// two-beat meter), lo-fi's kicks (A and B: 16ths of four beats, or three)
/// and which of its snares, whether polka's tambourine plays.
#[derive(Default)]
struct Drummer {
    perc: Vec<f32>,
    perc_kind: u8,
    kicks: [Vec<u16>; 2],
    snare_kind: u8,
    /// polka's tambourine
    tambourine: bool,
}

// -- polka

/// A note of a polka line: a 16th within its strain, a length in 16ths, a
/// pitch in semitones above the strain's tonic, and a grace note's offset
/// (0: none) played on the beat with the note itself just after.
#[derive(Clone, Copy)]
struct PNote {
    step: u16,
    len: u16,
    pitch: i32,
    vel: f32,
    grace: i8,
}

/// A strain: 8 or 16 bars of 2/4, a chord a bar, in a key (semitones from the
/// piece's tonic) and mode, with its tune and the other instrument's answers.
#[derive(Clone)]
struct Strain {
    key: i32,
    mode: usize,
    chords: Vec<Harm>,
    mel: Vec<PNote>,
    ans: Vec<PNote>,
}

/// The polka in play.
#[derive(Default)]
struct Polka {
    strains: Vec<Strain>,
    /// per section: the strain, which time through, its first bar
    map: Vec<(usize, u32, u32)>,
    /// this time through: the tune and answers (varied on repeats), whether
    /// the other instrument doubles a third below, which channel leads
    mel: Vec<PNote>,
    ans: Vec<PNote>,
    thirds: bool,
    lead: usize,
    /// the tune's tonic (MIDI), whether the hats play, the bar's chord (for the stats)
    tonic: i32,
    hats: bool,
    mask: u16,
    /// the other instrument leads the B strain
    b_other: bool,
    /// a dogfight: the two instruments trade two-bar phrases
    trade: bool,
}

/// Eight-bar chord plans: major A strains, B strains (away from the tonic
/// at first), and minor ones.
const POLKA_A: &[&str] = &[
    "I I V7 V7 V7 V7 I I",
    "I I I V7 V7 V7 V7 I",
    "I IV I V7 I IV V7 I",
    "I I V7/V V7 I I V7 I",
    "I I IV IV V7 V7 I I",
    "I V7 I V7 I IV V7 I",
    "I I ii V7 V7 V7 I I",
    "I I^1 IV V7 I vi V7/V^1 V7",
    "I vi ii V7 I vi V7 I",
    "I I7 IV IV I V7/V V7 I",
];
const POLKA_B: &[&str] = &[
    "V7 V7 I I V7 V7 I I",
    "IV IV I I V7 V7 I I",
    "V7/V V7/V V7 V7 V7 V7 I I",
    "IV I V7 I IV I V7 I",
    "vi vi V7/V V7 I IV V7 I",
    "V7 I V7 I IV IV V7 I",
    "IV iv I vi ii V7 I I",
    "V7/vi vi V7/V V7 I IV V7 I",
];
const POLKA_MINOR: &[&str] = &[
    "i i V7 V7 V7 V7 i i",
    "i iv V7 i i iv V7 i",
    "i i iv iv V7 V7 i i",
    "i bVI iv V7 i bVI V7 i",
    "i V7 i V7 iv i V7 i",
    "bIII bIII bVII bVII iv iv V7 V7",
];

/// Figures a bar of tune is made of.
const F_ARP_UP: u8 = 0;
const F_ARP_DOWN: u8 = 1;
const F_RUN: u8 = 2;
const F_RUN_HALF: u8 = 3;
const F_REPEAT: u8 = 4;
const F_DOTTED: u8 = 5;
const F_TURN: u8 = 6;
const F_QUARTERS: u8 = 7;
const F_QE: u8 = 8;
const F_LONG: u8 = 9;
const F_HALF: u8 = 10;
const MID_FIGS: [u8; 8] = [F_ARP_UP, F_ARP_DOWN, F_RUN, F_RUN_HALF, F_DOTTED, F_TURN, F_QE, F_REPEAT];

/// One bar of tune over chord `h`, starting near `prev`: arpeggios, scale
/// runs, repeated notes, dotted rhythms, turns, or a cadence (`target`, if
/// given, is where a cadence lands).
fn polka_figure(
    r: &mut Rng,
    kind: u8,
    h: Harm,
    mode: usize,
    prev: i32,
    target: Option<i32>,
    lo: i32,
    hi: i32,
) -> Vec<PNote> {
    let ct = in_range(&h.pcs(), lo, hi);
    let sc = in_range(&chord_scale(mode, h), lo, hi);
    let i = nearest(&ct, prev);
    let at = |v: &[i32], k: isize| v[k.clamp(0, v.len() as isize - 1) as usize];
    let j = nearest(&sc, ct[i]) as isize;
    let ii = i as isize;
    let up = (prev as f32) < (lo + hi) as f32 / 2.0 || r.chance(0.3);
    let d: isize = if up { 1 } else { -1 };
    let n = |step: u16, len: u16, pitch: i32| PNote { step, len, pitch, vel: 0.0, grace: 0 };
    let mut v = match kind {
        F_ARP_UP => {
            let b = ii.min(ct.len() as isize - 4).max(0);
            (0..4).map(|k| n(2 * k as u16, 2, at(&ct, b + k))).collect()
        }
        F_ARP_DOWN => {
            let b = ii.max(3);
            (0..4).map(|k| n(2 * k as u16, 2, at(&ct, b - k))).collect()
        }
        F_RUN => (0..8).map(|k| n(k as u16, 1, at(&sc, j + d * k as isize))).collect(),
        F_RUN_HALF => {
            let mut v: Vec<PNote> = (0..4).map(|k| n(k as u16, 1, at(&sc, j + d * k as isize))).collect();
            let land = at(&sc, j + d * 4);
            v.push(n(4, 4, ct[nearest(&ct, land)]));
            v
        }
        F_REPEAT => {
            let (p, q) = (ct[i], at(&ct, ii + d));
            if r.chance(0.5) {
                vec![n(0, 2, p), n(2, 2, p), n(4, 2, p), n(6, 2, p)]
            } else {
                vec![n(0, 2, p), n(2, 2, q), n(4, 2, p), n(6, 2, q)]
            }
        }
        F_DOTTED => {
            let p2 = at(&ct, ii + d);
            let j2 = nearest(&sc, p2) as isize;
            vec![n(0, 3, ct[i]), n(3, 1, at(&sc, j + d)), n(4, 3, p2), n(7, 1, at(&sc, j2 + d))]
        }
        F_TURN => vec![
            n(0, 1, at(&sc, j + 1)),
            n(1, 1, at(&sc, j)),
            n(2, 1, at(&sc, j - 1)),
            n(3, 1, at(&sc, j)),
            n(4, 4, at(&ct, ii + d)),
        ],
        F_QUARTERS => vec![n(0, 4, ct[i]), n(4, 4, at(&ct, ii + d))],
        F_QE => vec![n(0, 4, ct[i]), n(4, 2, at(&sc, j + d)), n(6, 2, at(&sc, j + 2 * d))],
        F_LONG => vec![n(0, 8, target.map_or(ct[i], |t| ct[nearest(&ct, t)]))],
        _ => vec![n(0, 4, target.map_or(ct[i], |t| ct[nearest(&ct, t)]))],
    };
    for x in v.iter_mut() {
        x.vel = match x.step {
            0 => 0.95,
            4 => 0.85,
            _ => 0.75,
        } + r.range(-0.05, 0.05);
    }
    v
}

/// A bar of tune moved onto another chord: chord tones to the nearest tone
/// of the new chord, the rest to the nearest tone of its scale.
fn polka_refit(notes: &[PNote], from: Harm, to: Harm, mode: usize, lo: i32, hi: i32) -> Vec<PNote> {
    if from == to {
        return notes.to_vec();
    }
    let (fc, ct, sc) = (from.pcs(), in_range(&to.pcs(), lo, hi), in_range(&chord_scale(mode, to), lo, hi));
    notes
        .iter()
        .map(|n| {
            let v = if fc.contains(&n.pitch.rem_euclid(12)) { &ct } else { &sc };
            PNote { pitch: v[nearest(v, n.pitch)], ..*n }
        })
        .collect()
}

/// The melody's range and the answering instrument's, semitones about the strain's tonic.
const PK_LO: i32 = -3;
const PK_HI: i32 = 14;
const PK_ALO: i32 = -9;
const PK_AHI: i32 = 9;

/// A strain's tune: four-bar phrases opening on a two-bar motif (which comes
/// back fitted to the chords), a free bar, then a cadence: half (on V) mid
/// strain, home at its end. A sixteen-bar strain repeats its first eight with
/// a new ending. Where the tune rests on the second beat of a cadence bar the
/// other instrument answers.
fn polka_tune(r: &mut Rng, chords: &[Harm], mode: usize) -> (Vec<PNote>, Vec<PNote>) {
    let bars = chords.len();
    let motif: [u8; 2] = r.pick(&[
        [F_RUN_HALF, F_REPEAT],
        [F_ARP_UP, F_DOTTED],
        [F_DOTTED, F_DOTTED],
        [F_REPEAT, F_ARP_DOWN],
        [F_TURN, F_QUARTERS],
        [F_ARP_UP, F_RUN],
        [F_RUN, F_QE],
        [F_QE, F_ARP_DOWN],
        [F_REPEAT, F_REPEAT],
        [F_DOTTED, F_RUN_HALF],
        [F_ARP_UP, F_ARP_DOWN],
    ]);
    let mut bar_notes: Vec<Vec<PNote>> = Vec::with_capacity(bars);
    let mut prev = r.pick(&[4, 7, 0, 12, 7]);
    for b in 0..bars {
        let pos = b % 4;
        let fig = if b >= 8 && b < bars - 2 {
            // the second half of a long strain: its first half again
            polka_refit(&bar_notes[b - 8], chords[b - 8], chords[b], mode, PK_LO, PK_HI)
        } else if b >= 4 && pos < 2 && r.chance(0.8) {
            polka_refit(&bar_notes[pos], chords[pos], chords[b], mode, PK_LO, PK_HI)
        } else {
            let (kind, target) = match pos {
                0 | 1 => (motif[pos], None),
                2 => (r.pick(&MID_FIGS), None),
                _ if b == bars - 1 => (if r.chance(0.6) { F_HALF } else { F_LONG }, Some(r.pick(&[0, 12, 4]))),
                _ => (if r.chance(0.7) { F_HALF } else { F_QUARTERS }, None),
            };
            polka_figure(r, kind, chords[b], mode, prev, target, PK_LO, PK_HI)
        };
        prev = fig.last().map_or(prev, |n| n.pitch);
        bar_notes.push(fig);
    }
    let mut mel = Vec::new();
    let mut ans = Vec::new();
    for (b, notes) in bar_notes.iter().enumerate() {
        let st0 = 8 * b as u16;
        mel.extend(notes.iter().map(|n| PNote { step: n.step + st0, ..*n }));
        // a free second beat: the other instrument answers, leading on
        if notes.iter().all(|n| n.step < 4 && n.step + n.len <= 4) && r.chance(0.8) {
            let h = chords[b];
            let ct = in_range(&h.pcs(), PK_ALO, PK_AHI);
            let sc = in_range(&chord_scale(mode, h), PK_ALO, PK_AHI);
            let start = nearest(&ct, notes[0].pitch - 7) as isize;
            let at = |v: &[i32], k: isize| v[k.clamp(0, v.len() as isize - 1) as usize];
            let fill: Vec<(u16, u16, i32)> = match r.below(3) {
                0 => (0..4).map(|k| (4 + k as u16, 1, at(&ct, start + k))).collect(),
                1 => {
                    let j = nearest(&sc, at(&ct, start + 2)) as isize;
                    (0..4).map(|k| (4 + k as u16, 1, at(&sc, j - k))).collect()
                }
                _ => vec![(4, 2, at(&ct, start + 1)), (6, 2, at(&ct, start + 2))],
            };
            ans.extend(fill.into_iter().map(|(st, len, pitch)| PNote {
                step: st0 + st,
                len,
                pitch,
                vel: 0.75 + r.range(-0.05, 0.05),
                grace: 0,
            }));
        }
    }
    (mel, ans)
}

/// Grace notes on a few strong, long enough notes: a semitone below or the
/// scale tone above.
fn polka_graces(r: &mut Rng, mel: &mut [PNote], p: f32) {
    for n in mel.iter_mut() {
        if n.step % 4 == 0 && n.len >= 2 && n.grace == 0 && r.chance(p) {
            n.grace = r.pick(&[-1, -1, 2, 1]);
        }
    }
}

/// A strain of `bars` (8 or 16) from the plans.
fn polka_strain(r: &mut Rng, plans: &[&str], bars: usize, key: i32, mode: usize) -> Strain {
    let mut chords = harms(r.pick(plans));
    if bars == 16 {
        // the first eight close on the dominant, the second at home
        chords[7] = Harm { root: 7, q: Q_DOM7, inv: 0 };
        chords[6] = r.pick(&[chords[6], Harm { root: 2, q: Q_DOM7, inv: 0 }, Harm { root: 0, q: Q_MAJ, inv: 0 }]);
        if mode != IONIAN && chords[6].root == 2 {
            chords[6] = Harm { root: 5, q: Q_MIN, inv: 0 };
        }
        let second = harms(r.pick(plans));
        chords.extend(second);
    }
    let (mut mel, ans) = polka_tune(r, &chords, mode);
    polka_graces(r, &mut mel, 0.08);
    Strain { key, mode, chords, mel, ans }
}

// -- tavern

/// Tavern dances: jig (6/8), slip jig (9/8), reel (4/4), a dance in 3/4,
/// and now and then a slow ballad.
const T_JIG: u8 = 0;
const T_SLIP: u8 = 1;
const T_REEL: u8 = 2;
const T_WALTZ: u8 = 3;
const T_BALLAD: u8 = 4;

/// Eight-bar strains for the modes, A and B (the B away from home at
/// first), with modal cadences (bVII-I, i-bVII-i).
const TAV_ION: [&[&str]; 2] = [
    &["I I IV V I I IV+V I", "I IV I V I IV V I", "I vi IV V I vi IV+V I", "I I V V I I IV+V I"],
    &["IV IV I I IV IV V V", "vi vi IV IV I I V I", "IV V I vi IV V I I"],
];
const TAV_MIX: [&[&str]; 2] = [
    &["I bVII I bVII I bVII IV I", "I I bVII bVII I I bVII+IV I", "I IV I bVII I IV bVII I"],
    &["bVII bVII I I bVII bVII IV I", "IV IV I I bVII bVII I I", "bVII IV I I bVII IV bVII I"],
];
const TAV_DOR: [&[&str]; 2] = [
    &["i bVII i bVII i bVII IV i", "i i bVII bVII i i IV+bVII i", "i IV i IV bIII bVII IV i", "i i IV IV i bVII i i"],
    &["bIII bIII bVII bVII IV IV i i", "bVII bVII i i bVII bVII IV+bVII i", "IV IV i i bIII bVII i i"],
];
const TAV_AEO: [&[&str]; 2] = [
    &[
        "i bVII bVI bVII i bVII bVI+bVII i",
        "i i bVII bVII bVI bVII i i",
        "i iv i bVII i iv bVII i",
        "i bVII i v i bVII bVI+v i",
    ],
    &["bIII bIII bVII bVII bVI bVI v i", "bVI bVII i i bVI bVII v+bVII i", "bIII bVII iv i bIII bVII v i"],
];

/// A tavern tune's strain: its chords (two a bar) and tune (and the answers).
#[derive(Clone)]
struct TStrain {
    chords: Vec<Harm>,
    mel: Vec<PNote>,
    ans: Vec<PNote>,
}

/// The tavern piece in play.
#[derive(Default)]
struct Tavern {
    kind: u8,
    /// 16ths a bar and a beat
    bar_steps: u32,
    beat_steps: u32,
    strains: Vec<TStrain>,
    /// per section: the strain, which time through
    map: Vec<(usize, u32)>,
    /// this time through: the tune (varied), answers, a harmony above (semitone
    /// steps of the chord-scale: 2 a third, 5 a sixth; 0 none), the B strain up an octave
    mel: Vec<PNote>,
    ans: Vec<PNote>,
    harmony: i32,
    up: i32,
    /// the bar's chords (for the stats), the lute's last voicing, the drone
    mask: u16,
    drone: bool,
}

/// One beat of a tavern tune: a rhythm cell (16th steps, lengths) for the
/// meter, its notes moving from `prev` by step in the chord-scale, by chord
/// tone, or repeated; the beat's first note on a chord tone.
fn tavern_beat(
    r: &mut Rng,
    kind: u8,
    beat_steps: u32,
    h: Harm,
    mode: usize,
    prev: i32,
    dir: i32,
    lo: i32,
    hi: i32,
) -> Vec<(u16, u16, i32)> {
    let cells: &[(&[(u16, u16)], f32)] = if beat_steps == 6 {
        &[
            (&[(0, 2), (2, 2), (4, 2)], 0.45),
            (&[(0, 4), (4, 2)], 0.3),
            (&[(0, 2), (2, 1), (3, 1), (4, 2)], 0.12),
            (&[(0, 3), (3, 1), (4, 2)], 0.08),
            (&[(0, 6)], 0.05),
        ]
    } else if kind == T_BALLAD {
        &[(&[(0, 4)], 0.45), (&[(0, 2), (2, 2)], 0.35), (&[(0, 3), (3, 1)], 0.2)]
    } else {
        &[(&[(0, 2), (2, 2)], 0.55), (&[(0, 4)], 0.2), (&[(0, 1), (1, 1), (2, 2)], 0.15), (&[(0, 3), (3, 1)], 0.1)]
    };
    let w: Vec<f32> = cells.iter().map(|c| c.1).collect();
    let cell = cells[r.weighted(&w)].0;
    let ct = in_range(&h.pcs(), lo, hi);
    let sc = in_range(&chord_scale(mode, h), lo, hi);
    let mut p = ct[nearest(&ct, prev + dir)];
    let mut out = Vec::with_capacity(cell.len());
    for (k, &(st, len)) in cell.iter().enumerate() {
        if k > 0 {
            let j = nearest(&sc, p) as isize;
            p = match r.weighted(&[0.6, 0.25, 0.15]) {
                0 => sc[(j + dir as isize).clamp(0, sc.len() as isize - 1) as usize],
                1 => {
                    let i = nearest(&ct, p) as isize;
                    ct[(i + dir as isize).clamp(0, ct.len() as isize - 1) as usize]
                }
                _ => p,
            };
        }
        out.push((st, len, p));
    }
    out
}

/// A tavern strain's tune: eight bars in two four-bar phrases; the first
/// two bars a motif that comes back fitted to the chords in bars 5 and 6;
/// bar 4 rests on a chord tone (half cadence), bar 8 comes home long. The
/// line climbs and falls in arcs. Where it rests the answering instrument
/// plays.
fn tavern_tune(r: &mut Rng, t: &Tavern, chords: &[Harm], mode: usize, lo: i32, hi: i32) -> (Vec<PNote>, Vec<PNote>) {
    let (bs, beat) = (t.bar_steps as u16, t.beat_steps as u16);
    let beats = bs / beat;
    let chord_at = |bar: usize, b: u16| chords[bar * 2 + ((b * 2) / beats) as usize];
    let mut bars: Vec<Vec<PNote>> = Vec::new();
    let mut prev = r.pick(&[0, 4, 7, 7, 12]).clamp(lo, hi);
    let mut dir = 1;
    for bar in 0..8 {
        let notes: Vec<PNote> = if (bar == 4 || bar == 5) && r.chance(0.8) {
            // the motif again, its notes moved onto these chords
            bars[bar - 4]
                .iter()
                .map(|n| {
                    let (from, to) = (chord_at(bar - 4, n.step / beat), chord_at(bar, n.step / beat));
                    if from == to {
                        *n
                    } else {
                        let set =
                            if from.pcs().contains(&n.pitch.rem_euclid(12)) { to.pcs() } else { chord_scale(mode, to) };
                        let v = in_range(&set, lo, hi);
                        PNote { pitch: v[nearest(&v, n.pitch)], ..*n }
                    }
                })
                .collect()
        } else if bar == 3 || bar == 7 {
            // a cadence: the first beat's figure, then a long note
            let h = chord_at(bar, 0);
            let mut v: Vec<PNote> = tavern_beat(r, t.kind, t.beat_steps, h, mode, prev, dir, lo, hi)
                .into_iter()
                .map(|(st, len, pitch)| PNote { step: st, len, pitch, vel: 0.85, grace: 0 })
                .collect();
            let hl = chord_at(bar, beats - 1);
            let target = if bar == 7 { [0, 12, 7] } else { [7, 4, 2] };
            let ct = in_range(&hl.pcs(), lo, hi);
            let want = target.iter().map(|&x| x + 12 * ((prev - x) as f32 / 12.0).round() as i32).collect::<Vec<_>>();
            let goal = want[nearest(&want, prev)].clamp(lo, hi);
            let land = ct[nearest(&ct, goal)];
            v.push(PNote { step: beat, len: bs - beat, pitch: land, vel: 0.9, grace: 0 });
            v
        } else {
            let mut v = Vec::new();
            for b in 0..beats {
                let h = chord_at(bar, b);
                for (st, len, pitch) in tavern_beat(r, t.kind, t.beat_steps, h, mode, prev, dir, lo, hi) {
                    v.push(PNote { step: b * beat + st, len, pitch, vel: if st == 0 { 0.9 } else { 0.75 }, grace: 0 });
                    prev = pitch;
                }
                // arcs: turn at the edges, sometimes on a whim
                if prev >= hi - 2 {
                    dir = -1;
                } else if prev <= lo + 2 {
                    dir = 1;
                } else if r.chance(0.25) {
                    dir = -dir;
                }
            }
            v
        };
        prev = notes.last().map_or(prev, |n| n.pitch);
        bars.push(notes);
    }
    let mut mel = Vec::new();
    let mut ans = Vec::new();
    for (bar, notes) in bars.iter().enumerate() {
        let st0 = bs * bar as u16;
        mel.extend(notes.iter().map(|n| PNote { step: n.step + st0, ..*n }));
        // the long note of a cadence: the other instrument answers under it
        if (bar == 3 || bar == 7) && r.chance(0.75) {
            let h = chords[bar * 2 + 1];
            let sc = in_range(&chord_scale(mode, h), lo - 7, hi - 5);
            let j = nearest(&sc, notes.last().map_or(0, |n| n.pitch) - 5) as isize;
            let up = r.bit_chance();
            for k in 0..(beats as isize * if beat == 6 { 3 } else { 2 } - 2) {
                let p = sc[(j + if up { k } else { -k }).clamp(0, sc.len() as isize - 1) as usize];
                ans.push(PNote { step: st0 + beat + 2 * k as u16, len: 2, pitch: p, vel: 0.7, grace: 0 });
            }
        }
    }
    (mel, ans)
}

/// Tavern ornaments on long notes: a cut (a grace a step above) or a roll
/// (cut, the note, a tap below, the note).
fn tavern_ornaments(r: &mut Rng, mel: &mut [PNote], p: f32) {
    for n in mel.iter_mut() {
        if n.len >= 3 && n.grace == 0 && r.chance(p) {
            n.grace = r.pick(&[2, 2, 1, 9]);
        }
    }
}

const P_TAV_LEAD: &[(&str, Make)] = &[
    ("fiddle", fiddle),
    ("tin whistle", whistle),
    ("recorder", recorder),
    ("wooden flute", flute),
    ("bagpipe chanter", chanter),
    ("hurdy-gurdy", gurdy),
    ("gemshorn", ocarina),
    ("crumhorn", crumhorn),
];
const P_TAV_SECOND: &[(&str, Make)] = &[
    ("recorder", recorder),
    ("tin whistle", whistle),
    ("fiddle", fiddle),
    ("harp", harp),
    ("hammered dulcimer", dulcimer),
    ("psaltery", psaltery),
];
const P_TAV_PLUCK: &[(&str, Make)] = &[
    ("lute", lute),
    ("cittern", cittern),
    ("harp", harp),
    ("hammered dulcimer", dulcimer),
    ("nylon-strung guitar", nylon),
];
const P_TAV_DRONE: &[(&str, Make)] =
    &[("hurdy-gurdy drone", drone_reed), ("bagpipe drone", pipe_drone), ("voices", drone_choir)];
const P_TAV_BASS: &[(&str, Make)] = &[("bass viol", viol), ("plucked bass", bass_upright)];
const K_TAV_KICK: &[(&str, MakeKit)] = &[("frame drum", kit_frame)];
const K_TAV_HAT: &[(&str, MakeKit)] = &[("shaker", kit_shaker)];
const K_TAV_PERC: &[(&str, MakeKit)] = &[("woodblock and bones", kit_bones)];

fn chanter(r: &mut Rng) -> Patch {
    // a bagpipe's chanter: a narrow pulse, nasal, loud and steady
    Patch {
        wave: Wave::Square,
        pw: r.range(0.2, 0.3),
        vowel: 2,
        cutoff: 3500.0,
        keytrack: 0.3,
        a: 0.01,
        d: 0.2,
        s: 1.0,
        r: 0.05,
        glide: 0.008,
        gain: 0.4,
        ..PATCH
    }
}

fn gurdy(r: &mut Rng) -> Patch {
    // a hurdy-gurdy's melody string: a saw buzzing through a nasal vowel
    Patch {
        unison: 2,
        detune: r.range(3.0, 6.0),
        width: 0.2,
        vowel: 3,
        noise: 0.03,
        cutoff: 3000.0,
        keytrack: 0.3,
        a: 0.03,
        d: 0.3,
        s: 0.95,
        r: 0.1,
        vib: 0.05,
        vib_rate: 6.0,
        glide: 0.015,
        gain: 0.45,
        ..PATCH
    }
}

fn crumhorn(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Square,
        pw: 0.5,
        cutoff: r.range(1200.0, 1600.0),
        res: 0.3,
        keytrack: 0.5,
        a: 0.02,
        d: 0.2,
        s: 0.95,
        r: 0.05,
        glide: 0.01,
        gain: 0.4,
        ..PATCH
    }
}

fn psaltery(r: &mut Rng) -> Patch {
    Patch { filter: false, ks: r.range(2.5, 3.5), ks_tone: 0.85, a: 0.001, d: 4.0, s: 1.0, r: 0.5, gain: 0.45, ..PATCH }
}

fn pipe_drone(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Square,
        pw: 0.35,
        unison: 2,
        detune: r.range(2.0, 4.0),
        vowel: 5,
        cutoff: 2200.0,
        keytrack: 0.0,
        a: 0.6,
        d: 1.0,
        s: 1.0,
        r: 1.0,
        gain: 0.4,
        ..PATCH
    }
}

fn viol(r: &mut Rng) -> Patch {
    Patch {
        vowel: 4,
        noise: 0.02,
        cutoff: r.range(900.0, 1300.0),
        keytrack: 0.4,
        a: 0.05,
        d: 0.4,
        s: 0.8,
        r: 0.15,
        vib: 0.08,
        vib_rate: 5.0,
        gain: 0.6,
        ..PATCH
    }
}

/// The bodhran (or a frame drum): [the open bass stroke, the muted high one].
fn kit_bodhran(r: &mut Rng) -> Vec<DrumPatch> {
    let bass = DrumPatch {
        tone: 1.0,
        f0: r.range(130.0, 150.0),
        f1: r.range(80.0, 95.0),
        pdec: 0.03,
        tdec: 0.35,
        noise: 0.5,
        nmode: 0,
        ncut: 400.0,
        ndec: 0.12,
        gain: 0.9,
        ..DRUM
    };
    let top = DrumPatch { f0: 260.0, f1: 220.0, tdec: 0.08, noise: 0.6, ncut: 1200.0, ndec: 0.05, gain: 0.55, ..bass };
    vec![bass, top]
}
fn kit_frame(r: &mut Rng) -> Vec<DrumPatch> {
    let bass = kit_kick_frame(r)[0];
    vec![bass, DrumPatch { f0: 240.0, f1: 200.0, tdec: 0.1, noise: 0.5, ncut: 1500.0, ndec: 0.06, gain: 0.5, ..bass }]
}
/// Percussion: [spoons, a low and a high frame-drum stroke, finger cymbals, shaker].
fn kit_bones(r: &mut Rng) -> Vec<DrumPatch> {
    let mut v = kit_spoons(r);
    v[0] = DrumPatch {
        tone: 0.6,
        f0: 1800.0,
        f1: 1700.0,
        tdec: 0.04,
        noise: 0.5,
        nmode: 1,
        ncut: 2500.0,
        ndec: 0.02,
        claps: 2,
        gain: 0.55,
        ..DRUM
    };
    v
}

fn band_tavern(r: &mut Rng, s: &Synth) -> Band {
    let mut b = Band::default();
    b.tone(r, s, CH_LEAD, P_TAV_LEAD, 74.0);
    b.tone(r, s, CH_BELL, P_TAV_SECOND, 74.0);
    b.tone(r, s, CH_KEYS, P_TAV_PLUCK, 60.0);
    b.tone(r, s, CH_DRONE, P_TAV_DRONE, 45.0);
    b.tone(r, s, CH_BASS, P_TAV_BASS, 43.0);
    b.kit(r, s, CH_KICK, K_TAV_KICK);
    b.kit(r, s, CH_HAT, K_TAV_HAT);
    b.kit(r, s, CH_PERC, K_TAV_PERC);
    b
}

fn sounds_tavern(r: &mut Rng, s: &mut Synth) {
    let ch = &mut s.ch;
    ch[CH_LEAD].patch = fiddle(r);
    ch[CH_LEAD].mono = true;
    ch[CH_BELL].patch = recorder(r);
    ch[CH_KEYS].patch = lute(r);
    ch[CH_DRONE].patch = drone_reed(r);
    ch[CH_BASS].patch = viol(r);
    ch[CH_BASS].mono = true;
    ch[CH_KICK].drums = kit_bodhran(r);
    ch[CH_HAT].drums = kit_tambourine(r);
    ch[CH_PERC].drums = kit_spoons(r);
    ch[CH_SNARE].drums = kit_hand_clap(r);
    mix(&mut ch[CH_KICK], 0.6, 0.0, 0.1, 0.0, 0.0);
    mix(&mut ch[CH_HAT], 0.3, 0.35, 0.15, 0.0, 0.0);
    mix(&mut ch[CH_PERC], 0.35, -0.35, 0.15, 0.0, 0.0);
    mix(&mut ch[CH_SNARE], 0.35, 0.1, 0.2, 0.0, 0.0);
    mix(&mut ch[CH_KEYS], 0.45, -0.2, 0.2, 0.0, 0.0);
    mix(&mut ch[CH_LEAD], 0.42, 0.1, 0.22, 0.05, 0.0);
    mix(&mut ch[CH_BELL], 0.32, 0.3, 0.22, 0.05, 0.0);
    mix(&mut ch[CH_DRONE], 0.22, -0.1, 0.2, 0.0, 0.0);
    mix(&mut ch[CH_BASS], 0.35, 0.0, 0.1, 0.0, 0.0);
    // a room, not a hall
    s.rev.fb = r.range(0.72, 0.78);
    s.rev.damp = 0.5;
    s.rev_wet = r.range(0.14, 0.2);
    s.dly.fb = 0.0;
    s.dly_rev = 0.0;
    s.r2mode = R2_OFF;
    s.fx.crackle = 0.0;
    s.fx.wind_to = 0.0;
    s.fx.rain_to = 0.0;
    s.m.gain = 2.1;
    s.m.target = 0.21;
}

// -- ska

/// What a ska section plays: the intro, the head, a solo, the breakdown
/// (the third wave's bridge), the tag, the third wave's horn break.
const SK_INTRO: u8 = 0;
const SK_HEAD: u8 = 1;
const SK_SOLO: u8 = 2;
const SK_BREAK: u8 = 3;
const SK_TAG: u8 = 4;
const SK_RUN: u8 = 5;
/// Ska's styles: the first wave, 2-tone, the third wave (ska-punk).
const SKS_FIRST: u8 = 0;
const SKS_TWO: u8 = 1;
const SKS_THIRD: u8 = 2;
/// Intros: the rhythm section's vamp (the horns' pickup into the head), a
/// horn riff over it, a bar of the drums alone first.
const SKI_VAMP: u8 = 0;
const SKI_RIFF: u8 = 1;
const SKI_DRUMS: u8 = 2;
/// Endings (and 0: a hit on the one): the Charleston (the one and the
/// and-of-two), a chord held over a roll and cut off.
const SKE_CHARLESTON: u8 = 1;
const SKE_HOLD: u8 = 2;
/// How the second and third horns go under the lead: octaves, a third
/// below, a sixth below, close harmony.
const SKH_OCT: u8 = 0;
const SKH_THIRD: u8 = 1;
const SKH_SIXTH: u8 = 2;
const SKH_TRIAD: u8 = 3;
/// The head's range (semitones about the tonic) and about where its tonic sits (MIDI).
const SK_LO: i32 = -3;
const SK_HI: i32 = 12;
const SK_TUNE: i32 = 67;
/// A note's grace marking a fall off its end.
const SKA_FALL: i8 = -12;

/// Ska strains, eight bars (a chord a bar; A+B two): A strains come home,
/// bridges (B) leave it and end on the dominant; major, mixolydian, minor
/// (2-tone's and the dorian first wave's).
const SKA_MAJ_A: &[&str] = &[
    "I I IV IV V7 V7 I I",
    "I vi IV V7 I vi IV+V7 I",
    "I IV I V7 I IV V7 I",
    "I I7 IV iv I V7/V V7 I",
    "I vi ii V7 I vi ii+V7 I",
    "I IV+V7 I IV+V7 I IV ii+V7 I",
    "I I IV IV I vi ii+V7 I",
    "I V7/vi vi IV I V7/V V7 I",
    "I+IV V7+IV I+IV V7+IV I+IV V7+IV ii+V7 I",
    "I I V7 V7 I I7 IV+iv I",
];
const SKA_MAJ_B: &[&str] = &[
    "IV IV I I IV IV V7 V7",
    "vi vi IV IV ii ii V7 V7",
    "IV iv I vi ii ii V7 V7",
    "I7 I7 IV IV V7/V V7/V V7 V7",
    "vi V7/vi vi V7/vi IV IV V7 V7",
    "IV V7 I vi IV V7 I V7",
    "ii ii V7 V7 I I V7/V V7",
];
const SKA_MIX_A: &[&str] = &[
    "I bVII IV I I bVII IV+V7 I",
    "I I bVII bVII IV IV I I",
    "I bVII I IV I bVII IV+bVII I",
    "I IV bVII IV I IV bVII+IV I",
    "I I7 IV bVII I IV V7 I",
];
const SKA_MIX_B: &[&str] = &["bVII bVII IV IV I I V7 V7", "IV IV bVII bVII I I V7 V7", "IV bVII I I IV bVII V7 V7"];
const SKA_MIN_A: &[&str] = &[
    "i i iv iv V7 V7 i i",
    "i bVII bVI V7 i bVII bVI+V7 i",
    "i iv i V7 i iv V7 i",
    "i i bVII bVII bVI bVI V7 i",
    "i+bVII bVI+V7 i+bVII bVI+V7 i+bVII bVI+V7 iv+V7 i",
    "i i IV IV i i V7 i",
    "i iv bVII bIII bVI iv V7 i",
    "i bIII iv V7 i bIII iv+V7 i",
    "i iv v iv i iv v i",
];
const SKA_MIN_B: &[&str] = &[
    "iv iv i i bVI bVI V7 V7",
    "bIII bIII bVII bVII iv iv V7 V7",
    "bVI bVII i i bVI bVII V7 V7",
    "iv bVII bIII bVI iim7b5 iim7b5 V7 V7",
    "bVI bVI bIII bIII iv iv V7 V7",
];
/// Four-bar vamps, for the intro.
const SKA_VAMP_MAJ: &[&str] = &["I+IV V7+IV I+IV V7+IV", "I I IV V7", "I IV V7 IV", "I vi IV V7"];
const SKA_VAMP_MIX: &[&str] = &["I bVII IV I", "I+bVII IV+I I+bVII IV+V7", "I I bVII IV"];
const SKA_VAMP_MIN: &[&str] = &["i+bVII bVI+V7 i+bVII bVI+V7", "i i iv V7", "i iv V7 iv", "i bVII bVI V7"];
/// The third wave's: verses (A, home again), choruses (B, ending on the
/// dominant or the bVII), bridges (C), minor mostly (i-bVI-bIII-bVII and
/// its kin), and intros.
const SKA_TW_MIN_A: &[&str] = &[
    "i bVI bIII bVII i bVI bVII i",
    "i i bVI bVI bIII bIII bVII i",
    "i bVII bVI bVII i bVII bVI+bVII i",
    "i iv bVII bIII bVI iv V7 i",
    "i+bVI bIII+bVII i+bVI bIII+bVII i+bVI bIII+bVII iv+bVII i",
    "i bIII bVII iv i bIII bVII+V7 i",
];
const SKA_TW_MIN_B: &[&str] = &[
    "bVI bVII i i bVI bVII V7 V7",
    "bVI bVII bIII bIII bVI bVII V7 V7",
    "iv iv i i bVI bVII V7 V7",
    "bIII bVII bVI bVI bIII bVII bVI bVII",
    "bVI bIII bVII i bVI bIII bVII bVII",
];
const SKA_TW_MIN_C: &[&str] = &[
    "bVI bVI bVII bVII i i bVII bVII",
    "iv iv bVI bVI bVII bVII V7 V7",
    "bIII bIII bVII bVII bVI bVI V7 V7",
    "i i bVI bVI iv iv V7 V7",
];
const SKA_TW_MAJ_A: &[&str] =
    &["I V vi IV I V IV+V I", "I IV vi V I IV V I", "I iii IV V I iii IV+V I", "I I IV IV vi vi V I"];
const SKA_TW_MAJ_B: &[&str] = &["IV V vi IV IV V V7 V7", "vi IV I V vi IV V7 V7", "IV IV I I vi vi V7 V7"];
const SKA_TW_MAJ_C: &[&str] = &["vi vi IV IV I I V V", "ii ii IV IV vi vi V7 V7", "IV IV vi vi IV IV V7 V7"];
const SKA_TW_VAMP_MIN: &[&str] = &["i bVI bIII bVII", "i bVII bVI bVII", "i i bVI bVII"];
const SKA_TW_VAMP_MAJ: &[&str] = &["I V vi IV", "I IV I V", "vi IV I V"];

/// Rhythms a bar of a head is made of (16th steps, lengths): quarters and
/// eighths, the 3+3+2 push, dotted figures, a bar starting on the
/// off-beat, one ending on the last off-beat (pushed onto the next one).
const SKA_CELLS: &[&[(u16, u16)]] = &[
    &[(0, 4), (4, 4), (8, 2), (10, 2), (12, 4)],
    &[(0, 3), (3, 3), (6, 2), (8, 4), (12, 2)],
    &[(0, 2), (2, 2), (4, 2), (6, 6), (14, 2)],
    &[(0, 6), (6, 2), (8, 2), (10, 4)],
    &[(2, 2), (4, 2), (6, 4), (10, 2), (12, 4)],
    &[(0, 3), (3, 1), (4, 3), (7, 1), (8, 4), (12, 2)],
    &[(0, 4), (6, 2), (8, 4), (14, 2)],
    &[(0, 2), (2, 4), (6, 2), (8, 2), (10, 4), (14, 2)],
    &[(4, 2), (6, 2), (8, 4), (12, 2), (14, 2)],
    &[(0, 2), (2, 2), (4, 4), (8, 2), (10, 2), (12, 2)],
];
/// A phrase's last bar: a long note, with room after it for an answer.
const SKA_CADENCE: &[&[(u16, u16)]] = &[
    &[(0, 4), (4, 4), (8, 4)],
    &[(0, 2), (2, 2), (4, 8)],
    &[(0, 3), (3, 3), (6, 6)],
    &[(0, 2), (2, 2), (4, 2), (6, 6)],
    &[(0, 8)],
];
/// A solo's bars: running eighths, a run of 16ths, three against four.
const SKA_SOLO_CELLS: &[&[(u16, u16)]] = &[
    &[(0, 2), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2), (12, 4)],
    &[(2, 2), (4, 1), (5, 1), (6, 2), (8, 4), (12, 2), (14, 2)],
    &[(0, 4), (4, 2), (6, 2), (8, 2), (10, 2), (12, 2), (14, 2)],
    &[(0, 1), (1, 1), (2, 1), (3, 1), (4, 2), (6, 2), (8, 6), (14, 2)],
    &[(0, 6), (6, 2), (8, 2), (10, 2), (12, 4)],
    &[(0, 2), (3, 3), (6, 2), (8, 2), (11, 3), (14, 2)],
    &[(0, 3), (3, 3), (6, 3), (9, 3), (12, 4)],
    &[(2, 2), (6, 2), (8, 1), (9, 1), (10, 2), (12, 2), (14, 2)],
];
/// A soloist's breath: nothing, or a pickup into the next bar.
const SKA_PICKUPS: &[&[(u16, u16)]] =
    &[&[], &[(12, 2), (14, 2)], &[(10, 2), (12, 1), (13, 1), (14, 2)], &[(9, 1), (10, 2), (12, 2), (14, 2)]];
/// A riff's bars: stabs on the off-beats, pushes.
const SKA_RIFFS: &[&[(u16, u16)]] = &[
    &[(2, 1), (4, 1), (6, 2), (10, 1), (12, 2)],
    &[(0, 2), (3, 1), (6, 2), (14, 2)],
    &[(2, 2), (6, 1), (8, 1), (10, 4)],
    &[(0, 1), (2, 1), (4, 1), (6, 4), (12, 1), (14, 1)],
    &[(6, 2), (8, 1), (10, 1), (12, 4)],
    &[(0, 3), (3, 3), (6, 2), (10, 2), (14, 2)],
];

/// The third wave's horn runs, a bar each: 16ths up or down the scale and
/// into eighths, eighths straight through, pairs of 16ths, a syncopated
/// line breaking into 16ths.
const SKA_TW_RUNS: &[&[(u16, u16)]] = &[
    &[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 2), (10, 2), (12, 4)],
    &[(0, 2), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2), (12, 2), (14, 2)],
    &[(0, 2), (2, 1), (3, 1), (4, 2), (6, 1), (7, 1), (8, 2), (10, 1), (11, 1), (12, 4)],
    &[(0, 3), (3, 3), (6, 2), (8, 1), (9, 1), (10, 1), (11, 1), (12, 4)],
    &[(2, 1), (3, 1), (4, 1), (5, 1), (6, 2), (8, 2), (10, 2), (12, 2), (14, 2)],
    &[(0, 2), (2, 2), (4, 1), (5, 1), (6, 1), (7, 1), (8, 4), (12, 1), (13, 1), (14, 2)],
];

/// A ska section: its part, the strain it plays (0 A, 1 B, the third
/// wave's bridge 2 C), which time through that strain, whether it is the
/// head out, its instruments (the band's A or B: `which`), its soloist;
/// whether it is in the third wave's punk drive, and in half time.
#[derive(Clone, Copy, Default)]
struct SkaSec {
    part: u8,
    strain: u8,
    rep: u8,
    out: bool,
    dress: u8,
    solo: u8,
    punk: bool,
    half: bool,
}

/// A ska strain: eight bars of chords (two a bar), the head over them and
/// the stabs answering it (16ths from the strain's start, semitones above
/// the tonic).
#[derive(Clone)]
struct SkaStrain {
    chords: Vec<Harm>,
    mel: Vec<PNote>,
    ans: Vec<PNote>,
}

/// The ska piece in play.
#[derive(Default)]
struct Ska {
    /// 16ths a bar and a beat, and the meter's (and the style's) name
    bar_steps: u32,
    beat_steps: u32,
    meter: &'static str,
    two_tone: bool,
    third: bool,
    /// how late the off-beat eighths come (16ths): the first wave's shuffle
    shuffle: f32,
    /// the kick on every beat, the open hat on every off-beat, the organ's
    /// bubble (two stabs an off-beat), the organ holding chords this section
    four: bool,
    open: bool,
    bubble: bool,
    hold: bool,
    /// the intro's kind and the ending's
    intro: u8,
    end: u8,
    strains: Vec<SkaStrain>,
    /// the intro's vamp, the head's strains in order (AB or AABA), the sections
    vamp: Vec<Harm>,
    form: Vec<u8>,
    map: Vec<SkaSec>,
    /// this time through: the head, the answers, how the horns voice it
    mel: Vec<PNote>,
    ans: Vec<PNote>,
    harmony: u8,
    /// two-bar riffs over the home chord: the intro's, and the one behind a solo
    riff: Vec<PNote>,
    back: Vec<PNote>,
    /// the soloists (as loud as the lead's horn) and their registers (MIDI, about)
    solos: Vec<(Patch, i32)>,
    /// the walking bass's last note, the soloist's (above the tonic) and
    /// which way it goes, the piano's voicing
    bass_prev: i32,
    solo_prev: i32,
    solo_dir: i32,
    piano_v: Vec<i32>,
    /// the third wave's: this section's punk beat (0 the skank beat, 1 a
    /// rock beat), whether its power chords ring (or are palm-muted), the
    /// horn break's line
    drive: u8,
    ring: bool,
    run: Vec<PNote>,
}

/// A first-wave or 2-tone piece's form: an intro (a vamp, a horn riff
/// over it, or the drums leading in), the head (AB or AABA), as many solo
/// choruses (and a breakdown) as a minute and a half to two and a half
/// holds at bars `bar_s` seconds long (an AABA too long comes back from the
/// bridge, or is AB), the head out and the tag. The sections, the head's
/// strains in order, the solos, the intro's kind.
fn ska_form(r: &mut Rng, bar_s: f64, two_tone: bool) -> (Vec<(Sec, SkaSec)>, Vec<u8>, u32, u8) {
    let tag = if r.bit_chance() { 3 } else { 5 };
    let mut aaba = r.chance(0.5);
    let mut from_bridge = aaba && r.chance(0.5);
    let (mut solos, brk);
    loop {
        let n = if aaba { 32 } else { 16 };
        let base = 4 + n + if from_bridge { 16 } else { n } + tag;
        solos = 1;
        while ((base + solos * n) as f64) * bar_s < 95.0 && solos < 3 {
            solos += 1;
        }
        let t = (base + solos * n) as f64 * bar_s;
        if t > 150.0 && aaba {
            (aaba, from_bridge) = (!from_bridge, !from_bridge);
            continue;
        }
        brk = t + 8.0 * bar_s <= 150.0 && r.chance(0.5);
        break;
    }
    let form: Vec<u8> = if aaba { vec![0, 0, 1, 0] } else { vec![0, 1] };
    let organ = if two_tone || r.chance(0.5) { L_PAD } else { 0 };
    let piano = if !two_tone || r.chance(0.35) { L_DRONE } else { 0 };
    let rhythm = L_KICK | L_SNARE | L_HAT | L_OHAT | L_BASS | L_KEYS;
    let horns = L_LEAD | L_BELL | L_ARP;
    let intro = r.weighted(&[1.0, 1.2, 0.8]) as u8;
    let sx = |part: u8, strain: u8, rep: u8, dress: u8| SkaSec { part, strain, rep, dress, ..SkaSec::default() };
    let riffing = if intro == SKI_RIFF { horns } else { 0 };
    let mut plan = vec![(
        Sec {
            kind: INTRO,
            bars: 4,
            mask: rhythm | organ | riffing | if intro == SKI_VAMP { 0 } else { L_PERC },
            energy: 0.65,
        },
        sx(SK_INTRO, 0, 0, 0),
    )];
    // the head: the organ joins after the first strain, the piano leaves the bridge to it
    let mut reps = [0u8; 2];
    for (k, &st) in form.iter().enumerate() {
        let keys =
            if k == 0 { piano | if piano == 0 { organ } else { 0 } } else { organ | if st == 0 { piano } else { 0 } };
        let sec = Sec { kind: MAIN, bars: 8, mask: rhythm | horns | keys, energy: 0.72 + 0.04 * k as f32 };
        plan.push((sec, sx(SK_HEAD, st, reps[st as usize], st)));
        reps[st as usize] += 1;
    }
    // the solos, a chorus each, the riffs behind them
    let n = 8 * form.len() as u32;
    for k in 0..solos {
        let mask = rhythm | L_PERC | L_LEAD | L_CTR | if organ != 0 { organ } else { piano };
        let x = SkaSec { part: SK_SOLO, solo: k as u8, dress: 1 + (k % 2) as u8, ..SkaSec::default() };
        plan.push((Sec { kind: MAIN, bars: n, mask, energy: 0.8 + 0.05 * k as f32 }, x));
    }
    // the breakdown: the bass, the guitar and the drums, quietly
    if brk {
        let mask = L_KICK | L_SNARE | L_HAT | L_BASS | L_KEYS | L_PERC;
        plan.push((Sec { kind: BREAK, bars: 8, mask, energy: 0.5 }, sx(SK_BREAK, 0, 0, 2)));
    }
    for (k, &st) in form.iter().enumerate().skip(if from_bridge { 2 } else { 0 }) {
        let sec =
            Sec { kind: MAIN, bars: 8, mask: rhythm | horns | organ | piano | L_PERC, energy: 0.86 + 0.03 * k as f32 };
        plan.push((sec, SkaSec { out: true, ..sx(SK_HEAD, st, reps[st as usize], st) }));
        reps[st as usize] += 1;
    }
    let mask = rhythm | horns | organ | piano | L_PERC;
    plan.push((Sec { kind: OUTRO, bars: tag, mask, energy: 1.0 }, sx(SK_TAG, 0, 0, 0)));
    (plan, form, solos, intro)
}

/// The third wave's form: a horn line over the band in the punk drive, two
/// verses (A, the skank under the horns' tune, once or twice through) each
/// answered by a chorus (B, in the punk drive), the horn break (runs over
/// the chorus's chords), the bridge (C: a half-time breakdown, or quiet
/// and building back), the last chorus (once or twice through) and one hit
/// to end it.
fn ska_third_form(r: &mut Rng) -> Vec<(Sec, SkaSec)> {
    let ska = L_KICK | L_SNARE | L_HAT | L_OHAT | L_BASS | L_KEYS;
    let punk = L_KICK | L_SNARE | L_HAT | L_OHAT | L_BASS | L_DRONE | L_PERC;
    let horns = L_LEAD | L_BELL | L_ARP | L_PAD;
    // the band's B instruments in the punk drive
    let sx = |part: u8, strain: u8, rep: u8, punk: bool| SkaSec {
        part,
        strain,
        rep,
        punk,
        dress: punk as u8,
        ..SkaSec::default()
    };
    let mut plan = vec![(Sec { kind: INTRO, bars: 4, mask: punk | horns, energy: 0.8 }, sx(SK_INTRO, 0, 0, true))];
    for k in 0..2u8 {
        let e = 0.05 * k as f32;
        plan.push((Sec { kind: MAIN, bars: 8, mask: ska | horns, energy: 0.7 + e }, sx(SK_HEAD, 0, k, false)));
        plan.push((Sec { kind: MAIN, bars: 8, mask: punk | horns, energy: 0.85 + e }, sx(SK_HEAD, 1, k, true)));
    }
    plan.push((Sec { kind: MAIN, bars: 8, mask: punk | horns, energy: 0.9 }, sx(SK_RUN, 1, 0, true)));
    let bridge = if r.chance(0.55) {
        let x = SkaSec { half: true, dress: 2, ..sx(SK_BREAK, 2, 0, true) };
        (Sec { kind: BREAK, bars: 8, mask: punk | horns, energy: 0.75 }, x)
    } else {
        let mask = L_KICK | L_SNARE | L_HAT | L_BASS | L_KEYS | horns;
        (Sec { kind: BREAK, bars: 8, mask, energy: 0.5 }, SkaSec { dress: 2, ..sx(SK_BREAK, 2, 0, false) })
    };
    plan.push(bridge);
    let x = SkaSec { out: true, ..sx(SK_HEAD, 1, 2, true) };
    plan.push((Sec { kind: MAIN, bars: 16, mask: punk | horns, energy: 1.0 }, x));
    plan.push((Sec { kind: OUTRO, bars: 1, mask: punk | horns, energy: 1.0 }, sx(SK_TAG, 0, 0, true)));
    plan
}

/// Which of a bar's chords a note sounds over: 0 the first half's, 1 the
/// second's, 2 the next bar's (a note pushed onto it from the last off-beat).
fn ska_half(st: u16, len: u16) -> usize {
    if st >= 14 && len >= 2 {
        2
    } else if st >= 8 {
        1
    } else {
        0
    }
}

/// How long a horn holds a note `len` 16ths long: the short ones clipped (ska's horns are punchy).
fn ska_len(len: u16) -> f32 {
    match len {
        0 | 1 => 0.7,
        2 => 1.5,
        3 => 2.4,
        l => l as f32 - 0.6,
    }
}

/// A bar of a horn line in `cell`'s rhythm over the bar's chords `c` (its
/// halves', and the next bar's for a note pushed onto it), from `prev`
/// going `dir`: the first note and the strong ones (on the beat, long, or
/// pushed) on chord tones (the same, the next one along, or the one after),
/// the others stepping through the chord-scale, leaping by chord tones or
/// repeating (a riff does); turning at the edges of [lo, hi], and now and
/// then on a whim.
#[allow(clippy::too_many_arguments)]
fn ska_figure(
    r: &mut Rng,
    cell: &[(u16, u16)],
    c: [Harm; 3],
    mode: usize,
    prev: i32,
    dir: &mut i32,
    lo: i32,
    hi: i32,
) -> Vec<PNote> {
    let at = |v: &[i32], k: isize| v[k.clamp(0, v.len() as isize - 1) as usize];
    let mut p = prev.clamp(lo, hi);
    let mut v = Vec::with_capacity(cell.len());
    for (k, &(st, len)) in cell.iter().enumerate() {
        let h = c[ska_half(st, len)];
        let ct = in_range(&h.pcs(), lo, hi);
        let sc = in_range(&chord_scale(mode, h), lo, hi);
        let strong = st.is_multiple_of(4) || len >= 4 || ska_half(st, len) == 2;
        let d = *dir as isize;
        p = if k == 0 || strong {
            let i = nearest(&ct, p) as isize;
            at(&ct, i + d * r.weighted(&[0.25, 0.55, 0.2]) as isize)
        } else {
            match r.weighted(&[0.5, 0.3, 0.2]) {
                0 => at(&sc, nearest(&sc, p) as isize + d),
                1 => at(&ct, nearest(&ct, p) as isize + d),
                _ => sc[nearest(&sc, p)],
            }
        };
        let vel = match st % 4 {
            0 => 0.9,
            2 => 0.86,
            _ => 0.78,
        } + r.range(-0.04, 0.04);
        v.push(PNote { step: st, len, pitch: p, vel, grace: 0 });
        if p >= hi - 2 {
            *dir = -1;
        } else if p <= lo + 2 {
            *dir = 1;
        } else if r.chance(0.2) {
            *dir = -*dir;
        }
    }
    v
}

/// A phrase's last note brought to rest on the nearest of `targets`
/// (semitones above the tonic, in any octave), a tone of its chord.
fn ska_land(v: &mut [PNote], c: [Harm; 3], targets: &[i32], lo: i32, hi: i32) {
    if let Some(n) = v.last_mut() {
        let ct = in_range(&c[ska_half(n.step, n.len)].pcs(), lo, hi);
        let want: Vec<i32> = targets.iter().map(|&x| x + 12 * ((n.pitch - x) as f32 / 12.0).round() as i32).collect();
        let goal = want[nearest(&want, n.pitch)].clamp(lo, hi);
        n.pitch = ct[nearest(&ct, goal)];
    }
}

/// A bar of a line (16ths within the bar) moved from chords `from` onto
/// `to` (each half's, and the next bar's for a pushed note): chord tones to
/// the nearest tone of the new chord, the rest to the nearest of its scale.
fn ska_refit(notes: &[PNote], from: [Harm; 3], to: [Harm; 3], mode: usize, lo: i32, hi: i32) -> Vec<PNote> {
    notes
        .iter()
        .map(|n| {
            let k = ska_half(n.step % 16, n.len);
            if from[k] == to[k] {
                return *n;
            }
            let set =
                if from[k].pcs().contains(&n.pitch.rem_euclid(12)) { to[k].pcs() } else { chord_scale(mode, to[k]) };
            let v = in_range(&set, lo, hi);
            PNote { pitch: v[nearest(&v, n.pitch)], ..*n }
        })
        .collect()
}

/// A strain's head: eight bars in two four-bar phrases, the first two bars
/// a motif that comes back fitted to the chords in bars five and six (bar
/// seven now and then in its rhythm again); bar four a half cadence, bar
/// eight home (a bridge's, on the dominant's tones). Where the head rests
/// at a bar's end (a cadence's, mostly) the low horns answer: two or three
/// stabs on the off-beats.
fn ska_tune(r: &mut Rng, chords: &[Harm], mode: usize, lo: i32, hi: i32, home: bool) -> (Vec<PNote>, Vec<PNote>) {
    let ch = |bar: usize| [chords[bar * 2], chords[bar * 2 + 1], chords[(bar * 2 + 2) % chords.len()]];
    let motif = [r.pick(SKA_CELLS), r.pick(SKA_CELLS)];
    let mut bars: Vec<Vec<PNote>> = Vec::with_capacity(8);
    let mut prev = r.pick(&[4, 7, 7, 12, 0]).clamp(lo, hi);
    let mut dir = if prev > (lo + hi) / 2 { -1 } else { 1 };
    for bar in 0..8 {
        let notes = if (bar == 4 || bar == 5) && r.chance(0.85) {
            ska_refit(&bars[bar - 4], ch(bar - 4), ch(bar), mode, lo, hi)
        } else if bar == 3 || bar == 7 {
            let cell = r.pick(SKA_CADENCE);
            let mut v = ska_figure(r, cell, ch(bar), mode, prev, &mut dir, lo, hi);
            let targets: &[i32] = match (bar, home) {
                (7, true) => &[0, 12, 4],
                (7, false) => &[7, 11, 2],
                _ => &[7, 4, 2],
            };
            ska_land(&mut v, ch(bar), targets, lo, hi);
            v
        } else {
            let cell = match bar {
                0 | 1 => motif[bar],
                6 if r.chance(0.4) => motif[0],
                _ => r.pick(SKA_CELLS),
            };
            ska_figure(r, cell, ch(bar), mode, prev, &mut dir, lo, hi)
        };
        prev = notes.last().map_or(prev, |n| n.pitch);
        bars.push(notes);
    }
    let mut mel = Vec::new();
    let mut ans = Vec::new();
    for (bar, notes) in bars.iter().enumerate() {
        let st0 = 16 * bar as u16;
        mel.extend(notes.iter().map(|n| PNote { step: n.step + st0, ..*n }));
        let end = notes.last().map_or(0, |n| n.step + n.len);
        let last = notes.last().map_or(prev, |n| n.pitch);
        let k: u16 = if end <= 10 && r.chance(0.4) { 3 } else { 2 };
        if end + 2 * k <= 16 && r.chance(if bar % 4 == 3 { 0.85 } else { 0.4 }) {
            let c = ch(bar);
            let shape = r.below(3);
            for j in 0..k {
                let st = 16 - 2 * (k - j);
                let ct = in_range(&c[ska_half(st, 1)].pcs(), lo - 7, hi - 3);
                let j0 = nearest(&ct, last - 4) as isize;
                let d = match shape {
                    0 => 0,
                    1 => (j + 1 == k) as isize,
                    _ => 1 - j as isize,
                };
                let pitch = ct[(j0 + d).clamp(0, ct.len() as isize - 1) as usize];
                ans.push(PNote { step: st0 + st, len: 1, pitch, vel: if j + 1 == k { 0.95 } else { 0.85 }, grace: 0 });
            }
        }
    }
    (mel, ans)
}

/// Scoops: a few long notes on the beat or the off-beat come up from a
/// semitone or two below.
fn ska_scoops(r: &mut Rng, mel: &mut [PNote], p: f32) {
    for n in mel.iter_mut() {
        if n.len >= 3 && n.step.is_multiple_of(2) && n.grace == 0 && r.chance(p) {
            n.grace = r.pick(&[-1, -1, -2]);
        }
    }
}

/// A two-bar riff over the home chord: a cell of `cells` (stabs and
/// pushes; the third wave's runs), its second bar the same rhythm from
/// another chord tone (or another cell).
fn ska_riff(r: &mut Rng, home: Harm, mode: usize, lo: i32, hi: i32, cells: &[&[(u16, u16)]]) -> Vec<PNote> {
    let c = [home; 3];
    let ct = in_range(&home.pcs(), lo + 3, hi - 3);
    let mut dir = 1;
    let cell = r.pick(cells);
    let p = ct[r.below(ct.len())];
    let a = ska_figure(r, cell, c, mode, p, &mut dir, lo, hi);
    let cell2 = if r.chance(0.6) { cell } else { r.pick(cells) };
    let p = ct[r.below(ct.len())];
    let b = ska_figure(r, cell2, c, mode, p, &mut dir, lo, hi);
    a.into_iter().chain(b.into_iter().map(|n| PNote { step: n.step + 16, ..n })).collect()
}

/// The third wave's horn break: eight bars of runs over a strain's chords
/// (the first two bars back in the fifth and sixth, fitted to theirs), each
/// four-bar phrase closing on a long note.
fn ska_runs(r: &mut Rng, chords: &[Harm], mode: usize, lo: i32, hi: i32) -> Vec<PNote> {
    let ch = |bar: usize| [chords[bar * 2], chords[bar * 2 + 1], chords[(bar * 2 + 2) % chords.len()]];
    let mut bars: Vec<Vec<PNote>> = Vec::with_capacity(8);
    let mut prev = r.pick(&[0, 4, 7, 12]).clamp(lo, hi);
    let mut dir = if prev > (lo + hi) / 2 { -1 } else { 1 };
    for bar in 0..8 {
        let notes = if (bar == 4 || bar == 5) && r.chance(0.7) {
            ska_refit(&bars[bar - 4], ch(bar - 4), ch(bar), mode, lo, hi)
        } else {
            let cell = if bar % 4 == 3 { r.pick(SKA_CADENCE) } else { r.pick(SKA_TW_RUNS) };
            ska_figure(r, cell, ch(bar), mode, prev, &mut dir, lo, hi)
        };
        prev = notes.last().map_or(prev, |n| n.pitch);
        bars.push(notes);
    }
    let mut run = Vec::new();
    for (bar, notes) in bars.iter().enumerate() {
        run.extend(notes.iter().map(|n| PNote { step: n.step + 16 * bar as u16, ..*n }));
    }
    run
}

/// A clean electric guitar's upstroke, muted after it speaks (a short
/// gate, a quick release): a bright string, the filter closing quickly.
fn ska_guitar_chop(r: &mut Rng) -> Patch {
    Patch {
        ks: r.range(0.35, 0.6),
        ks_tone: r.range(0.72, 0.85),
        cutoff: r.range(2800.0, 4000.0),
        res: 0.15,
        env_amt: 0.8,
        fdec: 0.06,
        keytrack: 0.3,
        a: 0.001,
        d: 0.12,
        s: 0.4,
        r: 0.035,
        gain: 0.55,
        ..PATCH
    }
}

/// The skank on a clean guitar left to ring a little (a bridge pickup's twang).
fn ska_guitar_clean(r: &mut Rng) -> Patch {
    Patch { filter: false, ks: r.range(0.5, 0.8), ks_tone: 0.88, a: 0.001, d: 0.3, s: 0.7, r: 0.07, gain: 0.5, ..PATCH }
}

/// The third wave's skank: a guitar's chop with some crunch (an amp turned up).
fn ska_guitar_crunch(r: &mut Rng) -> Patch {
    Patch {
        ks: r.range(0.3, 0.5),
        ks_tone: r.range(0.8, 0.9),
        cutoff: r.range(3000.0, 4200.0),
        res: 0.15,
        env_amt: 0.8,
        fdec: 0.06,
        keytrack: 0.3,
        a: 0.001,
        d: 0.12,
        s: 0.45,
        r: 0.035,
        drive: r.range(1.5, 3.0),
        gain: 0.45,
        ..PATCH
    }
}

/// A distorted guitar for power chords, double-tracked (two saws a few
/// cents apart, left and right, the amp's mids pushed ahead of it, an "a"
/// formant, and driven hard).
fn ska_guitar_dist(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Saw,
        unison: 2,
        detune: r.range(6.0, 10.0),
        width: 1.0,
        wave2: Wave::Square,
        mix2: 0.3,
        vowel: 1,
        cutoff: r.range(3000.0, 4000.0),
        res: 0.2,
        env_amt: 1.0,
        fdec: 0.08,
        keytrack: 0.4,
        a: 0.002,
        d: 0.6,
        s: 0.7,
        r: 0.06,
        drive: r.range(5.0, 8.0),
        gain: 0.3,
        ..PATCH
    }
}

/// A fuzzier take: squares, an "o" ahead of the amp, driven harder.
fn ska_guitar_fuzz(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Square,
        pw: r.range(0.3, 0.45),
        vowel: 4,
        unison: 2,
        detune: r.range(5.0, 9.0),
        width: 1.0,
        wave2: Wave::Saw,
        mix2: 0.25,
        semi2: 12.0,
        cutoff: r.range(1200.0, 1800.0),
        res: 0.25,
        env_amt: 1.2,
        fdec: 0.1,
        keytrack: 0.4,
        a: 0.002,
        d: 0.7,
        s: 0.75,
        r: 0.06,
        drive: r.range(8.0, 12.0),
        gain: 0.28,
        ..PATCH
    }
}

/// An upright piano, bright and a little hard (a high partial ringing on).
fn ska_piano_bright(r: &mut Rng) -> Patch {
    Patch {
        filter: false,
        fm: r.range(1.5, 2.0),
        ratio: 1.0,
        fm_dec: 0.5,
        fm2: 1.0,
        ratio2: 4.0,
        fm2_dec: 0.04,
        a: 0.002,
        d: 2.0,
        s: 0.0,
        r: 0.25,
        gain: 0.3,
        ..PATCH
    }
}

/// A Farfisa: thin, bright and buzzing (a narrow pulse, a saw an octave
/// up), a quick vibrato.
fn ska_farfisa(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Square,
        pw: r.range(0.22, 0.32),
        wave2: Wave::Saw,
        mix2: 0.35,
        semi2: 12.0,
        cutoff: r.range(4000.0, 5500.0),
        res: 0.1,
        keytrack: 0.3,
        a: 0.003,
        d: 0.1,
        s: 1.0,
        r: 0.04,
        vib: 0.06,
        vib_rate: 6.5,
        gain: 0.11,
        ..PATCH
    }
}

/// A Vox Continental: hollow (squares at the note and the octave), reedy, still.
fn ska_vox(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Square,
        pw: 0.5,
        wave2: Wave::Square,
        mix2: 0.45,
        semi2: 12.0,
        cutoff: r.range(3000.0, 4200.0),
        res: 0.12,
        keytrack: 0.3,
        a: 0.003,
        d: 0.1,
        s: 1.0,
        r: 0.05,
        gain: 0.11,
        ..PATCH
    }
}

/// A Hammond: drawbars (sines at the note and the octave, the sub-octave
/// under them), the Leslie's wobble.
fn ska_hammond(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Sine,
        wave2: Wave::Tri,
        mix2: r.range(0.4, 0.7),
        semi2: 12.0,
        sub: r.range(0.3, 0.6),
        cutoff: 5000.0,
        keytrack: 0.0,
        a: 0.002,
        d: 0.1,
        s: 1.0,
        r: 0.06,
        vib: 0.05,
        vib_rate: 6.8,
        trem: 0.12,
        drive: 0.3,
        gain: 0.16,
        ..PATCH
    }
}

/// An electric bass played with a pick: a saw's edge, the filter snapping shut.
fn ska_bass_pick(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Saw,
        wave2: Wave::Sine,
        mix2: 0.6,
        cutoff: r.range(500.0, 750.0),
        res: 0.15,
        env_amt: 1.6,
        fdec: 0.07,
        keytrack: 0.3,
        a: 0.002,
        d: 0.6,
        s: 0.45,
        r: 0.06,
        drive: 0.25,
        gain: 0.6,
        ..PATCH
    }
}

/// An electric bass played with the fingers: round, a little growl.
fn ska_bass_finger(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Tri,
        wave2: Wave::Saw,
        mix2: 0.15,
        sub: 0.35,
        cutoff: r.range(450.0, 650.0),
        res: 0.1,
        env_amt: 0.9,
        fdec: 0.12,
        keytrack: 0.3,
        a: 0.003,
        d: 0.9,
        s: 0.5,
        r: 0.07,
        drive: 0.2,
        gain: 0.75,
        ..PATCH
    }
}

/// Flatwound strings muted at the bridge: a thump and a short tone.
fn ska_bass_flat(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Sine,
        wave2: Wave::Tri,
        mix2: 0.35,
        cutoff: r.range(350.0, 500.0),
        env_amt: 0.8,
        fdec: 0.05,
        keytrack: 0.2,
        a: 0.002,
        d: 0.35,
        s: 0.3,
        r: 0.05,
        gain: 0.85,
        ..PATCH
    }
}

/// An alto sax: brighter and lighter than the tenor.
fn ska_alto(r: &mut Rng) -> Patch {
    Patch {
        fm: r.range(2.4, 3.0),
        ratio: 1.0,
        fm_dec: 2.0,
        cutoff: r.range(3200.0, 4000.0),
        res: 0.2,
        keytrack: 0.4,
        a: 0.03,
        d: 0.4,
        s: 0.85,
        r: 0.1,
        vib: 0.2,
        vib_rate: 5.8,
        glide: 0.025,
        gain: 0.26,
        ..PATCH
    }
}

/// A melodica: a free reed blown, plain and a little nasal.
fn ska_melodica(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Square,
        pw: r.range(0.38, 0.45),
        wave2: Wave::Saw,
        mix2: 0.25,
        vowel: 3,
        cutoff: 3200.0,
        keytrack: 0.3,
        a: 0.025,
        d: 0.3,
        s: 0.9,
        r: 0.06,
        trem: 0.06,
        vib_rate: 5.0,
        glide: 0.01,
        gain: 0.4,
        ..PATCH
    }
}

/// A flugelhorn: a trumpet's darker, rounder cousin.
fn ska_flugel(r: &mut Rng) -> Patch {
    Patch {
        unison: 2,
        detune: 3.0,
        width: 0.1,
        wave2: Wave::Sine,
        mix2: 0.5,
        cutoff: r.range(850.0, 1100.0),
        res: 0.1,
        env_amt: 1.0,
        fdec: 0.15,
        keytrack: 0.6,
        a: 0.03,
        d: 0.4,
        s: 0.85,
        r: 0.1,
        vib: 0.12,
        vib_rate: 5.2,
        glide: 0.02,
        gain: 0.33,
        ..PATCH
    }
}

/// A baritone sax: low and reedy, a growl in it.
fn ska_bari(r: &mut Rng) -> Patch {
    Patch {
        fm: r.range(2.8, 3.4),
        ratio: 1.0,
        fm_dec: 1.5,
        cutoff: r.range(1500.0, 2000.0),
        res: 0.15,
        keytrack: 0.3,
        a: 0.03,
        d: 0.4,
        s: 0.85,
        r: 0.1,
        vib: 0.08,
        vib_rate: 5.0,
        glide: 0.03,
        gain: 0.3,
        ..PATCH
    }
}

/// A jazz kit's bass drum, tuned and damped: a felt beater's thud.
fn ska_kit_kick(r: &mut Rng) -> Vec<DrumPatch> {
    vec![DrumPatch {
        noise: 0.12,
        nmode: 0,
        ncut: 900.0,
        ndec: 0.03,
        ..kick(r, (100.0, 125.0), (55.0, 64.0), (0.28, 0.4), 0.1, 0.25)
    }]
}

/// The cross-stick (the stick laid across the rim: a woody click with a
/// ring) and the snare, softly, for rolls and ghosts.
fn ska_kit_cross(r: &mut Rng) -> Vec<DrumPatch> {
    let f = r.range(430.0, 540.0);
    let cross = DrumPatch {
        tone: 0.8,
        f0: f,
        f1: f,
        pdec: 1.0,
        tdec: 0.06,
        tone2: 0.5,
        ratio2: 2.3,
        noise: 0.5,
        nmode: 1,
        ncut: r.range(1800.0, 2500.0),
        nres: 0.35,
        ndec: 0.025,
        click: 0.25,
        gain: 0.8,
        ..DRUM
    };
    vec![cross, ska_kit_snare(r)[1]]
}

/// A tight, high snare (a rimshot's crack) and its soft stroke.
fn ska_kit_snare(r: &mut Rng) -> Vec<DrumPatch> {
    let c = DrumPatch {
        tone: 0.5,
        f0: r.range(290.0, 330.0),
        f1: r.range(235.0, 260.0),
        pdec: 0.008,
        tdec: 0.08,
        tone2: 0.25,
        ratio2: 1.6,
        noise: 0.95,
        nmode: 1,
        ncut: r.range(3000.0, 4000.0),
        nres: 0.15,
        ndec: r.range(0.11, 0.15),
        click: 0.15,
        drive: 0.3,
        gain: 0.85,
        ..DRUM
    };
    vec![c, DrumPatch { tone: 0.3, ndec: 0.07, click: 0.0, gain: 0.55, ..c }]
}

/// Hi-hats, bright and tight; the open one a short "tss" with the skank.
fn ska_kit_hats(r: &mut Rng) -> Vec<DrumPatch> {
    hats(r, 8500.0, 0.035, 0.16, 0.35, 0.7)
}

/// Percussion: [the cross-stick, three toms, a tambourine's jingle].
fn ska_kit_toms(r: &mut Rng) -> Vec<DrumPatch> {
    let tom = |f: f32| DrumPatch {
        tone: 1.0,
        f0: f * 1.4,
        f1: f,
        pdec: 0.04,
        tdec: 0.35,
        noise: 0.15,
        nmode: 0,
        ncut: 1800.0,
        ndec: 0.05,
        gain: 0.75,
        ..DRUM
    };
    let base = r.range(95.0, 115.0);
    let jingle = kit_tambourine(r)[0];
    vec![ska_kit_cross(r)[0], tom(base), tom(base * 1.33), tom(base * 1.7), DrumPatch { gain: 0.5, ..jingle }]
}

/// A splash: a small cymbal, bright and quick.
fn ska_kit_splash(r: &mut Rng) -> Vec<DrumPatch> {
    vec![DrumPatch { ndec: r.range(0.7, 1.0), ncut: r.range(5500.0, 7000.0), ..crash(r)[0] }]
}

/// The third wave's trumpet: bright and hard-edged, a section player's.
fn ska_trumpet(r: &mut Rng) -> Patch {
    Patch {
        unison: 2,
        detune: 4.0,
        width: 0.1,
        cutoff: r.range(2200.0, 2800.0),
        res: 0.2,
        env_amt: 1.6,
        fdec: 0.08,
        keytrack: 0.6,
        a: 0.012,
        d: 0.3,
        s: 0.8,
        r: 0.06,
        vib: 0.08,
        vib_rate: 5.5,
        glide: 0.012,
        gain: 0.28,
        ..PATCH
    }
}

/// The third wave's tenor sax: more reed in it than the jazz band's.
fn ska_tenor(r: &mut Rng) -> Patch {
    Patch {
        fm: r.range(3.0, 3.8),
        ratio: 1.0,
        fm_dec: 2.0,
        cutoff: r.range(3500.0, 4500.0),
        res: 0.2,
        keytrack: 0.4,
        a: 0.025,
        d: 0.4,
        s: 0.85,
        r: 0.1,
        vib: 0.15,
        vib_rate: 5.5,
        glide: 0.02,
        gain: 0.26,
        ..PATCH
    }
}

/// The third wave's trombone: brassier, blown harder.
fn ska_bone(r: &mut Rng) -> Patch {
    Patch {
        cutoff: r.range(1500.0, 2000.0),
        res: 0.2,
        env_amt: 1.4,
        fdec: 0.12,
        keytrack: 0.5,
        a: 0.02,
        d: 0.4,
        s: 0.85,
        r: 0.08,
        vib: 0.06,
        vib_rate: 5.0,
        glide: 0.04,
        gain: 0.3,
        ..PATCH
    }
}

/// The third wave's rock kick: punchy, the beater's snap on top.
fn ska_kit_punk_kick(r: &mut Rng) -> Vec<DrumPatch> {
    vec![DrumPatch {
        noise: 0.5,
        nmode: 0,
        ncut: 6000.0,
        ndec: 0.02,
        ..kick(r, (130.0, 160.0), (50.0, 58.0), (0.22, 0.3), 1.0, 0.4)
    }]
}

/// The third wave's rock snare: a fat crack, the wires ringing; and its soft stroke.
fn ska_kit_punk_snare(r: &mut Rng) -> Vec<DrumPatch> {
    let c = DrumPatch {
        tone: 0.6,
        f0: r.range(220.0, 260.0),
        f1: r.range(180.0, 200.0),
        pdec: 0.01,
        tdec: 0.1,
        tone2: 0.3,
        ratio2: 1.5,
        noise: 1.0,
        nmode: 1,
        ncut: r.range(2500.0, 3500.0),
        nres: 0.1,
        ndec: r.range(0.16, 0.22),
        click: 0.2,
        drive: 0.4,
        gain: 0.9,
        ..DRUM
    };
    vec![c, DrumPatch { tone: 0.3, ndec: 0.08, click: 0.0, gain: 0.55, ..c }]
}

/// Ska's instruments: the skank's (guitars, pianos, an organ stab; the
/// first wave's guitars alone), the organs, the pianos, the basses, the
/// lead horns, the second horns, the low ones; the kits; the soloists (and
/// their registers).
const P_SKA_CHOP: &[(&str, Make)] = &[
    ("muted guitar chop", ska_guitar_chop),
    ("clean guitar skank", ska_guitar_clean),
    ("piano", piano),
    ("bright upright piano", ska_piano_bright),
    ("organ stab", stab_organ),
];
const P_SKA_GUITAR: &[(&str, Make)] =
    &[("muted guitar chop", ska_guitar_chop), ("clean guitar skank", ska_guitar_clean)];
/// The third wave's: the skank, the distorted guitars, the sax beside the
/// trumpet, the bass, the lead and second horns; its kick and snare.
const P_SKA_PUNK_CHOP: &[(&str, Make)] =
    &[("crunchy guitar skank", ska_guitar_crunch), ("muted guitar chop", ska_guitar_chop)];
const P_SKA_DIST: &[(&str, Make)] = &[("distorted guitar", ska_guitar_dist), ("fuzz guitar", ska_guitar_fuzz)];
const P_SKA_SAX: &[(&str, Make)] = &[("tenor sax", ska_tenor), ("alto sax", ska_alto)];
const P_SKA_PUNK_BASS: &[(&str, Make)] =
    &[("electric bass, picked", ska_bass_pick), ("electric bass, fingers", ska_bass_finger)];
const P_SKA_TW_LEAD: &[(&str, Make)] = &[("trumpet", ska_trumpet), ("alto sax", ska_alto)];
const P_SKA_TW_HORN: &[(&str, Make)] = &[("trombone", ska_bone), ("tenor sax", ska_tenor)];
const K_SKA_PUNK_KICK: &[(&str, MakeKit)] = &[("rock kick", ska_kit_punk_kick)];
const K_SKA_PUNK_SNARE: &[(&str, MakeKit)] = &[("rock snare", ska_kit_punk_snare), ("tight snare", ska_kit_snare)];
const P_SKA_ORGAN: &[(&str, Make)] =
    &[("Farfisa", ska_farfisa), ("Vox Continental", ska_vox), ("Hammond", ska_hammond), ("organ", pad_organ)];
const P_SKA_PIANO: &[(&str, Make)] =
    &[("piano", piano), ("bright upright piano", ska_piano_bright), ("Wurlitzer", ep_wurli)];
const P_SKA_BASS: &[(&str, Make)] = &[
    ("electric bass, picked", ska_bass_pick),
    ("electric bass, fingers", ska_bass_finger),
    ("flatwound bass", ska_bass_flat),
    ("upright bass", bass_upright),
];
const P_SKA_LEAD: &[(&str, Make)] = &[
    ("trumpet", trumpet),
    ("alto sax", ska_alto),
    ("tenor sax", sax),
    ("melodica", ska_melodica),
    ("flugelhorn", ska_flugel),
    ("muted trumpet", muted_trumpet),
];
const P_SKA_HORN: &[(&str, Make)] = &[
    ("trombone", trombone),
    ("tenor sax", sax),
    ("tenor horn", horn),
    ("alto sax", ska_alto),
    ("flugelhorn", ska_flugel),
];
const P_SKA_LOW: &[(&str, Make)] = &[("baritone sax", ska_bari), ("trombone", trombone), ("tenor sax", sax)];
const K_SKA_KICK: &[(&str, MakeKit)] = &[("jazz kick", ska_kit_kick), ("soft kick", kit_kick_soft)];
const K_SKA_SNARE: &[(&str, MakeKit)] = &[("cross-stick", ska_kit_cross), ("tight snare", ska_kit_snare)];
const K_SKA_HAT: &[(&str, MakeKit)] = &[("hi-hats", ska_kit_hats), ("ride", kit_ride)];
const K_SKA_PERC: &[(&str, MakeKit)] = &[("cross-stick, toms, tambourine", ska_kit_toms)];
const K_SKA_CYM: &[(&str, MakeKit)] = &[("crash", crash), ("splash", ska_kit_splash)];
const SKA_SOLOISTS: &[(&str, Make, i32)] =
    &[("tenor sax", sax, 62), ("trumpet", trumpet, 72), ("trombone", trombone, 58), ("alto sax", ska_alto, 67)];

fn band_ska(r: &mut Rng, s: &Synth, style: u8) -> Band {
    let third = style == SKS_THIRD;
    let mut b = Band::default();
    // the first wave's skank always a guitar's (the piano plays it on its own)
    let chop = match style {
        SKS_FIRST => P_SKA_GUITAR,
        SKS_TWO => P_SKA_CHOP,
        _ => P_SKA_PUNK_CHOP,
    };
    b.tone(r, s, CH_KEYS, chop, 64.0);
    b.tone(r, s, CH_PAD, if third { P_SKA_SAX } else { P_SKA_ORGAN }, 62.0);
    if third {
        b.tone(r, s, CH_DRONE, P_SKA_DIST, 48.0);
    } else {
        b.tone(r, s, CH_DRONE, P_SKA_PIANO, 60.0);
    }
    b.tone(r, s, CH_BASS, if third { P_SKA_PUNK_BASS } else { P_SKA_BASS }, 40.0);
    b.tone(r, s, CH_LEAD, if third { P_SKA_TW_LEAD } else { P_SKA_LEAD }, 72.0);
    b.tone(r, s, CH_BELL, if third { P_SKA_TW_HORN } else { P_SKA_HORN }, 62.0);
    b.tone(r, s, CH_ARP, P_SKA_LOW, 50.0);
    b.kit(r, s, CH_KICK, if third { K_SKA_PUNK_KICK } else { K_SKA_KICK });
    b.kit(r, s, CH_SNARE, if third { K_SKA_PUNK_SNARE } else { K_SKA_SNARE });
    b.kit(r, s, CH_HAT, K_SKA_HAT);
    b.kit(r, s, CH_PERC, K_SKA_PERC);
    b.kit(r, s, CH_CYM, K_SKA_CYM);
    b
}

/// Ska's sounds: a jazz kit (the cross-stick more often in the first
/// wave; the third wave's a rock kit), the muted guitar's chop (the third
/// wave's crunchy, and its distorted guitar), a Hammond (first wave) or a
/// Farfisa or Vox (2-tone), the piano (the third wave's sax instead of the
/// organ, its guitar instead of the piano), an upright or fingered bass (a
/// picked one after the first wave), trumpet or alto, trombone or tenor,
/// baritone; a room; the first wave's record a little dull and crackling,
/// light in the bass, the chop forward; 2-tone and the third wave spread wide.
fn sounds_ska(r: &mut Rng, s: &mut Synth, style: u8) {
    let (first, two_tone, third) = (style == SKS_FIRST, style == SKS_TWO, style == SKS_THIRD);
    let ch = &mut s.ch;
    ch[CH_KICK].drums = if third { ska_kit_punk_kick(r) } else { ska_kit_kick(r) };
    ch[CH_SNARE].drums = if third {
        ska_kit_punk_snare(r)
    } else if r.chance(if two_tone { 0.2 } else { 0.6 }) {
        ska_kit_cross(r)
    } else {
        ska_kit_snare(r)
    };
    ch[CH_HAT].drums = ska_kit_hats(r);
    ch[CH_PERC].drums = ska_kit_toms(r);
    ch[CH_CYM].drums = crash(r);
    ch[CH_BASS].patch = match (first, r.bit_chance()) {
        (false, true) => ska_bass_pick(r),
        (true, true) => bass_upright(r),
        _ => ska_bass_finger(r),
    };
    ch[CH_KEYS].patch = if third { ska_guitar_crunch(r) } else { ska_guitar_chop(r) };
    ch[CH_PAD].patch = match (style, r.bit_chance()) {
        (SKS_FIRST, _) => ska_hammond(r),
        (SKS_TWO, true) => ska_farfisa(r),
        (SKS_TWO, false) => ska_vox(r),
        (_, true) => ska_tenor(r),
        _ => ska_alto(r),
    };
    ch[CH_DRONE].patch = if third { ska_guitar_dist(r) } else { piano(r) };
    ch[CH_LEAD].patch = if third {
        ska_trumpet(r)
    } else if two_tone && r.bit_chance() {
        ska_alto(r)
    } else {
        trumpet(r)
    };
    ch[CH_BELL].patch = if third {
        ska_bone(r)
    } else if r.chance(0.6) {
        trombone(r)
    } else {
        sax(r)
    };
    ch[CH_ARP].patch = ska_bari(r);
    // each role as loud whichever instrument it drew (the band's others are
    // levelled to these); on a first-wave record the bass thinner and the
    // chop forward, up in the middle with the horns
    for (c, note, db) in [
        (CH_BASS, 40.0, if two_tone { -18.0 } else { -21.0 }),
        (
            CH_KEYS,
            64.0,
            match style {
                SKS_FIRST => -21.0,
                SKS_TWO => -26.0,
                _ => -25.0,
            },
        ),
        (CH_PAD, 62.0, -21.0),
        (CH_DRONE, if third { 48.0 } else { 60.0 }, if third { -21.0 } else { -19.0 }),
        (CH_LEAD, 72.0, -19.0),
        (CH_BELL, 62.0, -19.0),
        (CH_ARP, 50.0, -18.0),
    ] {
        let p = ch[c].patch;
        ch[c].patch = Patch { gain: p.gain * 10f32.powf(db / 20.0) / loudness(&p, note), ..p };
    }
    for (c, mono) in [
        (CH_BASS, true),
        (CH_LEAD, true),
        (CH_BELL, true),
        (CH_ARP, true),
        (CH_KEYS, false),
        (CH_PAD, third),
        (CH_DRONE, false),
    ] {
        ch[c].mono = mono;
    }
    // the first wave's jazz kit light (a felt beater's thud, the cross-stick
    // soft), the third wave's rock kit up front
    let (kick, snare) = match style {
        SKS_FIRST => (0.35, 0.32),
        SKS_TWO => (0.6, 0.45),
        _ => (0.45, 1.4),
    };
    mix(&mut ch[CH_KICK], kick, 0.0, 0.05, 0.0, 0.0);
    mix(&mut ch[CH_SNARE], snare, 0.08, 0.18, 0.0, 0.0);
    mix(&mut ch[CH_HAT], if third { 0.98 } else { 0.7 }, 0.25, 0.06, 0.0, 0.0);
    mix(&mut ch[CH_PERC], 0.35, -0.2, 0.12, 0.0, 0.0);
    mix(&mut ch[CH_CYM], 0.32, -0.3, 0.15, 0.0, 0.0);
    mix(&mut ch[CH_BASS], 0.5, 0.0, 0.04, 0.0, 0.0);
    mix(&mut ch[CH_KEYS], 0.4, -0.35, if third { 0.3 } else { 0.15 }, 0.04, 0.0);
    mix(&mut ch[CH_PAD], 0.5, 0.35, 0.18, 0.0, 0.0);
    mix(&mut ch[CH_DRONE], 0.4, 0.15, 0.15, 0.0, 0.0);
    mix(&mut ch[CH_LEAD], if third { 0.88 } else { 0.7 }, 0.12, 0.22, 0.05, 0.0);
    mix(&mut ch[CH_BELL], 0.5, -0.22, 0.22, 0.03, 0.0);
    mix(&mut ch[CH_ARP], 0.36, -0.05, 0.18, 0.0, 0.0);
    // 2-tone spread wide: the guitar and the organ either side, the horns
    // apart, the piano and the hats out; the third wave's horn section
    // spread across, its distorted guitar wide of itself (left and right);
    // each as loud as nearer the middle
    let pans: &[(usize, f32)] = match style {
        SKS_TWO => &[
            (CH_KEYS, -0.8),
            (CH_PAD, 0.8),
            (CH_LEAD, 0.4),
            (CH_BELL, -0.7),
            (CH_ARP, 0.5),
            (CH_DRONE, 0.55),
            (CH_HAT, 0.55),
            (CH_PERC, -0.45),
            (CH_CYM, -0.45),
        ],
        SKS_THIRD => &[
            (CH_KEYS, -0.6),
            (CH_PAD, -0.35),
            (CH_LEAD, 0.3),
            (CH_BELL, 0.55),
            (CH_ARP, -0.5),
            (CH_DRONE, 0.0),
            (CH_HAT, 0.45),
            (CH_PERC, -0.4),
            (CH_CYM, -0.35),
        ],
        _ => &[],
    };
    let power = |p: f32| ((1.0 - p.abs()).powi(2) + 1.0) / 2.0;
    for &(c, pan) in pans {
        ch[c].gain *= (power(ch[c].pan) / power(pan)).sqrt();
        ch[c].pan = pan;
    }
    // the distorted guitar's speaker
    ch[CH_DRONE].lp = if third { 6500.0 } else { 20000.0 };
    if first {
        mix(&mut ch[CH_FX], 0.8, 0.0, 0.0, 0.0, 0.0);
        s.m.set_lp(r.range(10000.0, 14000.0));
        s.fx.crackle = r.range(0.02, 0.05);
        s.fx.crackle_rate = r.range(3.0, 8.0);
        s.fx.hiss = r.range(0.0008, 0.0015);
    } else {
        s.m.set_lp(20000.0);
        s.fx.crackle = 0.0;
        s.fx.hiss = 0.0;
    }
    s.rev.fb = r.range(0.76, 0.82);
    s.rev.damp = 0.45;
    s.rev_wet = match style {
        SKS_FIRST => 0.24,
        SKS_TWO => 0.18,
        _ => 0.22,
    };
    s.dly.fb = 0.15;
    s.dly.lpk = 0.3;
    s.dly_rev = 0.2;
    s.r2mode = R2_OFF;
    s.fx.wind_to = 0.0;
    s.fx.rain_to = 0.0;
    s.m.wow_to = 0.0;
    s.m.gain = 1.3;
    s.m.target = 0.195;
}

/// A bassline changed in `k` places: a note on or off, another note, an
/// accent or a slide.
fn mutate_bass(r: &mut Rng, p: &[Step; 16], mode: usize, k: usize) -> [Step; 16] {
    let mut p = *p;
    for _ in 0..k {
        let i = r.below(16);
        let x = &mut p[i];
        match r.below(4) {
            0 => x.on = !x.on || i == 0,
            1 => x.note = deg_semi(mode, r.pick(&[0, 0, 0, 7, 7, 2, 4, 6, -1])),
            2 => x.accent = !x.accent,
            _ => x.slide = !x.slide,
        }
    }
    p
}

/// A bassline for techno: acid (16ths, accents, slides) or rolling (offbeats).
fn new_bass(r: &mut Rng, mode: usize, style: u8) -> [Step; 16] {
    let mut p = [Step::default(); 16];
    if style == 0 {
        let dens = r.range(0.5, 0.8);
        for (i, x) in p.iter_mut().enumerate() {
            let on = i == 0 || r.chance(dens * if i % 4 == 0 { 1.25 } else { 1.0 });
            let deg = r.pick(&[0, 0, 0, 0, 0, 0, 7, 7, 7, 2, 4, 6, -1]);
            *x = Step { on, note: deg_semi(mode, deg), accent: r.chance(0.28), slide: r.chance(0.2) };
        }
    } else {
        let cell: [bool; 4] =
            r.pick(&[[false, false, true, false], [false, false, true, true], [false, true, true, true]]);
        for (i, x) in p.iter_mut().enumerate() {
            let octave = r.chance(0.12);
            *x = Step { on: cell[i % 4], note: if octave { 12 } else { 0 }, accent: i % 4 == 2, slide: false };
        }
    }
    p
}

// ---------------------------------------------------------------- sounds

fn mix(c: &mut Channel, gain: f32, pan: f32, rev: f32, dly: f32, duck: f32) {
    c.gain = gain;
    c.pan = pan;
    c.rev = rev;
    c.dly = dly;
    c.rev2 = 0.0;
    c.duck = duck;
    c.lp_base = 20000.0;
}

fn kick(r: &mut Rng, f0: (f32, f32), f1: (f32, f32), tdec: (f32, f32), click: f32, drive: f32) -> DrumPatch {
    DrumPatch {
        tone: 1.0,
        f0: r.range(f0.0, f0.1),
        f1: r.range(f1.0, f1.1),
        pdec: r.range(0.022, 0.04),
        tdec: r.range(tdec.0, tdec.1),
        click,
        drive,
        ..DRUM
    }
}

fn hats(r: &mut Rng, cut: f32, closed: f32, open: f32, metal: f32, gain: f32) -> Vec<DrumPatch> {
    let c = DrumPatch {
        noise: 1.0,
        nmode: 2,
        ncut: cut * r.range(0.85, 1.15),
        nres: 0.15,
        ndec: closed * r.range(0.8, 1.25),
        metal: metal * r.range(0.6, 1.2),
        mf: r.range(1.0, 1.6),
        gain,
        ..DRUM
    };
    vec![c, DrumPatch { ndec: open * r.range(0.8, 1.25), gain: gain * 0.85, ..c }]
}

fn percs(r: &mut Rng) -> Vec<DrumPatch> {
    let rim = DrumPatch {
        tone: 0.7,
        f0: r.range(1500.0, 2200.0),
        pdec: 1.0,
        tdec: 0.04,
        noise: 0.5,
        nmode: 1,
        ncut: 2500.0,
        nres: 0.3,
        ndec: 0.025,
        gain: 0.6,
        ..DRUM
    };
    let rim = DrumPatch { f1: rim.f0, ..rim };
    let tom = |f: f32| DrumPatch {
        tone: 1.0,
        f0: f * 1.5,
        f1: f,
        pdec: 0.05,
        tdec: 0.45,
        noise: 0.12,
        nmode: 0,
        ncut: 1500.0,
        ndec: 0.06,
        gain: 0.8,
        ..DRUM
    };
    let base = r.range(90.0, 120.0);
    let shaker = DrumPatch {
        noise: 1.0,
        nmode: 1,
        ncut: r.range(5500.0, 7500.0),
        nres: 0.2,
        ndec: 0.08,
        natt: 0.01,
        gain: 0.6,
        ..DRUM
    };
    vec![rim, tom(base), tom(base * 1.33), tom(base * 1.78), shaker]
}

fn crash(r: &mut Rng) -> Vec<DrumPatch> {
    vec![DrumPatch {
        noise: 1.0,
        nmode: 2,
        ncut: r.range(3500.0, 5500.0),
        nres: 0.1,
        ndec: r.range(1.6, 2.6),
        natt: 0.001,
        metal: 0.5,
        mf: r.range(2.0, 3.0),
        gain: 0.7,
        ..DRUM
    }]
}

// ---------------------------------------------------------------- the instruments

/// An instrument of a palette: its name and how to make one (each a little
/// different).
type Make = fn(&mut Rng) -> Patch;
/// A drum kit for one channel, its kinds in the channel's order.
type MakeKit = fn(&mut Rng) -> Vec<DrumPatch>;

/// How loud a patch plays: the RMS of one note at `note`, velocity 0.8, over
/// 0.12 s after a short attack (the patch's own attack would only delay it);
/// a sampled one's loudest 0.12 s of its first 0.45 s.
fn loudness(p: &Patch, note: f32) -> f32 {
    loudness_in(p, note, samples::bank())
}

/// `loudness`, a sampled patch's zones from `bank`.
fn loudness_in(p: &Patch, note: f32, bank: Option<&Bank>) -> f32 {
    let q = Patch { a: p.a.min(0.03), ..*p };
    let mut v = VOICE;
    v.start(&q, note, 0.8, (0.5 * SR) as i64, false, 0x1234_5679, bank, 0);
    if v.sm.n > 0 {
        // a recording speaks in its own time (a bowed string's swell): its
        // loudest 0.12 s of the gate's first 0.45 s
        const N: usize = (0.45 * SR) as usize / BLOCK;
        let mut e = [0.0f32; N];
        let mut buf = [[0.0f32; 2]; BLOCK];
        for x in e.iter_mut() {
            buf.fill([0.0; 2]);
            v.render(1.0, &mut buf, bank);
            *x = buf.iter().map(|x| x[0] * x[0] + x[1] * x[1]).sum::<f32>();
        }
        let (skip, w) = ((0.03 * SR) as usize / BLOCK, (0.12 * SR) as usize / BLOCK);
        let most = e[skip..].windows(w).map(|x| x.iter().sum::<f32>()).fold(0.0, f32::max);
        return (most / (2 * w * BLOCK) as f32).sqrt().max(1e-5);
    }
    let mut buf = [[0.0f32; 2]; BLOCK];
    let (mut sum, mut n) = (0.0f32, 0usize);
    for k in 0..(0.15 * SR) as usize / BLOCK {
        buf.fill([0.0; 2]);
        v.render(1.0, &mut buf, bank);
        if k * BLOCK >= (0.03 * SR) as usize {
            sum += buf.iter().map(|x| x[0] * x[0] + x[1] * x[1]).sum::<f32>();
            n += 2 * BLOCK;
        }
    }
    (sum / n.max(1) as f32).sqrt().max(1e-5)
}

/// A piece's instruments: two for each tonal role (A and B: verses one,
/// choruses and bridges the other, give or take), and two kits for each drum
/// channel; swapped at section boundaries, each note keeping the patch it
/// began with. (And the note each tonal pair was levelled at.)
#[derive(Default)]
struct Band {
    tones: Vec<(usize, [Patch; 2], f32)>,
    kits: Vec<(usize, [Vec<DrumPatch>; 2])>,
}

/// `p` brought to the loudness of `orig` at `note`.
fn levelled(orig: &Patch, note: f32, p: Patch) -> Patch {
    Patch { gain: p.gain * (loudness(orig, note) / loudness(&p, note)).clamp(0.2, 5.0), ..p }
}

impl Band {
    /// Channel `ch`'s pair: the role's tuned original (sometimes) or a pick
    /// from `pal`, and another pick, both brought to the original's
    /// loudness at `note`.
    fn tone(&mut self, r: &mut Rng, s: &Synth, ch: usize, pal: &[(&str, Make)], note: f32) {
        let orig = s.ch[ch].patch;
        let level = |p: Patch| levelled(&orig, note, p);
        let ia = if r.chance(0.35) { None } else { Some(r.below(pal.len())) };
        let mut ib = r.below(pal.len());
        if Some(ib) == ia {
            ib = (ib + 1 + r.below(pal.len() - 1)) % pal.len();
        }
        let a = ia.map_or(orig, |i| level((pal[i].1)(r)));
        let b = level((pal[ib].1)(r));
        self.tones.push((ch, [a, b], note));
    }

    /// Channel `ch`'s kits: the original and one from `kits`.
    fn kit(&mut self, r: &mut Rng, s: &Synth, ch: usize, kits: &[(&str, MakeKit)]) {
        let orig = s.ch[ch].drums.clone();
        let alt = (r.pick(kits).1)(r);
        let [a, b] = if r.bit_chance() { [orig, alt] } else { [alt, orig] };
        self.kits.push((ch, [a, b]));
    }
}

// -- pads

fn pad_strings(r: &mut Rng) -> Patch {
    Patch {
        unison: 3,
        detune: r.range(18.0, 28.0),
        width: 1.0,
        cutoff: r.range(2500.0, 4000.0),
        res: 0.05,
        keytrack: 0.5,
        a: r.range(0.6, 1.2),
        d: 2.0,
        s: 0.9,
        r: 1.5,
        vib: 0.08,
        vib_rate: 5.0,
        gain: 0.14,
        ..PATCH
    }
}

fn pad_brass(r: &mut Rng) -> Patch {
    Patch {
        unison: 2,
        detune: 6.0,
        width: 0.5,
        cutoff: r.range(600.0, 900.0),
        res: 0.15,
        env_amt: 2.2,
        fdec: 0.6,
        keytrack: 0.4,
        a: 0.08,
        d: 1.0,
        s: 0.8,
        r: 0.6,
        gain: 0.15,
        ..PATCH
    }
}

fn pad_choir(r: &mut Rng) -> Patch {
    Patch {
        unison: 3,
        detune: r.range(8.0, 14.0),
        width: 0.8,
        vowel: r.pick(&[1, 4, 5]),
        cutoff: 3200.0,
        keytrack: 0.2,
        a: r.range(0.5, 1.2),
        d: 2.0,
        s: 0.9,
        r: 2.0,
        vib: 0.15,
        vib_rate: 5.0,
        gain: 0.3,
        ..PATCH
    }
}

fn pad_glass(r: &mut Rng) -> Patch {
    Patch {
        filter: false,
        fm: r.range(0.8, 1.4),
        ratio: r.pick(&[2.0, 3.0]),
        fm_dec: 3.0,
        a: r.range(0.5, 1.2),
        d: 3.0,
        s: 0.8,
        r: 2.5,
        gain: 0.2,
        ..PATCH
    }
}

fn pad_shimmer(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Tri,
        unison: 3,
        detune: 6.0,
        width: 1.0,
        wave2: Wave::Sine,
        mix2: 0.4,
        semi2: r.pick(&[19.0, 24.0]),
        cutoff: 6000.0,
        keytrack: 0.3,
        a: r.range(0.8, 1.6),
        d: 3.0,
        s: 0.8,
        r: 3.0,
        trem: 0.25,
        vib_rate: r.range(5.0, 7.0),
        gain: 0.14,
        ..PATCH
    }
}

fn pad_warm(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Tri,
        unison: 3,
        detune: r.range(5.0, 9.0),
        width: 0.8,
        wave2: Wave::Sine,
        mix2: 0.6,
        cutoff: 1200.0,
        keytrack: 0.4,
        a: r.range(0.8, 1.8),
        d: 3.0,
        s: 0.9,
        r: 2.5,
        gain: 0.16,
        ..PATCH
    }
}

fn pad_organ(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Square,
        pw: 0.5,
        wave2: Wave::Sine,
        mix2: 0.5,
        semi2: 12.0,
        sub: 0.3,
        cutoff: r.range(2000.0, 3000.0),
        keytrack: 0.3,
        a: 0.02,
        d: 0.1,
        s: 1.0,
        r: 0.15,
        trem: r.range(0.0, 0.15),
        vib_rate: 6.0,
        gain: 0.12,
        ..PATCH
    }
}

fn pad_dark(r: &mut Rng) -> Patch {
    Patch {
        unison: 3,
        detune: r.range(10.0, 18.0),
        width: 0.9,
        cutoff: r.range(400.0, 700.0),
        res: 0.25,
        keytrack: 0.3,
        a: 1.5,
        d: 2.0,
        s: 0.8,
        r: 2.5,
        gain: 0.16,
        ..PATCH
    }
}

// -- keys and mallets

fn ep_rhodes(r: &mut Rng) -> Patch {
    Patch {
        filter: false,
        fm: r.range(1.2, 2.0),
        ratio: 1.0,
        fm_dec: r.range(0.6, 1.2),
        fm2: r.range(0.4, 0.8),
        ratio2: r.pick(&[14.0, 13.0, 15.0]),
        fm2_dec: 0.05,
        a: 0.002,
        d: r.range(1.2, 2.0),
        s: r.range(0.3, 0.42),
        r: 0.35,
        trem: r.range(0.1, 0.25),
        vib_rate: r.range(3.0, 5.0),
        gain: 0.3,
        ..PATCH
    }
}

fn ep_wurli(r: &mut Rng) -> Patch {
    Patch {
        filter: false,
        fm: r.range(2.0, 3.0),
        ratio: 1.0,
        fm_dec: 0.4,
        a: 0.002,
        d: 1.5,
        s: 0.3,
        r: 0.25,
        trem: r.range(0.15, 0.3),
        vib_rate: 5.5,
        drive: 0.6,
        gain: 0.28,
        ..PATCH
    }
}

fn piano_felt(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Tri,
        wave2: Wave::Sine,
        mix2: 0.6,
        semi2: 12.0,
        cutoff: r.range(1000.0, 1600.0),
        env_amt: 1.0,
        fdec: 0.5,
        keytrack: 0.5,
        a: 0.003,
        d: 1.5,
        s: 0.35,
        r: 0.3,
        gain: 0.32,
        ..PATCH
    }
}

fn piano(r: &mut Rng) -> Patch {
    Patch {
        filter: false,
        fm: r.range(1.0, 1.5),
        ratio: 1.0,
        fm_dec: 0.8,
        fm2: 0.8,
        ratio2: 3.0,
        fm2_dec: 0.03,
        a: 0.002,
        d: 3.0,
        s: 0.0,
        r: 0.3,
        gain: 0.3,
        ..PATCH
    }
}

fn vibes(r: &mut Rng) -> Patch {
    Patch {
        filter: false,
        fm: r.range(0.8, 1.2),
        ratio: 4.0,
        fm_dec: 0.4,
        a: 0.002,
        d: 1.8,
        s: 0.0,
        r: 0.5,
        trem: 0.3,
        vib_rate: 5.0,
        gain: 0.3,
        ..PATCH
    }
}

fn marimba(r: &mut Rng) -> Patch {
    Patch {
        filter: false,
        fm: r.range(2.0, 3.0),
        ratio: 4.0,
        fm_dec: 0.05,
        a: 0.001,
        d: r.range(0.4, 0.7),
        s: 0.0,
        r: 0.2,
        gain: 0.35,
        ..PATCH
    }
}

fn celesta(r: &mut Rng) -> Patch {
    Patch {
        filter: false,
        fm: r.range(1.2, 1.8),
        ratio: r.pick(&[4.0, 7.0]),
        fm_dec: 0.3,
        a: 0.001,
        d: 1.5,
        s: 0.0,
        r: 0.5,
        gain: 0.28,
        ..PATCH
    }
}

fn kalimba(r: &mut Rng) -> Patch {
    Patch { filter: false, ks: r.range(0.6, 1.0), ks_tone: 0.75, a: 0.001, d: 3.0, s: 1.0, r: 0.2, gain: 0.5, ..PATCH }
}

fn harp(r: &mut Rng) -> Patch {
    Patch { filter: false, ks: r.range(2.0, 3.0), ks_tone: 0.55, a: 0.001, d: 4.0, s: 1.0, r: 0.6, gain: 0.5, ..PATCH }
}

fn music_box(r: &mut Rng) -> Patch {
    Patch {
        filter: false,
        fm: r.range(0.8, 1.2),
        ratio: 5.0,
        fm_dec: 0.15,
        a: 0.001,
        d: 1.2,
        s: 0.0,
        r: 0.4,
        gain: 0.28,
        ..PATCH
    }
}

fn glockenspiel(r: &mut Rng) -> Patch {
    Patch {
        filter: false,
        fm: r.range(1.5, 2.1),
        ratio: 3.5,
        fm_dec: 0.2,
        a: 0.001,
        d: 1.6,
        s: 0.0,
        r: 0.5,
        gain: 0.26,
        ..PATCH
    }
}

fn bell_fm(r: &mut Rng) -> Patch {
    Patch {
        filter: false,
        fm: r.range(1.5, 3.0),
        ratio: r.pick(&[3.5, 1.41, 2.0, 4.0, 2.76]),
        fm_dec: r.range(1.2, 2.5),
        a: 0.002,
        d: r.range(3.0, 5.0),
        s: 0.0,
        r: 2.0,
        gain: 0.3,
        ..PATCH
    }
}

fn dulcimer(r: &mut Rng) -> Patch {
    Patch { filter: false, ks: r.range(1.2, 1.8), ks_tone: 0.9, a: 0.001, d: 3.0, s: 1.0, r: 0.4, gain: 0.45, ..PATCH }
}

fn glass_arp(_: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Tri,
        wave2: Wave::Sine,
        mix2: 0.6,
        semi2: 12.0,
        cutoff: 2500.0,
        res: 0.1,
        keytrack: 0.2,
        a: 0.006,
        d: 1.2,
        s: 0.0,
        r: 0.8,
        gain: 0.25,
        ..PATCH
    }
}

// -- plucked strings

fn nylon(r: &mut Rng) -> Patch {
    Patch {
        ks: r.range(1.8, 2.5),
        ks_tone: 0.45,
        cutoff: 3000.0,
        keytrack: 0.3,
        a: 0.001,
        d: 3.0,
        s: 1.0,
        r: 0.3,
        gain: 0.5,
        ..PATCH
    }
}

fn lute(r: &mut Rng) -> Patch {
    Patch { filter: false, ks: r.range(1.4, 2.0), ks_tone: 0.5, a: 0.001, d: 3.0, s: 1.0, r: 0.3, gain: 0.5, ..PATCH }
}

fn cittern(r: &mut Rng) -> Patch {
    Patch { filter: false, ks: r.range(0.8, 1.2), ks_tone: 0.8, a: 0.001, d: 2.0, s: 1.0, r: 0.25, gain: 0.45, ..PATCH }
}

fn banjo(r: &mut Rng) -> Patch {
    Patch {
        filter: false,
        ks: r.range(0.5, 0.8),
        ks_tone: 0.92,
        drive: 0.4,
        a: 0.001,
        d: 1.5,
        s: 1.0,
        r: 0.15,
        gain: 0.45,
        ..PATCH
    }
}

fn pizzicato(r: &mut Rng) -> Patch {
    Patch { filter: false, ks: r.range(0.3, 0.45), ks_tone: 0.5, a: 0.001, d: 1.0, s: 1.0, r: 0.1, gain: 0.5, ..PATCH }
}

// -- leads

fn lead_saw(r: &mut Rng) -> Patch {
    Patch {
        unison: 3,
        detune: r.range(6.0, 10.0),
        width: 0.35,
        cutoff: r.range(2500.0, 4500.0),
        res: 0.15,
        env_amt: 1.0,
        fdec: 0.3,
        keytrack: 0.3,
        a: 0.015,
        d: 0.6,
        s: 0.75,
        r: 0.3,
        vib: r.range(0.18, 0.3),
        vib_rate: r.range(5.0, 6.0),
        glide: 0.06,
        gain: 0.3,
        ..PATCH
    }
}

fn lead_square(r: &mut Rng) -> Patch {
    Patch { wave: Wave::Square, pw: r.range(0.4, 0.5), unison: 2, detune: 6.0, width: 0.3, ..lead_saw(r) }
}

fn lead_sync(r: &mut Rng) -> Patch {
    // a tearing, sweeping lead (FM standing in for oscillator sync)
    Patch {
        fm: r.range(2.5, 3.5),
        ratio: r.pick(&[2.5, 3.5, 1.5]),
        fm_dec: r.range(0.25, 0.45),
        cutoff: 4000.0,
        keytrack: 0.3,
        a: 0.01,
        d: 0.6,
        s: 0.8,
        r: 0.25,
        vib: 0.15,
        vib_rate: 5.5,
        glide: 0.05,
        gain: 0.28,
        ..PATCH
    }
}

fn sax(r: &mut Rng) -> Patch {
    Patch {
        fm: r.range(1.8, 2.6),
        ratio: 1.0,
        fm_dec: 2.0,
        cutoff: r.range(2400.0, 3200.0),
        res: 0.2,
        keytrack: 0.4,
        a: 0.04,
        d: 0.4,
        s: 0.85,
        r: 0.12,
        vib: 0.25,
        vib_rate: 5.5,
        glide: 0.03,
        gain: 0.28,
        ..PATCH
    }
}

fn flute(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Sine,
        wave2: Wave::Tri,
        mix2: 0.4,
        noise: r.range(0.03, 0.05),
        cutoff: 2500.0,
        keytrack: 0.3,
        a: 0.05,
        d: 0.5,
        s: 0.8,
        r: 0.2,
        vib: 0.15,
        vib_rate: 5.0,
        gain: 0.3,
        ..PATCH
    }
}

fn ocarina(_: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Sine,
        wave2: Wave::Tri,
        mix2: 0.3,
        noise: 0.015,
        cutoff: 2500.0,
        keytrack: 0.3,
        a: 0.04,
        d: 0.5,
        s: 0.8,
        r: 0.4,
        vib: 0.12,
        vib_rate: 4.5,
        gain: 0.35,
        ..PATCH
    }
}

fn recorder(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Sine,
        wave2: Wave::Tri,
        mix2: 0.5,
        semi2: 12.0,
        noise: 0.03,
        cutoff: 4000.0,
        keytrack: 0.3,
        a: 0.025,
        d: 0.3,
        s: 0.9,
        r: 0.08,
        vib: r.range(0.05, 0.1),
        vib_rate: 5.0,
        glide: 0.012,
        gain: 0.3,
        ..PATCH
    }
}

fn whistle(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Sine,
        noise: r.range(0.04, 0.06),
        cutoff: 6000.0,
        keytrack: 0.2,
        a: 0.015,
        d: 0.3,
        s: 0.9,
        r: 0.06,
        vib: 0.12,
        vib_rate: 6.0,
        glide: 0.01,
        gain: 0.32,
        ..PATCH
    }
}

fn muted_trumpet(r: &mut Rng) -> Patch {
    Patch {
        vowel: 2,
        cutoff: r.range(1800.0, 2400.0),
        env_amt: 1.0,
        fdec: 0.1,
        keytrack: 0.5,
        a: 0.02,
        d: 0.4,
        s: 0.8,
        r: 0.08,
        vib: 0.1,
        vib_rate: 5.5,
        glide: 0.015,
        gain: 0.4,
        ..PATCH
    }
}

fn trumpet(r: &mut Rng) -> Patch {
    Patch {
        unison: 2,
        detune: 4.0,
        width: 0.1,
        cutoff: r.range(1200.0, 1600.0),
        res: 0.15,
        env_amt: 1.6,
        fdec: 0.09,
        keytrack: 0.6,
        a: 0.015,
        d: 0.4,
        s: 0.8,
        r: 0.07,
        vib: 0.1,
        vib_rate: 5.5,
        glide: 0.015,
        gain: 0.3,
        ..PATCH
    }
}

fn clarinet(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Square,
        pw: 0.5,
        cutoff: r.range(1800.0, 2600.0),
        res: 0.1,
        env_amt: 0.5,
        fdec: 0.1,
        keytrack: 0.6,
        noise: 0.012,
        a: 0.02,
        d: 0.3,
        s: 0.85,
        r: 0.06,
        vib: 0.08,
        vib_rate: 5.2,
        glide: 0.015,
        gain: 0.3,
        ..PATCH
    }
}

fn trombone(r: &mut Rng) -> Patch {
    Patch {
        cutoff: r.range(700.0, 1000.0),
        res: 0.15,
        env_amt: 1.2,
        fdec: 0.15,
        keytrack: 0.5,
        a: 0.03,
        d: 0.4,
        s: 0.85,
        r: 0.1,
        vib: 0.08,
        vib_rate: 5.0,
        glide: 0.05,
        gain: 0.32,
        ..PATCH
    }
}

fn fiddle(r: &mut Rng) -> Patch {
    Patch {
        unison: 2,
        detune: 4.0,
        width: 0.2,
        vowel: 2,
        noise: 0.02,
        cutoff: 4500.0,
        keytrack: 0.3,
        a: 0.05,
        d: 0.3,
        s: 0.9,
        r: 0.15,
        vib: r.range(0.2, 0.3),
        vib_rate: 5.8,
        glide: 0.02,
        gain: 0.5,
        ..PATCH
    }
}

fn accordion(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Square,
        pw: r.range(0.35, 0.45),
        unison: 3,
        detune: r.range(12.0, 22.0),
        width: 0.5,
        wave2: Wave::Saw,
        mix2: 0.3,
        semi2: if r.chance(0.4) { 12.0 } else { 0.0 },
        cutoff: r.range(2600.0, 3600.0),
        res: 0.25,
        keytrack: 0.5,
        a: 0.025,
        d: 0.4,
        s: 0.9,
        r: 0.08,
        gain: 0.32,
        ..PATCH
    }
}

fn bandoneon(r: &mut Rng) -> Patch {
    Patch { unison: 2, detune: r.range(2.0, 5.0), semi2: 0.0, cutoff: r.range(2000.0, 2600.0), ..accordion(r) }
}

fn harmonica(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Square,
        pw: 0.3,
        vowel: 3,
        cutoff: 3500.0,
        keytrack: 0.3,
        a: 0.03,
        d: 0.3,
        s: 0.85,
        r: 0.08,
        vib: r.range(0.1, 0.2),
        vib_rate: 5.5,
        trem: 0.1,
        gain: 0.45,
        ..PATCH
    }
}

fn horn(r: &mut Rng) -> Patch {
    Patch { cutoff: r.range(900.0, 1300.0), unison: 2, detune: 5.0, width: 0.2, ..trombone(r) }
}

fn hoover(r: &mut Rng) -> Patch {
    Patch {
        unison: 3,
        detune: r.range(25.0, 40.0),
        width: 1.0,
        wave2: Wave::Saw,
        mix2: 0.6,
        semi2: -12.0,
        cutoff: r.range(1500.0, 2200.0),
        res: 0.2,
        keytrack: 0.3,
        bend: -7.0,
        glide: 0.12,
        a: 0.02,
        d: 0.5,
        s: 0.8,
        r: 0.3,
        gain: 0.22,
        ..PATCH
    }
}

fn vox_chop(r: &mut Rng) -> Patch {
    Patch {
        vowel: r.pick(&[1, 2, 3, 4, 5]),
        cutoff: 4000.0,
        keytrack: 0.2,
        a: 0.005,
        d: 0.15,
        s: 0.0,
        r: 0.08,
        vib: 0.0,
        gain: 0.6,
        ..PATCH
    }
}

fn pluck_hook(r: &mut Rng) -> Patch {
    Patch {
        wave: if r.chance(0.5) { Wave::Square } else { Wave::Saw },
        pw: 0.3,
        cutoff: r.range(900.0, 1600.0),
        res: 0.35,
        env_amt: 2.5,
        fdec: 0.12,
        keytrack: 0.4,
        a: 0.002,
        d: 0.2,
        s: 0.0,
        r: 0.08,
        gain: 0.3,
        ..PATCH
    }
}

fn arp_pluck(r: &mut Rng) -> Patch {
    Patch {
        wave: if r.chance(0.5) { Wave::Square } else { Wave::Saw },
        pw: r.range(0.3, 0.5),
        cutoff: r.range(1000.0, 2200.0),
        res: r.range(0.2, 0.35),
        env_amt: 2.0,
        fdec: r.range(0.15, 0.3),
        keytrack: 0.4,
        a: 0.002,
        d: 0.3,
        s: 0.0,
        r: 0.12,
        gain: 0.3,
        ..PATCH
    }
}

fn glass_fm(r: &mut Rng) -> Patch {
    Patch {
        filter: false,
        fm: r.range(0.8, 1.6),
        ratio: r.pick(&[2.0, 3.0, 1.0]),
        fm_dec: 0.5,
        a: 0.003,
        d: 1.2,
        s: 0.2,
        r: 0.3,
        gain: 0.3,
        ..PATCH
    }
}

fn stab_saw(r: &mut Rng) -> Patch {
    Patch {
        unison: 2,
        detune: 12.0,
        width: 0.7,
        cutoff: r.range(600.0, 1200.0),
        res: 0.3,
        env_amt: 2.0,
        fdec: r.range(0.12, 0.25),
        keytrack: 0.3,
        a: 0.002,
        d: 0.35,
        s: 0.0,
        r: 0.2,
        gain: 0.3,
        ..PATCH
    }
}

fn stab_dub(r: &mut Rng) -> Patch {
    Patch { cutoff: r.range(400.0, 700.0), res: 0.4, env_amt: 1.5, fdec: 0.1, d: 0.25, ..stab_saw(r) }
}

fn stab_organ(r: &mut Rng) -> Patch {
    Patch { d: 0.2, s: 0.0, r: 0.1, ..pad_organ(r) }
}

// -- basses

fn bass_reese(r: &mut Rng) -> Patch {
    Patch {
        unison: 2,
        detune: r.range(14.0, 24.0),
        cutoff: r.range(450.0, 700.0),
        res: 0.2,
        keytrack: 0.3,
        a: 0.005,
        d: 0.5,
        s: 0.9,
        r: 0.08,
        drive: 0.5,
        gain: 0.6,
        ..PATCH
    }
}

fn bass_sub(_: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Sine,
        cutoff: 300.0,
        keytrack: 0.0,
        a: 0.004,
        d: 0.4,
        s: 0.8,
        r: 0.06,
        drive: 0.3,
        gain: 0.8,
        ..PATCH
    }
}

fn bass_fm(r: &mut Rng) -> Patch {
    Patch {
        fm: r.range(2.0, 3.0),
        ratio: r.pick(&[1.0, 2.0]),
        fm_dec: 0.15,
        cutoff: 1200.0,
        keytrack: 0.3,
        a: 0.002,
        d: 0.4,
        s: 0.5,
        r: 0.06,
        gain: 0.6,
        ..PATCH
    }
}

fn bass_slap(r: &mut Rng) -> Patch {
    Patch {
        fm: r.range(3.5, 4.5),
        ratio: 1.0,
        fm_dec: 0.05,
        cutoff: 2000.0,
        keytrack: 0.3,
        a: 0.002,
        d: 0.3,
        s: 0.4,
        r: 0.05,
        drive: 0.5,
        gain: 0.55,
        ..PATCH
    }
}

fn bass_seq(r: &mut Rng) -> Patch {
    Patch {
        cutoff: r.range(350.0, 600.0),
        res: 0.25,
        env_amt: 2.0,
        fdec: 0.1,
        keytrack: 0.3,
        a: 0.002,
        d: 0.2,
        s: 0.0,
        r: 0.04,
        gain: 0.6,
        ..PATCH
    }
}

fn bass_moog(r: &mut Rng) -> Patch {
    Patch {
        wave2: Wave::Square,
        mix2: 0.6,
        semi2: -12.0,
        poles4: true,
        cutoff: r.range(250.0, 400.0),
        res: 0.5,
        env_amt: 1.5,
        fdec: 0.2,
        keytrack: 0.3,
        a: 0.003,
        d: 0.3,
        s: 0.6,
        r: 0.06,
        drive: 0.4,
        gain: 0.6,
        ..PATCH
    }
}

fn bass_upright(r: &mut Rng) -> Patch {
    Patch {
        ks: r.range(1.0, 1.6),
        ks_tone: 0.35,
        cutoff: 900.0,
        keytrack: 0.2,
        a: 0.002,
        d: 2.0,
        s: 1.0,
        r: 0.12,
        gain: 0.8,
        ..PATCH
    }
}

fn bass_synth(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Tri,
        wave2: Wave::Saw,
        mix2: 0.25,
        sub: 0.4,
        cutoff: r.range(450.0, 700.0),
        res: 0.1,
        env_amt: 1.2,
        fdec: 0.15,
        keytrack: 0.3,
        a: 0.004,
        d: 0.8,
        s: 0.35,
        r: 0.08,
        drive: 0.6,
        gain: 0.8,
        ..PATCH
    }
}

fn bass_acid_sq(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Square,
        poles4: true,
        cutoff: r.range(180.0, 320.0),
        res: r.range(0.78, 0.9),
        env_amt: r.range(2.5, 4.0),
        fdec: r.range(0.2, 0.45),
        keytrack: 0.3,
        a: 0.002,
        d: 0.5,
        s: 0.6,
        r: 0.02,
        glide: r.range(0.04, 0.07),
        drive: r.range(1.0, 2.5),
        gain: 0.55,
        ..PATCH
    }
}

fn bass_acid_saw(r: &mut Rng) -> Patch {
    Patch { wave: Wave::Saw, ..bass_acid_sq(r) }
}

fn tuba(r: &mut Rng) -> Patch {
    Patch {
        wave2: Wave::Sine,
        mix2: 0.7,
        cutoff: r.range(350.0, 550.0),
        res: 0.15,
        env_amt: 1.4,
        fdec: 0.1,
        keytrack: 0.5,
        a: 0.015,
        d: 0.25,
        s: 0.6,
        r: 0.07,
        gain: 0.8,
        ..PATCH
    }
}

fn bass_accordion(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Square,
        pw: 0.4,
        cutoff: r.range(700.0, 1000.0),
        keytrack: 0.4,
        a: 0.01,
        d: 0.3,
        s: 0.9,
        r: 0.06,
        gain: 0.6,
        ..PATCH
    }
}

fn drone_hum(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Sine,
        wave2: Wave::Saw,
        mix2: r.range(0.08, 0.18),
        semi2: 0.08,
        cutoff: r.range(250.0, 450.0),
        res: 0.1,
        keytrack: 0.0,
        a: 4.0,
        d: 2.0,
        s: 1.0,
        r: 6.0,
        gain: 0.5,
        ..PATCH
    }
}

fn drone_choir(r: &mut Rng) -> Patch {
    Patch { vowel: r.pick(&[4, 5]), unison: 2, detune: 8.0, width: 0.6, a: 3.0, r: 5.0, keytrack: 0.0, ..pad_choir(r) }
}

fn drone_reed(r: &mut Rng) -> Patch {
    // a hurdy-gurdy or a bagpipe's drone: a buzzing, nasal saw
    Patch {
        unison: 2,
        detune: r.range(3.0, 6.0),
        vowel: r.pick(&[2, 3]),
        cutoff: r.range(1800.0, 2600.0),
        res: 0.2,
        keytrack: 0.0,
        a: 0.4,
        d: 1.0,
        s: 1.0,
        r: 1.0,
        gain: 0.4,
        ..PATCH
    }
}

// -- drum kits (each channel's kinds in their order)

fn kit_kick_808(r: &mut Rng) -> Vec<DrumPatch> {
    vec![DrumPatch { drive: 0.2, ..kick(r, (80.0, 100.0), (43.0, 48.0), (1.0, 1.5), 0.0, 0.0) }]
}
fn kit_kick_909(r: &mut Rng) -> Vec<DrumPatch> {
    vec![kick(r, (150.0, 210.0), (46.0, 54.0), (0.5, 0.75), 0.25, 0.8)]
}
fn kit_kick_dist(r: &mut Rng) -> Vec<DrumPatch> {
    vec![kick(r, (180.0, 230.0), (50.0, 56.0), (0.4, 0.55), 0.3, 3.0)]
}
fn kit_kick_deep(r: &mut Rng) -> Vec<DrumPatch> {
    vec![kick(r, (65.0, 80.0), (36.0, 42.0), (0.8, 1.0), 0.0, 0.3)]
}
fn kit_kick_linn(r: &mut Rng) -> Vec<DrumPatch> {
    vec![kick(r, (150.0, 170.0), (58.0, 64.0), (0.25, 0.32), 0.3, 0.6)]
}
fn kit_kick_soft(r: &mut Rng) -> Vec<DrumPatch> {
    vec![kick(r, (80.0, 100.0), (45.0, 50.0), (0.4, 0.6), 0.0, 0.0)]
}
fn kit_kick_frame(r: &mut Rng) -> Vec<DrumPatch> {
    vec![DrumPatch {
        tone: 1.0,
        f0: r.range(100.0, 120.0),
        f1: r.range(80.0, 95.0),
        pdec: 0.03,
        tdec: 0.6,
        noise: 0.35,
        nmode: 0,
        ncut: 800.0,
        ndec: 0.25,
        gain: 0.9,
        ..DRUM
    }]
}

/// Snares and claps: [the main hit, a softer one for rolls and ghosts].
fn kit_clap_808(r: &mut Rng) -> Vec<DrumPatch> {
    let c = DrumPatch {
        noise: 1.0,
        nmode: 1,
        ncut: r.range(800.0, 1000.0),
        nres: 0.3,
        ndec: 0.45,
        claps: 4,
        gain: 0.9,
        ..DRUM
    };
    vec![c, DrumPatch { claps: 2, ndec: 0.15, gain: 0.6, ..c }]
}
fn kit_snap(r: &mut Rng) -> Vec<DrumPatch> {
    let c = DrumPatch { noise: 1.0, nmode: 1, ncut: r.range(2200.0, 2800.0), nres: 0.4, ndec: 0.06, gain: 0.9, ..DRUM };
    vec![c, DrumPatch { gain: 0.6, ..c }]
}
fn kit_snare_linn(r: &mut Rng) -> Vec<DrumPatch> {
    let c = DrumPatch {
        tone: 0.55,
        f0: 200.0,
        f1: r.range(170.0, 185.0),
        pdec: 0.01,
        tdec: 0.1,
        noise: 0.9,
        nmode: 2,
        ncut: 1800.0,
        nres: 0.1,
        ndec: 0.15,
        gain: 0.8,
        ..DRUM
    };
    vec![c, DrumPatch { ndec: 0.08, gain: 0.6, ..c }]
}
fn kit_brush(r: &mut Rng) -> Vec<DrumPatch> {
    let c = DrumPatch {
        tone: 0.15,
        f0: 220.0,
        f1: 200.0,
        tdec: 0.1,
        noise: 1.0,
        nmode: 1,
        ncut: r.range(3000.0, 4000.0),
        nres: 0.1,
        ndec: 0.25,
        natt: 0.03,
        gain: 0.8,
        ..DRUM
    };
    vec![c, DrumPatch { ndec: 0.12, gain: 0.6, ..c }]
}
fn kit_hand_clap(r: &mut Rng) -> Vec<DrumPatch> {
    let c = DrumPatch {
        noise: 1.0,
        nmode: 1,
        ncut: r.range(1300.0, 1700.0),
        nres: 0.3,
        ndec: 0.12,
        claps: 2,
        gain: 0.8,
        ..DRUM
    };
    vec![c, DrumPatch { claps: 0, gain: 0.5, ..c }]
}

/// Hats: [closed, open].
fn kit_hats_808(r: &mut Rng) -> Vec<DrumPatch> {
    hats(r, 7000.0, 0.05, 0.4, 0.7, 0.8)
}
fn kit_hats_909(r: &mut Rng) -> Vec<DrumPatch> {
    hats(r, 9500.0, 0.04, 0.3, 0.15, 0.8)
}
fn kit_shaker(r: &mut Rng) -> Vec<DrumPatch> {
    let c = DrumPatch {
        noise: 1.0,
        nmode: 1,
        ncut: r.range(5500.0, 7000.0),
        nres: 0.2,
        ndec: 0.08,
        natt: 0.01,
        gain: 0.65,
        ..DRUM
    };
    vec![c, DrumPatch { ndec: 0.2, ..c }]
}
fn kit_tambourine(r: &mut Rng) -> Vec<DrumPatch> {
    let c = DrumPatch {
        noise: 1.0,
        nmode: 2,
        ncut: 6000.0,
        nres: 0.2,
        ndec: 0.12,
        metal: 0.7,
        mf: r.range(3.5, 4.5),
        natt: 0.002,
        gain: 0.7,
        ..DRUM
    };
    vec![c, DrumPatch { ndec: 0.35, ..c }]
}
fn kit_ride(r: &mut Rng) -> Vec<DrumPatch> {
    let c = DrumPatch {
        tone: 0.12,
        f0: 480.0,
        f1: 480.0,
        tdec: 0.8,
        noise: 1.0,
        nmode: 2,
        ncut: 5000.0,
        ndec: 0.9,
        metal: 0.6,
        mf: r.range(2.4, 2.8),
        gain: 0.5,
        ..DRUM
    };
    vec![DrumPatch { ndec: 0.25, ..c }, c]
}

/// Percussion: [rim (or a click), tom low, mid, high, shaker].
fn kit_perc_hand(r: &mut Rng) -> Vec<DrumPatch> {
    // clave, congas and a bongo, a shaker
    let conga = |f: f32| DrumPatch {
        tone: 1.0,
        f0: f * 1.25,
        f1: f,
        pdec: 0.02,
        tdec: 0.2,
        noise: 0.1,
        nmode: 1,
        ncut: 2000.0,
        ndec: 0.02,
        gain: 0.7,
        ..DRUM
    };
    let clave = DrumPatch { tone: 0.9, f0: 2500.0, f1: 2500.0, tdec: 0.05, gain: 0.6, ..DRUM };
    let base = r.range(260.0, 320.0);
    vec![clave, conga(base), conga(base * 1.3), conga(base * 1.8), kit_shaker(r)[0]]
}
fn kit_perc_metal(r: &mut Rng) -> Vec<DrumPatch> {
    // a cowbell, electronic toms, a woodblock
    let cow = DrumPatch {
        tone: 0.7,
        f0: 560.0,
        f1: 560.0,
        tdec: 0.18,
        tone2: 0.8,
        ratio2: 1.48,
        noise: 0.2,
        nmode: 1,
        ncut: 800.0,
        ndec: 0.05,
        drive: 1.5,
        gain: 0.45,
        ..DRUM
    };
    let tom = |f: f32| DrumPatch {
        tone: 1.0,
        f0: f * 2.0,
        f1: f,
        pdec: 0.08,
        tdec: 0.4,
        noise: 0.25,
        nmode: 0,
        ncut: 3000.0,
        ndec: 0.1,
        gain: 0.75,
        ..DRUM
    };
    let wood = DrumPatch {
        tone: 0.8,
        f0: 1000.0,
        f1: 950.0,
        tdec: 0.06,
        noise: 0.3,
        nmode: 1,
        ncut: 1000.0,
        ndec: 0.03,
        gain: 0.5,
        ..DRUM
    };
    let base = r.range(90.0, 120.0);
    vec![cow, tom(base), tom(base * 1.4), tom(base * 1.9), wood]
}
fn kit_spoons(r: &mut Rng) -> Vec<DrumPatch> {
    // spoons (two clicks), a frame drum's rim, finger cymbals
    let spoons = DrumPatch {
        noise: 1.0,
        nmode: 1,
        ncut: r.range(2800.0, 3500.0),
        nres: 0.3,
        ndec: 0.03,
        claps: 2,
        gain: 0.6,
        ..DRUM
    };
    let ding = DrumPatch { tone: 0.6, f0: 2600.0, f1: 2600.0, tdec: 1.2, tone2: 0.5, ratio2: 2.76, gain: 0.25, ..DRUM };
    let tom = |f: f32| DrumPatch {
        tone: 0.9,
        f0: f * 1.2,
        f1: f,
        pdec: 0.03,
        tdec: 0.3,
        noise: 0.2,
        nmode: 0,
        ncut: 900.0,
        ndec: 0.1,
        gain: 0.7,
        ..DRUM
    };
    vec![spoons, tom(140.0), tom(190.0), ding, kit_shaker(r)[0]]
}

// -- the genres' palettes (each role's tuned original comes on top)

const P_AMB_PAD: &[(&str, Make)] = &[
    ("string ensemble", pad_strings),
    ("choir", pad_choir),
    ("glassy FM pad", pad_glass),
    ("shimmer", pad_shimmer),
    ("warm triangle pad", pad_warm),
    ("organ pad", pad_organ),
];
const P_AMB_BELL: &[(&str, Make)] = &[
    ("FM bell", bell_fm),
    ("celesta", celesta),
    ("marimba", marimba),
    ("kalimba", kalimba),
    ("harp", harp),
    ("piano", piano),
    ("music box", music_box),
    ("vibraphone", vibes),
    ("hammered dulcimer", dulcimer),
];
const P_AMB_ARP: &[(&str, Make)] = &[
    ("glass pluck", glass_arp),
    ("kalimba", kalimba),
    ("harp", harp),
    ("marimba", marimba),
    ("nylon guitar", nylon),
    ("celesta", celesta),
    ("lute", lute),
];
const P_AMB_LEAD: &[(&str, Make)] =
    &[("flute", flute), ("recorder", recorder), ("ocarina", ocarina), ("tin whistle", whistle)];
const P_AMB_BASS: &[(&str, Make)] =
    &[("sub", bass_sub), ("upright bass", bass_upright), ("soft synth bass", bass_synth)];
const P_AMB_DRONE: &[(&str, Make)] =
    &[("choir drone", drone_choir), ("hum", drone_hum), ("reed organ drone", drone_reed)];
const K_AMB_KICK: &[(&str, MakeKit)] = &[("frame drum", kit_kick_frame), ("deep kick", kit_kick_deep)];
const K_AMB_PERC: &[(&str, MakeKit)] = &[("hand percussion", kit_perc_hand), ("spoons and finger cymbals", kit_spoons)];
const K_AMB_HAT: &[(&str, MakeKit)] = &[("shaker", kit_shaker), ("tambourine", kit_tambourine)];

const P_TECH_ACID: &[(&str, Make)] = &[("303 (saw)", bass_acid_saw), ("303 (square)", bass_acid_sq)];
const P_TECH_BASS: &[(&str, Make)] =
    &[("Reese", bass_reese), ("sub", bass_sub), ("FM bass", bass_fm), ("sequenced saw", bass_seq)];
const P_TECH_STAB: &[(&str, Make)] = &[("chord stab", stab_saw), ("dub chord", stab_dub), ("organ stab", stab_organ)];
const P_TECH_PAD: &[(&str, Make)] = &[("dark pad", pad_dark), ("strings", pad_strings), ("choir", pad_choir)];
const P_TECH_LEAD: &[(&str, Make)] = &[
    ("pluck", pluck_hook),
    ("hoover", hoover),
    ("vocal chop", vox_chop),
    ("marimba blip", marimba),
    ("sync lead", lead_sync),
];
const K_TECH_KICK: &[(&str, MakeKit)] = &[
    ("808 kick", kit_kick_808),
    ("distorted kick", kit_kick_dist),
    ("deep kick", kit_kick_deep),
    ("909 kick", kit_kick_909),
];
const K_TECH_SNARE: &[(&str, MakeKit)] =
    &[("808 clap", kit_clap_808), ("snap", kit_snap), ("Linn snare", kit_snare_linn)];
const K_TECH_HAT: &[(&str, MakeKit)] =
    &[("808 hats", kit_hats_808), ("909 hats", kit_hats_909), ("shaker", kit_shaker)];
const K_TECH_PERC: &[(&str, MakeKit)] =
    &[("cowbell, toms, woodblock", kit_perc_metal), ("clave, congas", kit_perc_hand)];
const K_CYM: &[(&str, MakeKit)] = &[("ride", kit_ride)];

const P_SW_BASS: &[(&str, Make)] =
    &[("Moog-ish bass", bass_moog), ("slap bass", bass_slap), ("sequenced bass", bass_seq), ("FM bass", bass_fm)];
const P_SW_PAD: &[(&str, Make)] =
    &[("strings", pad_strings), ("brass", pad_brass), ("choir", pad_choir), ("warm pad", pad_warm)];
const P_SW_ARP: &[(&str, Make)] = &[
    ("saw pluck", arp_pluck),
    ("glockenspiel", glockenspiel),
    ("electric piano", ep_rhodes),
    ("marimba", marimba),
    ("pizzicato strings", pizzicato),
];
const P_SW_LEAD: &[(&str, Make)] =
    &[("saw lead", lead_saw), ("square lead", lead_square), ("sync lead", lead_sync), ("sax", sax)];
const P_SW_BELL: &[(&str, Make)] =
    &[("glassy FM", glass_fm), ("electric piano", ep_rhodes), ("vibraphone", vibes), ("celesta", celesta)];
const K_SW_KICK: &[(&str, MakeKit)] =
    &[("Linn kick", kit_kick_linn), ("909 kick", kit_kick_909), ("808 kick", kit_kick_808)];
const K_SW_SNARE: &[(&str, MakeKit)] = &[("Linn snare", kit_snare_linn), ("808 clap", kit_clap_808)];
const K_SW_HAT: &[(&str, MakeKit)] = &[("808 hats", kit_hats_808), ("909 hats", kit_hats_909)];
const K_SW_PERC: &[(&str, MakeKit)] = &[("cowbell and electronic toms", kit_perc_metal), ("congas", kit_perc_hand)];

const P_LOFI_KEYS: &[(&str, Make)] = &[
    ("Rhodes", ep_rhodes),
    ("Wurlitzer", ep_wurli),
    ("felt piano", piano_felt),
    ("nylon guitar", nylon),
    ("piano", piano),
    ("organ", pad_organ),
];
const P_LOFI_BASS: &[(&str, Make)] = &[("upright bass", bass_upright), ("sub", bass_sub), ("synth bass", bass_synth)];
const P_LOFI_LEAD: &[(&str, Make)] = &[
    ("vibraphone", vibes),
    ("flute", flute),
    ("muted trumpet", muted_trumpet),
    ("sax", sax),
    ("harmonica", harmonica),
    ("trumpet", trumpet),
    ("clarinet", clarinet),
];
const K_LOFI_KICK: &[(&str, MakeKit)] = &[("Linn kick", kit_kick_linn), ("soft kick", kit_kick_soft)];
const K_LOFI_SNARE: &[(&str, MakeKit)] = &[("brushes", kit_brush), ("snap", kit_snap)];
const K_LOFI_HAT: &[(&str, MakeKit)] = &[("shaker", kit_shaker), ("808 hats", kit_hats_808)];
const K_LOFI_PERC: &[(&str, MakeKit)] = &[("clave, congas", kit_perc_hand)];

const P_POLKA_BASS: &[(&str, Make)] =
    &[("tuba", tuba), ("bass accordion", bass_accordion), ("upright bass", bass_upright)];
const P_POLKA_PAH: &[(&str, Make)] = &[("banjo", banjo), ("piano", piano), ("cittern", cittern)];
const P_POLKA_LEAD: &[(&str, Make)] =
    &[("musette accordion", accordion), ("bandoneon", bandoneon), ("harmonica", harmonica), ("fiddle", fiddle)];
const P_POLKA_OTHER: &[(&str, Make)] = &[
    ("clarinet", clarinet),
    ("trumpet", trumpet),
    ("trombone", trombone),
    ("tenor horn", horn),
    ("sax", sax),
    ("fiddle", fiddle),
    ("glockenspiel", glockenspiel),
    ("piccolo", whistle),
];
const K_POLKA_KICK: &[(&str, MakeKit)] = &[("deep bass drum", kit_kick_deep)];
const K_POLKA_SNARE: &[(&str, MakeKit)] = &[("brushes", kit_brush), ("hand claps", kit_hand_clap)];
const K_POLKA_HAT: &[(&str, MakeKit)] = &[("tambourine", kit_tambourine), ("ride", kit_ride)];

fn band_ambient(r: &mut Rng, s: &Synth) -> Band {
    let mut b = Band::default();
    b.tone(r, s, CH_PAD, P_AMB_PAD, 60.0);
    b.tone(r, s, CH_BELL, P_AMB_BELL, 76.0);
    b.tone(r, s, CH_ARP, P_AMB_ARP, 72.0);
    b.tone(r, s, CH_LEAD, P_AMB_LEAD, 67.0);
    b.tone(r, s, CH_BASS, P_AMB_BASS, 40.0);
    b.tone(r, s, CH_DRONE, P_AMB_DRONE, 38.0);
    b.kit(r, s, CH_KICK, K_AMB_KICK);
    b.kit(r, s, CH_PERC, K_AMB_PERC);
    b.kit(r, s, CH_HAT, K_AMB_HAT);
    b
}

fn band_techno(r: &mut Rng, s: &Synth, acid: bool) -> Band {
    let mut b = Band::default();
    b.tone(r, s, CH_BASS, if acid { P_TECH_ACID } else { P_TECH_BASS }, 40.0);
    b.tone(r, s, CH_KEYS, P_TECH_STAB, 62.0);
    b.tone(r, s, CH_PAD, P_TECH_PAD, 60.0);
    b.tone(r, s, CH_LEAD, P_TECH_LEAD, 70.0);
    b.kit(r, s, CH_KICK, K_TECH_KICK);
    b.kit(r, s, CH_SNARE, K_TECH_SNARE);
    b.kit(r, s, CH_HAT, K_TECH_HAT);
    b.kit(r, s, CH_PERC, K_TECH_PERC);
    b.kit(r, s, CH_CYM, K_CYM);
    b
}

fn band_synthwave(r: &mut Rng, s: &Synth) -> Band {
    let mut b = Band::default();
    b.tone(r, s, CH_BASS, P_SW_BASS, 40.0);
    b.tone(r, s, CH_PAD, P_SW_PAD, 62.0);
    b.tone(r, s, CH_ARP, P_SW_ARP, 70.0);
    b.tone(r, s, CH_LEAD, P_SW_LEAD, 74.0);
    b.tone(r, s, CH_BELL, P_SW_BELL, 64.0);
    b.kit(r, s, CH_KICK, K_SW_KICK);
    b.kit(r, s, CH_SNARE, K_SW_SNARE);
    b.kit(r, s, CH_HAT, K_SW_HAT);
    b.kit(r, s, CH_PERC, K_SW_PERC);
    b.kit(r, s, CH_CYM, K_CYM);
    b
}

fn band_lofi(r: &mut Rng, s: &Synth) -> Band {
    let mut b = Band::default();
    b.tone(r, s, CH_KEYS, P_LOFI_KEYS, 62.0);
    b.tone(r, s, CH_BASS, P_LOFI_BASS, 40.0);
    b.tone(r, s, CH_LEAD, P_LOFI_LEAD, 74.0);
    b.tone(r, s, CH_BELL, P_LOFI_LEAD, 66.0);
    b.kit(r, s, CH_KICK, K_LOFI_KICK);
    b.kit(r, s, CH_SNARE, K_LOFI_SNARE);
    b.kit(r, s, CH_HAT, K_LOFI_HAT);
    b.kit(r, s, CH_PERC, K_LOFI_PERC);
    b
}

fn band_polka(r: &mut Rng, s: &Synth) -> Band {
    let mut b = Band::default();
    b.tone(r, s, CH_BASS, P_POLKA_BASS, 40.0);
    b.tone(r, s, CH_KEYS, P_POLKA_PAH, 62.0);
    b.tone(r, s, CH_LEAD, P_POLKA_LEAD, 74.0);
    b.tone(r, s, CH_ARP, P_POLKA_OTHER, 70.0);
    b.kit(r, s, CH_KICK, K_POLKA_KICK);
    b.kit(r, s, CH_SNARE, K_POLKA_SNARE);
    b.kit(r, s, CH_HAT, K_POLKA_HAT);
    b
}

/// Techno's basses, the acid ones among them (a mix's).
const P_TECH_BASS_ANY: &[(&str, Make)] = &[
    ("303 (saw)", bass_acid_saw),
    ("303 (square)", bass_acid_sq),
    ("Reese", bass_reese),
    ("FM bass", bass_fm),
    ("sequenced saw", bass_seq),
];

/// Genre `g`'s instruments for the part channel `ch` plays in another
/// genre (a tune, an answer, chords, a bass, a drone), for a mix.
fn palette(g: usize, ch: usize) -> Option<&'static [(&'static str, Make)]> {
    Some(match (g, ch) {
        (AMBIENT, CH_PAD) => P_AMB_PAD,
        (AMBIENT, CH_DRONE) => P_AMB_DRONE,
        (AMBIENT, CH_BELL | CH_KEYS) => P_AMB_BELL,
        (AMBIENT, CH_ARP) => P_AMB_ARP,
        (AMBIENT, CH_LEAD) => P_AMB_LEAD,
        (AMBIENT, CH_BASS) => P_AMB_BASS,
        (TECHNO, CH_BASS) => P_TECH_BASS_ANY,
        (TECHNO, CH_KEYS) => P_TECH_STAB,
        (TECHNO, CH_PAD | CH_DRONE) => P_TECH_PAD,
        (TECHNO, CH_LEAD | CH_ARP | CH_BELL) => P_TECH_LEAD,
        (SYNTHWAVE, CH_BASS) => P_SW_BASS,
        (SYNTHWAVE, CH_PAD | CH_DRONE) => P_SW_PAD,
        (SYNTHWAVE, CH_ARP | CH_KEYS) => P_SW_ARP,
        (SYNTHWAVE, CH_LEAD) => P_SW_LEAD,
        (SYNTHWAVE, CH_BELL) => P_SW_BELL,
        (LOFI, CH_KEYS | CH_PAD) => P_LOFI_KEYS,
        (LOFI, CH_BASS) => P_LOFI_BASS,
        (LOFI, CH_LEAD | CH_BELL | CH_ARP) => P_LOFI_LEAD,
        (POLKA, CH_BASS) => P_POLKA_BASS,
        (POLKA, CH_KEYS) => P_POLKA_PAH,
        (POLKA, CH_LEAD | CH_PAD) => P_POLKA_LEAD,
        (POLKA, CH_ARP | CH_BELL) => P_POLKA_OTHER,
        (SKA, CH_KEYS) => P_SKA_CHOP,
        (SKA, CH_PAD | CH_DRONE) => P_SKA_ORGAN,
        (SKA, CH_LEAD) => P_SKA_LEAD,
        (SKA, CH_BELL | CH_ARP) => P_SKA_HORN,
        (SKA, CH_BASS) => P_SKA_BASS,
        (TAVERN, CH_LEAD) => P_TAV_LEAD,
        (TAVERN, CH_BELL | CH_ARP) => P_TAV_SECOND,
        (TAVERN, CH_KEYS) => P_TAV_PLUCK,
        (TAVERN, CH_DRONE | CH_PAD) => P_TAV_DRONE,
        (TAVERN, CH_BASS) => P_TAV_BASS,
        (MARIACHI, CH_LEAD) => P_MARIACHI_LEAD,
        (MARIACHI, CH_ARP | CH_BELL) => P_MARIACHI_SECOND,
        (MARIACHI, CH_KEYS) => P_MARIACHI_STRUM,
        (MARIACHI, CH_PAD | CH_DRONE) => P_MARIACHI_SUSTAIN,
        (MARIACHI, CH_BASS) => P_MARIACHI_BASS,
        _ => return None,
    })
}

/// Genre `g`'s kits for drum channel `ch`, for a mix.
fn kit_palette(g: usize, ch: usize) -> Option<&'static [(&'static str, MakeKit)]> {
    Some(match (g, ch) {
        (AMBIENT, CH_KICK) => K_AMB_KICK,
        (AMBIENT, CH_PERC) => K_AMB_PERC,
        (AMBIENT, CH_HAT) => K_AMB_HAT,
        (TECHNO, CH_KICK) => K_TECH_KICK,
        (TECHNO, CH_SNARE) => K_TECH_SNARE,
        (TECHNO, CH_HAT) => K_TECH_HAT,
        (TECHNO, CH_PERC) => K_TECH_PERC,
        (SYNTHWAVE, CH_KICK) => K_SW_KICK,
        (SYNTHWAVE, CH_SNARE) => K_SW_SNARE,
        (SYNTHWAVE, CH_HAT) => K_SW_HAT,
        (SYNTHWAVE, CH_PERC) => K_SW_PERC,
        (TECHNO | SYNTHWAVE | POLKA, CH_CYM) => K_CYM,
        (LOFI, CH_KICK) => K_LOFI_KICK,
        (LOFI, CH_SNARE) => K_LOFI_SNARE,
        (LOFI, CH_HAT) => K_LOFI_HAT,
        (LOFI, CH_PERC) => K_LOFI_PERC,
        (POLKA, CH_KICK) => K_POLKA_KICK,
        (POLKA, CH_SNARE) => K_POLKA_SNARE,
        (POLKA, CH_HAT) => K_POLKA_HAT,
        (SKA, CH_KICK) => K_SKA_KICK,
        (SKA, CH_SNARE) => K_SKA_SNARE,
        (SKA, CH_HAT) => K_SKA_HAT,
        (SKA, CH_PERC) => K_SKA_PERC,
        (SKA, CH_CYM) => K_SKA_CYM,
        (TAVERN, CH_KICK) => K_TAV_KICK,
        (TAVERN, CH_HAT) => K_TAV_HAT,
        (TAVERN, CH_PERC) => K_TAV_PERC,
        (MARIACHI, CH_KICK) => K_MARIACHI_KICK,
        (MARIACHI, CH_SNARE) => K_MARIACHI_SNARE,
        (MARIACHI, CH_HAT) => K_MARIACHI_HAT,
        (MARIACHI, CH_PERC) => K_MARIACHI_PERC,
        _ => return None,
    })
}

/// Genre `g`'s sounds for a piece, on `s` (a mix's drummer's, on its own synthesiser).
fn sounds(g: usize, r: &mut Rng, s: &mut Synth) {
    match g {
        AMBIENT => sounds_ambient(r, s),
        TECHNO => {
            let style = if r.chance(0.65) { 0 } else { 1 };
            sounds_techno(r, s, style)
        }
        SYNTHWAVE => sounds_synthwave(r, s),
        LOFI => sounds_lofi(r, s),
        POLKA => {
            let trumpet = r.bit_chance();
            sounds_polka(r, s, trumpet)
        }
        SKA => {
            let style = if r.bit_chance() { SKS_TWO } else { SKS_FIRST };
            sounds_ska(r, s, style)
        }
        MARIACHI => sounds_mariachi(r, s),
        _ => sounds_tavern(r, s),
    }
}

fn sounds_ambient(r: &mut Rng, s: &mut Synth) {
    let ch = &mut s.ch;
    ch[CH_PAD].patch = Patch {
        wave: if r.chance(0.65) { Wave::Saw } else { Wave::Tri },
        unison: 3,
        detune: r.range(6.0, 12.0),
        width: 1.0,
        wave2: r.pick(&[Wave::Tri, Wave::Sine]),
        mix2: r.range(0.4, 0.8),
        semi2: r.pick(&[0.0, 12.0, 12.0, 7.0]),
        cutoff: r.range(1000.0, 2200.0),
        res: r.range(0.1, 0.25),
        keytrack: 0.4,
        a: r.range(2.0, 4.0),
        d: 3.0,
        s: 0.9,
        r: r.range(3.0, 4.5),
        vib: 0.05,
        vib_rate: r.range(0.2, 0.4),
        gain: 0.16,
        ..PATCH
    };
    ch[CH_DRONE].patch = Patch {
        wave: Wave::Sine,
        wave2: Wave::Saw,
        mix2: r.range(0.08, 0.18),
        semi2: 0.08,
        cutoff: r.range(250.0, 450.0),
        res: 0.1,
        keytrack: 0.0,
        a: 4.0,
        d: 2.0,
        s: 1.0,
        r: 6.0,
        gain: 0.5,
        ..PATCH
    };
    ch[CH_BELL].patch = Patch {
        filter: false,
        fm: r.range(1.5, 3.0),
        ratio: r.pick(&[3.5, 1.41, 2.0, 4.0, 2.76]),
        fm_dec: r.range(1.2, 2.5),
        a: 0.002,
        d: r.range(3.0, 5.0),
        s: 0.0,
        r: 2.0,
        gain: 0.3,
        ..PATCH
    };
    ch[CH_ARP].patch = Patch {
        wave: Wave::Tri,
        wave2: Wave::Sine,
        mix2: 0.6,
        semi2: 12.0,
        cutoff: 2500.0,
        res: 0.1,
        keytrack: 0.2,
        a: 0.006,
        d: 1.2,
        s: 0.0,
        r: 0.8,
        gain: 0.25,
        ..PATCH
    };
    // the pulse bass: soft round plucks
    ch[CH_BASS].patch = Patch {
        wave: Wave::Sine,
        wave2: Wave::Tri,
        mix2: 0.3,
        cutoff: 450.0,
        env_amt: 0.8,
        fdec: 0.2,
        keytrack: 0.0,
        a: 0.008,
        d: 0.6,
        s: 0.4,
        r: 0.15,
        gain: 0.7,
        ..PATCH
    };
    ch[CH_BASS].mono = true;
    // the ocarina: a sine and a little triangle, breathy, its vibrato late
    ch[CH_LEAD].patch = Patch {
        wave: Wave::Sine,
        wave2: Wave::Tri,
        mix2: 0.3,
        noise: 0.015,
        cutoff: 2500.0,
        keytrack: 0.3,
        a: 0.04,
        d: 0.5,
        s: 0.8,
        r: 0.4,
        vib: 0.12,
        vib_rate: 4.5,
        gain: 0.35,
        ..PATCH
    };
    ch[CH_LEAD].mono = false;
    mix(&mut ch[CH_LEAD], 0.3, r.range(-0.3, 0.3), 0.6, 0.3, 0.0);
    ch[CH_KICK].drums = vec![kick(r, (80.0, 100.0), (45.0, 50.0), (0.4, 0.6), 0.0, 0.0)];
    ch[CH_HAT].drums = vec![DrumPatch {
        noise: 1.0,
        nmode: 1,
        ncut: r.range(5000.0, 7000.0),
        nres: 0.2,
        ndec: 0.09,
        natt: 0.01,
        gain: 0.6,
        ..DRUM
    }];
    mix(&mut ch[CH_PAD], 0.5, 0.0, 0.6, 0.0, 0.0);
    mix(&mut ch[CH_DRONE], 0.5, 0.0, 0.3, 0.0, 0.0);
    mix(&mut ch[CH_BELL], 0.4, r.range(-0.3, 0.3), 0.8, 0.4, 0.0);
    mix(&mut ch[CH_ARP], 0.3, r.range(-0.3, 0.3), 0.6, 0.45, 0.0);
    mix(&mut ch[CH_BASS], 0.5, 0.0, 0.1, 0.0, 0.0);
    mix(&mut ch[CH_KICK], 0.45, 0.0, 0.2, 0.0, 0.0);
    mix(&mut ch[CH_HAT], 0.25, 0.3, 0.4, 0.2, 0.0);
    ch[CH_PERC].drums = percs(r);
    mix(&mut ch[CH_PERC], 0.18, -0.3, 0.5, 0.25, 0.0);
    ch[CH_PERC].lp_base = 4000.0;
    ch[CH_PERC].lp = 4000.0;
    mix(&mut ch[CH_FX], 1.0, 0.0, 0.5, 0.0, 0.0);
    ch[CH_KICK].lp_base = 2500.0;
    ch[CH_KICK].lp = 2500.0;
    s.rev.fb = r.range(0.93, 0.96);
    s.rev.damp = r.range(0.35, 0.55);
    s.rev_wet = 0.5;
    s.dly.fb = r.range(0.45, 0.6);
    s.dly.lpk = 0.25;
    s.dly.target = r.range(0.45, 0.75) * SR;
    s.dly_rev = 0.4;
    s.r2mode = R2_OFF;
    s.fx.crackle = 0.0;
    s.m.gain = 1.6;
    s.m.target = 0.18;
}

fn sounds_techno(r: &mut Rng, s: &mut Synth, bass_style: u8) {
    let ch = &mut s.ch;
    let (click, drive) = (r.range(0.1, 0.3), r.range(0.4, 1.2));
    ch[CH_KICK].drums = vec![kick(r, (150.0, 210.0), (46.0, 54.0), (0.5, 0.75), click, drive)];
    let clap = DrumPatch {
        noise: 1.0,
        nmode: 1,
        ncut: r.range(1000.0, 1500.0),
        nres: 0.25,
        ndec: r.range(0.25, 0.4),
        claps: 3,
        gain: 0.9,
        ..DRUM
    };
    let snare = DrumPatch {
        tone: 0.5,
        f0: 260.0,
        f1: 200.0,
        pdec: 0.01,
        tdec: 0.12,
        tone2: 0.3,
        ratio2: 1.55,
        noise: 0.8,
        nmode: 2,
        ncut: 1800.0,
        nres: 0.1,
        ndec: 0.2,
        gain: 0.7,
        ..DRUM
    };
    ch[CH_SNARE].drums = vec![clap, snare];
    ch[CH_HAT].drums = hats(r, 8000.0, 0.045, 0.35, 0.45, 0.8);
    ch[CH_PERC].drums = percs(r);
    ch[CH_CYM].drums = crash(r);
    ch[CH_BASS].patch = if bass_style == 0 {
        Patch {
            wave: if r.chance(0.6) { Wave::Saw } else { Wave::Square },
            poles4: true,
            cutoff: r.range(180.0, 320.0),
            res: r.range(0.78, 0.9),
            env_amt: r.range(2.5, 4.0),
            fdec: r.range(0.2, 0.45),
            keytrack: 0.3,
            a: 0.002,
            d: 0.5,
            s: 0.6,
            r: 0.02,
            glide: r.range(0.04, 0.07),
            drive: r.range(1.0, 2.5),
            gain: 0.55,
            ..PATCH
        }
    } else {
        Patch {
            wave: Wave::Saw,
            sub: 0.6,
            cutoff: r.range(250.0, 400.0),
            res: 0.15,
            env_amt: 2.0,
            fdec: 0.12,
            keytrack: 0.3,
            a: 0.002,
            d: 0.22,
            s: 0.0,
            r: 0.03,
            gain: 0.7,
            ..PATCH
        }
    };
    ch[CH_BASS].mono = true;
    ch[CH_KEYS].patch = Patch {
        wave: Wave::Saw,
        unison: 2,
        detune: 12.0,
        width: 0.7,
        cutoff: r.range(600.0, 1200.0),
        res: 0.3,
        env_amt: 2.0,
        fdec: r.range(0.12, 0.25),
        keytrack: 0.3,
        a: 0.002,
        d: 0.35,
        s: 0.0,
        r: 0.2,
        gain: 0.3,
        ..PATCH
    };
    ch[CH_PAD].patch = Patch {
        wave: Wave::Saw,
        unison: 3,
        detune: r.range(10.0, 18.0),
        width: 0.9,
        cutoff: r.range(500.0, 900.0),
        res: 0.2,
        keytrack: 0.3,
        a: 1.5,
        d: 2.0,
        s: 0.8,
        r: 2.5,
        vib: 0.03,
        vib_rate: 0.2,
        gain: 0.16,
        ..PATCH
    };
    // the hook: a short bright pluck
    ch[CH_LEAD].patch = Patch {
        wave: if r.chance(0.5) { Wave::Square } else { Wave::Saw },
        pw: 0.3,
        cutoff: r.range(900.0, 1600.0),
        res: 0.35,
        env_amt: 2.5,
        fdec: 0.12,
        keytrack: 0.4,
        a: 0.002,
        d: 0.2,
        s: 0.0,
        r: 0.08,
        gain: 0.3,
        ..PATCH
    };
    ch[CH_LEAD].mono = false;
    mix(&mut ch[CH_LEAD], 0.3, r.range(-0.3, 0.3), 0.25, 0.4, 0.3);
    mix(&mut ch[CH_KICK], 0.6, 0.0, 0.0, 0.0, 0.0);
    mix(&mut ch[CH_SNARE], 0.5, 0.0, 0.25, 0.0, 0.0);
    mix(&mut ch[CH_HAT], 0.85, 0.15, 0.08, 0.0, 0.0);
    mix(&mut ch[CH_PERC], 0.55, r.range(-0.5, 0.5), 0.2, 0.25, 0.0);
    mix(&mut ch[CH_CYM], 0.45, -0.2, 0.3, 0.0, 0.0);
    mix(&mut ch[CH_BASS], 0.42, 0.0, 0.05, 0.08, 0.35);
    mix(&mut ch[CH_KEYS], 0.35, 0.0, 0.35, 0.45, 0.4);
    mix(&mut ch[CH_PAD], 0.45, 0.0, 0.5, 0.0, 0.6);
    mix(&mut ch[CH_FX], 0.5, 0.0, 0.4, 0.0, 0.0);
    s.rev.fb = r.range(0.82, 0.88);
    s.rev.damp = 0.45;
    s.rev_wet = 0.3;
    s.dly.fb = r.range(0.35, 0.5);
    s.dly.lpk = 0.3;
    s.dly_rev = 0.2;
    s.r2mode = R2_RUMBLE;
    s.rev2.fb = 0.92;
    s.rev2.damp = 0.7;
    s.r2gain = 0.8;
    s.sc_rel = (-1.0 / (0.13 * SR)).exp();
    s.fx.crackle = 0.0;
    s.m.gain = 1.05;
    s.m.target = 0.21;
}

fn sounds_synthwave(r: &mut Rng, s: &mut Synth) {
    let ch = &mut s.ch;
    ch[CH_KICK].drums = vec![kick(r, (170.0, 220.0), (50.0, 58.0), (0.45, 0.6), 0.2, 0.5)];
    let snare = DrumPatch {
        tone: 0.6,
        f0: 240.0,
        f1: r.range(180.0, 200.0),
        pdec: 0.015,
        tdec: 0.14,
        tone2: 0.4,
        ratio2: 1.6,
        noise: 1.0,
        nmode: 2,
        ncut: 1500.0,
        nres: 0.15,
        ndec: r.range(0.22, 0.3),
        gain: 0.8,
        ..DRUM
    };
    ch[CH_SNARE].drums = vec![snare, DrumPatch { ndec: 0.15, ..snare }];
    ch[CH_HAT].drums = hats(r, 8500.0, 0.04, 0.3, 0.25, 0.75);
    ch[CH_PERC].drums = percs(r);
    ch[CH_CYM].drums = crash(r);
    ch[CH_BASS].patch = Patch {
        wave: Wave::Saw,
        sub: 0.3,
        cutoff: r.range(300.0, 600.0),
        res: 0.2,
        env_amt: r.range(2.0, 3.0),
        fdec: r.range(0.15, 0.25),
        keytrack: 0.3,
        a: 0.002,
        d: 0.25,
        s: 0.45,
        r: 0.05,
        gain: 0.6,
        ..PATCH
    };
    ch[CH_BASS].mono = true;
    ch[CH_PAD].patch = Patch {
        wave: Wave::Saw,
        unison: 3,
        detune: r.range(12.0, 22.0),
        width: 1.0,
        wave2: Wave::Saw,
        mix2: 0.25,
        semi2: 12.0,
        cutoff: r.range(1500.0, 3000.0),
        res: 0.1,
        keytrack: 0.3,
        a: r.range(0.3, 0.9),
        d: 1.5,
        s: 0.85,
        r: 1.2,
        vib: 0.04,
        vib_rate: 0.25,
        gain: 0.15,
        ..PATCH
    };
    let sq = r.chance(0.4);
    ch[CH_ARP].patch = Patch {
        wave: if sq { Wave::Square } else { Wave::Saw },
        pw: r.range(0.3, 0.5),
        cutoff: r.range(1000.0, 2200.0),
        res: r.range(0.2, 0.35),
        env_amt: 2.0,
        fdec: r.range(0.15, 0.3),
        keytrack: 0.4,
        a: 0.002,
        d: 0.3,
        s: 0.0,
        r: 0.12,
        gain: 0.3,
        ..PATCH
    };
    ch[CH_LEAD].patch = Patch {
        wave: if sq { Wave::Saw } else { Wave::Square },
        unison: 3,
        detune: r.range(6.0, 10.0),
        width: 0.35,
        cutoff: r.range(2500.0, 4500.0),
        res: 0.15,
        env_amt: 1.0,
        fdec: 0.3,
        keytrack: 0.3,
        a: 0.015,
        d: 0.6,
        s: 0.75,
        r: 0.3,
        vib: r.range(0.18, 0.3),
        vib_rate: r.range(5.0, 6.0),
        glide: 0.06,
        gain: 0.3,
        ..PATCH
    };
    ch[CH_LEAD].mono = true;
    // the second voice: a glassy FM tone
    ch[CH_BELL].patch = Patch {
        filter: false,
        fm: r.range(0.8, 1.6),
        ratio: r.pick(&[2.0, 3.0, 1.0]),
        fm_dec: 0.5,
        a: 0.003,
        d: 1.2,
        s: 0.2,
        r: 0.3,
        gain: 0.3,
        ..PATCH
    };
    mix(&mut ch[CH_BELL], 0.3, r.range(-0.4, 0.4), 0.4, 0.35, 0.2);
    mix(&mut ch[CH_KICK], 0.8, 0.0, 0.0, 0.0, 0.0);
    mix(&mut ch[CH_SNARE], 0.4, 0.0, 0.15, 0.0, 0.0);
    ch[CH_SNARE].rev2 = 0.9;
    mix(&mut ch[CH_HAT], 1.0, 0.2, 0.1, 0.0, 0.0);
    mix(&mut ch[CH_PERC], 0.4, -0.15, 0.3, 0.0, 0.0);
    mix(&mut ch[CH_CYM], 0.45, -0.2, 0.3, 0.0, 0.0);
    mix(&mut ch[CH_BASS], 0.6, 0.0, 0.05, 0.0, 0.25);
    mix(&mut ch[CH_PAD], 0.42, 0.0, 0.45, 0.0, 0.5);
    mix(&mut ch[CH_ARP], 0.36, r.range(-0.3, 0.3), 0.3, 0.4, 0.2);
    mix(&mut ch[CH_LEAD], 0.28, 0.0, 0.35, 0.3, 0.0);
    s.rev.fb = r.range(0.85, 0.9);
    s.rev.damp = 0.3;
    s.rev_wet = 0.3;
    s.dly.fb = r.range(0.35, 0.5);
    s.dly.lpk = 0.35;
    s.dly_rev = 0.25;
    s.r2mode = R2_GATE;
    s.rev2.fb = 0.9;
    s.rev2.damp = 0.15;
    s.r2gain = 1.3;
    s.gate_hold = (r.range(0.2, 0.3) * SR) as u32;
    s.sc_rel = (-1.0 / (0.15 * SR)).exp();
    s.fx.crackle = 0.0;
    s.fx.wind_to = 0.0;
    s.m.gain = 1.3;
    s.m.target = 0.2;
}

fn sounds_lofi(r: &mut Rng, s: &mut Synth) {
    let ch = &mut s.ch;
    ch[CH_KICK].drums = vec![kick(r, (110.0, 140.0), (48.0, 56.0), (0.35, 0.5), 0.05, 0.8)];
    let snare = DrumPatch {
        tone: 0.5,
        f0: 210.0,
        f1: 180.0,
        pdec: 0.012,
        tdec: 0.1,
        tone2: 0.3,
        ratio2: 1.5,
        noise: 0.9,
        nmode: 1,
        ncut: r.range(1400.0, 2200.0),
        nres: 0.1,
        ndec: r.range(0.18, 0.25),
        drive: 0.5,
        gain: 0.8,
        ..DRUM
    };
    let rim = DrumPatch {
        tone: 0.6,
        f0: 1700.0,
        f1: 1700.0,
        tdec: 0.03,
        noise: 0.4,
        nmode: 1,
        ncut: 2200.0,
        nres: 0.3,
        ndec: 0.02,
        gain: 0.7,
        ..DRUM
    };
    ch[CH_SNARE].drums = vec![snare, rim];
    ch[CH_HAT].drums = {
        let c = DrumPatch {
            noise: 1.0,
            nmode: 1,
            ncut: r.range(6500.0, 8500.0),
            nres: 0.2,
            ndec: r.range(0.03, 0.05),
            gain: 0.7,
            ..DRUM
        };
        vec![c, DrumPatch { ndec: 0.2, ..c }]
    };
    ch[CH_PERC].drums = percs(r);
    ch[CH_KEYS].patch = if r.chance(0.7) {
        Patch {
            filter: false,
            fm: r.range(1.2, 2.0),
            ratio: 1.0,
            fm_dec: r.range(0.6, 1.2),
            fm2: r.range(0.4, 0.8),
            ratio2: r.pick(&[14.0, 13.0, 15.0]),
            fm2_dec: 0.05,
            a: 0.002,
            d: r.range(1.2, 2.0),
            s: r.range(0.3, 0.42),
            r: 0.35,
            trem: r.range(0.1, 0.25),
            vib_rate: r.range(3.0, 5.0),
            gain: 0.3,
            ..PATCH
        }
    } else {
        Patch {
            wave: Wave::Tri,
            wave2: Wave::Sine,
            mix2: 0.6,
            semi2: 12.0,
            cutoff: 1400.0,
            env_amt: 1.0,
            fdec: 0.5,
            keytrack: 0.5,
            a: 0.003,
            d: 1.5,
            s: 0.35,
            r: 0.3,
            gain: 0.32,
            ..PATCH
        }
    };
    ch[CH_BASS].patch = Patch {
        wave: Wave::Tri,
        wave2: Wave::Saw,
        mix2: 0.25,
        sub: 0.4,
        cutoff: r.range(450.0, 700.0),
        res: 0.1,
        env_amt: 1.2,
        fdec: 0.15,
        keytrack: 0.3,
        a: 0.004,
        d: 0.8,
        s: 0.35,
        r: 0.08,
        drive: 0.6,
        gain: 0.8,
        ..PATCH
    };
    ch[CH_BASS].mono = true;
    let vibes = Patch {
        filter: false,
        fm: 1.0,
        ratio: 4.0,
        fm_dec: 0.4,
        a: 0.002,
        d: 1.8,
        s: 0.0,
        r: 0.5,
        trem: 0.3,
        vib_rate: 5.0,
        gain: 0.3,
        ..PATCH
    };
    let flute = Patch {
        wave: Wave::Sine,
        wave2: Wave::Tri,
        mix2: 0.4,
        noise: 0.04,
        cutoff: 2500.0,
        keytrack: 0.3,
        a: 0.05,
        d: 0.5,
        s: 0.8,
        r: 0.2,
        vib: 0.15,
        vib_rate: 5.0,
        gain: 0.3,
        ..PATCH
    };
    // the second voice is whichever of vibes and flute the lead is not
    let (lead, second) = if r.chance(0.5) { (vibes, flute) } else { (flute, vibes) };
    ch[CH_LEAD].patch = lead;
    ch[CH_BELL].patch = second;
    mix(&mut ch[CH_BELL], 0.18, r.range(-0.3, 0.3), 0.3, 0.2, 0.1);
    ch[CH_LEAD].mono = false;
    let dusty = r.range(5000.0, 8000.0);
    mix(&mut ch[CH_KICK], 0.85, 0.0, 0.0, 0.0, 0.0);
    mix(&mut ch[CH_SNARE], 0.4, 0.05, 0.15, 0.0, 0.0);
    mix(&mut ch[CH_HAT], 0.75, -0.2, 0.05, 0.0, 0.0);
    mix(&mut ch[CH_PERC], 0.55, 0.3, 0.1, 0.0, 0.0);
    for c in [CH_KICK, CH_SNARE, CH_HAT, CH_PERC] {
        ch[c].lp_base = dusty;
        ch[c].lp = dusty;
    }
    mix(&mut ch[CH_KEYS], 0.5, 0.0, 0.25, 0.1, 0.25);
    mix(&mut ch[CH_BASS], 0.45, 0.0, 0.0, 0.0, 0.15);
    mix(&mut ch[CH_LEAD], 0.2, r.range(-0.2, 0.2), 0.3, 0.2, 0.1);
    mix(&mut ch[CH_FX], 1.0, 0.0, 0.0, 0.0, 0.0);
    s.rev.fb = r.range(0.78, 0.84);
    s.rev.damp = 0.5;
    s.rev_wet = 0.25;
    s.dly.fb = 0.3;
    s.dly.lpk = 0.2;
    s.dly_rev = 0.2;
    s.r2mode = R2_OFF;
    s.sc_rel = (-1.0 / (0.18 * SR)).exp();
    s.fx.crackle = r.range(0.05, 0.1);
    s.fx.crackle_rate = r.range(6.0, 16.0);
    s.fx.hiss = r.range(0.002, 0.005);
    s.fx.wind_to = 0.0;
    s.m.set_lp(r.range(7000.0, 9500.0));
    s.m.wow_to = r.range(0.0004, 0.0008) * SR;
    s.m.wrate = r.range(0.35, 0.7);
    s.m.gain = 1.4;
    s.m.target = 0.195;
}

fn sounds_polka(r: &mut Rng, s: &mut Synth, trumpet: bool) {
    let ch = &mut s.ch;
    ch[CH_KICK].drums = vec![kick(r, (100.0, 130.0), (52.0, 60.0), (0.25, 0.35), 0.08, 0.4)];
    let snare = DrumPatch {
        tone: 0.45,
        f0: 260.0,
        f1: 215.0,
        pdec: 0.01,
        tdec: 0.09,
        tone2: 0.3,
        ratio2: 1.5,
        noise: 0.9,
        nmode: 1,
        ncut: r.range(2500.0, 3500.0),
        nres: 0.15,
        ndec: r.range(0.12, 0.17),
        gain: 0.8,
        ..DRUM
    };
    ch[CH_SNARE].drums = vec![snare, DrumPatch { tone: 0.2, ndec: 0.07, gain: 0.6, ..snare }];
    ch[CH_HAT].drums = hats(r, 8000.0, 0.035, 0.25, 0.2, 0.7);
    ch[CH_CYM].drums = crash(r);
    // the tuba: a soft saw with a sine under it, a brassy bloom in the filter
    ch[CH_BASS].patch = Patch {
        wave: Wave::Saw,
        wave2: Wave::Sine,
        mix2: 0.7,
        cutoff: r.range(350.0, 550.0),
        res: 0.15,
        env_amt: 1.4,
        fdec: 0.1,
        keytrack: 0.5,
        a: 0.015,
        d: 0.25,
        s: 0.6,
        r: 0.07,
        gain: 0.8,
        ..PATCH
    };
    ch[CH_BASS].mono = true;
    // the pah: the accordion's chord reeds
    ch[CH_KEYS].patch = Patch {
        wave: Wave::Square,
        pw: 0.42,
        unison: 2,
        detune: r.range(6.0, 12.0),
        width: 0.4,
        wave2: Wave::Saw,
        mix2: 0.35,
        cutoff: r.range(1800.0, 2600.0),
        res: 0.2,
        env_amt: 0.8,
        fdec: 0.08,
        keytrack: 0.4,
        a: 0.006,
        d: 0.2,
        s: 0.4,
        r: 0.05,
        gain: 0.3,
        ..PATCH
    };
    // the accordion, musette: three reeds (one in tune, one sharp, one flat),
    // reedy (a narrowish pulse and a saw), sometimes an octave reed above
    ch[CH_LEAD].patch = Patch {
        wave: Wave::Square,
        pw: r.range(0.35, 0.45),
        unison: 3,
        detune: r.range(12.0, 22.0),
        width: 0.5,
        wave2: Wave::Saw,
        mix2: 0.3,
        semi2: if r.chance(0.4) { 12.0 } else { 0.0 },
        cutoff: r.range(2600.0, 3600.0),
        res: 0.25,
        keytrack: 0.5,
        a: 0.025,
        d: 0.4,
        s: 0.9,
        r: 0.08,
        gain: 0.32,
        ..PATCH
    };
    ch[CH_LEAD].mono = false;
    // a clarinet (a square: odd harmonics, a breath of noise) or a trumpet
    // (saws with a brassy filter bloom); graces slide into the note
    ch[CH_ARP].patch = if trumpet {
        Patch {
            wave: Wave::Saw,
            unison: 2,
            detune: 4.0,
            width: 0.1,
            cutoff: r.range(1200.0, 1600.0),
            res: 0.15,
            env_amt: 1.6,
            fdec: 0.09,
            keytrack: 0.6,
            a: 0.015,
            d: 0.4,
            s: 0.8,
            r: 0.07,
            vib: 0.1,
            vib_rate: 5.5,
            glide: 0.015,
            gain: 0.3,
            ..PATCH
        }
    } else {
        Patch {
            wave: Wave::Square,
            pw: 0.5,
            cutoff: r.range(1800.0, 2600.0),
            res: 0.1,
            env_amt: 0.5,
            fdec: 0.1,
            keytrack: 0.6,
            noise: 0.012,
            a: 0.02,
            d: 0.3,
            s: 0.85,
            r: 0.06,
            vib: 0.08,
            vib_rate: 5.2,
            glide: 0.015,
            gain: 0.3,
            ..PATCH
        }
    };
    ch[CH_ARP].mono = true;
    mix(&mut ch[CH_KICK], 0.7, 0.0, 0.05, 0.0, 0.0);
    mix(&mut ch[CH_SNARE], 0.45, 0.1, 0.15, 0.0, 0.0);
    mix(&mut ch[CH_HAT], 0.5, 0.2, 0.05, 0.0, 0.0);
    mix(&mut ch[CH_CYM], 0.4, -0.2, 0.2, 0.0, 0.0);
    mix(&mut ch[CH_BASS], 0.5, 0.0, 0.08, 0.0, 0.0);
    mix(&mut ch[CH_KEYS], 0.45, -0.1, 0.15, 0.0, 0.0);
    mix(&mut ch[CH_LEAD], 0.45, 0.1, 0.25, 0.05, 0.0);
    mix(&mut ch[CH_ARP], 0.4, -0.25, 0.25, 0.05, 0.0);
    // the bellows: a slow swell on the accordion's reeds
    let (depth, rate) = (r.range(0.08, 0.15), r.range(0.3, 0.6));
    for c in [CH_LEAD, CH_KEYS] {
        ch[c].swell = depth;
        ch[c].swell_rate = rate;
    }
    s.rev.fb = r.range(0.78, 0.84);
    s.rev.damp = 0.45;
    s.rev_wet = 0.2;
    s.dly.fb = 0.2;
    s.dly.lpk = 0.3;
    s.dly_rev = 0.2;
    s.r2mode = R2_OFF;
    s.fx.crackle = 0.0;
    s.fx.wind_to = 0.0;
    s.m.gain = 1.3;
    s.m.target = 0.2;
}

// -- mariachi

/// Mariachi's styles, one a piece: the son jalisciense (6/8 against 3/4, the
/// sesquialtera), the huapango (6/8, its strum syncopated, its tunes leaping
/// to long high notes), the ranchera valseada (3/4), the ranchera polkeada
/// (2/4) and the slow bolero ranchero (4/4).
const MARIACHI_SON: u8 = 0;
const MARIACHI_HUAPANGO: u8 = 1;
const MARIACHI_VALS: u8 = 2;
const MARIACHI_POLKA: u8 = 3;
const MARIACHI_BOLERO: u8 = 4;
/// Harmonic minor (its row in `MODES`): a minor piece's V7 has its leading tone.
const MARIACHI_MINOR: usize = 6;
/// Who leads a section: the violins, the trumpets, the two trading two-bar
/// phrases, or everyone (the trumpets, the violins an octave above them).
const MARIACHI_VIOLINS: u8 = 0;
const MARIACHI_TRUMPETS: u8 = 1;
const MARIACHI_TRADE: u8 = 2;
const MARIACHI_TUTTI: u8 = 3;
/// The vihuela's and guitar's strokes: down (low string to high), up (the
/// top strings, high to low), a chop (muted at once: the mánico's
/// percussive stroke), a rasgueo (a fan of three strokes in a 16th).
const MARIACHI_DOWN: u8 = 0;
const MARIACHI_UP: u8 = 1;
const MARIACHI_CHOP: u8 = 2;
const MARIACHI_RASGUEO: u8 = 3;
/// Ornaments, in a `PNote`'s grace (small values are grace notes, as in a
/// polka): a trill, a turn, a falsete (the huapango's break up to a high note).
const MARIACHI_TRILL: i8 = 20;
const MARIACHI_TURN: i8 = 21;
const MARIACHI_FALSETE: i8 = 22;
/// A tune's range, semitones about its tonic.
const MARIACHI_LO: i32 = -3;
const MARIACHI_HI: i32 = 14;
/// The guitarrón's lowest root (C2).
const MARIACHI_BASS_LO: i32 = 36;

/// Eight-bar plans (a chord a bar; A+B, two): verses (A); choruses (B), away
/// from home at first; the contrasting part (C: a major piece's to the
/// subdominant or the relative minor, a minor piece's to the relative
/// major), back on the dominant; and four-bar fanfares, to the dominant. A
/// minor piece's V7 has its leading tone (harmonic minor); I7 is V7/IV.
const MARIACHI_MAJ_A: &[&str] = &[
    "I I V7 V7 V7 V7 I I",
    "I I IV IV V7 V7 I I",
    "I IV V7 I I IV V7 I",
    "I I7 IV IV V7 V7 I I",
    "I V7 V7 I I V7 V7 I",
    "I I V7/V V7/V V7 V7 I I",
    "I vi ii V7 I vi V7 I",
    "I IV I V7 I IV I+V7 I",
];
const MARIACHI_MAJ_B: &[&str] = &[
    "IV IV I I V7 V7 I I",
    "I7 IV V7 I I7 IV V7 I",
    "IV I V7 I IV I V7 I",
    "V7 V7 I I V7 V7 I I",
    "vi vi V7/V V7 IV I V7 I",
    "I7 I7 IV IV V7/V V7/V V7 I",
    "IV IV I vi ii V7 I I",
];
const MARIACHI_MAJ_C: &[&str] = &[
    "vi vi V7/vi V7/vi V7/vi V7/vi vi V7",
    "I7 I7 IV IV iv iv I V7",
    "IV IV I I V7/V V7/V V7 V7",
    "vi V7/vi vi vi IV I V7/V V7",
    "I7 IV iv I vi V7/V V7 V7",
];
const MARIACHI_MIN_A: &[&str] = &[
    "i i V7 V7 V7 V7 i i",
    "i iv V7 i i iv V7 i",
    "i i iv iv V7 V7 i i",
    "i V7 i V7 iv i V7 i",
    "i I7 iv iv V7 V7 i i",
];
const MARIACHI_MIN_B: &[&str] = &[
    "iv iv i i V7 V7 i i",
    "I7 iv V7 i I7 iv V7 i",
    "bVI bVI V7 V7 iv i V7 i",
    "iv V7 i bVI iv i V7 i",
];
const MARIACHI_MIN_C: &[&str] = &[
    "bIII bIII bVII7 bVII7 bIII bIII V7 V7",
    "bVII7 bVII7 bIII bIII iv iv V7 V7",
    "bIII bVII7 bIII bVI iv i V7 V7",
    "iv iv bIII bIII bVI bVI V7 V7",
];
const MARIACHI_FANFARE: &[&str] = &["I I V7 V7", "I IV V7 V7", "I V7 I V7", "IV V7 I V7", "I I IV+V7 V7"];
const MARIACHI_FANFARE_MIN: &[&str] = &["i i V7 V7", "i iv V7 V7", "iv V7 i V7", "i i iv+V7 V7"];

/// A mariachi strain: its chords (two a bar), its tune and the other
/// section's answers (pitches in semitones about the tune's tonic).
#[derive(Clone)]
struct MariachiStrain {
    chords: Vec<Harm>,
    mel: Vec<PNote>,
    ans: Vec<PNote>,
}

/// A section of a mariachi piece: its strain, which time through, the
/// strain's bar it starts on, who leads, which of the instrument pairs.
#[derive(Clone, Copy, Default)]
struct MariachiSection {
    strain: usize,
    rep: u32,
    first: u32,
    role: u8,
    dress: usize,
}

/// The mariachi piece in play.
#[derive(Default)]
struct Mariachi {
    /// 16ths a bar and a beat, the meter's name, the style
    bar_steps: u32,
    beat_steps: u32,
    meter: &'static str,
    style: u8,
    /// which bars of each four turn to 3/4 against the 6/8 (bit k: bar k)
    hemi: u8,
    /// verses (A), choruses (B), the contrasting part (C), the fanfare
    strains: Vec<MariachiStrain>,
    map: Vec<MariachiSection>,
    /// this time through: the tune (varied) and answers, who leads, the
    /// harmony under the tune (chord-scale steps below: 2 a third, 5 a
    /// sixth; 0 none), whether the vihuela plays its second mánico
    mel: Vec<PNote>,
    ans: Vec<PNote>,
    role: u8,
    harm: usize,
    alt: bool,
    /// the tune's tonic (MIDI), the trumpets' octave (0 or -12), whether the
    /// piece ends on two stabs (else on a held chord)
    tonic: i32,
    tr_oct: i32,
    button: bool,
    /// the bar's chords (for the stats), the second violins' held notes
    mask: [u16; 2],
    sus_v: Vec<i32>,
    /// as a mix's drummer: whether this section's 6/8 bars alternate with 3/4
    dhemi: bool,
    /// the bars the fanfare plays alone before the band comes in
    call: u32,
}

/// A mánico: a style's strokes in a bar (16th, stroke, strength), the
/// vihuela's (two ways) or the guitar's; the son's (and a huapango's
/// cadence) in 6/8, or in 3/4 against it.
fn mariachi_manico(style: u8, hemi: bool, guitar: bool, alt: bool) -> &'static [(u8, u8, f32)] {
    const D: u8 = MARIACHI_DOWN;
    const U: u8 = MARIACHI_UP;
    const C: u8 = MARIACHI_CHOP;
    let six = matches!(style, MARIACHI_SON | MARIACHI_HUAPANGO);
    match (style, six && hemi, guitar, alt) {
        (_, true, false, false) => &[(0, D, 1.0), (2, U, 0.5), (4, D, 0.9), (6, C, 0.6), (8, D, 0.9), (10, U, 0.5)],
        (_, true, false, true) => &[(0, D, 1.0), (2, C, 0.55), (4, D, 0.92), (6, C, 0.55), (8, D, 0.92), (10, C, 0.55)],
        (_, true, true, _) => &[(0, D, 0.9), (4, D, 0.8), (8, D, 0.8)],
        (MARIACHI_SON, _, false, false) => {
            &[(0, D, 1.0), (2, U, 0.5), (4, D, 0.72), (6, D, 0.95), (8, C, 0.62), (10, U, 0.5)]
        }
        (MARIACHI_SON, _, false, true) => &[(0, D, 1.0), (2, C, 0.6), (4, U, 0.5), (6, D, 0.95), (8, C, 0.6), (10, U, 0.5)],
        (MARIACHI_SON, _, true, _) => &[(0, D, 0.9), (4, U, 0.45), (6, D, 0.85), (10, U, 0.45)],
        (MARIACHI_HUAPANGO, _, false, false) => &[
            (0, D, 1.0),
            (1, U, 0.4),
            (2, C, 0.55),
            (4, D, 0.9),
            (5, U, 0.4),
            (6, C, 0.55),
            (8, D, 0.9),
            (9, U, 0.4),
            (10, C, 0.55),
        ],
        (MARIACHI_HUAPANGO, _, false, true) => {
            &[(0, D, 1.0), (2, U, 0.45), (3, D, 0.6), (4, C, 0.6), (6, D, 0.95), (8, U, 0.45), (9, D, 0.6), (10, C, 0.6)]
        }
        (MARIACHI_HUAPANGO, _, true, _) => &[(0, D, 0.85), (3, U, 0.4), (6, D, 0.8), (9, U, 0.4)],
        (MARIACHI_VALS, _, false, false) => &[(0, C, 0.35), (4, D, 0.85), (6, U, 0.45), (8, D, 0.8), (10, U, 0.45)],
        (MARIACHI_VALS, _, false, true) => &[(4, D, 0.85), (6, U, 0.4), (8, D, 0.8), (10, C, 0.5), (11, U, 0.35)],
        (MARIACHI_VALS, _, true, _) => &[(4, D, 0.7), (8, D, 0.65)],
        (MARIACHI_POLKA, _, false, false) => &[(0, C, 0.35), (2, D, 0.85), (3, U, 0.4), (4, C, 0.35), (6, D, 0.85), (7, U, 0.4)],
        (MARIACHI_POLKA, _, false, true) => &[(2, D, 0.85), (3, U, 0.45), (6, D, 0.85), (7, U, 0.45)],
        (MARIACHI_POLKA, _, true, _) => &[(2, D, 0.7), (6, D, 0.7)],
        // the bolero's: down and up on the eighths, now and then (and the
        // guitar always) a 16th's pickup into the second and fourth beats
        (_, _, false, false) => &[
            (0, D, 0.45),
            (2, U, 0.36),
            (4, D, 0.6),
            (6, U, 0.42),
            (8, C, 0.42),
            (10, U, 0.4),
            (12, D, 0.6),
            (14, U, 0.42),
        ],
        (_, _, false, true) => &[
            (0, D, 0.45),
            (2, U, 0.36),
            (3, U, 0.3),
            (4, D, 0.6),
            (6, U, 0.42),
            (8, C, 0.42),
            (10, U, 0.38),
            (11, U, 0.3),
            (12, D, 0.6),
            (14, U, 0.42),
        ],
        _ => &[
            (0, D, 0.7),
            (3, U, 0.4),
            (4, D, 0.6),
            (6, U, 0.5),
            (8, D, 0.65),
            (11, U, 0.4),
            (12, D, 0.6),
            (14, U, 0.5),
        ],
    }
}

/// Rhythms a bar of tune is made of ((16th, length) lists): by style and
/// meter (`three`: a bar of three quarters, a valseada's or the son's 3/4
/// against its 6/8); a cadence's (its last note the landing); a fanfare's
/// (tattoos of repeated notes, dotted figures).
fn mariachi_rhythms(style: u8, bs: u16, three: bool, cadence: bool, fanfare: bool) -> &'static [&'static [(u16, u16)]] {
    match (bs, three, cadence, fanfare) {
        (8, _, true, _) => &[&[(0, 1), (1, 1), (2, 6)], &[(0, 2), (2, 6)], &[(0, 4), (4, 4)]],
        (8, _, _, true) => &[&[(0, 1), (1, 1), (2, 2), (4, 1), (5, 1), (6, 2)], &[(0, 3), (3, 1), (4, 4)], &[(0, 1), (1, 1), (2, 2), (4, 4)]],
        (8, _, _, _) => &[&[(0, 2), (2, 2), (4, 2), (6, 2)], &[(0, 4), (4, 2), (6, 2)], &[(0, 3), (3, 1), (4, 4)], &[(0, 2), (2, 2), (4, 4)]],
        (16, _, true, true) => &[&[(0, 1), (1, 1), (2, 2), (4, 12)], &[(0, 3), (3, 1), (4, 12)]],
        (16, _, true, _) => &[&[(0, 4), (4, 4), (8, 8)], &[(0, 2), (2, 2), (4, 12)], &[(0, 16)], &[(0, 6), (6, 10)]],
        (16, _, _, true) => &[
            &[(0, 1), (1, 1), (2, 2), (4, 4), (8, 1), (9, 1), (10, 2), (12, 4)],
            &[(0, 3), (3, 1), (4, 4), (8, 8)],
            &[(0, 2), (2, 1), (3, 1), (4, 4), (8, 2), (10, 2), (12, 4)],
        ],
        (16, _, _, _) => &[
            &[(0, 6), (6, 2), (8, 8)],
            &[(0, 3), (3, 3), (6, 2), (8, 8)],
            &[(0, 4), (4, 4), (8, 4), (12, 4)],
            &[(0, 2), (2, 4), (6, 2), (8, 6), (14, 2)],
            &[(0, 8), (8, 4), (12, 4)],
            &[(0, 4), (4, 2), (6, 2), (8, 8)],
            &[(2, 2), (4, 4), (8, 2), (10, 6)],
        ],
        (_, true, true, _) => &[&[(0, 4), (4, 8)], &[(0, 2), (2, 2), (4, 8)], &[(0, 12)]],
        (_, false, true, _) if style == MARIACHI_HUAPANGO => &[&[(0, 2), (2, 2), (4, 8)], &[(0, 12)], &[(0, 4), (4, 8)]],
        (_, false, true, _) => &[&[(0, 2), (2, 2), (4, 2), (6, 6)], &[(0, 4), (4, 2), (6, 6)], &[(0, 6), (6, 6)]],
        (_, true, _, true) => &[&[(0, 1), (1, 1), (2, 2), (4, 4), (8, 4)], &[(0, 3), (3, 1), (4, 4), (8, 4)], &[(0, 2), (2, 2), (4, 2), (6, 2), (8, 4)]],
        (_, false, _, true) => &[
            &[(0, 1), (1, 1), (2, 2), (4, 2), (6, 2), (8, 4)],
            &[(0, 2), (2, 2), (4, 2), (6, 6)],
            &[(0, 1), (1, 1), (2, 1), (3, 1), (4, 2), (6, 2), (8, 2), (10, 2)],
            &[(0, 4), (4, 2), (6, 4), (10, 2)],
        ],
        (_, true, _, _) if style == MARIACHI_VALS => &[
            &[(0, 8), (8, 4)],
            &[(0, 4), (4, 4), (8, 4)],
            &[(0, 6), (6, 2), (8, 4)],
            &[(0, 4), (4, 2), (6, 2), (8, 4)],
            &[(0, 2), (2, 2), (4, 4), (8, 4)],
            &[(0, 4), (4, 8)],
        ],
        (_, true, _, _) => &[
            &[(0, 4), (4, 4), (8, 4)],
            &[(0, 2), (2, 2), (4, 4), (8, 4)],
            &[(0, 4), (4, 2), (6, 2), (8, 4)],
            &[(0, 2), (2, 2), (4, 2), (6, 2), (8, 4)],
            &[(0, 4), (4, 4), (8, 2), (10, 2)],
        ],
        _ if style == MARIACHI_HUAPANGO => &[
            &[(0, 2), (2, 2), (4, 8)],
            &[(0, 2), (2, 4), (6, 2), (8, 4)],
            &[(0, 4), (4, 2), (6, 6)],
            &[(0, 6), (6, 6)],
            &[(0, 2), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2)],
            &[(0, 2), (2, 4), (6, 6)],
        ],
        _ => &[
            &[(0, 2), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2)],
            &[(0, 4), (4, 2), (6, 4), (10, 2)],
            &[(0, 2), (2, 2), (4, 2), (6, 4), (10, 2)],
            &[(0, 4), (4, 2), (6, 2), (8, 2), (10, 2)],
            &[(0, 3), (3, 1), (4, 2), (6, 3), (9, 1), (10, 2)],
            &[(0, 6), (6, 2), (8, 2), (10, 2)],
        ],
    }
}

/// A bar of tune in rhythm `tpl` over the chords (`chord_at` a 16th), from
/// near `prev`: its first note, and those on a beat (`beat` 16ths) or long,
/// on a chord tone (the next one along, mostly); the rest stepping through
/// the chord-scale (now and then skipping to a chord tone, or repeating),
/// turning at the range's edges. A fanfare's on chord tones alone, repeated
/// or climbing. A cadence's last note on the chord tone nearest `land`.
/// With `leaps` (a huapango's) a long note after a short one now and then
/// jumps a sixth or more, to come down after.
fn mariachi_figure(
    r: &mut Rng,
    tpl: &[(u16, u16)],
    chord_at: &dyn Fn(u16) -> Harm,
    mode: usize,
    prev: i32,
    dir: &mut i32,
    beat: u16,
    fanfare: bool,
    land: Option<i32>,
    leaps: bool,
) -> Vec<PNote> {
    let (lo, hi) = (MARIACHI_LO, MARIACHI_HI);
    let mut p = prev;
    let mut v = Vec::with_capacity(tpl.len());
    for (k, &(st, len)) in tpl.iter().enumerate() {
        let h = chord_at(st);
        let ct = in_range(&h.pcs(), lo, hi);
        let strong = st % beat == 0 || len >= beat;
        let along = |v: &[i32], from: i32, by: i32| v[(nearest(v, from) as i32 + by).clamp(0, v.len() as i32 - 1) as usize];
        p = match land {
            Some(goal) if k + 1 == tpl.len() => ct[nearest(&ct, goal)],
            _ if fanfare => {
                if k > 0 && r.chance(0.4) {
                    ct[nearest(&ct, p)]
                } else {
                    along(&ct, p, *dir)
                }
            }
            _ if k == 0 => ct[nearest(&ct, p + 2 * *dir)],
            _ if strong => match r.weighted(&[0.6, 0.15, 0.12, 0.13]) {
                0 => along(&ct, p, *dir),
                1 => along(&ct, p, -*dir),
                2 => along(&ct, p, 2 * *dir),
                _ => ct[nearest(&ct, p)],
            },
            _ => match r.weighted(&[0.6, 0.25, 0.15]) {
                0 => along(&in_range(&chord_scale(mode, h), lo, hi), p, *dir),
                1 => along(&ct, p, *dir),
                _ => p,
            },
        };
        let short_before = k > 0 && tpl[k - 1].1 < len;
        if leaps && len >= 4 && short_before && land.is_none_or(|_| k + 1 < tpl.len()) && r.chance(0.65) {
            let from = v.last().map_or(p, |n: &PNote| n.pitch);
            if let Some(&q) = ct.iter().find(|&&q| q >= from + 7) {
                p = q;
                *dir = -1;
            }
        }
        if p >= hi - 2 {
            *dir = -1;
        } else if p <= lo + 2 {
            *dir = 1;
        } else if strong && r.chance(0.2) {
            *dir = -*dir;
        }
        let vel = if st == 0 {
            0.92
        } else if st % beat == 0 {
            0.84
        } else {
            0.76
        } + r.range(-0.04, 0.04);
        v.push(PNote { step: st, len, pitch: p, vel, grace: 0 });
    }
    v
}

/// A bar of tune moved onto other chords: the tones of its chord to the
/// nearest of the new one's, the rest to the nearest of its scale.
fn mariachi_refit(notes: &[PNote], from: &dyn Fn(u16) -> Harm, to: &dyn Fn(u16) -> Harm, mode: usize) -> Vec<PNote> {
    notes
        .iter()
        .map(|n| {
            let (f, t) = (from(n.step), to(n.step));
            if f == t {
                return *n;
            }
            let set = if f.pcs().contains(&n.pitch.rem_euclid(12)) { t.pcs() } else { chord_scale(mode, t) };
            let v = in_range(&set, MARIACHI_LO, MARIACHI_HI);
            PNote { pitch: v[nearest(&v, n.pitch)], ..*n }
        })
        .collect()
}

/// A strain's tune: four-bar phrases, the first two bars a motif that comes
/// back fitted to the chords in bars 5 and 6, bar 4 resting (a half
/// cadence), bar 8 home; a sixteen-bar strain's first eight close on the
/// dominant and come again with a new ending; the son's bars in 3/4 where
/// its mánico turns (`hemi`). Where the tune holds a long note the other
/// section answers (scale runs or arpeggios, below it). A fanfare: chord
/// tones in tattoos and dotted figures, its second phrase echoing the first,
/// closing on the dominant, the answers in 16th runs. A chorus (`high`)
/// starts higher.
fn mariachi_tune(
    r: &mut Rng,
    style: u8,
    hemi: u8,
    chords: &[Harm],
    bs: u16,
    beat: u16,
    mode: usize,
    fanfare: bool,
    high: bool,
) -> (Vec<PNote>, Vec<PNote>) {
    let bars = chords.len() / 2;
    let chord_at = |bar: usize, st: u16| chords[bar * 2 + (st as usize * 2 / bs as usize).min(1)];
    let mut out: Vec<Vec<PNote>> = Vec::with_capacity(bars);
    let mut prev: i32 = if high { r.pick(&[9, 11, 12]) } else { r.pick(&[0, 4, 7, 7]) };
    let mut dir = 1;
    for bar in 0..bars {
        let pos = bar % 8;
        let hb = (hemi >> (bar % 4)) & 1 == 1;
        let three = hb || (bs == 12 && beat == 4);
        let last = bar + 1 == bars;
        let cad = last || (!fanfare && (pos == 3 || pos == 7));
        let back = if !fanfare && bar >= 8 && bar + 2 < bars && r.chance(0.85) {
            Some(bar - 8)
        } else if bar >= 4 && (fanfare || pos == 4 || pos == 5) && !last && r.chance(0.8) {
            Some(bar - 4)
        } else {
            None
        };
        let notes = match back {
            Some(src) => mariachi_refit(&out[src], &|st| chord_at(src, st), &|st| chord_at(bar, st), mode),
            None => {
                let tpl = r.pick(mariachi_rhythms(style, bs, three, cad, fanfare));
                // home at the end (a fanfare's end, and a part's on the
                // dominant, the nearest of its tones), else a fifth or a third
                let home = chord_at(bar, bs - 1).root == 0;
                let land = cad.then(|| {
                    let target: [i32; 3] = if last && home { [0, 12, 4] } else { [7, 4, 2] };
                    let want: Vec<i32> =
                        target.iter().map(|&x| x + 12 * ((prev - x) as f32 / 12.0).round() as i32).collect();
                    want[nearest(&want, prev)].clamp(MARIACHI_LO, MARIACHI_HI)
                });
                let eb = if three { 4 } else { beat };
                let leaps = style == MARIACHI_HUAPANGO && !fanfare;
                mariachi_figure(r, tpl, &|st| chord_at(bar, st), mode, prev, &mut dir, eb, fanfare, land, leaps)
            }
        };
        prev = notes.last().map_or(prev, |n| n.pitch);
        out.push(notes);
    }
    let mut mel = Vec::new();
    let mut ans = Vec::new();
    for (bar, notes) in out.iter().enumerate() {
        let st0 = bs * bar as u16;
        mel.extend(notes.iter().map(|n| PNote { step: n.step + st0, ..*n }));
        let Some(l) = notes.last() else { continue };
        let cad = bar % 8 == 3 || bar % 8 == 7 || bar + 1 == bars;
        if l.step > bs / 2 || l.len < 4 || !r.chance(if cad { 0.8 } else { 0.4 }) {
            continue;
        }
        let h = chord_at(bar, bs - 1);
        let run = fanfare || (bs == 16 && r.chance(0.4));
        let q = if run { 1 } else { 2 };
        let start = l.step + 2;
        let v = if run || r.bit_chance() {
            in_range(&chord_scale(mode, h), MARIACHI_LO - 7, MARIACHI_HI - 5)
        } else {
            in_range(&h.pcs(), MARIACHI_LO - 7, MARIACHI_HI - 5)
        };
        let j = nearest(&v, l.pitch - 5) as isize;
        let n = ((bs - start) / q).min(8);
        // up or down, whichever has room; turning back at the range's edge
        let (top, room) = (v.len() as isize - 1, n as isize - 1);
        let up = if j + room > top { false } else if j - room < 0 { true } else { r.bit_chance() };
        for k in 0..n as isize {
            let mut i = j + if up { k } else { -k };
            if i > top {
                i = 2 * top - i;
            } else if i < 0 {
                i = -i;
            }
            let pitch = v[i.clamp(0, top) as usize];
            ans.push(PNote { step: st0 + start + k as u16 * q, len: q, pitch, vel: 0.72 + r.range(-0.04, 0.04), grace: 0 });
        }
    }
    (mel, ans)
}

/// Ornaments on a tune's long notes, each with chance `p`: trills and turns,
/// grace notes; in a huapango a leap up to a long note breaks into falsete.
fn mariachi_ornaments(r: &mut Rng, mel: &mut [PNote], style: u8, p: f32) {
    let mut prev: Option<i32> = None;
    for n in mel.iter_mut() {
        if n.grace == 0 && n.len >= 4 {
            if style == MARIACHI_HUAPANGO && prev.is_some_and(|q| n.pitch - q >= 5) && r.chance(0.85) {
                n.grace = MARIACHI_FALSETE;
            } else if r.chance(p) {
                n.grace = r.pick(&[MARIACHI_TRILL, MARIACHI_TRILL, MARIACHI_TURN, -1, 2]);
            }
        }
        prev = Some(n.pitch);
    }
}

/// The tone `steps` chord-scale steps below `p` (2: a third, 5: a sixth) over `h`.
fn mariachi_below(mode: usize, h: Harm, p: i32, steps: usize) -> i32 {
    let sc = in_range(&chord_scale(mode, h), p - 14, p + 1);
    sc[nearest(&sc, p).saturating_sub(steps)]
}

// the instruments

fn mariachi_violins(r: &mut Rng) -> Patch {
    // a section: three saws a little apart, a bright body (the vowel e), the
    // bow's hiss, a wide quick vibrato
    Patch {
        unison: 3,
        detune: r.range(10.0, 16.0),
        width: 0.5,
        vowel: 2,
        noise: 0.015,
        cutoff: r.range(4500.0, 5500.0),
        keytrack: 0.3,
        a: 0.04,
        d: 0.3,
        s: 0.9,
        r: 0.18,
        vib: r.range(0.22, 0.3),
        vib_rate: r.range(5.8, 6.6),
        glide: 0.03,
        gain: 0.45,
        ..PATCH
    }
}

fn mariachi_violins_sweet(r: &mut Rng) -> Patch {
    Patch {
        detune: r.range(7.0, 11.0),
        vowel: 1,
        noise: 0.01,
        cutoff: r.range(3000.0, 3800.0),
        a: 0.06,
        vib: r.range(0.18, 0.24),
        vib_rate: r.range(5.4, 6.0),
        ..mariachi_violins(r)
    }
}

fn mariachi_viola(r: &mut Rng) -> Patch {
    Patch {
        unison: 2,
        detune: r.range(6.0, 9.0),
        vowel: 4,
        cutoff: r.range(2400.0, 3000.0),
        a: 0.06,
        vib: 0.16,
        vib_rate: 5.5,
        gain: 0.5,
        ..mariachi_violins(r)
    }
}

fn mariachi_violins_long(r: &mut Rng) -> Patch {
    Patch { a: r.range(0.25, 0.4), d: 1.0, r: 0.6, ..mariachi_violins_sweet(r) }
}

fn mariachi_trumpet(r: &mut Rng) -> Patch {
    // bright and open, a brassy bloom in the filter, a strong vibrato
    Patch {
        unison: 2,
        detune: r.range(4.0, 7.0),
        width: 0.15,
        cutoff: r.range(2200.0, 2800.0),
        res: 0.2,
        env_amt: 1.5,
        fdec: 0.12,
        keytrack: 0.6,
        a: 0.012,
        d: 0.35,
        s: 0.85,
        r: 0.08,
        vib: r.range(0.16, 0.24),
        vib_rate: r.range(5.8, 6.6),
        glide: 0.012,
        drive: 0.35,
        gain: 0.28,
        ..PATCH
    }
}

fn mariachi_flugel(r: &mut Rng) -> Patch {
    // a darker, rounder trumpet (a flugelhorn's)
    Patch {
        wave2: Wave::Sine,
        mix2: 0.3,
        cutoff: r.range(900.0, 1200.0),
        res: 0.1,
        env_amt: 1.2,
        fdec: 0.15,
        vib: r.range(0.1, 0.15),
        drive: 0.0,
        gain: 0.32,
        ..mariachi_trumpet(r)
    }
}

fn mariachi_vihuela(r: &mut Rng) -> Patch {
    // small and high, its strings short and bright, strummed hard
    Patch {
        ks: r.range(0.35, 0.55),
        ks_tone: r.range(0.86, 0.94),
        cutoff: 7000.0,
        keytrack: 0.0,
        drive: r.range(0.2, 0.45),
        a: 0.001,
        d: 2.0,
        s: 1.0,
        r: 0.05,
        gain: 0.5,
        ..PATCH
    }
}

fn mariachi_jarana(r: &mut Rng) -> Patch {
    Patch {
        filter: false,
        ks: r.range(0.6, 0.9),
        ks_tone: 0.8,
        drive: 0.1,
        a: 0.001,
        d: 2.0,
        s: 1.0,
        r: 0.06,
        gain: 0.5,
        ..PATCH
    }
}

fn mariachi_guitar(r: &mut Rng) -> Patch {
    // a Spanish guitar's nylon strings, damped quickly for the chops
    Patch {
        ks: r.range(1.4, 2.0),
        ks_tone: r.range(0.48, 0.58),
        cutoff: 4000.0,
        keytrack: 0.3,
        a: 0.001,
        d: 3.0,
        s: 1.0,
        r: 0.08,
        gain: 0.5,
        ..PATCH
    }
}

fn mariachi_requinto(r: &mut Rng) -> Patch {
    Patch {
        filter: false,
        ks: r.range(1.0, 1.4),
        ks_tone: 0.78,
        a: 0.001,
        d: 3.0,
        s: 1.0,
        r: 0.1,
        gain: 0.48,
        ..PATCH
    }
}

fn mariachi_guitarron(r: &mut Rng) -> Patch {
    // the deep acoustic bass: a string plucked hard and ringing, its snap a
    // brief opening of the filter, then dark (the filter following the
    // note: its body the fundamental and the first few harmonics), a little
    // grit
    Patch {
        ks: r.range(4.0, 5.0),
        ks_tone: r.range(0.3, 0.36),
        cutoff: r.range(600.0, 720.0),
        env_amt: 2.8,
        fdec: 0.035,
        keytrack: 1.0,
        drive: 0.12,
        a: 0.001,
        d: 2.0,
        s: 1.0,
        r: 0.12,
        gain: 0.85,
        ..PATCH
    }
}

fn mariachi_guitarron_round(r: &mut Rng) -> Patch {
    Patch { ks_tone: 0.3, cutoff: r.range(420.0, 500.0), env_amt: 1.0, drive: 0.0, ..mariachi_guitarron(r) }
}

fn mariachi_upright(r: &mut Rng) -> Patch {
    // an upright bass as dark as the guitarrón
    Patch { cutoff: 520.0, keytrack: 1.0, ..bass_upright(r) }
}

fn mariachi_arpa(r: &mut Rng) -> Patch {
    Patch { filter: false, ks: r.range(2.5, 3.5), ks_tone: 0.7, a: 0.001, d: 4.0, s: 1.0, r: 0.8, gain: 0.5, ..PATCH }
}

/// The tarima under the dancers' feet: [a heel stomp, a toe tap].
fn mariachi_kit_tarima(r: &mut Rng) -> Vec<DrumPatch> {
    let stomp = DrumPatch {
        tone: 1.0,
        f0: r.range(150.0, 180.0),
        f1: r.range(85.0, 100.0),
        pdec: 0.012,
        tdec: 0.12,
        noise: 0.7,
        nmode: 0,
        ncut: r.range(900.0, 1300.0),
        nres: 0.2,
        ndec: 0.05,
        click: 0.3,
        gain: 0.85,
        ..DRUM
    };
    let tap = DrumPatch {
        tone: 0.6,
        f0: 420.0,
        f1: 360.0,
        tdec: 0.04,
        noise: 0.6,
        nmode: 1,
        ncut: 2200.0,
        nres: 0.3,
        ndec: 0.025,
        click: 0.2,
        gain: 0.5,
        ..stomp
    };
    vec![stomp, tap]
}

/// A cajón, played with the heel and the hand: [the low slap, the high one].
fn mariachi_kit_cajon(r: &mut Rng) -> Vec<DrumPatch> {
    let low = DrumPatch {
        tone: 1.0,
        f0: r.range(130.0, 150.0),
        f1: r.range(70.0, 80.0),
        pdec: 0.015,
        tdec: 0.18,
        noise: 0.4,
        nmode: 1,
        ncut: 1500.0,
        nres: 0.2,
        ndec: 0.06,
        gain: 0.8,
        ..DRUM
    };
    vec![low, DrumPatch { tone: 0.4, f0: 300.0, f1: 260.0, tdec: 0.05, noise: 0.8, ncut: 3000.0, ndec: 0.05, gain: 0.5, ..low }]
}

/// Palmas, many hands: [a clap, a softer one].
fn mariachi_kit_palmas(r: &mut Rng) -> Vec<DrumPatch> {
    let c = DrumPatch {
        noise: 1.0,
        nmode: 1,
        ncut: r.range(1100.0, 1500.0),
        nres: 0.25,
        ndec: 0.09,
        claps: 3,
        gain: 0.8,
        ..DRUM
    };
    vec![c, DrumPatch { claps: 2, ndec: 0.05, gain: 0.5, ..c }]
}

/// Maracas: [a short shake, a long one].
fn mariachi_kit_maracas(r: &mut Rng) -> Vec<DrumPatch> {
    let c = DrumPatch {
        noise: 1.0,
        nmode: 2,
        ncut: r.range(4500.0, 6000.0),
        nres: 0.1,
        ndec: 0.05,
        natt: 0.012,
        gain: 0.55,
        ..DRUM
    };
    vec![c, DrumPatch { ndec: 0.16, natt: 0.03, ..c }]
}

/// Percussion: [a clave, a cajón low, mid and high, the guiro's scrape].
fn mariachi_kit_guiro(r: &mut Rng) -> Vec<DrumPatch> {
    let f = r.range(2300.0, 2700.0);
    let clave = DrumPatch { tone: 0.9, f0: f, f1: f, tdec: 0.05, gain: 0.55, ..DRUM };
    let cajon = |f: f32| DrumPatch {
        tone: 1.0,
        f0: f * 1.3,
        f1: f,
        pdec: 0.015,
        tdec: 0.12,
        noise: 0.35,
        nmode: 1,
        ncut: 1800.0,
        ndec: 0.04,
        gain: 0.7,
        ..DRUM
    };
    let guiro = DrumPatch {
        noise: 1.0,
        nmode: 1,
        ncut: r.range(2800.0, 3600.0),
        nres: 0.45,
        ndec: 0.03,
        claps: 8,
        gain: 0.5,
        ..DRUM
    };
    vec![clave, cajon(110.0), cajon(160.0), cajon(240.0), guiro]
}

// the palettes: a mariachi piece's own, by part
const P_MARIACHI_VIOLIN: &[(&str, Make)] = &[
    ("violin section", mariachi_violins),
    ("sweet violins", mariachi_violins_sweet),
    ("fiddle", fiddle),
    ("flute", flute),
];
const P_MARIACHI_VIOLIN2: &[(&str, Make)] = &[
    ("violin section", mariachi_violins),
    ("sweet violins", mariachi_violins_sweet),
    ("violas", mariachi_viola),
    ("string ensemble", pad_strings),
];
const P_MARIACHI_TRUMPET: &[(&str, Make)] = &[
    ("mariachi trumpets", mariachi_trumpet),
    ("flugelhorns", mariachi_flugel),
    ("trumpets", trumpet),
    ("muted trumpets", muted_trumpet),
];
const P_MARIACHI_VIHUELA: &[(&str, Make)] =
    &[("vihuela", mariachi_vihuela), ("jarana", mariachi_jarana), ("cittern", cittern)];
const P_MARIACHI_GUITAR: &[(&str, Make)] = &[
    ("Spanish guitar", mariachi_guitar),
    ("nylon guitar", nylon),
    ("requinto", mariachi_requinto),
    ("lute", lute),
];
const P_MARIACHI_BASS: &[(&str, Make)] = &[
    ("guitarrón", mariachi_guitarron),
    ("round guitarrón", mariachi_guitarron_round),
    ("upright bass", mariachi_upright),
];
const P_MARIACHI_HARP: &[(&str, Make)] = &[
    ("arpa", mariachi_arpa),
    ("harp", harp),
    ("salterio", psaltery),
    ("requinto", mariachi_requinto),
];
// lent to another genre's parts, in a mix
const P_MARIACHI_LEAD: &[(&str, Make)] = &[
    ("violin section", mariachi_violins),
    ("mariachi trumpet", mariachi_trumpet),
    ("sweet violins", mariachi_violins_sweet),
    ("flugelhorn", mariachi_flugel),
];
const P_MARIACHI_SECOND: &[(&str, Make)] = &[
    ("arpa", mariachi_arpa),
    ("mariachi trumpet", mariachi_trumpet),
    ("salterio", psaltery),
    ("requinto", mariachi_requinto),
];
const P_MARIACHI_STRUM: &[(&str, Make)] = &[
    ("vihuela", mariachi_vihuela),
    ("jarana", mariachi_jarana),
    ("Spanish guitar", mariachi_guitar),
    ("requinto", mariachi_requinto),
];
const P_MARIACHI_SUSTAIN: &[(&str, Make)] = &[
    ("violins, held", mariachi_violins_long),
    ("sweet violins", mariachi_violins_sweet),
    ("violas", mariachi_viola),
    ("string ensemble", pad_strings),
];
const K_MARIACHI_KICK: &[(&str, MakeKit)] = &[("tarima", mariachi_kit_tarima), ("cajón", mariachi_kit_cajon)];
const K_MARIACHI_SNARE: &[(&str, MakeKit)] = &[("palmas", mariachi_kit_palmas), ("hand claps", kit_hand_clap)];
const K_MARIACHI_HAT: &[(&str, MakeKit)] = &[("maracas", mariachi_kit_maracas), ("shaker", kit_shaker)];
const K_MARIACHI_PERC: &[(&str, MakeKit)] =
    &[("guiro, clave, cajón", mariachi_kit_guiro), ("clave, congas", kit_perc_hand)];

fn band_mariachi(r: &mut Rng, s: &Synth) -> Band {
    let mut b = Band::default();
    b.tone(r, s, CH_LEAD, P_MARIACHI_VIOLIN, 76.0);
    b.tone(r, s, CH_DRONE, P_MARIACHI_VIOLIN2, 70.0);
    b.tone(r, s, CH_ARP, P_MARIACHI_TRUMPET, 70.0);
    b.tone(r, s, CH_KEYS, P_MARIACHI_VIHUELA, 64.0);
    b.tone(r, s, CH_PAD, P_MARIACHI_GUITAR, 55.0);
    b.tone(r, s, CH_BASS, P_MARIACHI_BASS, 40.0);
    b.tone(r, s, CH_BELL, P_MARIACHI_HARP, 67.0);
    b.kit(r, s, CH_KICK, K_MARIACHI_KICK);
    b.kit(r, s, CH_SNARE, K_MARIACHI_SNARE);
    b.kit(r, s, CH_HAT, K_MARIACHI_HAT);
    b.kit(r, s, CH_PERC, K_MARIACHI_PERC);
    b
}

/// Mariachi's sounds: the violins (the first mono, sliding into their
/// ornaments; the second under them), the trumpets, the vihuela and guitar,
/// the guitarrón, the harp; the tarima, palmas, maracas and guiro; a room
/// with some air (a plaza's), a little echo on the trumpets. Spread as the
/// band stands: the first violins to one side, the trumpets to the other,
/// the second violins between the trumpets and the middle; the vihuela and
/// guitar a little either side, the guitarrón in the middle, loud (the
/// records' low end), the harp by the violins.
fn sounds_mariachi(r: &mut Rng, s: &mut Synth) {
    let ch = &mut s.ch;
    ch[CH_LEAD].patch = mariachi_violins(r);
    ch[CH_LEAD].mono = true;
    ch[CH_DRONE].patch = mariachi_violins(r);
    ch[CH_DRONE].mono = false;
    ch[CH_ARP].patch = mariachi_trumpet(r);
    ch[CH_ARP].mono = false;
    ch[CH_KEYS].patch = mariachi_vihuela(r);
    ch[CH_PAD].patch = mariachi_guitar(r);
    ch[CH_BASS].patch = mariachi_guitarron(r);
    ch[CH_BASS].mono = false;
    ch[CH_BELL].patch = mariachi_arpa(r);
    ch[CH_KICK].drums = mariachi_kit_tarima(r);
    ch[CH_SNARE].drums = mariachi_kit_palmas(r);
    ch[CH_HAT].drums = mariachi_kit_maracas(r);
    ch[CH_PERC].drums = mariachi_kit_guiro(r);
    mix(&mut ch[CH_LEAD], 0.93, 0.6, 0.25, 0.04, 0.0);
    mix(&mut ch[CH_DRONE], 0.15, -0.3, 0.28, 0.0, 0.0);
    mix(&mut ch[CH_ARP], 0.63, -0.6, 0.25, 0.06, 0.0);
    mix(&mut ch[CH_KEYS], 0.3, 0.3, 0.12, 0.0, 0.0);
    mix(&mut ch[CH_PAD], 0.21, -0.3, 0.12, 0.0, 0.0);
    mix(&mut ch[CH_BASS], 0.9, 0.0, 0.05, 0.0, 0.0);
    mix(&mut ch[CH_BELL], 0.34, 0.3, 0.25, 0.0, 0.0);
    mix(&mut ch[CH_KICK], 0.55, 0.0, 0.12, 0.0, 0.0);
    mix(&mut ch[CH_SNARE], 0.35, 0.1, 0.2, 0.0, 0.0);
    mix(&mut ch[CH_HAT], 0.3, 0.3, 0.1, 0.0, 0.0);
    mix(&mut ch[CH_PERC], 0.35, -0.3, 0.12, 0.0, 0.0);
    s.rev.fb = r.range(0.78, 0.83);
    s.rev.damp = 0.45;
    s.rev_wet = 0.2;
    s.dly.fb = 0.15;
    s.dly.lpk = 0.3;
    s.dly_rev = 0.15;
    s.r2mode = R2_OFF;
    s.fx.crackle = 0.0;
    s.fx.wind_to = 0.0;
    s.fx.rain_to = 0.0;
    s.m.gain = 1.9;
    s.m.target = 0.23;
}

// ---------------------------------------------------------------- the engine

/// A genre's music: the composer writing a bar at a time, the synthesiser playing it.
struct Engine {
    genre: usize,
    syn: Synth,
    song: Song,
    ev: Vec<(u64, Ev)>,
    evi: usize,
    t: u64,
    /// where the next bar starts, in samples
    pos: f64,
    /// the bar playing: start and length
    cur: (f64, f64),
    /// switched away from: it writes and plays nothing more, its sounds
    /// ringing out under the crossfade (two beats never run together)
    done: bool,
}

impl Engine {
    fn new(genre: usize, seed: u64) -> Engine {
        let genre = canon(genre);
        let mut syn = Synth::new(genre);
        let song = Song::new(genre, seed, &mut syn);
        Engine { genre, syn, song, ev: Vec::with_capacity(1024), evi: 0, t: 0, pos: 0.0, cur: (0.0, 1.0), done: false }
    }

    /// Fills `out` (interleaved stereo).
    fn render(&mut self, out: &mut [f32]) {
        let frames = out.len() / 2;
        let mut done = 0;
        while done < frames {
            if self.done {
                let o = &mut out[2 * done..];
                for c in o.chunks_mut(2 * BLOCK) {
                    self.syn.render(c);
                }
                self.t += (frames - done) as u64;
                return;
            }
            let next_bar = self.pos.ceil() as u64;
            if self.t >= next_bar {
                let start = self.pos;
                let len = self.song.bar(&mut self.syn, start, &mut self.ev);
                self.ev[self.evi..].sort_by_key(|e| e.0);
                self.cur = (start, len);
                self.pos += len;
                continue;
            }
            while self.evi < self.ev.len() && self.ev[self.evi].0 <= self.t {
                let e = self.ev[self.evi].1;
                self.syn.fire(e);
                self.evi += 1;
            }
            if self.evi >= 1024 {
                self.ev.drain(..self.evi);
                self.evi = 0;
            }
            let mut n = (frames - done).min(BLOCK).min((next_bar - self.t) as usize);
            if let Some(&(at, _)) = self.ev.get(self.evi) {
                n = n.min((at - self.t) as usize);
            }
            self.syn.render(&mut out[2 * done..2 * (done + n)]);
            self.t += n as u64;
            done += n;
        }
    }

    /// Where a genre switch can start: the next bar line, or the next beat
    /// when the bar line is more than 2.5 s away.
    fn switch_point(&self) -> u64 {
        let (start, len) = self.cur;
        let t = self.t as f64;
        let end = start + len;
        if end - t <= 2.5 * SR as f64 {
            return end.ceil() as u64;
        }
        let beat = len * self.song.beat_steps() as f64 / self.song.bar_steps() as f64;
        (start + ((t - start) / beat).ceil() * beat).ceil() as u64
    }
}

/// The music as heard: one engine, or two crossfading after a genre switch.
struct Mix {
    cur: Engine,
    next: Option<Engine>,
    fade: u32,
    at: Option<u64>,
    seed: u64,
    switches: u64,
    want: usize,
    tmp: Vec<f32>,
}

impl Mix {
    fn new(genre: usize, seed: u64) -> Mix {
        let cur = Engine::new(genre, seed);
        let want = cur.genre;
        Mix { cur, next: None, fade: 0, at: None, seed, switches: 0, want, tmp: vec![0.0; 2 * BLOCK] }
    }

    fn render(&mut self, out: &mut [f32]) {
        let _ftz = Ftz::on();
        let frames = out.len() / 2;
        let mut done = 0;
        while done < frames {
            let mut n = (frames - done).min(BLOCK);
            if self.next.is_none() {
                if canon(self.want) == self.cur.genre {
                    self.at = None;
                } else {
                    let at = match self.at {
                        Some(a) => a,
                        None => *self.at.insert(self.cur.switch_point()),
                    };
                    if self.cur.t >= at {
                        self.switches += 1;
                        let seed = self.seed.wrapping_add(self.switches.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                        self.next = Some(Engine::new(self.want, seed));
                        self.cur.done = true;
                        self.fade = 0;
                        self.at = None;
                    } else {
                        n = n.min((at - self.cur.t) as usize);
                    }
                }
            }
            let o = &mut out[2 * done..2 * (done + n)];
            self.cur.render(o);
            if let Some(nx) = &mut self.next {
                let t = &mut self.tmp[..2 * n];
                nx.render(t);
                for k in 0..n {
                    // smoothstep weights that sum to one: no louder than either side
                    let x = ((self.fade + k as u32) as f32 / FADE as f32).min(1.0);
                    let g = x * x * (3.0 - 2.0 * x);
                    o[2 * k] = o[2 * k] * (1.0 - g) + t[2 * k] * g;
                    o[2 * k + 1] = o[2 * k + 1] * (1.0 - g) + t[2 * k + 1] * g;
                }
                self.fade += n as u32;
                if self.fade >= FADE {
                    self.cur = self.next.take().expect("next");
                }
            }
            done += n;
        }
    }
}

/// Flush-to-zero and denormals-are-zero while it lives (decaying envelopes
/// and feedback loops otherwise crawl through denormals, many times slower),
/// then the caller's floating-point mode back.
struct Ftz(#[allow(dead_code)] u32);

impl Ftz {
    fn on() -> Ftz {
        #[cfg(target_arch = "x86_64")]
        {
            let mut old: u32 = 0;
            // SAFETY: stmxcsr / ldmxcsr read and write 4 bytes at a valid
            // u32; FTZ and DAZ (bits 15 and 6) only change how denormals round
            unsafe {
                std::arch::asm!("stmxcsr [{}]", in(reg) &mut old as *mut u32, options(nostack, preserves_flags));
                let new = old | 0x8040;
                std::arch::asm!("ldmxcsr [{}]", in(reg) &new as *const u32, options(nostack, preserves_flags));
            }
            Ftz(old)
        }
        #[cfg(not(target_arch = "x86_64"))]
        Ftz(0)
    }
}

impl Drop for Ftz {
    fn drop(&mut self) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above, restoring the value read then
        unsafe {
            std::arch::asm!("ldmxcsr [{}]", in(reg) &self.0 as *const u32, options(nostack, preserves_flags));
        }
    }
}

fn to_i16(x: f32) -> i16 {
    (x * 32767.0).round().clamp(-32768.0, 32767.0) as i16
}

// ---------------------------------------------------------------- the outside

/// Music playing through the speakers: made in the sound device's own
/// callback (SDL's real-time audio thread), or failing that on a thread of
/// its own writing to pacat.
pub struct Player {
    want: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    /// frames made so far (the tests watch it)
    #[cfg_attr(not(test), allow(dead_code))]
    frames: Arc<AtomicU64>,
    child: Option<Child>,
    thread: Option<JoinHandle<()>>,
    /// the sound device and what its callback plays from
    device: Option<(crate::sdl::AudioOut, Box<Live>)>,
}

/// What the device's callback plays: the mix (and its float buffer), and
/// what the app asks of it.
struct Live {
    mix: std::sync::Mutex<(Mix, Vec<f32>)>,
    want: Arc<AtomicUsize>,
    frames: Arc<AtomicU64>,
}

/// SDL's callback: the next `len` bytes of 16-bit stereo, made now.
unsafe extern "C" fn fill(user: *mut std::ffi::c_void, stream: *mut u8, len: std::ffi::c_int) {
    // SAFETY: `user` is the Live the Player keeps boxed until the device is
    // closed; SDL hands a buffer of `len` bytes, aligned for its samples
    let (live, out) = unsafe { (&*(user as *const Live), std::slice::from_raw_parts_mut(stream as *mut i16, len.max(0) as usize / 2)) };
    let mut g = live.mix.lock().unwrap_or_else(|e| e.into_inner());
    let (mix, buf) = &mut *g;
    buf.resize(out.len(), 0.0);
    mix.want = live.want.load(Ordering::Relaxed);
    mix.render(buf);
    for (o, &x) in out.iter_mut().zip(buf.iter()) {
        *o = to_i16(x);
    }
    live.frames.fetch_add(out.len() as u64 / 2, Ordering::Relaxed);
}

impl Player {
    /// Starts playing the music `genre` (an id: `mix_id`), seeded: through the
    /// sound device, else pacat. Error if neither plays.
    pub fn start(genre: usize, seed: u64) -> Result<Player, String> {
        Player::device(genre, seed).or_else(|e| {
            eprintln!("fleece: music through pacat ({e})");
            Player::spawn(
                &["pacat", "--raw", "--format=s16le", "--rate=48000", "--channels=2", "--latency-msec=200"],
                genre,
                seed,
            )
        })
    }

    /// Plays through SDL's default output, made in its callback (4096 frames,
    /// 85 ms, at a time). With PulseAudio's (PipeWire's) server SDL runs the
    /// callback on a real-time thread (round robin, priority 20 here).
    fn device(genre: usize, seed: u64) -> Result<Player, String> {
        let genre = canon(genre);
        let want = Arc::new(AtomicUsize::new(genre));
        let frames = Arc::new(AtomicU64::new(0));
        let live = Box::new(Live { mix: std::sync::Mutex::new((Mix::new(genre, seed), Vec::new())), want: want.clone(), frames: frames.clone() });
        // SAFETY: the Live is boxed (it never moves) and dropped only after the device
        let out = unsafe { crate::sdl::AudioOut::open(RATE, 4096, fill, &*live as *const Live as *mut std::ffi::c_void)? };
        eprintln!("fleece: music through {}", out.driver());
        Ok(Player { want, stop: Arc::new(AtomicBool::new(false)), frames, child: None, thread: None, device: Some((out, live)) })
    }

    /// The same, writing to `cmd`'s stdin.
    fn spawn(cmd: &[&str], genre: usize, seed: u64) -> Result<Player, String> {
        let mut child = Command::new(cmd[0])
            .args(&cmd[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("{}: {e}", cmd[0]))?;
        let mut pipe = child.stdin.take().expect("piped stdin");
        let genre = canon(genre);
        let want = Arc::new(AtomicUsize::new(genre));
        let stop = Arc::new(AtomicBool::new(false));
        let frames = Arc::new(AtomicU64::new(0));
        let (w, st, fr) = (want.clone(), stop.clone(), frames.clone());
        let thread = std::thread::Builder::new().name("music".into()).spawn(move || {
            let mut mix = Mix::new(genre, seed);
            let mut f = vec![0.0f32; 2 * CHUNK];
            let mut bytes = vec![0u8; 4 * CHUNK];
            // blocking writes pace the generator to the sound server
            while !st.load(Ordering::Relaxed) {
                mix.want = w.load(Ordering::Relaxed);
                mix.render(&mut f);
                for (o, &x) in bytes.chunks_exact_mut(2).zip(&f) {
                    o.copy_from_slice(&to_i16(x).to_le_bytes());
                }
                if pipe.write_all(&bytes).is_err() {
                    break;
                }
                fr.fetch_add(CHUNK as u64, Ordering::Relaxed);
            }
        });
        let thread = match thread {
            Ok(t) => t,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("music thread: {e}"));
            }
        };
        let mut p = Player { want, stop, frames, child: Some(child), thread: Some(thread), device: None };
        // a sound server that is not there shows as the player exiting at once
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_millis(300) {
            if let Some(c) = p.child.as_mut()
                && let Ok(Some(st)) = c.try_wait()
            {
                return Err(format!("{} exited ({st}): no PulseAudio / PipeWire server to play to", cmd[0]));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        Ok(p)
    }

    /// Switches the music (to an id: `mix_id`), at the next bar line, or beat
    /// when bars are long, with a crossfade.
    pub fn set_genre(&self, genre: usize) {
        self.want.store(canon(genre), Ordering::Relaxed);
    }

    pub fn genre(&self) -> usize {
        self.want.load(Ordering::Relaxed)
    }
}

impl Drop for Player {
    /// Stops the generator, kills and reaps the player (its pipe closing ends a
    /// blocked write), then joins the thread.
    fn drop(&mut self) {
        // the device first (no callback runs after), then what it played from
        if let Some((out, live)) = self.device.take() {
            drop(out);
            drop(live);
        }
        self.stop.store(true, Ordering::Relaxed);
        if let Some(c) = &mut self.child {
            let _ = c.kill();
            let _ = c.wait();
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Renders `secs` seconds of the music `genre` (an id) with `seed` to a 16-bit stereo 48 kHz WAV (deterministic for a given genre+seed).
pub fn render_wav(genre: usize, seed: u64, secs: f64, path: &str) -> Result<(), String> {
    let frames = (secs.max(0.0) * RATE as f64).round() as usize;
    let bytes = frames * 4;
    let mut data = Vec::with_capacity(44 + bytes);
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&(36 + bytes as u32).to_le_bytes());
    data.extend_from_slice(b"WAVEfmt ");
    data.extend_from_slice(&16u32.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes());
    data.extend_from_slice(&2u16.to_le_bytes());
    data.extend_from_slice(&RATE.to_le_bytes());
    data.extend_from_slice(&(RATE * 4).to_le_bytes());
    data.extend_from_slice(&4u16.to_le_bytes());
    data.extend_from_slice(&16u16.to_le_bytes());
    data.extend_from_slice(b"data");
    data.extend_from_slice(&(bytes as u32).to_le_bytes());
    let mut mix = Mix::new(genre, seed);
    let mut f = vec![0.0f32; 2 * CHUNK];
    let mut left = frames;
    while left > 0 {
        let n = left.min(CHUNK);
        mix.render(&mut f[..2 * n]);
        for &x in &f[..2 * n] {
            data.extend_from_slice(&to_i16(x).to_le_bytes());
        }
        left -= n;
    }
    std::fs::write(path, data).map_err(|e| format!("{path}: {e}"))
}

/// The pieces (start in seconds, tempo, tonic, mode) the first `secs` of a
/// genre and seed hold: the composer run alone, without the sound.
fn pieces(genre: usize, seed: u64, secs: f64) -> Vec<(f64, f64, i32, usize, &'static str)> {
    let mut e = Engine::new(genre, seed);
    let mut ev = Vec::new();
    let mut starts = vec![0.0];
    while e.pos < secs * SR as f64 {
        let n = e.song.log.len();
        let at = e.pos;
        e.pos += e.song.bar(&mut e.syn, e.pos, &mut ev);
        if e.song.log.len() > n {
            starts.push(at / SR as f64);
        }
        ev.clear();
    }
    starts.iter().zip(&e.song.log).map(|(&t, &(bpm, k, m, meter))| (t, bpm, k, m, meter)).collect()
}

/// What the composer wrote in a stretch of music, to compare versions by:
/// chord changes and distinct chords a minute, notes and drum hits a second,
/// and how alike bars 4 and 8 apart are (the mean Jaccard index of their
/// events, as (16th, instrument, pitch or drum); all events, then notes alone).
struct Stats {
    changes_per_min: f64,
    distinct_per_min: f64,
    notes_per_s: f64,
    hits_per_s: f64,
    sim: [f64; 4],
}

fn stats(genre: usize, seed: u64, secs: f64) -> Stats {
    let mut e = Engine::new(genre, seed);
    let mut ev = Vec::new();
    // per bar: piece, start (s), all events, notes alone
    let mut bars: Vec<(usize, f64, Vec<u32>, Vec<u32>)> = Vec::new();
    let mut chords: Vec<(f64, u16)> = Vec::new();
    let (mut notes, mut hits) = (0usize, 0usize);
    while e.pos < secs * SR as f64 {
        let t0 = e.pos;
        let len = e.song.bar(&mut e.syn, t0, &mut ev);
        let step = e.song.step_len();
        let (mut all, mut pitched) = (Vec::new(), Vec::new());
        for &(t, x) in &ev {
            let st = ((t as f64 - t0) / step).round().max(0.0) as u32;
            match x {
                Ev::Note { ch, note, .. } => {
                    let k = st << 16 | (ch as u32) << 8 | note as u32;
                    all.push(k);
                    pitched.push(k);
                    notes += 1;
                }
                Ev::Hit { ch, kind, .. } => {
                    all.push(st << 16 | (ch as u32 + 100) << 8 | kind as u32);
                    hits += 1;
                }
                Ev::Riser { .. } => {}
            }
        }
        for v in [&mut all, &mut pitched] {
            v.sort_unstable();
            v.dedup();
        }
        for m in e.song.bar_chords() {
            chords.push((t0 / SR as f64, m));
        }
        bars.push((e.song.log.len(), t0 / SR as f64, all, pitched));
        e.pos += len;
        ev.clear();
    }
    let jaccard = |a: &[u32], b: &[u32]| {
        let common = a.iter().filter(|x| b.binary_search(x).is_ok()).count();
        common as f64 / (a.len() + b.len() - common).max(1) as f64
    };
    let sim = |lag: usize, notes_only: bool| {
        let v: Vec<f64> = (0..bars.len().saturating_sub(lag))
            .filter(|&i| bars[i].0 == bars[i + lag].0)
            .map(|i| if notes_only { (&bars[i].3, &bars[i + lag].3) } else { (&bars[i].2, &bars[i + lag].2) })
            .filter(|(a, b)| !a.is_empty() || !b.is_empty())
            .map(|(a, b)| jaccard(a, b))
            .collect();
        v.iter().sum::<f64>() / v.len().max(1) as f64
    };
    let minutes = secs / 60.0;
    let changes = chords.windows(2).filter(|w| w[0].1 != w[1].1).count();
    let full = (secs / 60.0).floor().max(1.0) as usize;
    let distinct: usize = (0..full)
        .map(|m| {
            let mut v: Vec<u16> =
                chords.iter().filter(|c| c.0 >= m as f64 * 60.0 && c.0 < (m + 1) as f64 * 60.0).map(|c| c.1).collect();
            v.sort_unstable();
            v.dedup();
            v.len()
        })
        .sum();
    Stats {
        changes_per_min: changes as f64 / minutes,
        distinct_per_min: distinct as f64 / full as f64,
        notes_per_s: notes as f64 / secs,
        hits_per_s: hits as f64 / secs,
        sim: [sim(4, false), sim(8, false), sim(4, true), sim(8, true)],
    }
}

/// `--musicwav GENRE SECONDS OUT.wav`: renders and says how long it took.
pub fn cli(args: &[String], seed: u64) {
    let genre = args.first().and_then(|g| find(g));
    let secs = args.get(1).and_then(|s| s.parse::<f64>().ok());
    let (Some(genre), Some(secs), Some(path)) = (genre, secs, args.get(2)) else {
        eprintln!("fleece: --musicwav GENRE SECONDS OUT.wav, GENRE one of {} or two (POLKA+TECHNO: the first's music, the second's drums)", GENRES.join(", "));
        std::process::exit(2);
    };
    // the recorded instruments before the first note: a seed's music the same every time
    samples::load_now();
    let t0 = Instant::now();
    if let Err(e) = render_wav(genre, seed, secs, path) {
        eprintln!("fleece: {e}");
        std::process::exit(1);
    }
    let took = t0.elapsed().as_secs_f64();
    let pieces = pieces(genre, seed, secs);
    const KEYS: [&str; 12] = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
    const MODE: [&str; 7] = ["ionian", "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "harmonic minor"];
    let p: Vec<String> = pieces
        .iter()
        .map(|&(t, bpm, k, m, meter)| format!("{t:.0} s: {bpm:.1} bpm {meter} {} {}", KEYS[k as usize % 12], MODE[m]))
        .collect();
    println!(
        "{path}: {secs:.1} s of {} (seed {seed}) in {took:.2} s, {:.0}x real time; pieces: {}",
        name(genre),
        secs / took.max(1e-9),
        p.join(", ")
    );
    let st = stats(genre, seed, secs);
    println!(
        "  composition: {:.1} chord changes and {:.1} distinct chords a minute, {:.1} notes and {:.1} drum hits a second; \
         bars 4 / 8 apart alike {:.2} / {:.2} (notes alone {:.2} / {:.2})",
        st.changes_per_min,
        st.distinct_per_min,
        st.notes_per_s,
        st.hits_per_s,
        st.sim[0],
        st.sim[1],
        st.sim[2],
        st.sim[3]
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(genre: usize, seed: u64, secs: f32) -> Vec<f32> {
        let mut m = Mix::new(genre, seed);
        let mut v = vec![0.0; 2 * (secs * SR) as usize];
        m.render(&mut v);
        v
    }

    #[test]
    fn player_starts_and_drops_cleanly() {
        let p = Player::spawn(&["sh", "-c", "exec cat > /dev/null"], 1, 7).expect("player");
        std::thread::sleep(Duration::from_secs(1));
        p.set_genre(2);
        assert_eq!(p.genre(), 2);
        let pid = p.child.as_ref().expect("child").id();
        let frames = p.frames.clone();
        assert!(frames.load(Ordering::Relaxed) > RATE as u64 / 2, "it generated too little");
        let t0 = Instant::now();
        drop(p);
        assert!(t0.elapsed() < Duration::from_secs(2), "drop took {:?}", t0.elapsed());
        // the thread is gone (its clone of the counter with it), the child reaped
        assert_eq!(Arc::strong_count(&frames), 1, "the thread still runs");
        assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists(), "child {pid} not reaped");
    }

    #[test]
    fn player_fails_when_the_server_is_not_there() {
        assert!(Player::spawn(&["sh", "-c", "exit 1"], 0, 1).is_err());
        assert!(Player::spawn(&["/nonexistent/pacat"], 0, 1).is_err());
    }

    #[test]
    fn chord_pools_parse() {
        let tav = [TAV_ION, TAV_MIX, TAV_DOR, TAV_AEO].concat();
        for pool in [AMB_MAJ, AMB_MIN, TECH_MIN, SW_MAJ, SW_MIN, LOFI_MAJ, LOFI_MIN].into_iter().chain(tav) {
            for plan in pool {
                assert_eq!(plan.split_whitespace().count(), 8, "{plan}");
                for tok in plan.split_whitespace().flat_map(|t| t.split('+')) {
                    assert!(parse_harm(tok).is_some(), "{tok} in {plan}");
                }
            }
        }
        for pool in [POLKA_A, POLKA_B, POLKA_MINOR] {
            for plan in pool {
                assert_eq!(plan.split_whitespace().count(), 8, "{plan}");
                for tok in plan.split_whitespace() {
                    assert!(parse_harm(tok).is_some(), "{tok} in {plan}");
                }
            }
        }
        let mariachi = [MARIACHI_MAJ_A, MARIACHI_MAJ_B, MARIACHI_MAJ_C, MARIACHI_MIN_A, MARIACHI_MIN_B, MARIACHI_MIN_C];
        for (pool, bars) in mariachi.iter().map(|p| (p, 8)).chain([(&MARIACHI_FANFARE, 4), (&MARIACHI_FANFARE_MIN, 4)]) {
            for plan in pool.iter() {
                assert_eq!(plan.split_whitespace().count(), bars, "{plan}");
                for tok in plan.split_whitespace().flat_map(|t| t.split('+')) {
                    assert!(parse_harm(tok).is_some(), "{tok} in {plan}");
                }
            }
        }
        assert_eq!(parse_harm("V7/V"), Some(Harm { root: 2, q: Q_DOM7, inv: 0 }));
        assert_eq!(parse_harm("bVImaj7^1"), Some(Harm { root: 8, q: Q_MAJ7, inv: 1 }));
        assert_eq!(parse_harm("ii7"), Some(Harm { root: 2, q: Q_MIN7, inv: 0 }));
        assert!(parse_harm("X7").is_none() && parse_harm("I^5").is_none());
    }

    #[test]
    fn deterministic_per_seed() {
        assert_eq!(render(3, 5, 2.0), render(3, 5, 2.0));
        assert_ne!(render(3, 5, 2.0), render(3, 6, 2.0));
        assert_eq!(render(POLKA, 5, 2.0), render(POLKA, 5, 2.0));
        assert_eq!(render(TAVERN, 5, 2.0), render(TAVERN, 5, 2.0));
        assert_eq!(render(SKA, 5, 2.0), render(SKA, 5, 2.0));
        assert_ne!(render(SKA, 5, 2.0), render(SKA, 6, 2.0));
        assert_eq!(render(MARIACHI, 5, 2.0), render(MARIACHI, 5, 2.0));
    }

    #[test]
    fn genres_found_by_name() {
        assert_eq!(find("lofi"), Some(LOFI));
        assert_eq!(find("Lo-Fi"), Some(LOFI));
        assert_eq!(find("polka+techno"), Some(mix_id(POLKA, Some(TECHNO))));
        assert_eq!(find("techno polka"), Some(mix_id(TECHNO, Some(POLKA))));
        assert_eq!(find("lofi-tavern"), Some(mix_id(LOFI, Some(TAVERN))));
        assert_eq!(find("3"), Some(LOFI));
        assert_eq!(find("disco"), None);
        assert_eq!(find("polka+polka"), None);
        assert_eq!(name(mix_id(TAVERN, Some(LOFI))), "TAVERN+LO-FI");
        assert_eq!(mix_id(POLKA, Some(POLKA)), POLKA);
        for id in 0..IDS {
            assert_eq!(canon(id), mix_id(parts(id).0, parts(id).1));
            assert_ne!(parts(id).1, Some(parts(id).0));
        }
    }

    /// Every mix's bars: the drummer's beat in the writer's meter, every hit
    /// inside its bar, drums in every full section, the effects channel open;
    /// techno's kick on the one, polka+techno's clap on the second beat; the
    /// tavern writer's frame drum (its second stroke) gone under another
    /// drummer but for the piece's last bar (under a drummer whose own kick
    /// has a second stroke, mariachi's toe taps, it cannot be told apart).
    #[test]
    fn mixes_drum_in_the_writers_meter() {
        for p in 0..NG {
            for g in (0..NG).filter(|&g| g != p) {
                let id = mix_id(p, Some(g));
                for seed in [1u64, 2] {
                    let mut e = Engine::new(id, seed);
                    assert!(e.syn.ch[CH_FX].gain > 0.0, "{}: the riser's and the crackle's channel is shut", name(id));
                    let mut ev: Vec<(u64, Ev)> = Vec::new();
                    let (mut mains, mut claps) = (0, 0);
                    while e.pos < 300.0 * SR as f64 {
                        let t0 = e.pos;
                        let len = e.song.bar(&mut e.syn, t0, &mut ev);
                        e.pos += len;
                        let sec = e.song.plan[e.song.sec];
                        let ending = sec.kind == OUTRO && e.song.sec_bar == sec.bars;
                        let bs = e.song.bar_steps();
                        let step = len / bs as f64;
                        let (mut hits, mut one) = (0, false);
                        for &(t, x) in &ev {
                            if let Ev::Hit { ch, kind, .. } = x {
                                let at = (t as f64 - t0) / step;
                                assert!((-0.01..bs as f64).contains(&at), "{} seed {seed}: a hit at {at} of {bs}", name(id));
                                assert!(
                                    ending || !(p == TAVERN && !matches!(g, TAVERN | MARIACHI) && ch as usize == CH_KICK && kind == 1),
                                    "{}: the writer's frame drum",
                                    name(id)
                                );
                                hits += 1;
                                one |= ch as usize == CH_KICK && at.abs() < 0.01;
                                if ch as usize == CH_SNARE && kind == 0 && p == POLKA && g == TECHNO && (at - 4.0).abs() < 0.01 {
                                    claps += 1;
                                }
                            }
                        }
                        if sec.kind == MAIN {
                            mains += 1;
                            assert!(hits > 0, "{} seed {seed}: no drums", name(id));
                            assert!(g != TECHNO || one, "{} seed {seed}: no kick on the one", name(id));
                        }
                        ev.clear();
                    }
                    assert!(mains > 20, "{} seed {seed}: {mains} main bars", name(id));
                    assert!(p != POLKA || g != TECHNO || claps > mains / 2, "{} seed {seed}: {claps} claps in {mains} bars", name(id));
                }
            }
        }
    }

    /// Genres and mixes flipped every 3-7 s for four minutes, from a few
    /// seeds: the engines built for switches (with derived seeds) never
    /// panic, never produce a non-finite sample or one over the ceiling,
    /// never go silent for half a second (a mix's sparse beat can rest for
    /// a tenth between hits).
    #[test]
    #[ignore = "renders 16 minutes of audio: cargo test --bin fleece music -- --ignored"]
    fn genre_switching_stress() {
        for base in 1..=4u64 {
            let mut m = Mix::new(base as usize, base);
            let mut r = Rng::new(base * 77);
            let mut buf = vec![0.0f32; 2 * 4800];
            let mut next = r.range(3.0, 7.0);
            let mut quiet = 0;
            for k in 1..=2400 {
                m.render(&mut buf);
                assert!(buf.iter().all(|x| x.is_finite() && x.abs() <= 0.945), "seed {base}, {} s", k / 10);
                let rms = (buf.iter().map(|x| x * x).sum::<f32>() / buf.len() as f32).sqrt();
                quiet = if rms > 1e-3 { 0 } else { quiet + 1 };
                let (song, sec) = (&m.cur.song, m.cur.song.plan[m.cur.song.sec]);
                assert!(
                    k < 10 || quiet < 5,
                    "seed {base}: silent at {} s ({}, section {} kind {} bar {} mask {:b})",
                    k / 10,
                    name(m.cur.genre),
                    song.sec,
                    sec.kind,
                    song.sec_bar,
                    sec.mask
                );
                let t = k as f32 / 10.0;
                if t > next {
                    m.want = r.below(IDS);
                    next = t + r.range(3.0, 7.0);
                }
            }
        }
    }

    /// The composer alone through half an hour of many seeds (dozens of
    /// pieces each; overflow checks on in this profile), and the first 20 s
    /// of each as sound.
    #[test]
    #[ignore = "a sweep: cargo test --bin fleece music -- --ignored"]
    fn composer_sweep() {
        for genre in (0..IDS).filter(|&id| canon(id) == id) {
            for seed in 0..if genre < NG { 50 } else { 8u64 } {
                let mut e = Engine::new(genre, seed);
                let mut ev = Vec::new();
                while e.pos < 1800.0 * SR as f64 {
                    e.pos += e.song.bar(&mut e.syn, e.pos, &mut ev);
                    ev.clear();
                }
                assert!(e.song.log.len() >= 5, "genre {genre} seed {seed}: {} pieces", e.song.log.len());
                let v = render(genre, seed, 20.0);
                assert!(v.iter().all(|x| x.is_finite() && x.abs() <= 0.945), "genre {genre} seed {seed}");
            }
        }
    }

    #[test]
    fn genre_switch_crossfades_without_a_gap() {
        let mut m = Mix::new(TECHNO, 3);
        let mut a = vec![0.0; 2 * 6 * RATE as usize];
        m.render(&mut a);
        m.want = SYNTHWAVE;
        let mut b = vec![0.0; 2 * 8 * RATE as usize];
        m.render(&mut b);
        assert_eq!(m.cur.genre, SYNTHWAVE);
        assert!(m.next.is_none());
        // never silent for 100 ms, never over the ceiling
        for w in b.chunks(2 * 4800) {
            let rms = (w.iter().map(|x| x * x).sum::<f32>() / w.len() as f32).sqrt();
            assert!(rms > 1e-3, "a gap at the switch");
            assert!(w.iter().all(|x| x.is_finite() && x.abs() < 0.95));
        }
    }

    #[test]
    fn ska_pools_parse() {
        let strains = [
            SKA_MAJ_A,
            SKA_MAJ_B,
            SKA_MIX_A,
            SKA_MIX_B,
            SKA_MIN_A,
            SKA_MIN_B,
            SKA_TW_MIN_A,
            SKA_TW_MIN_B,
            SKA_TW_MIN_C,
            SKA_TW_MAJ_A,
            SKA_TW_MAJ_B,
            SKA_TW_MAJ_C,
        ];
        let vamps = [SKA_VAMP_MAJ, SKA_VAMP_MIX, SKA_VAMP_MIN, SKA_TW_VAMP_MIN, SKA_TW_VAMP_MAJ];
        for (pools, bars) in [(&strains[..], 8), (&vamps[..], 4)] {
            for plan in pools.iter().flat_map(|p| p.iter()) {
                assert_eq!(plan.split_whitespace().count(), bars, "{plan}");
                for tok in plan.split_whitespace().flat_map(|t| t.split('+')) {
                    assert!(parse_harm(tok).is_some(), "{tok} in {plan}");
                }
            }
        }
    }

    /// Ska alone over a few seeds draws all three styles, each in its tempo.
    #[test]
    fn ska_plays_its_three_styles() {
        let mut seen = [0; 3];
        for seed in 0..8u64 {
            let mut e = Engine::new(SKA, seed);
            let mut ev = Vec::new();
            while e.pos < 900.0 * SR as f64 {
                e.pos += e.song.bar(&mut e.syn, e.pos, &mut ev);
                ev.clear();
            }
            for &(bpm, _, _, meter) in &e.song.log {
                let (k, lo, hi) = match meter {
                    "4/4 first wave" => (0, 110.0, 130.0),
                    "4/4 2-tone" => (1, 132.0, 158.0),
                    "4/4 third wave" => (2, 135.0, 148.0),
                    _ => panic!("seed {seed}: a ska piece in {meter}"),
                };
                assert!((lo..=hi).contains(&bpm), "seed {seed}: {meter} at {bpm} bpm");
                seen[k] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n >= 3), "styles drawn: {seen:?}");
    }

    /// Ska alone, a few seeds, a few pieces each: every event (the horns'
    /// shuffled off-beats, scoops and falls, the drums' rolls) starts inside
    /// its bar (the engine fires a bar's events in order), every note a
    /// sensible pitch.
    #[test]
    fn ska_bars_hold_their_events() {
        for seed in 0..6u64 {
            let mut e = Engine::new(SKA, seed);
            let mut ev: Vec<(u64, Ev)> = Vec::new();
            while e.pos < 400.0 * SR as f64 {
                let t0 = e.pos;
                let len = e.song.bar(&mut e.syn, t0, &mut ev);
                e.pos += len;
                for &(t, x) in &ev {
                    let at = t as f64 - t0;
                    assert!((-1.0..len).contains(&at), "seed {seed}: an event {at} samples into a bar of {len}");
                    if let Ev::Note { note, .. } = x {
                        assert!((24.0..=100.0).contains(&note), "seed {seed}: note {note}");
                    }
                }
                ev.clear();
            }
            assert!(e.song.log.len() >= 3, "seed {seed}: {} pieces in 400 s", e.song.log.len());
        }
    }
}

#[cfg(test)]
mod device_tests {
    use super::*;

    /// The sound device path, against SDL's disk driver (no sound): it makes
    /// music in the callback and closes cleanly.
    #[test]
    #[ignore]
    fn device_plays_to_disk() {
        let path = format!("{}/.cache/fleece-work/music/disk.raw", std::env::var("HOME").unwrap());
        // SAFETY: set before SDL starts, on the test's own thread
        unsafe {
            std::env::set_var("SDL_AUDIODRIVER", "disk");
            std::env::set_var("SDL_DISKAUDIOFILE", &path);
        }
        let p = Player::device(1, 3).expect("device");
        std::thread::sleep(Duration::from_millis(1500));
        let frames = p.frames.load(Ordering::Relaxed);
        drop(p);
        let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let _ = std::fs::remove_file(&path);
        assert!(frames > RATE as u64, "made {frames} frames");
        assert!(bytes > 4 * RATE as u64, "wrote {bytes} bytes");
    }
}

/// Recorded instruments as the synthesiser plays them, from a bank of the
/// tests' own (never the installed one: tests run side by side).
#[cfg(test)]
mod sample_tests {
    use super::*;
    use crate::samples::testdata;

    /// `secs` of voice `v` (started), into a fresh buffer.
    fn run(v: &mut Voice, b: &Bank, secs: f32) -> Vec<[f32; 2]> {
        let mut out = vec![[0.0f32; 2]; (secs * SR) as usize];
        for c in out.chunks_mut(BLOCK) {
            v.render(1.0, c, Some(b));
        }
        out
    }

    fn rms(x: &[[f32; 2]]) -> f32 {
        (x.iter().map(|s| s[0] * s[0] + s[1] * s[1]).sum::<f32>() / (2 * x.len()) as f32).sqrt()
    }

    fn hz(note: f32) -> f64 {
        440.0 * ((note as f64 - 69.0) / 12.0).exp2()
    }

    #[test]
    fn sampled_notes_play_in_tune_and_release() {
        let d = testdata::synthetic("voice");
        let b = samples::load(&d).expect("load");
        let _ = std::fs::remove_dir_all(&d);
        let p = Patch { samp: b.multi("sine", "sus").unwrap(), gain: 1.0, ..SAMPLED };
        // at a root, transposed up and down from one, between roots
        for note in [69.0f32, 74.0, 66.0, 60.0, 78.5] {
            let mut v = VOICE;
            v.start(&p, note, 0.8, (2.0 * SR) as i64, false, 1, Some(&b), 0);
            assert!(v.sm.n > 0, "note {note}: not sampled");
            let out = run(&mut v, &b, 0.5);
            let l: Vec<f32> = out[4800..].iter().map(|x| x[0]).collect();
            let got = testdata::freq(&l);
            assert!((got / hz(note) - 1.0).abs() < 0.002, "note {note}: {got:.2} Hz, want {:.2}", hz(note));
            // the reference at velocity 0.8, scaled by CAL
            let want = 0.1 * 0.8 * samples::CAL;
            assert!((rms(&out[4800..]) / want - 1.0).abs() < 0.05, "note {note}: RMS {}", rms(&out[4800..]));
        }
        // a slide glides the pitch over the same sample, no new attack
        let mut v = VOICE;
        let p = Patch { glide: 0.02, ..p };
        v.start(&p, 69.0, 0.8, (2.0 * SR) as i64, false, 1, Some(&b), 0);
        run(&mut v, &b, 0.2);
        let z = v.sm;
        v.start(&p, 74.0, 0.8, (2.0 * SR) as i64, true, 1, Some(&b), 0);
        let out = run(&mut v, &b, 0.5);
        let l: Vec<f32> = out[12000..].iter().map(|x| x[0]).collect();
        assert!((testdata::freq(&l) / hz(74.0) - 1.0).abs() < 0.002, "slid to {:.2}", testdata::freq(&l));
        assert_eq!(v.sm.n, z.n);
        // a 0.3 s gate: the release (the instrument's own, 0.1 s to -60 dB),
        // then the voice is free
        let mut v = VOICE;
        let p = Patch { samp: b.multi("sine", "sus").unwrap(), gain: 1.0, ..SAMPLED };
        v.start(&p, 69.0, 0.8, (0.3 * SR) as i64, false, 1, Some(&b), 0);
        let out = run(&mut v, &b, 0.6);
        assert!(rms(&out[12000..14000]) > 0.3, "held");
        assert!(rms(&out[20000..]) < 1e-3 && !v.on, "released and free");
        // a sample that runs out with the gate open (the staccato's 0.2 s): it
        // fades and the voice ends
        let mut v = VOICE;
        let p = Patch { samp: b.multi("sine", "stac").unwrap(), ..SAMPLED };
        v.start(&p, 69.0, 0.8, (2.0 * SR) as i64, false, 1, Some(&b), 0);
        let out = run(&mut v, &b, 0.5);
        assert!(!v.on && out[9700..].iter().all(|x| x[0] == 0.0));
        // no bank, or an id it lacks: the synthesiser plays the patch
        let mut v = VOICE;
        v.start(&p, 69.0, 0.8, 4800, false, 1, None, 0);
        assert_eq!(v.sm.n, 0);
        v.start(&Patch { samp: 999, ..p }, 69.0, 0.8, 4800, false, 1, Some(&b), 0);
        assert_eq!(v.sm.n, 0);
    }

    #[test]
    fn sampled_drums_play_once_and_choke() {
        let d = testdata::dir("drums");
        let path = d.join(samples::SF2);
        std::fs::write(&path, testdata::tiny_sf2()).unwrap();
        let b = samples::load(&d).expect("load");
        let _ = std::fs::remove_dir_all(&d);
        let (closed, open) = (b.drum("Standard", 42).unwrap(), b.drum("Standard", 46).unwrap());
        let hats = vec![DrumPatch { samp: closed, ..DRUM }, DrumPatch { samp: open, ..DRUM }];
        let render = |c: &mut Channel, secs: f32| {
            let mut v = Vec::new();
            for _ in 0..(secs * SR) as usize / BLOCK {
                c.render(BLOCK, Some(&b));
                v.extend_from_slice(&c.buf);
            }
            v
        };
        // an open hat alone rings out, once through (0.47 s: key 46 from root 69)
        let mut c = Channel::new(0, 4);
        c.drums = hats.clone();
        c.hit(true, HAT_O, 0.8, 1, Some(&b));
        let out = render(&mut c, 0.4);
        assert!(rms(&out) > 0.05 && c.dv.iter().any(|d| d.on));
        let out = render(&mut c, 0.2);
        assert!(!c.dv.iter().any(|d| d.on) && out[4800..].iter().all(|x| x[0] == 0.0));
        // a closed hat on the hat channel chokes it; elsewhere, its
        // exclusive class does
        for hats_ch in [true, false] {
            let mut c = Channel::new(0, 4);
            c.drums = hats.clone();
            c.hit(hats_ch, HAT_O, 0.8, 1, Some(&b));
            render(&mut c, 0.05);
            c.hit(hats_ch, HAT_C, 0.8, 1, Some(&b));
            render(&mut c, 0.1);
            let open_on = c.dv.iter().filter(|d| d.on && d.sm.key == 46.0).count();
            assert_eq!(open_on, 0, "hat channel {hats_ch}: the open hat still rings");
        }
        // round robins cycle in order, each kind its own (a shaker's
        // strokes down, up), whatever else the channel plays between
        let d = testdata::held("rr");
        let b2 = samples::load(&d).expect("load");
        let _ = std::fs::remove_dir_all(&d);
        let mut c = Channel::new(0, 4);
        let shake = DrumPatch { samp: b2.multi("shake", "hit").unwrap(), ..DRUM };
        c.drums = vec![DrumPatch { samp: b2.multi("held", "sus").unwrap(), ..DRUM }, shake];
        let mut seen = Vec::new();
        for k in 0..7 {
            c.hit(false, (k % 2 == 0) as u8, 0.8, 1, Some(&b2));
            c.dv.iter_mut().for_each(|d| d.on = false);
            if k % 2 == 0 {
                // the zone the shaker played: its first frames' sign changes tell
                let mut v = vec![[0.0f32; 2]; 2400];
                let mut dr = DRUMV;
                dr.start(&shake, 1, 0.8, 1, Some(&b2), seen.len() as u32);
                dr.render(&mut v[..], Some(&b2));
                seen.push(v.windows(2).filter(|w| (w[0][0] > 0.0) != (w[1][0] > 0.0)).count());
            }
        }
        assert_eq!((c.rr[0], c.rr[1]), (3, 4));
        assert!(seen[0] < seen[1] && seen[1] < seen[2] && seen[3] == seen[0], "{seen:?}");
        // without the bank, the synthesiser's (silent here: no tone, no noise)
        let mut c = Channel::new(0, 4);
        c.drums = hats;
        c.hit(true, HAT_O, 0.8, 1, None);
        assert!(c.dv.iter().all(|d| d.sm.n == 0));
    }

    /// Where `samples::CAL` and `CAL_DRUM` come from: the synthesiser's plain
    /// saw against a sine at the samples' reference level; a dozen of its
    /// drums against MuseScore General's Standard kit's of the same kinds
    /// (RMS over the first second), when the SoundFont is there.
    #[test]
    fn calibration() {
        let d = testdata::synthetic("cal");
        let b = samples::load(&d).expect("load");
        let _ = std::fs::remove_dir_all(&d);
        let saw = loudness_in(&PATCH, 60.0, None);
        let sampled = loudness_in(&Patch { samp: b.multi("sine", "sus").unwrap(), ..SAMPLED }, 69.0, Some(&b));
        let cal = saw / (sampled / samples::CAL);
        eprintln!(
            "saw at gain 1, velocity 0.8: RMS {saw:.4}; the reference sampled: {:.4}; CAL {cal:.3}",
            sampled / samples::CAL
        );
        assert!((cal / samples::CAL - 1.0).abs() < 0.01, "CAL measures {cal}");
        let first_second = |p: &DrumPatch, bank: Option<&Bank>| {
            let mut dr = DRUMV;
            dr.start(p, 0, 0.8, 7, bank, 0);
            let mut buf = vec![[0.0f32; 2]; SR as usize];
            for c in buf.chunks_mut(BLOCK) {
                if dr.on {
                    dr.render(c, bank);
                }
            }
            20.0 * rms(&buf).log10()
        };
        // a recorded hit plays at the reference times CAL_DRUM at velocity 0.8
        let hit = first_second(&DrumPatch { samp: b.multi("hit", "one").unwrap(), ..DRUM }, Some(&b));
        let want = 20.0 * (0.08 * samples::CAL_DRUM).log10();
        assert!((hit - want).abs() < 0.2, "a sampled hit at {hit:.1} dBFS, not {want:.1}");
        // the synthesiser's drums against MuseScore General's Standard kit's
        let path = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(".local/share/fleece/samples")
            .join(samples::SF2);
        let Ok(kit) = testdata::sf2_bank(&path, &[], &["Standard"]) else {
            eprintln!("no {}: the drums' calibration skipped", path.display());
            return;
        };
        let mut r = Rng::new(5);
        let pairs: Vec<(&str, DrumPatch, u8)> = vec![
            ("ska kick", ska_kit_kick(&mut r)[0], 36),
            ("909 kick", kit_kick_909(&mut r)[0], 36),
            ("ska snare", ska_kit_snare(&mut r)[0], 38),
            ("punk snare", ska_kit_punk_snare(&mut r)[0], 38),
            ("hand clap", kit_hand_clap(&mut r)[0], 39),
            ("closed hat", ska_kit_hats(&mut r)[0], 42),
            ("open hat", ska_kit_hats(&mut r)[1], 46),
            ("shaker", kit_shaker(&mut r)[0], 82),
            ("ride", kit_ride(&mut r)[0], 51),
            ("crash", crash(&mut r)[0], 49),
            ("tom", ska_kit_toms(&mut r)[1], 45),
            ("conga", kit_perc_hand(&mut r)[1], 64),
        ];
        let mut sum = 0.0;
        for (name, p, key) in &pairs {
            let id = kit.drum("Standard", *key).expect("key");
            let (syn, smp) = (
                first_second(&DrumPatch { gain: 1.0, ..*p }, None),
                first_second(&DrumPatch { samp: id, ..DRUM }, Some(&kit)),
            );
            eprintln!("  {name:10} the synthesiser's {syn:6.1} dBFS, key {key}'s {smp:6.1}");
            sum += syn - smp;
        }
        let off = sum / pairs.len() as f32;
        let cal = samples::CAL_DRUM * 10f32.powf(off / 20.0);
        eprintln!("drums: the kit {off:+.1} dB from the synthesiser's on average: CAL_DRUM {cal:.2}");
        assert!(off.abs() < 0.3, "CAL_DRUM measures {cal}");
    }
}
