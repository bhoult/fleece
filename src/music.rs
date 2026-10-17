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
//! piece, usually in a neighbouring key.
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

/// Genre names, shown on screen in capitals (the app's font has only A-Z, 0-9 and " -.:/!+()[],%").
pub const GENRES: &[&str] = &["AMBIENT", "TECHNO", "SYNTHWAVE", "LO-FI", "POLKA", "TAVERN"];
const AMBIENT: usize = 0;
const TECHNO: usize = 1;
const SYNTHWAVE: usize = 2;
const LOFI: usize = 3;
const POLKA: usize = 4;
const TAVERN: usize = 5;

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
};

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
};

impl Voice {
    /// A note: `len` samples of gate. A slide into a voice whose gate is still
    /// open glides there without a new attack (the 303's slide).
    fn start(&mut self, p: &Patch, note: f32, vel: f32, len: i64, slide: bool, seed: u32) {
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
                if q.ks > 0.0 {
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
            // the output carried over the jump to the new source (no click)
            self.fix = p.ks > 0.0 || kind(p) != kind(&self.p);
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
        self.amp.set(p.a, p.d, p.s, p.r);
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

    /// Adds the voice into `buf`; `cut` scales the cutoff (the channel's drift).
    fn render(&mut self, cut: f32, buf: &mut [[f32; 2]]) {
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
        let gain = self.vel * p.gain * (1.0 - p.trem * (0.5 + 0.5 * lfo)) * (1.0 + 0.3 * self.accent);
        if self.amp.level == 0.0 {
            self.g = gain;
        }
        let stereo = p.unison > 1 && p.width > 0.0;
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
        if p.ks > 0.0 {
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
};

impl Drum {
    fn start(&mut self, p: &DrumPatch, kind: u8, vel: f32, seed: u32) {
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
            ta: if p.tone > 0.0 { 1.0 } else { 0.0 },
            tc: t60(p.tdec),
            na: if p.noise > 0.0 { 1.0 } else { 0.0 },
            nc: t60(p.ndec),
            burst: 1.0,
            bc: t60(0.02),
            natt: ((p.natt * SR) as u32).max(1),
            mph,
            noise: n,
            ..DRUMV
        };
        self.f.set(p.ncut, p.nres);
        self.f2.set(p.ncut, p.nres);
    }

    /// Fast fade (an open hat closed by a closed one).
    fn choke(&mut self) {
        self.tc = t60(0.012);
        self.nc = t60(0.012);
    }

    fn render(&mut self, buf: &mut [[f32; 2]]) {
        let p = self.p;
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
        self.time += (self.target - self.time) * 0.0001;
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
        let want = (self.target / (self.ms.sqrt() + 1e-6)).sqrt().clamp(0.5, 2.0);
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
        }
    }

    /// Renders the block into `buf`; whether anything sounded. (The swell
    /// keeps time whether or not the channel sounds.)
    fn render(&mut self, n: usize) -> bool {
        self.swell_s += (self.swell - self.swell_s) * 0.02;
        self.swell_ph = wrap(self.swell_ph + self.swell_rate * n as f32 / SR);
        let Channel { voices, dv, buf, .. } = self;
        let buf = &mut buf[..n];
        buf.fill([0.0; 2]);
        self.cut_s += (self.cut - self.cut_s) * 0.02;
        let mut any = false;
        for v in voices.iter_mut().filter(|v| v.on) {
            v.render(self.cut_s, buf);
            any = true;
        }
        for d in dv.iter_mut().filter(|d| d.on) {
            d.render(buf);
            any = true;
        }
        any
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
        let size = match genre {
            AMBIENT => 1.35,
            TECHNO => 1.0,
            SYNTHWAVE => 1.1,
            LOFI => 0.8,
            TAVERN => 0.65,
            _ => 0.9,
        };
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
            rev2: Reverb::new(if genre == TECHNO { 1.3 } else { 0.7 }),
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
        match e {
            Ev::Hit { ch, kind, vel } => {
                let ch = ch as usize;
                if ch == CH_KICK {
                    self.sc_att = (0.003 * SR) as u32;
                }
                if ch == CH_SNARE && kind == 0 {
                    self.gate_left = self.gate_hold;
                }
                let c = &mut self.ch[ch];
                if c.drums.is_empty() || c.dv.is_empty() {
                    return;
                }
                if ch == CH_HAT && kind == HAT_C {
                    for d in c.dv.iter_mut().filter(|d| d.on && d.kind == HAT_O) {
                        d.choke();
                    }
                }
                let p = c.drums[(kind as usize).min(c.drums.len() - 1)];
                let i =
                    c.dv.iter()
                        .position(|d| !d.on)
                        .unwrap_or_else(|| (0..c.dv.len()).max_by_key(|&i| c.dv[i].t).unwrap_or(0));
                c.dv[i].start(&p, kind, vel, seed);
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
                c.voices[i].start(&p, note, vel, (len * SR) as i64, slide, seed);
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
        for (ci, c) in ch.iter_mut().enumerate() {
            let mut any = c.render(n);
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
        let lilt = match (whole, (step as i32).rem_euclid(6)) {
            (true, 2) => self.lilt,
            (true, 4) => self.lilt * 0.5,
            _ => 0.0,
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
}

impl Song {
    fn new(genre: usize, seed: u64, s: &mut Synth) -> Song {
        let mut song = Song {
            genre,
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
            _ => 16,
        }
    }

    /// 16ths a beat: 6 in 6/8 and 9/8 (the beat a dotted quarter), else 4.
    fn beat_steps(&self) -> u32 {
        if self.genre == TAVERN { self.tv.beat_steps.max(1) } else { 4 }
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
        let lilt = if self.beat_steps() == 6 { 0.18 } else { 0.0 };
        let mut b = Bar { t0, step: self.step_len(), swing: self.swing, lilt, lift: self.lift, ev };
        if self.sec_bar.is_multiple_of(8) && self.sec_bar > 0 && sec.kind == MAIN {
            // now and then an optional layer sits out a phrase
            let opt: &[u32] = match self.genre {
                AMBIENT => &[L_BELL, L_DRONE, L_TEX, L_HAT, L_KICK],
                TECHNO => &[L_OHAT, L_PERC, L_KEYS, L_SNARE],
                SYNTHWAVE => &[L_ARP, L_OHAT, L_PAD, L_PERC],
                LOFI => &[L_LEAD, L_PERC, L_HAT],
                _ => &[L_HAT],
            };
            self.mute = if self.rng.chance(0.5) { self.rng.pick(opt) } else { 0 };
        }
        let m = sec.mask & !self.mute;
        for w in self.walk.iter_mut() {
            *w = walk(*w, &mut self.rng, 0.12);
        }
        match self.genre {
            AMBIENT => self.ambient(s, &mut b, sec, m),
            TECHNO => self.techno(s, &mut b, sec, m),
            SYNTHWAVE => self.synthwave(s, &mut b, sec, m),
            LOFI => self.lofi(s, &mut b, sec, m),
            TAVERN => self.tavern(s, &mut b, sec, m),
            _ => self.polka(&mut b, sec),
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
        match self.genre {
            AMBIENT => self.piece_ambient(s),
            TECHNO => self.piece_techno(s),
            SYNTHWAVE => self.piece_synthwave(s),
            LOFI => self.piece_lofi(s),
            TAVERN => self.piece_tavern(s),
            _ => self.piece_polka(s),
        }
        let meter = match (self.genre, self.tv.kind) {
            (POLKA, _) => "2/4",
            (TAVERN, T_JIG) => "6/8 jig",
            (TAVERN, T_SLIP) => "9/8 slip jig",
            (TAVERN, T_REEL) => "4/4 reel",
            (TAVERN, T_WALTZ) => "3/4",
            (TAVERN, _) if self.tv.bar_steps == 12 => "3/4 ballad",
            (TAVERN, _) => "4/4 ballad",
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
            let (ch, pair) = self.band.tones[k];
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
/// 0.12 s after a short attack (the patch's own attack would only delay it).
fn loudness(p: &Patch, note: f32) -> f32 {
    let q = Patch { a: p.a.min(0.03), ..*p };
    let mut v = VOICE;
    v.start(&q, note, 0.8, (0.5 * SR) as i64, false, 0x1234_5679);
    let mut buf = [[0.0f32; 2]; BLOCK];
    let (mut sum, mut n) = (0.0f32, 0usize);
    for k in 0..(0.15 * SR) as usize / BLOCK {
        buf.fill([0.0; 2]);
        v.render(1.0, &mut buf);
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
/// began with.
#[derive(Default)]
struct Band {
    tones: Vec<(usize, [Patch; 2])>,
    kits: Vec<(usize, [Vec<DrumPatch>; 2])>,
}

impl Band {
    /// Channel `ch`'s pair: the role's tuned original (sometimes) or a pick
    /// from `pal`, and another pick, both brought to the original's
    /// loudness at `note`.
    fn tone(&mut self, r: &mut Rng, s: &Synth, ch: usize, pal: &[(&str, Make)], note: f32) {
        let orig = s.ch[ch].patch;
        let target = loudness(&orig, note);
        let level = |p: Patch| Patch { gain: p.gain * (target / loudness(&p, note)).clamp(0.2, 5.0), ..p };
        let ia = if r.chance(0.35) { None } else { Some(r.below(pal.len())) };
        let mut ib = r.below(pal.len());
        if Some(ib) == ia {
            ib = (ib + 1 + r.below(pal.len() - 1)) % pal.len();
        }
        let a = ia.map_or(orig, |i| level((pal[i].1)(r)));
        let b = level((pal[ib].1)(r));
        self.tones.push((ch, [a, b]));
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
}

impl Engine {
    fn new(genre: usize, seed: u64) -> Engine {
        let genre = genre % GENRES.len();
        let mut syn = Synth::new(genre);
        let song = Song::new(genre, seed, &mut syn);
        Engine { genre, syn, song, ev: Vec::with_capacity(1024), evi: 0, t: 0, pos: 0.0, cur: (0.0, 1.0) }
    }

    /// Fills `out` (interleaved stereo).
    fn render(&mut self, out: &mut [f32]) {
        let frames = out.len() / 2;
        let mut done = 0;
        while done < frames {
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
                if self.want % GENRES.len() == self.cur.genre {
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
    /// Starts playing `GENRES[genre]` (index wraps), seeded: through the
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
        let genre = genre % GENRES.len();
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
        let genre = genre % GENRES.len();
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

    /// Switches genre (at the next bar line, or beat when bars are long, with a crossfade).
    pub fn set_genre(&self, genre: usize) {
        self.want.store(genre % GENRES.len(), Ordering::Relaxed);
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

/// Renders `secs` seconds of `GENRES[genre]` with `seed` to a 16-bit stereo 48 kHz WAV (deterministic for a given genre+seed).
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
    let genre = args.first().and_then(|g| {
        let g = g.to_uppercase();
        GENRES.iter().position(|&n| n == g || n.replace('-', "") == g).or_else(|| g.parse::<usize>().ok())
    });
    let secs = args.get(1).and_then(|s| s.parse::<f64>().ok());
    let (Some(genre), Some(secs), Some(path)) = (genre, secs, args.get(2)) else {
        eprintln!("fleece: --musicwav GENRE SECONDS OUT.wav, GENRE one of {}", GENRES.join(", "));
        std::process::exit(2);
    };
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
        GENRES[genre % GENRES.len()],
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
    }

    /// Genres flipped every 3-7 s for four minutes, from a few seeds: the
    /// engines built for switches (with derived seeds) never panic, never
    /// produce a non-finite sample or one over the ceiling, never go silent.
    #[test]
    #[ignore = "renders 16 minutes of audio: cargo test --bin fleece music -- --ignored"]
    fn genre_switching_stress() {
        for base in 1..=4u64 {
            let mut m = Mix::new(base as usize, base);
            let mut r = Rng::new(base * 77);
            let mut buf = vec![0.0f32; 2 * 4800];
            let mut next = r.range(3.0, 7.0);
            for k in 1..=2400 {
                m.render(&mut buf);
                assert!(buf.iter().all(|x| x.is_finite() && x.abs() <= 0.945), "seed {base}, {} s", k / 10);
                let rms = (buf.iter().map(|x| x * x).sum::<f32>() / buf.len() as f32).sqrt();
                assert!(k < 10 || rms > 1e-3, "seed {base}: silent at {} s", k / 10);
                let t = k as f32 / 10.0;
                if t > next {
                    m.want = r.below(GENRES.len());
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
        for genre in 0..GENRES.len() {
            for seed in 0..50u64 {
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

