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
pub const GENRES: &[&str] = &["AMBIENT", "TECHNO", "SYNTHWAVE", "LO-FI"];
const AMBIENT: usize = 0;
const TECHNO: usize = 1;
const SYNTHWAVE: usize = 2;

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
                if self.level < 1e-4 {
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
}

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
        if !self.on {
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
            self.lfo = self.noise.next() * 0.5 + 0.5;
            self.amp.level = 0.0;
        }
        self.p = *p;
        self.on = true;
        self.note = note;
        self.target = note;
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
        if p.fm > 0.0 {
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
                o[0] += x * a;
                o[1] += x * a;
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
                o[0] += l * a;
                o[1] += r * a;
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
            buf: [[0.0; 2]; BLOCK],
        }
    }

    /// Renders the block into `buf`; whether anything sounded.
    fn render(&mut self, n: usize) -> bool {
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
        let gl = self.gain_s * (1.0 - self.pan).min(1.0);
        let gr = self.gain_s * (1.0 + self.pan).min(1.0);
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
            _ => 0.8,
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
const DORIAN: usize = 1;
const PHRYGIAN: usize = 2;
const LYDIAN: usize = 3;
const MIXOLYDIAN: usize = 4;
const AEOLIAN: usize = 5;

/// Semitones above the tonic of scale degree `deg` (0: the tonic; 7: an octave up).
fn deg_semi(mode: usize, deg: i32) -> i32 {
    deg.div_euclid(7) * 12 + MODES[mode][deg.rem_euclid(7) as usize]
}

/// Chord progressions (scale degrees from 0) that stay on the mode's major
/// and minor chords.
fn progressions(mode: usize) -> &'static [&'static [i32]] {
    match mode {
        IONIAN => {
            &[&[0, 4, 5, 3], &[5, 3, 0, 4], &[0, 5, 3, 4], &[1, 4, 0, 0], &[0, 5, 1, 4], &[3, 2, 1, 0], &[0, 3, 0, 4]]
        }
        DORIAN => &[&[0, 3], &[0, 3, 0, 6], &[0, 6, 3, 0], &[0, 4, 3, 0], &[0, 2, 3, 0], &[0, 6, 2, 3]],
        PHRYGIAN => &[&[0, 1], &[0, 1, 0, 6], &[0, 6, 1, 0], &[0, 3, 1, 0]],
        LYDIAN => &[&[0, 1], &[0, 1, 0, 4], &[0, 6, 1, 0], &[5, 1, 0, 0], &[0, 2, 1, 0]],
        MIXOLYDIAN => &[&[0, 6], &[0, 6, 3, 0], &[0, 3, 6, 0], &[4, 3, 0, 0], &[0, 6, 4, 3]],
        AEOLIAN => {
            &[&[0, 5, 2, 6], &[0, 6, 5, 6], &[5, 6, 0, 0], &[0, 3, 5, 6], &[0, 5, 3, 6], &[0, 2, 5, 6], &[0, 3, 6, 2]]
        }
        _ => &[&[0, 5, 3, 4], &[0, 3, 4, 0], &[5, 3, 4, 0], &[0, 4, 0, 5]],
    }
}

const TRIAD: u8 = 0;
const SEV: u8 = 1;
const NINE: u8 = 2;
const SUS2: u8 = 3;
const SUS4: u8 = 4;
const ADD9: u8 = 5;

#[derive(Clone, Copy, PartialEq)]
struct Chord {
    deg: i32,
    kind: u8,
}

/// The chord's tones, as scale degrees above its root.
fn chord_offsets(kind: u8) -> &'static [i32] {
    match kind {
        SEV => &[0, 2, 4, 6],
        NINE => &[0, 2, 4, 6, 8],
        SUS2 => &[0, 1, 4],
        SUS4 => &[0, 3, 4],
        ADD9 => &[0, 2, 4, 8],
        _ => &[0, 2, 4],
    }
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

/// A four-bar phrase: call, the call with a new ending, the call again, and a
/// cadence (one-bar cells); or a two-bar call and its answer, which keeps the
/// rhythm, moves the line and comes home.
fn make_phrase(r: &mut Rng, density: f32, lo: i32, hi: i32) -> Vec<MNote> {
    let shift = |v: &[MNote], by: u16| v.iter().map(|n| MNote { step: n.step + by, ..*n }).collect::<Vec<_>>();
    if r.chance(0.5) {
        let a = make_cell(r, 16, density, lo, hi, false);
        let mut a2 = a.clone();
        let k = a2.len();
        for n in a2[k.saturating_sub(2)..].iter_mut() {
            n.deg = (n.deg + r.pick(&[-2, -1, 1, 2, 3])).clamp(lo, hi);
        }
        let cad = make_cell(r, 16, density * 0.8, lo, hi, true);
        let mut v = a.clone();
        v.extend(shift(&a2, 16));
        v.extend(shift(&a, 32));
        v.extend(shift(&cad, 48));
        v
    } else {
        let a = make_cell(r, 32, density, lo, hi, false);
        let home = make_cell(r, 32, density, lo, hi, true);
        // the answer: the call's rhythm (until its cadence), its pitches moved
        let by = r.pick(&[-2, 2, -1, 1, 3]);
        let mut b: Vec<MNote> =
            a.iter().filter(|n| n.step < 24).map(|n| MNote { deg: (n.deg + by).clamp(lo, hi), ..*n }).collect();
        b.extend(home.iter().filter(|n| n.step >= 24).copied());
        if b.last().is_none_or(|n| n.step < 24) {
            b.push(MNote { step: 24, len: 8, deg: 0, vel: 0.8 });
        }
        let mut v = a;
        v.extend(shift(&b, 32));
        v
    }
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
    /// semitones added to every note
    lift: i32,
    ev: &'a mut Vec<(u64, Ev)>,
}

impl Bar<'_> {
    fn at(&self, step: f32, late: f32) -> u64 {
        let odd = step.fract() == 0.0 && (step as i32) % 2 == 1;
        let st = step + if odd { self.swing } else { 0.0 };
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
    prog: Vec<Chord>,
    progs: [Vec<Chord>; 2],
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
    /// an optional layer sitting out this phrase
    mute: u32,
    anticipated: bool,
    bell_motif: bool,
    snare_kind: u8,
    /// each piece's tempo, tonic and mode, for --musicwav's summary
    log: Vec<(f64, i32, usize)>,
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
            prog: vec![Chord { deg: 0, kind: TRIAD }],
            progs: [vec![Chord { deg: 0, kind: TRIAD }], vec![Chord { deg: 0, kind: TRIAD }]],
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
            mute: 0,
            anticipated: false,
            bell_motif: false,
            snare_kind: 0,
            log: Vec::new(),
        };
        song.new_piece(s);
        song.section_start(s);
        song
    }

    /// A 16th, in samples.
    fn step_len(&self) -> f64 {
        60.0 / self.bpm * SR as f64 / 4.0
    }

    /// The tonic nearest `center` (MIDI), in [center - 6, center + 6).
    fn tonic_near(&self, center: i32) -> i32 {
        center - 6 + (self.root - (center - 6)).rem_euclid(12)
    }

    fn semi(&self, deg: i32) -> i32 {
        deg_semi(self.mode, deg)
    }

    fn chord(&self, half: u32) -> Chord {
        self.prog[half as usize % self.prog.len()]
    }

    /// The chord's root in [lo, lo + 12).
    fn bass_note(&self, c: Chord, lo: i32) -> i32 {
        let pc = self.root + self.semi(c.deg);
        lo + (pc - lo).rem_euclid(12)
    }

    /// Scale degree `deg` moved to the nearest tone of chord `c`.
    fn snap(&self, deg: i32, c: Chord) -> i32 {
        let offs = chord_offsets(c.kind);
        let is_tone = |d: i32| offs.iter().any(|&o| (d - c.deg - o).rem_euclid(7) == 0);
        for k in [0, -1, 1, -2, 2, -3, 3] {
            if is_tone(deg + k) {
                return deg + k;
            }
        }
        deg
    }

    /// A voicing of `c` within [lo, hi], moving as little as it can from
    /// `prev` (close position, or drop-2 when `open`).
    fn voicing(&self, c: Chord, lo: i32, hi: i32, prev: &[i32], open: bool) -> Vec<i32> {
        let pcs: Vec<i32> =
            chord_offsets(c.kind).iter().map(|&o| (self.root + self.semi(c.deg + o)).rem_euclid(12)).collect();
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

    /// Chord tones from `lo` upward over two octaves.
    fn arp_tones(&self, c: Chord, lo: i32) -> Vec<i32> {
        let mut v: Vec<i32> = chord_offsets(c.kind)
            .iter()
            .flat_map(|&o| {
                let n = lo + (self.root + self.semi(c.deg + o) - lo).rem_euclid(12);
                [n, n + 12]
            })
            .collect();
        v.sort();
        v.dedup();
        v
    }

    /// A progression for the genre: degrees from the mode's pool, chord
    /// colours and lengths (in half bars) the genre likes.
    fn draw_prog(&mut self) -> Vec<Chord> {
        let r = &mut self.rng;
        let pool = progressions(self.mode);
        let (degs, halves, kinds): (Vec<i32>, usize, Vec<u8>) = match self.genre {
            AMBIENT => {
                let mut d = r.pick(pool).to_vec();
                if r.chance(0.3) {
                    d.truncate(2);
                }
                (d, r.pick(&[4, 4, 8]), (0..4).map(|_| r.pick(&[ADD9, ADD9, SUS2, SEV, NINE, TRIAD])).collect())
            }
            TECHNO => {
                let d = if r.chance(0.45) {
                    vec![0]
                } else {
                    let p = r.pick(pool);
                    p[..r.pick(&[2, p.len()]).min(p.len())].to_vec()
                };
                let h = if d.len() == 1 { 8 } else { r.pick(&[4, 8]) };
                (d, h, vec![r.pick(&[SEV, SEV, NINE, SUS4])])
            }
            SYNTHWAVE => (
                r.pick(pool).to_vec(),
                r.pick(&[2, 2, 4]),
                (0..4).map(|_| r.pick(&[TRIAD, TRIAD, TRIAD, ADD9, SUS2, SEV])).collect(),
            ),
            _ => {
                let d = r.pick(pool).to_vec();
                let h = if d.len() == 2 { 4 } else { r.pick(&[2, 2, 4]) };
                (d, h, (0..4).map(|_| r.pick(&[SEV, SEV, NINE])).collect())
            }
        };
        degs.iter()
            .enumerate()
            .flat_map(|(i, &d)| std::iter::repeat_n(Chord { deg: d, kind: kinds[i % kinds.len()] }, halves))
            .collect()
    }

    /// The piece's two progressions (B another than A, when the pool allows).
    fn draw_progs(&mut self) {
        let a = self.draw_prog();
        let mut b = self.draw_prog();
        for _ in 0..6 {
            if b.iter().map(|c| c.deg).ne(a.iter().map(|c| c.deg)) {
                break;
            }
            b = self.draw_prog();
        }
        self.prog = a.clone();
        self.progs = [a, b];
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
        let len = 16.0 * self.step_len();
        let mut b = Bar { t0, step: len / 16.0, swing: self.swing, lift: self.lift, ev };
        let sec = self.plan[self.sec];
        if self.sec_bar.is_multiple_of(8) && self.sec_bar > 0 && sec.kind == MAIN {
            // now and then an optional layer sits out a phrase
            let opt: &[u32] = match self.genre {
                AMBIENT => &[L_BELL, L_DRONE, L_TEX, L_ARP],
                TECHNO => &[L_OHAT, L_PERC, L_KEYS, L_HAT, L_SNARE],
                SYNTHWAVE => &[L_ARP, L_OHAT, L_PAD, L_LEAD],
                _ => &[L_LEAD, L_PERC],
            };
            self.mute = if self.rng.chance(0.3) { self.rng.pick(opt) } else { 0 };
        }
        let m = sec.mask & !self.mute;
        for w in self.walk.iter_mut() {
            *w = walk(*w, &mut self.rng, 0.12);
        }
        match self.genre {
            AMBIENT => self.ambient(s, &mut b, sec, m),
            TECHNO => self.techno(s, &mut b, sec, m),
            SYNTHWAVE => self.synthwave(s, &mut b, sec, m),
            _ => self.lofi(s, &mut b, sec, m),
        }
        // the arc: an intro opens the melodic parts' filters, an outro closes them
        let x = (self.sec_bar as f32 + 1.0) / sec.bars as f32;
        let sweep = match sec.kind {
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
            _ => [3.0, 3.0, 0.0, 0.5, 1.0, 1.0, 0.0],
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
            _ => self.piece_lofi(s),
        }
        self.log.push((self.bpm, self.root, self.mode));
    }

    fn section_start(&mut self, s: &mut Synth) {
        let sec = self.plan[self.sec];
        self.mute = 0;
        // which progression: B for later main sections, choruses, some breaks
        let mains_before = self.plan[..self.sec].iter().filter(|x| x.kind == MAIN).count();
        let r = &mut self.rng;
        let use_b = match self.genre {
            AMBIENT => sec.kind != INTRO && self.sec % 2 == 0,
            TECHNO => (sec.kind == MAIN && mains_before >= 1) || (sec.kind == BREAK && r.chance(0.5)),
            SYNTHWAVE => sec.mask & L_LEAD != 0 && self.piece % 2 == 0,
            _ => sec.mask & L_LEAD != 0 && r.chance(0.6),
        };
        self.prog = self.progs[use_b as usize].clone();
        // synthwave's last chorus, up a step or a half (half the time)
        let last_main = self.plan.iter().rposition(|x| x.kind == MAIN) == Some(self.sec);
        if self.genre == SYNTHWAVE && last_main && sec.energy >= 1.0 && r.chance(0.5) {
            self.lift = r.pick(&[1, 2, 2]);
        }
        match self.genre {
            AMBIENT => {
                s.fx.wind_to = if sec.mask & L_TEX != 0 { r.range(0.06, 0.12) } else { 0.0 };
                if sec.kind == MAIN && r.chance(0.4) {
                    self.phrase = make_phrase(r, 0.3, 0, 11);
                }
                self.arp = make_arp(r);
            }
            TECHNO => {
                s.ch[CH_KICK].rev2 = if sec.mask & L_RUMBLE != 0 { 0.7 } else { 0.0 };
                s.fx.wind_to = if sec.kind == BREAK && r.chance(0.5) { 0.05 } else { 0.0 };
                if sec.kind == MAIN && r.chance(0.5) {
                    self.bass = new_bass(r, self.mode, self.bass_style);
                }
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
                if sec.kind == MAIN && r.chance(0.3) {
                    self.phrase = make_phrase(r, 0.55, -2, 9);
                }
            }
            _ => {
                if sec.kind == MAIN && r.chance(0.35) {
                    self.phrase = make_phrase(r, 0.35, -2, 8);
                }
            }
        }
    }

    // -- the parts all genres share

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

    /// The melody's bar: the phrase (varied each time round) against the chords.
    fn melody(&mut self, b: &mut Bar, ch: usize, vel: f32, late: f32, legato: f32) {
        let pb = (self.sec_bar % 4) as u16;
        if pb == 0 && self.sec_bar > 0 {
            self.vary_phrase();
        }
        let tonic = self.tonic_near(self.lead_center);
        for k in 0..self.phrase.len() {
            let n = self.phrase[k];
            if n.step / 16 != pb {
                continue;
            }
            let st = n.step % 16;
            let c = self.chord(self.sec_bar * 2 + (st >= 8) as u32);
            let mut d = n.deg + self.ptrans;
            if st.is_multiple_of(8) || n.len >= 4 {
                d = self.snap(d, c);
            }
            let j = self.rng.range(0.0, late);
            b.note(st as f32, j, ch, tonic + self.semi(d), n.vel * vel, n.len as f32 * legato, false);
        }
    }

    /// A new turn of the phrase: as it was, moved, rephrased or re-ended.
    fn vary_phrase(&mut self) {
        let r = &mut self.rng;
        match r.weighted(&[0.3, 0.25, 0.2, 0.15, 0.1]) {
            0 => self.ptrans = 0,
            1 => self.ptrans = r.pick(&[-2, 2, 1, -1, 3]),
            2 => {
                // a note nudged a 16th
                let k = r.below(self.phrase.len().max(1));
                if let Some(n) = self.phrase.get_mut(k)
                    && n.step % 16 != 0
                    && n.step % 16 < 15
                {
                    n.step += 1;
                    n.len = n.len.saturating_sub(1).max(1);
                }
            }
            3 => {
                // a couple of notes moved
                for _ in 0..2 {
                    let k = r.below(self.phrase.len().max(1));
                    if let Some(n) = self.phrase.get_mut(k) {
                        n.deg += r.pick(&[-2, -1, 1, 2]);
                    }
                }
            }
            _ => {
                // a new last bar, home again
                let cad = make_cell(r, 16, 0.4, self.prange.0, self.prange.1, true);
                self.phrase.retain(|n| n.step < 48);
                self.phrase.extend(cad.iter().map(|n| MNote { step: n.step + 48, ..*n }));
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
        self.bpm = r.range(60.0, 76.0) as f64;
        self.swing = 0.0;
        self.human = 0.01;
        self.plan = vec![
            Sec { kind: INTRO, bars: 8, mask: L_DRONE | L_PAD, energy: 0.3 },
            Sec { kind: MAIN, bars: 16, mask: L_DRONE | L_PAD | L_BELL, energy: 0.5 },
            Sec {
                kind: MAIN,
                bars: 16,
                mask: L_PAD | L_BELL | L_TEX | if r.chance(0.5) { L_ARP } else { 0 },
                energy: 0.6,
            },
        ];
        if r.chance(0.5) {
            self.plan.push(Sec {
                kind: MAIN,
                bars: 16,
                mask: L_PAD | L_BASS | L_ARP | L_KICK | L_HAT | L_DRONE,
                energy: 0.7,
            });
        } else {
            self.plan.push(Sec { kind: BREAK, bars: 8, mask: L_PAD | L_TEX, energy: 0.35 });
        }
        self.plan.push(Sec { kind: MAIN, bars: 16, mask: L_PAD | L_DRONE | L_BELL | L_TEX, energy: 0.5 });
        self.plan.push(Sec { kind: OUTRO, bars: 8, mask: L_DRONE | L_PAD | L_TEX, energy: 0.3 });
        self.phrase = make_phrase(r, 0.3, 0, 11);
        self.arp = make_arp(r);
        self.arp_rate = r.pick(&[2, 2, 4]);
        self.lead_center = 78;
        self.kick = [0.0; 16];
        self.kick[0] = 0.6;
        self.kick[8] = if r.chance(0.5) { 0.4 } else { 0.0 };
        for i in 0..16 {
            self.hat[i] = if i % 2 == 0 { if i % 4 == 0 { 0.5 } else { 0.35 } } else { 0.0 };
        }
        self.prange = (0, 11);
        sounds_ambient(r, s);
        self.draw_progs();
    }

    fn ambient(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let e = sec.energy;
        let i = self.sec_bar;
        s.ch[CH_PAD].cut = (0.9 * self.walk[0] + 1.2 * (e - 0.5)).exp2();
        s.ch[CH_BELL].cut = 1.0;
        s.rev_wet = 0.5 + 0.12 * self.walk[1];
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
            for h in 0..2 {
                let half = i * 2 + h;
                let c = self.chord(half);
                if half == 0 || self.chord(half - 1) != c {
                    let n = self.bass_note(c, 36);
                    b.note(h as f32 * 8.0, 0.0, CH_BASS, n, 0.8, 15.0, false);
                }
            }
        }
        if m & L_BELL != 0 {
            if i.is_multiple_of(2) {
                self.bell_motif = self.rng.chance(0.55);
            }
            if self.bell_motif {
                let pb = (i % 4) as u16;
                if pb == 0 && i > 0 {
                    self.vary_phrase();
                }
                let tonic = self.tonic_near(self.lead_center);
                for k in 0..self.phrase.len() {
                    let n = self.phrase[k];
                    if n.step / 16 != pb || self.rng.chance(0.15) {
                        continue;
                    }
                    let st = n.step % 16;
                    let c = self.chord(i * 2 + (st >= 8) as u32);
                    let mut d = self.penta(n.deg + self.ptrans);
                    if st.is_multiple_of(8) {
                        d = self.snap(d, c);
                    }
                    let j = self.rng.range(0.0, self.human);
                    b.note(st as f32, j, CH_BELL, tonic + self.semi(d), 0.35 + 0.35 * n.vel, 4.0, false);
                }
            } else {
                for beat in 0..4 {
                    if !self.rng.chance(0.16 + 0.22 * e) {
                        continue;
                    }
                    let st = beat * 4 + self.rng.pick(&[0, 0, 2]);
                    let c = self.chord(i * 2 + (st >= 8) as u32);
                    let offs = chord_offsets(c.kind);
                    let d = c.deg + self.rng.pick(offs) + 7 * self.rng.below(2) as i32;
                    let d = self.penta(d);
                    let note = self.tonic_near(72) + self.semi(d);
                    let j = self.rng.range(0.0, self.human);
                    b.note(st as f32, j, CH_BELL, note, self.rng.range(0.3, 0.65), 4.0, false);
                }
            }
        }
        if m & L_ARP != 0 {
            let rate = self.arp_rate;
            for st in (0..16).step_by(rate) {
                if !self.rng.chance(0.55 + 0.4 * e) {
                    continue;
                }
                let c = self.chord(i * 2 + (st >= 8) as u32);
                let tones = self.arp_tones(c, self.tonic_near(72) - 5);
                let k = self.arp[(st / rate) % self.arp.len()] as usize;
                let vel = if st % 8 == 0 { 0.6 } else { 0.42 };
                b.note(st as f32, 0.0, CH_ARP, tones[k % tones.len()], vel, rate as f32 * 2.0, false);
            }
        }
        if m & L_KICK != 0 {
            for st in 0..16 {
                if self.kick[st] > 0.0 {
                    b.hit(st as f32, 0.0, CH_KICK, 0, self.kick[st]);
                }
            }
        }
        if m & L_HAT != 0 {
            for st in 0..16 {
                if self.hat[st] > 0.0 {
                    let j = self.rng.range(0.0, self.human);
                    b.hit(st as f32, j, CH_HAT, HAT_C, self.hat[st] * self.rng.range(0.7, 1.0));
                }
            }
        }
    }

    // -- techno

    fn piece_techno(&mut self, s: &mut Synth) {
        let r = &mut self.rng;
        self.bpm = if self.piece == 1 {
            r.range(124.0, 132.0) as f64
        } else {
            (self.bpm + r.range(-2.0, 2.0) as f64).clamp(122.0, 134.0)
        };
        self.swing = r.pick(&[0.0, 0.0, 0.0, 0.08, 0.15]);
        self.human = 0.0;
        let rumble = if r.chance(0.6) { L_RUMBLE } else { 0 };
        let stabs = if r.chance(0.6) { L_KEYS } else { 0 };
        let drums = L_KICK | L_HAT | L_OHAT | L_SNARE | L_PERC;
        self.plan = vec![
            Sec { kind: INTRO, bars: 16, mask: L_KICK | L_HAT | if r.chance(0.5) { L_PERC } else { 0 }, energy: 0.35 },
            Sec { kind: BUILD, bars: r.pick(&[8, 16]), mask: L_KICK | L_HAT | L_OHAT | L_SNARE | L_BASS, energy: 0.55 },
            Sec { kind: MAIN, bars: 32, mask: drums | L_BASS | rumble | stabs, energy: 0.8 },
            Sec {
                kind: BREAK,
                bars: 16,
                mask: L_PAD | L_TEX | if r.chance(0.6) { L_BASS } else { 0 } | if r.chance(0.4) { L_HAT } else { 0 },
                energy: 0.3,
            },
            Sec { kind: MAIN, bars: 32, mask: drums | L_BASS | L_RUMBLE | L_KEYS, energy: 0.95 },
        ];
        if r.chance(0.4) {
            self.plan.push(Sec { kind: BREAK, bars: 8, mask: L_PAD | L_KEYS | L_HAT, energy: 0.4 });
            self.plan.push(Sec { kind: MAIN, bars: 16, mask: drums | L_BASS | rumble, energy: 0.9 });
        }
        self.plan.push(Sec { kind: OUTRO, bars: 16, mask: L_KICK | L_HAT | L_PERC | L_BASS, energy: 0.45 });
        self.kick = std::array::from_fn(|i| if i % 4 == 0 { 1.0 } else { 0.0 });
        self.snare = [0.0; 16];
        self.snare[4] = 0.9;
        self.snare[12] = 0.9;
        if r.chance(0.3) {
            self.snare[r.pick(&[7, 15, 14])] = 0.45;
        }
        let acc = r.range(0.7, 0.9);
        self.hat = std::array::from_fn(|i| [0.3, 0.2, acc, 0.22][i % 4]);
        self.ohat = std::array::from_fn(|i| if i % 4 == 2 { 0.55 } else { 0.0 });
        let e = euclid(r.pick(&[3, 5, 7]), 16, r.below(16));
        self.perc = std::array::from_fn(|i| if e[i] { 0.6 } else { 0.0 });
        self.perc_kind = r.pick(&[RIM, RIM, TOM_MID, SHAKER]);
        let st = euclid(r.pick(&[3, 3, 5]), r.pick(&[8, 16]), r.below(8));
        self.stab = std::array::from_fn(|i| if st[i % st.len()] { 0.8 } else { 0.0 });
        self.bass_style = if r.chance(0.65) { 0 } else { 1 };
        self.bass = new_bass(r, self.mode, self.bass_style);
        self.period = r.pick(&[16.0, 24.0, 32.0, 48.0]);
        sounds_techno(r, s, self.bass_style);
        s.dly.target = (self.step_len() * 3.0) as f32;
        self.draw_progs();
    }

    fn techno(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let e = sec.energy;
        let i = self.sec_bar;
        let left = sec.bars - i;
        let next_main = self.plan.get(self.sec + 1).is_some_and(|x| x.kind == MAIN);
        // the acid filter: a slow sweep over `period` bars, a walk, the energy
        self.phase = (self.phase + 1.0 / self.period).fract();
        let sweep = sin01(self.phase);
        let open = if self.bass_style == 0 { 1.5 * sweep + 0.6 * self.walk[0] } else { 0.4 * self.walk[0] };
        s.ch[CH_BASS].cut = (open + 2.0 * (e - 0.7)).exp2();
        s.ch[CH_PAD].cut = (0.7 * self.walk[1] + (left as f32 / sec.bars as f32 - 0.5) * -1.5).exp2();
        s.ch[CH_KEYS].cut = (0.8 * self.walk[2] + 0.5 * sweep).exp2();
        let pre_drop = next_main && left == 1 && sec.kind != OUTRO;
        if m & L_KICK != 0 {
            for st in 0..16 {
                if self.kick[st] > 0.0 && !(pre_drop && st >= 8) {
                    b.hit(st as f32, 0.0, CH_KICK, 0, self.kick[st]);
                }
            }
            if i % 8 == 7 && self.rng.chance(0.25) {
                b.hit(self.rng.pick(&[14.0, 15.0]), 0.0, CH_KICK, 0, 0.55);
            }
        }
        let rolling = next_main && ((sec.kind == BREAK && left <= 4) || (sec.kind == BUILD && left <= 2));
        if m & L_SNARE != 0 && !rolling {
            for st in 0..16 {
                if self.snare[st] > 0.0 {
                    b.hit(st as f32, 0.0, CH_SNARE, 0, self.snare[st]);
                }
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
            for st in 0..16 {
                let v = self.hat[st];
                if v > 0.0 && (st % 4 == 2 || self.rng.chance(0.2 + 0.75 * e)) {
                    b.hit(st as f32, 0.0, CH_HAT, HAT_C, v * self.rng.range(0.8, 1.0));
                }
            }
        }
        if m & L_OHAT != 0 {
            for st in 0..16 {
                if self.ohat[st] > 0.0 {
                    b.hit(st as f32, 0.0, CH_HAT, HAT_O, self.ohat[st]);
                }
            }
        }
        if m & L_PERC != 0 {
            for st in 0..16 {
                if self.perc[st] > 0.0 {
                    b.hit(st as f32, 0.0, CH_PERC, self.perc_kind, self.perc[st] * self.rng.range(0.8, 1.0));
                }
            }
        }
        if i == 0 && sec.kind == MAIN {
            b.hit(0.0, 0.0, CH_CYM, 0, 0.8);
        }
        if m & L_BASS != 0 {
            let c = self.chord(i * 2);
            let root = self.bass_note(c, self.tonic_near(36) - 3);
            let prev_slide = self.bass[15].on && self.bass[15].slide;
            for st in 0..16 {
                let x = self.bass[st];
                if !x.on {
                    continue;
                }
                let slide_in = if st == 0 { prev_slide } else { self.bass[st - 1].on && self.bass[st - 1].slide };
                let next_on = self.bass[(st + 1) % 16].on;
                let len = if x.slide && next_on { 1.15 } else { 0.55 };
                let vel = if x.accent { 1.0 } else { 0.72 };
                b.note(st as f32, 0.0, CH_BASS, root + x.note, vel, len, slide_in && self.bass_style == 0);
            }
        }
        if m & L_KEYS != 0 {
            let c = self.chord(i * 2);
            let v = self.voicing(c, 55, 74, &self.keys_v, false);
            for st in 0..16 {
                if self.stab[st] > 0.0 {
                    for &n in &v {
                        b.note(st as f32, 0.0, CH_KEYS, n, self.stab[st] * self.rng.range(0.85, 1.0), 1.0, false);
                    }
                }
            }
            self.keys_v = v;
        }
        if m & L_PAD != 0 {
            self.pads(b, CH_PAD, 50, 72, 0.6, false, 4.0);
        }
        if i % 8 == 7 {
            // phrase end: the line mutates a little
            let r = &mut self.rng;
            for _ in 0..r.pick(&[1, 1, 2, 3]) {
                let k = r.below(16);
                let x = &mut self.bass[k];
                match r.below(4) {
                    0 => x.on = !x.on || k == 0,
                    1 => x.note = deg_semi(self.mode, r.pick(&[0, 0, 0, 7, 7, 2, 4, 6, -1])),
                    2 => x.accent = !x.accent,
                    _ => x.slide = !x.slide,
                }
            }
            if r.chance(0.3) {
                let k = r.pick(&[1, 3, 5, 7, 9, 11, 13, 15]);
                self.hat[k] = if self.hat[k] > 0.0 { 0.0 } else { 0.22 };
            }
        }
    }

    // -- synthwave

    fn piece_synthwave(&mut self, s: &mut Synth) {
        let r = &mut self.rng;
        self.bpm = r.range(92.0, 112.0) as f64;
        self.swing = 0.0;
        self.human = 0.002;
        let lead = L_LEAD | L_OHAT;
        let beat = L_KICK | L_SNARE | L_HAT | L_BASS | L_PAD | L_ARP;
        self.plan = vec![
            Sec { kind: INTRO, bars: 8, mask: L_PAD | L_ARP | if r.chance(0.5) { L_BASS } else { 0 }, energy: 0.3 },
            Sec { kind: MAIN, bars: 16, mask: beat, energy: 0.6 },
            Sec { kind: MAIN, bars: 16, mask: beat | lead, energy: 0.9 },
            Sec {
                kind: BREAK,
                bars: 8,
                mask: L_PAD | if r.chance(0.5) { L_LEAD } else { L_ARP } | if r.chance(0.5) { L_BASS } else { 0 },
                energy: 0.4,
            },
            Sec { kind: MAIN, bars: 16, mask: beat | lead | L_PERC, energy: 1.0 },
        ];
        if r.chance(0.4) {
            self.plan.push(Sec { kind: MAIN, bars: 8, mask: beat & !L_ARP | L_LEAD, energy: 0.7 });
            self.plan.push(Sec { kind: MAIN, bars: 16, mask: beat | lead, energy: 1.0 });
        }
        self.plan.push(Sec { kind: OUTRO, bars: 8, mask: L_PAD | L_ARP | L_KICK, energy: 0.3 });
        self.kick = [0.0; 16];
        if r.chance(0.4) {
            for i in (0..16).step_by(4) {
                self.kick[i] = 0.95;
            }
        } else {
            self.kick[0] = 1.0;
            self.kick[8] = 0.95;
            if r.chance(0.5) {
                self.kick[r.pick(&[10, 7, 11])] = 0.75;
            }
        }
        self.snare = [0.0; 16];
        self.snare[4] = 0.95;
        self.snare[12] = 0.95;
        let sixteenths = r.chance(0.4);
        self.hat = std::array::from_fn(|i| match (i % 4, sixteenths) {
            (0, _) => 0.5,
            (2, _) => 0.42,
            (_, true) => 0.25,
            _ => 0.0,
        });
        self.ohat = [0.0; 16];
        self.ohat[14] = 0.45;
        if r.chance(0.5) {
            self.ohat[6] = 0.4;
        }
        self.perc = [0.0; 16];
        self.bass_style = r.pick(&[0, 0, 1, 2]) as u8;
        self.arp = make_arp(r);
        self.arp_rate = r.pick(&[1, 1, 2]);
        self.lead_center = r.pick(&[72, 74, 76]);
        self.phrase = make_phrase(r, 0.55, -2, 9);
        self.prange = (-2, 9);
        sounds_synthwave(r, s);
        s.dly.target = (self.step_len() * 3.0) as f32;
        self.draw_progs();
    }

    fn synthwave(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let e = sec.energy;
        let i = self.sec_bar;
        let fill = i % 8 == 7 && m & L_SNARE != 0;
        s.ch[CH_PAD].cut = (0.6 * self.walk[0] + (e - 0.7)).exp2();
        s.ch[CH_ARP].cut = (0.8 * self.walk[1] + 1.2 * (e - 0.6)).exp2();
        if m & L_KICK != 0 {
            for st in 0..16 {
                if self.kick[st] > 0.0 && !(fill && st >= 12) {
                    b.hit(st as f32, 0.0, CH_KICK, 0, self.kick[st]);
                }
            }
        }
        if m & L_SNARE != 0 {
            for st in 0..16 {
                if self.snare[st] > 0.0 && !(fill && st >= 12) {
                    b.hit(st as f32, 0.0, CH_SNARE, 0, self.snare[st]);
                }
            }
        }
        if fill {
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
        }
        if i == 0 && sec.kind == MAIN && m & L_KICK != 0 {
            b.hit(0.0, 0.0, CH_CYM, 0, 0.75);
        }
        if m & L_HAT != 0 {
            for st in 0..16 {
                if self.hat[st] > 0.0 {
                    b.hit(st as f32, 0.0, CH_HAT, HAT_C, self.hat[st] * self.rng.range(0.85, 1.0));
                }
            }
        }
        if m & L_OHAT != 0 {
            for st in 0..16 {
                if self.ohat[st] > 0.0 {
                    b.hit(st as f32, 0.0, CH_HAT, HAT_O, self.ohat[st]);
                }
            }
        }
        if m & L_PERC != 0 && i % 2 == 1 {
            b.hit(14.0, 0.0, CH_PERC, RIM, 0.5);
        }
        if m & L_BASS != 0 {
            let lo = self.tonic_near(36) - 4;
            for st in (0..16).step_by(if self.bass_style == 2 { 1 } else { 2 }) {
                let c = self.chord(i * 2 + (st >= 8) as u32);
                let root = self.bass_note(c, lo);
                let (note, len, vel) = match self.bass_style {
                    0 => (root + if st % 4 == 2 { 12 } else { 0 }, 1.6, 0.85),
                    1 => {
                        let next = self.chord(i * 2 + (st >= 8) as u32 + 1);
                        let n = if st % 8 == 6 && next != c {
                            root + self.semi(c.deg + 4) - self.semi(c.deg)
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
            let rate = self.arp_rate;
            for st in (0..16).step_by(rate) {
                let c = self.chord(i * 2 + (st >= 8) as u32);
                let tones = self.arp_tones(c, self.tonic_near(64) - 4);
                let k = self.arp[(st / rate) % self.arp.len()] as usize;
                let vel = if st % 4 == 0 { 0.85 } else { 0.65 };
                b.note(st as f32, 0.0, CH_ARP, tones[k % tones.len()], vel, rate as f32 * 0.8, false);
            }
        }
        if m & L_LEAD != 0 {
            self.melody(b, CH_LEAD, 0.9, 0.0, 0.95);
        }
    }

    // -- lo-fi

    fn piece_lofi(&mut self, s: &mut Synth) {
        let r = &mut self.rng;
        self.bpm = r.range(72.0, 90.0) as f64;
        self.swing = r.range(0.12, 0.3);
        self.human = 0.012;
        let drums = L_KICK | L_SNARE | L_HAT | L_BASS | L_KEYS | L_TEX;
        self.plan = vec![
            Sec { kind: INTRO, bars: 4, mask: L_KEYS | L_TEX | if r.chance(0.4) { L_BASS } else { 0 }, energy: 0.3 },
            Sec { kind: MAIN, bars: 16, mask: drums, energy: 0.6 },
            Sec { kind: MAIN, bars: 16, mask: drums | L_LEAD | L_PERC, energy: 0.75 },
            Sec {
                kind: BREAK,
                bars: r.pick(&[4, 8]),
                mask: L_KEYS | L_TEX | if r.chance(0.5) { L_LEAD } else { L_BASS | L_HAT },
                energy: 0.4,
            },
            Sec { kind: MAIN, bars: 16, mask: drums | L_LEAD, energy: 0.8 },
        ];
        if r.chance(0.4) {
            self.plan.push(Sec { kind: MAIN, bars: 8, mask: drums, energy: 0.65 });
        }
        self.plan.push(Sec { kind: OUTRO, bars: 4, mask: L_KEYS | L_TEX, energy: 0.3 });
        let kick: &[usize] = r.pick(&[&[0, 7, 10][..], &[0, 10, 11], &[0, 6, 9], &[0, 3, 10], &[0, 7, 9, 10]]);
        self.kick = [0.0; 16];
        for &k in kick {
            self.kick[k] = if k == 0 { 1.0 } else { 0.8 };
        }
        self.snare = [0.0; 16];
        self.snare[4] = 0.85;
        self.snare[12] = 0.9;
        self.snare_kind = if r.chance(0.3) { 1 } else { 0 };
        self.hat = std::array::from_fn(|i| match i % 4 {
            0 => 0.55,
            2 => 0.4,
            _ => 0.18,
        });
        self.perc = std::array::from_fn(|i| if i % 2 == 1 { 0.35 } else { 0.0 });
        self.lead_center = r.pick(&[72, 74, 76]);
        self.phrase = make_phrase(r, 0.35, -2, 8);
        self.anticipated = false;
        sounds_lofi(r, s);
        self.prange = (-2, 8);
        let echo = r.pick(&[4.0, 3.0, 6.0]);
        s.dly.target = (self.step_len() * echo) as f32;
        self.draw_progs();
    }

    fn lofi(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let e = sec.energy;
        let i = self.sec_bar;
        let h = self.human;
        let fill = i % 8 == 7;
        s.ch[CH_KEYS].cut = (0.5 * self.walk[0]).exp2();
        if m & L_KICK != 0 {
            for st in 0..16 {
                if self.kick[st] > 0.0 && !(fill && st >= 8 && self.rng.chance(0.5)) {
                    let j = self.rng.range(0.0, h * 0.3);
                    b.hit(st as f32, j, CH_KICK, 0, self.kick[st] * self.rng.range(0.9, 1.0));
                }
            }
        }
        if m & L_SNARE != 0 {
            for st in 0..16 {
                if self.snare[st] > 0.0 {
                    let j = self.rng.range(0.004, h + 0.004);
                    b.hit(st as f32, j, CH_SNARE, self.snare_kind, self.snare[st] * self.rng.range(0.85, 1.0));
                }
            }
            for st in [3usize, 7, 9, 15] {
                if self.rng.chance(0.12 + 0.15 * e) {
                    b.hit(st as f32, self.rng.range(0.0, h), CH_SNARE, 0, self.rng.range(0.12, 0.22));
                }
            }
        }
        if m & L_HAT != 0 {
            for st in 0..16 {
                if self.hat[st] > 0.0 && (st % 2 == 0 || self.rng.chance(0.25 + 0.4 * e)) {
                    let j = self.rng.range(0.0, h);
                    b.hit(st as f32, j, CH_HAT, HAT_C, self.hat[st] * self.rng.range(0.7, 1.0));
                }
            }
            if self.rng.chance(0.25) {
                b.hit(14.0, self.rng.range(0.0, h), CH_HAT, HAT_O, 0.4);
            }
        }
        if m & L_PERC != 0 {
            for st in 0..16 {
                if self.perc[st] > 0.0 && self.rng.chance(0.7) {
                    b.hit(st as f32, self.rng.range(0.0, h), CH_PERC, SHAKER, self.perc[st] * self.rng.range(0.6, 1.0));
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
            } else if self.rng.chance(0.55) {
                let st = self.rng.pick(&[6.0, 10.0, 11.0]);
                self.strum(b, st, c0, 16.0 - st, 0.42);
            }
            if next != c1 && self.rng.chance(0.3) && i + 1 < sec.bars {
                self.strum(b, 14.0, next, 16.0, 0.55);
                self.anticipated = true;
            }
        }
        if m & L_BASS != 0 {
            let lo = self.tonic_near(42) - 5;
            let next_root = self.bass_note(self.chord(i * 2 + 2), lo);
            let approach = self.rng.chance(0.35) && next_root != self.bass_note(self.chord(i * 2 + 1), lo);
            for hh in 0..2u32 {
                let c = self.chord(i * 2 + hh);
                let root = self.bass_note(c, lo);
                let st0 = hh as f32 * 8.0;
                if hh == 1 && c == self.chord(i * 2) && self.rng.chance(0.4) {
                    // the second half answers with the fifth
                    b.note(st0 + 2.0, h * 0.5, CH_BASS, root + 7, 0.65, 2.0, false);
                    continue;
                }
                b.note(st0, h * 0.3, CH_BASS, root, 0.85, 5.8, false);
                if !(hh == 1 && approach) && self.rng.chance(0.6) {
                    let n = root + self.rng.pick(&[7, 12, 7, 10]);
                    b.note(st0 + 6.0, h * 0.5, CH_BASS, n, 0.6, 1.8, false);
                }
            }
            if approach {
                let n = next_root + self.rng.pick(&[-1, 1, -2]);
                b.note(14.0, h * 0.5, CH_BASS, n, 0.6, 2.0, false);
            }
        }
        if m & L_LEAD != 0 {
            self.melody(b, CH_LEAD, 0.8, h, 0.9);
        }
    }

    /// The electric piano's chord, rolled upward.
    fn strum(&mut self, b: &mut Bar, step: f32, c: Chord, steps: f32, vel: f32) {
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
        r: r.range(5.0, 8.0),
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
    ch[CH_BASS].patch = Patch {
        wave: Wave::Sine,
        wave2: Wave::Tri,
        mix2: 0.3,
        cutoff: 400.0,
        keytrack: 0.0,
        a: 0.05,
        d: 2.0,
        s: 0.7,
        r: 1.0,
        gain: 0.6,
        ..PATCH
    };
    ch[CH_BASS].mono = true;
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
    ch[CH_LEAD].patch = if r.chance(0.5) {
        Patch {
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
        }
    } else {
        Patch {
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
        }
    };
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
        let beat = len / 4.0;
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

/// Music playing through the speakers (pacat), generated on a thread of its own.
pub struct Player {
    want: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    /// frames written so far (the tests watch it)
    #[cfg_attr(not(test), allow(dead_code))]
    frames: Arc<AtomicU64>,
    child: Option<Child>,
    thread: Option<JoinHandle<()>>,
}

impl Player {
    /// Starts playing `GENRES[genre]` (index wraps), seeded. Error if pacat is missing / exits at once.
    pub fn start(genre: usize, seed: u64) -> Result<Player, String> {
        Player::spawn(
            &["pacat", "--raw", "--format=s16le", "--rate=48000", "--channels=2", "--latency-msec=100"],
            genre,
            seed,
        )
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
        let mut p = Player { want, stop, frames, child: Some(child), thread: Some(thread) };
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
fn pieces(genre: usize, seed: u64, secs: f64) -> Vec<(f64, f64, i32, usize)> {
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
    starts.iter().zip(&e.song.log).map(|(&t, &(bpm, k, m))| (t, bpm, k, m)).collect()
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
        .map(|&(t, bpm, k, m)| format!("{t:.0} s: {bpm:.1} bpm {} {}", KEYS[k as usize % 12], MODE[m]))
        .collect();
    println!(
        "{path}: {secs:.1} s of {} (seed {seed}) in {took:.2} s, {:.0}x real time; pieces: {}",
        GENRES[genre % GENRES.len()],
        secs / took.max(1e-9),
        p.join(", ")
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
    fn deterministic_per_seed() {
        assert_eq!(render(3, 5, 2.0), render(3, 5, 2.0));
        assert_ne!(render(3, 5, 2.0), render(3, 6, 2.0));
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
