//! INDONESIAN: gamelan, in two styles. Javanese court gamelan: a gendhing
//! in lancaran, ketawang or ladrang form, slendro or pelog, its gongan
//! marked off by the gong ageng, kenong, kempul, ketuk and kempyang; the
//! balungan on the saron, demung and slenthem, the peking and bonang
//! elaborating it; the kendang leading the irama down (the balungan halving
//! its pace, the elaboration doubling) and the soft instruments coming in
//! there (gender, gambang, rebab, suling); a buka to open, a suwuk slowing
//! into the last gong. Balinese gong kebyar: an explosive kebyar to open
//! (unison byar strokes, runs, sudden stops), then the gong cycles with
//! the gangsa's interlocking kotekan (polos and sangsih), the reyong's
//! norot, the calung and jegogan's pokok, the kendang pair and the
//! cengceng, angsel breaks, a soft suling interlude, a faster pengecet and
//! a last angsel into the gong; each bronze pair tuned a few Hz apart (the
//! ombak's shimmer).
//!
//! The tunings are not equal-tempered: slendro's five near-equal steps and
//! pelog's uneven seven (a pathet's five of them), each piece's gamelan a
//! little different (its tones, its stretched octave, its pitch). Notes go
//! to the synthesiser at fractional pitches (`note_f`).

use super::*;

/// The styles.
const IDN_JAVA: u8 = 0;
const IDN_BALI: u8 = 1;

/// The forms: Javanese (16 or 32 balungan beats a gongan), Balinese (a
/// 16-beat tabuh, an 8-beat gilak).
const F_LANCARAN: u8 = 0;
const F_KETAWANG: u8 = 1;
const F_LADRANG: u8 = 2;
const F_TABUH: u8 = 3;

/// What a section plays.
const PT_BUKA: u8 = 0;
const PT_LOUD: u8 = 1;
const PT_SLOW: u8 = 2;
const PT_SOFT: u8 = 3;
const PT_SUWUK: u8 = 4;
const PT_KEBYAR: u8 = 5;
const PT_PENGAWAK: u8 = 6;
const PT_SULING: u8 = 7;
const PT_PENGECET: u8 = 8;
const PT_END: u8 = 9;

/// A balungan beat with nothing on it.
const REST: i32 = i32::MIN;

/// Slendro, pelog and Bali's pelog selisir, cents above the first tone (about
/// Surjodiningrat's and McPhee's measurements); their octaves stretched.
const SLENDRO: [f32; 5] = [0.0, 231.0, 474.0, 717.0, 955.0];
const SLENDRO_OCT: f32 = 1208.0;
const PELOG: [f32; 7] = [0.0, 120.0, 258.0, 538.0, 675.0, 785.0, 943.0];
const PELOG_OCT: f32 = 1206.0;
const SELISIR: [f32; 5] = [0.0, 128.0, 282.0, 672.0, 792.0];
const SELISIR_OCT: f32 = 1204.0;

/// The pathet: the tones a piece uses (of pelog's seven; slendro's are its
/// five) and its gong tones (degrees of those five), and its name.
struct Pathet {
    tones: [usize; 5],
    gongs: &'static [i32],
    name: usize,
}
const SL_PATHET: [Pathet; 3] = [
    Pathet { tones: [0, 1, 2, 3, 4], gongs: &[1, 4, 3], name: 0 },
    Pathet { tones: [0, 1, 2, 3, 4], gongs: &[3, 0, 1], name: 1 },
    Pathet { tones: [0, 1, 2, 3, 4], gongs: &[4, 1, 2], name: 2 },
];
const PL_PATHET: [Pathet; 3] = [
    Pathet { tones: [0, 1, 2, 4, 5], gongs: &[0, 3], name: 3 },
    Pathet { tones: [0, 1, 2, 4, 5], gongs: &[4, 3, 1], name: 4 },
    Pathet { tones: [1, 2, 4, 5, 6], gongs: &[3, 0, 4], name: 5 },
];
/// The meters as `--musicwav` names them: scale and pathet by form.
const JAVA_METERS: [[&str; 3]; 6] = [
    ["slendro nem, lancaran", "slendro nem, ketawang", "slendro nem, ladrang"],
    ["slendro sanga, lancaran", "slendro sanga, ketawang", "slendro sanga, ladrang"],
    ["slendro manyura, lancaran", "slendro manyura, ketawang", "slendro manyura, ladrang"],
    ["pelog lima, lancaran", "pelog lima, ketawang", "pelog lima, ladrang"],
    ["pelog nem, lancaran", "pelog nem, ketawang", "pelog nem, ladrang"],
    ["pelog barang, lancaran", "pelog barang, ketawang", "pelog barang, ladrang"],
];

/// A gamelan's tuning: its five tones (cents above the first), its octave
/// (cents) and the first tone's pitch (MIDI, fractional) in the demung's
/// octave.
#[derive(Clone, Copy, Default)]
struct Tuning {
    cents: [f32; 5],
    oct: f32,
    base: f32,
}

impl Tuning {
    /// Degree `d` (five a stretched octave, 0 the first tone in the demung's octave) as a pitch.
    fn pitch(&self, d: i32) -> f32 {
        let (o, k) = (d.div_euclid(5), d.rem_euclid(5) as usize);
        self.base + (o as f32 * self.oct + self.cents[k]) / 100.0
    }
}

/// `d` moved by octaves into `[lo, hi]` (at least five degrees wide).
fn fold(d: i32, lo: i32, hi: i32) -> i32 {
    let mut d = d;
    while d < lo {
        d += 5;
    }
    while d > hi {
        d -= 5;
    }
    d
}

/// A section's part, its gongan (Bali: its pokok), irama (1, or 2 where the
/// balungan's beats are twice as long), and its tempo from its first bar to
/// its last (the next section's first bar takes the next).
#[derive(Clone, Copy)]
struct ISec {
    part: u8,
    gongan: usize,
    irama: u8,
    bpm0: f64,
    bpm1: f64,
}

/// An Indonesian piece's state.
#[derive(Default)]
pub(super) struct Indonesian {
    pub(super) bar_steps: u32,
    pub(super) beat_steps: u32,
    pub(super) meter: &'static str,
    style: u8,
    form: u8,
    tun: Tuning,
    /// the gong tone (a degree)
    home: i32,
    /// the gongans' balungan (Bali: the pokok), a degree or `REST` a beat
    gongans: Vec<Vec<i32>>,
    map: Vec<ISec>,
    /// the irama playing (the outro keeps the one it comes from)
    irama: u8,
    /// the bronze pairs' beat (Hz; 0 none: the Javanese)
    ombak: f32,
    /// the kendang's patterns (a 16th each: 0 none, 1 dhung, 2 dhah, 3 tak,
    /// 4 tung, 5 ket): A, B, and the gatra into the gong
    kend: [[u8; 16]; 3],
    /// Bali: the cengceng's accents, which cycles end on an angsel
    cek: [f32; 16],
    angsel: Vec<bool>,
    /// the soft instruments' last notes (voice leading), the suling's phrase
    gender: i32,
    gambang: i32,
    /// the instruments for the loud parts and the soft (bonang and gender,
    /// peking and gambang, rebab or ugal and suling)
    loud: Vec<Patch>,
    soft: Vec<Patch>,
    /// a mix's drummer: whether its style is drawn, and which
    dready: bool,
    dstyle: u8,
    dpat: [[u8; 16]; 2],
}

/// The tempo a piece is pulled to in a mix (60 % of the way to the
/// drummer's, as `fuse` pulls the first bar), for all its sections.
fn idn_pull(guest: Option<usize>, bpm: f64) -> f64 {
    guest.map_or(bpm, |g| bpm + 0.6 * (MID_BPM[g] - bpm))
}

/// The colotomic instruments at beat `j` (0-based) of a `form`'s gongan of
/// `nb` beats: (kenong, kempul, ketuk, kempyang, gong).
fn colotomy(form: u8, j: u32, nb: u32) -> (bool, bool, bool, bool, bool) {
    let n = j + 1;
    let gong = n == nb;
    match form {
        F_LANCARAN => (n % 4 == 0, n % 4 == 2 && n != 2, n % 2 == 1, false, gong),
        F_KETAWANG => (n % 8 == 0, n == 12, n % 4 == 2, n % 2 == 1, gong),
        _ => (n % 8 == 0, n % 8 == 4 && n != 4, n % 4 == 2, n % 2 == 1, gong),
    }
}

/// A note at a fractional pitch (the gamelan's tunings), `steps` 16ths of gate.
fn note_f(b: &mut Bar, step: f32, late: f32, ch: usize, note: f32, vel: f32, steps: f32) {
    let t = b.at(step, late);
    let len = b.secs(steps);
    let note = note + b.lift as f32;
    b.ev.push((t, Ev::Note { ch: ch as u8, note, vel: vel.clamp(0.0, 1.0), len, slide: false }));
}

/// `note_f`, sliding from the voice's last note when it still sounds (a rebab's or suling's portamento).
fn note_slide(b: &mut Bar, step: f32, late: f32, ch: usize, note: f32, vel: f32, steps: f32) {
    let t = b.at(step, late);
    let len = b.secs(steps);
    let note = note + b.lift as f32;
    b.ev.push((t, Ev::Note { ch: ch as u8, note, vel: vel.clamp(0.0, 1.0), len, slide: true }));
}

/// A bronze pair's stroke: the pengumbang at `note`, the pengisep `hz` above
/// (the ombak's beat, the same Hz up the range), a few ms behind.
#[allow(clippy::too_many_arguments)]
fn ombak_pair(b: &mut Bar, step: f32, late: f32, ch: usize, note: f32, vel: f32, steps: f32, hz: f32) {
    note_f(b, step, late, ch, note, vel, steps);
    if hz > 0.0 {
        let up = 12.0 * (1.0 + hz / mtof(note)).log2();
        note_f(b, step, late + 0.003, ch, note + up, vel * 0.92, steps);
    }
}

/// The balungan's figures for a gatra: its first three beats against the
/// fourth (the seleh, its goal): runs, turns about it, neighbours.
const GATRA_FIGS: &[[i32; 3]] = &[
    [3, 2, 1],
    [-3, -2, -1],
    [1, 2, 1],
    [2, 1, 2],
    [-1, -2, -1],
    [1, -1, 1],
    [-1, 1, -1],
    [2, 1, -1],
    [-2, -1, 1],
    [1, 0, 1],
    [-1, 0, -1],
    [0, 1, 2],
    [2, 3, 1],
    [-2, -1, -2],
];

/// A gatra ending on `t`, its first beat near `prev`; `nibani`: the sparse
/// balungan (the first and third beats left empty).
fn gatra(r: &mut Rng, prev: i32, t: i32, nibani: bool) -> [i32; 4] {
    let cost = |f: &[i32; 3], r: &mut Rng| (t + f[0] - prev).abs() as f32 + r.range(0.0, 2.5);
    let mut best = 0;
    let mut bc = f32::MAX;
    for (k, f) in GATRA_FIGS.iter().enumerate() {
        let c = cost(f, r);
        if c < bc && (t + f.iter().max().unwrap_or(&0)) <= 9 && (t + f.iter().min().unwrap_or(&0)) >= -3 {
            bc = c;
            best = k;
        }
    }
    let f = GATRA_FIGS[best];
    if nibani { [REST, t + f[1], REST, t] } else { [t + f[0], t + f[1], t + f[2], t] }
}

/// A gongan's balungan: `gatras` of four beats, the seleh moving by small
/// steps between the pathet's strong tones (at the kenong), the last the
/// gong tone.
fn gongan(r: &mut Rng, gatras: usize, home: i32, strong: &[i32], kenong_every: usize, nibani: f32) -> Vec<i32> {
    let mut seleh = vec![0; gatras];
    seleh[gatras - 1] = home;
    let mut prev = home + r.pick(&[0, 1, -1, 2]);
    for (k, s) in seleh.iter_mut().enumerate().take(gatras - 1) {
        let on_kenong = (k + 1) % kenong_every == 0;
        let want = if on_kenong { r.pick(strong) } else { prev + r.pick(&[1, -1, 2, -2, 0, 1, -1]) };
        // the tone's octave nearest the last
        *s = fold(want, prev - 2, prev + 2).clamp(-1, 7);
        prev = *s;
    }
    // the gong tone in the octave nearest the line
    let h = fold(home, seleh[gatras.saturating_sub(2)] - 2, seleh[gatras.saturating_sub(2)] + 2).clamp(-1, 7);
    seleh[gatras - 1] = h;
    let mut out = Vec::with_capacity(gatras * 4);
    let mut p = h;
    let sparse = r.chance(nibani);
    for &t in &seleh {
        let thin = sparse && r.chance(0.7);
        let g = gatra(r, p, t, thin);
        out.extend_from_slice(&g);
        p = t;
    }
    out
}

/// The kendang's pattern for a gatra (irama I) or half a gatra (irama II,
/// busier); `gong`: the gatra into the gong (its strokes doubled, the seleh
/// struck hard).
fn kendang_pattern(r: &mut Rng, busy: bool, gong: bool) -> [u8; 16] {
    let mut p = [0u8; 16];
    p[12] = if gong || r.chance(0.6) { 2 } else { 1 };
    if r.chance(0.7) {
        p[if r.bit_chance() { 4 } else { 6 }] = 1;
    }
    for st in [2, 10] {
        if r.chance(0.6) {
            p[st] = 3;
        }
    }
    if r.chance(0.5) {
        p[14] = if r.bit_chance() { 3 } else { 5 };
    }
    if r.chance(0.4) {
        p[8] = 4;
    }
    let ghost = if busy { 0.35 } else { 0.15 };
    for st in (1..16).step_by(2) {
        if p[st] == 0 && r.chance(ghost) {
            p[st] = 5;
        }
    }
    if busy {
        for st in [0, 4, 6] {
            if p[st] == 0 && r.chance(0.5) {
                p[st] = r.pick(&[3, 4, 1]);
            }
        }
    }
    if gong {
        p[8] = 1;
        p[10] = 1;
    }
    p
}

impl Song {
    /// An Indonesian piece: Javanese two times in three, else Balinese.
    pub(super) fn piece_indonesian(&mut self, s: &mut Synth) {
        let style = if self.rng.chance(0.62) { IDN_JAVA } else { IDN_BALI };
        self.idn =
            Indonesian { bar_steps: 16, beat_steps: 4, meter: "gamelan", style, irama: 1, ..Indonesian::default() };
        self.swing = 0.0;
        self.human = if style == IDN_JAVA { 0.007 } else { 0.004 };
        self.lifts.clear();
        self.pivot = None;
        if style == IDN_JAVA {
            self.piece_java();
        } else {
            self.piece_bali();
        }
        // (the composer's key and mode, for the summary: the gong tone's
        // pitch class; the mode the nearest of the church modes)
        self.root = (self.idn.tun.pitch(self.idn.home).round() as i32).rem_euclid(12);
        sounds_gamelan(&mut self.rng, s, style);
        self.band = band_indonesian(&mut self.rng, s, &mut self.idn);
        // the recordings brought to the synthesiser's balance
        let table = if style == IDN_JAVA { JAVA_REC_TRIMS } else { BALI_REC_TRIMS };
        let trim = |ch: usize, p: &mut Patch| {
            if p.samp != 0 {
                p.gain *= 10f32.powf(trim_of(table, ch, p.samp) / 20.0);
            }
        };
        for (ch, pair, _) in self.band.tones.iter_mut() {
            pair.iter_mut().for_each(|p| trim(*ch, p));
        }
        for (k, ch) in [CH_KEYS, CH_ARP, CH_LEAD].into_iter().enumerate() {
            trim(ch, &mut self.idn.loud[k]);
            trim(ch, &mut self.idn.soft[k]);
        }
        for (ch, c) in s.ch.iter_mut().enumerate() {
            trim(ch, &mut c.patch);
        }
        self.idn.dready = false;
        let first = self.idn.map[0];
        self.bpm = first.bpm0;
        s.dly.target = (self.step_len() * 3.0) as f32;
    }

    /// A Javanese gendhing: its tuning, pathet, form and gongans, and the
    /// plan (buka; irama I; down to irama II, the soft instruments in; back
    /// up; the suwuk).
    fn piece_java(&mut self) {
        let guest = self.guest;
        let r = &mut self.rng;
        let pelog = r.chance(0.5);
        let pi = r.below(3);
        let p = if pelog { &PL_PATHET[pi] } else { &SL_PATHET[pi] };
        let mut cents = [0.0; 5];
        for (k, c) in cents.iter_mut().enumerate() {
            let t = if pelog { PELOG[p.tones[k]] } else { SLENDRO[p.tones[k]] };
            *c = t + if k == 0 { 0.0 } else { r.range(-12.0, 12.0) };
        }
        // (pelog barang's first tone is pelog's 2: its cents from there)
        let c0 = cents[0];
        for c in cents.iter_mut() {
            *c -= c0;
        }
        let oct = if pelog { PELOG_OCT } else { SLENDRO_OCT } + r.range(-4.0, 8.0);
        let base = r.range(58.5, 62.5);
        self.idn.tun = Tuning { cents, oct, base };
        self.idn.home = r.pick(p.gongs);
        let form = r.weighted(&[1.0, 0.8, 1.4]) as u8;
        self.idn.form = form;
        self.idn.meter = JAVA_METERS[p.name][form as usize];
        let strong: Vec<i32> = p.gongs.to_vec();
        let (gatras, kenong_every) = match form {
            F_LANCARAN => (4, 1),
            F_KETAWANG => (4, 2),
            _ => (8, 2),
        };
        let nibani = if form == F_LANCARAN { 0.45 } else { 0.15 };
        let home = self.idn.home;
        let n = if form == F_LANCARAN { 3 } else { 2 };
        self.idn.gongans = (0..n).map(|_| gongan(r, gatras, home, &strong, kenong_every, nibani)).collect();
        // the kendang's patterns
        self.idn.kend =
            [kendang_pattern(r, false, false), kendang_pattern(r, false, false), kendang_pattern(r, false, true)];
        // the tempo: irama I's balungan beat; irama II's at about half (the
        // elaboration a little faster than irama I's)
        let t1 = match form {
            F_LANCARAN => r.range(90.0, 110.0),
            F_KETAWANG => r.range(60.0, 70.0),
            _ => r.range(60.0, 72.0),
        } as f64;
        let t1 = idn_pull(guest, t1);
        let t2 = t1 * r.range(1.04, 1.16) as f64;
        let gb = gatras as u32;
        let mut plan: Vec<(Sec, ISec)> = Vec::new();
        let isec = |part, gongan, irama, bpm0, bpm1| ISec { part, gongan, irama, bpm0, bpm1 };
        let soft_buka = form == F_KETAWANG;
        plan.push((
            Sec { kind: INTRO, bars: 2, mask: if soft_buka { L_LEAD } else { L_KEYS }, energy: 0.5 },
            isec(PT_BUKA, 0, 1, t1 * 0.94, t1),
        ));
        let loud = L_BELL | L_BASS | L_ARP | L_KEYS | L_PAD | L_DRONE | L_PERC | L_KICK | L_SNARE;
        let soft = L_BELL | L_BASS | L_ARP | L_KEYS | L_PAD | L_DRONE | L_PERC | L_KICK | L_SNARE | L_LEAD;
        match form {
            F_LANCARAN => {
                // the gongans through, twice each, then faster and louder (seseg)
                for g in 0..n {
                    plan.push((
                        Sec { kind: MAIN, bars: gb * 2, mask: loud | L_HAT, energy: 0.7 },
                        isec(PT_LOUD, g, 1, t1, t1),
                    ));
                }
                for g in 0..2 {
                    let up = t1 * 1.12;
                    plan.push((
                        Sec { kind: MAIN, bars: gb * 2, mask: loud | L_HAT, energy: 0.9 },
                        isec(PT_LOUD, g, 1, if g == 0 { t1 } else { up }, up),
                    ));
                }
            }
            _ => {
                // irama I, the gongan once or twice; the kendang slowing
                // through the next into irama II; irama II's gongans (the
                // soft instruments), twice through; then back up
                let reps = if form == F_KETAWANG { 2 } else { 1 };
                plan.push((Sec { kind: MAIN, bars: gb * reps, mask: loud, energy: 0.7 }, isec(PT_LOUD, 0, 1, t1, t1)));
                plan.push((
                    Sec { kind: MAIN, bars: gb, mask: loud, energy: 0.6 },
                    isec(PT_SLOW, 1 % n, 1, t1, t2 * 0.5),
                ));
                for k in 0..2 {
                    plan.push((
                        Sec { kind: MAIN, bars: 2 * gb, mask: soft, energy: 0.5 },
                        isec(PT_SOFT, k % n, 2, t2, t2 * if k == 1 { 0.96 } else { 1.0 }),
                    ));
                }
                if form == F_LADRANG && r.chance(0.6) {
                    // up again (ngampat): the last gongan back in irama I
                    plan.push((
                        Sec { kind: MAIN, bars: gb, mask: loud, energy: 0.8 },
                        isec(PT_LOUD, 0, 1, t1 * 0.9, t1 * 1.08),
                    ));
                }
            }
        }
        plan.push((
            Sec {
                kind: OUTRO,
                bars: 2,
                mask: soft & !L_LEAD | if form == F_KETAWANG { L_LEAD } else { 0 },
                energy: 0.6,
            },
            isec(PT_SUWUK, 0, 1, t1, t1),
        ));
        self.plan = plan.iter().map(|x| x.0).collect();
        self.idn.map = plan.iter().map(|x| x.1).collect();
    }

    /// A Balinese gong kebyar piece: the kebyar, the pengawak's gong cycles
    /// (angsel now and then), a suling interlude, the faster pengecet, the
    /// last angsel into the gong.
    fn piece_bali(&mut self) {
        let guest = self.guest;
        let r = &mut self.rng;
        let mut cents = SELISIR;
        for c in cents.iter_mut().skip(1) {
            *c += r.range(-10.0, 10.0);
        }
        self.idn.tun = Tuning { cents, oct: SELISIR_OCT + r.range(-3.0, 6.0), base: r.range(59.0, 63.0) };
        self.idn.home = r.pick(&[0, 0, 3, 2]);
        self.idn.form = F_TABUH;
        self.idn.meter = "pelog selisir, gong kebyar";
        self.idn.ombak = r.range(5.0, 7.5);
        let home = self.idn.home;
        // the pengawak's pokok (a 16-beat cycle) and the pengecet's (8)
        let a = gongan(r, 4, home, &[0, 3, 2, 4], 2, 0.0);
        let b = gongan(r, 2, home, &[0, 3, 2], 1, 0.0);
        let c = gongan(r, 4, home, &[1, 3, 4], 2, 0.0);
        self.idn.gongans = vec![a, b, c];
        // cengceng accents and the kendang's interlock (wadon 1/2, lanang 3/4/5)
        for k in 0..16 {
            self.idn.cek[k] = if k % 4 == 2 {
                0.5
            } else if k % 2 == 1 {
                r.range(0.25, 0.4)
            } else {
                r.range(0.3, 0.42)
            };
        }
        for v in 0..2 {
            let mut p = [0u8; 16];
            for (k, x) in p.iter_mut().enumerate() {
                *x = match r.weighted(&[0.4, 0.4, 0.2]) {
                    0 => {
                        if k % 4 == 0 {
                            1
                        } else {
                            r.pick(&[1, 2, 2])
                        }
                    }
                    1 => r.pick(&[4, 4, 3, 5]),
                    _ => 0,
                };
            }
            p[12] = 1;
            self.idn.kend[v] = p;
        }
        self.idn.kend[2] = kendang_pattern(r, true, true);
        let t1 = idn_pull(guest, r.range(84.0, 100.0) as f64);
        let t2 = t1 * r.range(1.3, 1.5) as f64;
        let tk = t1 * r.range(1.05, 1.2) as f64;
        let isec = |part, gongan, bpm0, bpm1| ISec { part, gongan, irama: 1, bpm0, bpm1 };
        let all = L_BELL | L_ARP | L_KEYS | L_BASS | L_PAD | L_DRONE | L_PERC | L_KICK | L_SNARE | L_HAT;
        let mut plan: Vec<(Sec, ISec)> = vec![(
            Sec { kind: INTRO, bars: r.pick(&[4, 5, 6]), mask: all | L_LEAD | L_CTR, energy: 1.0 },
            isec(PT_KEBYAR, 0, tk, tk),
        )];
        let reps = r.pick(&[3, 4]);
        plan.push((Sec { kind: MAIN, bars: 4 * reps, mask: all | L_LEAD, energy: 0.75 }, isec(PT_PENGAWAK, 0, t1, t1)));
        if r.chance(0.7) {
            plan.push((
                Sec { kind: BREAK, bars: 8, mask: L_LEAD | L_PAD | L_BASS | L_DRONE | L_PERC | L_KICK, energy: 0.4 },
                isec(PT_SULING, 2, t1 * 0.85, t1 * 0.85),
            ));
            plan.push((
                Sec { kind: MAIN, bars: 8, mask: all | L_LEAD, energy: 0.8 },
                isec(PT_PENGAWAK, 2, t1, t1 * 1.1),
            ));
        }
        let reps = r.pick(&[6, 8]);
        plan.push((Sec { kind: MAIN, bars: 2 * reps, mask: all, energy: 0.95 }, isec(PT_PENGECET, 1, t2, t2 * 1.04)));
        plan.push((Sec { kind: OUTRO, bars: 2, mask: all, energy: 1.0 }, isec(PT_END, 1, t2, t2)));
        // which cycles end on an angsel (never the first)
        self.idn.angsel = (0..64).map(|k| k > 0 && r.chance(0.35)).collect();
        self.plan = plan.iter().map(|x| x.0).collect();
        self.idn.map = plan.iter().map(|x| x.1).collect();
    }

    /// An Indonesian section begins: its tempo; the outro keeps the irama it
    /// comes from and slows from the tempo it finds.
    pub(super) fn indonesian_section(&mut self) {
        let Some(&x) = self.idn.map.get(self.sec) else { return };
        if x.part == PT_SUWUK || x.part == PT_END {
            let from = self.bpm;
            let m = &mut self.idn.map[self.sec];
            m.bpm0 = from;
            m.bpm1 = from * 0.72;
            m.irama = self.idn.irama.max(1);
        } else {
            self.idn.irama = x.irama;
        }
        self.bpm = self.idn.map[self.sec].bpm0;
    }

    /// The tempo of bar `i` of section `k`.
    fn idn_tempo(&self, k: usize, i: u32) -> f64 {
        let x = self.idn.map[k];
        let n = self.plan[k].bars.max(2) - 1;
        let f = (i.min(n) as f64 / n as f64).powf(1.5);
        x.bpm0 + (x.bpm1 - x.bpm0) * f
    }

    /// A bar of gamelan.
    pub(super) fn indonesian(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let x = self.idn.map[self.sec];
        if self.sec_bar == 0 {
            // the loud instruments or the soft ones
            let soft = matches!(x.part, PT_SOFT | PT_SULING) || (x.part == PT_SUWUK && x.irama == 2);
            // (the Javanese soft parts' line the rebab's two times in three, else the suling's)
            let suling = soft && (self.idn.style == IDN_BALI || self.rng.chance(0.35));
            for (k, ch) in [CH_KEYS, CH_ARP, CH_LEAD].into_iter().enumerate() {
                let v = if (k < 2 && soft) || (k == 2 && suling) { &self.idn.soft } else { &self.idn.loud };
                if let Some(&p) = v.get(k) {
                    s.ch[ch].patch = p;
                }
            }
        }
        if self.idn.style == IDN_JAVA {
            self.java_bar(b, sec, m, x);
        } else {
            self.bali_bar(b, sec, m, x);
        }
        // the next bar's tempo (a section's first is set as it begins)
        if self.sec_bar + 1 < sec.bars {
            self.bpm = self.idn_tempo(self.sec, self.sec_bar + 1);
        }
    }

    /// A bar of Javanese gamelan.
    fn java_bar(&mut self, b: &mut Bar, sec: Sec, m: u32, x: ISec) {
        let irama = if x.part == PT_SUWUK { x.irama } else { x.irama.max(1) };
        let bal = if irama == 2 { 8u32 } else { 4 };
        let per_bar = 16 / bal;
        let g = self.idn.gongans[x.gongan.min(self.idn.gongans.len() - 1)].clone();
        let nb = g.len() as u32;
        let i = self.sec_bar;
        let bars = sec.bars;
        // where in the gongan: the buka and the suwuk are its last beats
        let j0 = match x.part {
            PT_BUKA | PT_SUWUK => (nb - per_bar * bars.min(nb / per_bar) + i * per_bar) % nb,
            _ => (i * per_bar) % nb,
        };
        let tun = self.idn.tun;
        let h = self.human;
        let soft = irama == 2;
        let e = sec.energy;
        let last_bar = i + 1 == bars;
        let ending = x.part == PT_SUWUK && last_bar;
        let buka = x.part == PT_BUKA;
        let home = self.idn.home;
        let form = self.idn.form;
        // the note at beat j (a rest: the last one sounded)
        let at = |j: u32| -> i32 {
            let mut k = j as i64;
            for _ in 0..8 {
                let v = g[k.rem_euclid(nb as i64) as usize];
                if v != REST {
                    return v;
                }
                k -= 1;
            }
            home
        };
        // the suwuk's last beats spread out (the slowing into the gong)
        let spread = |st: f32| -> f32 {
            if ending {
                // (the bar's last hit still inside it)
                let y = st / 16.0;
                st + 0.9 * y * y
            } else {
                st
            }
        };
        let vloud = if soft { 0.62 } else { 0.8 } * (0.85 + 0.2 * e);
        for k in 0..per_bar {
            let j = j0 + k;
            let jj = j % nb;
            let st0 = (k * bal) as f32;
            let st = spread(st0);
            let d = g[jj as usize];
            let cur = at(jj);
            let next = at(jj + 1);
            let next2 = at(jj + 2);
            let gong_beat = jj + 1 == nb;
            let (ken, kem, ketuk, kempyang, gong) = colotomy(form, jj, nb);
            let in_buka = buka && !(last_bar && gong_beat);
            let r = &mut self.rng;
            // the balungan: demung and saron, the slenthem an octave under (damped at the next)
            if d != REST && (m & L_BELL != 0 || buka) && !in_buka {
                let len = bal as f32 - 0.3;
                let v = vloud * if gong_beat || ken { 1.06 } else { 1.0 };
                let late = r.range(0.0, h);
                note_f(b, st, late, CH_BELL, tun.pitch(fold(d, 4, 10)), v, len);
                note_f(b, st, late + 0.004, CH_BELL, tun.pitch(fold(d, -1, 5)), v * 0.95, len);
            }
            if d != REST && (m & L_BASS != 0 || buka) && !in_buka {
                note_f(b, st, r.range(0.0, h), CH_BASS, tun.pitch(fold(d, -6, 0)), vloud * 0.9, bal as f32 * 1.8);
            }
            // the peking (irama I: the beat's note, then the next's; irama
            // II: each twice) or the gambang's runs in octaves
            if m & L_ARP != 0 && !in_buka {
                if soft {
                    let mut p = if self.idn.gambang == 0 { fold(cur, 5, 10) } else { self.idn.gambang };
                    let goal = fold(next, p - 2, p + 2).clamp(1, 12);
                    for q in 0..bal {
                        let left = bal - q;
                        let dir = (goal - p).signum();
                        p = if (goal - p).abs() >= left as i32 {
                            p + dir
                        } else if r.chance(0.5) {
                            p + if r.bit_chance() { 1 } else { -1 }
                        } else {
                            p + dir
                        };
                        p = p.clamp(0, 13);
                        let sq = spread(st0 + q as f32);
                        let v = 0.5 + if q % 2 == 0 { 0.1 } else { 0.0 };
                        note_f(b, sq, r.range(0.0, h * 0.5), CH_ARP, tun.pitch(p), v, 1.6);
                        note_f(b, sq, 0.002, CH_ARP, tun.pitch(p - 5), v * 0.8, 1.6);
                    }
                    self.idn.gambang = p;
                } else {
                    let notes = [cur, next];
                    for (q, &n) in notes.iter().enumerate() {
                        let sq = spread(st0 + 2.0 * q as f32);
                        note_f(b, sq, r.range(0.0, h * 0.5), CH_ARP, tun.pitch(fold(n, 9, 15)), vloud * 0.75, 1.9);
                    }
                }
            }
            // the bonang (mipil: this beat's note and the next's in turn, the
            // panerus twice as fast an octave up; into the gong a sekaran
            // walking to the gong tone, struck in octaves) or the gender
            if m & L_KEYS != 0 {
                if soft {
                    let mut p = if self.idn.gender == 0 { fold(cur, 2, 7) } else { self.idn.gender };
                    let goal = fold(next, p - 2, p + 2).clamp(0, 9);
                    for q in 0..bal {
                        let left = bal - q;
                        let dir = (goal - p).signum();
                        if (goal - p).abs() >= left as i32 || r.chance(0.55) {
                            p += dir;
                        } else if r.chance(0.6) {
                            p += if r.bit_chance() { 1 } else { -1 };
                        }
                        p = p.clamp(-1, 10);
                        let sq = spread(st0 + q as f32);
                        note_f(b, sq, r.range(0.0, h), CH_KEYS, tun.pitch(p), 0.42, 2.5);
                        if q % 4 == 0 {
                            // the left hand, an octave and more under
                            note_f(b, sq, r.range(0.0, h), CH_KEYS, tun.pitch(fold(next, -5, 0)), 0.36, 4.0);
                        }
                    }
                    self.idn.gender = p;
                } else {
                    let into_gong = jj + 4 >= nb && !buka;
                    if into_gong && !gong_beat {
                        // the sekaran: down (or up) the scale to the gong tone
                        let tgt = fold(home, 2, 7);
                        let mut p = fold(cur, tgt - 2, tgt + 3);
                        for q in 0..(bal / 2) {
                            let sq = spread(st0 + 2.0 * q as f32);
                            note_f(b, sq, r.range(0.0, h), CH_KEYS, tun.pitch(p), 0.6, 2.0);
                            note_f(b, sq, 0.003, CH_KEYS, tun.pitch(p + 5), 0.5, 2.0);
                            p += (tgt - p).signum();
                        }
                    } else if gong_beat {
                        let t = fold(home, 2, 7);
                        note_f(b, st, r.range(0.0, h), CH_KEYS, tun.pitch(t), 0.68, 6.0);
                        note_f(b, st, 0.003, CH_KEYS, tun.pitch(t + 5), 0.6, 6.0);
                    } else {
                        let (a, c) = if jj % 2 == 0 { (cur, next) } else { (next, next2) };
                        let (a, c) = (fold(a, 0, 8), fold(c, 0, 8));
                        let vb = if buka { 0.62 } else { 0.5 };
                        for q in 0..(bal / 2) {
                            let sq = spread(st0 + 2.0 * q as f32);
                            note_f(b, sq, r.range(0.0, h), CH_KEYS, tun.pitch(if q % 2 == 0 { a } else { c }), vb, 2.5);
                        }
                        if !buka {
                            for q in 0..bal {
                                let sq = spread(st0 + q as f32);
                                let n = if q % 2 == 0 { a } else { c };
                                note_f(b, sq, r.range(0.0, h), CH_KEYS, tun.pitch(n + 5), 0.36, 1.5);
                            }
                        }
                    }
                }
            }
            // the rebab: the seleh ahead, reached by a slide (or the suling
            // floating over the gongan's end)
            if m & L_LEAD != 0 {
                let goal = at((jj / 4) * 4 + 3);
                if buka || form == F_KETAWANG || soft {
                    let p = if jj % 4 == 3 {
                        fold(goal, 3, 10)
                    } else if jj % 2 == 1 {
                        fold(goal, 3, 10) + r.pick(&[1, -1, 0])
                    } else {
                        fold(next, 3, 10)
                    };
                    let len = bal as f32 * if jj % 4 == 3 { 1.5 } else { 1.0 };
                    let late = r.range(0.0, h * 2.0);
                    if r.chance(0.3) && jj % 4 != 3 {
                        // a grace from above
                        note_slide(b, st, late, CH_LEAD, tun.pitch(p + 1), 0.6, 1.0);
                        note_slide(b, st + 1.0, late, CH_LEAD, tun.pitch(p), 0.66, len - 1.0);
                    } else {
                        note_slide(b, st, late, CH_LEAD, tun.pitch(p), 0.66, len);
                    }
                }
            }
            // the colotomic instruments
            if (gong || (buka && last_bar && gong_beat)) && (m & L_DRONE != 0 || buka) {
                let t = fold(home, -14, -10);
                note_f(b, st, 0.0, CH_DRONE, tun.pitch(t), 0.95, 48.0);
            }
            if kem && m & L_DRONE != 0 && !in_buka {
                note_f(b, st, r.range(0.0, h), CH_DRONE, tun.pitch(fold(cur, -8, -4)), 0.62, bal as f32 * 4.0);
            }
            if (ken || (buka && last_bar && gong_beat)) && (m & L_PAD != 0 || buka) && !in_buka {
                note_f(b, st, r.range(0.0, h), CH_PAD, tun.pitch(fold(cur, 1, 6)), 0.6, bal as f32 * 6.0);
            }
            if m & L_PERC != 0 && !in_buka {
                if ketuk {
                    b.hit(st, r.range(0.0, h), CH_PERC, 0, 0.5);
                }
                if kempyang {
                    b.hit(st, r.range(0.0, h), CH_PERC, 1, 0.32);
                }
                if soft && (ketuk || kempyang) {
                    b.hit(spread(st0 + bal as f32 / 2.0), r.range(0.0, h), CH_PERC, 1, 0.26);
                }
            }
            // the kecer in the loud lancaran, on the off-beats
            if m & L_HAT != 0 && !in_buka {
                b.hit(spread(st0 + 2.0), r.range(0.0, h), CH_HAT, HAT_C, 0.32);
                if r.chance(0.3) {
                    b.hit(spread(st0 + 3.0), r.range(0.0, h), CH_HAT, HAT_C, 0.2);
                }
            }
        }
        // the kendang: its gatra pattern (half a gatra a bar in irama II),
        // the gatra into the gong's, and in the buka's last bar alone
        if m & (L_KICK | L_SNARE) != 0 || (buka && last_bar) {
            let into_gong = j0 + per_bar >= nb;
            let pat = if into_gong {
                self.idn.kend[2]
            } else if (j0 / 4) % 2 == 1 {
                self.idn.kend[1]
            } else {
                self.idn.kend[0]
            };
            let quiet = if soft { 0.8 } else { 1.0 };
            for (st, &c) in pat.iter().enumerate() {
                let sp = spread(st as f32);
                let late = self.rng.range(0.0, h);
                let v = quiet * if st % 4 == 0 { 0.8 } else { 0.62 };
                match c {
                    1 => b.hit(sp, late, CH_KICK, 0, v),
                    2 => {
                        b.hit(sp, late, CH_KICK, 1, v);
                        b.hit(sp, late, CH_SNARE, 0, v * 0.8);
                    }
                    3 => b.hit(sp, late, CH_SNARE, 0, v * 0.9),
                    4 => b.hit(sp, late, CH_SNARE, 1, v * 0.8),
                    5 => b.hit(sp, late, CH_SNARE, 2, v * 0.45),
                    _ => {}
                }
            }
        }
    }

    /// A bar of gong kebyar.
    fn bali_bar(&mut self, b: &mut Bar, sec: Sec, m: u32, x: ISec) {
        let g = self.idn.gongans[x.gongan.min(self.idn.gongans.len() - 1)].clone();
        let nb = g.len() as u32;
        let i = self.sec_bar;
        let bars = sec.bars;
        let last_bar = i + 1 == bars;
        let tun = self.idn.tun;
        let hz = self.idn.ombak;
        let h = self.human;
        let e = sec.energy;
        if x.part == PT_KEBYAR {
            self.kebyar(b, i, bars);
            return;
        }
        let j0 = match x.part {
            PT_END => (nb - 4 * bars.min(nb / 4) + i * 4) % nb,
            _ => (i * 4) % nb,
        };
        let cycle = (i * 4) / nb;
        let cycle_end = j0 + 4 >= nb;
        let angsel = (x.part == PT_END && last_bar)
            || (cycle_end
                && matches!(x.part, PT_PENGAWAK | PT_PENGECET)
                && !last_bar
                && self.idn.angsel.get(cycle as usize + self.sec * 7).copied().unwrap_or(false));
        let ending = x.part == PT_END && last_bar;
        let suling = x.part == PT_SULING;
        let at = |j: u32| -> i32 { g[(j % nb) as usize] };
        let spread = |st: f32| -> f32 {
            if ending {
                // (the bar's last hit still inside it)
                let y = st / 16.0;
                st + 0.9 * y * y
            } else {
                st
            }
        };
        let dyn_ = if suling { 0.55 } else { 0.75 + 0.25 * e };
        // where the angsel stops the music: the strokes it keeps
        let stop = |st: u32| angsel && st >= 6 && !matches!(st, 9 | 10 | 12);
        for k in 0..4u32 {
            let j = j0 + k;
            let jj = j % nb;
            let st0 = (k * 4) as f32;
            let st = spread(st0);
            let d = at(jj);
            let t = at(jj + 1);
            let (gong, kempur, kemong) = (jj + 1 == nb, jj + 1 == nb / 2, (jj + 1) % (nb / 4).max(1) == 0);
            let r = &mut self.rng;
            // the pokok: calung on every beat, jegogan every other
            if m & L_BASS != 0 && !stop(k * 4) {
                ombak_pair(b, st, r.range(0.0, h), CH_BASS, tun.pitch(fold(d, -1, 4)), 0.7 * dyn_, 3.8, hz);
            }
            if m & L_PAD != 0 && jj % 2 == 1 && !stop(k * 4) {
                ombak_pair(b, st, r.range(0.0, h), CH_PAD, tun.pitch(fold(d, -6, -2)), 0.72 * dyn_, 7.5, hz);
            }
            // the gangsa's kotekan: a figure of three tones about the coming
            // pokok note, polos the lower two, sangsih the upper two (the
            // middle both), in 16ths into it
            if m & (L_BELL | L_ARP) != 0 && !suling {
                let figs: &[[i32; 4]] = &[
                    [1, 0, -1, 0],
                    [-1, 0, 1, 0],
                    [0, 1, -1, 0],
                    [1, -1, 1, 0],
                    [0, -1, 1, 0],
                    [1, 0, 1, 0],
                    [-1, 1, -1, 0],
                ];
                let f = figs[((jj as usize) * 3 + x.gongan + (d.rem_euclid(5) as usize)) % figs.len()];
                let tg = fold(t, 3, 9);
                for (q, &o) in f.iter().enumerate() {
                    let sq = st0 + 1.0 + q as f32;
                    if stop(sq as u32) || (sq as u32 >= 16 && x.part == PT_END && last_bar) {
                        continue;
                    }
                    let n = tg + o;
                    let late = r.range(0.0, h * 0.5);
                    let v = (0.62 + if q == 3 { 0.1 } else { 0.0 }) * dyn_;
                    if o <= 0 && m & L_BELL != 0 {
                        ombak_pair(b, spread(sq), late, CH_BELL, tun.pitch(n), v, 1.2, hz);
                    }
                    if o >= 0 && m & L_ARP != 0 {
                        ombak_pair(b, spread(sq), late + 0.002, CH_ARP, tun.pitch(n), v * 0.95, 1.2, hz);
                    }
                }
            }
            // the reyong's norot: the coming note and the one above in turn,
            // in two octaves, the last 16th of the beat on the note
            if m & L_KEYS != 0 && !suling {
                let tg = fold(t, 4, 9);
                for q in 0..4u32 {
                    let sq = k * 4 + q + 1;
                    if stop(sq) || sq >= 16 && ending {
                        continue;
                    }
                    let n = if q % 2 == 0 { tg + 1 } else { tg };
                    let hi = q % 2 == 1;
                    let v = if q == 3 { 0.5 } else { 0.42 } * dyn_;
                    ombak_pair(
                        b,
                        spread(sq as f32),
                        r.range(0.0, h),
                        CH_KEYS,
                        tun.pitch(n + if hi { 5 } else { 0 }),
                        v,
                        1.4,
                        hz,
                    );
                }
            }
            // the ugal's melody (on the beat its note, between them the way
            // there), or the suling's long notes
            if m & L_LEAD != 0 && !stop(k * 4) {
                if suling {
                    if k % 2 == 0 {
                        let p = fold(t, 6, 13) + r.pick(&[0, 0, 1, -1]);
                        let late = r.range(0.0, 0.03);
                        if r.chance(0.4) {
                            note_slide(b, st, late, CH_LEAD, tun.pitch(p + 1), 0.55, 0.7);
                            note_slide(b, st + 0.7, late, CH_LEAD, tun.pitch(p), 0.6, 7.0);
                        } else {
                            note_slide(b, st, late, CH_LEAD, tun.pitch(p), 0.6, 7.6);
                        }
                    }
                } else {
                    let p = fold(d, 3, 9);
                    let q = fold(t, p - 2, p + 2);
                    note_f(b, st, r.range(0.0, h), CH_LEAD, tun.pitch(p), 0.66 * dyn_, 1.8);
                    let mid = if (q - p).abs() >= 2 { p + (q - p).signum() } else { p + r.pick(&[1, -1]) };
                    note_f(b, spread(st0 + 2.0), r.range(0.0, h), CH_LEAD, tun.pitch(mid), 0.58 * dyn_, 1.8);
                }
            }
            // the gong, kempur and kemong; the kempli on every beat
            if m & L_DRONE != 0 {
                if gong && !ending {
                    note_f(b, st, 0.0, CH_DRONE, tun.pitch(fold(self.idn.home, -14, -10)), 0.92, 48.0);
                } else if kempur {
                    note_f(b, st, r.range(0.0, h), CH_DRONE, tun.pitch(fold(d, -8, -4)), 0.6, 16.0);
                } else if kemong && !suling {
                    note_f(b, st, r.range(0.0, h), CH_DRONE, tun.pitch(fold(d, -3, 1)), 0.4, 8.0);
                }
            }
            if m & L_PERC != 0 && !stop(k * 4) {
                b.hit(st, r.range(0.0, h * 0.5), CH_PERC, 0, if suling { 0.3 } else { 0.44 });
            }
        }
        // the cengceng's 16ths, and the kendang pair's interlock
        let r = &mut self.rng;
        if m & L_HAT != 0 && !suling {
            for st in 0..16u32 {
                if stop(st) {
                    continue;
                }
                let v = self.idn.cek[st as usize] * (0.8 + 0.3 * e);
                b.hit(spread(st as f32), r.range(0.0, h * 0.5), CH_HAT, HAT_C, v);
            }
        }
        if m & (L_KICK | L_SNARE) != 0 {
            let pat =
                if suling { [1, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 1, 0] } else { self.idn.kend[(i % 2) as usize] };
            for (st, &c) in pat.iter().enumerate() {
                if stop(st as u32) {
                    continue;
                }
                let sp = spread(st as f32);
                let late = r.range(0.0, h * 0.5);
                let v = if suling {
                    0.45
                } else if st % 4 == 0 {
                    0.72
                } else {
                    0.55
                };
                match c {
                    1 => b.hit(sp, late, CH_KICK, 0, v),
                    2 => b.hit(sp, late, CH_KICK, 1, v * 0.8),
                    3 => b.hit(sp, late, CH_SNARE, 0, v),
                    4 => b.hit(sp, late, CH_SNARE, 1, v * 0.85),
                    5 => b.hit(sp, late, CH_SNARE, 2, v * 0.6),
                    _ => {}
                }
            }
        }
        // the angsel's strokes: everyone together, the cymbals open
        if angsel {
            let t = fold(self.idn.home, 3, 9);
            for st in [6u32, 9, 10, 12] {
                if ending && st != 12 && st != 6 {
                    continue;
                }
                let sp = spread(st as f32);
                let v = if st == 12 { 0.95 } else { 0.82 };
                self.byar(b, sp, t, v, ending && st == 12);
            }
        }
    }

    /// A byar: the whole gamelan's stroke (the gangsa, the reyong's
    /// cluster, the pokok, the cymbals; `gong`: the gong with it).
    fn byar(&mut self, b: &mut Bar, st: f32, t: i32, v: f32, gong: bool) {
        let tun = self.idn.tun;
        let hz = self.idn.ombak;
        let len = if gong { 24.0 } else { 3.0 };
        for (ch, n) in [(CH_BELL, t), (CH_BELL, t - 2), (CH_ARP, t + 2), (CH_ARP, t + 3)] {
            ombak_pair(b, st, 0.0, ch, tun.pitch(n), v * 0.8, len, hz);
        }
        for n in [t + 1, t + 2, t + 5] {
            ombak_pair(b, st, 0.002, CH_KEYS, tun.pitch(fold(n, 4, 12)), v * 0.6, len, hz);
        }
        ombak_pair(b, st, 0.0, CH_BASS, tun.pitch(fold(t, -1, 4)), v * 0.75, len, hz);
        if gong {
            note_f(b, st, 0.0, CH_DRONE, tun.pitch(fold(self.idn.home, -14, -10)), 0.95, 48.0);
            ombak_pair(b, st, 0.0, CH_PAD, tun.pitch(fold(t, -6, -2)), 0.8, 24.0, hz);
            note_f(b, st, 0.0, CH_LEAD, tun.pitch(fold(t, 3, 9)), 0.7, 12.0);
        }
        b.hit(st, 0.0, CH_CYM, 1, v * 0.8);
        b.hit(st, 0.0, CH_HAT, HAT_O, v * 0.7);
        b.hit(st, 0.0, CH_SNARE, 0, v * 0.8);
        b.hit(st, 0.0, CH_KICK, 0, v * 0.8);
    }

    /// The kebyar: byar strokes ringing, the ugal's free phrases, unison runs
    /// down the gangsa and reyong, accents in threes, a burst of kotekan
    /// stopped dead; the kendang leading into the pengawak.
    fn kebyar(&mut self, b: &mut Bar, i: u32, bars: u32) {
        let tun = self.idn.tun;
        let hz = self.idn.ombak;
        let home = self.idn.home;
        let t = fold(home, 3, 9);
        let last = i + 1 == bars;
        let fig = if i == 0 {
            0
        } else if last {
            4
        } else {
            1 + (i as usize + self.piece as usize) % 3
        };
        match fig {
            0 => {
                // the opening byar, ringing; the ugal's answer
                self.byar(b, 0.0, t, 0.95, true);
                let mut p = fold(home + 4, 5, 11);
                for (q, st) in [8.0, 9.5, 10.0, 11.5, 12.0, 14.0].iter().enumerate() {
                    note_f(b, *st, 0.0, CH_LEAD, tun.pitch(p), 0.7 - 0.03 * q as f32, 1.4);
                    p -= 1;
                }
            }
            1 => {
                // a unison run down from the top
                let mut p = t + 6;
                for st in 0..12 {
                    let v = if st % 3 == 0 { 0.85 } else { 0.62 };
                    ombak_pair(b, st as f32, 0.0, CH_BELL, tun.pitch(p), v, 1.2, hz);
                    ombak_pair(b, st as f32, 0.002, CH_ARP, tun.pitch(p), v * 0.9, 1.2, hz);
                    ombak_pair(b, st as f32, 0.003, CH_KEYS, tun.pitch(fold(p, 4, 13)), v * 0.6, 1.2, hz);
                    if st % 3 == 0 {
                        b.hit(st as f32, 0.0, CH_HAT, HAT_C, 0.5);
                        b.hit(st as f32, 0.0, CH_SNARE, 0, 0.6);
                    }
                    if st % 2 == 1 {
                        p -= 1;
                    }
                }
                self.byar(b, 12.0, fold(p, 3, 9), 0.9, false);
            }
            2 => {
                // accents in threes, the reyong between
                for (k, st) in [0u32, 3, 6, 9, 12].iter().enumerate() {
                    let n = t + [0, 2, -1, 1, 0][k];
                    self.byar(b, *st as f32, n, 0.8 + 0.03 * k as f32, false);
                    ombak_pair(b, *st as f32 + 1.0, 0.0, CH_KEYS, tun.pitch(fold(n + 1, 4, 13)), 0.45, 0.8, hz);
                    ombak_pair(b, *st as f32 + 2.0, 0.0, CH_KEYS, tun.pitch(fold(n, 4, 13)), 0.45, 0.8, hz);
                }
            }
            3 => {
                // a burst of kotekan, stopped dead (it rings)
                let tg = fold(home + 1, 3, 9);
                for st in 0..10u32 {
                    let o = [1, 0, -1, 0, 1, -1, 1, 0, -1, 0][st as usize];
                    let n = tg + o;
                    if o <= 0 {
                        ombak_pair(b, st as f32, 0.0, CH_BELL, tun.pitch(n), 0.66, 1.1, hz);
                    }
                    if o >= 0 {
                        ombak_pair(b, st as f32, 0.002, CH_ARP, tun.pitch(n), 0.62, 1.1, hz);
                    }
                    b.hit(st as f32, 0.0, CH_HAT, HAT_C, 0.4);
                    if st % 2 == 0 {
                        b.hit(st as f32, 0.0, CH_KICK, (st % 4 == 2) as u8, 0.55);
                    } else {
                        b.hit(st as f32, 0.0, CH_SNARE, 1, 0.5);
                    }
                }
                self.byar(b, 10.0, tg, 0.95, false);
                note_f(b, 12.0, 0.0, CH_LEAD, tun.pitch(tg + 2), 0.6, 3.5);
            }
            _ => {
                // the kendang leads in, the ugal on the pokok's first notes
                let g = &self.idn.gongans[0];
                for k in 0..4u32 {
                    let st = (k * 4) as f32;
                    b.hit(st, 0.0, CH_KICK, 0, 0.7);
                    b.hit(st + 2.0, 0.0, CH_SNARE, 1, 0.6);
                    b.hit(st + 3.0, 0.0, CH_SNARE, 0, 0.5);
                    let n = fold(g[(k as usize + 12) % g.len()], 3, 9);
                    note_f(b, st, 0.0, CH_LEAD, tun.pitch(n), 0.62, 3.6);
                    b.hit(st, 0.0, CH_PERC, 0, 0.4);
                }
                ombak_pair(b, 12.0, 0.0, CH_BASS, tun.pitch(fold(home, -1, 4)), 0.7, 4.0, hz);
            }
        }
    }

    /// The gamelan's drums under another genre's music, in its meter: the
    /// kendang's strokes (the low one on the last beat, the slaps between),
    /// the ketuk and kempyang, the gong closing every fourth bar (on the
    /// next one's downbeat, where the writer's phrase turns); a Balinese
    /// drummer's interlock and cengceng in 16ths.
    pub(super) fn indonesian_beat(&mut self, _s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let (bs, beat) = (self.bar_steps(), self.beat_steps());
        let beats = (bs / beat).max(1);
        if !self.idn.dready {
            self.idn.dready = true;
            let r = &mut self.rng;
            self.idn.dstyle = if r.chance(0.6) { IDN_JAVA } else { IDN_BALI };
            for v in 0..2 {
                self.idn.dpat[v] = kendang_pattern(r, v == 1, false);
            }
        }
        let (e, i, h) = (sec.energy, self.sec_bar, self.human);
        let full = matches!(sec.kind, MAIN | BUILD | OUTRO);
        let bali = self.idn.dstyle == IDN_BALI;
        let quiet = if sec.kind == BREAK && m & (L_KICK | L_HAT) == 0 { 0.6 } else { 1.0 };
        let r = &mut self.rng;
        // the gong where a phrase turns, a kempul halfway
        if i % 4 == 0 && (full || m & L_KICK != 0 || i == 0) {
            b.hit(0.0, 0.0, CH_CYM, 0, 0.85 * quiet);
        }
        let sub = if beat == 6 { 2 } else { 1 };
        let lastb = (beats - 1) * beat;
        if bali && (full || m & (L_HAT | L_SNARE) != 0) {
            // the interlock: the wadon's low strokes and the lanang's high ones in turn
            let p = self.idn.dpat[(i % 2) as usize];
            let mut k = 0;
            let mut st = 0;
            while st < bs {
                let c = p[k % 16];
                let v = quiet * if st % beat == 0 { 0.7 } else { 0.5 };
                match c {
                    1 | 2 => b.hit(st as f32, r.range(0.0, h), CH_KICK, (c == 2) as u8, v),
                    3 | 4 => b.hit(st as f32, r.range(0.0, h), CH_SNARE, c - 3, v * 0.9),
                    5 => b.hit(st as f32, r.range(0.0, h), CH_SNARE, 2, v * 0.5),
                    _ => {}
                }
                b.hit(
                    st as f32,
                    r.range(0.0, h * 0.5),
                    CH_HAT,
                    HAT_C,
                    quiet * (0.25 + 0.15 * (st % 2 == 0) as u8 as f32) * (0.8 + 0.3 * e),
                );
                st += sub;
                k += 1;
            }
            for k in 0..beats {
                b.hit((k * beat) as f32, r.range(0.0, h), CH_PERC, 0, 0.4 * quiet);
            }
            return;
        }
        // Javanese: the kendang's gatra pattern laid over the bar's beats
        // (its last on the bar's last beat), the ketuk between the beats,
        // the kempyang on them
        let p = self.idn.dpat[if full && e >= 0.8 { 1 } else { 0 }];
        for k in 0..beats {
            let at = k * beat;
            // the pattern's beat that lands here: the bar's last beat takes the pattern's last
            let pb = if k + 1 == beats { 3 } else { (k * 3 / beats.max(2)).min(2) };
            for q in 0..4u32 {
                let st = at as f32 + q as f32 * beat as f32 / 4.0;
                if st >= bs as f32 {
                    continue;
                }
                let c = p[(pb * 4 + q) as usize];
                let v = quiet * if q == 0 { 0.75 } else { 0.55 };
                let late = r.range(0.0, h);
                match c {
                    1 => b.hit(st, late, CH_KICK, 0, v),
                    2 => {
                        b.hit(st, late, CH_KICK, 1, v);
                        b.hit(st, late, CH_SNARE, 0, v * 0.7);
                    }
                    3 => b.hit(st, late, CH_SNARE, 0, v * 0.85),
                    4 => b.hit(st, late, CH_SNARE, 1, v * 0.8),
                    5 => b.hit(st, late, CH_SNARE, 2, v * 0.4),
                    _ => {}
                }
            }
            if full || m & L_PERC != 0 {
                b.hit(at as f32, r.range(0.0, h), CH_PERC, 1, 0.3 * quiet);
                let off = at + beat / 2;
                if off < bs {
                    b.hit(off as f32, r.range(0.0, h), CH_PERC, 0, 0.42 * quiet);
                }
            }
        }
        // something every bar (a sparse section's pattern can come up empty)
        b.hit(lastb as f32, r.range(0.0, h), CH_KICK, 0, 0.7 * quiet);
    }
}

// -- the instruments

/// A bronze bar (saron, demung, peking, gangsa, ugal): FM at a free bar's
/// partials (1 : 2.76 : 5.40), the clang dying first, then the ring.
fn bar_synth(r: &mut Rng, bright: f32, dec: f32, gain: f32) -> Patch {
    Patch {
        filter: false,
        fm: r.range(1.1, 1.5) * bright,
        ratio: 2.76,
        fm_dec: 0.45,
        fm2: 0.45 * bright,
        ratio2: 5.40,
        fm2_dec: 0.12,
        a: 0.001,
        d: dec,
        s: 0.0,
        r: 0.12,
        gain,
        ..PATCH
    }
}

/// The glockenspiel's recording (steel bars: a free bar's partials, as the
/// gamelan's bronze ones), played down a fifth to two octaves: as a saron's
/// or gangsa's, its ring the longer.
fn glock_bar(p: Patch, note: f32) -> Patch {
    let q = Patch { cutoff: 2000.0, keytrack: 0.8, ..p };
    rec_lp(q, multi_or("glockenspiel", "hit", "Glockenspiel"), note, 1.0, 0.0)
}

fn saron_synth(r: &mut Rng) -> Patch {
    bar_synth(r, 1.0, 3.5, 0.3)
}

/// The saron and demung (the balungan, in octaves): the glockenspiel down.
fn saron(r: &mut Rng) -> Patch {
    glock_bar(saron_synth(r), 72.0)
}

fn peking_synth(r: &mut Rng) -> Patch {
    bar_synth(r, 1.3, 2.5, 0.26)
}

/// The peking (the saron panerus): the glockenspiel as it is.
fn peking(r: &mut Rng) -> Patch {
    glock_bar(peking_synth(r), 84.0)
}

fn gangsa_synth(r: &mut Rng) -> Patch {
    Patch { r: 0.09, ..bar_synth(r, 1.4, 2.8, 0.26) }
}

/// The gangsa (and the ugal): the glockenspiel, damped after each 16th.
fn gangsa(r: &mut Rng) -> Patch {
    glock_bar(gangsa_synth(r), 76.0)
}

/// A thin bar over a tube (slenthem, gender, calung, jegogan): soft, long,
/// nearly pure (the synthesiser's).
fn tube_bar_synth(r: &mut Rng, dec: f32, gain: f32) -> Patch {
    Patch {
        filter: false,
        fm: r.range(0.3, 0.5),
        ratio: 2.76,
        fm_dec: 0.4,
        fm2: 0.1,
        ratio2: 1.003,
        fm2_dec: 3.0,
        a: 0.004,
        d: dec,
        s: 0.0,
        r: 0.4,
        gain,
        ..PATCH
    }
}

fn slenthem_synth(r: &mut Rng) -> Patch {
    tube_bar_synth(r, 6.0, 0.34)
}

/// The slenthem and the Balinese calung and jegogan: the vibraphone's
/// recording (bars over tubes, as theirs), its motor off.
fn slenthem(r: &mut Rng) -> Patch {
    rec(slenthem_synth(r), multi_or("vibraphone", "soft", "Vibraphone"), 52.0)
}

fn gender_synth(r: &mut Rng) -> Patch {
    Patch { r: 0.5, ..tube_bar_synth(r, 5.0, 0.3) }
}

/// The gender: the vibraphone, damped as the next note comes.
fn gender(r: &mut Rng) -> Patch {
    rec(gender_synth(r), multi_or("vibraphone", "soft", "Vibraphone"), 64.0)
}

/// A bossed gong (kenong, kempul, gong ageng, bonang, reyong): FM at the
/// gong's partials, the near-unison one beating against the note (the
/// ombak a gong has of its own).
fn gong_synth(r: &mut Rng, dec: f32, idx: f32, gain: f32) -> Patch {
    Patch {
        filter: false,
        cutoff: 2500.0,
        keytrack: 0.7,
        fm: idx * r.range(0.8, 1.2),
        ratio: 2.46,
        fm_dec: 0.5,
        fm2: 0.6,
        ratio2: 2.04,
        fm2_dec: dec * 0.5,
        a: 0.002,
        d: dec,
        s: 0.0,
        r: 0.4,
        gain,
        ..PATCH
    }
}

/// The bossed gong's recording (VCSL's Gong 1, tuned to C4), at any of the
/// gongs' pitches: the kettles' (bonang, reyong, kenong) above it under a
/// low-pass (the higher a recording is played, the brighter its noise), the
/// hanging gongs' below.
fn gong_rec(p: Patch, note: f32, lp: bool) -> Patch {
    let id = samples::multi("gong", "ring");
    if lp { rec_lp(p, id, note, 1.0, 0.0) } else { rec(p, id, note) }
}

fn bonang_synth(r: &mut Rng) -> Patch {
    Patch { r: 0.3, ..gong_synth(r, 2.2, 0.7, 0.3) }
}

/// The bonang (barung and panerus) and the reyong: small kettle gongs.
fn bonang(r: &mut Rng) -> Patch {
    gong_rec(bonang_synth(r), 72.0, true)
}

fn kenong_synth(r: &mut Rng) -> Patch {
    gong_synth(r, 5.0, 0.5, 0.32)
}

fn kenong(r: &mut Rng) -> Patch {
    gong_rec(kenong_synth(r), 66.0, true)
}

fn gong_ageng_synth(r: &mut Rng) -> Patch {
    Patch { r: 2.0, ..gong_synth(r, 14.0, 0.35, 0.45) }
}

/// The gong ageng and the kempul (one channel: the kempul's notes higher).
fn gong_ageng(r: &mut Rng) -> Patch {
    gong_rec(gong_ageng_synth(r), 48.0, false)
}

fn gambang_synth(r: &mut Rng) -> Patch {
    Patch {
        filter: false,
        fm: r.range(1.0, 1.6),
        ratio: 2.76,
        fm_dec: 0.04,
        a: 0.001,
        d: r.range(0.35, 0.5),
        s: 0.0,
        r: 0.15,
        gain: 0.32,
        ..PATCH
    }
}

/// The gambang (teak bars over a trough): the balafon's recording.
fn gambang(r: &mut Rng) -> Patch {
    rec(gambang_synth(r), multi_or("balafon", "hit", "Marimba"), 72.0)
}

fn rebab_synth(r: &mut Rng) -> Patch {
    Patch {
        unison: 2,
        detune: 5.0,
        width: 0.15,
        vowel: 3,
        noise: 0.03,
        cutoff: 2400.0,
        keytrack: 0.3,
        a: 0.06,
        d: 0.3,
        s: 0.9,
        r: 0.2,
        vib: r.range(0.12, 0.2),
        vib_rate: 5.2,
        glide: 0.07,
        gain: 0.45,
        ..PATCH
    }
}

/// The rebab (a two-string spike fiddle): the solo violin's recording through
/// the patch's nasal vowel and low-pass, sliding as the rebab does.
fn rebab(r: &mut Rng) -> Patch {
    let p = rebab_synth(r);
    let keep = |q| Patch { vowel: p.vowel, filter: true, cutoff: p.cutoff, keytrack: p.keytrack, glide: p.glide, ..q };
    rec_keep(p, multi_or("violin", "sus", "Fiddle"), 74.0, keep)
}

fn suling_synth(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Sine,
        wave2: Wave::Tri,
        mix2: 0.25,
        noise: r.range(0.07, 0.1),
        cutoff: 3200.0,
        keytrack: 0.3,
        a: 0.07,
        d: 0.5,
        s: 0.85,
        r: 0.25,
        vib: 0.1,
        vib_rate: 4.8,
        glide: 0.05,
        gain: 0.3,
        ..PATCH
    }
}

/// The suling (a ring-stopped bamboo flute): the shakuhachi's recording
/// (end-blown bamboo, as breathy), else the flute's.
fn suling(r: &mut Rng) -> Patch {
    let id = samples::gm("Shakuhachi").or_else(flute_rec);
    rec(suling_synth(r), id, 79.0)
}

/// The kendang's low head: [dhung (open), dhah (muted)].
fn kit_kendang_lo_synth(r: &mut Rng) -> Vec<DrumPatch> {
    let open = DrumPatch {
        tone: 1.0,
        f0: r.range(110.0, 125.0),
        f1: r.range(72.0, 82.0),
        pdec: 0.05,
        tdec: 0.42,
        noise: 0.25,
        nmode: 0,
        ncut: 700.0,
        ndec: 0.05,
        gain: 0.85,
        ..DRUM
    };
    let muted = DrumPatch { tdec: 0.12, noise: 0.5, nmode: 1, ncut: 1300.0, ndec: 0.04, gain: 0.7, ..open };
    vec![open, muted]
}

/// The kendang's low head: a tumba's open and muted strokes.
fn kit_kendang_lo(r: &mut Rng) -> Vec<DrumPatch> {
    rec_kit_db(
        kit_kendang_lo_synth(r),
        &[(samples::multi("conga", "tumba"), 5.2), (samples::multi("conga", "tumba_muted"), 5.2)],
    )
}

/// The kendang's high head: [tak (a slap), tung (open), ket (a small damped one)].
fn kit_kendang_hi_synth(r: &mut Rng) -> Vec<DrumPatch> {
    let tak = DrumPatch {
        tone: 0.6,
        f0: r.range(420.0, 480.0),
        f1: 400.0,
        pdec: 0.01,
        tdec: 0.05,
        noise: 0.7,
        nmode: 1,
        ncut: 2600.0,
        nres: 0.25,
        ndec: 0.03,
        gain: 0.6,
        ..DRUM
    };
    let tung = DrumPatch {
        tone: 1.0,
        f0: 340.0,
        f1: 310.0,
        pdec: 0.03,
        tdec: 0.28,
        noise: 0.12,
        ncut: 1500.0,
        gain: 0.6,
        ..tak
    };
    let ket = DrumPatch { tone: 0.8, f0: 520.0, f1: 500.0, tdec: 0.035, noise: 0.4, gain: 0.45, ..tak };
    vec![tak, tung, ket]
}

/// The kendang's high head: a bongo's muted slap, a quinto's open stroke, a conga's muted one.
fn kit_kendang_hi(r: &mut Rng) -> Vec<DrumPatch> {
    rec_kit_db(
        kit_kendang_hi_synth(r),
        &[
            (samples::multi("bongo", "high_muted"), 0.5),
            (samples::multi("conga", "quinto"), 0.5),
            (samples::multi("conga", "muted"), 0.5),
        ],
    )
}

/// The cengceng (or the kecer): [cek (closed), ceng (open)].
fn kit_cengceng_synth(r: &mut Rng) -> Vec<DrumPatch> {
    hats(r, 6500.0, 0.05, 0.4, 0.7, 0.5)
}

/// The cengceng: a hi-hat's closed strokes, a clash cymbal's short ones.
fn kit_cengceng(r: &mut Rng) -> Vec<DrumPatch> {
    rec_kit_db(
        kit_cengceng_synth(r),
        &[(samples::multi("hihat", "closed"), -10.0), (samples::multi("cymbal", "clash_short"), -8.0)],
    )
}

/// The ketuk (and Bali's kempli) and the kempyang: small damped kettles.
fn kit_ketuk_synth(r: &mut Rng) -> Vec<DrumPatch> {
    let ketuk = DrumPatch {
        tone: 1.0,
        f0: r.range(860.0, 940.0),
        f1: 880.0,
        pdec: 0.5,
        tdec: 0.2,
        tone2: 0.45,
        ratio2: 2.42,
        noise: 0.1,
        nmode: 1,
        ncut: 3000.0,
        ndec: 0.01,
        gain: 0.42,
        ..DRUM
    };
    let ketuk = DrumPatch { f1: ketuk.f0, ..ketuk };
    let kempyang = DrumPatch { f0: 1460.0, f1: 1460.0, tdec: 0.32, tone2: 0.35, ratio2: 2.71, gain: 0.32, ..ketuk };
    vec![ketuk, kempyang]
}

/// The ketuk and kempyang: the agogo bells' low and high bells.
fn kit_ketuk(r: &mut Rng) -> Vec<DrumPatch> {
    rec_kit_db(kit_ketuk_synth(r), &[(samples::multi("agogo", "low"), 3.6), (samples::multi("agogo", "high"), 3.6)])
}

/// The gong and the kebyar's clashed cymbals, as drums: [gong, clash].
fn kit_gong_synth(r: &mut Rng) -> Vec<DrumPatch> {
    let gong = DrumPatch {
        tone: 1.0,
        f0: r.range(46.0, 50.0),
        f1: 45.0,
        pdec: 0.3,
        tdec: 9.0,
        tone2: 0.5,
        ratio2: 1.028,
        noise: 0.08,
        nmode: 0,
        ncut: 400.0,
        ndec: 0.3,
        gain: 0.9,
        ..DRUM
    };
    let gong = DrumPatch { f1: gong.f0 * 0.98, ..gong };
    let mut v = vec![gong];
    v.extend(crash(r));
    v
}

/// The gong as a drum (a mix's): the synthesiser's (a recorded hit plays at
/// one pitch, and the gong's is a kempul's, not a gong ageng's); the clash
/// cymbal's recording.
fn kit_gong(r: &mut Rng) -> Vec<DrumPatch> {
    rec_kit_db(kit_gong_synth(r), &[(None, 0.0), (samples::multi("cymbal", "clash"), 5.2)])
}

// The recordings' trims (dB), against the synthesiser's patch each
// replaces: each channel alone over whole pieces (6 seeds' first 90 s, the
// leveller frozen), K-weighted mean power against the synthesiser's twin
// playing the same notes; the swapped parts (bonang and gender, peking and
// gambang, rebab and suling) measured with each forced in. Under half a dB
// left out.

/// Javanese.
const JAVA_REC_TRIMS: &[(usize, &str, f32)] = &[
    (CH_BELL, "glockenspiel:hit", -1.3),
    (CH_ARP, "glockenspiel:hit", -0.6),
    (CH_ARP, "balafon:hit", -1.3),
    (CH_KEYS, "gong:ring", -2.9),
    (CH_KEYS, "vibraphone:soft", -0.8),
    (CH_PAD, "gong:ring", -1.7),
    (CH_DRONE, "gong:ring", -3.6),
    (CH_LEAD, "violin:sus", 3.0),
    (CH_LEAD, "Shakuhachi", 3.0),
];

/// Balinese.
const BALI_REC_TRIMS: &[(usize, &str, f32)] = &[
    (CH_BELL, "glockenspiel:hit", -3.2),
    (CH_ARP, "glockenspiel:hit", -2.1),
    (CH_KEYS, "gong:ring", -1.5),
    (CH_PAD, "vibraphone:soft", 2.6),
    (CH_BASS, "vibraphone:soft", -1.6),
    (CH_DRONE, "gong:ring", -3.0),
    (CH_LEAD, "glockenspiel:hit", -2.4),
    (CH_LEAD, "Shakuhachi", 3.2),
];

/// The gamelan's instruments lent to another genre's parts, in a mix (at its pitches).
pub(super) const P_IDN_BARS: &[(&str, Make)] =
    &[("saron", saron), ("peking", peking), ("gangsa", gangsa), ("gender", gender), ("gambang", gambang)];
pub(super) const P_IDN_KETTLES: &[(&str, Make)] = &[("bonang", bonang), ("kenong", kenong), ("gender", gender)];
pub(super) const P_IDN_LEAD: &[(&str, Make)] = &[("suling", suling), ("rebab", rebab), ("ugal", gangsa)];
pub(super) const P_IDN_LOW: &[(&str, Make)] = &[("slenthem", slenthem), ("jegogan", slenthem)];
pub(super) const P_IDN_GONG: &[(&str, Make)] = &[("kempul", gong_ageng), ("kenong", kenong)];
pub(super) const K_IDN_KICK: &[(&str, MakeKit)] = &[("kendang (low head)", kit_kendang_lo)];
pub(super) const K_IDN_SNARE: &[(&str, MakeKit)] = &[("kendang (high head)", kit_kendang_hi)];
pub(super) const K_IDN_HAT: &[(&str, MakeKit)] = &[("cengceng", kit_cengceng)];
pub(super) const K_IDN_PERC: &[(&str, MakeKit)] = &[("ketuk and kempyang", kit_ketuk)];
pub(super) const K_IDN_CYM: &[(&str, MakeKit)] = &[("gong and cymbals", kit_gong)];

/// The band: the parts that swap between the loud style and the soft (the
/// bonang and the gender, the peking and the gambang, the rebab or ugal and
/// the suling), set by the composer as each section begins; the balungan's
/// and the bass's second instruments for the sections `dress` gives them.
/// The colotomic gongs stay out of it (a mix keeps them gongs).
fn band_indonesian(r: &mut Rng, s: &Synth, idn: &mut Indonesian) -> Band {
    let mut b = Band::default();
    let java = idn.style == IDN_JAVA;
    let lv = |ch: usize, note: f32, p: Patch| levelled(&s.ch[ch].patch, note, p);
    let keys = s.ch[CH_KEYS].patch;
    let arp = s.ch[CH_ARP].patch;
    let lead = s.ch[CH_LEAD].patch;
    let soft_keys = if java { lv(CH_KEYS, 64.0, gender(r)) } else { keys };
    let soft_arp = if java { lv(CH_ARP, 72.0, gambang(r)) } else { arp };
    let soft_lead = lv(CH_LEAD, 79.0, suling(r));
    idn.loud = vec![keys, arp, lead];
    idn.soft = vec![soft_keys, soft_arp, soft_lead];
    b.tones.push((CH_KEYS, [keys, soft_keys], 64.0));
    b.tones.push((CH_ARP, [arp, soft_arp], 72.0));
    b.tones.push((CH_LEAD, [lead, soft_lead], 76.0));
    let bell = s.ch[CH_BELL].patch;
    let bell2 = lv(CH_BELL, 72.0, if java { saron(r) } else { gangsa(r) });
    b.tones.push((CH_BELL, [bell, bell2], 72.0));
    let bass = s.ch[CH_BASS].patch;
    b.tones.push((CH_BASS, [bass, lv(CH_BASS, 52.0, slenthem(r))], 52.0));
    b.kit(r, s, CH_KICK, K_IDN_KICK);
    b.kit(r, s, CH_SNARE, K_IDN_SNARE);
    b.kit(r, s, CH_HAT, K_IDN_HAT);
    b.kit(r, s, CH_PERC, K_IDN_PERC);
    b
}

/// The gamelan's sounds (a mix's drummer's too, a style drawn): the bronze,
/// the gongs, the drums; a pavilion's air (a pendopo: open, the roof's
/// reflections, a moderate tail), the bronze spread across it.
pub(super) fn sounds_indonesian(r: &mut Rng, s: &mut Synth) {
    let style = if r.chance(0.62) { IDN_JAVA } else { IDN_BALI };
    sounds_gamelan(r, s, style)
}

fn sounds_gamelan(r: &mut Rng, s: &mut Synth, style: u8) {
    let java = style == IDN_JAVA;
    let ch = &mut s.ch;
    ch[CH_BELL].patch = if java { saron(r) } else { gangsa(r) };
    ch[CH_ARP].patch = if java { peking(r) } else { gangsa(r) };
    ch[CH_KEYS].patch = bonang(r);
    ch[CH_PAD].patch = if java { kenong(r) } else { slenthem(r) };
    ch[CH_BASS].patch = slenthem(r);
    ch[CH_DRONE].patch = gong_ageng(r);
    ch[CH_LEAD].patch = if java { rebab(r) } else { gangsa(r) };
    for c in [CH_BELL, CH_ARP, CH_KEYS, CH_PAD, CH_BASS, CH_DRONE, CH_LEAD] {
        ch[c].mono = false;
        ch[c].swell = 0.0;
        ch[c].width = 1.0;
    }
    ch[CH_LEAD].mono = true;
    ch[CH_KICK].drums = kit_kendang_lo(r);
    ch[CH_SNARE].drums = kit_kendang_hi(r);
    ch[CH_HAT].drums = kit_cengceng(r);
    ch[CH_PERC].drums = kit_ketuk(r);
    ch[CH_CYM].drums = kit_gong(r);
    // as the gamelan sits: the balungan to one side, the elaborating
    // instruments to the other, the gongs at the back in the middle
    // (measured against Javanese and Balinese records: the Javanese's
    // 1-3 kHz, the bonang's and peking's, to the fore, its low end and top
    // lighter; both near mono)
    let (bell_pan, arp_pan) = if java { (-0.2, 0.18) } else { (-0.35, 0.35) };
    mix(&mut ch[CH_BELL], 0.42, bell_pan, 0.25, 0.0, 0.0);
    mix(&mut ch[CH_ARP], if java { 0.42 } else { 0.34 }, arp_pan, 0.25, 0.0, 0.0);
    mix(&mut ch[CH_KEYS], if java { 0.42 } else { 0.34 }, if java { 0.25 } else { 0.1 }, 0.25, 0.0, 0.0);
    mix(&mut ch[CH_PAD], 0.36, if java { -0.1 } else { -0.2 }, 0.3, 0.0, 0.0);
    mix(&mut ch[CH_BASS], if java { 0.2 } else { 0.3 }, if java { -0.1 } else { 0.05 }, 0.25, 0.0, 0.0);
    mix(&mut ch[CH_DRONE], if java { 0.3 } else { 0.38 }, 0.05, 0.35, 0.0, 0.0);
    mix(&mut ch[CH_LEAD], 0.4, if java { 0.1 } else { 0.0 }, 0.28, 0.02, 0.0);
    mix(&mut ch[CH_KICK], 0.5, 0.05, 0.15, 0.0, 0.0);
    mix(&mut ch[CH_SNARE], 0.42, 0.05, 0.15, 0.0, 0.0);
    mix(&mut ch[CH_HAT], if java { 0.1 } else { 0.24 }, 0.2, 0.18, 0.0, 0.0);
    mix(&mut ch[CH_PERC], 0.36, -0.25, 0.25, 0.0, 0.0);
    mix(&mut ch[CH_CYM], 0.5, 0.0, 0.3, 0.0, 0.0);
    // the recordings' own stereo narrowed (the records are near mono)
    for c in [CH_BELL, CH_ARP, CH_KEYS, CH_PAD, CH_BASS, CH_DRONE, CH_LEAD] {
        ch[c].width = if ch[c].patch.samp != 0 { 0.3 } else { 0.7 };
    }
    s.rev.fb = r.range(0.8, 0.85);
    s.rev.damp = 0.4;
    s.rev_wet = r.range(0.2, 0.26);
    s.dly.fb = 0.0;
    s.dly_rev = 0.0;
    s.r2mode = R2_OFF;
    s.fx.crackle = 0.0;
    s.fx.wind_to = 0.0;
    s.fx.rain_to = 0.0;
    s.m.gain = 2.0;
    s.m.target = 0.22;
}
