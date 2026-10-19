//! CELTIC: the Irish and Scottish traditions as they are played now (the
//! tavern's are older: its jigs and reels a medieval band's). Three styles.
//! The session: a set of two or three tunes of one kind (reels, jigs,
//! hornpipes with their dotted swing, polkas, slides), each played AABB and
//! usually twice through, the key changing between them and the set lifting
//! at each change; the fiddle leading, the flute or whistle and the button
//! accordion, concertina or tenor banjo on the tune with it (each its own
//! ornaments: cuts, taps, rolls, trebles), the guitar in DADGAD or the
//! bouzouki strumming open-string chords with the off-beats leant on, the
//! bodhrán driving the backbeat with its rolling triplets. The pipe set:
//! the Highland pipes and their drones (two tenors and the bass), a march,
//! a strathspey (its Scotch snaps) and a reel, the gracenotes and doublings
//! between every note, the side drums' rolls, the tenor and bass drums. The
//! air: a slow air or an O'Carolan planxty on the harp, a fiddle, flute or
//! the uilleann pipes on the tune, the harp's rolled chords and arpeggios
//! under it, freely timed.

use super::*;

const CS_SESSION: u8 = 0;
const CS_PIPES: u8 = 1;
const CS_AIR: u8 = 2;

/// The tune types.
const CT_REEL: u8 = 0;
const CT_JIG: u8 = 1;
const CT_HORNPIPE: u8 = 2;
const CT_POLKA: u8 = 3;
const CT_SLIDE: u8 = 4;
const CT_MARCH: u8 = 5;
const CT_MARCH68: u8 = 6;
const CT_STRATHSPEY: u8 = 7;
const CT_AIR: u8 = 8;
const CT_PLANXTY: u8 = 9;

/// The ornaments (a `PNote`'s grace): a cut (a grace a step or two above),
/// a roll, a tap (a grace below), a treble (the note bowed three times).
const OR_ROLL: i8 = 9;
const OR_TAP: i8 = -2;
const OR_TREBLE: i8 = 11;

/// A tune type's 16ths a bar and a beat, and its name.
fn ct_meter(ty: u8) -> (u32, u32, &'static str) {
    match ty {
        CT_REEL => (16, 4, "4/4 reel"),
        CT_JIG => (12, 6, "6/8 jig"),
        CT_HORNPIPE => (16, 4, "4/4 hornpipe"),
        CT_POLKA => (8, 4, "2/4 polka"),
        CT_SLIDE => (24, 6, "12/8 slide"),
        CT_MARCH => (8, 4, "2/4 pipe march"),
        CT_MARCH68 => (12, 6, "6/8 pipe march"),
        CT_STRATHSPEY => (16, 4, "4/4 strathspey"),
        CT_AIR => (12, 4, "3/4 slow air"),
        _ => (12, 4, "3/4 planxty"),
    }
}

/// A beat's rhythms for a tune type (16th steps, lengths) and how often each comes.
fn ct_cells(ty: u8) -> &'static [(&'static [(u16, u16)], f32)] {
    match ty {
        CT_JIG | CT_MARCH68 => &[
            (&[(0, 2), (2, 2), (4, 2)], 0.6),
            (&[(0, 4), (4, 2)], 0.25),
            (&[(0, 2), (2, 1), (3, 1), (4, 2)], 0.08),
            (&[(0, 6)], 0.07),
        ],
        CT_SLIDE => &[(&[(0, 4), (4, 2)], 0.45), (&[(0, 2), (2, 2), (4, 2)], 0.4), (&[(0, 6)], 0.15)],
        CT_HORNPIPE => &[
            (&[(0, 2), (2, 2)], 0.62),
            (&[(0, 4)], 0.13),
            (&[(0, 2), (2, 1), (3, 1)], 0.15),
            (&[(0, 1), (1, 1), (2, 2)], 0.1),
        ],
        CT_POLKA => &[
            (&[(0, 2), (2, 2)], 0.55),
            (&[(0, 1), (1, 1), (2, 2)], 0.2),
            (&[(0, 4)], 0.15),
            (&[(0, 2), (2, 1), (3, 1)], 0.1),
        ],
        CT_MARCH => &[(&[(0, 3), (3, 1)], 0.45), (&[(0, 2), (2, 2)], 0.3), (&[(0, 4)], 0.15), (&[(0, 1), (1, 3)], 0.1)],
        CT_STRATHSPEY => {
            &[(&[(0, 1), (1, 3)], 0.35), (&[(0, 3), (3, 1)], 0.35), (&[(0, 2), (2, 2)], 0.15), (&[(0, 4)], 0.15)]
        }
        CT_AIR => &[(&[(0, 4)], 0.45), (&[(0, 2), (2, 2)], 0.3), (&[(0, 3), (3, 1)], 0.25)],
        CT_PLANXTY => &[(&[(0, 2), (2, 2)], 0.5), (&[(0, 4)], 0.3), (&[(0, 3), (3, 1)], 0.2)],
        _ => {
            &[(&[(0, 2), (2, 2)], 0.66), (&[(0, 4)], 0.1), (&[(0, 2), (2, 1), (3, 1)], 0.14), (&[(0, 3), (3, 1)], 0.1)]
        }
    }
}

/// Eight-bar parts' chords by mode: A parts, then B (higher, away first).
const CE_ION: [&[&str]; 2] = [
    &["I I V V I I IV+V I", "I IV I V I IV V I", "I I IV IV I I V+IV I", "I vi IV V I vi V I"],
    &["IV IV I I IV IV V V", "vi vi IV IV I I V I", "IV I V vi IV I V I"],
];
const CE_MIX: [&[&str]; 2] = [
    &["I I bVII bVII I I bVII+V I", "I bVII I bVII I IV bVII I", "I I IV bVII I I bVII+IV I"],
    &["bVII bVII I I bVII bVII IV I", "IV IV I I bVII bVII I I", "I IV bVII IV I IV bVII I"],
];
const CE_DOR: [&[&str]; 2] = [
    &[
        "i i bVII bVII i i bVII+v i",
        "i bVII i bVII i bVII IV+bVII i",
        "i i IV IV i i bVII i",
        "i bIII bVII IV i bIII bVII i",
    ],
    &["bIII bIII bVII bVII bIII bVII i i", "bVII bVII i i bVII bVII IV i", "IV IV i i bVII bVII i i"],
];
const CE_AEO: [&[&str]; 2] = [
    &["i i bVII bVII i i bVI+bVII i", "i bVI bVII i i bVI bVII+v i", "i i bIII bVII i iv bVII i"],
    &["bIII bIII bVII bVII bVI bVI bVII i", "bIII bVII i i bIII bVII bVI+bVII i", "bVI bVII i i bVI bVII v i"],
];

/// A part of a tune: its chords (two a bar) and tune.
#[derive(Clone)]
struct CPart {
    chords: Vec<Harm>,
    mel: Vec<PNote>,
}

/// A tune: its type, key (semitones above the piece's tonic) and mode, its
/// tempo, its A and B parts.
#[derive(Clone)]
struct CTune {
    ty: u8,
    key: i32,
    mode: usize,
    bpm: f64,
    parts: Vec<CPart>,
}

/// A section: its tune, part (0 A, 1 B), time through the tune, and whether
/// it repeats the part.
#[derive(Clone, Copy, Default)]
struct CSec {
    tune: usize,
    part: usize,
    time: u32,
    rep: bool,
}

/// A Celtic piece's state.
#[derive(Default)]
pub(super) struct Celtic {
    pub(super) bar_steps: u32,
    pub(super) beat_steps: u32,
    pub(super) meter: &'static str,
    style: u8,
    tunes: Vec<CTune>,
    map: Vec<CSec>,
    /// this time through: the lead's tune (ornamented), the second's and the
    /// third's (their own ornaments, a beat held here and there), the second
    /// an octave up
    mel: Vec<PNote>,
    mel2: Vec<PNote>,
    mel3: Vec<PNote>,
    up2: i32,
    /// the guitar's last voicing, the side drums' pattern (a bar's 16ths: 0
    /// none, 1 a tap, 2 an accent, 3 a roll's start)
    keys_v: Vec<i32>,
    drum: Vec<u8>,
    /// the off-beat eighths' delay (16ths): a reel's lilt, a hornpipe's swing
    lilt: f32,
    /// the tune playing (an outro the ending jumps to closes it)
    cur: usize,
    /// a mix's drummer: the bodhrán's triplets this piece
    dready: bool,
    dtrip: f32,
}

/// The notes of one beat: a rhythm cell for the type, moving from `prev` by
/// step in the chord-scale or by chord tone (more leaps in a jig), the
/// beat's first on a chord tone.
#[allow(clippy::too_many_arguments)]
fn ce_beat(r: &mut Rng, ty: u8, h: Harm, mode: usize, prev: i32, dir: i32, lo: i32, hi: i32) -> Vec<(u16, u16, i32)> {
    let cells = ct_cells(ty);
    let w: Vec<f32> = cells.iter().map(|c| c.1).collect();
    let cell = cells[r.weighted(&w)].0;
    let ct = in_range(&h.pcs(), lo, hi);
    let sc = in_range(&chord_scale(mode, h), lo, hi);
    let leap = match ty {
        CT_JIG | CT_SLIDE | CT_MARCH68 => 0.35,
        CT_REEL | CT_HORNPIPE => 0.28,
        _ => 0.2,
    };
    let mut p = ct[nearest(&ct, prev + dir)];
    let mut out = Vec::with_capacity(cell.len());
    for (k, &(st, len)) in cell.iter().enumerate() {
        if k > 0 {
            let j = nearest(&sc, p) as isize;
            p = match r.weighted(&[0.72 - leap, leap, 0.1]) {
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

/// An eight-bar part: a two-bar motif coming back in bars 5 and 6 (fitted
/// to their chords), bar 4 resting on a chord tone, bar 8 home; the line in
/// arcs. A hornpipe's last bar ends on three quarters; an air's phrases end long.
fn ce_part(r: &mut Rng, ty: u8, bs: u16, beat: u16, chords: &[Harm], mode: usize, lo: i32, hi: i32) -> Vec<PNote> {
    let beats = bs / beat;
    let chord_at = |bar: usize, b: u16| chords[bar * 2 + ((b * 2) / beats.max(1)) as usize];
    let mut bars: Vec<Vec<PNote>> = Vec::new();
    let mut prev = r.pick(&[0, 4, 7, 7, 12, 5]).clamp(lo, hi);
    let mut dir = if r.bit_chance() { 1 } else { -1 };
    for bar in 0..8usize {
        let notes: Vec<PNote> = if (bar == 4 || bar == 5) && r.chance(0.75) {
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
            let hl = chord_at(bar, beats - 1);
            let target = if bar == 7 { [0, 12, 7] } else { [7, 4, 2] };
            let ct = in_range(&hl.pcs(), lo, hi);
            let want: Vec<i32> = target.iter().map(|&x| x + 12 * ((prev - x) as f32 / 12.0).round() as i32).collect();
            let goal = want[nearest(&want, prev)].clamp(lo, hi);
            let land = ct[nearest(&ct, goal)];
            let mut v: Vec<PNote> = Vec::new();
            if ty == CT_HORNPIPE && bar == 7 && beats == 4 {
                // the hornpipe's ending: a beat's run, then three quarters home
                for (st, len, p) in ce_beat(r, ty, chord_at(bar, 0), mode, prev, dir, lo, hi) {
                    v.push(PNote { step: st, len, pitch: p, vel: 0.85, grace: 0 });
                }
                let sc = in_range(&chord_scale(mode, hl), lo, hi);
                let j = nearest(&sc, land);
                for (k, d) in [2isize, 1, 0].iter().enumerate() {
                    let p = sc[(j as isize + d).clamp(0, sc.len() as isize - 1) as usize];
                    v.push(PNote {
                        step: beat * (k as u16 + 1),
                        len: beat,
                        pitch: if *d == 0 { land } else { p },
                        vel: 0.9,
                        grace: 0,
                    });
                }
            } else {
                // a beat (or two) of figure, then the long note
                let lead = if beats >= 4 { 2 } else { 1 };
                for b in 0..lead {
                    for (st, len, p) in ce_beat(r, ty, chord_at(bar, b), mode, prev, dir, lo, hi) {
                        v.push(PNote { step: b * beat + st, len, pitch: p, vel: 0.85, grace: 0 });
                        prev = p;
                    }
                }
                v.push(PNote { step: lead * beat, len: bs - lead * beat, pitch: land, vel: 0.9, grace: 0 });
            }
            v
        } else {
            let mut v = Vec::new();
            for b in 0..beats {
                let h = chord_at(bar, b);
                for (st, len, p) in ce_beat(r, ty, h, mode, prev, dir, lo, hi) {
                    v.push(PNote {
                        step: b * beat + st,
                        len,
                        pitch: p,
                        vel: if st == 0 { 0.9 } else { 0.75 },
                        grace: 0,
                    });
                    prev = p;
                }
                if prev >= hi - 2 {
                    dir = -1;
                } else if prev <= lo + 2 {
                    dir = 1;
                } else if r.chance(0.3) {
                    dir = -dir;
                }
            }
            v
        };
        prev = notes.last().map_or(prev, |n| n.pitch);
        bars.push(notes);
    }
    let mut mel = Vec::new();
    for (bar, notes) in bars.iter().enumerate() {
        mel.extend(notes.iter().map(|n| PNote { step: n.step + bs * bar as u16, ..*n }));
    }
    mel
}

/// Ornaments on a time through: cuts and taps on the longer notes, rolls on
/// the longest (a dotted quarter's or a quarter's), the fiddle's trebles on
/// a few eighths; more each time.
fn ce_ornaments(r: &mut Rng, mel: &mut [PNote], p: f32, fiddle: bool) {
    for n in mel.iter_mut() {
        if n.grace != 0 {
            continue;
        }
        if n.len >= 4 && r.chance(p * 1.4) {
            n.grace = OR_ROLL;
        } else if n.len >= 3 && r.chance(p) {
            n.grace = r.pick(&[2, 2, 1, OR_TAP]);
        } else if fiddle && n.len == 2 && r.chance(p * 0.3) {
            n.grace = OR_TREBLE;
        }
    }
}

/// The second instrument's tune: the lead's, a beat now and then held
/// where the lead runs (its own way through it), its ornaments its own.
fn ce_heterophony(r: &mut Rng, mel: &[PNote], beat: u16, hold: f32) -> Vec<PNote> {
    let mut out: Vec<PNote> = Vec::with_capacity(mel.len());
    let mut skip_beat = u16::MAX;
    for n in mel {
        let b = n.step / beat;
        if b == skip_beat {
            continue;
        }
        if n.step % beat == 0 && n.len < beat && r.chance(hold) {
            skip_beat = b;
            out.push(PNote { len: beat, grace: 0, ..*n });
            continue;
        }
        out.push(PNote { grace: 0, ..*n });
    }
    out
}

impl Song {
    /// A Celtic piece: a session's set most often, else a pipe set or an air.
    pub(super) fn piece_celtic(&mut self, s: &mut Synth) {
        let style = self.rng.weighted(&[0.6, 0.2, 0.2]) as u8;
        self.ce = Celtic { style, ..Celtic::default() };
        self.swing = 0.0;
        self.human = if style == CS_PIPES { 0.004 } else { 0.01 };
        self.lifts.clear();
        self.pivot = None;
        let guest = self.guest;
        let pull = |bpm: f64| guest.map_or(bpm, |g| bpm + 0.6 * (MID_BPM[g] - bpm));
        let r = &mut self.rng;
        // the set's tunes: their types, keys and tempos
        let types: Vec<u8> = match style {
            CS_SESSION => {
                let ty = r.pick(&[CT_REEL, CT_REEL, CT_JIG, CT_JIG, CT_HORNPIPE, CT_POLKA, CT_SLIDE]);
                vec![ty; r.pick(&[2, 2, 3])]
            }
            CS_PIPES => vec![if r.bit_chance() { CT_MARCH } else { CT_MARCH68 }, CT_STRATHSPEY, CT_REEL],
            _ => vec![if r.bit_chance() { CT_AIR } else { CT_PLANXTY }],
        };
        // the session's keys: D, G, A, E minor, A dorian, B minor, D mixolydian...,
        // each tune in another; the pipes' A mixolydian (sounding near B flat)
        let keys: &[(i32, usize)] = &[
            (2, IONIAN),
            (7, IONIAN),
            (9, 1),
            (4, AEOLIAN),
            (4, 1),
            (11, AEOLIAN),
            (2, MIXOLYDIAN),
            (9, MIXOLYDIAN),
            (0, IONIAN),
        ];
        let mut tunes = Vec::new();
        let mut last = usize::MAX;
        let base = match style {
            CS_PIPES => (10, MIXOLYDIAN),
            _ => keys[r.below(keys.len())],
        };
        self.root = base.0;
        self.mode = base.1;
        for (t, &ty) in types.iter().enumerate() {
            let (pc, mode) = if style == CS_PIPES || t == 0 {
                base
            } else {
                let mut k = r.below(keys.len());
                if keys[k].0 == keys[last].0 || keys[k] == base {
                    k = (k + 1 + r.below(keys.len() - 1)) % keys.len();
                }
                last = k;
                keys[k]
            };
            if t == 0 {
                last = keys.iter().position(|&k| k == base).unwrap_or(0);
            }
            let bpm = match ty {
                CT_REEL if style == CS_PIPES => r.range(132.0, 148.0),
                CT_REEL => r.range(104.0, 120.0),
                CT_JIG => r.range(112.0, 126.0),
                CT_HORNPIPE => r.range(76.0, 90.0),
                CT_POLKA => r.range(126.0, 142.0),
                CT_SLIDE => r.range(124.0, 138.0),
                CT_MARCH => r.range(78.0, 90.0),
                CT_MARCH68 => r.range(98.0, 112.0),
                CT_STRATHSPEY => r.range(112.0, 128.0),
                CT_AIR => r.range(58.0, 70.0),
                _ => r.range(96.0, 112.0),
            } as f64;
            // the set lifts as it goes
            let bpm = pull(bpm * (1.0 + 0.025 * t as f64));
            let (bs, beat, _) = ct_meter(ty);
            let pools = match mode {
                IONIAN => CE_ION,
                MIXOLYDIAN => CE_MIX,
                AEOLIAN => CE_AEO,
                _ => CE_DOR,
            };
            let (lo, hi) = if style == CS_PIPES { (-2, 12) } else { (-3, 12) };
            let mut parts = Vec::new();
            for (k, pool) in pools.iter().enumerate() {
                let chords = halves(r.pick(pool));
                let (l, h) = if k == 0 { (lo, hi) } else { (lo + 2, hi + 3) };
                let mel = ce_part(
                    r,
                    ty,
                    bs as u16,
                    beat as u16,
                    &chords,
                    mode,
                    l,
                    h.min(if style == CS_PIPES { 12 } else { 15 }),
                );
                parts.push(CPart { chords, mel });
            }
            let key = (pc - base.0).rem_euclid(12);
            tunes.push(CTune { ty, key: if key > 6 { key - 12 } else { key }, mode, bpm, parts });
        }
        // the plan: a bar in, each tune AABB (twice through in a set of two,
        // an air once), a bar's cadence and the last chord out
        let mut plan: Vec<(Sec, CSec)> = Vec::new();
        let times = match style {
            CS_SESSION if tunes.len() == 2 => 2,
            CS_PIPES => 1,
            CS_AIR => 1,
            _ => 1,
        };
        let session = style == CS_SESSION;
        let intro_mask = match style {
            CS_PIPES => L_DRONE | L_SNARE | L_KICK | L_PERC,
            CS_AIR => L_KEYS | L_BASS,
            _ => L_KEYS | L_KICK,
        };
        plan.push((
            Sec { kind: INTRO, bars: if style == CS_SESSION { 1 } else { 2 }, mask: intro_mask, energy: 0.5 },
            CSec::default(),
        ));
        for (t, tune) in tunes.iter().enumerate() {
            let tt = if style == CS_PIPES && tune.ty != CT_REEL { 1 } else { times };
            for time in 0..tt {
                for (part, rep) in [(0, false), (0, true), (1, false), (1, true)] {
                    let q = (time * 4 + part as u32 * 2 + rep as u32) as f32 + 8.0 * t as f32;
                    let mut mask = L_LEAD;
                    match style {
                        CS_SESSION => {
                            // the leader alone at first, the others joining
                            if t > 0 || q >= 1.0 {
                                mask |= L_KEYS | L_BELL;
                            }
                            if t > 0 || q >= 1.0 {
                                mask |= L_KICK;
                            }
                            if t > 0 || q >= 2.0 {
                                mask |= L_ARP;
                            }
                            if t > 0 && q >= 12.0 && part == 1 {
                                mask |= L_PERC;
                            }
                        }
                        CS_PIPES => mask |= L_BELL | L_DRONE | L_SNARE | L_KICK | L_PERC,
                        _ => {
                            mask |= L_KEYS | L_BASS;
                            if q >= 1.0 {
                                mask |= L_BELL;
                            }
                            if q < 1.0 && r.chance(0.5) {
                                mask &= !L_LEAD;
                            }
                        }
                    }
                    let energy = (0.55 + 0.04 * q + if session { 0.1 * t as f32 } else { 0.0 }).min(1.0);
                    plan.push((Sec { kind: MAIN, bars: 8, mask, energy }, CSec { tune: t, part, time, rep }));
                }
            }
        }
        let end_mask = match style {
            CS_SESSION => L_LEAD | L_BELL | L_ARP | L_KEYS | L_KICK,
            CS_PIPES => L_LEAD | L_BELL | L_DRONE | L_SNARE | L_KICK | L_PERC,
            _ => L_LEAD | L_BELL | L_KEYS | L_BASS,
        };
        let lt = tunes.len() - 1;
        plan.push((
            Sec { kind: OUTRO, bars: 2, mask: end_mask, energy: 0.9 },
            CSec { tune: lt, part: 1, time: 0, rep: true },
        ));
        self.plan = plan.iter().map(|x| x.0).collect();
        self.ce.map = plan.iter().map(|x| x.1).collect();
        self.ce.tunes = tunes;
        let (bs, beat, meter) = ct_meter(self.ce.tunes[0].ty);
        (self.ce.bar_steps, self.ce.beat_steps, self.ce.meter) = (bs, beat, meter);
        self.bpm = self.ce.tunes[0].bpm;
        sounds_ce(&mut self.rng, s, style);
        self.band = band_celtic(&mut self.rng, s, style);
        s.dly.target = (self.step_len() * 3.0) as f32;
    }

    /// A Celtic section begins: the tune's meter and tempo (a new tune's
    /// lift), the tune this time through (more ornaments each time, a bar
    /// varied on the repeats), the second and third instruments' ways
    /// through it, the side drums' pattern.
    pub(super) fn celtic_section(&mut self) {
        let Some(&x) = self.ce.map.get(self.sec) else { return };
        let sec = self.plan[self.sec];
        // the outro closes the tune it comes from (its B part's cadence)
        let x = if sec.kind == OUTRO { CSec { tune: self.ce.cur, part: 1, time: 0, rep: true } } else { x };
        self.ce.map[self.sec] = x;
        self.ce.cur = x.tune;
        let tune = self.ce.tunes[x.tune].clone();
        let (bs, beat, meter) = ct_meter(tune.ty);
        (self.ce.bar_steps, self.ce.beat_steps, self.ce.meter) = (bs, beat, meter);
        if sec.kind != OUTRO {
            self.bpm = tune.bpm;
        }
        let style = self.ce.style;
        let r = &mut self.rng;
        self.ce.lilt = match tune.ty {
            CT_HORNPIPE => r.range(0.55, 0.75),
            CT_REEL if style == CS_SESSION => r.range(0.08, 0.2),
            CT_POLKA => r.range(0.05, 0.15),
            _ => 0.0,
        };
        let part = &tune.parts[x.part.min(tune.parts.len() - 1)];
        let mut mel = part.mel.clone();
        if sec.kind == MAIN && (x.rep || x.time > 0) && r.chance(0.6) {
            // a bar figured anew
            let bar = r.pick(&[0u16, 1, 2, 4, 5, 6]);
            let beats = (bs / beat) as u16;
            let mut p = mel.iter().filter(|n| n.step / bs as u16 == bar).map(|n| n.pitch).next().unwrap_or(0);
            let dir = if r.bit_chance() { 1 } else { -1 };
            let mut v = Vec::new();
            for k in 0..beats {
                let h = part.chords[bar as usize * 2 + ((k * 2) / beats) as usize];
                for (st, len, pitch) in ce_beat(r, tune.ty, h, tune.mode, p, dir, -3, 15) {
                    v.push(PNote {
                        step: bar * bs as u16 + k * beat as u16 + st,
                        len,
                        pitch,
                        vel: if st == 0 { 0.9 } else { 0.75 },
                        grace: 0,
                    });
                    p = pitch;
                }
            }
            mel.retain(|n| n.step / bs as u16 != bar);
            mel.extend(v);
            mel.sort_by_key(|n| n.step);
        }
        let orn = 0.1 + 0.05 * (x.time * 2 + x.rep as u32) as f32 + if style == CS_AIR { 0.1 } else { 0.0 };
        let mut m2 = ce_heterophony(r, &mel, beat as u16, if style == CS_PIPES { 0.0 } else { 0.18 });
        let mut m3 = ce_heterophony(r, &mel, beat as u16, 0.25);
        if style != CS_PIPES {
            ce_ornaments(r, &mut mel, orn, true);
            ce_ornaments(r, &mut m2, orn * 0.8, false);
        }
        self.ce.up2 = if style == CS_SESSION && r.chance(0.4) { 12 } else { 0 };
        self.ce.mel = mel;
        self.ce.mel2 = m2;
        self.ce.mel3 = m3.drain(..).collect();
        // the side drums: a bar's pattern of taps, accents and rolls, the
        // tune type's lilt in it
        let mut d = vec![0u8; bs as usize];
        for (st, v) in d.iter_mut().enumerate() {
            let on_beat = st as u32 % beat == 0;
            *v = if on_beat {
                if r.chance(0.3) { 3 } else { 2 }
            } else if r.chance(match tune.ty {
                CT_STRATHSPEY => 0.45,
                CT_MARCH | CT_MARCH68 => 0.6,
                _ => 0.75,
            }) {
                1
            } else {
                0
            };
        }
        self.ce.drum = d;
    }

    /// A bar of Celtic music.
    pub(super) fn celtic(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let x = self.ce.map[self.sec];
        let tune = &self.ce.tunes[x.tune];
        let (ty, mode, key) = (tune.ty, tune.mode, tune.key);
        let (bs, beat) = (self.ce.bar_steps, self.ce.beat_steps);
        let beats = bs / beat;
        let i = self.sec_bar;
        let style = self.ce.style;
        let part = &tune.parts[x.part.min(tune.parts.len() - 1)];
        let bi = (i % 8) as usize;
        let minor = MODES[mode][2] == 3;
        let home = Harm { root: 0, q: if minor { Q_MIN } else { Q_MAJ }, inv: 0 };
        let last_bar = sec.kind == OUTRO && i + 1 == sec.bars;
        let chords: [Harm; 2] = match sec.kind {
            MAIN => [part.chords[bi * 2], part.chords[bi * 2 + 1]],
            OUTRO if i == 0 => [part.chords[14], part.chords[15]],
            _ => [home, home],
        };
        let e = sec.energy;
        let h = self.human;
        let lilt = self.ce.lilt;
        // the off-beat eighths late (the lilt, the hornpipe's swing)
        let late_of = |st: f32| -> f32 { if beat == 4 && (st % 4.0 - 2.0).abs() < 0.01 { lilt } else { 0.0 } };
        let step_s = b.secs(1.0);
        let root = self.root;
        let tonic = |center: i32| center - 6 + (root + key - (center - 6)).rem_euclid(12);
        let tune_t = if style == CS_PIPES { tonic(70) } else { tonic(67) };
        let chord_at = |k: u32| chords[((k * 2) / beats.max(1)) as usize];
        // the tune for this bar (the outro: the B part's cadence, then the tonic)
        let pick = |v: &[PNote], bar: usize| -> Vec<PNote> {
            v.iter()
                .filter(|n| n.step as usize / bs as usize == bar)
                .map(|n| PNote { step: n.step % bs as u16, ..*n })
                .collect()
        };
        let (mel, mel2, mel3) = match sec.kind {
            MAIN => (pick(&self.ce.mel, bi), pick(&self.ce.mel2, bi), pick(&self.ce.mel3, bi)),
            OUTRO if i == 0 => {
                let v = pick(&part.mel, 7);
                (v.clone(), v.clone(), v)
            }
            OUTRO => {
                let v = vec![PNote { step: 0, len: bs as u16, pitch: 0, vel: 0.9, grace: 0 }];
                (v.clone(), v.clone(), v)
            }
            _ => (Vec::new(), Vec::new(), Vec::new()),
        };
        let up2 = self.ce.up2;
        let pipes = style == CS_PIPES;
        // the lead
        if m & L_LEAD != 0 {
            let vel = 0.7 + 0.22 * e;
            self.ce_line(b, CH_LEAD, &mel, tune_t, vel, step_s, &late_of, pipes, last_bar);
        }
        if m & L_BELL != 0 {
            let vel = 0.62 + 0.2 * e;
            let t2 = tune_t + if pipes { 0 } else { up2 };
            self.ce_line(b, CH_BELL, &mel2, t2, vel, step_s, &late_of, pipes, last_bar);
        }
        if m & L_ARP != 0 && sec.kind != INTRO {
            let vel = 0.55 + 0.2 * e;
            self.ce_line(b, CH_ARP, &mel3, tune_t, vel, step_s, &late_of, false, last_bar);
        }
        // the drones: the bass and the two tenors (or the uilleann pipes' regulators' tonic and fifth)
        if m & L_DRONE != 0 && (i % 4 == 0 || sec.kind == INTRO && i == 0) {
            let t = tonic(57) - 12;
            let len = if last_bar { bs as f32 } else { 4.0 * bs as f32 + 2.0 };
            b.note(0.0, 0.0, CH_DRONE, t, 0.62, len, false);
            b.note(0.0, 0.004, CH_DRONE, t + 12, 0.5, len, false);
            b.note(0.0, 0.008, CH_DRONE, t + 12, 0.48, len, false);
        }
        // the guitar (or bouzouki) strumming, the harp's chords and arpeggios
        if m & L_KEYS != 0 {
            if style == CS_AIR {
                self.ce_harp(b, &chords, tonic(48), last_bar, m & L_BASS != 0);
            } else {
                self.ce_strum(b, &chords, ty, tonic(50), last_bar, e);
            }
        }
        // the drums
        self.ce_drums(b, sec, m, ty, last_bar);
        let _ = (chord_at(0), h, s.ch.len());
    }

    /// A line on channel `ch`: the notes about `t`, each with its ornament
    /// (the pipes' gracenotes and doublings between every note instead).
    #[allow(clippy::too_many_arguments)]
    fn ce_line(
        &mut self,
        b: &mut Bar,
        ch: usize,
        mel: &[PNote],
        t: i32,
        vel: f32,
        step_s: f32,
        late_of: &dyn Fn(f32) -> f32,
        pipes: bool,
        last: bool,
    ) {
        let h = self.human;
        let mut prev = i32::MIN;
        for n in mel {
            let st = n.step as f32;
            let late0 = late_of(st) * step_s + self.rng.range(0.0, h);
            let len = if last { n.len as f32 * 2.0 } else { n.len as f32 * 0.94 - late_of(st) };
            let p = t + n.pitch;
            let v = n.vel * vel;
            if pipes {
                // a gracenote (high G) before a new note, a doubling on the
                // beat's strong ones; a G gracenote between repeated notes
                let g = t + 10;
                let gs = 0.022 / step_s;
                if n.step % 4 == 0 && n.len >= 2 && self.rng.chance(0.5) {
                    b.note(st, late0, ch, g + 2, v * 0.6, gs, false);
                    b.note(st, late0 + 0.022, ch, p, v * 0.7, gs, true);
                    b.note(st, late0 + 0.044, ch, p + 2, v * 0.6, gs, true);
                    b.note(st, late0 + 0.066, ch, p, v, len - 0.066 / step_s, true);
                } else if p != prev || self.rng.chance(0.6) {
                    b.note(st, late0, ch, if p >= g { g + 2 } else { g }, v * 0.6, gs, false);
                    b.note(st, late0 + 0.022, ch, p, v, len - gs, true);
                } else {
                    b.note(st, late0, ch, p, v, len, true);
                }
                prev = p;
                continue;
            }
            match n.grace {
                0 => b.note(st, late0, ch, p, v, len, false),
                OR_ROLL => {
                    // the note, a cut above, the note, a tap below, the note
                    let g = 0.04 / step_s;
                    b.note(st, late0, ch, p, v * 0.85, g, false);
                    b.note(st, late0 + 0.04, ch, p + 2, v * 0.6, g * 0.5, true);
                    b.note(st, late0 + 0.06, ch, p, v * 0.8, g, true);
                    b.note(st, late0 + 0.1, ch, p - 2, v * 0.6, g * 0.5, true);
                    b.note(st, late0 + 0.12, ch, p, v, len - 0.12 / step_s, true);
                }
                OR_TAP => {
                    b.note(st, late0, ch, p - 2, v * 0.6, 0.03 / step_s, false);
                    b.note(st, late0 + 0.03, ch, p, v, len - 0.03 / step_s, true);
                }
                OR_TREBLE => {
                    // three bows on the note, the first two short
                    let g = n.len as f32 / 4.0;
                    b.note(st, late0, ch, p, v * 0.85, g * 0.9, false);
                    b.note(st + g, late0, ch, p, v * 0.75, g * 0.9, false);
                    b.note(st + 2.0 * g, late0, ch, p, v, n.len as f32 * 0.45, false);
                }
                g => {
                    // a cut: a grace above, flicked into the note
                    b.note(st, late0, ch, p + g as i32, v * 0.65, 0.035 / step_s, false);
                    b.note(st, late0 + 0.035, ch, p, v, len - 0.035 / step_s, true);
                }
            }
            prev = p;
        }
        let _ = prev;
    }

    /// The guitar in DADGAD (or the bouzouki): the key's open strings (its
    /// tonic and fifth) under the chord's tones, strummed in eighths with
    /// the off-beats leant on (a jig's down-up-down a beat); the last chord rung.
    fn ce_strum(&mut self, b: &mut Bar, chords: &[Harm; 2], ty: u8, low: i32, last: bool, e: f32) {
        let (bs, beat) = (self.ce.bar_steps, self.ce.beat_steps);
        let beats = bs / beat;
        let h = self.human;
        let lilt = self.ce.lilt;
        let step_s = b.secs(1.0);
        for k in 0..beats {
            let c = chords[((k * 2) / beats.max(1)) as usize];
            // the open strings: the key's tonic and fifth, low; the chord over them
            let mut v = vec![low, low + 7, low + 12];
            let top = self.voicing(c, low + 14, low + 24, &self.ce.keys_v, false);
            v.extend(top.iter().copied());
            self.ce.keys_v = top;
            let at = (k * beat) as f32;
            if last {
                if k == 0 {
                    for (j, &n) in v.iter().enumerate() {
                        b.note(0.0, 0.012 * j as f32, CH_KEYS, n, 0.6, bs as f32 * 2.0, false);
                    }
                }
                continue;
            }
            let strokes: &[(f32, bool, f32)] = if beat == 6 {
                &[(0.0, true, 0.62), (2.0, false, 0.36), (4.0, true, 0.48)]
            } else if ty == CT_POLKA {
                &[(0.0, true, 0.42), (2.0, false, 0.62)]
            } else {
                &[(0.0, true, 0.48), (2.0, false, 0.58)]
            };
            for &(off, down, vel) in strokes {
                let st = at + off;
                let late = if beat == 4 && off == 2.0 { lilt * step_s } else { 0.0 } + self.rng.range(0.0, h);
                let acc = if beat == 4 && k % 2 == 1 { 1.08 } else { 1.0 };
                let gap = self.rng.range(0.006, 0.012);
                let strings: Vec<i32> = if down { v.clone() } else { v.iter().rev().take(4).copied().collect() };
                for (j, &n) in strings.iter().enumerate() {
                    let vv = vel * acc * (0.8 + 0.25 * e) * (1.0 - 0.04 * j as f32);
                    b.note(
                        st,
                        late + gap * j as f32,
                        CH_KEYS,
                        n,
                        vv,
                        beat as f32 * if off == 0.0 { 0.9 } else { 0.5 },
                        false,
                    );
                }
            }
        }
    }

    /// The harp: the bass on the beat (the left hand, on its own channel),
    /// a rolled chord on the bar's first beat and the arpeggio rising after
    /// it; the last chord rolled long.
    fn ce_harp(&mut self, b: &mut Bar, chords: &[Harm; 2], low: i32, last: bool, bass: bool) {
        let (bs, beat) = (self.ce.bar_steps, self.ce.beat_steps);
        let beats = bs / beat;
        let h = self.human;
        for k in 0..beats {
            let c = chords[((k * 2) / beats.max(1)) as usize];
            let at = (k * beat) as f32;
            let root = low + c.bass().rem_euclid(12);
            if bass && (k == 0 || (beats == 4 && k == 2)) {
                b.note(
                    at,
                    self.rng.range(0.0, h),
                    CH_BASS,
                    root - 12,
                    0.62,
                    if last { 24.0 } else { bs as f32 },
                    false,
                );
            }
            let tones = self.arp_tones(c, low + 12);
            if k == 0 {
                for (j, &n) in tones.iter().take(4).enumerate() {
                    b.note(
                        at,
                        0.03 * j as f32 + self.rng.range(0.0, h),
                        CH_KEYS,
                        n,
                        0.45,
                        if last { 24.0 } else { 8.0 },
                        false,
                    );
                }
            } else if !last {
                for (q, x) in [0.0, 2.0].iter().enumerate() {
                    let n = tones[(k as usize * 2 + q + 1) % tones.len()];
                    b.note(at + x, self.rng.range(0.0, h), CH_KEYS, n, 0.38, 6.0, false);
                }
            }
        }
    }

    /// The drums: the bodhrán (its eighths, the backbeat leant on, its
    /// rolling triplets, the top's tone now and then), the bones; the pipe
    /// band's side drums (their pattern's taps, accents and rolls), tenor
    /// and bass drums. The last bar: one stroke.
    fn ce_drums(&mut self, b: &mut Bar, sec: Sec, m: u32, ty: u8, last: bool) {
        let (bs, beat) = (self.ce.bar_steps, self.ce.beat_steps);
        let beats = bs / beat;
        let e = sec.energy;
        let h = self.human;
        let lilt = self.ce.lilt;
        let step_s = b.secs(1.0);
        let r = &mut self.rng;
        if last {
            if m & (L_KICK | L_SNARE) != 0 {
                b.hit(0.0, 0.0, CH_KICK, 0, 0.85);
            }
            if m & L_SNARE != 0 {
                b.hit(0.0, 0.0, CH_SNARE, 0, 0.9);
            }
            return;
        }
        if self.ce.style == CS_PIPES {
            if sec.kind == INTRO {
                // the roll-in: two three-pace rolls, the bass drum on the beats
                for st in 0..bs {
                    let x = st as f32 / bs as f32;
                    for q in 0..2 {
                        b.hit(st as f32 + 0.5 * q as f32, r.range(0.0, h), CH_SNARE, 1, 0.3 + 0.4 * x);
                    }
                }
                for k in 0..beats {
                    b.hit((k * beat) as f32, 0.0, CH_KICK, 0, 0.75);
                }
                return;
            }
            if m & L_SNARE != 0 {
                for (st, &c) in self.ce.drum.iter().enumerate() {
                    let at = st as f32;
                    let late = if beat == 4 && st % 4 == 2 { lilt * step_s } else { 0.0 };
                    match c {
                        1 => b.hit(at, late + r.range(0.0, h), CH_SNARE, 1, 0.35 + 0.1 * e),
                        2 => {
                            // a flam
                            b.hit(at, late, CH_SNARE, 1, 0.4);
                            b.hit(at, late + 0.012, CH_SNARE, 0, 0.82);
                        }
                        3 => {
                            // a roll into the beat
                            for q in 0..4 {
                                b.hit(at + 0.25 * q as f32, late, CH_SNARE, 1, 0.3 + 0.08 * q as f32);
                            }
                            b.hit(at + 1.0, late, CH_SNARE, 0, 0.75);
                        }
                        _ => {}
                    }
                }
            }
            if m & L_KICK != 0 {
                for k in 0..beats {
                    if k % 2 == 0 || beats <= 2 {
                        b.hit((k * beat) as f32, r.range(0.0, h), CH_KICK, 0, if k == 0 { 0.85 } else { 0.7 });
                    }
                }
            }
            if m & L_PERC != 0 {
                for k in 0..beats {
                    b.hit((k * beat) as f32, r.range(0.0, h), CH_PERC, TOM_MID, 0.45);
                }
            }
            return;
        }
        if m & L_KICK != 0 {
            // the bodhrán
            let trip = if ty == CT_HORNPIPE { 0.1 } else { 0.22 + 0.15 * e };
            for k in 0..beats {
                let at = (k * beat) as f32;
                let back = beats >= 4 && k % 2 == 1 || beats == 2 && k == 1;
                if sec.kind == INTRO {
                    if k + 1 == beats {
                        // the pickup: a triplet into the tune
                        for q in 0..3 {
                            b.hit(at + q as f32 * beat as f32 / 3.0, 0.0, CH_KICK, 0, 0.45 + 0.1 * q as f32);
                        }
                    }
                    continue;
                }
                let strokes: Vec<(f32, f32, u8)> = if beat == 6 {
                    let mut v = vec![(0.0, 0.78, 0u8), (2.0, 0.42, 0), (4.0, 0.55, 0)];
                    if r.chance(trip) {
                        v.pop();
                        v.extend([(4.0, 0.45, 0), (4.67, 0.38, 0), (5.33, 0.48, 0)]);
                    }
                    v
                } else {
                    let mut v = vec![(0.0, if back { 0.85 } else { 0.62 }, 0u8), (2.0 + lilt, 0.42, 0)];
                    if r.chance(trip) {
                        v.pop();
                        v.extend([(2.0, 0.42, 0), (2.67, 0.36, 0), (3.33, 0.46, 0)]);
                    }
                    if back && r.chance(0.3) {
                        v.push((1.0, 0.3, 1));
                    }
                    v
                };
                for (off, v, kind) in strokes {
                    b.hit(at + off, r.range(0.0, h), CH_KICK, kind, v);
                }
            }
        }
        if m & L_PERC != 0 {
            // the bones: the off-beat eighths and a rattle into the bar
            for k in 0..beats {
                let at = (k * beat) as f32;
                b.hit(at + 2.0, r.range(0.0, h), CH_PERC, 0, 0.4);
                if k + 1 == beats && r.chance(0.5) {
                    b.hit(at + 3.0, r.range(0.0, h), CH_PERC, 0, 0.32);
                }
            }
        }
    }

    /// The bodhrán under another genre's music, in its meter: its eighths
    /// (a compound beat's three), the backbeat leant on, a rolling triplet
    /// now and then, the top's tone; the bones on the off-beats in a full
    /// section.
    pub(super) fn celtic_beat(&mut self, _s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let (bs, beat) = (self.bar_steps(), self.beat_steps());
        let beats = (bs / beat).max(1);
        if !self.ce.dready {
            self.ce.dready = true;
            self.ce.dtrip = self.rng.range(0.15, 0.35);
        }
        let (e, h) = (sec.energy, self.human);
        let full = matches!(sec.kind, MAIN | BUILD | OUTRO);
        let quiet = if sec.kind == BREAK && m & L_KICK == 0 { 0.6 } else { 1.0 };
        let trip = self.ce.dtrip;
        let r = &mut self.rng;
        for k in 0..beats {
            let at = (k * beat) as f32;
            let back = beats >= 4 && k % 2 == 1 || beats == 2 && k == 1 || beats == 3 && k > 0;
            let mut strokes: Vec<(f32, f32, u8)> = if beat == 6 {
                vec![(0.0, 0.78, 0), (2.0, 0.42, 0), (4.0, 0.55, 0)]
            } else {
                vec![(0.0, if back { 0.85 } else { 0.62 }, 0), (2.0, 0.42, 0)]
            };
            if (full || m & L_HAT != 0) && r.chance(trip) {
                let s0 = if beat == 6 { 4.0 } else { 2.0 };
                strokes.pop();
                strokes.extend([(s0, 0.42, 0), (s0 + 0.67, 0.36, 0), (s0 + 1.33, 0.46, 0)]);
            }
            if back && r.chance(0.3) {
                strokes.push((1.0, 0.3, 1));
            }
            for (off, v, kind) in strokes {
                b.hit(at + off, r.range(0.0, h), CH_KICK, kind, v * quiet);
            }
            if (full && e >= 0.8) || m & L_PERC != 0 {
                b.hit(at + beat as f32 / 2.0, r.range(0.0, h), CH_PERC, 0, 0.38 * quiet);
            }
        }
    }
}

// -- the instruments

/// The tin whistle: the SoundFont's whistle (else the piccolo's).
fn ce_whistle(r: &mut Rng) -> Patch {
    let id = samples::gm("Whistle").or_else(|| samples::gm("Piccolo"));
    rec(whistle_synth(r), id, 81.0)
}

/// The uilleann pipes' chanter: the bagpipe's recording under a low-pass
/// (sweeter, quieter than the Highland pipes').
fn uilleann(r: &mut Rng) -> Patch {
    let p = Patch { cutoff: 1900.0, keytrack: 0.4, gain: 0.32, ..chanter_synth(r) };
    rec_lp(p, samples::gm("Bagpipe"), 72.0, 1.0, 0.0)
}

/// The concertina: the bandoneon's recording (a concertina's kin).
fn concertina(r: &mut Rng) -> Patch {
    bandoneon(r)
}

/// The button accordion.
fn box_accordion(r: &mut Rng) -> Patch {
    rec(accordion_synth(r), samples::gm("Accordion"), 72.0)
}

fn ce_guitar_synth(r: &mut Rng) -> Patch {
    Patch {
        ks: r.range(1.4, 2.0),
        ks_tone: 0.7,
        cutoff: 4500.0,
        keytrack: 0.3,
        a: 0.001,
        d: 2.5,
        s: 1.0,
        r: 0.25,
        gain: 0.42,
        ..PATCH
    }
}

/// The guitar in DADGAD: a steel-strung guitar's recording.
fn ce_guitar(r: &mut Rng) -> Patch {
    rec(ce_guitar_synth(r), samples::gm("Steel String Guitar"), 55.0)
}

fn bouzouki_synth(r: &mut Rng) -> Patch {
    Patch { unison: 2, detune: 6.0, ks: r.range(1.0, 1.4), ks_tone: 0.85, ..ce_guitar_synth(r) }
}

/// The Irish bouzouki: the twelve-string guitar's recording (paired courses, an octave's jangle).
fn bouzouki(r: &mut Rng) -> Patch {
    rec(bouzouki_synth(r), samples::gm("12-String Guitar"), 55.0)
}

/// The folk (lever) harp, else the concert harp.
fn folk_harp(r: &mut Rng) -> Patch {
    rec(harp_synth(r), multi_or("folk_harp", "pluck", "Harp"), 67.0)
}

/// The pipe band's side drum: [an accent, a tap (the roll's strokes)].
fn kit_side_drum_synth(r: &mut Rng) -> Vec<DrumPatch> {
    let acc = DrumPatch {
        tone: 0.5,
        f0: r.range(320.0, 360.0),
        f1: 260.0,
        pdec: 0.01,
        tdec: 0.06,
        noise: 1.0,
        nmode: 1,
        ncut: 4200.0,
        nres: 0.2,
        ndec: 0.12,
        gain: 0.6,
        ..DRUM
    };
    let tap = DrumPatch { tone: 0.3, ndec: 0.06, gain: 0.45, ..acc };
    vec![acc, tap]
}

/// The side drum: the kit's snare and its side stroke.
fn kit_side_drum(r: &mut Rng) -> Vec<DrumPatch> {
    rec_kit_db(kit_side_drum_synth(r), &[(samples::drum("Standard", 38), 0.0), (samples::drum("Standard", 40), -4.0)])
}

/// The pipe band's bass drum.
fn kit_pipe_bass_synth(r: &mut Rng) -> Vec<DrumPatch> {
    vec![kick(r, (70.0, 80.0), (48.0, 55.0), (0.5, 0.7), 0.1, 0.0)]
}

fn kit_pipe_bass(r: &mut Rng) -> Vec<DrumPatch> {
    rec_kit_db(
        kit_pipe_bass_synth(r),
        &[(samples::drum("Orchestra Kit", 36).or_else(|| samples::drum("Standard", 36)), 0.0)],
    )
}

/// The tenor drums (percussion's toms).
fn kit_tenor(r: &mut Rng) -> Vec<DrumPatch> {
    rec_percs(percs(r))
}

/// The session's instruments lent to another genre's parts, in a mix.
pub(super) const P_CE_LEAD: &[(&str, Make)] =
    &[("fiddle", fiddle), ("wooden flute", flute), ("tin whistle", ce_whistle), ("uilleann pipes", uilleann)];
pub(super) const P_CE_SECOND: &[(&str, Make)] = &[
    ("button accordion", box_accordion),
    ("concertina", concertina),
    ("tenor banjo", banjo),
    ("mandolin", cittern),
    ("harp", folk_harp),
];
pub(super) const P_CE_STRUM: &[(&str, Make)] =
    &[("guitar (DADGAD)", ce_guitar), ("bouzouki", bouzouki), ("harp", folk_harp)];
pub(super) const P_CE_DRONE: &[(&str, Make)] = &[("pipes' drones", pipe_drone)];
pub(super) const P_CE_BASS: &[(&str, Make)] = &[("harp's bass", folk_harp)];
pub(super) const K_CE_KICK: &[(&str, MakeKit)] = &[("bodhrán", kit_bodhran)];
pub(super) const K_CE_SNARE: &[(&str, MakeKit)] = &[("side drum", kit_side_drum)];
pub(super) const K_CE_PERC: &[(&str, MakeKit)] = &[("bones", kit_bones)];

/// The band: each melodic part's two (the A instrument verses, the B the
/// others: `dress`), by style.
fn band_celtic(r: &mut Rng, s: &Synth, style: u8) -> Band {
    let mut b = Band::default();
    match style {
        CS_SESSION => {
            b.tone(r, s, CH_LEAD, &[("fiddle", fiddle), ("wooden flute", flute)], 74.0);
            b.tone(
                r,
                s,
                CH_BELL,
                &[
                    ("tin whistle", ce_whistle),
                    ("wooden flute", flute),
                    ("fiddle", fiddle),
                    ("uilleann pipes", uilleann),
                ],
                76.0,
            );
            b.tone(r, s, CH_ARP, P_CE_SECOND, 72.0);
            b.tone(r, s, CH_KEYS, &[("guitar (DADGAD)", ce_guitar), ("bouzouki", bouzouki)], 55.0);
        }
        CS_PIPES => {}
        _ => {
            b.tone(r, s, CH_LEAD, &[("fiddle", fiddle), ("wooden flute", flute), ("uilleann pipes", uilleann)], 74.0);
            b.tone(r, s, CH_BELL, &[("wooden flute", flute), ("fiddle", fiddle), ("tin whistle", ce_whistle)], 74.0);
        }
    }
    b
}

/// Celtic's sounds for a mix's drummer: a session's.
pub(super) fn sounds_celtic(r: &mut Rng, s: &mut Synth) {
    sounds_ce(r, s, CS_SESSION)
}

/// Each style's instruments: the session's fiddle, flute or whistle,
/// accordion or banjo, guitar and bodhrán in a pub's room; the pipes and
/// drums outdoors; the harp and the air in a small hall.
fn sounds_ce(r: &mut Rng, s: &mut Synth, style: u8) {
    let ch = &mut s.ch;
    for c in [CH_LEAD, CH_BELL, CH_ARP, CH_KEYS, CH_BASS, CH_DRONE, CH_PAD] {
        ch[c].mono = false;
        ch[c].swell = 0.0;
        ch[c].width = 1.0;
    }
    match style {
        CS_PIPES => {
            ch[CH_LEAD].patch = chanter(r);
            ch[CH_BELL].patch = chanter(r);
            ch[CH_DRONE].patch = pipe_drone(r);
            ch[CH_ARP].patch = chanter(r);
            ch[CH_KEYS].patch = ce_guitar(r);
            ch[CH_BASS].patch = folk_harp(r);
            ch[CH_KICK].drums = kit_pipe_bass(r);
            ch[CH_SNARE].drums = kit_side_drum(r);
            ch[CH_PERC].drums = kit_tenor(r);
            ch[CH_HAT].drums = kit_trim(kit_tambourine(r), -4.8);
            mix(&mut ch[CH_LEAD], 0.4, -0.15, 0.2, 0.0, 0.0);
            mix(&mut ch[CH_BELL], 0.34, 0.2, 0.22, 0.0, 0.0);
            mix(&mut ch[CH_DRONE], 0.26, 0.0, 0.2, 0.0, 0.0);
            mix(&mut ch[CH_ARP], 0.3, 0.0, 0.2, 0.0, 0.0);
            mix(&mut ch[CH_KEYS], 0.3, 0.0, 0.2, 0.0, 0.0);
            mix(&mut ch[CH_BASS], 0.3, 0.0, 0.2, 0.0, 0.0);
            mix(&mut ch[CH_SNARE], 0.42, 0.1, 0.18, 0.0, 0.0);
            mix(&mut ch[CH_KICK], 0.55, 0.0, 0.15, 0.0, 0.0);
            mix(&mut ch[CH_PERC], 0.3, -0.2, 0.15, 0.0, 0.0);
            mix(&mut ch[CH_HAT], 0.2, 0.3, 0.1, 0.0, 0.0);
            ch[CH_LEAD].mono = true;
            ch[CH_BELL].mono = true;
            // outdoors: a short, faint reflection
            s.rev.fb = r.range(0.6, 0.68);
            s.rev.damp = 0.55;
            s.rev_wet = r.range(0.08, 0.12);
        }
        _ => {
            let air = style == CS_AIR;
            ch[CH_LEAD].patch = fiddle(r);
            ch[CH_BELL].patch = if air { flute(r) } else { ce_whistle(r) };
            ch[CH_ARP].patch = box_accordion(r);
            ch[CH_KEYS].patch = if air { folk_harp(r) } else { ce_guitar(r) };
            ch[CH_BASS].patch = folk_harp(r);
            ch[CH_DRONE].patch = pipe_drone(r);
            ch[CH_LEAD].mono = true;
            ch[CH_BELL].mono = true;
            ch[CH_ARP].mono = false;
            ch[CH_KICK].drums = kit_bodhran(r);
            ch[CH_SNARE].drums = kit_side_drum(r);
            ch[CH_PERC].drums = kit_bones(r);
            ch[CH_HAT].drums = kit_trim(kit_tambourine(r), -4.8);
            // around the table: the fiddle to one side, the flute or whistle
            // to the other, the box beside the fiddle, the guitar opposite
            mix(&mut ch[CH_LEAD], 0.42, 0.35, 0.2, 0.03, 0.0);
            mix(&mut ch[CH_BELL], 0.34, -0.45, 0.22, 0.03, 0.0);
            mix(&mut ch[CH_ARP], 0.3, 0.55, 0.18, 0.0, 0.0);
            mix(&mut ch[CH_KEYS], if air { 0.5 } else { 0.4 }, -0.3, 0.2, 0.0, 0.0);
            mix(&mut ch[CH_BASS], 0.42, -0.1, 0.2, 0.0, 0.0);
            mix(&mut ch[CH_DRONE], 0.18, 0.0, 0.2, 0.0, 0.0);
            mix(&mut ch[CH_KICK], 0.42, -0.1, 0.12, 0.0, 0.0);
            mix(&mut ch[CH_SNARE], 0.35, 0.1, 0.15, 0.0, 0.0);
            mix(&mut ch[CH_PERC], 0.3, -0.35, 0.15, 0.0, 0.0);
            mix(&mut ch[CH_HAT], 0.22, 0.3, 0.12, 0.0, 0.0);
            // a pub's room (an air's small hall)
            s.rev.fb = if air { r.range(0.8, 0.84) } else { r.range(0.7, 0.76) };
            s.rev.damp = 0.5;
            s.rev_wet = if air { r.range(0.22, 0.28) } else { r.range(0.13, 0.18) };
        }
    }
    for c in [CH_LEAD, CH_BELL, CH_ARP] {
        ch[c].width = if ch[c].patch.samp != 0 { 0.7 } else { 1.0 };
    }
    s.dly.fb = 0.0;
    s.dly_rev = 0.0;
    s.r2mode = R2_OFF;
    s.fx.crackle = 0.0;
    s.fx.wind_to = 0.0;
    s.fx.rain_to = 0.0;
    s.m.gain = 2.1;
    s.m.target = 0.21;
}
