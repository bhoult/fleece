//! STAR WARS: a space-opera film score in John Williams's manner, its tunes,
//! fanfares and progressions fleece's own (none of the films'): brass
//! fanfares in fourths and fifths over string ostinatos, a heroic march with
//! triplet pickups and a lyrical second theme on the strings with a horn
//! answering, a dark minor march on low brass over timpani and choir, and a
//! battle cue (a driving 4/4 or 6/8 ostinato, brass stabs, rising sequences,
//! a bar of 3/4 or 9/8 now and then); harp, celesta and glockenspiel for
//! colour, the choir for mystery, cymbal swells and rolls, a key change up a
//! step for the last statement and a final tutti hit.

use super::*;

/// The styles: the heroic march (a major key; in a minor one, the dark
/// march), the battle.
pub(super) const SWS_MARCH: u8 = 0;
pub(super) const SWS_BATTLE: u8 = 1;
pub(super) const SWS_DARK: u8 = 2;

/// A section's part: the opening fanfare, the main theme, the lyrical
/// second theme, the mystery (choir, harp, celesta), the build (a rising
/// sequence into the last statement), the battle's sequence, the finale.
const SP_FANFARE: u8 = 0;
const SP_THEME: u8 = 1;
const SP_LOVE: u8 = 2;
const SP_MYSTERY: u8 = 3;
const SP_BUILD: u8 = 4;
const SP_SEQ: u8 = 5;
const SP_FINALE: u8 = 6;

/// The cymbals' kinds (CH_CYM): a clash, a swell into the next downbeat
/// (2 s or 4 s), the gong.
pub(super) const CY_CLASH: u8 = 0;
const CY_SWELL2: u8 = 1;
const CY_SWELL4: u8 = 2;
pub(super) const CY_GONG: u8 = 3;

/// A note of a tune: its 16th from the tune's start (a fraction for a
/// triplet), its length in 16ths, its pitch (semitones above the tune's
/// tonic), how hard.
#[derive(Clone, Copy)]
pub(super) struct Tn {
    pub(super) st: f32,
    pub(super) len: f32,
    pub(super) p: i32,
    pub(super) v: f32,
}

/// A section of a Star Wars piece: its part, which of the instrument pairs,
/// which time through.
#[derive(Clone, Copy, Default)]
pub(super) struct SwPart {
    part: u8,
    dress: usize,
    rep: u32,
}

/// A Star Wars piece's state.
#[derive(Default)]
pub(super) struct StarWars {
    /// 16ths a bar (the section's usual) and a beat, the meter's name
    pub(super) bar_steps: u32,
    pub(super) beat_steps: u32,
    pub(super) meter: &'static str,
    style: u8,
    map: Vec<SwPart>,
    /// the section's bars, 16ths each (a battle's bar of 3/4 among its 4/4)
    bars: Vec<u32>,
    /// the progressions: the theme's, the second theme's, the mystery's, the
    /// sequence's (two chords a bar)
    progs: Vec<Vec<Harm>>,
    /// the main theme and the second (eight bars each), the fanfare (two)
    themes: [Vec<Tn>; 2],
    fanfare: Vec<Tn>,
    /// the fanfare's interval under its top line (5: fourths, 7: fifths)
    quartal: i32,
    /// the tunes' tonic (MIDI), the ostinato's figure
    tonic: i32,
    ost: u8,
    /// the brass's chord last time (voice leading)
    brass_v: Vec<i32>,
}

impl StarWars {
    /// The 16ths in bar `k` of the section (the parent's bar_steps()).
    pub(super) fn bs_at(&self, k: u32) -> u32 {
        self.bars.get(k as usize).copied().unwrap_or(self.bar_steps).max(1)
    }

    /// Which of the instrument pairs section `k` wears.
    pub(super) fn dress_at(&self, k: usize) -> usize {
        self.map.get(k).map_or(0, |x| x.dress)
    }
}

// -- the progressions (Roman numerals, a bar each, A+B two in the bar)

const SW_MARCH_A: &[&str] = &[
    "I I IV I bVI bVII I I",
    "I V^1 vi IV I bVII IV V",
    "I bVII IV I I bVI bVII V",
    "I IV+I I V I bVI IV V",
    "I iii IV I II II IV V",
    "I bIII IV I bVI IV bVII V",
];
const SW_MARCH_B: &[&str] =
    &["vi IV I V vi ii IV V", "IV V^1 vi iii IV I ii V", "I iii vi IV bVI IV V V", "IV I^1 ii V I vi bVI V"];
const SW_MYST_MAJ: &[&str] = &["vi vi IV IV bVI bVI V V", "I bVI I bVI iv iv V V", "vi iii IV I bVI bVI V V"];
const SW_MIN_A: &[&str] = &[
    "i i bvi bvi i i bII V",
    "i iv i bVI i bvi iv V",
    "i i iii iii i bVI bII V",
    "i bVI iv i i bII bVI V",
    "i iv bVII bIII bVI bII V V",
];
const SW_MIN_B: &[&str] = &["iv iv bVI bVI bvi bvi V V", "bVI bVI bVII bVII i i V V", "bVI iv bII bII bVI iv V V"];
const SW_SEQ: &[&str] = &["i bII bIII iv V bVI bVII V", "i i bII bII bIII bIII iv V", "iv V bVI bVII i bII bIII V"];
const SW_MYST_MIN: &[&str] = &["i i bvi bvi i i bvi V", "i bII i bII iv iv V V", "i bvi i iii iv iv V V"];

/// The pools parse (the chord-pool test's).
#[cfg(test)]
pub(super) const SW_POOLS: &[&[&str]] = &[SW_MARCH_A, SW_MARCH_B, SW_MYST_MAJ, SW_MIN_A, SW_MIN_B, SW_SEQ, SW_MYST_MIN];

// -- the tunes

/// A bar's rhythm cells: a leap (two long notes), a fall (a quarter, a
/// triplet falling, a half), dotted (a march's), held with a triplet pickup,
/// a climb, the end, lyrical (dotted, a step, a leap), a sigh.
const C_LEAP: u8 = 0;
const C_FALL: u8 = 1;
const C_DOT: u8 = 2;
const C_HOLD: u8 = 3;
const C_CLIMB: u8 = 4;
const C_END: u8 = 5;
const C_LYRIC: u8 = 6;
const C_SIGH: u8 = 7;

pub(super) const T3: f32 = 4.0 / 3.0;

/// A cell's notes (16th, length) in a bar of 16 (4/4) or 12 (6/8), and how
/// each moves from the last: scale steps, or M_CT the nearest chord tone,
/// M_LEAP a leap up to one, M_HOME the tonic, M_UNDER a fourth under the
/// note before (a pickup's repeated note).
fn sw_cell(bs: u32, kind: u8) -> (&'static [(f32, f32)], &'static [i32]) {
    let four = bs != 12;
    match (kind, four) {
        (C_LEAP, true) => (&[(0.0, 8.0), (8.0, 8.0)], &[M_CT, M_LEAP]),
        (C_LEAP, false) => (&[(0.0, 6.0), (6.0, 6.0)], &[M_CT, M_LEAP]),
        (C_FALL, true) => {
            (&[(0.0, 4.0), (4.0, T3), (4.0 + T3, T3), (4.0 + 2.0 * T3, T3), (8.0, 8.0)], &[1, -1, -1, -1, M_CT])
        }
        (C_FALL, false) => (&[(0.0, 3.0), (3.0, 1.0), (4.0, 1.0), (5.0, 1.0), (6.0, 6.0)], &[1, -1, -1, -1, M_CT]),
        (C_DOT, true) => (&[(0.0, 6.0), (6.0, 2.0), (8.0, 6.0), (14.0, 2.0)], &[M_CT, 1, M_CT, 1]),
        (C_DOT, false) => (&[(0.0, 3.0), (3.0, 1.0), (6.0, 4.0), (10.0, 2.0)], &[M_CT, 1, M_CT, 1]),
        (C_HOLD, true) => (&[(0.0, 12.0), (12.0, T3), (12.0 + T3, T3), (12.0 + 2.0 * T3, T3)], &[M_CT, M_UNDER, 0, 0]),
        (C_HOLD, false) => (&[(0.0, 6.0), (6.0, 2.0), (8.0, 2.0), (10.0, 2.0)], &[M_CT, M_UNDER, 0, 0]),
        (C_CLIMB, true) => (&[(0.0, 4.0), (4.0, 4.0), (8.0, 4.0), (12.0, 4.0)], &[1, 1, 1, M_CT]),
        (C_CLIMB, false) => (&[(0.0, 2.0), (2.0, 2.0), (4.0, 2.0), (6.0, 6.0)], &[1, 1, 1, M_CT]),
        (C_LYRIC, true) => (&[(0.0, 6.0), (6.0, 2.0), (8.0, 8.0)], &[M_CT, -1, M_LEAP]),
        (C_LYRIC, false) => (&[(0.0, 4.0), (4.0, 2.0), (6.0, 6.0)], &[M_CT, -1, M_LEAP]),
        (C_SIGH, true) => (&[(0.0, 4.0), (4.0, 4.0), (8.0, 6.0), (14.0, 2.0)], &[M_CT, 1, M_CT, -1]),
        (C_SIGH, false) => (&[(0.0, 4.0), (4.0, 2.0), (6.0, 4.0), (10.0, 2.0)], &[M_CT, 1, M_CT, -1]),
        (_, true) => (&[(0.0, 16.0)], &[M_HOME]),
        (_, false) => (&[(0.0, 12.0)], &[M_HOME]),
    }
}

pub(super) const M_CT: i32 = 50;
pub(super) const M_LEAP: i32 = 51;
pub(super) const M_HOME: i32 = 52;
pub(super) const M_UNDER: i32 = 53;

/// `p` moved `n` steps along the mode's scale (semitones above the tonic).
pub(super) fn sw_step(mode: usize, p: i32, n: i32) -> i32 {
    let sc = |q: i32| MODES[mode].contains(&q.rem_euclid(12));
    let mut q = p;
    for _ in 0..n.abs() {
        q += n.signum();
        while !sc(q) {
            q += n.signum();
        }
    }
    q
}

/// The tone of `h` nearest `p` (at or above it when `up`).
pub(super) fn sw_ct(h: Harm, p: i32, up: bool) -> i32 {
    let pcs = h.pcs();
    (0..12).flat_map(|k| [p + k, p - k]).find(|q| pcs.contains(&q.rem_euclid(12)) && (!up || *q >= p)).unwrap_or(p)
}

/// An eight-bar theme over `chords` (two a bar) in `mode`: bars of the cells
/// `kinds`, the first two coming back as the fifth and sixth, the third the
/// first moved up the scale (a sequence); between `lo` and `hi`. The leap
/// after a fifth never falls by a triplet into the octave (no film's theme).
pub(super) fn sw_theme(
    r: &mut Rng,
    chords: &[Harm],
    mode: usize,
    bs: u32,
    kinds: &[u8],
    lo: i32,
    hi: i32,
    start: i32,
) -> Vec<Tn> {
    let chord_at = |bar: usize, st: f32| chords[(2 * bar + (st * 2.0 >= bs as f32) as usize) % chords.len()];
    let mut v: Vec<Tn> = Vec::new();
    let mut cur = start;
    let mut last_leap = 0;
    let fb = bs as f32;
    for (bar, &k0) in kinds.iter().enumerate() {
        let t0 = bar as f32 * fb;
        if bar == 4 || bar == 5 || bar == 2 {
            // the motif again (bars 5, 6), or the first bar a step or two up
            let src = if bar == 2 { 0 } else { bar - 4 };
            let by = if bar == 2 { r.pick(&[1, 2, 1, -1]) } else { 0 };
            let notes: Vec<Tn> = v.iter().filter(|n| (n.st / fb) as usize == src).copied().collect();
            for n in notes {
                let st = n.st - src as f32 * fb;
                let mut p = sw_step(mode, n.p, by);
                if st == 0.0 || n.len >= fb / 2.0 {
                    p = sw_ct(chord_at(bar, st), p, false);
                }
                v.push(Tn { st: t0 + st, p: p.clamp(lo, hi), ..n });
                cur = p;
            }
            continue;
        }
        let k = if k0 == C_FALL && last_leap == 7 { C_DOT } else { k0 };
        let (cells, moves) = sw_cell(bs, k);
        last_leap = 0;
        for (&(st, len), &mv) in cells.iter().zip(moves) {
            let h = chord_at(bar, st);
            let strong = st == 0.0 || st * 2.0 == fb || len >= fb / 2.0;
            let mut p = match mv {
                M_CT => sw_ct(h, cur, false),
                M_HOME => {
                    let t = (cur as f32 / 12.0).round() as i32 * 12;
                    if t > hi {
                        t - 12
                    } else if t < lo {
                        t + 12
                    } else {
                        t
                    }
                }
                M_UNDER => sw_step(mode, cur, -3),
                M_LEAP => {
                    let iv = [5, 7, 9, 12][r.weighted(&[3.0, 1.0, 2.0, 1.5])];
                    let mut q = sw_ct(h, cur + iv, true);
                    if q > hi {
                        q = sw_ct(h, cur - 5, false);
                    }
                    last_leap = q - cur;
                    q
                }
                n => sw_step(mode, cur, n),
            };
            if strong && !matches!(mv, M_HOME | M_LEAP) {
                p = sw_ct(h, p, false);
            }
            while p > hi {
                p -= 12;
            }
            while p < lo {
                p += 12;
            }
            let vel = if strong { 0.95 } else { 0.8 } - if len < 2.0 { 0.06 } else { 0.0 };
            v.push(Tn { st: t0 + st, len, p, v: vel });
            cur = p;
        }
    }
    v
}

/// The fanfare: two bars of a brass call (its top line; the brass under it
/// in fourths or fifths), repeated notes in triplets leaping up a fourth or
/// a fifth, held.
fn sw_fanfare(r: &mut Rng, bs: u32) -> Vec<Tn> {
    let shapes: &[&[(f32, f32, i32)]] = &[
        &[
            (0.0, T3, 7),
            (T3, T3, 7),
            (2.0 * T3, T3, 7),
            (4.0, 12.0, 12),
            (16.0, T3, 12),
            (16.0 + T3, T3, 12),
            (16.0 + 2.0 * T3, T3, 12),
            (20.0, 4.0, 14),
            (24.0, 8.0, 7),
        ],
        &[
            (0.0, 8.0, 0),
            (8.0, 4.0, 5),
            (12.0, 4.0, 7),
            (16.0, 12.0, 12),
            (28.0, T3, 7),
            (28.0 + T3, T3, 7),
            (28.0 + 2.0 * T3, T3, 7),
        ],
        &[
            (0.0, 2.0, 7),
            (2.0, 2.0, 7),
            (4.0, 4.0, 12),
            (8.0, 2.0, 10),
            (10.0, 2.0, 12),
            (12.0, 4.0, 14),
            (16.0, 16.0, 12),
        ],
        &[
            (0.0, 6.0, 0),
            (6.0, 2.0, 7),
            (8.0, 8.0, 12),
            (16.0, T3, 10),
            (16.0 + T3, T3, 12),
            (16.0 + 2.0 * T3, T3, 14),
            (20.0, 12.0, 17),
        ],
        &[
            (0.0, T3, 0),
            (T3, T3, 0),
            (2.0 * T3, T3, 0),
            (4.0, 4.0, 5),
            (8.0, 8.0, 7),
            (16.0, 4.0, 12),
            (20.0, 4.0, 10),
            (24.0, 8.0, 7),
        ],
    ];
    let sh = r.pick(shapes);
    let k = bs as f32 / 16.0;
    sh.iter().map(|&(st, len, p)| Tn { st: st * k, len: len * k, p, v: if len >= 4.0 { 1.0 } else { 0.85 } }).collect()
}

// -- the instruments: each the synthesiser's patch first (the same draws
// with the samples or without), then its recording

fn sw_strings_synth(r: &mut Rng) -> Patch {
    // a section: three saws apart, a bowed body, a slow vibrato
    Patch {
        unison: 3,
        detune: r.range(9.0, 14.0),
        width: 0.7,
        vowel: 1,
        noise: 0.01,
        cutoff: r.range(3200.0, 4200.0),
        keytrack: 0.3,
        a: 0.09,
        d: 0.5,
        s: 0.9,
        r: 0.35,
        vib: r.range(0.12, 0.18),
        vib_rate: r.range(5.2, 5.8),
        glide: 0.02,
        gain: 0.3,
        ..PATCH
    }
}

fn sw_spic_synth(r: &mut Rng) -> Patch {
    Patch { a: 0.006, d: 0.12, s: 0.25, r: 0.08, vib: 0.0, ..sw_strings_synth(r) }
}

fn sw_low_strings_synth(r: &mut Rng) -> Patch {
    Patch { vowel: 4, cutoff: r.range(1800.0, 2400.0), keytrack: 0.6, ..sw_strings_synth(r) }
}

/// The violins: held, or spiccato under a quarter of a second (CC0 section).
pub(super) fn sw_violins(r: &mut Rng) -> Patch {
    rec(sw_strings_synth(r), multi_or("violins", "auto", "Strings Slow"), 74.0)
}

fn sw_violins_spic(r: &mut Rng) -> Patch {
    rec(sw_spic_synth(r), multi_or("violins", "stac", "Strings Pizzicato"), 72.0)
}

fn sw_violas_spic(r: &mut Rng) -> Patch {
    rec(sw_spic_synth(r), multi_or("violas", "stac", "Viola"), 62.0)
}

fn sw_cellos_spic(r: &mut Rng) -> Patch {
    let p = Patch { vowel: 4, cutoff: 2000.0, keytrack: 0.6, ..sw_spic_synth(r) };
    rec(p, multi_or("cellos", "stac", "Cello"), 48.0)
}

/// The tremolo: the violins' (CC0) or the SoundFont's strings.
pub(super) fn sw_tremolo(r: &mut Rng) -> Patch {
    let p = Patch { trem: 0.6, vib_rate: r.range(11.0, 13.0), vib: 0.0, ..sw_strings_synth(r) };
    rec(p, multi_or("violins_trem", "sus", "Strings Tremolo"), 72.0)
}

/// The string ensemble held (the SoundFont's two, the fast one brighter).
fn sw_ensemble(r: &mut Rng) -> Patch {
    rec(sw_strings_synth(r), samples::gm("Strings Fast"), 62.0)
}

fn sw_violas(r: &mut Rng) -> Patch {
    rec(sw_low_strings_synth(r), multi_or("violas", "sus", "Viola"), 62.0)
}

/// The basses: the cellos and the bowed contrabass (CC0).
pub(super) fn sw_cellos(r: &mut Rng) -> Patch {
    rec(sw_low_strings_synth(r), multi_or("cellos", "auto", "Cello"), 48.0)
}

fn sw_contrabass(r: &mut Rng) -> Patch {
    let p = Patch { cutoff: 900.0, keytrack: 1.0, ..sw_low_strings_synth(r) };
    rec(p, multi_or("contrabass", "sus", "Contrabass"), 40.0)
}

fn sw_brass_synth(r: &mut Rng) -> Patch {
    Patch { unison: 3, detune: r.range(6.0, 9.0), width: 0.5, a: 0.03, r: 0.25, ..horn_synth(r) }
}

/// The brass section held (the SoundFont's), the horns' section.
fn sw_brass(r: &mut Rng) -> Patch {
    rec(sw_brass_synth(r), samples::gm("Brass Section"), 60.0)
}

fn sw_horns(r: &mut Rng) -> Patch {
    rec(sw_brass_synth(r), samples::gm("French Horns"), 58.0)
}

/// The low brass's chords: trombones (CC0).
fn sw_trombones(r: &mut Rng) -> Patch {
    let p = Patch { unison: 2, detune: 6.0, width: 0.4, ..trombone_synth(r) };
    rec(p, multi_or("trombone", "sus", "Trombone"), 52.0)
}

fn sw_oboe_synth(r: &mut Rng) -> Patch {
    Patch {
        vowel: 2,
        cutoff: r.range(2600.0, 3200.0),
        res: 0.2,
        vib: 0.1,
        vib_rate: 5.5,
        gain: 0.25,
        ..clarinet_synth(r)
    }
}

fn sw_oboe(r: &mut Rng) -> Patch {
    rec(sw_oboe_synth(r), multi_or("oboe", "auto", "Oboe"), 72.0)
}

fn sw_bassoon(r: &mut Rng) -> Patch {
    let p = Patch { vowel: 4, cutoff: r.range(900.0, 1200.0), ..sw_oboe_synth(r) };
    rec(p, multi_or("bassoon", "auto", "Bassoon"), 48.0)
}

fn sw_timpani_synth(r: &mut Rng) -> Patch {
    // a membrane: a sine and an inharmonic partial falling from a little
    // sharp, the mallet's thud, ringing
    Patch {
        wave: Wave::Sine,
        fm: r.range(0.8, 1.2),
        ratio: 1.5,
        fm_dec: 0.08,
        noise: 0.08,
        cutoff: 900.0,
        keytrack: 0.5,
        a: 0.002,
        d: r.range(1.2, 1.6),
        s: 0.0,
        r: 0.9,
        bend: 0.35,
        glide: 0.05,
        gain: 0.7,
        ..PATCH
    }
}

/// The timpani (the SoundFont's, nine layers), on a tonal channel: tuned
/// to the piece's tonic and dominant, played by note.
pub(super) fn sw_timpani(r: &mut Rng) -> Patch {
    rec(sw_timpani_synth(r), samples::gm("Timpani"), 43.0)
}

/// The choir: "ah" (the SoundFont's choir), or the voices' "oo".
fn sw_choir(r: &mut Rng) -> Patch {
    let p = Patch { vowel: 1, ..pad_choir_synth(r) };
    rec(p, samples::gm("Choir Aahs"), 60.0)
}

fn sw_oohs(r: &mut Rng) -> Patch {
    let p = Patch { vowel: 5, ..pad_choir_synth(r) };
    rec(p, samples::gm("Voice Oohs"), 60.0)
}

// the percussion: kits on the drum channels

fn sw_kit_bass_drum_synth(r: &mut Rng) -> Vec<DrumPatch> {
    let k = DrumPatch {
        noise: 0.2,
        nmode: 0,
        ncut: 400.0,
        ndec: 0.4,
        gain: 0.9,
        ..kick(r, (60.0, 70.0), (42.0, 48.0), (0.9, 1.3), 0.0, 0.0)
    };
    vec![k]
}

/// The orchestra's bass drum (CC0, four layers).
pub(super) fn sw_kit_bass_drum(r: &mut Rng) -> Vec<DrumPatch> {
    rec_kit(sw_kit_bass_drum_synth(r), &[multi_or("bass_drum", "hit", "Concert Bass Drum")])
}

pub(super) fn sw_kit_snare_synth(r: &mut Rng) -> Vec<DrumPatch> {
    let sn = DrumPatch {
        tone: 0.35,
        f0: r.range(240.0, 280.0),
        f1: 200.0,
        pdec: 0.02,
        tdec: 0.12,
        noise: 0.9,
        nmode: 1,
        ncut: r.range(3500.0, 4500.0),
        nres: 0.15,
        ndec: 0.18,
        gain: 0.55,
        ..DRUM
    };
    vec![sn, DrumPatch { ndec: 0.1, gain: 0.4, ..sn }]
}

/// The concert snare (CC0, six layers), or the field drum's (the western's).
fn sw_kit_snare(r: &mut Rng) -> Vec<DrumPatch> {
    let id = multi_or("snare", "hit", "Brush");
    rec_kit(sw_kit_snare_synth(r), &[id, id])
}

fn sw_kit_triangle_synth(r: &mut Rng) -> Vec<DrumPatch> {
    let t = DrumPatch {
        tone: 0.6,
        f0: r.range(4800.0, 5600.0),
        f1: 0.0,
        pdec: 1.0,
        tdec: 2.2,
        tone2: 0.5,
        ratio2: 2.76,
        noise: 0.05,
        nmode: 2,
        ncut: 8000.0,
        ndec: 0.05,
        gain: 0.25,
        ..DRUM
    };
    vec![DrumPatch { f1: t.f0, ..t }]
}

fn sw_kit_triangle(r: &mut Rng) -> Vec<DrumPatch> {
    rec_kit(sw_kit_triangle_synth(r), &[samples::multi("triangle", "hit")])
}

pub(super) fn sw_kit_cymbals_synth(r: &mut Rng) -> Vec<DrumPatch> {
    let clash = DrumPatch {
        noise: 1.0,
        nmode: 2,
        ncut: r.range(3000.0, 4200.0),
        nres: 0.1,
        ndec: r.range(2.4, 3.2),
        metal: 0.6,
        mf: r.range(2.0, 3.0),
        gain: 0.6,
        ..DRUM
    };
    let gong = DrumPatch {
        tone: 0.7,
        f0: r.range(80.0, 95.0),
        f1: 0.0,
        pdec: 1.0,
        tdec: 5.0,
        tone2: 0.5,
        ratio2: 2.32,
        noise: 0.4,
        nmode: 0,
        ncut: 1200.0,
        ndec: 3.0,
        natt: 0.15,
        metal: 0.4,
        mf: 0.5,
        gain: 0.8,
        ..DRUM
    };
    // (the swells: a riser on the effects channel where nothing is recorded)
    let swell = DrumPatch { gain: 0.0, ..clash };
    vec![clash, swell, swell, DrumPatch { f1: gong.f0 * 0.98, ..gong }]
}

/// The clash cymbals, the suspended cymbal's swells, the gong (CC0).
fn sw_kit_cymbals(r: &mut Rng) -> Vec<DrumPatch> {
    let v = sw_kit_cymbals_synth(r);
    let swell = |a: &str| samples::multi("cymbal", a).or_else(|| samples::gm("Reverse Cymbal"));
    let mut k = rec_kit(
        v,
        &[samples::multi("cymbals", "clash"), swell("swell2"), swell("swell4"), samples::multi("gong", "ring")],
    );
    // the swells' manifest gains read their quiet first second (their
    // peaks come at the end): down to a clash's peak
    for d in k[1..3].iter_mut().filter(|d| d.samp != 0) {
        d.gain = 0.6 * 10f32.powf(-16.0 / 20.0);
    }
    k
}

// the palettes (each a role's alternates, the mixes' borrowings)

const P_SW_THEME: &[(&str, Make)] = &[("trumpets", trumpet), ("horns", horn), ("violins", sw_violins)];
const P_SW_BRASS: &[(&str, Make)] =
    &[("brass section", sw_brass), ("horn section", sw_horns), ("trombones", sw_trombones)];
const P_SW_OSTINATO: &[(&str, Make)] = &[
    ("violins, spiccato", sw_violins_spic),
    ("violas, spiccato", sw_violas_spic),
    ("cellos, spiccato", sw_cellos_spic),
];
const P_SW_WINDS: &[(&str, Make)] =
    &[("flute", flute), ("oboe", sw_oboe), ("clarinet", clarinet), ("bassoon", sw_bassoon)];
const P_SW_STRINGS: &[(&str, Make)] =
    &[("string ensemble", sw_ensemble), ("strings, slow", pad_strings), ("violas", sw_violas), ("tremolo", sw_tremolo)];
const P_SW_CHOIR: &[(&str, Make)] = &[("choir", sw_choir), ("voices", sw_oohs)];
const P_SW_COLOUR: &[(&str, Make)] = &[("harp", harp), ("celesta", celesta), ("glockenspiel", glockenspiel)];
const P_SW_BASS: &[(&str, Make)] = &[("cellos", sw_cellos), ("contrabass", sw_contrabass), ("tuba", tuba)];
const K_SW_KICK: &[(&str, MakeKit)] = &[("bass drum", sw_kit_bass_drum)];
const K_SW_SNARE: &[(&str, MakeKit)] = &[("snare drum", sw_kit_snare)];
const K_SW_HAT: &[(&str, MakeKit)] = &[("triangle", sw_kit_triangle)];
const K_SW_CYM: &[(&str, MakeKit)] = &[("cymbals and gong", sw_kit_cymbals)];

/// Star Wars's instruments for the part channel `ch` plays in another genre.
pub(super) fn palette_starwars(ch: usize) -> Option<&'static [(&'static str, Make)]> {
    Some(match ch {
        CH_LEAD => P_SW_THEME,
        CH_KEYS => P_SW_BRASS,
        CH_ARP => P_SW_OSTINATO,
        CH_PAD => P_SW_STRINGS,
        CH_DRONE => P_SW_CHOIR,
        CH_BELL => P_SW_COLOUR,
        CH_BASS => P_SW_BASS,
        _ => return None,
    })
}

/// Its kits for drum channel `ch`.
pub(super) fn kit_palette_starwars(ch: usize) -> Option<&'static [(&'static str, MakeKit)]> {
    Some(match ch {
        CH_KICK => K_SW_KICK,
        CH_SNARE => K_SW_SNARE,
        CH_HAT => K_SW_HAT,
        CH_CYM => K_SW_CYM,
        _ => return None,
    })
}

/// A channel's pair: its original (`s`'s patch) and `b`, or `a` and `b`,
/// both levelled to the original at `note`.
fn sw_pair(band: &mut Band, r: &mut Rng, s: &Synth, ch: usize, a: Option<Make>, b: Make, note: f32) {
    let orig = s.ch[ch].patch;
    let pa = a.map_or(orig, |f| levelled(&orig, note, f(r)));
    let pb = levelled(&orig, note, b(r));
    band.tones.push((ch, [pa, pb], note));
}

/// The band: for each role the verses' instrument (A) and the second
/// theme's and the mystery's (B).
fn band_starwars(r: &mut Rng, s: &Synth, style: u8) -> Band {
    let mut b = Band::default();
    match style {
        SWS_MARCH => {
            sw_pair(&mut b, r, s, CH_LEAD, None, sw_violins, 74.0);
            sw_pair(&mut b, r, s, CH_KEYS, None, sw_horns, 58.0);
            let w: Make = P_SW_WINDS[r.below(3)].1;
            sw_pair(&mut b, r, s, CH_ARP, None, w, 72.0);
        }
        SWS_DARK => {
            sw_pair(&mut b, r, s, CH_LEAD, None, sw_trombones, 58.0);
            sw_pair(&mut b, r, s, CH_KEYS, None, sw_horns, 52.0);
            sw_pair(&mut b, r, s, CH_ARP, None, sw_bassoon, 50.0);
        }
        _ => {
            sw_pair(&mut b, r, s, CH_LEAD, None, trumpet, 66.0);
            sw_pair(&mut b, r, s, CH_KEYS, None, sw_trombones, 56.0);
            sw_pair(&mut b, r, s, CH_ARP, None, sw_violas_spic, 52.0);
        }
    }
    sw_pair(&mut b, r, s, CH_PAD, None, if style == SWS_BATTLE { sw_tremolo } else { sw_choir }, 62.0);
    sw_pair(&mut b, r, s, CH_BELL, None, if style == SWS_BATTLE { glockenspiel } else { celesta }, 74.0);
    sw_pair(&mut b, r, s, CH_DRONE, None, sw_oohs, 45.0);
    sw_pair(&mut b, r, s, CH_BASS, None, if style == SWS_DARK { tuba } else { sw_contrabass }, 40.0);
    b
}

/// The recordings against the synthesiser's patches they replace, by style
/// and channel (dB): measured over whole pieces (each channel alone, the
/// leveller frozen, K-weighted mean power with the samples against
/// without, the same notes; 8 seeds' first 300 s, by the pieces' styles).
/// The levelling's one early note misreads the spiccato's short notes, the
/// held strings' swell and the low brass; the bass drum, the cymbals and
/// the gong ring otherwise than the synthesiser's.
fn sw_rec_trim(style: u8, ch: usize) -> f32 {
    match (style, ch) {
        (SWS_MARCH, CH_BASS) => 3.0,
        (SWS_MARCH, CH_PAD) => 1.5,
        (SWS_MARCH, CH_ARP) => -4.2,
        (SWS_MARCH, CH_BELL) => 1.1,
        (SWS_MARCH, CH_KEYS) => 0.9,
        (SWS_DARK, CH_BASS) => 5.1,
        (SWS_DARK, CH_LEAD) => 0.8,
        (SWS_DARK, CH_KEYS) => 2.1,
        (SWS_DARK, CH_DRONE) => 0.7,
        (SWS_BATTLE, CH_BASS) => 2.4,
        (SWS_BATTLE, CH_PAD) => 4.4,
        (SWS_BATTLE, CH_ARP) => -1.6,
        (SWS_BATTLE, CH_LEAD) => 2.2,
        (SWS_BATTLE, CH_BELL) => -2.2,
        (SWS_BATTLE, CH_KEYS) => 1.8,
        (SWS_BATTLE, CH_DRONE) => -0.9,
        _ => 0.0,
    }
}

/// `sw_rec_trim`'s for the kits.
fn sw_kit_trims(style: u8) -> [(usize, f32); 4] {
    match style {
        SWS_MARCH => [(CH_KICK, 7.7), (CH_SNARE, -0.7), (CH_HAT, 9.3), (CH_CYM, 5.5)],
        SWS_DARK => [(CH_KICK, 8.4), (CH_SNARE, -0.5), (CH_HAT, 9.3), (CH_CYM, 10.8)],
        _ => [(CH_KICK, 7.7), (CH_SNARE, -0.5), (CH_HAT, 9.3), (CH_CYM, 4.4)],
    }
}

/// Star Wars's sounds (a mix's drummer's: the march's or the battle's).
pub(super) fn sounds_starwars(r: &mut Rng, s: &mut Synth) {
    let style = if r.bit_chance() { SWS_MARCH } else { SWS_BATTLE };
    sw_sounds(r, s, style)
}

/// The orchestra as it sits: the first violins (or the theme's brass) to
/// the left, the brass right of centre, the low strings right, the harp
/// left, the timpani at the back; a large hall. Recorded where the samples
/// are in (each patch says which).
fn sw_sounds(r: &mut Rng, s: &mut Synth, style: u8) {
    let ch = &mut s.ch;
    let (lead, keys, arp): (Make, Make, Make) = match style {
        SWS_MARCH => (trumpet, sw_brass, sw_violins_spic),
        SWS_DARK => (horn, sw_trombones, sw_cellos_spic),
        _ => (horn, sw_brass, sw_cellos_spic),
    };
    ch[CH_LEAD].patch = lead(r);
    ch[CH_LEAD].mono = false;
    ch[CH_KEYS].patch = keys(r);
    ch[CH_ARP].patch = arp(r);
    ch[CH_ARP].mono = false;
    ch[CH_PAD].patch = if style == SWS_DARK { sw_choir(r) } else { sw_ensemble(r) };
    ch[CH_BELL].patch = harp(r);
    ch[CH_DRONE].patch = sw_timpani(r);
    ch[CH_DRONE].mono = false;
    ch[CH_BASS].patch = sw_cellos(r);
    ch[CH_BASS].mono = false;
    ch[CH_KICK].drums = sw_kit_bass_drum(r);
    ch[CH_SNARE].drums = sw_kit_snare(r);
    ch[CH_HAT].drums = sw_kit_triangle(r);
    ch[CH_CYM].drums = sw_kit_cymbals(r);
    // the recorded kits at the synthesiser's balance (`SW_REC_TRIMS`'s measure)
    for (c, db) in sw_kit_trims(style) {
        ch[c].drums = kit_trim(std::mem::take(&mut ch[c].drums), db);
    }
    // the balance (measured over whole pieces, K-weighted): the tune on
    // top, the brass 3 dB under it, the strings 4-6, the ostinato 6-8, the
    // timpani and the percussion well under (the low strings' ostinato
    // speaks quietly, so its channel is up)
    let (arp, bass, timp, snare) = match style {
        SWS_MARCH => (1.25, 0.95, 0.17, 0.76),
        SWS_DARK => (4.0, 0.75, 0.15, 0.6),
        _ => (2.5, 1.06, 0.21, 0.68),
    };
    mix(&mut ch[CH_LEAD], 0.8, -0.3, 0.32, 0.0, 0.0);
    mix(&mut ch[CH_KEYS], 0.55, 0.4, 0.35, 0.0, 0.0);
    mix(&mut ch[CH_ARP], arp, -0.45, 0.28, 0.0, 0.0);
    mix(&mut ch[CH_PAD], if style == SWS_BATTLE { 0.5 } else { 0.53 }, 0.0, 0.38, 0.0, 0.0);
    mix(&mut ch[CH_BELL], 0.5, -0.6, 0.4, 0.0, 0.0);
    mix(&mut ch[CH_DRONE], timp, 0.2, 0.35, 0.0, 0.0);
    mix(&mut ch[CH_BASS], bass, 0.45, 0.25, 0.0, 0.0);
    mix(&mut ch[CH_KICK], 0.3, 0.1, 0.35, 0.0, 0.0);
    mix(&mut ch[CH_SNARE], snare, 0.15, 0.3, 0.0, 0.0);
    mix(&mut ch[CH_HAT], 0.4, -0.3, 0.4, 0.0, 0.0);
    mix(&mut ch[CH_CYM], 0.42, 0.2, 0.4, 0.0, 0.0);
    mix(&mut ch[CH_FX], 0.35, 0.0, 0.5, 0.0, 0.0);
    // the hall's recordings narrowed to where they sit
    for (c, w) in [(CH_LEAD, 0.7), (CH_KEYS, 0.85), (CH_ARP, 0.85), (CH_BASS, 0.7), (CH_DRONE, 0.8)] {
        ch[c].width = if ch[c].patch.samp != 0 { w } else { 1.0 };
    }
    s.rev.fb = r.range(0.86, 0.9);
    s.rev.damp = 0.35;
    s.rev_wet = 0.32;
    s.dly.fb = 0.0;
    s.dly_rev = 0.0;
    s.r2mode = R2_OFF;
    s.fx.crackle = 0.0;
    s.fx.hiss = 0.0;
    s.fx.wind_to = 0.0;
    s.fx.rain_to = 0.0;
    s.m.gain = 1.6;
    s.m.target = 0.2;
}

impl Song {
    pub(super) fn piece_starwars(&mut self, s: &mut Synth) {
        // (no Dorian: its major fourth in a minor key is the films' own)
        if self.mode == 1 {
            self.mode = AEOLIAN;
        }
        let minor = MODES[self.mode][2] == 3;
        let mode = self.mode;
        let r = &mut self.rng;
        let style = if minor {
            if r.chance(0.55) { SWS_BATTLE } else { SWS_DARK }
        } else if r.chance(0.2) {
            SWS_BATTLE
        } else {
            SWS_MARCH
        };
        let six = style == SWS_BATTLE && r.chance(0.35);
        let (bs, beat, bpm, meter) = match style {
            SWS_MARCH => (16u32, 4u32, r.range(100.0, 118.0), "4/4 march"),
            SWS_DARK => (16, 4, r.range(86.0, 100.0), "4/4 dark march"),
            _ if six => (12, 6, r.range(96.0, 112.0), "6/8 battle"),
            _ => (16, 4, r.range(138.0, 156.0), "4/4 battle"),
        };
        self.bpm = bpm as f64;
        self.swing = 0.0;
        self.human = 0.006;
        let center = match style {
            SWS_MARCH => r.pick(&[67, 69, 70]),
            SWS_DARK => r.pick(&[57, 59, 60]),
            _ => r.pick(&[62, 64, 65]),
        };
        let tonic = center - 6 + (self.root - (center - 6)).rem_euclid(12);
        let pick = |r: &mut Rng, pool: &[&str]| halves(r.pick(pool));
        let (pa, pb, pm) =
            if minor { (SW_MIN_A, SW_MIN_B, SW_MYST_MIN) } else { (SW_MARCH_A, SW_MARCH_B, SW_MYST_MAJ) };
        let a = pick(r, pa);
        let mut bb = pick(r, pb);
        if style == SWS_BATTLE && r.chance(0.5) {
            bb = pick(r, SW_SEQ);
        }
        let progs = vec![a.clone(), bb.clone(), pick(r, pm), pick(r, SW_SEQ)];
        // the themes: heroic (the march's, the battle's), lyrical (the
        // second), dark
        let (hero, lyric, dark): (&[u8], &[u8], &[u8]) = (
            &[C_LEAP, C_FALL, C_LEAP, C_HOLD, C_LEAP, C_DOT, C_CLIMB, C_END],
            &[C_LYRIC, C_SIGH, C_LYRIC, C_HOLD, C_LYRIC, C_SIGH, C_CLIMB, C_END],
            &[C_DOT, C_HOLD, C_DOT, C_LEAP, C_DOT, C_HOLD, C_SIGH, C_END],
        );
        let ka = if style == SWS_DARK { dark } else { hero };
        let start = r.pick(&[0, -5, 4]);
        let t0 = sw_theme(r, &a, mode, bs, ka, -5, 14, start);
        let s1 = r.pick(&[4, 7, 0]);
        let t1 = sw_theme(r, &bb, mode, bs, if style == SWS_MARCH { lyric } else { ka }, -3, 16, s1);
        let fanfare = sw_fanfare(r, bs);
        let quartal = r.pick(&[5, 5, 7]);
        // the plan: (section, its part)
        let bar_s = bs as f64 / beat as f64 * 60.0 / self.bpm;
        let long = if bar_s < 1.4 { 16 } else { 8 };
        let mk = |kind: u8, bars: u32, mask: u32, energy: f32| Sec { kind, bars, mask, energy };
        let part = |part: u8, dress: usize, rep: u32| SwPart { part, dress, rep };
        let strings = L_PAD | L_BASS;
        let perc = L_KICK | L_SNARE;
        let intro = (mk(INTRO, 4, L_LEAD | L_KEYS | strings | L_DRONE | L_KICK, 0.75), part(SP_FANFARE, 0, 0));
        let theme = |rep: u32, e: f32| {
            let ctr = if rep > 0 { L_CTR } else { 0 };
            (
                mk(MAIN, long, L_LEAD | L_KEYS | L_ARP | strings | L_DRONE | L_BELL | perc | ctr, e),
                part(SP_THEME, 0, rep),
            )
        };
        let love =
            (mk(MAIN, long, L_LEAD | L_KEYS | L_ARP | strings | L_BELL | L_HAT | L_CTR, 0.6), part(SP_LOVE, 1, 0));
        let myst = (mk(BREAK, long.min(8), L_LEAD | strings | L_BELL | L_DRONE, 0.4), part(SP_MYSTERY, 2, 0));
        let build = (mk(BUILD, 4, L_ARP | L_KEYS | strings | L_DRONE | perc | L_LEAD, 0.85), part(SP_BUILD, 0, 0));
        let seq =
            (mk(MAIN, long, L_LEAD | L_KEYS | L_ARP | strings | L_DRONE | L_BELL | perc, 0.9), part(SP_SEQ, 1, 0));
        let fin = (
            mk(OUTRO, 3, L_LEAD | L_KEYS | L_ARP | strings | L_DRONE | L_BELL | perc | L_HAT, 1.0),
            part(SP_FINALE, 0, 0),
        );
        let plan = match style {
            SWS_MARCH if r.chance(0.5) => {
                vec![intro, theme(0, 0.8), theme(1, 0.9), love, myst, build, theme(2, 1.0), fin]
            }
            SWS_MARCH => vec![intro, theme(0, 0.8), love, theme(1, 0.9), myst, build, theme(2, 1.0), fin],
            SWS_DARK => vec![intro, theme(0, 0.75), theme(1, 0.85), myst, seq, build, theme(2, 1.0), fin],
            _ => vec![intro, theme(0, 0.85), seq, theme(1, 0.9), myst, build, theme(2, 1.0), fin],
        };
        self.plan = plan.iter().map(|x| x.0).collect();
        // the last statement and the finale up a step (a half step, the dark march's)
        let n = self.plan.len();
        self.lifts = vec![0; n];
        let up = if style == SWS_DARK { 1 } else { r.pick(&[2, 2, 1]) };
        for l in self.lifts[n - 2..].iter_mut() {
            *l = up;
        }
        self.progs = progs.clone();
        self.prog = progs[0].clone();
        self.sw = StarWars {
            bar_steps: bs,
            beat_steps: beat,
            meter,
            style,
            map: plan.iter().map(|x| x.1).collect(),
            bars: Vec::new(),
            progs,
            themes: [t0, t1],
            fanfare,
            quartal,
            tonic,
            ost: r.below(3) as u8,
            brass_v: Vec::new(),
        };
        sw_sounds(r, s, style);
        self.band = band_starwars(r, s, style);
        rec_trim(&mut self.band, s, |ch, _| sw_rec_trim(style, ch));
    }

    /// A section begins: its chords, its bars' lengths (the battle's bar of
    /// 3/4 or 9/8 closing a phrase now and then; the finale's last bar
    /// long, for its chord to ring).
    pub(super) fn starwars_section(&mut self) {
        let sec = self.plan[self.sec];
        let p = self.sw.map[self.sec];
        let (bs, beat) = (self.sw.bar_steps, self.sw.beat_steps);
        let which = match p.part {
            SP_LOVE => 1,
            SP_MYSTERY => 2,
            SP_SEQ | SP_BUILD => 3,
            _ => 0,
        };
        self.prog = self.sw.progs[which].clone();
        let mut bars = vec![bs; sec.bars as usize];
        if self.sw.style == SWS_BATTLE && matches!(p.part, SP_SEQ | SP_THEME) {
            for k in (7..sec.bars as usize).step_by(8) {
                if self.rng.chance(0.5) {
                    bars[k] = if beat == 6 { bs + 6 } else { bs - 4 };
                }
            }
        }
        if p.part == SP_FINALE {
            if let Some(l) = bars.last_mut() {
                *l = bs + 2 * beat;
            }
        }
        self.sw.bars = bars;
    }

    /// The bar's two chords.
    fn sw_chords(&self) -> [Harm; 2] {
        [self.chord(self.sec_bar * 2), self.chord(self.sec_bar * 2 + 1)]
    }

    /// A bar of Star Wars.
    pub(super) fn starwars(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let p = self.sw.map[self.sec];
        let i = self.sec_bar;
        let bs = self.sw.bs_at(i);
        let last = i + 1 == sec.bars;
        let mut c = self.sw_chords();
        if i == 0 {
            // a film score's arc: the leveller's target with the section's
            // weight (the mystery well under the last statement)
            s.m.target = 0.2 * (sec.energy * sec.energy).max(0.15);
        }
        if p.part == SP_FINALE {
            self.sw_finale(s, b, sec, m);
            return;
        }
        if p.part == SP_BUILD {
            // the rising sequence: the bar's chord a step up from the last
            let k = [0, 2, 3, 5, 7][(i as usize).min(4)];
            let q = if MODES[self.mode][2] == 3 { Q_MIN } else { Q_MAJ };
            c = [Harm { root: k, q, inv: 0 }; 2];
            if last {
                c[1] = Harm { root: 7, q: Q_DOM7, inv: 0 };
            }
        }
        let e = sec.energy;
        // the strings held; the tremolo in a build
        if m & L_PAD != 0 {
            let (lo, hi) = if self.sw.style == SWS_DARK { (50, 72) } else { (55, 79) };
            let v = 0.45 + 0.3 * e;
            if p.part == SP_BUILD || p.part == SP_FANFARE {
                self.sw_held(b, CH_PAD, c, lo, hi, v * (0.7 + 0.3 * (i as f32 + 1.0) / sec.bars as f32), bs);
            } else {
                self.pads(b, CH_PAD, lo, hi, v, false, 0.5);
            }
        }
        if m & L_BASS != 0 {
            self.sw_bass(s, b, c, p.part, bs, e);
        }
        if m & L_ARP != 0 {
            if p.part == SP_LOVE || p.part == SP_MYSTERY || s.ch[CH_ARP].patch.a > 0.08 {
                let center = self.tonic_near(if self.sw.style == SWS_DARK { 52 } else { 64 });
                if s.ch[CH_ARP].patch.a > 0.08 {
                    self.sw_held(b, CH_ARP, c, center - 7, center + 7, 0.4 + 0.2 * e, bs);
                } else {
                    self.counter(b, CH_ARP, center, 0.55 + 0.2 * e, false);
                }
            } else {
                self.sw_ostinato(b, c, bs, e, p.part);
            }
        }
        if m & L_KEYS != 0 {
            self.sw_brass(s, b, c, p, bs, e, m);
        }
        if m & L_LEAD != 0 {
            self.sw_lead(s, b, p, sec, bs, m);
        }
        if m & L_BELL != 0 {
            self.sw_colour(b, c, p.part, bs, e, last);
        }
        if m & L_DRONE != 0 {
            self.sw_timpani(s, b, c, p.part, bs, e, last);
        }
        self.sw_drums(s, b, sec, m, p.part, bs, last);
    }

    /// Chords held through the bar (each change voice-led) on `ch`.
    fn sw_held(&mut self, b: &mut Bar, ch: usize, c: [Harm; 2], lo: i32, hi: i32, vel: f32, bs: u32) {
        let two = c[1] != c[0];
        let half = bs as f32 / 2.0;
        for (h, &x) in c.iter().enumerate().take(if two { 2 } else { 1 }) {
            let v = self.voicing(x, lo, hi, &self.pad_v, false);
            for (k, &n) in v.iter().enumerate() {
                b.note(
                    h as f32 * half,
                    0.0,
                    ch,
                    n,
                    vel * (1.0 - 0.04 * k as f32),
                    if two { half } else { bs as f32 } + 0.3,
                    false,
                );
            }
            self.pad_v = v;
        }
    }

    /// The basses: the march's roots on the beat (the fifth between), the
    /// battle's driving eighths, the dark march's roots held, an octave
    /// under them on the strong beats.
    fn sw_bass(&mut self, s: &Synth, b: &mut Bar, c: [Harm; 2], part: u8, bs: u32, e: f32) {
        let beat = self.sw.beat_steps;
        let held = s.ch[CH_BASS].patch.a > 0.08 || matches!(part, SP_LOVE | SP_MYSTERY | SP_FANFARE);
        let root = |x: Harm| 36 + (self.root + x.bass() - 36).rem_euclid(12);
        let fifth = |x: Harm| {
            let r0 = root(x);
            let f = r0 + (x.root + 7 - x.bass()).rem_euclid(12);
            if f > 50 { f - 12 } else { f }
        };
        let half = bs / 2;
        let v = 0.7 + 0.25 * e;
        if held || self.sw.style == SWS_DARK {
            for h in 0..2u32 {
                let x = c[h as usize];
                if h == 1 && x == c[0] {
                    continue;
                }
                let len = if h == 0 && c[1] == c[0] { bs } else { half } as f32;
                b.note((h * half) as f32, 0.0, CH_BASS, root(x), v, len, false);
            }
            return;
        }
        if self.sw.style == SWS_BATTLE {
            // eighths, the bar's accents (3+3+2, or the compound beats) harder
            let acc: &[u32] = if beat == 6 { &[0, 6, 12] } else { &[0, 6, 12] };
            for st in (0..bs).step_by(2) {
                let x = c[(st >= half) as usize];
                let a = acc.contains(&st);
                b.note(
                    st as f32,
                    self.rng.range(0.0, self.human),
                    CH_BASS,
                    root(x),
                    v * if a { 1.0 } else { 0.72 },
                    1.6,
                    false,
                );
            }
            return;
        }
        // the march: root, fifth, root, fifth (quarters, the root's long)
        for (k, st) in (0..bs).step_by(beat as usize).enumerate() {
            let x = c[(st >= half) as usize];
            let p = if k % 2 == 0 { root(x) } else { fifth(x) };
            b.note(st as f32, 0.0, CH_BASS, p, v * if k % 2 == 0 { 1.0 } else { 0.85 }, beat as f32 * 0.9, false);
        }
    }

    /// The strings' ostinato: the march's triplets (up the chord, or a
    /// chord repeated), the battle's 16ths on the root (a neighbour now and
    /// then; in 6/8, the beat's three eighths), the dark march's eighths
    /// with the flat second.
    fn sw_ostinato(&mut self, b: &mut Bar, c: [Harm; 2], bs: u32, e: f32, part: u8) {
        let beat = self.sw.beat_steps;
        let half = bs as f32 / 2.0;
        let vel = 0.55 + 0.3 * e;
        let h = self.human;
        match self.sw.style {
            SWS_MARCH => {
                let lo = self.tonic_near(62) - 5;
                for bt in (0..bs).step_by(beat as usize) {
                    let x = c[(bt as f32 >= half) as usize];
                    let t = self.arp_tones(x, lo);
                    for j in 0..3 {
                        let st = bt as f32 + j as f32 * T3;
                        let ps: Vec<i32> = match self.sw.ost {
                            0 => vec![t[j.min(t.len() - 1)]],
                            1 => vec![t[0], t[2.min(t.len() - 1)]],
                            _ => vec![t[[0, 1, 2][j].min(t.len() - 1)] + if j == 2 { 12 } else { 0 }],
                        };
                        let v = vel * if j == 0 { 1.0 } else { 0.78 };
                        for q in ps {
                            b.note(st, self.rng.range(0.0, h), CH_ARP, q, v, T3 * 0.7, false);
                        }
                    }
                }
            }
            SWS_DARK => {
                let lo = 45;
                for st in (0..bs).step_by(2) {
                    let x = c[(st as f32 >= half) as usize];
                    let r0 = lo + (self.root + x.bass() - lo).rem_euclid(12);
                    let n = if st % 8 == 6 { r0 + 1 } else { r0 };
                    b.note(
                        st as f32,
                        self.rng.range(0.0, h),
                        CH_ARP,
                        n,
                        vel * if st % 8 == 0 { 1.0 } else { 0.75 },
                        1.5,
                        false,
                    );
                }
            }
            _ => {
                let lo = if part == SP_SEQ { 48 } else { 43 };
                let pat: &[i32] = match self.sw.ost {
                    0 => &[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0],
                    1 => &[0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 5, 3, 0, 0],
                    _ => &[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 2, 1, 0, 1],
                };
                let step: u32 = if beat == 6 { 2 } else { 1 };
                for st in (0..bs).step_by(step as usize) {
                    let x = c[(st as f32 >= half) as usize];
                    let r0 = lo + (self.root + x.bass() - lo).rem_euclid(12);
                    let k = pat[(st / step) as usize % pat.len()];
                    let n = if self.sw.ost == 1 { r0 + k } else { self.root + sw_step(self.mode, r0 - self.root, k) };
                    let acc = if beat == 6 { st % 6 == 0 } else { matches!(st, 0 | 6 | 12) };
                    b.note(
                        st as f32,
                        self.rng.range(0.0, h * 0.5),
                        CH_ARP,
                        n,
                        vel * if acc { 1.0 } else { 0.7 },
                        step as f32 * 0.75,
                        false,
                    );
                }
            }
        }
    }

    /// The brass: the fanfare's lower voices (fourths or fifths under the
    /// call); the theme harmonised under its tune (the second time, and
    /// in the last statement), else held chords; the battle's stabs; the
    /// second theme's horn answering (the counter-line).
    fn sw_brass(&mut self, s: &Synth, b: &mut Bar, c: [Harm; 2], p: SwPart, bs: u32, e: f32, m: u32) {
        let i = self.sec_bar;
        let vel = 0.55 + 0.35 * e;
        let tonic = self.sw.tonic;
        let slow = s.ch[CH_KEYS].patch.a > 0.12;
        let (lo, hi) = if self.sw.style == SWS_DARK { (43, 64) } else { (50, 70) };
        match p.part {
            SP_FANFARE => {
                let fb = self.sw.bar_steps as f32;
                let q = self.sw.quartal;
                let bar = (i % 2) as f32;
                let up = if i >= 2 { 5 } else { 0 };
                let notes: Vec<Tn> = self.sw.fanfare.iter().filter(|n| (n.st / fb).floor() == bar).copied().collect();
                for n in notes {
                    let st = n.st - bar * fb;
                    for d in [q, 12] {
                        let pt = tonic + n.p + up - d - 12;
                        b.note(st, 0.0, CH_KEYS, pt.max(lo - 5), n.v * vel, n.len * 0.92, false);
                    }
                }
                if i == 3 {
                    self.sw_held(b, CH_KEYS, c, lo, hi, vel * 0.8, bs);
                }
            }
            SP_LOVE if m & L_CTR != 0 => {
                let center = self.tonic_near(55);
                self.counter(b, CH_KEYS, center, 0.6 + 0.2 * e, false);
            }
            SP_THEME | SP_SEQ if !slow && (p.rep > 0 || self.sw.style == SWS_BATTLE) => {
                if self.sw.style == SWS_BATTLE && (p.part == SP_SEQ || i % 4 < 2) {
                    // stabs on the accents
                    let half = bs as f32 / 2.0;
                    let at: &[f32] =
                        if self.sw.beat_steps == 6 { &[0.0, 6.0, 10.0] } else { &[0.0, 3.0, 6.0, 10.0, 12.0] };
                    for &st in at.iter().filter(|&&st| st < bs as f32) {
                        let x = c[(st >= half) as usize];
                        let v = self.voicing(x, lo, hi, &self.sw.brass_v, false);
                        for &n in &v {
                            b.note(st, 0.0, CH_KEYS, n, vel * if st == 0.0 { 1.0 } else { 0.85 }, 1.2, false);
                        }
                        self.sw.brass_v = v;
                    }
                    return;
                }
                // the tune harmonised: each of its notes of an eighth or more
                // with the chord's tones under it
                let th = &self.sw.themes[(p.part == SP_SEQ) as usize];
                let tb = (th.len() > 0) as u32
                    * (th.last().map_or(1.0, |n| n.st / self.sw.bar_steps as f32).floor() as u32 + 1);
                let fb = self.sw.bar_steps as f32;
                let bar = (i % tb.max(1)) as f32;
                let notes: Vec<Tn> =
                    th.iter().filter(|n| (n.st / fb).floor() == bar && n.len >= 1.9).copied().collect();
                for n in notes {
                    let st = n.st - bar * fb;
                    if st >= bs as f32 {
                        continue;
                    }
                    let x = c[(st * 2.0 >= bs as f32) as usize];
                    let top = tonic + n.p - 12;
                    let under: Vec<i32> = in_range(&self.pcs(x), top - 9, top - 1).into_iter().rev().take(2).collect();
                    for (k, &q) in under.iter().enumerate() {
                        b.note(
                            st,
                            0.0,
                            CH_KEYS,
                            q.max(lo - 7),
                            n.v * vel * (0.9 - 0.08 * k as f32),
                            n.len * 0.9,
                            false,
                        );
                    }
                }
            }
            _ => self.sw_held(b, CH_KEYS, c, lo, hi, vel * 0.7, bs),
        }
    }

    /// The tune: the fanfare's call (its top two voices), the theme (or the
    /// second), the mystery's fragments of the second theme, softly; the
    /// build's call on its last two bars.
    fn sw_lead(&mut self, s: &Synth, b: &mut Bar, p: SwPart, sec: Sec, bs: u32, m: u32) {
        let i = self.sec_bar;
        let fb = self.sw.bar_steps as f32;
        let tonic = self.sw.tonic;
        let vel = 0.65 + 0.3 * sec.energy;
        let (notes, oct, legato): (Vec<Tn>, i32, f32) = match p.part {
            SP_FANFARE | SP_BUILD => {
                if p.part == SP_BUILD && i + 2 < sec.bars {
                    return;
                }
                let bar = (i % 2) as f32;
                let up = if p.part == SP_FANFARE && i >= 2 { 5 } else { 0 };
                let v = self
                    .sw
                    .fanfare
                    .iter()
                    .filter(|n| (n.st / fb).floor() == bar)
                    .map(|n| Tn { st: n.st - bar * fb, p: n.p + up, ..*n })
                    .collect();
                (v, 0, 0.9)
            }
            _ => {
                let th = &self.sw.themes[matches!(p.part, SP_LOVE | SP_MYSTERY | SP_SEQ) as usize];
                let tb = th.last().map_or(1.0, |n| (n.st / fb).floor() + 1.0) as u32;
                let bar = (i % tb.max(1)) as f32;
                // the mystery: the second theme's first and fifth bars, alone
                if p.part == SP_MYSTERY && !(i % 4 == 0 || i % 4 == 1) {
                    return;
                }
                let v = th
                    .iter()
                    .filter(|n| (n.st / fb).floor() == bar)
                    .map(|n| Tn { st: n.st - bar * fb, ..*n })
                    .collect();
                let oct = if p.part == SP_THEME && p.rep >= 2 && self.sw.style == SWS_MARCH { 0 } else { 0 };
                (v, oct, if p.part == SP_LOVE || p.part == SP_MYSTERY { 1.0 } else { 0.88 })
            }
        };
        let soft = if p.part == SP_MYSTERY { 0.6 } else { 1.0 };
        let dark = self.sw.style == SWS_DARK;
        for n in notes {
            if n.st >= bs as f32 {
                continue;
            }
            let len = (n.len.min(bs as f32 - n.st + 0.5)) * if n.len <= 2.0 { 0.75 } else { legato };
            let j = self.rng.range(0.0, self.human);
            let pt = tonic + n.p + oct;
            b.note(n.st, j, CH_LEAD, pt, n.v * vel * soft, len, false);
            if p.part == SP_FANFARE {
                // the call in fourths (or fifths)
                b.note(n.st, j, CH_LEAD, pt - self.sw.quartal, n.v * vel * 0.85, len, false);
            } else if p.part == SP_THEME && p.rep >= 2 && !dark {
                // the last statement: the tune in octaves
                b.note(n.st, j, CH_LEAD, pt - 12, n.v * vel * 0.8, len, false);
            }
        }
        let _ = (s, m);
    }

    /// The harp, the celesta, the glockenspiel: the chord's arpeggio up in
    /// eighths (the second theme's, the mystery's), a glissando into the
    /// next section's downbeat, the theme's downbeats touched.
    fn sw_colour(&mut self, b: &mut Bar, c: [Harm; 2], part: u8, bs: u32, e: f32, last: bool) {
        let half = bs as f32 / 2.0;
        let vel = 0.45 + 0.25 * e;
        if matches!(part, SP_LOVE | SP_MYSTERY) {
            for (h, &x) in c.iter().enumerate() {
                let t = self.arp_tones(x, self.tonic_near(60) - 5);
                let n = (half / 2.0) as usize;
                for k in 0..n {
                    let st = h as f32 * half + 2.0 * k as f32;
                    b.note(
                        st,
                        self.rng.range(0.0, self.human),
                        CH_BELL,
                        t[k % t.len()] + 12 * (k / t.len()) as i32,
                        vel * if k == 0 { 1.0 } else { 0.8 },
                        6.0,
                        false,
                    );
                }
            }
        } else if !last {
            let x = c[0];
            let t = self.arp_tones(x, self.tonic_near(72) - 5);
            if self.sec_bar % 2 == 0 {
                b.note(0.0, 0.0, CH_BELL, t[t.len() / 2], vel * 0.7, 8.0, false);
            }
        }
        if last {
            // a glissando: the chord-scale up two octaves in the bar's last beat
            let sc = chord_scale(self.mode, c[1]);
            let lo = self.tonic_near(55);
            let notes = in_range(&sc, lo - self.root, lo - self.root + 24);
            let beat = self.sw.beat_steps as f32;
            let n = notes.len().max(1) as f32;
            for (k, &q) in notes.iter().enumerate() {
                let st = bs as f32 - beat + k as f32 * beat / n;
                b.note(st, 0.0, CH_BELL, q + self.root, vel * (0.6 + 0.4 * k as f32 / n), 3.0, false);
            }
        }
    }

    /// The timpani, tuned to the tonic and the dominant (under it): the
    /// march's tonic and dominant on the strong beats, the battle's
    /// accents, the dark march's figure; a roll up to each section's end
    /// (a soft one under the mystery). A slow patch (a mix's, a choir)
    /// holds the two instead.
    fn sw_timpani(&mut self, s: &Synth, b: &mut Bar, c: [Harm; 2], part: u8, bs: u32, e: f32, last: bool) {
        let t = 38 + (self.root - 38).rem_euclid(12);
        let d = if t - 5 >= 36 { t - 5 } else { t + 7 };
        let drum = |x: Harm| if x.root == 7 || x.bass() == 7 { d } else { t };
        let vel = 0.55 + 0.35 * e;
        let h = self.human;
        if s.ch[CH_DRONE].patch.a > 0.05 {
            if self.sec_bar % 2 == 0 {
                for q in [t + 12, d + 12] {
                    b.note(0.0, 0.0, CH_DRONE, q, 0.4 + 0.2 * e, 2.0 * bs as f32, false);
                }
            }
            return;
        }
        let beat = self.sw.beat_steps;
        let half = bs as f32 / 2.0;
        let roll = |b: &mut Bar, from: f32, to: f32, p: i32, v0: f32, v1: f32| {
            let n = ((to - from) / 0.5).max(1.0) as usize;
            for k in 0..n {
                let x = k as f32 / n as f32;
                b.note(from + 0.5 * k as f32, 0.0, CH_DRONE, p, v0 + (v1 - v0) * x * x, 0.4, false);
            }
        };
        match part {
            SP_MYSTERY => {
                if self.sec_bar % 4 == 3 {
                    roll(b, 0.0, bs as f32, t, 0.15, 0.4);
                }
                return;
            }
            SP_LOVE => {
                if last {
                    roll(b, half, bs as f32, drum(c[1]), 0.25, 0.6);
                }
                return;
            }
            SP_FANFARE => {
                if self.sec_bar == 0 {
                    b.note(0.0, 0.0, CH_DRONE, t, vel, 4.0, false);
                }
                if self.sec_bar == 3 || self.sec_bar == 1 {
                    roll(b, 0.0, bs as f32, if self.sec_bar == 1 { d } else { t }, 0.3, vel);
                }
                return;
            }
            SP_BUILD => {
                roll(b, 0.0, bs as f32, drum(c[0]), 0.3 + 0.1 * self.sec_bar as f32, vel);
                return;
            }
            _ => {}
        }
        let end = if last { bs as f32 - beat as f32 } else { bs as f32 };
        match self.sw.style {
            SWS_MARCH => {
                let every = if e >= 0.9 { 2 } else { 4 } * beat as usize;
                for st in (0..bs).step_by(every).map(|x| x as f32).filter(|&st| st < end) {
                    let x = c[(st >= half) as usize];
                    b.note(
                        st,
                        self.rng.range(0.0, h),
                        CH_DRONE,
                        drum(x),
                        vel * if st == 0.0 { 1.0 } else { 0.8 },
                        3.0,
                        false,
                    );
                }
                if self.sec_bar % 4 == 3 && !last {
                    // a triplet into the next phrase
                    for k in 0..3 {
                        b.note(
                            bs as f32 - beat as f32 + k as f32 * T3,
                            0.0,
                            CH_DRONE,
                            if k == 2 { d } else { t },
                            vel * 0.75,
                            1.0,
                            false,
                        );
                    }
                }
            }
            SWS_DARK => {
                for (st, k) in [(0.0, 0), (10.0, 1), (12.0, 0), (14.0, 1)] {
                    if st < end && st < bs as f32 {
                        b.note(
                            st,
                            self.rng.range(0.0, h),
                            CH_DRONE,
                            if k == 0 { t } else { d },
                            vel * if st == 0.0 { 1.0 } else { 0.75 },
                            2.0,
                            false,
                        );
                    }
                }
            }
            _ => {
                let at: &[f32] = if beat == 6 { &[0.0, 6.0, 9.0, 10.0] } else { &[0.0, 6.0, 12.0, 14.0] };
                for &st in at.iter().filter(|&&st| st < end) {
                    let x = c[(st >= half) as usize];
                    b.note(
                        st,
                        self.rng.range(0.0, h),
                        CH_DRONE,
                        drum(x),
                        vel * if st == 0.0 { 1.0 } else { 0.8 },
                        2.0,
                        false,
                    );
                }
            }
        }
        if last {
            roll(b, bs as f32 - beat as f32, bs as f32, t, 0.35, vel);
        }
    }

    /// The percussion of a Star Wars bar: the march's snare (and its
    /// triplet drag into each phrase), the bass drum on the strong beats,
    /// the battle's driving snare; the triangle under the second theme;
    /// a clash on each theme's downbeat (the gong's in the dark march), a
    /// swell (or the riser) into the next section's.
    fn sw_drums(&mut self, s: &Synth, b: &mut Bar, sec: Sec, m: u32, part: u8, bs: u32, last: bool) {
        let i = self.sec_bar;
        let beat = self.sw.beat_steps;
        let e = sec.energy;
        let h = self.human;
        let style = self.sw.style;
        let r = &mut self.rng;
        if i == 0 && matches!(part, SP_THEME | SP_SEQ | SP_FANFARE) {
            if style == SWS_DARK {
                b.hit(0.0, 0.0, CH_CYM, CY_GONG, 0.7 + 0.2 * e);
            } else if part != SP_FANFARE || r.bit_chance() {
                b.hit(0.0, 0.0, CH_CYM, CY_CLASH, 0.75 + 0.2 * e);
            }
        } else if i % 4 == 0 && part == SP_THEME && sec.energy >= 0.9 {
            b.hit(0.0, 0.0, CH_CYM, CY_CLASH, 0.6);
        }
        // into the next section: a swell (2 s or 4 s, whichever fits) or the riser
        let next = self.sw.map.get(self.sec + 1).map(|x| x.part);
        if last && matches!(next, Some(SP_THEME | SP_SEQ | SP_FINALE)) {
            let bar_secs = b.secs(bs as f32);
            let recorded = s.ch[CH_CYM].drums.get(CY_SWELL2 as usize).is_some_and(|d| d.samp != 0);
            if recorded {
                let (kind, secs) = if bar_secs >= 4.1 { (CY_SWELL4, 4.0) } else { (CY_SWELL2, 2.0) };
                let st = (bs as f32 - secs / b.secs(1.0)).max(0.0);
                b.hit(st, 0.0, CH_CYM, kind, 0.85);
            } else {
                let at = (bs as f32 - 2.0 / b.secs(1.0)).max(0.0);
                let t = b.at(at, 0.0);
                b.ev.push((t, Ev::Riser { secs: b.secs(bs as f32 - at), vel: 0.25 }));
            }
        }
        if m & L_HAT != 0 && part == SP_LOVE && i % 2 == 0 {
            b.hit(0.0, r.range(0.0, h), CH_HAT, 0, 0.45);
        }
        if m & L_KICK != 0 && !matches!(part, SP_FANFARE | SP_LOVE | SP_MYSTERY) {
            let at: &[u32] = match style {
                SWS_BATTLE if beat == 6 => &[0, 6],
                SWS_BATTLE => &[0, 6, 12],
                _ => &[0, 8],
            };
            for &st in at.iter().filter(|&&st| st < bs) {
                b.hit(st as f32, r.range(0.0, h), CH_KICK, 0, (0.55 + 0.3 * e) * if st == 0 { 1.0 } else { 0.8 });
            }
        } else if m & L_KICK != 0 && part == SP_FANFARE && i == 0 {
            b.hit(0.0, 0.0, CH_KICK, 0, 0.9);
        }
        if m & L_SNARE != 0 && !matches!(part, SP_LOVE | SP_MYSTERY | SP_FANFARE) {
            let v = 0.35 + 0.3 * e;
            match style {
                SWS_BATTLE => {
                    for st in (0..bs).step_by(if beat == 6 { 2 } else { 1 }) {
                        let acc = if beat == 6 { st % 6 == 0 } else { matches!(st, 4 | 12) };
                        if acc || r.chance(if part == SP_BUILD { 0.9 } else { 0.45 }) {
                            b.hit(
                                st as f32,
                                r.range(0.0, h),
                                CH_SNARE,
                                if acc { 0 } else { 1 },
                                v * if acc { 1.0 } else { 0.5 },
                            );
                        }
                    }
                }
                _ if part == SP_BUILD => {
                    // a roll, growing
                    let x0 = i as f32 / sec.bars as f32;
                    for k in 0..bs * 2 {
                        let x = x0 + k as f32 / (2 * bs * sec.bars) as f32;
                        b.hit(k as f32 * 0.5, 0.0, CH_SNARE, 1, v * (0.35 + 0.6 * x));
                    }
                }
                _ => {
                    // the march: the beats, the "and"s now and then, a drag
                    // (a triplet) on the last beat of a phrase
                    for st in (0..bs).step_by(2) {
                        let on = st % beat == 0;
                        if on || r.chance(0.35) {
                            b.hit(
                                st as f32,
                                r.range(0.0, h),
                                CH_SNARE,
                                (!on) as u8,
                                v * if st == 0 {
                                    1.0
                                } else if on {
                                    0.75
                                } else {
                                    0.45
                                },
                            );
                        }
                    }
                    if i % 4 == 3 {
                        for k in 0..3 {
                            b.hit(
                                bs as f32 - beat as f32 + k as f32 * T3,
                                0.0,
                                CH_SNARE,
                                1,
                                v * (0.5 + 0.2 * k as f32),
                            );
                        }
                    }
                }
            }
        }
    }

    /// The finale: the heroic cadence (the sixth and seventh flattened,
    /// or the dark march's flat second and dominant, the battle's), the
    /// brass's call over a timpani roll and the strings' tremolo, a swell;
    /// then everyone on the tonic, held, the timpani's and the bass drum's
    /// stroke, the clash (and the gong).
    fn sw_finale(&mut self, s: &Synth, b: &mut Bar, sec: Sec, m: u32) {
        let i = self.sec_bar;
        let bs = self.sw.bs_at(i);
        let minor = MODES[self.mode][2] == 3;
        let style = self.sw.style;
        let tonic = self.sw.tonic;
        let q = if minor { Q_MIN } else { Q_MAJ };
        let home =
            Harm { root: 0, q: if style == SWS_BATTLE && minor && self.rng.chance(0.3) { Q_MAJ } else { q }, inv: 0 };
        let steps: [Harm; 2] = if style == SWS_DARK {
            [Harm { root: 1, q: Q_MAJ, inv: 0 }, Harm { root: 7, q: Q_DOM7, inv: 0 }]
        } else {
            [Harm { root: 8, q: Q_MAJ, inv: 0 }, Harm { root: 10, q: Q_MAJ, inv: 0 }]
        };
        if i + 1 < sec.bars {
            let c = [steps[i.min(1) as usize]; 2];
            let fake = SwPart { part: SP_BUILD, dress: 0, rep: 0 };
            if m & L_PAD != 0 {
                self.sw_held(b, CH_PAD, c, 55, 79, 0.75, bs);
            }
            if m & L_BASS != 0 {
                self.sw_bass(s, b, c, SP_FANFARE, bs, 1.0);
            }
            if m & L_KEYS != 0 {
                self.sw_held(b, CH_KEYS, c, 50, 70, 0.8, bs);
            }
            if m & L_ARP != 0 && s.ch[CH_ARP].patch.a <= 0.08 {
                self.sw_ostinato(b, c, bs, 1.0, SP_THEME);
            }
            if m & L_LEAD != 0 {
                // the call: the chord's fifth, root, third in triplets, held
                let tones = in_range(&self.pcs(c[0]), tonic - 3, tonic + 12);
                let k = tones.len();
                let fig = [tones[k.saturating_sub(3)], tones[k.saturating_sub(2)], tones[k - 1]];
                let beat = self.sw.beat_steps as f32;
                for (j, &pt) in fig.iter().enumerate() {
                    b.note(j as f32 * T3, 0.0, CH_LEAD, pt, 0.85, T3 * 0.8, false);
                }
                b.note(beat, 0.0, CH_LEAD, fig[2], 1.0, bs as f32 - beat, false);
                b.note(beat, 0.0, CH_LEAD, fig[2] - self.sw.quartal, 0.85, bs as f32 - beat, false);
            }
            if m & L_DRONE != 0 {
                self.sw_timpani(s, b, c, SP_BUILD, bs, 1.0, false);
            }
            self.sw_drums(
                s,
                b,
                Sec { energy: 1.0, ..sec },
                m,
                if i == 0 { SP_BUILD } else { SP_THEME },
                bs,
                i + 2 == sec.bars,
            );
            let _ = fake;
            return;
        }
        // the last chord
        let len = bs as f32 - 1.0;
        let v = self.voicing(home, 52, 79, &self.pad_v, false);
        if m & L_PAD != 0 {
            for &n in &v {
                b.note(0.0, 0.0, CH_PAD, n, 0.85, len, false);
            }
        }
        if m & L_KEYS != 0 {
            for n in self.voicing(home, 48, 70, &[], false) {
                b.note(0.0, 0.0, CH_KEYS, n, 0.95, len, false);
            }
        }
        if m & L_LEAD != 0 {
            b.note(0.0, 0.0, CH_LEAD, tonic + 12, 1.0, len, false);
            b.note(0.0, 0.0, CH_LEAD, tonic + 7, 0.85, len, false);
        }
        if m & L_ARP != 0 {
            for n in in_range(&self.pcs(home), 55, 72) {
                b.note(0.0, 0.0, CH_ARP, n, 0.8, if s.ch[CH_ARP].patch.a > 0.08 { len } else { 1.5 }, false);
            }
        }
        if m & L_BASS != 0 {
            let r0 = 36 + (self.root - 36).rem_euclid(12);
            b.note(0.0, 0.0, CH_BASS, r0, 1.0, len, false);
            b.note(0.0, 0.0, CH_BASS, r0 + 12, 0.8, len, false);
        }
        if m & L_BELL != 0 {
            for (k, n) in in_range(&self.pcs(home), 48, 84).into_iter().enumerate() {
                b.note(0.0, 0.02 * k as f32, CH_BELL, n, 0.6, len, false);
            }
        }
        if m & L_DRONE != 0 {
            let t = 38 + (self.root - 38).rem_euclid(12);
            if s.ch[CH_DRONE].patch.a > 0.05 {
                b.note(0.0, 0.0, CH_DRONE, t + 12, 0.7, len, false);
            } else {
                b.note(0.0, 0.0, CH_DRONE, t, 1.0, 4.0, false);
                // a roll under the held chord, dying, then the last stroke
                let beat = self.sw.beat_steps as f32;
                let n = ((len - 2.0 * beat) / 0.5) as usize;
                for k in 0..n {
                    b.note(beat + 0.5 * k as f32, 0.0, CH_DRONE, t, 0.55 - 0.25 * k as f32 / n as f32, 0.4, false);
                }
                b.note(bs as f32 - beat, 0.0, CH_DRONE, t, 0.95, 2.0, false);
            }
        }
        b.hit(0.0, 0.0, CH_CYM, if style == SWS_DARK { CY_GONG } else { CY_CLASH }, 1.0);
        if style != SWS_DARK && self.rng.bit_chance() {
            b.hit(0.0, 0.0, CH_CYM, CY_GONG, 0.6);
        }
        b.hit(0.0, 0.0, CH_KICK, 0, 1.0);
        b.hit(bs as f32 - self.sw.beat_steps as f32, 0.0, CH_KICK, 0, 0.9);
        b.hit(0.0, 0.0, CH_SNARE, 0, 0.7);
    }

    /// Star Wars's beat in any meter, for a mix: the bass drum on the
    /// downbeat (the middle of the bar too when it is strong), the snare's
    /// march (the beats, an "and" now and then, a drag closing a phrase),
    /// the battle's driving 16ths in a strong section's B groups; the
    /// triangle in a quiet one; a clash on a section's downbeat, a swell
    /// into a main section's.
    pub(super) fn starwars_beat(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let (bs, beat) = (self.bar_steps(), self.beat_steps());
        let i = self.sec_bar;
        let e = sec.energy;
        let full = sec.kind == MAIN;
        let vb = self.var_b();
        let h = self.human;
        let fill = if full || sec.kind == BUILD { self.fill(sec) } else { 0 };
        let r = &mut self.rng;
        let quiet = matches!(sec.kind, INTRO | BREAK) && m & (L_KICK | L_SNARE) == 0;
        if i == 0 && sec.kind != BREAK {
            b.hit(0.0, 0.0, CH_CYM, CY_CLASH, 0.6 + 0.3 * e);
        }
        if i + 1 == sec.bars && full && s.ch[CH_CYM].drums.get(CY_SWELL2 as usize).is_some_and(|d| d.samp != 0) {
            let st = (bs as f32 - 2.0 / b.secs(1.0)).max(0.0);
            b.hit(st, 0.0, CH_CYM, CY_SWELL2, 0.7);
        }
        if quiet {
            if i % 2 == 0 {
                b.hit(0.0, r.range(0.0, h), CH_HAT, 0, 0.35);
            }
            return;
        }
        b.hit(0.0, r.range(0.0, h), CH_KICK, 0, 0.6 + 0.3 * e);
        if (full && e >= 0.7) || bs >= 16 && m & L_KICK != 0 {
            let mid = (bs / beat / 2) * beat;
            if mid > 0 && mid < bs {
                b.hit(mid as f32, r.range(0.0, h), CH_KICK, 0, 0.5 + 0.25 * e);
            }
        }
        let v = 0.3 + 0.3 * e;
        let drive = full && e >= 0.85 && vb;
        for st in (0..bs).step_by(if drive { 1 } else { 2 }) {
            let on = st % beat == 0;
            if fill == 2 && st + beat >= bs {
                continue;
            }
            if on || r.chance(if drive { 0.5 } else { 0.3 }) {
                b.hit(
                    st as f32,
                    r.range(0.0, h),
                    CH_SNARE,
                    (!on) as u8,
                    v * if st == 0 {
                        1.0
                    } else if on {
                        0.75
                    } else {
                        0.45
                    },
                );
            }
        }
        if fill > 0 {
            let from = bs - beat;
            let n = if fill == 2 { 3 } else { 2 };
            let gap = beat as f32 / n as f32;
            for k in 0..n {
                b.hit(from as f32 + k as f32 * gap, 0.0, CH_SNARE, 1, v * (0.55 + 0.15 * k as f32));
            }
        }
        if !full && sec.kind != BUILD && i % 2 == 0 {
            b.hit(0.0, r.range(0.0, h), CH_HAT, 0, 0.3);
        }
    }
}

#[cfg(test)]
mod pools {
    use super::*;

    /// Every chord of the Star Wars and Western pools parses, and each plan
    /// is eight bars (sixteen chords).
    #[test]
    fn cinematic_pools_parse() {
        for pool in SW_POOLS.iter().chain(WN_POOLS) {
            for plan in pool.iter() {
                for tok in plan.split_whitespace().flat_map(|t| t.split('+')) {
                    assert!(parse_harm(tok).is_some(), "{tok} in {plan}");
                }
                assert_eq!(halves(plan).len(), 16, "{plan}");
            }
        }
    }
}

