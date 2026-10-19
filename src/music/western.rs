//! WESTERN: the spaghetti western's score in Ennio Morricone's manner, its
//! tunes fleece's own (none of the films'): a lonely whistled melody, a
//! twangy electric guitar (clean, with tremolo and a spring's reverb, on its
//! low strings), a Spanish guitar strumming, a trumpet's solo in the
//! mariachi way, a choir's wordless chants and grunts, the jaw harp,
//! harmonica and ocarina, a galloping ostinato on the bass and the drums,
//! church bells, the anvil, the whip, timpani; long builds to a stand-off's
//! climax, lonely open fifths, minor keys and Phrygian and Andalusian
//! cadences.

use super::*;

/// The styles: the duel (a slow 3/4 building to its climax), the ride (a
/// gallop in 2/4), the lament (a slow 3/4 ballad, quiet throughout).
pub(super) const WNS_DUEL: u8 = 0;
pub(super) const WNS_RIDE: u8 = 1;
pub(super) const WNS_LAMENT: u8 = 2;

/// A section's part: the lonely opening, the theme, the theme again with
/// more around it, the build (strings' tremolo, timpani, the choir), the
/// climax (the trumpet's solo, everyone), the ride's chorus, the end.
const WP_ALONE: u8 = 0;
const WP_THEME: u8 = 1;
const WP_MORE: u8 = 2;
const WP_BUILD: u8 = 3;
const WP_CLIMAX: u8 = 4;
const WP_END: u8 = 5;

/// The percussion's kinds on CH_PERC: the anvil, the whip, castanets, the
/// bell tree; on CH_CYM: the vibraslap, the gong.
const WK_ANVIL: u8 = 0;
const WK_WHIP: u8 = 1;
const WK_CAST: u8 = 2;
const WK_TREE: u8 = 3;
const WC_SLAP: u8 = 0;
const WC_GONG: u8 = 1;

#[derive(Clone, Copy, Default)]
pub(super) struct WnPart {
    part: u8,
    dress: usize,
    /// the tempo against the piece's (the duel quickening as it builds)
    pace: f32,
}

/// A Western piece's state.
#[derive(Default)]
pub(super) struct Western {
    pub(super) bar_steps: u32,
    pub(super) beat_steps: u32,
    pub(super) meter: &'static str,
    style: u8,
    bpm0: f64,
    map: Vec<WnPart>,
    bars: Vec<u32>,
    progs: Vec<Vec<Harm>>,
    /// the theme (eight bars), the twangy guitar's riff (two bars)
    theme: Vec<Tn>,
    riff: Vec<(f32, i32)>,
    tonic: i32,
}

impl Western {
    /// The 16ths in bar `k` of the section.
    pub(super) fn bs_at(&self, k: u32) -> u32 {
        self.bars.get(k as usize).copied().unwrap_or(self.bar_steps).max(1)
    }

    /// Which of the instrument pairs section `k` wears.
    pub(super) fn dress_at(&self, k: usize) -> usize {
        self.map.get(k).map_or(0, |x| x.dress)
    }
}

// the progressions: Andalusian, Phrygian, the open minor; the ballad's
const WN_MIN_A: &[&str] = &[
    "i bVII bVI V i bVII bVI V",
    "i i iv iv bVII bVII bIII V",
    "i bII i bII i iv bII V",
    "i iv bVI V i iv V i",
    "i v bVI bIII iv i bII V",
];
const WN_MIN_B: &[&str] = &["bVI bVI bVII bVII i i V V", "iv iv i i bVI bII V V", "bIII bIII bVII bVII iv iv V V"];
const WN_MAJ_A: &[&str] = &["I bVII IV I I bVII IV V", "I IV I V I IV V I", "I I bVII bVII IV IV I V"];
const WN_MAJ_B: &[&str] = &["IV IV I I bVII bVII V V", "vi vi IV IV I I V V"];
const WN_LAMENT: &[&str] = &["i iv i V i iv bVI V", "i bIII iv i bVI bIII iv V", "i i iv iv bVI bVI V V"];

#[cfg(test)]
pub(super) const WN_POOLS: &[&[&str]] = &[WN_MIN_A, WN_MIN_B, WN_MAJ_A, WN_MAJ_B, WN_LAMENT];

/// A bar's cells, as fractions of the bar (start, length) and moves (as
/// `sw_theme`'s): a call (a short pickup, a long note leapt to), held, a
/// fall in quarters, a turn, two halves leapt, the end.
fn wn_cell(kind: u8) -> (&'static [(f32, f32)], &'static [i32]) {
    match kind {
        0 => (&[(0.0, 0.25), (0.25, 0.75)], &[M_CT, M_LEAP]),
        1 => (&[(0.0, 1.0)], &[M_CT]),
        2 => (&[(0.0, 0.25), (0.25, 0.25), (0.5, 0.25), (0.75, 0.25)], &[-1, -1, -1, M_CT]),
        3 => (&[(0.0, 0.5), (0.5, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0), (5.0 / 6.0, 1.0 / 6.0)], &[M_CT, 1, -1, -1]),
        4 => (&[(0.0, 0.5), (0.5, 0.5)], &[M_CT, M_LEAP]),
        5 => (&[(0.0, 0.375), (0.375, 0.125), (0.5, 0.5)], &[M_CT, -1, -1]),
        _ => (&[(0.0, 1.0)], &[M_HOME]),
    }
}

/// An eight-bar theme: a call leaping up (an octave, a fifth, a sixth),
/// held long, falling by steps, turning; its first two bars again as the
/// fifth and sixth; between `lo` and `hi`.
fn wn_theme(r: &mut Rng, chords: &[Harm], mode: usize, bs: u32, lo: i32, hi: i32) -> Vec<Tn> {
    let fb = bs as f32;
    let chord_at = |bar: usize, st: f32| chords[(2 * bar + (st * 2.0 >= fb) as usize) % chords.len()];
    let kinds = [0u8, r.pick(&[1, 5]), r.pick(&[2, 3]), 1, 0, 1, r.pick(&[3, 4, 2]), 6];
    let mut v: Vec<Tn> = Vec::new();
    let mut cur = r.pick(&[0, -5, 0]);
    for (bar, &k) in kinds.iter().enumerate() {
        let t0 = bar as f32 * fb;
        if bar == 4 || bar == 5 {
            let src = bar - 4;
            let notes: Vec<Tn> = v.iter().filter(|n| (n.st / fb) as usize == src).copied().collect();
            for n in notes {
                v.push(Tn { st: n.st + 4.0 * fb, ..n });
                cur = n.p;
            }
            continue;
        }
        let (cells, moves) = wn_cell(k);
        for (&(a, l), &mv) in cells.iter().zip(moves) {
            let (st, len) = ((a * fb).round(), (l * fb).round().max(1.0));
            let h = chord_at(bar, st);
            let mut p = match mv {
                M_CT => sw_ct(h, cur, false),
                M_HOME => {
                    let t = (cur as f32 / 12.0).round() as i32 * 12;
                    if t > hi { t - 12 } else { t }
                }
                M_LEAP => {
                    // an octave, a fifth or a sixth (never a fourth up and back: the films' call)
                    let iv = [12, 7, 9][r.weighted(&[2.0, 2.0, 1.0])];
                    let q = sw_ct(h, cur + iv, true);
                    if q > hi { sw_ct(h, cur - 7, false) } else { q }
                }
                n => sw_step(mode, cur, n),
            };
            while p > hi {
                p -= 12;
            }
            while p < lo {
                p += 12;
            }
            v.push(Tn { st: t0 + st, len, p, v: if st == 0.0 { 0.95 } else { 0.8 } });
            cur = p;
        }
    }
    v
}

// -- the instruments (the synthesiser's patch first, then its recording)

/// The whistle: a sine, the breath's noise, a singer's vibrato, sliding
/// between notes. Synthesised whatever is installed: the SoundFont's
/// Whistle is one sample (at its top, C7), played two octaves down here,
/// slowed fourfold (its breath and its vibrato with it), and the CC0
/// ocarina's hollow, chiffy speech is a vessel flute's, not lips'.
fn wn_whistle(r: &mut Rng) -> Patch {
    Patch {
        noise: r.range(0.03, 0.045),
        cutoff: 5000.0,
        vib: r.range(0.16, 0.22),
        vib_rate: r.range(5.2, 6.0),
        glide: r.range(0.05, 0.08),
        a: 0.03,
        r: 0.12,
        gain: 0.34,
        ..whistle_synth(r)
    }
}

fn wn_twang_synth(r: &mut Rng) -> Patch {
    // a clean electric guitar's string, bright, through its amplifier's tremolo
    Patch {
        ks: r.range(1.6, 2.2),
        ks_tone: r.range(0.72, 0.82),
        cutoff: 4500.0,
        keytrack: 0.2,
        a: 0.001,
        d: 2.5,
        s: 1.0,
        r: 0.25,
        trem: r.range(0.3, 0.45),
        vib_rate: r.range(5.0, 6.5),
        drive: 0.15,
        gain: 0.5,
        ..PATCH
    }
}

/// The twangy guitar: the SoundFont's clean electric guitar, its tremolo kept.
fn wn_twang(r: &mut Rng) -> Patch {
    let p = wn_twang_synth(r);
    rec_keep(p, samples::gm("Clean Guitar"), 52.0, |q| Patch { trem: p.trem, vib_rate: p.vib_rate, ..q })
}

/// The jaw harp: a narrow pulse at the drone's pitch through a resonant
/// band the mouth sweeps (the filter's envelope), short. Synthesised: no
/// recording of one anywhere in the bank.
fn wn_jaw_harp(r: &mut Rng) -> Patch {
    Patch {
        wave: Wave::Square,
        pw: r.range(0.12, 0.18),
        cutoff: r.range(500.0, 700.0),
        res: 0.82,
        env_amt: r.range(2.2, 2.8),
        fdec: r.range(0.1, 0.16),
        keytrack: 0.0,
        a: 0.002,
        d: 0.4,
        s: 0.0,
        r: 0.08,
        drive: 0.25,
        gain: 0.42,
        ..PATCH
    }
}

/// The trumpet's solo: the CC0 trumpet with its vibrato (the deguello's).
fn wn_trumpet(r: &mut Rng) -> Patch {
    let p = Patch { vib: 0.2, vib_rate: 6.0, ..trumpet_synth(r) };
    rec(p, multi_or("trumpet", "vib", "Trumpet"), 72.0)
}

fn wn_bell_synth(r: &mut Rng) -> Patch {
    // a church bell: FM at an inharmonic ratio, ringing long
    Patch {
        wave: Wave::Sine,
        fm: r.range(1.5, 2.2),
        ratio: 2.76,
        fm_dec: 1.5,
        filter: false,
        a: 0.001,
        d: 4.0,
        s: 0.0,
        r: 2.0,
        gain: 0.3,
        ..PATCH
    }
}

/// The church bell (the SoundFont's), or the tubular bells.
fn wn_bell(r: &mut Rng) -> Patch {
    rec(wn_bell_synth(r), samples::gm("Church Bell"), 60.0)
}

fn wn_chimes(r: &mut Rng) -> Patch {
    rec(wn_bell_synth(r), samples::gm("Tubular Bells"), 67.0)
}

fn wn_electric_bass(r: &mut Rng) -> Patch {
    rec(bass_synth(r), samples::gm("Fingered Bass"), 40.0)
}

/// The choir chanting (the SoundFont's choir), its "ah" sung short for
/// the ride's grunts (a stand-in: nothing grunts in the bank).
fn wn_choir(r: &mut Rng) -> Patch {
    let p = Patch { vowel: 1, a: 0.08, ..pad_choir_synth(r) };
    rec(p, samples::gm("Choir Aahs"), 57.0)
}

fn wn_strings(r: &mut Rng) -> Patch {
    rec(pad_strings_synth(r), multi_or("violins", "sus", "Strings Slow"), 67.0)
}

/// The kits: the bass drum, the field drum (CC0), the tambourine (CC0),
/// the anvil, the whip, the castanets (the SoundFont's), the bell tree, the
/// vibraslap and the gong (CC0).
fn wn_kit_perc_synth(r: &mut Rng) -> Vec<DrumPatch> {
    let anvil = DrumPatch {
        tone: 0.8,
        f0: r.range(900.0, 1100.0),
        f1: 0.0,
        pdec: 1.0,
        tdec: 0.9,
        tone2: 0.6,
        ratio2: 2.71,
        noise: 0.2,
        nmode: 2,
        ncut: 4000.0,
        ndec: 0.03,
        gain: 0.45,
        ..DRUM
    };
    let whip = DrumPatch {
        noise: 1.0,
        nmode: 2,
        ncut: r.range(2500.0, 3500.0),
        nres: 0.3,
        ndec: 0.06,
        click: 0.6,
        gain: 0.8,
        ..DRUM
    };
    let cast = DrumPatch {
        tone: 0.4,
        f0: 2600.0,
        f1: 2600.0,
        pdec: 1.0,
        tdec: 0.03,
        noise: 0.6,
        nmode: 1,
        ncut: 3500.0,
        nres: 0.4,
        ndec: 0.02,
        gain: 0.45,
        ..DRUM
    };
    let tree =
        DrumPatch { noise: 0.8, nmode: 2, ncut: 7000.0, ndec: 1.2, natt: 0.25, metal: 0.8, mf: 3.0, gain: 0.3, ..DRUM };
    vec![DrumPatch { f1: anvil.f0, ..anvil }, whip, cast, tree]
}

fn wn_kit_perc(r: &mut Rng) -> Vec<DrumPatch> {
    let v = wn_kit_perc_synth(r);
    rec_kit(
        v,
        &[
            samples::multi("anvil", "hit"),
            samples::multi("whip", "crack"),
            samples::gm("Castanets"),
            samples::multi("bell_tree", "stroke"),
        ],
    )
}

fn wn_kit_cym_synth(r: &mut Rng) -> Vec<DrumPatch> {
    let slap =
        DrumPatch { noise: 0.9, nmode: 1, ncut: 3000.0, nres: 0.6, ndec: 0.9, metal: 0.3, mf: 1.2, gain: 0.35, ..DRUM };
    let gong = sw_kit_cymbals_synth(r)[CY_GONG as usize];
    vec![slap, gong]
}

fn wn_kit_cym(r: &mut Rng) -> Vec<DrumPatch> {
    rec_kit(wn_kit_cym_synth(r), &[samples::multi("vibraslap", "rattle"), samples::multi("gong", "ring")])
}

fn wn_kit_snare(r: &mut Rng) -> Vec<DrumPatch> {
    let id = multi_or("field_snare", "hit", "Brush");
    rec_kit(sw_kit_snare_synth(r), &[id, id])
}

fn wn_kit_tamb(r: &mut Rng) -> Vec<DrumPatch> {
    rec_kit_db(kit_tambourine_synth(r), &[tambourine_rec(1, -6.0), tambourine_rec(0, -6.0)])
}

const P_WN_LEAD: &[(&str, Make)] =
    &[("whistle", wn_whistle), ("harmonica", harmonica), ("ocarina", ocarina), ("trumpet", wn_trumpet)];
const P_WN_GUITAR: &[(&str, Make)] = &[("twangy guitar", wn_twang), ("Spanish guitar", nylon)];
const P_WN_CHOIR: &[(&str, Make)] = &[("choir", wn_choir), ("strings", wn_strings)];
const P_WN_BELL: &[(&str, Make)] = &[("church bell", wn_bell), ("tubular bells", wn_chimes), ("jaw harp", wn_jaw_harp)];
const P_WN_BASS: &[(&str, Make)] = &[("electric bass", wn_electric_bass), ("upright bass", bass_upright)];
const K_WN_KICK: &[(&str, MakeKit)] = &[("bass drum", sw_kit_bass_drum)];
const K_WN_SNARE: &[(&str, MakeKit)] = &[("field drum", wn_kit_snare)];
const K_WN_HAT: &[(&str, MakeKit)] = &[("tambourine", wn_kit_tamb)];
const K_WN_PERC: &[(&str, MakeKit)] = &[("anvil, whip, castanets, bell tree", wn_kit_perc)];
const K_WN_CYM: &[(&str, MakeKit)] = &[("vibraslap, gong", wn_kit_cym)];

pub(super) fn palette_western(ch: usize) -> Option<&'static [(&'static str, Make)]> {
    Some(match ch {
        CH_LEAD => P_WN_LEAD,
        CH_ARP | CH_KEYS => P_WN_GUITAR,
        CH_PAD | CH_DRONE => P_WN_CHOIR,
        CH_BELL => P_WN_BELL,
        CH_BASS => P_WN_BASS,
        _ => return None,
    })
}

pub(super) fn kit_palette_western(ch: usize) -> Option<&'static [(&'static str, MakeKit)]> {
    Some(match ch {
        CH_KICK => K_WN_KICK,
        CH_SNARE => K_WN_SNARE,
        CH_HAT => K_WN_HAT,
        CH_PERC => K_WN_PERC,
        CH_CYM => K_WN_CYM,
        _ => return None,
    })
}

fn wn_pair(band: &mut Band, r: &mut Rng, s: &Synth, ch: usize, b: Make, note: f32) {
    let orig = s.ch[ch].patch;
    let pb = levelled(&orig, note, b(r));
    band.tones.push((ch, [orig, pb], note));
}

fn band_western(r: &mut Rng, s: &Synth, style: u8) -> Band {
    let mut b = Band::default();
    let second: Make = match style {
        WNS_LAMENT => sw_violins,
        _ => wn_trumpet,
    };
    wn_pair(&mut b, r, s, CH_LEAD, second, 72.0);
    wn_pair(&mut b, r, s, CH_ARP, if style == WNS_LAMENT { harp } else { nylon }, 55.0);
    wn_pair(&mut b, r, s, CH_KEYS, if style == WNS_RIDE { wn_twang } else { harp }, 55.0);
    wn_pair(&mut b, r, s, CH_PAD, sw_tremolo, 64.0);
    wn_pair(&mut b, r, s, CH_DRONE, sw_timpani, 50.0);
    wn_pair(&mut b, r, s, CH_BELL, if style == WNS_RIDE { wn_chimes } else { wn_jaw_harp }, 60.0);
    wn_pair(&mut b, r, s, CH_BASS, if style == WNS_RIDE { bass_upright } else { wn_electric_bass }, 40.0);
    b
}

/// The recordings against the synthesiser's patches they replace, by style
/// and channel (dB), measured as Star Wars's are (`sw_rec_trim`): the
/// strings' swell and the choir read loud by the levelling's early note,
/// the harmonica and ocarina quiet; the bass drum, the vibraslap and the
/// gong ring otherwise than the synthesiser's.
fn wn_rec_trim(style: u8, ch: usize) -> f32 {
    match (style, ch) {
        (WNS_RIDE, CH_BASS) => 0.9,
        (WNS_RIDE, CH_PAD) => -0.9,
        (WNS_RIDE, CH_ARP) => -1.3,
        (WNS_RIDE, CH_KEYS) => -0.8,
        (WNS_RIDE, CH_DRONE) => -1.0,
        (WNS_DUEL, CH_BASS) => 1.0,
        (WNS_DUEL, CH_PAD) => -4.7,
        (WNS_DUEL, CH_ARP) => -2.6,
        (WNS_DUEL, CH_DRONE) => -2.3,
        (WNS_LAMENT, CH_BASS) => 2.4,
        (WNS_LAMENT, CH_PAD) => -9.8,
        (WNS_LAMENT, CH_ARP) => -1.0,
        (WNS_LAMENT, CH_LEAD) => 4.8,
        (WNS_LAMENT, CH_KEYS) => -2.9,
        (WNS_LAMENT, CH_DRONE) => -1.8,
        _ => 0.0,
    }
}

/// `wn_rec_trim`'s for the kits.
fn wn_kit_trims(style: u8) -> [(usize, f32); 4] {
    match style {
        WNS_RIDE => [(CH_KICK, 7.1), (CH_SNARE, -0.9), (CH_PERC, 4.2), (CH_CYM, 12.0)],
        _ => [(CH_KICK, 7.2), (CH_SNARE, -1.6), (CH_PERC, -4.8), (CH_CYM, 10.6)],
    }
}

/// The Western's sounds (a mix's drummer's: the ride's).
pub(super) fn sounds_western(r: &mut Rng, s: &mut Synth) {
    wn_sounds(r, s, WNS_RIDE)
}

/// The band: the lead (whistle, harmonica or ocarina) to the left, the
/// twangy guitar to the right, the Spanish guitar left of centre, the
/// strings and the choir wide, the bells far off; a spring's reverb (short,
/// bright) and a slap of echo on the guitar.
fn wn_sounds(r: &mut Rng, s: &mut Synth, style: u8) {
    let ch = &mut s.ch;
    let lead: Make = match style {
        WNS_LAMENT => r.pick(&[harmonica, ocarina, wn_whistle]),
        WNS_RIDE => r.pick(&[wn_whistle, harmonica, wn_whistle]),
        _ => wn_whistle,
    };
    ch[CH_LEAD].patch = lead(r);
    ch[CH_LEAD].mono = true;
    ch[CH_ARP].patch = if style == WNS_LAMENT { nylon(r) } else { wn_twang(r) };
    ch[CH_ARP].mono = false;
    ch[CH_KEYS].patch = nylon(r);
    ch[CH_PAD].patch = wn_strings(r);
    ch[CH_DRONE].patch = wn_choir(r);
    ch[CH_DRONE].mono = false;
    ch[CH_BELL].patch = if style == WNS_RIDE { wn_jaw_harp(r) } else { wn_bell(r) };
    ch[CH_BASS].patch = if style == WNS_RIDE { wn_electric_bass(r) } else { sw_cellos(r) };
    ch[CH_BASS].mono = false;
    ch[CH_KICK].drums = sw_kit_bass_drum(r);
    ch[CH_SNARE].drums = wn_kit_snare(r);
    ch[CH_HAT].drums = wn_kit_tamb(r);
    ch[CH_PERC].drums = wn_kit_perc(r);
    ch[CH_CYM].drums = wn_kit_cym(r);
    for (c, db) in wn_kit_trims(style) {
        ch[c].drums = kit_trim(std::mem::take(&mut ch[c].drums), db);
    }
    // the balance (measured over whole pieces, K-weighted): the tune on
    // top, the guitars and strings 6-10 dB under it, the gallop's bass and
    // drum under the guitar's riff
    let ride = style == WNS_RIDE;
    let (gtr, pad, bass) = match style {
        WNS_RIDE => (0.78, 0.48, 0.36),
        WNS_DUEL => (0.78, 0.6, 1.0),
        _ => (0.78, 0.54, 0.8),
    };
    mix(&mut ch[CH_LEAD], 0.75, -0.3, 0.35, 0.12, 0.0);
    mix(&mut ch[CH_ARP], gtr, 0.55, 0.45, 0.2, 0.0);
    mix(&mut ch[CH_KEYS], 0.7, -0.4, 0.25, 0.0, 0.0);
    mix(&mut ch[CH_PAD], pad, 0.1, 0.4, 0.0, 0.0);
    mix(&mut ch[CH_DRONE], 0.63, 0.0, 0.45, 0.0, 0.0);
    mix(&mut ch[CH_BELL], 0.35, 0.45, 0.55, 0.0, 0.0);
    mix(&mut ch[CH_BASS], bass, 0.0, 0.15, 0.0, 0.0);
    mix(&mut ch[CH_KICK], if ride { 0.28 } else { 0.4 }, 0.0, 0.3, 0.0, 0.0);
    mix(&mut ch[CH_SNARE], 0.7, 0.15, 0.3, 0.0, 0.0);
    mix(&mut ch[CH_HAT], 0.48, -0.5, 0.25, 0.0, 0.0);
    mix(&mut ch[CH_PERC], 0.45, 0.3, 0.4, 0.0, 0.0);
    mix(&mut ch[CH_CYM], 0.35, -0.2, 0.45, 0.0, 0.0);
    mix(&mut ch[CH_FX], 0.3, 0.0, 0.5, 0.0, 0.0);
    for (c, w) in [(CH_PAD, 1.0), (CH_DRONE, 1.0), (CH_BASS, 0.6)] {
        ch[c].width = if ch[c].patch.samp != 0 { w } else { 1.0 };
    }
    s.rev.fb = r.range(0.8, 0.84);
    s.rev.damp = 0.2;
    s.rev_wet = 0.3;
    s.dly.fb = 0.2;
    s.dly.lpk = 0.4;
    s.dly_rev = 0.3;
    s.r2mode = R2_OFF;
    s.fx.crackle = 0.0;
    s.fx.hiss = 0.0;
    s.fx.wind_to = 0.0;
    s.fx.rain_to = 0.0;
    s.m.gain = 1.7;
    s.m.target = 0.2;
}

impl Song {
    pub(super) fn piece_western(&mut self, s: &mut Synth) {
        let minor = MODES[self.mode][2] == 3;
        let mode = self.mode;
        let r = &mut self.rng;
        let style = r.weighted(&[3.0, 3.0, 1.5]) as u8;
        let (bs, beat, bpm, meter) = match style {
            WNS_DUEL => (12u32, 4u32, r.range(62.0, 74.0), "3/4 duel"),
            WNS_RIDE => (8, 4, r.range(116.0, 136.0), "2/4 gallop"),
            _ => (12, 4, r.range(66.0, 80.0), "3/4 lament"),
        };
        self.bpm = bpm as f64;
        self.swing = 0.0;
        self.human = 0.008;
        let tonic = self.tonic_near(r_center(style));
        let r = &mut self.rng;
        let pick = |r: &mut Rng, pool: &[&str]| halves(r.pick(pool));
        let (pa, pb) = match (style, minor) {
            (WNS_LAMENT, true) => (WN_LAMENT, WN_MIN_B),
            (_, true) => (WN_MIN_A, WN_MIN_B),
            _ => (WN_MAJ_A, WN_MAJ_B),
        };
        let a = pick(r, pa);
        let b = pick(r, pb);
        let theme = wn_theme(r, &a, mode, bs, -5, 14);
        // the twangy guitar's riff: two bars of eighths on its low strings
        // (the root, the flat second, the fifth under it)
        let cells: &[&[i32]] = &[
            &[0, 0, 1, 0, -5, 0, 1, 3],
            &[0, 3, 5, 3, 0, -2, 0, 1],
            &[0, 0, 7, 5, 3, 1, 0, -5],
            &[0, 1, 0, -5, 0, 1, 3, 1],
        ];
        let cell = r.pick(cells);
        let riff: Vec<(f32, i32)> = cell.iter().enumerate().map(|(k, &p)| ((2 * k) as f32, p)).collect();
        let bar_s = bs as f64 / beat as f64 * 60.0 / self.bpm;
        let long = if bar_s < 1.5 { 16 } else { 8 };
        let mk = |kind: u8, bars: u32, mask: u32, energy: f32| Sec { kind, bars, mask, energy };
        let part = |part: u8, dress: usize, pace: f32| WnPart { part, dress, pace };
        let base = L_LEAD | L_KEYS | L_BASS;
        let plan = match style {
            WNS_DUEL => vec![
                (mk(INTRO, 4, L_LEAD | L_PAD | L_BELL | L_BASS, 0.4), part(WP_ALONE, 0, 1.0)),
                (mk(MAIN, long, base | L_ARP | L_PERC, 0.55), part(WP_THEME, 0, 1.0)),
                (mk(MAIN, long, base | L_ARP | L_PAD | L_DRONE | L_PERC | L_SNARE, 0.7), part(WP_MORE, 0, 1.04)),
                (
                    mk(BUILD, 8, L_KEYS | L_PAD | L_DRONE | L_BASS | L_SNARE | L_BELL | L_KICK, 0.85),
                    part(WP_BUILD, 2, 1.08),
                ),
                (
                    mk(MAIN, long, base | L_ARP | L_PAD | L_DRONE | L_PERC | L_SNARE | L_KICK | L_BELL, 1.0),
                    part(WP_CLIMAX, 1, 1.12),
                ),
                (mk(OUTRO, 2, base | L_PAD | L_DRONE | L_BELL | L_KICK, 1.0), part(WP_END, 1, 1.12)),
            ],
            WNS_RIDE => vec![
                (mk(INTRO, 4, L_BASS | L_KICK | L_HAT | L_ARP | L_BELL, 0.6), part(WP_ALONE, 0, 1.0)),
                (
                    mk(MAIN, long, base | L_ARP | L_KICK | L_HAT | L_SNARE | L_PERC | L_BELL, 0.75),
                    part(WP_THEME, 0, 1.0),
                ),
                (
                    mk(MAIN, long, base | L_ARP | L_PAD | L_DRONE | L_KICK | L_HAT | L_SNARE | L_PERC, 0.9),
                    part(WP_MORE, 1, 1.0),
                ),
                (
                    mk(MAIN, long, base | L_ARP | L_KICK | L_HAT | L_SNARE | L_PERC | L_BELL, 0.8),
                    part(WP_THEME, 0, 1.0),
                ),
                (mk(BREAK, long / 2, L_BASS | L_KICK | L_ARP | L_BELL | L_DRONE, 0.6), part(WP_BUILD, 0, 1.0)),
                (
                    mk(MAIN, long, base | L_ARP | L_PAD | L_DRONE | L_KICK | L_HAT | L_SNARE | L_PERC, 1.0),
                    part(WP_CLIMAX, 1, 1.0),
                ),
                (mk(OUTRO, 2, base | L_PAD | L_DRONE | L_KICK | L_PERC, 1.0), part(WP_END, 1, 1.0)),
            ],
            _ => vec![
                (mk(INTRO, 4, L_KEYS | L_BELL | L_BASS, 0.35), part(WP_ALONE, 0, 1.0)),
                (mk(MAIN, long, base | L_PAD, 0.5), part(WP_THEME, 0, 1.0)),
                (mk(MAIN, long, base | L_PAD | L_ARP | L_DRONE | L_BELL, 0.6), part(WP_MORE, 1, 1.0)),
                (mk(MAIN, long, base | L_PAD | L_ARP, 0.55), part(WP_THEME, 0, 1.0)),
                (mk(OUTRO, 2, L_KEYS | L_PAD | L_BASS | L_BELL | L_LEAD, 0.5), part(WP_END, 0, 1.0)),
            ],
        };
        self.plan = plan.iter().map(|x| x.0).collect();
        let n = self.plan.len();
        self.lifts = vec![0; n];
        self.progs = vec![a.clone(), b.clone()];
        self.prog = a;
        self.wn = Western {
            bar_steps: bs,
            beat_steps: beat,
            meter,
            style,
            bpm0: self.bpm,
            map: plan.iter().map(|x| x.1).collect(),
            bars: Vec::new(),
            progs: self.progs.clone(),
            theme,
            riff,
            tonic,
        };
        let r = &mut self.rng;
        wn_sounds(r, s, style);
        self.band = band_western(r, s, style);
        rec_trim(&mut self.band, s, |ch, _| wn_rec_trim(style, ch));
        s.dly.target = (self.step_len() * 3.0) as f32;
    }

    /// A section begins: its chords, its pace (the duel quickening), its
    /// bars (the end's last one long, its chord left to ring).
    pub(super) fn western_section(&mut self) {
        let sec = self.plan[self.sec];
        let p = self.wn.map[self.sec];
        self.prog = self.wn.progs
            [(p.part == WP_MORE || p.part == WP_CLIMAX) as usize & (self.wn.style == WNS_RIDE) as usize]
            .clone();
        self.bpm = self.wn.bpm0 * p.pace as f64;
        let bs = self.wn.bar_steps;
        let mut bars = vec![bs; sec.bars as usize];
        if p.part == WP_END
            && let Some(l) = bars.last_mut()
        {
            *l = bs * 2;
        }
        self.wn.bars = bars;
    }

    /// A bar of the Western.
    pub(super) fn western(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let p = self.wn.map[self.sec];
        let i = self.sec_bar;
        let bs = self.wn.bs_at(i);
        let last = i + 1 == sec.bars;
        let e = sec.energy;
        if i == 0 {
            s.dly.target = (self.step_len() * 3.0) as f32;
            // the long build: the leveller's target with the section's weight
            s.m.target = 0.2 * (e * e).max(0.15);
        }
        if p.part == WP_END {
            self.wn_end(s, b, sec, m);
            return;
        }
        let c = [self.chord(i * 2), self.chord(i * 2 + 1)];
        let style = self.wn.style;
        // the strings: lonely open fifths at first, then the chords; the
        // tremolo building
        if m & L_PAD != 0 {
            let lo = self.tonic_near(55);
            let v = 0.35 + 0.35 * e;
            if p.part == WP_ALONE || (style == WNS_DUEL && p.part == WP_THEME) {
                if i % 2 == 0 {
                    let r0 = lo + (self.root + c[0].root - lo).rem_euclid(12);
                    for q in [r0 - 12, r0 - 5, r0] {
                        b.note(0.0, 0.0, CH_PAD, q, v * 0.8, 2.0 * bs as f32, false);
                    }
                }
            } else {
                let vv = if p.part == WP_BUILD { v * (0.6 + 0.4 * (i as f32 + 1.0) / sec.bars as f32) } else { v };
                self.pads(b, CH_PAD, 52, 76, vv, false, 0.5);
            }
        }
        if m & L_BASS != 0 {
            self.wn_bass(s, b, c, p.part, bs, e);
        }
        if m & L_KEYS != 0 {
            self.wn_guitar(s, b, c, p.part, bs, e);
        }
        if m & L_ARP != 0 {
            self.wn_twang(s, b, c, p.part, bs, e);
        }
        if m & L_LEAD != 0 {
            self.wn_lead(b, p.part, sec, bs);
        }
        if m & L_DRONE != 0 {
            self.wn_choir(s, b, c, p.part, bs, e, last);
        }
        if m & L_BELL != 0 {
            self.wn_bells(s, b, c, p.part, bs, e);
        }
        self.wn_drums(b, sec, m, p.part, bs, last);
    }

    /// The bass: the duel's and the lament's roots and fifths (open,
    /// lonely), the ride's gallop (an eighth and two 16ths).
    fn wn_bass(&mut self, s: &Synth, b: &mut Bar, c: [Harm; 2], part: u8, bs: u32, e: f32) {
        let lo = 36;
        let root = |x: Harm| lo + (self.root + x.bass() - lo).rem_euclid(12);
        let v = 0.7 + 0.25 * e;
        if self.wn.style == WNS_RIDE && part != WP_ALONE || (self.wn.style == WNS_RIDE && self.sec_bar >= 2) {
            if s.ch[CH_BASS].patch.a > 0.08 {
                b.note(0.0, 0.0, CH_BASS, root(c[0]), v, bs as f32, false);
                return;
            }
            for beat in (0..bs).step_by(4) {
                let x = c[(beat * 2 >= bs) as usize];
                let r0 = root(x);
                for (k, (d, l)) in [(0.0, 1.8), (2.0, 0.9), (3.0, 0.9)].iter().enumerate() {
                    let q = if k == 0 { r0 } else { r0 + if beat % 8 == 4 && k == 2 { 7 } else { 0 } };
                    b.note(
                        beat as f32 + d,
                        self.rng.range(0.0, self.human * 0.5),
                        CH_BASS,
                        q,
                        v * if k == 0 { 1.0 } else { 0.75 },
                        *l,
                        false,
                    );
                }
            }
            return;
        }
        if part == WP_ALONE && self.sec_bar % 2 == 1 {
            return;
        }
        let x = c[0];
        b.note(0.0, 0.0, CH_BASS, root(x), v, 8.0, false);
        let f = root(x) + 7;
        if part != WP_ALONE {
            b.note(8.0, 0.0, CH_BASS, if f > 50 { f - 12 } else { f }, v * 0.8, (bs - 8) as f32, false);
        }
    }

    /// The Spanish guitar: the slow styles' fingerpicking (the bass, then
    /// the chord's tones in eighths), a rasgueo on each chord in the
    /// climax; the ride's strum on the eighths, down hard on the beats.
    fn wn_guitar(&mut self, s: &Synth, b: &mut Bar, c: [Harm; 2], part: u8, bs: u32, e: f32) {
        let v = 0.5 + 0.3 * e;
        let slow = s.ch[CH_KEYS].patch.a > 0.08;
        let half = bs as f32 / 2.0;
        if slow {
            self.wn_held(b, CH_KEYS, c, 50, 72, v * 0.6, bs);
            return;
        }
        let voic = |me: &Song, x: Harm| {
            let lo = 40 + (me.root + x.root - 40).rem_euclid(12);
            let mut w = in_range(&me.pcs(x), lo, lo + 19);
            w.truncate(5);
            w
        };
        if self.wn.style == WNS_RIDE || part == WP_CLIMAX {
            for st in (0..bs).step_by(2) {
                let x = c[(st as f32 >= half) as usize];
                let w = voic(self, x);
                let on = st % 4 == 0;
                let late = self.rng.range(0.0, self.human);
                let gap = 0.008;
                let ord: Vec<i32> = if on { w.clone() } else { w.iter().rev().take(3).copied().collect() };
                for (k, &q) in ord.iter().enumerate() {
                    b.note(st as f32, late + gap * k as f32, CH_KEYS, q, v * if on { 1.0 } else { 0.6 }, 1.8, false);
                }
            }
            return;
        }
        for st in (0..bs).step_by(2) {
            let x = c[(st as f32 >= half) as usize];
            let w = voic(self, x);
            let k = (st / 2) as usize;
            let q = if k == 0 || (st as f32 == half && c[1] != c[0]) {
                w[0]
            } else {
                w[1 + [2, 1, 2, 0, 2, 1][k % 6].min(w.len() - 2)]
            };
            b.note(
                st as f32,
                self.rng.range(0.0, self.human),
                CH_KEYS,
                q,
                v * if k == 0 { 1.0 } else { 0.75 },
                6.0,
                false,
            );
        }
    }

    fn wn_held(&mut self, b: &mut Bar, ch: usize, c: [Harm; 2], lo: i32, hi: i32, vel: f32, bs: u32) {
        let v = self.voicing(c[0], lo, hi, &self.pad_v, false);
        for &n in &v {
            b.note(0.0, 0.0, ch, n, vel, bs as f32 + 0.3, false);
        }
        self.pad_v = v;
    }

    /// The twangy guitar: the riff on its low strings (the ride's, under
    /// everything; the duel's in its theme), the tune an octave down in the
    /// duel's second time; the lament's harp or nylon arpeggio.
    fn wn_twang(&mut self, s: &Synth, b: &mut Bar, c: [Harm; 2], part: u8, bs: u32, e: f32) {
        let v = 0.55 + 0.3 * e;
        if s.ch[CH_ARP].patch.a > 0.08 {
            self.wn_held(b, CH_ARP, c, 48, 67, v * 0.5, bs);
            return;
        }
        let lo = 40 + (self.root - 40).rem_euclid(12);
        if self.wn.style == WNS_LAMENT {
            let t = self.arp_tones(c[0], lo + 12);
            for (k, st) in (0..bs).step_by(2).enumerate() {
                b.note(st as f32, self.rng.range(0.0, self.human), CH_ARP, t[k % t.len()] + 12, v * 0.7, 5.0, false);
            }
            return;
        }
        if self.wn.style == WNS_DUEL && part == WP_MORE {
            // the tune, an octave and more under the whistle
            let fb = self.wn.bar_steps as f32;
            let bar = (self.sec_bar % 8) as f32;
            let notes: Vec<Tn> = self.wn.theme.iter().filter(|n| (n.st / fb).floor() == bar).copied().collect();
            for n in notes {
                b.note(n.st - bar * fb, 0.0, CH_ARP, self.wn.tonic + n.p - 12, n.v * v, n.len * 0.9, false);
            }
            return;
        }
        let half = bs as f32 / 2.0;
        let bar = (self.sec_bar % 2) as usize;
        let per = (bs / 2) as usize;
        for (k, &(st, iv)) in self.wn.riff.iter().enumerate().skip(bar * per).take(per) {
            let st = st - (bar * per * 2) as f32;
            if st >= bs as f32 {
                continue;
            }
            let x = c[(st >= half) as usize];
            let r0 = lo + (self.root + x.root - lo).rem_euclid(12);
            let q = self.root + sw_step(self.mode, r0 - self.root, 0) + iv;
            let acc = k % 4 == 0;
            b.note(st, self.rng.range(0.0, self.human * 0.5), CH_ARP, q, v * if acc { 1.0 } else { 0.8 }, 1.8, false);
        }
    }

    /// The tune: the whistle's (or the harmonica's, the ocarina's) alone at
    /// first, sliding into its long notes; the trumpet's in the climax, an
    /// octave up, its long notes shaken.
    fn wn_lead(&mut self, b: &mut Bar, part: u8, sec: Sec, bs: u32) {
        let fb = self.wn.bar_steps as f32;
        let i = self.sec_bar;
        if part == WP_ALONE && self.wn.style == WNS_RIDE {
            return;
        }
        let bar = (i % 8) as f32;
        let notes: Vec<Tn> = self.wn.theme.iter().filter(|n| (n.st / fb).floor() == bar).copied().collect();
        let vel = 0.65 + 0.3 * sec.energy;
        let up = if part == WP_CLIMAX && self.wn.style == WNS_DUEL { 0 } else { 0 };
        let mut prev_end = -1.0f32;
        for n in notes {
            let st = n.st - bar * fb;
            if st >= bs as f32 {
                continue;
            }
            let slide = (st - prev_end).abs() < 0.6 && n.len >= 2.0;
            let len = n.len * 0.97;
            b.note(st, self.rng.range(0.0, self.human), CH_LEAD, self.wn.tonic + n.p + up, n.v * vel, len, slide);
            prev_end = st + n.len;
        }
    }

    /// The choir: held "ah"s under the theme's second time and the climax;
    /// the ride's grunts on the beats of its second bar; the timpani (the
    /// pair's other) rolling in the build.
    fn wn_choir(&mut self, s: &Synth, b: &mut Bar, c: [Harm; 2], part: u8, bs: u32, e: f32, last: bool) {
        let perc = s.ch[CH_DRONE].patch.a < 0.05;
        let t = 38 + (self.root - 38).rem_euclid(12);
        if perc {
            // timpani: a roll swelling through the bar, a stroke on its downbeat
            b.note(0.0, 0.0, CH_DRONE, t, 0.5 + 0.4 * e, 3.0, false);
            let n = (bs as f32 / 0.5) as usize;
            let x0 = self.sec_bar as f32 / 8.0;
            for k in 1..n {
                let x = (x0 + k as f32 / (8 * n) as f32).min(1.0);
                b.note(
                    0.5 * k as f32,
                    0.0,
                    CH_DRONE,
                    if last && k > n / 2 { t + 7 - 12 + 12 } else { t },
                    0.2 + 0.55 * x,
                    0.4,
                    false,
                );
            }
            return;
        }
        let lo = self.tonic_near(55) - 3;
        if self.wn.style == WNS_RIDE && part != WP_BUILD {
            // "hu!": short, low, on the beats of every other bar
            if self.sec_bar % 2 == 1 {
                for st in (0..bs).step_by(4) {
                    for q in [self.wn.tonic - 12, self.wn.tonic - 5] {
                        b.note(st as f32, 0.0, CH_DRONE, q, 0.8 + 0.15 * e, 0.8, false);
                    }
                }
            }
            return;
        }
        if self.sec_bar % 2 == 0 || c[0] != self.chord(self.sec_bar * 2 - 1) {
            let v = self.voicing(c[0], lo, lo + 16, &self.keys_v, false);
            for &q in &v {
                b.note(0.0, 0.0, CH_DRONE, q, 0.4 + 0.3 * e, 2.0 * bs as f32 - 1.0, false);
            }
            self.keys_v = v;
        }
    }

    /// The bells: the church bell tolling the tonic at each phrase's start
    /// (far off), the tubular bells' chord tones; the ride's jaw harp
    /// boinging on the beats.
    fn wn_bells(&mut self, s: &Synth, b: &mut Bar, c: [Harm; 2], part: u8, bs: u32, e: f32) {
        let jaw = s.ch[CH_BELL].patch.s == 0.0 && s.ch[CH_BELL].patch.res > 0.5;
        if jaw {
            let p = 40 + (self.root - 40).rem_euclid(12);
            for st in (0..bs).step_by(4) {
                b.note(st as f32, self.rng.range(0.0, self.human), CH_BELL, p, 0.75 + 0.2 * e, 1.5, false);
                if self.var_b() {
                    b.note(st as f32 + 2.0, 0.0, CH_BELL, p, 0.55, 1.0, false);
                }
            }
            return;
        }
        let i = self.sec_bar;
        if i % 4 == 0 || (part == WP_BUILD && i % 2 == 0) {
            let t = self.tonic_near(60);
            b.note(0.0, 0.0, CH_BELL, t, 0.55 + 0.3 * e, 2.0 * bs as f32, false);
            if part == WP_CLIMAX {
                let r0 = t + (c[0].root) % 12;
                b.note(self.wn.beat_steps as f32, 0.0, CH_BELL, r0 - 12, 0.5, bs as f32, false);
            }
        }
    }

    /// The percussion: the gallop's bass drum and field drum, its
    /// tambourine in eighths, the anvil on the off-beats, a whip crack
    /// closing a phrase; the duel's castanets, its snare's roll in the
    /// build, the vibraslap and gong at its turns.
    fn wn_drums(&mut self, b: &mut Bar, sec: Sec, m: u32, part: u8, bs: u32, last: bool) {
        let i = self.sec_bar;
        let e = sec.energy;
        let h = self.human;
        let style = self.wn.style;
        let r = &mut self.rng;
        if i == 0 && part == WP_CLIMAX {
            b.hit(0.0, 0.0, CH_CYM, WC_GONG, 0.8);
        }
        if i == 0 && part == WP_ALONE && style == WNS_DUEL {
            b.hit(0.0, 0.0, CH_CYM, WC_SLAP, 0.6);
        }
        if style == WNS_RIDE {
            if m & L_KICK != 0 {
                for st in (0..bs).step_by(4) {
                    b.hit(st as f32, r.range(0.0, h), CH_KICK, 0, (0.6 + 0.3 * e) * if st == 0 { 1.0 } else { 0.85 });
                }
            }
            if m & L_SNARE != 0 {
                for st in 0..bs {
                    let x = st % 4;
                    if x == 0 || x == 2 || x == 3 {
                        b.hit(
                            st as f32,
                            r.range(0.0, h),
                            CH_SNARE,
                            (x != 0) as u8,
                            (0.3 + 0.25 * e) * if st == 4 { 1.0 } else { 0.55 },
                        );
                    }
                }
            }
            if m & L_HAT != 0 {
                for st in (0..bs).step_by(2) {
                    b.hit(st as f32, r.range(0.0, h), CH_HAT, (st % 4 == 2) as u8, 0.4 + 0.15 * e);
                }
            }
            if m & L_PERC != 0 {
                if part == WP_CLIMAX || r.chance(0.5) {
                    b.hit(6.0f32.min(bs as f32 - 1.0), r.range(0.0, h), CH_PERC, WK_ANVIL, 0.5 + 0.2 * e);
                }
                if i % 8 == 7 || last {
                    b.hit(bs as f32 - 2.0, 0.0, CH_PERC, WK_WHIP, 0.85);
                }
            }
            return;
        }
        // the slow styles
        if m & L_PERC != 0 && part != WP_ALONE {
            for st in (4..bs).step_by(2) {
                if r.chance(0.7) {
                    b.hit(
                        st as f32,
                        r.range(0.0, h),
                        CH_PERC,
                        WK_CAST,
                        (0.35 + 0.25 * e) * if st % 4 == 0 { 1.0 } else { 0.7 },
                    );
                }
            }
            if i % 8 == 7 {
                b.hit(0.0, 0.0, CH_PERC, WK_TREE, 0.5);
            }
            if part == WP_CLIMAX && i % 4 == 2 {
                b.hit(4.0, 0.0, CH_PERC, WK_ANVIL, 0.6);
            }
        }
        if m & L_SNARE != 0 {
            if part == WP_BUILD {
                let x0 = i as f32 / sec.bars as f32;
                for k in 0..bs * 2 {
                    let x = x0 + k as f32 / (2 * bs * sec.bars) as f32;
                    b.hit(k as f32 * 0.5, 0.0, CH_SNARE, 1, 0.15 + 0.5 * x);
                }
            } else if i % 2 == 1 {
                b.hit(bs as f32 - 4.0, r.range(0.0, h), CH_SNARE, 0, 0.3 + 0.2 * e);
            }
        }
        if m & L_KICK != 0 && part != WP_BUILD {
            b.hit(0.0, r.range(0.0, h), CH_KICK, 0, 0.6 + 0.3 * e);
        }
        if last && matches!(part, WP_BUILD | WP_MORE) {
            b.hit(bs as f32 - 4.0, 0.0, CH_CYM, WC_SLAP, 0.55);
        }
    }

    /// The end: the dominant (a bar of the Andalusian or Phrygian close: the
    /// sixth, or the flat second, then the fifth), then everyone on the
    /// tonic, held over a long last bar: the bell, the gong, the bass drum,
    /// a whip (the ride's); the lament's softly.
    fn wn_end(&mut self, s: &Synth, b: &mut Bar, sec: Sec, m: u32) {
        let i = self.sec_bar;
        let bs = self.wn.bs_at(i);
        let style = self.wn.style;
        let minor = MODES[self.mode][2] == 3;
        let tonic = self.wn.tonic;
        let soft = if style == WNS_LAMENT { 0.6 } else { 1.0 };
        if i + 1 < sec.bars {
            let pre =
                if self.mode == 2 { Harm { root: 1, q: Q_MAJ, inv: 0 } } else { Harm { root: 8, q: Q_MAJ, inv: 0 } };
            let dom = Harm { root: 7, q: Q_MAJ, inv: 0 };
            let c = [pre, dom];
            if m & L_PAD != 0 {
                self.wn_held(b, CH_PAD, c, 52, 76, 0.6 * soft, bs);
            }
            if m & L_KEYS != 0 {
                self.wn_guitar(s, b, c, WP_CLIMAX, bs, sec.energy * soft);
            }
            if m & L_BASS != 0 {
                for (h, &x) in c.iter().enumerate() {
                    let r0 = 36 + (self.root + x.root - 36).rem_euclid(12);
                    b.note((h as u32 * bs / 2) as f32, 0.0, CH_BASS, r0, 0.85 * soft, (bs / 2) as f32, false);
                }
            }
            if m & L_LEAD != 0 {
                // the tune's last call: the fifth's leading note up to the dominant, held
                b.note(0.0, 0.0, CH_LEAD, tonic + 8, 0.85 * soft, (bs / 2) as f32, false);
                b.note((bs / 2) as f32, 0.0, CH_LEAD, tonic + 7, 0.9 * soft, (bs / 2) as f32, true);
            }
            if m & L_DRONE != 0 && s.ch[CH_DRONE].patch.a >= 0.05 {
                for q in [tonic - 5, tonic - 1, tonic + 2] {
                    b.note(0.0, 0.0, CH_DRONE, q, 0.6 * soft, bs as f32, false);
                }
            }
            if style != WNS_LAMENT {
                let n = (bs as f32 / 0.5) as usize;
                for k in 0..n {
                    b.hit(0.5 * k as f32, 0.0, CH_SNARE, 1, 0.2 + 0.5 * k as f32 / n as f32);
                }
            }
            return;
        }
        let q = if minor { Q_MIN } else { Q_MAJ };
        let home = Harm { root: 0, q, inv: 0 };
        let len = bs as f32 - 1.0;
        if m & L_PAD != 0 {
            for n in self.voicing(home, 52, 76, &self.pad_v, false) {
                b.note(0.0, 0.0, CH_PAD, n, 0.75 * soft, len, false);
            }
        }
        if m & L_KEYS != 0 {
            let lo = 40 + (self.root - 40).rem_euclid(12);
            for (k, n) in in_range(&self.pcs(home), lo, lo + 19).into_iter().enumerate() {
                b.note(0.0, 0.012 * k as f32, CH_KEYS, n, 0.85 * soft, len, false);
            }
        }
        if m & L_BASS != 0 {
            let r0 = 36 + (self.root - 36).rem_euclid(12);
            b.note(0.0, 0.0, CH_BASS, r0, soft, len, false);
        }
        if m & L_LEAD != 0 {
            b.note(0.0, 0.0, CH_LEAD, tonic + if style == WNS_LAMENT { 0 } else { 12 }, 0.95 * soft, len, false);
        }
        if m & L_DRONE != 0 {
            if s.ch[CH_DRONE].patch.a >= 0.05 {
                for n in self.voicing(home, 50, 67, &[], false) {
                    b.note(0.0, 0.0, CH_DRONE, n, 0.7 * soft, len, false);
                }
            } else {
                let t = 38 + (self.root - 38).rem_euclid(12);
                b.note(0.0, 0.0, CH_DRONE, t, 0.95, 4.0, false);
            }
        }
        if m & L_BELL != 0 {
            b.note(0.0, 0.0, CH_BELL, self.tonic_near(60), 0.8 * soft, len, false);
        }
        if style == WNS_LAMENT {
            return;
        }
        b.hit(0.0, 0.0, CH_KICK, 0, 1.0);
        b.hit(0.0, 0.0, CH_CYM, WC_GONG, 0.85);
        if style == WNS_RIDE {
            b.hit(0.0, 0.0, CH_PERC, WK_WHIP, 0.9);
            b.hit(0.0, 0.0, CH_PERC, WK_ANVIL, 0.8);
        }
    }

    /// The Western's beat in any meter, for a mix: the gallop (the bass
    /// drum on each beat, the field drum's eighth and two 16ths, the
    /// tambourine's eighths), the anvil off the beat and a whip crack
    /// closing a phrase in a strong section; a quiet one's castanets; the
    /// vibraslap's rattle at a section's turn.
    pub(super) fn western_beat(&mut self, _s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        let (bs, beat) = (self.bar_steps(), self.beat_steps());
        let i = self.sec_bar;
        let e = sec.energy;
        let full = sec.kind == MAIN;
        let h = self.human;
        let fill = if full || sec.kind == BUILD { self.fill(sec) } else { 0 };
        let r = &mut self.rng;
        // a section's turn: the vibraslap's rattle
        if i == 0 && matches!(sec.kind, INTRO | MAIN) {
            b.hit(0.0, 0.0, CH_CYM, WC_SLAP, 0.55);
        }
        if matches!(sec.kind, INTRO | BREAK) && m & (L_KICK | L_SNARE) == 0 {
            for st in (beat..bs).step_by(2) {
                if r.chance(0.6) {
                    b.hit(st as f32, r.range(0.0, h), CH_PERC, WK_CAST, 0.35);
                }
            }
            return;
        }
        for st in (0..bs).step_by(beat as usize) {
            b.hit(st as f32, r.range(0.0, h), CH_KICK, 0, (0.55 + 0.3 * e) * if st == 0 { 1.0 } else { 0.8 });
            // the gallop: the beat's eighth and two 16ths (a compound beat's three eighths)
            let g: &[u32] = if beat == 6 { &[0, 2, 4] } else { &[0, 2, 3] };
            for &d in g.iter().filter(|&&d| st + d < bs) {
                b.hit(
                    (st + d) as f32,
                    r.range(0.0, h),
                    CH_SNARE,
                    (d != 0) as u8,
                    (0.28 + 0.2 * e) * if d == 0 { 1.0 } else { 0.6 },
                );
            }
        }
        // the tambourine in eighths with the gallop (shaken on the off-beats)
        for st in (0..bs).step_by(2) {
            let on = st % beat == 0;
            b.hit(st as f32, r.range(0.0, h), CH_HAT, (!on) as u8, (0.35 + 0.1 * e) * if on { 1.0 } else { 0.8 });
        }
        if full && e >= 0.7 && bs > 2 {
            b.hit((bs - 2) as f32, r.range(0.0, h), CH_PERC, WK_ANVIL, 0.45);
        }
        if fill == 2 || (full && i % 8 == 7) {
            b.hit((bs - 1) as f32, 0.0, CH_PERC, WK_WHIP, 0.8);
        }
    }
}

fn r_center(style: u8) -> i32 {
    match style {
        WNS_RIDE => 69,
        WNS_LAMENT => 67,
        _ => 70,
    }
}
