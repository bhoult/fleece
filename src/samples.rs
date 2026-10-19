//! Recorded instruments for the music: multisampled WAVs listed in a
//! manifest, and a General MIDI SoundFont's presets and drum kits, loaded
//! once (never on the audio thread), held read-only and played by the
//! synthesiser's voices in place of their oscillators.
//!
//! The samples directory (`~/.local/share/fleece/samples`, or `--samples
//! DIR`; `tools/get-samples.sh` fills it) holds `MuseScore_General.sf2`,
//! and `manifest.tsv` with the WAVs it lists under `multi/`. Genre code asks
//! for an instrument by name when it builds a patch (`multi`, `gm`, `drum`)
//! and gets an id, or None (no samples, or no such instrument) and keeps its
//! synthesiser patch. A note on a sampled patch is then an index lookup: the
//! id's table of note x velocity layer gives the zones to play.
//!
//! Levels: every instrument is brought to one reference, its loudest layer
//! at -20 dBFS RMS over the first second of a middle note (the manifest's
//! `gain_db` does it for the WAVs, each root and velocity layer evened to
//! the instrument's level as well; `normalise` for the SoundFont's presets;
//! a kit keeps its maker's balance, brought to the reference at its snare),
//! and a sampled voice is then scaled by `CAL` (a drum by `CAL_DRUM`),
//! measured so that a sampled note sounds about as loud as the
//! synthesiser's at the same patch gain and velocity.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Once, OnceLock};
use std::time::Instant;

/// The engine's sample rate.
const RATE: f32 = 48000.0;

/// The SoundFont, the manifest, in the samples directory.
pub const SF2: &str = "MuseScore_General.sf2";
const MANIFEST: &str = "manifest.tsv";

/// The SoundFont's melodic presets loaded, by name (as its phdr has them,
/// bank 0, else the lowest melodic bank: Funk Guitar is bank 8's; never the
/// "Expr." variants, whose dynamics follow a controller). The pianos are
/// one set of samples (100 MB).
pub const GM_PRESETS: &[&str] = &[
    // guitars
    "Nylon String Guitar",
    "Steel String Guitar",
    "Jazz Guitar",
    "Clean Guitar",
    "Palm Muted Guitar",
    "Overdrive Guitar",
    "Distortion Guitar",
    "Funk Guitar",
    "Mandolin",
    "Banjo",
    // organs
    "Drawbar Organ",
    "Percussive Organ",
    "Rock Organ",
    "Church Organ",
    "Reed Organ",
    // pianos and keys
    "Grand Piano",
    "Bright Grand Piano",
    "Mellow Grand Piano",
    "Honky-Tonk Piano",
    "Tine Electric Piano",
    "Detuned Tine EP",
    "FM Electric Piano",
    "Harpsichord",
    "Clavinet",
    // mallets and bells
    "Celesta",
    "Glockenspiel",
    "Music Box",
    "Vibraphone",
    "Marimba",
    "Xylophone",
    "Tubular Bells",
    "Dulcimer",
    "Kalimba",
    "Tinker Bell",
    "Steel Drums",
    // free reeds
    "Accordion",
    "Italian Accordion",
    "Bandoneon",
    "Harmonica",
    // basses
    "Acoustic Bass",
    "Fingered Bass",
    "Picked Bass",
    "Fretless Bass",
    "Slap Bass",
    "Synth Bass 1",
    "Synth Bass 2",
    "Synth Bass 3",
    "Synth Bass 4",
    // saxes and reeds
    "Soprano Sax",
    "Alto Sax",
    "Tenor Sax",
    "Baritone Sax",
    "Oboe",
    "English Horn",
    "Bassoon",
    "Clarinet",
    "Bagpipe",
    "Shenai",
    // pipes
    "Piccolo",
    "Flute",
    "Recorder",
    "Pan Flute",
    "Whistle",
    "Ocarina",
    // brass
    "Trumpet",
    "Harmon Mute Trumpet",
    "Trombone",
    "Tuba",
    "French Horns",
    "Brass Section",
    "Synth Brass 1",
    "Synth Brass 2",
    // strings (General MIDI's String Ensembles 1 and 2 are "Strings Fast"
    // and "Strings Slow" here)
    "Violin",
    "Viola",
    "Cello",
    "Fiddle",
    "Strings Fast",
    "Strings Slow",
    "Strings Pizzicato",
    "Synth Strings 1",
    "Synth Strings 2",
    "Harp",
    "Contrabass",
    // voices
    "Choir Aahs",
    "Voice Oohs",
    "Synth Voice",
    "Orchestra Hit",
    // synthesisers: leads
    "Square Lead",
    "Saw Lead",
    "Calliope Lead",
    "Chiffer Lead",
    "Charang",
    "Solo Vox",
    "5th Saw Wave",
    "Bass & Lead",
    "Detuned Saw",
    "Sine Wave",
    // pads
    "Fantasia",
    "Warm Pad",
    "Polysynth",
    "Space Voice",
    "Bowed Glass",
    "Metal Pad",
    "Halo Pad",
    "Sweep Pad",
    // effects
    "Ice Rain",
    "Soundtrack",
    "Crystal",
    "Atmosphere",
    "Brightness",
    "Echo Drops",
    "Star Theme",
    // percussion played by note
    "Agogo",
    "Woodblock",
    "Taiko Drum",
    "Melodic Tom",
    "Synth Drum",
    "Castanets",
    "Concert Bass Drum",
    "808 Tom",
    "Timpani",
    // -- star wars / western: the strings' tremolo, a cymbal's swell (a
    // reversed crash), a church bell
    "Strings Tremolo",
    "Reverse Cymbal",
    "Church Bell",
    // -- end star wars / western
    // -- indonesian / celtic: the suling's stand-in (an end-blown bamboo
    // flute), the bouzouki's (a twelve-string's octave courses)
    "Shakuhachi",
    "12-String Guitar",
    // -- end indonesian / celtic
];

/// Its drum kits loaded (bank 128), by name.
pub const GM_KITS: &[&str] = &["Standard", "Room", "Power", "Jazz", "Brush", "Electronic", "TR-808", "Orchestra Kit"];

/// The reference every instrument is brought to: -20 dBFS RMS (full scale 1).
const REF_RMS: f32 = 0.1;

/// A sampled voice's level over the reference. Measured (the test
/// `calibration` in music.rs): the plain saw `PATCH` at gain 1, velocity
/// 0.8, note 60 has an RMS of 0.325 by `loudness` (30-150 ms), a sine at
/// the reference played the same way 0.08; so 0.325 / 0.08 (+12.2 dB).
pub const CAL: f32 = 4.06;

/// A sampled drum's level over the reference. Measured (the test
/// `calibration`): a dozen of the synthesiser's drums (two kicks, two
/// snares, a clap, closed and open hats, a shaker, a ride, a crash, a tom,
/// a conga) at gain 1, velocity 0.8, against MuseScore General's Standard
/// kit's keys of the same kinds (the kit brought to the reference at its
/// snare), RMS over the first second: 4.4 dB under the reference on
/// average (each kind within 7 dB of that).
pub const CAL_DRUM: f32 = 0.60;

/// Gates shorter than this (seconds) take the short articulation of an
/// "auto" instrument.
pub const AUTO_SHORT: f32 = 0.25;

/// The window (seconds from the onset) an "auto" instrument's short
/// articulation's level is taken over, to match the long one's (over its
/// first second, the reference's window, mostly steady): a short note's
/// momentary loudness (EBU R128's 400 ms).
const AUTO_MATCH: f32 = 0.4;

/// Zones sounding at once in one voice: a stereo pair (the SoundFont keeps
/// a stereo sample as two zones panned apart), or layers (the pianos have
/// two stereo pairs over some velocities).
const MAXZ: usize = 4;

/// 32.32 fixed point's one.
const FX: f64 = 4294967296.0;

/// The fade a sample that runs out ends with (10 ms, samples).
const END_FADE: f64 = 480.0;

/// A sustained recording ("sus", "vib") at least `LOOP_MIN` seconds long
/// loops `LOOP` (seconds from its onset) while the gate is open, the last
/// `LOOP_XF` of each time round crossfaded into the start's: the manifest's
/// stop at 5.5 s, faded, with no loop points of their own.
const LOOP_MIN: f32 = 5.0;
const LOOP: (f32, f32) = (1.5, 4.5);
const LOOP_XF: f32 = 0.25;

/// One sample as played over some keys and velocities.
#[derive(Clone, Copy, Debug)]
struct Zone {
    /// where its frames start in the bank's data, how many, and its
    /// channels (1, or 2 interleaved)
    off: usize,
    len: u32,
    chans: u8,
    /// the key that plays it at its own rate, tuning on top (semitones), and
    /// semitones per key (0: every key the same pitch)
    root: f32,
    tune: f32,
    scale: f32,
    /// its sample rate over the engine's
    rate: f32,
    /// the loop (frames from its start, the end exclusive) and its mode: 0
    /// none, 1 always, 3 until release; frames before its end crossfaded
    /// into those before its start (0: none, a SoundFont's loop is made to
    /// join)
    ls: u32,
    le: u32,
    mode: u8,
    xf: u32,
    /// the left and right gains (level, normalisation, pan, 1/32768)
    gl: f32,
    gr: f32,
    /// the SoundFont's exclusive class (0 none): a hit chokes its class
    excl: u8,
    /// its own volume envelope (the SoundFont's): hold and decay
    /// (timecents) and how they shorten up the keyboard (per key from 60),
    /// the sustain level, the release (seconds to -60 dB)
    hold_tc: f32,
    dec_tc: f32,
    key_hold: f32,
    key_dec: f32,
    sus: f32,
    rel: f32,
}

/// Where a note and layer's zones are in `Bank::picks`: `rr` round robins
/// of `per` zones each (all of them to sound together).
#[derive(Clone, Copy, Default, Debug)]
struct Cell {
    first: u32,
    rr: u8,
    per: u8,
}

/// An instrument as the voices see it: velocity to layer, and a cell for
/// every note (or a kit's one key) and layer.
struct Inst {
    layer: [u8; 128],
    layers: usize,
    cells: Vec<Cell>,
    /// a kit's key (its one note), None for an instrument played by note
    key: Option<u8>,
    /// "auto": the short articulation's id and the long one's, and the
    /// short one's gain (matching the long one's level)
    auto: Option<(u16, u16, f32)>,
    /// its zones (each instrument has its own, contiguous)
    zones: std::ops::Range<u32>,
}

/// Every sample loaded, and what plays them. Read-only once loaded.
pub struct Bank {
    data: Vec<i16>,
    zones: Vec<Zone>,
    picks: Vec<u32>,
    /// ids are 1 + an index here (0 is the synthesiser)
    insts: Vec<Inst>,
    multis: Vec<(String, String, u16)>,
    gms: Vec<(String, u16)>,
    kits: Vec<(String, [u16; 128])>,
    /// what loading skipped, and why
    pub warnings: Vec<String>,
}

impl Bank {
    fn new() -> Bank {
        Bank {
            data: Vec::new(),
            zones: Vec::new(),
            picks: Vec::new(),
            insts: Vec::new(),
            multis: Vec::new(),
            gms: Vec::new(),
            kits: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// A multisampled instrument's articulation ("sus", "stac", "pizz",
    /// "vib", ... or "auto": by each note's gate, see `AUTO_SHORT`).
    pub fn multi(&self, instrument: &str, articulation: &str) -> Option<u16> {
        self.multis.iter().find(|m| m.0 == instrument && m.1 == articulation).map(|m| m.2)
    }

    /// A SoundFont preset, by name (one of `GM_PRESETS`).
    #[allow(dead_code)] // the genres' hook
    pub fn gm(&self, preset: &str) -> Option<u16> {
        self.gms.iter().find(|g| g.0 == preset).map(|g| g.1)
    }

    /// One key of a SoundFont kit (one of `GM_KITS`), General MIDI's
    /// numbering (36 kick, 38 snare, 42 closed hat, 46 open hat, ...).
    #[allow(dead_code)] // the genres' hook
    pub fn drum(&self, kit: &str, key: u8) -> Option<u16> {
        let id = self.kits.iter().find(|k| k.0 == kit)?.1.get(key as usize).copied()?;
        (id != 0).then_some(id)
    }

    /// Bytes held: the samples, and the tables.
    pub fn bytes(&self) -> usize {
        self.data.len() * 2
            + self.zones.len() * std::mem::size_of::<Zone>()
            + self.picks.len() * 4
            + self.insts.iter().map(|i| i.cells.len() * std::mem::size_of::<Cell>() + 160).sum::<usize>()
    }

    /// Instruments, presets and kits held.
    pub fn counts(&self) -> (usize, usize, usize) {
        (self.multis.iter().filter(|m| m.1 != "auto").count(), self.gms.len(), self.kits.len())
    }

    /// The zones for `id` at `note`, velocity `vel` (0-1), `len` samples of
    /// gate, round robin `rr`; a kit's key; a gain on them (an "auto"
    /// instrument's short articulation's).
    fn pick(&self, id: u16, note: f32, vel: f32, len: i64, rr: u32) -> Option<(&[u32], Option<u8>, f32)> {
        let mut inst = self.insts.get((id as usize).checked_sub(1)?)?;
        let mut g = 1.0;
        if let Some((short, long, gs)) = inst.auto {
            let sub = if len < (AUTO_SHORT * RATE) as i64 {
                g = gs;
                short
            } else {
                long
            };
            inst = self.insts.get((sub as usize).checked_sub(1)?)?;
        }
        let n = if inst.key.is_some() { 0 } else { note.round().clamp(0.0, 127.0) as usize };
        let layer = *inst.layer.get((vel * 127.0).round().clamp(0.0, 127.0) as usize)? as usize;
        let c = inst.cells.get(n * inst.layers + layer)?;
        if c.per == 0 || c.rr == 0 {
            return None;
        }
        let s = c.first as usize + (rr % c.rr as u32) as usize * c.per as usize;
        Some((self.picks.get(s..s + c.per as usize)?, inst.key, g))
    }

    /// A zone's frames.
    #[inline]
    fn frames(&self, z: &Zone) -> &[i16] {
        self.data.get(z.off..z.off + z.len as usize * z.chans as usize).unwrap_or(&[])
    }

    /// RMS over the first second of `id` at `note`, its loudest layer (both
    /// channels, as the synthesiser's `loudness` measures).
    fn rms_1s(&self, id: u16, note: f32) -> f32 {
        self.rms_over(id, note, 1.0)
    }

    /// RMS over the first `secs` of `id` at `note`, its loudest layer.
    fn rms_over(&self, id: u16, note: f32, secs: f32) -> f32 {
        let mut s = SAMPLER;
        if !s.start(self, id, note, 1.0, i64::MAX, 0) {
            return 0.0;
        }
        let pitch = s.key;
        let mut buf = [[0.0f32; 2]; 64];
        let mut sum = 0.0f64;
        let blocks = (secs * RATE) as usize / 64;
        for _ in 0..blocks {
            let live = s.fill(self, pitch, true, &mut buf);
            sum += buf.iter().map(|x| (x[0] * x[0] + x[1] * x[1]) as f64).sum::<f64>();
            if !live {
                break;
            }
        }
        (sum / (128 * blocks.max(1)) as f64).sqrt() as f32
    }

    /// Brings `id` to the reference (its loudest layer at `note`).
    fn normalise(&mut self, id: u16, note: f32, what: &str) -> bool {
        let rms = self.rms_1s(id, note);
        if !(rms > 1e-6) {
            self.warnings.push(format!("{what}: silent"));
            return false;
        }
        let k = REF_RMS / rms;
        let r = self.insts[id as usize - 1].zones.clone();
        for z in &mut self.zones[r.start as usize..r.end as usize] {
            z.gl *= k;
            z.gr *= k;
        }
        true
    }

    /// Adds an instrument made of `zones`, each covering keys and
    /// velocities (`covers`, in step). `nearest`: each note plays the zones
    /// whose root is nearest (ties: the lower), the ones of that root all
    /// round robins in their order; else (the SoundFont's) every zone whose
    /// range holds the note sounds (`MAXZ` at most), and a note none holds
    /// takes the nearest note's. Its id.
    fn add(&mut self, zones: Vec<Zone>, covers: &[Cover], nearest: bool, key: Option<u8>) -> u16 {
        let base = self.zones.len() as u32;
        self.zones.extend(zones);
        // layers: wherever a zone's velocities start or end
        let mut cuts: Vec<u8> = vec![0];
        for c in covers {
            cuts.push(c.vlo);
            if c.vhi < 127 {
                cuts.push(c.vhi + 1);
            }
        }
        cuts.sort_unstable();
        cuts.dedup();
        let layers = cuts.len();
        let mut layer = [0u8; 128];
        for (v, l) in layer.iter_mut().enumerate() {
            *l = (cuts.partition_point(|&c| c as usize <= v) - 1) as u8;
        }
        let notes: Vec<u8> = match key {
            Some(k) => vec![k],
            None => (0..128).collect(),
        };
        let mut cells = vec![Cell::default(); notes.len() * layers];
        let mut here: Vec<usize> = Vec::new();
        for (ni, &n) in notes.iter().enumerate() {
            for (li, &v) in cuts.iter().enumerate() {
                here.clear();
                here.extend(
                    (0..covers.len())
                        .filter(|&i| covers[i].vlo <= v && v <= covers[i].vhi)
                        .filter(|&i| nearest || (covers[i].klo <= n && n <= covers[i].khi)),
                );
                if here.is_empty() {
                    continue;
                }
                let first = self.picks.len() as u32;
                if nearest {
                    let d = |i: usize| ((covers[i].root as i32 - n as i32).abs(), covers[i].root);
                    let best = covers[here.iter().copied().min_by_key(|&i| d(i)).unwrap_or(here[0])].root;
                    here.retain(|&i| covers[i].root == best);
                    here.sort_by_key(|&i| covers[i].rr);
                    here.truncate(255);
                    self.picks.extend(here.iter().map(|&i| base + i as u32));
                    cells[ni * layers + li] = Cell { first, rr: here.len() as u8, per: 1 };
                } else {
                    here.truncate(MAXZ);
                    self.picks.extend(here.iter().map(|&i| base + i as u32));
                    cells[ni * layers + li] = Cell { first, rr: 1, per: here.len() as u8 };
                }
            }
        }
        // gaps: the nearest note's cell in the layer, then the nearest layer's
        let nn = notes.len();
        for li in 0..layers {
            let full: Vec<usize> = (0..nn).filter(|&ni| cells[ni * layers + li].per > 0).collect();
            if full.is_empty() {
                continue;
            }
            for ni in 0..nn {
                if cells[ni * layers + li].per == 0 {
                    let near = full.iter().copied().min_by_key(|&f| (f.abs_diff(ni), f)).unwrap_or(full[0]);
                    cells[ni * layers + li] = cells[near * layers + li];
                }
            }
        }
        for ni in 0..nn {
            let full: Vec<usize> = (0..layers).filter(|&li| cells[ni * layers + li].per > 0).collect();
            for li in 0..layers {
                if cells[ni * layers + li].per == 0 {
                    if let Some(&near) = full.iter().min_by_key(|&&f| (f.abs_diff(li), f)) {
                        cells[ni * layers + li] = cells[ni * layers + near];
                    }
                }
            }
        }
        self.insts.push(Inst { layer, layers, cells, key, auto: None, zones: base..self.zones.len() as u32 });
        self.insts.len() as u16
    }
}

/// A zone's keys, velocities, root and round robin, as the table is made.
#[derive(Clone, Copy)]
struct Cover {
    klo: u8,
    khi: u8,
    vlo: u8,
    vhi: u8,
    root: u8,
    rr: u32,
}

// ---------------------------------------------------------------- playing

/// What a voice playing samples keeps: its zones (a stereo pair, or
/// layers), their read positions, and their own envelope (the SoundFont's
/// hold, decay and sustain; 1 throughout for a WAV).
#[derive(Clone, Copy)]
pub struct Sampler {
    /// zones sounding (0: none, the synthesiser plays)
    pub n: u8,
    z: [u32; MAXZ],
    pos: [f64; MAXZ],
    live: [bool; MAXZ],
    env: f32,
    hold: u32,
    dec: f32,
    sus: f32,
    /// the note a kit's key sounds (its pitch), else the note started
    pub key: f32,
    /// the instrument's own release (seconds to -60 dB)
    pub rel: f32,
    /// the zones' exclusive class (0 none)
    pub excl: u8,
    /// whether looped zones loop (a drum's hit plays once through)
    pub loops: bool,
    /// a gain on the zones (an "auto" instrument's short articulation's)
    g: f32,
}

pub const SAMPLER: Sampler = Sampler {
    n: 0,
    z: [0; MAXZ],
    pos: [0.0; MAXZ],
    live: [false; MAXZ],
    env: 1.0,
    hold: 0,
    dec: 1.0,
    sus: 1.0,
    key: 60.0,
    rel: 0.1,
    excl: 0,
    loops: true,
    g: 1.0,
};

/// Frames `i` and `j` (mono or interleaved stereo) `fr` of the way between.
#[inline(always)]
fn lerp(d: &[i16], chans: u8, i: usize, j: usize, fr: f32) -> (f32, f32) {
    if chans == 2 {
        let (a, b) = (d[2 * i] as f32, d[2 * j] as f32);
        let (c, e) = (d[2 * i + 1] as f32, d[2 * j + 1] as f32);
        (a + (b - a) * fr, c + (e - c) * fr)
    } else {
        let (a, b) = (d[i] as f32, d[j] as f32);
        let m = a + (b - a) * fr;
        (m, m)
    }
}

/// Timecents to seconds.
fn tc(t: f32) -> f32 {
    if t <= -32000.0 { 0.0 } else { (t / 1200.0).exp2() }
}

impl Sampler {
    /// Starts `id` at `note` (a kit's key sounds whatever the note),
    /// velocity `vel` (0-1), `len` samples of gate (an "auto" instrument's
    /// articulation), round robin `rr`. False (nothing sounding) when the
    /// bank has nothing for it.
    pub fn start(&mut self, b: &Bank, id: u16, note: f32, vel: f32, len: i64, rr: u32) -> bool {
        self.n = 0;
        let Some((zs, key, g)) = b.pick(id, note, vel, len, rr) else {
            return false;
        };
        let n = zs.len().min(MAXZ);
        for (k, &z) in zs.iter().take(n).enumerate() {
            self.z[k] = z;
            self.pos[k] = 0.0;
            self.live[k] = true;
        }
        let Some(z0) = b.zones.get(zs[0] as usize) else {
            return false;
        };
        self.n = n as u8;
        self.key = key.map_or(note, |k| k as f32);
        // the hold and decay shorten up the keyboard as the zone says
        let k = self.key - 60.0;
        self.hold = (tc(z0.hold_tc - z0.key_hold * k).min(20.0) * RATE) as u32;
        let dec = tc(z0.dec_tc - z0.key_dec * k).clamp(0.001, 100.0);
        // -96 dB over the decay (FluidSynth's), down to the sustain
        self.dec = 10f32.powf(-4.8 / (dec * RATE));
        self.sus = z0.sus;
        self.env = 1.0;
        self.rel = z0.rel;
        self.excl = z0.excl;
        self.loops = true;
        self.g = g;
        true
    }

    /// The zones at `pitch` (a note, fractional: glides and vibrato) into
    /// `out` (overwritten); `gate`: still held (a loop until release keeps
    /// looping). False once they have all ended or died away.
    #[inline]
    pub fn fill(&mut self, b: &Bank, pitch: f32, gate: bool, out: &mut [[f32; 2]]) -> bool {
        out.fill([0.0; 2]);
        let mut any = false;
        for k in 0..self.n as usize {
            if !self.live[k] {
                continue;
            }
            let Some(z) = b.zones.get(self.z[k] as usize) else {
                self.live[k] = false;
                continue;
            };
            let d = b.frames(z);
            if d.len() < 2 * z.chans as usize {
                self.live[k] = false;
                continue;
            }
            let step = ((((pitch - z.root) * z.scale + z.tune) / 12.0).exp2() * z.rate).clamp(1e-3, 64.0) as f64;
            let looping = self.loops && (z.mode == 1 || (z.mode == 3 && gate));
            let (ls, le) = (z.ls as f64, z.le as f64);
            let end = (z.len - 1) as f64;
            let fade = 1.0 / (END_FADE * step);
            let (gl, gr) = (z.gl * self.g, z.gr * self.g);
            let xf = if looping { z.xf as f64 } else { 0.0 };
            let xs = le - xf;
            let mut pos = self.pos[k];
            let n = out.len();
            let mut at = 0;
            while at < n {
                // plain reading as far as it goes: short of the loop's end
                // (or its crossfade) or of the end's fade, a frame to spare
                let limit = if looping { if xf > 0.0 { xs } else { le } } else { end - END_FADE * step } - 1.0;
                if pos < limit {
                    let m = (((limit - pos) / step).ceil() as usize).clamp(1, n - at);
                    // in 32.32 fixed point from the frame under it
                    let base = pos.floor();
                    let b0 = base as usize;
                    let (mut fx, st) = (((pos - base) * FX) as u64, (step * FX) as u64);
                    if z.chans == 2 {
                        for o in out[at..at + m].iter_mut() {
                            let i = 2 * (b0 + (fx >> 32) as usize);
                            let fr = (fx as u32) as f32 * (1.0 / FX as f32);
                            let (a, b) = (d[i] as f32, d[i + 2] as f32);
                            let (c, e) = (d[i + 1] as f32, d[i + 3] as f32);
                            o[0] += (a + (b - a) * fr) * gl;
                            o[1] += (c + (e - c) * fr) * gr;
                            fx += st;
                        }
                    } else {
                        for o in out[at..at + m].iter_mut() {
                            let i = b0 + (fx >> 32) as usize;
                            let fr = (fx as u32) as f32 * (1.0 / FX as f32);
                            let (a, b) = (d[i] as f32, d[i + 1] as f32);
                            let x = a + (b - a) * fr;
                            o[0] += x * gl;
                            o[1] += x * gr;
                            fx += st;
                        }
                    }
                    pos = base + fx as f64 / FX;
                    at += m;
                    continue;
                }
                // a sample at a time: the loop's wrap and crossfade, the end's fade
                if looping {
                    if pos >= le {
                        pos = ls + (pos - le) % (le - ls);
                    }
                } else if pos >= end {
                    self.live[k] = false;
                    break;
                }
                let i = pos as usize;
                let fr = (pos - i as f64) as f32;
                let j = if looping && i + 1 >= z.le as usize { z.ls as usize } else { i + 1 };
                let (mut l, mut r) = lerp(d, z.chans, i, j, fr);
                if xf > 0.0 && pos >= xs {
                    // equal power, into the same stretch before the loop's start
                    let q = pos - (le - ls);
                    let qi = q as usize;
                    let (l2, r2) = lerp(d, z.chans, qi, qi + 1, (q - qi as f64) as f32);
                    let t = ((pos - xs) / xf) as f32;
                    let (a, b) = ((1.0 - t).sqrt(), t.sqrt());
                    l = l * a + l2 * b;
                    r = r * a + r2 * b;
                }
                if !looping {
                    let left = ((end - pos) * fade) as f32;
                    if left < 1.0 {
                        l *= left;
                        r *= left;
                    }
                }
                out[at][0] += l * gl;
                out[at][1] += r * gr;
                pos += step;
                at += 1;
            }
            self.pos[k] = pos;
            any |= self.live[k];
        }
        if self.hold == 0 && self.env == self.sus {
            if self.env != 1.0 {
                for o in out.iter_mut() {
                    o[0] *= self.env;
                    o[1] *= self.env;
                }
            }
        } else {
            // held, then decaying to the sustain: a ramp to where the block ends
            let n = out.len();
            let h = (self.hold as usize).min(n);
            self.hold -= h as u32;
            let e0 = self.env;
            let e1 = if n > h { (e0 * self.dec.powi((n - h) as i32)).max(self.sus) } else { e0 };
            let de = (e1 - e0) / n.max(1) as f32;
            let mut e = e0;
            for o in out.iter_mut() {
                e += de;
                o[0] *= e;
                o[1] *= e;
            }
            self.env = e1;
        }
        any && self.env >= 1e-3
    }
}

// ---------------------------------------------------------------- WAV

/// A WAV's frames (16-bit PCM, mono or stereo, interleaved), its rate and
/// channels. Every chunk is bounds-checked; anything else is an error.
fn read_wav(b: &[u8]) -> Result<(Vec<i16>, u32, u8), String> {
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("not a RIFF WAVE file".into());
    }
    let mut fmt: Option<(u16, u16, u32, u16, u16)> = None;
    let mut data: Option<&[u8]> = None;
    let mut p = 12usize;
    while p + 8 <= b.len() {
        let id = &b[p..p + 4];
        let size = u32::from_le_bytes([b[p + 4], b[p + 5], b[p + 6], b[p + 7]]) as usize;
        let body = p + 8;
        let end = body.checked_add(size).filter(|&e| e <= b.len()).ok_or("a chunk runs past the end")?;
        let c = &b[body..end];
        match id {
            b"fmt " => {
                if c.len() < 16 {
                    return Err("fmt chunk too short".into());
                }
                let u16_at = |i: usize| u16::from_le_bytes([c[i], c[i + 1]]);
                let mut tag = u16_at(0);
                if tag == 0xFFFE {
                    // WAVE_FORMAT_EXTENSIBLE: the subformat's first two bytes
                    if c.len() < 26 {
                        return Err("extensible fmt chunk too short".into());
                    }
                    tag = u16_at(24);
                }
                let rate = u32::from_le_bytes([c[4], c[5], c[6], c[7]]);
                fmt = Some((tag, u16_at(2), rate, u16_at(12), u16_at(14)));
            }
            b"data" => data = Some(c),
            _ => {}
        }
        p = end + (size & 1);
    }
    let (tag, chans, rate, align, bits) = fmt.ok_or("no fmt chunk")?;
    let data = data.ok_or("no data chunk")?;
    if tag != 1 || bits != 16 {
        return Err(format!("format {tag}, {bits} bits: only 16-bit PCM"));
    }
    if !(1..=2).contains(&chans) || align != 2 * chans {
        return Err(format!("{chans} channels, block {align}: only mono or stereo"));
    }
    if !(4000..=192_000).contains(&rate) {
        return Err(format!("rate {rate}"));
    }
    let frames = data.len() / align as usize;
    if frames < 2 {
        return Err("no audio".into());
    }
    let v = data[..frames * align as usize].chunks_exact(2).map(|s| i16::from_le_bytes([s[0], s[1]])).collect();
    Ok((v, rate, chans as u8))
}

// ---------------------------------------------------------------- manifest

/// One manifest row.
struct Row {
    instrument: String,
    articulation: String,
    root: u8,
    vlo: u8,
    vhi: u8,
    rr: u32,
    gain_db: f32,
    file: String,
}

fn parse_row(f: &[&str]) -> Result<Row, String> {
    if f.len() != 8 {
        return Err(format!("{} fields, not 8", f.len()));
    }
    let num = |s: &str, what: &str| s.trim().parse::<u32>().map_err(|_| format!("{what} {s:?}"));
    let (root, vlo, vhi) = (num(f[2], "root")?, num(f[3], "vel_lo")?, num(f[4], "vel_hi")?);
    if root > 127 || vhi > 127 || vlo > vhi {
        return Err(format!("root {root}, velocities {vlo}..{vhi}"));
    }
    let gain_db = f[6].trim().parse::<f32>().map_err(|_| format!("gain_db {:?}", f[6]))?;
    if !gain_db.is_finite() || gain_db.abs() > 96.0 {
        return Err(format!("gain_db {gain_db}"));
    }
    let file = f[7].trim();
    // inside the samples directory, nowhere else
    if file.is_empty() || !Path::new(file).components().all(|c| matches!(c, Component::Normal(_))) {
        return Err(format!("file {file:?}"));
    }
    let (instrument, articulation) = (f[0].trim(), f[1].trim());
    if instrument.is_empty() || articulation.is_empty() || articulation == "auto" {
        return Err("no instrument or articulation".into());
    }
    Ok(Row {
        instrument: instrument.into(),
        articulation: articulation.into(),
        root: root as u8,
        vlo: vlo as u8,
        vhi: vhi as u8,
        rr: num(f[5], "rr")?,
        gain_db,
        file: file.into(),
    })
}

/// The manifest's instruments, and an "auto" for each: short notes (gates
/// under `AUTO_SHORT`) the staccato (or pizzicato, when that is all there
/// is), long ones the vibrato (else the sustain); either, failing that,
/// whatever there is. The short articulation is brought to the long one's
/// level (its RMS over the first `AUTO_MATCH` to the long one's over the
/// first second, at the long one's middle root, loudest layer): the
/// manifest levels each by its first second, which makes a short note hot.
/// One root does for all, as the manifest evens each articulation's roots.
fn load_multi(b: &mut Bank, dir: &Path) -> Result<(), String> {
    let path = dir.join(MANIFEST);
    let text = read_file(&path, 16 << 20)?;
    let text = String::from_utf8(text).map_err(|_| format!("{}: not UTF-8", path.display()))?;
    let mut rows: Vec<Row> = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f[0].trim() == "instrument" {
            continue;
        }
        match parse_row(&f) {
            Ok(r) => rows.push(r),
            Err(e) => b.warnings.push(format!("{MANIFEST} line {}: {e}", ln + 1)),
        }
    }
    let mut groups: Vec<(String, String)> = Vec::new();
    for r in &rows {
        if !groups.iter().any(|g| g.0 == r.instrument && g.1 == r.articulation) {
            groups.push((r.instrument.clone(), r.articulation.clone()));
        }
    }
    // each file read once, however many rows name it
    let mut wavs: HashMap<String, Option<(usize, u32, u8, u32)>> = HashMap::new();
    for (inst, art) in groups {
        let (mut zones, mut covers) = (Vec::new(), Vec::new());
        for r in rows.iter().filter(|r| r.instrument == inst && r.articulation == art) {
            let w = match wavs.get(&r.file) {
                Some(w) => *w,
                None => {
                    let w = match read_file(&dir.join(&r.file), 256 << 20).and_then(|v| read_wav(&v)) {
                        Ok((v, rate, chans)) => {
                            let off = b.data.len();
                            b.data.extend_from_slice(&v);
                            Some((off, (v.len() / chans as usize) as u32, chans, rate))
                        }
                        Err(e) => {
                            b.warnings.push(format!("{}: {e}", r.file));
                            None
                        }
                    };
                    wavs.insert(r.file.clone(), w);
                    w
                }
            };
            let Some((off, len, chans, rate)) = w else { continue };
            let g = 10f32.powf(r.gain_db / 20.0) / 32768.0;
            // a held note outlasting a sustained recording (5.5 s, its last
            // 0.5 s faded): its steady middle looped, crossfaded
            let sec = |t: f32| (t * rate as f32) as u32;
            let held = matches!(r.articulation.as_str(), "sus" | "vib") && len >= sec(LOOP_MIN);
            let (ls, le) = if held { (sec(LOOP.0), sec(LOOP.1)) } else { (0, 0) };
            zones.push(Zone {
                off,
                len,
                chans,
                root: r.root as f32,
                tune: 0.0,
                scale: 1.0,
                rate: rate as f32 / RATE,
                ls,
                le,
                mode: if held { 3 } else { 0 },
                xf: if held { sec(LOOP_XF) } else { 0 },
                gl: g,
                gr: g,
                excl: 0,
                hold_tc: -32768.0,
                dec_tc: -32768.0,
                key_hold: 0.0,
                key_dec: 0.0,
                sus: 1.0,
                rel: 0.1,
            });
            covers.push(Cover { klo: 0, khi: 127, vlo: r.vlo, vhi: r.vhi, root: r.root, rr: r.rr });
        }
        if zones.is_empty() {
            continue;
        }
        let id = b.add(zones, &covers, true, None);
        b.multis.push((inst, art, id));
    }
    // "auto" for each instrument
    let mut names: Vec<String> = Vec::new();
    for m in &b.multis {
        if !names.contains(&m.0) {
            names.push(m.0.clone());
        }
    }
    for name in names {
        let find = |arts: &[&str]| arts.iter().find_map(|a| b.multi(&name, a));
        let any = b.multis.iter().find(|m| m.0 == name).map(|m| m.2);
        let long = find(&["vib", "sus"]).or(find(&["stac", "pizz"])).or(any);
        let short = find(&["stac", "pizz"]).or(long);
        if let (Some(s), Some(l)) = (short, long) {
            let mut g = 1.0;
            if s != l {
                let r = b.insts[l as usize - 1].zones.clone();
                let mut roots: Vec<f32> = b.zones[r.start as usize..r.end as usize].iter().map(|z| z.root).collect();
                roots.sort_by(f32::total_cmp);
                let mid = roots[roots.len() / 2];
                let (rs, rl) = (b.rms_over(s, mid, AUTO_MATCH), b.rms_1s(l, mid));
                if rs > 1e-6 && rl > 1e-6 {
                    g = (rl / rs).clamp(0.25, 4.0);
                }
            }
            b.insts.push(Inst {
                layer: [0; 128],
                layers: 0,
                cells: Vec::new(),
                key: None,
                auto: Some((s, l, g)),
                zones: 0..0,
            });
            b.multis.push((name, "auto".into(), b.insts.len() as u16));
        }
    }
    Ok(())
}

/// A file's bytes, refused past `max` bytes.
fn read_file(path: &Path, max: u64) -> Result<Vec<u8>, String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let len = f.metadata().map_err(|e| format!("{}: {e}", path.display()))?.len();
    if len > max {
        return Err(format!("{}: {len} bytes", path.display()));
    }
    let mut v = Vec::with_capacity(len as usize);
    f.read_to_end(&mut v).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(v)
}

// ---------------------------------------------------------------- SoundFont

/// A generator as stored: its operator and amount.
type Gen = (u16, [u8; 2]);

/// Generators: a zone's values (as stored) and which it sets.
#[derive(Clone, Copy)]
struct Gens {
    v: [i16; 61],
    set: u64,
}

const G_START: usize = 0;
const G_END: usize = 1;
const G_LOOP_START: usize = 2;
const G_LOOP_END: usize = 3;
const G_START_COARSE: usize = 4;
const G_END_COARSE: usize = 12;
const G_PAN: usize = 17;
const G_HOLD: usize = 35;
const G_DECAY: usize = 36;
const G_SUSTAIN: usize = 37;
const G_RELEASE: usize = 38;
const G_KEY_HOLD: usize = 39;
const G_KEY_DECAY: usize = 40;
const G_INSTRUMENT: u16 = 41;
const G_KEYS: usize = 43;
const G_VELS: usize = 44;
const G_LOOP_START_COARSE: usize = 45;
const G_ATTEN: usize = 48;
const G_LOOP_END_COARSE: usize = 50;
const G_COARSE: usize = 51;
const G_FINE: usize = 52;
const G_SAMPLE: u16 = 53;
const G_MODES: usize = 54;
const G_SCALE: usize = 56;
const G_EXCL: usize = 57;
const G_ROOT: usize = 58;

impl Gens {
    /// None set: a preset zone's (its values add to the instrument's).
    fn none() -> Gens {
        Gens { v: [0; 61], set: 0 }
    }

    /// The SoundFont's defaults: an instrument zone's before its own.
    fn defaults() -> Gens {
        let mut g = Gens::none();
        g.v[G_KEYS] = 0x7F00;
        g.v[G_VELS] = 0x7F00;
        g.v[G_ROOT] = -1;
        g.v[G_SCALE] = 100;
        for t in [33, 34, G_HOLD, G_DECAY, G_RELEASE] {
            g.v[t] = -12000;
        }
        g
    }

    fn put(&mut self, (op, amt): Gen) {
        if (op as usize) < 61 {
            self.v[op as usize] = i16::from_le_bytes(amt);
            self.set |= 1 << op;
        }
    }

    fn has(&self, op: usize) -> bool {
        self.set & (1 << op) != 0
    }

    fn range(&self, op: usize) -> (u8, u8) {
        let a = self.v[op] as u16;
        ((a & 0xFF) as u8, (a >> 8) as u8)
    }

    fn i(&self, op: usize) -> i32 {
        self.v[op] as i32
    }
}

struct Shdr {
    start: u32,
    end: u32,
    ls: u32,
    le: u32,
    rate: u32,
    pitch: u8,
    corr: i8,
    kind: u16,
}

/// The SoundFont's articulation data (pdta), and where its samples are.
struct Sf2 {
    file: File,
    /// the smpl chunk: offset in the file, frames
    smpl: (u64, u64),
    phdr: Vec<(String, u16, u16, u16)>,
    pbag: Vec<u16>,
    pgen: Vec<Gen>,
    inst: Vec<u16>,
    ibag: Vec<u16>,
    igen: Vec<Gen>,
    shdr: Vec<Shdr>,
}

/// A zone as the SoundFont resolves it (instrument and preset generators
/// combined), before its sample is loaded.
struct Raw {
    sample: usize,
    klo: u8,
    khi: u8,
    vlo: u8,
    vhi: u8,
    root: u8,
    tune: f32,
    scale: f32,
    mode: u8,
    /// address offsets: start, end, loop start, loop end (frames)
    offs: [i64; 4],
    atten: i32,
    pan: i32,
    excl: u8,
    hold: i32,
    decay: i32,
    key_hold: i32,
    key_decay: i32,
    sus: i32,
    rel: i32,
}

fn read_at(f: &mut File, len_file: u64, off: u64, len: usize) -> Result<Vec<u8>, String> {
    if off.checked_add(len as u64).is_none_or(|e| e > len_file) {
        return Err("a chunk runs past the end".into());
    }
    f.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
    let mut v = vec![0u8; len];
    f.read_exact(&mut v).map_err(|e| e.to_string())?;
    Ok(v)
}

/// A chunk: its id and body.
type Chunk<'a> = ([u8; 4], &'a [u8]);

/// The sub-chunks of a chunk's body, bounds-checked.
fn sub_chunks(b: &[u8]) -> Result<Vec<Chunk<'_>>, String> {
    let mut v = Vec::new();
    let mut p = 0usize;
    while p + 8 <= b.len() {
        let id = [b[p], b[p + 1], b[p + 2], b[p + 3]];
        let size = u32::from_le_bytes([b[p + 4], b[p + 5], b[p + 6], b[p + 7]]) as usize;
        let end = (p + 8).checked_add(size).filter(|&e| e <= b.len()).ok_or("a sub-chunk runs past the end")?;
        v.push((id, &b[p + 8..end]));
        p = end + (size & 1);
    }
    Ok(v)
}

/// A NUL-padded name.
fn name(b: &[u8]) -> String {
    let n = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    b[..n].iter().map(|&c| c as char).collect::<String>().trim().to_string()
}

impl Sf2 {
    fn open(path: &Path) -> Result<Sf2, String> {
        let mut f = File::open(path).map_err(|e| e.to_string())?;
        let flen = f.metadata().map_err(|e| e.to_string())?.len();
        let h = read_at(&mut f, flen, 0, 12)?;
        if &h[0..4] != b"RIFF" || &h[8..12] != b"sfbk" {
            return Err("not a SoundFont".into());
        }
        let riff_end = (8 + u32::from_le_bytes([h[4], h[5], h[6], h[7]]) as u64).min(flen);
        let (mut smpl, mut pdta) = (None, None);
        let mut p = 12u64;
        while p + 8 <= riff_end {
            let h = read_at(&mut f, flen, p, 8)?;
            let size = u32::from_le_bytes([h[4], h[5], h[6], h[7]]) as u64;
            let (body, end) = (p + 8, p + 8 + size);
            if end > riff_end {
                return Err("a chunk runs past the end".into());
            }
            if &h[0..4] == b"LIST" && size >= 4 {
                match &read_at(&mut f, flen, body, 4)?[..] {
                    b"sdta" => {
                        let mut q = body + 4;
                        while q + 8 <= end {
                            let s = read_at(&mut f, flen, q, 8)?;
                            let n = u32::from_le_bytes([s[4], s[5], s[6], s[7]]) as u64;
                            if q + 8 + n > end {
                                return Err("smpl runs past the end".into());
                            }
                            if &s[0..4] == b"smpl" {
                                smpl = Some((q + 8, n / 2));
                            }
                            q += 8 + n + (n & 1);
                        }
                    }
                    b"pdta" => {
                        if size > 64 << 20 {
                            return Err(format!("pdta of {size} bytes"));
                        }
                        pdta = Some(read_at(&mut f, flen, body + 4, size as usize - 4)?);
                    }
                    _ => {}
                }
            }
            p = end + (size & 1);
        }
        let smpl = smpl.ok_or("no smpl chunk")?;
        let pdta = pdta.ok_or("no pdta chunk")?;
        let subs = sub_chunks(&pdta)?;
        let get = |id: &[u8; 4], rec: usize, min: usize| -> Result<Vec<&[u8]>, String> {
            let c =
                subs.iter().find(|s| &s.0 == id).ok_or_else(|| format!("no {} chunk", String::from_utf8_lossy(id)))?.1;
            if c.len() % rec != 0 || c.len() / rec < min {
                return Err(format!("{} chunk of {} bytes", String::from_utf8_lossy(id), c.len()));
            }
            Ok(c.chunks_exact(rec).collect())
        };
        let u16_at = |r: &[u8], i: usize| u16::from_le_bytes([r[i], r[i + 1]]);
        let u32_at = |r: &[u8], i: usize| u32::from_le_bytes([r[i], r[i + 1], r[i + 2], r[i + 3]]);
        let gens = |r: Vec<&[u8]>| r.iter().map(|g| (u16_at(g, 0), [g[2], g[3]])).collect();
        Ok(Sf2 {
            phdr: get(b"phdr", 38, 2)?
                .iter()
                .map(|r| (name(&r[..20]), u16_at(r, 20), u16_at(r, 22), u16_at(r, 24)))
                .collect(),
            pbag: get(b"pbag", 4, 1)?.iter().map(|r| u16_at(r, 0)).collect(),
            pgen: gens(get(b"pgen", 4, 1)?),
            inst: get(b"inst", 22, 2)?.iter().map(|r| u16_at(r, 20)).collect(),
            ibag: get(b"ibag", 4, 1)?.iter().map(|r| u16_at(r, 0)).collect(),
            igen: gens(get(b"igen", 4, 1)?),
            shdr: get(b"shdr", 46, 1)?
                .iter()
                .map(|r| Shdr {
                    start: u32_at(r, 20),
                    end: u32_at(r, 24),
                    ls: u32_at(r, 28),
                    le: u32_at(r, 32),
                    rate: u32_at(r, 36),
                    pitch: r[40],
                    corr: r[41] as i8,
                    kind: u16_at(r, 44),
                })
                .collect(),
            file: f,
            smpl,
        })
    }

    /// Zones `a..b` of a bag list: each one's generators.
    fn zones<'a>(bags: &[u16], gens: &'a [Gen], a: usize, b: usize) -> Result<Vec<&'a [Gen]>, String> {
        (a..b)
            .map(|z| {
                let (g0, g1) =
                    (*bags.get(z).ok_or("bag index")? as usize, *bags.get(z + 1).ok_or("bag index")? as usize);
                gens.get(g0..g1).ok_or_else(|| "generator index".to_string())
            })
            .collect()
    }

    /// Preset `pi`'s zones, resolved: instrument generators absolute (a
    /// global zone's under each zone's own), the preset's added to them
    /// (its global zone's under each zone's own), ranges intersected.
    fn resolve(&self, pi: usize) -> Result<Vec<Raw>, String> {
        let (a, b) = (self.phdr[pi].3 as usize, self.phdr.get(pi + 1).ok_or("preset index")?.3 as usize);
        let mut out = Vec::new();
        let mut pglobal = Gens::none();
        for (k, z) in Sf2::zones(&self.pbag, &self.pgen, a, b)?.iter().enumerate() {
            if z.last().map(|g| g.0) != Some(G_INSTRUMENT) {
                if k == 0 {
                    z.iter().for_each(|&g| pglobal.put(g));
                }
                continue;
            }
            let mut pg = pglobal;
            z.iter().for_each(|&g| pg.put(g));
            let ii = pg.v[G_INSTRUMENT as usize] as u16 as usize;
            let (ia, ib) = (
                *self.inst.get(ii).ok_or("instrument index")? as usize,
                *self.inst.get(ii + 1).ok_or("instrument index")? as usize,
            );
            let mut iglobal = Gens::defaults();
            for (m, iz) in Sf2::zones(&self.ibag, &self.igen, ia, ib)?.iter().enumerate() {
                if iz.last().map(|g| g.0) != Some(G_SAMPLE) {
                    if m == 0 {
                        iz.iter().for_each(|&g| iglobal.put(g));
                    }
                    continue;
                }
                let mut ig = iglobal;
                iz.iter().for_each(|&g| ig.put(g));
                let full = (0u8, 127u8);
                let both = |op: usize| {
                    let (i, p) = (ig.range(op), if pg.has(op) { pg.range(op) } else { full });
                    (i.0.max(p.0), i.1.min(p.1).min(127))
                };
                let ((klo, khi), (vlo, vhi)) = (both(G_KEYS), both(G_VELS));
                if klo > khi || vlo > vhi {
                    continue;
                }
                let sample = ig.v[G_SAMPLE as usize] as u16 as usize;
                let Some(sh) = self.shdr.get(sample) else { continue };
                let add = |op: usize| ig.i(op) + pg.i(op);
                let root = match ig.i(G_ROOT) {
                    r @ 0..=127 => r as u8,
                    _ if sh.pitch <= 127 => sh.pitch,
                    _ => 60,
                };
                let addr = |fine: usize, coarse: usize| ig.i(fine) as i64 + 32768 * ig.i(coarse) as i64;
                out.push(Raw {
                    sample,
                    klo,
                    khi,
                    vlo,
                    vhi,
                    root,
                    tune: add(G_COARSE) as f32 + (add(G_FINE) as f32 + sh.corr as f32) / 100.0,
                    scale: add(G_SCALE) as f32 / 100.0,
                    mode: (ig.i(G_MODES) & 3) as u8,
                    offs: [
                        addr(G_START, G_START_COARSE),
                        addr(G_END, G_END_COARSE),
                        addr(G_LOOP_START, G_LOOP_START_COARSE),
                        addr(G_LOOP_END, G_LOOP_END_COARSE),
                    ],
                    atten: add(G_ATTEN),
                    pan: add(G_PAN),
                    excl: ig.i(G_EXCL).clamp(0, 255) as u8,
                    hold: add(G_HOLD),
                    decay: add(G_DECAY),
                    key_hold: add(G_KEY_HOLD),
                    key_decay: add(G_KEY_DECAY),
                    sus: add(G_SUSTAIN),
                    rel: add(G_RELEASE),
                });
            }
        }
        Ok(out)
    }

    /// A preset by name: in bank 128 for a kit, else bank 0 (or the lowest
    /// melodic bank that has it).
    fn find(&self, preset: &str, kit: bool) -> Option<usize> {
        let n = self.phdr.len() - 1;
        (0..n).filter(|&i| self.phdr[i].0 == preset && (self.phdr[i].2 == 128) == kit).min_by_key(|&i| self.phdr[i].2)
    }
}

/// The presets of a SoundFont: bank, program, name.
pub fn sf2_presets(path: &Path) -> Result<Vec<(u16, u16, String)>, String> {
    let s = Sf2::open(path)?;
    let mut v: Vec<(u16, u16, String)> = s.phdr[..s.phdr.len() - 1].iter().map(|p| (p.2, p.1, p.0.clone())).collect();
    v.sort();
    Ok(v)
}

/// Loads `melodic` presets and `kits` from a SoundFont into `b`: only the
/// samples they use, read from the file one by one.
fn load_sf2(b: &mut Bank, path: &Path, melodic: &[&str], kits: &[&str]) -> Result<(), String> {
    let mut s = Sf2::open(path)?;
    let mut want: Vec<(String, bool, Vec<Raw>)> = Vec::new();
    for (names, kit) in [(melodic, false), (kits, true)] {
        for &p in names {
            match s.find(p, kit).map(|i| s.resolve(i)) {
                Some(Ok(z)) if !z.is_empty() => want.push((p.to_string(), kit, z)),
                Some(Ok(_)) => b.warnings.push(format!("{p}: no zones")),
                Some(Err(e)) => b.warnings.push(format!("{p}: {e}")),
                None => b.warnings.push(format!("{p}: not in the SoundFont")),
            }
        }
    }
    // each sample read once: where it went in the data, its frames
    let mut loaded: HashMap<usize, Option<(usize, u32)>> = HashMap::new();
    let flen = s.file.metadata().map_err(|e| e.to_string())?.len();
    for (_, _, raws) in &want {
        for r in raws {
            if loaded.contains_key(&r.sample) {
                continue;
            }
            let sh = &s.shdr[r.sample];
            let ok =
                sh.kind & 0x8000 == 0 && sh.start < sh.end && sh.end as u64 <= s.smpl.1 && sh.end - sh.start <= 1 << 25;
            let got = if ok {
                let n = (sh.end - sh.start) as usize;
                read_at(&mut s.file, flen, s.smpl.0 + 2 * sh.start as u64, 2 * n).ok().map(|v| {
                    let off = b.data.len();
                    b.data.extend(v.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])));
                    (off, n as u32)
                })
            } else {
                None
            };
            loaded.insert(r.sample, got);
        }
    }
    let zone = |r: &Raw, s: &Sf2| -> Option<Zone> {
        let (off, slen) = loaded.get(&r.sample).copied().flatten()?;
        let sh = &s.shdr[r.sample];
        let slen = slen as i64;
        if slen < 4 {
            return None;
        }
        let start = r.offs[0].clamp(0, slen - 2);
        let end = (slen + r.offs[1]).clamp(start + 2, slen);
        let ls = sh.ls as i64 - sh.start as i64 + r.offs[2] - start;
        let le = sh.le as i64 - sh.start as i64 + r.offs[3] - start;
        let len = end - start;
        let looped = (r.mode == 1 || r.mode == 3) && ls >= 0 && ls + 2 <= le && le <= len;
        // FluidSynth's (and the EMU hardware's) attenuation: 0.4 of a centibel a unit
        let g = 10f32.powf(-(r.atten.clamp(0, 1440) as f32 * 0.4) / 200.0) / 32768.0;
        let pan = (r.pan as f32 / 500.0).clamp(-1.0, 1.0);
        Some(Zone {
            off: off + start as usize,
            len: len as u32,
            chans: 1,
            root: r.root as f32,
            tune: r.tune,
            scale: r.scale,
            rate: sh.rate.clamp(4000, 192_000) as f32 / RATE,
            ls: if looped { ls as u32 } else { 0 },
            le: if looped { le as u32 } else { 0 },
            mode: if looped { r.mode } else { 0 },
            xf: 0,
            gl: g * (1.0 - pan).min(1.0),
            gr: g * (1.0 + pan).min(1.0),
            excl: r.excl,
            hold_tc: r.hold as f32,
            dec_tc: r.decay as f32,
            key_hold: r.key_hold as f32,
            key_dec: r.key_decay as f32,
            sus: if r.sus >= 1000 { 0.0 } else { 10f32.powf(-(r.sus.max(0) as f32) / 200.0) },
            // -96 dB over the release (FluidSynth's): to -60 dB in 0.625 of it
            rel: (tc(r.rel as f32) * 0.625).clamp(0.003, 10.0),
        })
    };
    let cover = |r: &Raw| Cover { klo: r.klo, khi: r.khi, vlo: r.vlo, vhi: r.vhi, root: r.root, rr: 0 };
    for (p, kit, raws) in &want {
        if !kit {
            let (zones, covers): (Vec<Zone>, Vec<Cover>) =
                raws.iter().filter_map(|r| Some((zone(r, &s)?, cover(r)))).unzip();
            if zones.is_empty() {
                b.warnings.push(format!("{p}: no samples"));
                continue;
            }
            // the middle note: 60 if the preset has it, else its range's middle
            let mid = if covers.iter().any(|c| c.klo <= 60 && 60 <= c.khi) {
                60.0
            } else {
                let lo = covers.iter().map(|c| c.klo).min().unwrap_or(60) as f32;
                let hi = covers.iter().map(|c| c.khi).max().unwrap_or(60) as f32;
                ((lo + hi) / 2.0).round()
            };
            let id = b.add(zones, &covers, false, None);
            if b.normalise(id, mid, p) {
                b.gms.push((p.clone(), id));
            }
        } else {
            // each key its own instrument
            let mut keys = [0u16; 128];
            let first = b.zones.len();
            for (key, slot) in keys.iter_mut().enumerate() {
                let k = key as u8;
                let (zones, covers): (Vec<Zone>, Vec<Cover>) = raws
                    .iter()
                    .filter(|r| r.klo <= k && k <= r.khi)
                    .filter_map(|r| Some((zone(r, &s)?, cover(r))))
                    .unzip();
                if !zones.is_empty() {
                    *slot = b.add(zones, &covers, false, Some(k));
                }
            }
            // the kit as its maker balanced it, brought to the reference at
            // its snare (38; else the kick, else its first key)
            let Some(at) = [38, 36].into_iter().chain(0..128).find(|&k| keys[k] != 0) else {
                b.warnings.push(format!("{p}: no samples"));
                continue;
            };
            let rms = b.rms_1s(keys[at], at as f32);
            if !(rms > 1e-6) {
                b.warnings.push(format!("{p}: silent"));
                continue;
            }
            for z in &mut b.zones[first..] {
                z.gl *= REF_RMS / rms;
                z.gr *= REF_RMS / rms;
            }
            b.kits.push((p.clone(), keys));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- loading

/// Loads the samples directory: the manifest's WAVs and the SoundFont's
/// `GM_PRESETS` and `GM_KITS`. Either may be missing (a warning); an error
/// when neither loads.
pub fn load(dir: &Path) -> Result<Bank, String> {
    if !dir.is_dir() {
        return Err(format!("no directory {}", dir.display()));
    }
    let mut b = Bank::new();
    let (m, s) = (dir.join(MANIFEST), dir.join(SF2));
    let mut any = false;
    if m.is_file() {
        match load_multi(&mut b, dir) {
            Ok(()) => any = true,
            Err(e) => b.warnings.push(e),
        }
    } else {
        b.warnings.push(format!("no {MANIFEST}"));
    }
    if s.is_file() {
        match load_sf2(&mut b, &s, GM_PRESETS, GM_KITS) {
            Ok(()) => any = true,
            Err(e) => b.warnings.push(format!("{SF2}: {e}")),
        }
    } else {
        b.warnings.push(format!("no {SF2}"));
    }
    if !any || b.insts.is_empty() {
        return Err(format!("nothing loaded from {} ({})", dir.display(), b.warnings.join("; ")));
    }
    Ok(b)
}

static BANK: OnceLock<Bank> = OnceLock::new();
static CONFIG: OnceLock<(PathBuf, bool)> = OnceLock::new();
static LOAD: Once = Once::new();
static SPAWNED: AtomicBool = AtomicBool::new(false);

/// Makes `bank` the one the music plays (the first installed stays).
pub fn install(bank: Bank) {
    let _ = BANK.set(bank);
}

/// Whether samples are installed.
pub fn ready() -> bool {
    BANK.get().is_some()
}

/// The installed bank (an atomic load: the audio thread may ask).
#[inline]
pub fn bank() -> Option<&'static Bank> {
    BANK.get()
}

/// `Bank::multi` of the installed bank (None: no samples).
#[allow(dead_code)] // the genres' hook
pub fn multi(instrument: &str, articulation: &str) -> Option<u16> {
    bank()?.multi(instrument, articulation)
}

/// `Bank::gm` of the installed bank.
#[allow(dead_code)] // the genres' hook
pub fn gm(preset: &str) -> Option<u16> {
    bank()?.gm(preset)
}

/// `Bank::drum` of the installed bank.
#[allow(dead_code)] // the genres' hook
pub fn drum(kit: &str, key: u8) -> Option<u16> {
    bank()?.drum(kit, key)
}

/// `--samples DIR` (None: `~/.local/share/fleece/samples`) and
/// `--no-samples`, before anything loads.
pub fn configure(dir: Option<String>, off: bool) {
    let dir = dir.map(PathBuf::from).unwrap_or_else(default_dir);
    let _ = CONFIG.set((dir, off));
}

fn default_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    Path::new(&home).join(".local/share/fleece/samples")
}

fn config() -> &'static (PathBuf, bool) {
    CONFIG.get_or_init(|| (default_dir(), false))
}

/// Loads and installs the samples now (once; later calls wait for it):
/// for `--musicwav` and a video's soundtrack, so a seed gives the same
/// music. A missing directory is one line on stderr, and the synthesiser.
pub fn load_now() -> bool {
    LOAD.call_once(|| {
        let (dir, off) = config();
        if *off {
            return;
        }
        if !dir.is_dir() {
            eprintln!(
                "fleece: no samples in {} (tools/get-samples.sh fetches them): the music is all synthesiser",
                dir.display()
            );
            return;
        }
        let t0 = Instant::now();
        match load(dir) {
            Ok(b) => {
                let (m, g, k) = b.counts();
                eprintln!(
                    "fleece: samples: {m} multisampled instruments, {g} presets, {k} kits ({:.0} MB) in {:.1} s{}",
                    b.bytes() as f64 / 1e6,
                    t0.elapsed().as_secs_f64(),
                    if b.warnings.is_empty() {
                        String::new()
                    } else {
                        format!("; {} skipped (--samples-list)", b.warnings.len())
                    }
                );
                install(b);
            }
            Err(e) => eprintln!("fleece: samples: {e}: the music is all synthesiser"),
        }
    });
    ready()
}

/// Loads them on a thread of their own (the live music: never in the audio
/// callback); pieces begun once `ready()` use them.
pub fn load_in_background() {
    if config().1 || ready() || SPAWNED.swap(true, Ordering::Relaxed) {
        return;
    }
    let _ = std::thread::Builder::new().name("samples".into()).spawn(|| {
        load_now();
    });
}

/// Whether loading is over (in vain or not), or never asked for.
pub fn settled() -> bool {
    LOAD.is_completed() || !SPAWNED.load(Ordering::Relaxed)
}

/// `--samples-list`: the SoundFont's presets, then what loads, and what not.
pub fn list_cli() {
    let (dir, _) = config();
    println!("samples directory: {}", dir.display());
    match sf2_presets(&dir.join(SF2)) {
        Ok(v) => {
            println!("{SF2}: {} presets (bank program name):", v.len());
            for (bank, prog, name) in v {
                println!("  {bank:3} {prog:3} {name}");
            }
        }
        Err(e) => println!("{SF2}: {e}"),
    }
    let t0 = Instant::now();
    match load(dir) {
        Ok(b) => {
            println!("loaded in {:.2} s, {:.1} MB", t0.elapsed().as_secs_f64(), b.bytes() as f64 / 1e6);
            for (i, a, id) in &b.multis {
                let inst = &b.insts[*id as usize - 1];
                let (z, l) = (inst.zones.end - inst.zones.start, inst.layers);
                if let Some((s, l, g)) = inst.auto {
                    let art = |id: u16| b.multis.iter().find(|m| m.2 == id).map_or("?", |m| m.1.as_str());
                    println!(
                        "  multi {i:12} {a:18} gates under {AUTO_SHORT} s {} ({:+.1} dB), else {}",
                        art(s),
                        20.0 * g.log10(),
                        art(l)
                    );
                } else {
                    println!("  multi {i:12} {a:18} {z} samples, {l} velocity layers");
                }
            }
            for (p, id) in &b.gms {
                let inst = &b.insts[*id as usize - 1];
                println!("  gm    {p:20} {} zones, {} velocity layers", inst.zones.end - inst.zones.start, inst.layers);
            }
            for (k, keys) in &b.kits {
                let ks: Vec<String> = (0..128).filter(|&i| keys[i] != 0).map(|i| i.to_string()).collect();
                println!("  kit   {k:8} keys {}", ks.join(" "));
            }
            for w in &b.warnings {
                println!("  skipped: {w}");
            }
        }
        Err(e) => println!("nothing loaded: {e} (tools/get-samples.sh fetches the samples)"),
    }
}

// ---------------------------------------------------------------- tests

/// Test material: a directory written under the target directory.
#[cfg(test)]
pub mod testdata {
    use std::path::{Path, PathBuf};

    /// A 16-bit WAV of `frames` frames from `f(frame, channel)`.
    pub fn wav(rate: u32, chans: u16, frames: usize, f: impl Fn(usize, usize) -> f32) -> Vec<u8> {
        let bytes = frames * chans as usize * 2;
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36 + bytes as u32).to_le_bytes());
        v.extend_from_slice(b"WAVEfmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&chans.to_le_bytes());
        v.extend_from_slice(&rate.to_le_bytes());
        v.extend_from_slice(&(rate * 2 * chans as u32).to_le_bytes());
        v.extend_from_slice(&(2 * chans).to_le_bytes());
        v.extend_from_slice(&16u16.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&(bytes as u32).to_le_bytes());
        for i in 0..frames {
            for c in 0..chans as usize {
                v.extend_from_slice(&((f(i, c) * 32767.0).round() as i16).to_le_bytes());
            }
        }
        v
    }

    /// A sine at `hz`, its RMS the reference (-20 dBFS).
    pub fn sine(rate: u32, chans: u16, hz: f64, secs: f64) -> Vec<u8> {
        let a = 0.1 * std::f64::consts::SQRT_2;
        wav(rate, chans, (secs * rate as f64) as usize, |i, _| {
            (a * (std::f64::consts::TAU * hz * i as f64 / rate as f64).sin()) as f32
        })
    }

    /// A bank of a SoundFont's presets and kits alone.
    pub fn sf2_bank(path: &Path, melodic: &[&str], kits: &[&str]) -> Result<super::Bank, String> {
        let mut b = super::Bank::new();
        super::load_sf2(&mut b, path, melodic, kits)?;
        Ok(b)
    }

    /// A fresh directory for test `name`.
    pub fn dir(name: &str) -> PathBuf {
        let d = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("samples-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("multi")).expect("test dir");
        d
    }

    /// Frequency by upward zero crossings (interpolated), over `x`.
    pub fn freq(x: &[f32]) -> f64 {
        let ups: Vec<f64> = (1..x.len())
            .filter(|&i| x[i - 1] < 0.0 && x[i] >= 0.0)
            .map(|i| (i - 1) as f64 + (-x[i - 1] / (x[i] - x[i - 1])) as f64)
            .collect();
        (ups.len() - 1) as f64 * 48000.0 / (ups[ups.len() - 1] - ups[0])
    }

    /// A small SoundFont: one sample (a sine at 440 Hz, 32 kHz, looped),
    /// an instrument with a global zone (loop until release, attenuation) and
    /// two zones (the lower keys tuned +100 cents with their root at 72),
    /// a preset ("Trumpet") adding a fine tune of +50 cents; and a kit
    /// ("Standard", bank 128) with keys 42 and 46 in exclusive class 1.
    pub fn tiny_sf2() -> Vec<u8> {
        let rate = 32000u32;
        let period = 32000.0 / 440.0;
        let n = 4000usize;
        let smp: Vec<i16> =
            (0..n).map(|i| (8000.0 * (std::f64::consts::TAU * i as f64 / period).sin()) as i16).collect();
        let chunk = |id: &[u8], body: &[u8]| {
            let mut v = id.to_vec();
            v.extend_from_slice(&(body.len() as u32).to_le_bytes());
            v.extend_from_slice(body);
            if body.len() & 1 == 1 {
                v.push(0);
            }
            v
        };
        let list = |typ: &[u8], parts: &[Vec<u8>]| {
            let mut body = typ.to_vec();
            for p in parts {
                body.extend_from_slice(p);
            }
            chunk(b"LIST", &body)
        };
        let name20 = |s: &str| {
            let mut v = s.as_bytes().to_vec();
            v.resize(20, 0);
            v
        };
        let gn = |op: u16, amt: i16| [op.to_le_bytes(), amt.to_le_bytes()].concat();
        let range = |op: u16, lo: u8, hi: u8| [op.to_le_bytes(), [lo, hi]].concat();
        let bag = |g: u16| [g.to_le_bytes(), 0u16.to_le_bytes()].concat();
        // presets: "Trumpet" (bank 0) zone 0 -> instrument 0; "Standard" (bank 128) zone 1 -> instrument 1
        let mut phdr = Vec::new();
        for (nm, prog, bank, bagi) in [("Trumpet", 56u16, 0u16, 0u16), ("Standard", 0, 128, 1), ("EOP", 0, 0, 2)] {
            phdr.extend(name20(nm));
            phdr.extend(prog.to_le_bytes());
            phdr.extend(bank.to_le_bytes());
            phdr.extend(bagi.to_le_bytes());
            phdr.extend([0u8; 12]);
        }
        let pgen = [gn(52, 50), gn(41, 0), gn(41, 1), gn(0, 0)].concat();
        let pbag = [bag(0), bag(2), bag(3)].concat();
        // instrument 0: global (mode 3, attenuation 60), zone keys 0-59 (root 72, coarse +1), zone keys 60-127
        // instrument 1: key 42 and key 46, class 1, mode 0
        let igen = [
            gn(54, 3),
            gn(48, 60),
            range(43, 0, 59),
            gn(58, 72),
            gn(51, 1),
            gn(53, 0),
            range(43, 60, 127),
            gn(53, 0),
            range(43, 42, 42),
            gn(57, 1),
            gn(53, 0),
            range(43, 46, 46),
            gn(57, 1),
            gn(53, 0),
            gn(0, 0),
        ]
        .concat();
        let ibag = [bag(0), bag(2), bag(6), bag(8), bag(11), bag(14)].concat();
        let mut inst = Vec::new();
        for (nm, b) in [("Sine", 0u16), ("Hats", 3), ("EOI", 5)] {
            inst.extend(name20(nm));
            inst.extend(b.to_le_bytes());
        }
        let mut shdr = Vec::new();
        for (nm, s, e) in [("sine", 0u32, n as u32), ("EOS", 0, 0)] {
            shdr.extend(name20(nm));
            for x in [s, e, 800, 800 + (period * 10.0).round() as u32, rate] {
                shdr.extend(x.to_le_bytes());
            }
            shdr.extend([69u8, 0]);
            shdr.extend(0u16.to_le_bytes());
            shdr.extend(1u16.to_le_bytes());
        }
        let mut sbytes: Vec<u8> = smp.iter().flat_map(|s| s.to_le_bytes()).collect();
        sbytes.extend([0u8; 92]);
        let mut body = b"sfbk".to_vec();
        body.extend(list(b"INFO", &[chunk(b"ifil", &[2, 0, 1, 0])]));
        body.extend(list(b"sdta", &[chunk(b"smpl", &sbytes)]));
        body.extend(list(
            b"pdta",
            &[
                chunk(b"phdr", &phdr),
                chunk(b"pbag", &pbag),
                chunk(b"pmod", &[0; 10]),
                chunk(b"pgen", &pgen),
                chunk(b"inst", &inst),
                chunk(b"ibag", &ibag),
                chunk(b"imod", &[0; 10]),
                chunk(b"igen", &igen),
                chunk(b"shdr", &shdr),
            ],
        ));
        chunk(b"RIFF", &body)
    }

    /// A samples directory: "held" sus (one 5.5 s sine at 69, 22.05 kHz,
    /// its last 0.5 s faded, as the manifest's are) and "shake" hit (three
    /// round robins, 50 ms each, 1, 2 and 3 cycles of a square).
    pub fn held(name: &str) -> PathBuf {
        let d = dir(name);
        let a = 0.1 * std::f64::consts::SQRT_2;
        let long = wav(22050, 1, 22050 * 11 / 2, |i, _| {
            let t = i as f64 / 22050.0;
            (a * (std::f64::consts::TAU * 440.0 * t).sin() * ((5.5 - t) / 0.5).min(1.0)) as f32
        });
        std::fs::write(d.join("multi/held.wav"), long).expect("write");
        let mut m = String::from("instrument\tarticulation\troot\tvel_lo\tvel_hi\trr\tgain_db\tfile\n");
        m += "held\tsus\t69\t0\t127\t1\t0\tmulti/held.wav\n";
        for k in 1..=3 {
            let sq = wav(48000, 1, 2400, |i, _| if (i * k / 1200) % 2 == 0 { 0.5 } else { -0.5 });
            std::fs::write(d.join(format!("multi/shake{k}.wav")), sq).expect("write");
            m += &format!("shake\thit\t60\t0\t127\t{k}\t0\tmulti/shake{k}.wav\n");
        }
        std::fs::write(d.join("manifest.tsv"), m).expect("manifest");
        d
    }

    /// A samples directory: "sine" sus (roots 57 at 44.1 kHz, 69 stereo,
    /// two round robins of 81 loud and one soft) and stac (69 only, 0.2 s),
    /// "hit" one (a second of middle C, for a drum), and a manifest naming
    /// them; all at the reference level.
    pub fn synthetic(name: &str) -> PathBuf {
        let d = dir(name);
        let hz = |n: f64| 440.0 * ((n - 69.0) / 12.0).exp2();
        let put = |f: &str, b: Vec<u8>| std::fs::write(d.join(f), b).expect("write");
        put("multi/s57.wav", sine(44100, 1, hz(57.0), 1.0));
        put("multi/s69.wav", sine(48000, 2, hz(69.0), 1.0));
        put("multi/s81a.wav", sine(48000, 1, hz(81.0), 1.0));
        put("multi/s81b.wav", sine(48000, 1, hz(81.0), 1.0));
        put("multi/s81soft.wav", sine(48000, 1, hz(81.0), 1.0));
        put("multi/stac.wav", sine(48000, 1, hz(69.0), 0.2));
        put("multi/hit.wav", sine(48000, 1, hz(60.0), 1.0));
        let m = "# a test\ninstrument\tarticulation\troot\tvel_lo\tvel_hi\trr\tgain_db\tfile\n\
                 sine\tsus\t57\t0\t127\t0\t0\tmulti/s57.wav\n\
                 sine\tsus\t69\t0\t127\t0\t0\tmulti/s69.wav\n\
                 sine\tsus\t81\t64\t127\t0\t0\tmulti/s81a.wav\n\
                 sine\tsus\t81\t64\t127\t1\t0\tmulti/s81b.wav\n\
                 sine\tsus\t81\t0\t63\t0\t-6\tmulti/s81soft.wav\n\
                 sine\tstac\t69\t0\t127\t0\t0\tmulti/stac.wav\n\
                 hit\tone\t60\t0\t127\t0\t0\tmulti/hit.wav\n";
        std::fs::write(d.join("manifest.tsv"), m).expect("manifest");
        d
    }
}

#[cfg(test)]
mod tests {
    use super::testdata::*;
    use super::*;

    fn play(b: &Bank, id: u16, note: f32, vel: f32, secs: f32) -> Vec<[f32; 2]> {
        let mut s = SAMPLER;
        assert!(s.start(b, id, note, vel, (secs * RATE) as i64, 0));
        let mut v = vec![[0.0f32; 2]; (secs * RATE) as usize];
        for c in v.chunks_mut(32) {
            s.fill(b, note, true, c);
        }
        v
    }

    #[test]
    fn synthetic_dir_plays_in_tune() {
        let d = synthetic("tune");
        let b = load(&d).expect("load");
        let _ = std::fs::remove_dir_all(&d);
        let id = b.multi("sine", "sus").expect("sine sus");
        // at a root, a fifth over one (from 57, recorded at 44.1 kHz), seven
        // semitones under one, between two roots, far past the last
        for note in [69.0f32, 64.0, 62.0, 75.0, 76.0, 90.0, 40.0] {
            let v = play(&b, id, note, 0.9, 0.5);
            let l: Vec<f32> = v[2400..].iter().map(|x| x[0]).collect();
            let want = 440.0 * ((note as f64 - 69.0) / 12.0).exp2();
            let got = freq(&l);
            assert!((got / want - 1.0).abs() < 0.002, "note {note}: {got:.2} Hz, want {want:.2}");
            // the level: the reference, both channels
            let rms = (v[2400..].iter().map(|x| x[0] * x[0] + x[1] * x[1]).sum::<f32>()
                / (2 * (v.len() - 2400)) as f32)
                .sqrt();
            assert!((rms / REF_RMS - 1.0).abs() < 0.05, "note {note}: RMS {rms}");
        }
        // nearest roots: 62 from 57 (5 away, not 69's 7), 63 a tie (57)
        for (note, root) in [(62.0, 57.0), (63.0, 57.0), (64.0, 69.0), (76.0, 81.0), (20.0, 57.0)] {
            let (z, _, _) = b.pick(id, note, 0.9, 48000, 0).expect("pick");
            assert_eq!(b.zones[z[0] as usize].root, root, "note {note}");
        }
        // round robins rotate at 81 (loud), the soft layer has one
        let rr: Vec<u32> = (0..4).map(|k| b.pick(id, 81.0, 0.9, 48000, k).unwrap().0[0]).collect();
        assert!(rr[0] != rr[1] && rr[0] == rr[2] && rr[1] == rr[3], "{rr:?}");
        let soft: Vec<u32> = (0..2).map(|k| b.pick(id, 81.0, 0.3, 48000, k).unwrap().0[0]).collect();
        assert!(soft[0] == soft[1] && soft[0] != rr[0] && soft[0] != rr[1]);
        // auto: short gates the staccato, long the sustain
        let auto = b.multi("sine", "auto").expect("auto");
        let stac = b.multi("sine", "stac").expect("stac");
        assert_eq!(b.pick(auto, 69.0, 0.9, 4800, 0).unwrap().0, b.pick(stac, 69.0, 0.9, 4800, 0).unwrap().0);
        assert_eq!(b.pick(auto, 69.0, 0.9, 48000, 0).unwrap().0, b.pick(id, 69.0, 0.9, 48000, 0).unwrap().0);
        // the short one's first 0.4 s matched to the long one's first second
        // at its middle root (81): the staccato's 0.2 s, an octave up, lasts
        // 0.1 s, the sustain all of it; so twice as loud
        let g = b.pick(auto, 69.0, 0.9, 4800, 0).unwrap().2;
        assert!((g / 2.0 - 1.0).abs() < 0.05 && b.pick(auto, 69.0, 0.9, 48000, 0).unwrap().2 == 1.0, "{g}");
        assert!(b.multi("sine", "pizz").is_none() && b.gm("Trumpet").is_none() && b.drum("Standard", 36).is_none());
    }

    #[test]
    fn a_sample_that_runs_out_fades_and_ends() {
        let d = synthetic("end");
        let b = load(&d).expect("load");
        let _ = std::fs::remove_dir_all(&d);
        let id = b.multi("sine", "stac").unwrap();
        let mut s = SAMPLER;
        assert!(s.start(&b, id, 69.0, 0.9, 48000, 0));
        let mut v = vec![[0.0f32; 2]; 48000];
        let mut live = 0;
        for (k, c) in v.chunks_mut(32).enumerate() {
            if s.fill(&b, 69.0, true, c) {
                live = k;
            }
        }
        // 0.2 s long: over by then, the last 10 ms fading
        assert!((live * 32) as f32 / RATE < 0.205 && (live * 32) as f32 / RATE > 0.19, "ended at block {live}");
        let peak = |a: usize, b: usize| v[a..b].iter().map(|x| x[0].abs()).fold(0.0, f32::max);
        assert!(peak(9000, 9200) > 0.1 && peak(9580, 9600) < 0.02 && peak(9700, 48000) == 0.0);
    }

    /// A held note outlasting a sustained recording: its middle loops,
    /// crossfaded, steady to within a few dB, until the gate closes.
    #[test]
    fn held_notes_loop_past_the_recording() {
        let d = held("held");
        let b = load(&d).expect("load");
        let _ = std::fs::remove_dir_all(&d);
        let id = b.multi("held", "sus").unwrap();
        let mut s = SAMPLER;
        assert!(s.start(&b, id, 69.0, 0.9, 12 * 48000, 0));
        let mut v = vec![[0.0f32; 2]; 12 * 48000];
        for c in v.chunks_mut(32) {
            assert!(s.fill(&b, 69.0, true, c));
        }
        // 50 ms windows from 0.1 s to 12 s
        for w in v[4800..].chunks(2400) {
            let rms = (w.iter().map(|x| x[0] * x[0]).sum::<f32>() / w.len() as f32).sqrt();
            assert!(rms > 0.1 / 1.5 && rms < 0.1 * 1.5, "a window's RMS {rms}");
        }
        let l: Vec<f32> = v[5 * 48000..].iter().map(|x| x[0]).collect();
        assert!((freq(&l) / 440.0 - 1.0).abs() < 0.002);
        // released, it plays out to its end (a drum's hit never loops)
        let mut out = [[0.0f32; 2]; 32];
        let mut blocks = 0;
        while s.fill(&b, 69.0, false, &mut out) {
            blocks += 1;
        }
        assert!(blocks * 32 < 6 * 48000);
    }

    #[test]
    fn malformed_wavs_are_errors() {
        let good = sine(48000, 1, 440.0, 0.01);
        assert!(read_wav(&good).is_ok());
        // every truncation, and every byte of the header broken
        for n in 0..good.len() {
            let _ = read_wav(&good[..n]);
        }
        for i in 0..44 {
            for x in [0u8, 0xFF, 0x7F, 0x80] {
                let mut b = good.clone();
                b[i] = x;
                let _ = read_wav(&b);
            }
        }
        let mut big = good.clone();
        big[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(read_wav(&big).is_err(), "a data chunk past the end");
        let mut eight = good.clone();
        eight[34] = 8;
        assert!(read_wav(&eight).is_err(), "8-bit");
        let mut three = good.clone();
        three[22] = 3;
        assert!(read_wav(&three).is_err(), "three channels");
        assert!(read_wav(b"RIFF").is_err() && read_wav(&[]).is_err());
        assert!(read_wav(&wav(48000, 1, 1, |_, _| 0.0)).is_err(), "one frame");
    }

    #[test]
    fn malformed_manifests_skip_rows() {
        let d = dir("bad");
        std::fs::write(d.join("multi/ok.wav"), sine(48000, 1, 440.0, 0.1)).unwrap();
        std::fs::write(d.join("multi/short.wav"), &sine(48000, 1, 440.0, 0.1)[..30]).unwrap();
        let m = "instrument\tarticulation\troot\tvel_lo\tvel_hi\trr\tgain_db\tfile\n\
                 a\tsus\t69\t0\t127\t0\t0\tmulti/ok.wav\n\
                 a\tsus\t70\t0\t127\t0\t0\n\
                 a\tsus\tx\t0\t127\t0\t0\tmulti/ok.wav\n\
                 a\tsus\t69\t100\t20\t0\t0\tmulti/ok.wav\n\
                 a\tsus\t200\t0\t127\t0\t0\tmulti/ok.wav\n\
                 a\tsus\t69\t0\t127\t0\tNaN\tmulti/ok.wav\n\
                 a\tsus\t69\t0\t127\t0\t0\t../outside.wav\n\
                 a\tsus\t69\t0\t127\t0\t0\t/etc/passwd\n\
                 b\tsus\t69\t0\t127\t0\t0\tmulti/missing.wav\n\
                 c\tsus\t69\t0\t127\t0\t0\tmulti/short.wav\n\
                 \n\
                 garbage line without tabs\n";
        std::fs::write(d.join("manifest.tsv"), m).unwrap();
        let b = load(&d).expect("the good row loads");
        assert!(b.multi("a", "sus").is_some() && b.multi("b", "sus").is_none() && b.multi("c", "sus").is_none());
        assert_eq!(b.warnings.iter().filter(|w| w.contains("line")).count(), 8, "{:?}", b.warnings);
        assert!(
            b.warnings.iter().any(|w| w.contains("missing.wav")) && b.warnings.iter().any(|w| w.contains("short.wav"))
        );
        // nothing loadable at all, and no directory: errors
        std::fs::write(d.join("manifest.tsv"), "garbage\n\u{0}\u{1}").unwrap();
        assert!(load(&d).is_err());
        std::fs::write(d.join("manifest.tsv"), [0xFFu8, 0xFE, 0x00]).unwrap();
        assert!(load(&d).is_err());
        let _ = std::fs::remove_dir_all(&d);
        assert!(load(&d).is_err());
    }

    #[test]
    fn a_tiny_soundfont_resolves() {
        let d = dir("sf2");
        let path = d.join("tiny.sf2");
        let sf = tiny_sf2();
        std::fs::write(&path, &sf).unwrap();
        let mut b = Bank::new();
        load_sf2(&mut b, &path, &["Trumpet", "Nothing"], &["Standard"]).expect("tiny sf2");
        assert!(b.warnings.iter().any(|w| w.contains("Nothing")), "{:?}", b.warnings);
        let id = b.gm("Trumpet").expect("Trumpet");
        let z = b.zones[b.pick(id, 64.0, 0.8, 48000, 0).unwrap().0[0] as usize];
        assert_eq!((z.mode, z.root, z.tune), (3, 69.0, 0.5), "the global zone's mode, the preset's fine tune");
        let z = b.zones[b.pick(id, 50.0, 0.8, 48000, 0).unwrap().0[0] as usize];
        assert_eq!((z.root, z.tune), (72.0, 1.5), "the zone's root and coarse tune over the preset's fine");
        // in tune through the loop: 440 Hz sounds where the root less the
        // tuning is (68.5 above key 60, 70.5 below)
        for note in [69.5f32, 76.5, 60.0, 50.0] {
            let v = play(&b, id, note, 0.8, 0.5);
            let l: Vec<f32> = v.iter().map(|x| x[0]).collect();
            let at = if note >= 60.0 { 68.5 } else { 70.5 };
            let want = 440.0 * ((note as f64 - at) / 12.0).exp2();
            let got = freq(&l[1000..]);
            assert!((got / want - 1.0).abs() < 0.003, "note {note}: {got:.2} Hz, want {want:.2}");
            let rms = (l[1000..].iter().map(|x| x * x).sum::<f32>() / (l.len() - 1000) as f32).sqrt();
            assert!((rms / REF_RMS - 1.0).abs() < 0.1, "note {note}: RMS {rms} (normalised)");
        }
        // the loop holds past the sample's 0.125 s while the gate is on
        let v = play(&b, id, 68.5, 0.8, 0.5);
        assert!(v[20000..].iter().map(|x| x[0].abs()).fold(0.0, f32::max) > 0.1);
        // the kit: two keys, one class, nothing at 36
        let (h, o) = (b.drum("Standard", 42).expect("42"), b.drum("Standard", 46).expect("46"));
        assert!(b.drum("Standard", 36).is_none() && h != o);
        let mut s = SAMPLER;
        assert!(s.start(&b, o, 0.0, 0.8, 0, 0));
        assert_eq!((s.excl, s.key), (1, 46.0));
        // broken: every truncation, and bytes of the structure overwritten
        for n in (0..sf.len()).step_by(7) {
            std::fs::write(&path, &sf[..n]).unwrap();
            let mut b = Bank::new();
            let _ = load_sf2(&mut b, &path, &["Trumpet"], &["Standard"]);
        }
        let pdta = sf.windows(4).position(|w| w == b"pdta").unwrap();
        for i in (pdta..sf.len()).step_by(3) {
            for x in [0u8, 0xFF, 0x80] {
                let mut s = sf.clone();
                s[i] = x;
                std::fs::write(&path, &s).unwrap();
                let mut b = Bank::new();
                if load_sf2(&mut b, &path, &["Trumpet"], &["Standard"]).is_ok() {
                    // whatever loaded plays without a panic
                    for &(_, id) in &b.gms {
                        let mut v = [[0.0f32; 2]; 64];
                        let mut sm = SAMPLER;
                        if sm.start(&b, id, 64.0, 0.8, 4800, 0) {
                            for _ in 0..50 {
                                sm.fill(&b, 64.0, true, &mut v);
                            }
                        }
                    }
                }
            }
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn timgm6mb_parses_when_present() {
        let path = Path::new("/usr/share/sounds/sf2/TimGM6mb.sf2");
        if !path.is_file() {
            eprintln!("no {}: skipped", path.display());
            return;
        }
        let presets = sf2_presets(path).expect("presets");
        assert!(presets.len() > 128, "{} presets", presets.len());
        let melodic: Vec<&str> = presets.iter().filter(|p| p.0 == 0 && p.1 < 6).map(|p| p.2.as_str()).collect();
        assert!(presets.iter().any(|p| p.0 == 128 && p.2 == "Standard"));
        let mut b = Bank::new();
        load_sf2(&mut b, path, &melodic, &["Standard"]).expect("load");
        assert_eq!(b.gms.len(), melodic.len(), "{:?}", b.warnings);
        for (p, id) in &b.gms {
            let (z, _, _) = b.pick(*id, 60.0, 0.8, 48000, 0).expect(p);
            assert!(!z.is_empty() && z.iter().all(|&z| b.frames(&b.zones[z as usize]).iter().any(|&x| x != 0)), "{p}");
            let rms = b.rms_1s(*id, 60.0);
            assert!((rms / REF_RMS - 1.0).abs() < 0.01, "{p}: {rms}");
        }
        for key in [36u8, 38, 42, 46] {
            assert!(b.drum("Standard", key).is_some(), "key {key}");
        }
        assert!(b.bytes() > 100_000);
    }
}
