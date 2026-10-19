//! INDONESIAN: gamelan (a placeholder: lo-fi's music until it is written).

use super::*;

/// An Indonesian piece's state: its meter (16ths a bar and a beat, and its name).
#[derive(Default)]
pub(super) struct Indonesian {
    pub(super) bar_steps: u32,
    pub(super) beat_steps: u32,
    pub(super) meter: &'static str,
}

impl Song {
    pub(super) fn piece_indonesian(&mut self, s: &mut Synth) {
        self.piece_lofi(s);
        self.idn = Indonesian { bar_steps: 16, beat_steps: 4, meter: "4/4" };
    }

    pub(super) fn indonesian_section(&mut self) {}

    pub(super) fn indonesian(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        self.lofi(s, b, sec, m)
    }

    /// Indonesian's beat in any meter, for a mix.
    pub(super) fn indonesian_beat(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        self.lofi_beat(s, b, sec, m)
    }
}

pub(super) fn sounds_indonesian(r: &mut Rng, s: &mut Synth) {
    sounds_lofi(r, s)
}
