//! CELTIC: the Irish and Scottish session (a placeholder: lo-fi's music until it is written).

use super::*;

/// A Celtic piece's state: its meter (16ths a bar and a beat, and its name).
#[derive(Default)]
pub(super) struct Celtic {
    pub(super) bar_steps: u32,
    pub(super) beat_steps: u32,
    pub(super) meter: &'static str,
}

impl Song {
    pub(super) fn piece_celtic(&mut self, s: &mut Synth) {
        self.piece_lofi(s);
        self.ce = Celtic { bar_steps: 16, beat_steps: 4, meter: "4/4" };
    }

    pub(super) fn celtic_section(&mut self) {}

    pub(super) fn celtic(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        self.lofi(s, b, sec, m)
    }

    /// Celtic's beat in any meter, for a mix.
    pub(super) fn celtic_beat(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        self.lofi_beat(s, b, sec, m)
    }
}

pub(super) fn sounds_celtic(r: &mut Rng, s: &mut Synth) {
    sounds_lofi(r, s)
}
