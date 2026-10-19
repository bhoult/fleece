//! WESTERN: the spaghetti western's score (a placeholder: lo-fi's music until it is written).

use super::*;

/// A Western piece's state: its meter (16ths a bar and a beat, and its name).
#[derive(Default)]
pub(super) struct Western {
    pub(super) bar_steps: u32,
    pub(super) beat_steps: u32,
    pub(super) meter: &'static str,
}

impl Song {
    pub(super) fn piece_western(&mut self, s: &mut Synth) {
        self.piece_lofi(s);
        self.wn = Western { bar_steps: 16, beat_steps: 4, meter: "4/4" };
    }

    pub(super) fn western_section(&mut self) {}

    pub(super) fn western(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        self.lofi(s, b, sec, m)
    }

    /// Western's beat in any meter, for a mix.
    pub(super) fn western_beat(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        self.lofi_beat(s, b, sec, m)
    }
}

impl Western {
    pub(super) fn bs_at(&self, _k: u32) -> u32 {
        self.bar_steps.max(1)
    }
    pub(super) fn dress_at(&self, _k: usize) -> usize {
        0
    }
}

pub(super) fn palette_western(_ch: usize) -> Option<&'static [(&'static str, Make)]> {
    None
}

pub(super) fn kit_palette_western(_ch: usize) -> Option<&'static [(&'static str, MakeKit)]> {
    None
}

pub(super) fn sounds_western(r: &mut Rng, s: &mut Synth) {
    sounds_lofi(r, s)
}
