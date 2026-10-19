//! STAR WARS: a space-opera film score (a placeholder: lo-fi's music until it is written).

use super::*;

/// A Star Wars piece's state: its meter (16ths a bar and a beat, and its name).
#[derive(Default)]
pub(super) struct StarWars {
    pub(super) bar_steps: u32,
    pub(super) beat_steps: u32,
    pub(super) meter: &'static str,
}

impl Song {
    pub(super) fn piece_starwars(&mut self, s: &mut Synth) {
        self.piece_lofi(s);
        self.sw = StarWars { bar_steps: 16, beat_steps: 4, meter: "4/4" };
    }

    pub(super) fn starwars_section(&mut self) {}

    pub(super) fn starwars(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        self.lofi(s, b, sec, m)
    }

    /// Star Wars's beat in any meter, for a mix.
    pub(super) fn starwars_beat(&mut self, s: &mut Synth, b: &mut Bar, sec: Sec, m: u32) {
        self.lofi_beat(s, b, sec, m)
    }
}

pub(super) fn sounds_starwars(r: &mut Rng, s: &mut Synth) {
    sounds_lofi(r, s)
}
