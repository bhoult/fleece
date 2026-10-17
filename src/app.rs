//! The live show: a window (or the offscreen driver for tests and video),
//! the frame loop that keeps the GPU's iteration budget filling the frame,
//! the HUD and the keys.

use crate::gpu::{self, Gpu};
use crate::hud::{self, Hud, Item, Style};
use crate::render::{Renderer, Stats};
use crate::sample::Sample;
use crate::sdl::{self, Window};
use crate::show::{self, Look, Show};
use crate::{Opts, png, tone_of};
use std::time::Instant;

/// Time samples per frame (motion blur) and the shutter in Electric Sheep frames.
const BLUR_SAMPLES: usize = 64;
const SHUTTER_FRAMES: f64 = 1.0;
/// Animation speeds (Left / Right).
const SPEEDS: [f64; 9] = [0.0625, 0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0];

/// fleece's own additions to the show (v2). `CLASSIC` is Electric Sheep's
/// show as it was; `V2` the recommended set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fx {
    /// glow: 0 off, 1 subtle, 2 soft, 3 strong
    pub glow: u8,
    /// frames blended over time: 0 off, 1 short (a smoother picture), 2 long (trails)
    pub trails: u8,
    /// auto-exposure
    pub levels: bool,
    /// camera drift, palette drift, motion for every sheep
    pub drift: bool,
    pub hue: bool,
    pub wobble: bool,
    /// 3D camera: 0 flat, 1 tilted, 2 tilted with depth of field
    pub tilt: u8,
    /// how strongly the music drives the show: 0 off, 1 low, 2 medium, 3 high
    pub audio: u8,
}

impl Fx {
    pub const CLASSIC: Fx = Fx { glow: 0, trails: 0, levels: false, drift: false, hue: false, wobble: false, tilt: 0, audio: 0 };
    pub const V2: Fx = Fx { glow: 2, trails: 1, levels: true, drift: true, hue: false, wobble: true, tilt: 0, audio: 0 };
}

/// Z: steps through cheaper settings until the show keeps up with the
/// display's rate, measuring each for OPT_SETTLE seconds and keeping it only
/// if it helped.
struct Optimize {
    /// the step being measured (None: things as they were)
    step: Option<usize>,
    /// the settings before it, and the frame rate they gave
    before: Settings,
    fps_before: f64,
    frame0: u64,
    t0: Instant,
    kept: Vec<&'static str>,
}

/// What the optimizer changes.
#[derive(Clone, Copy)]
struct Settings {
    budget: f32,
    supersample: u32,
    use_cpu: bool,
    fixed: Option<(u32, u32)>,
    scale: f32,
    rw: u32,
    rh: u32,
}

const OPT_SETTLE: f64 = 1.5;
/// The optimizer's steps, the least visible first: fewer samples per frame
/// (trails hide the extra noise), no supersampling, fewer still, 1080 lines
/// at most, the GPU alone (when waiting on the CPU costs more than its share).
const OPT_STEPS: [&str; 6] = ["ITERATIONS 70%", "SUPERSAMPLE 1X", "ITERATIONS 55%", "1080P", "CPU OFF", "ITERATIONS 40%"];

/// How strongly the music moves the picture (A: off, low, medium, high): a
/// factor on every response to it.
const AUDIO_STRENGTH: [f64; 4] = [0.0, 0.5, 1.0, 1.8];
/// How long the audio analysis takes to see a beat (seconds): its parec's
/// 20 ms chunks and the onset showing in its 2048-sample window. With a frame
/// of rendering, the picture of a recording ran 1-3 frames behind its sound.
const ANALYSIS_LAG: f64 = 0.04;
/// Frame blending's time constants (animation seconds) for short and long trails.
const TRAILS: [f64; 3] = [0.0, 0.06, 0.35];
/// Glow (subtle, soft, strong): its strength, and the most light it adds,
/// so a bright region keeps its detail instead of going white.
const GLOW: [(f32, f32); 4] = [(0.0, 0.0), (0.6, 0.3), (1.2, 0.6), (2.4, 1.2)];
/// Auto-exposure aims the 90th percentile of the lit pixels' luma at the
/// real sheep's median (80 genebank sheep: 0.56, quartiles 0.43 and 0.70),
/// leaves alone sheep within 15 % of it, and pulls down those saturating
/// more than a tenth of their lit pixels (the top quarter).
const LEVEL_TARGET: f64 = 0.56;
const LEVEL_DEADBAND: f64 = 0.15;
const LEVEL_MAX_SATURATED: f64 = 0.10;
/// Supersampling choices (X cycles them; 0 is automatic: 2x while the
/// histogram stays small enough to sit in the GPU's cache, else 1x).
const SUPERSAMPLES: [u32; 4] = [0, 1, 2, 4];

pub struct App {
    pub win: Window,
    pub r: Renderer,
    pub show: Show,
    samples: Vec<Sample>,
    /// GPU rounds per frame (adapted to the frame time)
    dispatches: f32,
    max_dispatches: u32,
    /// the share of a display frame the GPU's iterations get (Z may lower it)
    budget: f32,
    /// Z: the optimizer at work, and what it last reported (and when)
    opt: Option<Optimize>,
    opt_note: Option<(String, Instant)>,
    speed: usize,
    paused: bool,
    blur: bool,
    hud: bool,
    hud_at: Instant,
    /// the HUD wants redrawing now (a key, the mouse over another button)
    hud_dirty: bool,
    /// the mouse (drawable pixels), the HUD's buttons (overlay pixels), the
    /// one under the mouse, and until when the pointer shows in fullscreen
    /// after the mouse moved
    mouse: Option<(f64, f64)>,
    buttons: Vec<hud::Button>,
    hover: Option<char>,
    pointer_until: Option<Instant>,
    cursor: bool,
    /// the display's frame time, seconds
    period: f64,
    /// per-second rates for the HUD
    rate_gpu: f64,
    rate_cpu: f64,
    fps: f64,
    quality: f64,
    scale: f32,
    /// a fixed render size (Electric Sheep's), scaled to the window
    fixed: Option<(u32, u32)>,
    /// samples per pixel every frame (--quality), else as many as 30 fps allows
    target_q: Option<f32>,
    rw: u32,
    rh: u32,
    frame: u64,
    headless: bool,
    /// the sheep the window title names
    titled: u32,
    /// recording the show (R)
    rec: Option<crate::video::Recorder>,
    /// burn the credit into recorded frames (--credit off / K: metadata only)
    burn_credit: bool,
    /// seconds the burned-in credit shows at the start of a recording
    credit_secs: f64,
    /// milliseconds spent building the frame's samples (smoothed)
    pub build_ms: f64,
    /// v2's effects, and the set V brings back after classic
    pub fx: Fx,
    fx_saved: Fx,
    look: Look,
    /// supersampling: 0 automatic, else buckets per pixel on each axis
    supersample: u32,
    /// auto-exposure: ln of the brightness multiplier, and whether it has
    /// yet to be set (the first measure sets it outright: no fade-in)
    exposure: f64,
    exposure_unset: bool,
    /// the music's own exposure (natural log, like `exposure`), this frame and
    /// the last (whose picture the next brightness counts come from)
    audio_ev: [f64; 2],
    /// a fixed delay for a recording's sound (--sound-delay), else measured
    sound_delay: Option<f64>,
    /// when the optimizer starts by itself (the window, once warmed up)
    auto_optimize: Option<Instant>,
    /// the music's lead on the camera drift and the hues (show::Look)
    drift_lead: f64,
    hue_lead: f64,
    /// what the speakers play (while audio is on), or a soundtrack's levels
    /// every hop (a video made to music) and the time in it
    audio: Option<crate::audio::Audio>,
    track: Option<(Vec<crate::audio::Levels>, f64)>,
    track_t: f64,
    /// music fleece makes and plays (N), and its seed
    music: Option<crate::music::Player>,
    music_seed: u64,
}

impl App {
    pub fn new(o: &Opts, headless: bool) -> App {
        let win = Window::open("fleece", o.w, o.h, headless, o.fullscreen).unwrap_or_else(|e| {
            eprintln!("fleece: {e}");
            std::process::exit(1)
        });
        let g = crate::gl::Gl::load(Window::proc_address).unwrap_or_else(|e| {
            eprintln!("fleece: {e}");
            std::process::exit(1)
        });
        let (dw, dh) = if headless { (o.w, o.h) } else { win.drawable_size() };
        let (rw, rh) = o.render.unwrap_or_else(|| render_size(dw, dh, o.scale));
        let mut gpu = Gpu::new(g, rw, rh).unwrap_or_else(|e| {
            eprintln!("fleece: shader: {e}");
            std::process::exit(1)
        });
        let supersample = o.supersample.unwrap_or(0);
        fit_histogram(&mut gpu, rw, rh, supersample);
        let vsync = !headless && win.set_vsync(true);
        let period = if headless { 1.0 / o.fps } else { 1.0 / win.refresh_hz().unwrap_or(60.0) };
        let mut r = Renderer::new(gpu, o.workers);
        r.use_gpu = o.gpu;
        let t0 = Instant::now();
        // the window keeps its flock between runs; tests and videos don't touch it
        let saved = if headless || o.fresh {
            Vec::new()
        } else {
            std::fs::read_to_string(crate::flock_path()).map(|x| crate::flam3::parse(&x).0).unwrap_or_default()
        };
        if !saved.is_empty() {
            eprintln!("fleece: {} sheep in the flock from {}", saved.len(), crate::flock_path());
        }
        let genebank = crate::load_genebank();
        if !genebank.0.is_empty() {
            eprintln!("fleece: {} real sheep in the genebank ({})", genebank.0.len(), crate::genebank_source());
        }
        let show = Show::new(o.seed, saved, crate::load_files(&o.load), genebank);
        {
            let mut f = show.flock.lock().unwrap();
            f.use_taste = o.taste;
            f.new_variations = o.fx != Fx::CLASSIC;
            // the window's taste carries on from last time; tests start from the prior
            if !headless && !o.fresh {
                f.taste = crate::taste::Taste::load(&crate::taste_path());
            }
        }
        // stderr: stdout carries the frames in --video
        eprintln!(
            "fleece: {} driver, {}x{} render ({:.0} Hz{}), {} MB video memory free, {} CPU workers, first sheep in {:.2} s: {}",
            win.driver(),
            rw,
            rh,
            1.0 / period,
            if vsync { ", vsync" } else { "" },
            r.gpu.free_vram_mb().map_or("?".into(), |m| m.to_string()),
            r.workers(),
            t0.elapsed().as_secs_f64(),
            show.cur.genome.name
        );
        App {
            win,
            r,
            show,
            samples: Vec::new(),
            dispatches: 2.0,
            max_dispatches: 256,
            budget: 0.85,
            opt: None,
            opt_note: None,
            speed: 4,
            paused: false,
            blur: true,
            hud: o.hud,
            hud_at: Instant::now(),
            hud_dirty: true,
            mouse: None,
            buttons: Vec::new(),
            hover: None,
            pointer_until: None,
            cursor: !o.fullscreen,
            period,
            rate_gpu: 0.0,
            rate_cpu: 0.0,
            fps: 0.0,
            quality: 0.0,
            scale: o.scale,
            fixed: o.render,
            target_q: o.live_quality,
            rw,
            rh,
            frame: 0,
            headless,
            titled: 0,
            rec: None,
            burn_credit: o.credit,
            credit_secs: o.credit_secs,
            build_ms: 0.0,
            fx: o.fx,
            fx_saved: if o.fx == Fx::CLASSIC { Fx::V2 } else { o.fx },
            look: Look::default(),
            supersample,
            exposure: 0.0,
            exposure_unset: true,
            audio_ev: [0.0; 2],
            sound_delay: o.sound_delay,
            auto_optimize: None,
            drift_lead: 0.0,
            hue_lead: 0.0,
            audio: None,
            track: None,
            track_t: 0.0,
            music: None,
            music_seed: o.seed as u64,
        }
    }

    /// One frame: advance the show by `dt` real seconds, start the GPU and the
    /// CPU on it, finish (tone-map) the previous one. Fixed `dispatches`
    /// overrides the adaptive budget (video).
    pub fn frame(&mut self, dt: f64, fixed: Option<u32>) -> Option<Stats> {
        let audio = self.audio_levels();
        // a recording takes the sound from when audio or music is on
        let sound = (self.fx.audio > 0 || self.music.is_some()) && self.track.is_none();
        if let Some(rec) = self.rec.as_mut().filter(|r| !r.has_sound() && sound) {
            if let Err(e) = rec.capture_sound() {
                eprintln!("fleece: recording without sound: {e}");
            }
        }
        let k = AUDIO_STRENGTH[self.fx.audio as usize % 4];
        let [loud, _, _, _, high, beat] = audio.map(|v| v as f64 * k);
        // loud music turns the sheep faster
        let dt_show = if self.paused { 0.0 } else { dt * SPEEDS[self.speed] * (1.0 + 0.6 * loud) };
        if !self.paused {
            self.show.advance(dt_show);
        }
        // the music runs the drift ahead when loud, and turns the hues with
        // its highs and a step on each beat (~0.03 of a turn)
        self.drift_lead += dt_show * 3.0 * loud;
        self.hue_lead += dt_show * (6.0 * high + 58.0 * beat);
        self.look = Look {
            drift: self.fx.drift,
            hue: self.fx.hue,
            wobble: self.fx.wobble,
            tilt: self.fx.tilt,
            audio,
            react: k as f32,
            drift_lead: self.drift_lead,
            hue_lead: self.hue_lead,
        };
        let tau = TRAILS[self.fx.trails as usize % 3];
        self.r.decay = if tau > 0.0 { (-dt_show / tau).exp() as f32 } else { 0.0 };
        let (mut glow, mut cap) = GLOW[self.fx.glow as usize % 4];
        if glow > 0.0 {
            // the glow swells with the music and flares on the beats
            glow *= (1.0 + 0.6 * loud - 0.3 * k + 0.8 * beat) as f32;
            cap *= (1.0 + 0.6 * beat) as f32;
        }
        (self.r.gpu.glow, self.r.gpu.glow_cap) = (glow, cap);
        let n = if self.blur { BLUR_SAMPLES } else { 1 };
        let shutter = SHUTTER_FRAMES / show::ES_FPS;
        let t_build = Instant::now();
        let (bw, bh) = (self.r.gpu.bw, self.r.gpu.bh);
        let mid = self.show.samples(n, shutter, bw, bh, &self.look, &mut self.samples);
        self.build_ms += (t_build.elapsed().as_secs_f64() * 1000.0 - self.build_ms) * 0.1;
        let mut tone = tone_of(&mid);
        // the music's brightness, on top of auto-levels (which leaves it out
        // of what it steers): louder is brighter, each beat a flash; the
        // picture moves as exposure^(1 / gamma), hence the gamma
        let g = mid.gamma.max(1.0) as f64;
        let ev = g * (0.25 * (loud - 0.5 * k) + 0.25 * beat);
        self.audio_ev = [ev, self.audio_ev[0]];
        let levels = if self.fx.levels { self.exposure } else { 0.0 };
        tone.exposure = (levels + self.audio_ev[0]).exp() as f32;
        let ppu = mid.scale * bw as f32;
        // a shader with just this frame's variations (once compiled), and
        // the ones coming up compiling in the background
        let ids = crate::sample::variation_ids(&self.samples);
        self.r.gpu.use_variations(&ids, fixed.is_some());
        if self.frame.is_multiple_of(15) {
            for mut set in self.show.upcoming() {
                // the look's 3D camera tilts every sheep
                if self.fx.tilt > 0 {
                    set.push(crate::sample::THREE_D);
                }
                self.r.gpu.prepare(&set);
            }
        }
        let per_round = gpu::WALKERS as f32 * gpu::ITERS as f32;
        let d = fixed
            .or(self.target_q.map(|q| ((q * self.rw as f32 * self.rh as f32) / per_round).ceil().clamp(1.0, 4096.0) as u32))
            .unwrap_or(self.dispatches.round().max(1.0) as u32);
        let st = self.r.step(&self.samples, tone, ppu * ppu, d, self.paused, false);
        if fixed.is_none() && self.target_q.is_none() {
            self.adapt();
        }
        self.levels(dt, mid.gamma);
        self.frame += 1;
        // the window optimizes itself once the first sheep's own shader runs
        // (or 5 s later at the latest)
        let now = Instant::now();
        if self.auto_optimize.is_some_and(|t| now >= t && (self.r.gpu.specialised() || now >= t + std::time::Duration::from_secs(5))) {
            self.auto_optimize = None;
            if self.opt.is_none() {
                self.optimize();
            }
        }
        self.optimize_tick();
        if let Some(s) = st {
            let k = 0.1;
            let secs = s.secs.max(1e-4);
            self.rate_gpu += (s.gpu_iters as f64 / secs - self.rate_gpu) * k;
            self.rate_cpu += (s.cpu_iters as f64 / secs - self.rate_cpu) * k;
            self.fps += (1.0 / secs - self.fps) * k;
            self.quality = self.r.total_iters() / (self.rw as f64 * self.rh as f64);
        }
        st
    }

    /// Auto-exposure: moves the brightness so the lit pixels' 90th
    /// percentile comes near LEVEL_TARGET, without saturating more than
    /// LEVEL_MAX_SATURATED of them, settling over a second or two.
    fn levels(&mut self, dt: f64, gamma: f32) {
        let Some(st) = self.r.gpu.frame_stats.take() else { return };
        if !self.fx.levels {
            self.exposure = 0.0;
            return;
        }
        // a nearly black frame (an edge through darkness) says little
        if st.lit < st.pixels / 500 {
            return;
        }
        let p90 = st.quantile(0.9).max(0.02) as f64;
        let sat = st.saturated as f64 / st.lit as f64;
        // brightness b moves the picture about as b^(1 / gamma): the sheep's
        // own level, without the exposure, and the exposure that would take
        // it to the target (none inside the dead band)
        let g = gamma.max(1.0) as f64;
        let own = p90 * (-(self.exposure + self.audio_ev[1]) / g).exp();
        let e = (LEVEL_TARGET / own).ln();
        let mut want = g * e.signum() * (e.abs() - LEVEL_DEADBAND).max(0.0);
        if sat > LEVEL_MAX_SATURATED {
            want = want.min(self.exposure - g * 0.5 * (sat / LEVEL_MAX_SATURATED).ln());
        }
        let step = if self.exposure_unset { want - self.exposure } else { ((want - self.exposure) * dt * 2.0).clamp(-0.3, 0.3) };
        self.exposure_unset = false;
        self.exposure = (self.exposure + step).clamp(-(8f64.ln()), 16f64.ln());
    }

    /// The music's levels for the look (zeros unless audio is on): the
    /// soundtrack's at this point of the video, else what the speakers play
    /// (parec runs only while audio is on).
    fn audio_levels(&mut self) -> [f32; 6] {
        if self.fx.audio == 0 {
            self.audio = None;
            return [0.0; 6];
        }
        let l = match &self.track {
            Some((tl, hop)) => tl.get((self.track_t / hop) as usize).copied().unwrap_or_default(),
            None => {
                if self.audio.is_none() {
                    match crate::audio::Audio::start() {
                        Ok(a) => self.audio = Some(a),
                        Err(e) => {
                            eprintln!("fleece: no audio: {e}");
                            self.fx.audio = 0;
                            return [0.0; 6];
                        }
                    }
                }
                self.audio.as_ref().unwrap().levels()
            }
        };
        [l.rms, l.bands[0], l.bands[1], l.bands[2], l.bands[3], l.beat]
    }

    /// Keeps the GPU's iterations at about 85 % of the display's frame time.
    fn adapt(&mut self) {
        if let Some((ms, d)) = self.r.gpu.last_ms {
            if d > 0 && ms > 0.0 {
                let per = ms / d as f32;
                let target = (self.period * 1000.0) as f32 * self.budget - 1.5;
                let want = (target / per).clamp(1.0, self.max_dispatches as f32);
                // move a third of the way: the measurement is a few frames old
                self.dispatches += (want - self.dispatches) / 3.0;
            }
        }
    }

    pub fn present(&mut self) {
        // right after a resize nothing is finished yet: the screen keeps the last picture
        if !self.r.gpu.has_frame {
            return;
        }
        let (dw, dh) = self.screen();
        if !self.headless && self.show.cur.id != self.titled {
            self.titled = self.show.cur.id;
            self.win.set_title(&format!("fleece: sheep {} ({})", self.show.cur.id, self.show.cur.genome.name));
        }
        // hidden, the HUD still shows that a recording is on
        let shown = self.hud || self.rec.is_some();
        if !self.hud && shown && (self.hud_dirty || self.hud_at.elapsed().as_secs_f64() > 0.25) {
            let h = Hud::new(&self.rec_line().into_iter().collect::<Vec<_>>());
            self.r.gpu.set_hud(h.w, h.h, &h.px);
            self.hud_at = Instant::now();
            self.hud_dirty = false;
        }
        if self.hud && (self.hud_dirty || self.hud_at.elapsed().as_secs_f64() > 0.25) {
            // wrapped to the window, so nothing is cut off
            let max = ((dw / (dh / 540).max(1)).saturating_sub(40) / 6) as usize;
            let mut rows: Vec<Vec<Item>> = self
                .hud_lines()
                .into_iter()
                .flat_map(|l| {
                    let (style, text) = match l.chars().next() {
                        Some('~') => (Style::Dim, l[1..].to_string()),
                        Some('*') => (Style::Red, l[1..].to_string()),
                        _ => (Style::Plain, l),
                    };
                    wrap(&text, max).into_iter().map(move |w| vec![Item::text(w.trim_start(), style)])
                })
                .collect();
            rows.extend(self.button_rows());
            let hover = self.mouse.and_then(|(x, y)| self.hud_point(x, y));
            let h = Hud::panel(&rows, max as u32 * 6, hover);
            self.r.gpu.set_hud(h.w, h.h, &h.px);
            self.buttons = h.buttons;
            self.hud_at = Instant::now();
            self.hud_dirty = false;
        }
        // full screen hides the mouse pointer until the mouse moves
        let cursor = !self.win.is_fullscreen() || self.pointer_until.is_some_and(|t| Instant::now() < t);
        if !self.headless && cursor != self.cursor {
            self.win.show_cursor(cursor);
            self.cursor = cursor;
        }
        self.record_frame();
        let s = (dh / 540).max(1);
        // the picture's shape kept, with black bars, as Electric Sheep's player shows its 800x592 videos
        let (ar, aw) = (self.rw as f32 / self.rh as f32, dw as f32 / dh as f32);
        let (w, h) = if ar > aw { (2.0, 2.0 * aw / ar) } else { (2.0 * ar / aw, 2.0) };
        self.r.gpu.present(dw, dh, [-w / 2.0, -h / 2.0, w, h], shown.then_some(s));
        if !self.headless {
            self.win.swap();
        }
    }

    fn hud_lines(&self) -> Vec<String> {
        let st = self.show.status();
        let phase = match st.edge {
            Some(e) => format!("EDGE {:.0}%", e * 100.0),
            None => format!("LOOP {}/{}", st.loop_no.min(show::LOOPS), show::LOOPS),
        };
        let rating = if st.rating != 0 { format!("  RATING {:+}", st.rating) } else { String::new() };
        let rating = match st.taste {
            Some(t) => format!("{rating}  TASTE {:.0}%", t * 100.0),
            None => rating,
        };
        let g = |v: f64| if v >= 1e9 { format!("{:.2} G", v / 1e9) } else { format!("{:.0} M", v / 1e6) };
        vec![
            format!("SHEEP {}  {}  {}{}", st.id, st.name, phase, rating),
            credit(&st.name, &st.nick, st.nc),
            format!(
                "GPU {}/S{}   CPU {}/S ({} THREADS{})   QUALITY {:.0}{}   {:.0} FPS   {}X{}{}",
                g(self.rate_gpu),
                if self.r.use_gpu { "" } else { " (OFF)" },
                g(self.rate_cpu),
                self.r.workers(),
                if self.r.use_cpu { "" } else { ", OFF" },
                self.quality,
                if self.target_q.is_some() { " (FIXED)" } else { "" },
                self.fps,
                self.rw,
                self.rh,
                if self.fixed == Some(crate::ES_SIZE) { " (ELECTRIC SHEEP)" } else { "" }
            ),
            format!(
                "HISTORY {}/{}   FLOCK {}   BRED {}  REJECTED {}   SHADER {}   SPEED {}X{}{}{}",
                self.show.place().0,
                self.show.place().1,
                st.flock,
                st.tested - st.rejected,
                st.rejected,
                if self.r.gpu.specialised() { "TUNED" } else { "GENERAL" },
                SPEEDS[self.speed],
                if self.blur { "" } else { "   NO BLUR" },
                if self.show.mutating() { "   BREEDING A MUTATION" } else { "" },
                if self.paused { "   PAUSED (REFINING)" } else { "" }
            ),
        ]
        .into_iter()
        .chain(self.opt.as_ref().map(|o| {
            let fps = (self.frame - o.frame0) as f64 / o.t0.elapsed().as_secs_f64().max(0.1);
            match o.step {
                Some(i) => format!("*OPTIMIZING FOR {:.0} FPS: TRYING {} ({fps:.0} FPS)", 1.0 / self.period, OPT_STEPS[i]),
                None => format!("*OPTIMIZING FOR {:.0} FPS: MEASURING ({fps:.0} FPS)", 1.0 / self.period),
            }
        }))
        .chain(self.opt_note.as_ref().filter(|n| n.1.elapsed().as_secs() < 10).map(|n| format!("*{}", n.0)))
        .chain(self.rec_line())
        .collect()
    }

    /// The HUD's line while recording.
    fn rec_line(&self) -> Option<String> {
        self.rec.as_ref().map(|r| {
            format!(
                "*REC {}:{:02}  {}{}  CREDIT {}  (R TO STOP)",
                r.seconds() as u64 / 60,
                r.seconds() as u64 % 60,
                r.path,
                if r.has_sound() { "  WITH SOUND" } else { "" },
                if self.burn_credit { "ON" } else { "OFF (METADATA ONLY)" }
            )
        })
    }

    /// The HUD's buttons, each labelled with its key and lit when its setting
    /// is on: the show, v2's effects, the renderer.
    fn button_rows(&self) -> Vec<Vec<Item>> {
        let f = &self.fx;
        let classic = *f == Fx::CLASSIC;
        let lines = |n: u32| self.fixed.is_none() && self.rh == n;
        vec![
            vec![
                Item::button("LEFT", "PREVIOUS", '<', false),
                Item::button("RIGHT", "NEXT", '>', false),
                Item::button("UP", "LIKE", '^', false),
                Item::button("DOWN", "DISLIKE", '_', false),
                Item::button("M", "MUTATE", 'm', self.show.mutating()),
                Item::button("P", "PAUSE", 'p', self.paused),
                Item::button("-", "SLOWER", '-', false),
                Item::text(format!("SPEED {}X", SPEEDS[self.speed]), Style::Dim),
                Item::button("+", "FASTER", '+', false),
                Item::button("R", "RECORD", 'r', self.rec.is_some()),
                Item::button("S", "SAVE", 's', false),
                Item::button("F", "FULLSCREEN", 'f', !self.headless && self.win.is_fullscreen()),
                Item::button("H", "HIDE", 'h', false),
                Item::button("ESC", "QUIT", 'q', false),
            ],
            vec![
                Item::button("V", if classic { "CLASSIC" } else { "V2" }, 'v', !classic),
                Item::button("O", format!("GLOW {}", ["OFF", "SUBTLE", "SOFT", "STRONG"][f.glow as usize % 4]), 'o', f.glow > 0),
                Item::button("T", format!("TRAILS {}", ["OFF", "SHORT", "LONG"][f.trails as usize % 3]), 't', f.trails > 0),
                Item::button("X", format!("SUPERSAMPLE {}", self.ss_label()), 'x', self.r.gpu.os > 1),
                Item::button("L", if f.levels { format!("LEVELS {:+.1}", self.exposure / 2f64.ln()) } else { "LEVELS".into() }, 'l', f.levels),
                Item::button("D", "DRIFT", 'd', f.drift),
                Item::button("U", "HUE", 'u', f.hue),
                Item::button("W", "WOBBLE", 'w', f.wobble),
                Item::button("3", ["3D", "3D TILT", "3D TILT+DOF"][f.tilt as usize % 3], '3', f.tilt > 0),
                Item::button("A", format!("AUDIO {}", ["OFF", "LOW", "MEDIUM", "HIGH"][f.audio as usize % 4]), 'a', f.audio > 0),
                Item::button(
                    "N",
                    match &self.music {
                        Some(p) => format!("MUSIC {}", crate::music::GENRES[p.genre()]),
                        None => "MUSIC".into(),
                    },
                    'n',
                    self.music.is_some(),
                ),
            ],
            vec![
                Item::button("C", "CPU", 'c', self.r.use_cpu && self.r.workers() > 0),
                Item::button("G", "GPU", 'g', self.r.use_gpu),
                Item::button("B", "BLUR", 'b', self.blur),
                Item::button("E", "800X592", 'e', self.fixed == Some(crate::ES_SIZE)),
                Item::button("1", "1080P", '1', lines(1080)),
                Item::button("2", "2K", '2', lines(1440)),
                Item::button("4", "4K", '4', lines(2160)),
                Item::button("K", "CREDIT IN RECORDINGS", 'k', self.burn_credit),
                Item::button("Z", format!("OPTIMIZE FOR {:.0} FPS", 1.0 / self.period), 'z', self.opt.is_some()),
            ],
        ]
    }

    /// Drawable pixel (`x`, `y`) on the HUD overlay, if past its corner.
    fn hud_point(&self, x: f64, y: f64) -> Option<(u32, u32)> {
        let (_, dh) = self.screen();
        // as present() draws it: 12 pixels in, `s` drawable pixels per overlay pixel
        let s = (dh / 540).max(1) as f64;
        let (hx, hy) = ((x - 12.0) / s, (y - 12.0) / s);
        (hx >= 0.0 && hy >= 0.0).then_some((hx as u32, hy as u32))
    }

    /// The window's mouse coordinates in drawable pixels.
    fn to_drawable(&self, (x, y): (i32, i32)) -> (f64, f64) {
        let ((dw, dh), (ww, wh)) = (self.win.drawable_size(), self.win.window_size());
        (x as f64 * dw as f64 / ww as f64, y as f64 * dh as f64 / wh as f64)
    }

    /// The mouse moved to drawable pixel `p`: the pointer shows for a while
    /// (in fullscreen), and the button under it lights.
    fn mouse_moved(&mut self, p: (f64, f64)) {
        self.mouse = Some(p);
        self.pointer_until = Some(Instant::now() + std::time::Duration::from_secs(3));
        let over = if self.hud { self.hud_point(p.0, p.1).and_then(|(x, y)| Hud::hit(&self.buttons, x, y)) } else { None };
        if over != self.hover {
            self.hover = over;
            self.hud_dirty = true;
        }
    }

    /// A left click at drawable pixel `p` (`clicks` 2: a double click): a
    /// button does what its key does (none while the HUD is hidden: only H
    /// brings it back), a double click elsewhere toggles fullscreen. False
    /// to quit.
    pub fn click(&mut self, p: (f64, f64), clicks: u8) -> bool {
        self.mouse_moved(p);
        if let Some(c) = self.hud_point(p.0, p.1).and_then(|(x, y)| Hud::hit(&self.buttons, x, y)).filter(|_| self.hud) {
            return self.key(c);
        }
        if clicks == 2 {
            self.toggle_fullscreen();
        }
        true
    }

    /// Where the button for key `c` is (drawable pixels), for testing clicks.
    pub fn button_centre(&self, c: char) -> Option<(f64, f64)> {
        let (_, dh) = self.screen();
        let s = (dh / 540).max(1) as f64;
        let b = self.buttons.iter().find(|b| b.action == c)?;
        Some((12.0 + s * (b.x0 + b.x1) as f64 / 2.0, 12.0 + s * (b.y0 + b.y1) as f64 / 2.0))
    }

    fn ss_label(&self) -> String {
        format!("{}X{}", self.r.gpu.os, if self.supersample == 0 { " (AUTO)" } else { "" })
    }

    /// Handles the window's events; false to quit.
    pub fn events(&mut self) -> bool {
        while let Some(ev) = self.win.poll() {
            match ev.kind {
                sdl::SDL_QUIT => return false,
                sdl::SDL_WINDOWEVENT if ev.window_event() == sdl::SDL_WINDOWEVENT_SIZE_CHANGED => self.fit(),
                sdl::SDL_MOUSEMOTION => {
                    let p = self.to_drawable(ev.mouse_xy());
                    self.mouse_moved(p);
                }
                sdl::SDL_MOUSEBUTTONDOWN => {
                    let (button, clicks) = ev.button();
                    if button == sdl::SDL_BUTTON_LEFT && !self.click(self.to_drawable(ev.mouse_xy()), clicks) {
                        return false;
                    }
                }
                sdl::SDL_MOUSEWHEEL => {
                    let y = ev.wheel_y();
                    if y > 0 {
                        self.speed = (self.speed + 1).min(SPEEDS.len() - 1);
                    } else if y < 0 {
                        self.speed = self.speed.saturating_sub(1);
                    }
                }
                sdl::SDL_KEYDOWN => {
                    let k = ev.key();
                    if ev.repeat() && !matches!(k, sdl::K_KP_MINUS | sdl::K_KP_PLUS) && k != b'-' as i32 && k != b'=' as i32 {
                        continue;
                    }
                    match k {
                        sdl::K_ESCAPE => return false,
                        sdl::K_UP => {
                            self.show.vote(1);
                            self.save_flock();
                        }
                        sdl::K_DOWN => {
                            self.show.vote(-1);
                            self.save_flock();
                        }
                        sdl::K_LEFT => self.show.back(),
                        sdl::K_RIGHT => self.show.skip(),
                        sdl::K_KP_MINUS => self.speed = self.speed.saturating_sub(1),
                        sdl::K_KP_PLUS => self.speed = (self.speed + 1).min(SPEEDS.len() - 1),
                        sdl::K_F11 => self.toggle_fullscreen(),
                        // letters, digits and the rest arrive as their ASCII codes
                        k if (32..127).contains(&k) => {
                            if !self.key(k as u8 as char) {
                                return false;
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        true
    }

    fn toggle_fullscreen(&mut self) {
        self.win.toggle_fullscreen();
        self.fit();
    }

    /// Matches the render size to the window (unless it is fixed).
    fn fit(&mut self) {
        let (dw, dh) = self.win.drawable_size();
        let (rw, rh) = self.fixed.unwrap_or_else(|| render_size(dw, dh, self.scale));
        self.set_render(rw, rh);
        if let Some(hz) = self.win.refresh_hz() {
            self.period = 1.0 / hz;
        }
    }

    /// The drawable's size (headless: the hidden window's, or the picture's).
    fn screen(&self) -> (u32, u32) {
        if self.headless {
            let (w, h) = self.win.drawable_size();
            if w > 1 && h > 1 { (w, h) } else { (self.rw, self.rh) }
        } else {
            self.win.drawable_size()
        }
    }

    /// Renders `lines` tall at the window's shape (1080p, 2K, 4K), whatever the
    /// window's own size: smaller windows get a sharper picture, larger
    /// ones an upscaled one.
    pub fn render_lines(&mut self, lines: u32) {
        self.fixed = None;
        let (dw, dh) = if self.headless { (self.rw, self.rh) } else { self.win.drawable_size() };
        self.scale = lines as f32 / dh.max(1) as f32;
        let (rw, rh) = render_size(dw, dh, self.scale);
        self.set_render(rw, rh);
    }

    fn set_render(&mut self, rw: u32, rh: u32) {
        if (rw, rh) != (self.rw, self.rh) {
            // a video keeps one frame size: a new size ends the recording
            if self.rec.is_some() {
                self.stop_recording();
            }
            // no worker may still be writing into the histograms being freed
            self.r.quiesce();
            fit_histogram(&mut self.r.gpu, rw, rh, self.supersample);
            self.rw = rw;
            self.rh = rh;
        }
    }

    /// Z: starts the optimizer (or stops it).
    fn optimize(&mut self) {
        if self.opt.take().is_some() {
            self.opt_note = Some(("OPTIMIZE STOPPED".into(), Instant::now()));
            return;
        }
        let before = self.settings();
        self.opt = Some(Optimize { step: None, before, fps_before: 0.0, frame0: self.frame, t0: Instant::now(), kept: Vec::new() });
        self.opt_note = None;
    }

    /// The optimizer's frame rate goal: the display's (a little under, as
    /// measured rates wobble).
    fn opt_target(&self) -> f64 {
        0.97 / self.period
    }

    fn settings(&self) -> Settings {
        Settings {
            budget: self.budget,
            supersample: self.supersample,
            use_cpu: self.r.use_cpu,
            fixed: self.fixed,
            scale: self.scale,
            rw: self.rw,
            rh: self.rh,
        }
    }

    fn restore(&mut self, s: Settings) {
        self.budget = s.budget;
        self.r.use_cpu = s.use_cpu;
        if (s.supersample, s.rw, s.rh) != (self.supersample, self.rw, self.rh) {
            (self.fixed, self.scale, self.supersample) = (s.fixed, s.scale, s.supersample);
            self.r.quiesce();
            fit_histogram(&mut self.r.gpu, s.rw, s.rh, s.supersample);
            (self.rw, self.rh) = (s.rw, s.rh);
        }
    }

    /// Applies optimizer step `i` if it can change anything.
    fn apply_step(&mut self, i: usize) -> bool {
        let mut budget = |b: f32| {
            let lower = self.budget > b + 0.01;
            if lower {
                self.budget = b;
            }
            lower
        };
        match i {
            0 => budget(0.70),
            1 if self.r.gpu.os > 1 => {
                self.supersample = 1;
                self.r.quiesce();
                fit_histogram(&mut self.r.gpu, self.rw, self.rh, 1);
                true
            }
            2 => budget(0.55),
            // a new size would end a recording
            3 if self.rec.is_none() && self.fixed.is_none() && self.rh > 1080 => {
                self.render_lines(1080);
                true
            }
            4 if self.r.use_cpu && self.r.use_gpu && self.r.workers() > 0 => {
                self.r.use_cpu = false;
                true
            }
            5 => budget(0.40),
            _ => false,
        }
    }

    /// Once a step has run OPT_SETTLE seconds: keeps it if the frame rate
    /// rose (or reached the goal), else undoes it; then the next step, until
    /// the goal or the last step.
    fn optimize_tick(&mut self) {
        let Some(o) = &self.opt else { return };
        let secs = o.t0.elapsed().as_secs_f64();
        if secs < OPT_SETTLE {
            return;
        }
        let fps = (self.frame - o.frame0) as f64 / secs;
        let target = self.opt_target();
        let mut o = self.opt.take().unwrap();
        match o.step {
            Some(i) if fps >= o.fps_before + 1.0 || fps >= target => {
                o.kept.push(OPT_STEPS[i]);
                o.fps_before = fps;
            }
            Some(_) => self.restore(o.before),
            None => o.fps_before = fps,
        }
        let mut next = o.step.map_or(0, |i| i + 1);
        while o.fps_before < target && next < OPT_STEPS.len() {
            o.before = self.settings();
            if self.apply_step(next) {
                (o.step, o.frame0, o.t0) = (Some(next), self.frame, Instant::now());
                self.opt = Some(o);
                return;
            }
            next += 1;
        }
        let what = if o.kept.is_empty() { "NOTHING CHANGED".to_string() } else { o.kept.join(", ") };
        let note = if o.fps_before >= target {
            format!("OPTIMIZED: {:.0} FPS ({what})", o.fps_before)
        } else {
            format!("OPTIMIZED AS FAR AS IT GOES: {:.0} FPS ({what})", o.fps_before)
        };
        println!("fleece: {}", note.to_lowercase());
        self.opt_note = Some((note, Instant::now()));
    }

    /// N: music off, then each genre in turn, then off.
    fn cycle_music(&mut self) {
        match &self.music {
            None => self.start_music(0),
            Some(p) if p.genre() + 1 < crate::music::GENRES.len() => p.set_genre(p.genre() + 1),
            Some(_) => self.music = None,
        }
    }

    /// Plays generated music of `genre`, and turns audio on so the show follows it.
    pub fn start_music(&mut self, genre: usize) {
        if self.headless {
            // never sound from a run without a window
            eprintln!("fleece: no music without a window (a video takes --music as its soundtrack)");
            return;
        }
        match crate::music::Player::start(genre, self.music_seed) {
            Ok(p) => {
                self.music = Some(p);
                if self.fx.audio == 0 {
                    self.fx.audio = 2;
                }
            }
            Err(e) => eprintln!("fleece: no music: {e}"),
        }
    }

    /// X: the next supersampling setting (automatic, 1x, 2x, 4x).
    fn cycle_supersample(&mut self) {
        let i = SUPERSAMPLES.iter().position(|&s| s == self.supersample).unwrap_or(0);
        self.supersample = SUPERSAMPLES[(i + 1) % SUPERSAMPLES.len()];
        self.r.quiesce();
        fit_histogram(&mut self.r.gpu, self.rw, self.rh, self.supersample);
    }

    /// A key, in the window or from --keys: false to quit.
    pub fn key(&mut self, c: char) -> bool {
        let next = |v: u8, n: u8| (v + 1) % n;
        self.hud_dirty = true;
        match c {
            'q' => return false,
            ' ' | '>' => self.show.skip(),
            '<' => self.show.back(),
            // the HUD's like and dislike buttons (the arrow keys in the window)
            '^' | '_' => {
                self.show.vote(if c == '^' { 1 } else { -1 });
                self.save_flock();
            }
            '-' => self.speed = self.speed.saturating_sub(1),
            '=' | '+' => self.speed = (self.speed + 1).min(SPEEDS.len() - 1),
            'f' => self.toggle_fullscreen(),
            'm' => self.show.mutate_next(),
            'p' => self.paused = !self.paused,
            'b' => self.blur = !self.blur,
            'h' => self.hud = !self.hud,
            'c' => {
                if self.r.workers() > 0 {
                    self.r.use_cpu = !self.r.use_cpu || !self.r.use_gpu;
                }
            }
            'g' => self.r.use_gpu = !self.r.use_gpu || (!self.r.use_cpu || self.r.workers() == 0),
            's' => self.save(),
            'r' => self.toggle_recording(),
            'k' => self.burn_credit = !self.burn_credit,
            '1' => self.render_lines(1080),
            '2' => self.render_lines(1440),
            '4' => self.render_lines(2160),
            'e' => {
                self.fixed = Some(crate::ES_SIZE);
                self.set_render(crate::ES_SIZE.0, crate::ES_SIZE.1);
            }
            // v2's effects
            'v' => {
                if self.fx == Fx::CLASSIC {
                    self.fx = self.fx_saved;
                } else {
                    self.fx_saved = self.fx;
                    self.fx = Fx::CLASSIC;
                }
                // classic breeds from flam3's variations alone
                self.show.flock.lock().unwrap().new_variations = self.fx != Fx::CLASSIC;
            }
            'o' => self.fx.glow = next(self.fx.glow, 4),
            't' => self.fx.trails = next(self.fx.trails, 3),
            'x' => self.cycle_supersample(),
            'l' => self.fx.levels = !self.fx.levels,
            'd' => self.fx.drift = !self.fx.drift,
            'u' => self.fx.hue = !self.fx.hue,
            'w' => self.fx.wobble = !self.fx.wobble,
            '3' => self.fx.tilt = next(self.fx.tilt, 3),
            'a' => self.fx.audio = next(self.fx.audio, 4),
            'z' => self.optimize(),
            'n' => self.cycle_music(),
            _ => {}
        }
        true
    }

    /// R: starts recording the show (the picture without the controls, with
    /// the sheep's credit) to ~/Videos, or stops and finishes the file.
    pub fn toggle_recording(&mut self) {
        if self.rec.is_some() {
            self.stop_recording();
            return;
        }
        let dir = std::env::var("HOME").map(|h| format!("{h}/Videos")).unwrap_or_else(|_| ".".into());
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let path = format!("{dir}/fleece-{t}.mp4");
        // the video runs at the display's rate, one frame per frame shown
        let fps = (1.0 / self.period).round().max(1.0);
        match crate::video::Recorder::start(&path, self.rw, self.rh, fps) {
            Ok(r) => {
                println!("recording to {path}");
                self.rec = Some(r);
            }
            Err(e) => eprintln!("fleece: cannot record: {e}"),
        }
    }

    pub fn stop_recording(&mut self) {
        if let Some(mut r) = self.rec.take() {
            // the picture follows the sound a frame late (a frame is shown
            // once the next has started) plus the analysis' own lag
            r.sound_delay = self.sound_delay.unwrap_or(1.0 / self.fps.max(1.0) + ANALYSIS_LAG);
            let (path, frames, secs, fps) = (r.path.clone(), r.frames, r.seconds(), r.fps);
            match r.stop() {
                Ok(()) => println!("recorded {path}: {secs:.1} s, {frames} frames at {fps} fps (credits in its metadata and {path}.credits.txt)"),
                Err(e) => eprintln!("fleece: the recording failed: {e}"),
            }
        }
    }

    /// Every frame while recording: the picture, the credit burned in
    /// unless turned off, the sheep noted for the file's credits.
    fn record_frame(&mut self) {
        if self.rec.is_none() {
            return;
        }
        let mut rgb = self.r.gpu.read_rgb();
        let st = self.show.status();
        if self.burn_credit {
            let t = self.rec.as_ref().map_or(0.0, |r| r.seconds());
            let fade = credit_fade(t, self.credit_secs);
            stamp_credit(&mut rgb, self.rw as usize, self.rh as usize, &st.name, &st.nick, st.nc, fade);
        }
        if let Some(rec) = self.rec.as_mut() {
            rec.credits.add(&st.name, &st.nick, st.nc);
            // with sound, as many copies as keep the picture with it
            let n = rec.due();
            for _ in 1..n {
                rec.offer(rgb.clone());
            }
            if n > 0 {
                rec.offer(rgb);
            }
        }
    }

    /// The animation time a frame advances: the wall clock, or exactly one
    /// display frame while recording without sound (so the video has every
    /// frame; with sound it keeps to the clock, as the sound does).
    pub fn step_time(&self, wall: f64) -> f64 {
        if self.rec.as_ref().is_some_and(|r| !r.has_sound()) { self.period } else { wall }
    }

    /// The flock and its ratings, for next time (the window only).
    pub fn save_flock(&self) {
        if self.headless {
            return;
        }
        let (snap, taste) = {
            let f = self.show.flock.lock().unwrap();
            (f.snapshot(), f.taste.clone())
        };
        if let Err(e) = crate::flam3::write_flock(&crate::flock_path(), &snap) {
            eprintln!("fleece: could not save the flock: {e}");
        }
        if let Err(e) = taste.save(&crate::taste_path()) {
            eprintln!("fleece: could not save the taste: {e}");
        }
    }

    /// The frame on screen as a PNG and its genome as flam3 XML, in ~/Pictures/fleece.
    pub fn save(&mut self) {
        let dir = std::env::var("HOME").map(|h| format!("{h}/Pictures/fleece")).unwrap_or_else(|_| ".".into());
        let _ = std::fs::create_dir_all(&dir);
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let base = format!("{dir}/sheep-{t}-{}", self.show.cur.id);
        let rgb = self.r.gpu.read_rgb();
        let ok = std::fs::write(format!("{base}.png"), png::encode(self.rw, self.rh, &rgb)).is_ok()
            && std::fs::write(format!("{base}.flam3"), self.show.current(&self.look).to_flam3(self.rw, self.rh)).is_ok();
        println!("{} {base}.png / .flam3", if ok { "saved" } else { "could not save" });
    }
}

/// The histogram size for a `dw` x `dh` drawable: `scale` times it, or with
/// scale 0 the drawable capped at 1080 lines (a 4K screen gets a 2x upscale
/// and four times the samples per pixel).
/// The credit for a sheep, in parts. Electric Sheep's reuse policy: "artwork
/// by Scott Draves and the Electric Sheep", the licence (evolved sheep CC
/// BY-NC, designed ones CC BY) and the designer; fleece's own sheep credit flam3.
pub fn credit_parts(name: &str, nick: &str, nc: bool) -> Vec<String> {
    if name.contains("electricsheep.") {
        let lic = if nc { "CC BY-NC" } else { "CC BY" };
        let by = if nick.is_empty() { String::new() } else { format!("  DESIGN BY {nick}") };
        vec!["ARTWORK BY SCOTT DRAVES AND THE ELECTRIC SHEEP".into(), format!("ELECTRICSHEEP.ORG  {lic}{by}")]
    } else {
        vec!["FRACTAL FLAMES: FLAM3 BY SCOTT DRAVES (GPL)".into(), "NEW SHEEP BRED BY FLEECE".into()]
    }
}

pub fn credit(name: &str, nick: &str, nc: bool) -> String {
    credit_parts(name, nick, nc).join("  ")
}

/// Word-wraps `text` to `max` characters a line (a word longer than a line is cut).
fn wrap(text: &str, max: usize) -> Vec<String> {
    let max = max.max(8);
    let mut out = Vec::new();
    let mut line = String::new();
    for word in text.split(' ') {
        let word: String = word.chars().take(max).collect();
        if !line.is_empty() && line.len() + 1 + word.len() > max {
            out.push(std::mem::take(&mut line));
        }
        if !line.is_empty() || word.is_empty() {
            line.push(' ');
        }
        line.push_str(&word);
    }
    if !line.trim().is_empty() {
        out.push(line);
    }
    out
}

/// The credit burned into a video frame: the sheep's name (its generation and
/// number, which the policy asks for), then the credit, wrapped to the frame.
/// How strongly the burned-in credit shows `t` seconds into a video: fully
/// for the first `secs`, then fading out over one second.
pub fn credit_fade(t: f64, secs: f64) -> f32 {
    (1.0 - (t - secs)).clamp(0.0, 1.0) as f32
}

fn stamp_credit(rgb: &mut [u8], w: usize, h: usize, name: &str, nick: &str, nc: bool, fade: f32) {
    if fade <= 0.0 {
        return;
    }
    let s = if h >= 1000 { 2 } else { 1 };
    let max = (w.saturating_sub(24) / (6 * s)).max(8);
    let mut lines = Vec::new();
    if name.contains("electricsheep.") {
        lines.extend(wrap(name, max));
    }
    for p in credit_parts(name, nick, nc) {
        lines.extend(wrap(&p, max));
    }
    stamp(rgb, w, h, &lines, fade);
}

/// Draws `lines` on a dim panel at the bottom left of an RGB8 frame (twice
/// the size from 1080 lines up).
fn stamp(rgb: &mut [u8], w: usize, h: usize, lines: &[String], fade: f32) {
    let hud = Hud::new(lines);
    let s = if h >= 1000 { 2 } else { 1 };
    let (x0, y0) = (8usize, h.saturating_sub(hud.h as usize * s + 8));
    for y in 0..hud.h as usize * s {
        for x in 0..hud.w as usize * s {
            let (px, py) = (x0 + x, y0 + y);
            if px >= w || py >= h {
                continue;
            }
            let src = &hud.px[((y / s) * hud.w as usize + x / s) * 4..][..4];
            let a = (src[3] as f32 * fade) as u32;
            let d = &mut rgb[(py * w + px) * 3..][..3];
            for k in 0..3 {
                d[k] = ((src[k] as u32 * a + d[k] as u32 * (255 - a)) / 255) as u8;
            }
        }
    }
}

/// Sizes the histogram for a `rw` x `rh` picture at supersampling `ss` (0:
/// automatic), stepping down when the video memory runs short.
pub fn fit_histogram(gpu: &mut Gpu, rw: u32, rh: u32, ss: u32) {
    let mut os = match ss {
        // 2x while the histogram (16 bytes a bucket) fits the GPU's cache;
        // past that the scattered atomics slow to memory speed
        0 if (rw * 2) as u64 * (rh * 2) as u64 * 16 <= 40 << 20 => 2,
        0 => 1,
        s => s.clamp(1, 4),
    };
    loop {
        match gpu.resize(rw, rh, os) {
            Ok(()) => return,
            Err(e) if os > 1 => {
                eprintln!("fleece: {e}; trying {}x supersampling", os / 2);
                os /= 2;
            }
            Err(e) => {
                eprintln!("fleece: {e}");
                std::process::exit(1)
            }
        }
    }
}

fn render_size(dw: u32, dh: u32, scale: f32) -> (u32, u32) {
    let s = if scale > 0.0 { scale } else { (1080.0 / dh as f32).min(1.0) };
    (((dw as f32 * s).round() as u32).max(64), ((dh as f32 * s).round() as u32).max(36))
}

/// The interactive window.
pub fn window(o: &Opts) {
    let mut app = App::new(o, false);
    if o.optimize {
        app.auto_optimize = Some(Instant::now() + std::time::Duration::from_secs(3));
    }
    if let Some(g) = o.music {
        app.start_music(g);
    }
    let mut last = Instant::now();
    while app.events() {
        let now = Instant::now();
        let dt = (now - last).as_secs_f64().min(0.25);
        last = now;
        app.frame(app.step_time(dt), None);
        app.present();
    }
    // the music's sound device closes before the window takes SDL down
    app.music = None;
    app.stop_recording();
    app.r.idle();
    app.save_flock();
}

/// The same loop with no display: `frames` frames at `o.fps`, stats on
/// stdout, every `o.every`-th frame saved as a PNG in `o.out`.
pub fn headless(o: &Opts) {
    let mut app = App::new(o, true);
    app.hud = o.hud;
    let dt = 1.0 / o.fps;
    let t0 = Instant::now();
    for f in 0..o.frames {
        for k in &o.keys {
            if k.0 == f {
                match k.1 {
                    ' ' | '>' => app.show.skip(),
                    '<' => app.show.back(),
                    'm' => app.show.mutate_next(),
                    '+' => app.show.vote(1),
                    '-' => app.show.vote(-1),
                    // a resize mid-run, as the window's F key does
                    'r' => {
                        let (w, h) = (app.rw, app.rh);
                        app.set_render(w * 3 / 4, h * 3 / 4);
                    }
                    'R' => app.toggle_recording(),
                    // other capitals click the HUD's button for that key
                    c if c.is_ascii_uppercase() => match app.button_centre(c.to_ascii_lowercase()) {
                        Some(p) => {
                            println!("click {c} at {:.0},{:.0}", p.0, p.1);
                            app.click(p, 1);
                        }
                        None => println!("click {c}: no button"),
                    },
                    c => {
                        app.key(c);
                    }
                }
            }
        }
        let st = app.frame(dt, None);
        app.present();
        app.r.gpu.finish();
        if let Some(s) = st {
            if f % 10 == 0 || f + 1 == o.frames {
                println!(
                    "frame {f:4}: gpu {:5.0} M ({:4.1} ms, {:3.0} rounds) cpu {:4.0} M  quality {:4.0}  build {:.1} ms  {}",
                    s.gpu_iters as f64 / 1e6,
                    s.gpu_ms,
                    app.dispatches,
                    s.cpu_iters as f64 / 1e6,
                    app.quality,
                    app.build_ms,
                    app.hud_lines()[0]
                );
            }
        }
        if o.every > 0 && f % o.every == 0 && !o.out.is_empty() && app.r.gpu.has_frame {
            let (sw, sh) = app.screen();
            let (rgb, w, h) = if app.hud { (app.r.gpu.read_screen(sw, sh), sw, sh) } else { (app.r.gpu.read_rgb(), app.rw, app.rh) };
            let _ = std::fs::write(format!("{}/f{f:05}.png", o.out), png::encode(w, h, &rgb));
        }
    }
    app.stop_recording();
    app.r.idle();
    println!("{} frames in {:.1} s", o.frames, t0.elapsed().as_secs_f64());
    if !o.out.is_empty() {
        // the flock, as the window would keep it (never the user's file)
        let snap = app.show.flock.lock().unwrap().snapshot();
        let ratings: Vec<i32> = snap.iter().map(|s| s.1).collect();
        let path = format!("{}/flock.flam3", o.out);
        match crate::flam3::write_flock(&path, &snap) {
            Ok(()) => println!("flock of {} written to {path}, ratings {ratings:?}", snap.len()),
            Err(e) => eprintln!("fleece: {path}: {e}"),
        }
    }
}

/// A video: `frames` frames at `o.fps`, each iterated to `o.quality`
/// samples per pixel, into `o.out` (any ffmpeg format by its extension,
/// `.y4m` without ffmpeg, `-` for raw RGB24 on stdout).
pub fn video(o: &Opts) {
    let mut app = App::new(o, true);
    app.hud = false;
    let dt = 1.0 / o.fps;
    let per_round = gpu::WALKERS as f64 * gpu::ITERS as f64;
    let rounds = ((o.quality as f64 * app.rw as f64 * app.rh as f64) / per_round).ceil().max(1.0) as u32;
    let mut frames = o.frames;
    // generated music: rendered first (a second longer than the video), then
    // made to like any soundtrack, and deleted once in the file
    let made = o.music.filter(|_| o.soundtrack.is_none()).map(|g| {
        let dir = std::env::var("HOME").map(|h| format!("{h}/.cache/fleece")).unwrap_or_else(|_| ".".into());
        let _ = std::fs::create_dir_all(&dir);
        let path = format!("{dir}/music-{}.wav", std::process::id());
        let secs = frames as f64 / o.fps + 1.0;
        eprintln!("fleece: making {:.0} s of {} music", secs, crate::music::GENRES[g].to_lowercase());
        if let Err(e) = crate::music::render_wav(g, o.seed as u64, secs, &path) {
            eprintln!("fleece: music: {e}");
            std::process::exit(1)
        }
        path
    });
    let soundtrack = o.soundtrack.clone().or(made.clone());
    if let Some(f) = &soundtrack {
        match crate::audio::timeline(f) {
            Ok((tl, hop)) => {
                if frames == 0 {
                    frames = (tl.len() as f64 * hop * o.fps).round() as u64;
                }
                eprintln!("fleece: made to {f} ({:.1} s)", tl.len() as f64 * hop);
                app.track = Some((tl, hop));
            }
            Err(e) => {
                eprintln!("fleece: {e}");
                std::process::exit(1)
            }
        }
    }
    let mut sink = crate::video::Sink::open(&o.out, app.rw, app.rh, o.fps, o.crf, "slow", soundtrack.as_deref()).unwrap_or_else(|e| {
        eprintln!("fleece: {e}");
        std::process::exit(1)
    });
    eprintln!(
        "fleece: {} frames ({:.1} s at {} fps), {}x{}, {} samples per pixel, to {}",
        frames,
        frames as f64 / o.fps,
        o.fps,
        app.rw,
        app.rh,
        o.quality,
        if o.out == "-" { "stdout" } else { &o.out }
    );
    let t0 = Instant::now();
    let mut said = Instant::now();
    let mut credits = crate::video::Credits::default();
    // a few frames before the first, so auto-exposure has measured it (its
    // counts come two frames late) and trails have something to blend with
    for _ in 0..4 {
        app.track_t = 0.0;
        app.frame(0.0, Some(rounds));
        app.r.gpu.finish();
    }
    // one frame of latency: frame f is on screen after step f + 1
    for f in 0..=frames {
        for k in &o.keys {
            if k.0 == f {
                match k.1 {
                    ' ' | '>' => app.show.skip(),
                    '<' => app.show.back(),
                    'm' => app.show.mutate_next(),
                    // v2's effects (not sizes: the video keeps its frame)
                    c if "votldu3aw".contains(c) => {
                        app.key(c);
                    }
                    _ => {}
                }
            }
        }
        // the samples built now are the video's frame f, at f / fps
        app.track_t = f as f64 / o.fps;
        app.frame(if f == 0 { 0.0 } else { dt }, Some(rounds));
        app.r.gpu.finish();
        if f == 0 {
            continue;
        }
        app.present();
        let mut rgb = app.r.gpu.read_rgb();
        let st = app.show.status();
        credits.add(&st.name, &st.nick, st.nc);
        if o.credit {
            let fade = credit_fade((f - 1) as f64 / o.fps, o.credit_secs);
            stamp_credit(&mut rgb, app.rw as usize, app.rh as usize, &st.name, &st.nick, st.nc, fade);
        }
        if let Err(e) = sink.write(&rgb) {
            eprintln!("fleece: writing the video: {e}");
            break;
        }
        if said.elapsed().as_secs_f64() > 2.0 || f == frames {
            said = Instant::now();
            let el = t0.elapsed().as_secs_f64();
            let eta = el / f as f64 * (frames - f) as f64;
            eprint!("\rframe {f}/{}  {:.0} s elapsed, about {:.0} s to go  {}          ", frames, el, eta, app.show.cur.genome.name);
        }
    }
    eprintln!();
    app.r.idle();
    let done = sink.finish().and_then(|_| credits.embed(&o.out));
    if let Some(m) = &made {
        let _ = std::fs::remove_file(m);
    }
    match done {
        Ok(()) if o.out != "-" => eprintln!(
            "fleece: wrote {} in {:.0} s (credits in its metadata and {}.credits.txt)",
            o.out,
            t0.elapsed().as_secs_f64(),
            o.out
        ),
        Ok(()) => {}
        Err(e) => {
            eprintln!("fleece: {e}");
            std::process::exit(1)
        }
    }
}
