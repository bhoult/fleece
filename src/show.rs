//! The flock on screen, the way the Electric Sheep client plays it: each
//! sheep is a loop (its animated xforms turn 360 degrees in 128 frames at 23
//! fps) shown twice, then an edge (128 frames: both sheep keep turning while
//! one morphs into the other, eased with flam3's smoother) leads to the next
//! sheep. A breeder thread keeps new sheep coming from the flock.
//!
//! Every frame is several time samples spread over one Electric Sheep frame
//! (flam3's temporal samples): motion blur.

use crate::breed::{Flock, Sheep, first_good, mutate};
use crate::genome::Genome;
use crate::interp;
use crate::rng::Rng;
use crate::sample::Sample;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// Electric Sheep: loops and edges are 128 frames, played at 23 fps.
pub const ES_FRAMES: f64 = 128.0;
pub const ES_FPS: f64 = 23.0;
pub const LOOP_SECS: f64 = ES_FRAMES / ES_FPS;
/// Each loop plays twice before the walk moves on.
pub const LOOPS: u32 = 2;
/// Sheep remembered for going back.
const HISTORY: usize = 500;

/// What fleece adds to the sheep as they play (all off: Electric Sheep's
/// show). Each is a function of the show's own clock, so it runs on across
/// loops and edges without a jump.
#[derive(Clone, Copy, Debug, Default)]
pub struct Look {
    /// camera drift: a slow zoom, pan and turn
    pub drift: bool,
    /// palette drift: the hues turn
    pub hue: bool,
    /// play fleece's extra motion elements (motion for every sheep)
    pub wobble: bool,
    /// a 3D camera: 0 flat, 1 tilted in perspective, 2 that with depth of field
    pub tilt: u8,
    /// the music (all 0 without): loudness, bass, low mid, mid, high, beat
    pub audio: [f32; 6],
}

/// Seconds (of animation time) for the palette's hues to turn once.
const HUE_TURN: f64 = 96.0;

#[derive(Clone)]
enum Phase {
    Loop,
    /// the two sheep padded to the same xforms, and where `b` came from
    Edge { a: Box<Genome>, b: Box<Genome> },
}

pub struct Show {
    pub cur: Sheep,
    next: Option<Sheep>,
    phase: Phase,
    /// animation seconds: `turn` drives the rotation and never jumps; `seg`
    /// is the time into the current loop or edge
    turn: f64,
    seg: f64,
    loops_done: u32,
    ready: Arc<Mutex<VecDeque<Sheep>>>,
    pub flock: Arc<Mutex<Flock>>,
    quit: Arc<AtomicBool>,
    breeder: Option<JoinHandle<()>>,
    /// a mutation of the current sheep is being bred (M); the breeder
    /// takes the parent from `mut_req` and leaves the child in `mut_out`
    mutating: bool,
    mut_req: Arc<Mutex<Option<Genome>>>,
    mut_out: Arc<Mutex<Option<Sheep>>>,
    /// a step back (-1) or forward (+1) asked for during an edge, taken when it ends
    pending: Option<i32>,
    /// the sheep shown so far, oldest first, and where the show is in it
    history: Vec<Sheep>,
    pos: usize,
    /// the history entry the edge is heading for (None: a new sheep)
    next_pos: Option<usize>,
    pub shown: u32,
}

/// What the HUD says about the show.
pub struct Status {
    pub name: String,
    pub nick: String,
    pub nc: bool,
    pub id: u32,
    pub rating: i32,
    pub edge: Option<f32>,
    pub loop_no: u32,
    pub flock: usize,
    pub tested: u32,
    pub rejected: u32,
    /// its taste score, if measured
    pub taste: Option<f32>,
}

impl Show {
    /// `saved`: the flock from last time (sheep and ratings); `play`: flames
    /// to show first, in order (they join the flock rated 1).
    pub fn new(seed: u32, saved: Vec<(Genome, i32)>, play: Vec<Genome>, genebank: (Vec<Genome>, Vec<f32>)) -> Show {
        let mut flock = Flock::new(seed);
        (flock.genebank, flock.bank_weight) = genebank;
        for (g, rating) in saved {
            let id = flock.next_id();
            flock.add(Sheep::new(g, rating, id));
        }
        let mut queued: VecDeque<Sheep> = play
            .into_iter()
            .map(|g| {
                let id = flock.next_id();
                let s = Sheep::new(g, 1, id);
                flock.add(s.clone());
                s
            })
            .collect();
        let first = match queued.pop_front() {
            Some(s) => s,
            None => {
                let s = flock.opener();
                flock.add(s.clone());
                s
            }
        };
        let flock = Arc::new(Mutex::new(flock));
        let ready = Arc::new(Mutex::new(queued));
        let quit = Arc::new(AtomicBool::new(false));
        let mut_req = Arc::new(Mutex::new(None));
        let mut_out = Arc::new(Mutex::new(None));
        let breeder = {
            let (flock, ready, quit) = (flock.clone(), ready.clone(), quit.clone());
            let (req, out) = (mut_req.clone(), mut_out.clone());
            std::thread::Builder::new()
                .name("breeder".into())
                .spawn(move || breed_loop(flock, ready, quit, req, out))
                .ok()
        };
        let history = vec![first.clone()];
        Show {
            cur: first,
            next: None,
            phase: Phase::Loop,
            turn: 0.0,
            seg: 0.0,
            loops_done: 0,
            ready,
            flock,
            quit,
            breeder,
            mutating: false,
            mut_req,
            mut_out,
            pending: None,
            history,
            pos: 0,
            next_pos: None,
            shown: 1,
        }
    }

    /// Advances the animation by `dt` seconds of Electric Sheep time.
    pub fn advance(&mut self, dt: f64) {
        self.turn = (self.turn + dt) % (LOOP_SECS * 1e6);
        self.seg += dt;
        // a bred mutation starts its edge as soon as a loop is playing
        if self.mutating && matches!(self.phase, Phase::Loop) {
            let bred = self.mut_out.lock().unwrap().take();
            if let Some(s) = bred {
                self.mutating = false;
                self.begin_edge(s, None);
                self.seg = 0.0;
            }
        }
        while self.seg >= LOOP_SECS {
            self.seg -= LOOP_SECS;
            match self.phase {
                Phase::Loop => {
                    self.loops_done += 1;
                    // after going back, the show plays forward through the
                    // history; at its end, a new sheep (with nothing bred yet,
                    // or a mutation on its way, the loop plays again)
                    if self.loops_done >= LOOPS && !self.mutating {
                        if self.pos + 1 < self.history.len() {
                            let n = self.history[self.pos + 1].clone();
                            self.begin_edge(n, Some(self.pos + 1));
                        } else {
                            self.start_edge();
                        }
                    }
                }
                Phase::Edge { .. } => {
                    self.cur = self.next.take().expect("an edge has a next sheep");
                    self.phase = Phase::Loop;
                    self.loops_done = 0;
                    self.shown += 1;
                    match self.next_pos.take() {
                        Some(i) => self.pos = i,
                        None => {
                            // a new sheep: like a browser, it replaces anything ahead
                            self.history.truncate(self.pos + 1);
                            self.history.push(self.cur.clone());
                            if self.history.len() > HISTORY {
                                self.history.remove(0);
                            }
                            self.pos = self.history.len() - 1;
                        }
                    }
                    if let Some(d) = self.pending.take() {
                        self.step(d);
                    }
                }
            }
        }
    }

    /// Starts the edge to the next bred sheep. False if none is ready.
    fn start_edge(&mut self) -> bool {
        let n = self.ready.lock().unwrap().pop_front();
        let Some(n) = n else { return false };
        self.begin_edge(n, None);
        true
    }

    /// An edge to `n`: a history entry (`to`), or a new sheep (None).
    fn begin_edge(&mut self, n: Sheep, to: Option<usize>) {
        let (a, b) = interp::align(&self.cur.genome, &n.genome);
        self.phase = Phase::Edge { a: Box::new(a), b: Box::new(b) };
        self.next = Some(n);
        self.next_pos = to;
    }

    /// Next sheep: forward through the history, past its end a new sheep.
    /// The edge starts now, from the current turn.
    pub fn skip(&mut self) {
        self.step(1);
    }

    /// Previous sheep: back through the history.
    pub fn back(&mut self) {
        self.step(-1);
    }

    /// A step through the history (-1 back, +1 forward); during an edge it
    /// waits for the edge to end. False if there is nowhere to go.
    fn step(&mut self, d: i32) -> bool {
        if matches!(self.phase, Phase::Edge { .. }) {
            self.pending = Some(d);
            return true;
        }
        let to = self.pos as i64 + d as i64;
        let started = if to < 0 {
            false
        } else if (to as usize) < self.history.len() {
            let n = self.history[to as usize].clone();
            self.begin_edge(n, Some(to as usize));
            true
        } else {
            self.start_edge()
        };
        if started {
            self.seg = 0.0;
        }
        started
    }

    /// Where the show is in its history: (1-based position, length).
    pub fn place(&self) -> (usize, usize) {
        (self.pos + 1, self.history.len())
    }

    /// Has the breeder make a mutation of the sheep on screen (tested like
    /// any new sheep); it morphs in when ready.
    pub fn mutate_next(&mut self) {
        let parent = match (&self.phase, &self.next) {
            (Phase::Edge { .. }, Some(n)) => n.genome.clone(),
            _ => self.cur.genome.clone(),
        };
        self.mut_out.lock().unwrap().take();
        *self.mut_req.lock().unwrap() = Some(parent);
        self.mutating = true;
    }

    pub fn mutating(&self) -> bool {
        self.mutating
    }

    /// A vote for the sheep the HUD names: the one on screen, or during an
    /// edge the one coming in. A dislike moves on to another sheep (during an
    /// edge: as soon as it ends).
    pub fn vote(&mut self, v: i32) {
        let s = match (&self.phase, self.next.as_mut()) {
            (Phase::Edge { .. }, Some(n)) => n,
            _ => &mut self.cur,
        };
        s.rating += v;
        let id = s.id;
        // the taste learns from the vote (sheep from the flock file get measured now)
        let mut f = self.flock.lock().unwrap();
        if s.taste.is_none() {
            let seed = f.seed();
            s.measure(&f.taste, seed);
        }
        f.vote(id, v, s.taste.map(|t| t.0));
        drop(f);
        if v < 0 {
            if matches!(self.phase, Phase::Edge { .. }) {
                self.pending = Some(1);
            } else {
                self.skip();
            }
        }
    }

    /// The genome `off` seconds from now, and the angle to turn it by
    /// (fleece's extra motion played with `wobble`).
    fn genome_at(&self, off: f64, wobble: bool) -> (Genome, f32) {
        let phase = ((self.turn + off) / LOOP_SECS).fract();
        let angle = (2.0 * std::f64::consts::PI * phase) as f32;
        // flam3's sheep_loop / sheep_edge: motion elements first, at the loop's phase
        let moved = |g: &Genome| if g.has_motion() { g.moved(phase as f32, wobble) } else { g.clone() };
        match &self.phase {
            Phase::Loop => (moved(&self.cur.genome), angle),
            Phase::Edge { a, b } => {
                let t = interp::smoother(((self.seg + off) / LOOP_SECS) as f32);
                if a.has_motion() || b.has_motion() {
                    (interp::edge(&moved(a), &moved(b), angle, t), 0.0)
                } else {
                    (interp::edge(a, b, angle, t), 0.0)
                }
            }
        }
    }

    /// `n` time samples spread over `shutter` seconds around now, for a
    /// `w` x `h` histogram, dressed in `look`; returns the middle genome too
    /// (for the tone map).
    pub fn samples(&self, n: usize, shutter: f64, w: u32, h: u32, look: &Look, out: &mut Vec<Sample>) -> Genome {
        out.resize(n.max(1), Sample::default());
        let mut mid = None;
        for (i, s) in out.iter_mut().enumerate() {
            let off = if n > 1 { (i as f64 / (n - 1) as f64 - 0.5) * shutter } else { 0.0 };
            let (mut g, angle) = self.genome_at(off, look.wobble);
            dress(&mut g, self.turn + off, look);
            g.sample(angle, w, h, s);
            if i == n / 2 {
                mid = Some(g);
            }
        }
        mid.unwrap_or_else(|| self.cur.genome.clone())
    }

    pub fn status(&self) -> Status {
        let f = self.flock.lock().unwrap();
        let (s, edge) = match &self.phase {
            Phase::Loop => (&self.cur, None),
            Phase::Edge { .. } => (self.next.as_ref().unwrap(), Some((self.seg / LOOP_SECS) as f32)),
        };
        let (g, id, rating) = (&s.genome, s.id, s.rating);
        Status {
            name: g.name.clone(),
            nick: g.nick.clone(),
            nc: g.nc,
            id,
            rating,
            edge,
            loop_no: self.loops_done + 1,
            flock: f.sheep.len(),
            tested: f.tested,
            rejected: f.rejected,
            taste: s.taste.map(|t| t.1),
        }
    }

    /// Variation sets the GPU will want soon: the next sheep with the
    /// current one (its edge) and alone (its loops).
    pub fn upcoming(&self) -> Vec<Vec<i32>> {
        let ids = |g: &Genome| -> Vec<i32> {
            let mut v: Vec<i32> =
                g.xforms.iter().chain(g.fin.iter()).flat_map(|x| x.vars.iter().filter(|v| v.w != 0.0).map(|v| v.id)).collect();
            // a 3D camera needs the 3D variant too
            if !g.cam.is_flat() {
                v.push(crate::sample::THREE_D);
            }
            v
        };
        let q = self.ready.lock().unwrap();
        let Some(n) = q.front() else { return Vec::new() };
        let mut both = ids(&self.cur.genome);
        both.extend(ids(&n.genome));
        vec![both, ids(&n.genome)]
    }

    /// The genome on screen now (for saving), as `look` shows it.
    pub fn current(&self, look: &Look) -> Genome {
        let (mut g, _) = self.genome_at(0.0, look.wobble);
        dress(&mut g, self.turn, look);
        g
    }
}

impl Drop for Show {
    fn drop(&mut self) {
        self.quit.store(true, Ordering::Release);
        if let Some(t) = self.breeder.take() {
            let _ = t.join();
        }
    }
}

/// The look's changes to a genome `t` seconds into the show.
fn dress(g: &mut Genome, t: f64, look: &Look) {
    use std::f64::consts::TAU;
    let wave = |period: f64, phase: f64| (TAU * t / period + phase).sin() as f32;
    if look.drift {
        // zoom in by up to a third and back, pan by a tenth of the picture,
        // turn a few degrees: periods that never line up, slower than a loop
        let zoom = 0.5 * (1.0 - wave(29.0, 1.3));
        g.scale *= (0.28 * zoom).exp();
        let (px, py) = (0.10 * wave(47.0, 0.0), 0.07 * wave(37.0, 1.0));
        g.center[0] += px / g.scale;
        g.center[1] += py / g.scale;
        g.rotate += 5.0 * wave(61.0, 2.0);
    }
    let [loud, bass, _lowmid, mid, high, beat] = look.audio;
    if look.hue || mid > 0.0 {
        let turn = if look.hue { (t / HUE_TURN).fract() as f32 } else { 0.0 };
        let dh = 6.0 * (turn + 0.04 * mid);
        for c in g.palette.iter_mut() {
            let mut v = crate::genome::rgb2hsv(*c);
            v[0] = (v[0] + dh).rem_euclid(6.0);
            *c = crate::genome::hsv2rgb(v);
        }
    }
    if look.audio != [0.0; 6] {
        // the bass and the beats push the camera in; each band swells some xforms
        g.scale *= 1.0 + 0.05 * bass + 0.07 * beat;
        let bands = [bass, look.audio[2], mid, high];
        for (i, x) in g.xforms.iter_mut().enumerate().filter(|(_, x)| x.animate) {
            x.weight *= 1.0 + 0.6 * (bands[i % 4] - 0.4);
        }
        g.brightness *= 1.0 + 0.25 * loud;
    }
    if look.tilt > 0 {
        // the flame as a plane leaning back, swaying a little; the zoom keeps
        // it filling the frame
        let pitch = 0.75 + 0.12 * wave(43.0, 0.5);
        g.scale *= 1.0 / pitch.cos().max(0.3);
        // the perspective in units of the picture, whatever the sheep's own
        // zoom: the top edge a third further, the bottom a third nearer
        // (the depth blur grows with the same distance, so it needs no such scaling)
        let edge = 0.37 / g.scale * pitch.sin();
        g.cam = crate::genome::Cam3 {
            pitch,
            yaw: 0.15 * wave(53.0, 2.5),
            persp: 0.35 / edge,
            zpos: 0.0,
            dof: if look.tilt > 1 { 0.4 } else { 0.0 },
        };
    }
}

/// New sheep the breeder keeps ready (a few skips in a row need no wait).
const READY: usize = 2;

/// Keeps tested sheep ready, and breeds mutations on request (first).
fn breed_loop(
    flock: Arc<Mutex<Flock>>,
    ready: Arc<Mutex<VecDeque<Sheep>>>,
    quit: Arc<AtomicBool>,
    mut_req: Arc<Mutex<Option<Genome>>>,
    mut_out: Arc<Mutex<Option<Sheep>>>,
) {
    while !quit.load(Ordering::Acquire) {
        let parent = mut_req.lock().unwrap().take();
        if parent.is_none() && ready.lock().unwrap().len() >= READY {
            std::thread::sleep(std::time::Duration::from_millis(20));
            continue;
        }
        // conceive under the lock, test outside it (votes stay responsive);
        // a new sheep is the taste's pick of a few, a mutation the one asked for
        let (mut tested, mut rejected) = (0, 0);
        let (mut r, taste, use_taste) = {
            let mut f = flock.lock().unwrap();
            (Rng::new(f.seed()), f.taste.clone(), f.use_taste)
        };
        let n = if parent.is_none() && use_taste { crate::breed::TASTE_CANDIDATES } else { 1 };
        let mut best: Option<Sheep> = None;
        for _ in 0..n {
            let g = first_good(
                || match &parent {
                    Some(p) => {
                        let g = mutate(p, &mut r);
                        (g, r.next_u32())
                    }
                    None => {
                        let mut f = flock.lock().unwrap();
                        (f.conceive(), f.seed())
                    }
                },
                &mut tested,
                &mut rejected,
            );
            let mut s = Sheep::new(g, 0, 0);
            s.measure(&taste, r.next_u32());
            if best.as_ref().is_none_or(|b| s.score() > b.score()) {
                best = Some(s);
            }
            if quit.load(Ordering::Acquire) {
                return;
            }
        }
        let mut s = best.unwrap();
        crate::breed::add_wobble(&mut s.genome, r.next_u32());
        let sheep = {
            let mut f = flock.lock().unwrap();
            f.tested += tested;
            f.rejected += rejected;
            s.id = f.next_id();
            f.add(s.clone());
            s
        };
        if parent.is_some() {
            *mut_out.lock().unwrap() = Some(sheep);
        } else {
            ready.lock().unwrap().push_back(sheep);
        }
    }
}
