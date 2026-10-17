//! The CPU side of the chaos game: one worker per core, eight points per
//! AVX2 lane group, plotting into a histogram the GPU reads directly.
//!
//! Lanes pick their xforms independently, so per-xform values are looked up
//! per lane (two vpermps over the 16 xforms). The variations are evaluated
//! once per variation the flame uses, for all eight lanes, weighted by each
//! lane's xform's weight for it (zero for xforms without it).

use crate::cpuvar::{self, VarIn};
use crate::fmath::Rng8;
use crate::sample::{MAXX, NV, PICKN, Sample};
use crate::simd::{F8, I8, M8};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::*};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;

/// Sixteen per-xform values as two vectors.
#[derive(Clone, Copy)]
struct Lanes([F8; 2]);

impl Lanes {
    fn new(v: [f32; 16]) -> Lanes {
        let mut lo = [0.0; 8];
        let mut hi = [0.0; 8];
        lo.copy_from_slice(&v[..8]);
        hi.copy_from_slice(&v[8..]);
        Lanes([F8::from_array(lo), F8::from_array(hi)])
    }
    #[inline(always)]
    fn at(&self, k: I8, high: M8) -> F8 {
        F8::select(high, self.0[1].permute(k), self.0[0].permute(k))
    }
}

struct Slot {
    id: i32,
    w: Lanes,
    p: [Lanes; 8],
    /// parameters this variation reads (the rest stay zero)
    np: usize,
}

struct FinalXf {
    c: [F8; 6],
    post: Option<[F8; 6]>,
    vars: Vec<(i32, F8, [F8; 8])>,
    color: F8,
    speed: F8,
}

/// One time sample, laid out for the lanes.
pub struct Table {
    c: [Lanes; 6],
    color: Lanes,
    speed: Lanes,
    opacity: Lanes,
    any_opacity: bool,
    post: Option<[Lanes; 6]>,
    /// pre_blur's weight per xform, if any xform has it
    preblur: Option<Lanes>,
    slots: Vec<Slot>,
    fin: Option<FinalXf>,
    pick: Vec<u8>,
    xaos: bool,
    cam: [f32; 6],
    pal: [u32; 256],
    /// colour index -> RGBA8 in 1/16 steps between entries (palette_mode linear)
    pal_fine: Option<Vec<u32>>,
}

impl Table {
    pub fn new(s: &Sample) -> Table {
        let nx = (s.info[0] as usize).clamp(1, MAXX);
        let col = |f: &dyn Fn(usize) -> f32| {
            let mut v = [0.0; 16];
            for (k, x) in v.iter_mut().enumerate().take(nx) {
                *x = f(k);
            }
            Lanes::new(v)
        };
        let c = [0, 1, 2, 3, 4, 5].map(|j| col(&|k| s.xf[k].c[j]));
        let any_post = (0..nx).any(|k| s.xf[k].post[7] != 0.0);
        let post = any_post.then(|| {
            [0, 1, 2, 3, 4, 5].map(|j| col(&|k| if s.xf[k].post[7] != 0.0 { s.xf[k].post[j] } else { [1.0, 0.0, 0.0, 1.0, 0.0, 0.0][j] }))
        });
        // the union of the variations, each with its weight per xform
        let mut slots: Vec<Slot> = Vec::new();
        let mut ids: Vec<i32> = Vec::new();
        for k in 0..nx {
            for j in 0..NV {
                let id = s.xf[k].vid[j];
                if id >= 0 && !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        ids.sort();
        let preblur = ids.contains(&67).then(|| col(&|k| (0..NV).filter(|&j| s.xf[k].vid[j] == 67).map(|j| s.xf[k].vw[j]).sum()));
        ids.retain(|&id| id != 67);
        for id in ids {
            let find = |k: usize| (0..NV).find(|&j| s.xf[k].vid[j] == id);
            // a variation listed twice counts twice (with the first one's parameters)
            let w = col(&|k| (0..NV).filter(|&j| s.xf[k].vid[j] == id).map(|j| s.xf[k].vw[j]).sum());
            let p = std::array::from_fn(|i| col(&|k| find(k).map_or(0.0, |j| s.xf[k].vp[j][i])));
            slots.push(Slot { id, w, p, np: crate::var::gpu_param_count(id) });
        }
        let fin = (s.info[1] != 0).then(|| {
            let f = &s.xf[MAXX];
            FinalXf {
                c: [0, 1, 2, 3, 4, 5].map(|j| F8::splat(f.c[j])),
                post: (f.post[7] != 0.0).then(|| [0, 1, 2, 3, 4, 5].map(|j| F8::splat(f.post[j]))),
                vars: (0..NV)
                    .filter(|&j| f.vid[j] >= 0)
                    .map(|j| (f.vid[j], F8::splat(f.vw[j]), f.vp[j].map(F8::splat)))
                    .collect(),
                color: F8::splat(f.c[6]),
                speed: F8::splat(f.c[7]),
            }
        });
        let mut pick = s.pick.to_vec();
        pick.extend_from_slice(&[0; 4]);
        Table {
            c,
            color: col(&|k| s.xf[k].c[6]),
            speed: col(&|k| s.xf[k].c[7]),
            opacity: col(&|k| s.xf[k].post[6]),
            any_opacity: (0..nx).any(|k| s.xf[k].post[6] < 1.0),
            post,
            preblur,
            slots,
            fin,
            pick,
            xaos: s.info[2] != 0,
            cam: [s.cam[0], s.cam[1], s.cam[2], s.cam[4], s.cam[5], s.cam[6]],
            pal: s.pal,
            pal_fine: (s.info[3] != 0).then(|| {
                let ch = |c: u32, k: u32| ((c >> (8 * k)) & 255) as f32;
                (0..256 * 16)
                    .map(|j| {
                        let (i0, fr) = if j / 16 >= 255 { (254, 1.0) } else { (j / 16, (j % 16) as f32 / 16.0) };
                        let (a, b) = (s.pal[i0], s.pal[i0 + 1]);
                        (0..3).fold(255 << 24, |acc, k| acc | (((ch(a, k) * (1.0 - fr) + ch(b, k) * fr + 0.5) as u32) << (8 * k)))
                    })
                    .collect()
            }),
        }
    }
}

/// Eight walkers.
#[derive(Clone, Copy)]
struct Walk {
    x: F8,
    y: F8,
    c: F8,
    prev: I8,
    fuse: I8,
}

const FUSE: i32 = 20;
const GROUPS: usize = 4;

impl Walk {
    fn scatter(rng: &mut Rng8) -> Walk {
        Walk {
            x: rng.f01() * 2.0 - 1.0,
            y: rng.f01() * 2.0 - 1.0,
            c: rng.f01(),
            prev: I8::zero(),
            fuse: I8::splat(FUSE),
        }
    }
}

/// A histogram: w * h * (r, g, b, count).
#[derive(Clone, Copy)]
pub struct Hist {
    pub ptr: *mut u32,
    pub w: u32,
    pub h: u32,
}
unsafe impl Send for Hist {}
unsafe impl Sync for Hist {}

/// One chaos-game step of eight walkers. Returns the points to plot (after
/// the final xform), their colour indices, and which lanes plot.
#[inline(always)]
fn step(t: &Table, w: &mut Walk, rng: &mut Rng8) -> (F8, F8, F8, M8) {
    // pick: the table row of the previous xform (with xaos), a random column
    let r = rng.next();
    let mut e = r.shr::<22>();
    if t.xaos {
        e = e + w.prev.mul_lo(I8::splat(PICKN as i32));
    }
    let k = gather_u8(&t.pick, e);
    let hi = k.gt(I8::splat(7));
    let (c00, c01, c10, c11, c20, c21) =
        (t.c[0].at(k, hi), t.c[1].at(k, hi), t.c[2].at(k, hi), t.c[3].at(k, hi), t.c[4].at(k, hi), t.c[5].at(k, hi));
    let mut tx = c00.mul_add(w.x, c10.mul_add(w.y, c20));
    let mut ty = c01.mul_add(w.x, c11.mul_add(w.y, c21));
    if let Some(pb) = &t.preblur {
        let pw = pb.at(k, hi);
        if !pw.eq(F8::zero()).all() {
            (tx, ty) = cpuvar::pre_blur(tx, ty, pw, rng);
        }
    }
    let sumsq = tx.mul_add(tx, ty * ty);
    let sq = sumsq.sqrt();
    let mut px = F8::zero();
    let mut py = F8::zero();
    for sl in &t.slots {
        let wt = sl.w.at(k, hi);
        let used = wt.ne(F8::zero());
        if used.none() {
            continue;
        }
        let v = VarIn {
            tx,
            ty,
            sumsq,
            sq,
            w: wt,
            p: std::array::from_fn(|i| if i < sl.np { sl.p[i].at(k, hi) } else { F8::zero() }),
            c10,
            c11,
            c20,
            c21,
        };
        let (dx, dy) = cpuvar::eval(sl.id, &v, rng);
        px += dx.and(used);
        py += dy.and(used);
    }
    if let Some(p) = &t.post {
        let (a, b) = (px, py);
        px = p[0].at(k, hi).mul_add(a, p[2].at(k, hi).mul_add(b, p[4].at(k, hi)));
        py = p[1].at(k, hi).mul_add(a, p[3].at(k, hi).mul_add(b, p[5].at(k, hi)));
    }
    let c = (t.color.at(k, hi) - w.c).mul_add(t.speed.at(k, hi), w.c);
    let bad = px.bad() | py.bad();
    // bad lanes restart from a random point and fuse again
    let fuse_on = w.fuse.gt(I8::zero());
    w.x = F8::select(bad, rng.f01() * 2.0 - 1.0, px);
    w.y = F8::select(bad, rng.f01() * 2.0 - 1.0, py);
    w.c = c;
    w.prev = I8::select(bad, w.prev, k);
    w.fuse = I8::select(bad, I8::splat(FUSE), I8::select(fuse_on, w.fuse - 1, w.fuse));
    let mut plot = !(bad | fuse_on);
    if plot.none() {
        return (w.x, w.y, w.c, plot);
    }
    let (mut fx, mut fy, mut fc) = (w.x, w.y, w.c);
    if let Some(f) = &t.fin {
        let mut tx = f.c[0].mul_add(fx, f.c[2].mul_add(fy, f.c[4]));
        let mut ty = f.c[1].mul_add(fx, f.c[3].mul_add(fy, f.c[5]));
        for &(id, wt, _) in &f.vars {
            if id == 67 {
                (tx, ty) = cpuvar::pre_blur(tx, ty, wt, rng);
            }
        }
        let sumsq = tx.mul_add(tx, ty * ty);
        let sq = sumsq.sqrt();
        let (mut ox, mut oy) = (F8::zero(), F8::zero());
        for &(id, wt, p) in &f.vars {
            let v = VarIn { tx, ty, sumsq, sq, w: wt, p, c10: f.c[2], c11: f.c[3], c20: f.c[4], c21: f.c[5] };
            let (dx, dy) = cpuvar::eval(id, &v, rng);
            ox += dx;
            oy += dy;
        }
        if let Some(p) = &f.post {
            let (a, b) = (ox, oy);
            ox = p[0].mul_add(a, p[2].mul_add(b, p[4]));
            oy = p[1].mul_add(a, p[3].mul_add(b, p[5]));
        }
        fc = (f.color - fc).mul_add(f.speed, fc);
        plot &= !(ox.bad() | oy.bad());
        fx = ox;
        fy = oy;
    }
    if t.any_opacity {
        plot &= rng.f01().lt(t.opacity.at(k, hi));
    }
    (fx, fy, fc, plot)
}

/// `n` steps of the walkers in `wk` with table `t`, plotting into `h`.
#[inline(never)]
fn run(t: &Table, wk: &mut [Walk; GROUPS], rng: &mut Rng8, h: Hist, n: usize) {
    let hist = h.ptr as *const AtomicU32;
    let (wf, hf) = (F8::splat(h.w as f32), F8::splat(h.h as f32));
    let cam = t.cam.map(F8::splat);
    for _ in 0..n {
        for w in wk.iter_mut() {
            let (fx, fy, fc, plot) = step(t, w, rng);
            if plot.none() {
                continue;
            }
            let sx = cam[0].mul_add(fx, cam[1].mul_add(fy, cam[2]));
            let sy = cam[3].mul_add(fx, cam[4].mul_add(fy, cam[5]));
            let inside = plot & sx.ge(F8::zero()) & sy.ge(F8::zero()) & sx.lt(wf) & sy.lt(hf);
            let mut bits = inside.bits();
            if bits == 0 {
                continue;
            }
            let idx = (sy.to_i8().mul_lo(I8::splat(h.w as i32)) + sx.to_i8()).to_array();
            let ci = match &t.pal_fine {
                Some(_) => (fc * 4096.0).to_i8().max(I8::zero()).min(I8::splat(4095)).to_array(),
                None => (fc * 256.0).to_i8().max(I8::zero()).min(I8::splat(255)).to_array(),
            };
            while bits != 0 {
                let l = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let col = match &t.pal_fine {
                    Some(f) => f[ci[l] as usize],
                    None => t.pal[ci[l] as usize],
                };
                let i = idx[l] as usize * 4;
                // racy read-modify-write: a rare lost hit is invisible, a locked add costs more
                unsafe {
                    let b = &*hist.add(i);
                    b.store(b.load(Relaxed).wrapping_add(col & 255), Relaxed);
                    let b = &*hist.add(i + 1);
                    b.store(b.load(Relaxed).wrapping_add((col >> 8) & 255), Relaxed);
                    let b = &*hist.add(i + 2);
                    b.store(b.load(Relaxed).wrapping_add((col >> 16) & 255), Relaxed);
                    let b = &*hist.add(i + 3);
                    b.store(b.load(Relaxed).wrapping_add(1), Relaxed);
                }
            }
        }
    }
}

/// Bytes `t[e]` per lane (the table has 4 bytes of padding at the end).
#[inline(always)]
fn gather_u8(t: &[u8], e: I8) -> I8 {
    use core::arch::x86_64::*;
    unsafe { I8(_mm256_i32gather_epi32::<1>(t.as_ptr() as *const i32, e.0)) & 0xff }
}

/// What the workers do this frame.
pub struct Job {
    pub epoch: u64,
    pub tables: Vec<Table>,
    pub hist: Hist,
    /// scatter the walkers first (a cut to an unrelated flame)
    pub reset: bool,
}

struct Shared {
    job: RwLock<Option<Arc<Job>>>,
    /// per worker: the epoch it is working on
    seen: Vec<AtomicU64>,
    /// per worker: (epoch, iterations) of the last epoch it finished
    done: Vec<(AtomicU64, AtomicU64)>,
    quit: AtomicBool,
    idle: Mutex<bool>,
    wake: Condvar,
}

pub struct Pool {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
    pub workers: usize,
    /// microseconds `switched` waits for stragglers
    pub patience: AtomicU64,
}

/// Steps per batch: each batch re-reads the job (about 0.1-0.3 ms of work).
const BATCH: usize = 256;

impl Pool {
    pub fn new(workers: usize) -> Pool {
        let shared = Arc::new(Shared {
            job: RwLock::new(None),
            seen: (0..workers).map(|_| AtomicU64::new(0)).collect(),
            done: (0..workers).map(|_| (AtomicU64::new(0), AtomicU64::new(0))).collect(),
            quit: AtomicBool::new(false),
            idle: Mutex::new(true),
            wake: Condvar::new(),
        });
        let threads = (0..workers)
            .map(|id| {
                let sh = shared.clone();
                std::thread::Builder::new()
                    .name(format!("chaos{id}"))
                    .spawn(move || worker(id, sh))
                    .expect("spawning a worker")
            })
            .collect();
        Pool { shared, threads, workers, patience: AtomicU64::new(2000) }
    }

    /// Hands out the next job (or none: the workers sleep). Returns at once.
    pub fn publish(&self, job: Option<Job>) {
        let idle = job.is_none();
        *self.shared.job.write().unwrap() = job.map(Arc::new);
        let mut g = self.shared.idle.lock().unwrap();
        *g = idle;
        self.shared.wake.notify_all();
    }

    /// Waits (at most 2 ms) for the workers to move on to `epoch`; returns
    /// the iterations reported for `epoch - 1`. On a loaded machine a worker
    /// can be descheduled for tens of milliseconds; the frame does not wait
    /// for it: the rest of its batch (a few thousand points) lands in the
    /// histogram's next use and its count is dropped.
    pub fn switched(&self, epoch: u64) -> u64 {
        let sh = &self.shared;
        let t0 = std::time::Instant::now();
        let limit = self.patience.load(Relaxed);
        while !sh.seen.iter().all(|s| s.load(Acquire) >= epoch) && (t0.elapsed().as_micros() as u64) < limit {
            std::hint::spin_loop();
            std::thread::yield_now();
        }
        sh.done.iter().filter(|(e, _)| e.load(Acquire) == epoch - 1).map(|(_, n)| n.load(Acquire)).sum()
    }
}

impl Pool {
    /// Stops every worker and waits (up to 2 s) until none is mid-batch:
    /// after this no worker touches any histogram (they may be freed).
    pub fn drain(&self) {
        self.publish(None);
        let t0 = std::time::Instant::now();
        while !self.shared.seen.iter().all(|s| s.load(Acquire) == u64::MAX) && t0.elapsed().as_secs() < 2 {
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.quit.store(true, Release);
        self.publish(None);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

fn worker(id: usize, sh: Arc<Shared>) {
    lower_priority();
    let mut rng = Rng8::new(0x1234_5678 ^ (id as u32).wrapping_mul(0x9E37_79B9));
    let mut wk = [Walk::scatter(&mut rng); GROUPS];
    for w in wk.iter_mut() {
        *w = Walk::scatter(&mut rng);
    }
    let mut epoch = 0u64;
    let mut count = 0u64;
    let mut batch = id;
    while !sh.quit.load(Acquire) {
        let job = sh.job.read().unwrap().clone();
        let Some(job) = job else {
            // idle: report the last epoch as finished and sleep
            if count > 0 {
                sh.done[id].1.store(count, Release);
                sh.done[id].0.store(epoch, Release);
                count = 0;
            }
            sh.seen[id].store(u64::MAX, Release);
            let g = sh.idle.lock().unwrap();
            let _g = sh.wake.wait_while(g, |idle| *idle && !sh.quit.load(Acquire)).unwrap();
            continue;
        };
        if job.epoch != epoch {
            sh.done[id].1.store(count, Release);
            sh.done[id].0.store(epoch, Release);
            count = 0;
            epoch = job.epoch;
            if job.reset {
                for w in wk.iter_mut() {
                    *w = Walk::scatter(&mut rng);
                }
            }
            sh.seen[id].store(epoch, Release);
        }
        if job.tables.is_empty() {
            std::thread::yield_now();
            continue;
        }
        let t = &job.tables[batch % job.tables.len()];
        batch = batch.wrapping_add(1);
        run(t, &mut wk, &mut rng, job.hist, BATCH);
        count += (BATCH * GROUPS * 8) as u64;
    }
}

/// Nice 5: the workers take every idle cycle but yield to the render thread
/// and to the desktop.
fn lower_priority() {
    unsafe extern "C" {
        fn setpriority(which: i32, who: u32, prio: i32) -> i32;
        fn gettid() -> i32;
    }
    unsafe {
        setpriority(0, gettid() as u32, 5);
    }
}

/// Runs the chaos game on this thread alone: `n` iterations of `s`, the
/// plotted points (after the final xform, before the camera) into `out`
/// with their colour indices (genome tests and framing).
pub fn sample_points(s: &Sample, n: usize, seed: u32, out: &mut Vec<[f32; 3]>) {
    let t = Table::new(s);
    let mut rng = Rng8::new(seed);
    let mut w = Walk::scatter(&mut rng);
    for _ in 0..n / 8 {
        let (fx, fy, fc, plot) = step(&t, &mut w, &mut rng);
        let mut bits = plot.bits();
        if bits == 0 {
            continue;
        }
        let (xs, ys, cs) = (fx.to_array(), fy.to_array(), fc.to_array());
        while bits != 0 {
            let l = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            out.push([xs[l], ys[l], cs[l]]);
        }
    }
}
