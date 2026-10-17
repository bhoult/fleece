//! One frame on both processors. The CPU workers and the GPU run the chaos
//! game on the same time samples at once; at the next frame's start the
//! workers move to the new samples (and the other CPU histogram), and the
//! finished frame's two histograms are merged, tone-mapped and shown while
//! the GPU starts on the new one. The picture is one frame behind.

use crate::cpu::{self, Hist, Job, Pool};
use crate::gpu::{Gpu, ITERS, Tone, WALKERS};
use crate::sample::Sample;
use std::time::Instant;

/// What went into a finished frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub gpu_iters: u64,
    pub cpu_iters: u64,
    /// seconds the frame was being iterated (both sides)
    pub secs: f64,
    /// GPU milliseconds for its iterations (from timer queries, a few frames old)
    pub gpu_ms: f32,
}

struct Pending {
    tone: Tone,
    /// pixels per unit squared, for the density scale
    ppu2: f32,
    gpu_iters: u64,
    cpu_hist: Option<usize>,
    started: Instant,
    /// the first frame after a cut: it replaces what was there
    reset: bool,
}

pub struct Renderer {
    pub gpu: Gpu,
    pub pool: Option<Pool>,
    epoch: u64,
    /// the CPU histogram the workers are filling
    cur: usize,
    pending: Option<Pending>,
    /// accumulated iterations behind the acc texture (progressive mode)
    total: f64,
    pub use_gpu: bool,
    pub use_cpu: bool,
    pub last: Stats,
    /// frame blending: how much of the frames before stays in this one (0:
    /// none; a cut always starts afresh, a refining still keeps all)
    pub decay: f32,
}

impl Renderer {
    pub fn new(gpu: Gpu, workers: usize) -> Renderer {
        let pool = (workers > 0).then(|| Pool::new(workers));
        Renderer {
            gpu,
            use_cpu: pool.is_some(),
            pool,
            epoch: 0,
            cur: 0,
            pending: None,
            total: 0.0,
            use_gpu: true,
            last: Stats::default(),
            decay: 0.0,
        }
    }

    pub fn workers(&self) -> usize {
        self.pool.as_ref().map_or(0, |p| p.workers)
    }

    /// Starts frame n+1 on `samples` and finishes frame n into the output
    /// texture: `keep` adds frame n to what is already there (a still that
    /// refines). `reset` scatters every walker (a cut, not a morph).
    /// Returns the finished frame's stats, if there was one.
    pub fn step(&mut self, samples: &[Sample], tone: Tone, ppu2: f32, dispatches: u32, keep: bool, reset: bool) -> Option<Stats> {
        self.epoch += 1;
        // 1. the workers move to the new samples and the other histogram
        let mut cpu_iters = 0;
        let mut next_cpu = None;
        if let Some(pool) = &self.pool {
            if self.use_cpu {
                let i = self.cur ^ 1;
                self.gpu.wait_cpu_hist(i);
                let (ptr, _) = self.gpu.cpu_hist(i);
                let job = Job {
                    epoch: self.epoch,
                    tables: samples.iter().map(cpu::Table::new).collect(),
                    hist: Hist { ptr, w: self.gpu.bw, h: self.gpu.bh },
                    reset,
                };
                pool.publish(Some(job));
                next_cpu = Some(i);
            } else {
                pool.publish(None);
            }
            cpu_iters = pool.switched(self.epoch);
        }
        // 2. finish the previous frame
        let done = self.close(cpu_iters, keep);
        if done.is_none() {
            self.gpu.clear_hist();
        }
        // 3. the GPU starts the new frame
        if reset {
            self.gpu.reset_walkers(self.epoch as u32 ^ 0x5eed_1234);
        }
        self.gpu.upload(samples);
        let d = if self.use_gpu { dispatches } else { 0 };
        if d > 0 {
            self.gpu.iterate(d);
        }
        self.pending = Some(Pending {
            tone,
            ppu2,
            gpu_iters: d as u64 * WALKERS as u64 * ITERS as u64,
            cpu_hist: next_cpu,
            started: Instant::now(),
            reset,
        });
        self.cur = next_cpu.unwrap_or(self.cur);
        done
    }

    /// Merges and tone-maps the pending frame.
    fn close(&mut self, cpu_iters: u64, keep: bool) -> Option<Stats> {
        let p = self.pending.take()?;
        let decay = if p.reset {
            0.0
        } else if keep {
            1.0
        } else {
            self.decay
        };
        let iters = p.gpu_iters as f64 + cpu_iters as f64;
        // the iterations behind acc, for the density: they fade with it
        self.total = self.total * decay as f64 + iters;
        self.gpu.merge(p.cpu_hist, decay);
        let mut tone = p.tone;
        tone.density = (p.ppu2 as f64 / self.total.max(1.0)) as f32;
        self.gpu.tone(&tone);
        let s = Stats {
            gpu_iters: p.gpu_iters,
            cpu_iters,
            secs: p.started.elapsed().as_secs_f64(),
            gpu_ms: self.gpu.last_ms.map_or(0.0, |(ms, _)| ms),
        };
        self.last = s;
        Some(s)
    }

    /// Finishes the pending frame without starting another (the workers stop).
    pub fn finish(&mut self, keep: bool) -> Option<Stats> {
        self.epoch += 1;
        let mut cpu_iters = 0;
        if let Some(pool) = &self.pool {
            pool.publish(None);
            cpu_iters = pool.switched(self.epoch);
        }
        self.close(cpu_iters, keep)
    }

    /// Finishes the pending frame and waits until no worker is writing: the
    /// histograms can be freed (a resize).
    pub fn quiesce(&mut self) {
        self.finish(false);
        if let Some(p) = &self.pool {
            p.drain();
        }
    }

    /// Stops the workers (for good, or until the next `step`).
    pub fn idle(&mut self) {
        if let Some(p) = &self.pool {
            p.publish(None);
        }
    }

    pub fn total_iters(&self) -> f64 {
        self.total
    }
}
