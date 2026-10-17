//! The GPU side: the chaos game in a compute shader (iterate.comp), adding
//! the CPU's histogram in (merge.comp), the log-density tone map with
//! density estimation (tone.comp) and the window blit (present.*).
//!
//! The CPU's histograms live in two persistent, coherent, client-storage
//! buffers: cached system memory the worker threads write straight into and
//! the merge pass reads over the bus, so nothing is copied per frame.

use crate::gl::*;
use crate::sample::{MAX_SAMPLES, MAXX, NV, PICKN, Sample, THREE_D};
use std::ffi::c_void;

/// Chaos-game walkers on the GPU (256 per work group).
pub const WALKERS: u32 = 1 << 18;
const GROUP: u32 = 256;
/// Iterations each walker runs per dispatch.
pub const ITERS: u32 = 64;
/// Iterations after a (re)start before a walker plots (flam3's fuse).
pub const FUSE: u32 = 20;
const QUERY_RING: usize = 4;

/// The tone map's settings for one frame (flam3's names and defaults).
#[derive(Clone, Copy, Debug)]
pub struct Tone {
    pub brightness: f32,
    pub gamma: f32,
    pub gamma_threshold: f32,
    pub vibrancy: f32,
    pub highlight: f32,
    pub background: [f32; 3],
    /// density estimation: estimator radius (9; 0 = off), minimum (0) and
    /// curve (0.4)
    pub de_radius: f32,
    pub de_min: f32,
    pub de_curve: f32,
    /// pixels per unit squared over the iterations behind the histogram:
    /// hits times this is the density in world units (flam3's A k2)
    pub density: f32,
}

/// An iterate program and its uniforms.
#[derive(Clone, Copy)]
struct IterProg {
    prog: GLuint,
    sample0: GLint,
    nsamples: GLint,
    iters: GLint,
    size: GLint,
}

enum Prog {
    /// compiling on the driver's threads (program, shader)
    Pending(GLuint, GLuint),
    Ready(IterProg),
    Failed,
}

impl IterProg {
    fn of(gl: &Gl, prog: GLuint) -> IterProg {
        IterProg {
            prog,
            sample0: gl.uniform(prog, "u_sample0"),
            nsamples: gl.uniform(prog, "u_nsamples"),
            iters: gl.uniform(prog, "u_iters"),
            size: gl.uniform(prog, "u_size"),
        }
    }
}

/// variations.glsl with only the cases for `ids` (all of them for None): a
/// small switch compiles to far fewer registers.
fn specialize(src: &str, ids: Option<&[i32]>) -> String {
    let Some(ids) = ids else { return src.to_string() };
    let open = "    switch (id) {\n";
    let close = "    }\n    return vec2(0.0);\n}";
    let (Some(a), Some(b)) = (src.find(open), src.rfind(close)) else { return src.to_string() };
    let body = &src[a + open.len()..b];
    let mut out = String::from(&src[..a + open.len()]);
    let mut keep = false;
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("    case ") {
            let id: i32 = rest.split(':').next().unwrap_or("").trim().parse().unwrap_or(-1);
            keep = ids.contains(&id);
        }
        if keep {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.push_str(&src[b..]);
    out
}

struct Locs {
    rs_seed: GLint,
    mg_keep: GLint,
    mg_size: GLint,
    tn: [GLint; 9],
    pr_rect: GLint,
    pr_mode: GLint,
}

pub struct Gpu {
    pub gl: Gl,
    pub w: u32,
    pub h: u32,
    p_iter: IterProg,
    /// iterate programs specialised to a set of variations
    progs: std::collections::HashMap<Vec<i32>, Prog>,
    parallel: bool,
    iter_src: String,
    /// the program for the samples uploaded last
    cur: IterProg,
    pub compile_ms: f32,
    p_reset: GLuint,
    p_merge: GLuint,
    p_tone: GLuint,
    p_present: GLuint,
    loc: Locs,
    samples: GLuint,
    walk: GLuint,
    aux: GLuint,
    hist: GLuint,
    cpu: [GLuint; 2],
    cpu_ptr: [*mut u32; 2],
    fence: [GLsync; 2],
    acc: GLuint,
    out: GLuint,
    knorm: GLuint,
    hud: GLuint,
    hud_size: (u32, u32),
    vao: GLuint,
    queries: [[GLuint; 2]; QUERY_RING],
    query_frame: usize,
    /// GPU milliseconds of the last measured iterate batch, and its dispatch count
    pub last_ms: Option<(f32, u32)>,
    query_dispatches: [u32; QUERY_RING],
    nsamples: u32,
    sample_cursor: u32,
}

const ITERATE: &str = include_str!("shaders/iterate.comp");
const VARIATIONS: &str = include_str!("shaders/variations.glsl");
const RESET: &str = include_str!("shaders/reset.comp");
const MERGE: &str = include_str!("shaders/merge.comp");
const TONE: &str = include_str!("shaders/tone.comp");
const PRESENT_VS: &str = include_str!("shaders/present.vert");
const PRESENT_FS: &str = include_str!("shaders/present.frag");

fn header() -> String {
    format!(
        "#version 460\n#define MAXX {MAXX}\n#define NV {NV}\n#define PICKN {PICKN}\n#define WALKERS {WALKERS}u\n#define FUSE {FUSE}u\n#define ITERS_MAX 4096\n"
    )
}

impl Gpu {
    pub fn new(gl: Gl, w: u32, h: u32) -> Result<Gpu, String> {
        let hd = header();
        // 64-bit atomics (NVIDIA) halve the histogram traffic
        let ext = if std::env::var_os("FLEECE_NO_INT64").is_none()
            && gl.has_extension("GL_NV_shader_atomic_int64")
            && gl.has_extension("GL_NV_gpu_shader5")
        {
            "#extension GL_NV_gpu_shader5 : require\n#extension GL_NV_shader_atomic_int64 : require\n#define HIST64 1\n"
        } else {
            ""
        };
        // FLEECE_EXP=noplot,linear,warpxf,randaddr,oneatomic: measurement
        // switches for --bench (see iterate.comp); they break the picture
        let exp: String = std::env::var("FLEECE_EXP")
            .unwrap_or_default()
            .split(',')
            .filter(|e| !e.is_empty())
            .map(|e| format!("#define EXP_{}\n#extension GL_KHR_shader_subgroup_ballot : enable\n", e.to_uppercase()))
            .collect();
        let iter_src = format!("{hd}{ext}{exp}{ITERATE}");
        let p_iter = Gpu::iter_prog(&gl, &iter_src, None)?;
        let p_reset = gl.program("reset.comp", &[(GL_COMPUTE_SHADER, &format!("{hd}{RESET}"))])?;
        let p_merge = gl.program("merge.comp", &[(GL_COMPUTE_SHADER, &format!("{hd}{MERGE}"))])?;
        let p_tone = gl.program("tone.comp", &[(GL_COMPUTE_SHADER, &format!("{hd}{TONE}"))])?;
        let p_present = gl.program(
            "present",
            &[(GL_VERTEX_SHADER, &format!("{hd}{PRESENT_VS}")), (GL_FRAGMENT_SHADER, &format!("{hd}{PRESENT_FS}"))],
        )?;
        let tn_names = ["u_size", "u_k", "u_gamma", "u_vib", "u_bg", "u_de", "u_unused", "u_highlight", "u_lin"];
        let loc = Locs {
            rs_seed: gl.uniform(p_reset, "u_seed"),
            mg_keep: gl.uniform(p_merge, "u_keep"),
            mg_size: gl.uniform(p_merge, "u_size"),
            tn: tn_names.map(|n| gl.uniform(p_tone, n)),
            pr_rect: gl.uniform(p_present, "u_rect"),
            pr_mode: gl.uniform(p_present, "u_mode"),
        };
        let mut g = Gpu {
            gl,
            w: 0,
            h: 0,
            p_iter,
            progs: std::collections::HashMap::new(),
            parallel: false,
            iter_src,
            cur: p_iter,
            compile_ms: 0.0,
            p_reset,
            p_merge,
            p_tone,
            p_present,
            loc,
            samples: 0,
            walk: 0,
            aux: 0,
            hist: 0,
            cpu: [0; 2],
            cpu_ptr: [std::ptr::null_mut(); 2],
            fence: [std::ptr::null(); 2],
            acc: 0,
            out: 0,
            knorm: 0,
            hud: 0,
            hud_size: (0, 0),
            vao: 0,
            queries: [[0; 2]; QUERY_RING],
            query_frame: 0,
            last_ms: None,
            query_dispatches: [0; QUERY_RING],
            nsamples: 1,
            sample_cursor: 0,
        };
        unsafe {
            let gl = &g.gl;
            gl.glCreateBuffers(1, &mut g.samples);
            gl.glNamedBufferStorage(
                g.samples,
                (MAX_SAMPLES * size_of::<Sample>()) as isize,
                std::ptr::null(),
                GL_DYNAMIC_STORAGE_BIT,
            );
            gl.glCreateBuffers(1, &mut g.walk);
            gl.glNamedBufferStorage(g.walk, WALKERS as isize * 16, std::ptr::null(), 0);
            gl.glCreateBuffers(1, &mut g.aux);
            // prev | fuse << 8, and z
            gl.glNamedBufferStorage(g.aux, WALKERS as isize * 8, std::ptr::null(), 0);
            gl.glCreateVertexArrays(1, &mut g.vao);
            // density estimation kernels: the sum of gauss(1.5 d / h) over the
            // pixels within h (and within ceil(h) - 1 on each axis), h = i / 16
            let norm: Vec<f32> = (0..512)
                .map(|i| {
                    let h = (i as f32 / 16.0).max(1e-3);
                    let reach = h.ceil() as i32 - 1;
                    let mut s = 0.0f32;
                    for dy in -12i32..=12 {
                        for dx in -12i32..=12 {
                            let d2 = (dx * dx + dy * dy) as f32;
                            if d2 <= h * h && dx.abs().max(dy.abs()) <= reach {
                                s += (-4.5 * d2 / (h * h)).exp();
                            }
                        }
                    }
                    s.max(1e-6)
                })
                .collect();
            gl.glCreateBuffers(1, &mut g.knorm);
            gl.glNamedBufferStorage(g.knorm, 512 * 4, norm.as_ptr() as *const c_void, 0);
            for q in g.queries.iter_mut() {
                gl.glCreateQueries(GL_TIMESTAMP, 2, q.as_mut_ptr());
            }
            gl.check("GPU setup");
        }
        g.reset_walkers(0x5eed);
        g.resize(w, h);
        if g.gl.has_extension("GL_KHR_parallel_shader_compile") {
            unsafe { g.gl.glMaxShaderCompilerThreadsKHR(4) };
            g.parallel = true;
        }
        Ok(g)
    }

    fn iter_source(&self, ids: Option<&[i32]>) -> String {
        let src = self.iter_src.replace("#include \"variations.glsl\"", &specialize(VARIATIONS, ids));
        // the 3D variant (walker z, the 3D camera, direct colour)
        match ids {
            Some(k) if k.contains(&THREE_D) => src.replacen('\n', &format!("\n{}", crate::var::three_d_glsl()), 1),
            _ => src,
        }
    }

    fn iter_prog(gl: &Gl, src: &str, ids: Option<&[i32]>) -> Result<IterProg, String> {
        let src = src.replace("#include \"variations.glsl\"", &specialize(VARIATIONS, ids));
        let prog = gl.program("iterate.comp", &[(GL_COMPUTE_SHADER, &src)])?;
        Ok(IterProg::of(gl, prog))
    }

    fn key(ids: &[i32]) -> Vec<i32> {
        let mut key = ids.to_vec();
        // 3D and dc_ variations need the 3D variant (a 3D camera alone is
        // marked by sample::variation_ids)
        if ids.iter().any(|&id| crate::var::needs_3d(id)) {
            key.push(THREE_D);
        }
        key.sort();
        key.dedup();
        key
    }

    /// Starts compiling the program for these variations (if it isn't).
    pub fn prepare(&mut self, ids: &[i32]) {
        let key = Gpu::key(ids);
        if self.progs.contains_key(&key) {
            return;
        }
        if self.progs.len() >= 48 {
            // forget the programs not in use
            let cur = self.cur.prog;
            let gl = &self.gl;
            self.progs.retain(|_, p| match p {
                Prog::Ready(ip) if ip.prog != cur => {
                    unsafe { gl.glDeleteProgram(ip.prog) };
                    false
                }
                _ => true,
            });
        }
        let src = self.iter_source(Some(&key));
        let (p, s) = self.gl.program_async(&src);
        self.progs.insert(key, Prog::Pending(p, s));
    }

    /// Uses the program for exactly these variations once it is compiled,
    /// the general one until then. With `wait`, compiles it now.
    pub fn use_variations(&mut self, ids: &[i32], wait: bool) {
        let key = Gpu::key(ids);
        if !self.progs.contains_key(&key) {
            self.prepare(&key);
        }
        let entry = self.progs.get_mut(&key).unwrap();
        if let Prog::Pending(p, s) = *entry {
            // the general program is flat: a 3D flame waits for its own
            if wait || key.contains(&THREE_D) || !self.parallel || self.gl.ready(p) {
                let t = std::time::Instant::now();
                *entry = match self.gl.finish_program(p, s) {
                    Ok(p) => Prog::Ready(IterProg::of(&self.gl, p)),
                    Err(e) => {
                        eprintln!("fleece: specialised shader failed, using the general one: {e}");
                        Prog::Failed
                    }
                };
                self.compile_ms = t.elapsed().as_secs_f32() * 1000.0;
            }
        }
        self.cur = match self.progs[&key] {
            Prog::Ready(p) => p,
            _ => self.p_iter,
        };
    }

    /// Whether the specialised program is the one in use.
    pub fn specialised(&self) -> bool {
        self.cur.prog != self.p_iter.prog
    }

    /// Back to the program with every variation.
    pub fn use_all_variations(&mut self) {
        self.cur = self.p_iter;
    }

    /// (Re)allocates everything sized by the histogram.
    pub fn resize(&mut self, w: u32, h: u32) {
        if (w, h) == (self.w, self.h) {
            return;
        }
        let gl = &self.gl;
        unsafe {
            gl.glFinish();
            if self.hist != 0 {
                gl.glDeleteBuffers(1, &self.hist);
                gl.glDeleteBuffers(2, self.cpu.as_ptr());
                gl.glDeleteTextures(1, &self.acc);
                gl.glDeleteTextures(1, &self.out);
                for f in self.fence.iter_mut() {
                    if !f.is_null() {
                        gl.glDeleteSync(*f);
                        *f = std::ptr::null();
                    }
                }
            }
            let n = w as isize * h as isize * 16;
            gl.glCreateBuffers(1, &mut self.hist);
            gl.glNamedBufferStorage(self.hist, n, std::ptr::null(), GL_DYNAMIC_STORAGE_BIT);
            let flags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
            gl.glCreateBuffers(2, self.cpu.as_mut_ptr());
            for i in 0..2 {
                gl.glNamedBufferStorage(self.cpu[i], n, std::ptr::null(), flags | GL_CLIENT_STORAGE_BIT);
                self.cpu_ptr[i] = gl.glMapNamedBufferRange(self.cpu[i], 0, n, flags) as *mut u32;
                std::ptr::write_bytes(self.cpu_ptr[i] as *mut u8, 0, n as usize);
            }
            gl.glCreateTextures(GL_TEXTURE_2D, 1, &mut self.acc);
            gl.glTextureStorage2D(self.acc, 1, GL_RGBA32F, w as i32, h as i32);
            gl.glTextureParameteri(self.acc, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            gl.glTextureParameteri(self.acc, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            gl.glTextureParameteri(self.acc, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            gl.glTextureParameteri(self.acc, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            gl.glCreateTextures(GL_TEXTURE_2D, 1, &mut self.out);
            gl.glTextureStorage2D(self.out, 1, GL_RGBA8, w as i32, h as i32);
            gl.glTextureParameteri(self.out, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            gl.glTextureParameteri(self.out, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            gl.glTextureParameteri(self.out, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            gl.glTextureParameteri(self.out, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            let e = gl.glGetError();
            if e != 0 {
                panic!("OpenGL error 0x{e:04x} allocating a {w}x{h} histogram (out of video memory?)");
            }
        }
        self.w = w;
        self.h = h;
        self.clear_hist();
    }

    /// The CPU histogram `i` (w * h * 4 u32: r, g, b, count) for the workers.
    pub fn cpu_hist(&self, i: usize) -> (*mut u32, usize) {
        (self.cpu_ptr[i], self.w as usize * self.h as usize * 4)
    }

    /// Blocks until the GPU has finished reading and clearing CPU histogram `i`.
    pub fn wait_cpu_hist(&mut self, i: usize) {
        let f = std::mem::replace(&mut self.fence[i], std::ptr::null());
        if !f.is_null() {
            unsafe {
                self.gl.glClientWaitSync(f, GL_SYNC_FLUSH_COMMANDS_BIT, 1_000_000_000);
                self.gl.glDeleteSync(f);
            }
        }
    }

    /// New walker positions everywhere (a cut to an unrelated flame).
    pub fn reset_walkers(&mut self, seed: u32) {
        let gl = &self.gl;
        unsafe {
            gl.glUseProgram(self.p_reset);
            gl.glProgramUniform1ui(self.p_reset, self.loc.rs_seed, seed);
            gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.walk);
            gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.aux);
            gl.glDispatchCompute(WALKERS / GROUP, 1, 1);
            gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    }

    pub fn upload(&mut self, samples: &[Sample]) {
        let n = samples.len().min(MAX_SAMPLES);
        let b = Sample::as_bytes(&samples[..n]);
        unsafe { self.gl.glNamedBufferSubData(self.samples, 0, b.len() as isize, b.as_ptr() as *const c_void) };
        self.nsamples = n as u32;
    }

    pub fn clear_hist(&mut self) {
        let zero = 0u32;
        unsafe {
            self.gl.glClearNamedBufferData(self.hist, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero as *const u32 as _)
        };
    }

    /// `dispatches` rounds of the chaos game: WALKERS x ITERS iterations each.
    pub fn iterate(&mut self, dispatches: u32) {
        let gl = &self.gl;
        let q = self.query_frame % QUERY_RING;
        unsafe {
            gl.glQueryCounter(self.queries[q][0], GL_TIMESTAMP);
            let p = self.cur;
            gl.glUseProgram(p.prog);
            gl.glProgramUniform1i(p.prog, p.iters, ITERS as i32);
            gl.glProgramUniform1i(p.prog, p.nsamples, self.nsamples as i32);
            gl.glProgramUniform2i(p.prog, p.size, self.w as i32, self.h as i32);
            gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.samples);
            gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.walk);
            gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.aux);
            gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.hist);
            for _ in 0..dispatches {
                gl.glProgramUniform1i(p.prog, p.sample0, self.sample_cursor as i32);
                self.sample_cursor = self.sample_cursor.wrapping_add(7);
                gl.glDispatchCompute(WALKERS / GROUP, 1, 1);
                gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }
            gl.glQueryCounter(self.queries[q][1], GL_TIMESTAMP);
        }
        self.query_dispatches[q] = dispatches;
        self.query_frame += 1;
        // the oldest query in the ring is (almost always) ready by now
        if self.query_frame >= QUERY_RING {
            let old = self.query_frame % QUERY_RING;
            let mut ready = 0;
            unsafe { gl.glGetQueryObjectiv(self.queries[old][1], GL_QUERY_RESULT_AVAILABLE, &mut ready) };
            if ready != 0 && self.query_dispatches[old] > 0 {
                let (mut t0, mut t1) = (0u64, 0u64);
                unsafe {
                    gl.glGetQueryObjectui64v(self.queries[old][0], GL_QUERY_RESULT, &mut t0);
                    gl.glGetQueryObjectui64v(self.queries[old][1], GL_QUERY_RESULT, &mut t1);
                }
                self.last_ms = Some((t1.saturating_sub(t0) as f32 * 1e-6, self.query_dispatches[old]));
            }
        }
    }

    /// acc = (keep ? acc : 0) + GPU histogram + CPU histogram `cpu`; then the
    /// CPU histogram is cleared for reuse and the GPU's for the next frame.
    pub fn merge(&mut self, cpu: Option<usize>, keep: bool) {
        let gl = &self.gl;
        unsafe {
            gl.glUseProgram(self.p_merge);
            gl.glProgramUniform1i(self.p_merge, self.loc.mg_keep, keep as i32 | (cpu.is_some() as i32) << 1);
            gl.glProgramUniform2i(self.p_merge, self.loc.mg_size, self.w as i32, self.h as i32);
            gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.hist);
            gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.cpu[cpu.unwrap_or(0)]);
            gl.glBindImageTexture(0, self.acc, 0, 0, 0, GL_READ_WRITE, GL_RGBA32F);
            gl.glDispatchCompute(self.w.div_ceil(16), self.h.div_ceil(16), 1);
            gl.glMemoryBarrier(GL_ALL_BARRIER_BITS);
            let zero = 0u32;
            gl.glClearNamedBufferData(self.hist, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero as *const u32 as _);
            if let Some(i) = cpu {
                gl.glClearNamedBufferData(self.cpu[i], GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero as *const u32 as _);
                if !self.fence[i].is_null() {
                    gl.glDeleteSync(self.fence[i]);
                }
                self.fence[i] = gl.glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            }
        }
    }

    pub fn tone(&mut self, t: &Tone) {
        let gl = &self.gl;
        let l = &self.loc.tn;
        unsafe {
            gl.glUseProgram(self.p_tone);
            gl.glProgramUniform2i(self.p_tone, l[0], self.w as i32, self.h as i32);
            gl.glProgramUniform2f(self.p_tone, l[1], t.brightness * 268.0 / 256.0, t.density);
            gl.glProgramUniform1f(self.p_tone, l[2], 1.0 / t.gamma.max(0.1));
            gl.glProgramUniform1f(self.p_tone, l[3], t.vibrancy);
            gl.glProgramUniform4f(self.p_tone, l[4], t.background[0], t.background[1], t.background[2], 0.0);
            let cmax = if t.de_radius > 0.0 { t.de_radius.min(12.0) + 1.0 } else { 0.0 };
            gl.glProgramUniform4f(self.p_tone, l[5], cmax, t.de_min + 1.0, t.de_curve.max(0.01), 0.0);
            gl.glProgramUniform1f(self.p_tone, l[7], t.highlight);
            gl.glProgramUniform1f(self.p_tone, l[8], t.gamma_threshold);
            gl.glBindTextureUnit(0, self.acc);
            gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, self.knorm);
            gl.glBindImageTexture(1, self.out, 0, 0, 0, GL_WRITE_ONLY, GL_RGBA8);
            gl.glDispatchCompute(self.w.div_ceil(16), self.h.div_ceil(16), 1);
            gl.glMemoryBarrier(GL_ALL_BARRIER_BITS);
        }
    }

    /// The text overlay, RGBA8, drawn at the top left.
    pub fn set_hud(&mut self, w: u32, h: u32, rgba: &[u8]) {
        let gl = &self.gl;
        unsafe {
            if self.hud_size != (w, h) {
                if self.hud != 0 {
                    gl.glDeleteTextures(1, &self.hud);
                }
                gl.glCreateTextures(GL_TEXTURE_2D, 1, &mut self.hud);
                gl.glTextureStorage2D(self.hud, 1, GL_RGBA8, w as i32, h as i32);
                gl.glTextureParameteri(self.hud, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                gl.glTextureParameteri(self.hud, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                self.hud_size = (w, h);
            }
            gl.glTextureSubImage2D(self.hud, 0, 0, 0, w as i32, h as i32, GL_RGBA, GL_UNSIGNED_BYTE, rgba.as_ptr() as _);
        }
    }

    /// Draws the frame into `rect` (x, y, w, h in clip space) of the drawable
    /// (`dw` x `dh`, black around it), and the HUD at `hud_scale` if there is one.
    pub fn present(&mut self, dw: u32, dh: u32, rect: [f32; 4], hud_scale: Option<u32>) {
        let gl = &self.gl;
        unsafe {
            gl.glViewport(0, 0, dw as i32, dh as i32);
            gl.glClearColor(0.0, 0.0, 0.0, 1.0);
            gl.glClear(GL_COLOR_BUFFER_BIT);
            gl.glUseProgram(self.p_present);
            gl.glBindVertexArray(self.vao);
            gl.glBindTextureUnit(0, self.out);
            gl.glProgramUniform1i(self.p_present, self.loc.pr_mode, 0);
            gl.glProgramUniform4f(self.p_present, self.loc.pr_rect, rect[0], rect[1], rect[2], rect[3]);
            gl.glDrawArrays(GL_TRIANGLES, 0, 6);
            if let (Some(s), true) = (hud_scale, self.hud != 0) {
                let (w, h) = self.hud_size;
                let sx = 2.0 * (w * s) as f32 / dw as f32;
                let sy = 2.0 * (h * s) as f32 / dh as f32;
                let m = 2.0 * 12.0 / dw as f32;
                let my = 2.0 * 12.0 / dh as f32;
                gl.glEnable(GL_BLEND);
                gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                gl.glBindTextureUnit(0, self.hud);
                gl.glProgramUniform1i(self.p_present, self.loc.pr_mode, 1);
                gl.glProgramUniform4f(self.p_present, self.loc.pr_rect, -1.0 + m, 1.0 - my - sy, sx, sy);
                gl.glDrawArrays(GL_TRIANGLES, 0, 6);
                gl.glDisable(GL_BLEND);
            }
        }
    }

    /// The tone-mapped frame, RGB8, top row first.
    pub fn read_rgb(&self) -> Vec<u8> {
        let (w, h) = (self.w as usize, self.h as usize);
        let mut rgba = vec![0u8; w * h * 4];
        unsafe {
            self.gl.glGetTextureImage(self.out, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.len() as i32, rgba.as_mut_ptr() as _)
        };
        let mut rgb = Vec::with_capacity(w * h * 3);
        for row in 0..h {
            for px in rgba[row * w * 4..(row + 1) * w * 4].chunks_exact(4) {
                rgb.extend_from_slice(&px[..3]);
            }
        }
        rgb
    }

    /// What `present` drew (the default framebuffer), RGB8, top row first.
    pub fn read_screen(&self, w: u32, h: u32) -> Vec<u8> {
        let (w, h) = (w as usize, h as usize);
        let mut rgba = vec![0u8; w * h * 4];
        unsafe { self.gl.glReadPixels(0, 0, w as i32, h as i32, GL_RGBA, GL_UNSIGNED_BYTE, rgba.as_mut_ptr() as _) };
        let mut rgb = Vec::with_capacity(w * h * 3);
        for row in (0..h).rev() {
            for px in rgba[row * w * 4..(row + 1) * w * 4].chunks_exact(4) {
                rgb.extend_from_slice(&px[..3]);
            }
        }
        rgb
    }

    /// The raw summed histogram (r, g, b, count per pixel) for tests.
    pub fn read_acc(&self) -> Vec<f32> {
        let n = self.w as usize * self.h as usize * 4;
        let mut v = vec![0f32; n];
        unsafe { self.gl.glGetTextureImage(self.acc, 0, GL_RGBA, GL_FLOAT, (n * 4) as i32, v.as_mut_ptr() as _) };
        v
    }

    pub fn finish(&self) {
        unsafe { self.gl.glFinish() };
    }

    /// Free video memory in MB (NVIDIA only; None elsewhere).
    pub fn free_vram_mb(&self) -> Option<u32> {
        let mut kb = -1;
        unsafe {
            self.gl.glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &mut kb);
            self.gl.glGetError();
        }
        (kb > 0).then_some(kb as u32 / 1024)
    }
}
