//! fleece: fractal flames in the manner of Electric Sheep, rendered live on
//! the GPU and every CPU core at once.
// nested ifs read better here; `!(a > b)` is how NaN is kept out
#![allow(clippy::collapsible_if, clippy::neg_cmp_op_on_partial_ord, clippy::new_without_default)]

mod app;
mod breed;
mod cpu;
mod cpuvar;
mod flam3;
mod fmath;
mod genome;
mod gl;
mod gpu;
mod hud;
mod interp;
mod png;
mod render;
mod rng;
mod sample;
mod sdl;
mod show;
mod simd;
mod var;
mod video;

use genome::Genome;
use gpu::{Gpu, Tone};
use render::Renderer;
use sample::Sample;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Window,
    Shot,
    VarCheck,
    MutCheck,
    Frames,
    Video,
    Bench,
}

pub struct Opts {
    mode: Mode,
    pub out: String,
    pub w: u32,
    pub h: u32,
    pub seed: u32,
    pub quality: f32,
    pub cpu: bool,
    pub gpu: bool,
    angle: f32,
    breed: u32,
    /// --bench: runs per flame (the best counts: a shared GPU only slows)
    repeat: u32,
    pub workers: usize,
    pub fullscreen: bool,
    /// render size relative to the window (0: at most 1080 lines)
    pub scale: f32,
    /// frames per second of animation time (headless and video)
    pub fps: f64,
    pub hud: bool,
    pub frames: u64,
    pub every: u64,
    /// (frame, key) presses for the headless run
    pub keys: Vec<(u64, char)>,
    /// .flam3 files to show first
    pub load: Vec<String>,
    /// start a new flock (the window keeps its flock between runs otherwise)
    pub fresh: bool,
    /// a fixed render size, scaled to the window (Electric Sheep's 800x592 by default)
    pub render: Option<(u32, u32)>,
    /// --quality in the window: samples per pixel every frame, at whatever frame rate that takes
    pub live_quality: Option<f32>,
    /// --video: x264 / VP9 quality (lower is better; 17 is near lossless)
    pub crf: u32,
    /// burn the sheep's credit into video and recorded frames (--credit off:
    /// only the file's metadata and the .credits.txt carry it)
    pub credit: bool,
    /// seconds the burned-in credit shows before fading (--credit-secs)
    pub credit_secs: f64,
    /// buckets per pixel on each axis (--supersample; None: the genome's own
    /// for stills and videos, the live setting in the window)
    pub supersample: Option<u32>,
}

/// Electric Sheep's frame sizes: the standard flock and the HD (gold) one.
pub const ES_SIZE: (u32, u32) = (800, 592);
pub const ES_HD_SIZE: (u32, u32) = (1280, 720);

fn usage() -> ! {
    eprintln!("usage: fleece [WxH | 720p | 1080p | 2k | 4k] [--fullscreen] [--seed N] [--no-hud]");
    eprintln!("              [--es | --es-hd | --render WxH | --native | --scale F] [--quality N]");
    eprintln!("              [--load FILE.flam3]... [--fresh] [--cpu-only | --gpu-only] [--threads N]");
    eprintln!("       (default: Electric Sheep's 800x592 frame, scaled to the window, at 30 fps)");
    eprintln!("       fleece --shot OUT.png [--quality Q] [--angle DEG] [--breed N] [size]   one still");
    eprintln!("       fleece --frames N [--fps F] [--out DIR --every K] [--keys F:K,...]   headless run");
    eprintln!("       fleece --video OUT.mp4 [--seconds S | --frames N] [--fps F] [--quality Q] [--crf N] [size]");
    eprintln!("              [--credit on|off]   a video (any ffmpeg format by extension; .y4m needs no ffmpeg;");
    eprintln!("              - is raw RGB24 on stdout); the credit is burned in for the first --credit-secs (10)");
    eprintln!("              unless off, and always in the metadata");
    eprintln!("       fleece --varcheck   CPU against GPU, every variation");
    std::process::exit(2)
}

fn parse() -> Opts {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut o = Opts {
        mode: Mode::Window,
        out: String::new(),
        w: 1280,
        h: 720,
        seed: (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u32)),
        quality: 200.0,
        cpu: true,
        gpu: true,
        angle: 0.0,
        breed: 0,
        repeat: 5,
        workers: threads.saturating_sub(1).max(1),
        fullscreen: false,
        scale: 0.0,
        fps: 30.0,
        hud: true,
        frames: 0,
        every: 0,
        keys: Vec::new(),
        load: Vec::new(),
        fresh: false,
        render: None,
        live_quality: None,
        crf: 17,
        credit: true,
        credit_secs: 10.0,
        supersample: None,
    };
    let mut seconds: Option<f64> = None;
    // the render size: Electric Sheep's unless a size or scale says otherwise
    let mut render: Option<Option<(u32, u32)>> = None;
    let mut quality_given = false;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    // a size asked for is rendered at that size; otherwise at most 1080 lines
    let mut sized = false;
    let num = |i: usize| -> f64 { args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| usage()) };
    while i < args.len() {
        match args[i].as_str() {
            "--shot" => {
                o.mode = Mode::Shot;
                i += 1;
                o.out = args.get(i).cloned().unwrap_or_else(|| usage());
            }
            "--seed" => {
                i += 1;
                o.seed = num(i) as u32;
            }
            "--quality" => {
                i += 1;
                o.quality = num(i) as f32;
                quality_given = true;
            }
            "--es" => render = Some(Some(ES_SIZE)),
            "--es-hd" => render = Some(Some(ES_HD_SIZE)),
            "--native" => render = Some(None),
            "--render" => {
                i += 1;
                let s = args.get(i).cloned().unwrap_or_else(|| usage());
                let mut it = s.split('x').map(|v| v.parse::<u32>().ok());
                match (it.next().flatten(), it.next().flatten()) {
                    (Some(w), Some(h)) if w >= 16 && h >= 16 => render = Some(Some((w, h))),
                    _ => usage(),
                }
            }
            "--angle" => {
                i += 1;
                o.angle = num(i) as f32;
            }
            "--breed" => {
                i += 1;
                o.breed = num(i) as u32;
            }
            "--repeat" => {
                i += 1;
                o.repeat = num(i) as u32;
            }
            "--threads" => {
                i += 1;
                o.workers = num(i) as usize;
            }
            "--varcheck" => o.mode = Mode::VarCheck,
            "--mutcheck" => o.mode = Mode::MutCheck,
            "--out" => {
                i += 1;
                o.out = args.get(i).cloned().unwrap_or_else(|| usage());
            }
            "--fullscreen" => o.fullscreen = true,
            "--fresh" => o.fresh = true,
            "--load" => {
                i += 1;
                o.load.push(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "--no-hud" => o.hud = false,
            "--scale" => {
                i += 1;
                o.scale = (num(i) as f32).clamp(0.1, 2.0);
            }
            "--fps" => {
                i += 1;
                o.fps = num(i).clamp(1.0, 240.0);
            }
            "--frames" => {
                i += 1;
                o.frames = num(i) as u64;
                if o.mode == Mode::Window {
                    o.mode = Mode::Frames;
                }
            }
            "--every" => {
                i += 1;
                o.every = num(i) as u64;
            }
            "--video" => {
                o.mode = Mode::Video;
                i += 1;
                o.out = args.get(i).cloned().unwrap_or_else(|| usage());
            }
            "--seconds" => {
                i += 1;
                seconds = Some(num(i).max(0.0));
            }
            "--credit" => {
                // --credit on (the default) / off; a bare --credit means on
                match args.get(i + 1).map(|s| s.as_str()) {
                    Some("on") => i += 1,
                    Some("off") => {
                        o.credit = false;
                        i += 1;
                    }
                    _ => {}
                }
            }
            "--credit-secs" => {
                i += 1;
                o.credit_secs = num(i).max(0.0);
            }
            "--supersample" => {
                i += 1;
                o.supersample = Some((num(i) as u32).clamp(1, 4));
            }
            "--crf" => {
                i += 1;
                o.crf = (num(i) as u32).min(63);
            }
            "--bench" => o.mode = Mode::Bench,
            "--keys" => {
                i += 1;
                for kv in args.get(i).unwrap_or_else(|| usage()).split(',') {
                    let (f, k) = kv.split_once(':').unwrap_or_else(|| usage());
                    o.keys.push((f.parse().unwrap_or_else(|_| usage()), k.chars().next().unwrap_or(' ')));
                }
            }
            "--cpu-only" => o.gpu = false,
            "--gpu-only" => o.cpu = false,
            "720p" => (o.w, o.h, sized) = (1280, 720, true),
            "1080p" => (o.w, o.h, sized) = (1920, 1080, true),
            "1440p" | "2k" => (o.w, o.h, sized) = (2560, 1440, true),
            "4k" | "2160p" => (o.w, o.h, sized) = (3840, 2160, true),
            s if s.contains('x') => {
                let mut it = s.split('x').map(|v| v.parse::<u32>().ok());
                match (it.next().flatten(), it.next().flatten()) {
                    (Some(w), Some(h)) if w >= 16 && h >= 16 => (o.w, o.h, sized) = (w, h, true),
                    _ => usage(),
                }
            }
            _ => usage(),
        }
        i += 1;
    }
    if !o.cpu {
        o.workers = 0;
    }
    if sized && o.scale == 0.0 {
        o.scale = 1.0;
    }
    // the window and headless runs default to Electric Sheep's frame size,
    // scaled to the window, unless a window size or --scale was given
    o.render = match render {
        Some(r) => r,
        None if !sized && o.scale == 0.0 && matches!(o.mode, Mode::Window | Mode::Frames) => Some(ES_SIZE),
        None => None,
    };
    if let (Some((rw, rh)), false) = (o.render, sized) {
        // a window of the picture's shape
        o.h = 720;
        o.w = (720.0 * rw as f32 / rh as f32).round() as u32;
    }
    if quality_given && matches!(o.mode, Mode::Window | Mode::Frames) {
        o.live_quality = Some(o.quality.max(1.0));
    }
    if o.mode == Mode::Video {
        // Electric Sheep's frame and quality unless asked otherwise
        if !sized {
            (o.w, o.h) = ES_SIZE;
        }
        if !quality_given {
            o.quality = 2000.0;
        }
        if let Some(s) = seconds {
            o.frames = (s * o.fps).round() as u64;
        }
        if o.frames == 0 {
            o.frames = (60.0 * o.fps) as u64;
        }
    }
    o
}

/// Where the window keeps its flock between runs.
pub fn flock_path() -> String {
    let base = std::env::var("XDG_DATA_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOME").ok().map(|h| format!("{h}/.local/share")))
        .unwrap_or_else(|| ".".into());
    format!("{base}/fleece/flock.flam3")
}

/// Where the genebank lives: every .flam3 file there (real Electric Sheep
/// genomes) feeds the breeding, as the server's genebank did.
pub fn genebank_dir() -> String {
    flock_path().trim_end_matches("flock.flam3").to_string() + "genebank"
}

/// The genebank to use: the user's (~/.local/share/fleece/genebank) if it
/// has sheep, else the `sheep/` directory that comes with the source.
pub fn genebank_source() -> String {
    let user = genebank_dir();
    let has = |d: &str| {
        std::fs::read_dir(d).is_ok_and(|mut it| it.any(|e| e.is_ok_and(|e| e.path().extension().is_some_and(|x| x == "flam3"))))
    };
    if has(&user) {
        return user;
    }
    let bundled = concat!(env!("CARGO_MANIFEST_DIR"), "/sheep").to_string();
    if has(&bundled) { bundled } else { user }
}

/// The genebank's flames, marked curated (kept as they are when cloned), and
/// their weights (the square root of their Electric Sheep rating, plus one).
pub fn load_genebank() -> (Vec<Genome>, Vec<f32>) {
    let mut files: Vec<String> = std::fs::read_dir(genebank_source())
        .map(|d| {
            d.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "flam3"))
                .map(|p| p.to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    let (mut out, mut weight) = (Vec::new(), Vec::new());
    for f in files {
        if let Ok(xml) = std::fs::read_to_string(&f) {
            for (mut g, rating) in flam3::parse(&xml).0 {
                g.curated = true;
                // genebank sheep go by their Electric Sheep file name when the
                // flame's own name doesn't say (Draves' genebank, render jobs)
                let stem = std::path::Path::new(&f).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                if !g.name.contains("electricsheep.") && stem.starts_with("electricsheep.") {
                    g.name = stem;
                }
                g.nc = g.nick.is_empty();
                out.push(g);
                weight.push((rating.max(0) as f32).sqrt() + 1.0);
            }
        }
    }
    (out, weight)
}

/// The flames in the --load files.
pub fn load_files(files: &[String]) -> Vec<Genome> {
    let mut out = Vec::new();
    for f in files {
        match std::fs::read_to_string(f) {
            Ok(xml) => {
                let (gs, unknown) = flam3::parse(&xml);
                let rated: Vec<i32> = gs.iter().map(|g| g.1).collect();
                eprintln!("{f}: {} flames{}", gs.len(), if rated.iter().any(|&r| r != 0) { format!(", ratings {rated:?}") } else { String::new() });
                if !unknown.is_empty() {
                    eprintln!("  ignored (unsupported here): {}", unknown.join(", "));
                }
                out.extend(gs.into_iter().map(|(g, _)| g));
            }
            Err(e) => eprintln!("fleece: {f}: {e}"),
        }
    }
    out
}

pub fn tone_of(g: &Genome) -> Tone {
    Tone {
        brightness: g.brightness,
        gamma: g.gamma,
        gamma_threshold: g.gamma_threshold,
        vibrancy: g.vibrancy,
        highlight: g.highlight,
        background: g.background,
        de_radius: g.de[0],
        de_min: g.de[1],
        de_curve: g.de[2],
        density: 0.0,
        filter: g.filter,
        exposure: 1.0,
    }
}

fn open_gl(w: u32, h: u32, headless: bool) -> (sdl::Window, Gpu) {
    let win = sdl::Window::open("fleece", w, h, headless, false).unwrap_or_else(|e| {
        eprintln!("fleece: {e}");
        std::process::exit(1)
    });
    let g = gl::Gl::load(sdl::Window::proc_address).unwrap_or_else(|e| {
        eprintln!("fleece: {e}");
        std::process::exit(1)
    });
    let gpu = Gpu::new(g, w, h).unwrap_or_else(|e| {
        eprintln!("fleece: shader: {e}");
        std::process::exit(1)
    });
    (win, gpu)
}

fn shot(o: &Opts) {
    let (_win, gpu) = open_gl(o.w, o.h, true);
    let mut r = Renderer::new(gpu, o.workers);
    r.use_gpu = o.gpu;
    let mut flock = breed::Flock::new(o.seed);
    let t0 = std::time::Instant::now();
    let mut sheep = flock.breed();
    for _ in 0..o.breed {
        flock.add(sheep);
        sheep = flock.breed();
    }
    let mut g = sheep.genome;
    if let Some(l) = load_files(&o.load).into_iter().next() {
        g = l;
    }
    println!(
        "{}: {} xforms{}, sym {}, palette {}, {} tested / {} rejected in {:.2} s",
        g.name,
        g.xforms.len(),
        if g.fin.is_some() { " + final" } else { "" },
        g.symmetry,
        g.palette_index,
        flock.tested,
        flock.rejected,
        t0.elapsed().as_secs_f64()
    );
    for (i, x) in g.xforms.iter().enumerate() {
        let vs: Vec<String> = x.vars.iter().map(|v| format!("{} {:.2}", var::info(v.id).name, v.w)).collect();
        println!("  xf{i} w {:.2} c {:.2}: {}", x.weight, x.color, vs.join(", "));
    }
    // the loop's phase: motion elements first, then the turn (flam3's sheep_loop)
    if g.has_motion() {
        g = g.moved(o.angle.rem_euclid(360.0) / 360.0);
    }
    // flam3's supersampling: the genome's own (the server's sheep say 4) unless asked
    let os = o.supersample.unwrap_or(g.supersample).clamp(1, 4);
    r.gpu.resize(o.w, o.h, os).unwrap_or_else(|e| {
        eprintln!("fleece: {e}");
        std::process::exit(1)
    });
    let (bw, bh) = (r.gpu.bw, r.gpu.bh);
    let mut s = vec![Sample::default()];
    g.sample(o.angle.to_radians(), bw, bh, &mut s[0]);
    let tone = tone_of(&g);
    let ppu = g.scale * bw as f32;
    // the shader tuned to this flame, as the live show uses (compiled before the clock starts)
    r.gpu.use_variations(&sample::variation_ids(&s), true);
    let target = o.quality as f64 * o.w as f64 * o.h as f64;
    let per = gpu::WALKERS as f64 * gpu::ITERS as f64;
    let t1 = std::time::Instant::now();
    let mut first = true;
    let (mut gi, mut ci) = (0u64, 0u64);
    loop {
        if let Some(st) = r.step(&s, tone, ppu * ppu, 8, true, first) {
            gi += st.gpu_iters;
            ci += st.cpu_iters;
        }
        first = false;
        r.gpu.finish();
        let est = r.total_iters() + if o.gpu { 8.0 * per } else { 0.0 };
        if est >= target {
            break;
        }
        if t1.elapsed().as_secs() > 120 {
            break;
        }
    }
    if let Some(st) = r.finish(true) {
        gi += st.gpu_iters;
        ci += st.cpu_iters;
    }
    // the density estimation and tone map too
    r.gpu.finish();
    let secs = t1.elapsed().as_secs_f64();
    println!(
        "{:.0} M iterations ({:.1} per pixel, supersample {os}) in {:.2} s: GPU {:.0} M/s, CPU {:.0} M/s ({} threads)",
        (gi + ci) as f64 / 1e6,
        (gi + ci) as f64 / (o.w as f64 * o.h as f64),
        secs,
        gi as f64 / secs / 1e6,
        ci as f64 / secs / 1e6,
        r.workers()
    );
    let rgb = r.gpu.read_rgb();
    // the genome too, as flam3 XML (with this quality), for flam3-render
    let xml = g.to_flam3(o.w, o.h).replacen("<flame ", &format!("<flame quality=\"{}\" ", o.quality), 1);
    let _ = std::fs::write(format!("{}.flam3", o.out.trim_end_matches(".png")), xml);
    std::fs::write(&o.out, png::encode(o.w, o.h, &rgb)).unwrap_or_else(|e| {
        eprintln!("fleece: {}: {e}", o.out);
        std::process::exit(1)
    });
}

/// Renders a small flame made of each variation on the CPU alone and on the
/// GPU alone, and compares where the points landed: the two chaos games
/// must agree on every variation.
fn varcheck(o: &Opts) {
    let (w, h) = (320u32, 180u32);
    let (_win, gpu) = open_gl(w, h, true);
    let mut r = Renderer::new(gpu, o.workers.max(1));
    // every point of a frame in that frame: wait for the slowest worker
    if let Some(p) = &r.pool {
        p.patience.store(if std::env::var_os("FLEECE_IMPATIENT").is_some() { 2000 } else { 5_000_000 }, std::sync::atomic::Ordering::Relaxed);
    }
    let mut worst = Vec::new();
    for (n, vi) in var::VARS.iter().enumerate() {
        if o.breed > 0 && vi.id != o.breed as i32 - 1 {
            continue;
        }
        let mut rg = rng::Rng::new(o.seed ^ (vi.id as u32 * 7919));
        let mut g = Genome::random(rg.next_u32());
        g.fin = None;
        g.xforms.truncate(3);
        while g.xforms.len() < 3 {
            g.xforms.push(genome::Xform::new());
        }
        for x in g.xforms.iter_mut() {
            x.weight = 1.0;
            x.post = genome::IDENTITY;
            x.vars = vec![
                genome::VarUse { id: vi.id, w: 0.8, p: var::random_params(vi.id, &mut rg) },
                genome::VarUse { id: 0, w: 0.2, p: [0.0; crate::genome::NP] },
            ];
        }
        breed::frame(&mut g, &mut rg);
        if !(g.scale > 0.02 && g.scale < 100.0) {
            g.center = [0.0, 0.0];
            g.scale = 0.25;
        }
        let mut s = vec![Sample::default()];
        g.sample(0.3, w, h, &mut s[0]);
        let tone = tone_of(&g);
        let ppu = g.scale * w as f32;
        let mut dens = [vec![], vec![]];
        let mut rate = [0f64; 2];
        #[allow(clippy::needless_range_loop)]
        for side in 0..2 {
            r.use_gpu = side == 1;
            r.use_cpu = side == 0;
            let mut first = true;
            for _ in 0..if side == 0 { 40 } else { 2 } {
                r.step(&s, tone, ppu * ppu, 2, true, first);
                first = false;
                r.gpu.finish();
            }
            r.finish(true);
            let acc = r.gpu.read_acc();
            if o.breed > 0 {
                let f = format!("{}/vc-{}-{}.png", o.out, vi.name, if side == 0 { "cpu" } else { "gpu" });
                let _ = std::fs::write(&f, png::encode(w, h, &r.gpu.read_rgb()));
                let hits: f64 = acc.iter().skip(3).step_by(4).map(|&v| v as f64).sum();
                println!("{} side {}: {:.0} hits on screen of {:.0} iterations; scale {} centre {:?} bright {} gamma {}", vi.name, side, hits, r.total_iters(), g.scale, g.center, g.brightness, g.gamma);
            }
            // 32 x 18 cells of the hit counts, normalised
            let mut cells = vec![0f64; 32 * 18];
            for y in 0..h as usize {
                for x in 0..w as usize {
                    cells[(y * 18 / h as usize) * 32 + x * 32 / w as usize] += acc[(y * w as usize + x) * 4 + 3] as f64;
                }
            }
            let t: f64 = cells.iter().sum();
            rate[side] = t / r.total_iters().max(1.0);
            dens[side] = cells.iter().map(|c| c / t.max(1.0)).collect::<Vec<f64>>();
            // too few points on screen to compare (a flame that escapes)
            if t < 200_000.0 {
                dens[side] = vec![-1.0];
            }
        }
        let empty = [dens[0].len() == 1, dens[1].len() == 1];
        let d = match empty {
            [true, true] => {
                println!("{:3} {:2} {:14} too few points on screen on both sides (skipped)", n, vi.id, vi.name);
                continue;
            }
            [false, false] => dens[0].iter().zip(&dens[1]).map(|(a, b)| (a - b).abs()).sum::<f64>(),
            // one side short of points: agreeing if both land the same share on screen
            _ if (rate[0] / rate[1].max(1e-12) - 1.0).abs() < 0.3 => {
                println!("{:3} {:2} {:14} same share on screen ({:.2} % / {:.2} %), too few points to compare more", n, vi.id, vi.name, rate[0] * 100.0, rate[1] * 100.0);
                continue;
            }
            _ => f64::NAN,
        };
        println!(
            "{:3} {:2} {:14} L1 {:.3}{}",
            n,
            vi.id,
            vi.name,
            d,
            if d.is_nan() {
                format!("   <-- only the {} side has points", if empty[0] { "GPU" } else { "CPU" })
            } else if d >= 0.15 {
                "   <--".into()
            } else {
                String::new()
            }
        );
        worst.push((d, vi.name));
    }
    let bad = worst.iter().filter(|(d, _)| !(*d < 0.15)).count();
    println!("{} of {} variations differ (L1 >= 0.15 or no points)", bad, worst.len());
}

/// Chaos-game throughput of each side alone, over a few bred sheep.
fn bench(o: &Opts) {
    let (w, h) = (o.w, o.h);
    let (_win, gpu) = open_gl(w, h, true);
    let mut r = Renderer::new(gpu, o.workers);
    let mut flock = breed::Flock::new(o.seed);
    let (mut tg, mut tc) = (0.0, 0.0);
    let n = 6;
    for i in 0..n {
        let g = flock.breed().genome;
        let mut s = vec![Sample::default(); 16];
        for (k, smp) in s.iter_mut().enumerate() {
            g.sample(k as f32 * 0.003, w, h, smp);
        }
        let tone = tone_of(&g);
        let ppu = g.scale * w as f32;
        // GPU alone: 24 rounds after a warm-up, with the general shader and
        // with one specialised to this sheep's variations
        r.use_gpu = true;
        r.use_cpu = false;
        let mut ids: Vec<i32> = g.xforms.iter().chain(g.fin.iter()).flat_map(|x| x.vars.iter().map(|v| v.id)).collect();
        ids.sort();
        ids.dedup();
        let mut rates = [0.0; 2];
        for (k, rate) in rates.iter_mut().enumerate() {
            if k == 0 {
                r.gpu.use_all_variations();
            } else {
                r.gpu.use_variations(&ids, true);
            }
            r.step(&s, tone, ppu * ppu, 4, false, true);
            r.gpu.finish();
            // the best of several runs: the GPU is shared, interference only slows
            for _ in 0..o.repeat.max(1) {
                let t = std::time::Instant::now();
                r.step(&s, tone, ppu * ppu, 24, false, false);
                r.gpu.finish();
                *rate = f64::max(*rate, 24.0 * gpu::WALKERS as f64 * gpu::ITERS as f64 / t.elapsed().as_secs_f64());
            }
            r.finish(false);
        }
        let gs = rates[1];
        print!("[general {:5.2}, specialised {:5.2} G/s, compiled in {:.0} ms] ", rates[0] / 1e9, rates[1] / 1e9, r.gpu.compile_ms);
        // CPU alone: two seconds
        r.use_gpu = false;
        r.use_cpu = r.workers() > 0;
        let mut cs = 0.0;
        if r.use_cpu {
            r.step(&s, tone, ppu * ppu, 0, false, true);
            let t = std::time::Instant::now();
            std::thread::sleep(std::time::Duration::from_secs(2));
            let st = r.finish(false).unwrap();
            cs = st.cpu_iters as f64 / t.elapsed().as_secs_f64();
        }
        let vars: usize = g.xforms.iter().map(|x| x.vars.len()).sum();
        println!(
            "sheep {i}: {:2} xforms, {:2} variations{}: GPU {:6.2} G/s   CPU {:5.2} G/s",
            g.xforms.len(),
            vars,
            if g.fin.is_some() { " + final" } else { "        " },
            gs / 1e9,
            cs / 1e9
        );
        tg += gs;
        tc += cs;
    }
    println!("mean: GPU {:.2} G iterations/s, CPU {:.2} G/s ({} threads)", tg / n as f64 / 1e9, tc / n as f64 / 1e9, r.workers());
}

/// How often a mutation (as the M key makes them) comes out dark on screen:
/// untested (framed only) against put through the breeder's tests.
fn mutcheck(o: &Opts) {
    let mut flock = breed::Flock::new(o.seed);
    let mut r = rng::Rng::new(o.seed ^ 77);
    let n = o.frames.max(100) as u32;
    let (mut dark_raw, mut dark_tested, mut dark_parent) = (0, 0, 0);
    let (mut tested, mut rejected) = (0, 0);
    for i in 0..n {
        let p = flock.breed().genome;
        if breed::view_brightness(&p, i) < breed::MIN_VIEW {
            dark_parent += 1;
        }
        let mut m = breed::mutate(&p, &mut r);
        breed::frame(&mut m, &mut r);
        if breed::view_brightness(&m, i ^ 0x55) < breed::MIN_VIEW {
            dark_raw += 1;
        }
        let t = breed::first_good(
            || {
                let g = breed::mutate(&p, &mut r);
                (g, r.next_u32())
            },
            &mut tested,
            &mut rejected,
        );
        if breed::view_brightness(&t, i ^ 0xaa) < breed::MIN_VIEW {
            dark_tested += 1;
        }
    }
    let pc = |d: u32| 100.0 * d as f32 / n as f32;
    println!("{n} parents bred with the new tests: {dark_parent} dark ({:.1} %)", pc(dark_parent));
    println!("mutations, untested: {dark_raw} dark ({:.1} %)", pc(dark_raw));
    println!("mutations, tested:   {dark_tested} dark ({:.1} %); {tested} tried, {rejected} rejected", pc(dark_tested));
}

fn main() {
    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
        eprintln!("fleece: this CPU lacks AVX2 / FMA");
        std::process::exit(1);
    }
    let o = parse();
    match o.mode {
        Mode::Shot => shot(&o),
        Mode::VarCheck => varcheck(&o),
        Mode::MutCheck => mutcheck(&o),
        Mode::Window => app::window(&o),
        Mode::Frames => app::headless(&o),
        Mode::Bench => bench(&o),
        Mode::Video => app::video(&o),
    }
}
