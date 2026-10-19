//! fleece: fractal flames in the manner of Electric Sheep, rendered live on
//! the GPU and every CPU core at once.
// nested ifs read better here; `!(a > b)` is how NaN is kept out
#![allow(clippy::collapsible_if, clippy::neg_cmp_op_on_partial_ord, clippy::new_without_default)]

mod app;
mod audio;
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
mod music;
mod png;
mod render;
mod rng;
mod sample;
mod samples;
mod sdl;
mod show;
mod simd;
mod taste;
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
    TasteFit,
    AudioCheck,
    MusicWav,
    SamplesList,
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
    /// for stills, automatic for the show and videos)
    pub supersample: Option<u32>,
    /// v2's effects to start with (--classic: none)
    pub fx: app::Fx,
    /// new sheep are the taste's pick of a few candidates (--taste off: the first)
    pub taste: bool,
    /// --audio FILE: a recording the video is made to (and carries)
    pub soundtrack: Option<String>,
    /// a recording's sound this much later (seconds; else the live lag)
    pub sound_delay: Option<f64>,
    /// the window optimizes itself for the display's rate at the start
    pub optimize: bool,
    /// --musicwav: genre, seconds, file
    pub musicwav: Vec<String>,
    /// generated music (a music::mix_id: a genre, or a primary and a secondary): played in the window, the soundtrack of a video
    pub music: Option<usize>,
    /// --samples DIR: the recorded instruments' directory (else ~/.local/share/fleece/samples)
    samples: Option<String>,
    /// --no-samples: the music all synthesiser
    no_samples: bool,
}

/// Electric Sheep's frame sizes: the standard flock and the HD (gold) one.
pub const ES_SIZE: (u32, u32) = (800, 592);
pub const ES_HD_SIZE: (u32, u32) = (1280, 720);

fn usage() -> ! {
    eprintln!("usage: fleece [WxH | 720p | 1080p | 2k | 4k] [--fullscreen] [--seed N] [--no-hud]");
    eprintln!("              [--es | --es-hd | --render WxH | --native | --scale F] [--quality N]");
    eprintln!("              [--load FILE.flam3]... [--fresh] [--cpu-only | --gpu-only] [--threads N]");
    eprintln!("              [--classic] [--glow 0-3] [--trails 0-2] [--levels|--drift|--hue|--wobble on|off]");
    eprintln!("              [--tilt 0-2] [--audio [MUSIC]] [--audio-strength 0-3] [--music GENRE] [--sound-delay MS]");
    eprintln!("              [--optimize on|off] [--supersample 1-4] [--taste on|off]");
    eprintln!("       (default: Electric Sheep's 800x592 frame, scaled to the window, at 30 fps, with v2's");
    eprintln!("       effects: glow, short trails, auto levels, camera drift, motion for every sheep; --classic: none)");
    eprintln!("       --music GENRE: generated music the show follows ({}; or two, as polka+techno: the first's music, the second's drums); a video's soundtrack", music::cli_names());
    eprintln!("       fleece --shot OUT.png [--quality Q] [--angle DEG] [--breed N] [size]   one still");
    eprintln!("       fleece --frames N [--fps F] [--out DIR --every K] [--keys F:K,...]   headless run");
    eprintln!("       fleece --video OUT.mp4 [--seconds S | --frames N] [--fps F] [--quality Q] [--crf N] [size]");
    eprintln!("              [--credit on|off]   a video (any ffmpeg format by extension; .y4m needs no ffmpeg;");
    eprintln!("              - is raw RGB24 on stdout); the credit is burned in for the first --credit-secs (10)");
    eprintln!("              unless off, and always in the metadata");
    eprintln!("       fleece --varcheck   CPU against GPU, every variation");
    eprintln!("       fleece --audiocheck [FILE.wav | -] [--seconds S]   audio levels of a WAV (-: stdin), or live from");
    eprintln!("              what the speakers play for S seconds (10)");
    eprintln!("       fleece --musicwav GENRE SECONDS OUT.wav [--seed N]   generated music to a WAV");
    eprintln!("       --samples DIR: recorded instruments for the music (~/.local/share/fleece/samples by default;");
    eprintln!("              tools/get-samples.sh fetches them); --no-samples: all synthesiser");
    eprintln!("       fleece --samples-list   the samples directory's SoundFont presets and what loads");
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
        fx: app::Fx::V2,
        taste: true,
        soundtrack: None,
        sound_delay: None,
        optimize: true,
        musicwav: Vec::new(),
        music: None,
        samples: None,
        no_samples: false,
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
            "--tastefit" => o.mode = Mode::TasteFit,
            "--audiocheck" => {
                o.mode = Mode::AudioCheck;
                // a WAV file to analyse (- for stdin); none: live
                if let Some(f) = args.get(i + 1).filter(|s| !s.starts_with("--")) {
                    o.out = f.clone();
                    i += 1;
                }
            }
            "--musicwav" => {
                // GENRE SECONDS OUT.wav: generated music to a file
                o.mode = Mode::MusicWav;
                o.musicwav = args.get(i + 1..i + 4).unwrap_or_else(|| usage()).to_vec();
                i += 3;
            }
            "--out" => {
                i += 1;
                o.out = args.get(i).cloned().unwrap_or_else(|| usage());
            }
            "--samples" => {
                i += 1;
                o.samples = Some(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "--no-samples" => o.no_samples = true,
            "--samples-list" => o.mode = Mode::SamplesList,
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
            "--classic" => o.fx = app::Fx::CLASSIC,
            "--glow" => {
                i += 1;
                o.fx.glow = (num(i) as u8).min(3);
            }
            "--trails" => {
                i += 1;
                o.fx.trails = (num(i) as u8).min(2);
            }
            "--tilt" => {
                i += 1;
                o.fx.tilt = (num(i) as u8).min(2);
            }
            "--sound-delay" => {
                i += 1;
                o.sound_delay = Some(num(i) / 1000.0);
            }
            "--optimize" => match args.get(i + 1).map(|s| s.as_str()) {
                Some("off") => {
                    o.optimize = false;
                    i += 1;
                }
                Some("on") => i += 1,
                _ => {}
            },
            "--audio-strength" => {
                i += 1;
                o.fx.audio = (num(i) as u8).min(3);
            }
            "--audio" => {
                if o.fx.audio == 0 {
                    o.fx.audio = 2;
                }
                // a recording to make the video to (else: what the speakers play, live)
                if let Some(f) = args.get(i + 1).filter(|f| !f.starts_with("--") && std::path::Path::new(f.as_str()).is_file()) {
                    o.soundtrack = Some(f.clone());
                    i += 1;
                }
            }
            "--music" => {
                i += 1;
                match args.get(i).and_then(|g| music::find(g)) {
                    Some(g) => {
                        o.music = Some(g);
                        if o.fx.audio == 0 {
                            o.fx.audio = 2;
                        }
                    }
                    None => {
                        eprintln!("fleece: --music: one of {}, or two (polka+techno: the first's music, the second's drums)", music::cli_names());
                        std::process::exit(2)
                    }
                }
            }
            "--taste" => match args.get(i + 1).map(|s| s.as_str()) {
                Some("off") => {
                    o.taste = false;
                    i += 1;
                }
                Some("on") => i += 1,
                _ => {}
            },
            "--levels" | "--drift" | "--hue" | "--wobble" => {
                let flag = args[i].clone();
                let on = match args.get(i + 1).map(|s| s.as_str()) {
                    Some("off") => {
                        i += 1;
                        false
                    }
                    Some("on") => {
                        i += 1;
                        true
                    }
                    _ => true,
                };
                match flag.as_str() {
                    "--levels" => o.fx.levels = on,
                    "--drift" => o.fx.drift = on,
                    "--hue" => o.fx.hue = on,
                    _ => o.fx.wobble = on,
                }
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
        // a minute, or as long as the soundtrack
        if o.frames == 0 && o.soundtrack.is_none() {
            o.frames = (60.0 * o.fps) as u64;
        }
    }
    if o.mode == Mode::AudioCheck {
        // the live check's length, in tenths of a second
        o.frames = (seconds.unwrap_or(10.0) * 10.0).round() as u64;
    }
    o
}

/// Where the window keeps the taste its votes have taught.
pub fn taste_path() -> String {
    flock_path().replace("flock.flam3", "taste.txt")
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
    let f = taste::features(&g, o.seed);
    let t = taste::Taste::prior();
    let ms: Vec<String> = taste::NAMES.iter().zip(f).map(|(n, v)| format!("{n} {v:.3}")).collect();
    println!("  taste {:.2}: {}", t.score(&f), ms.join(", "));
    for (i, x) in g.xforms.iter().enumerate() {
        let vs: Vec<String> = x.vars.iter().map(|v| format!("{} {:.2}", var::info(v.id).name, v.w)).collect();
        println!("  xf{i} w {:.2} c {:.2}: {}", x.weight, x.color, vs.join(", "));
    }
    // the loop's phase: motion elements first, then the turn (flam3's sheep_loop)
    if g.has_motion() {
        g = g.moved(o.angle.rem_euclid(360.0) / 360.0, true);
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
/// GPU alone, and compares where the points landed and their colours: the
/// two chaos games must agree on every variation. The 3D and direct-colour
/// ones are seen through a tilted camera, so z matters (with zblur feeding
/// z in), as are a few of flam3's, and one with depth of field.
fn varcheck(o: &Opts) {
    let (w, h) = (320u32, 180u32);
    let (_win, gpu) = open_gl(w, h, true);
    let mut r = Renderer::new(gpu, o.workers.max(1));
    // every point of a frame in that frame: wait for the slowest worker
    if let Some(p) = &r.pool {
        p.patience.store(if std::env::var_os("FLEECE_IMPATIENT").is_some() { 2000 } else { 5_000_000 }, std::sync::atomic::Ordering::Relaxed);
    }
    let mut worst = Vec::new();
    let tilt = genome::Cam3 { pitch: 0.6, persp: 0.3, ..Default::default() };
    let mut cases: Vec<(&var::VarInfo, genome::Cam3, &str)> = var::VARS
        .iter()
        .map(|vi| if var::needs_3d(vi.id) { (vi, tilt, "") } else { (vi, genome::Cam3::default(), "") })
        .collect();
    // flam3's z: passed on (spherical), bubble's, cylinder's, pre_blur's
    for id in [2, 28, 29, 67] {
        cases.push((var::info(id), tilt, " tilted"));
    }
    let dof = genome::Cam3 { pitch: 0.5, yaw: 0.4, persp: 0.2, zpos: 0.1, dof: 0.3 };
    cases.push((var::info(101), dof, " dof"));
    for (n, &(vi, cam, tag)) in cases.iter().enumerate() {
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
        // tilted, the second variation is julia (z = 0), so that z settles;
        // pre_blur keeps linear (it adds nothing of its own)
        let second = if cam.is_flat() || vi.id == 67 { 0 } else { 13 };
        for x in g.xforms.iter_mut() {
            x.weight = 1.0;
            x.post = genome::IDENTITY;
            x.vars = vec![
                genome::VarUse { id: vi.id, w: 0.8, p: var::random_params(vi.id, &mut rg) },
                genome::VarUse { id: second, w: 0.2, p: [0.0; crate::genome::NP] },
            ];
        }
        if !cam.is_flat() {
            g.xforms[0].vars.push(genome::VarUse { id: 105, w: 0.3, p: [0.0; crate::genome::NP] });
        }
        breed::frame(&mut g, &mut rg);
        if !(g.scale > 0.02 && g.scale < 100.0) {
            g.center = [0.0, 0.0];
            g.scale = 0.25;
        }
        // framed flat, then tilted about the centre of that view, focused on
        // the middle of the flame's depth (bubble's and dc_bubble's z is 2 or more)
        if !cam.is_flat() {
            g.cam = genome::Cam3 { zpos: cam.zpos + median_z(&g, 0.3, rg.next_u32()), ..cam };
        }
        let mut s = vec![Sample::default()];
        g.sample(0.3, w, h, &mut s[0]);
        let tone = tone_of(&g);
        let ppu = g.scale * w as f32;
        // the shader specialised to it, as the show uses (3D: its 3D variant)
        r.gpu.use_variations(&sample::variation_ids(&s), true);
        let mut dens = [vec![], vec![]];
        let mut cols = [vec![], vec![]];
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
            // 32 x 18 cells of the hit counts and of r, g and b, normalised
            let mut cells = vec![0f64; 32 * 18];
            let mut rgb = vec![0f64; 32 * 18 * 3];
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let c = (y * 18 / h as usize) * 32 + x * 32 / w as usize;
                    let a = &acc[(y * w as usize + x) * 4..][..4];
                    cells[c] += a[3] as f64;
                    for k in 0..3 {
                        rgb[c * 3 + k] += a[k] as f64;
                    }
                }
            }
            let t: f64 = cells.iter().sum();
            let tc: f64 = rgb.iter().sum();
            rate[side] = t / r.total_iters().max(1.0);
            dens[side] = cells.iter().map(|c| c / t.max(1.0)).collect::<Vec<f64>>();
            cols[side] = rgb.iter().map(|c| c / tc.max(1.0)).collect::<Vec<f64>>();
            // too few points on screen to compare (a flame that escapes)
            if t < 200_000.0 {
                dens[side] = vec![-1.0];
            }
        }
        let empty = [dens[0].len() == 1, dens[1].len() == 1];
        let name = format!("{}{tag}", vi.name);
        let l1 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| (a - b).abs()).sum::<f64>();
        let (d, dc) = match empty {
            [true, true] => {
                println!("{:3} {:3} {:20} too few points on screen on both sides (skipped)", n, vi.id, name);
                continue;
            }
            [false, false] => (l1(&dens[0], &dens[1]), l1(&cols[0], &cols[1])),
            // one side short of points: agreeing if both land the same share on screen
            _ if (rate[0] / rate[1].max(1e-12) - 1.0).abs() < 0.3 => {
                println!("{:3} {:3} {:20} same share on screen ({:.2} % / {:.2} %), too few points to compare more", n, vi.id, name, rate[0] * 100.0, rate[1] * 100.0);
                continue;
            }
            _ => (f64::NAN, f64::NAN),
        };
        println!(
            "{:3} {:3} {:20} L1 {:.3}, colour {:.3}{}",
            n,
            vi.id,
            name,
            d,
            dc,
            if d.is_nan() {
                format!("   <-- only the {} side has points", if empty[0] { "GPU" } else { "CPU" })
            } else if d.max(dc) >= 0.15 {
                "   <--".into()
            } else {
                String::new()
            }
        );
        // where the points land, and their colours
        worst.push((d.max(dc), name));
    }
    let bad = worst.iter().filter(|(d, _)| !(*d < 0.15)).count();
    println!("{} of {} cases differ (L1 >= 0.15 or no points)", bad, worst.len());
}

/// The median z of a flame's points at `angle`: seen edge-on (a quarter-turn
/// pitch, no perspective) the 3D camera puts the centre's y minus z on y.
fn median_z(g: &Genome, angle: f32, seed: u32) -> f32 {
    let mut probe = g.clone();
    probe.cam = genome::Cam3 { pitch: core::f32::consts::FRAC_PI_2, ..Default::default() };
    let mut s = Box::new(Sample::default());
    probe.sample(angle, 128, 128, &mut s);
    let mut pts = Vec::with_capacity(50_000);
    cpu::sample_points(&s, 50_000, seed, &mut pts);
    let mut zs: Vec<f32> = pts.iter().map(|p| g.center[1] - p[1]).filter(|z| z.is_finite()).collect();
    if zs.is_empty() {
        return 0.0;
    }
    zs.sort_by(f32::total_cmp);
    zs[zs.len() / 2]
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

/// Fits taste::Taste's prior by ridge regression and prints it (with its
/// cross-validated correlation) to paste into taste.rs: the measures of the
/// genebank's real sheep (1) against new random sheep that passed the tests
/// (0), so the score says how much a candidate looks like a survivor of
/// Electric Sheep's flock. With --breed 1: against ln(1 + rating) among the
/// rated real sheep instead.
fn tastefit(o: &Opts) {
    let (bank, weight) = load_genebank();
    let threads = o.workers.max(1);
    let t0 = std::time::Instant::now();
    let rated: Vec<(Genome, f32)> = if o.breed == 1 {
        bank.into_iter().zip(weight).filter(|(_, w)| *w > 1.0).map(|(g, w)| (g, ((w - 1.0) * (w - 1.0)).ln_1p())).collect()
    } else {
        let m = bank.len().min(3000);
        let per = m.div_ceil(threads);
        let bred: Vec<Genome> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    sc.spawn(move || {
                        let mut f = breed::Flock::new(o.seed ^ (t as u32 * 7919 + 1));
                        // candidates as they come, not already chosen by a taste
                        f.use_taste = false;
                        (0..per).map(|_| f.breed().genome).collect::<Vec<_>>()
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        println!("{} random sheep bred in {:.1} s", bred.len(), t0.elapsed().as_secs_f64());
        let step = (bank.len() / m).max(1);
        bank.into_iter().step_by(step).take(m).map(|g| (g, 1.0)).chain(bred.into_iter().map(|g| (g, 0.0))).collect()
    };
    let n = rated.len();
    let feats: Vec<[f32; taste::NF]> = std::thread::scope(|sc| {
        let chunks: Vec<_> = rated
            .chunks(n.div_ceil(threads).max(1))
            .enumerate()
            .map(|(c, ch)| {
                sc.spawn(move || ch.iter().enumerate().map(|(i, (g, _))| taste::features(g, (c * 100_000 + i) as u32 ^ o.seed)).collect::<Vec<_>>())
            })
            .collect();
        chunks.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let ys: Vec<f64> = rated.iter().map(|r| r.1 as f64).collect();
    println!("{n} rated sheep measured in {:.1} s", t0.elapsed().as_secs_f64());
    let nf = taste::NF;
    let mut mean = [0f32; taste::NF];
    let mut std = [0f32; taste::NF];
    for i in 0..nf {
        let m = feats.iter().map(|f| f[i] as f64).sum::<f64>() / n as f64;
        let v = feats.iter().map(|f| (f[i] as f64 - m).powi(2)).sum::<f64>() / n as f64;
        mean[i] = m as f32;
        std[i] = v.sqrt().max(1e-6) as f32;
    }
    let z = |f: &[f32; taste::NF]| -> Vec<f64> { (0..nf).map(|i| (((f[i] - mean[i]) / std[i]) as f64).clamp(-4.0, 4.0)).collect() };
    // ridge regression on rows `idx`: (Z'Z + lambda I) w = Z'y, with the bias unpenalised
    let fit = |idx: &[usize]| -> (Vec<f64>, f64) {
        let ym = idx.iter().map(|&i| ys[i]).sum::<f64>() / idx.len() as f64;
        let mut a = vec![vec![0f64; nf + 1]; nf];
        for &r in idx {
            let zr = z(&feats[r]);
            for i in 0..nf {
                for j in 0..nf {
                    a[i][j] += zr[i] * zr[j];
                }
                a[i][nf] += zr[i] * (ys[r] - ym);
            }
        }
        for (i, row) in a.iter_mut().enumerate() {
            row[i] += 0.05 * idx.len() as f64;
        }
        // Gauss-Jordan
        for c in 0..nf {
            let p = (c..nf).max_by(|&x, &y| a[x][c].abs().total_cmp(&a[y][c].abs())).unwrap();
            a.swap(c, p);
            let d = a[c][c];
            for v in a[c].iter_mut() {
                *v /= d;
            }
            for r in 0..nf {
                if r != c {
                    let k = a[r][c];
                    let row_c = a[c].clone();
                    for (v, rc) in a[r].iter_mut().zip(row_c) {
                        *v -= k * rc;
                    }
                }
            }
        }
        ((0..nf).map(|i| a[i][nf]).collect(), ym)
    };
    let corr = |p: &[f64], y: &[f64]| {
        let (mp, my) = (p.iter().sum::<f64>() / p.len() as f64, y.iter().sum::<f64>() / y.len() as f64);
        let num: f64 = p.iter().zip(y).map(|(a, b)| (a - mp) * (b - my)).sum();
        let den = (p.iter().map(|a| (a - mp).powi(2)).sum::<f64>() * y.iter().map(|b| (b - my).powi(2)).sum::<f64>()).sqrt();
        num / den.max(1e-12)
    };
    // five-fold cross-validation
    let (mut pred, mut truth) = (Vec::new(), Vec::new());
    for k in 0..5 {
        let train: Vec<usize> = (0..n).filter(|i| i % 5 != k).collect();
        let (w, b) = fit(&train);
        for i in (0..n).filter(|i| i % 5 == k) {
            pred.push(b + z(&feats[i]).iter().zip(&w).map(|(a, b)| a * b).sum::<f64>());
            truth.push(ys[i]);
        }
    }
    let all: Vec<usize> = (0..n).collect();
    let (w, b) = fit(&all);
    println!("cross-validated correlation with ln(1 + rating): {:.3}", corr(&pred, &truth));
    for i in 0..nf {
        let single: Vec<f64> = feats.iter().map(|f| f[i] as f64).collect();
        println!("  {:11} mean {:7.4} std {:7.4} weight {:+.4}  (alone: r {:+.3})", taste::NAMES[i], mean[i], std[i], w[i], corr(&single, &ys));
    }
    // each measure's spread in each class (1: real sheep or the high rated, 0: the rest)
    for i in 0..nf {
        let q = |label: bool| {
            let mut v: Vec<f32> = feats.iter().zip(&ys).filter(|(_, y)| (**y > 0.5) == label).map(|(f, _)| f[i]).collect();
            v.sort_by(f32::total_cmp);
            let at = |p: f64| v.get(((v.len() as f64 - 1.0) * p) as usize).copied().unwrap_or(0.0);
            format!("{:.3} {:.3} {:.3} {:.3}", at(0.01), at(0.5), at(0.99), at(0.999))
        };
        println!("  {:11} p1 p50 p99 p99.9  real {}   random {}", taste::NAMES[i], q(true), q(false));
    }
    let f = |v: &[f32]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(", ");
    println!("mean: [{}],", f(&mean));
    println!("std: [{}],", f(&std));
    println!("w: [{}],", f(&w.iter().map(|&x| x as f32).collect::<Vec<_>>()));
    println!("bias: {b:.4}");
}

fn main() {
    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
        eprintln!("fleece: this CPU lacks AVX2 / FMA");
        std::process::exit(1);
    }
    let o = parse();
    samples::configure(o.samples.clone(), o.no_samples);
    match o.mode {
        Mode::Shot => shot(&o),
        Mode::VarCheck => varcheck(&o),
        Mode::MutCheck => mutcheck(&o),
        Mode::Window => app::window(&o),
        Mode::Frames => app::headless(&o),
        Mode::Bench => bench(&o),
        Mode::Video => app::video(&o),
        Mode::TasteFit => tastefit(&o),
        Mode::AudioCheck => audio::check(&o.out, o.frames),
        Mode::MusicWav => music::cli(&o.musicwav, o.seed as u64),
        Mode::SamplesList => samples::list_cli(),
    }
}
