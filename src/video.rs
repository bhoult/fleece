//! Where `--video` frames go: a file encoded by ffmpeg (MP4, MKV, WebM, MOV,
//! GIF..., chosen by the extension), a YUV4MPEG2 file written here with no
//! encoder at all (`.y4m`), or raw RGB24 on stdout (`-`).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::{Child, ChildStdin, Command, Stdio};

pub enum Sink {
    Raw(BufWriter<std::io::Stdout>),
    Y4m { out: BufWriter<File>, w: usize, h: usize, frame: Vec<u8> },
    Ffmpeg { child: Child, stdin: BufWriter<ChildStdin> },
}

impl Sink {
    /// Opens `path` for `w` x `h` frames at `fps`.
    /// `preset`: x264's speed / size trade ("slow" for --video, faster for
    /// recording the live show).
    /// `audio`: a soundtrack to put in the file with the frames (ffmpeg only).
    pub fn open(path: &str, w: u32, h: u32, fps: f64, crf: u32, preset: &str, audio: Option<&str>) -> Result<Sink, String> {
        if path == "-" {
            return Ok(Sink::Raw(BufWriter::with_capacity(1 << 22, std::io::stdout())));
        }
        let ext = std::path::Path::new(path).extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        if let Some(dir) = std::path::Path::new(path).parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        if ext == "y4m" {
            if !w.is_multiple_of(2) || !h.is_multiple_of(2) {
                return Err("a .y4m video needs an even width and height".into());
            }
            let mut out = BufWriter::with_capacity(1 << 22, File::create(path).map_err(|e| format!("{path}: {e}"))?);
            let (num, den) = rational(fps);
            writeln!(out, "YUV4MPEG2 W{w} H{h} F{num}:{den} Ip A1:1 C420jpeg XYSCSS=420JPEG").map_err(|e| e.to_string())?;
            return Ok(Sink::Y4m { out, w: w as usize, h: h as usize, frame: Vec::new() });
        }
        // everything else: ffmpeg, picked by the extension
        let mut args: Vec<String> = ["-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"].map(String::from).to_vec();
        args.extend(["-s".into(), format!("{w}x{h}"), "-r".into(), format!("{fps}"), "-i".into(), "-".into()]);
        if let Some(a) = audio.filter(|_| ext != "gif") {
            args.extend(["-i", a, "-map", "0:v", "-map", "1:a", "-shortest"].map(String::from));
            let codec = if ext == "webm" { ["-c:a", "libopus", "-b:a", "160k"] } else { ["-c:a", "aac", "-b:a", "192k"] };
            args.extend(codec.map(String::from));
        }
        if !w.is_multiple_of(2) || !h.is_multiple_of(2) {
            args.extend(["-vf".into(), "pad=ceil(iw/2)*2:ceil(ih/2)*2".into()]);
        }
        match ext.as_str() {
            "webm" => args.extend(
                ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", &(crf + 13).to_string(), "-row-mt", "1", "-pix_fmt", "yuv420p"]
                    .map(String::from),
            ),
            "gif" => {}
            _ => {
                args.extend(["-c:v", "libx264", "-preset", preset, "-crf", &crf.to_string(), "-pix_fmt", "yuv420p"].map(String::from));
                if ext == "mp4" || ext == "mov" || ext == "m4v" {
                    args.extend(["-movflags", "+faststart"].map(String::from));
                }
            }
        }
        args.push(path.to_string());
        let mut child = Command::new("ffmpeg")
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .map_err(|e| format!("ffmpeg: {e} (install ffmpeg, or write a .y4m file, which needs no encoder)"))?;
        let stdin = BufWriter::with_capacity(1 << 22, child.stdin.take().unwrap());
        Ok(Sink::Ffmpeg { child, stdin })
    }

    /// One frame, RGB8, top row first.
    pub fn write(&mut self, rgb: &[u8]) -> std::io::Result<()> {
        match self {
            Sink::Raw(out) => out.write_all(rgb),
            Sink::Ffmpeg { stdin, .. } => stdin.write_all(rgb),
            Sink::Y4m { out, w, h, frame } => {
                to_yuv420(rgb, *w, *h, frame);
                out.write_all(b"FRAME\n")?;
                out.write_all(frame)
            }
        }
    }

    /// Flushes and, for ffmpeg, waits for the encoder.
    pub fn finish(self) -> Result<(), String> {
        match self {
            Sink::Raw(mut out) => out.flush().map_err(|e| e.to_string()),
            Sink::Y4m { mut out, .. } => out.flush().map_err(|e| e.to_string()),
            Sink::Ffmpeg { mut child, stdin } => {
                drop(stdin.into_inner().map_err(|e| e.to_string())?);
                let st = child.wait().map_err(|e| e.to_string())?;
                if st.success() { Ok(()) } else { Err(format!("ffmpeg failed ({st})")) }
            }
        }
    }
}

fn rational(fps: f64) -> (u64, u64) {
    let den = if (fps - fps.round()).abs() < 1e-6 { 1 } else { 1000 };
    ((fps * den as f64).round() as u64, den)
}

/// BT.601 studio-range Y'CbCr, 4:2:0 (2 x 2 chroma averages).
fn to_yuv420(rgb: &[u8], w: usize, h: usize, out: &mut Vec<u8>) {
    out.clear();
    out.resize(w * h + 2 * (w / 2) * (h / 2), 0);
    let (yp, rest) = out.split_at_mut(w * h);
    let (up, vp) = rest.split_at_mut((w / 2) * (h / 2));
    for (i, px) in rgb.chunks_exact(3).enumerate().take(w * h) {
        let (r, g, b) = (px[0] as f32, px[1] as f32, px[2] as f32);
        yp[i] = (16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0).round().clamp(16.0, 235.0) as u8;
    }
    for cy in 0..h / 2 {
        for cx in 0..w / 2 {
            let (mut r, mut g, mut b) = (0.0f32, 0.0f32, 0.0f32);
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let o = ((cy * 2 + dy) * w + cx * 2 + dx) * 3;
                r += rgb[o] as f32;
                g += rgb[o + 1] as f32;
                b += rgb[o + 2] as f32;
            }
            let (r, g, b) = (r / 4.0, g / 4.0, b / 4.0);
            let i = cy * (w / 2) + cx;
            up[i] = (128.0 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0).round().clamp(16.0, 240.0) as u8;
            vp[i] = (128.0 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0).round().clamp(16.0, 240.0) as u8;
        }
    }
}

/// The sheep that appear in a video, for its credits: (name, designer,
/// non-commercial).
#[derive(Default)]
pub struct Credits {
    pub sheep: Vec<(String, String, bool)>,
}

impl Credits {
    pub fn add(&mut self, name: &str, nick: &str, nc: bool) {
        if !self.sheep.iter().any(|(n, ..)| n == name) {
            self.sheep.push((name.to_string(), nick.to_string(), nc));
        }
    }

    fn es(&self) -> bool {
        self.sheep.iter().any(|(n, ..)| n.contains("electricsheep."))
    }

    /// One line per sheep, with its licence (Electric Sheep's reuse policy:
    /// evolved sheep CC BY-NC 3.0 US, designed ones CC BY 3.0 US).
    fn sheep_lines(&self) -> Vec<String> {
        self.sheep
            .iter()
            .map(|(n, nick, nc)| {
                let lic = if *nc { "CC BY-NC 3.0 US" } else { "CC BY 3.0 US" };
                if !n.contains("electricsheep.") {
                    format!("{n}: bred by fleece (flam3 algorithms)")
                } else if nick.is_empty() {
                    format!("{n}: {lic}")
                } else {
                    format!("{n}, designed by {nick}: {lic}")
                }
            })
            .collect()
    }

    fn comment(&self) -> String {
        let mut s = if self.es() {
            "Artwork by Scott Draves and the Electric Sheep (electricsheep.org), shared under Creative Commons licences. Sheep: ".to_string()
        } else {
            "Sheep: ".to_string()
        };
        s += &self.sheep_lines().join("; ");
        s += ". Rendered by fleece (GPL v3) with the flam3 algorithms by Scott Draves.";
        s
    }

    fn artist(&self) -> String {
        let mut designers: Vec<&str> = self.sheep.iter().map(|(_, k, _)| k.as_str()).filter(|k| !k.is_empty()).collect();
        designers.sort();
        designers.dedup();
        let base = if self.es() { "Scott Draves and the Electric Sheep" } else { "fleece (flam3 by Scott Draves)" };
        if designers.is_empty() { base.to_string() } else { format!("{base}; designs by {}", designers.join(", ")) }
    }

    /// Writes `<video>.credits.txt` and, for files ffmpeg can tag, puts the
    /// credits into the file's metadata (a stream-copy remux, no re-encode).
    pub fn embed(&self, path: &str) -> Result<(), String> {
        if path == "-" || self.sheep.is_empty() {
            return Ok(());
        }
        let mut txt = String::new();
        if self.es() {
            txt += "Artwork by Scott Draves and the Electric Sheep - electricsheep.org\n";
            txt += "Evolved sheep: CC BY-NC 3.0 US; sheep designed by people: CC BY 3.0 US (credited by name).\n";
            txt += "Commercial use needs a licence from Spotworks (info@spotworks.com).\n\n";
        }
        txt += "Sheep in this video, in order of appearance:\n";
        for l in self.sheep_lines() {
            txt += &format!("  {l}\n");
        }
        txt += "\nRendered by fleece (GPL v3), using the flam3 algorithms by Scott Draves.\n";
        std::fs::write(format!("{path}.credits.txt"), txt).map_err(|e| format!("{path}.credits.txt: {e}"))?;
        let ext = std::path::Path::new(path).extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        if ext == "y4m" || ext == "gif" {
            return Ok(());
        }
        let tmp = format!("{path}.tagging.{ext}");
        let mut args: Vec<String> = ["-loglevel", "error", "-y", "-i", path, "-map", "0", "-c", "copy"].map(String::from).to_vec();
        if matches!(ext.as_str(), "mp4" | "mov" | "m4v") {
            args.extend(["-movflags", "+faststart"].map(String::from));
        }
        let title = if self.es() { "Electric Sheep, rendered live by fleece" } else { "fleece" };
        for (k, v) in [
            ("title", title.to_string()),
            ("artist", self.artist()),
            ("comment", self.comment()),
            ("description", self.comment()),
            ("copyright", if self.es() { "Electric Sheep artwork under Creative Commons; see comment".to_string() } else { String::new() }),
        ] {
            if !v.is_empty() {
                args.extend(["-metadata".to_string(), format!("{k}={v}")]);
            }
        }
        args.push(tmp.clone());
        let st = Command::new("ffmpeg").args(&args).stdin(Stdio::null()).status().map_err(|e| format!("ffmpeg: {e}"))?;
        if !st.success() {
            let _ = std::fs::remove_file(&tmp);
            return Err("ffmpeg could not tag the file (the credits are in the .credits.txt)".into());
        }
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }
}

/// Recording the live show. Every frame goes to the file: the show steps
/// exactly one frame of time per frame while recording, and a busy encoder
/// makes the display wait rather than lose frames.
pub struct Recorder {
    tx: Option<std::sync::mpsc::SyncSender<Vec<u8>>>,
    thread: Option<std::thread::JoinHandle<Result<(), String>>>,
    pub path: String,
    pub fps: f64,
    pub frames: u64,
    pub credits: Credits,
    /// what the speakers play, captured from some frame on
    sound: Option<Sound>,
    sound_tried: bool,
    /// how far the picture runs behind the sound it follows (seconds): the
    /// sound goes that much later into the file, so the two line up again
    pub sound_delay: f64,
}

/// The speakers' sound (the default output's monitor: other programs' sound
/// and fleece's own music alike) going into a raw file while recording.
struct Sound {
    child: Child,
    path: String,
    /// the video frame it starts at, and when that was
    frame: u64,
    t0: std::time::Instant,
}

const SOUND_RATE: u32 = 48000;

impl Recorder {
    pub fn start(path: &str, w: u32, h: u32, fps: f64) -> Result<Recorder, String> {
        let mut sink = Sink::open(path, w, h, fps, 18, "veryfast", None)?;
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(16);
        let thread = std::thread::Builder::new()
            .name("recorder".into())
            .spawn(move || {
                for f in rx {
                    sink.write(&f).map_err(|e| e.to_string())?;
                }
                sink.finish()
            })
            .map_err(|e| e.to_string())?;
        Ok(Recorder {
            tx: Some(tx),
            thread: Some(thread),
            path: path.to_string(),
            fps,
            frames: 0,
            credits: Credits::default(),
            sound: None,
            sound_tried: false,
            sound_delay: 0.0,
        })
    }

    /// Starts recording the sound too, from the next frame on (once: an error
    /// is not retried). From then on the frames keep to the clock, so the
    /// picture stays with the sound.
    pub fn capture_sound(&mut self) -> Result<(), String> {
        if self.sound_tried {
            return Ok(());
        }
        self.sound_tried = true;
        let dir = std::env::var("HOME").map(|h| format!("{h}/.cache/fleece")).unwrap_or_else(|_| ".".into());
        std::fs::create_dir_all(&dir).map_err(|e| format!("{dir}: {e}"))?;
        let path = format!("{dir}/sound-{}-{}.raw", std::process::id(), self.frames);
        let file = File::create(&path).map_err(|e| format!("{path}: {e}"))?;
        let child = Command::new("parec")
            .args(["-d", "@DEFAULT_MONITOR@", "--raw", "--format=s16le", "--channels=2", "--latency-msec=20"])
            .arg(format!("--rate={SOUND_RATE}"))
            .stdin(Stdio::null())
            .stdout(file)
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("parec: {e}"))?;
        self.sound = Some(Sound { child, path, frame: self.frames, t0: std::time::Instant::now() });
        Ok(())
    }

    pub fn has_sound(&self) -> bool {
        self.sound.is_some()
    }

    /// How many times to write the next frame: once, or with sound as many
    /// times as keep the video with the clock (0 when ahead, more when the
    /// rendering fell behind).
    pub fn due(&self) -> u64 {
        match &self.sound {
            None => 1,
            Some(s) => (s.frame + (s.t0.elapsed().as_secs_f64() * self.fps).round() as u64 + 1).saturating_sub(self.frames),
        }
    }

    /// The next frame (waits while the encoder is busy; never drops).
    pub fn offer(&mut self, rgb: Vec<u8>) {
        if let Some(tx) = &self.tx {
            if tx.send(rgb).is_ok() {
                self.frames += 1;
            }
        }
    }

    /// Seconds of video so far.
    pub fn seconds(&self) -> f64 {
        self.frames as f64 / self.fps
    }

    /// Ends the file (waits for the encoder) and tags it with the credits.
    pub fn stop(mut self) -> Result<(), String> {
        drop(self.tx.take());
        let video = match self.thread.take().map(|t| t.join()) {
            Some(Ok(r)) => r,
            _ => Err("the recorder thread failed".into()),
        };
        if let Some(mut s) = self.sound.take() {
            let _ = s.child.kill();
            let _ = s.child.wait();
            let muxed = video.as_ref().map_err(|e| e.clone()).and_then(|_| self.add_sound(&s));
            let _ = std::fs::remove_file(&s.path);
            muxed?;
        }
        video?;
        self.credits.embed(&self.path)
    }

    /// Puts the captured sound into the finished video, starting where it began.
    fn add_sound(&self, s: &Sound) -> Result<(), String> {
        let ext = std::path::Path::new(&self.path).extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        let tmp = format!("{}.sound.{ext}", self.path);
        let offset = format!("{:.4}", s.frame as f64 / self.fps + self.sound_delay);
        let rate = SOUND_RATE.to_string();
        let mut args: Vec<&str> = vec!["-loglevel", "error", "-y", "-i", &self.path, "-itsoffset", &offset];
        args.extend(["-f", "s16le", "-ar", &rate, "-ac", "2", "-i", &s.path, "-map", "0:v", "-map", "1:a", "-c:v", "copy"]);
        args.extend(if ext == "webm" { ["-c:a", "libopus", "-b:a", "160k"] } else { ["-c:a", "aac", "-b:a", "192k"] });
        if matches!(ext.as_str(), "mp4" | "mov" | "m4v") {
            args.extend(["-movflags", "+faststart"]);
        }
        args.push(&tmp);
        let st = Command::new("ffmpeg").args(&args).stdin(Stdio::null()).status().map_err(|e| format!("ffmpeg: {e}"))?;
        if !st.success() {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!("ffmpeg could not add the sound to {}", self.path));
        }
        std::fs::rename(&tmp, &self.path).map_err(|e| format!("{}: {e}", self.path))
    }
}
