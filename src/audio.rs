//! What the speakers play, as levels the show can follow: the default sink's
//! monitor (PulseAudio, or PipeWire's PulseAudio server) read through `parec`,
//! analysed ~94 times a second into loudness, four bands and a beat impulse,
//! each brought to 0..1 by its own automatic gain.

use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The capture rate asked of parec.
const RATE: u32 = 48000;
/// Analysis frame and hop, in samples (2048 / 512: 43 ms frames, ~94 a second at 48 kHz).
const N: usize = 2048;
const HOP: usize = 512;
/// The bands, in Hz: bass, low-mid, mid, high.
const BANDS: [(f32, f32); 4] = [(20.0, 150.0), (150.0, 600.0), (600.0, 2500.0), (2500.0, 10000.0)];
/// Smoothing of the levels: fast attack, slower release (time constants, seconds).
const ATTACK: f32 = 0.010;
const RELEASE: f32 = 0.200;
/// Automatic gain: each level is divided by its running peak, which decays
/// with this time constant (seconds; 4.3 dB a second) so quiet music comes up too...
const PEAK_DECAY: f32 = 2.0;
/// ...but never by less than these (RMS, full scale 1: rms -50 dBFS, each band
/// -60 dBFS), so silence and hiss read ~0 rather than amplified noise.
const FLOOR: [f32; 5] = [3e-3, 1e-3, 1e-3, 1e-3, 1e-3];
/// Nor is a band divided by less than this share of the loudest band's peak
/// (-20 dB), so one far below the rest (a kick's click in the highs) stays low.
const REL: f32 = 0.1;
/// The RMS that counts as sound for `active` (-60 dBFS), and how long it stays active after.
const SOUND: f32 = 1e-3;
const HOLD: f32 = 1.0;
/// Onsets: spectral flux of log magnitudes, ln(1 + COMPRESS * amplitude) (so a
/// change counts by its ratio, and anything under ~-60 dBFS hardly at all),
/// mean per bin of each band, weighted mostly to the bass and low-mid, and
/// averaged over two hops (the bass's few bins make it noisy).
const COMPRESS: f32 = 1000.0;
const FLUX_WEIGHT: [f32; 4] = [1.0, 0.7, 0.2, 0.2];
/// An onset is flux above mean + K * std of the last second, above RATIO times
/// that mean, and above MIN_FLUX; at most one per REFRACTORY seconds.
const K: f32 = 2.0;
const RATIO: f32 = 2.0;
const MIN_FLUX: f32 = 0.1;
const REFRACTORY: f32 = 0.1;
/// The beat impulse's decay (time constant, seconds: ~0.05 left after 150 ms).
const BEAT_DECAY: f32 = 0.05;

/// The latest levels, each roughly 0..1.
#[derive(Clone, Copy, Debug, Default)]
pub struct Levels {
    /// overall loudness
    pub rms: f32,
    /// bass (20-150 Hz), low-mid (150-600), mid (600-2500), high (2500-10000)
    pub bands: [f32; 4],
    /// 1 on an onset (a kick, mostly), decaying over ~150 ms
    pub beat: f32,
    /// sound in the last second
    pub active: bool,
}

/// Audio analysed on a thread of its own; `levels()` reads the latest.
pub struct Audio {
    /// the latest levels, and whether any input has come yet
    shared: Arc<Mutex<(Levels, bool)>>,
    child: Option<Child>,
    thread: Option<JoinHandle<()>>,
}

impl Audio {
    /// Listens to what the speakers play: parec on the default sink's monitor.
    /// An error if parec is missing or exits at once (no sound server).
    pub fn start() -> Result<Audio, String> {
        let mut child = Command::new("parec")
            .args(["-d", "@DEFAULT_MONITOR@", "--raw", "--format=s16le", "--channels=1", "--latency-msec=20"])
            .arg(format!("--rate={RATE}"))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("parec: {e}"))?;
        let out = child.stdout.take().unwrap();
        let mut a = Audio::from_reader(out, RATE);
        a.child = Some(child);
        // wait for the first samples or parec's end; a server that is up but
        // sends nothing yet (a suspended sink) gets the benefit of the doubt
        let t0 = Instant::now();
        while !a.shared.lock().unwrap_or_else(|e| e.into_inner()).1 && t0.elapsed() < Duration::from_millis(500) {
            if let Ok(Some(st)) = a.child.as_mut().unwrap().try_wait() {
                return Err(format!("parec exited ({st}): no PulseAudio / PipeWire monitor to listen to"));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok(a)
    }

    /// Analyses s16le mono at `rate` from `r` until it ends (for tests; dropping
    /// the Audio waits for that end).
    pub fn from_reader(mut r: impl Read + Send + 'static, rate: u32) -> Audio {
        let shared = Arc::new(Mutex::new((Levels::default(), false)));
        let sh = shared.clone();
        let thread = std::thread::Builder::new()
            .name("audio".into())
            .spawn(move || {
                let mut an = Analyser::new(rate);
                let mut buf = [0u8; 4096];
                // a sample split between two reads
                let mut odd: Option<u8> = None;
                loop {
                    let n = match r.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => n,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(_) => break,
                    };
                    let mut bytes = &buf[..n];
                    let mut new = false;
                    if let Some(lo) = odd.take() {
                        new |= an.push(i16::from_le_bytes([lo, bytes[0]]) as f32 / 32768.0);
                        bytes = &bytes[1..];
                    }
                    let pairs = bytes.chunks_exact(2);
                    odd = pairs.remainder().first().copied();
                    for s in pairs {
                        new |= an.push(i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0);
                    }
                    let mut s = sh.lock().unwrap_or_else(|e| e.into_inner());
                    s.1 = true;
                    if new {
                        s.0 = an.levels;
                    }
                }
                // the input ended: no sound from here on
                sh.lock().unwrap_or_else(|e| e.into_inner()).0 = Levels::default();
            })
            .expect("audio thread");
        Audio { shared, child: None, thread: Some(thread) }
    }

    /// The latest levels: a copy under a lock the analysis holds only to write
    /// them, never while reading audio.
    pub fn levels(&self) -> Levels {
        self.shared.lock().unwrap_or_else(|e| e.into_inner()).0
    }
}

impl Drop for Audio {
    /// Stops parec (its output then ends, which ends the thread) and waits for both.
    fn drop(&mut self) {
        if let Some(c) = &mut self.child {
            let _ = c.kill();
            let _ = c.wait();
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Radix-2 FFT, in place, of one fixed power-of-two size.
struct Fft {
    rev: Vec<u32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Fft {
    fn new(n: usize) -> Fft {
        assert!(n.is_power_of_two() && n >= 2);
        let bits = n.trailing_zeros();
        let rev = (0..n as u32).map(|i| i.reverse_bits() >> (32 - bits)).collect();
        let a = |k: usize| -2.0 * std::f64::consts::PI * k as f64 / n as f64;
        let cos = (0..n / 2).map(|k| a(k).cos() as f32).collect();
        let sin = (0..n / 2).map(|k| a(k).sin() as f32).collect();
        Fft { rev, cos, sin }
    }

    /// The forward transform (e^-2πi kn/N) of `re` + i `im`.
    fn run(&self, re: &mut [f32], im: &mut [f32]) {
        let n = self.rev.len();
        for i in 0..n {
            let j = self.rev[i] as usize;
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let (half, step) = (len / 2, n / len);
            for s in (0..n).step_by(len) {
                for k in 0..half {
                    let (c, si) = (self.cos[k * step], self.sin[k * step]);
                    let (a, b) = (s + k, s + k + half);
                    let tr = re[b] * c - im[b] * si;
                    let ti = re[b] * si + im[b] * c;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len *= 2;
        }
    }
}

/// The analysis itself, one sample at a time: a frame every HOP samples.
struct Analyser {
    fft: Fft,
    window: Vec<f32>,
    /// the last N samples, oldest first; the newest HOP fill in at the end
    buf: Vec<f32>,
    fresh: usize,
    re: Vec<f32>,
    im: Vec<f32>,
    /// each band's bins, [lo, hi)
    bins: [(usize, usize); 4],
    /// |X|^2 to the band's share of the mean square, and |X| to a sine's amplitude
    pscale: f32,
    ascale: f32,
    /// the previous frame's log magnitudes, for the flux
    prev: Vec<f32>,
    /// the last frame's flux, and the last second's (two-hop averages), a ring
    last_flux: f32,
    hist: Vec<f32>,
    at: usize,
    /// the running peaks the gain divides by: rms, then the bands
    peak: [f32; 5],
    /// per-hop factors: attack, release, peak decay, beat decay
    k_att: f32,
    k_rel: f32,
    k_peak: f32,
    k_beat: f32,
    hop_secs: f32,
    since_onset: f32,
    since_sound: f32,
    levels: Levels,
    /// an onset in the last frame
    onset: bool,
}

impl Analyser {
    fn new(rate: u32) -> Analyser {
        let rate = rate.max(1) as f32;
        let window: Vec<f32> =
            (0..N).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / N as f32).cos()).collect();
        let (wsum, wsq) = (window.iter().sum::<f32>(), window.iter().map(|w| w * w).sum::<f32>());
        let bin = |f: f32| ((f * N as f32 / rate).ceil() as usize).clamp(1, N / 2);
        let bins = BANDS.map(|(lo, hi)| (bin(lo), bin(hi)));
        let hop_secs = HOP as f32 / rate;
        let per_hop = |tau: f32| (-hop_secs / tau).exp();
        Analyser {
            fft: Fft::new(N),
            window,
            buf: vec![0.0; N],
            fresh: 0,
            re: vec![0.0; N],
            im: vec![0.0; N],
            bins,
            pscale: 2.0 / (N as f32 * wsq),
            ascale: 2.0 / wsum,
            prev: vec![0.0; N / 2 + 1],
            last_flux: 0.0,
            hist: vec![0.0; (1.0 / hop_secs).round().max(8.0) as usize],
            at: 0,
            peak: FLOOR,
            k_att: 1.0 - per_hop(ATTACK),
            k_rel: 1.0 - per_hop(RELEASE),
            k_peak: per_hop(PEAK_DECAY),
            k_beat: per_hop(BEAT_DECAY),
            hop_secs,
            since_onset: REFRACTORY,
            since_sound: f32::INFINITY,
            levels: Levels::default(),
            onset: false,
        }
    }

    /// One sample (full scale ±1); true when it completed a hop and
    /// `levels` and `onset` are new.
    fn push(&mut self, x: f32) -> bool {
        self.buf[N - HOP + self.fresh] = x;
        self.fresh += 1;
        if self.fresh < HOP {
            return false;
        }
        self.frame();
        self.buf.copy_within(HOP.., 0);
        self.fresh = 0;
        true
    }

    fn frame(&mut self) {
        let rms = (self.buf.iter().map(|x| x * x).sum::<f32>() / N as f32).sqrt();
        for i in 0..N {
            self.re[i] = self.buf[i] * self.window[i];
            self.im[i] = 0.0;
        }
        self.fft.run(&mut self.re, &mut self.im);
        // each band's RMS, and the rise of its log magnitudes since the last frame
        let mut raw = [rms, 0.0, 0.0, 0.0, 0.0];
        let mut flux = 0.0;
        for (b, &(lo, hi)) in self.bins.iter().enumerate() {
            let (mut p, mut f) = (0.0, 0.0);
            for k in lo..hi {
                let pw = self.re[k] * self.re[k] + self.im[k] * self.im[k];
                p += pw;
                let l = (1.0 + COMPRESS * self.ascale * pw.sqrt()).ln();
                f += (l - self.prev[k]).max(0.0);
                self.prev[k] = l;
            }
            raw[b + 1] = (p * self.pscale).sqrt();
            flux += FLUX_WEIGHT[b] * f / (hi - lo).max(1) as f32;
        }
        // automatic gain, then fast attack and slow release
        for (i, (r, pk)) in raw.iter().zip(self.peak.iter_mut()).enumerate() {
            *pk = (*pk * self.k_peak).max(*r).max(FLOOR[i]);
        }
        let top = self.peak[1..].iter().fold(0.0f32, |a, &b| a.max(b));
        let l = &mut self.levels;
        for (i, (r, pk)) in raw.iter().zip(self.peak).enumerate() {
            let x = r / if i == 0 { pk } else { pk.max(REL * top) };
            let y = if i == 0 { &mut l.rms } else { &mut l.bands[i - 1] };
            *y += (x - *y) * if x > *y { self.k_att } else { self.k_rel };
        }
        // an onset: flux well above the last second's
        (flux, self.last_flux) = ((flux + self.last_flux) / 2.0, flux);
        let n = self.hist.len() as f32;
        let mean = self.hist.iter().sum::<f32>() / n;
        let std = (self.hist.iter().map(|f| (f - mean) * (f - mean)).sum::<f32>() / n).sqrt();
        self.since_onset += self.hop_secs;
        self.onset = flux > (mean + K * std).max(RATIO * mean).max(MIN_FLUX) && self.since_onset >= REFRACTORY;
        if self.onset {
            self.since_onset = 0.0;
            l.beat = 1.0;
        } else {
            l.beat *= self.k_beat;
        }
        self.hist[self.at] = flux;
        self.at = (self.at + 1) % self.hist.len();
        self.since_sound = if rms > SOUND { 0.0 } else { self.since_sound + self.hop_secs };
        l.active = self.since_sound < HOLD;
    }
}

/// A WAV file (`-`: on stdin) of 16-bit PCM, any rate and channel count:
/// its samples mixed to mono, its rate and its channel count.
fn read_wav(path: &str) -> Result<(Vec<f32>, u32, u16), String> {
    let data = if path == "-" {
        let mut v = Vec::new();
        std::io::stdin().read_to_end(&mut v).map(|_| v)
    } else {
        std::fs::read(path)
    }
    .map_err(|e| format!("{path}: {e}"))?;
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(format!("{path}: not a WAV file"));
    }
    let u16_at = |i: usize| u16::from_le_bytes([data[i], data[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
    // (format, channels, rate, bits)
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut at = 12;
    while at + 8 <= data.len() {
        let (id, size, body) = (&data[at..at + 4], u32_at(at + 4) as usize, at + 8);
        if id == b"fmt " && size >= 16 && body + 16 <= data.len() {
            let mut tag = u16_at(body);
            // WAVE_FORMAT_EXTENSIBLE: the real format leads the subformat GUID
            if tag == 0xFFFE && size >= 26 && body + 26 <= data.len() {
                tag = u16_at(body + 24);
            }
            fmt = Some((tag, u16_at(body + 2), u32_at(body + 4), u16_at(body + 14)));
        } else if id == b"data" {
            let (tag, ch, rate, bits) = fmt.ok_or_else(|| format!("{path}: no format before the data"))?;
            if tag != 1 || bits != 16 || ch == 0 || rate == 0 {
                return Err(format!("{path}: only 16-bit PCM is read here (format {tag}, {bits} bits, {ch} channels)"));
            }
            // a WAV streamed through a pipe gives no size: read to the end
            let end =
                if size == 0 || size >= u32::MAX as usize - 8 { data.len() } else { (body + size).min(data.len()) };
            let mono = data[body..end]
                .chunks_exact(2 * ch as usize)
                .map(|f| f.chunks_exact(2).map(|s| i16::from_le_bytes([s[0], s[1]]) as f32).sum::<f32>())
                .map(|s| s / (ch as f32 * 32768.0))
                .collect();
            return Ok((mono, rate, ch));
        }
        at = body + size + (size & 1);
    }
    Err(format!("{path}: no audio data"))
}

/// A meter of `v` (0..1), ten characters wide.
fn bar(v: f32) -> String {
    let n = (v.clamp(0.0, 1.0) * 10.0).round() as usize;
    "#".repeat(n) + &".".repeat(10 - n)
}

fn meters(t: f64, l: &Levels, beat: f32, onset: bool) -> String {
    let b = &l.bands;
    format!(
        "{t:7.2} s  rms {}  bass {}  lo-mid {}  mid {}  high {}  beat {}{}{}",
        bar(l.rms),
        bar(b[0]),
        bar(b[1]),
        bar(b[2]),
        bar(b[3]),
        bar(beat),
        if onset { "  ONSET" } else { "" },
        if l.active { "" } else { "  (silent)" }
    )
}

/// `--audiocheck`: meters for a WAV file (`-`: on stdin), analysed as fast as
/// it goes, a line per 50 ms of its time and a summary; or, with no file,
/// live from what the speakers play, ten lines a second for `tenths` / 10 seconds.
pub fn check(path: &str, tenths: u64) {
    let r = if path.is_empty() { live(tenths) } else { file(path) };
    if let Err(e) = r {
        eprintln!("fleece: {e}");
        std::process::exit(1);
    }
}

fn file(path: &str) -> Result<(), String> {
    let (samples, rate, ch) = read_wav(path)?;
    let mut out = std::io::stdout().lock();
    let mut an = Analyser::new(rate);
    let every = (rate as usize / 20).max(1);
    let (mut beat, mut onset) = (0.0f32, false);
    let mut onsets = Vec::new();
    let (mut sum, mut hops, mut active) = ([0.0f64; 5], 0u32, 0u32);
    for (i, &x) in samples.iter().enumerate() {
        let t = (i + 1) as f64 / rate as f64;
        if an.push(x) {
            let l = &an.levels;
            beat = beat.max(l.beat);
            if an.onset {
                onset = true;
                onsets.push(t);
            }
            for (s, v) in sum.iter_mut().zip([l.rms, l.bands[0], l.bands[1], l.bands[2], l.bands[3]]) {
                *s += v as f64;
            }
            hops += 1;
            active += l.active as u32;
        }
        if (i + 1) % every == 0 {
            if writeln!(out, "{}", meters(t, &an.levels, beat, onset)).is_err() {
                return Ok(());
            }
            (beat, onset) = (0.0, false);
        }
    }
    let secs = samples.len() as f64 / rate as f64;
    let _ = writeln!(out, "{path}: {secs:.2} s, {rate} Hz, {ch} channel{}", if ch == 1 { "" } else { "s" });
    let m: Vec<String> = sum.iter().map(|s| format!("{:.2}", s / hops.max(1) as f64)).collect();
    let _ = writeln!(
        out,
        "mean levels: rms {}  bass {}  lo-mid {}  mid {}  high {};  active {:.0} % of the time",
        m[0],
        m[1],
        m[2],
        m[3],
        m[4],
        100.0 * active as f64 / hops.max(1) as f64
    );
    let times: Vec<String> = onsets.iter().map(|t| format!("{t:.3}")).collect();
    let _ = writeln!(out, "onsets: {}{}{}", onsets.len(), if onsets.is_empty() { "" } else { " at " }, times.join(" "));
    if onsets.len() > 1 {
        let gaps: Vec<f64> = onsets.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let (lo, hi) = gaps.iter().fold((f64::MAX, 0.0f64), |(lo, hi), &g| (lo.min(g), hi.max(g)));
        let _ = writeln!(out, "intervals: mean {mean:.3} s ({:.1} BPM), min {lo:.3}, max {hi:.3}", 60.0 / mean);
    }
    Ok(())
}

fn live(tenths: u64) -> Result<(), String> {
    let a = Audio::start()?;
    let mut out = std::io::stdout().lock();
    let t0 = Instant::now();
    for n in 1..=tenths {
        let due = t0 + Duration::from_millis(100 * n);
        if let Some(d) = due.checked_duration_since(Instant::now()) {
            std::thread::sleep(d);
        }
        let l = a.levels();
        if writeln!(out, "{}", meters(t0.elapsed().as_secs_f64(), &l, l.beat, false)).is_err() {
            break;
        }
    }
    Ok(())
}
