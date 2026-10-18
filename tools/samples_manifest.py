#!/usr/bin/env python3
"""Builds fleece's multi-sample directory from the CC0 Versilian libraries.

Called by tools/get-samples.sh; Python standard library only, plus ffmpeg/ffprobe.

  samples_manifest.py repos                  print "name<TAB>url<TAB>commit" per source
  samples_manifest.py folders NAME           print the folders to check out from source NAME
  samples_manifest.py uptodate DEST          exit 0 if DEST's multi/ was built by this exact
                                             selection and its files are all there
  samples_manifest.py build SRC DEST         convert SRC/<source>/... into DEST/multi, and write
                                             DEST/manifest.tsv, sources.tsv, SOURCES.md, licenses/
  samples_manifest.py summary DEST           validate DEST/manifest.tsv and print a table

The source files are treated as data: they are only read by ffprobe/ffmpeg (as file: URLs)
and by a header check here; nothing in SRC is ever executed or imported.

Output format (the interface fleece's sampler reads):
  multi/<instrument>/<articulation>/<name>.wav: 48 kHz, 16-bit PCM, mono or stereo as recorded;
    leading silence trimmed so the onset (the first sample within 50 dB of the peak) is 1 ms
    from the start; the tail cut where it stays below -60 dB relative to the peak, with a 50 ms
    fade (a quarter of the sample for samples under 200 ms); sustained articulations capped at
    MAX_SECONDS after the onset, with a FADE_CAPPED fade.
  manifest.tsv: instrument articulation root vel_lo vel_hi rr gain_db file (tab-separated).
"""

import hashlib
import json
import math
import operator
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor

REPOS = {
    "VSCO-2-CE": ("https://github.com/sgossner/VSCO-2-CE", "440300901dfe9275fd84e0b7763af1f8443ae62e"),
    "VCSL": ("https://github.com/sgossner/VCSL", "c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e"),
}

RATE = 48000
ONSET_DB = -50.0     # the onset: the first sample within 50 dB of the peak
PREROLL = 48         # frames kept before the onset (1 ms), faded in
TAIL_DB = -60.0      # the tail is cut where every 10 ms block stays below this (re the peak)
FADE = 0.050         # the tail fade
# caps, in seconds after the onset, per articulation (pluck: the harp), for the disk budget
MAX_SECONDS = {"sus": 5.5, "vib": 5.5, "pluck": 4.5}
FADE_CAPPED = 0.5    # the fade of a capped sample
TARGET_DB = -20.0    # the loudest layer's median RMS (first 1 s after the onset), in dBFS

# --------------------------------------------------------------------------------------------
# Root corrections. Every pitched sample's root was verified by pitch estimation on a steady
# 200 ms after the attack (YIN, and a harmonic-spectrum test: which candidate's harmonics, odd
# ones included, explain the spectrum), run with numpy outside this repository on 2026-10-18,
# and cross-checked against the pitch_keycenter of the libraries' own SFZ maps where they exist
# (VSCO-2-CE branch SFZ, commit 6dd651d).
#
# Most folders name their notes one octave low (the "C3 = middle C" convention): their files
# sound an octave above their names, so +12 for the whole folder. The others are named in
# scientific pitch (C4 = middle C = MIDI 60) and need nothing.
ROOT_FIX_FOLDER = {
    ("VSCO-2-CE", "Strings/Violin Section/susVib"): 12,
    ("VSCO-2-CE", "Strings/Violin Section/Spic"): 12,
    ("VSCO-2-CE", "Strings/Violin Section/Pizz"): 12,
    ("VSCO-2-CE", "Strings/Solo Violin/Arco Vib"): 0,
    ("VSCO-2-CE", "Strings/Solo Violin/spic"): 12,
    ("VSCO-2-CE", "Strings/Solo Violin/Pizz"): 0,
    ("VSCO-2-CE", "Brass/Trumpet/sus"): 12,
    ("VSCO-2-CE", "Brass/Trumpet/susvib"): 12,
    ("VSCO-2-CE", "Brass/Trumpet/stac"): 12,
    ("VSCO-2-CE", "Brass/Tenor Trombone/sus"): 12,
    ("VSCO-2-CE", "Brass/Tenor Trombone/vib"): 12,
    ("VSCO-2-CE", "Brass/Tenor Trombone/stac"): 12,
    ("VSCO-2-CE", "Strings/Solo Contrabass/Pizz"): 12,
    ("VCSL", "Chordophones/Composite Chordophones/Concert Harp"): 0,
    ("VCSL", "Aerophones/Reed Aerophones/Tenor Saxophone/Non-Vibrato"): 12,
    ("VCSL", "Aerophones/Reed Aerophones/Tenor Saxophone/Vibrato"): 12,
    ("VCSL", "Aerophones/Reed Aerophones/Tenor Saxophone/Staccato"): 12,
}
# Per-file corrections on top of the folder's: (source, folder, file) -> the sounding MIDI note.
# None were needed. These files' estimates disagreed and were checked by hand from their
# spectra; each sounds as its folder's rule says (a weak fundamental, a strong resonance or an
# attack the estimator misread):
#   Violin Section/Pizz  A2_v2_rr1, B4_v1_rr2, D5_v1_rr1, D5_v1_rr2, G2_v2_rr1, G2_v2_rr2;
#   Violin Section/Spic  A2_v1_rr1;  Solo Violin/Arco Vib G3_p;  Solo Violin/spic C4_v1_rr2;
#   Solo Violin/Pizz C5_f_RR1, C5_f_RR2, C5_p_RR2;  Solo Contrabass/Pizz C#2_v1_rr1, C#2_v1_rr2,
#   E1_v1_rr1, G#1_v1_rr1, G#1_v1_rr2, G0_v1_rr1, G0_v1_rr2;  Concert Harp F7_f1 (30 cents flat);
#   Tenor Saxophone/Non-Vibrato C3_vl2.
ROOT_FIX_FILE = {}
# Left out: Pizz C7_f has no clear pitch (its two strongest partials near C7 are 57 cents
# apart); Pizz A3_p and A4_p are 12-21 dB below their neighbours, with noise 40-52 dB under
# their peaks (G3/C4 and G4/C5 cover them).
EXCLUDE = {("VSCO-2-CE", "Strings/Solo Violin/Pizz", f) for f in (
    "LLVln_Pizz_C7_f.wav", "LLVln_Pizz_A3_p_RR1.wav", "LLVln_Pizz_A3_p_RR2.wav",
    "LLVln_Pizz_A4_p_RR1.wav", "LLVln_Pizz_A4_p_RR2.wav")}

# --------------------------------------------------------------------------------------------
# The selection. A pitched set: (instrument, articulation, source, folder, layers, keep) where
# layers lists (name, regex on the file name) from the softest to the loudest, and keep (a
# regex, or None for all) picks the files. The note is read from the file name.
NOTE_RE = re.compile(r"(?<![A-Za-z])([A-G])(#|b)?(-?\d)(?=_|\.|$)")

PITCHED = [
    ("violins", "sus", "VSCO-2-CE", "Strings/Violin Section/susVib", [("v1", r"_v1\."), ("v2", r"_v2\.")], None),
    ("violins", "stac", "VSCO-2-CE", "Strings/Violin Section/Spic", [("v1", r"_v1_"), ("v2", r"_v2_")], None),
    ("violins", "pizz", "VSCO-2-CE", "Strings/Violin Section/Pizz", [("v1", r"_v1_"), ("v2", r"_v2_")], None),
    ("violin", "sus", "VSCO-2-CE", "Strings/Solo Violin/Arco Vib", [("p", r"_p\."), ("f", r"_f\.")], None),
    ("violin", "stac", "VSCO-2-CE", "Strings/Solo Violin/spic", [("v1", r"_v1_"), ("v2", r"_v2_")], None),
    ("violin", "pizz", "VSCO-2-CE", "Strings/Solo Violin/Pizz", [("p", r"_p_RR"), ("f", r"_f_RR")], None),
    ("trumpet", "sus", "VSCO-2-CE", "Brass/Trumpet/sus", [("v1", r"_v1_"), ("v3", r"_v3_")], None),
    ("trumpet", "vib", "VSCO-2-CE", "Brass/Trumpet/susvib", [("v1", r"_v1_"), ("v2", r"_v2_")], None),
    ("trumpet", "stac", "VSCO-2-CE", "Brass/Trumpet/stac", [("v1", r"_v1_"), ("v2", r"_v2_"), ("v3", r"_v3_")], None),
    # v1 (pp) left out: 10 dB below v2, with noise 50 dB under its peak
    ("trombone", "sus", "VSCO-2-CE", "Brass/Tenor Trombone/sus", [("v2", r"_v2_"), ("v3", r"_v3_")], None),
    ("trombone", "vib", "VSCO-2-CE", "Brass/Tenor Trombone/vib", [("v1", r"_v1_")], None),
    ("trombone", "stac", "VSCO-2-CE", "Brass/Tenor Trombone/stac",
     [("v1", r"_v1_"), ("v2", r"_v2_"), ("v3", r"_v3_"), ("v4", r"_v4_")], None),
    # v3 exists only for the six lowest notes, so one layer
    ("contrabass", "pizz", "VSCO-2-CE", "Strings/Solo Contrabass/Pizz", [("v1", r"_v1_")], None),
    # the soft layer: mf (G1 has only mp); D7 and F7's p samples (17 dB below their neighbours)
    # are left out, so B6 mf covers the top
    ("harp", "pluck", "VCSL", "Chordophones/Composite Chordophones/Concert Harp",
     [("mf", r"_(mf\d|mp\d)\."), ("f", r"_f\d\.")], None),
    # whole-tone sampled: every other note (4 semitones apart), G#2..E6 sounding
    ("tenor_sax", "sus", "VCSL", "Aerophones/Reed Aerophones/Tenor Saxophone/Non-Vibrato",
     [("vl2", r"_vl2_"), ("vl3", r"_vl3_")], r"_Main_(G#|C|E)\d_"),
    # one layer; 12 of its 19 notes, 2-4 semitones apart
    ("tenor_sax", "vib", "VCSL", "Aerophones/Reed Aerophones/Tenor Saxophone/Vibrato",
     [("var", r"_var\d\.")], r"_Main_(A#1|D2|F#2|A#2|D3|F#3|A#3|D4|E4|G4|A#4|D5)_"),
    ("tenor_sax", "stac", "VCSL", "Aerophones/Reed Aerophones/Tenor Saxophone/Staccato",
     [("vl1", r"_vl1_"), ("vl2", r"_vl2_")], None),
]

# Unpitched sets (root 60): (instrument, articulation, source, folder, [layer, ...]) where each
# layer, softest first, is a list of files (its round robins).
G = "Idiophones/Struck Idiophones/"
UNPITCHED = [
    ("guiro", "short", "VCSL", G + "Guiro/Legacy",
     [["guiro_short.wav", "guiro_short2.wav", "guiro_short3.wav", "guiro_short4.wav", "guiro_short5.wav"]]),
    ("guiro", "long", "VCSL", G + "Guiro/Legacy", [["guiro_long.wav", "guiro_long2.wav", "gurio_long4.wav"]]),
    ("guiro", "hit", "VCSL", G + "Guiro", [["Guiro_Hit_rr1_Mid.wav", "Guiro_Hit_rr2_Mid.wav"]]),
    ("guiro", "fast", "VCSL", G + "Guiro", [["Guiro_Fast_rr1_Mid.wav"]]),
    ("guiro", "medium", "VCSL", G + "Guiro", [["Guiro_Med_rr1_Mid.wav"]]),
    ("guiro", "slow", "VCSL", G + "Guiro", [["Guiro_Slow_rr2_Mid.wav"]]),
    # small and large: the single strokes, down and up alternating in round-robin order (so an
    # engine that cycles its round robins alternates); the other shaker articulations split them
    ("shaker", "small", "VCSL", G + "Shaker, Small",
     [["Mid_ShakerHighFaster_Down_rr1.wav", "Mid_ShakerHighFaster_Up_rr1.wav",
       "Mid_ShakerLowFaster_Down_rr1.wav", "Mid_ShakerLowFaster_Up_rr1.wav",
       "Mid_ShakerHighFaster_Down_rr2.wav", "Mid_ShakerHighFaster_Up_rr2.wav",
       "Mid_ShakerLowFaster_Down_rr2.wav", "Mid_ShakerLowFaster_Up_rr2.wav"]]),
    ("shaker", "large", "VCSL", G + "Shaker, Large",
     [["LShaker_Shake1D_rr2_Mid.wav", "LShaker_Shake1U_rr1_Mid.wav",
       "LShaker_Shake1D_rr3_Mid.wav", "LShaker_Shake1U_rr4_Mid.wav"]]),
    ("shaker", "small_down", "VCSL", G + "Shaker, Small",
     [["Mid_ShakerHighFaster_Down_rr1.wav", "Mid_ShakerHighFaster_Down_rr2.wav",
       "Mid_ShakerLowFaster_Down_rr1.wav", "Mid_ShakerLowFaster_Down_rr2.wav"]]),
    ("shaker", "small_up", "VCSL", G + "Shaker, Small",
     [["Mid_ShakerHighFaster_Up_rr1.wav", "Mid_ShakerHighFaster_Up_rr2.wav",
       "Mid_ShakerLowFaster_Up_rr1.wav", "Mid_ShakerLowFaster_Up_rr2.wav"]]),
    ("shaker", "small_double_down", "VCSL", G + "Shaker, Small",
     [["Mid_ShakerDouble_Down_rr1.wav", "Mid_ShakerDouble_Down_rr2.wav"]]),
    ("shaker", "small_double_up", "VCSL", G + "Shaker, Small",
     [["Mid_ShakerDouble_Up_rr1.wav", "Mid_ShakerDouble_Up_rr2.wav"]]),
    ("shaker", "small_roll", "VCSL", G + "Shaker, Small",
     [["Mid_Shaker_Roll_Fast_rr1.wav", "Mid_Shaker_Roll_Fast_rr2.wav"]]),
    ("shaker", "small_slap", "VCSL", G + "Shaker, Small", [["Mid_Shaker_Slap_rr1.wav", "Mid_Shaker_Slap_rr2.wav"]]),
    ("shaker", "large_hit", "VCSL", G + "Shaker, Large", [["LShaker_Hit_rr1_Mid.wav", "LShaker_Hit_rr2_Mid.wav"]]),
    ("shaker", "large_down", "VCSL", G + "Shaker, Large",
     [["LShaker_Shake1D_rr2_Mid.wav", "LShaker_Shake1D_rr3_Mid.wav"]]),
    ("shaker", "large_up", "VCSL", G + "Shaker, Large",
     [["LShaker_Shake1U_rr1_Mid.wav", "LShaker_Shake1U_rr4_Mid.wav"]]),
    ("claps", "group", "VCSL", G + "Claps", [["Clap_rr%d.wav" % i for i in range(1, 7)]]),
    ("claps", "solo", "VCSL", G + "Claps", [["SoloClap_vl%d.wav" % i] for i in range(1, 5)]),
    ("cajon", "hit1", "VCSL", G + "Cajon",
     [["Cajon_hit1_%s_rr1.wav" % d, "Cajon_hit1_%s_rr2.wav" % d] for d in ("mp", "f", "fff")]),
    ("cajon", "hit2", "VCSL", G + "Cajon",
     [["Cajon_hit2_%s_rr1.wav" % d, "Cajon_hit2_%s_rr2.wav" % d] for d in ("pp", "mp", "f")]),
    ("cajon", "hit3", "VCSL", G + "Cajon",
     [["Cajon_hit3_%s_rr1.wav" % d, "Cajon_hit3_%s_rr2.wav" % d] for d in ("p", "mp", "f")]),
]

UNPITCHED_NAMES = {u[0] for u in UNPITCHED}
LICENSE_FILES = {"VSCO-2-CE": ["LICENSE", "README.md", "Readme.txt"], "VCSL": ["LICENSE", "README.md"]}
MUSESCORE_DOCS = ["MuseScore_General_License.md", "MuseScore_General_Readme.md"]

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(m):
    return "%s%d" % (NAMES[m % 12], m // 12 - 1)


def log(*a):
    print(*a, flush=True)


def die(msg):
    print("samples_manifest: " + msg, file=sys.stderr)
    sys.exit(1)


def stamp():
    """Identifies this selection and processing: the hash of this file."""
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def vel_ranges(n):
    return [(round(128 * i / n), round(128 * (i + 1) / n) - 1) for i in range(n)]


# --------------------------------------------------------------------------------------------
def plan(src):
    """Every output file: dicts with instrument, articulation, root, layer, nlayers, rr, src."""
    out = []
    for inst, art, repo, folder, layers, keep in PITCHED:
        d = os.path.join(src, repo, folder)
        if not os.path.isdir(d):
            die("missing source folder %s" % d)
        fix = ROOT_FIX_FOLDER[(repo, folder)]
        groups = {}
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(".wav") or (keep and not re.search(keep, fn)):
                continue
            if (repo, folder, fn) in EXCLUDE:
                continue
            li = [i for i, (_, rx) in enumerate(layers) if re.search(rx, fn)]
            if len(li) != 1:
                continue
            m = NOTE_RE.search(fn)
            if not m:
                die("no note in %s/%s" % (folder, fn))
            named = (int(m.group(3)) + 1) * 12 + "C D EF G A B".index(m.group(1)) \
                + {"#": 1, "b": -1}.get(m.group(2), 0)
            root = ROOT_FIX_FILE.get((repo, folder, fn), named + fix)
            groups.setdefault((li[0], root), []).append((fn, named))
        if not groups:
            die("nothing selected in %s" % folder)
        for (li, root), files in sorted(groups.items()):
            for rr, (fn, named) in enumerate(files, 1):
                out.append(dict(instrument=inst, articulation=art, root=root, layer=li, nlayers=len(layers),
                                layer_name=layers[li][0], rr=rr, repo=repo, folder=folder, src=fn,
                                named=note_name(named), pitched=True))
    for inst, art, repo, folder, layers in UNPITCHED:
        for li, files in enumerate(layers):
            for rr, fn in enumerate(files, 1):
                out.append(dict(instrument=inst, articulation=art, root=60, layer=li, nlayers=len(layers),
                                layer_name="v%d" % (li + 1), rr=rr, repo=repo, folder=folder, src=fn,
                                named="", pitched=False))
    for o in out:
        o["vel_lo"], o["vel_hi"] = vel_ranges(o["nlayers"])[o["layer"]]
        base = ("%03d_v%d_rr%d.wav" % (o["root"], o["layer"] + 1, o["rr"])) if o["pitched"] \
            else ("v%d_rr%d.wav" % (o["layer"] + 1, o["rr"]))
        o["file"] = "multi/%s/%s/%s" % (o["instrument"], o["articulation"], base)
    return out


def folders(name):
    """The folders to check out from source name (in cone mode a folder brings its subfolders)."""
    fs = {f for _, _, r, f, _, _ in PITCHED if r == name} | {f for _, _, r, f, _ in UNPITCHED if r == name}
    return sorted(f for f in fs if not any(f.startswith(g + "/") for g in fs))


# --------------------------------------------------------------------------------------------
def check_wav_header(path):
    with open(path, "rb") as f:
        h = f.read(64)
    if h.startswith(b"version https://git-lfs"):
        die("%s is a Git LFS pointer, not audio. Install git-lfs and run 'git lfs pull' in its clone." % path)
    if h[:4] != b"RIFF" or h[8:12] != b"WAVE":
        die("%s is not a RIFF/WAVE file" % path)


RESAMPLER = None


def resampler():
    global RESAMPLER
    if RESAMPLER is None:
        t = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                            "-t", "0.01", "-af", "aresample=48000:resampler=soxr", "-f", "null", "-"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        RESAMPLER = "resampler=soxr:precision=28:" if t.returncode == 0 else "filter_size=64:"
    return RESAMPLER


def decode(path):
    """The file's audio as 48 kHz 16-bit interleaved samples: (array('h'), channels)."""
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
                        "-of", "csv=p=0", "file:" + path], capture_output=True, text=True)
    if p.returncode != 0 or not p.stdout.strip():
        die("ffprobe failed on %s: %s" % (path, p.stderr.strip()))
    ch = int(p.stdout.strip().split(",")[0])
    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", "file:" + path, "-map", "0:a:0",
                        "-af", "aresample=%d:%sdither_method=triangular" % (RATE, resampler()),
                        "-ac", str(ch), "-ar", str(RATE), "-f", "s16le", "-c:a", "pcm_s16le", "-"],
                       capture_output=True)
    if p.returncode != 0:
        die("ffmpeg failed on %s: %s" % (path, p.stderr.decode(errors="replace").strip()))
    a = array("h")
    a.frombytes(p.stdout[: len(p.stdout) // (2 * ch) * 2 * ch])
    if sys.byteorder == "big":
        a.byteswap()
    return a, ch


def peak_of(a):
    return max(max(a), -min(a)) if len(a) else 0


def onset_frame(a, ch, thr):
    """The first frame with a sample at or above thr."""
    blk = 48 * ch
    for i in range(0, len(a), blk):
        b = a[i:i + blk]
        if max(b) >= thr or -min(b) >= thr:
            for j in range(i, min(i + blk, len(a))):
                if abs(a[j]) >= thr:
                    return j // ch
    return 0


def tail_frame(a, ch, thr):
    """The end of the last 10 ms block with a sample at or above thr."""
    frames = len(a) // ch
    blk = 480
    for k in range((frames + blk - 1) // blk - 1, -1, -1):
        b = a[k * blk * ch:(k + 1) * blk * ch]
        if max(b) >= thr or -min(b) >= thr:
            return min(frames, (k + 1) * blk)
    return frames


def sumsq(a):
    try:
        return math.sumprod(a, a)
    except AttributeError:
        return sum(map(operator.mul, a, a))


def convert(job, src, stage):
    path = os.path.join(src, job["repo"], job["folder"], job["src"])
    check_wav_header(path)
    a, ch = decode(path)
    frames = len(a) // ch
    pk = peak_of(a)
    if frames < 480 or pk < 64:
        die("%s is empty or silent" % path)
    on = onset_frame(a, ch, pk * 10 ** (ONSET_DB / 20))
    start = max(0, on - PREROLL)
    end = tail_frame(a, ch, pk * 10 ** (TAIL_DB / 20))
    end = max(end, min(frames, on + 480))
    cap = MAX_SECONDS.get(job["articulation"])
    capped = cap is not None and end - on > cap * RATE
    if capped:
        end = on + int(cap * RATE)
        fade = int(FADE_CAPPED * RATE)
    else:
        fade = min(int(FADE * RATE), (end - start) // 4)
    out = a[start * ch:end * ch]
    n = end - start
    # fade the preroll in (only when there was something before it to cut)
    if start > 0:
        k = on - start
        for f in range(k):
            g = f / k
            for c in range(ch):
                out[f * ch + c] = int(round(out[f * ch + c] * g))
    # fade out: a raised cosine reaching zero at the last frame
    for f in range(fade):
        g = 0.5 * (1 + math.cos(math.pi * (f + 1) / fade))
        i = (n - fade + f) * ch
        for c in range(ch):
            out[i + c] = int(round(out[i + c] * g))
    first_s = out[(on - start) * ch:(on - start + RATE) * ch]
    rms = math.sqrt(sumsq(first_s) / len(first_s)) if len(first_s) else 0.0
    dst = os.path.join(stage, job["file"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if sys.byteorder == "big":
        out.byteswap()
    with wave.open(dst + ".part", "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(out.tobytes())
    os.replace(dst + ".part", dst)
    job.update(channels=ch, seconds=n / RATE, onset_ms=1000.0 * (on - start) / RATE,
               rms_db=20 * math.log10(rms / 32768) if rms > 0 else -200.0,
               peak_db=20 * math.log10(pk / 32768), capped=capped, cut_s=(frames - end) / RATE,
               lead_s=start / RATE, src_seconds=frames / RATE)
    return job


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


# --------------------------------------------------------------------------------------------
def build(src, dest):
    t0 = time.time()
    jobs = plan(src)
    log("converting %d samples from %s (48 kHz, 16-bit; resampler: %s)"
        % (len(jobs), src, "soxr" if "soxr" in resampler() else "swr"))
    stage = os.path.join(dest, ".multi.new")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    done = [0]

    def run(j):
        r = convert(j, src, stage)
        done[0] += 1
        if done[0] % 100 == 0:
            log("  %d/%d" % (done[0], len(jobs)))
        return r

    with ThreadPoolExecutor(max_workers=max(1, min(12, os.cpu_count() or 1))) as ex:
        jobs = list(ex.map(run, jobs))
    # gain per instrument/articulation: the loudest layer's median RMS to TARGET_DB
    gains = {}
    for key in sorted({(j["instrument"], j["articulation"]) for j in jobs}):
        top = [j for j in jobs if (j["instrument"], j["articulation"]) == key and j["layer"] == j["nlayers"] - 1]
        gains[key] = round(TARGET_DB - median([j["rms_db"] for j in top]), 1)
    for j in jobs:
        j["gain_db"] = gains[(j["instrument"], j["articulation"])]
    jobs.sort(key=lambda j: (j["instrument"], j["articulation"], j["vel_lo"], j["root"], j["rr"]))
    with open(os.path.join(stage, "manifest.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# fleece multi-samples: generated by tools/get-samples.sh (samples_manifest.py %s)\n" % stamp())
        f.write("# root: MIDI note (verified by pitch estimation; 60 for unpitched); vel_lo..vel_hi tile 0..127 per\n"
                "# instrument/articulation; rr: round robin within root and layer; gain_db: one per\n"
                "# instrument/articulation, putting the loudest layer's median RMS (first 1 s) at %g dBFS\n" % TARGET_DB)
        f.write("instrument\tarticulation\troot\tvel_lo\tvel_hi\trr\tgain_db\tfile\n")
        for j in jobs:
            f.write("%s\t%s\t%d\t%d\t%d\t%d\t%.1f\t%s\n" % (j["instrument"], j["articulation"], j["root"],
                                                         j["vel_lo"], j["vel_hi"], j["rr"], j["gain_db"], j["file"]))
    with open(os.path.join(stage, "sources.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("file\tsource\tsource_path\tnamed_note\troot\troot_name\tlayer\tchannels\tseconds\tcapped\t"
                "rms_db_first_1s\tpeak_db\n")
        for j in jobs:
            f.write("%s\t%s\t%s/%s\t%s\t%d\t%s\t%s\t%d\t%.3f\t%s\t%.1f\t%.1f\n" % (
                j["file"], j["repo"], j["folder"], j["src"], j["named"], j["root"],
                note_name(j["root"]) if j["pitched"] else "", j["layer_name"], j["channels"], j["seconds"],
                "yes" if j["capped"] else "no", j["rms_db"], j["peak_db"]))
    # licenses, copied verbatim
    lic = os.path.join(stage, "licenses")
    for repo, files in LICENSE_FILES.items():
        os.makedirs(os.path.join(lic, repo), exist_ok=True)
        for fn in files:
            shutil.copyfile(os.path.join(src, repo, fn), os.path.join(lic, repo, fn))
    os.makedirs(os.path.join(lic, "MuseScore_General"), exist_ok=True)
    for fn in MUSESCORE_DOCS:
        shutil.copyfile(os.path.join(dest, fn), os.path.join(lic, "MuseScore_General", fn))
    with open(os.path.join(stage, "SOURCES.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(sources_md(jobs, gains, dest))
    # swap the new tree in
    for name in ("multi", "licenses"):
        old = os.path.join(dest, "." + name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        if os.path.exists(os.path.join(dest, name)):
            os.rename(os.path.join(dest, name), old)
        os.rename(os.path.join(stage, name), os.path.join(dest, name))
        shutil.rmtree(old, ignore_errors=True)
    for fn in ("manifest.tsv", "sources.tsv", "SOURCES.md"):
        os.replace(os.path.join(stage, fn), os.path.join(dest, fn))
    shutil.rmtree(stage, ignore_errors=True)
    files = {j["file"]: os.path.getsize(os.path.join(dest, j["file"])) for j in jobs}
    with open(os.path.join(dest, ".multi-stamp"), "w") as f:
        json.dump({"stamp": stamp(), "files": files}, f)
    log("converted in %.0f s" % (time.time() - t0))


def uptodate(dest):
    try:
        with open(os.path.join(dest, ".multi-stamp")) as f:
            st = json.load(f)
    except (OSError, ValueError):
        return False
    if st.get("stamp") != stamp():
        return False
    for fn in ("manifest.tsv", "sources.tsv", "SOURCES.md"):
        if not os.path.isfile(os.path.join(dest, fn)):
            return False
    for repo, files in list(LICENSE_FILES.items()) + [("MuseScore_General", MUSESCORE_DOCS)]:
        for fn in files:
            if not os.path.isfile(os.path.join(dest, "licenses", repo, fn)):
                return False
    for fn, size in st.get("files", {}).items():
        p = os.path.join(dest, fn)
        if not os.path.isfile(p) or os.path.getsize(p) != size:
            return False
    return bool(st.get("files"))


# --------------------------------------------------------------------------------------------
def sources_md(jobs, gains, dest):
    by = {}
    for j in jobs:
        by.setdefault((j["instrument"], j["articulation"]), []).append(j)
    L = []
    w = L.append
    w("# fleece samples: sources\n")
    w("Generated by `tools/get-samples.sh` (fleece), selection `%s`, on %s.\n"
      % (stamp(), time.strftime("%Y-%m-%d")))
    w("`manifest.tsv` maps every file to its instrument, articulation, MIDI root, velocity range, round robin and"
      " gain; `sources.tsv` gives each file's source file and levels; `licenses/` holds each source's license and"
      " readme, copied verbatim.\n")
    w("## MuseScore General 0.2 (General MIDI SoundFont)\n")
    w("- URL: https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/ (files dated 2020-07-10)")
    w("- License: MIT (see `MuseScore_General_License.md`): FluidR3 by Frank Wen, FluidR3Mono by Michael Cowgill,"
      " adaptation by S. Christian Collins, Temple Blocks by Ethan Winer, Drumline Cymbals by Michael Schorsch."
      " Its acknowledgements and copyright notices must be included in any derivative work.")
    w("- Taken: `MuseScore_General.sf2` whole, with `MuseScore_General_License.md`, `MuseScore_General_Readme.md`"
      " and `MuseScore_General_Sample_Sources.csv`, unchanged.\n")
    for repo, title, lic in (
            ("VSCO-2-CE", "VS Chamber Orchestra 2: Community Edition (Versilian Studios)",
             "CC0 1.0 Universal (`LICENSE`). Its `Readme.txt` asks for credit to Versilian Studios / Sam Gossner"
             " and Ivy Audio / Simon Dalzell where applicable, and a link to the VSCO: CE homepage, and that the"
             " samples not be sold on their own; recorded by Sam Gossner and Simon Dalzell, cut by Elan Hickler"
             " / Soundemote."),
            ("VCSL", "Versilian Community Sample Library (Versilian Studios)", "CC0 1.0 Universal (`LICENSE`).")):
        url, commit = REPOS[repo]
        w("## %s\n" % title)
        w("- URL: %s" % url)
        w("- Commit: `%s`" % commit)
        w("- License: %s" % lic)
        w("- Taken (folder: instrument/articulation, files):")
        seen = {}
        for j in jobs:
            if j["repo"] == repo:
                k = (j["folder"], j["instrument"], j["articulation"])
                seen[k] = seen.get(k, 0) + 1
        for (folder, inst, art), n in sorted(seen.items()):
            w("  - `%s`: %s/%s, %d file%s" % (folder, inst, art, n, "" if n == 1 else "s"))
        w("")
    w("## Processing\n")
    w("- Decoded and resampled to 48 kHz with ffmpeg (soxr where available), 16-bit with triangular dither;"
      " channels as recorded (all of these are stereo).")
    w("- Leading silence trimmed: the onset is the first sample within %g dB of the file's peak, and the file"
      " starts 1 ms before it (that 1 ms faded in)." % -ONSET_DB)
    w("- Tail cut at the end of the last 10 ms that reaches %g dB relative to the peak, with a 50 ms raised-cosine"
      " fade (a quarter of the sample for samples under 200 ms). To fit the disk budget, samples of the"
      " articulations %s are capped at that many seconds after the onset, the last %g s faded out."
      % (TAIL_DB, ", ".join("%s %g s" % kv for kv in sorted(MAX_SECONDS.items())), FADE_CAPPED))
    w("- Roots: the note in each file name, corrected per folder (most VSCO-2-CE and VCSL folders name notes one"
      " octave low, the C3 = middle C convention) and verified by pitch estimation of every sample;"
      " see `ROOT_FIX_FOLDER` in `tools/samples_manifest.py`. Unpitched percussion uses root 60.")
    w("- Velocity layers split 0-127 evenly, softest first; `gain_db` is one value per instrument/articulation,"
      " putting the loudest layer's median RMS over the first 1 s after the onset at %g dBFS, so the layers keep"
      " their recorded loudness relationship." % TARGET_DB)
    w("- Octave fixes applied (folder: semitones added to the named note):")
    for (repo, folder), fix in sorted(ROOT_FIX_FOLDER.items()):
        w("  - %s `%s`: %+d" % (repo, folder, fix))
    w("\n## Contents\n")
    w("| instrument/articulation | roots (MIDI) | layers | round robins | files | MB | gain dB |")
    w("|---|---|---|---|---|---|---|")
    for (inst, art), js in sorted(by.items()):
        roots = [j["root"] for j in js]
        mb = sum(os.path.getsize(os.path.join(dest, ".multi.new", j["file"])) for j in js) / 1e6
        rng = ("%s-%s (%d-%d)" % (note_name(min(roots)), note_name(max(roots)), min(roots), max(roots))) \
            if js[0]["pitched"] else "60 (unpitched)"
        w("| %s/%s | %s | %d | %d | %d | %.1f | %+.1f |" % (inst, art, rng, js[0]["nlayers"], max(j["rr"] for j in js),
                                                         len(js), mb, gains[(inst, art)]))
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------
HEADER = "instrument\tarticulation\troot\tvel_lo\tvel_hi\trr\tgain_db\tfile"


def summary(dest):
    """Validates manifest.tsv against the interface and prints a table."""
    rows = []
    with open(os.path.join(dest, "manifest.tsv"), encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
    if not lines or lines[0] != HEADER:
        die("manifest.tsv: bad header")
    name_ok = re.compile(r"^[a-z0-9_]+$")
    for ln in lines[1:]:
        if not ln:
            continue
        p = ln.split("\t")
        if len(p) != 8:
            die("manifest.tsv: bad line %r" % ln)
        r = dict(instrument=p[0], articulation=p[1], root=int(p[2]), vel_lo=int(p[3]), vel_hi=int(p[4]),
                 rr=int(p[5]), gain_db=float(p[6]), file=p[7])
        if not (name_ok.match(r["instrument"]) and name_ok.match(r["articulation"])):
            die("manifest.tsv: bad name in %r" % ln)
        rows.append(r)
    errors = []
    by = {}
    for r in rows:
        by.setdefault((r["instrument"], r["articulation"]), []).append(r)
    total = 0
    log("%-28s %-22s %6s %6s %6s %8s %7s" % ("instrument/articulation", "roots", "layers", "rr", "files", "MB", "gain"))
    for key, rs in sorted(by.items()):
        if len({r["gain_db"] for r in rs}) != 1:
            errors.append("%s/%s: more than one gain_db" % key)
        layers = sorted({(r["vel_lo"], r["vel_hi"]) for r in rs})
        lo = 0
        for a, b in layers:
            if a != lo or b < a:
                errors.append("%s/%s: velocity layers do not tile 0-127" % key)
            lo = b + 1
        if lo != 128:
            errors.append("%s/%s: velocity layers do not reach 127" % key)
        groups = {}
        for r in rs:
            groups.setdefault((r["root"], r["vel_lo"]), []).append(r["rr"])
        for g, rr in groups.items():
            if sorted(rr) != list(range(1, len(rr) + 1)):
                errors.append("%s/%s: round robins of %s are %s" % (key + (g, rr)))
        size = 0
        for r in rs:
            path = os.path.join(dest, r["file"])
            try:
                with wave.open(path, "rb") as w:
                    ch = w.getnchannels()
                    if w.getframerate() != RATE or w.getsampwidth() != 2 or ch not in (1, 2):
                        errors.append("%s: not 48 kHz 16-bit mono/stereo" % r["file"])
                        continue
                    a = array("h")
                    a.frombytes(w.readframes(w.getnframes()))
            except (OSError, wave.Error, EOFError) as e:
                errors.append("%s: %s" % (r["file"], e))
                continue
            if sys.byteorder == "big":
                a.byteswap()
            on = onset_frame(a, ch, peak_of(a) * 10 ** (ONSET_DB / 20))
            if on > 0.005 * RATE:
                errors.append("%s: onset %.1f ms from the start" % (r["file"], 1000.0 * on / RATE))
            size += os.path.getsize(path)
        total += size
        roots = sorted({r["root"] for r in rs})
        rng = "60 (unpitched)" if key[0] in UNPITCHED_NAMES else \
            "%s-%s (%d-%d)" % (note_name(roots[0]), note_name(roots[-1]), roots[0], roots[-1])
        log("%-28s %-22s %6d %6d %6d %8.1f %+7.1f" % ("%s/%s" % key, rng, len(layers),
                                                      max(r["rr"] for r in rs), len(rs), size / 1e6, rs[0]["gain_db"]))
    log("%-28s %-22s %6s %6s %6d %8.1f" % ("total", "", "", "", len(rows), total / 1e6))
    if errors:
        for e in errors:
            print("error: " + e, file=sys.stderr)
        sys.exit(1)


def main():
    a = sys.argv[1:]
    if a[:1] == ["repos"]:
        for name, (url, commit) in REPOS.items():
            print("%s\t%s\t%s" % (name, url, commit))
    elif a[:1] == ["folders"] and len(a) == 2:
        for f in folders(a[1]):
            print(f)
    elif a[:1] == ["uptodate"] and len(a) == 2:
        sys.exit(0 if uptodate(a[1]) else 1)
    elif a[:1] == ["build"] and len(a) == 3:
        build(os.path.abspath(a[1]), os.path.abspath(a[2]))
    elif a[:1] == ["summary"] and len(a) == 2:
        summary(os.path.abspath(a[1]))
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
