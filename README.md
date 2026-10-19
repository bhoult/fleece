# fleece

A live fractal flame animator in the manner of [Electric Sheep](https://electricsheep.org). The flames are rendered in real time on the GPU and on every CPU core at once. It is written in Rust with no crates: the window and OpenGL 4.6 context come from libSDL2, and every GL call is looked up at run time.

```
cargo build --release
target/release/fleece                 # Electric Sheep's 800x592 frame, scaled to the window, 30 fps
target/release/fleece --fullscreen    # the same, filling the screen (or press F)
target/release/fleece --quality 2000  # Electric Sheep's own quality every frame, at whatever frame rate that takes
target/release/fleece 2k              # render at a size instead: 1080p, 2k (2560x1440), 4k (3840x2160), WxH
```

By default fleece renders Electric Sheep's standard frame, 800x592, and scales it to the window with black bars, as the Electric Sheep player shows its 800x592 videos. At that size this machine gets roughly 700–2,500 samples per pixel per frame at 30 fps without supersampling, and about 60 % of that at the automatic 2x; the server renders each frame at 2,000. `--quality N` holds every frame at N samples per pixel (2,000 runs at about 14–20 fps here). `--es-hd` renders the HD flock's 1280x720, `--render WxH` any fixed size, and `--native` the window's own size.

## Version 2

v2 adds ten things to the show. Electric Sheep's show as it was is still there: `V` switches between the two, and `--classic` starts without them.

| | key | flag | default | what it does |
| --- | --- | --- | --- | --- |
| Supersampling | X | `--supersample N` | automatic | The histogram holds 2x2 or 4x4 buckets a pixel, and flam3's gaussian spatial filter brings them down to pixels, as the Electric Sheep server rendered (its sheep say `supersample="4" filter="1"`). Thin lines come out smooth instead of stair-stepped. Automatic means 2x while the histogram fits the GPU's cache (800x592 does) and 1x above that, where the scattered atomics slow to memory speed. Stills (`--shot`) use the sheep's own setting. |
| Glow | O | `--glow 0-3` | soft | The bright parts blurred wide, through six half-size levels, and added back with a screen blend, like light scattering in a lens. The light it adds is softly limited, so bright regions keep their detail instead of going white. Off, subtle, soft or strong (`--glow 2` is soft; before the subtle step it was `--glow 1`). |
| Trails | T | `--trails 0-2` | short | Frame blending: each frame keeps a fading share of the ones before. Short trails (60 ms) mostly smooth the picture; long ones (350 ms) smear the motion into streaks. A paused picture still refines forever, and a cut starts afresh. |
| Levels | L | `--levels on/off` | on | Auto-exposure. The GPU counts each frame's lit pixels by brightness. A dim or blown-out sheep is moved towards the real sheep's typical level, measured on 80 genebank sheep. Sheep near it are left alone, and only the most saturated are pulled down. |
| Drift | D | `--drift on/off` | on | A slow camera move: zoom in by up to a third, pan a tenth of the frame and turn a few degrees, on periods of 29 to 61 seconds that never line up. |
| Hue | U | `--hue on/off` | off | Palette drift: the palette's hues turn once every 96 seconds. |
| Wobble | W | `--wobble on/off` | on | Motion for every sheep. Each new sheep, real or bred, gets one or two of fleece's own `<motion>` elements: a variation's weight, an xform's size, position or colour swinging one or two times a loop, so the loop still closes. They are tested at four phases of the loop like any new sheep, and dropped if they make it go dark. They are saved with the flock (marked `fleece_extra`) and play only while wobble is on. |
| 3D | 3 | `--tilt 0-2` | off | A 3D camera: the flame tilted back in perspective, swaying slowly, and with depth of field at 2. See "3D" below. |
| Audio | A | `--audio [MUSIC]`, `--audio-strength 0-3` | off | The music drives the show (see below): off, low, medium (what `--audio` gives) or high. |
| Music | N, J | `--music GENRE[+GENRE]` | off | Experimental: music fleece makes up as it plays, which the show then follows (see below); J adds a secondary genre that drums for the primary. |
| Recorded instruments | | `--samples DIR`, `--no-samples` | on once fetched | Every genre plays real recordings, fetched by `tools/get-samples.sh` (see below); without them, the synthesiser. |
| Taste | | `--taste on/off` | on | New sheep are the best of three candidates by a taste score (see below). |
| New variations | | | on | Half of the random new sheep ("random+") draw from 31 more variations: 3D, direct-colour and Apophysis plugins (see "3D" below). Classic breeds from flam3's 99 alone. |

**3D.** fleece reads the 3D camera of Apophysis 7X and Fractorium: `cam_pitch` and `cam_yaw` (radians), `cam_perspective`, `cam_zpos`, `cam_dof`.
- **Points carry a z.** The affine passes it on. flam3's variations keep it or drop it as Fractorium's do.
- **13 3D variations** shape the z: linear3D, Spherical3D, julia3D, julia3Dz, hemisphere, zcone, zblur, ztranslate, zscale, blur3D, curl3D, pie3D and sinusoidal3D.
- **The projection** follows Fractorium. After the final xform, each point is tilted and turned about the centre of the view and seen in perspective. Points behind the camera are dropped, and each point is blurred with its distance from the focal plane.
- **14 Apophysis plugins**, as Fractorium implements them: eclipse, epispiral, circlize, glynnia, SphericalN, xheart, petal, squirrel, hole, flipcircle, flipy, bMod, bSwirl and collideoscope.
- **Direct colour:** dc_linear, dc_bubble, dc_carpet and dc_cylinder colour a point by where it lands. dc_ztransl lifts it by its colour.
- **The 3 key** tilts any sheep: it leans back about 43°, swaying, with the same lean whatever the sheep's own zoom. Pressed again, it adds depth of field: the middle of the plane stays sharp while the near and far edges blur.
- **Speed:** a flat flame without these runs exactly the chaos game it always did. 3D is a separate variant of the GPU shader and of the CPU loop. A general 3D shader is compiled in the background at start, so the first tilted frame doesn't wait for a compile.
- `--varcheck` covers the new variations with a tilted camera, and one case with depth of field.

**Taste.** Each candidate is measured on a small render of its first frame: colourfulness (Hasler and Süsstrunk), coverage, brightness entropy, detail, symmetry, distinct colours, mean brightness and streaks. Streaks measures how much the picture's colour gradients point one way. A linear score of those says how much it looks like a real Electric Sheep. `fleece --tastefit` fitted the score to 3,000 genebank sheep against 3,000 random sheep that pass flam3's tests, with a cross-validated correlation of 0.58. Real sheep are more colourful and more detailed and use more distinct colours. A fixed penalty applies past the streaks of 99% of real sheep. Without it, a direct-colour flame painting stripes across the frame would win on colourfulness. The HUD shows each sheep's score. Every Up or Down vote nudges the score towards what you like, and the window keeps it in `~/.local/share/fleece/taste.txt`. (Fitted to the genebank's own Electric Sheep ratings instead, `--tastefit --breed 1`, the same measures predict little: r 0.12. Those sheep had all survived the flock already.)

**Audio.** `A` (or `--audio`) listens to what the speakers play: PulseAudio or PipeWire's monitor of the default output, through `parec`, which runs only while audio is on. No microphone is used. Each hop of 512 samples (about 94 a second) gives the loudness, four bands (bass, low mid, mid, high, each with its own automatic gain) and beats (onsets from spectral flux). Bass and beats push the camera in. Each band swells some of the xforms. Loudness turns the sheep faster, and the mid band shifts the hues. How strongly is A's setting (`--audio-strength`): low is half of medium, high nearly twice, on everything the music moves. The music also drives the other effects: the picture brightens with loudness and flashes on each beat (on top of auto-levels, which leaves the music out of what it steers); the glow, when on, swells with the music and flares on beats; the camera drift, when on, runs faster and swings wider in loud passages; and the palette drift, when on, turns with the highs and steps on each beat. `--audio MUSIC` on a video makes a music video. The track (any format ffmpeg reads) is analysed ahead, the frames follow it exactly, it goes into the file, and the video runs as long as the music unless `--seconds` says otherwise:

```
target/release/fleece --video dance.mp4 --audio song.mp3
target/release/fleece --audiocheck song.wav      # the analysis as text meters (no file: live, for --seconds)
```

**Music (experimental).** `N` plays music fleece makes up as it goes, and turns audio on so the show follows it: off, then ambient, techno, synthwave, lo-fi, polka, tavern (medieval: jigs, reels and dances on lute, fiddle, whistle, hurdy-gurdy and frame drum), ska (Jamaican first wave, with a shuffle, a jazz band's horns, Hammond and piano; faster, straighter 2-tone with a bubbling Farfisa or Vox; or third-wave ska-punk, its ska verses breaking into choruses of distorted power chords under four horns in close harmony: the guitar's chop on every off-beat over a walking bass, a horn head in unison or harmony with stabs answering it, solos over the changes with riffs behind them, and a tag ending on a final hit) and mariachi (Jalisco: each piece a son jalisciense, huapango, ranchera valseada or polkeada, or bolero ranchero; violins and trumpets trade the tune in thirds and sixths over the strummed vihuela and guitar, the guitarrón and a harp, with a trumpet fanfare to open and between verses, the odd zapateado, and a V7-I flourish to close) in turn, then off. That is the primary genre. `J` picks a secondary one: none at first, then each genre but the primary in turn (when N lands on the secondary, the secondary clears); a change starts a new piece at the next bar. With a secondary the primary writes the music (its form, meter, chords and tunes) and the secondary plays the drums (its beat, fitted to the primary's meter, with its kits and effects, a tempo 60 % of the way to its own, and drums wherever either genre's arrangement has them), and each part has an instrument from each genre. So polka with techno might pass its tune between an accordion and a hoover and its oom-pah between a tuba and a 303, over a four-on-the-floor kick with claps, rumble and sidechain; tavern with lo-fi might give its jigs and reels to a fiddle and a sax, a lute and a Rhodes, over a boom-bap beat with vinyl crackle and tape wow. Any of the 56 pairs plays. A composer writes pieces of about two minutes (64-158 bpm), each with its own key, mode, tempo, chord progressions (chords every bar or two, borrowed chords and secondary dominants, the odd key change), drum patterns, bassline, a 16-bar melody and a second voice answering it, laid out in sections with something changing every four bars; within a piece the bass mutates at phrase ends, the melody comes back varied, and filters drift; the next piece usually moves to a neighbouring key. Each genre has 28-44 instruments and kit pieces; a piece picks two for each part and swaps them between sections. It never loops, though each genre keeps its style. Recordings of real instruments play it (see below), or without them a small synthesiser (band-limited oscillators, resonant filters, FM bells and electric piano, synthesised drums), through reverb, delay, sidechain and a limiter, at well under 1 % of one core. It is made in SDL's audio callback, which runs on a real-time thread (round robin, through PulseAudio's or PipeWire's server; `pacat` if SDL cannot open a device), and the chaos-game workers run at nice 10, so a busy machine does not make it stutter. A genre switch waits for the next bar and crossfades. Recordings carry it like any other sound. `--music` takes a genre's name or number, or two names, primary first (`polka+techno`: polka drummed by techno). On a video, `--music GENRE` is the soundtrack: rendered first (the same genre and `--seed` always give the same music), followed like `--audio MUSIC`, and put in the file. `--musicwav GENRE SECONDS OUT.wav` writes it to a file alone.

```
target/release/fleece --music ambient
target/release/fleece --video dance.mp4 --music techno --seconds 60
```

**Recorded instruments.** Every genre plays real recordings once `tools/get-samples.sh` has fetched them: about 755 MB into `~/.local/share/fleece/samples`, plus about 2 GB of git clones in `~/.cache/fleece/samples-src` that can be deleted afterwards (it needs git, curl, ffmpeg and python3). The acoustic instruments are recordings of themselves: grand and honky-tonk pianos, electric pianos and organs; vibraphone, marimba, glockenspiel, celesta, kalimba, music box, dulcimer and tubular bells; guitars, lute, mandolin, banjo, harp and psaltery; violins, fiddle, cello and contrabass, bowed and plucked; flute, piccolo, recorder, ocarina, saxes, clarinet and bagpipes; trumpets, muted trumpet, trombone, horn and tuba; accordions, bandoneon and harmonica; choirs; and drum kits, brushes, frame drum, tambourine, shakers, congas, castanets, claps, guiro and cajón. The electronic genres play the SoundFont's synthesisers and drum machines (saw and square leads, synth basses, pads, the 808 and an electronic kit), each through its patch's own filter, drive and envelope, so the plucks and stabs still die away. Where nothing was recorded, the nearest recorded instrument stands in: an accordion for the Farfisa, a reed organ for the Vox, a harmonica for the melodica, an FM electric piano for the Wurlitzer, a fiddle through a nasal filter for the hurdy-gurdy, a clarinet through a resonant filter for the crumhorn, and a cajón for the zapateado. Only the sub-bass (the SoundFont's sine wave is the same sine) and the noise textures (risers, wind, vinyl crackle, hiss) stay synthesised. A sampler plays them: the SoundFont's zones, loops and envelopes, and multisampled recordings with velocity layers and round robins, short notes on staccato recordings and long ones on sustained ones with vibrato. Each is levelled to the synthesised instrument it replaces and goes through the same mix, and other genres borrow them in mixes. In the window they load in the background when music first starts and play from the next piece on; `--musicwav` and a video's soundtrack load them first, so a seed still gives the same music. They take about 730 MB of memory. `--no-samples` keeps everything synthesised, as does a missing directory; `--samples DIR` reads another directory, and `--samples-list` prints what loads. The recordings come from MuseScore General 0.2 (MIT: FluidR3 by Frank Wen, FluidR3Mono by Michael Cowgill, adapted by S. Christian Collins) and from Versilian Studios' VS Chamber Orchestra 2: Community Edition and Versilian Community Sample Library (CC0); `SOURCES.md` and `licenses/` in the samples directory list everything taken.

```
tools/get-samples.sh                                  # fetch and prepare the recordings (once)
target/release/fleece --music mariachi --no-samples   # the synthesiser alone
```

**Faithfulness.** With supersampling, fleece's renders of real sheep match flam3-render's at 37–40 dB PSNR (v1: 20–28). At the server's own settings (4x, quality 2,000) a frame takes 0.8–0.9 s against flam3's 4.7–8.6 s. Both were timed while another program kept the GPU busy and the machine loaded. 4x is the slow case: its histogram no longer fits in the GPU's cache. The density estimation gathers up to 16 buckets away, the most the GPU's shared memory allows. flam3's kernels at 4x reach further, so the faintest haze is a little noisier than flam3's.

## What Electric Sheep does, and what fleece does

Electric Sheep's pictures are ordinary flam3 renders: the chaos game, log-density tone mapping and density estimation. What makes it Electric Sheep is everything around the renders. fleece follows flam3 and the Electric Sheep sources (scottdraves/flam3 and scottdraves/electricsheep, plus the 2005 server code) for each part.

**Sheep and loops**
- A sheep is a loop: every animated xform's linear part turns 360° over 128 frames at 23 fps, about 5.6 s (`flam3_rotate`, `sheep_loop`).
- Each loop plays twice.

**Edges**
- An edge is a 128-frame morph into the next sheep (`sheep_edge`).
- Both sheep are padded to the same xforms with flam3's padding rules. Spherical-like variations get a turned identity of weight -1; fan and rings get a swapped one.
- Each affine column is interpolated in log-polar form, with flam3's wind angles when one side is a symmetry xform.
- Palettes are blended in HSV with the hue taking the short way round.
- Everything else is linear. The blend is eased with flam3's `3t² - 2t³` while both sheep keep turning, so loop, edge and loop join smoothly.

**Motion blur**
- Each frame is 64 time samples spread over one Electric Sheep frame. flam3 used 1,000 temporal samples.

**The flock and evolution**
- New sheep come from the flock as on the server:
  - 70% are descendants: a third flam3 mutations (seven kinds, flam3's odds), the rest crossovers (union, blend or alternating xforms, and a palette splice 40% of the time);
  - the rest are flam3-genome random genomes;
  - 20% get a symmetry "singularity".
- Parents are picked by rating.
- Every candidate, mutations from the M key included, goes through:
  - flam3-genome's 128×128 test render (mean channel ≥ 20, at least 1% black pixels, at most 5% white);
  - its bounding-box framing;
  - a check of the framed view at four turns of the loop. The mean channel must stay at 8 or more; that is fleece's own rule.
- flam3's test looks at [-1, 1]² at one angle only. About one sheep in ten that passes it, and one untested mutation in five, goes nearly black on screen somewhere in its loop. With the view check, none did in 300 of each (`--mutcheck`).
- Each candidate gets up to ten tries and keeps the best. Mutations are bred on the breeder thread too; the HUD says BREEDING A MUTATION until it morphs in.
- Votes: Up likes the sheep on screen, Down dislikes it and moves on. The 32-sheep flock and its ratings are saved to `~/.local/share/fleece/flock.flam3` and carry over to the next run (`--fresh` starts over).

**Real sheep (the genebank)**
- Every `.flam3` file in `~/.local/share/fleece/genebank/` is a real sheep the show draws on, as the server drew on its genebank. Real sheep are shown as designed (their own framing and colours) or crossed with each other and with sheep you like. They are picked in proportion to the square root of their Electric Sheep rating.
- The repository's `sheep/` directory holds 5,925 of them (with `sheep/README.md` on their licence); fleece uses it when `~/.local/share/fleece/genebank/` is empty. The genebank installed here holds 5,923 sheep. They come from generations 165–248 of the live server (v3d0.sheepserver.net still serves generations 247 and 248), the Wayback Machine, and GitHub copies, including Scott Draves' own genebank file. 4,945 of them have ratings.
- fleece reads what real genomes use:
  - per-sheep brightness, gamma, highlight power, vibrancy, density estimation and the linear palette mode;
  - the older `symmetry` attribute (colour speed and a non-turning xform) and `<symmetry kind=N/>`;
  - xaos (`chaos`) and opacity;
  - `<motion>` elements: values that swing over each loop (sin, triangle or hill, applied before the turn, as flam3's sheep_loop does). A motion sheep checked against flam3-genome's loop frames lines up: PSNR 26.8 dB at the same phase, 18 dB a quarter loop apart.
- Not done: the server's 4x supersampling. Each sheep's own `size` is not used either; the show renders every sheep into the same frame, framed by width.

**Rendering**
- All 99 of flam3's variations, numbered and named as in flam3, with flam3's random parameter ranges.
- flam3's 701 palettes, hue-turned one time in eight.
- flam3's tone curve: brightness, gamma 4 with the linear range, vibrancy, highlight power, background.
- Density estimation: radius 9, curve 0.4, and each pixel's kernel narrowing with its hit count. It runs as a shared-memory gather on the GPU.

**Checked against the real thing**
- flam3 3.1.1 (built from the source) rendered six of fleece's genomes. fleece's frames match them in structure, colour and brightness; flam3's are grainier at equal quality.
- `--varcheck` renders a flame made of each variation on the CPU alone and on the GPU alone and compares where the points land.
  - Over three seeds, every variation agrees (L1 distance under 0.15, mostly under 0.03) except rings2 at a tiny ring width, where points sit on a `trunc()` boundary and the GPU's faster division rounds some the other way.
  - The one exception is a "sin" flame that sends 99.99% of its points off to infinity, leaving a few thousand on screen.
  - Flames with no points on screen on either side are skipped.

Electric Sheep's client plays pre-rendered videos. fleece renders every frame live instead, and it has no server: breeding happens in a background thread.

## CPU and GPU

Both run the chaos game on the same frame and add into one picture.

- **GPU:** an OpenGL compute shader runs 262,144 walkers, 64 iterations a dispatch.
  - Each work group caches its time sample's xforms, palette and xform-choice table in shared memory.
  - Points are added with 64-bit atomics: two per point instead of four (NVIDIA).
  - For every sheep (and every edge) a shader with only the variations it uses is compiled in the background (KHR_parallel_shader_compile). It is up to about 2x faster than the general one. The HUD says SHADER TUNED once it is in use.
  - The number of dispatches per frame adapts to keep the GPU busy for about 85% of the display's frame time.
- **CPU:** one worker per core except one, at nice 5, so the desktop and the render thread come first.
  - The workers use AVX2: eight walkers per vector, each picking its own xform. Per-xform values are looked up with vpermps, and sin, cos, atan2, ln, exp and pow are eight-lane polynomials accurate to about 1 ulp.
  - They write into a persistent, coherent, client-storage GL buffer. That is cached system memory, which the GPU's merge pass reads directly, so nothing is copied per frame. Two such buffers alternate between frames.
- **Merging:** at the next frame's start the workers move to the new frame's samples. The finished frame's GPU and CPU histograms are merged, density-estimated and tone-mapped while the GPU starts the next frame, so the picture is one frame behind.

Measured here while another program kept the GPU at 100% and about 14 cores busy (load average 40-60):

| | iterations / s |
| --- | --- |
| GPU (RTX 5080), specialised shader | about 5-10 G, depending on the flame and the other load |
| CPU (Ultra 9 285K, 23 workers) | about 0.6-1.0 G |

At 1280x720 and 30 Hz that is about 150-350 samples per pixel per frame. Electric Sheep used 500, with 2x oversampling; density estimation hides the difference.

`C` and `G` turn either side off, so you can see what each one adds.

## Keys

| key | |
| --- | --- |
| Left / Right | previous / next sheep: a morph back through the sheep already shown (up to 500), or forward; past the newest, a new sheep |
| Space | next sheep (as Right) |
| Up | like this sheep (its rating +1: it breeds more often) |
| Down | dislike it (rating -1, skip; at a negative rating it leaves the flock) |
| M | morph into a (tested) mutation of the sheep the HUD names |
| P | pause: the picture keeps refining until P again |
| - / +, mouse wheel | animation speed, 1/16x to 4x |
| B | motion blur on / off |
| C / G | the CPU's / the GPU's share on / off |
| E | render Electric Sheep's 800x592 frame, scaled to the window |
| 1 / 2 / 4 | render 1080 / 1440 (2K) / 2160 (4K) lines, at the window's shape |
| S | save the frame (PNG) and its genome (flam3 XML) to ~/Pictures/fleece |
| R | start / stop recording the show to ~/Videos/fleece-TIME.mp4 (see below) |
| K | the credit burned into recordings on / off (it is in the file's metadata either way) |
| V | v2's effects off (classic Electric Sheep) / back on |
| O / T | glow (off, subtle, soft, strong) / trails (off, short, long) |
| X | supersampling: automatic, 1x, 2x, 4x |
| L / D / U / W | auto levels / camera drift / palette (hue) drift / wobble (motion for every sheep) |
| 3 | 3D camera: flat, tilted, tilted with depth of field |
| A | audio: the music drives the show, off / low / medium / high |
| N | music: fleece's own, off / ambient / techno / synthwave / lo-fi / polka / tavern / ska / mariachi (the primary genre) |
| J | music's secondary genre (drums and half the instruments): none, then each genre but the primary |
| Z | optimize: find settings that keep up with the display's refresh rate (below) |
| H | hide the text and buttons / show them again (a recording's REC line stays) |
| F, F11, double click | fullscreen |
| Esc, Q | quit |

**Optimize (Z).** The GPU's share of each frame adapts to the display's frame time, but the rest of the frame (density estimation, the spatial filter, glow, waiting for the CPU) comes on top, so a busy frame can miss the display's refresh. Z measures the frame rate for a second and a half and, while it is below the display's rate, tries cheaper settings one at a time, least visible first: fewer samples per frame (70 %, then 55 %, then 40 % of the frame time; trails hide the extra noise), no supersampling, at most 1080 lines, the GPU alone. Each is measured for a second and a half and kept only if the frame rate rose; the HUD shows what it is trying, and then what it kept. Z again stops it. The window does this by itself a few seconds after it opens, once the first sheep's own shader runs (`--optimize off` leaves the settings alone).

**Mouse.** Under the text are three rows of buttons, one for each key above, each labelled with its key and lit when its setting is on: the show (previous, next, like, dislike, mutate, pause, speed, record, save, fullscreen, hide, quit), v2's effects, and the renderer (CPU, GPU, blur, render size, credit). Clicking one does what its key does. Once hidden (H or its button), only the H key brings them back. In fullscreen the pointer shows only while the mouse moves. In a headless run, a capital letter in `--keys` clicks that key's button (`--keys "30:O"` clicks GLOW), which tests the buttons without a display.

A size given on the command line (`1080p`, `2k`, `4k`, `WxH`) is rendered at full resolution, and so is fullscreen started from it. With `--native`, the window and fullscreen are rendered at most 1080 lines tall: a 4K screen gets a 2x upscale and four times the samples per pixel. `--scale F` sets the render size relative to the window, for example `4k --fullscreen --scale 0.5`.

## Other modes

```
target/release/fleece --load sheep.flam3            # show these flames first (flam3, Electric Sheep, Apophysis, Fractorium XML)
target/release/fleece --shot out.png --seed 7 --quality 500 1080p   # one still (also writes out.flam3)
target/release/fleece --frames 300 --out DIR --every 30 --keys "100: ,200:m"   # headless run, PNGs, key presses
target/release/fleece --bench                       # chaos-game rate of each side alone
target/release/fleece --varcheck                    # CPU against GPU for every variation
target/release/fleece --mutcheck                    # how often mutations go dark, untested and tested
target/release/fleece --tastefit                    # refit the taste score (real sheep against random ones)
target/release/fleece --audiocheck [song.wav]       # the audio analysis as text meters
target/release/fleece --video sheep.mp4 --seconds 60 --credit   # a video file (headless): ES frame and quality
target/release/fleece --video sheep.webm 1080p --quality 500     # any size / quality; the format follows the extension
target/release/fleece --video sheep.y4m --seconds 10             # .y4m: uncompressed, written without ffmpeg
tools/video.sh -s 30 -l 1080                                     # the same through a script: ~/Videos/fleece-demo.mp4
```

**Recording the live show (R).**
- The file is ~/Videos/fleece-TIME.mp4: H.264, at the render size (800x592 by default), at the display's frame rate.
- Without sound, every frame goes into the file. While recording, the show steps exactly one display frame of animation per frame, and a busy encoder makes the display wait rather than drop a frame. The video plays at true speed with nothing skipped or repeated. If the machine cannot keep up live, the live view slows down instead of the video.
- With audio on (A), the recording has sound: what the speakers play, which is any other program's sound and fleece's own music alike, captured from the default output's monitor with `parec` and added to the file when recording stops. Turning audio on during a recording starts the sound from that point. From then on the video keeps to the clock, as the sound does: a frame the renderer was late with is shown again rather than the video drifting off the sound. The HUD's REC line says WITH SOUND. Live, the picture follows the sound a little late (a frame of rendering, about 40 ms of analysis: 1-3 frames on a 30 Hz display), which a recording makes up: its sound goes into the file that much later, so picture and sound line up. `--sound-delay MS` sets that delay by hand.
- It holds the picture without the controls, with the sheep's credit burned into its first 10 seconds (K turns that on or off; `--credit off` starts with it off).
- Changing the render size (E, 1, 2, 4, or fullscreen without a fixed size) ends the recording. So does quitting, which finishes the file.

Videos get v2's look by default, like the window: automatic supersampling, glow, short trails, levels, drift and wobble. `--classic` renders the show as Electric Sheep's server did, and `--classic --supersample 4` matches its renders most closely.

`--video FILE` renders frames at a fixed quality, `--quality` samples per pixel (default 2,000, Electric Sheep's), at `--fps` (30) for `--seconds` (60) or `--frames`, at Electric Sheep's 800x592 unless a size is given.
- ffmpeg encodes the file, chosen by its extension: H.264 for .mp4/.mkv/.mov (`--crf`, default 17), VP9 for .webm. `.y4m` is written directly and `-` streams raw RGB24 to stdout.
- Credits:
  - The sheep's name and credit are burned into the bottom left of the first 10 seconds of each video and recording, wrapped to fit, then fade out over a second (`--credit-secs N` changes the 10). `--credit off` leaves the frames clean.
  - Either way, every video and recording carries the credits for every sheep that appeared: in the file's metadata (title, artist, comment, description, copyright; a stream-copy remux after encoding) and in `FILE.credits.txt` next to it.
  - Licences follow the sheep: designed sheep are CC BY, evolved ones CC BY-NC, and anything bred from an evolved sheep stays CC BY-NC.
- The HUD's second line always shows the credit: "artwork by Scott Draves and the Electric Sheep", the licence (CC BY-NC for evolved sheep, CC BY for designed ones) and the designer. For fleece's own random sheep it credits flam3 instead.

The headless modes use SDL's offscreen driver, so the context is made on the GPU through EGL and no display is needed. They never touch the saved flock.

## Files

| file | contents |
| --- | --- |
| `src/app.rs` | the window, frame loop, adaptive GPU budget, HUD, keys, headless and video modes |
| `src/show.rs` | the playback: loops, edges, motion-blur samples, the breeder thread |
| `src/breed.rs` | the flock, mutation, crossover, framing, the fitness test, wobble |
| `src/taste.rs` | the taste measures and score |
| `src/audio.rs` | listening (parec), the FFT bands and beats, soundtracks |
| `src/music.rs` | generated music (experimental): the composer, the synthesiser, the player |
| `src/genome.rs` | genomes, flam3_random, symmetry, rotation, the flattening for the chaos game, XML output |
| `src/interp.rs` | edges: padding, log-polar affines, HSV palettes |
| `src/var.rs` | the variation table (flam3's 99, then fleece's 3D, plugin and direct-colour ones), random parameters, precalcs |
| `src/flam3.rs` | reading flam3 XML; the flock file |
| `src/render.rs` | one frame on both sides: publish, merge, tone |
| `src/gpu.rs`, `src/shaders/` | buffers, specialised shaders, the passes (iterate, merge with frame blending, de: log density and density estimation per bucket, finish: flam3's spatial filter, gamma and the brightness counts, glow, present) |
| `src/cpu.rs`, `src/cpuvar.rs` | the AVX2 chaos game, the worker pool, the variations in eight lanes |
| `src/fmath.rs`, `src/simd.rs` | eight-lane math (simd.rs from FRAKTAL256's Rust port) |
| `src/gl.rs`, `src/sdl.rs` | the OpenGL loader and the SDL2 calls |
| `src/hud.rs`, `src/png.rs` | the 5x7 text and a PNG writer |
| `assets/flam3-palettes.bin` | flam3's 701 palettes (from flam3-palettes.xml) |

## Requirements and licence

- An x86-64 CPU with AVX2 and FMA.
- An OpenGL 4.6 GPU. The 64-bit atomics and the parallel shader compiles are used when the driver has them (NVIDIA does).
- libSDL2 (`SDLLIB=/path/to/libSDL2-2.0.so.0` at build time if it is elsewhere).

The palettes and the algorithms (variations, tone mapping, density estimation, interpolation, mutation, crossover) come from flam3, which is GPL v3 or later (© Spotworks LLC / Scott Draves). Anything distributed from this tree therefore falls under the GPL v3 too. `src/simd.rs` and the HUD font come from FRAKTAL256's Rust port.

The genomes in `sheep/` are not covered by the GPL. They are Electric Sheep artwork under Creative Commons (see `sheep/README.md`), and the Electric Sheep reuse policy applies to them and to anything rendered from them:
- sheep evolved by the algorithm are CC Attribution-NonCommercial;
- sheep designed by people are CC Attribution and carry the designer's name;
- credit "artwork by Scott Draves and the Electric Sheep" (and the sheep's generation and number);
- commercial use needs a licence from Spotworks.
