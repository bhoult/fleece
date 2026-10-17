//! The few SDL2 calls used: a window with an OpenGL 4.6 core context (or a
//! hidden one through the offscreen driver, which makes its context with EGL
//! on the GPU and needs no display), events and the GL loader.

use std::ffi::{CStr, CString, c_char, c_int, c_void};

pub const SDL_INIT_VIDEO: u32 = 0x20;
pub const SDL_INIT_EVENTS: u32 = 0x4000;
const SDL_INIT_AUDIO: u32 = 0x10;
const AUDIO_S16LSB: u16 = 0x8010;
pub const SDL_WINDOW_FULLSCREEN_DESKTOP: u32 = 0x1001;
pub const SDL_WINDOW_OPENGL: u32 = 0x2;
pub const SDL_WINDOW_HIDDEN: u32 = 0x8;
pub const SDL_WINDOW_RESIZABLE: u32 = 0x20;
pub const SDL_WINDOWPOS_CENTERED: c_int = 0x2FFF_0000;

const SDL_GL_DOUBLEBUFFER: c_int = 5;
const SDL_GL_DEPTH_SIZE: c_int = 6;
const SDL_GL_CONTEXT_MAJOR_VERSION: c_int = 17;
const SDL_GL_CONTEXT_MINOR_VERSION: c_int = 18;
const SDL_GL_CONTEXT_PROFILE_MASK: c_int = 21;
const SDL_GL_CONTEXT_PROFILE_CORE: c_int = 1;

pub const SDL_QUIT: u32 = 0x100;
pub const SDL_WINDOWEVENT: u32 = 0x200;
pub const SDL_KEYDOWN: u32 = 0x300;
pub const SDL_MOUSEMOTION: u32 = 0x400;
pub const SDL_MOUSEBUTTONDOWN: u32 = 0x401;
pub const SDL_MOUSEWHEEL: u32 = 0x403;
pub const SDL_BUTTON_LEFT: u8 = 1;
pub const SDL_WINDOWEVENT_SIZE_CHANGED: u8 = 6;

pub const K_ESCAPE: i32 = 27;
pub const K_RIGHT: i32 = 0x4000_004F;
pub const K_LEFT: i32 = 0x4000_0050;
pub const K_DOWN: i32 = 0x4000_0051;
pub const K_UP: i32 = 0x4000_0052;
pub const K_F11: i32 = 0x4000_0044;
pub const K_KP_MINUS: i32 = 0x4000_0056;
pub const K_KP_PLUS: i32 = 0x4000_0057;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Event {
    pub kind: u32,
    raw: [u8; 52],
}

impl Event {
    pub fn new() -> Event {
        Event { kind: 0, raw: [0; 52] }
    }
    fn i32_at(&self, off: usize) -> i32 {
        let o = off - 4;
        i32::from_le_bytes([self.raw[o], self.raw[o + 1], self.raw[o + 2], self.raw[o + 3]])
    }
    /// SDL_KEYDOWN: the key code (keysym.sym)
    pub fn key(&self) -> i32 {
        self.i32_at(20)
    }
    /// SDL_KEYDOWN: held down and repeating
    pub fn repeat(&self) -> bool {
        self.raw[13 - 4] != 0
    }
    /// SDL_WINDOWEVENT: the event id
    pub fn window_event(&self) -> u8 {
        self.raw[12 - 4]
    }
    /// SDL_MOUSEWHEEL: y
    pub fn wheel_y(&self) -> i32 {
        self.i32_at(20)
    }
    /// SDL_MOUSEMOTION, SDL_MOUSEBUTTONDOWN: where, in window coordinates
    pub fn mouse_xy(&self) -> (i32, i32) {
        (self.i32_at(20), self.i32_at(24))
    }
    /// SDL_MOUSEBUTTONDOWN: which button (1 left), and 2 for a double click
    pub fn button(&self) -> (u8, u8) {
        (self.raw[16 - 4], self.raw[18 - 4])
    }
}

unsafe extern "C" {
    fn SDL_Init(flags: u32) -> c_int;
    fn SDL_InitSubSystem(flags: u32) -> c_int;
    fn SDL_QuitSubSystem(flags: u32);
    fn SDL_OpenAudioDevice(device: *const c_char, iscapture: c_int, desired: *const AudioSpec, obtained: *mut AudioSpec, allowed: c_int) -> u32;
    fn SDL_PauseAudioDevice(dev: u32, pause: c_int);
    fn SDL_CloseAudioDevice(dev: u32);
    fn SDL_GetCurrentAudioDriver() -> *const c_char;
    fn SDL_Quit();
    fn SDL_GetError() -> *const c_char;
    fn SDL_SetHint(name: *const c_char, value: *const c_char) -> c_int;
    fn SDL_GL_SetAttribute(attr: c_int, value: c_int) -> c_int;
    fn SDL_CreateWindow(title: *const c_char, x: c_int, y: c_int, w: c_int, h: c_int, flags: u32) -> *mut c_void;
    fn SDL_DestroyWindow(win: *mut c_void);
    fn SDL_SetWindowFullscreen(win: *mut c_void, flags: u32) -> c_int;
    fn SDL_SetWindowTitle(win: *mut c_void, title: *const c_char);
    fn SDL_GL_CreateContext(win: *mut c_void) -> *mut c_void;
    fn SDL_GL_DeleteContext(ctx: *mut c_void);
    fn SDL_GL_GetProcAddress(name: *const c_char) -> *mut c_void;
    fn SDL_GL_SetSwapInterval(interval: c_int) -> c_int;
    fn SDL_GL_SwapWindow(win: *mut c_void);
    fn SDL_GL_GetDrawableSize(win: *mut c_void, w: *mut c_int, h: *mut c_int);
    fn SDL_GetWindowSize(win: *mut c_void, w: *mut c_int, h: *mut c_int);
    fn SDL_GetCurrentVideoDriver() -> *const c_char;
    fn SDL_PollEvent(ev: *mut Event) -> c_int;
    fn SDL_ShowCursor(toggle: c_int) -> c_int;
    fn SDL_GetWindowDisplayIndex(win: *mut c_void) -> c_int;
    fn SDL_GetCurrentDisplayMode(index: c_int, mode: *mut DisplayMode) -> c_int;
}

/// SDL's audio callback: fill `len` bytes at `stream`.
pub type AudioCallback = unsafe extern "C" fn(user: *mut c_void, stream: *mut u8, len: c_int);

#[repr(C)]
struct AudioSpec {
    freq: c_int,
    format: u16,
    channels: u8,
    silence: u8,
    samples: u16,
    padding: u16,
    size: u32,
    callback: Option<AudioCallback>,
    userdata: *mut c_void,
}

/// A sound output: SDL calls the callback for 16-bit stereo on its own audio
/// thread, which it runs at real-time priority where the system allows
/// (RealtimeKit), ahead of everything else on a busy machine.
pub struct AudioOut {
    id: u32,
}

impl AudioOut {
    /// Opens the default output at `rate`, `frames` per callback, playing at once.
    ///
    /// # Safety
    /// `user` must stay valid, and `callback` safe to call with it from
    /// another thread, until the AudioOut is dropped.
    pub unsafe fn open(rate: u32, frames: u16, callback: AudioCallback, user: *mut c_void) -> Result<AudioOut, String> {
        unsafe {
            SDL_SetHint(c"SDL_THREAD_FORCE_REALTIME_TIME_CRITICAL".as_ptr(), c"1".as_ptr());
            SDL_SetHint(c"SDL_AUDIO_DEVICE_APP_NAME".as_ptr(), c"fleece".as_ptr());
            SDL_SetHint(c"SDL_AUDIO_DEVICE_STREAM_NAME".as_ptr(), c"music".as_ptr());
            if SDL_InitSubSystem(SDL_INIT_AUDIO) != 0 {
                return Err(error("SDL audio"));
            }
            let want = AudioSpec {
                freq: rate as c_int,
                format: AUDIO_S16LSB,
                channels: 2,
                silence: 0,
                samples: frames,
                padding: 0,
                size: 0,
                callback: Some(callback),
                userdata: user,
            };
            let mut have = AudioSpec { callback: None, userdata: std::ptr::null_mut(), ..want };
            // no allowed changes: SDL converts to the device's format itself
            let id = SDL_OpenAudioDevice(std::ptr::null(), 0, &want, &mut have, 0);
            if id == 0 {
                let e = error("SDL_OpenAudioDevice");
                SDL_QuitSubSystem(SDL_INIT_AUDIO);
                return Err(e);
            }
            SDL_PauseAudioDevice(id, 0);
            Ok(AudioOut { id })
        }
    }

    /// The audio driver in use (pipewire, pulseaudio, alsa...).
    pub fn driver(&self) -> String {
        let d = unsafe { SDL_GetCurrentAudioDriver() };
        if d.is_null() { "?".into() } else { unsafe { CStr::from_ptr(d) }.to_string_lossy().into_owned() }
    }
}

impl Drop for AudioOut {
    /// Closes the device (SDL waits for a callback under way), then the subsystem.
    fn drop(&mut self) {
        unsafe {
            SDL_CloseAudioDevice(self.id);
            SDL_QuitSubSystem(SDL_INIT_AUDIO);
        }
    }
}

#[repr(C)]
#[derive(Default)]
struct DisplayMode {
    format: u32,
    w: c_int,
    h: c_int,
    refresh: c_int,
    driverdata: usize,
}

pub fn error(what: &str) -> String {
    let e = unsafe { SDL_GetError() };
    let msg = if e.is_null() { "?".into() } else { unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned() };
    format!("{what}: {msg}")
}

/// A window and its current GL context.
pub struct Window {
    win: *mut c_void,
    ctx: *mut c_void,
    fullscreen: bool,
}

impl Window {
    /// `headless`: the offscreen driver (no display; EGL on the GPU).
    pub fn open(title: &str, w: u32, h: u32, headless: bool, fullscreen: bool) -> Result<Window, String> {
        unsafe {
            if headless {
                SDL_SetHint(c"SDL_VIDEODRIVER".as_ptr(), c"offscreen".as_ptr());
            }
            // the CPU threads are busy; don't let SDL pin anything to the compositor's taste
            SDL_SetHint(c"SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR".as_ptr(), c"0".as_ptr());
            if SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0 {
                return Err(error("SDL_Init"));
            }
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
            SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
            SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
            let mut flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE;
            if headless {
                flags |= SDL_WINDOW_HIDDEN;
            } else if fullscreen {
                flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
            }
            let t = CString::new(title).unwrap();
            let win = SDL_CreateWindow(
                t.as_ptr(),
                SDL_WINDOWPOS_CENTERED,
                SDL_WINDOWPOS_CENTERED,
                w as c_int,
                h as c_int,
                flags,
            );
            if win.is_null() {
                let e = error("SDL_CreateWindow");
                SDL_Quit();
                return Err(e);
            }
            let ctx = SDL_GL_CreateContext(win);
            if ctx.is_null() {
                let e = error("SDL_GL_CreateContext (OpenGL 4.6 core)");
                SDL_DestroyWindow(win);
                SDL_Quit();
                return Err(e);
            }
            if !headless {
                SDL_ShowCursor(if fullscreen { 0 } else { 1 });
            }
            Ok(Window { win, ctx, fullscreen: fullscreen && !headless })
        }
    }

    pub fn driver(&self) -> String {
        let d = unsafe { SDL_GetCurrentVideoDriver() };
        if d.is_null() { "?".into() } else { unsafe { CStr::from_ptr(d) }.to_string_lossy().into_owned() }
    }

    pub fn proc_address(name: &CStr) -> *const c_void {
        unsafe { SDL_GL_GetProcAddress(name.as_ptr()) }
    }

    /// 1: wait for the vertical blank; 0: don't.
    pub fn set_vsync(&self, on: bool) -> bool {
        unsafe { SDL_GL_SetSwapInterval(if on { 1 } else { 0 }) == 0 }
    }

    /// The refresh rate of the display the window is on.
    pub fn refresh_hz(&self) -> Option<f64> {
        let mut m = DisplayMode::default();
        let i = unsafe { SDL_GetWindowDisplayIndex(self.win) };
        (i >= 0 && unsafe { SDL_GetCurrentDisplayMode(i, &mut m) } == 0 && m.refresh > 0).then_some(m.refresh as f64)
    }

    pub fn swap(&self) {
        unsafe { SDL_GL_SwapWindow(self.win) }
    }

    pub fn drawable_size(&self) -> (u32, u32) {
        let (mut w, mut h) = (0, 0);
        unsafe { SDL_GL_GetDrawableSize(self.win, &mut w, &mut h) };
        (w.max(1) as u32, h.max(1) as u32)
    }

    /// The window's size in its own coordinates (the mouse's; the drawable
    /// may have more pixels on a scaled display).
    pub fn window_size(&self) -> (u32, u32) {
        let (mut w, mut h) = (0, 0);
        unsafe { SDL_GetWindowSize(self.win, &mut w, &mut h) };
        (w.max(1) as u32, h.max(1) as u32)
    }

    pub fn is_fullscreen(&self) -> bool {
        self.fullscreen
    }

    pub fn show_cursor(&self, on: bool) {
        unsafe { SDL_ShowCursor(if on { 1 } else { 0 }) };
    }

    pub fn set_title(&self, title: &str) {
        let t = CString::new(title).unwrap();
        unsafe { SDL_SetWindowTitle(self.win, t.as_ptr()) }
    }

    pub fn toggle_fullscreen(&mut self) {
        self.fullscreen = !self.fullscreen;
        unsafe {
            SDL_SetWindowFullscreen(self.win, if self.fullscreen { SDL_WINDOW_FULLSCREEN_DESKTOP } else { 0 });
            SDL_ShowCursor(if self.fullscreen { 0 } else { 1 });
        }
    }

    pub fn poll(&self) -> Option<Event> {
        let mut ev = Event::new();
        if unsafe { SDL_PollEvent(&mut ev) } != 0 { Some(ev) } else { None }
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        unsafe {
            SDL_GL_DeleteContext(self.ctx);
            SDL_DestroyWindow(self.win);
            SDL_Quit();
        }
    }
}
