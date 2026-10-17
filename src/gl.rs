//! OpenGL 4.6 core, loaded through SDL_GL_GetProcAddress: the entry points
//! and constants used, and shader / program helpers.
#![allow(non_snake_case, dead_code, clippy::too_many_arguments, clippy::missing_safety_doc)]

use std::ffi::{CStr, CString, c_char, c_void};

pub type GLenum = u32;
pub type GLuint = u32;
pub type GLint = i32;
pub type GLsizei = i32;
pub type GLbitfield = u32;
pub type GLintptr = isize;
pub type GLsizeiptr = isize;
pub type GLsync = *const c_void;

macro_rules! gl_fns {
    ($($name:ident ( $($an:ident : $at:ty),* ) $(-> $ret:ty)? ;)*) => {
        pub struct Gl {
            $($name: unsafe extern "C" fn($($at),*) $(-> $ret)?,)*
        }
        impl Gl {
            pub fn load(get: impl Fn(&CStr) -> *const c_void) -> Result<Gl, String> {
                Ok(Gl {
                    $($name: {
                        let n = CString::new(stringify!($name)).unwrap();
                        let p = get(&n);
                        if p.is_null() {
                            return Err(format!("OpenGL entry point {} is missing", stringify!($name)));
                        }
                        unsafe { std::mem::transmute::<*const c_void, unsafe extern "C" fn($($at),*) $(-> $ret)?>(p) }
                    },)*
                })
            }
            $(
                #[inline]
                pub unsafe fn $name(&self, $($an: $at),*) $(-> $ret)? {
                    unsafe { (self.$name)($($an),*) }
                }
            )*
        }
    };
}

gl_fns! {
    glGetString(name: GLenum) -> *const u8;
    glGetStringi(name: GLenum, i: GLuint) -> *const u8;
    glGetIntegerv(name: GLenum, data: *mut GLint);
    glGetError() -> GLenum;
    glViewport(x: GLint, y: GLint, w: GLsizei, h: GLsizei);
    glClearColor(r: f32, g: f32, b: f32, a: f32);
    glClear(mask: GLbitfield);
    glEnable(cap: GLenum);
    glDisable(cap: GLenum);
    glBlendFunc(s: GLenum, d: GLenum);
    glFinish();
    glFlush();
    glCreateShader(kind: GLenum) -> GLuint;
    glShaderSource(s: GLuint, n: GLsizei, src: *const *const c_char, len: *const GLint);
    glCompileShader(s: GLuint);
    glGetShaderiv(s: GLuint, p: GLenum, out: *mut GLint);
    glGetShaderInfoLog(s: GLuint, max: GLsizei, len: *mut GLsizei, log: *mut c_char);
    glDeleteShader(s: GLuint);
    glCreateProgram() -> GLuint;
    glAttachShader(p: GLuint, s: GLuint);
    glLinkProgram(p: GLuint);
    glGetProgramiv(p: GLuint, q: GLenum, out: *mut GLint);
    glGetProgramInfoLog(p: GLuint, max: GLsizei, len: *mut GLsizei, log: *mut c_char);
    glUseProgram(p: GLuint);
    glDeleteProgram(p: GLuint);
    glGetUniformLocation(p: GLuint, name: *const c_char) -> GLint;
    glProgramUniform1i(p: GLuint, l: GLint, v: GLint);
    glProgramUniform1ui(p: GLuint, l: GLint, v: GLuint);
    glProgramUniform1f(p: GLuint, l: GLint, v: f32);
    glProgramUniform2i(p: GLuint, l: GLint, a: GLint, b: GLint);
    glProgramUniform2f(p: GLuint, l: GLint, a: f32, b: f32);
    glProgramUniform4f(p: GLuint, l: GLint, a: f32, b: f32, c: f32, d: f32);
    glCreateBuffers(n: GLsizei, out: *mut GLuint);
    glDeleteBuffers(n: GLsizei, b: *const GLuint);
    glNamedBufferStorage(b: GLuint, size: GLsizeiptr, data: *const c_void, flags: GLbitfield);
    glNamedBufferSubData(b: GLuint, off: GLintptr, size: GLsizeiptr, data: *const c_void);
    glGetNamedBufferSubData(b: GLuint, off: GLintptr, size: GLsizeiptr, data: *mut c_void);
    glClearNamedBufferData(b: GLuint, ifmt: GLenum, fmt: GLenum, kind: GLenum, data: *const c_void);
    glMapNamedBufferRange(b: GLuint, off: GLintptr, len: GLsizeiptr, access: GLbitfield) -> *mut c_void;
    glBindBufferBase(target: GLenum, index: GLuint, b: GLuint);
    glCreateTextures(target: GLenum, n: GLsizei, out: *mut GLuint);
    glDeleteTextures(n: GLsizei, t: *const GLuint);
    glTextureStorage2D(t: GLuint, levels: GLsizei, ifmt: GLenum, w: GLsizei, h: GLsizei);
    glTextureSubImage2D(t: GLuint, lvl: GLint, x: GLint, y: GLint, w: GLsizei, h: GLsizei, fmt: GLenum, kind: GLenum, px: *const c_void);
    glTextureParameteri(t: GLuint, p: GLenum, v: GLint);
    glGenerateTextureMipmap(t: GLuint);
    glGetTextureImage(t: GLuint, lvl: GLint, fmt: GLenum, kind: GLenum, size: GLsizei, px: *mut c_void);
    glBindTextureUnit(unit: GLuint, t: GLuint);
    glBindImageTexture(unit: GLuint, t: GLuint, lvl: GLint, layered: u8, layer: GLint, access: GLenum, fmt: GLenum);
    glDispatchCompute(x: GLuint, y: GLuint, z: GLuint);
    glMemoryBarrier(bits: GLbitfield);
    glCreateVertexArrays(n: GLsizei, out: *mut GLuint);
    glBindVertexArray(v: GLuint);
    glDrawArrays(mode: GLenum, first: GLint, n: GLsizei);
    glFenceSync(cond: GLenum, flags: GLbitfield) -> GLsync;
    glClientWaitSync(s: GLsync, flags: GLbitfield, timeout: u64) -> GLenum;
    glDeleteSync(s: GLsync);
    glCreateQueries(target: GLenum, n: GLsizei, out: *mut GLuint);
    glQueryCounter(q: GLuint, target: GLenum);
    glGetQueryObjectiv(q: GLuint, p: GLenum, out: *mut GLint);
    glGetQueryObjectui64v(q: GLuint, p: GLenum, out: *mut u64);
    glMaxShaderCompilerThreadsKHR(count: GLuint);
    glReadPixels(x: GLint, y: GLint, w: GLsizei, h: GLsizei, fmt: GLenum, kind: GLenum, px: *mut c_void);
    glProgramUniform1fv(p: GLuint, l: GLint, n: GLsizei, v: *const f32);
}

pub const GL_VENDOR: GLenum = 0x1F00;
pub const GL_RENDERER: GLenum = 0x1F01;
pub const GL_VERSION: GLenum = 0x1F02;
pub const GL_EXTENSIONS: GLenum = 0x1F03;
pub const GL_NUM_EXTENSIONS: GLenum = 0x821D;
pub const GL_COLOR_BUFFER_BIT: GLbitfield = 0x4000;
pub const GL_BLEND: GLenum = 0x0BE2;
pub const GL_SRC_ALPHA: GLenum = 0x0302;
pub const GL_ONE_MINUS_SRC_ALPHA: GLenum = 0x0303;
pub const GL_FRAMEBUFFER_SRGB: GLenum = 0x8DB9;
pub const GL_TRIANGLES: GLenum = 0x0004;
pub const GL_COMPUTE_SHADER: GLenum = 0x91B9;
pub const GL_VERTEX_SHADER: GLenum = 0x8B31;
pub const GL_FRAGMENT_SHADER: GLenum = 0x8B30;
pub const GL_COMPILE_STATUS: GLenum = 0x8B81;
pub const GL_LINK_STATUS: GLenum = 0x8B82;
pub const GL_INFO_LOG_LENGTH: GLenum = 0x8B84;
pub const GL_COMPLETION_STATUS_KHR: GLenum = 0x91B1;
pub const GL_SHADER_STORAGE_BUFFER: GLenum = 0x90D2;
pub const GL_UNIFORM_BUFFER: GLenum = 0x8A11;
pub const GL_DYNAMIC_STORAGE_BIT: GLbitfield = 0x0100;
pub const GL_MAP_READ_BIT: GLbitfield = 0x0001;
pub const GL_MAP_WRITE_BIT: GLbitfield = 0x0002;
pub const GL_MAP_PERSISTENT_BIT: GLbitfield = 0x0040;
pub const GL_MAP_COHERENT_BIT: GLbitfield = 0x0080;
pub const GL_CLIENT_STORAGE_BIT: GLbitfield = 0x0200;
pub const GL_R32UI: GLenum = 0x8236;
pub const GL_RED_INTEGER: GLenum = 0x8D94;
pub const GL_UNSIGNED_INT: GLenum = 0x1405;
pub const GL_UNSIGNED_BYTE: GLenum = 0x1401;
pub const GL_FLOAT: GLenum = 0x1406;
pub const GL_RGBA: GLenum = 0x1908;
pub const GL_RGBA8: GLenum = 0x8058;
pub const GL_RGBA16F: GLenum = 0x881A;
pub const GL_RGBA32F: GLenum = 0x8814;
pub const GL_TEXTURE_2D: GLenum = 0x0DE1;
pub const GL_TEXTURE_MIN_FILTER: GLenum = 0x2801;
pub const GL_TEXTURE_MAG_FILTER: GLenum = 0x2800;
pub const GL_TEXTURE_WRAP_S: GLenum = 0x2802;
pub const GL_TEXTURE_WRAP_T: GLenum = 0x2803;
pub const GL_CLAMP_TO_EDGE: GLint = 0x812F;
pub const GL_LINEAR: GLint = 0x2601;
pub const GL_NEAREST: GLint = 0x2600;
pub const GL_LINEAR_MIPMAP_LINEAR: GLint = 0x2703;
pub const GL_READ_ONLY: GLenum = 0x88B8;
pub const GL_WRITE_ONLY: GLenum = 0x88B9;
pub const GL_READ_WRITE: GLenum = 0x88BA;
pub const GL_SHADER_STORAGE_BARRIER_BIT: GLbitfield = 0x2000;
pub const GL_SHADER_IMAGE_ACCESS_BARRIER_BIT: GLbitfield = 0x0020;
pub const GL_TEXTURE_FETCH_BARRIER_BIT: GLbitfield = 0x0008;
pub const GL_TEXTURE_UPDATE_BARRIER_BIT: GLbitfield = 0x0100;
pub const GL_BUFFER_UPDATE_BARRIER_BIT: GLbitfield = 0x0200;
pub const GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT: GLbitfield = 0x4000;
pub const GL_ALL_BARRIER_BITS: GLbitfield = 0xFFFF_FFFF;
pub const GL_SYNC_GPU_COMMANDS_COMPLETE: GLenum = 0x9117;
pub const GL_SYNC_FLUSH_COMMANDS_BIT: GLbitfield = 0x1;
pub const GL_TIMEOUT_EXPIRED: GLenum = 0x911B;
pub const GL_WAIT_FAILED: GLenum = 0x911D;
pub const GL_TIMESTAMP: GLenum = 0x8E28;
pub const GL_QUERY_RESULT: GLenum = 0x8866;
pub const GL_QUERY_RESULT_AVAILABLE: GLenum = 0x8867;
pub const GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX: GLenum = 0x9049;
pub const GL_MAX_COMPUTE_SHARED_MEMORY_SIZE: GLenum = 0x8262;
pub const GL_OUT_OF_MEMORY: GLenum = 0x0505;

impl Gl {
    pub fn string(&self, name: GLenum) -> String {
        let p = unsafe { self.glGetString(name) };
        if p.is_null() { String::new() } else { unsafe { CStr::from_ptr(p as *const c_char) }.to_string_lossy().into_owned() }
    }

    pub fn has_extension(&self, ext: &str) -> bool {
        let mut n = 0;
        unsafe { self.glGetIntegerv(GL_NUM_EXTENSIONS, &mut n) };
        (0..n as u32).any(|i| {
            let p = unsafe { self.glGetStringi(GL_EXTENSIONS, i) };
            !p.is_null() && unsafe { CStr::from_ptr(p as *const c_char) }.to_bytes() == ext.as_bytes()
        })
    }

    /// Panics with the message if a GL error is pending (a bug, or out of video memory).
    pub fn check(&self, what: &str) {
        let e = unsafe { self.glGetError() };
        if e != 0 {
            panic!("OpenGL error 0x{e:04x} after {what}");
        }
    }

    pub fn shader(&self, kind: GLenum, name: &str, src: &str) -> Result<GLuint, String> {
        unsafe {
            let s = self.glCreateShader(kind);
            let c = CString::new(src).unwrap();
            let p = c.as_ptr();
            self.glShaderSource(s, 1, &p, std::ptr::null());
            self.glCompileShader(s);
            let mut ok = 0;
            self.glGetShaderiv(s, GL_COMPILE_STATUS, &mut ok);
            if ok == 0 {
                let log = self.shader_log(s);
                self.glDeleteShader(s);
                return Err(format!("{name}: {log}"));
            }
            Ok(s)
        }
    }

    fn shader_log(&self, s: GLuint) -> String {
        unsafe {
            let mut n = 0;
            self.glGetShaderiv(s, GL_INFO_LOG_LENGTH, &mut n);
            let mut buf = vec![0u8; n.max(1) as usize];
            self.glGetShaderInfoLog(s, n, std::ptr::null_mut(), buf.as_mut_ptr() as *mut c_char);
            String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string()
        }
    }

    pub fn program(&self, name: &str, shaders: &[(GLenum, &str)]) -> Result<GLuint, String> {
        unsafe {
            let p = self.glCreateProgram();
            let mut ids = Vec::new();
            for &(kind, src) in shaders {
                let s = self.shader(kind, name, src)?;
                self.glAttachShader(p, s);
                ids.push(s);
            }
            self.glLinkProgram(p);
            for s in ids {
                self.glDeleteShader(s);
            }
            let mut ok = 0;
            self.glGetProgramiv(p, GL_LINK_STATUS, &mut ok);
            if ok == 0 {
                let mut n = 0;
                self.glGetProgramiv(p, GL_INFO_LOG_LENGTH, &mut n);
                let mut buf = vec![0u8; n.max(1) as usize];
                self.glGetProgramInfoLog(p, n, std::ptr::null_mut(), buf.as_mut_ptr() as *mut c_char);
                return Err(format!("{name}: link: {}", String::from_utf8_lossy(&buf).trim_end_matches('\0')));
            }
            Ok(p)
        }
    }

    /// Starts compiling and linking a compute program without waiting
    /// (KHR_parallel_shader_compile); `ready` and `finish_program` follow up.
    pub fn program_async(&self, src: &str) -> (GLuint, GLuint) {
        unsafe {
            let s = self.glCreateShader(GL_COMPUTE_SHADER);
            let c = CString::new(src).unwrap();
            let ptr = c.as_ptr();
            self.glShaderSource(s, 1, &ptr, std::ptr::null());
            self.glCompileShader(s);
            let p = self.glCreateProgram();
            self.glAttachShader(p, s);
            self.glLinkProgram(p);
            (p, s)
        }
    }

    /// Whether an async program has finished compiling and linking.
    pub fn ready(&self, p: GLuint) -> bool {
        let mut done = 0;
        unsafe { self.glGetProgramiv(p, GL_COMPLETION_STATUS_KHR, &mut done) };
        done != 0
    }

    /// The linked program, or the compile / link log.
    pub fn finish_program(&self, p: GLuint, s: GLuint) -> Result<GLuint, String> {
        unsafe {
            let mut ok = 0;
            self.glGetProgramiv(p, GL_LINK_STATUS, &mut ok);
            if ok != 0 {
                self.glDeleteShader(s);
                return Ok(p);
            }
            let log = self.shader_log(s);
            self.glDeleteShader(s);
            self.glDeleteProgram(p);
            Err(log)
        }
    }

    pub fn uniform(&self, p: GLuint, name: &str) -> GLint {
        let c = CString::new(name).unwrap();
        unsafe { self.glGetUniformLocation(p, c.as_ptr()) }
    }
}
