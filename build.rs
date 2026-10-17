// libSDL2 is the only library linked besides libc: it opens the window and
// the OpenGL context (GLX, or EGL for the headless modes), and every GL entry
// point is looked up through SDL_GL_GetProcAddress.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SDLLIB");
    let sdl = std::env::var("SDLLIB").unwrap_or_else(|_| "/usr/lib/x86_64-linux-gnu/libSDL2-2.0.so.0".into());
    println!("cargo:rustc-link-arg-bin=fleece={sdl}");
}
