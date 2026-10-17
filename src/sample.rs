//! One instant of a flame, flattened for the chaos game: what the GPU's
//! compute shader reads (an std430 struct, `Sample` in iterate.comp) and what
//! the CPU workers turn into lane tables. A frame carries several of them,
//! spread over the shutter time, for motion blur.

/// Xforms per flame, not counting the final one.
pub const MAXX: usize = 16;
/// Variations per xform (a random xform has at most four; a blend of two
/// xforms can have up to eight).
pub const NV: usize = 8;
/// Entries per xform-choice table (flam3's CHOOSE_XFORM_GRAIN is 16384).
pub const PICKN: usize = 1024;
/// Time samples per frame (motion blur).
pub const MAX_SAMPLES: usize = 64;

/// One xform (std430 `Xf`, 384 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct XfGpu {
    /// flam3's coefficients c00 c01 c10 c11 c20 c21 (x' = c00 x + c10 y + c20,
    /// y' = c01 x + c11 y + c21), then the colour index and colour speed
    pub c: [f32; 8],
    /// the post transform's six, then opacity and 1 if there is a post transform
    pub post: [f32; 8],
    /// variation ids (-1: unused, and the rest are too), weights and eight
    /// parameters each
    pub vid: [i32; NV],
    pub vw: [f32; NV],
    pub vp: [[f32; 8]; NV],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Sample {
    /// world to histogram pixels: px = cam[0] x + cam[1] y + cam[2],
    /// py = cam[4] x + cam[5] y + cam[6]
    pub cam: [f32; 8],
    /// the 3D camera, applied before `cam` (Fractorium's projection, turned
    /// about the view's centre): m00 m10 cx cy, m01 m11 m21 zpos, m02 m12 m22
    /// perspective, dof blur (0.1 cam_dof), 1 if it projects, unused x2
    pub proj: [f32; 16],
    /// xforms, 1 if there is a final xform, 1 if the choice depends on the
    /// previous xform (xaos), unused
    pub info: [i32; 4],
    /// MAXX xforms, then the final one
    pub xf: [XfGpu; MAXX + 1],
    /// per previous xform (row 0 only without xaos), PICKN xform indices
    pub pick: [u8; MAXX * PICKN],
    /// the palette, RGBA8
    pub pal: [u32; 256],
}

impl Default for Sample {
    fn default() -> Sample {
        Sample {
            cam: [0.0; 8],
            proj: [0.0; 16],
            info: [0; 4],
            xf: [XfGpu::default(); MAXX + 1],
            pick: [0; MAXX * PICKN],
            pal: [0; 256],
        }
    }
}

const _: () = assert!(core::mem::size_of::<XfGpu>() == 384);
const _: () = assert!(core::mem::size_of::<Sample>().is_multiple_of(16));

impl Sample {
    pub fn as_bytes(s: &[Sample]) -> &[u8] {
        unsafe { core::slice::from_raw_parts(s.as_ptr() as *const u8, core::mem::size_of_val(s)) }
    }
}

/// In a program key: the 3D variant (walker z, the 3D camera, direct colour).
pub const THREE_D: i32 = 1 << 16;

/// The variations the samples use (for a shader specialised to them), and
/// THREE_D if they need the 3D variant.
pub fn variation_ids(samples: &[Sample]) -> Vec<i32> {
    let mut ids = Vec::new();
    if samples.iter().any(|s| s.proj[13] != 0.0) {
        ids.push(THREE_D);
    }
    for s in samples {
        let n = (s.info[0] as usize).min(MAXX);
        let fin = if s.info[1] != 0 { Some(MAXX) } else { None };
        for k in (0..n).chain(fin) {
            for &id in s.xf[k].vid.iter().take_while(|&&id| id >= 0) {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
    }
    if !ids.contains(&THREE_D) && ids.iter().any(|&id| crate::var::needs_3d(id)) {
        ids.push(THREE_D);
    }
    ids.sort();
    ids
}
