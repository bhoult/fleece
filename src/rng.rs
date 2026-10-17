//! xorshift32 for the genome generator (FRAKTAL256's rng.rs), seeded through
//! a hash so neighbouring seeds give unrelated streams.

#[derive(Clone, Copy, Debug)]
pub struct Rng(u32);

pub fn mix(a: u32, b: u32) -> u32 {
    let mut h = a ^ b.wrapping_mul(0x9E37_79B9);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^ (h >> 16)
}

impl Rng {
    pub fn new(seed: u32) -> Rng {
        Rng(mix(seed, 0x2545_F491) | 1)
    }
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    /// [0, 1)
    pub fn f01(&mut self) -> f32 {
        f32::from_bits((self.next_u32() >> 9) | 0x3F80_0000) - 1.0
    }
    /// [-1, 1)
    pub fn signed(&mut self) -> f32 {
        self.f01() * 2.0 - 1.0
    }
    /// [0, n)
    pub fn below(&mut self, n: u32) -> u32 {
        ((self.next_u32() as u64 * n as u64) >> 32) as u32
    }
    pub fn bit(&mut self) -> bool {
        self.next_u32() & 0x8000_0000 != 0
    }
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f01()
    }
    /// An entry of `t` (flam3's random_distrib).
    pub fn pick<T: Copy>(&mut self, t: &[T]) -> T {
        t[self.below(t.len() as u32) as usize]
    }
}
