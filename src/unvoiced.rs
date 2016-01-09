use std;
use std::f32::consts::PI;

use arrayvec::ArrayVec;
use num::complex::Complex32;
use collect_slice::CollectSlice;

use consts::SAMPLES;
use descramble::VoiceDecisions;
use enhance::EnhancedSpectrals;
use noise::Noise;
use params::BaseParams;
use window;

// Result of equation 121
const SCALING_COEF: f32 = 146.6432708443356;

pub struct UnvoicedDFT {
    dfts: ArrayVec<[Complex32; 256]>,
    scale: f32,
}

impl UnvoicedDFT {
    pub fn new(lower: isize, upper: isize, noise: &Noise, window: &window::Window)
        -> UnvoicedDFT
    {
        let dfts = (lower..upper).map(|m| {
            let mut nclone = noise.clone();

            (-104..105).map(|n| {
                nclone.next() * window.get(n) *
                    Complex32::new(0.0, -2.0 / 256.0 * PI * m as f32 * n as f32).exp()
            }).fold(Complex32::new(0.0, 0.0), |s, x| s + x)
        }).collect::<ArrayVec<[Complex32; 256]>>();

        let sum = dfts.iter().map(|dft| dft.norm_sqr()).fold(0.0, |s, x| s + x);

        UnvoicedDFT {
            dfts: dfts,
            scale: SCALING_COEF / (sum / (upper - lower) as f32).sqrt(),
        }
    }

    pub fn scale(&self, spectral: f32) -> f32 { self.scale * spectral }
}

impl std::ops::Deref for UnvoicedDFT {
    type Target = ArrayVec<[Complex32; 256]>;
    fn deref(&self) -> &Self::Target { &self.dfts }
}

fn edges(l: usize, params: &BaseParams) -> (isize, isize) {
    let edge = |inner: f32| {
        256.0 / (2.0 * PI) * inner * params.fundamental
    };

    (
        edge(l as f32 - 0.5).ceil() as isize,
        edge(l as f32 + 0.5).ceil() as isize,
    )
}

pub struct UnvoicedParts([Complex32; 256]);

impl UnvoicedParts {
    pub fn new(params: &BaseParams, voice: &VoiceDecisions, enhanced: &EnhancedSpectrals)
        -> UnvoicedParts
    {
        let mut parts = [Complex32::new(0.0, 0.0); 256];

        let noise = Noise::new();
        let window = window::synthesis_trunc();

        for (l, &m) in enhanced.iter().enumerate() {
            let l = l + 1;

            if voice.is_voiced(l) {
                continue;
            }

            let (lower, upper) = edges(l, params);

            let pos = UnvoicedDFT::new(lower, upper, &noise, &window);
            let pscale = pos.scale(m);

            let neg = UnvoicedDFT::new(-upper+1, -lower+1, &noise, &window);
            let nscale = neg.scale(m);

            pos.iter()
               .map(|&dft| pscale * dft)
               .collect_slice(&mut parts[128 + lower as usize..]);

            neg.iter()
               .map(|&dft| nscale * dft)
               .collect_slice(&mut parts[129 - upper as usize..]);
        }

        UnvoicedParts(parts)
    }

    // -128 <= m < 128
    fn get(&self, m: isize) -> Complex32 { self.0[(m + 128) as usize] }

    pub fn idft(&self, n: isize) -> f32 {
        if n < -128 || n > 127 {
            return 0.0;
        }

        (-128..128).map(|m| {
            self.get(m) *
                Complex32::new(0.0, 2.0 / 256.0 * PI * m as f32 * n as f32).exp()
        }).fold(Complex32::new(0.0, 0.0), |s, x| s + x).re / 256.0
    }
}

impl Default for UnvoicedParts {
    fn default() -> UnvoicedParts {
        UnvoicedParts([Complex32::new(0.0, 0.0); 256])
    }
}

pub struct Unvoiced<'a, 'b> {
    cur: &'a UnvoicedParts,
    prev: &'b UnvoicedParts,
    window: window::Window,
}

impl<'a, 'b> Unvoiced<'a, 'b> {
    pub fn new(cur: &'a UnvoicedParts, prev: &'b UnvoicedParts) -> Unvoiced<'a, 'b> {
        Unvoiced {
            cur: cur,
            prev: prev,
            window: window::synthesis_full(),
        }
    }

    pub fn get(&self, n: usize) -> f32 {
        let n = n as isize;

        let numer = self.window.get(n) * self.prev.idft(n) +
            self.window.get(n - SAMPLES as isize) * self.cur.idft(n - SAMPLES as isize);
        let denom = self.window.get(n).powi(2) +
            self.window.get(n - SAMPLES as isize).powi(2);

        numer / denom
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use super::edges;
    use descramble::Bootstrap;
    use params::BaseParams;

    #[test]
    fn test_edges() {
        let chunks = [
            0b001000010010,
            0b110011001100,
            0b111000111000,
            0b111111111111,
            0b10100110101,
            0b00101111010,
            0b01110111011,
            0b00001000,
        ];

        let b = Bootstrap::new(&chunks);
        let p = BaseParams::new(b.unwrap_period());

        assert!((p.fundamental - 0.17575344).abs() < 0.000001);

        let (lower, upper) = edges(1, &p);
        assert_eq!(lower, 4);
        assert_eq!(upper, 11);

        let (lower, upper) = edges(2, &p);
        assert_eq!(lower, 11);
        assert_eq!(upper, 18);

        let (lower, upper) = edges(5, &p);
        assert_eq!(lower, 33);
        assert_eq!(upper, 40);

        let (lower, upper) = edges(8, &p);
        assert_eq!(lower, 54);
        assert_eq!(upper, 61);

        let (lower, upper) = edges(16, &p);
        assert_eq!(lower, 111);
        assert_eq!(upper, 119);
    }
}
