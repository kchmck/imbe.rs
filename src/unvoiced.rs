use std::f32::consts::PI;

use num::complex::Complex32;
use num::traits::Zero;

use consts::SAMPLES;
use descramble::VoiceDecisions;
use enhance::EnhancedSpectrals;
use noise::Noise;
use params::BaseParams;
use window;

// Result of equation 121
const SCALING_COEF: f32 = 146.6432708443356;

const DFT_SIZE: usize = 256;
const IDFT_SIZE: usize = 256;

const DFT_HALF: usize = DFT_SIZE / 2;
const IDFT_HALF: usize = IDFT_SIZE / 2;

fn edges(l: usize, params: &BaseParams) -> (usize, usize) {
    let common = DFT_SIZE as f32 / (2.0 * PI) * params.fundamental;

    (
        (common * (l as f32 - 0.5)).ceil() as usize,
        (common * (l as f32 + 0.5)).ceil() as usize,
    )
}

pub struct UnvoicedDFT([Complex32; DFT_HALF]);

impl UnvoicedDFT {
    pub fn new(params: &BaseParams, voice: &VoiceDecisions, enhanced: &EnhancedSpectrals)
        -> UnvoicedDFT
    {
        let mut dft = [Complex32::zero(); DFT_HALF];

        let noise = Noise::new();
        let window = window::synthesis_trunc();

        for (l, &spectral) in enhanced.iter().enumerate() {
            let l = l + 1;

            if voice.is_voiced(l) {
                continue;
            }

            let (lower, upper) = edges(l, params);

            for m in lower..upper {
                let mut nclone = noise.clone();

                dft[m] = (-104..105).map(|n| {
                    let inner = 2.0 / DFT_SIZE as f32 * PI * m as f32 * n as f32;
                    let (sin, cos) = inner.sin_cos();

                    nclone.next() * window.get(n) * Complex32::new(cos, -sin)
                }).fold(Complex32::zero(), |s, x| s + x)
            }

            let energy = (lower..upper)
                .map(|m| dft[m].norm_sqr())
                .fold(0.0, |s, x| s + x);
            let power = energy / (upper - lower) as f32;

            let scale = SCALING_COEF * spectral / power.sqrt();

            for m in lower..upper {
                dft[m] = scale * dft[m];
            }
        }

        UnvoicedDFT(dft)
    }

    pub fn idft(&self, n: isize) -> f32 {
        if n < -(IDFT_HALF as isize) || n >= IDFT_HALF as isize {
            return 0.0;
        }

        2.0 / (DFT_SIZE as f32 * IDFT_SIZE as f32).sqrt() * (0..DFT_HALF).map(|m| {
            let (sin, cos) = (2.0 / IDFT_SIZE as f32 * PI * m as f32 * n as f32).sin_cos();
            self.0[m].re * cos - self.0[m].im * sin
        }).fold(0.0, |s, x| s + x)
    }
}

impl Default for UnvoicedDFT {
    fn default() -> UnvoicedDFT {
        UnvoicedDFT([Complex32::zero(); DFT_HALF])
    }
}

pub struct Unvoiced<'a, 'b> {
    cur: &'a UnvoicedDFT,
    prev: &'b UnvoicedDFT,
    window: window::Window,
}

impl<'a, 'b> Unvoiced<'a, 'b> {
    pub fn new(cur: &'a UnvoicedDFT, prev: &'b UnvoicedDFT) -> Unvoiced<'a, 'b> {
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
