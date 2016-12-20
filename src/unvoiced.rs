//! Unvoiced spectrum synthesis.
//!
//! At a high level, the unvoiced signal is generated with these steps:
//!
//! 1. Construct the frequency-domain representation of a white noise signal using the
//!    Discrete Fourier Transform (DFT).
//! 2. Bandpass the spectrum to contain only the frequencies marked as unvoiced in the
//!    current frame.
//! 3. Perform an Inverse Discrete Fourier Transform (IDFT) on this spectrum to produce a
//!    white noise signal containing only the unvoiced frequency content.
//!
//! Rather than performing both DFT and IDFT operations, which are relatively expensive
//! and were found to be a bottleneck, this implementation computes an equivalent result
//! using only a partial IDFT.
//!
//! ## DFT of Noise
//!
//! [Under certain circumstances][dft-noise] the DFT of a noise signal can have its points
//! sampled from a probability distribution, rather than each computed with a O(N)
//! procedure.
//!
//! According to the standard [p58], the DFT is generated from a windowed white noise
//! signal u(n). The standard specifies that the signal can have arbitrary mean but doesn't
//! specify if there are any constraints on the variance or if the signal must be real or
//! complex (although the given example u(n) is real with sample mean μ<sub>x</sub> ≈
//! 26562 and variance σ<sub>x</sub><sup>2</sup> ≈ 235198690.)
//!
//! This implementation assumes the white noise signal is real with samples pulled from a
//! Gaussian distribution having sample mean μ<sub>x</sub> = 0 and sample variance
//! σ<sub>x</sub><sup>2</sup> = 1, i.e., u(n) ~ *N*(0, 1). The resulting DFT of this
//! signal then has points with real and complex parts that are sampled from a Gaussian
//! distribution with mean μ = 0 and variance σ<sup>2</sup> =
//! E<sub>w</sub>σ<sub>x</sub><sup>2</sup> / 2 = E<sub>w</sub> / 2, i.e.,
//! Re[U<sub>w</sub>(m)], Im[U<sub>w</sub>(m)] ~ *N*(0, E<sub>w</sub> / 2), where
//! E<sub>w</sub> is the energy of the speech synthesis window w<sub>s</sub>(n).
//!
//! Note that [the given source][dft-noise] only derives this result for a complex signal
//! with equal real and imaginary sample variances, but empirical evaluations show that
//! the result is the same with a real signal.
//!
//! [dft-noise]: http://users.ece.gatech.edu/mrichard/DFT%20of%20Noise.pdf
//!
//! ## DFT Symmetry
//!
//! According to Eq 125, the IDFT is defined as
//!
//! > u<sub>w</sub>(n) = [U<sub>w</sub>(-128) exp(*j* 2π(-128)n/256) +
//! >           U<sub>w</sub>(-127) exp(*j* 2π(-127)n/256) + ··· +
//! >           U<sub>w</sub>(126) exp(*j* π(126)n/256) +
//! >           U<sub>w</sub>(127) exp(*j* 2π(127)n/256)] / 256
//!
//! Since u(n) is a real signal, U<sub>w</sub>(m) = U<sub>w</sub>(-m)<sup>\*</sup> (i.e.,
//! the complex conjugate), and the DFT magnitude is symmetric
//! around DC: for all m, 0 ≤ m ≤ 127,
//!
//! > U<sub>w</sub>(m) exp(*j* mφ) + U<sub>w</sub>(-m) exp(*-j* mφ) =
//! >
//! > (a + *j* b)(cos mφ + *j* sin mφ) + (a - *j* b)(cos mφ - sin mφ) =
//! >
//! > 2a cos mφ - 2b sin mφ = 2 Re[U(m)] cos mφ - 2 Im[U(m)] sin mφ
//!
//! Additionally, the definition of a<sub>l</sub> and b<sub>l</sub> in Eqs 122 and 123
//! guarantees that for all L and ω<sub>0</sub> parameters, a<sub>1</sub> ≥ 2 and
//! b<sub>L</sub> ≤ 125. So according to Eq 124, every frame has at least
//! U<sub>w</sub>(-128) = U<sub>w</sub>(0) = 0.
//!
//! Using these results, it can be seen that the sum for u<sub>w</sub>(n) can be
//! "simplified" to
//!
//! > u<sub>w</sub>(n) =
//! >
//! > [0 + U<sub>w</sub>(-127) exp(*j* 2π(-127)n/256) + ··· +
//! >   U<sub>w</sub>(-1) exp(*j* 2π(-1)n/256) + 0 +
//! >   U<sub>w</sub>(1) exp(*j* π(1)n/256) + ··· +
//! >   U<sub>w</sub>(127) exp(*j* 2π(127)n/256)] / 256 =
//! >
//! > 2 [Re[U(0)] cos(2π(0)n/256) - Im[U(0)] sin(2π(0)n/256) +
//! >   Re[U(1)] cos(2π(1)n/256) - Im[U(1)] sin(2π(1)n/256) + ··· +
//! >   Re[U(127)] cos(2π(127)n/256) - Im[U(127)] sin(2π(127)n/256)] / 256
//!
//! which requires half as many U<sub>w</sub>(m) values and performs no complex
//! arithmetic.

use std::f32::consts::PI;

use map_in_place::MapInPlace;
use num::complex::Complex32;
use num::traits::Zero;
use quad_osc::QuadOsc;
use rand::distributions::IndependentSample;
use rand::distributions::normal::Normal;
use rand::Rng;

use consts::SAMPLES_PER_FRAME;
use descramble::VoiceDecisions;
use enhance::EnhancedSpectrals;
use params::BaseParams;
use window;

/// Unvoiced scaling coefficient γ<sub>w</sub> computed from Eq 121.
const SCALING_COEF: f32 = 146.6432708443356;

/// Number of points in the generated discrete Fourier transform.
const DFT_SIZE: usize = 256;
/// Number of points in the generated inverse DFT.
const IDFT_SIZE: usize = 256;

/// Number of points in real half of DFT.
const DFT_HALF: usize = DFT_SIZE / 2;
/// Number of points in real half of IDFT.
const IDFT_HALF: usize = IDFT_SIZE / 2;

/// Constructs unvoiced DFT/IDFT.
pub struct UnvoicedDFT([Complex32; DFT_HALF]);

impl UnvoicedDFT {
    /// Construct a new `UnvoicedDFT` from the given frame parameters and noise generator.
    pub fn new<R: Rng>(params: &BaseParams, voice: &VoiceDecisions,
                       enhanced: &EnhancedSpectrals, mut rng: R)
        -> Self
    {
        // DFT values default to 0 according to Eqs 119 and 124.
        let mut dft = [Complex32::default(); DFT_HALF];

        // Create a Gaussian distribution with mean μ = 0 and variance σ^2 = E_w / 2.
        let gaus = Normal::new(0.0, (window::ENERGY_SYNTHESIS / 2.0).sqrt() as f64);

        for (l, &amplitude) in enhanced.iter().enumerate() {
            let l = l + 1;

            if voice.is_voiced(l) {
                continue;
            }

            // Compute the lower and upper frequency bands for the current harmonic.
            let (lower, upper) = edges(l, params);

            // Populate the current band with random spectrum.
            for m in lower..upper {
                dft[m] = Complex32::new(gaus.ind_sample(&mut rng) as f32,
                                        gaus.ind_sample(&mut rng) as f32);
            }

            // Compute energy of current band according to Eq 120.
            let energy = (lower..upper)
                .map(|m| dft[m].norm_sqr())
                .fold(0.0, |s, x| s + x);
            // Compute power of current band according to Eq 120.
            let power = energy / (upper - lower) as f32;
            // Compute scale for current enhanced spectral amplitude according to Eq 120.
            let scale = SCALING_COEF * amplitude / power.sqrt();

            // Scale the band according to Eq 120.
            (&mut dft[lower..upper]).map_in_place(|&x| scale * x);
        }

        UnvoicedDFT(dft)
    }

    /// Compute the IDFT u<sub>w</sub>(n) at the given point n.
    pub fn idft(&self, n: isize) -> f32 {
        // The IDFT is zero outside the defined range [p59].
        if n < -(IDFT_HALF as isize) || n >= IDFT_HALF as isize {
            return 0.0;
        }

        let mut osc = QuadOsc::new(0.0, 2.0 / IDFT_SIZE as f32 * PI * n as f32);

        2.0 / IDFT_SIZE as f32 * self.0.iter().map(|x| {
            let (sin, cos) = osc.next();
            x.re * cos - x.im * sin
        }).fold(0.0, |s, x| s + x)
    }
}

impl Default for UnvoicedDFT {
    /// Create a new `UnvoicedDFT` in the default state.
    fn default() -> Self {
        // By default all IDFT values are zero [p64]. Setting the DFT values to zero will
        // derive this effect.
        UnvoicedDFT([Complex32::zero(); DFT_HALF])
    }
}

/// Synthesizes unvoiced spectrum signal s<sub>uv</sub>(n).
pub struct Unvoiced<'a, 'b> {
    /// Unvoiced DFT/IDFT for current frame.
    cur: &'a UnvoicedDFT,
    /// Unvoiced DFT/IDFT for previous frame.
    prev: &'b UnvoicedDFT,
    /// Synthesis window w<sub>s</sub>(n) for "weighted overlap add".
    window: window::Window,
}

impl<'a, 'b> Unvoiced<'a, 'b> {
    /// Create a new `Unvoiced` from the given unvoiced spectrums of the current and
    /// previous frames.
    pub fn new(cur: &'a UnvoicedDFT, prev: &'b UnvoicedDFT) -> Self {
        Unvoiced {
            cur: cur,
            prev: prev,
            window: window::synthesis(),
        }
    }

    /// Compute the unvoiced signal sample s<sub>uv</sub>(n) for the given n, 0 ≤ n < N.
    pub fn get(&self, n: usize) -> f32 {
        debug_assert!(n < SAMPLES_PER_FRAME);

        let n = n as isize;

        // Compute numerator in Eq 126.
        let numer = self.window.get(n) * self.prev.idft(n) +
            self.window.get(n - SAMPLES_PER_FRAME as isize) *
                self.cur.idft(n - SAMPLES_PER_FRAME as isize);

        // Compute denominator in Eq 126.
        let denom = self.window.get(n).powi(2) +
            self.window.get(n - SAMPLES_PER_FRAME as isize).powi(2);

        // Compute Eq 126.
        numer / denom
    }
}

/// Determine the lower and upper band edges (a<sub>l</sub>, b<sub>l</sub>) for the given
/// harmonic of the fundamental frequency.
fn edges(l: usize, params: &BaseParams) -> (usize, usize) {
    let common = DFT_SIZE as f32 / (2.0 * PI) * params.fundamental;

    (
        // Compute Eq 122.
        (common * (l as f32 - 0.5)).ceil() as usize,
        // Compute Eq 123.
        (common * (l as f32 + 0.5)).ceil() as usize,
    )
}

#[cfg(test)]
mod test {
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
