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
    use super::*;
    use descramble::{VoiceDecisions, Bootstrap};
    use params::BaseParams;
    use num::complex::Complex32;
    use rand::XorShiftRng;

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

    #[test]
    fn test_dft() {
        // Verify results computed with standalone python script.

        let p = BaseParams::new(42);
        assert_eq!(p.fundamental, 0.1541886);
        assert_eq!(p.harmonics, 18);
        assert_eq!(p.bands, 6);

        let mut voice = VoiceDecisions::new(0b101001, &p);
        voice.force_voiced(5);
        voice.force_voiced(13);
        voice.force_voiced(14);
        assert!(voice.is_voiced(1));
        assert!(voice.is_voiced(2));
        assert!(voice.is_voiced(3));
        assert!(!voice.is_voiced(4));
        assert!(voice.is_voiced(5));
        assert!(!voice.is_voiced(6));
        assert!(voice.is_voiced(7));
        assert!(voice.is_voiced(8));
        assert!(voice.is_voiced(9));
        assert!(!voice.is_voiced(10));
        assert!(!voice.is_voiced(11));
        assert!(!voice.is_voiced(12));
        assert!(voice.is_voiced(13));
        assert!(voice.is_voiced(14));
        assert!(!voice.is_voiced(15));
        assert!(voice.is_voiced(16));
        assert!(voice.is_voiced(17));
        assert!(voice.is_voiced(18));

        let mut amps = EnhancedSpectrals::default();
        amps.push(2.0);
        amps.push(1.0);
        amps.push(4.0);
        amps.push(6.0);
        amps.push(42.0);
        amps.push(8.0);
        amps.push(1.5);
        amps.push(0.5);
        amps.push(24.0);
        amps.push(32.0);
        amps.push(3.0);
        amps.push(7.0);
        amps.push(13.0);
        amps.push(5.0);
        amps.push(4.2);
        amps.push(11.0);
        amps.push(9.0);
        amps.push(18.0);

        let dft = UnvoicedDFT::new(&p, &voice, &amps, XorShiftRng::new_unseeded());

        assert_eq!(dft.0[0], Complex32::zero());
        assert_eq!(dft.0[1], Complex32::zero());
        assert_eq!(dft.0[2], Complex32::zero());
        assert_eq!(dft.0[3], Complex32::zero());
        assert_eq!(dft.0[4], Complex32::zero());
        assert_eq!(dft.0[5], Complex32::zero());
        assert_eq!(dft.0[6], Complex32::zero());
        assert_eq!(dft.0[7], Complex32::zero());
        assert_eq!(dft.0[8], Complex32::zero());
        assert_eq!(dft.0[9], Complex32::zero());
        assert_eq!(dft.0[10], Complex32::zero());
        assert_eq!(dft.0[11], Complex32::zero());
        assert_eq!(dft.0[12], Complex32::zero());
        assert_eq!(dft.0[13], Complex32::zero());
        assert_eq!(dft.0[14], Complex32::zero());
        assert_eq!(dft.0[15], Complex32::zero());
        assert_eq!(dft.0[16], Complex32::zero());
        assert_eq!(dft.0[17], Complex32::zero());
        assert_eq!(dft.0[18], Complex32::zero());
        assert_eq!(dft.0[19], Complex32::zero());
        assert_eq!(dft.0[20], Complex32::zero());
        assert_eq!(dft.0[21], Complex32::zero());
        assert!((dft.0[22].re - 703.42907837950860994169488549232483).abs() < 1e-4);
        assert!((dft.0[23].re - -173.75184149287974832986947149038315).abs() < 1e-4);
        assert!((dft.0[24].re - 884.56073381600754146347753703594208).abs() < 1e-4);
        assert!((dft.0[25].re - 359.56583225630549804918700829148293).abs() < 1e-4);
        assert!((dft.0[26].re - 419.68425681671448046472505666315556).abs() < 1e-4);
        assert!((dft.0[27].re - -313.48020621968555587955052033066750).abs() < 1e-4);
        assert!((dft.0[28].re - -80.15363595778023864113492891192436).abs() < 1e-4);
        assert!((dft.0[22].im - 487.86348415050247240287717431783676).abs() < 1e-4);
        assert!((dft.0[23].im - -612.26361186515930512541672214865685).abs() < 1e-4);
        assert!((dft.0[24].im - 556.24827162778478850668761879205704).abs() < 1e-4);
        assert!((dft.0[25].im - -341.51983854534915963085950352251530).abs() < 1e-4);
        assert!((dft.0[26].im - 753.90749652367856015189317986369133).abs() < 1e-4);
        assert!((dft.0[27].im - 181.36998232764872795996780041605234).abs() < 1e-4);
        assert!((dft.0[28].im - -1435.72522260304094743332825601100922).abs() < 1e-3);
        assert_eq!(dft.0[29], Complex32::zero());
        assert_eq!(dft.0[30], Complex32::zero());
        assert_eq!(dft.0[31], Complex32::zero());
        assert_eq!(dft.0[32], Complex32::zero());
        assert_eq!(dft.0[33], Complex32::zero());
        assert_eq!(dft.0[34], Complex32::zero());
        assert!((dft.0[35].re - -627.73957694347848246252397075295448).abs() < 1e-4);
        assert!((dft.0[36].re - 224.68153342605762645689537748694420).abs() < 1e-4);
        assert!((dft.0[37].re - -83.87732477554284571397147374227643).abs() < 1e-4);
        assert!((dft.0[38].re - 438.92579701409601966588525101542473).abs() < 1e-4);
        assert!((dft.0[39].re - -142.51808002534366437430435325950384).abs() < 1e-4);
        assert!((dft.0[40].re - 504.35744820956358580588130280375481).abs() < 1e-4);
        assert!((dft.0[35].im - 2240.90859048500396966119296848773956).abs() < 1e-3);
        assert!((dft.0[36].im - 552.52898268735350484348600730299950).abs() < 1e-4);
        assert!((dft.0[37].im - 1317.61451609264986473135650157928467).abs() < 1e-4);
        assert!((dft.0[38].im - 263.53759853221021103308885358273983).abs() < 1e-4);
        assert!((dft.0[39].im - -3.79034320072740271712063986342400).abs() < 1e-4);
        assert!((dft.0[40].im - -454.07011653497653469457873143255711).abs() < 1e-4);
        assert_eq!(dft.0[41], Complex32::zero());
        assert_eq!(dft.0[42], Complex32::zero());
        assert_eq!(dft.0[43], Complex32::zero());
        assert_eq!(dft.0[44], Complex32::zero());
        assert_eq!(dft.0[45], Complex32::zero());
        assert_eq!(dft.0[46], Complex32::zero());
        assert_eq!(dft.0[47], Complex32::zero());
        assert_eq!(dft.0[48], Complex32::zero());
        assert_eq!(dft.0[49], Complex32::zero());
        assert_eq!(dft.0[50], Complex32::zero());
        assert_eq!(dft.0[51], Complex32::zero());
        assert_eq!(dft.0[52], Complex32::zero());
        assert_eq!(dft.0[53], Complex32::zero());
        assert_eq!(dft.0[54], Complex32::zero());
        assert_eq!(dft.0[55], Complex32::zero());
        assert_eq!(dft.0[56], Complex32::zero());
        assert_eq!(dft.0[57], Complex32::zero());
        assert_eq!(dft.0[58], Complex32::zero());
        assert_eq!(dft.0[59], Complex32::zero());
        assert!((dft.0[60].re - -807.27968274095837841741740703582764).abs() < 1e-4);
        assert!((dft.0[61].re - -184.04927898596756108418048825114965).abs() < 1e-4);
        assert!((dft.0[62].re - 1231.35904130671747225278522819280624).abs() < 1e-4);
        assert!((dft.0[63].re - -2152.75908030796881575952284038066864).abs() < 1e-4);
        assert!((dft.0[64].re - 463.18344138081107530524604953825474).abs() < 1e-4);
        assert!((dft.0[65].re - 872.27138323153656074282480403780937).abs() < 1e-4);
        assert!((dft.0[60].im - 341.97000614873280710526159964501858).abs() < 1e-4);
        assert!((dft.0[61].im - -2480.24168770538062744890339672565460).abs() < 1e-3);
        assert!((dft.0[62].im - -8670.05218478608367149718105792999268).abs() < 1e-4);
        assert!((dft.0[63].im - -2928.43248922020438840263523161411285).abs() < 1e-4);
        assert!((dft.0[64].im - -1236.39650873955588394892401993274689).abs() < 1e-4);
        assert!((dft.0[65].im - 5724.31455587039272359106689691543579).abs() < 1e-3);
        assert!((dft.0[66].re - 191.02411659862829651501670014113188).abs() < 1e-4);
        assert!((dft.0[67].re - -6.74512550788298792525665703578852).abs() < 1e-4);
        assert!((dft.0[68].re - -35.38223977789849783448516973294318).abs() < 1e-4);
        assert!((dft.0[69].re - 70.29753459358614975371892796829343).abs() < 1e-4);
        assert!((dft.0[70].re - 561.29313016445246375951683148741722).abs() < 1e-4);
        assert!((dft.0[71].re - -167.48315002876381640817271545529366).abs() < 1e-4);
        assert!((dft.0[72].re - -205.27578304307121470628771930932999).abs() < 1e-4);
        assert!((dft.0[66].im - 58.58786551482015170222439337521791).abs() < 1e-4);
        assert!((dft.0[67].im - 353.73457258774232059295172803103924).abs() < 1e-4);
        assert!((dft.0[68].im - -360.46865539617647300474345684051514).abs() < 1e-4);
        assert!((dft.0[69].im - 730.13489750555811497179092839360237).abs() < 1e-4);
        assert!((dft.0[70].im - 46.66449616613434159262396860867739).abs() < 1e-4);
        assert!((dft.0[71].im - 364.57070422201223891534027643501759).abs() < 1e-4);
        assert!((dft.0[72].im - 10.71977054628772130229208414675668).abs() < 1e-4);
        assert!((dft.0[73].re - 576.07366417374544198537478223443031).abs() < 1e-4);
        assert!((dft.0[74].re - -576.68331857178111476969206705689430).abs() < 1e-4);
        assert!((dft.0[75].re - -840.80257678851728542213095352053642).abs() < 1e-4);
        assert!((dft.0[76].re - 366.74649363974822335876524448394775).abs() < 1e-4);
        assert!((dft.0[77].re - 88.95099604809327331622625933960080).abs() < 1e-4);
        assert!((dft.0[78].re - 77.27541723753962799037253716960549).abs() < 1e-4);
        assert!((dft.0[73].im - -216.96917183271762041840702295303345).abs() < 1e-4);
        assert!((dft.0[74].im - -1193.06624048973094431858044117689133).abs() < 1e-4);
        assert!((dft.0[75].im - -618.26994076338291961292270570993423).abs() < 1e-4);
        assert!((dft.0[76].im - 1237.21964174349841414368711411952972).abs() < 1e-4);
        assert!((dft.0[77].im - 1033.60011705569399964588228613138199).abs() < 1e-4);
        assert!((dft.0[78].im - 592.20510179746065659855958074331284).abs() < 1e-4);
        assert_eq!(dft.0[79], Complex32::zero());
        assert_eq!(dft.0[80], Complex32::zero());
        assert_eq!(dft.0[81], Complex32::zero());
        assert_eq!(dft.0[82], Complex32::zero());
        assert_eq!(dft.0[83], Complex32::zero());
        assert_eq!(dft.0[84], Complex32::zero());
        assert_eq!(dft.0[85], Complex32::zero());
        assert_eq!(dft.0[86], Complex32::zero());
        assert_eq!(dft.0[87], Complex32::zero());
        assert_eq!(dft.0[88], Complex32::zero());
        assert_eq!(dft.0[89], Complex32::zero());
        assert_eq!(dft.0[90], Complex32::zero());
        assert_eq!(dft.0[91], Complex32::zero());
        assert!((dft.0[92].re - 935.56376272982060982030816376209259).abs() < 1e-3);
        assert!((dft.0[93].re - 333.49202949917577143423841334879398).abs() < 1e-4);
        assert!((dft.0[94].re - 261.34768766092463465611217543482780).abs() < 1e-4);
        assert!((dft.0[95].re - -49.61336789901637445154847227968276).abs() < 1e-4);
        assert!((dft.0[96].re - 183.98171796944663469730585347861052).abs() < 1e-4);
        assert!((dft.0[97].re - -61.28501891046433058818365680053830).abs() < 1e-4);
        assert!((dft.0[92].im - -561.18768653114136668591527268290520).abs() < 1e-4);
        assert!((dft.0[93].im - -397.64666578373777383603737689554691).abs() < 1e-4);
        assert!((dft.0[94].im - -293.13862638558407525124493986368179).abs() < 1e-4);
        assert!((dft.0[95].im - 315.02157304752563504735007882118225).abs() < 1e-4);
        assert!((dft.0[96].im - 647.70130654677518577955197542905807).abs() < 1e-4);
        assert!((dft.0[97].im - 321.56440910349215300811920315027237).abs() < 1e-4);
        assert_eq!(dft.0[98], Complex32::zero());
        assert_eq!(dft.0[99], Complex32::zero());
        assert_eq!(dft.0[100], Complex32::zero());
        assert_eq!(dft.0[101], Complex32::zero());
        assert_eq!(dft.0[102], Complex32::zero());
        assert_eq!(dft.0[103], Complex32::zero());
        assert_eq!(dft.0[104], Complex32::zero());
        assert_eq!(dft.0[105], Complex32::zero());
        assert_eq!(dft.0[106], Complex32::zero());
        assert_eq!(dft.0[107], Complex32::zero());
        assert_eq!(dft.0[108], Complex32::zero());
        assert_eq!(dft.0[109], Complex32::zero());
        assert_eq!(dft.0[110], Complex32::zero());
        assert_eq!(dft.0[111], Complex32::zero());
        assert_eq!(dft.0[112], Complex32::zero());
        assert_eq!(dft.0[113], Complex32::zero());
        assert_eq!(dft.0[114], Complex32::zero());
        assert_eq!(dft.0[115], Complex32::zero());
        assert_eq!(dft.0[116], Complex32::zero());
        assert_eq!(dft.0[117], Complex32::zero());
        assert_eq!(dft.0[118], Complex32::zero());
        assert_eq!(dft.0[119], Complex32::zero());
        assert_eq!(dft.0[120], Complex32::zero());
        assert_eq!(dft.0[121], Complex32::zero());
        assert_eq!(dft.0[122], Complex32::zero());
        assert_eq!(dft.0[123], Complex32::zero());
        assert_eq!(dft.0[124], Complex32::zero());
        assert_eq!(dft.0[125], Complex32::zero());
        assert_eq!(dft.0[126], Complex32::zero());
        assert_eq!(dft.0[127], Complex32::zero());

        assert!((dft.idft(-128) - 64.42259519017054003597877454012632).abs() < 1e-2);
        assert!((dft.idft(-127) - 121.63306983925482995800848584622145).abs() < 1e-2);
        assert!((dft.idft(-126) - -9.41794603101527982857987808529288).abs() < 1e-2);
        assert!((dft.idft(-125) - -92.96327345695668498137820279225707).abs() < 1e-2);
        assert!((dft.idft(-124) - -27.11388172953701669598558510188013).abs() < 1e-2);
        assert!((dft.idft(-123) - 40.10162308211403825453089666552842).abs() < 1e-2);
        assert!((dft.idft(-122) - -44.78156399712668900292555917985737).abs() < 1e-2);
        assert!((dft.idft(-121) - -88.96566036518859732495911885052919).abs() < 1e-2);
        assert!((dft.idft(-120) - 67.81889485307867460051056696102023).abs() < 1e-2);
        assert!((dft.idft(-119) - 131.33253141012846754165366291999817).abs() < 1e-2);
        assert!((dft.idft(-118) - -12.78550701705875169977844052482396).abs() < 1e-2);
        assert!((dft.idft(-117) - -75.67484814274409643530816538259387).abs() < 1e-2);
        assert!((dft.idft(-116) - 21.52016771348176149558639735914767).abs() < 1e-2);
        assert!((dft.idft(-115) - 61.97029523185187827039044350385666).abs() < 1e-2);
        assert!((dft.idft(-114) - -46.54606576267670448032731655985117).abs() < 1e-2);
        assert!((dft.idft(-113) - -94.39120075504547457967419177293777).abs() < 1e-2);
        assert!((dft.idft(-112) - 26.60374398393446071509060857351869).abs() < 1e-2);
        assert!((dft.idft(-111) - 66.95948341669280523547058692201972).abs() < 1e-2);
        assert!((dft.idft(-110) - -42.05002726100411791776423342525959).abs() < 1e-2);
        assert!((dft.idft(-109) - -54.32205388807943080564655247144401).abs() < 1e-2);
        assert!((dft.idft(-108) - 49.59203880167189026906271465122700).abs() < 1e-2);
        assert!((dft.idft(-107) - 80.59420057525966285538743250072002).abs() < 1e-2);
        assert!((dft.idft(-106) - 4.13458027955707052569778170436621).abs() < 1e-2);
        assert!((dft.idft(-105) - -53.12246379464443890583424945361912).abs() < 1e-2);
        assert!((dft.idft(-104) - -6.57349261437057297285946333431639).abs() < 1e-2);
        assert!((dft.idft(-103) - 37.45698563477338183247411507181823).abs() < 1e-2);
        assert!((dft.idft(-102) - -33.55147581873112727635088958777487).abs() < 1e-2);
        assert!((dft.idft(-101) - -87.50731679787858752206375356763601).abs() < 1e-2);
        assert!((dft.idft(-100) - 2.21241342381550598616968272835948).abs() < 1e-2);
        assert!((dft.idft(-99) - 87.02676043527513627395819639787078).abs() < 1e-2);
        assert!((dft.idft(-98) - 31.12218106605313749923880095593631).abs() < 1e-2);
        assert!((dft.idft(-97) - -40.98734438648934741422635852359235).abs() < 1e-2);
        assert!((dft.idft(-96) - 6.87944784090122674058420670917258).abs() < 1e-2);
        assert!((dft.idft(-95) - 63.54596160798504200784009299241006).abs() < 1e-2);
        assert!((dft.idft(-94) - -25.35811316039172069736196135636419).abs() < 1e-2);
        assert!((dft.idft(-93) - -106.66870832950812086892256047576666).abs() < 1e-2);
        assert!((dft.idft(-92) - -3.30837405442931764554259643773548).abs() < 1e-2);
        assert!((dft.idft(-91) - 78.74440445506851915524748619645834).abs() < 1e-2);
        assert!((dft.idft(-90) - 4.77252277251996215312601634650491).abs() < 1e-2);
        assert!((dft.idft(-89) - -28.05283314937748784245741262566298).abs() < 1e-2);
        assert!((dft.idft(-88) - 23.92050395611651936178532196208835).abs() < 1e-2);
        assert!((dft.idft(-87) - 27.19742531178921751688903896138072).abs() < 1e-2);
        assert!((dft.idft(-86) - -30.19482831340211959059161017648876).abs() < 1e-2);
        assert!((dft.idft(-85) - -53.01765072037472492638698895461857).abs() < 1e-2);
        assert!((dft.idft(-84) - 6.45437571065289361627037578728050).abs() < 1e-2);
        assert!((dft.idft(-83) - 39.25991587461555099025645176880062).abs() < 1e-2);
        assert!((dft.idft(-82) - 7.37216230992549803602287283865735).abs() < 1e-2);
        assert!((dft.idft(-81) - 6.09912317433270967370617654523812).abs() < 1e-2);
        assert!((dft.idft(-80) - 1.08535712194478994874202726350632).abs() < 1e-2);
        assert!((dft.idft(-79) - -14.72957729453700181920794420875609).abs() < 1e-2);
        assert!((dft.idft(-78) - 5.86980107282878282859428509254940).abs() < 1e-2);
        assert!((dft.idft(-77) - -5.39034345937808012649838929064572).abs() < 1e-2);
        assert!((dft.idft(-76) - -18.77381677314189190042270638514310).abs() < 1e-2);
        assert!((dft.idft(-75) - 6.41540459814678420968903083121404).abs() < 1e-2);
        assert!((dft.idft(-74) - -0.20899481316456097745870579274197).abs() < 1e-2);
        assert!((dft.idft(-73) - -8.13246309644936715699259366374463).abs() < 1e-2);
        assert!((dft.idft(-72) - 6.12267112839593252715530979912728).abs() < 1e-2);
        assert!((dft.idft(-71) - 7.64205550095107533081772999139503).abs() < 1e-2);
        assert!((dft.idft(-70) - 13.77850843024883253917778347386047).abs() < 1e-2);
        assert!((dft.idft(-69) - 23.60498813160100439745292533189058).abs() < 1e-2);
        assert!((dft.idft(-68) - 18.59888711342443912144517526030540).abs() < 1e-2);
        assert!((dft.idft(-67) - -29.47835502140341645826993044465780).abs() < 1e-2);
        assert!((dft.idft(-66) - -67.46607738868371484386443626135588).abs() < 1e-2);
        assert!((dft.idft(-65) - 1.43844526261078575757323960715439).abs() < 1e-2);
        assert!((dft.idft(-64) - 43.16664591550821228338463697582483).abs() < 1e-2);
        assert!((dft.idft(-63) - -30.87636974467304895597408176399767).abs() < 1e-2);
        assert!((dft.idft(-62) - -23.37533177141544626920222071930766).abs() < 1e-2);
        assert!((dft.idft(-61) - 83.51341662745107896625995635986328).abs() < 1e-2);
        assert!((dft.idft(-60) - 66.16016762166988485205365577712655).abs() < 1e-2);
        assert!((dft.idft(-59) - -62.50124119171376690928809694014490).abs() < 1e-2);
        assert!((dft.idft(-58) - -68.04600392941850373063061852008104).abs() < 1e-2);
        assert!((dft.idft(-57) - 36.51714785039450106296499143354595).abs() < 1e-2);
        assert!((dft.idft(-56) - 27.30262400380399512300755304750055).abs() < 1e-2);
        assert!((dft.idft(-55) - -66.54287027930232056860404554754496).abs() < 1e-2);
        assert!((dft.idft(-54) - -54.20008112991858695295377401635051).abs() < 1e-2);
        assert!((dft.idft(-53) - 62.80257088023483191818741033785045).abs() < 1e-2);
        assert!((dft.idft(-52) - 105.26905415876184690660011256113648).abs() < 1e-2);
        assert!((dft.idft(-51) - -16.81584175151880344856181181967258).abs() < 1e-2);
        assert!((dft.idft(-50) - -89.01332324577558097189466934651136).abs() < 1e-2);
        assert!((dft.idft(-49) - 25.11020481110049118456117867026478).abs() < 1e-2);
        assert!((dft.idft(-48) - 76.19880877595294066395581467077136).abs() < 1e-2);
        assert!((dft.idft(-47) - -60.00733246721515712351902038790286).abs() < 1e-2);
        assert!((dft.idft(-46) - -130.42945656114602570596616715192795).abs() < 1e-2);
        assert!((dft.idft(-45) - 45.54990704483925867407378973439336).abs() < 1e-2);
        assert!((dft.idft(-44) - 170.68998779438121005114226136356592).abs() < 1e-2);
        assert!((dft.idft(-43) - -34.97638919467912188565605902113020).abs() < 1e-2);
        assert!((dft.idft(-42) - -148.27933430484969790086324792355299).abs() < 1e-2);
        assert!((dft.idft(-41) - 68.07239751762944024449097923934460).abs() < 1e-2);
        assert!((dft.idft(-40) - 119.41460420817486465239198878407478).abs() < 1e-2);
        assert!((dft.idft(-39) - -71.13039170081452766680740751326084).abs() < 1e-2);
        assert!((dft.idft(-38) - -111.46235389980245145125081762671471).abs() < 1e-2);
        assert!((dft.idft(-37) - 38.88178378031586390761731308884919).abs() < 1e-2);
        assert!((dft.idft(-36) - 114.88428948165289966709678992629051).abs() < 1e-2);
        assert!((dft.idft(-35) - -41.96831753251464647291868459433317).abs() < 1e-2);
        assert!((dft.idft(-34) - -133.22845736152103768290544394403696).abs() < 1e-2);
        assert!((dft.idft(-33) - 38.51696164793128929204613086767495).abs() < 1e-2);
        assert!((dft.idft(-32) - 126.15921062440004618565581040456891).abs() < 1e-2);
        assert!((dft.idft(-31) - 23.53593553194738774436700623482466).abs() < 1e-2);
        assert!((dft.idft(-30) - -77.15745201647371231956640258431435).abs() < 1e-2);
        assert!((dft.idft(-29) - -52.19076280799291112089122179895639).abs() < 1e-2);
        assert!((dft.idft(-28) - 87.86549080322538429754786193370819).abs() < 1e-2);
        assert!((dft.idft(-27) - 14.88671081481284552694432932185009).abs() < 1e-2);
        assert!((dft.idft(-26) - -171.22560727602404995195684023201466).abs() < 1e-2);
        assert!((dft.idft(-25) - -16.71302274722966529907353105954826).abs() < 1e-2);
        assert!((dft.idft(-24) - 173.47173392336944175440294202417135).abs() < 1e-2);
        assert!((dft.idft(-23) - 58.92271527655534413270288496278226).abs() < 1e-2);
        assert!((dft.idft(-22) - -92.12029917696305858498817542567849).abs() < 1e-2);
        assert!((dft.idft(-21) - -58.52270847259715935706481104716659).abs() < 1e-2);
        assert!((dft.idft(-20) - 84.93720528112550027799443341791630).abs() < 1e-2);
        assert!((dft.idft(-19) - 34.65807826494620513813060824759305).abs() < 1e-2);
        assert!((dft.idft(-18) - -134.26283044992609916334913577884436).abs() < 1e-2);
        assert!((dft.idft(-17) - -28.91679840136026768959709443151951).abs() < 1e-2);
        assert!((dft.idft(-16) - 120.94162526795366829901468008756638).abs() < 1e-2);
        assert!((dft.idft(-15) - 43.50772031123560168452968355268240).abs() < 1e-2);
        assert!((dft.idft(-14) - -67.41424415055631413906667148694396).abs() < 1e-2);
        assert!((dft.idft(-13) - -84.05860323194413297187566058710217).abs() < 1e-2);
        assert!((dft.idft(-12) - 38.82701449642151914076748653315008).abs() < 1e-2);
        assert!((dft.idft(-11) - 96.69968695397385261003364576026797).abs() < 1e-2);
        assert!((dft.idft(-10) - -50.93025066148771173857312533073127).abs() < 1e-2);
        assert!((dft.idft(-9) - -51.84281200675548717526908149011433).abs() < 1e-2);
        assert!((dft.idft(-8) - 87.69660695522080118280427996069193).abs() < 1e-2);
        assert!((dft.idft(-7) - 49.87522038161620940854845684953034).abs() < 1e-2);
        assert!((dft.idft(-6) - -72.05538529258362245855096261948347).abs() < 1e-2);
        assert!((dft.idft(-5) - -99.79630202649343573284568265080452).abs() < 1e-2);
        assert!((dft.idft(-4) - -0.32507098649241650267782688388252).abs() < 1e-2);
        assert!((dft.idft(-3) - 83.52356932488631002797774272039533).abs() < 1e-2);
        assert!((dft.idft(-2) - 11.79304219252217400537574576446787).abs() < 1e-2);
        assert!((dft.idft(-1) - -31.93931117908418571005313424393535).abs() < 1e-2);
        assert!((dft.idft(0) - 25.30613912637092255408788332715631).abs() < 1e-2);
        assert!((dft.idft(1) - 40.68574731296158120130712632089853).abs() < 1e-2);
        assert!((dft.idft(2) - -0.60689811897838064069787833432201).abs() < 1e-2);
        assert!((dft.idft(3) - -68.96947427724329315879003843292594).abs() < 1e-2);
        assert!((dft.idft(4) - -48.63313630127792919211060507223010).abs() < 1e-2);
        assert!((dft.idft(5) - 80.80983316194502208418271038681269).abs() < 1e-2);
        assert!((dft.idft(6) - 65.51969448781612470611435128375888).abs() < 1e-2);
        assert!((dft.idft(7) - -79.38919635090361737184139201417565).abs() < 1e-2);
        assert!((dft.idft(8) - -73.48633356316136655550508294254541).abs() < 1e-2);
        assert!((dft.idft(9) - 39.18541662106756007233343552798033).abs() < 1e-2);
        assert!((dft.idft(10) - 44.23472494784491004793380852788687).abs() < 1e-2);
        assert!((dft.idft(11) - -22.58105416280769617287660366855562).abs() < 1e-2);
        assert!((dft.idft(12) - -9.95203325223520529618781438330188).abs() < 1e-2);
        assert!((dft.idft(13) - 72.89446727156263250435586087405682).abs() < 1e-2);
        assert!((dft.idft(14) - 75.86112597487067432666663080453873).abs() < 1e-2);
        assert!((dft.idft(15) - -62.27284363359093788403697544708848).abs() < 1e-2);
        assert!((dft.idft(16) - -148.94976992345098665282421279698610).abs() < 1e-2);
        assert!((dft.idft(17) - -26.47729956488647218293408514000475).abs() < 1e-2);
        assert!((dft.idft(18) - 100.55850753435925071244128048419952).abs() < 1e-2);
        assert!((dft.idft(19) - 34.10443987175787583510100375860929).abs() < 1e-2);
        assert!((dft.idft(20) - -60.64277213933897314745991025120020).abs() < 1e-2);
        assert!((dft.idft(21) - 27.91899710814645985124116123188287).abs() < 1e-2);
        assert!((dft.idft(22) - 132.95718540690864983844221569597721).abs() < 1e-2);
        assert!((dft.idft(23) - -36.35296025652959173157796612940729).abs() < 1e-2);
        assert!((dft.idft(24) - -192.13042787821046886165277101099491).abs() < 1e-2);
        assert!((dft.idft(25) - 17.34338560508507498525432310998440).abs() < 1e-2);
        assert!((dft.idft(26) - 160.93170866633019500113732647150755).abs() < 1e-2);
        assert!((dft.idft(27) - -16.21906942441147947420176933519542).abs() < 1e-2);
        assert!((dft.idft(28) - -103.09803049068808888932835543528199).abs() < 1e-2);
        assert!((dft.idft(29) - 8.09964243755235635546796402195469).abs() < 1e-2);
        assert!((dft.idft(30) - 77.66221322193239018361055059358478).abs() < 1e-2);
        assert!((dft.idft(31) - -7.45076283804888639394903293577954).abs() < 1e-2);
        assert!((dft.idft(32) - -100.09379041996801618097379105165601).abs() < 1e-2);
        assert!((dft.idft(33) - 12.03811075143823572375367803033441).abs() < 1e-2);
        assert!((dft.idft(34) - 125.56641197492167805194185348227620).abs() < 1e-2);
        assert!((dft.idft(35) - 27.65675312949167263809613359626383).abs() < 1e-2);
        assert!((dft.idft(36) - -99.91799237534850419706344837322831).abs() < 1e-2);
        assert!((dft.idft(37) - -90.86598924764670925924292532727122).abs() < 1e-2);
        assert!((dft.idft(38) - 71.09972662817878585883590858429670).abs() < 1e-2);
        assert!((dft.idft(39) - 108.55373002329876896965288324281573).abs() < 1e-2);
        assert!((dft.idft(40) - -105.11094556182037251801375532522798).abs() < 1e-2);
        assert!((dft.idft(41) - -75.32517437666794535289227496832609).abs() < 1e-2);
        assert!((dft.idft(42) - 162.80658110277201444660022389143705).abs() < 1e-2);
        assert!((dft.idft(43) - 53.79464066647766173900890862569213).abs() < 1e-2);
        assert!((dft.idft(44) - -172.22222720330262291099643334746361).abs() < 1e-2);
        assert!((dft.idft(45) - -80.91563527570096425733936484903097).abs() < 1e-2);
        assert!((dft.idft(46) - 131.70968133209586881093855481594801).abs() < 1e-2);
        assert!((dft.idft(47) - 94.86099127694916433028993196785450).abs() < 1e-2);
        assert!((dft.idft(48) - -109.11231822800488089342252351343632).abs() < 1e-2);
        assert!((dft.idft(49) - -48.50991309125067374452555668540299).abs() < 1e-2);
        assert!((dft.idft(50) - 133.62844168091007190923846792429686).abs() < 1e-2);
        assert!((dft.idft(51) - 21.75355409405599260708186193369329).abs() < 1e-2);
        assert!((dft.idft(52) - -125.65114813961059780922369100153446).abs() < 1e-2);
        assert!((dft.idft(53) - -59.12145159621906032043625600636005).abs() < 1e-2);
        assert!((dft.idft(54) - 58.04754734289151940629380987957120).abs() < 1e-2);
        assert!((dft.idft(55) - 85.67394571789466795053158421069384).abs() < 1e-2);
        assert!((dft.idft(56) - -36.72026364852642643654689891263843).abs() < 1e-2);
        assert!((dft.idft(57) - -83.69168571816541657426569145172834).abs() < 1e-2);
        assert!((dft.idft(58) - 78.10993710235796072538505541160703).abs() < 1e-2);
        assert!((dft.idft(59) - 97.16002521976638206524512497708201).abs() < 1e-2);
        assert!((dft.idft(60) - -81.05889755235230609287100378423929).abs() < 1e-2);
        assert!((dft.idft(61) - -110.01977076269766087079915450885892).abs() < 1e-2);
        assert!((dft.idft(62) - 50.56196484376166466745416983030736).abs() < 1e-2);
        assert!((dft.idft(63) - 107.83191291989137994278280530124903).abs() < 1e-2);
        assert!((dft.idft(64) - -56.80213680543050713822594843804836).abs() < 1e-2);
        assert!((dft.idft(65) - -105.85005589130587111412751255556941).abs() < 1e-2);
        assert!((dft.idft(66) - 82.62061709931710140608629444614053).abs() < 1e-2);
        assert!((dft.idft(67) - 91.42297016134381237861816771328449).abs() < 1e-2);
        assert!((dft.idft(68) - -87.43479372379130154513404704630375).abs() < 1e-2);
        assert!((dft.idft(69) - -68.08589879677900569276971509680152).abs() < 1e-2);
        assert!((dft.idft(70) - 68.62637956033780994857806945219636).abs() < 1e-2);
        assert!((dft.idft(71) - 57.01231262722411941012978786602616).abs() < 1e-2);
        assert!((dft.idft(72) - -41.18368171840808855677096289582551).abs() < 1e-2);
        assert!((dft.idft(73) - -46.79758326626389219882184988819063).abs() < 1e-2);
        assert!((dft.idft(74) - 27.09289098775152382359010516665876).abs() < 1e-2);
        assert!((dft.idft(75) - 32.56553269247952897558207041583955).abs() < 1e-2);
        assert!((dft.idft(76) - -20.34787907116844607458006066735834).abs() < 1e-2);
        assert!((dft.idft(77) - -23.73015547930431523582228692248464).abs() < 1e-2);
        assert!((dft.idft(78) - 5.57046550323321731923442712286487).abs() < 1e-2);
        assert!((dft.idft(79) - 23.64820544784612366129294969141483).abs() < 1e-2);
        assert!((dft.idft(80) - 10.32656628339723248188875004416332).abs() < 1e-2);
        assert!((dft.idft(81) - -38.26484632951154196689458331093192).abs() < 1e-2);
        assert!((dft.idft(82) - -35.12646505946358388428052421659231).abs() < 1e-2);
        assert!((dft.idft(83) - 39.24199889576057387330365600064397).abs() < 1e-2);
        assert!((dft.idft(84) - 49.13871294174068538040955900214612).abs() < 1e-2);
        assert!((dft.idft(85) - -16.89959546783107668943557655438781).abs() < 1e-2);
        assert!((dft.idft(86) - -15.29605323788375947913209529360756).abs() < 1e-2);
        assert!((dft.idft(87) - 33.85500241629342355054177460260689).abs() < 1e-2);
        assert!((dft.idft(88) - 2.98095344998257250068718349211849).abs() < 1e-2);
        assert!((dft.idft(89) - -60.76241915197672938120376784354448).abs() < 1e-2);
        assert!((dft.idft(90) - -48.51371998061103596455723163671792).abs() < 1e-2);
        assert!((dft.idft(91) - 12.12010686281630533756015211110935).abs() < 1e-2);
        assert!((dft.idft(92) - 36.35642595318564218587198411114514).abs() < 1e-2);
        assert!((dft.idft(93) - 20.48040201455224718074532574974000).abs() < 1e-2);
        assert!((dft.idft(94) - 21.35542319132686017724154226016253).abs() < 1e-2);
        assert!((dft.idft(95) - 26.67974501128360742541190120391548).abs() < 1e-2);
        assert!((dft.idft(96) - 5.38736092384732678794989624293521).abs() < 1e-2);
        assert!((dft.idft(97) - -14.58469909131014041747675946680829).abs() < 1e-2);
        assert!((dft.idft(98) - -45.43582357603801114009911543689668).abs() < 1e-2);
        assert!((dft.idft(99) - -56.40334336085416566675121430307627).abs() < 1e-2);
        assert!((dft.idft(100) - 14.25460440394217798143472464289516).abs() < 1e-2);
        assert!((dft.idft(101) - 48.54664360507891984752859571017325).abs() < 1e-2);
        assert!((dft.idft(102) - -9.37674054328559591908742731902748).abs() < 1e-2);
        assert!((dft.idft(103) - -18.02771113247451850725155964028090).abs() < 1e-2);
        assert!((dft.idft(104) - 30.18994016744491304393704922404140).abs() < 1e-2);
        assert!((dft.idft(105) - 34.43876983471905361966491909697652).abs() < 1e-2);
        assert!((dft.idft(106) - -13.19411131355785116170409310143441).abs() < 1e-2);
        assert!((dft.idft(107) - -34.71816937742666908661703928373754).abs() < 1e-2);
        assert!((dft.idft(108) - 27.12159171047665040532592684030533).abs() < 1e-2);
        assert!((dft.idft(109) - 47.53535468294820987011917168274522).abs() < 1e-2);
        assert!((dft.idft(110) - -46.71186368092463681023218668997288).abs() < 1e-2);
        assert!((dft.idft(111) - -83.27763119692434656826662831008434).abs() < 1e-2);
        assert!((dft.idft(112) - -10.62384075874326327948438120074570).abs() < 1e-2);
        assert!((dft.idft(113) - 48.43253999147198385344381676986814).abs() < 1e-2);
        assert!((dft.idft(114) - 31.19271148988846675820241216570139).abs() < 1e-2);
        assert!((dft.idft(115) - -17.81344881169568594714291975833476).abs() < 1e-2);
        assert!((dft.idft(116) - 29.84811379984226675787795102223754).abs() < 1e-2);
        assert!((dft.idft(117) - 85.79371303691222294673934811726213).abs() < 1e-2);
        assert!((dft.idft(118) - -35.87368120591880682468399754725397).abs() < 1e-2);
        assert!((dft.idft(119) - -122.64756863911628670393838547170162).abs() < 1e-2);
        assert!((dft.idft(120) - -2.81774274230703269950026879087090).abs() < 1e-2);
        assert!((dft.idft(121) - 81.89409420480171775125199928879738).abs() < 1e-2);
        assert!((dft.idft(122) - 4.18814385106759434762579985545017).abs() < 1e-2);
        assert!((dft.idft(123) - -68.42950090791062223161134170368314).abs() < 1e-2);
        assert!((dft.idft(124) - 4.54994343608337636908345302799717).abs() < 1e-2);
        assert!((dft.idft(125) - 75.54024218153713832180073950439692).abs() < 1e-2);
        assert!((dft.idft(126) - -34.64023523716051045084896031767130).abs() < 1e-2);
        assert!((dft.idft(127) - -88.51185254733691465389711083844304).abs() < 1e-2);
    }
}
