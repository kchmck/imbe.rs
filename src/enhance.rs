//! Frame repeat/muting, spectral amplitude enhancement, and adaptive smoothing.

use std;
use std::f32::consts::PI;

use arrayvec::ArrayVec;
use slice_mip::MapInPlace;

use consts::MAX_HARMONICS;
use frame::Errors;
use descramble::VoiceDecisions;
use params::BaseParams;
use spectral::Spectrals;

/// Values derived from error correction decoding.
pub struct EnhanceErrors {
    /// Total number of errors corrected in the current frame, ϵ<sub>T</sub> [p45].
    pub total: usize,
    /// Error rate tracking term, ϵ<sub>R</sub> [p45].
    pub rate: f32,
    /// Errors corrected in first (u<sub>0</sub>) Golay-coded chunk, ϵ<sub>0</sub>.
    pub golay_init: usize,
    /// Errors corrected in first (u<sub>4</sub>) Hamming-coded chunk, ϵ<sub>4</sub>.
    pub hamming_init: usize,
}

impl EnhanceErrors {
    /// Create a new `EnhanceErrors` from the errors corrected in the current frame,
    /// ϵ<sub>i</sub>, and the previous frame's ϵ<sub>R</sub> value.
    pub fn new(errors: &Errors, prev_rate: f32) -> EnhanceErrors {
        let total = errors.iter().fold(0, |s, &e| s + e);

        EnhanceErrors {
            total: total,
            // Compute Eq 96.
            rate: 0.95 * prev_rate + 0.000365 * total as f32,
            golay_init: errors[0],
            hamming_init: errors[4],
        }
    }
}

/// Energy-related parameters for a voice frame.
pub struct FrameEnergy {
    /// Spectral amplitude energy, R<sub>M0</sub>.
    pub energy: f32,
    /// Scaled energy value, R<sub>M1</sub>.
    pub scaled: f32,
    /// Moving average energy tracker, S<sub>E</sub>.
    pub tracking: f32,
}

impl FrameEnergy {
    /// Create a new `FrameEnergy` from the given spectral amplitudes M<sub>l</sub>,
    /// previous frame energy values, and current frame parameters.
    pub fn new(spectrals: &Spectrals, prev: &FrameEnergy, params: &BaseParams)
        -> FrameEnergy
    {
        // Compute energy of spectral amplitudes according to Eq 105.
        let energy = spectrals.iter()
            .map(|&m| m.powi(2))
            .fold(0.0, |s, x| s + x);

        // Compute scaled energies according to Eq 106.
        let scaled = spectrals.iter().enumerate()
            .map(|(l, &m)| m.powi(2) * (params.fundamental * (l + 1) as f32).cos())
            .fold(0.0, |s, x| s + x);

        FrameEnergy {
            energy: energy,
            scaled: scaled,
            // Compute energy tracking EWMA according to Eq 111.
            tracking: (0.95 * prev.tracking + 0.05 * energy).max(10000.0),
        }
    }
}

impl Default for FrameEnergy {
    /// Create a new `FrameEnergy` with default initial values.
    fn default() -> FrameEnergy {
        // The first two values are arbitrary as they aren't saved across frames, and the
        // third value is taken from [p64].
        FrameEnergy {
            energy: 0.0,
            scaled: 0.0,
            tracking: 75000.0,
        }
    }
}

/// Enhanced spectral amplitudes, "overbar" M<sub>l</sub>, are derived from the decoded
/// spectral amplitudes, "tilde" M<sub>l</sub>.
#[derive(Clone)]
pub struct EnhancedSpectrals(ArrayVec<[f32; MAX_HARMONICS]>);

impl EnhancedSpectrals {
    /// Create a new `EnhancedSpectrals` from the given base spectral amplitudes
    /// M<sub>l</sub> and current frame energy values and parameters.
    pub fn new(spectrals: &Spectrals, fen: &FrameEnergy, params: &BaseParams)
        -> EnhancedSpectrals
    {
        // Compute R_M0^2.
        let energy_sqr = fen.energy.powi(2);
        // Compute R_M1^2.
        let scaled_sqr = fen.scaled.powi(2);
        // Compute denominator term of Eq 107.
        let denom = params.fundamental * fen.energy * (energy_sqr - scaled_sqr);

        let mut enhanced = spectrals.iter().enumerate().map(|(l, &m)| {
            let l = l + 1;

            // Handle fast-path case in Eq 108.
            if 8 * l as u32 <= params.harmonics {
                return m;
            }

            // Compute Eq 107.
            let weight = m.sqrt() * (
                0.96 * PI * (
                    energy_sqr + scaled_sqr - 2.0 * fen.energy * fen.scaled *
                        (params.fundamental * l as f32).cos()
                ) / denom
            ).powf(0.25);

            // Scale current spectral amplitude according to Eq 108.
            m * weight.max(0.5).min(1.2)
        }).collect::<ArrayVec<[f32; MAX_HARMONICS]>>();

        // Compute root ratio of energies according to Eq 109.
        let scale = (
            fen.energy / enhanced.iter().fold(0.0, |s, &m| s + m.powi(2))
        ).sqrt();

        // Perform second scaling pass according to Eq 110.
        enhanced.map_in_place(|&m| m * scale);

        EnhancedSpectrals(enhanced)
    }

    /// Retrieve the enhanced spectral amplitude M<sub>l</sub>, 1 ≤ l ≤ L.
    pub fn get(&self, l: usize) -> f32 {
        assert!(l >= 1);

        match self.0.get(l - 1) {
            Some(&s) => s,
            // Out-of-bounds amplitudes are zero [p60].
            None => 0.0,
        }
    }
}

impl std::ops::Deref for EnhancedSpectrals {
    type Target = ArrayVec<[f32; MAX_HARMONICS]>;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl std::ops::DerefMut for EnhancedSpectrals {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl Default for EnhancedSpectrals {
    /// Create a new `EnhancedSpectrals` with default initial values.
    fn default() -> EnhancedSpectrals {
        // By default, all enhanced amplitudes are 0 [p64].
        EnhancedSpectrals(ArrayVec::new())
    }
}

/// Compute the spectral amplitude threshold τ<sub>M</sub> used in adaptive smoothing from
/// the given error characteristics and previous amplitude threshold.
pub fn amp_thresh(errors: &EnhanceErrors, prev: f32) -> f32 {
    // Compute Eq 115.
    if errors.rate <= 0.005 && errors.total <= 6 {
        20480.0
    } else {
        6000.0 - 300.0 * errors.total as f32 + prev
    }
}

/// Smooth the given enhanced spectral amplitudes M<sub>l</sub> and voiced/unvoiced
/// decisions v<sub>l</sub> based on the given error characteristics, current frame
/// energy, and spectral amplitude threshold τ<sub>M</sub> for the current frame.
pub fn smooth(enhanced: &mut EnhancedSpectrals, voiced: &mut VoiceDecisions,
              errors: &EnhanceErrors, fen: &FrameEnergy, amp_thresh: f32)
{
    // Compute Eq 112.
    let thresh = if errors.rate <= 0.005 && errors.total <= 4 {
        std::f32::MAX
    } else if errors.rate <= 0.0125 && errors.hamming_init == 0 {
        45.255 * fen.tracking.powf(0.375) / (277.26 * errors.rate).exp()
    } else {
        1.414 * fen.tracking.powf(0.375)
    };

    // Update voiced/unvoiced decisions according to Eq 113.
    for (l, &m) in enhanced.iter().enumerate() {
        if m > thresh {
            voiced.force_voiced(l + 1);
        }
    }

    // Compute amplitude sum in Eq 114.
    let amp = enhanced.iter().fold(0.0, |s, &m| s + m);
    // Compute scale factor in Eq 116.
    let scale = (amp_thresh / amp).min(1.0);

    // Scale each enhanced M_l [p50].
    enhanced.map_in_place(|&m| m * scale);
}

/// Check whether the current frame should be discarded and the previous repeated based on
/// the given error characteristics of the current frame.
pub fn should_repeat(errors: &EnhanceErrors) -> bool {
    // Check the conditions in Eqs 97 and 98.
    errors.golay_init >= 2 && errors.total as f32 >= 10.0 + 40.0 * errors.rate
}

/// Check if the current frame should be discarded and replaced with silence/comfort noise
/// based on the given error characteristics of the current frame.
pub fn should_mute(errors: &EnhanceErrors) -> bool {
    // Check the condition on [p47].
    errors.rate > 0.0875
}

#[cfg(test)]
mod tests {
    use super::*;
    use descramble::{Bootstrap, descramble};
    use gain::Gains;
    use spectral::Spectrals;
    use coefs::Coefficients;
    use params::BaseParams;
    use prev::{PrevFrame};

    #[test]
    fn test_errors() {
        let e = EnhanceErrors::new(&[1, 2, 3, 4, 5, 6, 7], 0.5);

        assert_eq!(e.total, 28);
        assert!((e.rate - 0.48522).abs() < 0.00001);
        assert_eq!(e.golay_init, 1);
        assert_eq!(e.hamming_init, 5);
    }

    #[test]
    fn test_frame_energy() {
        // Test against results computed with standalone python script.

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
        assert_eq!(p.harmonics, 16);
        let (amps, _, gain_idx) = descramble(&chunks, &p);
        let g = Gains::new(gain_idx, &amps, &p);
        let c = Coefficients::new(&g, &amps, &p);
        let prev = PrevFrame::default();
        let s = Spectrals::new(&c, &p, &prev);

        assert!((s.get(0) - 1.0).abs() < 0.000001);
        assert!((s.get(1) - 0.5306769781475001).abs() < 0.000001);
        assert!((s.get(2) - 0.3535007196676076).abs() < 0.000001);
        assert!((s.get(3) - 0.9173875577243951).abs() < 0.000001);
        assert!((s.get(4) - 0.13169278935782622).abs() < 0.000001);
        assert!((s.get(5) - 4.438599873836148).abs() < 0.00001);
        assert!((s.get(6) - 0.6796441620283217).abs() < 0.000001);
        assert!((s.get(7) - 0.9439604687782126).abs() < 0.000001);
        assert!((s.get(8) - 10.646341109175768).abs() < 0.000001);
        assert!((s.get(9) - 1.3058035207362984).abs() < 0.000001);
        assert!((s.get(10) - 1.566168983101457).abs() < 0.000001);
        assert!((s.get(11) - 8.325591838823936).abs() < 0.00001);
        assert!((s.get(12) - 1.5943520720633444).abs() < 0.000001);
        assert!((s.get(13) - 1.556318849691248).abs() < 0.000001);
        assert!((s.get(14) - 15.882295806333524).abs() < 0.00001);
        assert!((s.get(15) - 11.386537444591385).abs() < 0.00001);
        assert!((s.get(16) - 12.498792573794846).abs() < 0.00001);
        assert!((s.get(17) - 12.498792573794846).abs() < 0.00001);
        assert!((s.get(30) - 12.498792573794846).abs() < 0.00001);
        assert!((s.get(56) - 12.498792573794846).abs() < 0.00001);

        let pfe = FrameEnergy::default();
        let fe = FrameEnergy::new(&s, &pfe, &p);

        assert!((fe.energy - 752.2228257467827461).abs() < 0.0001);
        assert!((fe.scaled - -452.3599305372189292).abs() < 0.0001);
        assert!((fe.tracking - 71287.6111412873433437).abs() < 0.0001);

        let fe = FrameEnergy::new(&s, &fe, &p);

        assert!((fe.energy - 752.2228257467827461).abs() < 0.0001);
        assert!((fe.scaled - -452.3599305372189292).abs() < 0.0001);
        assert!((fe.tracking - 67760.8417255103122443).abs() < 0.1);
    }

    #[test]
    fn test_spectrals() {
        // Test against results computed with standalone python script.

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
        let (amps, _, gain_idx) = descramble(&chunks, &p);
        let g = Gains::new(gain_idx, &amps, &p);
        let c = Coefficients::new(&g, &amps, &p);
        let prev = PrevFrame::default();
        let s = Spectrals::new(&c, &p, &prev);
        let pfe = FrameEnergy::default();
        let fe = FrameEnergy::new(&s, &pfe, &p);
        let e = EnhancedSpectrals::new(&s, &fe, &p);

        assert_eq!(e.0.len(), 16);

        assert!((e.get(1) - 0.46407439358059288103675044112606).abs() < 1e-6);
        assert!((e.get(2) - 0.30913461049404472591461967567739).abs() < 1e-6);
        assert!((e.get(3) - 0.41587411398142343221806527253648).abs() < 1e-6);
        assert!((e.get(4) - 0.05758233878573310732251755439393).abs() < 1e-6);
        assert!((e.get(5) - 4.29488628936570737693045884952880).abs() < 1e-6);
        assert!((e.get(6) - 0.29717281253740035484867121340358).abs() < 1e-6);
        assert!((e.get(7) - 0.41274456522882696507537048091763).abs() < 1e-6);
        assert!((e.get(8) - 11.17220652652428292128661269089207).abs() < 1e-6);
        assert!((e.get(9) - 0.61137412496551313267900695791468).abs() < 1e-6);
        assert!((e.get(10) - 0.76976839668918539683062363110366).abs() < 1e-6);
        assert!((e.get(11) - 8.73682419143608512968057766556740).abs() < 1e-6);
        assert!((e.get(12) - 0.71114975707359529000228803852224).abs() < 1e-6);
        assert!((e.get(13) - 0.68049652809620897464526478870539).abs() < 1e-6);
        assert!((e.get(14) - 16.66679545607086865288692933972925).abs() < 1e-6);
        assert!((e.get(15) - 10.89394258052602992847823770716786).abs() < 1e-6);
        assert!((e.get(16) - 11.55358738738138946189337730174884).abs() < 1e-6);
        assert_eq!(e.get(17), 0.0);
        assert_eq!(e.get(30), 0.0);
        assert_eq!(e.get(56), 0.0);
    }

    #[test]
    fn test_smooth() {
        // Test against results from standalone python script.

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
        let (amps, mut voice, gain_idx) = descramble(&chunks, &p);
        let g = Gains::new(gain_idx, &amps, &p);
        let c = Coefficients::new(&g, &amps, &p);
        let prev = PrevFrame::default();
        let s = Spectrals::new(&c, &p, &prev);
        let pfe = FrameEnergy::default();
        let mut fe = FrameEnergy::new(&s, &pfe, &p);
        let mut amps = EnhancedSpectrals::new(&s, &fe, &p);

        let err = EnhanceErrors {
            total: 0,
            rate: 0.0250,
            golay_init: 0,
            hamming_init: 0,
        };

        // Dummy value just to get a good adaptive threshold.
        fe.tracking = 100.0;

        assert!(voice.is_voiced(1));
        assert!(voice.is_voiced(2));
        assert!(voice.is_voiced(3));
        assert!(!voice.is_voiced(4));
        assert!(!voice.is_voiced(5));
        assert!(!voice.is_voiced(6));
        assert!(voice.is_voiced(7));
        assert!(voice.is_voiced(8));
        assert!(voice.is_voiced(9));
        assert!(!voice.is_voiced(10));
        assert!(!voice.is_voiced(11));
        assert!(!voice.is_voiced(12));
        assert!(!voice.is_voiced(13));
        assert!(!voice.is_voiced(14));
        assert!(!voice.is_voiced(15));
        assert!(voice.is_voiced(16));

        smooth(&mut amps, &mut voice, &err, &fe, 42.0);

        assert!(voice.is_voiced(1));
        assert!(voice.is_voiced(2));
        assert!(voice.is_voiced(3));
        assert!(!voice.is_voiced(4));
        assert!(!voice.is_voiced(5));
        assert!(!voice.is_voiced(6));
        assert!(voice.is_voiced(7));
        assert!(voice.is_voiced(8));
        assert!(voice.is_voiced(9));
        assert!(!voice.is_voiced(10));
        assert!(voice.is_voiced(11));
        assert!(!voice.is_voiced(12));
        assert!(!voice.is_voiced(13));
        assert!(voice.is_voiced(14));
        assert!(voice.is_voiced(15));
        assert!(voice.is_voiced(16));

        assert!((amps.get(1) - 0.28643362145733153312221475061961).abs() < 1e-6);
        assert!((amps.get(2) - 0.19080248172803679351794414742471).abs() < 1e-6);
        assert!((amps.get(3) - 0.25668369163611542971281664904382).abs() < 1e-6);
        assert!((amps.get(4) - 0.03554067636251981299189139917871).abs() < 1e-6);
        assert!((amps.get(5) - 2.65086772859580133143708735588007).abs() < 1e-6);
        assert!((amps.get(6) - 0.18341948202958965885578379584331).abs() < 1e-6);
        assert!((amps.get(7) - 0.25475208757622064270620398929168).abs() < 1e-6);
        assert!((amps.get(8) - 6.89565211812498901622348057571799).abs() < 1e-6);
        assert!((amps.get(9) - 0.37734920758726891998335872813186).abs() < 1e-6);
        assert!((amps.get(10) - 0.47511250910851310358395949151600).abs() < 1e-6);
        assert!((amps.get(11) - 5.39249790077991697501147427828982).abs() < 1e-6);
        assert!((amps.get(12) - 0.43893221245295155341636927914806).abs() < 1e-6);
        assert!((amps.get(13) - 0.42001258338742591957881700182043).abs() < 1e-6);
        assert!((amps.get(14) - 10.28699416862334992117666843114421).abs() < 1e-6);
        assert!((amps.get(15) - 6.72390346990002374383266214863397).abs() < 1e-6);
        assert!((amps.get(16) - 7.13104606064994861469585885060951).abs() < 1e-4);
        assert_eq!(amps.get(17), 0.0);
        assert_eq!(amps.get(30), 0.0);
        assert_eq!(amps.get(56), 0.0);
    }
}
