//! Voiced spectrum synthesis.

use std::cmp::max;
use std::f32::consts::PI;

use collect_slice::CollectSlice;
use map_in_place::MapInPlace;
use rand::Rng;

use consts::{SAMPLES_PER_FRAME, MAX_HARMONICS};
use descramble::VoiceDecisions;
use enhance::EnhancedSpectrals;
use params::BaseParams;
use prev::PrevFrame;
use window;

/// Computes the base phase offsets Ψ<sub>l</sub>.
pub struct PhaseBase([f32; MAX_HARMONICS]);

impl PhaseBase {
    /// Create a new `PhaseBase` from the given current and previous frame parameters.
    pub fn new(params: &BaseParams, prev: &PrevFrame) -> Self {
        let mut base = [0.0; MAX_HARMONICS];

        // Compute common scaling factor in Eq 139.
        let scale = (prev.params.fundamental + params.fundamental) *
            SAMPLES_PER_FRAME as f32 / 2.0;

        // Compute Eq 139.
        (1...MAX_HARMONICS).map(|l| {
            prev.phase_base.get(l) + scale * l as f32
        }).collect_slice_checked(&mut base[..]);

        PhaseBase(base)
    }

    /// Retrieve the phase term Ψ<sub>l</sub>, 1 ≤ l ≤ 56.
    pub fn get(&self, l: usize) -> f32 { self.0[l - 1] }
}

impl Default for PhaseBase {
    /// Create a new `PhaseBase` in the default state.
    fn default() -> Self {
        // By default all phase terms are 0 [p64].
        PhaseBase([0.0; MAX_HARMONICS])
    }
}

/// Computes the random phase terms Φ<sub>l</sub>.
pub struct Phase([f32; MAX_HARMONICS]);

impl Phase {
    /// Create a new `Phase` building on the given base phase terms.
    pub fn new<R: Rng>(base: &PhaseBase, params: &BaseParams, prev: &PrevFrame,
                       voice: &VoiceDecisions, mut noise: R)
        -> Self
    {
        let mut phase = [0.0; MAX_HARMONICS];

        // Derive phase terms from base phase offsets according to Eq 140.
        (&mut phase[..]).copy_from_slice(&base.0[..]);

        // Compute bounds for modification used in Eq 140.
        let start = params.harmonics as usize / 4;
        let stop = max(params.harmonics, prev.params.harmonics) as usize;

        // Compute common scaling factor in Eq 140.
        let scale = voice.unvoiced_count() as f32 / params.harmonics as f32;

        // Modify Ψ_l from start + 1 ≤ l ≤ stop. Since i = l - 1, start ≤ i ≤ stop - 1.
        (&mut phase[start..stop]).map_in_place(|&x| {
            // Compute Eq 140.
            x + scale * noise.gen_range(-PI, PI)
        });

        Phase(phase)
    }

    /// Retrieve the phase term Φ<sub>l</sub>, 1 ≤ l ≤ 56.
    pub fn get(&self, l: usize) -> f32 { self.0[l - 1] }
}

impl Default for Phase {
    /// Create a new `Phase` in the default state.
    fn default() -> Self {
        // By default all phase terms are 0 [p64].
        Phase([0.0; MAX_HARMONICS])
    }
}

/// Synthesizes voiced spectrum signal s<sub>v</sub>(n).
pub struct Voiced<'a, 'b, 'c, 'd> {
    prev: &'a PrevFrame,
    phase: &'b Phase,
    enhanced: &'c EnhancedSpectrals,
    voice: &'d VoiceDecisions,
    /// Synthesis window w<sub>s</sub> for combining voiced/unvoiced frames.
    window: window::Window,
    /// Fundamental frequency of current frame.
    fundamental: f32,
    /// Number of harmonics that make up each signal sample.
    end: usize,
}

impl<'a, 'b, 'c, 'd> Voiced<'a, 'b, 'c, 'd> {
    pub fn new(params: &BaseParams, prev: &'a PrevFrame, phase: &'b Phase,
               enhanced: &'c EnhancedSpectrals, voice: &'d VoiceDecisions)
        -> Self
    {
        Voiced {
            prev: prev,
            phase: phase,
            enhanced: enhanced,
            voice: voice,
            window: window::synthesis(),
            fundamental: params.fundamental,
            // Compute the sum bound in Eq 127.
            end: max(params.harmonics, prev.params.harmonics) as usize,
        }
    }

    /// Compute s<sub>v,l</sub>(n), the signal level at sample n for the l'th spectral
    /// amplitude.
    fn get_pair(&self, l: usize, n: isize) -> f32 {
        match (self.voice.is_voiced(l), self.prev.voice.is_voiced(l)) {
            // Use Eq 130.
            (false, false) => 0.0,
            // Use Eq 131.
            (false, true) => self.sig_prev(l, n),
            // Use Eq 132.
            (true, false) => self.sig_cur(l, n),
            // Use Eq 133. The Eq 134 form for voiced/voiced frames isn't used due to its
            // complexity and lack of rationale.
            (true, true) => self.sig_prev(l, n) + self.sig_cur(l, n)
        }
    }

    /// Compute s<sub>v,l</sub>(n) for a voiced current frame and unvoiced previous frame.
    fn sig_cur(&self, l: usize, n: isize) -> f32 {
        // Compute Eq 132.
        self.window.get(n - SAMPLES_PER_FRAME as isize) * self.enhanced.get(l) * (
            self.fundamental * (n - SAMPLES_PER_FRAME as isize) as f32 * l as f32 +
                self.phase.get(l)
        ).cos()
    }

    /// Compute s<sub>v,l</sub>(n) for an unvoiced current frame and voiced previous frame.
    fn sig_prev(&self, l: usize, n: isize) -> f32 {
        // Compute Eq 131.
        self.window.get(n) * self.prev.enhanced.get(l) * (
            self.prev.params.fundamental * n as f32 * l as f32 +
                self.prev.phase.get(l)
        ).cos()
    }

    /// Compute the voiced signal sample s<sub>v</sub>(n) for the given sample n, 0 ≤ n <
    /// 160.
    pub fn get(&self, n: usize) -> f32 {
        debug_assert!(n < SAMPLES_PER_FRAME);

        // Compute Eq 127
        2.0 * (1...self.end)
            .map(|l| self.get_pair(l, n as isize))
            .fold(0.0, |s, x| s + x)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use params::BaseParams;
    use prev::PrevFrame;
    use descramble::Bootstrap;

    #[test]
    fn test_phase_base() {
        let chunks = [
            0b001000010010,
            0b110011001100,
            0b111000111000,
            0b111111111111,
            0b10101110101,
            0b00101111010,
            0b01110111011,
            0b00001000,
        ];

        let b = Bootstrap::new(&chunks);
        let p = BaseParams::new(b.unwrap_period());
        let prev = PrevFrame::default();

        assert!((p.fundamental - 0.17575344).abs() < 0.000001);
        assert!((prev.params.fundamental - 0.0937765407).abs() < 0.000001);

        let pb = PhaseBase::new(&p, &prev);

        assert!((pb.get(1) - 21.56239846).abs() < 0.0001);
        assert!((pb.get(2) - 43.12479691).abs() < 0.0001);
        assert!((pb.get(3) - 64.68719537).abs() < 0.0001);
        assert!((pb.get(4) - 86.24959382).abs() < 0.0001);
        assert!((pb.get(5) - 107.8119923).abs() < 0.0001);
        assert!((pb.get(6) - 129.3743907).abs() < 0.0001);
        assert!((pb.get(20) - 431.2479691).abs() < 0.0001);
        assert!((pb.get(56) - 1207.494314).abs() < 0.001);
    }
}
