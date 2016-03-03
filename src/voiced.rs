use std::cmp::max;

use collect_slice::CollectSlice;

use consts::SAMPLES;
use descramble::VoiceDecisions;
use enhance::EnhancedSpectrals;
use noise::Noise;
use params::BaseParams;
use prev::PrevFrame;
use window;

pub struct PhaseBase([f32; 56]);

impl PhaseBase {
    pub fn new(params: &BaseParams, prev: &PrevFrame) -> PhaseBase {
        let mut phase_base = [0.0; 56];
        let common = (prev.params.fundamental + params.fundamental) *
            SAMPLES as f32 / 2.0;

        (1..57).map(|l| {
            prev.phase_base.get(l) + common * l as f32
        }).collect_slice_checked(&mut phase_base[..]);

        PhaseBase(phase_base)
    }

    pub fn get(&self, l: usize) -> f32 { self.0[l - 1] }
}


impl Default for PhaseBase {
    fn default() -> PhaseBase {
        PhaseBase([0.0; 56])
    }
}

pub struct Phase([f32; 56]);

impl Phase {
    pub fn new(params: &BaseParams, voice: &VoiceDecisions, base: &PhaseBase) -> Phase {
        let mut noise = Noise::new();
        let mut phase = [0.0; 56];

        let trans = params.harmonics as usize / 4;

        (1..trans+1).map(|l| {
            base.get(l)
        }).chain((trans+1..57).map(|l| {
            base.get(l) + voice.unvoiced_count as f32 * noise.next_phase() /
                params.harmonics as f32
        })).collect_slice_checked(&mut phase[..]);

        Phase(phase)
    }

    pub fn get(&self, l: usize) -> f32 { self.0[l - 1] }
}

impl Default for Phase {
    fn default() -> Phase {
        Phase([0.0; 56])
    }
}

pub struct Voiced<'a, 'b, 'c, 'd> {
    prev: &'a PrevFrame,
    phase: &'b Phase,
    enhanced: &'c EnhancedSpectrals,
    voice: &'d VoiceDecisions,
    window: window::Window,
    fundamental: f32,
    end: usize,
}

impl<'a, 'b, 'c, 'd> Voiced<'a, 'b, 'c, 'd> {
    pub fn new(params: &BaseParams, prev: &'a PrevFrame, phase: &'b Phase,
               enhanced: &'c EnhancedSpectrals, voice: &'d VoiceDecisions)
        -> Voiced<'a, 'b, 'c, 'd>
    {
        Voiced {
            prev: prev,
            phase: phase,
            enhanced: enhanced,
            voice: voice,
            window: window::synthesis_full(),
            fundamental: params.fundamental,
            end: max(params.harmonics, prev.params.harmonics) as usize + 1,
        }
    }

    fn get_pair(&self, l: usize, n: isize) -> f32 {
        match (self.voice.is_voiced(l), self.prev.voice.is_voiced(l)) {
            (false, false) => 0.0,
            (false, true) => self.sig_prev(l, n),
            (true, false) => self.sig_cur(l, n),
            (true, true) => self.sig_prev(l, n) + self.sig_cur(l, n)
        }
    }

    fn sig_cur(&self, l: usize, n: isize) -> f32 {
        self.window.get(n - SAMPLES as isize) * self.enhanced.get(l) * (
            self.fundamental * (n - SAMPLES as isize) as f32 * l as f32 +
                self.phase.get(l)
        ).cos()
    }

    fn sig_prev(&self, l: usize, n: isize) -> f32 {
        self.window.get(n) * self.prev.enhanced.get(l) * (
            self.prev.params.fundamental * n as f32 * l as f32 +
                self.prev.phase.get(l)
        ).cos()
    }

    pub fn get(&self, n: usize) -> f32 {
        2.0 * (1..self.end)
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
