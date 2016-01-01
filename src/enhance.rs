use std;
use std::f32::consts::PI;

use arrayvec::ArrayVec;

use descramble::VoiceDecisions;
use errors::Errors;
use params::BaseParams;
use spectral::Spectrals;

pub struct FrameEnergy {
    pub spectral: f32,
    pub cos: f32,
    pub tracking: f32,
}

impl FrameEnergy {
    pub fn new(spectrals: &Spectrals, prev: &FrameEnergy, params: &BaseParams)
        -> FrameEnergy
    {
        let m0 = spectrals.iter().map(|&m| {
            m.powi(2)
        }).fold(0.0, |s, x| s + x);

        let m1 = spectrals.iter().enumerate().map(|(l, &m)| {
            let l = l + 1;
            m.powi(2) * (params.fundamental * l as f32).cos()
        }).fold(0.0, |s, x| s + x);

        FrameEnergy {
            spectral: m0,
            cos: m1,
            tracking: (0.95 * prev.tracking + 0.05 * m0).max(10000.0),
        }
    }
}

impl Default for FrameEnergy {
    fn default() -> FrameEnergy {
        FrameEnergy {
            spectral: 0.0,
            cos: 0.0,
            tracking: 75000.0,
        }
    }
}

#[derive(Clone)]
pub struct EnhancedSpectrals(ArrayVec<[f32; 56]>);

impl EnhancedSpectrals {
    pub fn new(spectrals: &Spectrals, energy: &FrameEnergy, params: &BaseParams)
        -> EnhancedSpectrals
    {
        let spectral_sqr = energy.spectral.powi(2);
        let cos_sqr = energy.cos.powi(2);

        let mut enhanced = spectrals.iter().enumerate().map(|(l, &m)| {
            let l = l + 1;

            let weight = m.sqrt() * (
                0.96 * PI * (
                    spectral_sqr + cos_sqr - 2.0 * energy.spectral * energy.cos *
                        (params.fundamental * l as f32).cos()
                ) / (params.fundamental * energy.spectral * (spectral_sqr - cos_sqr))
            ).powf(0.25);

            if 8 * l as u32 <= params.harmonics {
                m
            } else {
                m * weight.max(0.5).min(1.2)
            }
        }).collect::<ArrayVec<[f32; 56]>>();

        let scale = (
            energy.spectral / enhanced.iter().fold(0.0, |s, &m| s + m.powi(2))
        ).sqrt();

        for m in enhanced.iter_mut() {
            *m *= scale;
        }

        EnhancedSpectrals(enhanced)
    }

    pub fn get(&self, l: usize) -> f32 {
        if l > self.0.len() {
            0.0
        } else {
            self.0[l - 1]
        }
    }
}

impl std::ops::Deref for EnhancedSpectrals {
    type Target = ArrayVec<[f32; 56]>;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl std::ops::DerefMut for EnhancedSpectrals {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl Default for EnhancedSpectrals {
    fn default() -> EnhancedSpectrals {
        EnhancedSpectrals(ArrayVec::new())
    }
}

pub fn amp_thresh(errors: &Errors, prev: f32) -> f32 {
    if errors.rate <= 0.005 && errors.total <= 6 {
        20480.0
    } else {
        6000.0 - 300.0 * errors.total as f32 + prev
    }
}

pub fn smooth(enhanced: &mut EnhancedSpectrals, voiced: &mut VoiceDecisions,
              errors: &Errors, energy: &FrameEnergy, amp_thresh: f32)
{
    let thresh = if errors.rate <= 0.005 && errors.total <= 4 {
        std::f32::MAX
    } else if errors.rate <= 0.0125 && errors.hamming_init == 0 {
        45.255 * energy.tracking.powf(0.375) / (277.26 * errors.rate).exp()
    } else {
        1.414 * energy.tracking.powf(0.375)
    };

    for (l, &m) in enhanced.iter().enumerate() {
        if m > thresh {
            voiced.force_voiced(l + 1);
        }
    }

    let amp = enhanced.iter().fold(0.0, |s, &m| s + m);
    let scale = (amp_thresh / amp).max(1.0);

    for m in enhanced.iter_mut() {
        *m *= scale;
    }
}

pub fn should_repeat(errors: &Errors) -> bool {
    errors.golay_init >= 2 && errors.total as f32 >= 10.0 + 40.0 * errors.rate
}

pub fn should_mute(errors: &Errors) -> bool {
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
    fn test_spectrals() {
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

        assert!((e.get(17) - 0.0).abs() < 0.00001);
        assert!((e.get(30) - 0.0).abs() < 0.00001);
        assert!((e.get(56) - 0.0).abs() < 0.00001);
        assert!((e.get(56) - 0.0).abs() < 0.00001);
    }
}
