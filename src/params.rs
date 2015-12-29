use std;

use std::cmp;
use std::f32::consts::PI;

#[derive(Copy, Clone)]
pub struct BaseParams {
    /// w_0
    pub fundamental: f32,
    /// L
    pub harmonics: u32,
    /// K
    pub bands: u32,
}

impl BaseParams {
    pub fn new(first_chunk: u8) -> BaseParams {
        Self::from_float(first_chunk as f32)
    }

    // Assuming period is valid
    fn from_float(period: f32) -> BaseParams {
        let f = 4.0 * PI / (period + 39.5);
        let h = (0.9254 * (PI / f + 0.25).floor()) as u32;
        let b = std::cmp::min((h + 2) / 3, 12);

        BaseParams {
            fundamental: f,
            harmonics: h,
            bands: b,
        }
    }
}

impl Default for BaseParams {
    fn default() -> BaseParams {
        BaseParams {
            fundamental: 0.02985 * PI,
            harmonics: 30,
            bands: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params() {
        let p = BaseParams::new(207);

        assert!((p.fundamental - 0.050979191).abs() < 0.00000001);
        assert_eq!(p.harmonics, 56);
        assert_eq!(p.bands, 12);

        let p = BaseParams::new(0);

        assert!((p.fundamental - 0.318135965).abs() < 0.00000001);
        assert_eq!(p.harmonics, 9);
        assert_eq!(p.bands, 3);

        let p = BaseParams::new(104);

        assert!((p.fundamental - 0.087570527).abs() < 0.00000001);
        assert_eq!(p.harmonics, 33);
        assert_eq!(p.bands, 11);
    }
}
