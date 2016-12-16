use std::cmp::min;
use std::f32::consts::PI;

/// Basic parameters of the current frame.
#[derive(Copy, Clone)]
pub struct BaseParams {
    /// Fundamental frequency ω<sub>0</sub> the frame is derived from.
    pub fundamental: f32,
    /// Number of harmonics L of the fundamental frequency present in the frame.
    pub harmonics: u32,
    /// Number of frequency bands K in the frame, each of which is classified as either
    /// voiced or unvoiced.
    ///
    /// Each band contains 3 harmonics of the fundamental frequency, except the last which
    /// may contain less [p20].
    pub bands: u32,
}

impl BaseParams {
    /// Create a new `BaseParams` from the given period b<sub>0</sub>.
    pub fn new(period: u8) -> BaseParams {
        Self::from_float(period as f32)
    }

    /// Create a new `BaseParams` from the given floating-point period b<sub>0</sub>.
    fn from_float(period: f32) -> BaseParams {
        // Compute Eq 46.
        let f = 4.0 * PI / (period + 39.5);
        // Compute Eq 47.
        let h = (0.9254 * (PI / f + 0.25).floor()) as u32;
        // Compute Eq 48.
        let b = min((h + 2) / 3, 12);

        BaseParams {
            fundamental: f,
            harmonics: h,
            bands: b,
        }
    }
}

impl Default for BaseParams {
    /// Create a new `BaseParams` with initial default values.
    fn default() -> BaseParams {
        // Taken from [p64].
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
