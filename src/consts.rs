/// Audio samples per second
pub const SAMPLE_RATE: usize = 8000;
/// Samples per voiced/unvoiced frame
pub const SAMPLES_PER_FRAME: usize = 160;

/// Number of harmonics L when the fundamental frequency ω<sub>0</sub> is maximum.
pub const MIN_HARMONICS: usize = 9;
/// Number of harmonics L when the fundamental frequency ω<sub>0</sub> is minimum.
pub const MAX_HARMONICS: usize = 56;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_duration() {
        // Verify it's 20ms.
        assert_eq!(SAMPLES_PER_FRAME * 1000 / SAMPLE_RATE, 20);
    }
}
