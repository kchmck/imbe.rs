//! Constants used in the codec.

/// Audio samples per second
pub const SAMPLE_RATE: usize = 8000;
/// Samples per voiced/unvoiced frame
pub const SAMPLES_PER_FRAME: usize = 160;

/// Number of harmonics L when the fundamental frequency ω<sub>0</sub> is maximum.
pub const MIN_HARMONICS: usize = 9;
/// Number of harmonics L when the fundamental frequency ω<sub>0</sub> is minimum.
pub const MAX_HARMONICS: usize = 56;
/// Number of discrete values that exist for the harmonics parameter.
pub const NUM_HARMONICS: usize = MAX_HARMONICS - MIN_HARMONICS + 1;

/// Maximum number of quantized amplitudes.
///
/// Since b<sub>m</sub>, 3 ≤ m ≤ L + 1, must be stored and 9 ≤ L ≤ 56, the maximum
/// capacity required is 57 - 3 + 1 = 56 - 1.
pub const MAX_QUANTIZED_AMPS: usize = MAX_HARMONICS - 1;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_duration() {
        // Verify it's 20ms.
        assert_eq!(SAMPLES_PER_FRAME * 1000 / SAMPLE_RATE, 20);
    }
}
