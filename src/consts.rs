// Audio samples per second
pub const SAMPLE_RATE: usize = 8000;

// Samples per voiced/unvoiced segment
pub const SAMPLES: usize = 160;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_duration() {
        // Verify it's 20ms.
        assert_eq!(SAMPLES * 1000 / SAMPLE_RATE, 20);
    }
}
