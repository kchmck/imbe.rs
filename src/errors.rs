use frame;

/// Values derived from error correction decoding.
pub struct Errors {
    /// Total number of errors corrected in the current frame, ϵ<sub>T</sub> [p45].
    pub total: usize,
    /// Error rate tracking term, ϵ<sub>R</sub> [p45].
    pub rate: f32,
    /// Errors corrected in first (u<sub>0</sub>) Golay-coded chunk, ϵ<sub>0</sub>.
    pub golay_init: usize,
    /// Errors corrected in first (u<sub>4</sub>) Hamming-coded chunk, ϵ<sub>4</sub>.
    pub hamming_init: usize,
}

impl Errors {
    /// Create a new `Errors` from the errors corrected in the current frame,
    /// ϵ<sub>i</sub>, and the previous frame's ϵ<sub>R</sub> value.
    pub fn new(errors: &frame::Errors, prev_rate: f32) -> Errors {
        let total = errors.iter().fold(0, |s, &e| s + e);

        Errors {
            total: total,
            // Compute Eq 96.
            rate: 0.95 * prev_rate + 0.000365 * total as f32,
            golay_init: errors[0],
            hamming_init: errors[4],
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_errors() {
        let e = Errors::new(&[1, 2, 3, 4, 5, 6, 7], 0.5);

        assert_eq!(e.total, 28);
        assert!((e.rate - 0.48522).abs() < 0.00001);
        assert_eq!(e.golay_init, 1);
        assert_eq!(e.hamming_init, 5);
    }
}
