pub struct Errors {
    pub total: usize,
    pub rate: f32,
    pub golay_init: usize,
    pub hamming_init: usize,
}

impl Errors {
    pub fn new(errors: &[usize; 7], prev_rate: f32) -> Errors {
        let total = errors.iter().fold(0, |s, &e| s + e);

        Errors {
            total: total,
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
