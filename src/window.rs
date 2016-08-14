pub fn synthesis_full() -> Window {
    Window::new(&WINDOW_SYNTHESIS[..])
}

pub struct Window {
    coefs: &'static [f32],
    offset: isize,
}

impl Window {
    pub fn new(coefs: &'static [f32]) -> Window {
        Window {
            coefs: coefs,
            offset: coefs.len() as isize / 2,
        }
    }

    pub fn get(&self, n: isize) -> f32 {
        match self.coefs.get((n + self.offset) as usize) {
            Some(&coef) => coef,
            None => 0.0,
        }
    }
}

// w_S(n)
static WINDOW_SYNTHESIS: [f32; 211] = [
    0.000000,
    0.020000,
    0.040000,
    0.060000,
    0.080000,
    0.100000,
    0.120000,
    0.140000,
    0.160000,
    0.180000,
    0.200000,
    0.220000,
    0.240000,
    0.260000,
    0.280000,
    0.300000,
    0.320000,
    0.340000,
    0.360000,
    0.380000,
    0.400000,
    0.420000,
    0.440000,
    0.460000,
    0.480000,
    0.500000,
    0.520000,
    0.540000,
    0.560000,
    0.580000,
    0.60000,
    0.620000,
    0.640000,
    0.660000,
    0.680000,
    0.700000,
    0.720000,
    0.740000,
    0.760000,
    0.780000,
    0.800000,
    0.820000,
    0.840000,
    0.860000,
    0.880000,
    0.900000,
    0.920000,
    0.940000,
    0.960000,
    0.980000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.00000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    1.000000,
    0.980000,
    0.960000,
    0.940000,
    0.920000,
    0.900000,
    0.880000,
    0.860000,
    0.840000,
    0.820000,
    0.800000,
    0.780000,
    0.760000,
    0.740000,
    0.720000,
    0.700000,
    0.680000,
    0.660000,
    0.640000,
    0.620000,
    0.600000,
    0.580000,
    0.560000,
    0.540000,
    0.520000,
    0.500000,
    0.480000,
    0.460000,
    0.440000,
    0.420000,
    0.400000,
    0.380000,
    0.360000,
    0.340000,
    0.320000,
    0.300000,
    0.280000,
    0.260000,
    0.240000,
    0.220000,
    0.200000,
    0.180000,
    0.160000,
    0.140000,
    0.120000,
    0.100000,
    0.080000,
    0.060000,
    0.040000,
    0.020000,
    0.00000,
];

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_syn_full() {
        let w = synthesis_full();

        assert_eq!(w.get(-200), 0.0);
        assert_eq!(w.get(-106), 0.0);
        assert_eq!(w.get(-105), 0.0);
        assert_eq!(w.get(-104), 0.02);
        assert_eq!(w.get(-68), 0.74);
        assert_eq!(w.get(0), 1.0);
        assert_eq!(w.get(77), 0.56);
        assert_eq!(w.get(104), 0.02);
        assert_eq!(w.get(105), 0.0);
        assert_eq!(w.get(106), 0.0);
        assert_eq!(w.get(200), 0.0);
    }
}
