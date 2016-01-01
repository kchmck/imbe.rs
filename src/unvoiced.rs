use std::f32::consts::PI;

use num::complex::Complex32;
use collect_slice::CollectSlice;

use consts::SAMPLES;
use descramble::VoiceDecisions;
use enhance::EnhancedSpectrals;
use noise::Noise;
use params::BaseParams;
use window;

// Result of equation 121
const SCALING_COEF: f32 = 146.6432708443356;

pub struct UnvoicedDFT([Complex32; 256]);

impl UnvoicedDFT {
    pub fn new() -> UnvoicedDFT {
        let mut dft = [Complex32::new(0.0, 0.0); 256];
        let window = window::synthesis_trunc();

        (-128..128).map(|m| {
            let mut noise = Noise::new();

            (-104..105).map(|n| {
                noise.next() as f32 * window.get(n) *
                    Complex32::new(0.0, -2.0 / 256.0 * PI * m as f32 * n as f32).exp()
            }).fold(Complex32::new(0.0, 0.0), |s, x| s + x)
        }).collect_slice_checked(&mut dft[..]);

        UnvoicedDFT(dft)
    }

    // -128 <= m < 128
    pub fn get(&self, m: isize) -> Complex32 { self.0[(m + 128) as usize] }

    pub fn scale(&self, lower: isize, upper: isize, spectral: f32) -> f32 {
        let sum = (lower..upper).map(|n| {
           self.get(n).norm_sqr()
        }).fold(0.0, |s, x| s + x);

        SCALING_COEF * spectral /
            (sum / (upper - lower) as f32).sqrt()
    }
}

fn edges(l: usize, params: &BaseParams) -> (isize, isize) {
    let edge = |inner: f32| {
        256.0 / (2.0 * PI) * inner * params.fundamental
    };

    (
        edge(l as f32 - 0.5).ceil() as isize,
        edge(l as f32 + 0.5).ceil() as isize,
    )
}

pub struct UnvoicedParts([Complex32; 256]);

impl UnvoicedParts {
    pub fn new(dft: &UnvoicedDFT, params: &BaseParams, voice: &VoiceDecisions,
               enhanced: &EnhancedSpectrals)
        -> UnvoicedParts
    {
        let mut parts = [Complex32::new(0.0, 0.0); 256];

        for (l, &m) in enhanced.iter().enumerate() {
            let l = l + 1;

            if voice.is_voiced(l) {
                continue;
            }

            let (lower, upper) = edges(l, params);
            let scale = dft.scale(lower, upper, m);

            (lower..upper).map(|m| {
                scale * dft.get(m)
            }).collect_slice_checked(&mut parts[128 + lower as usize..128 + upper as usize]);

            (lower..upper).rev().map(|m| {
                scale * dft.get(-m)
            }).collect_slice_checked(&mut parts[128 - upper as usize + 1..128 - lower as usize + 1]);
        }

        UnvoicedParts(parts)
    }

    // -128 <= m < 128
    fn get(&self, m: isize) -> Complex32 { self.0[(m + 128) as usize] }

    pub fn idft(&self, n: isize) -> f32 {
        if n < -128 || n > 127 {
            return 0.0;
        }

        (-128..128).map(|m| {
            self.get(m) *
                Complex32::new(0.0, 2.0 / 256.0 * PI * m as f32 * n as f32).exp()
        }).fold(Complex32::new(0.0, 0.0), |s, x| s + x).re / 256.0
    }
}

impl Default for UnvoicedParts {
    fn default() -> UnvoicedParts {
        UnvoicedParts([Complex32::new(0.0, 0.0); 256])
    }
}

pub struct Unvoiced<'a, 'b> {
    cur: &'a UnvoicedParts,
    prev: &'b UnvoicedParts,
    window: window::Window,
}

impl<'a, 'b> Unvoiced<'a, 'b> {
    pub fn new(cur: &'a UnvoicedParts, prev: &'b UnvoicedParts) -> Unvoiced<'a, 'b> {
        Unvoiced {
            cur: cur,
            prev: prev,
            window: window::synthesis_full(),
        }
    }

    pub fn get(&self, n: usize) -> f32 {
        let n = n as isize;

        let numer = self.window.get(n) * self.prev.idft(n) +
            self.window.get(n - SAMPLES as isize) * self.cur.idft(n - SAMPLES as isize);
        let denom = self.window.get(n).powi(2) +
            self.window.get(n - SAMPLES as isize).powi(2);

        numer / denom
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use super::{edges};
    use spectral::Spectrals;
    use descramble::{descramble, Bootstrap};
    use params::BaseParams;
    use gain::Gains;
    use coefs::Coefficients;
    use prev::{PrevFrame};

    #[test]
    fn test_dft() {
        let d = UnvoicedDFT::new();

        assert!((d.scale(54, 61, 0.5306769781475001) - 0.0003583616704623).abs() < 0.000001);
        assert!((d.scale(26, 33, 0.13169278935782622) - 0.0001140735292).abs() < 0.000001);

        assert!((d.get(-128).re - 235025.0).abs() < 0.1);
        assert!((d.get(-127).re - 204767.0).abs() < 10.0);
        assert!((d.get(-126).re - -16972.0).abs() < 10.0);
        assert!((d.get(-125).re - -155453.0).abs() < 1.0);
        assert!((d.get(-124).re - 28858.0).abs() < 1.0);
        assert!((d.get(-123).re - 53777.0).abs() < 1.0);
        assert!((d.get(-122).re - -115639.0).abs() < 1.0);
        assert!((d.get(-121).re - -54318.0).abs() < 1.0);
        assert!((d.get(-120).re - 206756.0).abs() < 10.0);
        assert!((d.get(-119).re - 244318.0).abs() < 10.0);
        assert!((d.get(-118).re - 11846.0).abs() < 10.0);
        assert!((d.get(-117).re - -13590.0).abs() < 1.0);
        assert!((d.get(-116).re - 150657.0).abs() < 1.0);
        assert!((d.get(-115).re - 133462.0).abs() < 10.0);
        assert!((d.get(-114).re - 36336.0).abs() < 1.0);
        assert!((d.get(-113).re - 36149.0).abs() < 1.0);
        assert!((d.get(-112).re - 48478.0).abs() < 1.0);
        assert!((d.get(-111).re - -46891.0).abs() < 1.0);
        assert!((d.get(-110).re - -143162.0).abs() < 1.0);
        assert!((d.get(-109).re - -199670.0).abs() < 1.0);
        assert!((d.get(-108).re - -329288.0).abs() < 1.0);
        assert!((d.get(-107).re - -222079.0).abs() < 10.0);
        assert!((d.get(-106).re - 107314.0).abs() < 1.0);
        assert!((d.get(-105).re - 93590.0).abs() < 10.0);
        assert!((d.get(-104).re - -93289.0).abs() < 1.0);
        assert!((d.get(-103).re - -53311.0).abs() < 1.0);
        assert!((d.get(-102).re - 29021.0).abs() < 1.0);
        assert!((d.get(-101).re - 43992.0).abs() < 1.0);
        assert!((d.get(-100).re - 67228.0).abs() < 1.0);
        assert!((d.get(-99).re - 4294.0).abs() < 1.0);
        assert!((d.get(-98).re - -124886.0).abs() < 1.0);
        assert!((d.get(-97).re - -83746.0).abs() < 1.0);
        assert!((d.get(-96).re - -16072.0).abs() < 1.0);
        assert!((d.get(-95).re - -99393.0).abs() < 1.0);
        assert!((d.get(-94).re - -52143.0).abs() < 1.0);
        assert!((d.get(-93).re - 40618.0).abs() < 1.0);
        assert!((d.get(-92).re - 28599.0).abs() < 1.0);
        assert!((d.get(-91).re - 97223.0).abs() < 10.0);
        assert!((d.get(-90).re - 38978.0).abs() < 1.0);
        assert!((d.get(-89).re - -76216.0).abs() < 1.0);
        assert!((d.get(-88).re - 23148.0).abs() < 1.0);
        assert!((d.get(-87).re - -27079.0).abs() < 10.0);
        assert!((d.get(-86).re - -127182.0).abs() < 1.0);
        assert!((d.get(-85).re - 29030.0).abs() < 1.0);
        assert!((d.get(-84).re - -6887.0).abs() < 10.0);
        assert!((d.get(-83).re - -171482.0).abs() < 1.0);
        assert!((d.get(-82).re - 11076.0).abs() < 1.0);
        assert!((d.get(-81).re - 171115.0).abs() < 1.0);
        assert!((d.get(-80).re - -2239.0).abs() < 10.0);
        assert!((d.get(-79).re - -128365.0).abs() < 1.0);
        assert!((d.get(-78).re - -64421.0).abs() < 1.0);
        assert!((d.get(-77).re - -3439.0).abs() < 10.0);
        assert!((d.get(-76).re - 7615.0).abs() < 1.0);
        assert!((d.get(-75).re - 7714.0).abs() < 1.0);
        assert!((d.get(-74).re - 29699.0).abs() < 1.0);
        assert!((d.get(-73).re - 38681.0).abs() < 1.0);
        assert!((d.get(-72).re - -37952.0).abs() < 1.0);
        assert!((d.get(-71).re - -141903.0).abs() < 1.0);
        assert!((d.get(-70).re - -119887.0).abs() < 1.0);
        assert!((d.get(-69).re - 61739.0).abs() < 1.0);
        assert!((d.get(-68).re - 110538.0).abs() < 1.0);
        assert!((d.get(-67).re - -96271.0).abs() < 1.0);
        assert!((d.get(-66).re - -173994.0).abs() < 1.0);
        assert!((d.get(-65).re - -101631.0).abs() < 1.0);
        assert!((d.get(-64).re - -155533.0).abs() < 10.0);
        assert!((d.get(-63).re - -165171.0).abs() < 1.0);
        assert!((d.get(-62).re - -38513.0).abs() < 1.0);
        assert!((d.get(-61).re - 10481.0).abs() < 1.0);
        assert!((d.get(-60).re - -48193.0).abs() < 1.0);
        assert!((d.get(-59).re - -104421.0).abs() < 1.0);
        assert!((d.get(-58).re - -187070.0).abs() < 1.0);
        assert!((d.get(-57).re - -262793.0).abs() < 1.0);
        assert!((d.get(-56).re - -149096.0).abs() < 1.0);
        assert!((d.get(-55).re - -34737.0).abs() < 1.0);
        assert!((d.get(-54).re - -89815.0).abs() < 1.0);
        assert!((d.get(-53).re - -57320.0).abs() < 1.0);
        assert!((d.get(-52).re - 15252.0).abs() < 1.0);
        assert!((d.get(-51).re - -2038.0).abs() < 1.0);
        assert!((d.get(-50).re - -18749.0).abs() < 1.0);
        assert!((d.get(-49).re - -54613.0).abs() < 1.0);
        assert!((d.get(-48).re - -810.0).abs() < 1.0);
        assert!((d.get(-47).re - 105401.0).abs() < 1.0);
        assert!((d.get(-46).re - 15181.0).abs() < 1.0);
        assert!((d.get(-45).re - -102133.0).abs() < 1.0);
        assert!((d.get(-44).re - -39030.0).abs() < 1.0);
        assert!((d.get(-43).re - 10075.0).abs() < 1.0);
        assert!((d.get(-42).re - -3576.0).abs() < 1.0);
        assert!((d.get(-41).re - 83869.0).abs() < 1.0);
        assert!((d.get(-40).re - 136404.0).abs() < 1.0);
        assert!((d.get(-39).re - 3723.0).abs() < 1.0);
        assert!((d.get(-38).re - -173305.0).abs() < 1.0);
        assert!((d.get(-37).re - -188825.0).abs() < 1.0);
        assert!((d.get(-36).re - 677.0).abs() < 1.0);
        assert!((d.get(-35).re - 82362.0).abs() < 1.0);
        assert!((d.get(-34).re - 8002.0).abs() < 1.0);
        assert!((d.get(-33).re - 48220.0).abs() < 1.0);
        assert!((d.get(-32).re - 70574.0).abs() < 1.0);
        assert!((d.get(-31).re - 63688.0).abs() < 1.0);
        assert!((d.get(-30).re - 120615.0).abs() < 1.0);
        assert!((d.get(-29).re - 38190.0).abs() < 1.0);
        assert!((d.get(-28).re - -160753.0).abs() < 1.0);
        assert!((d.get(-27).re - -244761.0).abs() < 1.0);
        assert!((d.get(-26).re - -76669.0).abs() < 1.0);
        assert!((d.get(-25).re - 89170.0).abs() < 1.0);
        assert!((d.get(-24).re - -12639.0).abs() < 1.0);
        assert!((d.get(-23).re - -46685.0).abs() < 1.0);
        assert!((d.get(-22).re - 90717.0).abs() < 1.0);
        assert!((d.get(-21).re - 79243.0).abs() < 1.0);
        assert!((d.get(-20).re - -42392.0).abs() < 1.0);
        assert!((d.get(-19).re - -123994.0).abs() < 1.0);
        assert!((d.get(-18).re - -208514.0).abs() < 1.0);
        assert!((d.get(-17).re - -177209.0).abs() < 1.0);
        assert!((d.get(-16).re - 38245.0).abs() < 1.0);
        assert!((d.get(-15).re - 165885.0).abs() < 1.0);
        assert!((d.get(-14).re - 99505.0).abs() < 1.0);
        assert!((d.get(-13).re - -64928.0).abs() < 1.0);
        assert!((d.get(-12).re - -148374.0).abs() < 1.0);
        assert!((d.get(-11).re - -95575.0).abs() < 1.0);
        assert!((d.get(-10).re - -91385.0).abs() < 1.0);
        assert!((d.get(-9).re - -2725.0).abs() < 1.0);
        assert!((d.get(-8).re - 32137.0).abs() < 1.0);
        assert!((d.get(-7).re - 3768.0).abs() < 1.0);
        assert!((d.get(-6).re - 169325.0).abs() < 1.0);
        assert!((d.get(-5).re - 110058.0).abs() < 1.0);
        assert!((d.get(-4).re - 165417.0).abs() < 1.0);
        assert!((d.get(-3).re - -143750.0).abs() < 1.0);
        assert!((d.get(-2).re - -623185.0).abs() < 1.0);
        assert!((d.get(-1).re - 1899880.0).abs() < 1.0);
        assert!((d.get(0).re - 4367745.0).abs() < 1.0);
        assert!((d.get(1).re - 1899880.0).abs() < 1.0);
        assert!((d.get(2).re - -623185.0).abs() < 1.0);
        assert!((d.get(3).re - -143750.0).abs() < 1.0);
        assert!((d.get(4).re - 165417.0).abs() < 1.0);
        assert!((d.get(5).re - 110058.0).abs() < 1.0);
        assert!((d.get(6).re - 169325.0).abs() < 1.0);
        assert!((d.get(7).re - 3768.0).abs() < 1.0);
        assert!((d.get(8).re - 32137.0).abs() < 1.0);
        assert!((d.get(9).re - -2725.0).abs() < 1.0);
        assert!((d.get(10).re - -91385.0).abs() < 1.0);
        assert!((d.get(11).re - -95575.0).abs() < 1.0);
        assert!((d.get(12).re - -148374.0).abs() < 1.0);
        assert!((d.get(13).re - -64928.0).abs() < 1.0);
        assert!((d.get(14).re - 99505.0).abs() < 1.0);
        assert!((d.get(15).re - 165885.0).abs() < 1.0);
        assert!((d.get(16).re - 38245.0).abs() < 1.0);
        assert!((d.get(17).re - -177209.0).abs() < 1.0);
        assert!((d.get(18).re - -208514.0).abs() < 1.0);
        assert!((d.get(19).re - -123994.0).abs() < 1.0);
        assert!((d.get(20).re - -42392.0).abs() < 1.0);
        assert!((d.get(21).re - 79243.0).abs() < 1.0);
        assert!((d.get(22).re - 90717.0).abs() < 1.0);
        assert!((d.get(23).re - -46685.0).abs() < 1.0);
        assert!((d.get(24).re - -12639.0).abs() < 1.0);
        assert!((d.get(25).re - 89170.0).abs() < 1.0);
        assert!((d.get(26).re - -76669.0).abs() < 1.0);
        assert!((d.get(27).re - -244761.0).abs() < 1.0);
        assert!((d.get(28).re - -160753.0).abs() < 1.0);
        assert!((d.get(29).re - 38190.0).abs() < 1.0);
        assert!((d.get(30).re - 120615.0).abs() < 1.0);
        assert!((d.get(31).re - 63688.0).abs() < 1.0);
        assert!((d.get(32).re - 70574.0).abs() < 1.0);
        assert!((d.get(33).re - 48220.0).abs() < 1.0);
        assert!((d.get(34).re - 8002.0).abs() < 1.0);
        assert!((d.get(35).re - 82362.0).abs() < 1.0);
        assert!((d.get(36).re - 677.0).abs() < 1.0);
        assert!((d.get(37).re - -188825.0).abs() < 1.0);
        assert!((d.get(38).re - -173305.0).abs() < 1.0);
        assert!((d.get(39).re - 3723.0).abs() < 1.0);
        assert!((d.get(40).re - 136404.0).abs() < 1.0);
        assert!((d.get(41).re - 83869.0).abs() < 1.0);
        assert!((d.get(42).re - -3576.0).abs() < 1.0);
        assert!((d.get(43).re - 10075.0).abs() < 1.0);
        assert!((d.get(44).re - -39030.0).abs() < 1.0);
        assert!((d.get(45).re - -102133.0).abs() < 1.0);
        assert!((d.get(46).re - 15181.0).abs() < 1.0);
        assert!((d.get(47).re - 105401.0).abs() < 1.0);
        assert!((d.get(48).re - -810.0).abs() < 1.0);
        assert!((d.get(49).re - -54613.0).abs() < 1.0);
        assert!((d.get(50).re - -18749.0).abs() < 1.0);
        assert!((d.get(51).re - -2038.0).abs() < 1.0);
        assert!((d.get(52).re - 15252.0).abs() < 1.0);
        assert!((d.get(53).re - -57320.0).abs() < 1.0);
        assert!((d.get(54).re - -89815.0).abs() < 1.0);
        assert!((d.get(55).re - -34737.0).abs() < 1.0);
        assert!((d.get(56).re - -149096.0).abs() < 1.0);
        assert!((d.get(57).re - -262793.0).abs() < 1.0);
        assert!((d.get(58).re - -187070.0).abs() < 1.0);
        assert!((d.get(59).re - -104421.0).abs() < 1.0);
        assert!((d.get(60).re - -48193.0).abs() < 1.0);
        assert!((d.get(61).re - 10481.0).abs() < 1.0);
        assert!((d.get(62).re - -38513.0).abs() < 1.0);
        assert!((d.get(63).re - -165171.0).abs() < 1.0);
        assert!((d.get(64).re - -155533.0).abs() < 10.0);
        assert!((d.get(65).re - -101631.0).abs() < 1.0);
        assert!((d.get(66).re - -173994.0).abs() < 1.0);
        assert!((d.get(67).re - -96271.0).abs() < 1.0);
        assert!((d.get(68).re - 110538.0).abs() < 1.0);
        assert!((d.get(69).re - 61739.0).abs() < 1.0);
        assert!((d.get(70).re - -119887.0).abs() < 1.0);
        assert!((d.get(71).re - -141903.0).abs() < 1.0);
        assert!((d.get(72).re - -37952.0).abs() < 1.0);
        assert!((d.get(73).re - 38681.0).abs() < 1.0);
        assert!((d.get(74).re - 29699.0).abs() < 1.0);
        assert!((d.get(75).re - 7714.0).abs() < 1.0);
        assert!((d.get(76).re - 7615.0).abs() < 1.0);
        assert!((d.get(77).re - -3439.0).abs() < 10.0);
        assert!((d.get(78).re - -64421.0).abs() < 1.0);
        assert!((d.get(79).re - -128365.0).abs() < 1.0);
        assert!((d.get(80).re - -2239.0).abs() < 10.0);
        assert!((d.get(81).re - 171115.0).abs() < 1.0);
        assert!((d.get(82).re - 11076.0).abs() < 1.0);
        assert!((d.get(83).re - -171482.0).abs() < 1.0);
        assert!((d.get(84).re - -6887.0).abs() < 10.0);
        assert!((d.get(85).re - 29030.0).abs() < 1.0);
        assert!((d.get(86).re - -127182.0).abs() < 1.0);
        assert!((d.get(87).re - -27079.0).abs() < 10.0);
        assert!((d.get(88).re - 23148.0).abs() < 1.0);
        assert!((d.get(89).re - -76216.0).abs() < 1.0);
        assert!((d.get(90).re - 38978.0).abs() < 1.0);
        assert!((d.get(91).re - 97223.0).abs() < 10.0);
        assert!((d.get(92).re - 28599.0).abs() < 1.0);
        assert!((d.get(93).re - 40618.0).abs() < 1.0);
        assert!((d.get(94).re - -52143.0).abs() < 1.0);
        assert!((d.get(95).re - -99393.0).abs() < 1.0);
        assert!((d.get(96).re - -16072.0).abs() < 1.0);
        assert!((d.get(97).re - -83746.0).abs() < 1.0);
        assert!((d.get(98).re - -124886.0).abs() < 1.0);
        assert!((d.get(99).re - 4294.0).abs() < 1.0);
        assert!((d.get(100).re - 67228.0).abs() < 1.0);
        assert!((d.get(101).re - 43992.0).abs() < 1.0);
        assert!((d.get(102).re - 29021.0).abs() < 1.0);
        assert!((d.get(103).re - -53311.0).abs() < 1.0);
        assert!((d.get(104).re - -93289.0).abs() < 1.0);
        assert!((d.get(105).re - 93590.0).abs() < 10.0);
        assert!((d.get(106).re - 107314.0).abs() < 1.0);
        assert!((d.get(107).re - -222079.0).abs() < 10.0);
        assert!((d.get(108).re - -329288.0).abs() < 1.0);
        assert!((d.get(109).re - -199670.0).abs() < 1.0);
        assert!((d.get(110).re - -143162.0).abs() < 1.0);
        assert!((d.get(111).re - -46891.0).abs() < 1.0);
        assert!((d.get(112).re - 48478.0).abs() < 1.0);
        assert!((d.get(113).re - 36149.0).abs() < 1.0);
        assert!((d.get(114).re - 36336.0).abs() < 1.0);
        assert!((d.get(115).re - 133462.0).abs() < 10.0);
        assert!((d.get(116).re - 150657.0).abs() < 1.0);
        assert!((d.get(117).re - -13590.0).abs() < 1.0);
        assert!((d.get(118).re - 11846.0).abs() < 10.0);
        assert!((d.get(119).re - 244318.0).abs() < 10.0);
        assert!((d.get(120).re - 206756.0).abs() < 10.0);
        assert!((d.get(121).re - -54318.0).abs() < 1.0);
        assert!((d.get(122).re - -115639.0).abs() < 1.0);
        assert!((d.get(123).re - 53777.0).abs() < 1.0);
        assert!((d.get(124).re - 28858.0).abs() < 1.0);
        assert!((d.get(125).re - -155453.0).abs() < 1.0);
        assert!((d.get(126).re - -16972.0).abs() < 10.0);
        assert!((d.get(127).re - 204767.0).abs() < 10.0);
    }

    #[test]
    fn test_edges() {
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

        assert!((p.fundamental - 0.17575344).abs() < 0.000001);

        let (lower, upper) = edges(1, &p);
        assert_eq!(lower, 4);
        assert_eq!(upper, 11);

        let (lower, upper) = edges(2, &p);
        assert_eq!(lower, 11);
        assert_eq!(upper, 18);

        let (lower, upper) = edges(5, &p);
        assert_eq!(lower, 33);
        assert_eq!(upper, 40);

        let (lower, upper) = edges(8, &p);
        assert_eq!(lower, 54);
        assert_eq!(upper, 61);

        let (lower, upper) = edges(16, &p);
        assert_eq!(lower, 111);
        assert_eq!(upper, 119);
    }

    #[test]
    fn test_parts() {
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
        let (amps, voice, gain_idx) = descramble(&chunks, &p);

        assert!(voice.is_voiced(1));
        assert!(voice.is_voiced(2));
        assert!(voice.is_voiced(3));
        assert!(!voice.is_voiced(4));
        assert!(!voice.is_voiced(5));
        assert!(!voice.is_voiced(6));
        assert!(voice.is_voiced(7));
        assert!(voice.is_voiced(8));
        assert!(voice.is_voiced(9));
        assert!(!voice.is_voiced(10));
        assert!(!voice.is_voiced(11));
        assert!(!voice.is_voiced(12));
        assert!(voice.is_voiced(13));
        assert!(voice.is_voiced(14));
        assert!(voice.is_voiced(15));
        assert!(voice.is_voiced(16));

        let g = Gains::new(gain_idx, &amps, &p);
        let c = Coefficients::new(&g, &amps, &p);
        let prev = PrevFrame::default();
        let s = Spectrals::new(&c, &p, &prev);

        let dft = UnvoicedDFT::new();
        let u = UnvoicedParts::new(&dft, &p, &voice, &s);

        for i in 0..26 {
            assert_eq!(u.get(i).re, 0.0);
            assert_eq!(u.get(-i).re, 0.0);
        }

        assert!((u.get(26).re - -8.745903410231604).abs() < 0.0001);
        assert!((u.get(-26).re - -8.745903410231604).abs() < 0.0001);
        assert!((u.get(27).re - -27.920751080510996).abs() < 0.0001);
        assert!((u.get(-27).re - -27.920751080510996).abs() < 0.0001);
        assert!((u.get(28).re - -18.3376620394809).abs() < 0.0001);
        assert!((u.get(-28).re - -18.3376620394809).abs() < 0.0001);
        assert!((u.get(29).re - 4.356468080146408).abs() < 0.0001);
        assert!((u.get(-29).re - 4.356468080146408).abs() < 0.0001);
        assert!((u.get(30).re - 13.758978724452971).abs() < 0.0001);
        assert!((u.get(-30).re - 13.758978724452971).abs() < 0.0001);
        assert!((u.get(31).re - 7.265114927686945).abs() < 0.0001);
        assert!((u.get(-31).re - 7.265114927686945).abs() < 0.0001);
        assert!((u.get(32).re - 8.050625249757857).abs() < 0.0001);
        assert!((u.get(-32).re - 8.050625249757857).abs() < 0.0001);

        assert!((u.get(33).re - 139.75211285496735).abs() < 0.01);
        assert!((u.get(-33).re - 139.75211285496735).abs() < 0.01);
        assert!((u.get(34).re - 23.19154722242739).abs() < 0.01);
        assert!((u.get(-34).re - 23.19154722242739).abs() < 0.01);
        assert!((u.get(35).re - 238.70310076650395).abs() < 0.01);
        assert!((u.get(-35).re - 238.70310076650395).abs() < 0.01);
        assert!((u.get(36).re - 1.9620941601578783).abs() < 0.01);
        assert!((u.get(-36).re - 1.9620941601578783).abs() < 0.01);
        assert!((u.get(37).re - -547.2561739908588).abs() < 0.01);
        assert!((u.get(-37).re - -547.2561739908588).abs() < 0.01);
        assert!((u.get(38).re - -502.27581746848017).abs() < 0.01);
        assert!((u.get(-38).re - -502.27581746848017).abs() < 0.01);
        assert!((u.get(39).re - 10.790068771444284).abs() < 0.01);
        assert!((u.get(-39).re - 10.790068771444284).abs() < 0.01);

        assert!((u.get(40).re - 108.69373521423314).abs() < 0.01);
        assert!((u.get(-40).re - 108.69373521423314).abs() < 0.01);
        assert!((u.get(41).re - 66.83114042610569).abs() < 0.01);
        assert!((u.get(-41).re - 66.83114042610569).abs() < 0.01);
        assert!((u.get(42).re - -2.84954104810781).abs() < 0.01);
        assert!((u.get(-42).re - -2.84954104810781).abs() < 0.01);
        assert!((u.get(43).re - 8.028279099464818).abs() < 0.01);
        assert!((u.get(-43).re - 8.028279099464818).abs() < 0.01);
        assert!((u.get(44).re - -31.101114962988767).abs() < 0.01);
        assert!((u.get(-44).re - -31.101114962988767).abs() < 0.01);
        assert!((u.get(45).re - -81.38483665167645).abs() < 0.01);
        assert!((u.get(-45).re - -81.38483665167645).abs() < 0.01);
        assert!((u.get(46).re - 12.097002978558352).abs() < 0.01);
        assert!((u.get(-46).re - 12.097002978558352).abs() < 0.01);

        for i in 47..69 {
            assert_eq!(u.get(i).re, 0.0);
            assert_eq!(u.get(-i).re, 0.0);
        }

        assert!((u.get(69).re - 84.4070392887328).abs() < 0.01);
        assert!((u.get(-69).re - 84.4070392887328).abs() < 0.01);
        assert!((u.get(70).re - -163.90461003916988).abs() < 0.01);
        assert!((u.get(-70).re - -163.90461003916988).abs() < 0.01);
        assert!((u.get(71).re - -194.0039860734552).abs() < 0.01);
        assert!((u.get(-71).re - -194.0039860734552).abs() < 0.01);
        assert!((u.get(72).re - -51.88642438468371).abs() < 0.01);
        assert!((u.get(-72).re - -51.88642438468371).abs() < 0.01);
        assert!((u.get(73).re - 52.883083411255015).abs() < 0.01);
        assert!((u.get(-73).re - 52.883083411255015).abs() < 0.01);
        assert!((u.get(74).re - 40.60325984930231).abs() < 0.01);
        assert!((u.get(-74).re - 40.60325984930231).abs() < 0.01);
        assert!((u.get(75).re - 10.546265748931548).abs() < 0.01);
        assert!((u.get(-75).re - 10.546265748931548).abs() < 0.01);

        assert!((u.get(76).re - 52.02174571873675).abs() < 0.01);
        assert!((u.get(-76).re - 52.02174571873675).abs() < 0.01);
        assert!((u.get(77).re - -23.49347124448269).abs() < 0.01);
        assert!((u.get(-77).re - -23.49347124448269).abs() < 0.01);
        assert!((u.get(78).re - -440.0909889621458).abs() < 0.01);
        assert!((u.get(-78).re - -440.0909889621458).abs() < 0.01);
        assert!((u.get(79).re - -876.9233603658099).abs() < 0.01);
        assert!((u.get(-79).re - -876.9233603658099).abs() < 0.01);
        assert!((u.get(80).re - -15.295691223145315).abs() < 0.1);
        assert!((u.get(-80).re - -15.295691223145315).abs() < 0.1);
        assert!((u.get(81).re - 1168.969273625954).abs() < 0.01);
        assert!((u.get(-81).re - 1168.969273625954).abs() < 0.01);
        assert!((u.get(82).re - 75.66550959694396).abs() < 0.01);
        assert!((u.get(-82).re - 75.66550959694396).abs() < 0.01);

        assert!((u.get(83).re - -251.85161934148522).abs() < 0.01);
        assert!((u.get(-83).re - -251.85161934148522).abs() < 0.01);
        assert!((u.get(84).re - -10.11477649202137).abs() < 0.01);
        assert!((u.get(-84).re - -10.11477649202137).abs() < 0.01);
        assert!((u.get(85).re - 42.635684850207696).abs() < 0.01);
        assert!((u.get(-85).re - 42.635684850207696).abs() < 0.01);
        assert!((u.get(86).re - -186.78924115119239).abs() < 0.01);
        assert!((u.get(-86).re - -186.78924115119239).abs() < 0.01);
        assert!((u.get(87).re - -39.77029659175936).abs() < 0.01);
        assert!((u.get(-87).re - -39.77029659175936).abs() < 0.01);
        assert!((u.get(88).re - 33.99692845031374).abs() < 0.01);
        assert!((u.get(-88).re - 33.99692845031374).abs() < 0.01);
        assert!((u.get(89).re - -111.93666402147537).abs() < 0.01);
        assert!((u.get(-89).re - -111.93666402147537).abs() < 0.01);

        for i in 90..127 {
            assert_eq!(u.get(i).re, 0.0);
            assert_eq!(u.get(-i).re, 0.0);
        }

        assert_eq!(u.get(-128).re, 0.0);

        assert!((u.idft(-128) - -5.79664917).abs() < 0.1);
        assert!((u.idft(-127) - -9.39344502).abs() < 0.1);
        assert!((u.idft(-126) - 5.06817731).abs() < 0.1);
        assert!((u.idft(-125) - -0.96754878).abs() < 0.1);
        assert!((u.idft(-124) - -5.36275853).abs() < 0.1);
        assert!((u.idft(-123) - 9.25248228).abs() < 0.1);
        assert!((u.idft(-122) - 5.14178501).abs() < 0.1);
        assert!((u.idft(-121) - -6.82315977).abs() < 0.1);
        assert!((u.idft(-120) - 0.24390933).abs() < 0.1);
        assert!((u.idft(-119) - 0.32498515).abs() < 0.1);
        assert!((u.idft(-118) - -5.59282565).abs() < 0.1);
        assert!((u.idft(-117) - 1.65833539).abs() < 0.1);
        assert!((u.idft(-116) - 2.69111583).abs() < 0.1);
        assert!((u.idft(-115) - -0.68879576).abs() < 0.1);
        assert!((u.idft(-114) - 4.19714704).abs() < 0.1);
        assert!((u.idft(-113) - 0.04222748).abs() < 0.1);
        assert!((u.idft(-112) - -4.25194871).abs() < 0.1);
        assert!((u.idft(-111) - 2.73856548).abs() < 0.1);
        assert!((u.idft(-110) - -2.18867362).abs() < 0.1);
        assert!((u.idft(-109) - -6.98326203).abs() < 0.1);
        assert!((u.idft(-108) - 5.93328155).abs() < 0.1);
        assert!((u.idft(-107) - 5.10875953).abs() < 0.1);
        assert!((u.idft(-106) - -4.92751877).abs() < 0.1);
        assert!((u.idft(-105) - 4.22577597).abs() < 0.1);
        assert!((u.idft(-104) - 3.63757811).abs() < 0.1);
        assert!((u.idft(-103) - -10.83188742).abs() < 0.1);
        assert!((u.idft(-102) - -0.90741249).abs() < 0.1);
        assert!((u.idft(-101) - 7.79073650).abs() < 0.1);
        assert!((u.idft(-100) - -6.53558677).abs() < 0.1);
        assert!((u.idft(-99) - 0.08444139).abs() < 0.1);
        assert!((u.idft(-98) - 12.41775487).abs() < 0.1);
        assert!((u.idft(-97) - -7.42828181).abs() < 0.1);
        assert!((u.idft(-96) - -7.48810311).abs() < 0.1);
        assert!((u.idft(-95) - 14.28093969).abs() < 0.1);
        assert!((u.idft(-94) - -5.62534996).abs() < 0.1);
        assert!((u.idft(-93) - -16.41403428).abs() < 0.1);
        assert!((u.idft(-92) - 16.53935107).abs() < 0.1);
        assert!((u.idft(-91) - 5.46098470).abs() < 0.1);
        assert!((u.idft(-90) - -20.66761836).abs() < 0.1);
        assert!((u.idft(-89) - 14.99340076).abs() < 0.1);
        assert!((u.idft(-88) - 17.24977364).abs() < 0.1);
        assert!((u.idft(-87) - -28.02218840).abs() < 0.1);
        assert!((u.idft(-86) - -2.68024504).abs() < 0.1);
        assert!((u.idft(-85) - 24.07850215).abs() < 0.1);
        assert!((u.idft(-84) - -19.97721902).abs() < 0.1);
        assert!((u.idft(-83) - -10.01992239).abs() < 0.1);
        assert!((u.idft(-82) - 33.25768816).abs() < 0.1);
        assert!((u.idft(-81) - -5.69569598).abs() < 0.1);
        assert!((u.idft(-80) - -23.28797157).abs() < 0.1);
        assert!((u.idft(-79) - 20.74487609).abs() < 0.1);
        assert!((u.idft(-78) - -0.20039937).abs() < 0.1);
        assert!((u.idft(-77) - -27.81453662).abs() < 0.1);
        assert!((u.idft(-76) - 16.51788741).abs() < 0.1);
        assert!((u.idft(-75) - 15.26632971).abs() < 0.1);
        assert!((u.idft(-74) - -19.82224706).abs() < 0.1);
        assert!((u.idft(-73) - 11.41134754).abs() < 0.1);
        assert!((u.idft(-72) - 16.13466736).abs() < 0.1);
        assert!((u.idft(-71) - -27.07550193).abs() < 0.1);
        assert!((u.idft(-70) - -3.84579676).abs() < 0.1);
        assert!((u.idft(-69) - 18.71595906).abs() < 0.1);
        assert!((u.idft(-68) - -19.02463959).abs() < 0.1);
        assert!((u.idft(-67) - -2.08856073).abs() < 0.1);
        assert!((u.idft(-66) - 34.26453815).abs() < 0.1);
        assert!((u.idft(-65) - -7.33770705).abs() < 0.1);
        assert!((u.idft(-64) - -20.92081647).abs() < 0.1);
        assert!((u.idft(-63) - 17.07589621).abs() < 0.1);
        assert!((u.idft(-62) - -7.30952593).abs() < 0.1);
        assert!((u.idft(-61) - -30.93552991).abs() < 0.1);
        assert!((u.idft(-60) - 16.89977115).abs() < 0.1);
        assert!((u.idft(-59) - 27.97775631).abs() < 0.1);
        assert!((u.idft(-58) - -8.14555832).abs() < 0.1);
        assert!((u.idft(-57) - 0.50629296).abs() < 0.1);
        assert!((u.idft(-56) - 12.02032378).abs() < 0.1);
        assert!((u.idft(-55) - -21.82736521).abs() < 0.1);
        assert!((u.idft(-54) - -29.15495045).abs() < 0.1);
        assert!((u.idft(-53) - 9.83596318).abs() < 0.1);
        assert!((u.idft(-52) - 23.97022383).abs() < 0.1);
        assert!((u.idft(-51) - 6.66392125).abs() < 0.1);
        assert!((u.idft(-50) - 7.57935615).abs() < 0.1);
        assert!((u.idft(-49) - 8.83441304).abs() < 0.1);
        assert!((u.idft(-48) - -21.85346156).abs() < 0.1);
        assert!((u.idft(-47) - -33.31551231).abs() < 0.1);
        assert!((u.idft(-46) - 1.48361254).abs() < 0.1);
        assert!((u.idft(-45) - 21.28575517).abs() < 0.1);
        assert!((u.idft(-44) - 12.04514954).abs() < 0.1);
        assert!((u.idft(-43) - 13.92062457).abs() < 0.1);
        assert!((u.idft(-42) - 9.29144790).abs() < 0.1);
        assert!((u.idft(-41) - -20.00954053).abs() < 0.1);
        assert!((u.idft(-40) - -29.53746184).abs() < 0.1);
        assert!((u.idft(-39) - -5.43448076).abs() < 0.1);
        assert!((u.idft(-38) - 10.76036684).abs() < 0.1);
        assert!((u.idft(-37) - 12.84723250).abs() < 0.1);
        assert!((u.idft(-36) - 20.64713636).abs() < 0.1);
        assert!((u.idft(-35) - 11.14764130).abs() < 0.1);
        assert!((u.idft(-34) - -18.20908302).abs() < 0.1);
        assert!((u.idft(-33) - -19.61802293).abs() < 0.1);
        assert!((u.idft(-32) - -8.19806102).abs() < 0.1);
        assert!((u.idft(-31) - -11.74537548).abs() < 0.1);
        assert!((u.idft(-30) - 14.92574659).abs() < 0.1);
        assert!((u.idft(-29) - 37.24382873).abs() < 0.1);
        assert!((u.idft(-28) - -2.25353612).abs() < 0.1);
        assert!((u.idft(-27) - -12.97207479).abs() < 0.1);
        assert!((u.idft(-26) - 7.91625844).abs() < 0.1);
        assert!((u.idft(-25) - -31.59942016).abs() < 0.1);
        assert!((u.idft(-24) - -29.35035223).abs() < 0.1);
        assert!((u.idft(-23) - 41.27631878).abs() < 0.1);
        assert!((u.idft(-22) - 19.89886234).abs() < 0.1);
        assert!((u.idft(-21) - -17.59164273).abs() < 0.1);
        assert!((u.idft(-20) - 26.07102151).abs() < 0.1);
        assert!((u.idft(-19) - -0.08594049).abs() < 0.1);
        assert!((u.idft(-18) - -51.18277665).abs() < 0.1);
        assert!((u.idft(-17) - 5.22913077).abs() < 0.1);
        assert!((u.idft(-16) - 22.92406853).abs() < 0.1);
        assert!((u.idft(-15) - -16.16033735).abs() < 0.1);
        assert!((u.idft(-14) - 17.72598191).abs() < 0.1);
        assert!((u.idft(-13) - 22.18516070).abs() < 0.1);
        assert!((u.idft(-12) - -25.11108388).abs() < 0.1);
        assert!((u.idft(-11) - -1.18699777).abs() < 0.1);
        assert!((u.idft(-10) - 9.03229592).abs() < 0.1);
        assert!((u.idft(-9) - -26.45728235).abs() < 0.1);
        assert!((u.idft(-8) - 4.56024994).abs() < 0.1);
        assert!((u.idft(-7) - 22.08178310).abs() < 0.1);
        assert!((u.idft(-6) - -17.59171648).abs() < 0.1);
        assert!((u.idft(-5) - 7.02802722).abs() < 0.1);
        assert!((u.idft(-4) - 31.28951287).abs() < 0.1);
        assert!((u.idft(-3) - -22.50191079).abs() < 0.1);
        assert!((u.idft(-2) - -23.46011347).abs() < 0.1);
        assert!((u.idft(-1) - 19.15931422).abs() < 0.1);
        assert!((u.idft(0) - -10.78672773).abs() < 0.1);
        assert!((u.idft(1) - -20.38247464).abs() < 0.1);
        assert!((u.idft(2) - 36.34924449).abs() < 0.1);
        assert!((u.idft(3) - 22.10862316).abs() < 0.1);
        assert!((u.idft(4) - -28.15892880).abs() < 0.1);
        assert!((u.idft(5) - 1.26585774).abs() < 0.1);
        assert!((u.idft(6) - 8.29548529).abs() < 0.1);
        assert!((u.idft(7) - -37.32592318).abs() < 0.1);
        assert!((u.idft(8) - -2.38641534).abs() < 0.1);
        assert!((u.idft(9) - 39.03278393).abs() < 0.1);
        assert!((u.idft(10) - -4.53718360).abs() < 0.1);
        assert!((u.idft(11) - 1.00653030).abs() < 0.1);
        assert!((u.idft(12) - 28.29558432).abs() < 0.1);
        assert!((u.idft(13) - -32.32720506).abs() < 0.1);
        assert!((u.idft(14) - -35.78007419).abs() < 0.1);
        assert!((u.idft(15) - 23.47296407).abs() < 0.1);
        assert!((u.idft(16) - -7.50319279).abs() < 0.1);
        assert!((u.idft(17) - -4.56669882).abs() < 0.1);
        assert!((u.idft(18) - 61.67557092).abs() < 0.1);
        assert!((u.idft(19) - 2.58401083).abs() < 0.1);
        assert!((u.idft(20) - -54.33887744).abs() < 0.1);
        assert!((u.idft(21) - 8.45111307).abs() < 0.1);
        assert!((u.idft(22) - -8.41016845).abs() < 0.1);
        assert!((u.idft(23) - -45.46104310).abs() < 0.1);
        assert!((u.idft(24) - 47.93722226).abs() < 0.1);
        assert!((u.idft(25) - 60.94331766).abs() < 0.1);
        assert!((u.idft(26) - -32.98283133).abs() < 0.1);
        assert!((u.idft(27) - -13.33703293).abs() < 0.1);
        assert!((u.idft(28) - 13.37965781).abs() < 0.1);
        assert!((u.idft(29) - -52.48454058).abs() < 0.1);
        assert!((u.idft(30) - -20.67155015).abs() < 0.1);
        assert!((u.idft(31) - 63.73105350).abs() < 0.1);
        assert!((u.idft(32) - 18.06583597).abs() < 0.1);
        assert!((u.idft(33) - -23.93905051).abs() < 0.1);
        assert!((u.idft(34) - 20.94158838).abs() < 0.1);
        assert!((u.idft(35) - -7.67853185).abs() < 0.1);
        assert!((u.idft(36) - -56.08384708).abs() < 0.1);
        assert!((u.idft(37) - 12.36090374).abs() < 0.1);
        assert!((u.idft(38) - 42.37205721).abs() < 0.1);
        assert!((u.idft(39) - -19.98470969).abs() < 0.1);
        assert!((u.idft(40) - 3.27695989).abs() < 0.1);
        assert!((u.idft(41) - 37.29797601).abs() < 0.1);
        assert!((u.idft(42) - -34.98240265).abs() < 0.1);
        assert!((u.idft(43) - -32.57432802).abs() < 0.1);
        assert!((u.idft(44) - 39.63282676).abs() < 0.1);
        assert!((u.idft(45) - -5.32238891).abs() < 0.1);
        assert!((u.idft(46) - -31.48791066).abs() < 0.1);
        assert!((u.idft(47) - 42.58494771).abs() < 0.1);
        assert!((u.idft(48) - 14.05462803).abs() < 0.1);
        assert!((u.idft(49) - -47.67627480).abs() < 0.1);
        assert!((u.idft(50) - 15.92570804).abs() < 0.1);
        assert!((u.idft(51) - 27.20430115).abs() < 0.1);
        assert!((u.idft(52) - -42.43731851).abs() < 0.1);
        assert!((u.idft(53) - -1.55333517).abs() < 0.1);
        assert!((u.idft(54) - 43.08442969).abs() < 0.1);
        assert!((u.idft(55) - -23.08501026).abs() < 0.1);
        assert!((u.idft(56) - -21.50273291).abs() < 0.1);
        assert!((u.idft(57) - 41.22292240).abs() < 0.1);
        assert!((u.idft(58) - -1.70982605).abs() < 0.1);
        assert!((u.idft(59) - -37.33879987).abs() < 0.1);
        assert!((u.idft(60) - 21.25092718).abs() < 0.1);
        assert!((u.idft(61) - 11.02694795).abs() < 0.1);
        assert!((u.idft(62) - -39.50723940).abs() < 0.1);
        assert!((u.idft(63) - 13.12599041).abs() < 0.1);
        assert!((u.idft(64) - 39.94185348).abs() < 0.1);
        assert!((u.idft(65) - -20.49338938).abs() < 0.1);
        assert!((u.idft(66) - -7.99384162).abs() < 0.1);
        assert!((u.idft(67) - 23.22920912).abs() < 0.1);
        assert!((u.idft(68) - -30.75540642).abs() < 0.1);
        assert!((u.idft(69) - -25.80438587).abs() < 0.1);
        assert!((u.idft(70) - 36.18347261).abs() < 0.1);
        assert!((u.idft(71) - 10.11299966).abs() < 0.1);
        assert!((u.idft(72) - -11.47388509).abs() < 0.1);
        assert!((u.idft(73) - 23.53641926).abs() < 0.1);
        assert!((u.idft(74) - -5.95409133).abs() < 0.1);
        assert!((u.idft(75) - -39.10226269).abs() < 0.1);
        assert!((u.idft(76) - 4.46990113).abs() < 0.1);
        assert!((u.idft(77) - 16.19724527).abs() < 0.1);
        assert!((u.idft(78) - -5.88640965).abs() < 0.1);
        assert!((u.idft(79) - 14.80633473).abs() < 0.1);
        assert!((u.idft(80) - 16.78517421).abs() < 0.1);
        assert!((u.idft(81) - -18.00587911).abs() < 0.1);
        assert!((u.idft(82) - -16.48159939).abs() < 0.1);
        assert!((u.idft(83) - 3.10990610).abs() < 0.1);
        assert!((u.idft(84) - -1.56448955).abs() < 0.1);
        assert!((u.idft(85) - -0.76036963).abs() < 0.1);
        assert!((u.idft(86) - 14.90803775).abs() < 0.1);
        assert!((u.idft(87) - 10.69050309).abs() < 0.1);
        assert!((u.idft(88) - -9.49789330).abs() < 0.1);
        assert!((u.idft(89) - -8.97859300).abs() < 0.1);
        assert!((u.idft(90) - 0.66039865).abs() < 0.1);
        assert!((u.idft(91) - -6.85950458).abs() < 0.1);
        assert!((u.idft(92) - -3.15069207).abs() < 0.1);
        assert!((u.idft(93) - 14.73447430).abs() < 0.1);
        assert!((u.idft(94) - 6.88624525).abs() < 0.1);
        assert!((u.idft(95) - -6.22978603).abs() < 0.1);
        assert!((u.idft(96) - 0.40578185).abs() < 0.1);
        assert!((u.idft(97) - -2.73334780).abs() < 0.1);
        assert!((u.idft(98) - -8.88664136).abs() < 0.1);
        assert!((u.idft(99) - 1.49680170).abs() < 0.1);
        assert!((u.idft(100) - 5.17957994).abs() < 0.1);
        assert!((u.idft(101) - 1.84958020).abs() < 0.1);
        assert!((u.idft(102) - 2.84081354).abs() < 0.1);
        assert!((u.idft(103) - -0.91272964).abs() < 0.1);
        assert!((u.idft(104) - -1.05605999).abs() < 0.1);
        assert!((u.idft(105) - 0.72137813).abs() < 0.1);
        assert!((u.idft(106) - -5.89032597).abs() < 0.1);
        assert!((u.idft(107) - -1.93045269).abs() < 0.1);
        assert!((u.idft(108) - 5.28647828).abs() < 0.1);
        assert!((u.idft(109) - -2.42853363).abs() < 0.1);
        assert!((u.idft(110) - 0.82354931).abs() < 0.1);
        assert!((u.idft(111) - 9.65877294).abs() < 0.1);
        assert!((u.idft(112) - -2.99434770).abs() < 0.1);
        assert!((u.idft(113) - -7.70205682).abs() < 0.1);
        assert!((u.idft(114) - 2.72333581).abs() < 0.1);
        assert!((u.idft(115) - -3.27493606).abs() < 0.1);
        assert!((u.idft(116) - -5.29250082).abs() < 0.1);
        assert!((u.idft(117) - 9.21202681).abs() < 0.1);
        assert!((u.idft(118) - 5.42562770).abs() < 0.1);
        assert!((u.idft(119) - -5.07678868).abs() < 0.1);
        assert!((u.idft(120) - 2.66070712).abs() < 0.1);
        assert!((u.idft(121) - 0.48852727).abs() < 0.1);
        assert!((u.idft(122) - -10.71014176).abs() < 0.1);
        assert!((u.idft(123) - 0.03002930).abs() < 0.1);
        assert!((u.idft(124) - 7.93482508).abs() < 0.1);
        assert!((u.idft(125) - -3.23895896).abs() < 0.1);
        assert!((u.idft(126) - 1.32746514).abs() < 0.1);
        assert!((u.idft(127) - 9.82312305).abs() < 0.1);
    }

    #[test]
    fn test_unvoiced() {
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
        let (amps, voice, gain_idx) = descramble(&chunks, &p);
        let g = Gains::new(gain_idx, &amps, &p);
        let c = Coefficients::new(&g, &amps, &p);
        let prev = PrevFrame::default();
        let s = Spectrals::new(&c, &p, &prev);
        let dft = UnvoicedDFT::new();
        let parts = UnvoicedParts::new(&dft, &p, &voice, &s);
        let u = Unvoiced::new(&parts, &parts);

        assert!((u.get(0) - -10.78672773).abs() < 0.1);
        assert!((u.get(1) - -20.38247464).abs() < 0.1);
        assert!((u.get(2) - 36.34924449).abs() < 0.1);
        assert!((u.get(3) - 22.10862316).abs() < 0.1);
        assert!((u.get(4) - -28.15892880).abs() < 0.1);
        assert!((u.get(5) - 1.26585774).abs() < 0.1);
        assert!((u.get(6) - 8.29548529).abs() < 0.1);
        assert!((u.get(7) - -37.32592318).abs() < 0.1);
        assert!((u.get(8) - -2.38641534).abs() < 0.1);
        assert!((u.get(9) - 39.03278393).abs() < 0.1);
        assert!((u.get(10) - -4.53718360).abs() < 0.1);
        assert!((u.get(11) - 1.00653030).abs() < 0.1);
        assert!((u.get(12) - 28.29558432).abs() < 0.1);
        assert!((u.get(13) - -32.32720506).abs() < 0.1);
        assert!((u.get(14) - -35.78007419).abs() < 0.1);
        assert!((u.get(15) - 23.47296407).abs() < 0.1);
        assert!((u.get(16) - -7.50319279).abs() < 0.1);
        assert!((u.get(17) - -4.56669882).abs() < 0.1);
        assert!((u.get(18) - 61.67557092).abs() < 0.1);
        assert!((u.get(19) - 2.58401083).abs() < 0.1);
        assert!((u.get(20) - -54.33887744).abs() < 0.1);
        assert!((u.get(21) - 8.45111307).abs() < 0.1);
        assert!((u.get(22) - -8.41016845).abs() < 0.1);
        assert!((u.get(23) - -45.46104310).abs() < 0.1);
        assert!((u.get(24) - 47.93722226).abs() < 0.1);
        assert!((u.get(25) - 60.94331766).abs() < 0.1);
        assert!((u.get(26) - -32.98283133).abs() < 0.1);
        assert!((u.get(27) - -13.33703293).abs() < 0.1);
        assert!((u.get(28) - 13.37965781).abs() < 0.1);
        assert!((u.get(29) - -52.48454058).abs() < 0.1);
        assert!((u.get(30) - -20.67155015).abs() < 0.1);
        assert!((u.get(31) - 63.73105350).abs() < 0.1);
        assert!((u.get(32) - 18.06583597).abs() < 0.1);
        assert!((u.get(33) - -23.93905051).abs() < 0.1);
        assert!((u.get(34) - 20.94158838).abs() < 0.1);
        assert!((u.get(35) - -7.67853185).abs() < 0.1);
        assert!((u.get(36) - -56.08384708).abs() < 0.1);
        assert!((u.get(37) - 12.36090374).abs() < 0.1);
        assert!((u.get(38) - 42.37205721).abs() < 0.1);
        assert!((u.get(39) - -19.98470969).abs() < 0.1);
        assert!((u.get(40) - 3.27695989).abs() < 0.1);
        assert!((u.get(41) - 37.29797601).abs() < 0.1);
        assert!((u.get(42) - -34.98240265).abs() < 0.1);
        assert!((u.get(43) - -32.57432802).abs() < 0.1);
        assert!((u.get(44) - 39.63282676).abs() < 0.1);
        assert!((u.get(45) - -5.32238891).abs() < 0.1);
        assert!((u.get(46) - -31.48791066).abs() < 0.1);
        assert!((u.get(47) - 42.58494771).abs() < 0.1);
        assert!((u.get(48) - 14.05462803).abs() < 0.1);
        assert!((u.get(49) - -47.67627480).abs() < 0.1);
        assert!((u.get(50) - 15.92570804).abs() < 0.1);
        assert!((u.get(51) - 27.20430115).abs() < 0.1);
        assert!((u.get(52) - -42.43731851).abs() < 0.1);
        assert!((u.get(53) - -1.55333517).abs() < 0.1);
        assert!((u.get(54) - 43.08442969).abs() < 0.1);
        assert!((u.get(55) - -23.08501026).abs() < 0.1);
        assert!((u.get(56) - -21.85670971).abs() < 0.1);
        assert!((u.get(57) - 42.39680460).abs() < 0.1);
        assert!((u.get(58) - -1.87294999).abs() < 0.1);
        assert!((u.get(59) - -39.55023096).abs() < 0.1);
        assert!((u.get(60) - 22.52716559).abs() < 0.1);
        assert!((u.get(61) - 12.31471496).abs() < 0.1);
        assert!((u.get(62) - -42.46277687).abs() < 0.1);
        assert!((u.get(63) - 13.45364723).abs() < 0.1);
        assert!((u.get(64) - 44.55797573).abs() < 0.1);
        assert!((u.get(65) - -19.90959348).abs() < 0.1);
        assert!((u.get(66) - -11.37754789).abs() < 0.1);
        assert!((u.get(67) - 21.59135817).abs() < 0.1);
        assert!((u.get(68) - -30.00450175).abs() < 0.1);
        assert!((u.get(69) - -28.56917243).abs() < 0.1);
        assert!((u.get(70) - 32.97956090).abs() < 0.1);
        assert!((u.get(71) - 20.67055244).abs() < 0.1);
        assert!((u.get(72) - -3.09840552).abs() < 0.1);
        assert!((u.get(73) - 9.22722646).abs() < 0.1);
        assert!((u.get(74) - -8.90701539).abs() < 0.1);
        assert!((u.get(75) - -26.59607068).abs() < 0.1);
        assert!((u.get(76) - -11.30633646).abs() < 0.1);
        assert!((u.get(77) - 9.19103214).abs() < 0.1);
        assert!((u.get(78) - 24.08560283).abs() < 0.1);
        assert!((u.get(79) - 9.91485620).abs() < 0.1);
        assert!((u.get(80) - -6.50279736).abs() < 0.1);
        assert!((u.get(81) - 4.28217570).abs() < 0.1);
        assert!((u.get(82) - -15.28169988).abs() < 0.1);
        assert!((u.get(83) - -28.01218814).abs() < 0.1);
        assert!((u.get(84) - 17.40110976).abs() < 0.1);
        assert!((u.get(85) - 17.03009610).abs() < 0.1);
        assert!((u.get(86) - -12.52787223).abs() < 0.1);
        assert!((u.get(87) - 20.68220241).abs() < 0.1);
        assert!((u.get(88) - 13.46080685).abs() < 0.1);
        assert!((u.get(89) - -37.68500544).abs() < 0.1);
        assert!((u.get(90) - -4.29989334).abs() < 0.1);
        assert!((u.get(91) - 19.36130905).abs() < 0.1);
        assert!((u.get(92) - -24.21556118).abs() < 0.1);
        assert!((u.get(93) - 3.06827405).abs() < 0.1);
        assert!((u.get(94) - 42.99834610).abs() < 0.1);
        assert!((u.get(95) - -10.46488654).abs() < 0.1);
        assert!((u.get(96) - -24.23670370).abs() < 0.1);
        assert!((u.get(97) - 19.01862304).abs() < 0.1);
        assert!((u.get(98) - -9.91875934).abs() < 0.1);
        assert!((u.get(99) - -34.28454629).abs() < 0.1);
        assert!((u.get(100) - 19.18018540).abs() < 0.1);
        assert!((u.get(101) - 30.35588910).abs() < 0.1);
        assert!((u.get(102) - -8.43820560).abs() < 0.1);
        assert!((u.get(103) - 0.48692814).abs() < 0.1);
        assert!((u.get(104) - 12.23854716).abs() < 0.1);
        assert!((u.get(105) - -21.82736521).abs() < 0.1);
        assert!((u.get(106) - -29.15495045).abs() < 0.1);
        assert!((u.get(107) - 9.83596318).abs() < 0.1);
        assert!((u.get(108) - 23.97022383).abs() < 0.1);
        assert!((u.get(109) - 6.66392125).abs() < 0.1);
        assert!((u.get(110) - 7.57935615).abs() < 0.1);
        assert!((u.get(111) - 8.83441304).abs() < 0.1);
        assert!((u.get(112) - -21.85346156).abs() < 0.1);
        assert!((u.get(113) - -33.31551231).abs() < 0.1);
        assert!((u.get(114) - 1.48361254).abs() < 0.1);
        assert!((u.get(115) - 21.28575517).abs() < 0.1);
        assert!((u.get(116) - 12.04514954).abs() < 0.1);
        assert!((u.get(117) - 13.92062457).abs() < 0.1);
        assert!((u.get(118) - 9.29144790).abs() < 0.1);
        assert!((u.get(119) - -20.00954053).abs() < 0.1);
        assert!((u.get(120) - -29.53746184).abs() < 0.1);
        assert!((u.get(121) - -5.43448076).abs() < 0.1);
        assert!((u.get(122) - 10.76036684).abs() < 0.1);
        assert!((u.get(123) - 12.84723250).abs() < 0.1);
        assert!((u.get(124) - 20.64713636).abs() < 0.1);
        assert!((u.get(125) - 11.14764130).abs() < 0.1);
        assert!((u.get(126) - -18.20908302).abs() < 0.1);
        assert!((u.get(127) - -19.61802293).abs() < 0.1);
        assert!((u.get(128) - -8.19806102).abs() < 0.1);
        assert!((u.get(129) - -11.74537548).abs() < 0.1);
        assert!((u.get(130) - 14.92574659).abs() < 0.1);
        assert!((u.get(131) - 37.24382873).abs() < 0.1);
        assert!((u.get(132) - -2.25353612).abs() < 0.1);
        assert!((u.get(133) - -12.97207479).abs() < 0.1);
        assert!((u.get(134) - 7.91625844).abs() < 0.1);
        assert!((u.get(135) - -31.59942016).abs() < 0.1);
        assert!((u.get(136) - -29.35035223).abs() < 0.1);
        assert!((u.get(137) - 41.27631878).abs() < 0.1);
        assert!((u.get(138) - 19.89886234).abs() < 0.1);
        assert!((u.get(139) - -17.59164273).abs() < 0.1);
        assert!((u.get(140) - 26.07102151).abs() < 0.1);
        assert!((u.get(141) - -0.08594049).abs() < 0.1);
        assert!((u.get(142) - -51.18277665).abs() < 0.1);
        assert!((u.get(143) - 5.22913077).abs() < 0.1);
        assert!((u.get(144) - 22.92406853).abs() < 0.1);
        assert!((u.get(145) - -16.16033735).abs() < 0.1);
        assert!((u.get(146) - 17.72598191).abs() < 0.1);
        assert!((u.get(147) - 22.18516070).abs() < 0.1);
        assert!((u.get(148) - -25.11108388).abs() < 0.1);
        assert!((u.get(149) - -1.18699777).abs() < 0.1);
        assert!((u.get(150) - 9.03229592).abs() < 0.1);
        assert!((u.get(151) - -26.45728235).abs() < 0.1);
        assert!((u.get(152) - 4.56024994).abs() < 0.1);
        assert!((u.get(153) - 22.08178310).abs() < 0.1);
        assert!((u.get(154) - -17.59171648).abs() < 0.1);
        assert!((u.get(155) - 7.02802722).abs() < 0.1);
        assert!((u.get(156) - 31.28951287).abs() < 0.1);
        assert!((u.get(157) - -22.50191079).abs() < 0.1);
        assert!((u.get(158) - -23.46011347).abs() < 0.1);
        assert!((u.get(159) - 19.15931422).abs() < 0.1);
    }
}
