use std::f32::consts::PI;

use arrayvec::ArrayVec;

use allocs::allocs;
use descramble::QuantizedAmplitudes;
use gain::Gains;
use params::BaseParams;

pub struct Coefficients(ArrayVec<[f32; 56]>);

impl Coefficients {
    pub fn new(gains: &Gains, amps: &QuantizedAmplitudes, params: &BaseParams)
        -> Coefficients
    {
        let mut coefs = ArrayVec::new();

        for block in 1..7 {
            let c = CoefBlock::new(block, gains, amps, params);
            coefs.extend((1...c.len()).map(|j| c.idct(j)));
        }

        Coefficients(coefs)
    }

    pub fn get(&self, l: usize) -> f32 { self.0[l - 1] }
}

struct CoefBlock(ArrayVec<[f32; 10]>);

impl CoefBlock {
    pub fn new(block: usize, gains: &Gains, amps: &QuantizedAmplitudes, params: &BaseParams)
        -> CoefBlock
    {
        assert!(block >= 1 && block <= 6);

        let mut coefs = ArrayVec::new();

        let (_, alloc) = allocs(params.harmonics);
        let amps_used = AMPS_USED[params.harmonics as usize - 9];

        coefs.push(gains.idct(block));

        let block = block - 1;

        let m_start = 8 + (0..block).map(|k| amps_used[k]).fold(0, |s, x| s + x) as usize;
        let m_end = m_start + amps_used[block] as usize;

        coefs.extend((m_start..m_end).enumerate().map(|(k, m)| {
            let bits = alloc[m - 3] as i32;

            if bits == 0 {
                0.0
            } else {
                DCT_STEP_SIZE[bits as usize - 1] * DCT_STD_DEV[k] *
                    (amps.get(m as usize) as f32 - (2.0f32).powi(bits as i32 - 1) + 0.5)
            }
        }));

        CoefBlock(coefs)
    }

    pub fn len(&self) -> usize { self.0.len() }

    pub fn idct(&self, j: usize) -> f32 {
        assert!(j >= 1 && j <= self.len());

        self.0[0] + 2.0 * (2...self.len()).map(|k| {
            self.0[k - 1] * (
                PI * (k as f32 - 1.0) * (j as f32 - 0.5) / self.len() as f32
            ).cos()
        }).fold(0.0, |s, x| s + x)
    }
}

// AMPS_USED[l] is J_1,..,,J_6
static AMPS_USED: [[u8; 6]; 48] = [
    [0, 0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 2, 2],
    [1, 1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2, 2],
    [1, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 3],
    [2, 2, 2, 2, 3, 3],
    [2, 2, 2, 3, 3, 3],
    [2, 2, 3, 3, 3, 3],
    [2, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 4],
    [3, 3, 3, 3, 4, 4],
    [3, 3, 3, 4, 4, 4],
    [3, 3, 4, 4, 4, 4],
    [3, 4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4, 5],
    [4, 4, 4, 4, 5, 5],
    [4, 4, 4, 5, 5, 5],
    [4, 4, 5, 5, 5, 5],
    [4, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 6],
    [5, 5, 5, 5, 6, 6],
    [5, 5, 5, 6, 6, 6],
    [5, 5, 6, 6, 6, 6],
    [5, 6, 6, 6, 6, 6],
    [6, 6, 6, 6, 6, 6],
    [6, 6, 6, 6, 6, 7],
    [6, 6, 6, 6, 7, 7],
    [6, 6, 6, 7, 7, 7],
    [6, 6, 7, 7, 7, 7],
    [6, 7, 7, 7, 7, 7],
    [7, 7, 7, 7, 7, 7],
    [7, 7, 7, 7, 7, 8],
    [7, 7, 7, 7, 8, 8],
    [7, 7, 7, 8, 8, 8],
    [7, 7, 8, 8, 8, 8],
    [7, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 9],
    [8, 8, 8, 8, 9, 9],
];

// DCT_STEP_SIZE[i] is for i number
// of bits,
// 1<=i<=10
const DCT_STEP_SIZE: [f32; 10] = [
    1.2,
    0.85,
    0.65,
    0.40,
    0.28,
    0.15,
    0.08,
    0.04,
    0.02,
    0.01,
];

// DCT_STD_DEV[k] is stddev for
// C[.][k],
// 2<=k<=10
const DCT_STD_DEV: [f32; 9] = [
    0.307,
    0.241,
    0.207,
    0.190,
    0.179,
    0.173,
    0.165,
    0.170,
    0.170,
];

#[cfg(test)]
mod tests {
    use super::*;
    use super::CoefBlock;
    use descramble::{Bootstrap, descramble};
    use params::BaseParams;
    use gain::Gains;

    #[test]
    fn test_coefs() {
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

        assert!((c.get(1) - -0.9140941318413551).abs() < 0.000001);
        assert!((c.get(2) - -1.5002149427668843).abs() < 0.000001);
        assert!((c.get(3) - -0.1243967542115918).abs() < 0.000001);
        assert!((c.get(4) - -2.924751739744676).abs() < 0.000001);
        assert!((c.get(5) - 2.1501046599919884).abs() < 0.000001);
        assert!((c.get(6) - -0.557148495647667).abs() < 0.000001);
        assert!((c.get(7) - -0.08320165128732226).abs() < 0.000001);
        assert!((c.get(8) - 3.412285791008137).abs() < 0.000001);
        assert!((c.get(9) - 0.38493783640665913).abs() < 0.000001);
        assert!((c.get(10) - 0.6472398818051812).abs() < 0.000001);
        assert!((c.get(11) - 3.0575528322544274).abs() < 0.000001);
        assert!((c.get(12) - 0.672970246978134).abs() < 0.000001);
        assert!((c.get(13) - 0.638137661701841).abs() < 0.000001);
        assert!((c.get(14) - 3.9893475658703124).abs() < 0.000001);
        assert!((c.get(15) - 3.5092571965451276).abs() < 0.000001);
        assert!((c.get(16) - 3.643716827219943).abs() < 0.000001);
    }

    #[test]
    fn test_coefs_9() {
        let chunks = [
            0b000000010010,
            0b110011001100,
            0b111000111000,
            0b111111111111,
            0b11010110101,
            0b00101111010,
            0b01110111011,
            0b00001000,
        ];

        let b = Bootstrap::new(&chunks);
        let p = BaseParams::new(b.unwrap_period());
        let (amps, _, gain_idx) = descramble(&chunks, &p);
        let g = Gains::new(gain_idx, &amps, &p);

        let c = CoefBlock::new(1, &g, &amps, &p);
        assert_eq!(c.len(), 1);
        assert!((c.0[0] - 0.8519942560055926).abs() < 0.000001);
        assert!((c.idct(1) - 0.8519942560055926).abs() < 0.000001);

        let c = CoefBlock::new(2, &g, &amps, &p);
        assert_eq!(c.len(), 1);
        assert!((c.0[0] - -0.13083074772702047).abs() < 0.000001);
        assert!((c.idct(1) - -0.13083074772702047).abs() < 0.000001);

        let c = CoefBlock::new(3, &g, &amps, &p);
        assert_eq!(c.len(), 1);
        assert!((c.0[0] - -0.014229409757043066).abs() < 0.000001);
        assert!((c.idct(1) - -0.014229409757043066).abs() < 0.000001);

        let c = CoefBlock::new(4, &g, &amps, &p);
        assert_eq!(c.len(), 2);
        assert!((c.0[0] - 2.0309896309891773).abs() < 0.000001);
        assert!((c.0[1] - -0.45129).abs() < 0.000001);
        assert!((c.idct(1) - 1.3927691924258232).abs() < 0.000001);
        assert!((c.idct(2) - 2.6692100695525314).abs() < 0.000001);

        let c = CoefBlock::new(5, &g, &amps, &p);
        assert_eq!(c.len(), 2);
        assert!((c.0[0] - 0.5902767477270205).abs() < 0.000001);
        assert!((c.0[1] - -0.8289).abs() < 0.000001);
        assert!((c.idct(1) - -0.581964874124038).abs() < 0.000001);
        assert!((c.idct(2) - 1.7625183695780788).abs() < 0.000001);

        let c = CoefBlock::new(6, &g, &amps, &p);
        assert_eq!(c.len(), 2);
        assert!((c.0[0] - 1.0951375227622724).abs() < 0.000001);
        assert!((c.0[1] - 1.24028).abs() < 0.000001);
        assert!((c.idct(1) - 2.849158319902375).abs() < 0.000001);
        assert!((c.idct(2) - -0.6588832743778299).abs() < 0.000001);
    }

    #[test]
    fn test_coefs_16() {
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

        let c = CoefBlock::new(1, &g, &amps, &p);
        assert_eq!(c.len(), 2);
        assert!((c.0[0] - -1.2071545373041197).abs() < 0.000001);
        assert!((c.0[1] - 0.207225).abs() < 0.000001);
        assert!((c.idct(1) - -0.9140941318413551).abs() < 0.000001);
        assert!((c.idct(2) - -1.5002149427668843).abs() < 0.000001);

        let c = CoefBlock::new(2, &g, &amps, &p);
        assert_eq!(c.len(), 2);
        assert!((c.0[0] - -1.524574246978134).abs() < 0.000001);
        assert!((c.0[1] - 0.990075).abs() < 0.000001);
        assert!((c.idct(1) - -0.1243967542115918).abs() < 0.000001);
        assert!((c.idct(2) - -2.924751739744676).abs() < 0.000001);

        let c = CoefBlock::new(3, &g, &amps, &p);
        assert_eq!(c.len(), 3);
        assert!((c.0[0] - 0.5032515043523329).abs() < 0.000001);
        assert!((c.0[1] - 0.6447).abs() < 0.000001);
        assert!((c.0[2] - 0.5302).abs() < 0.000001);
        assert!((c.idct(1) - 2.1501046599919884).abs() < 0.000001);
        assert!((c.idct(2) - -0.557148495647667).abs() < 0.000001);
        assert!((c.idct(3) - -0.08320165128732226).abs() < 0.000001);

        let c = CoefBlock::new(4, &g, &amps, &p);
        assert_eq!(c.len(), 3);
        assert!((c.0[0] - 1.481487836406659).abs() < 0.000001);
        assert!((c.0[1] - 0.7982).abs() < 0.000001);
        assert!((c.0[2] - 0.548275).abs() < 0.000001);
        assert!((c.idct(1) - 3.412285791008137).abs() < 0.000001);
        assert!((c.idct(2) - 0.38493783640665913).abs() < 0.000001);
        assert!((c.idct(3) - 0.6472398818051812).abs() < 0.000001);

        let c = CoefBlock::new(5, &g, &amps, &p);
        assert_eq!(c.len(), 3);
        assert!((c.0[0] - 1.456220246978134).abs() < 0.000001);
        assert!((c.0[1] - 0.698425).abs() < 0.000001);
        assert!((c.0[2] - 0.391625).abs() < 0.000001);
        assert!((c.idct(1) - 3.0575528322544274).abs() < 0.000001);
        assert!((c.idct(2) - 0.672970246978134).abs() < 0.000001);
        assert!((c.idct(3) - 0.638137661701841).abs() < 0.000001);

        let c = CoefBlock::new(6, &g, &amps, &p);
        assert_eq!(c.len(), 3);
        assert!((c.0[0] - 3.7141071965451276).abs() < 0.000001);
        assert!((c.0[1] - 0.099775).abs() < 0.000001);
        assert!((c.0[2] - 0.102425).abs() < 0.000001);
        assert!((c.idct(1) - 3.9893475658703124).abs() < 0.000001);
        assert!((c.idct(2) - 3.5092571965451276).abs() < 0.000001);
        assert!((c.idct(3) - 3.643716827219943).abs() < 0.000001);
    }
}
