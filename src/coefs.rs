//! Higher order DCT coefficients.

use std::f32::consts::PI;

use arrayvec::ArrayVec;

use allocs::allocs;
use consts::{MIN_HARMONICS, MAX_HARMONICS};
use descramble::QuantizedAmplitudes;
use gain::Gains;
use params::BaseParams;

/// Higher order DCT coefficients vector T<sub>l</sub>, 1 ≤ l ≤ L.
pub struct Coefficients(ArrayVec<[f32; MAX_HARMONICS]>);

impl Coefficients {
    /// Create a new `Coefficients` vector from the given gains G<sub>m</sub>, quantized
    /// amplitudes b<sub>m</sub>, and frame parameters.
    pub fn new(gains: &Gains, amps: &QuantizedAmplitudes, params: &BaseParams)
        -> Coefficients
    {
        let mut coefs = ArrayVec::new();

        // Tracks the starting quantized amplitude b_m to be inserted into the current
        // coefficient block. For the first block this is always b_8 [p34].
        let mut cur = 8;

        // Generate blocks for 1 ≤ i ≤ 6.
        for block in 1...6 {
            let b = CoefBlock::new(block, cur, gains, amps, params);
            coefs.extend((1...b.len()).map(|j| b.idct(j)));

            // The first coefficient C_i,1 in each block doesn't count towards quantized
            // amplitude usage.
            cur += b.len() - 1;
        }

        Coefficients(coefs)
    }

    /// Retrieve T<sub>l</sub>, 1 ≤ l ≤ L + 1.
    pub fn get(&self, l: usize) -> f32 { self.0[l - 1] }
}

/// Block of coeffients C<sub>i,k</sub>, 1 ≤ i ≤ 6 and 1 ≤ k ≤ J<sub>i</sub>.
struct CoefBlock(ArrayVec<[f32; 10]>);

impl CoefBlock {
    /// Create a new `CoefBlock` from the given block i, the starting quantized amplitude
    /// number, gains G<sub>m</sub>, quantized amplitudes b<sub>m</sub>, and frame
    /// parameters.
    pub fn new(block: usize, cur: usize, gains: &Gains, amps: &QuantizedAmplitudes,
               params: &BaseParams)
        -> CoefBlock
    {
        assert!(block >= 1 && block <= 6);

        let mut coefs = ArrayVec::new();

        let (_, alloc) = allocs(params.harmonics);
        let blocks = &AMPS_USED[params.harmonics as usize - MIN_HARMONICS];

        // C_i,1 = R_i.
        coefs.push(gains.idct(block));

        // Compute the starting and ending m for the b_m's to use for this block.
        let start = cur;
        let stop = start + blocks[block - 1];

        // Generate C_i,2, ..., C_i,Ji.
        coefs.extend((start..stop).enumerate().map(|(k, m)| {
            // Retrieve the bit allocation B_m for the current quantized amplitude b_m.
            let bits = alloc[m - 3] as i32;

            // Compute C_i,k.
            if bits == 0 {
                0.0
            } else {
                DCT_STEP_SIZE[bits as usize - 1] * DCT_STD_DEV[k] *
                    (amps.get(m) as f32 - (1 << (bits - 1)) as f32 + 0.5)
            }
        }));

        CoefBlock(coefs)
    }

    /// Retrieve the number of coeffiients in this block, J<sub>i</sub>.
    pub fn len(&self) -> usize { self.0.len() }

    /// Compute the IDCT c<sub>i,j</sub> for the current block i and 1 ≤ j ≤
    /// J<sub>i</sub>.
    pub fn idct(&self, j: usize) -> f32 {
        assert!(j >= 1 && j <= self.len());

        self.0[0] + 2.0 * (2...self.len()).map(|k| {
            self.0[k - 1] * (
                PI * (k as f32 - 1.0) * (j as f32 - 0.5) / self.len() as f32
            ).cos()
        }).fold(0.0, |s, x| s + x)
    }
}

/// Each AMPS_USED[l] gives J<sub>1</sub> - 1,..,,J<sub>6</sub> - 1 for harmonics
/// parameter l = L - 9. Each J<sub>i</sub> - 1 represents the number of quantized
/// amplitudes used in coefficient block i.
static AMPS_USED: [[usize; 6]; 48] = [
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

/// Each DCT_STEP_SIZE[b] is the "uniform quantizer step size" [p31] for the bit
/// allocation b = B<sub>m</sub> - 1.
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

/// Each DCT_STD_DEV[j] is the DCT standard deviation [p32] for the coefficient
/// C<sub>i,j+2</sub>.
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
    fn verify_amps() {
        // Verify the constaints on J_i [p33].

        for l in 9...56 {
            let amps = &AMPS_USED[l as usize - 9];

            let sum = amps.iter().fold(0, |sum, &x| sum + x + 1);
            assert_eq!(sum, l);

            assert!(l / 6 <= amps[0] + 1);

            for i in 0...4 {
                assert!(amps[i] <= amps[i + 1]);
            }

            assert!(amps[5] + 1 <= (l as f32 / 6.0).ceil() as usize);
        }
    }

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

        let c = CoefBlock::new(1, 8, &g, &amps, &p);
        assert_eq!(c.len(), 1);
        assert!((c.0[0] - 0.8519942560055926).abs() < 0.000001);
        assert!((c.idct(1) - 0.8519942560055926).abs() < 0.000001);

        let c = CoefBlock::new(2, 8, &g, &amps, &p);
        assert_eq!(c.len(), 1);
        assert!((c.0[0] - -0.13083074772702047).abs() < 0.000001);
        assert!((c.idct(1) - -0.13083074772702047).abs() < 0.000001);

        let c = CoefBlock::new(3, 8, &g, &amps, &p);
        assert_eq!(c.len(), 1);
        assert!((c.0[0] - -0.014229409757043066).abs() < 0.000001);
        assert!((c.idct(1) - -0.014229409757043066).abs() < 0.000001);

        let c = CoefBlock::new(4, 8, &g, &amps, &p);
        assert_eq!(c.len(), 2);
        assert!((c.0[0] - 2.0309896309891773).abs() < 0.000001);
        assert!((c.0[1] - -0.45129).abs() < 0.000001);
        assert!((c.idct(1) - 1.3927691924258232).abs() < 0.000001);
        assert!((c.idct(2) - 2.6692100695525314).abs() < 0.000001);

        let c = CoefBlock::new(5, 9, &g, &amps, &p);
        assert_eq!(c.len(), 2);
        assert!((c.0[0] - 0.5902767477270205).abs() < 0.000001);
        assert!((c.0[1] - -0.8289).abs() < 0.000001);
        assert!((c.idct(1) - -0.581964874124038).abs() < 0.000001);
        assert!((c.idct(2) - 1.7625183695780788).abs() < 0.000001);

        let c = CoefBlock::new(6, 10, &g, &amps, &p);
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

        let c = CoefBlock::new(1, 8, &g, &amps, &p);
        assert_eq!(c.len(), 2);
        assert!((c.0[0] - -1.2071545373041197).abs() < 0.000001);
        assert!((c.0[1] - 0.207225).abs() < 0.000001);
        assert!((c.idct(1) - -0.9140941318413551).abs() < 0.000001);
        assert!((c.idct(2) - -1.5002149427668843).abs() < 0.000001);

        let c = CoefBlock::new(2, 9, &g, &amps, &p);
        assert_eq!(c.len(), 2);
        assert!((c.0[0] - -1.524574246978134).abs() < 0.000001);
        assert!((c.0[1] - 0.990075).abs() < 0.000001);
        assert!((c.idct(1) - -0.1243967542115918).abs() < 0.000001);
        assert!((c.idct(2) - -2.924751739744676).abs() < 0.000001);

        let c = CoefBlock::new(3, 10, &g, &amps, &p);
        assert_eq!(c.len(), 3);
        assert!((c.0[0] - 0.5032515043523329).abs() < 0.000001);
        assert!((c.0[1] - 0.6447).abs() < 0.000001);
        assert!((c.0[2] - 0.5302).abs() < 0.000001);
        assert!((c.idct(1) - 2.1501046599919884).abs() < 0.000001);
        assert!((c.idct(2) - -0.557148495647667).abs() < 0.000001);
        assert!((c.idct(3) - -0.08320165128732226).abs() < 0.000001);

        let c = CoefBlock::new(4, 12, &g, &amps, &p);
        assert_eq!(c.len(), 3);
        assert!((c.0[0] - 1.481487836406659).abs() < 0.000001);
        assert!((c.0[1] - 0.7982).abs() < 0.000001);
        assert!((c.0[2] - 0.548275).abs() < 0.000001);
        assert!((c.idct(1) - 3.412285791008137).abs() < 0.000001);
        assert!((c.idct(2) - 0.38493783640665913).abs() < 0.000001);
        assert!((c.idct(3) - 0.6472398818051812).abs() < 0.000001);

        let c = CoefBlock::new(5, 14, &g, &amps, &p);
        assert_eq!(c.len(), 3);
        assert!((c.0[0] - 1.456220246978134).abs() < 0.000001);
        assert!((c.0[1] - 0.698425).abs() < 0.000001);
        assert!((c.0[2] - 0.391625).abs() < 0.000001);
        assert!((c.idct(1) - 3.0575528322544274).abs() < 0.000001);
        assert!((c.idct(2) - 0.672970246978134).abs() < 0.000001);
        assert!((c.idct(3) - 0.638137661701841).abs() < 0.000001);

        let c = CoefBlock::new(6, 16, &g, &amps, &p);
        assert_eq!(c.len(), 3);
        assert!((c.0[0] - 3.7141071965451276).abs() < 0.000001);
        assert!((c.0[1] - 0.099775).abs() < 0.000001);
        assert!((c.0[2] - 0.102425).abs() < 0.000001);
        assert!((c.idct(1) - 3.9893475658703124).abs() < 0.000001);
        assert!((c.idct(2) - 3.5092571965451276).abs() < 0.000001);
        assert!((c.idct(3) - 3.643716827219943).abs() < 0.000001);
    }
}
