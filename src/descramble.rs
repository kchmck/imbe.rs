//! Initial descrambling of prioritized chunks.

use arrayvec::ArrayVec;

use allocs::allocs;
use frame::Chunks;
use params::BaseParams;
use scan::{ScanSep, ScanBits, ScanChunks};

/// Descramble the given prioritized chunks u<sub>i</sub> into the underlying quantized
/// amplitudes b<sub>m</sub>, voiced/unvoiced decisions v<sub>l</sub>, and initial gain
/// index b<sub>2</sub>.
pub fn descramble(chunks: &Chunks, params: &BaseParams) ->
    (QuantizedAmplitudes, VoiceDecisions, usize)
{
    // Extract the data in between the two scans.
    let parts = ScanSep::new(chunks, params);

    (
        QuantizedAmplitudes::new(ScanBits::new(
            ScanChunks::new(chunks, parts.scanned, &params)), params),
        VoiceDecisions::new(parts.voiced, params),
        gain_idx(chunks, parts.idx_part),
    )
}

/// Decodes the bootstrap value b<sub>0</sub>.
#[derive(Copy, Clone)]
pub enum Bootstrap {
    /// Frame contains voiced/unvoiced data derived from enclosed b<sub>0</sub> parameter.
    Period(u8),
    /// Frame is silence.
    Silence,
    /// Invalid b<sub>0</sub> value was detected.
    Invalid,
}

impl Bootstrap {
    /// Parse a `Bootstrap` value from the given chunks.
    pub fn new(chunks: &Chunks) -> Bootstrap {
        match period(chunks) {
            p @ 0...207 => Bootstrap::Period(p),
            216...219 => Bootstrap::Silence,
            // Rest of the values are invalid [p46].
            _ => Bootstrap::Invalid,
        }
    }

    #[cfg(test)]
    pub fn unwrap_period(&self) -> u8 {
        if let Bootstrap::Period(period) = *self {
            period
        } else {
            panic!("attempted unwrap of invalid/silence period");
        }
    }
}

/// Compute the 8-bit quantized period b<sub>0</sub> from the given u<sub>0</sub>, ...,
/// s<sub>7</sub>.
fn period(chunks: &Chunks) -> u8 {
    // Concatenate MSBs of u_0 with bits 2 and 1 of u_7 [p39].
    (chunks[0] >> 4) as u8 & 0b11111100 | (chunks[7] >> 1) as u8 & 0b11
}

/// Compute the 6-bit index, b<sub>2</sub>, used to look up the quantized gain value [p30].
fn gain_idx(chunks: &Chunks, idx_part: u32) -> usize {
    // Concatenate bits 5 through 3 of u_0, the two index bits, and bit 3 of u_7 [p39].
    (chunks[0] & 0b111000 | idx_part << 1 | chunks[7] >> 3 & 1) as usize
}

/// Reconstructs quantized amplitudes b<sub>3</sub>, ..., b<sub>L+1</sub>.
pub struct QuantizedAmplitudes(ArrayVec<[u32; 64]>);

impl QuantizedAmplitudes {
    /// Reconstruct quantized amplitudes from the given bit scan.
    fn new(mut scan: ScanBits, params: &BaseParams) -> QuantizedAmplitudes {
        // Since 3 ≤ m ≤ L + 1, let i = m - 3. Then 0 ≤ i ≤ L + 1 - 3 = L - 2.
        //
        // The underlying array has a maximum length of 64 because that's the closest impl
        // provided by arrayvec.
        let mut amps: ArrayVec<[u32; 64]> = (1..params.harmonics).map(|_| 0).collect();

        let (bits, max) = allocs(params.harmonics);

        // Iterate through bit levels, MSB to LSB.
        for idx in (0..max).rev() {
            // Iterate over all b_i.
            for i in 0..amps.len() {
                // Skip this b_i if there are no bits allocated to it at this bit level.
                if bits[i] <= idx {
                    continue;
                }

                // Shift the next scanned bit onto the LSB.
                amps[i] <<= 1;
                amps[i] |= scan.next().unwrap();
            }
        }

        assert!(scan.next().is_none());

        QuantizedAmplitudes(amps)
    }

    /// Retrieve the quantized amplitude b<sub>m</sub>, 3 ≤ m ≤ L + 1.
    pub fn get(&self, m: usize) -> u32 { self.0[m - 3] }
}

/// Tracks harmonic voiced/unvoiced decisions.
///
/// The decisions are initialized from the received "tilde" v<sub>l</sub> vector [p25-26],
/// and then may be modified by adaptive smoothing to force certain harmonics as voiced
/// [p49], which forms the "overbar" v<sub>l</sub> vector.
#[derive(Copy, Clone)]
pub struct VoiceDecisions {
    /// Parameters for the current frame.
    params: BaseParams,
    /// The voiced/unvoiced bitmap, b<sub>1</sub>.
    ///
    /// Each bit indicates whether the harmonic l, 1 ≤ l ≤ 56, is voiced.
    voiced: u64,
}

impl VoiceDecisions {
    /// Create a new `VoiceDecisions` from the given band voiced/unvoiced bitmap,
    /// b<sub>1</sub>, and frame parameters.
    pub fn new(voiced: u32, params: &BaseParams) -> VoiceDecisions {
        VoiceDecisions {
            params: params.clone(),
            voiced: gen_harmonics_bitmap(voiced, params),
        }
    }

    /// Force the given harmonic to be voiced.
    pub fn force_voiced(&mut self, l: usize) {
        self.voiced |= self.mask(l);
    }

    /// Compute the number of unvoiced spectral amplitudes/harmonics, L<sub>uv</sub>.
    pub fn unvoiced_count(&self) -> u32 {
        self.params.harmonics - self.voiced.count_ones()
    }

    /// Check if the given harmonic is voiced.
    pub fn is_voiced(&self, l: usize) -> bool {
        if l as u32 > self.params.harmonics {
            false
        } else {
            self.voiced & self.mask(l) != 0
        }
    }

    /// Create a 1-bit mask for the bit position of the given harmonic.
    fn mask(&self, l: usize) -> u64 {
        1 << (self.params.harmonics as usize - l)
    }
}

impl Default for VoiceDecisions {
    /// Create a new `VoiceDecisions` in default state.
    fn default() -> Self {
        // As default, all harmonics are unvoiced [p64].
        VoiceDecisions::new(0, &BaseParams::default())
    }
}

/// Create a bitmap for harmonic voiced/unvoiced decisions from the given voiced/unvoiced
/// band bitmap b<sub>1</sub> and frame parameters.
///
/// The resulting bitmap has the LSB represent harmonic L, and so on through more
/// significant bits until harmonic 1.
fn gen_harmonics_bitmap(voiced: u32, params: &BaseParams) -> u64 {
    // Each voiced/unvoiced band (except possibly the last) contains 3 harmonics.
    let bits = (0..params.bands).rev().fold(0, |bits, i| {
        bits << 3 | if voiced >> i & 1 == 1 {
            0b111
        } else {
            0
        }
    });

    // Check if the last band contains less than 3 harmonics.
    let rem = params.harmonics % 3;

    if rem == 0 {
        bits
    } else {
        // Strip off extra harmonics.
        bits >> 3 - rem
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use params::BaseParams;

    #[test]
    #[should_panic]
    fn test_invalid_period() {
        let chunks = [
            0b111111000000,
            0, 0, 0, 0, 0, 0,
            0b00000010,
        ];

        Bootstrap::new(&chunks).unwrap_period();
    }

    #[test]
    fn test_descramble_16() {
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
        let (amps, voice, gain_idx) = descramble(&chunks, &p);

        assert_eq!(p.harmonics, 16);
        assert_eq!(p.bands, 6);
        assert_eq!(gain_idx, 0b010101);

        assert_eq!(amps.get(3),  0b000110);
        assert_eq!(amps.get(4),  0b100010);
        assert_eq!(amps.get(5),  0b011011);
        assert_eq!(amps.get(6),  0b11001);
        assert_eq!(amps.get(7),  0b01111);
        assert_eq!(amps.get(8),  0b100100);
        assert_eq!(amps.get(9),  0b110101);
        assert_eq!(amps.get(10), 0b10111);
        assert_eq!(amps.get(11), 0b1101);
        assert_eq!(amps.get(12), 0b1110);
        assert_eq!(amps.get(13), 0b111);
        assert_eq!(amps.get(14), 0b111);
        assert_eq!(amps.get(15), 0b110);
        assert_eq!(amps.get(16), 0b100);
        assert_eq!(amps.get(17), 0b10);

        assert_eq!(voice.unvoiced_count(), 9);

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
        assert!(!voice.is_voiced(13));
        assert!(!voice.is_voiced(14));
        assert!(!voice.is_voiced(15));
        assert!(voice.is_voiced(16));

        for l in 17..63 {
            assert!(!voice.is_voiced(l));
        }
    }

    #[test]
    fn test_descramble_10() {
        let chunks = [
            0b000001010010,
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
        let (amps, voice, gain_idx) = descramble(&chunks, &p);

        assert_eq!(p.harmonics, 10);
        assert_eq!(p.bands, 4);
        assert_eq!(gain_idx, 0b010011);

        assert_eq!(amps.get(3),  0b010101011);
        assert_eq!(amps.get(4),  0b110101101);
        assert_eq!(amps.get(5),  0b01001011);
        assert_eq!(amps.get(6),  0b01011000);
        assert_eq!(amps.get(7),  0b10011101);
        assert_eq!(amps.get(8),  0b010111011);
        assert_eq!(amps.get(9),  0b1111110);
        assert_eq!(amps.get(10), 0b110110);
        assert_eq!(amps.get(11), 0b11100);

        assert_eq!(voice.unvoiced_count(), 3);

        assert!(voice.is_voiced(1));
        assert!(voice.is_voiced(2));
        assert!(voice.is_voiced(3));
        assert!(voice.is_voiced(4));
        assert!(voice.is_voiced(5));
        assert!(voice.is_voiced(6));
        assert!(!voice.is_voiced(7));
        assert!(!voice.is_voiced(8));
        assert!(!voice.is_voiced(9));
        assert!(voice.is_voiced(10));

        for l in 11...63 {
            assert!(!voice.is_voiced(l));
        }
    }

    #[test]
    fn test_descramble_16_2() {
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
        let (_, voice, _) = descramble(&chunks, &p);

        assert_eq!(p.harmonics, 16);
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

        for l in 17...63 {
            assert!(!voice.is_voiced(l));
        }
    }

    #[test]
    fn test_voice() {
        let p = BaseParams::new(32);
        assert_eq!(p.harmonics, 16);
        assert_eq!(p.bands, 6);
        let v = VoiceDecisions::new(0b101011, &p);
        assert_eq!(v.unvoiced_count(), 6);
        assert_eq!(v.voiced, 0b1110001110001111);
        let v = VoiceDecisions::new(0b101010, &p);
        assert_eq!(v.unvoiced_count(), 7);
        assert_eq!(v.voiced, 0b1110001110001110);
        let v = VoiceDecisions::new(0b001001, &p);
        assert_eq!(v.unvoiced_count(), 12);
        assert_eq!(v.voiced, 0b0000001110000001);
        let v = VoiceDecisions::new(0b101000, &p);
        assert_eq!(v.unvoiced_count(), 10);
        assert_eq!(v.voiced, 0b1110001110000000);
        let v = VoiceDecisions::new(0b000000, &p);
        assert_eq!(v.unvoiced_count(), 16);
        assert_eq!(v.voiced, 0b0000000000000000);
        let v = VoiceDecisions::new(0b111111, &p);
        assert_eq!(v.unvoiced_count(), 0);
        assert_eq!(v.voiced, 0b1111111111111111);

        let p = BaseParams::new(36);
        assert_eq!(p.harmonics, 17);
        assert_eq!(p.bands, 6);
        let v = VoiceDecisions::new(0b101011, &p);
        assert_eq!(v.unvoiced_count(), 6);
        assert_eq!(v.voiced, 0b11100011100011111);
        let v = VoiceDecisions::new(0b101010, &p);
        assert_eq!(v.unvoiced_count(), 8);
        assert_eq!(v.voiced, 0b11100011100011100);
        let v = VoiceDecisions::new(0b001001, &p);
        assert_eq!(v.unvoiced_count(), 12);
        assert_eq!(v.voiced, 0b00000011100000011);
        let v = VoiceDecisions::new(0b101000, &p);
        assert_eq!(v.unvoiced_count(), 11);
        assert_eq!(v.voiced, 0b11100011100000000);
        let v = VoiceDecisions::new(0b000000, &p);
        assert_eq!(v.unvoiced_count(), 17);
        assert_eq!(v.voiced, 0b00000000000000000);
        let v = VoiceDecisions::new(0b111111, &p);
        assert_eq!(v.unvoiced_count(), 0);
        assert_eq!(v.voiced, 0b11111111111111111);

        let p = BaseParams::new(40);
        assert_eq!(p.harmonics, 18);
        assert_eq!(p.bands, 6);
        let v = VoiceDecisions::new(0b101011, &p);
        assert_eq!(v.unvoiced_count(), 6);
        assert_eq!(v.voiced, 0b111000111000111111);
        let v = VoiceDecisions::new(0b101010, &p);
        assert_eq!(v.unvoiced_count(), 9);
        assert_eq!(v.voiced, 0b111000111000111000);
        let v = VoiceDecisions::new(0b001001, &p);
        assert_eq!(v.unvoiced_count(), 12);
        assert_eq!(v.voiced, 0b000000111000000111);
        let v = VoiceDecisions::new(0b101000, &p);
        assert_eq!(v.unvoiced_count(), 12);
        assert_eq!(v.voiced, 0b111000111000000000);
        let v = VoiceDecisions::new(0b000000, &p);
        assert_eq!(v.unvoiced_count(), 18);
        assert_eq!(v.voiced, 0b000000000000000000);
        let v = VoiceDecisions::new(0b111111, &p);
        assert_eq!(v.unvoiced_count(), 0);
        assert_eq!(v.voiced, 0b111111111111111111);
    }

    #[test]
    #[should_panic]
    fn test_amp_bounds() {
        let chunks = [
            0b000001010010,
            0b110011001100,
            0b111000111000,
            0b111111111111,
            0b11110110101,
            0b00101111010,
            0b01110111011,
            0b00001000,
        ];

        let b = Bootstrap::new(&chunks);
        let p = BaseParams::new(b.unwrap_period());
        let (amps, _, _) = descramble(&chunks, &p);

        assert_eq!(p.harmonics, 10);
        amps.get(12);
    }

    #[test]
    fn test_gain_idx() {
        let chunks = [
            0b111111010111,
            0b111111111111,
            0b111111111111,
            0b111111111111,
            0b11111111111,
            0b11111111111,
            0b11111111111,
            0b1110111,
        ];

        assert_eq!(gain_idx(&chunks, 0b01), 0b010010);
    }

    #[test]
    fn test_period() {
        let chunks = [
            0b010101111111,
            0b111111111111,
            0b111111111111,
            0b111111111111,
            0b11111111111,
            0b11111111111,
            0b11111111111,
            0b1111011,
        ];

        assert_eq!(period(&chunks), 0b01010101);
    }
}
