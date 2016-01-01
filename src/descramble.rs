use arrayvec::ArrayVec;

use std::cmp::min;

use params::BaseParams;
use scan::{ScanBits, ScanChunks};
use chunk::{Chunks, ChunkParts};
use allocs::allocs;

pub fn descramble(chunks: &Chunks, params: &BaseParams) ->
    (QuantizedAmplitudes, VoiceDecisions, usize)
{
    let parts = ChunkParts::new(chunks, params);
    let scan = ScanBits::new(ScanChunks::new(chunks, parts.scanned(), &params));

    (
        QuantizedAmplitudes::new(scan, params),
        VoiceDecisions::new(parts.voiced(), params),
        gain_idx(chunks, parts.idx_part()),
    )
}

// pub fn descramble(...) -> (u32, usize

#[derive(Copy, Clone)]
pub enum Bootstrap {
    Period(u8),
    Silence,
    Invalid,
}

impl Bootstrap {
    pub fn new(chunks: &Chunks) -> Bootstrap {
        match (chunks[0] >> 4) as u8 & 0xFC | (chunks[7] >> 1) as u8 & 0b11 {
            period @ 0...207 => Bootstrap::Period(period),
            216...219 => Bootstrap::Silence,
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

pub fn gain_idx(chunks: &Chunks, idx_part: u32) -> usize {
    (chunks[0] & 0x38 | idx_part << 1 | chunks[7] >> 3 & 1) as usize
}

pub struct QuantizedAmplitudes(ArrayVec<[u32; 64]>);

impl QuantizedAmplitudes {
    fn new(mut scan: ScanBits, params: &BaseParams) -> QuantizedAmplitudes {
        let mut amps: ArrayVec<[u32; 64]> = (0..params.harmonics-1).map(|_| 0).collect();

        let (max, bits) = allocs(params.harmonics);

        for idx in (0..max).rev() {
            // Visit indexes [3, L+1].
            for i in 0..amps.len() {
                if bits[i] <= idx {
                    continue;
                }

                amps[i] <<= 1;
                amps[i] |= scan.next().unwrap();
            }
        }

        assert!(scan.next().is_none());

        QuantizedAmplitudes(amps)
    }

    // 1-based index starting at 3
    pub fn get(&self, idx: usize) -> u32 { self.0[idx - 3] }
}

#[derive(Copy, Clone)]
pub struct VoiceDecisions {
    params: BaseParams,
    voiced: u32,
    pub unvoiced_count: u32,
}

impl VoiceDecisions {
    pub fn new(voiced: u32, params: &BaseParams) -> VoiceDecisions {
        VoiceDecisions {
            params: params.clone(),
            voiced: voiced,
            unvoiced_count: (params.bands - voiced.count_ones()) * 3,
        }
    }

    pub fn force_voiced(&mut self, l: usize) {
        self.voiced |= self.mask(l)
    }

    pub fn is_voiced(&self, l: usize) -> bool {
        if l as u32 > self.params.harmonics {
            false
        } else {
            self.voiced & self.mask(l) != 0
        }
    }

    fn mask(&self, l: usize) -> u32 {
        self.band_mask(min((l + 2) / 3, 12))
    }

    fn band_mask(&self, idx: usize) -> u32 {
        assert!(idx >= 1 && idx <= self.params.bands as usize);
        1 << (self.params.bands as usize - idx)
    }
}

impl Default for VoiceDecisions {
    fn default() -> VoiceDecisions {
        let params = BaseParams::default();

        VoiceDecisions {
            params: params,
            unvoiced_count: params.bands,
            voiced: 0,
        }
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

        assert_eq!(voice.unvoiced_count, 9);

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

        assert_eq!(voice.unvoiced_count, 3);

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

        assert!(!voice.is_voiced(11));
        assert!(!voice.is_voiced(12));
        assert!(!voice.is_voiced(13));
        assert!(!voice.is_voiced(14));
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
}
