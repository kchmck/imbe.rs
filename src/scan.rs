use std::ops::Range;

use params::BaseParams;
use frame::Chunks;

/// Decodes voiced/unvoiced decisions and the quantized gain index fragment from
/// prioritized chunks.
///
/// Between the chunks created by the two scanning procedures, there is a 22-bit
/// vector made up of u<sub>4</sub> and u<sub>5</sub> that contains the
/// voiced/unvoiced vector, part of the quantized gain index, and the initial bits in
/// the second scanning procedure.
#[derive(Copy, Clone)]
pub struct ScanSep {
    /// Voiced/Unvoiced Boolean bit vector, b<sub>1</sub> [p25].
    pub voiced: u32,
    /// Bits 1 an 2 of the 6-bit quantized gain index, b<sub>2</sub> [p30].
    pub idx_part: u32,
    /// Chunk of some b<sub>m</sub> used in the scanning procedure [p39].
    pub scanned: u32,
}

impl ScanSep {
    /// Create a new `ScanSep` decoder from the given chunks and frame parameters.
    pub fn new(chunks: &Chunks, params: &BaseParams) -> ScanSep {
        // Concatenate u_4 and u_5 into a 22-bit vector.
        let parts = chunks[4] << 11 | chunks[5];

        ScanSep {
            // Take first K MSBs as the voiced/unvoiced vector [p39].
            voiced: parts >> (22 - params.bands),
            // Take next 2 bits as bit 1 and 2 of b_2 [p39].
            idx_part: parts >> (20 - params.bands) & 0b11,
            // Take the remaining K-2 LSBs as part of the scanning procedure [p39].
            // Since the underlying word is 32 bits, (32-22) + K+2 = 12 + K.
            scanned: parts & !0 >> (12 + params.bands),
        }
    }
}

pub struct ScanChunks<'a> {
    pos: Range<u8>,
    chunks: &'a Chunks,
    scanned: u32,
    scanned_len: u8,
}

impl<'a> ScanChunks<'a> {
    pub fn new(chunks: &'a Chunks, scanned: u32, params: &BaseParams) -> ScanChunks<'a> {
        ScanChunks {
            pos: 0..7,
            chunks: chunks,
            scanned: scanned,
            scanned_len: (20 - params.bands) as u8,
        }
    }
}

impl<'a> Iterator for ScanChunks<'a> {
    type Item = (u32, u8);

    fn next(&mut self) -> Option<Self::Item> {
        self.pos.next().map(|n| {
            match n {
                0 => (self.chunks[0] & 0b111, 3),
                1 => (self.chunks[1], 12),
                2 => (self.chunks[2], 12),
                3 => (self.chunks[3], 12),
                4 => (self.scanned, self.scanned_len),
                5 => (self.chunks[6], 11),
                6 => (self.chunks[7] >> 4, 3),
                _ => unreachable!()
            }
        })
    }
}

pub struct ScanBits<'a> {
    chunks: ScanChunks<'a>,
    chunk: u32,
    remain: u8,
}

impl<'a> ScanBits<'a> {
    pub fn new(chunks: ScanChunks<'a>) -> ScanBits<'a> {
        ScanBits {
            chunks: chunks,
            chunk: 0,
            remain: 0,
        }
    }
}

impl<'a> Iterator for ScanBits<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remain == 0 {
            let (c, r) = match self.chunks.next() {
                Some(x) => x,
                None => return None,
            };

            self.chunk = c;
            self.remain = r;

            self.chunk <<= 32 - self.remain;
        }

        let bit = self.chunk >> 31;
        self.chunk <<= 1;
        self.remain -= 1;

        Some(bit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use params::BaseParams;

    #[test]
    fn test_chunks_16() {
        let chunks = [
            0b101,
            0b101010101010,
            0b101010101010,
            0b101010101010,
            0b11111111111,
            0b01010101010,
            0b10101010101,
            0b1010000,
        ];

        let p = BaseParams::new(32);

        assert_eq!(p.harmonics, 16);
        assert_eq!(p.bands, 6);

        let parts = ScanSep::new(&chunks, &p);
        let mut c = ScanChunks::new(&chunks, parts.scanned, &p);

        assert_eq!(c.next().unwrap(), (0b101, 3));
        assert_eq!(c.next().unwrap(), (0b101010101010, 12));
        assert_eq!(c.next().unwrap(), (0b101010101010, 12));
        assert_eq!(c.next().unwrap(), (0b101010101010, 12));
        assert_eq!(c.next().unwrap(), (0b11101010101010, 14));
        assert_eq!(c.next().unwrap(), (0b10101010101, 11));
        assert_eq!(c.next().unwrap(), (0b101, 3));
    }

    #[test]
    fn test_chunks_10() {
        let chunks = [
            0b111111111101,
            0b101010101010,
            0b101010101010,
            0b101010101010,
            0b11111111111,
            0b01010101010,
            0b10101010101,
            0b1010000,
        ];

        let p = BaseParams::new(4);

        assert_eq!(p.harmonics, 10);
        assert_eq!(p.bands, 4);

        let parts = ScanSep::new(&chunks, &p);
        let mut c = ScanChunks::new(&chunks, parts.scanned, &p);

        assert_eq!(c.next().unwrap(), (0b101, 3));
        assert_eq!(c.next().unwrap(), (0b101010101010, 12));
        assert_eq!(c.next().unwrap(), (0b101010101010, 12));
        assert_eq!(c.next().unwrap(), (0b101010101010, 12));
        assert_eq!(c.next().unwrap(), (0b1111101010101010, 16));
        assert_eq!(c.next().unwrap(), (0b10101010101, 11));
        assert_eq!(c.next().unwrap(), (0b101, 3));
    }

    #[test]
    fn test_bits_16() {
        let chunks = [
            0b111111111101,
            0b010101010101,
            0b010101010101,
            0b010101010101,
            0b11111111111,
            0b01010101010,
            0b10101010101,
            0b1010000,
        ];

        let p = BaseParams::new(32);

        assert_eq!(p.harmonics, 16);
        assert_eq!(p.bands, 6);

        let parts = ScanSep::new(&chunks, &p);
        let mut c = ScanBits::new(ScanChunks::new(&chunks, parts.scanned, &p));

        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);

        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);

        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);

        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);

        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);

        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);

        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
    }

    #[test]
    fn test_bits_10() {
        let chunks = [
            0b111111111101,
            0b010101010101,
            0b010101010101,
            0b010101010101,
            0b11111111111,
            0b01010101010,
            0b10101010101,
            0b1010000,
        ];

        let p = BaseParams::new(4);

        assert_eq!(p.harmonics, 10);
        assert_eq!(p.bands, 4);

        let parts = ScanSep::new(&chunks, &p);
        let mut c = ScanBits::new(ScanChunks::new(&chunks, parts.scanned, &p));

        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);

        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);

        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);

        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);

        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);

        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);

        assert_eq!(c.next().unwrap(), 1);
        assert_eq!(c.next().unwrap(), 0);
        assert_eq!(c.next().unwrap(), 1);
    }

    #[test]
    fn test_parts_16() {
        let chunks = [
            0, 0, 0, 0,
            0b11110110101,
            0b00001111010,
            0, 0,
        ];

        let p = BaseParams::new(32);

        assert_eq!(p.harmonics, 16);
        assert_eq!(p.bands, 6);

        let c = ScanSep::new(&chunks, &p);

        assert_eq!(c.voiced, 0b111101);
        assert_eq!(c.idx_part, 0b10);
        assert_eq!(c.scanned, 0b10100001111010);
    }

    #[test]
    fn test_parts_10() {
        let chunks = [
            0, 0, 0, 0,
            0b11110110101,
            0b00001111010,
            0, 0,
        ];

        let p = BaseParams::new(4);

        assert_eq!(p.harmonics, 10);
        assert_eq!(p.bands, 4);

        let c = ScanSep::new(&chunks, &p);

        assert_eq!(c.voiced, 0b1111);
        assert_eq!(c.idx_part, 0b01);
        assert_eq!(c.scanned, 0b1010100001111010);
    }
}
