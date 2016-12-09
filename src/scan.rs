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

/// Iterates over the chunks covered in the scanning procedure [p39].
pub struct ScanChunks<'a> {
    /// Chunks to scan.
    chunks: &'a Chunks,
    /// Separator chunk and its length in bits.
    sep: (u32, u8),
    /// Current chunk in scan.
    pos: Range<u8>,
}

impl<'a> ScanChunks<'a> {
    /// Create a new `ScanChunks` iterator over the given chunks.
    pub fn new(chunks: &'a Chunks, sep: u32, params: &BaseParams) -> Self {
        ScanChunks {
            chunks: chunks,
            // 22 - (K + 2) = 20 - K.
            sep: (sep, (20 - params.bands) as u8),
            // Each scan is made up of 7 chunks, each in whole or partial form.
            pos: 0..7,
        }
    }
}

/// At each iteration, yield a chunk of bits along with the number of LSBs to use from the
/// chunk.
impl<'a> Iterator for ScanChunks<'a> {
    type Item = (u32, u8);

    fn next(&mut self) -> Option<Self::Item> {
        self.pos.next().map(|n| {
            match n {
                // Last 3 LSBs of u_0.
                0 => (self.chunks[0] & 0b111, 3),
                // All of u_1, u_2, and u_3.
                1 => (self.chunks[1], 12),
                2 => (self.chunks[2], 12),
                3 => (self.chunks[3], 12),
                // Some LSBs of u_4/u_5.
                4 => self.sep,
                // All of u_6.
                5 => (self.chunks[6], 11),
                // First 3 MSBs of u_7.
                6 => (self.chunks[7] >> 4, 3),
                _ => unreachable!()
            }
        })
    }
}

/// Sequentially extracts the bits scanned into prioritized chunks.
pub struct ScanBits<'a> {
    /// Chunks in the scan.
    chunks: ScanChunks<'a>,
    /// Current chunk bits, stored starting at the MSB.
    chunk: u32,
    /// Bits remaining to yield from the current chunk.
    remain: u8,
}

impl<'a> ScanBits<'a> {
    /// Create a new `ScanBits` iterator over the given scan chunks.
    pub fn new(chunks: ScanChunks<'a>) -> Self {
        ScanBits {
            chunks: chunks,
            chunk: 0,
            remain: 0,
        }
    }
}

/// At each iteration, yield the next single bit in the scan.
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

            // Move bits to MSB of 32-bit word.
            self.chunk <<= 32 - self.remain;
        }

        // Extract the MSB and shift it off.
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
