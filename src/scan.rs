use std;
use std::ops::Range;

use params::BaseParams;
use chunk::{Chunks, ChunkParts};

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
        Some(match self.pos.next() {
            Some(n) => match n {
                0 => (self.chunks[0] & 0b111, 3),
                1 => (self.chunks[1], 12),
                2 => (self.chunks[2], 12),
                3 => (self.chunks[3], 12),
                4 => (self.scanned, self.scanned_len),
                5 => (self.chunks[6], 11),
                6 => (self.chunks[7] >> 4, 3),
                _ => unreachable!()
            },
            None => return None,
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
    use super::{ScanChunks, ScanBits};
    use chunk::ChunkParts;
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

        let parts = ChunkParts::new(&chunks, &p);
        let mut c = ScanChunks::new(&chunks, parts.scanned(), &p);

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

        let parts = ChunkParts::new(&chunks, &p);
        let mut c = ScanChunks::new(&chunks, parts.scanned(), &p);

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

        let parts = ChunkParts::new(&chunks, &p);
        let mut c = ScanBits::new(ScanChunks::new(&chunks, parts.scanned(), &p));

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

        let parts = ChunkParts::new(&chunks, &p);
        let mut c = ScanBits::new(ScanChunks::new(&chunks, parts.scanned(), &p));

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
}
