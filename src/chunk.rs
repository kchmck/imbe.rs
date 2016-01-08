use params::BaseParams;

pub type Chunks = [u32; 8];

#[derive(Copy, Clone)]
pub struct ChunkParts {
    pub voiced: u32,
    pub idx_part: u32,
    pub scanned: u32,
}

impl ChunkParts {
    pub fn new(chunks: &Chunks, params: &BaseParams) -> ChunkParts {
        let parts = chunks[4] << 11 | chunks[5];

        ChunkParts {
            voiced: parts >> (22 - params.bands),
            idx_part: parts >> (20 - params.bands) & 0b11,
            scanned: parts & !0 >> (12 + params.bands),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use params::BaseParams;

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

        let c = ChunkParts::new(&chunks, &p);

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

        let c = ChunkParts::new(&chunks, &p);

        assert_eq!(c.voiced, 0b1111);
        assert_eq!(c.idx_part, 0b01);
        assert_eq!(c.scanned, 0b1010100001111010);
    }
}
