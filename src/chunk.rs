use params::BaseParams;

pub type Chunks = [u32; 8];

#[derive(Copy, Clone)]
pub struct ChunkParts {
    parts: u32,
    voiced_len: u32,
}

impl ChunkParts {
    pub fn new(chunks: &Chunks, params: &BaseParams) -> ChunkParts {
        ChunkParts {
            parts: chunks[4] << 11 | chunks[5],
            voiced_len: params.bands,
        }
    }

    pub fn voiced(&self) -> u32 {
        self.parts >> (22 - self.voiced_len)
    }

    pub fn idx_part(&self) -> u32 {
        self.parts >> (20 - self.voiced_len) & 0b11
    }

    pub fn scanned(&self) -> u32 {
        self.parts & !0 >> (12 + self.voiced_len)
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

        assert_eq!(c.voiced(), 0b111101);
        assert_eq!(c.idx_part(), 0b10);
        assert_eq!(c.scanned(), 0b10100001111010);
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

        assert_eq!(c.voiced(), 0b1111);
        assert_eq!(c.idx_part(), 0b01);
        assert_eq!(c.scanned(), 0b1010100001111010);
    }
}
