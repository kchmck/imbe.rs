use params::BaseParams;

/// Represents the bit vectors u<sub>0</sub>, ..., u<sub>7</sub>, in that order.
pub type Chunks = [u32; 8];

/// Decodes voiced/unvoiced decisions and the quantized gain index fragment from
/// prioritized chunks.
///
/// Between the chunks created by the two scanning procedures, there is a 22-bit
/// vector made up of u<sub>4</sub> and u<sub>5</sub> that contains the
/// voiced/unvoiced vector, part of the quantized gain index, and the initial bits in
/// the second scanning procedure.
#[derive(Copy, Clone)]
pub struct ChunkParts {
    /// Voiced/Unvoiced Boolean bit vector, b<sub>1</sub> [p25].
    pub voiced: u32,
    /// Bits 1 an 2 of the 6-bit quantized gain index, b<sub>2</sub> [p30].
    pub idx_part: u32,
    /// Chunk of some b<sub>m</sub> used in the scanning procedure [p39].
    pub scanned: u32,
}

impl ChunkParts {
    /// Create a new `ChunkParts` decoder from the given chunks and frame parameters.
    pub fn new(chunks: &Chunks, params: &BaseParams) -> ChunkParts {
        // Concatenate u_4 and u_5 into a 22-bit vector.
        let parts = chunks[4] << 11 | chunks[5];

        ChunkParts {
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
