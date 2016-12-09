/// Represents the bit vectors u<sub>0</sub>, ..., u<sub>7</sub>, in that order.
pub type Chunks = [u32; 8];

/// Represents the number of detected Hamming/Golay bit errors, ϵ<sub>0</sub>, ...,
/// ϵ<sub>6</sub>, corresponding to the chunks u<sub>0</sub>, ..., u<sub>6</sub>.
pub type Errors = [usize; 7];

/// A received IMBE voice frame.
pub struct ReceivedFrame {
    /// Prioritized bit vector chunks, u<sub>0</sub>, ..., u<sub>7</sub>.
    pub chunks: Chunks,
    /// Error correction counts, ϵ<sub>0</sub>, ..., ϵ<sub>6</sub>.
    pub errors: Errors,
}

impl ReceivedFrame {
    /// Create a new `ReceivedFrame` from the given chunks u<sub>0</sub>, ...,
    /// u<sub>7</sub> and error counts ϵ<sub>0</sub>, ..., ϵ<sub>6</sub>.
    pub fn new(chunks: Chunks, errors: Errors) -> ReceivedFrame {
        // First 4 chunks must have at most 12 bits.
        for i in 0...3 {
            assert!(chunks[i] >> 12 == 0);
        }

        // Next 3 chunks must have at most 11 bits.
        for i in 4...6 {
            assert!(chunks[i] >> 11 == 0);
        }

        // Final chunk must have at most 7 bits.
        assert!(chunks[7] >> 7 == 0);

        ReceivedFrame {
            chunks: chunks,
            errors: errors,
        }
    }
}
