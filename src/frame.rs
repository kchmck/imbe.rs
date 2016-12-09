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
        ReceivedFrame {
            chunks: chunks,
            errors: errors,
        }
    }
}
