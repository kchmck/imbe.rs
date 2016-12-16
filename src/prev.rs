//! Previous frame saved parameters.

use descramble::VoiceDecisions;
use enhance::{FrameEnergy, EnhancedSpectrals};
use params::BaseParams;
use spectral::Spectrals;
use unvoiced::UnvoicedDFT;
use voiced::{Phase, PhaseBase};

/// Various parameters saved from the previous frame, used when constructing the current
/// frame.
pub struct PrevFrame {
    pub params: BaseParams,
    pub spectrals: Spectrals,
    pub enhanced: EnhancedSpectrals,
    pub voice: VoiceDecisions,
    pub err_rate: f32,
    pub energy: FrameEnergy,
    pub amp_thresh: f32,
    pub unvoiced: UnvoicedDFT,
    pub phase_base: PhaseBase,
    pub phase: Phase,
}

impl Default for PrevFrame {
    /// Create a new `PrevFrame` suitable for decoding the very first IMBE frame in a
    /// stream.
    fn default() -> PrevFrame {
        PrevFrame {
            params: BaseParams::default(),
            spectrals: Spectrals::default(),
            enhanced: EnhancedSpectrals::default(),
            voice: VoiceDecisions::default(),
            // Taken from [p64].
            err_rate: 0.0,
            energy: FrameEnergy::default(),
            // This value is arbitrary and not given in the standard.
            amp_thresh: 0.0,
            unvoiced: UnvoicedDFT::default(),
            phase_base: PhaseBase::default(),
            phase: Phase::default(),
        }
    }
}
