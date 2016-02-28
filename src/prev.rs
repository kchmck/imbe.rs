use descramble::VoiceDecisions;
use enhance::{FrameEnergy, EnhancedSpectrals};
use params::BaseParams;
use spectral::Spectrals;
use unvoiced::UnvoicedDFT;
use voiced::{Phase, PhaseBase};

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
    fn default() -> PrevFrame {
        PrevFrame {
            params: BaseParams::default(),
            spectrals: Spectrals::default(),
            enhanced: EnhancedSpectrals::default(),
            voice: VoiceDecisions::default(),
            err_rate: 0.0,
            energy: FrameEnergy::default(),
            amp_thresh: 0.0, // not in document
            unvoiced: UnvoicedDFT::default(),
            phase_base: PhaseBase::default(),
            phase: Phase::default(),
        }
    }
}
