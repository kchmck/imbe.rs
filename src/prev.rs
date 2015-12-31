use descramble::VoicedDecisions;
use enhance::{FrameEnergy, EnhancedSpectrals};
use params::BaseParams;
use spectral::Spectrals;
use unvoiced::UnvoicedParts;
use voice::{Phase, PhaseBase};

pub struct PrevFrame {
    pub params: BaseParams,
    pub spectrals: Spectrals,
    pub enhanced: EnhancedSpectrals,
    pub voice: VoicedDecisions,
    pub err_rate: f32,
    pub energy: FrameEnergy,
    pub amp_thresh: f32,
    pub unvoiced: UnvoicedParts,
    pub phase_base: PhaseBase,
    pub phase: Phase,
}

impl Default for PrevFrame {
    fn default() -> PrevFrame {
        PrevFrame {
            params: BaseParams::default(),
            spectrals: Spectrals::default(),
            enhanced: EnhancedSpectrals::default(),
            voice: VoicedDecisions::default(),
            err_rate: 0.0,
            energy: FrameEnergy::default(),
            amp_thresh: 0.0, // not in document
            unvoiced: UnvoicedParts::default(),
            phase_base: PhaseBase::default(),
            phase: Phase::default(),
        }
    }
}
