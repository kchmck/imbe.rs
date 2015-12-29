use descramble::VoicedDecisions;
use params::BaseParams;
use spectral::Spectrals;
// use unvoiced::UnvoicedParts;

pub struct PrevFrame {
    pub params: BaseParams,
    pub spectrals: Spectrals,
    pub voiced: VoicedDecisions,
    pub err_rate: f32,
    pub energy: f32,
    pub amp_thresh: f32,
    // pub unvoiced: Unvoiced,
    pub phase_change: [f32; 56],
    pub phase: [f32; 56],
}

impl Default for PrevFrame {
    fn default() -> PrevFrame {
        PrevFrame {
            params: BaseParams::default(),
            spectrals: Spectrals::default(),
            voiced: VoicedDecisions::default(),
            err_rate: 0.0,
            energy: 75000.0,
            amp_thresh: 0.0, // not in document
            // unvoiced: [0.0; 256],
            phase_change: [0.0; 56],
            phase: [0.0; 56],
        }
    }
}
