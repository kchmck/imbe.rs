use coefs::Coefficients;
use descramble::{descramble, Bootstrap};
use enhance::{self, EnhancedSpectrals, FrameEnergy};
use error::{IMBEResult, IMBEError};
use errors::Errors;
use gain::Gains;
use params::BaseParams;
use prev::PrevFrame;
use spectral::Spectrals;
use unvoiced::{UnvoicedDFT, UnvoicedParts, Unvoiced};
use voiced::{Phase, PhaseBase, Voiced};

pub struct CAIFrame {
    chunks: [u32; 8],
    errors: [usize; 7],
}

pub struct IMBEDecoder {
    prev: PrevFrame,
}

impl IMBEDecoder {
    pub fn new() -> IMBEDecoder {
        IMBEDecoder {
            prev: PrevFrame::default(),
        }
    }

    pub fn decode(&mut self, frame: CAIFrame) -> IMBEResult<()> {
        let period = match Bootstrap::new(&frame.chunks) {
            Bootstrap::Period(p) => p,
            Bootstrap::Invalid => {
                // repeat
                return Err(IMBEError::Derp)
            },
            Bootstrap::Silence => {
                // output silence
                return Ok(())
            },
        };

        let errors = Errors::new(&frame.errors, self.prev.err_rate);

        if enhance::should_repeat(&errors) {
            // repeat
            return Ok(())
        }

        if enhance::should_mute(&errors) {
            // output silence
            return Ok(())
        }

        let params = BaseParams::new(period);
        let (amps, mut voice, gain_idx) = descramble(&frame.chunks, &params);
        let gains = Gains::new(gain_idx, &amps, &params);
        let coefs = Coefficients::new(&gains, &amps, &params);
        let spectrals = Spectrals::new(&coefs, &params, &self.prev);
        let energy = FrameEnergy::new(&spectrals, &self.prev.energy, &params);

        let mut enhanced = EnhancedSpectrals::new(&spectrals, &energy, &params);
        enhance::smooth(&mut enhanced, &mut voice, &errors, &energy, self.prev.amp_thresh);

        // separate out following

        let udft = UnvoicedDFT::new();
        let uparts = UnvoicedParts::new(&udft, &params, &voice, &spectrals);
        let unvoiced = Unvoiced::new(&uparts, &self.prev.unvoiced);

        let vbase = PhaseBase::new(&params, &self.prev);
        let vphase = Phase::new(&params, &voice, &vbase, &self.prev);
        let voiced = Voiced::new(&params, &self.prev, &vphase, &enhanced, &voice);

        self.prev.params = params;
        self.spectrals = spectrals;
        self.enhanced = enhanced;
        self.voice = voice;

        Ok(())
    }
}
