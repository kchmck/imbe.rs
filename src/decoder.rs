use coefs::Coefficients;
use consts::SAMPLES;
use descramble::{descramble, Bootstrap};
use enhance::{self, EnhancedSpectrals, FrameEnergy};
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

impl CAIFrame {
    pub fn new(chunks: [u32; 8], errors: [usize; 7]) -> CAIFrame {
        CAIFrame {
            chunks: chunks,
            errors: errors,
        }
    }
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

    pub fn decode<F: FnMut(f32)>(&mut self, frame: CAIFrame, mut cb: F) {
        let period = match Bootstrap::new(&frame.chunks) {
            Bootstrap::Period(p) => p,
            Bootstrap::Invalid => {
                self.repeat(cb);
                return;
            },
            Bootstrap::Silence => {
                self.silence(cb);
                return;
            },
        };

        let errors = Errors::new(&frame.errors, self.prev.err_rate);

        if enhance::should_repeat(&errors) {
            self.repeat(cb);
            return;
        }

        if enhance::should_mute(&errors) {
            self.silence(cb);
            return;
        }

        let params = BaseParams::new(period);
        let (amps, mut voice, gain_idx) = descramble(&frame.chunks, &params);
        let gains = Gains::new(gain_idx, &amps, &params);
        let coefs = Coefficients::new(&gains, &amps, &params);
        let spectrals = Spectrals::new(&coefs, &params, &self.prev);
        let energy = FrameEnergy::new(&spectrals, &self.prev.energy, &params);

        let mut enhanced = EnhancedSpectrals::new(&spectrals, &energy, &params);
        let amp_thresh = enhance::amp_thresh(&errors, self.prev.amp_thresh);
        enhance::smooth(&mut enhanced, &mut voice, &errors, &energy, amp_thresh);

        let udft = UnvoicedDFT::new();
        let uparts = UnvoicedParts::new(&udft, &params, &voice, &enhanced);

        let vbase = PhaseBase::new(&params, &self.prev);
        let vphase = Phase::new(&params, &voice, &vbase);

        {
            let unvoiced = Unvoiced::new(&uparts, &self.prev.unvoiced);
            let voiced = Voiced::new(&params, &self.prev, &vphase, &enhanced, &voice);

            for n in 0..SAMPLES {
                cb(unvoiced.get(n) + voiced.get(n));
            }
        }

        self.prev = PrevFrame {
            params: params,
            spectrals: spectrals,
            enhanced: enhanced,
            voice: voice,
            err_rate: errors.rate,
            energy: energy,
            amp_thresh: amp_thresh,
            unvoiced: uparts,
            phase_base: vbase,
            phase: vphase,
        };
    }

    fn silence<F: FnMut(f32)>(&self, mut cb: F) {
        for _ in 0..SAMPLES {
            cb(0.0);
        }
    }

    fn repeat<F: FnMut(f32)>(&self, mut cb: F) {
        let params = self.prev.params.clone();
        let voice = self.prev.voice.clone();
        let enhanced = self.prev.enhanced.clone();

        let udft = UnvoicedDFT::new();
        let uparts = UnvoicedParts::new(&udft, &params, &voice, &enhanced);

        let vbase = PhaseBase::new(&params, &self.prev);
        let vphase = Phase::new(&params, &voice, &vbase);

        let unvoiced = Unvoiced::new(&uparts, &self.prev.unvoiced);
        let voiced = Voiced::new(&params, &self.prev, &vphase, &enhanced, &voice);

        for n in 0..SAMPLES {
            cb(unvoiced.get(n) + voiced.get(n));
        }
    }
}
