use std::sync::Arc;

use arrayvec::ArrayVec;
use collect_slice::CollectSlice;
use thread_scoped;

use coefs::Coefficients;
use consts::SAMPLES_PER_FRAME;
use descramble::{descramble, Bootstrap};
use enhance::{self, EnhancedSpectrals, FrameEnergy};
use errors::Errors;
use gain::Gains;
use params::BaseParams;
use prev::PrevFrame;
use spectral::Spectrals;
use unvoiced::{UnvoicedDFT, Unvoiced};
use voiced::{Phase, PhaseBase, Voiced};

/// Number of threads to spin up per frame.
const THREADS: usize = 4;
/// Number of samples to process in each thread.
const SAMPLES_PER_THREAD: usize = SAMPLES_PER_FRAME / THREADS;

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

    pub fn decode(&mut self, frame: CAIFrame, buf: &mut [f32; SAMPLES_PER_FRAME]) {
        let period = match Bootstrap::new(&frame.chunks) {
            Bootstrap::Period(p) => p,
            Bootstrap::Invalid => {
                self.repeat(buf);
                return;
            },
            Bootstrap::Silence => {
                self.silence(buf);
                return;
            },
        };

        let errors = Errors::new(&frame.errors, self.prev.err_rate);

        if enhance::should_repeat(&errors) {
            self.repeat(buf);
            return;
        }

        if enhance::should_mute(&errors) {
            self.silence(buf);
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

        let udft = UnvoicedDFT::new(&params, &voice, &enhanced);
        let vbase = PhaseBase::new(&params, &self.prev);
        let vphase = Phase::new(&params, &voice, &vbase);

        {
            let unvoiced = Arc::new(Unvoiced::new(&udft, &self.prev.unvoiced));
            let voiced = Arc::new(Voiced::new(&params, &self.prev, &vphase, &enhanced, &voice));

            let mut threads = buf.chunks_mut(SAMPLES_PER_THREAD).enumerate().map(|(i, chunk)| {
                let u = unvoiced.clone();
                let v = voiced.clone();

                let start = i * SAMPLES_PER_THREAD;
                let stop = start + SAMPLES_PER_THREAD;

                unsafe {
                    thread_scoped::scoped(move || {
                        (start..stop)
                            .map(|n| u.get(n) + v.get(n))
                            .collect_slice(&mut chunk[..]);
                    })
                }
            }).collect::<ArrayVec<[thread_scoped::JoinGuard<()>; THREADS]>>();

            for thread in threads.drain(..) {
                thread.join();
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
            unvoiced: udft,
            phase_base: vbase,
            phase: vphase,
        };
    }

    fn silence(&self, buf: &mut [f32; SAMPLES_PER_FRAME]) {
        (0..SAMPLES_PER_FRAME).map(|_| 0.0).collect_slice(&mut buf[..]);
    }

    fn repeat(&self, buf: &mut [f32; SAMPLES_PER_FRAME]) {
        let params = self.prev.params.clone();
        let voice = self.prev.voice.clone();
        let enhanced = self.prev.enhanced.clone();

        let udft = UnvoicedDFT::new(&params, &voice, &enhanced);
        let vbase = PhaseBase::new(&params, &self.prev);
        let vphase = Phase::new(&params, &voice, &vbase);

        let unvoiced = Unvoiced::new(&udft, &self.prev.unvoiced);
        let voiced = Voiced::new(&params, &self.prev, &vphase, &enhanced, &voice);

        (0..SAMPLES_PER_FRAME)
            .map(|n| unvoiced.get(n) + voiced.get(n))
            .collect_slice(&mut buf[..]);
    }
}

#[cfg(test)]
mod test {
    use super::THREADS;
    use consts::SAMPLES_PER_FRAME;

    #[test]
    fn verify_threads() {
        assert!(SAMPLES_PER_FRAME % THREADS == 0);
    }
}
