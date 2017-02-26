//! Decode IMBE frames into an audio signal.

use std::sync::Arc;

use collect_slice::CollectSlice;
use crossbeam;
use rand;

use coefs::Coefficients;
use consts::SAMPLES_PER_FRAME;
use descramble::{descramble, Bootstrap};
use enhance::{self, EnhancedSpectrals, FrameEnergy, EnhanceErrors};
use frame::{AudioBuf, ReceivedFrame};
use gain::Gains;
use params::BaseParams;
use prev::PrevFrame;
use spectral::Spectrals;
use unvoiced::{UnvoicedDft, Unvoiced};
use voiced::{Phase, PhaseBase, Voiced};

/// Number of threads to spin up per frame.
const THREADS: usize = 4;
/// Number of samples to process in each thread.
const SAMPLES_PER_THREAD: usize = SAMPLES_PER_FRAME / THREADS;

/// Decodes a stream of IMBE frames.
pub struct ImbeDecoder {
    /// Tracks saved parameters across frames.
    prev: PrevFrame,
}

impl ImbeDecoder {
    /// Create a new `ImbeDecoder` in the default state.
    pub fn new() -> ImbeDecoder {
        ImbeDecoder {
            prev: PrevFrame::default(),
        }
    }

    /// Decode the given frame into the given audio sample buffer.
    pub fn decode(&mut self, frame: ReceivedFrame, buf: &mut AudioBuf) {
        let period = match Bootstrap::new(&frame.chunks) {
            Bootstrap::Period(p) => p,
            Bootstrap::Invalid => {
                // Repeat previous frame on invalid period [p46].
                self.repeat(buf);
                return;
            },
            Bootstrap::Silence => {
                self.silence(buf);
                return;
            },
        };

        let errors = EnhanceErrors::new(&frame.errors, self.prev.err_rate);

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

        let udft = UnvoicedDft::new(&params, &voice, &enhanced, rand::weak_rng());
        let vbase = PhaseBase::new(&params, &self.prev);
        let vphase = Phase::new(&vbase, &params, &self.prev, &voice, rand::weak_rng());

        crossbeam::scope(|scope| {
            let unvoiced = Arc::new(Unvoiced::new(&udft, &self.prev.unvoiced));
            let voiced = Arc::new(Voiced::new(&params, &self.prev, &vphase, &enhanced, &voice));

            for (i, chunk) in buf.chunks_mut(SAMPLES_PER_THREAD).enumerate() {
                let u = unvoiced.clone();
                let v = voiced.clone();

                let start = i * SAMPLES_PER_THREAD;
                let stop = start + SAMPLES_PER_THREAD;

                // Compute Eq 142 for this chunk.
                scope.spawn(move || {
                    (start..stop)
                        .map(|n| u.get(n) + v.get(n))
                        .collect_slice_checked(&mut chunk[..]);
                });
            }
        });

        // Save current parameters.
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

    /// Fill the given audio buffer with silence.
    fn silence(&self, buf: &mut AudioBuf) {
        (0..SAMPLES_PER_FRAME).map(|_| 0.0).collect_slice_checked(&mut buf[..]);
    }

    /// Repeat the previous frame into the given audio buffer.
    fn repeat(&self, buf: &mut AudioBuf) {
        // Apply Eqs 99 through 104.
        let params = self.prev.params.clone();
        let voice = self.prev.voice.clone();
        let enhanced = self.prev.enhanced.clone();

        let udft = UnvoicedDft::new(&params, &voice, &enhanced, rand::weak_rng());
        let vbase = PhaseBase::new(&params, &self.prev);
        let vphase = Phase::new(&vbase, &params, &self.prev, &voice, rand::weak_rng());

        let unvoiced = Unvoiced::new(&udft, &self.prev.unvoiced);
        let voiced = Voiced::new(&params, &self.prev, &vphase, &enhanced, &voice);

        // Repeat frame using previous parameters [p47].
        (0..SAMPLES_PER_FRAME)
            .map(|n| unvoiced.get(n) + voiced.get(n))
            .collect_slice_checked(&mut buf[..]);
    }
}

#[cfg(test)]
mod test {
    use super::THREADS;
    use consts::SAMPLES_PER_FRAME;

    #[test]
    fn verify_threads() {
        // Verify samples are split cleanly over threads.
        assert!(SAMPLES_PER_FRAME % THREADS == 0);
    }
}
